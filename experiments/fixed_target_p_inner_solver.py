#!/usr/bin/env python3
"""Diagnose the production P inner solver against immutable Bellman targets.

The experiment starts from reported fixed-point cycles 1, 5, and 10, which
correspond to cycle-start checkpoints 00, 04, and 09.  It loads the exact
frozen EP2 pre-PV bank recorded in the fixed-point config, constructs each P
target cache once, and then runs independent production-objective and pure
value-regression fits from the same initialization and deterministic RNG tape.

This runner never updates Q, SDF/FC1, BP, the teacher, or the target cache and
never calls simulation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis.checkpoint_loader import load_analysis_checkpoint  # noqa: E402
from evaluation.bp_diagnostics import (  # noqa: E402
    _checkpoint_economic_config,
    build_frozen_transition_data,
)
from evaluation.fixed_target_p_inner_solver import (  # noqa: E402
    SELECTED_REPORTED_CYCLES,
    classify_diagnosis,
    cycle_start_index,
    evaluate_fixed_cache,
    parameter_max_abs_difference,
    parameter_state_sha256,
    parameter_update_norms,
    pure_value_regression_loss,
    snapshot_named_parameters,
    surface_complexity,
    write_json,
)
from evaluation.grids import build_frozen_grid, load_reference_state  # noqa: E402
from evaluation.pq_fixed_point import (  # noqa: E402
    checkpoint_provenance,
    load_frozen_batch_bank,
    model_state_sha256,
    object_sha256,
    seed_fixed_mapping,
    validate_checkpoint_bank_provenance,
)
from experiments.frozen_env_pq_fixed_point import (  # noqa: E402
    _build_p_caches,
    _canonical_batch,
    _canonical_p_teacher_targets,
    _make_episode,
    _prepare_cycle_batch_plan,
    _subset_hashes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-point-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cycles", type=int, nargs="+", default=list(SELECTED_REPORTED_CYCLES))
    parser.add_argument("--budgets", type=int, nargs="+", default=[100, 200, 500, 1000])
    parser.add_argument("--memorization-epochs", type=int, default=1000)
    parser.add_argument("--memorization-batches", type=int, nargs="+", default=[1, 4, 16, 0])
    parser.add_argument("--seed", type=int, default=24681357)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--determinism-cycle", type=int, default=1)
    parser.add_argument("--determinism-budget", type=int, default=100)
    parser.add_argument("--determinism-tolerance", type=float, default=1e-7)
    parser.add_argument("--fit-tolerance", type=float, default=0.05)
    parser.add_argument("--memorization-tolerance", type=float, default=0.01)
    parser.add_argument("--b-points", type=int, default=41)
    parser.add_argument("--z-points", type=int, default=41)
    parser.add_argument("--n-child-shocks", type=int, default=64)
    parser.add_argument("--shock-seed", type=int, default=12345)
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_policy_state(path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    models = payload.get("models", {})
    state = models.get("policy_value")
    if not isinstance(state, Mapping):
        raise ValueError(f"checkpoint lacks models.policy_value: {path}")
    return {str(key): value.detach().cpu().clone() for key, value in state.items()}


def _cache_target_stats(cache: Sequence[Any]) -> pd.DataFrame:
    rows = []
    for name in ("p0_value_target", "pi_value_target"):
        values = torch.cat([getattr(item, name).detach().cpu().reshape(-1) for item in cache])
        array = values.numpy().astype(np.float64)
        rows.append({
            "target": name,
            "n": int(array.size),
            "mean": float(array.mean()),
            "std": float(array.std()),
            "rms": float(np.sqrt(np.mean(np.square(array)))),
            "min": float(array.min()),
            "p01": float(np.quantile(array, 0.01)),
            "p50": float(np.quantile(array, 0.50)),
            "p99": float(np.quantile(array, 0.99)),
            "max": float(array.max()),
            "dynamic_range": float(array.max() - array.min()),
        })
    return pd.DataFrame(rows)


def _cache_hashes(episode: Any, train_cache: Sequence[Any], val_cache: Sequence[Any]) -> Dict[str, str]:
    return {
        "train_target_cache": episode._pq_value_cache_hash(list(train_cache)),
        "validation_target_cache": episode._pq_value_cache_hash(list(val_cache)),
    }


@contextmanager
def _pure_value_scope(episode: Any):
    model = episode.models["policy_value"]
    prefixes = ("value_encoder", "v0_head", "vi_head")
    original = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    try:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(
                name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes
            )
        yield
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = original[name]


def _pure_validation_score(episode: Any, batches: Sequence[Any], cache: Sequence[Any]) -> float:
    model = episode.models["policy_value"]
    was_training = model.training
    values = []
    try:
        model.eval()
        with torch.no_grad():
            for batch, item in zip(batches, cache):
                total, _ = pure_value_regression_loss(episode, batch, item)
                values.append(float(total.detach().item()))
    finally:
        model.train(was_training)
    return float(np.mean(values)) if values else float("inf")


def _mean_records(records: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    keys = sorted({key for record in records for key in record})
    result = {}
    for key in keys:
        values = np.asarray([float(record[key]) for record in records if key in record])
        values = values[np.isfinite(values)]
        if values.size:
            result[key] = float(values.mean())
    return result


def _training_params(episode: Any, objective: str) -> list[torch.nn.Parameter]:
    if objective == "production":
        return episode._policy_value_stage_params("p")
    model = episode.models["policy_value"]
    return episode._unique_params([model.value_encoder, model.v0_head, model.vi_head])


def run_fixed_target_fit(
    *,
    loaded: Any,
    start_state: Mapping[str, torch.Tensor],
    train_batches: Sequence[Mapping[str, Any]],
    validation_batches: Sequence[Mapping[str, Any]],
    canonical_batches: Sequence[Mapping[str, Any]],
    train_cache: Sequence[Any],
    validation_cache: Sequence[Any],
    canonical_cache: Sequence[Any],
    target_cache_hashes: Mapping[str, str],
    objective: str,
    epochs: int,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    if objective not in {"production", "pure"}:
        raise ValueError("objective must be production or pure")
    seed_fixed_mapping(seed)
    model = copy.deepcopy(loaded.models["policy_value"]).to(device)
    model.load_state_dict(start_state, strict=True)
    episode = _make_episode(loaded, model, device)
    params = _training_params(episode, objective)
    optimizer = episode._make_policy_value_stage_optimizer(params)
    model_start_hash = parameter_state_sha256(model)
    immutable_start = _subset_hashes(episode)
    teacher_hash = str(next(iter(train_cache)).teacher_hash)
    best_score = float("inf")
    best_checkpoint = None
    best_epoch = None
    accepted_epochs = 0
    rejected_epochs = 0
    optimizer_steps = 0
    attempted_optimizer_steps = 0
    hard_spike_count = 0
    soft_spike_count = 0
    nonfinite_count = 0
    curve: list[dict[str, Any]] = []
    max_skip_ratio = float(getattr(episode.hyperparams, "pv_epoch_max_skip_ratio", 0.05))
    hard_threshold = float(getattr(episode.hyperparams, "pv_grad_hard_threshold", 1000.0))
    soft_threshold = float(getattr(episode.hyperparams, "pv_grad_soft_threshold", 100.0))
    clip_norm = float(getattr(episode.hyperparams, "pv_eval_grad_clip_norm", 10.0))
    prefixes = ("value_encoder", "v0_head", "vi_head", "barz_model", "bari_model")
    scope = episode._policy_value_train_scope("p") if objective == "production" else _pure_value_scope(episode)
    started = time.perf_counter()
    model.train()
    with scope:
        for epoch in range(1, int(epochs) + 1):
            epoch_checkpoint = episode._policy_value_stage_checkpoint(optimizer, None)
            rng_state = episode._capture_rng_state()
            parameter_before = snapshot_named_parameters(model, prefixes)
            batch_records: list[Dict[str, float]] = []
            raw_norms: list[float] = []
            clipped_norms: list[float] = []
            epoch_steps = 0
            epoch_hard = 0
            epoch_soft = 0
            epoch_nonfinite = 0
            for batch, item in zip(train_batches, train_cache):
                optimizer.zero_grad(set_to_none=True)
                if objective == "production":
                    total, terms = episode._compute_cached_value_loss(batch, item)
                else:
                    total, terms = pure_value_regression_loss(episode, batch, item)
                if not torch.isfinite(total):
                    epoch_nonfinite += 1
                    nonfinite_count += 1
                    continue
                total.backward()
                grad_groups = episode._policy_value_grad_group_norms()
                raw_norm, clipped_norm = episode._clip_params_with_raw_norm(params, clip_norm)
                if not np.isfinite(raw_norm):
                    optimizer.zero_grad(set_to_none=True)
                    epoch_nonfinite += 1
                    nonfinite_count += 1
                    continue
                if raw_norm > hard_threshold:
                    optimizer.zero_grad(set_to_none=True)
                    epoch_hard += 1
                    hard_spike_count += 1
                    continue
                if raw_norm > soft_threshold:
                    epoch_soft += 1
                    soft_spike_count += 1
                optimizer.step()
                attempted_optimizer_steps += 1
                optimizer_steps += 1
                epoch_steps += 1
                raw_norms.append(raw_norm)
                clipped_norms.append(clipped_norm)
                batch_records.append({**terms, **grad_groups})

            skip_ratio = float(epoch_hard + epoch_nonfinite) / max(len(train_batches), 1)
            if epoch_steps == 0 or skip_ratio > max_skip_ratio:
                episode._restore_policy_value_stage_checkpoint(optimizer, epoch_checkpoint, None)
                episode._restore_rng_state(rng_state)
                optimizer_steps -= epoch_steps
                rejected_epochs += 1
                curve.append({
                    "epoch": epoch,
                    "accepted": False,
                    "reason": "no_optimizer_steps" if epoch_steps == 0 else "skip_ratio_exceeded",
                    "skip_ratio": skip_ratio,
                })
                continue

            train_eval = evaluate_fixed_cache(episode, train_batches, train_cache)
            validation_eval = evaluate_fixed_cache(episode, validation_batches, validation_cache)
            canonical_eval = evaluate_fixed_cache(episode, canonical_batches, canonical_cache)
            score = (
                episode._evaluate_cached_p_score(list(validation_batches), list(validation_cache))[0]
                if objective == "production"
                else _pure_validation_score(episode, validation_batches, validation_cache)
            )
            if not np.isfinite(score):
                episode._restore_policy_value_stage_checkpoint(optimizer, epoch_checkpoint, None)
                episode._restore_rng_state(rng_state)
                optimizer_steps -= epoch_steps
                rejected_epochs += 1
                curve.append({"epoch": epoch, "accepted": False, "reason": "nonfinite_validation"})
                continue
            accepted_epochs += 1
            row: Dict[str, Any] = {
                "epoch": epoch,
                "accepted": True,
                "optimizer_steps": epoch_steps,
                "optimizer_steps_cumulative": optimizer_steps,
                "validation_score": score,
                "raw_grad_norm_mean": float(np.mean(raw_norms)),
                "raw_grad_norm_max": float(np.max(raw_norms)),
                "clipped_grad_norm_mean": float(np.mean(clipped_norms)),
                "clip_ratio": float(np.mean(np.asarray(raw_norms) > clip_norm)),
                "skip_ratio": skip_ratio,
                "hard_spike_count": epoch_hard,
                "soft_spike_count": epoch_soft,
                "nonfinite_count": epoch_nonfinite,
                **parameter_update_norms(model, parameter_before),
            }
            row.update({f"train_{key}": value for key, value in train_eval.metrics.items()})
            row.update({f"validation_{key}": value for key, value in validation_eval.metrics.items()})
            row.update({f"canonical_{key}": value for key, value in canonical_eval.metrics.items()})
            row.update({f"train_objective_{key}": value for key, value in train_eval.objective.items()})
            row.update({f"validation_objective_{key}": value for key, value in validation_eval.objective.items()})
            row.update({f"batch_{key}": value for key, value in _mean_records(batch_records).items()})
            curve.append(row)
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_checkpoint = episode._policy_value_stage_checkpoint(optimizer, None)

    if best_checkpoint is None:
        raise RuntimeError(f"{objective} fixed-target run produced no valid checkpoint")
    episode._restore_policy_value_stage_checkpoint(optimizer, best_checkpoint, None)
    final_train = evaluate_fixed_cache(episode, train_batches, train_cache)
    final_validation = evaluate_fixed_cache(episode, validation_batches, validation_cache)
    final_canonical = evaluate_fixed_cache(episode, canonical_batches, canonical_cache)
    cache_hashes_after = _cache_hashes(episode, train_cache, validation_cache)
    if dict(target_cache_hashes) != cache_hashes_after:
        raise RuntimeError("fixed target cache changed during inner-solver run")
    immutable_end = _subset_hashes(episode)
    for component in ("q", "bp", "sdf_fc1"):
        if immutable_end[component] != immutable_start[component]:
            raise RuntimeError(f"{objective} inner solve changed immutable component {component}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curve).to_csv(output_dir / "learning_curve.csv", index=False)
    final_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    torch.save(
        {"objective": objective, "epochs": epochs, "best_epoch": best_epoch, "models": {"policy_value": final_state}},
        output_dir / "best_checkpoint.pt",
    )
    summary: Dict[str, Any] = {
        "objective": objective,
        "epochs_requested": int(epochs),
        "accepted_epochs": accepted_epochs,
        "rejected_epochs": rejected_epochs,
        "optimizer_steps": optimizer_steps,
        "attempted_optimizer_steps": attempted_optimizer_steps,
        "hard_spike_count": hard_spike_count,
        "soft_spike_count": soft_spike_count,
        "nonfinite_count": nonfinite_count,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "teacher_hash": teacher_hash,
        "model_start_hash": model_start_hash,
        "model_final_hash": parameter_state_sha256(model),
        "target_cache_hashes": dict(target_cache_hashes),
        "runtime_sec": time.perf_counter() - started,
    }
    for bank, evaluation in (
        ("train", final_train),
        ("validation", final_validation),
        ("canonical", final_canonical),
    ):
        summary.update({f"{bank}_{key}": value for key, value in evaluation.metrics.items()})
        summary.update({f"{bank}_objective_{key}": value for key, value in evaluation.objective.items()})
    write_json(output_dir / "summary.json", summary)
    return {"summary": summary, "final_state": final_state, "curve": curve}


def _plot_learning_curves(
    summary: pd.DataFrame,
    output: Path,
    *,
    bank: str,
    objective: str | None = None,
) -> None:
    if objective is not None:
        summary = summary[summary.objective == objective]
    fig, axis = plt.subplots(figsize=(8, 5))
    for (cycle, objective, budget), group in summary.groupby(["reported_cycle", "objective", "budget"]):
        curve = pd.read_csv(group.iloc[0]["curve_path"])
        column = f"{bank}_combined_normalized_rms"
        if column in curve:
            axis.plot(curve["epoch"], curve[column], label=f"c{cycle} {objective} b{budget}")
    axis.axhline(0.05, color="black", linestyle="--", linewidth=1)
    axis.set_yscale("log")
    axis.set_xlabel("epoch")
    axis.set_ylabel(f"{bank} combined normalized RMS")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_budget_comparison(summary: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for (cycle, objective), group in summary.groupby(["reported_cycle", "objective"]):
        ordered = group.sort_values("budget")
        axis.plot(
            ordered["budget"], ordered["validation_combined_normalized_rms"],
            marker="o", label=f"cycle {cycle} {objective}",
        )
    axis.axhline(0.05, color="black", linestyle="--", linewidth=1)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("independent epoch budget")
    axis.set_ylabel("validation normalized RMS")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_train_validation(summary: pd.DataFrame, output: Path) -> None:
    production = summary[summary.objective == "production"]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for cycle, group in production.groupby("reported_cycle"):
        ordered = group.sort_values("budget")
        axis.plot(
            ordered["budget"], ordered["train_combined_normalized_rms"],
            marker="o", label=f"cycle {cycle} train",
        )
        axis.plot(
            ordered["budget"], ordered["validation_combined_normalized_rms"],
            marker="s", linestyle="--", label=f"cycle {cycle} validation",
        )
    axis.axhline(0.05, color="black", linestyle="--", linewidth=1)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("independent epoch budget")
    axis.set_ylabel("combined normalized RMS")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_target_complexity(frame: pd.DataFrame, output: Path) -> None:
    selected = frame[frame.surface.isin(["p0_value_target", "pi_value_target"])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for surface, group in selected.groupby("surface"):
        ordered = group.sort_values("reported_cycle")
        label = surface.replace("_value_target", "")
        axes[0].plot(ordered.reported_cycle, ordered.total_variation, marker="o", label=label)
        axes[1].plot(ordered.reported_cycle, ordered.curvature_b, marker="o", label=f"{label} b")
        axes[1].plot(ordered.reported_cycle, ordered.curvature_z, marker="s", label=f"{label} z")
        axes[2].plot(ordered.reported_cycle, ordered.local_max_gradient, marker="o", label=label)
    for axis, title in zip(axes, ("total variation", "mean absolute curvature", "local max gradient")):
        axis.set_title(title)
        axis.set_xlabel("reported cycle")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_selected_curve_columns(
    summary: pd.DataFrame,
    output: Path,
    *,
    columns: Sequence[str],
    cycle: int,
    objective: str = "production",
) -> None:
    candidates = summary[
        (summary.reported_cycle == cycle) & (summary.objective == objective)
    ]
    selected = candidates.loc[candidates.budget.idxmax()]
    curve = pd.read_csv(selected.curve_path)
    fig, axis = plt.subplots(figsize=(8, 5))
    for column in columns:
        if column in curve and curve[column].notna().any():
            axis.plot(curve.epoch, curve[column], label=column)
    axis.set_xlabel("epoch")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_surface_triptych(
    targets: Mapping[int, Mapping[str, np.ndarray]],
    key: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4), squeeze=False)
    for axis, cycle in zip(axes[0], sorted(targets)):
        image = axis.imshow(targets[cycle][key].T, origin="lower", aspect="auto")
        axis.set_title(f"Cycle {cycle} {key}")
        axis.set_xlabel("b index")
        axis.set_ylabel("z index")
        fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    usable = frame.loc[:, [column for column in columns if column in frame]].copy()
    if usable.empty:
        return ["No rows available."]
    lines = [
        "| " + " | ".join(usable.columns) + " |",
        "| " + " | ".join(["---"] * len(usable.columns)) + " |",
    ]
    for _, row in usable.iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_report(
    output: Path,
    diagnosis: Mapping[str, Any],
    audit: Mapping[str, Any],
    run_frame: pd.DataFrame,
    memorization_frame: pd.DataFrame,
    complexity_frame: pd.DataFrame,
) -> None:
    recommendation = {
        "A": "Replace fixed 100 epochs with target-fit convergence stopping on the frozen validation cache.",
        "B": "Separate or reweight the identified production penalty before changing network capacity.",
        "C": "Audit gradients, value scaling, clipping, optimizer state, and parameter routing.",
        "D": "Only now test additional value-network capacity against the same frozen targets.",
        "E": "Improve state-bank coverage or projection diagnostics before changing the Bellman operator.",
        "F": "Control target geometry across outer updates before resuming fixed-point analysis.",
        "G": "Collect the missing discriminating diagnostic; do not change the model yet.",
    }[diagnosis["primary"]]
    max_budget = int(run_frame.budget.max())
    comparison = run_frame[run_frame.budget == max_budget].sort_values(
        ["reported_cycle", "objective"]
    )
    complexity_selected = complexity_frame[
        complexity_frame.surface.isin(["p0_value_target", "pi_value_target", "p0_bp_star", "pi_bp_star"])
    ].sort_values(["reported_cycle", "surface"])
    lines = [
        "# Fixed-Target P Inner-Solver Diagnostic",
        "",
        "## 1. Experimental audit",
        "",
        f"- Selected reported cycles: `{audit['reported_cycles']}`.",
        f"- Start checkpoints: `{audit['start_checkpoints']}`.",
        "- Q, BP, SDF/FC1, simulation, teacher, and target caches remain frozen.",
        "",
        "## 2. Frozen-target verification",
        "",
        "See `audit.json`, `provenance.json`, and each cycle target-cache hash record.",
        "",
        "## 3. Production objective learning curves",
        "",
        "See `run_summary.csv` and `figs/production_train_fit_vs_epoch.png`.",
        "",
        "## 4. Does longer training solve the problem?",
        "",
        "Independent budgets restart from the same cycle-start state; they are not continuations.",
        "The complete budget path is in `run_summary.csv`.",
        "",
        "## 5. Production vs pure target regression",
        "",
        "Pure regression keeps production Huber and value-scale semantics but removes z/b penalties.",
        "",
        f"Best-checkpoint comparison at the largest independent budget ({max_budget} epochs):",
        "",
        *_markdown_table(
            comparison,
            (
                "reported_cycle", "objective", "best_epoch",
                "train_combined_normalized_rms",
                "validation_combined_normalized_rms",
                "canonical_combined_normalized_rms",
            ),
        ),
        "",
        "## 6. Small-subset memorization",
        "",
        *_markdown_table(
            memorization_frame,
            (
                "subset", "best_epoch", "train_combined_normalized_rms",
                "validation_combined_normalized_rms", "optimizer_steps",
            ),
        ),
        "",
        "## 7. Gradient / optimizer diagnostics",
        "",
        "Per-epoch raw/clipped norms, clipping shares, skipped batches, and module update norms are in each learning curve.",
        "",
        "## 8. Target complexity: Cycle 1 vs 5 vs 10",
        "",
        *_markdown_table(
            complexity_selected,
            (
                "reported_cycle", "surface", "std", "dynamic_range",
                "tv_b", "tv_z", "curvature_b", "curvature_z",
                "local_max_gradient", "local_max_curvature", "argmax_jump_share",
            ),
        ),
        "",
        "## 9. Train vs validation vs canonical",
        "",
        "All three spaces are reported separately in `run_summary.csv`.",
        "",
        "## 10. Primary diagnosis",
        "",
        f"**{diagnosis['primary']}. {diagnosis['label']}: {diagnosis['reason']}**",
        "",
        "## 11. Recommended next modification",
        "",
        recommendation,
    ]
    (output / "fixed_target_p_inner_solver_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("formal fixed-target P inner-solver diagnostic requires CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    run = args.fixed_point_run.expanduser().resolve()
    config_path = run / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"missing fixed-point config: {config_path}")
    fixed_config = _load_json(config_path)
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run / "fixed_target_p_inner_solver"
    )
    (output / "figs").mkdir(parents=True, exist_ok=True)
    base_checkpoint = Path(fixed_config["checkpoint"])
    frozen_batch_bank = Path(fixed_config["frozen_batch_bank"])
    reference_firm = Path(fixed_config["reference_state"]["source"])
    reference_macro = Path(fixed_config["reference_state"]["macro_source"])
    for path in (base_checkpoint, frozen_batch_bank, reference_firm, reference_macro):
        if not path.is_file():
            raise SystemExit(f"required provenance input does not exist: {path}")

    loaded = load_analysis_checkpoint(base_checkpoint, device=device)
    train_batches, validation_batches, dataset_meta = load_frozen_batch_bank(
        frozen_batch_bank, device=device
    )
    checkpoint_meta = checkpoint_provenance(base_checkpoint)
    provenance_guard = validate_checkpoint_bank_provenance(
        checkpoint_meta, dataset_meta["provenance"]
    )
    expected_dataset = _load_json(run / "dataset_hashes.json")
    if expected_dataset["dataset_sha256"] != dataset_meta["dataset_sha256"]:
        raise RuntimeError("frozen batch bank hash differs from fixed-point run")
    expected_environment = _load_json(run / "environment_hashes.json")
    if expected_environment["sdf_fc1"] != model_state_sha256(loaded.models["sdf_fc1"]):
        raise RuntimeError("SDF/FC1 hash differs from fixed-point environment")

    with _checkpoint_economic_config(loaded.economic_config):
        _, reference = load_reference_state(reference_firm, macro_path=reference_macro)
        grid = build_frozen_grid(
            reference, b_min=0.0, b_max=1.0, b_points=args.b_points,
            z_min=-2.0, z_max=2.0, z_points=args.z_points, device=device,
        )
        transition = build_frozen_transition_data(
            loaded.models["sdf_fc1"], grid.base_states, reference,
            loaded.hyperparams, loaded.economic_config,
            n_child_shocks=args.n_child_shocks, shock_seed=args.shock_seed,
            shock_bank_max_child_shocks=args.n_child_shocks,
        )
        canonical_batch = _canonical_batch(grid, transition)
        all_results: list[dict[str, Any]] = []
        production_map: Dict[int, Dict[int, Dict[str, float]]] = {}
        pure_map: Dict[int, Dict[int, Dict[str, float]]] = {}
        complexity_map: Dict[int, Dict[str, Dict[str, float]]] = {}
        canonical_target_map: Dict[int, Dict[str, np.ndarray]] = {}
        cycle_artifacts: Dict[int, Dict[str, Any]] = {}

        for reported_cycle in args.cycles:
            start_index = cycle_start_index(reported_cycle)
            checkpoint_path = run / "checkpoints" / f"cycle_{start_index:02d}.pt"
            if not checkpoint_path.is_file():
                raise RuntimeError(f"missing cycle-start checkpoint: {checkpoint_path}")
            start_state = _load_policy_state(checkpoint_path)
            start_model = copy.deepcopy(loaded.models["policy_value"]).to(device)
            start_model.load_state_dict(start_state, strict=True)
            episode = _make_episode(loaded, start_model, device)
            plan = _prepare_cycle_batch_plan(episode, train_batches, validation_batches)
            p_train = plan["p_train_batches"]
            p_validation = plan["p_validation_batches"] or p_train
            teacher = copy.deepcopy(start_model).to(device)
            teacher.eval()
            teacher.requires_grad_(False)
            teacher_hash_before = parameter_state_sha256(teacher)
            q_hash_before = _subset_hashes(episode)["q"]
            train_cache = _build_p_caches(episode, p_train, teacher, teacher)
            validation_cache = _build_p_caches(episode, p_validation, teacher, teacher)
            canonical_cache = _build_p_caches(episode, [canonical_batch], teacher, teacher)
            cache_hashes = _cache_hashes(episode, train_cache, validation_cache)
            cycle_dir = output / f"cycle{reported_cycle:02d}_target"
            cycle_dir.mkdir(parents=True, exist_ok=True)
            _cache_target_stats(train_cache).to_csv(cycle_dir / "train_target_stats.csv", index=False)
            _cache_target_stats(validation_cache).to_csv(
                cycle_dir / "validation_target_stats.csv", index=False
            )
            targets_flat = _canonical_p_teacher_targets(episode, canonical_batch, teacher)
            targets = {key: value.reshape(grid.shape) for key, value in targets_flat.items()}
            canonical_target_map[reported_cycle] = targets
            np.savez_compressed(cycle_dir / "canonical_targets.npz", **targets)
            complexity_map[reported_cycle] = {}
            complexity_rows = []
            for key, values in targets.items():
                stats = surface_complexity(
                    values,
                    jump_threshold=0.1 if key.endswith("bp_star") else None,
                )
                complexity_map[reported_cycle][key] = stats
                complexity_rows.append({"reported_cycle": reported_cycle, "surface": key, **stats})
            pd.DataFrame(complexity_rows).to_csv(cycle_dir / "canonical_target_complexity.csv", index=False)
            target_record = {
                "reported_cycle": reported_cycle,
                "cycle_start_index": start_index,
                "cycle_start_checkpoint": str(checkpoint_path),
                "cycle_start_model_hash": parameter_state_sha256(start_model),
                "teacher_hash_before": teacher_hash_before,
                "teacher_hash_after": parameter_state_sha256(teacher),
                "q_hash_before": q_hash_before,
                "q_hash_after": _subset_hashes(episode)["q"],
                "target_cache_hashes": cache_hashes,
                "original_train_hash": plan["q_train_original_hash"],
                "original_validation_hash": plan["q_validation_original_hash"],
                "balanced_train_hash": plan["p_train_balanced_hash"],
                "balanced_validation_hash": plan["p_validation_balanced_hash"],
                "p_train_eta_summary": plan["p_train_eta_summary"],
                "p_validation_eta_summary": plan["p_validation_eta_summary"],
            }
            if target_record["teacher_hash_before"] != target_record["teacher_hash_after"]:
                raise RuntimeError("teacher changed while constructing frozen target cache")
            if target_record["q_hash_before"] != target_record["q_hash_after"]:
                raise RuntimeError("Q changed while constructing frozen target cache")
            write_json(cycle_dir / "target_cache_hash.json", target_record)
            cycle_artifacts[reported_cycle] = {
                "start_state": start_state,
                "train_batches": p_train,
                "validation_batches": p_validation,
                "train_cache": train_cache,
                "validation_cache": validation_cache,
                "canonical_cache": canonical_cache,
                "cache_hashes": cache_hashes,
            }
            production_map[reported_cycle] = {}
            pure_map[reported_cycle] = {}
            for objective, result_map in (("production", production_map), ("pure", pure_map)):
                for budget in args.budgets:
                    run_dir = cycle_dir / ("production" if objective == "production" else "pure_regression") / f"budget_{budget}"
                    result = run_fixed_target_fit(
                        loaded=loaded, start_state=start_state,
                        train_batches=p_train, validation_batches=p_validation,
                        canonical_batches=[canonical_batch], train_cache=train_cache,
                        validation_cache=validation_cache, canonical_cache=canonical_cache,
                        target_cache_hashes=cache_hashes, objective=objective,
                        epochs=budget, seed=args.seed, device=device, output_dir=run_dir,
                    )
                    result_map[reported_cycle][budget] = result["summary"]
                    all_results.append({
                        "reported_cycle": reported_cycle,
                        "cycle_start_index": start_index,
                        "objective": objective,
                        "budget": budget,
                        "curve_path": str(run_dir / "learning_curve.csv"),
                        **result["summary"],
                    })

        complexity_frame = pd.DataFrame([
            {"reported_cycle": cycle, "surface": surface, **stats}
            for cycle, surfaces in complexity_map.items()
            for surface, stats in surfaces.items()
        ])
        complexity_frame.to_csv(output / "target_complexity.csv", index=False)
        run_frame = pd.DataFrame(all_results)
        run_frame.to_csv(output / "run_summary.csv", index=False)

        worst_cycle = max(args.cycles)
        artifacts = cycle_artifacts[worst_cycle]
        memorization: Dict[str, Dict[str, float]] = {}
        memorization_rows = []
        full_budget = max(args.budgets)
        for count in args.memorization_batches:
            label = "full" if int(count) == 0 else f"{int(count)}_batch" if int(count) == 1 else f"{int(count)}_batches"
            if int(count) == 0 and args.memorization_epochs == full_budget:
                summary = dict(pure_map[worst_cycle][full_budget])
                summary["reused_pure_full_run"] = True
            else:
                n = len(artifacts["train_batches"]) if int(count) == 0 else min(int(count), len(artifacts["train_batches"]))
                subset_batches = artifacts["train_batches"][:n]
                subset_cache = artifacts["train_cache"][:n]
                subset_hashes = {
                    "train_target_cache": _make_episode(
                        loaded,
                        copy.deepcopy(loaded.models["policy_value"]).to(device),
                        device,
                    )._pq_value_cache_hash(list(subset_cache)),
                    "validation_target_cache": _make_episode(
                        loaded,
                        copy.deepcopy(loaded.models["policy_value"]).to(device),
                        device,
                    )._pq_value_cache_hash(list(subset_cache)),
                }
                result = run_fixed_target_fit(
                    loaded=loaded, start_state=artifacts["start_state"],
                    train_batches=subset_batches, validation_batches=subset_batches,
                    canonical_batches=[canonical_batch], train_cache=subset_cache,
                    validation_cache=subset_cache, canonical_cache=artifacts["canonical_cache"],
                    target_cache_hashes=subset_hashes, objective="pure",
                    epochs=args.memorization_epochs, seed=args.seed, device=device,
                    output_dir=output / f"cycle{worst_cycle:02d}_target" / "memorization" / label,
                )
                summary = result["summary"]
            memorization[label] = summary
            memorization_rows.append({"subset": label, **summary})
        pd.DataFrame(memorization_rows).to_csv(output / "memorization_summary.csv", index=False)

        deterministic_artifacts = cycle_artifacts[args.determinism_cycle]
        replica_dir = output / "determinism_replica"
        replica = run_fixed_target_fit(
            loaded=loaded, start_state=deterministic_artifacts["start_state"],
            train_batches=deterministic_artifacts["train_batches"],
            validation_batches=deterministic_artifacts["validation_batches"],
            canonical_batches=[canonical_batch],
            train_cache=deterministic_artifacts["train_cache"],
            validation_cache=deterministic_artifacts["validation_cache"],
            canonical_cache=deterministic_artifacts["canonical_cache"],
            target_cache_hashes=deterministic_artifacts["cache_hashes"],
            objective="production", epochs=args.determinism_budget,
            seed=args.seed, device=device, output_dir=replica_dir,
        )
        reference_checkpoint = torch.load(
            output / f"cycle{args.determinism_cycle:02d}_target" / "production"
            / f"budget_{args.determinism_budget}" / "best_checkpoint.pt",
            map_location="cpu",
        )["models"]["policy_value"]
        max_difference = parameter_max_abs_difference(
            reference_checkpoint, replica["final_state"]
        )
        determinism = {
            "reported_cycle": args.determinism_cycle,
            "budget": args.determinism_budget,
            "max_parameter_abs_difference": max_difference,
            "tolerance": args.determinism_tolerance,
            "passed": max_difference <= args.determinism_tolerance,
        }
        write_json(output / "determinism_check.json", determinism)
        if not determinism["passed"]:
            raise RuntimeError("fixed-target deterministic replicas differ")

        diagnosis = classify_diagnosis(
            production=production_map, pure=pure_map,
            memorization=memorization, complexity=complexity_map,
            fit_tolerance=args.fit_tolerance,
            memorization_tolerance=args.memorization_tolerance,
        )
        write_json(output / "diagnosis.json", diagnosis)
        audit = {
            "reported_cycles": list(args.cycles),
            "start_checkpoints": {
                str(cycle): f"cycle_{cycle_start_index(cycle):02d}.pt" for cycle in args.cycles
            },
            "budgets": list(args.budgets),
            "same_initialization_per_budget": True,
            "target_cache_constructed_once_per_cycle": True,
            "target_cache_refreshed_during_training": False,
            "production_objective_terms": [
                "p0_cached_value_loss", "pi_cached_value_loss",
                "p0_cached_penalty_z", "pi_cached_penalty_z", "pi_cached_penalty_b",
            ],
            "pure_objective_terms": ["p0_cached_value_loss", "pi_cached_value_loss"],
            "production_trainable_modules": [
                "value_encoder", "v0_head", "vi_head", "barz_model", "bari_model"
            ],
            "pure_trainable_modules": ["value_encoder", "v0_head", "vi_head"],
            "bar_models_used_by_fixed_cache_value_forward": False,
        }
        write_json(output / "audit.json", audit)
        write_json(output / "provenance.json", {
            "fixed_point_run": str(run),
            "fixed_point_config": fixed_config,
            "checkpoint_provenance": checkpoint_meta,
            "bank_provenance": dataset_meta["provenance"],
            "provenance_guard": provenance_guard,
            "dataset_sha256": dataset_meta["dataset_sha256"],
            "environment_hashes": expected_environment,
        })
        _plot_learning_curves(
            run_frame,
            output / "figs" / "production_train_fit_vs_epoch.png",
            bank="train",
            objective="production",
        )
        _plot_learning_curves(
            run_frame,
            output / "figs" / "production_validation_fit_vs_epoch.png",
            bank="validation",
            objective="production",
        )
        _plot_train_validation(run_frame, output / "figs" / "production_train_vs_validation.png")
        _plot_budget_comparison(run_frame, output / "figs" / "production_vs_pure_regression.png")
        for key, filename in (
            ("p0_value_target", "P0_target_surfaces.png"),
            ("pi_value_target", "PI_target_surfaces.png"),
            ("p0_bp_star", "bp0_star_surfaces.png"),
            ("pi_bp_star", "bpI_star_surfaces.png"),
        ):
            _plot_surface_triptych(canonical_target_map, key, output / "figs" / filename)
        _plot_target_complexity(
            complexity_frame, output / "figs" / "target_complexity_by_cycle.png"
        )
        _plot_selected_curve_columns(
            run_frame,
            output / "figs" / "loss_component_curves.png",
            cycle=max(args.cycles),
            columns=(
                "train_objective_p0_cached_value_loss",
                "train_objective_pi_cached_value_loss",
                "train_objective_p0_cached_penalty_z",
                "train_objective_pi_cached_penalty_z",
                "train_objective_pi_cached_penalty_b",
                "train_objective_total",
            ),
        )
        _plot_selected_curve_columns(
            run_frame,
            output / "figs" / "gradient_norm_curves.png",
            cycle=max(args.cycles),
            columns=(
                "raw_grad_norm_mean",
                "raw_grad_norm_max",
                "clipped_grad_norm_mean",
                "value_encoder_update_norm",
                "v0_head_update_norm",
                "vi_head_update_norm",
            ),
        )
        mem_frame = pd.DataFrame(memorization_rows)
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.plot(np.arange(len(mem_frame)), mem_frame["train_combined_normalized_rms"], marker="o")
        axis.set_xticks(np.arange(len(mem_frame)), mem_frame["subset"], rotation=20)
        axis.set_yscale("log")
        axis.set_ylabel("train normalized RMS")
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "figs" / "memorization_fit_vs_dataset_size.png", dpi=160)
        plt.close(fig)
        _write_report(
            output,
            diagnosis,
            audit,
            run_frame,
            mem_frame,
            complexity_frame,
        )
        print(json.dumps({"output_dir": str(output), "diagnosis": diagnosis}, indent=2))


if __name__ == "__main__":
    main()
