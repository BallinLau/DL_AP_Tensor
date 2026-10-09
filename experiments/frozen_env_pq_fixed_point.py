#!/usr/bin/env python3
"""Frozen-environment fitted P/Q fixed-point experiment.

The runner composes the production Episode P-only and Q-regime stages.  It does
not simulate data, update SDF/FC1/BP, or reproduce a Bellman equation locally.
An exact, provenance-labelled EP2 pre-PV batch bank is a mandatory input.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Iterable, Mapping

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
from config import Config  # noqa: E402
from evaluation.bellman_diagnostics import evaluate_bellman_residuals  # noqa: E402
from evaluation.bp_diagnostics import (  # noqa: E402
    _checkpoint_economic_config,
    build_frozen_transition_data,
)
from evaluation.grids import build_frozen_grid, load_reference_state  # noqa: E402
from evaluation.pq_fixed_point import (  # noqa: E402
    absolute_gap_summary,
    checkpoint_provenance,
    choose_verdict,
    evaluate_grid_p_fixed_point_residuals,
    fixed_rms_scale,
    function_drift,
    load_frozen_batch_bank,
    model_state_sha256,
    normalized_state_distance,
    object_sha256,
    parameter_subset_sha256,
    seed_fixed_mapping,
    update_cosine,
    validate_checkpoint_bank_provenance,
    validate_cycle_teacher_hash,
    verify_production_pq_method_fingerprints,
    write_json,
)
from training.episode import Episode  # noqa: E402
from training.bp_grid_teacher import BPGridTeacher  # noqa: E402


EXPECTED_CODE_COMMIT = "4b236eaacf47b2ac7cb506508e54b21a199e89b0"
DELIVERY_BASE_COMMIT = "5195ad058077f5fd45865c0ce8ed050be0f4fb6a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-batch-bank", type=Path, required=True)
    parser.add_argument("--reference-firm-data", type=Path, required=True)
    parser.add_argument("--reference-macro-data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--fixed-point-seed", type=int, default=24681357)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--b-points", type=int, default=41)
    parser.add_argument("--z-points", type=int, default=41)
    parser.add_argument("--n-child-shocks", type=int, default=64)
    parser.add_argument("--shock-seed", type=int, default=12345)
    parser.add_argument("--eval-chunk-size", type=int, default=8192)
    parser.add_argument("--determinism-tolerance", type=float, default=1e-7)
    parser.add_argument("--fit-normalized-tolerance", type=float, default=5e-2)
    parser.add_argument("--distance-tolerance", type=float, default=1e-3)
    parser.add_argument("--residual-improvement-ratio", type=float, default=0.8)
    parser.add_argument("--boundary-drift-tolerance", type=float, default=1e-3)
    parser.add_argument("--two-cycle-ratio-threshold", type=float, default=0.5)
    parser.add_argument("--skip-determinism-replica", action="store_true")
    parser.add_argument("--allow-code-commit-mismatch", action="store_true")
    return parser.parse_args()


def _git_commit() -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _git_is_ancestor(commit: str) -> bool:
    import subprocess

    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode == 0


def _field(output: Any, name: str) -> torch.Tensor:
    return output[name] if isinstance(output, dict) else getattr(output, name)


def _forward_surfaces(
    model: torch.nn.Module,
    states: torch.Tensor,
    *,
    grid_shape: tuple[int, int],
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    fields = ("P", "P0", "PI", "Phat", "Q", "bar_z", "bar_i")
    pieces: Dict[str, list[torch.Tensor]] = {name: [] for name in fields}
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, states.shape[0], max(1, int(chunk_size))):
                output = model(states[start:start + max(1, int(chunk_size))])
                for name in fields:
                    pieces[name].append(_field(output, name).detach().cpu())
    finally:
        model.train(was_training)
    return {
        name: torch.cat(values, dim=0).reshape(grid_shape).numpy().astype(np.float64)
        for name, values in pieces.items()
    }


def _canonical_batch(grid: Any, transition: Any) -> Dict[str, Any]:
    """Convert the exact-eta evaluator bank back to production's continuous-J input.

    ``Episode`` performs exact eta expansion itself.  The evaluator transition is
    already expanded in adjacent eta pairs, so select one member of each pair;
    the exogenous state and M are identical within a pair.
    """
    children_tensor = transition.children_tensor
    m_raw_tensor = transition.m_raw_tensor
    if children_tensor is None or m_raw_tensor is None:
        raise RuntimeError("canonical transition is missing tensorized children/M")
    if children_tensor.shape[1] % 2:
        raise RuntimeError("exact-eta canonical children must contain adjacent eta pairs")
    continuous_children = children_tensor[:, 0::2, :].clone()
    continuous_m = m_raw_tensor[:, 0::2, :].clone()
    children = [
        torch.cat([continuous_children[:, index, :], continuous_m[:, index, :]], dim=1)
        for index in range(continuous_children.shape[1])
    ]
    parent_m = continuous_m.mean(dim=1)
    return {
        "parent": torch.cat([grid.base_states, parent_m], dim=1),
        "children": children,
    }


def _subset_hashes(episode: Episode) -> Dict[str, str]:
    model = episode.models["policy_value"]
    result = {}
    for name in ("p", "q", "bp"):
        params = episode._policy_value_stage_params(name)
        result[name] = parameter_subset_sha256(model, (id(param) for param in params))
    result["policy_value"] = model_state_sha256(model)
    result["sdf_fc1"] = model_state_sha256(episode.models["sdf_fc1"])
    return result


def _assert_stage_isolation(
    before: Mapping[str, str],
    after: Mapping[str, str],
    *,
    allowed: str,
    stage: str,
) -> None:
    for name in ("p", "q", "bp", "sdf_fc1"):
        if name == allowed:
            continue
        if before[name] != after[name]:
            raise RuntimeError(f"{stage} changed forbidden component {name}")


def _cache_fit(
    episode: Episode,
    batches: list[dict[str, Any]],
    cache: list[Any],
) -> Dict[str, Any]:
    model = episode.models["policy_value"]
    p0_pred: list[np.ndarray] = []
    p0_target: list[np.ndarray] = []
    pi_pred: list[np.ndarray] = []
    pi_target: list[np.ndarray] = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch, item in zip(batches, cache):
                parent = batch["parent"]
                state = parent[:, :7] if parent.shape[1] > 7 else parent
                value_fn = getattr(model, "_value_outputs", None)
                if callable(value_fn):
                    pred0, predi = value_fn(state)
                else:
                    output = model(state)
                    pred0, predi = _field(output, "P0"), _field(output, "PI")
                p0_pred.append(pred0.detach().cpu().numpy())
                pi_pred.append(predi.detach().cpu().numpy())
                p0_target.append(item.p0_value_target.detach().cpu().numpy())
                pi_target.append(item.pi_value_target.detach().cpu().numpy())
    finally:
        model.train(was_training)
    p0 = absolute_gap_summary(np.concatenate(p0_pred), np.concatenate(p0_target))
    pi = absolute_gap_summary(np.concatenate(pi_pred), np.concatenate(pi_target))
    both_pred = np.concatenate([np.concatenate(p0_pred), np.concatenate(pi_pred)])
    both_target = np.concatenate([np.concatenate(p0_target), np.concatenate(pi_target)])
    combined = absolute_gap_summary(both_pred, both_target)
    combined["rms"] = float(
        np.sqrt(np.mean(np.square(both_pred.reshape(-1) - both_target.reshape(-1))))
    )
    return {"p0": p0, "pi": pi, "combined": combined}


def _build_p_caches(
    episode: Episode,
    batches: list[dict[str, Any]],
    teacher: torch.nn.Module,
    q_teacher: torch.nn.Module,
) -> list[Any]:
    combined = object_sha256(
        {"equity": teacher.state_dict(), "q": q_teacher.state_dict()}
    )
    cache, _ = episode._build_pq_value_target_cache(
        batches,
        teacher,
        q_target_model=q_teacher,
        teacher_hash=combined,
        grid_config_hash=episode._pq_grid_config_hash(),
    )
    return cache


def _prepare_cycle_batch_plan(
    episode: Episode,
    train_batches: list[dict[str, Any]],
    validation_batches: list[dict[str, Any]],
) -> Dict[str, Any]:
    """Mirror production P-only eta balancing while preserving Q's frozen bank."""
    p_train_batches, p_train_eta_summary = episode._balance_p_current_eta_batches(
        train_batches,
        stream="train",
    )
    balance_validation = bool(
        getattr(episode.hyperparams, "pv_current_eta_balance_validation", True)
    )
    if balance_validation:
        p_validation_batches, p_validation_eta_summary = (
            episode._balance_p_current_eta_batches(
                validation_batches,
                stream="validation",
            )
        )
        if not validation_batches:
            p_validation_eta_summary = dict(p_train_eta_summary)
            p_validation_eta_summary["stream"] = "train_fallback"
    else:
        p_validation_batches = validation_batches
        counts = episode._current_eta_counts_from_batches(validation_batches)
        p_validation_eta_summary = {
            "enabled": False,
            "stream": "validation",
            **counts,
            "eta0_count_after": counts["eta0_count"],
            "eta1_count_after": counts["eta1_count"],
            "eta1_share_after": counts["eta1_share"],
        }
    return {
        "p_train_batches": p_train_batches,
        "p_validation_batches": p_validation_batches,
        "q_train_batches": train_batches,
        "q_validation_batches": validation_batches,
        "p_train_eta_summary": p_train_eta_summary,
        "p_validation_eta_summary": p_validation_eta_summary,
        "p_train_balanced_hash": object_sha256(p_train_batches),
        "p_validation_balanced_hash": object_sha256(p_validation_batches),
        "q_train_original_hash": object_sha256(train_batches),
        "q_validation_original_hash": object_sha256(validation_batches),
    }


def _canonical_p_teacher_targets(
    episode: Episode,
    canonical_batch: dict[str, Any],
    teacher: torch.nn.Module,
) -> Dict[str, np.ndarray]:
    """Record the exact BPGridTeacher objects used to build the P target cache."""
    parent_state, children, m_list, _parent_hash, _children_hash, _m_hash = (
        episode._policy_batch_hash_components(canonical_batch)
    )
    children, _raw, m_list, child_weights = episode._expand_policy_expectation_children(
        children, m_list, m_list
    )
    grid_teacher = BPGridTeacher.from_hyperparams(
        teacher,
        episode.loss_fns["p0"],
        episode.loss_fns["pi"],
        episode.hyperparams,
        q_target_model=teacher,
    )
    output: Dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for branch in ("p0", "pi"):
            result = grid_teacher.compute_value_target(
                parent_state=parent_state,
                children=children,
                m_list=m_list,
                branch=branch,
                child_weights=child_weights,
            )
            output[f"{branch}_value_target"] = (
                result["value_star"].detach().cpu().numpy().reshape(-1)
            )
            output[f"{branch}_bp_star"] = (
                result["bp_star"].detach().cpu().numpy().reshape(-1)
            )
    return output


@contextmanager
def _capture_q_phase_fit(
    episode: Episode,
    on_batches: list[dict[str, Any]],
    canonical_batches: list[dict[str, Any]],
):
    original = episode._run_q_regime_phase
    records: list[dict[str, Any]] = []

    def wrapped(self: Episode, **kwargs: Any) -> Dict[str, Any]:
        phase = str(kwargs["phase"])
        frozen_p = kwargs["frozen_p_model"]
        q_target = kwargs["q_target_model"]
        entry: Dict[str, Any] = {
            "phase": phase,
            "q_target_hash": self._state_dict_hash(q_target),
            "frozen_p_hash": self._state_dict_hash(frozen_p),
        }
        on_bank = []
        canonical_bank = []
        if phase in {"survival", "polish"}:
            on_bank = self._build_q_validation_bank(on_batches, frozen_p)
            canonical_bank = self._build_q_validation_bank(canonical_batches, frozen_p)
        summary = original(**kwargs)
        entry["training_summary"] = summary
        if on_bank:
            entry["on_distribution"] = self._evaluate_q_validation_bank(
                on_bank, frozen_p_model=frozen_p, q_target_model=q_target
            )
        if canonical_bank:
            entry["canonical"] = self._evaluate_q_validation_bank(
                canonical_bank, frozen_p_model=frozen_p, q_target_model=q_target
            )
        records.append(entry)
        return summary

    episode._run_q_regime_phase = MethodType(wrapped, episode)
    try:
        yield records
    finally:
        episode._run_q_regime_phase = original


def _q_fit_from_records(records: list[dict[str, Any]], bank: str) -> Dict[str, float]:
    usable = [item for item in records if bank in item]
    if not usable:
        return {
            key: float("nan")
            for key in ("mean", "p50", "p90", "p99", "max", "rms", "rms_raw")
        }
    selected = next(
        (item for item in reversed(usable) if item["phase"] == "polish"),
        usable[-1],
    )[bank]
    metrics = selected.get("metrics", {})
    mean = float(selected.get("score_primary", float("nan")))
    normalize = bool(metrics.get("q_bellman_normalized_by_target_scale", False))
    suffix = "normalized" if normalize else "raw"
    mean_square = float(
        metrics.get(f"q_claim_bellman_mean_square_{suffix}", float("nan"))
    )
    mean_square_raw = float(
        metrics.get("q_claim_bellman_mean_square_raw", float("nan"))
    )
    rms = float(np.sqrt(mean_square)) if np.isfinite(mean_square) and mean_square >= 0 else float("nan")
    rms_raw = (
        float(np.sqrt(mean_square_raw))
        if np.isfinite(mean_square_raw) and mean_square_raw >= 0
        else float("nan")
    )
    return {
        "mean": mean,
        "p50": float(metrics.get(f"q_claim_bellman_abs_p50_{suffix}", float("nan"))),
        "p90": float(metrics.get(f"q_claim_bellman_abs_p90_{suffix}", float("nan"))),
        "p99": float(metrics.get(f"q_claim_bellman_abs_p99_{suffix}", float("nan"))),
        "max": float(metrics.get(f"q_claim_bellman_abs_max_{suffix}", float("nan"))),
        "rms": rms,
        "rms_raw": rms_raw,
    }


def _plot_lines(frame: pd.DataFrame, columns: Iterable[str], output: Path, *, hline: float | None = None) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for column in columns:
        if column in frame:
            axis.plot(frame["cycle"], frame[column], marker="o", label=column)
    if hline is not None:
        axis.axhline(float(hline), color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("cycle")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_surface(values: np.ndarray, b_values: np.ndarray, z_values: np.ndarray, output: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 4.8))
    image = axis.pcolormesh(b_values, z_values, values.T, shading="auto")
    axis.set_xlabel("b")
    axis.set_ylabel("z")
    axis.set_title(title)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _make_episode(loaded: Any, policy_model: torch.nn.Module, device: torch.device) -> Episode:
    models = {
        "policy_value": policy_model,
        "sdf_fc1": loaded.models["sdf_fc1"],
    }
    if "fc2" in loaded.models:
        models["fc2"] = loaded.models["fc2"]
    target = copy.deepcopy(policy_model).to(device)
    target.eval()
    target.requires_grad_(False)
    return Episode(
        models=models,
        optimizers={},
        config=Config,
        hyperparams=copy.deepcopy(loaded.hyperparams),
        device=device,
        episode_id=2,
        firm_target=target,
        q_checkpoint_loaded=True,
    )


def _run_cycle(
    episode: Episode,
    train_batches: list[dict[str, Any]],
    validation_batches: list[dict[str, Any]],
    canonical_batch: dict[str, Any],
    *,
    fixed_seed: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    seed_fixed_mapping(fixed_seed)
    model = episode.models["policy_value"]
    teacher = copy.deepcopy(model).to(episode.device)
    teacher.eval()
    teacher.requires_grad_(False)
    teacher_hash = episode._state_dict_hash(teacher)
    start_hash = episode._state_dict_hash(model)
    if teacher_hash != start_hash:
        raise RuntimeError("cycle teacher does not match X^k start snapshot")
    episode.firm_target = teacher

    batch_plan = _prepare_cycle_batch_plan(
        episode,
        train_batches,
        validation_batches,
    )
    p_train_batches = batch_plan["p_train_batches"]
    p_validation_batches = batch_plan["p_validation_batches"]

    on_p_cache = _build_p_caches(episode, p_validation_batches, teacher, teacher)
    canonical_p_cache = _build_p_caches(episode, [canonical_batch], teacher, teacher)
    canonical_p_targets = _canonical_p_teacher_targets(
        episode, canonical_batch, teacher
    )
    hashes_before_p = _subset_hashes(episode)
    p_summary = episode._run_policy_value_evaluation_stage(
        p_train_batches,
        p_validation_batches,
        teacher,
        int(getattr(episode.hyperparams, "pv_eval_epochs", 1)),
        q_target_model=teacher,
    )
    hashes_after_p = _subset_hashes(episode)
    _assert_stage_isolation(hashes_before_p, hashes_after_p, allowed="p", stage="P stage")
    if p_summary.get("status") != "accepted":
        raise RuntimeError(f"P stage was not accepted: {p_summary.get('status')}")
    p_fit_on = _cache_fit(episode, p_validation_batches, on_p_cache)
    p_fit_canonical = _cache_fit(episode, [canonical_batch], canonical_p_cache)

    frozen_p = copy.deepcopy(model).to(episode.device)
    frozen_p.eval()
    frozen_p.requires_grad_(False)
    frozen_p_hash = episode._state_dict_hash(frozen_p)
    hashes_before_q = _subset_hashes(episode)
    with _capture_q_phase_fit(
        episode,
        batch_plan["q_validation_batches"],
        [canonical_batch],
    ) as q_records:
        q_summary = episode._run_q_regime_training(
            batch_plan["q_train_batches"],
            frozen_p,
            validation_batches=batch_plan["q_validation_batches"],
        )
    hashes_after_q = _subset_hashes(episode)
    _assert_stage_isolation(hashes_before_q, hashes_after_q, allowed="q", stage="Q stage")
    if q_summary.get("status") not in {"accepted", "accepted_improved", "accepted_reverted"}:
        raise RuntimeError(f"Q stage was not accepted: {q_summary.get('status')}")
    if episode._state_dict_hash(frozen_p) != frozen_p_hash:
        raise RuntimeError("frozen P snapshot changed during Q stage")
    episode.firm_target = copy.deepcopy(model).to(episode.device)
    episode.firm_target.eval()
    episode.firm_target.requires_grad_(False)
    result = {
        "teacher_hash": teacher_hash,
        "cycle_start_hash": start_hash,
        "cycle_end_hash": episode._state_dict_hash(model),
        "component_hashes_before_p": hashes_before_p,
        "component_hashes_after_p": hashes_after_p,
        "component_hashes_after_q": hashes_after_q,
        "p_summary": p_summary,
        "q_summary": q_summary,
        "q_phase_fit": q_records,
        "p_fit_on": p_fit_on,
        "p_fit_canonical": p_fit_canonical,
        "q_fit_on": _q_fit_from_records(q_records, "on_distribution"),
        "q_fit_canonical": _q_fit_from_records(q_records, "canonical"),
        "canonical_p_targets": canonical_p_targets,
        "p_train_balanced_hash": batch_plan["p_train_balanced_hash"],
        "p_validation_balanced_hash": batch_plan["p_validation_balanced_hash"],
        "q_train_original_hash": batch_plan["q_train_original_hash"],
        "q_validation_original_hash": batch_plan["q_validation_original_hash"],
        "p_train_eta_summary": batch_plan["p_train_eta_summary"],
        "p_validation_eta_summary": batch_plan["p_validation_eta_summary"],
        "p_changed": hashes_before_p["p"] != hashes_after_p["p"],
        "q_changed": hashes_before_q["q"] != hashes_after_q["q"],
    }
    return result, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _report(
    output: Path,
    verdict: Mapping[str, Any],
    audit: Mapping[str, Any],
    blocker: str | None = None,
) -> None:
    lines = [
        "# Frozen-Environment P-Q Fixed-Point Test",
        "",
        "## 1. Audit",
        "",
        f"- P stage trainable set: `{audit['p_trainable']}`",
        f"- Q stage trainable set: `{audit['q_trainable']}`",
        f"- Q phases: `{audit['q_phases']}`",
        f"- Q target refresh: `{audit['q_target_refresh_mode']}`",
        "- Each cycle snapshots current X^k; P uses that snapshot for equity continuation and Q pricing; Q then freezes P^{k+1}.",
        "",
        "## 2. Frozen environment verification",
        "",
        "The training and validation batches are loaded from one hash-checked EP2 pre-PV batch bank. No simulation is called.",
        "",
        "## 3. Per-cycle stage fit",
        "",
        "See `stage_fit_metrics.csv` and per-cycle phase JSON records.",
        "",
        "## 4. Function drift",
        "",
        "See `pq_fixed_point_cycles.csv`.",
        "",
        "## 5. Contraction ratios",
        "",
        "Ratios are reported cycle by cycle; a single ratio below one is not treated as proof of contraction.",
        "",
        "## 6. Oscillation / two-cycle test",
        "",
        "Update cosine and two-step normalized distance are reported separately.",
        "",
        "## 7. Fixed-point residuals",
        "",
        "The main P residual is the contemporaneous BPGridTeacher argmax residual. "
        "The BP-head action residual is reported separately as a compatibility diagnostic "
        "and does not enter the verdict. Q keeps its production Bellman residual.",
        "",
        "## 8. Default-boundary stability",
        "",
        "DefaultShare, D-to-S, S-to-D, and bar_z drift are reported on the fixed canonical bank.",
        "",
        "## 9. On-distribution vs canonical projection",
        "",
        "P and Q fitted-stage metrics are reported for both banks.",
        "",
        "## 10. Final verdict",
        "",
        f"**{verdict['primary']}. {verdict['label']}: {verdict['reason']}.**",
        "",
        "Thresholds:",
        "",
        "```json",
        json.dumps(verdict["thresholds"], indent=2, sort_keys=True),
        "```",
        "",
        "Early/tail statistics:",
        "",
        "```json",
        json.dumps(verdict["statistics"], indent=2, sort_keys=True),
        "```",
    ]
    if blocker:
        lines.extend(["", "## Provenance blocker", "", blocker])
    (output / "pq_fixed_point_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("Frozen P-Q fixed-point experiment requires CUDA")
    current_commit = _git_commit()
    if not _git_is_ancestor(DELIVERY_BASE_COMMIT) and not args.allow_code_commit_mismatch:
        raise SystemExit(
            f"GitHub delivery base {DELIVERY_BASE_COMMIT} is not an ancestor of {current_commit}"
        )
    pq_method_fingerprints = verify_production_pq_method_fingerprints(Episode)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for directory in (
        output / "checkpoints",
        output / "figs" / "canonical_P_surfaces",
        output / "figs" / "canonical_Q_surfaces",
        output / "canonical_surfaces",
        output / "cycle_phase_diagnostics",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    train_batches, validation_batches, dataset_meta = load_frozen_batch_bank(
        args.frozen_batch_bank, device=device
    )
    if dataset_meta["provenance"].get("source_commit") != EXPECTED_CODE_COMMIT:
        raise SystemExit("frozen batch bank source_commit does not match the fixed code commit")
    checkpoint_meta = checkpoint_provenance(args.checkpoint)
    provenance_guard = validate_checkpoint_bank_provenance(
        checkpoint_meta,
        dataset_meta["provenance"],
    )
    loaded = load_analysis_checkpoint(args.checkpoint, device=device)
    _, reference = load_reference_state(
        args.reference_firm_data,
        macro_path=args.reference_macro_data,
    )
    with _checkpoint_economic_config(loaded.economic_config):
        grid = build_frozen_grid(
            reference,
            b_min=0.0,
            b_max=1.0,
            b_points=args.b_points,
            z_min=-2.0,
            z_max=2.0,
            z_points=args.z_points,
            device=device,
        )
        transition = build_frozen_transition_data(
            loaded.models["sdf_fc1"],
            grid.base_states,
            reference,
            loaded.hyperparams,
            loaded.economic_config,
            n_child_shocks=args.n_child_shocks,
            shock_seed=args.shock_seed,
            shock_bank_max_child_shocks=args.n_child_shocks,
        )
        canonical_batch = _canonical_batch(grid, transition)
        initial_policy = loaded.models["policy_value"]
        episode = _make_episode(loaded, initial_policy, device)
        audit = {
            "p_trainable": [name for name, parameter in initial_policy.named_parameters()
                            if id(parameter) in {id(item) for item in episode._policy_value_stage_params("p")}],
            "q_trainable": [name for name, parameter in initial_policy.named_parameters()
                            if id(parameter) in {id(item) for item in episode._policy_value_stage_params("q")}],
            "q_phases": ["zero", "default", "survival", "polish"],
            "q_target_refresh_mode": str(getattr(loaded.hyperparams, "q_target_refresh_mode", "phase")),
            "pv_eval_epochs": int(getattr(loaded.hyperparams, "pv_eval_epochs", 0)),
            "q_phase_epochs": {
                "zero": int(getattr(loaded.hyperparams, "q_zero_boundary_epochs", 0)),
                "default": int(getattr(loaded.hyperparams, "q_default_pretrain_epochs", 0)),
                "survival": int(getattr(loaded.hyperparams, "q_survival_aio_epochs", 0)),
                "polish": int(getattr(loaded.hyperparams, "q_mixed_polish_epochs", 0)),
            },
        }
        config = {
            "code_commit": current_commit,
            "delivery_base_commit": DELIVERY_BASE_COMMIT,
            "grid_run_source_commit": EXPECTED_CODE_COMMIT,
            "production_pq_method_fingerprints": pq_method_fingerprints,
            "checkpoint": str(args.checkpoint.resolve()),
            "frozen_batch_bank": str(args.frozen_batch_bank.resolve()),
            "checkpoint_provenance": checkpoint_meta,
            "bank_provenance": dataset_meta["provenance"],
            "provenance_guard": provenance_guard,
            "cycles": args.cycles,
            "fixed_point_seed": args.fixed_point_seed,
            "canonical_grid": {"b": [0.0, 1.0, args.b_points], "z": [-2.0, 2.0, args.z_points]},
            "n_child_shocks": args.n_child_shocks,
            "shock_seed": args.shock_seed,
            "reference_state": reference.to_dict(),
            "audit": audit,
            "hyperparams": vars(loaded.hyperparams),
            "economic_config": loaded.economic_config.to_dict(),
            "verdict_thresholds": {
                "fit_normalized_tolerance": args.fit_normalized_tolerance,
                "distance_tolerance": args.distance_tolerance,
                "residual_improvement_ratio": args.residual_improvement_ratio,
                "boundary_drift_tolerance": args.boundary_drift_tolerance,
                "two_cycle_ratio_threshold": args.two_cycle_ratio_threshold,
            },
        }
        write_json(output / "config.json", config)
        write_json(output / "dataset_hashes.json", dataset_meta)
        immutable_hashes = {
            "sdf_fc1": model_state_sha256(episode.models["sdf_fc1"]),
            "dataset": dataset_meta["dataset_sha256"],
            "economic_config": object_sha256(loaded.economic_config.to_dict()),
            "canonical_transition": object_sha256({
                "children": transition.children_tensor,
                "m_raw": transition.m_raw_tensor,
                "weights": transition.branch_weights,
            }),
        }
        write_json(output / "environment_hashes.json", immutable_hashes)

        surfaces = [_forward_surfaces(initial_policy, grid.base_states, grid_shape=grid.shape, chunk_size=args.eval_chunk_size)]
        torch.save(
            {"cycle": 0, "models": {"policy_value": initial_policy.state_dict()}, "config": config},
            output / "checkpoints" / "cycle_00.pt",
        )
        p_scale = fixed_rms_scale(surfaces[0]["P"])
        q_scale = fixed_rms_scale(surfaces[0]["Q"])
        p_stage_scale = fixed_rms_scale(surfaces[0]["P0"], surfaces[0]["PI"])
        config["fixed_cycle0_scales"] = {
            "P_function_drift_scale": p_scale,
            "P_stage_fit_scale_from_P0_PI": p_stage_scale,
            "Q_stage_and_drift_scale": q_scale,
        }
        write_json(output / "config.json", config)
        np.savez_compressed(output / "canonical_surfaces" / "cycle_00.npz", **surfaces[0])
        rows: list[dict[str, Any]] = []
        residual_rows: list[dict[str, Any]] = []
        stage_rows: list[dict[str, Any]] = []
        checkpoint_rows: list[dict[str, Any]] = []
        delta_vectors: list[np.ndarray] = []
        sdf_initial_hash = immutable_hashes["sdf_fc1"]
        previous_teacher_hash: str | None = None
        previous_cycle_end_hash: str | None = None
        expected_p_batch_hashes: tuple[str, str] | None = None

        if not args.skip_determinism_replica:
            replica_a = _make_episode(loaded, copy.deepcopy(initial_policy).to(device), device)
            replica_b = _make_episode(loaded, copy.deepcopy(initial_policy).to(device), device)
            _, state_a = _run_cycle(
                replica_a, train_batches, validation_batches, canonical_batch,
                fixed_seed=args.fixed_point_seed,
            )
            _, state_b = _run_cycle(
                replica_b, train_batches, validation_batches, canonical_batch,
                fixed_seed=args.fixed_point_seed,
            )
            max_difference = max(
                float((state_a[key] - state_b[key]).abs().max().item()) for key in state_a
            )
            write_json(output / "determinism_check.json", {
                "max_parameter_abs_difference": max_difference,
                "tolerance": args.determinism_tolerance,
                "passed": max_difference <= args.determinism_tolerance,
            })
            if max_difference > args.determinism_tolerance:
                raise RuntimeError("cycle-1 deterministic replicas differ")

        for cycle in range(args.cycles):
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            start_hash = episode._state_dict_hash(episode.models["policy_value"])
            result, state = _run_cycle(
                episode,
                train_batches,
                validation_batches,
                canonical_batch,
                fixed_seed=args.fixed_point_seed,
            )
            validate_cycle_teacher_hash(
                cycle_start_hash=start_hash,
                teacher_hash=result["teacher_hash"],
                previous_teacher_hash=previous_teacher_hash,
                previous_cycle_end_hash=previous_cycle_end_hash,
            )
            if model_state_sha256(episode.models["sdf_fc1"]) != sdf_initial_hash:
                raise RuntimeError("SDF/FC1 changed during frozen P/Q iteration")
            current_p_batch_hashes = (
                result["p_train_balanced_hash"],
                result["p_validation_balanced_hash"],
            )
            if expected_p_batch_hashes is None:
                expected_p_batch_hashes = current_p_batch_hashes
            elif current_p_batch_hashes != expected_p_batch_hashes:
                raise RuntimeError("P current-eta balanced batches changed across cycles")
            current = _forward_surfaces(
                episode.models["policy_value"],
                grid.base_states,
                grid_shape=grid.shape,
                chunk_size=args.eval_chunk_size,
            )
            surfaces.append(current)
            np.savez_compressed(
                output / "canonical_surfaces" / f"cycle_{cycle + 1:02d}.npz",
                **current,
            )
            drift = function_drift(surfaces[-2], current, p_scale=p_scale, q_scale=q_scale)
            delta_vectors.append(drift.delta_vector)
            rho = (
                drift.d_joint / rows[-1]["d_joint"]
                if rows and float(rows[-1]["d_joint"]) > 1e-15 else float("nan")
            )
            cosine = (
                update_cosine(delta_vectors[-1], delta_vectors[-2])
                if len(delta_vectors) >= 2 else float("nan")
            )
            two_step = (
                normalized_state_distance(surfaces[-3], current, p_scale=p_scale, q_scale=q_scale)
                if len(surfaces) >= 3 else float("nan")
            )
            residual_surfaces, residual_summary = evaluate_bellman_residuals(
                episode.models["policy_value"],
                grid,
                transition,
                loaded.economic_config,
                chunk_size=args.eval_chunk_size,
            )
            grid_p_residual = evaluate_grid_p_fixed_point_residuals(
                episode,
                canonical_batch,
                current_model=episode.models["policy_value"],
            )
            grid_p_summary = grid_p_residual["summary"]
            default = current["Phat"] <= 0.0
            previous_default = surfaces[-2]["Phat"] <= 0.0
            bar_z_drift = np.abs(current["bar_z"] - surfaces[-2]["bar_z"])
            runtime = time.perf_counter() - started
            row = {
                "cycle": cycle + 1,
                "P_mean_abs_drift": drift.p_stats["mean"],
                "P_p90_abs_drift": drift.p_stats["p90"],
                "P_max_abs_drift": drift.p_stats["max"],
                "Q_mean_abs_drift": drift.q_stats["mean"],
                "Q_p90_abs_drift": drift.q_stats["p90"],
                "Q_max_abs_drift": drift.q_stats["max"],
                "dP": drift.d_p,
                "dQ": drift.d_q,
                "d_joint": drift.d_joint,
                "rho": rho,
                "cos_theta": cosine,
                "two_step_distance": two_step,
                "P_stage_fit_on_mean": result["p_fit_on"]["combined"]["mean"],
                "P_stage_fit_on_p90": result["p_fit_on"]["combined"]["p90"],
                "P_stage_fit_on_norm": result["p_fit_on"]["combined"]["rms"] / p_stage_scale,
                "P_stage_fit_canonical_mean": result["p_fit_canonical"]["combined"]["mean"],
                "P_stage_fit_canonical_p90": result["p_fit_canonical"]["combined"]["p90"],
                "P_stage_fit_canonical_norm": result["p_fit_canonical"]["combined"]["rms"] / p_stage_scale,
                "Q_stage_fit_on_mean": result["q_fit_on"]["mean"],
                "Q_stage_fit_on_p50": result["q_fit_on"]["p50"],
                "Q_stage_fit_on_p90": result["q_fit_on"]["p90"],
                "Q_stage_fit_on_p99": result["q_fit_on"]["p99"],
                "Q_stage_fit_on_max": result["q_fit_on"]["max"],
                "Q_stage_fit_on_norm": result["q_fit_on"]["rms_raw"] / q_scale,
                "Q_stage_fit_canonical_mean": result["q_fit_canonical"]["mean"],
                "Q_stage_fit_canonical_p50": result["q_fit_canonical"]["p50"],
                "Q_stage_fit_canonical_p90": result["q_fit_canonical"]["p90"],
                "Q_stage_fit_canonical_p99": result["q_fit_canonical"]["p99"],
                "Q_stage_fit_canonical_max": result["q_fit_canonical"]["max"],
                "Q_stage_fit_canonical_norm": result["q_fit_canonical"]["rms_raw"] / q_scale,
                "P_head_policy_residual_mean": np.mean([
                    residual_summary["p0_residual_abs_mean"],
                    residual_summary["pi_residual_abs_mean"],
                ]),
                "P_head_policy_residual_p90": max(
                    residual_summary["p0_residual_abs_p90"],
                    residual_summary["pi_residual_abs_p90"],
                ),
                "P_grid_fixed_point_residual_mean": grid_p_summary[
                    "P_grid_fixed_point_residual_mean"
                ],
                "P_grid_fixed_point_residual_p90": grid_p_summary[
                    "P_grid_fixed_point_residual_p90"
                ],
                "P0_grid_residual_mean": grid_p_summary[
                    "p0_grid_residual_abs_mean"
                ],
                "P0_grid_residual_p90": grid_p_summary[
                    "p0_grid_residual_abs_p90"
                ],
                "PI_grid_residual_mean": grid_p_summary[
                    "pi_grid_residual_abs_mean"
                ],
                "PI_grid_residual_p90": grid_p_summary[
                    "pi_grid_residual_abs_p90"
                ],
                "Q_fixed_point_residual_mean": residual_summary["q_residual_abs_mean"],
                "Q_fixed_point_residual_p90": residual_summary["q_residual_abs_p90"],
                "DefaultShare": float(default.mean()),
                "D_to_S_share": float((previous_default & ~default).mean()),
                "S_to_D_share": float((~previous_default & default).mean()),
                "boundary_switch_share": float(
                    ((previous_default & ~default) | (~previous_default & default)).mean()
                ),
                "bar_z_mean_abs_drift": float(bar_z_drift.mean()),
                "runtime_sec": runtime,
                "peak_gpu_memory": int(torch.cuda.max_memory_allocated(device)),
            }
            row["boundary_drift"] = max(
                float(row["boundary_switch_share"]),
                float(row["bar_z_mean_abs_drift"]),
            )
            rows.append(row)
            residual_detail = {"cycle": cycle + 1, **grid_p_summary}
            for key, value in residual_summary.items():
                if key.startswith("p0_"):
                    residual_detail[f"p0_head_policy_{key[3:]}"] = value
                elif key.startswith("pi_"):
                    residual_detail[f"pi_head_policy_{key[3:]}"] = value
                elif key.startswith("q_"):
                    residual_detail[key] = value
            residual_rows.append(residual_detail)
            stage_rows.append({
                "cycle": cycle + 1,
                **{key: value for key, value in row.items() if "stage_fit" in key},
            })
            checkpoint_rows.append({
                "cycle": cycle + 1,
                "teacher_hash": result["teacher_hash"],
                "cycle_start_hash": result["cycle_start_hash"],
                "cycle_end_hash": result["cycle_end_hash"],
                "p_hash": result["component_hashes_after_q"]["p"],
                "q_hash": result["component_hashes_after_q"]["q"],
                "bp_hash": result["component_hashes_after_q"]["bp"],
                "sdf_fc1_hash": result["component_hashes_after_q"]["sdf_fc1"],
                "dataset_hash": dataset_meta["dataset_sha256"],
                "p_train_balanced_hash": result["p_train_balanced_hash"],
                "p_validation_balanced_hash": result["p_validation_balanced_hash"],
                "q_train_original_hash": result["q_train_original_hash"],
                "q_validation_original_hash": result["q_validation_original_hash"],
                "p_changed": result["p_changed"],
                "q_changed": result["q_changed"],
            })
            previous_teacher_hash = result["teacher_hash"]
            previous_cycle_end_hash = result["cycle_end_hash"]
            np.savez_compressed(
                output / "cycle_phase_diagnostics" / f"cycle_{cycle + 1:02d}_p_targets.npz",
                **result.pop("canonical_p_targets"),
            )
            write_json(output / "cycle_phase_diagnostics" / f"cycle_{cycle + 1:02d}.json", result)
            torch.save(
                {"cycle": cycle + 1, "models": {"policy_value": state}, "config": config},
                output / "checkpoints" / f"cycle_{cycle + 1:02d}.pt",
            )
            if cycle + 1 in {1, 2, 5, 10}:
                _plot_surface(current["P"], grid.b_values, grid.z_values,
                              output / "figs" / "canonical_P_surfaces" / f"cycle_{cycle + 1:02d}.png",
                              f"P cycle {cycle + 1}")
                _plot_surface(current["Q"], grid.b_values, grid.z_values,
                              output / "figs" / "canonical_Q_surfaces" / f"cycle_{cycle + 1:02d}.png",
                              f"Q cycle {cycle + 1}")

        _plot_surface(surfaces[0]["P"], grid.b_values, grid.z_values,
                      output / "figs" / "canonical_P_surfaces" / "cycle_00.png", "P cycle 0")
        _plot_surface(surfaces[0]["Q"], grid.b_values, grid.z_values,
                      output / "figs" / "canonical_Q_surfaces" / "cycle_00.png", "Q cycle 0")
        frame = pd.DataFrame(rows)
        frame.to_csv(output / "pq_fixed_point_cycles.csv", index=False)
        frame.to_csv(output / "cycle_metrics.csv", index=False)
        pd.DataFrame(stage_rows).to_csv(output / "stage_fit_metrics.csv", index=False)
        pd.DataFrame(residual_rows).to_csv(output / "fixed_point_residuals.csv", index=False)
        pd.DataFrame(checkpoint_rows).to_csv(output / "cycle_checkpoint_hashes.csv", index=False)
        _plot_lines(frame, ("dP", "dQ", "d_joint"), output / "figs" / "joint_distance_by_cycle.png")
        _plot_lines(frame, ("rho",), output / "figs" / "contraction_ratio.png", hline=1.0)
        _plot_lines(frame, ("P_grid_fixed_point_residual_mean", "Q_fixed_point_residual_mean"),
                    output / "figs" / "fixed_point_residuals.png")
        _plot_lines(frame, ("P_head_policy_residual_mean",),
                    output / "figs" / "p_head_policy_compatibility_residual.png")
        _plot_lines(frame, ("P_stage_fit_on_mean", "P_grid_fixed_point_residual_mean",
                            "Q_stage_fit_on_mean", "Q_fixed_point_residual_mean"),
                    output / "figs" / "stage_fit_vs_fixed_point_gap.png")
        _plot_lines(frame, ("cos_theta",), output / "figs" / "update_cosine.png", hline=0.0)
        _plot_lines(frame, ("DefaultShare", "D_to_S_share", "S_to_D_share"),
                    output / "figs" / "default_share_by_cycle.png")
        verdict = choose_verdict(
            rows,
            fit_normalized_tolerance=args.fit_normalized_tolerance,
            distance_tolerance=args.distance_tolerance,
            residual_improvement_ratio=args.residual_improvement_ratio,
            boundary_drift_tolerance=args.boundary_drift_tolerance,
            two_cycle_ratio_threshold=args.two_cycle_ratio_threshold,
        )
        write_json(output / "verdict.json", verdict)
        _report(output, verdict, audit)
        print(json.dumps({"verdict": verdict, "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
