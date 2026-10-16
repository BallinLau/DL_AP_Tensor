#!/usr/bin/env python3
"""Fixed-P Q frozen-target versus differentiable full-residual root diagnostic."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.analyze_p_only_movement_channels import (  # noqa: E402
    P_PREFIXES,
    BP_PREFIXES,
    _load_policy_state,
    _load_setup,
)
from analysis.test_frozen_policy_root_solve import (  # noqa: E402
    _clone_state,
    _restore_state,
    _state_hash,
    _tensor_digest,
)
from losses.q_loss import classify_q_parent_regimes  # noqa: E402
from training.inner_fixed_point import (  # noqa: E402
    evaluate_inner_operator_snapshot,
    named_parameter_hash,
)


Q_PREFIXES = ("q_encoder", "q_head")


@dataclass(frozen=True)
class QSurvivalBatch:
    parent: torch.Tensor
    child_states: torch.Tensor
    m: torch.Tensor
    bar_i: torch.Tensor
    bar_z: torch.Tensor
    production_targets: torch.Tensor
    production_scales: torch.Tensor
    m_source: str
    regime: torch.Tensor

    def digest(self) -> str:
        return _tensor_digest([
            ("parent", self.parent),
            ("child_states", self.child_states),
            ("m", self.m),
            ("bar_i", self.bar_i),
            ("bar_z", self.bar_z),
            ("production_targets", self.production_targets),
            ("production_scales", self.production_scales),
            ("regime", self.regime),
        ])


def _claim(model: torch.nn.Module, state: torch.Tensor) -> torch.Tensor:
    fn = getattr(model, "_q_claim_output", None)
    return fn(state) if callable(fn) else model._q_output(state)


def _q_scope(episode: Any, model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[str]]:
    original = episode.models["policy_value"]
    episode.models["policy_value"] = model
    try:
        params = episode._policy_value_stage_params("q")
    finally:
        episode.models["policy_value"] = original
    allowed = {id(param) for param in params}
    names: list[str] = []
    for name, param in model.named_parameters():
        param.requires_grad_(id(param) in allowed)
        if id(param) in allowed:
            names.append(name)
    return params, names


def _non_q_hash(model: torch.nn.Module) -> str:
    return named_parameter_hash(model, tuple(P_PREFIXES) + tuple(BP_PREFIXES))


def _stats(value: torch.Tensor, prefix: str) -> dict[str, float]:
    flat = value.detach().reshape(-1).to(torch.float64)
    finite = flat[torch.isfinite(flat)]
    absolute = finite.abs()
    return {
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_mae": float(absolute.mean()),
        f"{prefix}_p50": float(torch.quantile(absolute, 0.50)),
        f"{prefix}_p90": float(torch.quantile(absolute, 0.90)),
        f"{prefix}_p99": float(torch.quantile(absolute, 0.99)),
        f"{prefix}_max": float(absolute.max()),
    }


def _used_m(
    episode: Any,
    children: list[torch.Tensor],
    mode: str,
) -> tuple[list[torch.Tensor], str]:
    if mode == "legacy_m1":
        return [torch.ones_like(child[:, 0:1]) for child in children], "LEGACY_M1"
    if mode not in {"corrected_child", "current_sdf_child"}:
        raise ValueError(f"unknown M mode: {mode!r}")
    if any(child.shape[1] <= 7 for child in children):
        raise ValueError(f"{mode} M requires child transition column 7")
    if mode == "current_sdf_child":
        return [child[:, 7:8].detach() for child in children], "CURRENT_SDF_CHILD"
    lo = float(getattr(episode.hyperparams, "q_m_clamp_min", 0.5))
    hi = float(getattr(episode.hyperparams, "q_m_clamp_max", 1.5))
    return [child[:, 7:8].clamp(lo, hi).detach() for child in children], (
        f"CORRECTED_CHILD_CLAMPED_{lo:g}_{hi:g}"
    )


def q_dynamic_target_components(
    model: torch.nn.Module,
    episode: Any,
    batch: QSurvivalBatch,
    *,
    detach_child: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_parent, n_child, state_dim = batch.child_states.shape
    child_q = _claim(model, batch.child_states.reshape(-1, state_dim)).reshape(
        n_parent, n_child, 1
    )
    if detach_child:
        child_q = child_q.detach()
    components = episode.loss_fns["q"].compute_main_components(
        _claim(model, batch.parent),
        batch.parent[:, 0:1],
        batch.bar_i,
        list(batch.m.unbind(dim=1)),
        list(child_q.unbind(dim=1)),
        list(batch.bar_z.unbind(dim=1)),
        list(batch.child_states[:, :, 4:5].unbind(dim=1)),
        list(batch.child_states[:, :, 1:2].unbind(dim=1)),
    )
    targets = episode._collapse_policy_eta_pairs(
        [component["q_target_total"] for component in components]
    )
    branches = torch.stack(targets, dim=1)
    target = branches.mean(dim=1)
    scale = (1.0 + branches.abs().mean(dim=1)).detach().clamp_min(1e-8)
    if batch.regime.numel():
        regime = batch.regime.reshape(-1)
        zero = regime == 0
        default = regime == 1
        if zero.any():
            target = target.clone()
            scale = scale.clone()
            branches = branches.clone()
            target[zero] = 0.0
            scale[zero] = 1.0
            branches[zero] = 0.0
        if default.any():
            if target.data_ptr() == branches.data_ptr():
                target = target.clone()
            target_default = episode.loss_fns["q"].compute_current_recovery(
                batch.parent[default, 0:1],
                batch.parent[default, 4:5],
                batch.parent[default, 1:2],
            )
            target[default] = target_default
            scale = scale.clone()
            scale[default] = 1.0 + target_default.abs()
            branches = branches.clone()
            branches[default] = target_default.unsqueeze(1)
    return target, scale, branches


def q_dynamic_targets(
    model: torch.nn.Module,
    episode: Any,
    batch: QSurvivalBatch,
    *,
    detach_child: bool,
) -> torch.Tensor:
    return q_dynamic_target_components(
        model, episode, batch, detach_child=detach_child
    )[0]


def q_residual_tensors(
    model: torch.nn.Module,
    episode: Any,
    batch: QSurvivalBatch,
    *,
    frozen_target: bool = False,
    detach_child: bool = False,
) -> dict[str, torch.Tensor]:
    parent_q = _claim(model, batch.parent)
    if frozen_target:
        target, scale = batch.production_targets, batch.production_scales
    else:
        target, scale, _ = q_dynamic_target_components(
            model, episode, batch, detach_child=detach_child
        )
    residual = target - parent_q
    normalized = residual / scale
    return {
        "residual": residual,
        "normalized": normalized,
        "target": target,
        "parent_q": parent_q,
        "scale": scale,
    }


def q_root_loss(
    model: torch.nn.Module,
    episode: Any,
    batch: QSurvivalBatch,
    *,
    frozen_target: bool = False,
    detach_child: bool = False,
) -> torch.Tensor:
    return q_residual_tensors(
        model,
        episode,
        batch,
        frozen_target=frozen_target,
        detach_child=detach_child,
    )["normalized"].square().mean()


def refresh_q_targets(
    model: torch.nn.Module,
    episode: Any,
    problem: Sequence[QSurvivalBatch],
) -> list[QSurvivalBatch]:
    with torch.no_grad():
        refreshed = []
        for batch in problem:
            target, scale, _ = q_dynamic_target_components(
                model, episode, batch, detach_child=True
            )
            refreshed.append(replace(
                batch,
                production_targets=target.detach(),
                production_scales=scale.detach(),
            ))
        return refreshed


def evaluate_q_problem(
    model: torch.nn.Module,
    episode: Any,
    problem: Sequence[QSurvivalBatch],
    *,
    frozen_target: bool = False,
) -> dict[str, float]:
    residuals, normalized, parent_q, gains, regimes = [], [], [], [], []
    with torch.no_grad():
        for batch in problem:
            values = q_residual_tensors(
                model, episode, batch, frozen_target=frozen_target
            )
            residuals.append(values["residual"].cpu())
            normalized.append(values["normalized"].cpu())
            parent_q.append(values["parent_q"].cpu())
            multiplier = batch.bar_i * (float(episode.loss_fns["q"].g) - 1.0) + 1.0
            survival = batch.regime.reshape(-1) == 2
            if survival.any():
                gains.append((
                    batch.m[survival]
                    * multiplier[survival].unsqueeze(1)
                    * (1.0 - batch.bar_z[survival])
                ).cpu())
            regimes.append(batch.regime.cpu())
    raw = torch.cat(residuals)
    norm = torch.cat(normalized)
    q = torch.cat(parent_q)
    gain = torch.cat(gains) if gains else torch.empty(0)
    regime = torch.cat(regimes).reshape(-1)
    result = {}
    result.update(_stats(raw, "raw_residual"))
    result.update(_stats(norm, "normalized_residual"))
    result.update(_stats(q, "q"))
    if gain.numel():
        result.update(_stats(gain, "recursion_gain"))
    for index, name in enumerate(("zero", "default", "survival")):
        mask = regime == index
        if mask.any():
            result.update(_stats(norm[mask], f"{name}_normalized_residual"))
            result[f"{name}_count"] = int(mask.sum())
    result["normalized_mae"] = result["normalized_residual_mae"]
    result["finite"] = bool(torch.isfinite(raw).all() and torch.isfinite(q).all())
    return result


def build_q_survival_problem(
    *,
    episode: Any,
    initial_model: torch.nn.Module,
    frozen_p_model: torch.nn.Module,
    parents: torch.Tensor,
    children: torch.Tensor,
    batch_size: int,
    m_mode: str,
    include_all_regimes: bool = False,
) -> tuple[list[QSurvivalBatch], dict[str, int]]:
    problem: list[QSurvivalBatch] = []
    counts = {"zero": 0, "default": 0, "survival": 0, "unclassified": 0}
    for start in range(0, parents.shape[0], batch_size):
        stop = min(start + batch_size, parents.shape[0])
        parent = parents[start:stop, :7]
        raw_children = children[start:stop]
        with torch.no_grad():
            phat = frozen_p_model.forward_equity(parent)["Phat"]
            masks = classify_q_parent_regimes(parent[:, 0:1], phat)
        for name in counts:
            counts[name] += int(masks[name].sum())
        regime = torch.full(
            (parent.shape[0],), -1, device=parent.device, dtype=torch.long
        )
        regime[masks["zero"].reshape(-1)] = 0
        regime[masks["default"].reshape(-1)] = 1
        regime[masks["survival"].reshape(-1)] = 2
        selected = regime >= 0 if include_all_regimes else regime == 2
        if not bool(selected.any()):
            continue
        parent = parent[selected]
        raw_children = raw_children[selected]
        regime = regime[selected]
        child_list = list(raw_children.unbind(dim=1))
        used_m, m_source = _used_m(episode, child_list, m_mode)
        expanded_children, _, expanded_m, _ = episode._expand_policy_expectation_children(
            child_list,
            used_m,
            used_m,
        )
        child_state = torch.stack([value[:, :7] for value in expanded_children], dim=1)
        m = torch.stack(expanded_m, dim=1).detach()
        with torch.no_grad():
            parent_out = frozen_p_model(parent)
            bar_i = parent_out.bar_i.detach()
            if episode._pv_use_fixed_policy():
                bar_i = torch.zeros_like(bar_i)
            multiplier = bar_i * (float(episode.loss_fns["q"].g) - 1.0) + 1.0
            b_sp = parent[:, 0:1] / multiplier.clamp_min(1e-6)
            child_state = child_state.clone()
            child_state[:, :, 0:1] = b_sp.unsqueeze(1)
            flat = child_state.reshape(-1, child_state.shape[-1])
            bar_z = frozen_p_model.forward_equity(flat)["bar_z"].reshape(
                child_state.shape[0], child_state.shape[1], 1
            ).detach()
        shell = QSurvivalBatch(
            parent=parent.detach(),
            child_states=child_state.detach(),
            m=m,
            bar_i=bar_i,
            bar_z=bar_z,
            production_targets=torch.empty(0, device=parent.device),
            production_scales=torch.empty(0, device=parent.device),
            m_source=m_source,
            regime=regime.detach(),
        )
        with torch.no_grad():
            target, scale, _ = q_dynamic_target_components(
                initial_model, episode, shell, detach_child=True
            )
        problem.append(replace(
            shell,
            production_targets=target.detach(),
            production_scales=scale.detach(),
        ))
    if not problem:
        raise RuntimeError("Q survival problem has no surviving positive-debt parents")
    return problem, counts


def initial_q_target_parity(
    model: torch.nn.Module,
    episode: Any,
    batch: QSurvivalBatch,
    production_target: torch.Tensor,
    production_scale: torch.Tensor,
) -> dict[str, float | bool]:
    with torch.no_grad():
        target, scale, _ = q_dynamic_target_components(
            model, episode, batch, detach_child=True
        )
    error = (target - production_target).abs().cpu().reshape(-1)
    scale_error = (scale - production_scale).abs().cpu().reshape(-1)
    return {
        "mean_abs_error": float(error.mean()),
        "p90_abs_error": float(torch.quantile(error, 0.90)),
        "max_abs_error": float(error.max()),
        "scale_mean_abs_error": float(scale_error.mean()),
        "scale_max_abs_error": float(scale_error.max()),
        "passed": bool(float(error.max()) <= 1e-5 and float(scale_error.max()) <= 1e-5),
    }


def _grad_norm(params: Sequence[torch.nn.Parameter]) -> float:
    return math.sqrt(sum(
        float(param.grad.detach().square().sum())
        for param in params if param.grad is not None
    ))


def _param_norm(params: Sequence[torch.nn.Parameter]) -> float:
    return math.sqrt(sum(float(param.detach().square().sum()) for param in params))


def run_q_adam(
    *,
    model: torch.nn.Module,
    episode: Any,
    train: Sequence[QSurvivalBatch],
    validation: Sequence[QSurvivalBatch],
    steps: int,
    lr: float,
    frozen_target: bool,
    eval_every: int,
    phase: str,
) -> list[dict[str, Any]]:
    params, _ = _q_scope(episode, model)
    optimizer = torch.optim.Adam(params, lr=lr)
    history = []
    for step in range(1, steps + 1):
        batch = train[(step - 1) % len(train)]
        optimizer.zero_grad(set_to_none=True)
        loss = q_root_loss(
            model, episode, batch, frozen_target=frozen_target
        )
        loss.backward()
        raw_grad = _grad_norm(params)
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        clipped_grad = _grad_norm(params)
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == steps:
            val = evaluate_q_problem(model, episode, validation)
            history.append({
                "phase": phase,
                "step": step,
                "loss": float(loss.detach()),
                "raw_grad_norm": raw_grad,
                "clipped_grad_norm": clipped_grad,
                "parameter_norm": _param_norm(params),
                **{f"val_{key}": value for key, value in val.items()},
            })
    return history


def run_q_lbfgs(
    *,
    model: torch.nn.Module,
    episode: Any,
    train: Sequence[QSurvivalBatch],
    validation: Sequence[QSurvivalBatch],
    max_iter: int,
    chunk_iterations: int,
    history_size: int,
    target_residual: Optional[float],
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, float], str]:
    params, _ = _q_scope(episode, model)
    best = evaluate_q_problem(model, episode, validation)
    best_state = _clone_state(model)
    history: list[dict[str, Any]] = []
    budget = 0
    total_calls = 0
    stop_reason = "optimizer_limit"
    while budget < max_iter:
        block = min(chunk_iterations, max_iter - budget)
        optimizer = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=block,
            history_size=history_size,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
        )
        calls = 0
        started = time.perf_counter()

        def closure() -> torch.Tensor:
            nonlocal calls, total_calls
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=train[0].parent.device)
            for batch in train:
                loss = q_root_loss(model, episode, batch) / len(train)
                loss.backward()
                total = total + loss.detach()
            calls += 1
            total_calls += 1
            return total

        optimizer.step(closure)
        if train[0].parent.is_cuda:
            torch.cuda.synchronize(train[0].parent.device)
        elapsed = time.perf_counter() - started
        budget += block
        val = evaluate_q_problem(model, episode, validation)
        history.append({
            "phase": "root_lbfgs",
            "step": budget,
            "closure_calls": calls,
            "total_closure_calls": total_calls,
            "seconds": elapsed,
            "seconds_per_closure": elapsed / max(calls, 1),
            **{f"val_{key}": value for key, value in val.items()},
        })
        if val["normalized_mae"] < best["normalized_mae"]:
            best, best_state = val, _clone_state(model)
        if not bool(val["finite"]):
            stop_reason = "nonfinite_validation"
            break
        if target_residual is not None and val["normalized_mae"] <= target_residual:
            stop_reason = "target_residual_reached"
            break
    return history, best_state, best, stop_reason


def run_q_picard(
    *,
    model: torch.nn.Module,
    episode: Any,
    train: Sequence[QSurvivalBatch],
    validation: Sequence[QSurvivalBatch],
    outer_steps: int,
    fit_steps: int,
    lr: float,
) -> tuple[list[dict[str, Any]], torch.nn.Module]:
    rows = []
    for outer in range(1, outer_steps + 1):
        fixed_train = refresh_q_targets(model, episode, train)
        fixed_validation = refresh_q_targets(model, episode, validation)
        started = time.perf_counter()
        run_q_adam(
            model=model,
            episode=episode,
            train=fixed_train,
            validation=fixed_validation,
            steps=fit_steps,
            lr=lr,
            frozen_target=True,
            eval_every=fit_steps,
            phase="frozen_target_fit",
        )
        fresh = evaluate_q_problem(model, episode, validation)
        fit = evaluate_q_problem(
            model, episode, fixed_validation, frozen_target=True
        )
        rows.append({
            "outer": outer,
            "fresh_residual": fresh["normalized_mae"],
            "frozen_fit_residual": fit["normalized_mae"],
            "q_mean": fresh["q_mean"],
            "q_p90": fresh["q_p90"],
            "q_max": fresh["q_max"],
            "seconds": time.perf_counter() - started,
        })
        if not bool(fresh["finite"]):
            break
    return rows, model


def _plot(
    history: pd.DataFrame,
    picard: pd.DataFrame,
    output: Path,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if not history.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for phase, frame in history.groupby("phase"):
            ax.semilogy(frame["step"], frame["val_normalized_mae"], marker="o", label=phase)
        ax.set(xlabel="optimizer step/budget", ylabel="fresh normalized Q residual")
        ax.legend(); fig.tight_layout(); fig.savefig(figures / "q_root_residual.png", dpi=180); plt.close(fig)
    if not picard.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(picard["outer"], picard["fresh_residual"], marker="o", label="fresh")
        ax.semilogy(picard["outer"], picard["frozen_fit_residual"], marker="o", label="frozen fit")
        ax.set(xlabel="frozen-target outer", ylabel="normalized Q residual")
        ax.legend(); fig.tight_layout(); fig.savefig(figures / "q_picard_residual.png", dpi=180); plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--p-checkpoint", type=Path)
    parser.add_argument("--sweep", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--m-mode", choices=["corrected_child", "legacy_m1"], default="corrected_child")
    parser.add_argument("--q-regime", choices=["survival", "full"], default="survival")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--train-limit", type=int, default=8192)
    parser.add_argument("--validation-limit", type=int, default=2048)
    parser.add_argument("--picard-outer", type=int, default=5)
    parser.add_argument("--picard-fit-steps", type=int, default=100)
    parser.add_argument("--adam-steps", type=int, default=100)
    parser.add_argument("--adam-lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--lbfgs-max-iter", type=int, default=200)
    parser.add_argument("--lbfgs-chunk-iterations", type=int, default=20)
    parser.add_argument("--lbfgs-history-size", type=int, default=50)
    parser.add_argument("--target-residual", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(20261016)
    device = torch.device(args.device)
    output = args.output_dir.expanduser().resolve()
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    run = args.run.expanduser().resolve()
    episode, _, models, _, _, _, _, provenance, _ = _load_setup(
        run, device, None, args.source_run, args.source_checkpoint
    )
    model = models["policy_value"]
    sweep_checkpoint = run / "checkpoints" / f"sweep_{args.sweep:03d}.pt"
    payload = _load_policy_state(model, sweep_checkpoint, device)
    if args.p_checkpoint is not None:
        p_payload = torch.load(args.p_checkpoint, map_location=device, weights_only=False)
        _restore_state(model, p_payload["policy_value"])
    initial_model = deepcopy(model).to(device)
    frozen_p = deepcopy(model).to(device)
    frozen_p.eval()
    for parameter in frozen_p.parameters():
        parameter.requires_grad_(False)
    train_payload = torch.load(run / "fixed_point/collocation_train.pt", map_location="cpu", weights_only=False)
    validation_payload = torch.load(run / "fixed_point/collocation_val.pt", map_location="cpu", weights_only=False)
    child_payload = torch.load(run / "fixed_point/child_shock_bank.pt", map_location="cpu", weights_only=False)
    n_train = min(args.train_limit, train_payload["parent_states"].shape[0])
    n_validation = min(args.validation_limit, validation_payload["parent_states"].shape[0])
    train, train_regimes = build_q_survival_problem(
        episode=episode,
        initial_model=initial_model,
        frozen_p_model=frozen_p,
        parents=train_payload["parent_states"][:n_train].to(device),
        children=child_payload["train"]["children"][:n_train].to(device),
        batch_size=args.batch_size,
        m_mode=args.m_mode,
        include_all_regimes=args.q_regime == "full",
    )
    validation, validation_regimes = build_q_survival_problem(
        episode=episode,
        initial_model=initial_model,
        frozen_p_model=frozen_p,
        parents=validation_payload["parent_states"][:n_validation].to(device),
        children=child_payload["validation"]["children"][:n_validation].to(device),
        batch_size=args.batch_size,
        m_mode=args.m_mode,
        include_all_regimes=args.q_regime == "full",
    )
    # True cross-implementation parity: compare the new differentiable RHS
    # against the production operator snapshot, not against a target generated
    # by this diagnostic itself.  Keep this batch within BPGridTeacher's hard
    # cap because the production snapshot also evaluates P/BP targets.
    parity_n = min(512, n_validation)
    parity_problem, parity_regimes = build_q_survival_problem(
        episode=episode,
        initial_model=initial_model,
        frozen_p_model=frozen_p,
        parents=validation_payload["parent_states"][:parity_n].to(device),
        children=child_payload["validation"]["children"][:parity_n].to(device),
        batch_size=parity_n,
        m_mode=args.m_mode,
        include_all_regimes=args.q_regime == "full",
    )
    production_batch = {
        "parent": validation_payload["parent_states"][:parity_n].to(device),
        "children": list(
            child_payload["validation"]["children"][:parity_n].to(device).unbind(dim=1)
        ),
    }
    _, _, production_snapshot = evaluate_inner_operator_snapshot(
        episode, [production_batch], model=initial_model
    )
    production_selected = (
        torch.ones_like(production_snapshot["q_regime"], dtype=torch.bool)
        if args.q_regime == "full"
        else production_snapshot["q_regime"] == 2
    )
    if int(production_selected.sum()) != parity_problem[0].parent.shape[0]:
        raise RuntimeError(
            "Q production parity row count mismatch: "
            f"production={int(production_selected.sum())}, "
            f"diagnostic={parity_problem[0].parent.shape[0]}"
        )
    parity = initial_q_target_parity(
        initial_model,
        episode,
        parity_problem[0],
        production_snapshot["q_target"][production_selected].to(device),
        production_snapshot["q_scale"][production_selected].to(device),
    )
    parity["production_parity_rows"] = int(production_selected.sum())
    parity["production_parity_regime_counts"] = str(parity_regimes)
    pd.DataFrame([parity]).to_csv(output / "q_initial_parity.csv", index=False)
    if not bool(parity["passed"]):
        raise RuntimeError(f"Q initial target parity failed: {parity}")

    params, trainable_names = _q_scope(episode, initial_model)
    probe = q_dynamic_targets(initial_model, episode, train[0], detach_child=False)
    probe_grad = torch.autograd.grad(probe.sum(), params, allow_unused=True)
    child_gradient_nonzero = any(
        grad is not None and bool((grad.abs() > 0).any()) for grad in probe_grad
    )
    detached_probe = q_dynamic_targets(
        initial_model, episode, train[0], detach_child=True
    )
    detached_child_gradient = detached_probe.requires_grad
    if not child_gradient_nonzero or detached_child_gradient:
        raise RuntimeError("Q child gradient-path invariant failed")

    initial = evaluate_q_problem(initial_model, episode, validation)
    non_q_before = _non_q_hash(initial_model)
    q_before = _state_hash(initial_model, Q_PREFIXES)
    picard_model = deepcopy(initial_model).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    picard_rows, picard_model = run_q_picard(
        model=picard_model,
        episode=episode,
        train=train,
        validation=validation,
        outer_steps=args.picard_outer,
        fit_steps=args.picard_fit_steps,
        lr=args.adam_lr,
    )
    picard_final = evaluate_q_problem(picard_model, episode, validation)
    picard_peak_vram = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        if device.type == "cuda" else float("nan")
    )

    root_model = deepcopy(initial_model).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    adam_history = run_q_adam(
        model=root_model,
        episode=episode,
        train=train,
        validation=validation,
        steps=args.adam_steps,
        lr=args.adam_lr,
        frozen_target=False,
        eval_every=args.eval_every,
        phase="root_adam",
    )
    adam_final = evaluate_q_problem(root_model, episode, validation)
    lbfgs_history, best_state, best, stop_reason = run_q_lbfgs(
        model=root_model,
        episode=episode,
        train=train,
        validation=validation,
        max_iter=args.lbfgs_max_iter,
        chunk_iterations=args.lbfgs_chunk_iterations,
        history_size=args.lbfgs_history_size,
        target_residual=args.target_residual,
    )
    _restore_state(root_model, best_state)
    root_final = evaluate_q_problem(root_model, episode, validation)
    elapsed = time.perf_counter() - started
    root_peak_vram = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        if device.type == "cuda" else float("nan")
    )
    if _non_q_hash(root_model) != non_q_before:
        raise RuntimeError("Q root solver changed P/BP parameters")
    if _state_hash(root_model, Q_PREFIXES) == q_before:
        raise RuntimeError("Q root solver did not update Q parameters")

    history = pd.DataFrame(adam_history + lbfgs_history)
    picard = pd.DataFrame(picard_rows)
    summary = pd.DataFrame([
        {
            "method": "Frozen-target Picard",
            "initial_residual": initial["normalized_mae"],
            "final_residual": picard_final["normalized_mae"],
            "best_residual": min([initial["normalized_mae"]] + [row["fresh_residual"] for row in picard_rows]),
            "q_mean": picard_final["q_mean"],
            "q_p90": picard_final["q_p90"],
            "q_max": picard_final["q_max"],
            "seconds": sum(row["seconds"] for row in picard_rows),
            "peak_vram_mib": picard_peak_vram,
            "status": "completed",
        },
        {
            "method": "Full-residual Adam+LBFGS",
            "initial_residual": initial["normalized_mae"],
            "adam_residual": adam_final["normalized_mae"],
            "final_residual": root_final["normalized_mae"],
            "best_residual": best["normalized_mae"],
            "q_mean": root_final["q_mean"],
            "q_p90": root_final["q_p90"],
            "q_max": root_final["q_max"],
            "seconds": elapsed,
            "peak_vram_mib": root_peak_vram,
            "status": stop_reason,
        },
    ])
    history.to_csv(output / "q_root_history.csv", index=False)
    picard.to_csv(output / "q_picard_history.csv", index=False)
    summary.to_csv(output / "q_root_summary.csv", index=False)
    torch.save({"policy_value": _clone_state(root_model)}, checkpoints / "q_root_best.pt")
    m_all = torch.cat([batch.m.reshape(-1).cpu() for batch in train + validation])
    metadata: Mapping[str, Any] = {
        "args": vars(args),
        "source_git_commit": provenance.get("source_git_commit"),
        "sweep_checkpoint": str(sweep_checkpoint),
        "sweep_checkpoint_stage": payload.get("stage"),
        "p_checkpoint": str(args.p_checkpoint) if args.p_checkpoint else None,
        "train_regime_counts": train_regimes,
        "validation_regime_counts": validation_regimes,
        "trainable_parameter_names": trainable_names,
        "child_q_gradient_nonzero": child_gradient_nonzero,
        "detached_child_target_requires_grad": detached_child_gradient,
        "non_q_hash_before": non_q_before,
        "non_q_hash_after": _non_q_hash(root_model),
        "q_hash_before": q_before,
        "q_hash_after": _state_hash(root_model, Q_PREFIXES),
        "m_source": train[0].m_source,
        "q_regime": args.q_regime,
        "m_min": float(m_all.min()),
        "m_mean": float(m_all.mean()),
        "m_max": float(m_all.max()),
        "m_hash": sha256(m_all.contiguous().numpy().tobytes()).hexdigest(),
        "train_problem_hash": _tensor_digest((f"train.{i}", torch.tensor(list(bytes.fromhex(batch.digest())), dtype=torch.uint8)) for i, batch in enumerate(train)),
        "validation_problem_hash": _tensor_digest((f"validation.{i}", torch.tensor(list(bytes.fromhex(batch.digest())), dtype=torch.uint8)) for i, batch in enumerate(validation)),
    }
    (output / "q_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str, sort_keys=True), encoding="utf-8"
    )
    _plot(history, picard, output)
    print(summary.to_string(index=False))
    print(json.dumps({key: metadata[key] for key in ("m_source", "m_min", "m_mean", "m_max", "m_hash")}, indent=2))
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
