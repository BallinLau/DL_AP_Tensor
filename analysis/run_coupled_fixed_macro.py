#!/usr/bin/env python3
"""Bounded fixed-macro P-Q block Gauss-Seidel diagnostic."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.run_neural_policy_iteration import (  # noqa: E402
    PolicyView,
    _i_grid_values,
    _solve_frozen_policy,
    _state_hash,
    _tensor_digest,
    build_policy_view,
    detect_two_cycle,
    policy_transition_arrays,
    policy_transition_metrics,
    value_magnitude_metrics,
)
from analysis.run_q_root_solve_fixed_macro import (  # noqa: E402
    BP_PREFIXES,
    P_PREFIXES,
    Q_PREFIXES,
    _clone_state,
    _load_policy_state,
    _load_setup,
    _non_q_hash,
    _restore_state,
    build_q_survival_problem,
    evaluate_q_problem,
    run_q_adam,
    run_q_lbfgs,
)
from analysis.test_frozen_policy_root_solve import (  # noqa: E402
    build_problem,
    evaluate_problem,
)
from config.constants import SIMMODEL  # noqa: E402


@dataclass(frozen=True)
class CoupledView:
    policy: PolicyView
    q: torch.Tensor
    phat: torch.Tensor

    def digest(self) -> str:
        return _tensor_digest([
            ("policy", torch.tensor(list(bytes.fromhex(self.policy.digest())), dtype=torch.uint8)),
            ("q", self.q),
            ("phat", self.phat),
        ])


def _abs_stats(value: torch.Tensor, prefix: str) -> dict[str, float]:
    flat = value.detach().reshape(-1).to(torch.float64)
    absolute = flat[torch.isfinite(flat)].abs()
    return {
        f"{prefix}_mean": float(absolute.mean()),
        f"{prefix}_p50": float(torch.quantile(absolute, 0.50)),
        f"{prefix}_p90": float(torch.quantile(absolute, 0.90)),
        f"{prefix}_p99": float(torch.quantile(absolute, 0.99)),
        f"{prefix}_max": float(absolute.max()),
    }


def build_coupled_view(model: torch.nn.Module, problem: Sequence[Any]) -> CoupledView:
    policy = build_policy_view(model, problem)
    with torch.no_grad():
        q = model._q_claim_output(policy.parent)
        p0_grid, pi_grid = _i_grid_values(model, policy.parent.unsqueeze(1))
        phat = torch.maximum(p0_grid, pi_grid).mean(dim=0)
    return CoupledView(policy=policy, q=q.detach(), phat=phat.detach())


def coupled_transition_metrics(old: CoupledView, new: CoupledView) -> dict[str, float]:
    result = policy_transition_metrics(old.policy, new.policy)
    result.update(_abs_stats(
        (new.policy.p0 - old.policy.p0) / old.policy.scale,
        "p0_function_change",
    ))
    result.update(_abs_stats(
        (new.policy.pi - old.policy.pi) / old.policy.scale,
        "pi_function_change",
    ))
    result.update(_abs_stats((new.q - old.q) / (1.0 + old.q.abs()), "q_movement"))
    result.update(_abs_stats((new.phat - old.phat) / old.policy.scale, "phat_movement"))
    return result


def detect_coupled_two_cycle(
    current: CoupledView,
    previous: CoupledView,
    two_back: CoupledView,
) -> dict[str, Any]:
    policy = detect_two_cycle(current.policy, previous.policy, two_back.policy)
    one = coupled_transition_metrics(previous, current)
    two = coupled_transition_metrics(two_back, current)
    q_cycle = two["q_movement_p90"] < 0.01 and one["q_movement_p90"] > 0.02
    p_cycle = (
        two["p0_function_change_p90"] < 0.01
        and two["pi_function_change_p90"] < 0.01
        and (
            one["p0_function_change_p90"] > 0.02
            or one["pi_function_change_p90"] > 0.02
        )
    )
    return {
        **policy,
        "q_two_cycle": bool(q_cycle),
        "p_function_two_cycle": bool(p_cycle),
        "possible_two_cycle": bool(policy["possible_two_cycle"] or q_cycle or p_cycle),
        "one_step_q_p90": one["q_movement_p90"],
        "two_step_q_p90": two["q_movement_p90"],
    }


def assert_block_invariants(
    *,
    block: str,
    frozen_before: str,
    frozen_after: str,
    macro_before: str,
    macro_after: str,
    parent_before: str,
    parent_after: str,
    child_before: str,
    child_after: str,
) -> None:
    checks = {
        f"{block} frozen-parameter hash": (frozen_before, frozen_after),
        "macro hash": (macro_before, macro_after),
        "parent bank hash": (parent_before, parent_after),
        "child bank hash": (child_before, child_after),
    }
    changed = {name: pair for name, pair in checks.items() if pair[0] != pair[1]}
    if changed:
        raise RuntimeError(f"coupled block invariant changed: {changed}")


def bounded_coupled_loop(initial: Any, outer_max: int, p_step, q_step) -> list[Any]:
    """Minimal testable warm-start/max-outer skeleton used by the real pilot."""
    if not 1 <= outer_max <= 5:
        raise ValueError("coupled outer max must be in [1, 5]")
    current = initial
    states = [current]
    for outer in range(1, outer_max + 1):
        after_p = p_step(current, outer)
        current = q_step(after_p, outer)
        states.append(current)
    return states


def _models_hash(models: Mapping[str, torch.nn.Module]) -> str:
    entries = []
    for model_name, model in sorted(models.items()):
        if model_name == "policy_value":
            continue
        for name, tensor in sorted(model.state_dict().items()):
            entries.append((f"{model_name}.{name}", tensor))
    return _tensor_digest(entries)


def _build_p_problem(
    episode, model, parents, children, batch_size, *, require_legacy_m1=True
):
    return build_problem(
        episode, model, parents, children, batch_size,
        snapshot_batch_size=min(512, batch_size),
        require_legacy_m1=require_legacy_m1,
    )


def solve_p_block(
    *, model, episode, train_parents, train_children, validation_parents,
    validation_children, batch_size, policy_outer_max, adam_steps, adam_lr,
    eval_every, lbfgs_max_iter, lbfgs_chunk_iterations, target_residual,
    require_legacy_m1=True,
) -> tuple[torch.nn.Module, list[dict[str, Any]], Sequence[Any], Sequence[Any]]:
    current = model
    rows = []
    final_train = final_validation = None
    for policy_outer in range(policy_outer_max):
        train = _build_p_problem(
            episode, current, train_parents, train_children, batch_size,
            require_legacy_m1=require_legacy_m1,
        )
        validation = _build_p_problem(
            episode, current, validation_parents, validation_children, batch_size,
            require_legacy_m1=require_legacy_m1,
        )
        view_before = build_policy_view(current, validation)
        started = time.perf_counter()
        solved, history, info = _solve_frozen_policy(
            current, episode, train, validation,
            adam_steps=adam_steps, adam_lr=adam_lr, eval_every=eval_every,
            lbfgs_max_iter=lbfgs_max_iter, lbfgs_history_size=50,
            lbfgs_chunk_iterations=lbfgs_chunk_iterations,
            target_residual=target_residual,
        )
        final_train = _build_p_problem(
            episode, solved, train_parents, train_children, batch_size,
            require_legacy_m1=require_legacy_m1,
        )
        final_validation = _build_p_problem(
            episode, solved, validation_parents, validation_children, batch_size,
            require_legacy_m1=require_legacy_m1,
        )
        greedy = evaluate_problem(solved, final_validation)
        view_after = build_policy_view(solved, final_validation)
        movement = policy_transition_metrics(view_before, view_after)
        rows.append({
            "policy_outer": policy_outer,
            "initial_frozen_residual": info["initial"]["normalized_mae"],
            "final_frozen_residual": info["final"]["normalized_mae"],
            "full_greedy_residual": greedy["normalized_mae"],
            "optimizer_rows": len(history),
            "stop_reason": info["stop_reason"],
            "seconds": time.perf_counter() - started,
            **movement,
        })
        current = solved
        if greedy["normalized_mae"] <= target_residual:
            break
    assert final_train is not None and final_validation is not None
    return current, rows, final_train, final_validation


def solve_q_block(
    *, model, episode, train_parents, train_children, validation_parents,
    validation_children, batch_size, adam_steps, adam_lr, eval_every,
    lbfgs_max_iter, lbfgs_chunk_iterations, target_residual,
    m_mode="corrected_child",
) -> tuple[torch.nn.Module, dict[str, Any], Sequence[Any], Sequence[Any]]:
    started = time.perf_counter()
    frozen_p = deepcopy(model).to(train_parents.device)
    for parameter in frozen_p.parameters():
        parameter.requires_grad_(False)
    train, train_regimes = build_q_survival_problem(
        episode=episode, initial_model=model, frozen_p_model=frozen_p,
        parents=train_parents, children=train_children, batch_size=batch_size,
        m_mode=m_mode, include_all_regimes=True,
    )
    validation, validation_regimes = build_q_survival_problem(
        episode=episode, initial_model=model, frozen_p_model=frozen_p,
        parents=validation_parents, children=validation_children,
        batch_size=batch_size, m_mode=m_mode, include_all_regimes=True,
    )
    root = deepcopy(model).to(train_parents.device)
    initial = evaluate_q_problem(root, episode, validation)
    adam_history = run_q_adam(
        model=root, episode=episode, train=train, validation=validation,
        steps=adam_steps, lr=adam_lr, frozen_target=False,
        eval_every=eval_every, phase="root_adam",
    )
    adam = evaluate_q_problem(root, episode, validation)
    lbfgs_history, best_state, best, stop_reason = run_q_lbfgs(
        model=root, episode=episode, train=train, validation=validation,
        max_iter=lbfgs_max_iter, chunk_iterations=lbfgs_chunk_iterations,
        history_size=50, target_residual=target_residual,
    )
    _restore_state(root, best_state)
    final = evaluate_q_problem(root, episode, validation)
    return root, {
        "initial": initial,
        "adam": adam,
        "best": best,
        "final": final,
        "stop_reason": stop_reason,
        "seconds": time.perf_counter() - started,
        "adam_history": adam_history,
        "lbfgs_history": lbfgs_history,
        "train_regimes": train_regimes,
        "validation_regimes": validation_regimes,
    }, train, validation


def _binned_mean(x, y, value, bins=20):
    x_edges = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)
    y_edges = np.linspace(np.nanmin(y), np.nanmax(y), bins + 1)
    total, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges), weights=value)
    count, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    return mean.T, x_edges, y_edges


def _plot_map(value, view, title, path):
    parents = view.policy.parent.detach().cpu().numpy()
    b = parents[:, int(SIMMODEL.B)]
    z = parents[:, int(SIMMODEL.Z)]
    eta = parents[:, int(SIMMODEL.ETA)]
    for eta_value in (0, 1):
        mask = np.isclose(eta, eta_value)
        if not mask.any():
            continue
        grid, b_edges, z_edges = _binned_mean(b[mask], z[mask], value[mask])
        bc = 0.5 * (b_edges[:-1] + b_edges[1:])
        zc = 0.5 * (z_edges[:-1] + z_edges[1:])
        bb, zz = np.meshgrid(bc, zc)
        fig = plt.figure(figsize=(12, 5))
        ax3 = fig.add_subplot(1, 2, 1, projection="3d")
        axh = fig.add_subplot(1, 2, 2)
        ax3.plot_surface(bb, zz, grid, cmap="viridis", linewidth=0)
        image = axh.pcolormesh(b_edges, z_edges, grid, shading="auto", cmap="viridis")
        fig.colorbar(image, ax=axh)
        ax3.set(xlabel="b", ylabel="z", zlabel="bin mean")
        axh.set(xlabel="b", ylabel="z")
        fig.suptitle(f"{title}, eta={eta_value}")
        fig.tight_layout()
        fig.savefig(path.with_name(f"{path.stem}_eta{eta_value}.png"), dpi=160)
        plt.close(fig)


def plot_level_maps(view: CoupledView, outer: int, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    policy = view.policy
    metrics = {
        "p0": policy.p0.squeeze(-1).cpu().numpy(),
        "pi": policy.pi.squeeze(-1).cpu().numpy(),
        "q": view.q.squeeze(-1).cpu().numpy(),
        "phat": view.phat.squeeze(-1).cpu().numpy(),
        "bp0": policy.bp0.squeeze(-1).cpu().numpy(),
        "bpI": policy.bpI.squeeze(-1).cpu().numpy(),
    }
    for name, value in metrics.items():
        _plot_map(value, view, f"outer {outer}: {name}", output / f"outer_{outer:03d}_{name}.png")


def plot_transition_maps(old: CoupledView, new: CoupledView, outer: int, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    arrays = policy_transition_arrays(old.policy, new.policy)
    metrics = {
        **{name: tensor.cpu().numpy() for name, tensor in arrays.items()},
        "delta_q": ((new.q - old.q) / (1.0 + old.q.abs())).squeeze(-1).cpu().numpy(),
        "delta_phat": ((new.phat - old.phat) / old.policy.scale).squeeze(-1).cpu().numpy(),
    }
    for name, value in metrics.items():
        _plot_map(
            value, old, f"outer {outer - 1} to {outer}: {name}",
            output / f"outer_{outer - 1:03d}_to_{outer:03d}_{name}.png",
        )


def plot_summary(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(14, 20))
    x = frame["outer"]
    panels = (
        (("p_residual",), "P residual"),
        (("q_residual",), "Q residual"),
        (("p0_function_change_p90", "pi_function_change_p90"), "P movement p90"),
        (("q_movement_p90",), "Q movement p90"),
        (("investment_selector_flip_share", "default_survival_flip_share"), "Policy flip"),
        (("pi_active_share_new",), "PI-active share"),
        (("total_p_p90", "total_p_max"), "P magnitude"),
        (("q_p90", "q_max"), "Q magnitude"),
        (("bp0_p90", "bpI_p90"), "BP movement p90"),
        (("p_seconds", "q_seconds", "outer_seconds"), "Wall-clock seconds"),
    )
    for axis, (columns, title) in zip(axes.reshape(-1), panels):
        for column in columns:
            axis.plot(x, frame[column], marker="o", label=column)
        axis.set(title=title, xlabel="coupled outer")
        axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / "coupled_outer_summary.png", dpi=180); plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sweep", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--coupled-outer-max", type=int, default=5)
    parser.add_argument("--p-policy-outer-max", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--train-limit", type=int, default=8192)
    parser.add_argument("--validation-limit", type=int, default=2048)
    parser.add_argument("--p-adam-steps", type=int, default=50)
    parser.add_argument("--p-lbfgs-iterations", type=int, default=100)
    parser.add_argument("--p-target", type=float, default=0.02)
    parser.add_argument("--q-adam-steps", type=int, default=50)
    parser.add_argument("--q-lbfgs-iterations", type=int, default=100)
    parser.add_argument("--q-target", type=float, default=0.02)
    parser.add_argument("--adam-lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--lbfgs-chunk-iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.coupled_outer_max <= 5:
        raise ValueError("coupled-outer-max must be in [1, 5]")
    torch.manual_seed(20261016)
    device = torch.device(args.device)
    output = args.output_dir.expanduser().resolve()
    checkpoints = output / "checkpoints"
    figures = output / "figures"
    level_maps = figures / "levels"
    transition_maps = figures / "transitions"
    for path in (output, checkpoints, figures, level_maps, transition_maps):
        path.mkdir(parents=True, exist_ok=True)

    run = args.run.expanduser().resolve()
    episode, _, models, _, _, _, _, provenance, _ = _load_setup(
        run, device, None, args.source_run, args.source_checkpoint
    )
    model = models["policy_value"]
    _load_policy_state(model, run / "checkpoints" / f"sweep_{args.sweep:03d}.pt", device)
    initial_payload = torch.load(args.initial_checkpoint, map_location=device, weights_only=False)
    _restore_state(model, initial_payload["policy_value"])

    train_payload = torch.load(run / "fixed_point/collocation_train.pt", map_location="cpu", weights_only=False)
    validation_payload = torch.load(run / "fixed_point/collocation_val.pt", map_location="cpu", weights_only=False)
    child_payload = torch.load(run / "fixed_point/child_shock_bank.pt", map_location="cpu", weights_only=False)
    n_train = min(args.train_limit, train_payload["parent_states"].shape[0])
    n_validation = min(args.validation_limit, validation_payload["parent_states"].shape[0])
    train_parents = train_payload["parent_states"][:n_train].to(device)
    validation_parents = validation_payload["parent_states"][:n_validation].to(device)
    train_children = child_payload["train"]["children"][:n_train].to(device)
    validation_children = child_payload["validation"]["children"][:n_validation].to(device)
    parent_hash = _tensor_digest([("train", train_parents), ("validation", validation_parents)])
    child_hash = _tensor_digest([("train", train_children), ("validation", validation_children)])
    macro_hash = _models_hash(models)

    initial_p_val = _build_p_problem(
        episode, model, validation_parents, validation_children, args.batch_size
    )
    initial_q_val, _ = build_q_survival_problem(
        episode=episode, initial_model=model, frozen_p_model=deepcopy(model),
        parents=validation_parents, children=validation_children,
        batch_size=args.batch_size, m_mode="corrected_child", include_all_regimes=True,
    )
    current_view = build_coupled_view(model, initial_p_val)
    views = [current_view]
    rows = []
    p_initial = evaluate_problem(model, initial_p_val)
    q_initial = evaluate_q_problem(model, episode, initial_q_val)
    initial_magnitude = value_magnitude_metrics(current_view.policy)
    rows.append({
        "outer": 0,
        "p_residual": p_initial["normalized_mae"],
        "q_residual": q_initial["normalized_mae"],
        "p0_function_change_p90": 0.0, "pi_function_change_p90": 0.0,
        "q_movement_p90": 0.0, "phat_movement_p90": 0.0,
        "investment_selector_flip_share": 0.0,
        "default_survival_flip_share": 0.0,
        "bp0_p90": 0.0, "bpI_p90": 0.0,
        "pi_active_share_new": float(current_view.policy.investment_selector.float().mean()),
        "q_mean": q_initial["q_mean"], "q_p90": q_initial["q_p90"], "q_max": q_initial["q_max"],
        "p_seconds": 0.0, "q_seconds": 0.0, "outer_seconds": 0.0,
        "two_cycle_flag": False, "stop_reason": "initial",
        **initial_magnitude,
    })
    torch.save({"policy_value": _clone_state(model), "outer": 0}, checkpoints / "outer_000.pt")
    plot_level_maps(current_view, 0, level_maps)

    stop_reason = "outer_limit"
    for outer in range(1, args.coupled_outer_max + 1):
        outer_started = time.perf_counter()
        q_before_p = (
            f"{_state_hash(model, Q_PREFIXES)}:{_state_hash(model, BP_PREFIXES)}"
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        p_started = time.perf_counter()
        p_model, p_rows, _, p_validation = solve_p_block(
            model=model, episode=episode,
            train_parents=train_parents, train_children=train_children,
            validation_parents=validation_parents, validation_children=validation_children,
            batch_size=args.batch_size, policy_outer_max=args.p_policy_outer_max,
            adam_steps=args.p_adam_steps, adam_lr=args.adam_lr,
            eval_every=args.eval_every, lbfgs_max_iter=args.p_lbfgs_iterations,
            lbfgs_chunk_iterations=args.lbfgs_chunk_iterations,
            target_residual=args.p_target,
        )
        p_seconds = time.perf_counter() - p_started
        p_peak_vram = (
            float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            if device.type == "cuda" else float("nan")
        )
        assert_block_invariants(
            block="P", frozen_before=q_before_p,
            frozen_after=(
                f"{_state_hash(p_model, Q_PREFIXES)}:"
                f"{_state_hash(p_model, BP_PREFIXES)}"
            ),
            macro_before=macro_hash, macro_after=_models_hash(models),
            parent_before=parent_hash,
            parent_after=_tensor_digest([("train", train_parents), ("validation", validation_parents)]),
            child_before=child_hash,
            child_after=_tensor_digest([("train", train_children), ("validation", validation_children)]),
        )
        p_after = evaluate_problem(p_model, p_validation)

        p_before_q = _non_q_hash(p_model)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        q_model, q_info, _, _ = solve_q_block(
            model=p_model, episode=episode,
            train_parents=train_parents, train_children=train_children,
            validation_parents=validation_parents, validation_children=validation_children,
            batch_size=args.batch_size, adam_steps=args.q_adam_steps,
            adam_lr=args.adam_lr, eval_every=args.eval_every,
            lbfgs_max_iter=args.q_lbfgs_iterations,
            lbfgs_chunk_iterations=args.lbfgs_chunk_iterations,
            target_residual=args.q_target,
        )
        assert_block_invariants(
            block="Q", frozen_before=p_before_q, frozen_after=_non_q_hash(q_model),
            macro_before=macro_hash, macro_after=_models_hash(models),
            parent_before=parent_hash,
            parent_after=_tensor_digest([("train", train_parents), ("validation", validation_parents)]),
            child_before=child_hash,
            child_after=_tensor_digest([("train", train_children), ("validation", validation_children)]),
        )
        q_peak_vram = (
            float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            if device.type == "cuda" else float("nan")
        )
        final_p_validation = _build_p_problem(
            episode, q_model, validation_parents, validation_children, args.batch_size
        )
        p_final = evaluate_problem(q_model, final_p_validation)
        view = build_coupled_view(q_model, final_p_validation)
        transition = coupled_transition_metrics(current_view, view)
        magnitude = value_magnitude_metrics(view.policy)
        two_cycle = {"possible_two_cycle": False}
        if len(views) >= 2:
            two_cycle = detect_coupled_two_cycle(view, views[-1], views[-2])
        row = {
            "outer": outer,
            "p_residual_after_p": p_after["normalized_mae"],
            "p_residual": p_final["normalized_mae"],
            "q_residual_before_q": q_info["initial"]["normalized_mae"],
            "q_residual": q_info["final"]["normalized_mae"],
            "p_seconds": p_seconds,
            "q_seconds": q_info["seconds"],
            "p_peak_vram_mib": p_peak_vram,
            "q_peak_vram_mib": q_peak_vram,
            "outer_seconds": time.perf_counter() - outer_started,
            "p_policy_iterations": len(p_rows),
            "p_stop_reason": p_rows[-1]["stop_reason"],
            "q_stop_reason": q_info["stop_reason"],
            "two_cycle_flag": bool(two_cycle["possible_two_cycle"]),
            "stop_reason": "continue",
            **transition,
            **magnitude,
            "q_mean": q_info["final"]["q_mean"],
            "q_p90": q_info["final"]["q_p90"],
            "q_max": q_info["final"]["q_max"],
        }
        rows.append(row)
        model = q_model
        views.append(view)
        current_view = view
        torch.save({"policy_value": _clone_state(model), "outer": outer}, checkpoints / f"outer_{outer:03d}.pt")
        pd.DataFrame(p_rows).to_csv(output / f"p_block_outer_{outer:03d}.csv", index=False)
        pd.DataFrame(q_info["adam_history"] + q_info["lbfgs_history"]).to_csv(
            output / f"q_block_outer_{outer:03d}.csv", index=False
        )
        plot_level_maps(view, outer, level_maps)
        plot_transition_maps(views[-2], views[-1], outer, transition_maps)
        pd.DataFrame(rows).to_csv(output / "coupled_outer_summary.csv", index=False)
        print(json.dumps({"outer_complete": outer, **row}, sort_keys=True), flush=True)
        if not math.isfinite(row["p_residual"]) or not math.isfinite(row["q_residual"]):
            stop_reason = "nonfinite"
            break
        if row["two_cycle_flag"]:
            stop_reason = "possible_two_cycle"
            break
        if (
            row["p_residual"] < 0.02 and row["q_residual"] < 0.02
            and row["investment_selector_flip_share"] < 0.01
            and row["default_survival_flip_share"] < 0.01
            and row["bp0_p90"] < 0.025 and row["bpI_p90"] < 0.025
            and row["p0_function_change_p90"] < 0.01
            and row["pi_function_change_p90"] < 0.01
            and row["q_movement_p90"] < 0.01
        ):
            stop_reason = "coupled_convergence_gate"
            break

    rows[-1]["stop_reason"] = stop_reason
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "coupled_outer_summary.csv", index=False)
    plot_summary(summary, figures)
    metadata = {
        "args": vars(args),
        "source_git_commit": provenance.get("source_git_commit"),
        "initial_checkpoint": str(args.initial_checkpoint),
        "macro_hash_initial": macro_hash,
        "macro_hash_final": _models_hash(models),
        "parent_bank_hash": parent_hash,
        "child_bank_hash": child_hash,
        "completed_coupled_steps": len(rows) - 1,
        "stop_reason": stop_reason,
        "fixed_macro_invariant": macro_hash == _models_hash(models),
        "m_semantics": {
            "p": "LEGACY_M1 production P operator",
            "q": "CORRECTED_CHILD clamped identically in train/eval",
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
