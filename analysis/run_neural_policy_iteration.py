#!/usr/bin/env python3
"""Analysis-only Howard-style neural policy-iteration diagnostic.

The experiment keeps Q, the parent/child collocation banks, exact eta
expansion, and the legacy M=1 operator fixed.  A production greedy policy is
refreshed only at outer boundaries.  Within each outer evaluation the policy
snapshot is immutable and both parent and child P evaluations remain attached
to the current value network.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.test_frozen_policy_root_solve import (  # noqa: E402
    BP_PREFIXES,
    Q_PREFIXES,
    FrozenPolicyBatch,
    _clone_state,
    _i_grid_values,
    _load_policy_state,
    _load_setup,
    _quantile,
    _restore_state,
    _state_hash,
    _tensor_digest,
    assert_legacy_m1,
    build_problem,
    evaluate_problem,
    residual_tensors,
    root_loss,
    run_adam,
    run_lbfgs,
)
from config.constants import SIMMODEL  # noqa: E402


CONVERGENCE = {
    "full_greedy_p0_norm_mae": 0.02,
    "full_greedy_pi_norm_mae": 0.02,
    "investment_selector_flip_share": 0.01,
    "default_survival_flip_share": 0.01,
    "bp0_p90": 0.025,
    "bpI_p90": 0.025,
}


@dataclass(frozen=True)
class PolicyView:
    parent: torch.Tensor
    scale: torch.Tensor
    p0: torch.Tensor
    pi: torch.Tensor
    investment_selector: torch.Tensor
    survival: torch.Tensor
    bp0: torch.Tensor
    bpI: torch.Tensor

    def digest(self) -> str:
        return _tensor_digest([
            ("parent", self.parent),
            ("scale", self.scale),
            ("investment_selector", self.investment_selector.to(torch.uint8)),
            ("survival", self.survival.to(torch.uint8)),
            ("bp0", self.bp0),
            ("bpI", self.bpI),
        ])


def problem_digest(problem: Sequence[FrozenPolicyBatch], prefix: str = "batch") -> str:
    return _tensor_digest([
        (f"{prefix}.{index}", torch.tensor(list(bytes.fromhex(batch.digest())), dtype=torch.uint8))
        for index, batch in enumerate(problem)
    ])


def state_dict_digest(state: Mapping[str, torch.Tensor]) -> str:
    return _tensor_digest(state.items())


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_policy_view(model: torch.nn.Module, problem: Sequence[FrozenPolicyBatch]) -> PolicyView:
    parents = torch.cat([batch.parent for batch in problem])
    scales = torch.cat([batch.scale for batch in problem])
    bp0 = torch.cat([batch.p0.bp_star for batch in problem])
    bpI = torch.cat([batch.pi.bp_star for batch in problem])
    with torch.no_grad():
        p0, pi = model._value_outputs(parents)
        p0_grid, pi_grid = _i_grid_values(model, parents.unsqueeze(1))
        selector = (pi_grid > p0_grid).squeeze(-1).transpose(0, 1)
        phat = torch.maximum(p0_grid, pi_grid).mean(dim=0).squeeze(-1)
        survival = phat > 0.0
    return PolicyView(
        parent=parents.detach(),
        scale=scales.detach(),
        p0=p0.detach(),
        pi=pi.detach(),
        investment_selector=selector.detach(),
        survival=survival.detach(),
        bp0=bp0.detach(),
        bpI=bpI.detach(),
    )


def _abs_stats(value: torch.Tensor, prefix: str) -> dict[str, float]:
    flat = value.detach().reshape(-1).to(torch.float64)
    finite = flat[torch.isfinite(flat)].abs()
    if not finite.numel():
        return {f"{prefix}_{name}": float("nan") for name in ("mean", "l2", "p50", "p90", "p99", "max")}
    return {
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_l2": float(torch.sqrt(torch.mean(finite.square()))),
        f"{prefix}_p50": float(torch.quantile(finite, 0.50)),
        f"{prefix}_p90": float(torch.quantile(finite, 0.90)),
        f"{prefix}_p99": float(torch.quantile(finite, 0.99)),
        f"{prefix}_max": float(finite.max()),
    }


def function_transition_metrics(old: PolicyView, new: PolicyView) -> dict[str, float]:
    if old.parent.shape != new.parent.shape or not torch.equal(old.parent, new.parent):
        raise RuntimeError("function transitions require the identical ordered validation bank")
    return {
        **_abs_stats((new.p0 - old.p0) / old.scale, "p0_function_change"),
        **_abs_stats((new.pi - old.pi) / old.scale, "pi_function_change"),
    }


def policy_transition_arrays(old: PolicyView, new: PolicyView) -> dict[str, torch.Tensor]:
    if old.parent.shape != new.parent.shape or not torch.equal(old.parent, new.parent):
        raise RuntimeError("policy transitions require the identical ordered validation bank")
    return {
        "delta_p0": ((new.p0 - old.p0) / old.scale).squeeze(-1),
        "delta_pi": ((new.pi - old.pi) / old.scale).squeeze(-1),
        "investment_flip": (old.investment_selector != new.investment_selector).to(torch.float32).mean(dim=1),
        "default_flip": (old.survival != new.survival).to(torch.float32),
        "delta_bp0": (new.bp0 - old.bp0).squeeze(-1),
        "delta_bpI": (new.bpI - old.bpI).squeeze(-1),
    }


def policy_transition_metrics(old: PolicyView, new: PolicyView) -> dict[str, float]:
    arrays = policy_transition_arrays(old, new)
    result = {
        "investment_selector_flip_share": float(arrays["investment_flip"].mean()),
        "default_survival_flip_share": float(arrays["default_flip"].mean()),
        "pi_active_share_old": float(old.investment_selector.to(torch.float32).mean()),
        "pi_active_share_new": float(new.investment_selector.to(torch.float32).mean()),
        "default_share_old": float((~old.survival).to(torch.float32).mean()),
        "default_share_new": float((~new.survival).to(torch.float32).mean()),
    }
    result.update(_abs_stats(arrays["delta_bp0"], "bp0"))
    result.update(_abs_stats(arrays["delta_bpI"], "bpI"))
    return result


def value_magnitude_metrics(view: PolicyView, prefix: str = "") -> dict[str, float]:
    p0 = view.p0.squeeze(-1)
    pi = view.pi.squeeze(-1)
    total = torch.maximum(p0, pi)
    result: dict[str, float] = {}
    for name, value in (("p0", p0), ("pi", pi), ("total_p", total)):
        result[f"{prefix}{name}_mean"] = float(value.mean())
        result[f"{prefix}{name}_p90"] = _quantile(value.abs(), 0.90)
        result[f"{prefix}{name}_max"] = float(value.abs().max())
    result[f"{prefix}default_share"] = float((~view.survival).to(torch.float32).mean())
    result[f"{prefix}survival_share"] = float(view.survival.to(torch.float32).mean())
    return result


def detect_two_cycle(
    current: PolicyView,
    previous: PolicyView,
    two_back: PolicyView,
) -> dict[str, Any]:
    one = policy_transition_metrics(previous, current)
    two = policy_transition_metrics(two_back, current)
    near_two = (
        two["investment_selector_flip_share"] < 0.01
        and two["default_survival_flip_share"] < 0.01
        and two["bp0_p90"] < 0.025
        and two["bpI_p90"] < 0.025
        and abs(two["pi_active_share_new"] - two["pi_active_share_old"]) < 0.01
    )
    far_one = (
        one["investment_selector_flip_share"] > 0.02
        or one["default_survival_flip_share"] > 0.02
        or one["bp0_p90"] > 0.05
        or one["bpI_p90"] > 0.05
        or abs(one["pi_active_share_new"] - one["pi_active_share_old"]) > 0.02
    )
    return {
        "possible_two_cycle": bool(near_two and far_one),
        "one_step_investment_flip": one["investment_selector_flip_share"],
        "two_step_investment_flip": two["investment_selector_flip_share"],
        "one_step_default_flip": one["default_survival_flip_share"],
        "two_step_default_flip": two["default_survival_flip_share"],
        "one_step_bp0_p90": one["bp0_p90"],
        "two_step_bp0_p90": two["bp0_p90"],
        "one_step_bpI_p90": one["bpI_p90"],
        "two_step_bpI_p90": two["bpI_p90"],
    }


def convergence_gate(row: Mapping[str, Any]) -> bool:
    return all(float(row[name]) < threshold for name, threshold in CONVERGENCE.items())


def assert_outer_invariants(
    *,
    q_before: str,
    q_after: str,
    bp_before: str,
    bp_after: str,
    parent_bank_before: str,
    parent_bank_after: str,
    child_bank_before: str,
    child_bank_after: str,
    legacy_m1: bool,
) -> None:
    checks = {
        "Q hash": (q_before, q_after),
        "BP/policy parameter hash": (bp_before, bp_after),
        "parent bank hash": (parent_bank_before, parent_bank_after),
        "child bank hash": (child_bank_before, child_bank_after),
    }
    changed = {name: values for name, values in checks.items() if values[0] != values[1]}
    if changed:
        raise RuntimeError(f"outer invariant changed: {changed}")
    if not legacy_m1:
        raise RuntimeError("outer policy iteration requires the legacy M=1 operator")


def run_boundary_only_loop(
    initial: Any,
    outer_max: int,
    build_policy_fn: Callable[[Any], Any],
    solve_fn: Callable[[Any, Any, int], Any],
    policy_hash_fn: Callable[[Any], str],
) -> list[tuple[Any, Any]]:
    """Small testable skeleton enforcing refresh-only-at-boundaries semantics."""
    model = initial
    records = []
    for outer in range(outer_max):
        policy = build_policy_fn(model)
        before = policy_hash_fn(policy)
        root = solve_fn(model, policy, outer)
        if policy_hash_fn(policy) != before:
            raise RuntimeError("root optimizer mutated the frozen policy")
        refreshed = build_policy_fn(root)
        records.append((policy, refreshed))
        model = root
    return records


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _combined_residual(metrics: Mapping[str, Any]) -> float:
    return 0.5 * (float(metrics["p0_norm_mae"]) + float(metrics["pi_norm_mae"]))


def _plot_outer_evolution(summary: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    outer = summary["outer"]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.semilogy(outer, summary["final_frozen_p0_norm_mae"], marker="o", label="P0")
    ax.semilogy(outer, summary["final_frozen_pi_norm_mae"], marker="o", label="PI")
    ax.semilogy(outer, summary["final_frozen_residual"], marker="o", label="overall")
    ax.set(xlabel="outer", ylabel="frozen-policy validation residual")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(figures / "01_frozen_residual_by_outer.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.semilogy(outer, summary["full_greedy_p0_norm_mae"], marker="o", label="P0 full greedy")
    ax.semilogy(outer, summary["full_greedy_pi_norm_mae"], marker="o", label="PI full greedy")
    ax.set(xlabel="outer", ylabel="full-greedy normalized MAE")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(figures / "02_full_greedy_residual_by_outer.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.semilogy(outer, summary["final_frozen_residual"], marker="o", label="frozen after solve")
    ax.semilogy(outer, summary["full_greedy_residual"], marker="o", label="full greedy after refresh")
    ax.set(xlabel="outer", ylabel="validation normalized MAE")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(figures / "03_frozen_vs_greedy_residual.png", dpi=180); plt.close(fig)

    transitions = summary[summary["outer"] > 0]
    labels = [f"{int(value)-1}→{int(value)}" for value in transitions["outer"]]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for column, label in (
        ("p0_function_change_mean", "P0 MAE"),
        ("p0_function_change_p90", "P0 p90"),
        ("pi_function_change_mean", "PI MAE"),
        ("pi_function_change_p90", "PI p90"),
    ):
        ax.plot(labels, transitions[column], marker="o", label=label)
    ax.set(xlabel="root transition", ylabel="normalized function change")
    ax.grid(alpha=.25); ax.legend(ncol=2); fig.tight_layout()
    fig.savefig(figures / "04_function_change_by_transition.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(outer, summary["investment_selector_flip_share"], marker="o", label="investment selector")
    ax.plot(outer, summary["default_survival_flip_share"], marker="o", label="default/survival")
    ax.set(xlabel="outer", ylabel="flip share", ylim=(-.01, 1.01))
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(figures / "05_policy_flip_by_outer.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(outer, summary["bp0_p90"], marker="o", label="bp0 p90")
    ax.plot(outer, summary["bpI_p90"], marker="o", label="bpI p90")
    ax.plot(outer, summary["bp0_p99"], linestyle="--", marker=".", label="bp0 p99")
    ax.plot(outer, summary["bpI_p99"], linestyle="--", marker=".", label="bpI p99")
    ax.set(xlabel="outer", ylabel="absolute BP change")
    ax.grid(alpha=.25); ax.legend(ncol=2); fig.tight_layout()
    fig.savefig(figures / "06_bp_change_by_outer.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(outer, summary["pi_active_share_old"], marker="o", label="before refresh")
    ax.plot(outer, summary["pi_active_share_new"], marker="o", label="after refresh")
    ax.set(xlabel="outer", ylabel="PI-active share", ylim=(-.01, 1.01))
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(figures / "07_pi_active_share_by_outer.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(outer, summary["default_share"], marker="o", label="default")
    ax.plot(outer, summary["survival_share"], marker="o", label="survival")
    ax.set(xlabel="outer", ylabel="share", ylim=(-.01, 1.01))
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(figures / "08_default_survival_share_by_outer.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for name, style in (("p0", "-"), ("pi", "--"), ("total_p", ":")):
        for statistic in ("mean", "p90", "max"):
            ax.plot(outer, summary[f"{name}_{statistic}"], linestyle=style, marker="o",
                    label=f"{name} {statistic}")
    ax.set(xlabel="outer", ylabel="physical value magnitude")
    ax.grid(alpha=.25); ax.legend(ncol=3, fontsize=8); fig.tight_layout()
    fig.savefig(figures / "09_value_magnitude_by_outer.png", dpi=180); plt.close(fig)


def _plot_transition(
    old: PolicyView,
    new: PolicyView,
    old_outer: int,
    new_outer: int,
    output: Path,
) -> None:
    arrays = policy_transition_arrays(old, new)
    metrics = policy_transition_metrics(old, new)
    suffix = f"outer_{old_outer:03d}_to_{new_outer:03d}"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    sample = torch.linspace(0, old.p0.shape[0] - 1, min(old.p0.shape[0], 2048)).long()
    for axis, before, after, title in (
        (axes[0, 0], old.p0[sample].cpu(), new.p0[sample].cpu(), "P0 old vs new"),
        (axes[0, 1], old.pi[sample].cpu(), new.pi[sample].cpu(), "PI old vs new"),
    ):
        axis.scatter(before, after, s=5, alpha=.35)
        lower = float(torch.minimum(before.min(), after.min()))
        upper = float(torch.maximum(before.max(), after.max()))
        axis.plot([lower, upper], [lower, upper], color="black", linewidth=1)
        axis.set(xlabel="old", ylabel="new", title=title)
    axes[0, 2].hist(arrays["delta_p0"].cpu().numpy(), bins=60)
    axes[0, 2].set(title="normalized ΔP0", xlabel="ΔP0")
    axes[0, 3].hist(arrays["delta_pi"].cpu().numpy(), bins=60)
    axes[0, 3].set(title="normalized ΔPI", xlabel="ΔPI")
    axes[1, 0].hist(arrays["investment_flip"].cpu().numpy(), bins=20, range=(0, 1))
    axes[1, 0].set(title=f"investment flip, mean={metrics['investment_selector_flip_share']:.3f}")
    axes[1, 1].hist(arrays["default_flip"].cpu().numpy(), bins=2, range=(0, 1))
    axes[1, 1].set(title=f"default flip, mean={metrics['default_survival_flip_share']:.3f}")
    axes[1, 2].hist(arrays["delta_bp0"].cpu().numpy(), bins=60)
    axes[1, 2].set(title="Δbp0", xlabel="new-old")
    axes[1, 3].hist(arrays["delta_bpI"].cpu().numpy(), bins=60)
    axes[1, 3].set(title="ΔbpI", xlabel="new-old")
    fig.suptitle(f"Neural policy iteration transition {old_outer}→{new_outer}")
    fig.tight_layout(); fig.savefig(output / f"{suffix}.png", dpi=170); plt.close(fig)


def _binned_mean(x: np.ndarray, y: np.ndarray, value: np.ndarray, bins: int = 20):
    x_edges = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)
    y_edges = np.linspace(np.nanmin(y), np.nanmax(y), bins + 1)
    total, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges), weights=value)
    count, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    return mean.T, x_edges, y_edges


def _plot_state_maps(
    old: PolicyView,
    new: PolicyView,
    old_outer: int,
    new_outer: int,
    output: Path,
) -> None:
    arrays = policy_transition_arrays(old, new)
    parents = old.parent.detach().cpu().numpy()
    b = parents[:, int(SIMMODEL.B)]
    z = parents[:, int(SIMMODEL.Z)]
    eta = parents[:, int(SIMMODEL.ETA)]
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "delta_p0": arrays["delta_p0"],
        "delta_pi": arrays["delta_pi"],
        "investment_flip": arrays["investment_flip"],
        "default_flip": arrays["default_flip"],
        "delta_bp0": arrays["delta_bp0"],
        "delta_bpI": arrays["delta_bpI"],
    }
    for eta_value in (0, 1):
        mask = np.isclose(eta, eta_value)
        if not mask.any():
            continue
        for name, tensor in metrics.items():
            values = tensor.detach().cpu().numpy()[mask]
            grid, b_edges, z_edges = _binned_mean(b[mask], z[mask], values)
            fig, ax = plt.subplots(figsize=(6.5, 5))
            image = ax.pcolormesh(b_edges, z_edges, grid, shading="auto", cmap="coolwarm")
            fig.colorbar(image, ax=ax, label=f"bin mean {name}")
            ax.set(xlabel="b", ylabel="z", title=f"outer {old_outer}→{new_outer}, eta={eta_value}")
            fig.tight_layout()
            fig.savefig(
                output / f"outer_{old_outer:03d}_to_{new_outer:03d}_eta{eta_value}_{name}.png",
                dpi=170,
            )
            plt.close(fig)


def _slice_states(parents: torch.Tensor, b: torch.Tensor, z: torch.Tensor, eta: float, i_value: float) -> torch.Tensor:
    template = torch.median(parents, dim=0).values
    states = template.reshape(1, -1).expand(b.numel(), -1).clone()
    states[:, int(SIMMODEL.B)] = b
    states[:, int(SIMMODEL.Z)] = z
    states[:, int(SIMMODEL.ETA)] = eta
    states[:, int(SIMMODEL.I)] = i_value
    return states


def _plot_surfaces_and_boundaries(
    model: torch.nn.Module,
    state_paths: Sequence[Path],
    parents: torch.Tensor,
    figures: Path,
) -> None:
    surfaces = figures / "surfaces"
    boundaries = figures / "boundaries"
    surfaces.mkdir(parents=True, exist_ok=True)
    boundaries.mkdir(parents=True, exist_ok=True)
    device = parents.device
    b_grid = torch.linspace(
        float(torch.quantile(parents[:, int(SIMMODEL.B)], .02)),
        float(torch.quantile(parents[:, int(SIMMODEL.B)], .98)), 100, device=device,
    )
    z_values = torch.quantile(parents[:, int(SIMMODEL.Z)], torch.tensor([.1, .5, .9], device=device))
    i_values = torch.quantile(parents[:, int(SIMMODEL.I)], torch.tensor([.1, .5, .9], device=device))
    labels = [f"outer{index}" for index in range(len(state_paths))]

    states_by_outer = [torch.load(path, map_location=device, weights_only=False)["policy_value"] for path in state_paths]
    for eta in (0, 1):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for column, z_value in enumerate(z_values):
            states = _slice_states(parents, b_grid, torch.full_like(b_grid, z_value), eta, float(i_values[1]))
            for label, state in zip(labels, states_by_outer):
                _restore_state(model, state)
                with torch.no_grad():
                    p0, _ = model._value_outputs(states)
                axes[column].plot(b_grid.cpu(), p0.squeeze(-1).cpu(), label=label)
            axes[column].set(title=f"z q{[10, 50, 90][column]}", xlabel="b")
            axes[column].grid(alpha=.2)
        axes[0].set_ylabel("P0")
        axes[-1].legend(fontsize=8)
        fig.suptitle(f"P0 surface evolution, eta={eta}")
        fig.tight_layout(); fig.savefig(surfaces / f"p0_vs_b_eta{eta}.png", dpi=180); plt.close(fig)

        fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
        for row, i_value in enumerate(i_values):
            for column, z_value in enumerate(z_values):
                states = _slice_states(parents, b_grid, torch.full_like(b_grid, z_value), eta, float(i_value))
                for label, state in zip(labels, states_by_outer):
                    _restore_state(model, state)
                    with torch.no_grad():
                        _, pi = model._value_outputs(states)
                    axes[row, column].plot(b_grid.cpu(), pi.squeeze(-1).cpu(), label=label)
                axes[row, column].set(title=f"i q{[10, 50, 90][row]}, z q{[10, 50, 90][column]}")
                axes[row, column].grid(alpha=.2)
                if row == 2:
                    axes[row, column].set_xlabel("b")
        axes[1, 0].set_ylabel("PI")
        axes[0, -1].legend(fontsize=8)
        fig.suptitle(f"PI surface evolution, eta={eta}")
        fig.tight_layout(); fig.savefig(surfaces / f"pi_vs_b_eta{eta}.png", dpi=180); plt.close(fig)

    b_contour = torch.linspace(float(b_grid.min()), float(b_grid.max()), 70, device=device)
    z_contour = torch.linspace(
        float(torch.quantile(parents[:, int(SIMMODEL.Z)], .02)),
        float(torch.quantile(parents[:, int(SIMMODEL.Z)], .98)), 70, device=device,
    )
    bb, zz = torch.meshgrid(b_contour, z_contour, indexing="ij")
    for eta in (0, 1):
        states = _slice_states(
            parents, bb.reshape(-1), zz.reshape(-1), eta, float(i_values[1])
        )
        fig_i, ax_i = plt.subplots(figsize=(7, 5.5))
        fig_d, ax_d = plt.subplots(figsize=(7, 5.5))
        colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(states_by_outer)))
        investment_handles = []
        default_handles = []
        for label, state, color in zip(labels, states_by_outer, colors):
            _restore_state(model, state)
            with torch.no_grad():
                p0, pi = model._value_outputs(states)
                p0_grid, pi_grid = _i_grid_values(model, states.unsqueeze(1))
                phat = torch.maximum(p0_grid, pi_grid).mean(dim=0)
            investment = (pi - p0).reshape(bb.shape).cpu().numpy()
            default = phat.reshape(bb.shape).cpu().numpy()
            if np.nanmin(investment) <= 0 <= np.nanmax(investment):
                ax_i.contour(bb.cpu(), zz.cpu(), investment, levels=[0], colors=[color], linewidths=1.5)
                investment_handles.append(Line2D([0], [0], color=color, label=label))
            if np.nanmin(default) <= 0 <= np.nanmax(default):
                ax_d.contour(bb.cpu(), zz.cpu(), default, levels=[0], colors=[color], linewidths=1.5)
                default_handles.append(Line2D([0], [0], color=color, label=label))
        ax_i.set(xlabel="b", ylabel="z", title=f"Investment boundary PI-P0=0, eta={eta}")
        ax_d.set(xlabel="b", ylabel="z", title=f"Default boundary Phat=0, eta={eta}")
        if investment_handles:
            ax_i.legend(handles=investment_handles)
        if default_handles:
            ax_d.legend(handles=default_handles)
        fig_i.tight_layout(); fig_i.savefig(boundaries / f"investment_boundary_eta{eta}.png", dpi=180); plt.close(fig_i)
        fig_d.tight_layout(); fig_d.savefig(boundaries / f"default_boundary_eta{eta}.png", dpi=180); plt.close(fig_d)


def validate_outer0_reuse(
    reuse: Path,
    *,
    sweep: int,
    train_rows: int,
    validation_rows: int,
    train_fingerprint: Any,
    validation_fingerprint: Any,
    train_child_fingerprint: Any,
    validation_child_fingerprint: Any,
    frozen_policy_hash: str,
    q_hash: str,
    bp_hash: str,
) -> dict[str, Any]:
    metadata_path = reuse / "metadata.json"
    checkpoint = reuse / "checkpoints" / "root_adam_lbfgs_best.pt"
    summary_path = reuse / "root_solve_summary.csv"
    for path in (metadata_path, checkpoint, summary_path):
        if not path.exists():
            raise FileNotFoundError(f"outer0 reuse artifact missing: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "sweep": sweep,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "train_bank_fingerprint": train_fingerprint,
        "validation_bank_fingerprint": validation_fingerprint,
        "train_child_fingerprint": train_child_fingerprint,
        "validation_child_fingerprint": validation_child_fingerprint,
        "frozen_policy_hash": frozen_policy_hash,
        "q_hash_before": q_hash,
        "bp_hash_before": bp_hash,
        "legacy_m1_max_abs_error": 0.0,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items() if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"outer0 reuse provenance mismatch: {mismatches}")
    metadata["checkpoint_sha256"] = file_sha256(checkpoint)
    return metadata


def _solve_frozen_policy(
    model: torch.nn.Module,
    episode: Any,
    train_problem: Sequence[FrozenPolicyBatch],
    validation_problem: Sequence[FrozenPolicyBatch],
    *,
    adam_steps: int,
    adam_lr: float,
    eval_every: int,
    lbfgs_max_iter: int,
    lbfgs_history_size: int,
) -> tuple[torch.nn.Module, pd.DataFrame, dict[str, Any]]:
    initial = evaluate_problem(model, validation_problem)
    root = deepcopy(model).to(validation_problem[0].parent.device)
    adam_history, adam_state, adam_best = run_adam(
        model=root, episode=episode, train=train_problem, validation=validation_problem,
        steps=adam_steps, lr=adam_lr, clip=10.0, eval_every=eval_every,
    )
    _restore_state(root, adam_state)
    lbfgs_history, best_state, best = run_lbfgs(
        model=root, episode=episode, train=train_problem, validation=validation_problem,
        max_iter=lbfgs_max_iter, history_size=lbfgs_history_size,
    )
    _restore_state(root, best_state)
    final = evaluate_problem(root, validation_problem)
    history = pd.DataFrame(adam_history + lbfgs_history)
    stop_reason = "optimizer_limit"
    if final["normalized_mae"] < 0.02:
        stop_reason = "frozen_validation_below_0.02_after_optimizer"
    elif initial["normalized_mae"] / max(final["normalized_mae"], 1e-30) >= 5.0:
        stop_reason = "fivefold_reduction_after_optimizer"
    return root, history, {
        "initial": initial,
        "adam_best": adam_best,
        "best": best,
        "final": final,
        "stop_reason": stop_reason,
    }


def _historical_comparison(
    *,
    episode: Any,
    model: torch.nn.Module,
    historical_run: Path,
    validation_parents: torch.Tensor,
    validation_children: torch.Tensor,
    batch_size: int,
    policy_summary: pd.DataFrame,
    output: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    views: list[PolicyView] = []
    for sweep in range(5, 11):
        checkpoint = historical_run / "checkpoints" / f"sweep_{sweep:03d}.pt"
        if not checkpoint.exists():
            continue
        historical = deepcopy(model).to(validation_parents.device)
        _load_policy_state(historical, checkpoint, validation_parents.device)
        problem = build_problem(episode, historical, validation_parents, validation_children, batch_size)
        residual = evaluate_problem(historical, problem)
        view = build_policy_view(historical, problem)
        row = {
            "iteration_type": "historical Picard sweep",
            "iteration": sweep,
            "p0_residual": residual["p0_norm_mae"],
            "pi_residual": residual["pi_norm_mae"],
            "overall_residual": residual["normalized_mae"],
            "pi_active_share": float(view.investment_selector.to(torch.float32).mean()),
            **value_magnitude_metrics(view),
        }
        if views:
            row.update(function_transition_metrics(views[-1], view))
        rows.append(row)
        views.append(view)
    for _, item in policy_summary.iterrows():
        rows.append({
            "iteration_type": "neural policy iteration outer",
            "iteration": int(item["outer"]),
            "p0_residual": item["full_greedy_p0_norm_mae"],
            "pi_residual": item["full_greedy_pi_norm_mae"],
            "overall_residual": item["full_greedy_residual"],
            "p0_function_change_mean": item.get("p0_function_change_mean", np.nan),
            "p0_function_change_p90": item.get("p0_function_change_p90", np.nan),
            "pi_function_change_mean": item.get("pi_function_change_mean", np.nan),
            "pi_function_change_p90": item.get("pi_function_change_p90", np.nan),
            "pi_active_share": item["pi_active_share_new"],
            "default_share": item["default_share"],
            "p0_mean": item["p0_mean"], "p0_p90": item["p0_p90"], "p0_max": item["p0_max"],
            "pi_mean": item["pi_mean"], "pi_p90": item["pi_p90"], "pi_max": item["pi_max"],
            "total_p_mean": item["total_p_mean"], "total_p_p90": item["total_p_p90"],
            "total_p_max": item["total_p_max"],
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "historical_vs_policy_iteration.csv", index=False)
    figures = output / "figures" / "historical_vs_policy_iteration"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    panels = (
        ("p0_residual", "P0 greedy residual"),
        ("pi_residual", "PI greedy residual"),
        ("overall_residual", "overall greedy residual"),
        ("p0_function_change_p90", "P0 movement p90"),
        ("pi_function_change_p90", "PI movement p90"),
        ("pi_active_share", "PI-active share"),
        ("default_share", "default share"),
        ("total_p_p90", "P p90"),
        ("total_p_max", "P max"),
    )
    for axis, (column, title) in zip(axes.reshape(-1), panels):
        for label, group in frame.groupby("iteration_type"):
            axis.plot(group["iteration"], group[column], marker="o", label=label)
        axis.set(title=title, xlabel="sweep or outer")
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Historical Picard sweep vs neural policy iteration outer")
    fig.tight_layout()
    fig.savefig(figures / "historical_picard_vs_neural_policy_iteration.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-run", type=Path)
    parser.add_argument("--reuse-outer0-dir", type=Path)
    parser.add_argument("--sweep", type=int, default=5)
    parser.add_argument("--outer-max", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-limit", type=int, default=8192)
    parser.add_argument("--validation-limit", type=int, default=2048)
    parser.add_argument("--adam-steps", type=int, default=100)
    parser.add_argument("--adam-lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--lbfgs-max-iter", type=int, default=200)
    parser.add_argument("--lbfgs-history-size", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.outer_max < 1:
        raise ValueError("outer-max must be positive")
    torch.manual_seed(20261015)
    device = torch.device(args.device)
    output = args.output_dir.expanduser().resolve()
    checkpoints = output / "checkpoints"
    figures = output / "figures"
    transitions_dir = figures / "transitions"
    state_maps_dir = figures / "state_maps"
    for path in (output, checkpoints, figures, transitions_dir, state_maps_dir):
        path.mkdir(parents=True, exist_ok=True)

    run = args.run.expanduser().resolve()
    source_run = args.source_run.expanduser().resolve()
    source_checkpoint = args.source_checkpoint.expanduser().resolve()
    sweep_checkpoint = run / "checkpoints" / f"sweep_{args.sweep:03d}.pt"
    required = [
        sweep_checkpoint, source_run, source_checkpoint,
        run / "fixed_point" / "collocation_train.pt",
        run / "fixed_point" / "collocation_val.pt",
        run / "fixed_point" / "child_shock_bank.pt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"policy-iteration inputs missing: {missing}")

    episode, hp, models, _, _, _, _, provenance, _ = _load_setup(
        run, device, None, source_run, source_checkpoint
    )
    model = models["policy_value"]
    payload = _load_policy_state(model, sweep_checkpoint, device)
    if int(payload["sweep"]) != args.sweep:
        raise RuntimeError("historical initialization sweep mismatch")
    q_hash = _state_hash(model, Q_PREFIXES)
    bp_hash = _state_hash(model, BP_PREFIXES)
    initial_state_hash = state_dict_digest(_clone_state(model))

    train_payload = torch.load(run / "fixed_point" / "collocation_train.pt", map_location="cpu", weights_only=False)
    validation_payload = torch.load(run / "fixed_point" / "collocation_val.pt", map_location="cpu", weights_only=False)
    child_payload = torch.load(run / "fixed_point" / "child_shock_bank.pt", map_location="cpu", weights_only=False)
    n_train = min(args.train_limit, train_payload["parent_states"].shape[0])
    n_validation = min(args.validation_limit, validation_payload["parent_states"].shape[0])
    train_parents = train_payload["parent_states"][:n_train].to(device)
    validation_parents = validation_payload["parent_states"][:n_validation].to(device)
    train_children = child_payload["train"]["children"][:n_train].to(device)
    validation_children = child_payload["validation"]["children"][:n_validation].to(device)
    parent_bank_hash = _tensor_digest([("train", train_parents), ("validation", validation_parents)])
    child_bank_hash = _tensor_digest([("train", train_children), ("validation", validation_children)])

    started = time.time()
    summary_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    policy_views: list[PolicyView] = []
    root_views: list[PolicyView] = []
    root_checkpoints: list[Path] = []
    current_model = model
    current_train_problem = build_problem(episode, current_model, train_parents, train_children, args.batch_size)
    current_validation_problem = build_problem(
        episode, current_model, validation_parents, validation_children, args.batch_size
    )
    train_problem_hash = problem_digest(current_train_problem, "train")
    validation_problem_hash = problem_digest(current_validation_problem, "val")
    frozen_policy_hash = f"{train_problem_hash}:{validation_problem_hash}"
    policy_views.append(build_policy_view(current_model, current_validation_problem))

    outer0_reuse_metadata = None
    for outer in range(args.outer_max):
        outer_started = time.time()
        initial_checkpoint = checkpoints / f"outer_{outer:03d}_initial.pt"
        root_checkpoint = checkpoints / f"outer_{outer:03d}_root_best.pt"
        torch.save({"policy_value": _clone_state(current_model), "outer": outer}, initial_checkpoint)
        current_train_hash = problem_digest(current_train_problem, "train")
        current_validation_hash = problem_digest(current_validation_problem, "val")
        before_q = _state_hash(current_model, Q_PREFIXES)
        before_bp = _state_hash(current_model, BP_PREFIXES)
        initial_frozen = evaluate_problem(current_model, current_validation_problem)

        if outer == 0 and args.reuse_outer0_dir is not None:
            reuse = args.reuse_outer0_dir.expanduser().resolve()
            outer0_reuse_metadata = validate_outer0_reuse(
                reuse,
                sweep=args.sweep,
                train_rows=n_train,
                validation_rows=n_validation,
                train_fingerprint=train_payload.get("fingerprint"),
                validation_fingerprint=validation_payload.get("fingerprint"),
                train_child_fingerprint=child_payload["train"].get("fingerprint"),
                validation_child_fingerprint=child_payload["validation"].get("fingerprint"),
                frozen_policy_hash=frozen_policy_hash,
                q_hash=q_hash,
                bp_hash=bp_hash,
            )
            root_model = deepcopy(current_model).to(device)
            reused = torch.load(
                reuse / "checkpoints" / "root_adam_lbfgs_best.pt",
                map_location=device,
                weights_only=False,
            )
            _restore_state(root_model, reused["policy_value"])
            final_frozen = evaluate_problem(root_model, current_validation_problem)
            solve_info = {
                "initial": initial_frozen,
                "adam_best": None,
                "best": final_frozen,
                "final": final_frozen,
                "stop_reason": "reused_verified_full_k5",
            }
            history = pd.read_csv(reuse / "root_solve_history.csv")
        else:
            root_model, history, solve_info = _solve_frozen_policy(
                current_model, episode, current_train_problem, current_validation_problem,
                adam_steps=args.adam_steps, adam_lr=args.adam_lr, eval_every=args.eval_every,
                lbfgs_max_iter=args.lbfgs_max_iter, lbfgs_history_size=args.lbfgs_history_size,
            )
            final_frozen = solve_info["final"]
        history.insert(0, "outer", outer)
        history.to_csv(output / f"optimizer_history_outer_{outer:03d}.csv", index=False)
        torch.save({"policy_value": _clone_state(root_model), "outer": outer}, root_checkpoint)
        root_checkpoints.append(root_checkpoint)

        if (problem_digest(current_train_problem, "train") != current_train_hash
                or problem_digest(current_validation_problem, "val") != current_validation_hash):
            raise RuntimeError("root optimizer mutated a frozen policy snapshot")
        assert_outer_invariants(
            q_before=q_hash, q_after=_state_hash(root_model, Q_PREFIXES),
            bp_before=bp_hash, bp_after=_state_hash(root_model, BP_PREFIXES),
            parent_bank_before=parent_bank_hash,
            parent_bank_after=_tensor_digest([("train", train_parents), ("validation", validation_parents)]),
            child_bank_before=child_bank_hash,
            child_bank_after=_tensor_digest([("train", train_children), ("validation", validation_children)]),
            legacy_m1=True,
        )
        if before_q != q_hash or before_bp != bp_hash:
            raise RuntimeError("outer warm start already violated Q/BP invariants")
        next_train_problem = build_problem(episode, root_model, train_parents, train_children, args.batch_size)
        next_validation_problem = build_problem(
            episode, root_model, validation_parents, validation_children, args.batch_size
        )
        next_view = build_policy_view(root_model, next_validation_problem)
        current_policy_view = policy_views[-1]
        policy_metrics = policy_transition_metrics(current_policy_view, next_view)
        full_greedy = evaluate_problem(root_model, next_validation_problem)
        magnitudes = value_magnitude_metrics(next_view)

        function_metrics = {
            name: float("nan") for name in (
                "p0_function_change_mean", "p0_function_change_l2", "p0_function_change_p50",
                "p0_function_change_p90", "p0_function_change_p99", "p0_function_change_max",
                "pi_function_change_mean", "pi_function_change_l2", "pi_function_change_p50",
                "pi_function_change_p90", "pi_function_change_p99", "pi_function_change_max",
            )
        }
        if root_views:
            function_metrics = function_transition_metrics(root_views[-1], next_view)
            transition = {
                "from_outer": outer - 1,
                "to_outer": outer,
                **function_metrics,
                **policy_metrics,
                "full_greedy_residual_old": summary_rows[-1]["full_greedy_residual"],
                "full_greedy_residual_new": full_greedy["normalized_mae"],
                "default_share_old": summary_rows[-1]["default_share"],
                "default_share_new": magnitudes["default_share"],
                **{f"old_{name}": summary_rows[-1][name] for name in (
                    "p0_mean", "p0_p90", "p0_max", "pi_mean", "pi_p90", "pi_max",
                    "total_p_mean", "total_p_p90", "total_p_max",
                )},
                **{f"new_{name}": magnitudes[name] for name in (
                    "p0_mean", "p0_p90", "p0_max", "pi_mean", "pi_p90", "pi_max",
                    "total_p_mean", "total_p_p90", "total_p_max",
                )},
            }
            transition_rows.append(transition)
            _plot_transition(root_views[-1], next_view, outer - 1, outer, transitions_dir)
            _plot_state_maps(root_views[-1], next_view, outer - 1, outer, state_maps_dir)

        policy_views.append(next_view)
        root_views.append(next_view)
        two_cycle = {"possible_two_cycle": False}
        if outer >= 2:
            two_cycle = detect_two_cycle(policy_views[-1], policy_views[-2], policy_views[-3])
        row = {
            "outer": outer,
            "initial_frozen_residual": initial_frozen["normalized_mae"],
            "initial_frozen_p0_norm_mae": initial_frozen["p0_norm_mae"],
            "initial_frozen_pi_norm_mae": initial_frozen["pi_norm_mae"],
            "initial_frozen_p0_norm_p90": initial_frozen["p0_norm_p90"],
            "initial_frozen_pi_norm_p90": initial_frozen["pi_norm_p90"],
            "best_frozen_residual": solve_info["best"]["normalized_mae"],
            "final_frozen_residual": final_frozen["normalized_mae"],
            "final_frozen_p0_norm_mae": final_frozen["p0_norm_mae"],
            "final_frozen_pi_norm_mae": final_frozen["pi_norm_mae"],
            "final_frozen_p0_norm_p90": final_frozen["p0_norm_p90"],
            "final_frozen_pi_norm_p90": final_frozen["pi_norm_p90"],
            "final_frozen_p0_norm_p99": final_frozen["p0_norm_p99"],
            "final_frozen_pi_norm_p99": final_frozen["pi_norm_p99"],
            "full_greedy_residual": full_greedy["normalized_mae"],
            "full_greedy_p0_norm_mae": full_greedy["p0_norm_mae"],
            "full_greedy_pi_norm_mae": full_greedy["pi_norm_mae"],
            "full_greedy_p0_norm_p90": full_greedy["p0_norm_p90"],
            "full_greedy_pi_norm_p90": full_greedy["pi_norm_p90"],
            **policy_metrics,
            **function_metrics,
            **magnitudes,
            "two_cycle_flag": bool(two_cycle["possible_two_cycle"]),
            "stop_reason": solve_info["stop_reason"],
            "elapsed_seconds": time.time() - outer_started,
        }
        summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(output / "policy_iteration_summary.csv", index=False)
        pd.DataFrame(transition_rows).to_csv(output / "outer_transition_metrics.csv", index=False)
        metadata = {
            "outer": outer,
            "source_checkpoint": str(source_checkpoint),
            "source_git_commit": provenance.get("source_git_commit"),
            "historical_sweep_checkpoint": str(sweep_checkpoint),
            "historical_sweep_checkpoint_sha256": file_sha256(sweep_checkpoint),
            "initial_checkpoint": str(initial_checkpoint),
            "initial_checkpoint_sha256": file_sha256(initial_checkpoint),
            "root_checkpoint": str(root_checkpoint),
            "root_checkpoint_sha256": file_sha256(root_checkpoint),
            "previous_outer_checkpoint_hash": file_sha256(root_checkpoints[-2]) if len(root_checkpoints) > 1 else initial_state_hash,
            "current_policy_hash": current_policy_view.digest(),
            "next_policy_hash": next_view.digest(),
            "q_hash": q_hash,
            "bp_policy_parameter_hash": bp_hash,
            "train_bank_hash": parent_bank_hash,
            "child_bank_hash": child_bank_hash,
            "train_problem_hash": current_train_hash,
            "validation_problem_hash": current_validation_hash,
            "legacy_m1_invariant": True,
            "legacy_m1_max_abs_error": 0.0,
            "optimizer": {
                "adam_steps": args.adam_steps, "adam_lr": args.adam_lr,
                "eval_every": args.eval_every, "lbfgs_max_iter": args.lbfgs_max_iter,
                "lbfgs_history_size": args.lbfgs_history_size,
            },
            "reuse_outer0": outer0_reuse_metadata if outer == 0 else None,
            "two_cycle": two_cycle,
            "summary": row,
        }
        _write_json(output / f"metadata_outer_{outer:03d}.json", metadata)
        print(json.dumps({"outer_complete": outer, **row}, sort_keys=True), flush=True)

        current_model = root_model
        current_train_problem = next_train_problem
        current_validation_problem = next_validation_problem
        if convergence_gate(row):
            summary_rows[-1]["stop_reason"] = "outer_convergence_gate"
            break
        if two_cycle["possible_two_cycle"]:
            summary_rows[-1]["stop_reason"] = "possible_two_cycle"
            break

    summary = pd.DataFrame(summary_rows)
    transitions = pd.DataFrame(transition_rows)
    summary.to_csv(output / "policy_iteration_summary.csv", index=False)
    transitions.to_csv(output / "outer_transition_metrics.csv", index=False)
    _plot_outer_evolution(summary, figures)
    _plot_surfaces_and_boundaries(current_model, root_checkpoints, validation_parents, figures)
    if args.historical_run is not None:
        _historical_comparison(
            episode=episode, model=current_model,
            historical_run=args.historical_run.expanduser().resolve(),
            validation_parents=validation_parents, validation_children=validation_children,
            batch_size=args.batch_size, policy_summary=summary, output=output,
        )
    _write_json(output / "metadata.json", {
        "args": vars(args),
        "elapsed_seconds": time.time() - started,
        "completed_outers": len(summary_rows),
        "source_git_commit": provenance.get("source_git_commit"),
        "q_hash": q_hash,
        "bp_policy_parameter_hash": bp_hash,
        "train_parent_bank_fingerprint": train_payload.get("fingerprint"),
        "validation_parent_bank_fingerprint": validation_payload.get("fingerprint"),
        "train_child_bank_fingerprint": child_payload["train"].get("fingerprint"),
        "validation_child_bank_fingerprint": child_payload["validation"].get("fingerprint"),
        "train_bank_hash": parent_bank_hash,
        "child_bank_hash": child_bank_hash,
        "legacy_m1_invariant": True,
        "convergence_gate": CONVERGENCE,
    })
    print(summary.to_string(index=False), flush=True)
    print(f"Artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
