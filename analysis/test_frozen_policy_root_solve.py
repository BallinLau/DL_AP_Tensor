#!/usr/bin/env python3
"""Frozen-policy Bellman neural root-solve diagnostic.

This is deliberately an analysis-only program.  It freezes the production
BP-grid policy and the hard investment/default active sets at a historical
P-only sweep, while keeping both parent and child P evaluations connected to
the *same current* network during root optimization.
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
from typing import Any, Iterable, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.analyze_p_only_movement_channels import (  # noqa: E402
    BP_PREFIXES,
    P_PREFIXES,
    Q_PREFIXES,
    _expanded_inputs,
    _i_grid_values,
    _load_policy_state,
    _load_setup,
    _teacher,
)
from config import Config  # noqa: E402
from training.bp_grid_teacher import _expand_grid_children  # noqa: E402
from training.inner_fixed_point import named_parameter_hash  # noqa: E402


LEGACY_M1_OPERATOR = True
PARITY_TOL = 1e-5


def _tensor_digest(named: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = sha256()
    for name, tensor in named:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _state_hash(model: torch.nn.Module, prefixes: Sequence[str]) -> str:
    return named_parameter_hash(model, tuple(prefixes))


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _restore_state(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


def _quantile(value: torch.Tensor, q: float) -> float:
    flat = value.detach().reshape(-1).to(torch.float64)
    flat = flat[torch.isfinite(flat)]
    return float(torch.quantile(flat, q)) if flat.numel() else float("nan")


def _stats(value: torch.Tensor, prefix: str) -> dict[str, float]:
    flat = value.detach().reshape(-1).to(torch.float64)
    finite = flat[torch.isfinite(flat)]
    absolute = finite.abs()
    if not finite.numel():
        return {f"{prefix}_{key}": float("nan") for key in (
            "mae", "rmse", "p50", "p90", "p99", "max", "finite_share"
        )}
    return {
        f"{prefix}_mae": float(absolute.mean()),
        f"{prefix}_rmse": float(torch.sqrt(torch.mean(finite.square()))),
        f"{prefix}_p50": float(torch.quantile(absolute, 0.50)),
        f"{prefix}_p90": float(torch.quantile(absolute, 0.90)),
        f"{prefix}_p99": float(torch.quantile(absolute, 0.99)),
        f"{prefix}_max": float(absolute.max()),
        f"{prefix}_finite_share": float(torch.isfinite(flat).to(torch.float32).mean()),
    }


def assert_legacy_m1(used_m: torch.Tensor, tol: float = 1e-8) -> float:
    error = float((used_m.detach() - 1.0).abs().max().cpu())
    if error >= tol:
        raise RuntimeError(
            f"This experiment requires the legacy M=1 operator; max_abs(M-1)={error:.3e}"
        )
    return error


@dataclass(frozen=True)
class FrozenBranchPolicy:
    name: str
    child_states: torch.Tensor  # [B,J,7], with branch-specific frozen bp
    child_weights: torch.Tensor  # [B,J]
    selector_pi: torch.Tensor  # [I,B,J], fixed at theta_k
    survival: torch.Tensor  # [B,J], hard production survival at theta_k
    cashflow: torch.Tensor  # [B,1], production branch cash flow at bp_star
    bp_star: torch.Tensor  # [B,1]
    production_target: torch.Tensor  # [B,1]
    continuation_factor: float
    # Discount factor used by the Bellman continuation term.  ``None`` keeps
    # backward compatibility with the historical M=1 frozen-policy fixtures.
    m: Optional[torch.Tensor] = None  # [B,J]
    # Static [I*B*J, state_dim] tensor used by the fast residual engine.  It is
    # optional so older serialized/test fixtures remain valid.
    expanded_child_states: Optional[torch.Tensor] = None

    def digest_items(self, prefix: str) -> list[tuple[str, torch.Tensor]]:
        return [
            (f"{prefix}.child_states", self.child_states),
            (f"{prefix}.child_weights", self.child_weights),
            (f"{prefix}.selector_pi", self.selector_pi),
            (f"{prefix}.survival", self.survival),
            (f"{prefix}.cashflow", self.cashflow),
            (f"{prefix}.bp_star", self.bp_star),
            (f"{prefix}.production_target", self.production_target),
            *(
                [(f"{prefix}.m", self.m)]
                if self.m is not None else []
            ),
            *(
                [(f"{prefix}.expanded_child_states", self.expanded_child_states)]
                if self.expanded_child_states is not None else []
            ),
        ]


@dataclass(frozen=True)
class FrozenPolicyBatch:
    parent: torch.Tensor
    scale: torch.Tensor
    p0: FrozenBranchPolicy
    pi: FrozenBranchPolicy

    def digest(self) -> str:
        return _tensor_digest(
            [("parent", self.parent), ("scale", self.scale)]
            + self.p0.digest_items("p0")
            + self.pi.digest_items("pi")
        )


def _normalized_weights(weights: Optional[torch.Tensor], n_parent: int, n_child: int,
                        device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if weights is None:
        return torch.full((n_parent, n_child), 1.0 / n_child, device=device, dtype=dtype)
    value = weights.to(device=device, dtype=dtype)
    if value.ndim == 1:
        value = value.reshape(1, -1).expand(n_parent, -1)
    return value / value.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _frozen_equity_components(
    model: torch.nn.Module,
    child_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p0_grid, pi_grid = _i_grid_values(model, child_states)
    selector_pi = pi_grid > p0_grid
    phat = torch.where(selector_pi, pi_grid, p0_grid).mean(dim=0)
    survival = (phat > 0.0).to(phat.dtype)
    return selector_pi, survival, survival * phat


def _expand_i_grid_states(
    model: torch.nn.Module,
    child_states: torch.Tensor,
) -> torch.Tensor:
    """Materialize the frozen i-grid states once, without changing semantics."""
    i_grid = torch.linspace(
        0.0,
        float(model.i_threshold),
        steps=max(int(model.i_grid_size), 2),
        device=child_states.device,
        dtype=child_states.dtype,
    )
    expanded = child_states.unsqueeze(0).expand(
        i_grid.numel(), *child_states.shape
    ).clone()
    expanded[..., 3] = i_grid[:, None, None]
    return expanded.reshape(-1, child_states.shape[-1]).detach()


def _make_branch_snapshot(
    *,
    teacher: Any,
    model: torch.nn.Module,
    parent: torch.Tensor,
    children: torch.Tensor,
    used_m: torch.Tensor,
    child_weights: Optional[torch.Tensor],
    branch: str,
) -> FrozenBranchPolicy:
    production = teacher.compute_value_target(
        parent, children, used_m, branch=branch, child_weights=child_weights
    )
    bp_star = production["bp_star"].detach()
    fixed_eval = teacher._evaluate_grid(
        parent, children, used_m, bp_star, branch=branch, child_weights=child_weights
    )
    child_grid, _ = _expand_grid_children(
        children, bp_star, parent[:, 0:1], parent[:, 2:3]
    )
    child_states = child_grid[:, 0].detach()
    selector, survival, old_equity = _frozen_equity_components(model, child_states)
    weights = _normalized_weights(
        child_weights, parent.shape[0], child_states.shape[1], parent.device, parent.dtype
    )
    branch_m = used_m.squeeze(-1).detach()
    continuation = (weights * branch_m * old_equity).sum(dim=1, keepdim=True)
    factor = 1.0 if branch == "p0" else float(Config.G)
    reconstructed = fixed_eval["cashflow_grid_mean"][:, 0:1] + factor * continuation
    target = production["value_star"].detach()
    if not torch.allclose(reconstructed, target, atol=PARITY_TOL, rtol=1e-6):
        error = float((reconstructed - target).abs().max())
        raise RuntimeError(f"{branch} production reconstruction failed: max_abs={error:.3e}")
    return FrozenBranchPolicy(
        name=branch,
        child_states=child_states,
        child_weights=weights.detach(),
        selector_pi=selector.detach(),
        survival=survival.detach(),
        cashflow=fixed_eval["cashflow_grid_mean"][:, 0:1].detach(),
        bp_star=bp_star,
        production_target=target,
        continuation_factor=factor,
        m=branch_m,
        expanded_child_states=_expand_i_grid_states(model, child_states),
    )


def build_frozen_policy_batch(
    episode: Any,
    model: torch.nn.Module,
    parent: torch.Tensor,
    children_raw: torch.Tensor,
    *,
    require_legacy_m1: bool = True,
) -> FrozenPolicyBatch:
    if require_legacy_m1:
        children, used_m, weights = _expanded_inputs(episode, parent, children_raw)
        assert_legacy_m1(used_m)
    else:
        if children_raw.shape[-1] <= 7:
            raise ValueError("current-SDF P operator requires child transition column 7")
        child_list = list(children_raw.unbind(dim=1))
        common_m = [child[:, 7:8].detach() for child in child_list]
        expanded, _, expanded_m, weights = episode._expand_policy_expectation_children(
            child_list, common_m, common_m
        )
        children = torch.stack([child[:, :7] for child in expanded], dim=1)
        used_m = torch.stack(expanded_m, dim=1)
    teacher = _teacher(episode, model)
    with torch.no_grad():
        p0 = _make_branch_snapshot(
            teacher=teacher, model=model, parent=parent, children=children,
            used_m=used_m, child_weights=weights, branch="p0"
        )
        pi = _make_branch_snapshot(
            teacher=teacher, model=model, parent=parent, children=children,
            used_m=used_m, child_weights=weights, branch="pi"
        )
        normalize = bool(
            getattr(episode.hyperparams, "pv_bellman_normalize_by_value_scale", False)
        )
        scale = (
            model.equity_value_scale(parent).detach().clamp_min(1e-8)
            if normalize else torch.ones((parent.shape[0], 1), device=parent.device, dtype=parent.dtype)
        )
    return FrozenPolicyBatch(parent=parent.detach(), scale=scale, p0=p0, pi=pi)


def build_problem(
    episode: Any,
    model: torch.nn.Module,
    parents: torch.Tensor,
    children: torch.Tensor,
    batch_size: int,
    snapshot_batch_size: Optional[int] = None,
    require_legacy_m1: bool = True,
) -> list[FrozenPolicyBatch]:
    build_batch_size = min(
        batch_size,
        int(snapshot_batch_size) if snapshot_batch_size is not None else batch_size,
    )
    problem = []
    for start in range(0, parents.shape[0], build_batch_size):
        stop = min(start + build_batch_size, parents.shape[0])
        problem.append(build_frozen_policy_batch(
            episode, model, parents[start:stop], children[start:stop],
            require_legacy_m1=require_legacy_m1,
        ))
    if build_batch_size == batch_size:
        return problem
    return coalesce_problem(problem, batch_size)


def _merge_frozen_branches(
    branches: Sequence[FrozenBranchPolicy],
) -> FrozenBranchPolicy:
    if not branches:
        raise ValueError("cannot merge an empty frozen branch list")
    first = branches[0]
    if any(branch.name != first.name for branch in branches):
        raise ValueError("frozen branch names differ during coalescing")
    if any(branch.continuation_factor != first.continuation_factor for branch in branches):
        raise ValueError("frozen continuation factors differ during coalescing")
    expanded: Optional[torch.Tensor] = None
    if all(branch.expanded_child_states is not None for branch in branches):
        rows = []
        for branch in branches:
            shape = (
                branch.selector_pi.shape[0],
                branch.child_states.shape[0],
                branch.child_states.shape[1],
                branch.child_states.shape[2],
            )
            rows.append(branch.expanded_child_states.reshape(shape))
        expanded = torch.cat(rows, dim=1).reshape(-1, first.child_states.shape[-1])
    return FrozenBranchPolicy(
        name=first.name,
        child_states=torch.cat([branch.child_states for branch in branches], dim=0),
        child_weights=torch.cat([branch.child_weights for branch in branches], dim=0),
        selector_pi=torch.cat([branch.selector_pi for branch in branches], dim=1),
        survival=torch.cat([branch.survival for branch in branches], dim=0),
        cashflow=torch.cat([branch.cashflow for branch in branches], dim=0),
        bp_star=torch.cat([branch.bp_star for branch in branches], dim=0),
        production_target=torch.cat(
            [branch.production_target for branch in branches], dim=0
        ),
        continuation_factor=first.continuation_factor,
        m=(
            torch.cat([branch.m for branch in branches], dim=0)
            if all(branch.m is not None for branch in branches) else None
        ),
        expanded_child_states=expanded,
    )


def coalesce_problem(
    problem: Sequence[FrozenPolicyBatch],
    batch_size: int,
) -> list[FrozenPolicyBatch]:
    """Merge already-frozen snapshots without rerunning policy search."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    result: list[FrozenPolicyBatch] = []
    pending: list[FrozenPolicyBatch] = []
    pending_rows = 0
    for batch in problem:
        rows = int(batch.parent.shape[0])
        if pending and pending_rows + rows > batch_size:
            result.append(FrozenPolicyBatch(
                parent=torch.cat([item.parent for item in pending], dim=0),
                scale=torch.cat([item.scale for item in pending], dim=0),
                p0=_merge_frozen_branches([item.p0 for item in pending]),
                pi=_merge_frozen_branches([item.pi for item in pending]),
            ))
            pending, pending_rows = [], 0
        pending.append(batch)
        pending_rows += rows
        if pending_rows == batch_size:
            result.append(FrozenPolicyBatch(
                parent=torch.cat([item.parent for item in pending], dim=0),
                scale=torch.cat([item.scale for item in pending], dim=0),
                p0=_merge_frozen_branches([item.p0 for item in pending]),
                pi=_merge_frozen_branches([item.pi for item in pending]),
            ))
            pending, pending_rows = [], 0
    if pending:
        result.append(FrozenPolicyBatch(
            parent=torch.cat([item.parent for item in pending], dim=0),
            scale=torch.cat([item.scale for item in pending], dim=0),
            p0=_merge_frozen_branches([item.p0 for item in pending]),
            pi=_merge_frozen_branches([item.pi for item in pending]),
        ))
    return result


def current_frozen_child_equity_loop(
    model: torch.nn.Module,
    branch: FrozenBranchPolicy,
    *,
    detach_child: bool = False,
) -> torch.Tensor:
    flat = branch.child_states.reshape(-1, branch.child_states.shape[-1])
    i_grid = torch.linspace(
        0.0, float(model.i_threshold), steps=max(int(model.i_grid_size), 2),
        device=flat.device, dtype=flat.dtype,
    )
    selected = []
    for index, i_value in enumerate(i_grid):
        states = flat.clone()
        states[:, 3] = i_value
        p0, pi = model._value_outputs(states)
        shape = branch.child_states.shape[:2]
        p0 = p0.reshape(shape)
        pi = pi.reshape(shape)
        selected.append(torch.where(branch.selector_pi[index], pi, p0))
    phat = torch.stack(selected, dim=0).mean(dim=0)
    equity = branch.survival * phat
    return equity.detach() if detach_child else equity


def current_frozen_child_equity_vectorized(
    model: torch.nn.Module,
    branch: FrozenBranchPolicy,
    *,
    detach_child: bool = False,
    use_cache: bool = True,
) -> torch.Tensor:
    """Evaluate the complete frozen i-grid in one network call."""
    states = (
        branch.expanded_child_states
        if use_cache and branch.expanded_child_states is not None
        else _expand_i_grid_states(model, branch.child_states)
    )
    p0, pi = model._value_outputs(states)
    shape = (branch.selector_pi.shape[0],) + tuple(branch.child_states.shape[:2])
    p0_grid = p0.reshape(shape)
    pi_grid = pi.reshape(shape)
    phat = torch.where(branch.selector_pi, pi_grid, p0_grid).mean(dim=0)
    equity = branch.survival * phat
    return equity.detach() if detach_child else equity


def current_frozen_child_equity_pair(
    model: torch.nn.Module,
    p0_branch: FrozenBranchPolicy,
    pi_branch: FrozenBranchPolicy,
    *,
    detach_child: bool = False,
    use_cache: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate both frozen branches in one concatenated network call."""
    states0 = (
        p0_branch.expanded_child_states
        if use_cache and p0_branch.expanded_child_states is not None
        else _expand_i_grid_states(model, p0_branch.child_states)
    )
    statesi = (
        pi_branch.expanded_child_states
        if use_cache and pi_branch.expanded_child_states is not None
        else _expand_i_grid_states(model, pi_branch.child_states)
    )
    if states0.shape[1:] != statesi.shape[1:]:
        raise ValueError("P0/PI expanded child state dimensions do not match")
    joined = torch.cat([states0, statesi], dim=0)
    joined_p0, joined_pi = model._value_outputs(joined)
    split = states0.shape[0]

    def _equity(
        branch: FrozenBranchPolicy,
        flat_p0: torch.Tensor,
        flat_pi: torch.Tensor,
    ) -> torch.Tensor:
        shape = (branch.selector_pi.shape[0],) + tuple(branch.child_states.shape[:2])
        p0_grid = flat_p0.reshape(shape)
        pi_grid = flat_pi.reshape(shape)
        phat = torch.where(branch.selector_pi, pi_grid, p0_grid).mean(dim=0)
        return branch.survival * phat

    equity0 = _equity(p0_branch, joined_p0[:split], joined_pi[:split])
    equityi = _equity(pi_branch, joined_p0[split:], joined_pi[split:])
    if detach_child:
        return equity0.detach(), equityi.detach()
    return equity0, equityi


# Public default is the fast, mathematically identical implementation.  The
# loop implementation stays available for parity and performance controls.
current_frozen_child_equity = current_frozen_child_equity_vectorized


def dynamic_rhs(
    model: torch.nn.Module,
    branch: FrozenBranchPolicy,
    *,
    detach_child: bool = False,
    implementation: str = "fast",
    use_cache: bool = True,
) -> torch.Tensor:
    if implementation == "legacy":
        equity = current_frozen_child_equity_loop(
            model, branch, detach_child=detach_child
        )
    elif implementation == "fast":
        equity = current_frozen_child_equity_vectorized(
            model, branch, detach_child=detach_child, use_cache=use_cache
        )
    else:
        raise ValueError(f"Unknown residual implementation: {implementation!r}")
    m = branch.m if branch.m is not None else torch.ones_like(branch.child_weights)
    continuation = (branch.child_weights * m * equity).sum(dim=1, keepdim=True)
    return branch.cashflow + branch.continuation_factor * continuation


def residual_tensors(
    model: torch.nn.Module,
    batch: FrozenPolicyBatch,
    *,
    detach_child: bool = False,
    implementation: str = "fast",
    use_cache: bool = True,
) -> dict[str, torch.Tensor]:
    p0, pi = model._value_outputs(batch.parent)
    if implementation == "legacy":
        rhs0 = dynamic_rhs(
            model, batch.p0, detach_child=detach_child, implementation="legacy"
        )
        rhsi = dynamic_rhs(
            model, batch.pi, detach_child=detach_child, implementation="legacy"
        )
    elif implementation == "fast":
        equity0, equityi = current_frozen_child_equity_pair(
            model,
            batch.p0,
            batch.pi,
            detach_child=detach_child,
            use_cache=use_cache,
        )
        m0 = batch.p0.m if batch.p0.m is not None else torch.ones_like(batch.p0.child_weights)
        mi = batch.pi.m if batch.pi.m is not None else torch.ones_like(batch.pi.child_weights)
        continuation0 = (batch.p0.child_weights * m0 * equity0).sum(dim=1, keepdim=True)
        continuationi = (batch.pi.child_weights * mi * equityi).sum(dim=1, keepdim=True)
        rhs0 = batch.p0.cashflow + batch.p0.continuation_factor * continuation0
        rhsi = batch.pi.cashflow + batch.pi.continuation_factor * continuationi
    else:
        raise ValueError(f"Unknown residual implementation: {implementation!r}")
    r0, ri = p0 - rhs0, pi - rhsi
    return {
        "p0": r0,
        "pi": ri,
        "p0_normalized": r0 / batch.scale,
        "pi_normalized": ri / batch.scale,
        "p0_output": p0,
        "pi_output": pi,
        "p0_rhs": rhs0,
        "pi_rhs": rhsi,
    }


def root_loss(
    model: torch.nn.Module,
    batch: FrozenPolicyBatch,
    *,
    implementation: str = "fast",
    use_cache: bool = True,
) -> torch.Tensor:
    values = residual_tensors(
        model, batch, implementation=implementation, use_cache=use_cache
    )
    return values["p0_normalized"].square().mean() + values["pi_normalized"].square().mean()


def evaluate_problem(model: torch.nn.Module, problem: Sequence[FrozenPolicyBatch]) -> dict[str, float]:
    accum: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for batch in problem:
            values = residual_tensors(model, batch)
            for key, value in values.items():
                accum.setdefault(key, []).append(value.detach().cpu())
    p0 = torch.cat(accum["p0"])
    pi = torch.cat(accum["pi"])
    p0n = torch.cat(accum["p0_normalized"])
    pin = torch.cat(accum["pi_normalized"])
    p0out = torch.cat(accum["p0_output"])
    piout = torch.cat(accum["pi_output"])
    result = {}
    result.update(_stats(p0, "p0"))
    result.update(_stats(pi, "pi"))
    result.update(_stats(p0n, "p0_norm"))
    result.update(_stats(pin, "pi_norm"))
    result.update({
        "normalized_mae": 0.5 * (result["p0_norm_mae"] + result["pi_norm_mae"]),
        "normalized_rmse": math.sqrt(0.5 * (
            result["p0_norm_rmse"] ** 2 + result["pi_norm_rmse"] ** 2
        )),
        "p0_output_mean": float(p0out.mean()),
        "p0_output_p90": _quantile(p0out.abs(), 0.90),
        "p0_output_max": float(p0out.abs().max()),
        "pi_output_mean": float(piout.mean()),
        "pi_output_p90": _quantile(piout.abs(), 0.90),
        "pi_output_max": float(piout.abs().max()),
        "p_max": float(torch.maximum(p0out.abs(), piout.abs()).max()),
        "finite": bool(torch.isfinite(p0).all() and torch.isfinite(pi).all()),
    })
    return result


def initial_parity(problem: Sequence[FrozenPolicyBatch], model: torch.nn.Module) -> list[dict[str, Any]]:
    rows = []
    with torch.no_grad():
        for label in ("p0", "pi"):
            errors = []
            for batch in problem:
                branch = getattr(batch, label)
                errors.append((dynamic_rhs(model, branch) - branch.production_target).abs().cpu())
            value = torch.cat(errors)
            rows.append({
                "branch": label,
                "mean_abs_error": float(value.mean()),
                "p90_abs_error": _quantile(value, 0.90),
                "max_abs_error": float(value.max()),
                "tolerance": PARITY_TOL,
                "passed": bool(float(value.max()) <= PARITY_TOL),
            })
    return rows


def _p_scope(episode: Any, model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[str]]:
    original = episode.models["policy_value"]
    episode.models["policy_value"] = model
    try:
        params = episode._policy_value_stage_params("p")
    finally:
        episode.models["policy_value"] = original
    allowed = {id(param) for param in params}
    names = []
    for name, param in model.named_parameters():
        param.requires_grad_(id(param) in allowed)
        if id(param) in allowed:
            names.append(name)
    return params, names


def _grad_norm(params: Sequence[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in params:
        if param.grad is not None:
            total += float(param.grad.detach().square().sum())
    return math.sqrt(total)


def _param_norm(params: Sequence[torch.nn.Parameter]) -> float:
    return math.sqrt(sum(float(param.detach().square().sum()) for param in params))


def run_adam(
    *, model: torch.nn.Module, episode: Any, train: Sequence[FrozenPolicyBatch],
    validation: Sequence[FrozenPolicyBatch], steps: int, lr: float,
    clip: float, eval_every: int, phase: str = "adam",
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, float]]:
    params, _ = _p_scope(episode, model)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.0)
    history = []
    best = evaluate_problem(model, validation)
    best_state = _clone_state(model)
    for step in range(1, steps + 1):
        batch = train[(step - 1) % len(train)]
        optimizer.zero_grad(set_to_none=True)
        loss = root_loss(model, batch)
        loss.backward()
        raw_grad = _grad_norm(params)
        torch.nn.utils.clip_grad_norm_(params, clip)
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == steps:
            val = evaluate_problem(model, validation)
            train_metrics = evaluate_problem(model, train)
            row = {
                "phase": phase, "step": step, "loss": float(loss.detach()),
                "raw_grad_norm": raw_grad, "parameter_norm": _param_norm(params),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val.items()},
            }
            history.append(row)
            if val["normalized_mae"] < best["normalized_mae"]:
                best, best_state = val, _clone_state(model)
        if not torch.isfinite(loss):
            break
    return history, best_state, best


def run_lbfgs(
    *, model: torch.nn.Module, episode: Any, train: Sequence[FrozenPolicyBatch],
    validation: Sequence[FrozenPolicyBatch], max_iter: int, history_size: int,
    implementation: str = "fast", validate_in_closure: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, float]]:
    params, _ = _p_scope(episode, model)
    optimizer = torch.optim.LBFGS(
        params, lr=1.0, max_iter=max_iter, history_size=history_size,
        line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12,
    )
    calls = 0
    history = []
    best = evaluate_problem(model, validation)
    best_state = _clone_state(model)

    def closure() -> torch.Tensor:
        nonlocal calls, best, best_state
        optimizer.zero_grad(set_to_none=True)
        # Accumulate full-bank gradients batch by batch.  Returning a detached
        # scalar is valid for LBFGS because the closure has already populated
        # every parameter gradient; this avoids retaining one enormous graph
        # spanning the complete bank.
        loss = torch.zeros((), device=train[0].parent.device)
        for batch in train:
            batch_loss = root_loss(
                model, batch, implementation=implementation
            ) / len(train)
            batch_loss.backward()
            loss = loss + batch_loss.detach()
        calls += 1
        if validate_in_closure and (calls == 1 or calls % 5 == 0):
            val = evaluate_problem(model, validation)
            row = {
                "phase": "lbfgs", "step": calls, "loss": float(loss.detach()),
                "raw_grad_norm": _grad_norm(params), "parameter_norm": _param_norm(params),
                **{f"val_{k}": v for k, v in val.items()},
            }
            history.append(row)
            if val["normalized_mae"] < best["normalized_mae"]:
                best, best_state = val, _clone_state(model)
        return loss

    if max_iter > 0:
        optimizer.step(closure)
    final = evaluate_problem(model, validation)
    history.append({
        "phase": "lbfgs", "step": calls, "loss": float("nan"),
        "raw_grad_norm": _grad_norm(params), "parameter_norm": _param_norm(params),
        "validation_timing": "after_optimizer_block",
        **{f"val_{k}": v for k, v in final.items()},
    })
    if final["normalized_mae"] < best["normalized_mae"]:
        best, best_state = final, _clone_state(model)
    return history, best_state, best


def run_lbfgs_chunked(
    *,
    model: torch.nn.Module,
    episode: Any,
    train: Sequence[FrozenPolicyBatch],
    validation: Sequence[FrozenPolicyBatch],
    max_iter: int,
    chunk_iterations: int,
    history_size: int,
    implementation: str = "fast",
    line_search_fn: Optional[str] = "strong_wolfe",
    lr: float = 1.0,
    target_residual: Optional[float] = None,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, float], str]:
    """Run LBFGS in bounded chunks and validate only between chunks.

    The residual objective and line-search semantics within every optimizer
    block are unchanged.  Chunking keeps expensive full validation out of the
    strong-Wolfe closure and provides an explicit inexact policy-evaluation
    stopping point.
    """
    if chunk_iterations <= 0:
        raise ValueError("chunk_iterations must be positive")
    params, _ = _p_scope(episode, model)
    best = evaluate_problem(model, validation)
    best_state = _clone_state(model)
    history: list[dict[str, Any]] = []
    total_calls = 0
    completed_budget = 0
    stop_reason = "optimizer_limit"
    while completed_budget < max_iter:
        block_budget = min(chunk_iterations, max_iter - completed_budget)
        optimizer = torch.optim.LBFGS(
            params,
            lr=lr,
            max_iter=block_budget,
            history_size=history_size,
            line_search_fn=line_search_fn,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
        )
        block_calls = 0
        started = time.perf_counter()

        def closure() -> torch.Tensor:
            nonlocal block_calls, total_calls
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=train[0].parent.device)
            for batch in train:
                batch_loss = root_loss(
                    model, batch, implementation=implementation
                ) / len(train)
                batch_loss.backward()
                loss = loss + batch_loss.detach()
            block_calls += 1
            total_calls += 1
            return loss

        optimizer.step(closure)
        if train[0].parent.is_cuda:
            torch.cuda.synchronize(train[0].parent.device)
        elapsed = time.perf_counter() - started
        completed_budget += block_budget
        val = evaluate_problem(model, validation)
        row = {
            "phase": "lbfgs_chunk",
            "optimizer_budget": completed_budget,
            "block_budget": block_budget,
            "block_closure_calls": block_calls,
            "total_closure_calls": total_calls,
            "seconds": elapsed,
            "seconds_per_closure": elapsed / max(block_calls, 1),
            "raw_grad_norm": _grad_norm(params),
            "parameter_norm": _param_norm(params),
            **{f"val_{key}": value for key, value in val.items()},
        }
        history.append(row)
        if val["normalized_mae"] < best["normalized_mae"]:
            best, best_state = val, _clone_state(model)
        if not bool(val["finite"]):
            stop_reason = "nonfinite_validation"
            break
        if target_residual is not None and val["normalized_mae"] <= target_residual:
            stop_reason = "target_residual_reached"
            break
    return history, best_state, best, stop_reason


def run_picard(
    *, model: torch.nn.Module, episode: Any, train: Sequence[FrozenPolicyBatch],
    validation: Sequence[FrozenPolicyBatch], outer_steps: int, fit_steps: int,
    lr: float,
) -> list[dict[str, Any]]:
    params, _ = _p_scope(episode, model)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.0)
    history = []
    for outer in range(1, outer_steps + 1):
        targets = []
        with torch.no_grad():
            for batch in train:
                targets.append((dynamic_rhs(model, batch.p0).detach(), dynamic_rhs(model, batch.pi).detach()))
        for inner in range(fit_steps):
            index = inner % len(train)
            batch = train[index]
            target0, targeti = targets[index]
            optimizer.zero_grad(set_to_none=True)
            p0, pi = model._value_outputs(batch.parent)
            loss = ((p0 - target0) / batch.scale).square().mean()
            loss = loss + ((pi - targeti) / batch.scale).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            optimizer.step()
        val = evaluate_problem(model, validation)
        history.append({"outer": outer, **{f"val_{k}": v for k, v in val.items()}})
        if not val["finite"]:
            break
    return history


def _status(initial: float, best: float, p0: float, pi: float, finite: bool) -> str:
    reduction = initial / max(best, 1e-30)
    if finite and ((p0 < 1e-3 and pi < 1e-3) or reduction >= 100.0):
        return "strong_success"
    if finite and reduction >= 10.0:
        return "partial_success"
    return "optimizer_did_not_find_low_residual_root"


def _summary_row(method: str, sweep: int, initial: Mapping[str, float], best: Mapping[str, float],
                 final: Mapping[str, float]) -> dict[str, Any]:
    return {
        "method": method, "sweep": sweep,
        "initial_val_residual": initial["normalized_mae"],
        "best_val_residual": best["normalized_mae"],
        "final_val_residual": final["normalized_mae"],
        "reduction_factor": initial["normalized_mae"] / max(best["normalized_mae"], 1e-30),
        "P0_p90": final["p0_norm_p90"], "PI_p90": final["pi_norm_p90"],
        "P_max": final["p_max"], "finite": final["finite"],
        "status": _status(initial["normalized_mae"], best["normalized_mae"],
                          best["p0_norm_mae"], best["pi_norm_mae"], bool(best["finite"])),
    }


def policy_refresh_metrics(
    episode: Any, old_model: torch.nn.Module, root_model: torch.nn.Module,
    problem: Sequence[FrozenPolicyBatch],
) -> dict[str, Any]:
    bp0_change, bpi_change, greedy_r0, greedy_ri = [], [], [], []
    selector_flip, default_flip, active_old, active_new = [], [], [], []
    teacher = _teacher(episode, root_model)
    with torch.no_grad():
        for batch in problem:
            # Preserve the exact exogenous children already stored in each branch.
            # The two branches differ only through their frozen bp.  For the full
            # greedy refresh we recover the original child shocks from P0 states;
            # the production teacher overwrites leverage for each candidate.
            children = batch.p0.child_states.clone()
            used_m = torch.ones(
                (*children.shape[:2], 1), device=children.device, dtype=children.dtype
            )
            new0 = teacher.compute_value_target(batch.parent, children, used_m, branch="p0",
                                                child_weights=batch.p0.child_weights)
            newi = teacher.compute_value_target(batch.parent, children, used_m, branch="pi",
                                                child_weights=batch.p0.child_weights)
            p0, pi = root_model._value_outputs(batch.parent)
            greedy_r0.append(((p0 - new0["value_star"]) / batch.scale).cpu())
            greedy_ri.append(((pi - newi["value_star"]) / batch.scale).cpu())
            bp0_change.append((new0["bp_star"] - batch.p0.bp_star).abs().cpu())
            bpi_change.append((newi["bp_star"] - batch.pi.bp_star).abs().cpu())
            old0, oldi = _i_grid_values(old_model, batch.parent.unsqueeze(1))
            new0v, newiv = _i_grid_values(root_model, batch.parent.unsqueeze(1))
            old_sel, new_sel = oldi > old0, newiv > new0v
            selector_flip.append((old_sel != new_sel).to(torch.float32).cpu())
            old_phat = torch.maximum(old0, oldi).mean(0)
            new_phat = torch.maximum(new0v, newiv).mean(0)
            default_flip.append(((old_phat > 0) != (new_phat > 0)).to(torch.float32).cpu())
            active_old.append(old_sel.to(torch.float32).cpu())
            active_new.append(new_sel.to(torch.float32).cpu())
    b0, bi = torch.cat(bp0_change), torch.cat(bpi_change)
    r0, ri = torch.cat(greedy_r0), torch.cat(greedy_ri)
    return {
        "bp0_mean_abs_change": float(b0.mean()), "bp0_p50": _quantile(b0, .5),
        "bp0_p90": _quantile(b0, .9), "bp0_p99": _quantile(b0, .99),
        "bpI_mean_abs_change": float(bi.mean()), "bpI_p50": _quantile(bi, .5),
        "bpI_p90": _quantile(bi, .9), "bpI_p99": _quantile(bi, .99),
        "investment_selector_flip_share": float(torch.cat(selector_flip).mean()),
        "default_survival_flip_share": float(torch.cat(default_flip).mean()),
        "pi_active_share_old": float(torch.cat(active_old).mean()),
        "pi_active_share_new": float(torch.cat(active_new).mean()),
        "full_greedy_p0_norm_mae": float(r0.abs().mean()),
        "full_greedy_pi_norm_mae": float(ri.abs().mean()),
    }


def _plot(history: pd.DataFrame, picard: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if not history.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for phase, frame in history.groupby("phase"):
            ax.semilogy(frame["step"], frame["val_normalized_mae"], marker="o", label=phase)
        ax.set(xlabel="optimizer/closure step", ylabel="validation normalized MAE")
        ax.legend(); fig.tight_layout(); fig.savefig(figures / "root_residual.png", dpi=160); plt.close(fig)
        for metric, name in (("val_p0_norm_p90", "p0_p90"), ("val_pi_norm_p90", "pi_p90"),
                             ("val_p_max", "p_magnitude")):
            if metric in history:
                fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(history["step"], history[metric])
                ax.set(xlabel="step", ylabel=metric); fig.tight_layout()
                fig.savefig(figures / f"{name}.png", dpi=160); plt.close(fig)
    if not picard.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(picard["outer"], picard["val_normalized_mae"], marker="o", label="Picard")
        if not history.empty:
            ax.semilogy(history["step"], history["val_normalized_mae"], marker=".", label="root")
        ax.set(xlabel="iteration", ylabel="validation normalized MAE"); ax.legend()
        fig.tight_layout(); fig.savefig(figures / "picard_vs_root.png", dpi=160); plt.close(fig)
    if not summary.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(summary["method"], summary["best_val_residual"])
        ax.set_yscale("log"); ax.tick_params(axis="x", rotation=20)
        fig.tight_layout(); fig.savefig(figures / "method_summary.png", dpi=160); plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--sweep", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-limit", type=int, default=2048)
    parser.add_argument("--validation-limit", type=int, default=2048)
    parser.add_argument("--adam-steps", type=int, default=300)
    parser.add_argument("--adam-lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--lbfgs-max-iter", type=int, default=200)
    parser.add_argument("--lbfgs-history-size", type=int, default=50)
    parser.add_argument("--picard-outer", type=int, default=5)
    parser.add_argument("--picard-fit-steps", type=int, default=100)
    parser.add_argument("--run-optimization", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(20261015)
    device = torch.device(args.device)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    run = args.run.expanduser().resolve()
    checkpoint = run / "checkpoints" / f"sweep_{args.sweep:03d}.pt"
    required = [checkpoint, run / "fixed_point/collocation_train.pt",
                run / "fixed_point/collocation_val.pt", run / "fixed_point/child_shock_bank.pt",
                args.source_run, args.source_checkpoint]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Legacy run artifact is incomplete: {missing}")

    episode, hp, models, _, _, _, _, provenance, _ = _load_setup(
        run, device, None, args.source_run, args.source_checkpoint
    )
    model = models["policy_value"]
    payload = _load_policy_state(model, checkpoint, device)
    if int(payload["sweep"]) != args.sweep:
        raise RuntimeError("checkpoint sweep mismatch")
    initial_state = _clone_state(model)
    train_payload = torch.load(run / "fixed_point/collocation_train.pt", map_location="cpu", weights_only=False)
    val_payload = torch.load(run / "fixed_point/collocation_val.pt", map_location="cpu", weights_only=False)
    child_payload = torch.load(run / "fixed_point/child_shock_bank.pt", map_location="cpu", weights_only=False)
    n_train = min(args.train_limit, train_payload["parent_states"].shape[0])
    n_val = min(args.validation_limit, val_payload["parent_states"].shape[0])
    train_parents = train_payload["parent_states"][:n_train].to(device)
    val_parents = val_payload["parent_states"][:n_val].to(device)
    train_children = child_payload["train"]["children"][:n_train].to(device)
    val_children = child_payload["validation"]["children"][:n_val].to(device)

    started = time.time()
    train_problem = build_problem(episode, model, train_parents, train_children, args.batch_size)
    val_problem = build_problem(episode, model, val_parents, val_children, args.batch_size)
    frozen_hash = _tensor_digest((f"train.{i}", torch.tensor(list(bytes.fromhex(batch.digest())), dtype=torch.uint8))
                                 for i, batch in enumerate(train_problem))
    frozen_hash += ":" + _tensor_digest((f"val.{i}", torch.tensor(list(bytes.fromhex(batch.digest())), dtype=torch.uint8))
                                        for i, batch in enumerate(val_problem))
    parity_rows = initial_parity(train_problem + val_problem, model)
    pd.DataFrame(parity_rows).to_csv(output / "initial_parity.csv", index=False)
    if not all(row["passed"] for row in parity_rows):
        raise RuntimeError(f"Initial production parity failed: {parity_rows}")

    initial = evaluate_problem(model, val_problem)
    q_hash = _state_hash(model, Q_PREFIXES)
    bp_hash = _state_hash(model, BP_PREFIXES)
    params, trainable_names = _p_scope(episode, model)
    metadata: dict[str, Any] = {
        "LEGACY_M1_OPERATOR": True, "legacy_m1_max_abs_error": 0.0,
        "sweep": args.sweep, "checkpoint": str(checkpoint), "checkpoint_stage": payload.get("stage"),
        "source_run": str(args.source_run.resolve()), "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_git_commit": provenance.get("source_git_commit"),
        "train_rows": n_train, "validation_rows": n_val,
        "train_bank_fingerprint": train_payload.get("fingerprint"),
        "validation_bank_fingerprint": val_payload.get("fingerprint"),
        "train_child_fingerprint": child_payload["train"].get("fingerprint"),
        "validation_child_fingerprint": child_payload["validation"].get("fingerprint"),
        "frozen_policy_hash": frozen_hash,
        "q_hash_before": q_hash, "bp_hash_before": bp_hash,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": sum(param.numel() for param in params),
        "runtime_value_parameterization": {
            **model.value_parameterization_metadata(),
            "bellman_normalization": bool(
                getattr(hp, "pv_bellman_normalize_by_value_scale", False)
            ),
        },
        "initial_validation": initial,
        "args": vars(args),
    }
    history_rows: list[dict[str, Any]] = []
    picard_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []

    if args.run_optimization:
        picard_model = deepcopy(model).to(device)
        picard_rows = run_picard(
            model=picard_model, episode=episode, train=train_problem, validation=val_problem,
            outer_steps=args.picard_outer, fit_steps=args.picard_fit_steps, lr=args.adam_lr,
        )
        picard_final = evaluate_problem(picard_model, val_problem)
        picard_best = min(
            [initial, picard_final] + [
                {**initial, "normalized_mae": row["val_normalized_mae"],
                 "p0_norm_mae": row["val_p0_norm_mae"], "pi_norm_mae": row["val_pi_norm_mae"],
                 "finite": row["val_finite"]} for row in picard_rows
            ], key=lambda row: row["normalized_mae"]
        )
        summary_rows.append(_summary_row("Picard-fixed-policy", args.sweep, initial, picard_best, picard_final))

        root_model = deepcopy(model).to(device)
        adam_history, adam_best_state, adam_best = run_adam(
            model=root_model, episode=episode, train=train_problem, validation=val_problem,
            steps=args.adam_steps, lr=args.adam_lr, clip=10.0, eval_every=args.eval_every,
        )
        history_rows.extend(adam_history)
        adam_final = evaluate_problem(root_model, val_problem)
        summary_rows.append(_summary_row("Root-Adam", args.sweep, initial, adam_best, adam_final))
        torch.save({"policy_value": adam_best_state, "sweep": args.sweep},
                   output / "checkpoints/root_adam_best.pt")

        _restore_state(root_model, adam_best_state)
        lbfgs_history, root_best_state, root_best = run_lbfgs(
            model=root_model, episode=episode, train=train_problem, validation=val_problem,
            max_iter=args.lbfgs_max_iter, history_size=args.lbfgs_history_size,
        )
        history_rows.extend(lbfgs_history)
        _restore_state(root_model, root_best_state)
        root_final = evaluate_problem(root_model, val_problem)
        summary_rows.append(_summary_row("Root-Adam+LBFGS", args.sweep, initial, root_best, root_final))
        torch.save({"policy_value": root_best_state, "sweep": args.sweep},
                   output / "checkpoints/root_adam_lbfgs_best.pt")
        policy_rows.append({"sweep": args.sweep, **policy_refresh_metrics(
            episode, model, root_model, val_problem
        )})

        if _state_hash(root_model, Q_PREFIXES) != q_hash:
            raise RuntimeError("root solver changed Q parameters")
        if _state_hash(root_model, BP_PREFIXES) != bp_hash:
            raise RuntimeError("root solver changed BP/policy parameters")
        after_frozen_hash = _tensor_digest((f"train.{i}", torch.tensor(list(bytes.fromhex(batch.digest())), dtype=torch.uint8))
                                           for i, batch in enumerate(train_problem))
        after_frozen_hash += ":" + _tensor_digest((f"val.{i}", torch.tensor(list(bytes.fromhex(batch.digest())), dtype=torch.uint8))
                                                  for i, batch in enumerate(val_problem))
        if after_frozen_hash != frozen_hash:
            raise RuntimeError("frozen policy changed during optimization")
        metadata.update({
            "q_hash_after": _state_hash(root_model, Q_PREFIXES),
            "bp_hash_after": _state_hash(root_model, BP_PREFIXES),
            "frozen_policy_hash_after": after_frozen_hash,
        })
    else:
        summary_rows.append(_summary_row("invariant-only", args.sweep, initial, initial, initial))

    history = pd.DataFrame(history_rows)
    picard = pd.DataFrame(picard_rows)
    summary = pd.DataFrame(summary_rows)
    policy = pd.DataFrame(policy_rows)
    history.to_csv(output / "root_solve_history.csv", index=False)
    picard.to_csv(output / "picard_history.csv", index=False)
    summary.to_csv(output / "root_solve_summary.csv", index=False)
    policy.to_csv(output / "policy_change_after_root.csv", index=False)
    metadata["elapsed_seconds"] = time.time() - started
    metadata["initial_checkpoint_state_hash"] = _tensor_digest(initial_state.items())
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str, sort_keys=True), encoding="utf-8"
    )
    _plot(history, picard, summary, output)
    print(summary.to_string(index=False))
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
