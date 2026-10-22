"""Frozen-policy dynamic Bellman residual P solver (b291 root-solve semantics).

This module replaces the Fixed Target Cache (Huber regression) P stage with a
policy-iteration solver: each iteration freezes the *policy* (greedy Grid-BP
leverage ``bp_star``, child investment selector, child survival mask, Q,
FC1/SDF) and solves the dynamic Bellman residual of the **online** P0/PI,
including the *online* child continuation (no ``detach`` on child P):

    R0 = P0_online - CF0_frozen - sum_j w_j * M_j * P_child_online_j
    RI = PI_online - CFI_frozen - g * sum_j w_j * M_j * P_child_online_j
    loss = mean((R0 / value_scale) ** 2) + mean((RI / value_scale) ** 2)

Only the optimisation method is new.  Every economic quantity keeps the EP3
production semantics of ``BPGridTeacher._branch_objective_grid`` /
``P0Loss`` / ``PILoss``:

* cashflow: ``compute_cashflow_p0`` / ``compute_cashflow_pi`` with the frozen
  Q claim at the parent and at the issued-debt state ``b := bp_star``;
* continuation: EP3 clipped ``M_used`` (``pv_use_clipped_m``), eta exact
  integration weights from ``_expand_policy_expectation_children``, normalized
  exactly once like ``normalize_child_weights``;
* child equity: ``survival_frozen * mean_i where(selector_frozen, PI_online,
  P0_online)`` over the same ``cal_phats`` investment-cost grid, i.e. the
  frozen decomposition of EP3 ``P = clamp_min(Phat, 0)``;
* investment branch continuation carries the growth factor ``Config.G``.

Parity contract: at the snapshot the frozen-policy RHS must reproduce the
production Bellman value ``value_star`` reported by the grid teacher.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from utils.firm_transition import normalize_child_weights

from .bp_grid_teacher import (
    BPGridTeacher,
    _expand_grid_children,
    _slice_child_axis,
    _stack_children,
    _target_q_claim,
    resolve_grid_chunk_plan,
)


PARITY_RTOL = 1e-5
PARITY_ATOL = 1e-5


def _module_state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _quantile(value: torch.Tensor, q: float) -> float:
    flat = value.detach().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return float("nan")
    return float(torch.quantile(flat, q))


@dataclass(frozen=True)
class FrozenBranchBatch:
    """Frozen policy + frozen cashflow for one Bellman branch of one batch."""

    name: str
    parent: torch.Tensor                 # [B,7]
    child_states: torch.Tensor           # [B,J,7] child states at frozen bp
    child_weights: torch.Tensor          # [B,J] normalized
    m_used: torch.Tensor                 # [B,J] EP3 clipped M
    raw_m: torch.Tensor                  # [B,J] raw M (diagnostics only)
    selector_pi: torch.Tensor            # [I,B,J] frozen investment selector
    survival: torch.Tensor               # [B,J] frozen hard survival mask
    cashflow: torch.Tensor               # [B,1] frozen branch cashflow
    continuation_factor: float           # 1.0 (p0) or Config.G (pi)
    scale: torch.Tensor                  # [B,1] Bellman residual scale
    bp_star: torch.Tensor                # [B,1] frozen greedy leverage
    production_value: torch.Tensor       # [B,1] value_star parity anchor
    production_cashflow_mean: torch.Tensor  # [B,1]
    production_continuation_mean: torch.Tensor  # [B,1]
    expanded_child_states: torch.Tensor  # [I*B*J,7] detached i-grid states
    i_grid_size: int

    @property
    def n_rows(self) -> int:
        return int(self.parent.shape[0])

    @property
    def n_children(self) -> int:
        return int(self.child_states.shape[1])


@dataclass(frozen=True)
class FrozenPolicyBatch:
    """P0/PI frozen branches of one EP3 training batch."""

    batch_id: int
    p0: FrozenBranchBatch
    pi: FrozenBranchBatch
    parity_max_abs_error: float
    parity_passed: bool

    @property
    def parent(self) -> torch.Tensor:
        return self.p0.parent

    @property
    def scale(self) -> torch.Tensor:
        return self.p0.scale


def _i_grid_values(model: torch.nn.Module, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate (V0, VI) over the ``cal_phats`` investment-cost grid.

    Returns ``[I,B,J]`` tensors for ``states`` of shape ``[B,J,7]``.
    """
    n_parent, n_children, state_dim = states.shape
    i_grid = torch.linspace(
        0.0,
        float(model.i_threshold),
        steps=max(int(model.i_grid_size), 2),
        device=states.device,
        dtype=states.dtype,
    )
    n_i = int(i_grid.numel())
    expanded = (
        states.reshape(1, n_parent * n_children, state_dim)
        .expand(n_i, n_parent * n_children, state_dim)
        .clone()
    )
    expanded[:, :, 3] = i_grid[:, None].expand(n_i, n_parent * n_children)
    flat = expanded.reshape(n_i * n_parent * n_children, state_dim)
    v0, vi = model._value_outputs(flat)
    v0 = v0.reshape(n_i, n_parent, n_children)
    vi = vi.reshape(n_i, n_parent, n_children)
    return v0, vi


def _expand_i_grid_states(model: torch.nn.Module, child_states: torch.Tensor) -> torch.Tensor:
    """Materialize the frozen i-grid child states once: ``[I*B*J, 7]``."""
    n_parent, n_children, state_dim = child_states.shape
    i_grid = torch.linspace(
        0.0,
        float(model.i_threshold),
        steps=max(int(model.i_grid_size), 2),
        device=child_states.device,
        dtype=child_states.dtype,
    )
    n_i = int(i_grid.numel())
    expanded = (
        child_states.reshape(1, n_parent * n_children, state_dim)
        .expand(n_i, n_parent * n_children, state_dim)
        .clone()
    )
    expanded[:, :, 3] = i_grid[:, None].expand(n_i, n_parent * n_children)
    return expanded.reshape(-1, state_dim).detach()


def frozen_child_equity(
    model: torch.nn.Module,
    branch: FrozenBranchBatch,
    *,
    detach_child: bool = False,
) -> torch.Tensor:
    """Online child equity under the frozen selector and survival mask.

    ``equity_j = survival_j * mean_i where(selector_j[i], PI_online, P0_online)``
    which equals EP3 ``P = clamp_min(Phat, 0)`` when the selector/survival are
    the model's own (they are frozen here).
    """
    flat = branch.expanded_child_states
    n_i = int(branch.i_grid_size)
    v0, vi = model._value_outputs(flat)
    v0 = v0.reshape(n_i, branch.n_rows, branch.n_children)
    vi = vi.reshape(n_i, branch.n_rows, branch.n_children)
    phat = torch.where(branch.selector_pi, vi, v0).mean(dim=0)
    equity = branch.survival * phat
    return equity.detach() if detach_child else equity


def dynamic_rhs(
    model: torch.nn.Module,
    branch: FrozenBranchBatch,
    *,
    detach_child: bool = False,
) -> torch.Tensor:
    """Frozen-policy Bellman RHS with the online child continuation."""
    equity = frozen_child_equity(model, branch, detach_child=detach_child)
    continuation = (branch.child_weights * branch.m_used * equity).sum(dim=1, keepdim=True)
    return branch.cashflow + branch.continuation_factor * continuation


def residual_tensors(
    model: torch.nn.Module,
    batch: FrozenPolicyBatch,
) -> Dict[str, torch.Tensor]:
    p0_online, pi_online = model._value_outputs(batch.parent)
    rhs0 = dynamic_rhs(model, batch.p0)
    rhsi = dynamic_rhs(model, batch.pi)
    r0 = p0_online - rhs0
    ri = pi_online - rhsi
    return {
        "p0": r0,
        "pi": ri,
        "p0_normalized": r0 / batch.scale.clamp_min(1e-12),
        "pi_normalized": ri / batch.scale.clamp_min(1e-12),
        "p0_output": p0_online,
        "pi_output": pi_online,
        "p0_rhs": rhs0,
        "pi_rhs": rhsi,
    }


def root_loss(model: torch.nn.Module, batch: FrozenPolicyBatch) -> torch.Tensor:
    values = residual_tensors(model, batch)
    return (
        values["p0_normalized"].square().mean()
        + values["pi_normalized"].square().mean()
    )


def evaluate_problem(
    model: torch.nn.Module,
    problem: Sequence[FrozenPolicyBatch],
) -> Dict[str, float]:
    def _stats(value: torch.Tensor, prefix: str) -> Dict[str, float]:
        flat = value.detach().reshape(-1)
        finite = flat[torch.isfinite(flat)]
        if finite.numel() == 0:
            return {
                f"{prefix}_mae": float("nan"),
                f"{prefix}_rmse": float("nan"),
                f"{prefix}_p90": float("nan"),
                f"{prefix}_max": float("nan"),
            }
        return {
            f"{prefix}_mae": float(finite.abs().mean()),
            f"{prefix}_rmse": float(finite.square().mean().sqrt()),
            f"{prefix}_p90": _quantile(finite.abs(), 0.9),
            f"{prefix}_max": float(finite.abs().max()),
        }

    accum: Dict[str, List[torch.Tensor]] = {}
    with torch.no_grad():
        for batch in problem:
            values = residual_tensors(model, batch)
            for key, value in values.items():
                accum.setdefault(key, []).append(value.detach().cpu())
    if not accum:
        return {
            "p0_norm_mae": float("nan"),
            "pi_norm_mae": float("nan"),
            "normalized_mae": float("nan"),
            "normalized_rmse": float("nan"),
            "finite": False,
        }
    p0n = torch.cat(accum["p0_normalized"])
    pin = torch.cat(accum["pi_normalized"])
    result: Dict[str, float] = {}
    result.update(_stats(torch.cat(accum["p0"]), "p0"))
    result.update(_stats(torch.cat(accum["pi"]), "pi"))
    result.update(_stats(p0n, "p0_norm"))
    result.update(_stats(pin, "pi_norm"))
    result.update(
        {
            "normalized_mae": 0.5 * (result["p0_norm_mae"] + result["pi_norm_mae"]),
            "normalized_rmse": math.sqrt(
                0.5 * (result["p0_norm_rmse"] ** 2 + result["pi_norm_rmse"] ** 2)
            ),
            "p0_output_mean": float(torch.cat(accum["p0_output"]).mean()),
            "pi_output_mean": float(torch.cat(accum["pi_output"]).mean()),
            "finite": bool(
                torch.isfinite(torch.cat(accum["p0"])).all()
                and torch.isfinite(torch.cat(accum["pi"])).all()
            ),
        }
    )
    return result


# ---------------------------------------------------------------------------
# Frozen policy batch construction on the EP3 production operator
# ---------------------------------------------------------------------------


def _frozen_equity_components(
    model: torch.nn.Module,
    child_states: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Freeze the EP3 ``cal_phats`` investment selector and survival mask.

    Returns ``(selector, survival, equity_snapshot)`` where
    ``equity_snapshot = survival * mean_i where(selector, VI, V0)`` equals the
    EP3 production child equity ``P = clamp_min(Phat, 0)`` at the snapshot.
    """
    v0, vi = _i_grid_values(model, child_states)
    selector = (vi > v0).detach()
    phat = torch.maximum(v0, vi).mean(dim=0)
    # EP3 ``P = clamp_min(Phat, 0)`` equals ``1{Phat>0} * Phat`` exactly.
    survival = (phat > 0.0).to(phat.dtype).detach()
    equity_snapshot = survival * torch.where(selector, vi, v0).mean(dim=0)
    return selector, survival, equity_snapshot.detach()


def _evaluate_grid_chunked(
    teacher: BPGridTeacher,
    parent_state: torch.Tensor,
    children: Sequence[torch.Tensor],
    m_list: torch.Tensor,
    bp_grid: torch.Tensor,
    *,
    branch: str,
    child_weights: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Call ``teacher._evaluate_grid`` under its parent chunk hard cap.

    ``_evaluate_grid`` refuses batches above the resolved expanded-state cap;
    real training batches are larger, so slice parents exactly like
    ``_compute_value_target_grid_batch`` does.
    """
    n_parent = int(parent_state.shape[0])
    n_children = int(_stack_children(children).shape[1])
    plan = resolve_grid_chunk_plan(
        n_parent=n_parent,
        n_grid=int(bp_grid.shape[1]),
        n_children=n_children,
        configured_parent_chunk=n_parent,
        configured_candidate_chunk=teacher.candidate_chunk_size,
        max_expanded_states=teacher.max_expanded_states,
    )
    chunk = max(1, int(plan.parent_chunk_effective))
    if chunk >= n_parent:
        return teacher._evaluate_grid(
            parent_state, children, m_list, bp_grid,
            branch=branch, child_weights=child_weights,
        )
    outputs = []
    for start in range(0, n_parent, chunk):
        stop = min(start + chunk, n_parent)
        weight_chunk = (
            child_weights
            if child_weights is not None and child_weights.ndim == 1
            else child_weights[start:stop] if child_weights is not None else None
        )
        outputs.append(
            teacher._evaluate_grid(
                parent_state[start:stop],
                _slice_child_axis(children, start, stop),
                _slice_child_axis(m_list, start, stop),
                bp_grid[start:stop],
                branch=branch,
                child_weights=weight_chunk,
            )
        )
    keys = ("cashflow_grid_mean", "continuation_grid_mean", "value_grid")
    return {key: torch.cat([out[key] for out in outputs], dim=0) for key in keys}


def build_frozen_branch(
    *,
    teacher: BPGridTeacher,
    model: torch.nn.Module,
    q_model: torch.nn.Module,
    p0_loss: Any,
    pi_loss: Any,
    parent: torch.Tensor,
    children: Sequence[torch.Tensor],
    m_used: torch.Tensor,
    raw_m: torch.Tensor,
    child_weights: Optional[torch.Tensor],
    branch: str,
    normalize_by_value_scale: bool,
) -> Tuple[FrozenBranchBatch, float, bool]:
    """Build one frozen branch; returns (branch, parity_max_abs_error, parity_ok)."""
    if branch not in {"p0", "pi"}:
        raise ValueError(f"Unknown branch: {branch!r}")
    parent_state = parent[:, :7] if parent.shape[1] > 7 else parent
    b_parent = parent_state[:, 0:1]
    eta_current = parent_state[:, 2:3].clamp(0.0, 1.0)
    weights = normalize_child_weights(
        child_weights,
        n_parent=int(parent_state.shape[0]),
        n_child=int(len(children)),
        device=parent_state.device,
        dtype=parent_state.dtype,
    )
    with torch.no_grad():
        production = teacher.compute_value_target(
            parent_state=parent_state,
            children=children,
            m_list=m_used,
            branch=branch,
            child_weights=child_weights,
        )
        bp_star = production["bp_star"].detach()
        value_star = production["value_star"].detach()
        fixed_eval = _evaluate_grid_chunked(
            teacher,
            parent_state,
            children,
            m_used,
            bp_star,
            branch=branch,
            child_weights=child_weights,
        )
        cashflow_mean = fixed_eval["cashflow_grid_mean"][:, 0:1].detach()
        continuation_mean = fixed_eval["continuation_grid_mean"][:, 0:1].detach()

        child_states, _child_b = _expand_grid_children(
            children, bp_star, b_parent, eta_current
        )
        child_states = (
            child_states[:, 0].detach() if child_states.dim() == 4 else child_states.detach()
        )
        selector, survival, equity_snapshot = _frozen_equity_components(
            model, child_states
        )

        # Frozen Q claim at the parent and at the issued-debt state.
        q_current = _target_q_claim(q_model, parent_state).detach()
        issue_state = parent_state.clone()
        issue_state[:, 0:1] = bp_star
        q_issue = _target_q_claim(q_model, issue_state).detach()
        if branch == "p0":
            cashflow = p0_loss.compute_cashflow_p0(
                parent_state[:, 4:5],
                parent_state[:, 1:2],
                b_parent,
                q_current,
                q_issue,
                eta_current,
            ).detach()
            continuation_factor = 1.0
        else:
            cashflow = pi_loss.compute_cashflow_pi(
                parent_state[:, 4:5],
                parent_state[:, 1:2],
                b_parent,
                parent_state[:, 3:4],
                q_current,
                q_issue,
                eta_current,
            ).detach()
            continuation_factor = float(getattr(pi_loss, "g", 1.0))

        m_tensor = m_used if torch.is_tensor(m_used) else torch.stack(list(m_used), dim=1)
        raw_tensor = raw_m if torch.is_tensor(raw_m) else torch.stack(list(raw_m), dim=1)
        m_tensor = m_tensor.detach().reshape(parent_state.shape[0], -1)
        raw_tensor = raw_tensor.detach().reshape(parent_state.shape[0], -1)
        # Parity: the frozen decomposition must reproduce the production RHS.
        rhs_snapshot = cashflow + continuation_factor * (
            weights * m_tensor * equity_snapshot
        ).sum(dim=1, keepdim=True)
        parity_error = float((rhs_snapshot - value_star).abs().max())
        parity_ok = bool(
            torch.allclose(rhs_snapshot, value_star, atol=PARITY_ATOL, rtol=PARITY_RTOL)
        )

        scale = (
            model.equity_value_scale(parent_state).detach().clamp_min(1e-8)
            if normalize_by_value_scale
            else torch.ones(
                (parent_state.shape[0], 1),
                device=parent_state.device,
                dtype=parent_state.dtype,
            )
        )
        expanded_child_states = _expand_i_grid_states(model, child_states)

    return (
        FrozenBranchBatch(
            name=branch,
            parent=parent_state.detach(),
            child_states=child_states,
            child_weights=weights.detach(),
            m_used=m_tensor,
            raw_m=raw_tensor,
            selector_pi=selector,
            survival=survival,
            cashflow=cashflow,
            continuation_factor=continuation_factor,
            scale=scale,
            bp_star=bp_star,
            production_value=value_star,
            production_cashflow_mean=cashflow_mean,
            production_continuation_mean=continuation_mean,
            expanded_child_states=expanded_child_states,
            i_grid_size=int(max(int(model.i_grid_size), 2)),
        ),
        parity_error,
        parity_ok,
    )


def build_frozen_policy_batch(
    episode: Any,
    model: torch.nn.Module,
    teacher: BPGridTeacher,
    batch: Dict[str, torch.Tensor],
    *,
    batch_id: int = 0,
    q_model: Optional[torch.nn.Module] = None,
) -> FrozenPolicyBatch:
    """Freeze one EP3 parent/child batch into a P0/PI policy-iteration problem."""
    q_model = q_model if q_model is not None else model
    parent = batch["parent"]
    children = batch.get("children", [])
    if not children and batch.get("child0") is not None and batch.get("child1") is not None:
        children = [batch["child0"], batch["child1"]]
    if not children:
        raise ValueError("frozen policy batch requires children")
    m_lo = float(getattr(episode.hyperparams, "pv_m_clamp_min", 0.7))
    m_hi = float(getattr(episode.hyperparams, "pv_m_clamp_max", 1.3))
    raw_m_list, m_list = episode._build_policy_m_lists(parent, children, m_lo, m_hi)
    children, raw_expanded, m_expanded, child_weights = (
        episode._expand_policy_expectation_children(children, raw_m_list, m_list)
    )
    normalize_by_value_scale = bool(
        getattr(episode.hyperparams, "pv_bellman_normalize_by_value_scale", False)
    )
    raw_tensor = torch.stack([m.reshape(parent.shape[0], -1) for m in raw_expanded], dim=1)
    used_tensor = torch.stack([m.reshape(parent.shape[0], -1) for m in m_expanded], dim=1)
    parent_state = parent[:, :7] if parent.shape[1] > 7 else parent

    p0_branch, err0, ok0 = build_frozen_branch(
        teacher=teacher,
        model=model,
        q_model=q_model,
        p0_loss=episode.loss_fns["p0"],
        pi_loss=episode.loss_fns["pi"],
        parent=parent_state,
        children=children,
        m_used=used_tensor,
        raw_m=raw_tensor,
        child_weights=child_weights,
        branch="p0",
        normalize_by_value_scale=normalize_by_value_scale,
    )
    pi_branch, erri, oki = build_frozen_branch(
        teacher=teacher,
        model=model,
        q_model=q_model,
        p0_loss=episode.loss_fns["p0"],
        pi_loss=episode.loss_fns["pi"],
        parent=parent_state,
        children=children,
        m_used=used_tensor,
        raw_m=raw_tensor,
        child_weights=child_weights,
        branch="pi",
        normalize_by_value_scale=normalize_by_value_scale,
    )
    return FrozenPolicyBatch(
        batch_id=int(batch_id),
        p0=p0_branch,
        pi=pi_branch,
        parity_max_abs_error=max(err0, erri),
        parity_passed=bool(ok0 and oki),
    )


def build_problem(
    episode: Any,
    model: torch.nn.Module,
    teacher: BPGridTeacher,
    batches: Sequence[Dict[str, torch.Tensor]],
    *,
    q_model: Optional[torch.nn.Module] = None,
) -> Tuple[List[FrozenPolicyBatch], Dict[str, float]]:
    problem: List[FrozenPolicyBatch] = []
    parity_errors: List[float] = []
    parity_flags: List[bool] = []
    was_training = model.training
    model.eval()
    try:
        for batch_id, batch in enumerate(batches):
            frozen = build_frozen_policy_batch(
                episode,
                model,
                teacher,
                batch,
                batch_id=batch_id,
                q_model=q_model,
            )
            problem.append(frozen)
            parity_errors.append(frozen.parity_max_abs_error)
            parity_flags.append(frozen.parity_passed)
    finally:
        model.train(was_training)
    max_error = max(parity_errors) if parity_errors else 0.0
    return problem, {
        "parity_batches": int(len(problem)),
        "parity_max_abs_error": float(max_error),
        "parity_tolerance": PARITY_ATOL,
        "parity_passed": bool(all(parity_flags)) if parity_flags else False,
    }


# ---------------------------------------------------------------------------
# Optimizers (b291 root-solve schedule)
# ---------------------------------------------------------------------------


def _grad_norm(params: Sequence[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in params:
        if param.grad is not None:
            total += float(param.grad.detach().square().sum())
    return math.sqrt(total)


def _param_norm(params: Sequence[torch.nn.Parameter]) -> float:
    return math.sqrt(sum(float(param.detach().square().sum()) for param in params))


def _clone_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _restore_state(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state)


def run_adam(
    *,
    model: torch.nn.Module,
    params: Sequence[torch.nn.Parameter],
    train: Sequence[FrozenPolicyBatch],
    validation: Sequence[FrozenPolicyBatch],
    steps: int,
    lr: float,
    clip: float,
    eval_every: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor], Dict[str, float], int]:
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.0)
    history: List[Dict[str, Any]] = []
    best = evaluate_problem(model, validation)
    best_state = _clone_state(model)
    completed_steps = 0
    for step in range(1, steps + 1):
        batch = train[(step - 1) % len(train)]
        optimizer.zero_grad(set_to_none=True)
        loss = root_loss(model, batch)
        if not torch.isfinite(loss):
            history.append({"phase": "adam", "step": step, "loss": float("nan"), "nonfinite": 1})
            break
        loss.backward()
        raw_grad = _grad_norm(params)
        torch.nn.utils.clip_grad_norm_(params, clip)
        optimizer.step()
        completed_steps = step
        if step == 1 or step % eval_every == 0 or step == steps:
            val = evaluate_problem(model, validation)
            history.append(
                {
                    "phase": "adam",
                    "step": step,
                    "loss": float(loss.detach()),
                    "raw_grad_norm": raw_grad,
                    "parameter_norm": _param_norm(params),
                    "val_normalized_mae": val["normalized_mae"],
                    "val_p0_norm_mae": val["p0_norm_mae"],
                    "val_pi_norm_mae": val["pi_norm_mae"],
                }
            )
            if val["normalized_mae"] < best["normalized_mae"]:
                best, best_state = val, _clone_state(model)
    return history, best_state, best, completed_steps


def run_lbfgs_chunked(
    *,
    model: torch.nn.Module,
    params: Sequence[torch.nn.Parameter],
    train: Sequence[FrozenPolicyBatch],
    validation: Sequence[FrozenPolicyBatch],
    max_iter: int,
    chunk_iterations: int,
    history_size: int,
    target_residual: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor], Dict[str, float], str, int]:
    if chunk_iterations <= 0:
        raise ValueError("chunk_iterations must be positive")
    best = evaluate_problem(model, validation)
    best_state = _clone_state(model)
    history: List[Dict[str, Any]] = []
    total_calls = 0
    completed_budget = 0
    stop_reason = "optimizer_limit"
    while completed_budget < max_iter:
        block_budget = min(chunk_iterations, max_iter - completed_budget)
        optimizer = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=block_budget,
            history_size=history_size,
            line_search_fn="strong_wolfe",
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
                batch_loss = root_loss(model, batch) / len(train)
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
        history.append(
            {
                "phase": "lbfgs_chunk",
                "optimizer_budget": completed_budget,
                "block_budget": block_budget,
                "block_closure_calls": block_calls,
                "total_closure_calls": total_calls,
                "seconds": elapsed,
                "raw_grad_norm": _grad_norm(params),
                "parameter_norm": _param_norm(params),
                "val_normalized_mae": val["normalized_mae"],
                "val_p0_norm_mae": val["p0_norm_mae"],
                "val_pi_norm_mae": val["pi_norm_mae"],
            }
        )
        if val["normalized_mae"] < best["normalized_mae"]:
            best, best_state = val, _clone_state(model)
        if not bool(val["finite"]):
            stop_reason = "nonfinite_validation"
            break
        if target_residual is not None and val["normalized_mae"] <= target_residual:
            stop_reason = "target_residual_reached"
            break
    return history, best_state, best, stop_reason, total_calls


# ---------------------------------------------------------------------------
# One P policy iteration
# ---------------------------------------------------------------------------


def fresh_greedy_residual(
    episode: Any,
    model: torch.nn.Module,
    teacher: BPGridTeacher,
    batches: Sequence[Dict[str, torch.Tensor]],
    *,
    q_model: Optional[torch.nn.Module] = None,
) -> Dict[str, float]:
    """Recompute the greedy policy at the current P and report its residual.

    ``value_star`` is the production RHS under the fresh greedy policy, so
    ``P - value_star`` is the fresh-greedy Bellman residual.
    """
    q_model = q_model if q_model is not None else model
    r0, ri, bp0_change, bpi_change = [], [], [], []
    normalize = bool(
        getattr(episode.hyperparams, "pv_bellman_normalize_by_value_scale", False)
    )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                parent = batch["parent"]
                children = batch.get("children", [])
                if not children and batch.get("child0") is not None:
                    children = [batch["child0"], batch["child1"]]
                parent_state = parent[:, :7] if parent.shape[1] > 7 else parent
                m_lo = float(getattr(episode.hyperparams, "pv_m_clamp_min", 0.7))
                m_hi = float(getattr(episode.hyperparams, "pv_m_clamp_max", 1.3))
                _raw_m, m_list = episode._build_policy_m_lists(parent, children, m_lo, m_hi)
                children, _raw_expanded, m_expanded, child_weights = (
                    episode._expand_policy_expectation_children(children, m_list, m_list)
                )
                scale = (
                    model.equity_value_scale(parent_state).clamp_min(1e-8)
                    if normalize
                    else torch.ones(
                        (parent_state.shape[0], 1),
                        device=parent_state.device,
                        dtype=parent_state.dtype,
                    )
                )
                p0_online, pi_online = model._value_outputs(parent_state)
                for branch, online, sink in (
                    ("p0", p0_online, r0),
                    ("pi", pi_online, ri),
                ):
                    target = teacher.compute_value_target(
                        parent_state=parent_state,
                        children=children,
                        m_list=m_expanded,
                        branch=branch,
                        child_weights=child_weights,
                    )
                    sink.append(((online - target["value_star"]) / scale).detach().cpu())
    finally:
        model.train(was_training)
    if not r0:
        return {"fresh_greedy_p0_norm_mae": float("nan"), "fresh_greedy_pi_norm_mae": float("nan")}
    r0_flat = torch.cat(r0).reshape(-1)
    ri_flat = torch.cat(ri).reshape(-1)
    return {
        "fresh_greedy_p0_norm_mae": float(r0_flat.abs().mean()),
        "fresh_greedy_pi_norm_mae": float(ri_flat.abs().mean()),
        "fresh_greedy_p0_norm_p90": _quantile(r0_flat.abs(), 0.9),
        "fresh_greedy_pi_norm_p90": _quantile(ri_flat.abs(), 0.9),
    }


def p_root_params(episode: Any, model: torch.nn.Module) -> List[torch.nn.Parameter]:
    """Trainable P scope of the frozen-policy residual solver.

    按 Prompt 3.5：只更新 ``value_encoder`` / ``v0_head`` / ``vi_head``。旧的
    ``barz_model``/``bari_model`` 派生头不在 Dynamic Residual 的计算图内
    （策略与掩码已冻结），因此也不属于本次 P Solver 的学习范围。
    """
    modules = [
        getattr(model, "value_encoder", None),
        getattr(model, "v0_head", None),
        getattr(model, "vi_head", None),
    ]
    if any(module is None for module in modules):
        raise ValueError("policy_value model lacks value_encoder/v0_head/vi_head")
    params: List[torch.nn.Parameter] = []
    seen = set()
    for module in modules:
        for param in module.parameters():
            if id(param) not in seen:
                seen.add(id(param))
                params.append(param)
    return params


def run_p_policy_iteration(
    episode: Any,
    model: torch.nn.Module,
    *,
    train_batches: Sequence[Dict[str, torch.Tensor]],
    validation_batches: Sequence[Dict[str, torch.Tensor]],
    q_model: Optional[torch.nn.Module],
    inner_index: int,
    iteration_index: int,
    adam_steps: int,
    adam_lr: float,
    adam_clip: float,
    eval_every: int,
    lbfgs_max_iter: int,
    lbfgs_chunk_iterations: int,
    lbfgs_history_size: int,
    target_residual: Optional[float] = None,
) -> Dict[str, Any]:
    """One frozen-policy P iteration: greedy refresh -> residual solve -> best state."""
    device = next(model.parameters()).device
    params = p_root_params(episode, model)
    root_ids = {id(param) for param in params}
    p_scope_all = episode._policy_value_stage_params("p")
    non_root_p = [param for param in p_scope_all if id(param) not in root_ids]
    non_root_p_snapshot = episode._snapshot_params(non_root_p)
    q_params = episode._policy_value_stage_params("q")
    bp_params = episode._policy_value_stage_params("bp")
    q_snapshot = episode._snapshot_params(q_params)
    bp_snapshot = episode._snapshot_params(bp_params)
    sdf_model = episode.models.get("sdf_fc1")
    sdf_hash_before = _module_state_hash(sdf_model) if sdf_model is not None else None
    q_hash_before = episode._params_content_hash(q_params)
    start_state = _clone_state(model)
    started = time.perf_counter()

    teacher = BPGridTeacher.from_hyperparams(
        model,
        episode.loss_fns["p0"],
        episode.loss_fns["pi"],
        episode.hyperparams,
        q_target_model=q_model,
    )
    train_problem, train_parity = build_problem(
        episode, model, teacher, train_batches, q_model=q_model
    )
    validation_problem, val_parity = build_problem(
        episode, model, teacher, validation_batches or train_batches, q_model=q_model
    )
    initial = evaluate_problem(model, validation_problem)
    with torch.no_grad():
        start_p0, start_pi = model._value_outputs(validation_problem[0].parent)

    adam_history, adam_state, adam_best, adam_steps_done = run_adam(
        model=model,
        params=params,
        train=train_problem,
        validation=validation_problem,
        steps=adam_steps,
        lr=adam_lr,
        clip=adam_clip,
        eval_every=eval_every,
    )
    _restore_state(model, adam_state)
    if target_residual is not None and adam_best["normalized_mae"] <= target_residual:
        lbfgs_history: List[Dict[str, Any]] = []
        best_state, best = adam_state, adam_best
        stop_reason = "target_residual_reached_after_adam"
        lbfgs_calls = 0
    else:
        lbfgs_history, best_state, best, stop_reason, lbfgs_calls = run_lbfgs_chunked(
            model=model,
            params=params,
            train=train_problem,
            validation=validation_problem,
            max_iter=lbfgs_max_iter,
            chunk_iterations=lbfgs_chunk_iterations,
            history_size=lbfgs_history_size,
            target_residual=target_residual,
        )
    _restore_state(model, best_state)
    final = evaluate_problem(model, validation_problem)

    # 冻结策略残差 + 重新 Greedy 残差（独立固定 Validation）。
    fresh = fresh_greedy_residual(
        episode, model, teacher, validation_batches or train_batches, q_model=q_model
    )

    # Greedy BP drift: frozen iteration policy vs fresh greedy at the accepted
    # state, on every train batch.
    bp0_drift, bpi_drift = [], []
    with torch.no_grad():
        for frozen_batch, source_batch in zip(train_problem, train_batches):
            for branch_name, drift_sink in (("p0", bp0_drift), ("pi", bpi_drift)):
                fresh_bp = _recompute_bp_star(
                    episode, model, teacher, source_batch, branch_name, q_model
                )
                frozen_branch = getattr(frozen_batch, branch_name)
                drift_sink.append(
                    (frozen_branch.bp_star - fresh_bp).abs().reshape(-1)
                )
        end_p0, end_pi = model._value_outputs(validation_problem[0].parent)
        drift_p0 = float((end_p0 - start_p0).abs().mean())
        drift_pi = float((end_pi - start_pi).abs().mean())

    q_hash_after = episode._params_content_hash(q_params)
    if q_hash_after != q_hash_before:
        raise RuntimeError("P policy iteration mutated Q parameters")
    if episode._param_max_change_from_snapshot(q_params, q_snapshot) != 0.0:
        raise RuntimeError("P policy iteration changed Q parameter values")
    if episode._param_max_change_from_snapshot(bp_params, bp_snapshot) != 0.0:
        raise RuntimeError("P policy iteration changed BP policy parameters")
    if episode._param_max_change_from_snapshot(non_root_p, non_root_p_snapshot) != 0.0:
        raise RuntimeError(
            "P policy iteration changed non-root P-scope parameters "
            "(barz/bari derived heads must stay frozen)"
        )
    sdf_hash_after = _module_state_hash(sdf_model) if sdf_model is not None else None
    if sdf_hash_after != sdf_hash_before:
        raise RuntimeError("P policy iteration mutated the frozen SDF/FC1 model")

    elapsed = time.perf_counter() - started
    peak_memory = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))

    summary: Dict[str, Any] = {
        "inner_index": int(inner_index),
        "iteration_index": int(iteration_index),
        "status": "accepted",
        "optimizer_steps": int(adam_steps_done),
        "adam_steps": int(adam_steps_done),
        "lbfgs_closure_calls": int(lbfgs_calls),
        "lbfgs_optimizer_budget": int(lbfgs_max_iter),
        "stop_reason": stop_reason,
        "frozen_policy_p0_residual": {
            "mae": final["p0_norm_mae"],
            "rmse": final["p0_norm_rmse"],
            "p90": final["p0_norm_p90"],
        },
        "frozen_policy_pi_residual": {
            "mae": final["pi_norm_mae"],
            "rmse": final["pi_norm_rmse"],
            "p90": final["pi_norm_p90"],
        },
        "initial_frozen_policy_residual": {
            "p0_mae": initial["p0_norm_mae"],
            "pi_mae": initial["pi_norm_mae"],
            "normalized_mae": initial["normalized_mae"],
        },
        "fresh_greedy_residual": fresh,
        "function_drift": {
            "p0_output_mean_abs": drift_p0,
            "pi_output_mean_abs": drift_pi,
        },
        "greedy_bp_drift": {
            "p0_mean_abs": (
                float(torch.cat(bp0_drift).mean()) if bp0_drift else float("nan")
            ),
            "pi_mean_abs": (
                float(torch.cat(bpi_drift).mean()) if bpi_drift else float("nan")
            ),
            "p0_p90": _quantile(torch.cat(bp0_drift), 0.9) if bp0_drift else float("nan"),
            "pi_p90": _quantile(torch.cat(bpi_drift), 0.9) if bpi_drift else float("nan"),
        },
        "parity": {
            "train": train_parity,
            "validation": val_parity,
        },
        "module_hashes": {
            "q_params": q_hash_after,
            "sdf_fc1": sdf_hash_after,
        },
        "p_scope_modules": ["value_encoder", "v0_head", "vi_head"],
        "elapsed_seconds": float(elapsed),
        "peak_gpu_memory_bytes": peak_memory,
        "adam_history_len": len(adam_history),
        "lbfgs_history_len": len(lbfgs_history),
    }
    return summary


def _recompute_bp_star(
    episode: Any,
    model: torch.nn.Module,
    teacher: BPGridTeacher,
    batch: Dict[str, torch.Tensor],
    branch: str,
    q_model: Optional[torch.nn.Module],
) -> torch.Tensor:
    parent = batch["parent"]
    children = batch.get("children", [])
    if not children and batch.get("child0") is not None:
        children = [batch["child0"], batch["child1"]]
    parent_state = parent[:, :7] if parent.shape[1] > 7 else parent
    m_lo = float(getattr(episode.hyperparams, "pv_m_clamp_min", 0.7))
    m_hi = float(getattr(episode.hyperparams, "pv_m_clamp_max", 1.3))
    _raw_m, m_list = episode._build_policy_m_lists(parent, children, m_lo, m_hi)
    children, _raw_expanded, m_expanded, child_weights = (
        episode._expand_policy_expectation_children(children, m_list, m_list)
    )
    target = teacher.compute_value_target(
        parent_state=parent_state,
        children=children,
        m_list=m_expanded,
        branch=branch,
        child_weights=child_weights,
    )
    return target["bp_star"]