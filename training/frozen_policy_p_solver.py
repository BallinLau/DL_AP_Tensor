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
import logging
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


logger = logging.getLogger(__name__)


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
    parity_max_abs_error: float = 0.0
    parity_error_rows: Tuple[int, ...] = ()

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
    # 每个 Parent 行的来源标记（0=Simulated / 1=Coverage / -1=未知），仅用于
    # 诊断分组，绝不参与任何经济计算或验收判定。
    source_id: Optional[torch.Tensor] = None

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
    equity0 = frozen_child_equity(model, batch.p0)
    equityi = frozen_child_equity(model, batch.pi)
    cont0 = (batch.p0.child_weights * batch.p0.m_used * equity0).sum(dim=1, keepdim=True)
    conti = (batch.pi.child_weights * batch.pi.m_used * equityi).sum(dim=1, keepdim=True)
    rhs0 = batch.p0.cashflow + batch.p0.continuation_factor * cont0
    rhsi = batch.pi.cashflow + batch.pi.continuation_factor * conti
    r0 = p0_online - rhs0
    ri = pi_online - rhsi
    r0n = r0 / batch.scale.clamp_min(1e-12)
    rin = ri / batch.scale.clamp_min(1e-12)
    return {
        "p0": r0,
        "pi": ri,
        "p0_normalized": r0n,
        "pi_normalized": rin,
        "p0_output": p0_online,
        "pi_output": pi_online,
        "p0_rhs": rhs0,
        "pi_rhs": rhsi,
        "p0_continuation": cont0,
        "pi_continuation": conti,
        "row_loss": r0n.square() + rin.square(),
    }


def root_loss(model: torch.nn.Module, batch: FrozenPolicyBatch) -> torch.Tensor:
    values = residual_tensors(model, batch)
    return (
        values["p0_normalized"].square().mean()
        + values["pi_normalized"].square().mean()
    )


# 正式接受判定必须覆盖的逐行张量：残差、归一化残差、Parent P 输出、
# Child Continuation 与逐行 Loss。
FINITE_CHECK_KEYS = (
    "p0",
    "pi",
    "p0_normalized",
    "pi_normalized",
    "p0_output",
    "pi_output",
    "p0_continuation",
    "pi_continuation",
    "row_loss",
)


def _eta_stratified_stats(
    values: torch.Tensor,
    eta: torch.Tensor,
    prefix: str,
) -> Dict[str, float]:
    """Per-current-eta strata stats (diagnostics only; never used for gating).

    ``eta`` is the parent current-eta column aligned row-wise with ``values``.
    Reports MAE / P90 / counts / nonfinite ratio for the eta=0 and eta=1 strata.
    """
    out: Dict[str, float] = {}
    for label, mask in (("eta0", eta <= 0.0), ("eta1", eta > 0.0)):
        n_total = int(mask.sum())
        subset = values[mask]
        finite = subset[torch.isfinite(subset)]
        n_finite = int(finite.numel())
        out[f"{prefix}_{label}_n_total"] = float(n_total)
        out[f"{prefix}_{label}_n_finite"] = float(n_finite)
        out[f"{prefix}_{label}_nonfinite_count"] = float(n_total - n_finite)
        out[f"{prefix}_{label}_nonfinite_ratio"] = (
            float((n_total - n_finite) / n_total) if n_total else 0.0
        )
        if n_finite == 0:
            out[f"{prefix}_{label}_mae"] = float("nan")
            out[f"{prefix}_{label}_p90"] = float("nan")
        else:
            out[f"{prefix}_{label}_mae"] = float(finite.abs().mean())
            out[f"{prefix}_{label}_p90"] = _quantile(finite.abs(), 0.9)
    return out


def evaluate_problem_rows(
    model: torch.nn.Module,
    problem: Sequence[FrozenPolicyBatch],
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """``evaluate_problem`` + 逐 Parent 行的残差/η/source 行数据（CPU）。

    行数据仅供诊断分组（Balanced / Original Mixture / Simulated-only）复用，
    不进入任何正式摘要，也绝不参与 Acceptance 判定。统计输出与
    ``evaluate_problem`` 完全一致。
    """
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
    eta_rows: List[torch.Tensor] = []
    source_rows: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in problem:
            values = residual_tensors(model, batch)
            for key, value in values.items():
                accum.setdefault(key, []).append(value.detach().reshape(-1).cpu())
            eta_rows.append(
                batch.parent[:, 2].detach().reshape(-1).to(torch.float32).cpu()
            )
            if batch.source_id is not None:
                source_rows.append(
                    batch.source_id.detach().reshape(-1).to(torch.long).cpu()
                )
            else:
                source_rows.append(
                    torch.full(
                        (int(batch.parent.shape[0]),), -1, dtype=torch.long
                    )
                )
    rows: Dict[str, torch.Tensor] = {
        "p0_norm": torch.empty(0),
        "pi_norm": torch.empty(0),
        "p0": torch.empty(0),
        "pi": torch.empty(0),
        "eta": torch.empty(0),
        "source": torch.empty(0, dtype=torch.long),
        "row_finite": torch.empty(0, dtype=torch.bool),
    }
    if not accum:
        result: Dict[str, Any] = {
            "p0_norm_mae": float("nan"),
            "pi_norm_mae": float("nan"),
            "normalized_mae": float("nan"),
            "normalized_rmse": float("nan"),
            "finite": False,
            "n_total": 0,
            "n_finite": 0,
            "nonfinite_count": 0,
            "nonfinite_ratio": 0.0,
            "all_finite": False,
        }
        empty = torch.empty(0)
        for prefix in ("p0", "pi", "p0_norm", "pi_norm"):
            result.update(_eta_stratified_stats(empty, empty, prefix))
        return result, rows
    # 逐行 finite 判定覆盖全部正式验收张量；finite-only 统计仅作诊断。
    row_finite: Optional[torch.Tensor] = None
    for key in FINITE_CHECK_KEYS:
        flat = torch.isfinite(torch.cat(accum[key]))
        row_finite = flat if row_finite is None else (row_finite & flat)
    n_total = int(row_finite.numel())
    n_finite = int(row_finite.sum())
    p0n = torch.cat(accum["p0_normalized"])
    pin = torch.cat(accum["pi_normalized"])
    result: Dict[str, Any] = {}
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
            "finite": bool(n_finite == n_total),
            "n_total": n_total,
            "n_finite": n_finite,
            "nonfinite_count": n_total - n_finite,
            "nonfinite_ratio": (
                float((n_total - n_finite) / n_total) if n_total else 0.0
            ),
            "all_finite": bool(n_finite == n_total) and n_total > 0,
        }
    )
    # 分 η 口径（诊断专用，不参与任何验收判定）：P0/PI 的 Frozen-Policy
    # 残差在 eta=0/eta=1 两个 Parent 状态层内独立统计。
    eta_all = torch.cat(eta_rows) if eta_rows else torch.empty(0)
    if eta_all.numel() == p0n.numel():
        result.update(_eta_stratified_stats(torch.cat(accum["p0"]), eta_all, "p0"))
        result.update(_eta_stratified_stats(torch.cat(accum["pi"]), eta_all, "pi"))
        result.update(_eta_stratified_stats(p0n, eta_all, "p0_norm"))
        result.update(_eta_stratified_stats(pin, eta_all, "pi_norm"))
    rows = {
        "p0_norm": p0n,
        "pi_norm": pin,
        "p0": torch.cat(accum["p0"]),
        "pi": torch.cat(accum["pi"]),
        "eta": eta_all,
        "source": (
            torch.cat(source_rows)
            if source_rows
            else torch.empty(0, dtype=torch.long)
        ),
        "row_finite": row_finite,
    }
    return result, rows


def evaluate_problem(
    model: torch.nn.Module,
    problem: Sequence[FrozenPolicyBatch],
) -> Dict[str, Any]:
    return evaluate_problem_rows(model, problem)[0]


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
        row_error = (rhs_snapshot - value_star).abs().reshape(-1)
        row_tol = PARITY_ATOL + PARITY_RTOL * value_star.abs().reshape(-1)
        parity_error_rows = tuple(
            int(index)
            for index in torch.nonzero(
                ~(row_error <= row_tol), as_tuple=False
            ).reshape(-1).tolist()
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
            parity_max_abs_error=parity_error,
            parity_error_rows=parity_error_rows,
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
    source_ids = batch.get("source_id")
    if torch.is_tensor(source_ids) and source_ids.numel() == parent_state.shape[0]:
        source_ids = source_ids.detach().reshape(-1).to(torch.long)
    else:
        source_ids = torch.full(
            (parent_state.shape[0],), -1, dtype=torch.long, device=parent_state.device
        )
    return FrozenPolicyBatch(
        batch_id=int(batch_id),
        p0=p0_branch,
        pi=pi_branch,
        parity_max_abs_error=max(err0, erri),
        parity_passed=bool(ok0 and oki),
        source_id=source_ids,
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
    parity_failures: List[Dict[str, Any]] = []
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
            if not frozen.parity_passed:
                for branch_name, branch in (("p0", frozen.p0), ("pi", frozen.pi)):
                    if branch.parity_error_rows or branch.parity_max_abs_error > 0.0:
                        parity_failures.append({
                            "batch_id": int(batch_id),
                            "branch": branch_name,
                            "parity_max_abs_error": float(branch.parity_max_abs_error),
                            "failing_parent_rows": list(branch.parity_error_rows)[:20],
                            "failing_row_count": len(branch.parity_error_rows),
                        })
    finally:
        model.train(was_training)
    max_error = max(parity_errors) if parity_errors else 0.0
    return problem, {
        "parity_batches": int(len(problem)),
        "parity_max_abs_error": float(max_error),
        "parity_tolerance": PARITY_ATOL,
        "parity_passed": bool(all(parity_flags)) if parity_flags else False,
        "parity_failures": parity_failures,
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


def _params_finite(params: Sequence[torch.nn.Parameter]) -> bool:
    for param in params:
        if param.grad is not None or param.requires_grad:
            if not bool(torch.isfinite(param.detach()).all()):
                return False
    return True


class LBFGSNonFiniteClosureError(RuntimeError):
    """L-BFGS closure hit nonfinite loss/gradient/parameters.

    Raised from inside the closure and caught outside ``optimizer.step()`` so
    a nonfinite value is never fed to the Strong-Wolfe line search and the
    failure is never silently absorbed.
    """


def full_problem_loss(
    model: torch.nn.Module,
    train: Sequence[FrozenPolicyBatch],
    *,
    total_parents: Optional[int] = None,
) -> torch.Tensor:
    """Parent-count weighted full-problem objective for the L-BFGS closure.

    ``root_loss`` already averages over the parents of one batch, so batch
    equal weighting would make the objective depend on the minibatch
    partition.  Weight each batch by ``n_j / N`` so every parent carries the
    same weight:

        L = sum_j (n_j / N) * L_j

    ``total_parents`` lets callers compute the same objective over a *group*
    of batches with the GLOBAL parent count ``N`` (for fused closure passes);
    the resulting group losses sum to the exact full-problem loss.
    """
    if not train:
        raise ValueError("full_problem_loss requires at least one batch")
    if total_parents is None:
        total_parents = sum(int(batch.parent.shape[0]) for batch in train)
    if total_parents <= 0:
        raise ValueError("full_problem_loss requires positive parent counts")
    total = torch.zeros((), device=train[0].parent.device)
    for batch in train:
        weight = float(batch.parent.shape[0]) / float(total_parents)
        total = total + root_loss(model, batch) * weight
    return total


# L-BFGS closure: fuse this many parents into one graph per backward pass.
# Pure execution detail (fewer kernel launches on the tiny MLP); the weighted
# objective is exactly sum_j (n_j/N) L_j regardless of grouping.
CLOSURE_FUSE_PARENTS = 131072


def _grads_finite(params: Sequence[torch.nn.Parameter]) -> bool:
    for param in params:
        if param.grad is not None and not bool(torch.isfinite(param.grad).all()):
            return False
    return True


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
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor], Dict[str, float], int, str]:
    """Adam block with full numerical protection.

    Returns ``(history, best_state, best, completed_steps, status)`` where
    ``status`` is ``ok`` or the explicit numerical failure kind.  A state is
    only eligible as Best when the fixed Validation is entirely finite.
    """
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.0)
    history: List[Dict[str, Any]] = []
    best = evaluate_problem(model, validation)
    best_state = _clone_state(model)
    completed_steps = 0
    status = "ok"
    for step in range(1, steps + 1):
        batch = train[(step - 1) % len(train)]
        optimizer.zero_grad(set_to_none=True)
        loss = root_loss(model, batch)
        if not torch.isfinite(loss):
            status = "nonfinite_loss"
            history.append(
                {"phase": "adam", "step": step, "loss": float("nan"), "failure": status}
            )
            break
        loss.backward()
        raw_grad = _grad_norm(params)
        if not math.isfinite(raw_grad) or not _grads_finite(params):
            status = "nonfinite_gradient"
            history.append(
                {
                    "phase": "adam",
                    "step": step,
                    "loss": float(loss.detach()),
                    "raw_grad_norm": raw_grad,
                    "failure": status,
                }
            )
            # 非有限梯度绝不执行 optimizer.step()，也不保存为 Best。
            optimizer.zero_grad(set_to_none=True)
            break
        torch.nn.utils.clip_grad_norm_(params, clip)
        optimizer.step()
        if not _params_finite(params):
            status = "nonfinite_parameters"
            history.append(
                {"phase": "adam", "step": step, "failure": status}
            )
            break
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
                    "val_all_finite": bool(val["all_finite"]),
                }
            )
            if (
                val["all_finite"]
                and math.isfinite(val["normalized_mae"])
                and val["normalized_mae"] < best["normalized_mae"]
            ):
                best, best_state = val, _clone_state(model)
    return history, best_state, best, completed_steps, status


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
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor], Dict[str, float], str, Dict[str, int]]:
    """L-BFGS in bounded chunks with closure-level numerical protection.

    Returns ``(history, best_state, best, stop_reason, counters)`` where
    ``counters`` distinguishes the iteration budget from actual iterations
    and closure calls.  A nonfinite closure or nonfinite chunk result restores
    the last finite Best state and stops; the validation is re-evaluated on
    the restored state before returning.
    """
    if chunk_iterations <= 0:
        raise ValueError("chunk_iterations must be positive")
    best = evaluate_problem(model, validation)
    best_state = _clone_state(model)
    history: List[Dict[str, Any]] = []
    counters = {
        "LBFGS_ITERATION_BUDGET": int(max_iter),
        "LBFGS_ACTUAL_ITERATIONS": 0,
        "LBFGS_CLOSURE_CALLS": 0,
        "LBFGS_NONFINITE_CLOSURES": 0,
        "LBFGS_NONFINITE_CLOSURE_ABORT": 0,
    }
    total_parents = sum(int(batch.parent.shape[0]) for batch in train)
    # 执行层融合：每个 backward 覆盖约 CLOSURE_FUSE_PARENTS 个 parent 的
    # 组合图（等权公式完全不变），减少小 MLP 的 kernel launch 开销。
    fused_groups: List[List[FrozenPolicyBatch]] = []
    group: List[FrozenPolicyBatch] = []
    group_parents = 0
    for batch in train:
        group.append(batch)
        group_parents += int(batch.parent.shape[0])
        if group_parents >= CLOSURE_FUSE_PARENTS:
            fused_groups.append(group)
            group, group_parents = [], 0
    if group:
        fused_groups.append(group)
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
        block_nonfinite = 0
        started = time.perf_counter()

        def closure() -> torch.Tensor:
            nonlocal block_calls, block_nonfinite
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=train[0].parent.device)
            # Parent-count weighting: each parent carries weight 1/N regardless
            # of the minibatch partition (root_loss averages within a batch).
            # 分组仅是执行层融合，权重仍用全局 N。
            for group_batches in fused_groups:
                group_loss = full_problem_loss(
                    model, group_batches, total_parents=total_parents
                )
                group_loss.backward()
                loss = loss + group_loss.detach()
            block_calls += 1
            counters["LBFGS_CLOSURE_CALLS"] += 1
            # 非有限 Loss/梯度/参数必须立即中止求解：绝不清空梯度后继续把
            # NaN/Inf 返回给 Strong Wolfe Line Search。
            if (
                not torch.isfinite(loss)
                or not _grads_finite(params)
                or not _params_finite(params)
            ):
                block_nonfinite += 1
                counters["LBFGS_NONFINITE_CLOSURES"] += 1
                optimizer.zero_grad(set_to_none=True)
                raise LBFGSNonFiniteClosureError(
                    "L-BFGS closure produced nonfinite loss/gradient/parameters"
                )
            return loss

        try:
            optimizer.step(closure)
        except LBFGSNonFiniteClosureError:
            # 保守回退：停止本 Chunk 与剩余 L-BFGS，恢复最近的有限 Best
            # State 并验证恢复，再重新评价 Validation。
            counters["LBFGS_NONFINITE_CLOSURE_ABORT"] += 1
            if train[0].parent.is_cuda:
                torch.cuda.synchronize(train[0].parent.device)
            _restore_state(model, best_state)
            final = evaluate_problem(model, validation)
            history.append(
                {
                    "phase": "lbfgs_nonfinite_closure_abort",
                    "block_budget": block_budget,
                    "block_closure_calls": block_calls,
                    "block_nonfinite_closures": block_nonfinite,
                    "restored_best_state": True,
                    "val_normalized_mae": final["normalized_mae"],
                    "val_all_finite": bool(final["all_finite"]),
                }
            )
            stop_reason = "lbfgs_nonfinite_closure"
            break
        if train[0].parent.is_cuda:
            torch.cuda.synchronize(train[0].parent.device)
        elapsed = time.perf_counter() - started
        # torch LBFGS 维护每个 block 的真实迭代计数；closure calls ≠ iterations。
        actual_iter = 0
        for param in params:
            state_entry = optimizer.state.get(param, {})
            if isinstance(state_entry, dict) and "n_iter" in state_entry:
                actual_iter = max(actual_iter, int(state_entry["n_iter"]))
                break
        counters["LBFGS_ACTUAL_ITERATIONS"] += int(actual_iter)
        completed_budget += block_budget
        # 防御性双保险：即使 closure 未触发异常，本 Chunk 内出现过任何非有限
        # closure 也必须按失败处理，不能仅依赖最终 val["all_finite"]。
        if block_nonfinite > 0:
            counters["LBFGS_NONFINITE_CLOSURE_ABORT"] += 1
            _restore_state(model, best_state)
            final = evaluate_problem(model, validation)
            history.append(
                {
                    "phase": "lbfgs_nonfinite_closure_abort",
                    "block_nonfinite_closures": block_nonfinite,
                    "restored_best_state": True,
                    "val_normalized_mae": final["normalized_mae"],
                    "val_all_finite": bool(final["all_finite"]),
                }
            )
            stop_reason = "lbfgs_nonfinite_closure"
            break
        val = evaluate_problem(model, validation)
        history.append(
            {
                "phase": "lbfgs_chunk",
                "optimizer_budget": completed_budget,
                "block_budget": block_budget,
                "block_closure_calls": block_calls,
                "block_nonfinite_closures": block_nonfinite,
                "block_actual_iterations": int(actual_iter),
                "seconds": elapsed,
                "raw_grad_norm": _grad_norm(params),
                "parameter_norm": _param_norm(params),
                "val_normalized_mae": val["normalized_mae"],
                "val_p0_norm_mae": val["p0_norm_mae"],
                "val_pi_norm_mae": val["pi_norm_mae"],
                "val_all_finite": bool(val["all_finite"]),
            }
        )
        if (
            val["all_finite"]
            and math.isfinite(val["normalized_mae"])
            and val["normalized_mae"] < best["normalized_mae"]
        ):
            best, best_state = val, _clone_state(model)
        if not bool(val["all_finite"]) or not _params_finite(params):
            # 非有限 chunk 结果绝不污染 Best State：恢复最近一次有限
            # Best 并重新评价后退出。
            stop_reason = "nonfinite_chunk_restored"
            _restore_state(model, best_state)
            final = evaluate_problem(model, validation)
            history.append(
                {
                    "phase": "lbfgs_restore",
                    "restored_best_state": True,
                    "val_normalized_mae": final["normalized_mae"],
                    "val_all_finite": bool(final["all_finite"]),
                }
            )
            if final["all_finite"] and math.isfinite(final["normalized_mae"]):
                if final["normalized_mae"] < best["normalized_mae"]:
                    best = final
            break
        if target_residual is not None and val["normalized_mae"] <= target_residual:
            stop_reason = "target_residual_reached"
            break
    return history, best_state, best, stop_reason, counters


# ---------------------------------------------------------------------------
# One P policy iteration
# ---------------------------------------------------------------------------


def _fresh_greedy_rows(
    episode: Any,
    model: torch.nn.Module,
    teacher: BPGridTeacher,
    batches: Sequence[Dict[str, torch.Tensor]],
    *,
    q_model: Optional[torch.nn.Module] = None,
) -> Dict[str, torch.Tensor]:
    """逐 Parent 行的 Fresh-Greedy 归一化残差（CPU）+ η/source/finite 掩码。

    与 ``fresh_greedy_residual`` 同一套数学；行数据供 Natural Validation
    诊断分组（Simulated-only 等）复用，不进入正式摘要。
    """
    q_model = q_model if q_model is not None else model
    r0: List[torch.Tensor] = []
    ri: List[torch.Tensor] = []
    eta_sinks: List[torch.Tensor] = []
    source_sinks: List[torch.Tensor] = []
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
                    sink.append(
                        ((online - target["value_star"]) / scale)
                        .detach()
                        .reshape(-1)
                        .cpu()
                    )
                n_rows = int(parent_state.shape[0])
                eta_sinks.append(
                    parent_state[:, 2].detach().reshape(-1).to(torch.float32).cpu()
                )
                source_ids = batch.get("source_id")
                if torch.is_tensor(source_ids) and source_ids.numel() == n_rows:
                    source_sinks.append(
                        source_ids.detach().reshape(-1).to(torch.long).cpu()
                    )
                else:
                    source_sinks.append(torch.full((n_rows,), -1, dtype=torch.long))
    finally:
        model.train(was_training)
    r0_flat = torch.cat(r0) if r0 else torch.empty(0)
    ri_flat = torch.cat(ri) if ri else torch.empty(0)
    return {
        "fg_p0_norm": r0_flat,
        "fg_pi_norm": ri_flat,
        "eta": (
            torch.cat(eta_sinks).reshape(-1) if eta_sinks else torch.empty(0)
        ),
        "source": (
            torch.cat(source_sinks).reshape(-1)
            if source_sinks
            else torch.empty(0, dtype=torch.long)
        ),
        "row_finite": (
            torch.isfinite(r0_flat) & torch.isfinite(ri_flat)
            if r0_flat.numel()
            else torch.empty(0, dtype=torch.bool)
        ),
    }


def _fresh_stats_from_rows(rows: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    r0_flat = rows["fg_p0_norm"]
    ri_flat = rows["fg_pi_norm"]
    eta_flat = rows["eta"]
    if r0_flat.numel() == 0:
        result: Dict[str, Any] = {
            "fresh_greedy_p0_norm_mae": float("nan"),
            "fresh_greedy_pi_norm_mae": float("nan"),
            "all_finite": False,
        }
        empty = torch.empty(0)
        result.update(_eta_stratified_stats(empty, empty, "fresh_greedy_p0_norm"))
        result.update(_eta_stratified_stats(empty, empty, "fresh_greedy_pi_norm"))
        return result
    all_finite = bool(
        torch.isfinite(r0_flat).all() and torch.isfinite(ri_flat).all()
    )
    result = {
        "fresh_greedy_p0_norm_mae": float(r0_flat.abs().mean()),
        "fresh_greedy_pi_norm_mae": float(ri_flat.abs().mean()),
        "fresh_greedy_p0_norm_p90": _quantile(r0_flat.abs(), 0.9),
        "fresh_greedy_pi_norm_p90": _quantile(ri_flat.abs(), 0.9),
        "all_finite": all_finite,
    }
    # 分 η 口径（诊断专用）：Fresh-Greedy 残差在 eta=0/1 层内独立统计。
    if eta_flat.numel() == r0_flat.numel():
        result.update(
            _eta_stratified_stats(r0_flat, eta_flat, "fresh_greedy_p0_norm")
        )
        result.update(
            _eta_stratified_stats(ri_flat, eta_flat, "fresh_greedy_pi_norm")
        )
    return result


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
    return _fresh_stats_from_rows(
        _fresh_greedy_rows(episode, model, teacher, batches, q_model=q_model)
    )


# ---------------------------------------------------------------------------
# Natural Validation diagnostic（纯诊断，绝不参与 P Acceptance）
# ---------------------------------------------------------------------------
#
# 三种 Validation 分布（Prompt §4）：
#   * Balanced Validation      — P-only η Balance 后的主 Validation（正式口径）；
#   * Original Mixture         — 未 η 重分配的原始 80/20 混合分布；
#   * Simulated-only           — Original Mixture 中 source_id=0 的子集。
# 诊断只从 Original Validation 里取有上限的固定 Parent 子集（固定 Seed、
# Parent/Child 完整配对、不重采样 η、不消耗全局 RNG），分块构建/评估并在
# 优化期间把问题存放在 CPU，绝不把完整诊断问题常驻 GPU。

NOT_EVALUABLE = "not_evaluable"


def _tree_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        seq = [_tree_to_device(item, device) for item in obj]
        return tuple(seq) if isinstance(obj, tuple) else seq
    if isinstance(obj, dict):
        return {key: _tree_to_device(value, device) for key, value in obj.items()}
    return obj


def _frozen_batch_to(
    batch: FrozenPolicyBatch, device: torch.device
) -> FrozenPolicyBatch:
    kwargs = {
        name: _tree_to_device(getattr(batch, name), device)
        for name in batch.__dataclass_fields__
    }
    return FrozenPolicyBatch(**kwargs)


def _batch_dict_to_device(
    batch: Dict[str, Any], device: torch.device
) -> Dict[str, Any]:
    return {key: _tree_to_device(value, device) for key, value in batch.items()}


def _batch_dict_to_cpu(batch: Dict[str, Any]) -> Dict[str, Any]:
    return _batch_dict_to_device(batch, torch.device("cpu"))


def _select_diagnostic_batches(
    batches: Sequence[Dict[str, Any]],
    *,
    max_parents: int,
    seed: int,
    episode_id: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """从 Original Validation 里选固定的有上限 Parent 行子集（CPU、可复现）。

    * 固定 Seed 的 CPU ``torch.Generator``，不消耗全局 RNG；
    * Parent 与全部 Child 行整体配对切片；
    * 不重采样 η，不改变任何行的内容；
    * Parent/Child 行数不匹配的 batch 整体丢弃并如实计数，绝不凭空补样本。
    """
    pairable: List[Tuple[int, Dict[str, Any], int]] = []
    requested_parents = 0
    dropped_unpairable_batches = 0
    for batch in batches:
        parent = batch.get("parent")
        if not torch.is_tensor(parent) or parent.ndim < 2 or parent.shape[0] == 0:
            continue
        n_rows = int(parent.shape[0])
        children = batch.get("children")
        if not children and batch.get("child0") is not None:
            children = [batch.get("child0"), batch.get("child1")]
        children = children or []
        if not children or any(
            (not torch.is_tensor(child)) or int(child.shape[0]) != n_rows
            for child in children
        ):
            dropped_unpairable_batches += 1
            continue
        requested_parents += n_rows
        pairable.append((n_rows, batch, len(pairable)))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(episode_id) * 100003)
    budget = max(1, int(max_parents))
    if requested_parents <= budget:
        chosen = {
            index: torch.arange(n_rows, dtype=torch.long)
            for n_rows, _batch, index in pairable
        }
    else:
        flat = torch.randperm(requested_parents, generator=generator)[:budget]
        chosen_lists: Dict[int, List[int]] = {}
        offset = 0
        for n_rows, _batch, index in pairable:
            local = flat[(flat >= offset) & (flat < offset + n_rows)] - offset
            if int(local.numel()) > 0:
                chosen_lists[index] = local.sort().values.tolist()
            offset += n_rows
        chosen = {
            index: torch.tensor(rows, dtype=torch.long)
            for index, rows in chosen_lists.items()
        }

    subset: List[Dict[str, Any]] = []
    source_counts = {"source0": 0, "source1": 0, "unknown": 0}
    selected_parents = 0
    for n_rows, batch, index in pairable:
        rows = chosen.get(index)
        if rows is None:
            continue
        rows = rows.to(dtype=torch.long)
        sliced: Dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[0]) == n_rows:
                sliced[key] = value.detach().index_select(0, rows.to(value.device)).cpu()
            elif (
                isinstance(value, (list, tuple))
                and value
                and all(
                    torch.is_tensor(item) and int(item.shape[0]) == n_rows
                    for item in value
                )
            ):
                sliced[key] = [
                    item.detach().index_select(0, rows.to(item.device)).cpu()
                    for item in value
                ]
            else:
                sliced[key] = value
        subset.append(sliced)
        selected_parents += int(rows.numel())
        source_ids = batch.get("source_id")
        if torch.is_tensor(source_ids) and int(source_ids.numel()) == n_rows:
            picked = source_ids.detach().reshape(-1).index_select(
                0, rows.to(source_ids.device)
            )
            source_counts["source0"] += int((picked == 0).sum())
            source_counts["source1"] += int((picked == 1).sum())
            source_counts["unknown"] += int(
                (picked != 0).sum() - (picked == 1).sum()
            )
        else:
            source_counts["unknown"] += int(rows.numel())

    sampling = {
        "sampling_method": "uniform_parent_row_subset_without_replacement",
        "stratified": False,
        "eta_rebalanced": False,
        "preserves_global_rng": True,
        "requested_parents": int(requested_parents),
        "selected_parents": int(selected_parents),
        "sampling_ratio": (
            float(selected_parents / requested_parents)
            if requested_parents
            else 0.0
        ),
        "max_parents": int(budget),
        "seed": int(seed),
        "episode_id": int(episode_id),
        "source_counts": dict(source_counts),
        "dropped_unpairable_batches": int(dropped_unpairable_batches),
    }
    return subset, sampling


def _chunk_batch_dicts(
    batches: Sequence[Dict[str, Any]], chunk_size: int
) -> List[List[Dict[str, Any]]]:
    """把子集 batch 按 Parent 行数切成 ≤ chunk_size 的构建/评估块。"""
    limit = max(1, int(chunk_size))
    chunks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_rows = 0
    for batch in batches:
        n_rows = int(batch["parent"].shape[0])
        if current and current_rows + n_rows > limit:
            chunks.append(current)
            current = []
            current_rows = 0
        if n_rows <= limit:
            current.append(batch)
            current_rows += n_rows
            continue
        # 单个 batch 超过 chunk 上限时按行再切。
        for start in range(0, n_rows, limit):
            stop = min(start + limit, n_rows)
            sub: Dict[str, Any] = {}
            for key, value in batch.items():
                if torch.is_tensor(value) and int(value.shape[0]) == n_rows:
                    sub[key] = value[start:stop]
                elif (
                    isinstance(value, (list, tuple))
                    and value
                    and all(
                        torch.is_tensor(item) and int(item.shape[0]) == n_rows
                        for item in value
                    )
                ):
                    sub[key] = [item[start:stop] for item in value]
                else:
                    sub[key] = value
            if current:
                chunks.append(current)
                current = []
                current_rows = 0
            chunks.append([sub])
    if current:
        chunks.append(current)
    return chunks


def _eval_frozen_rows_chunked(
    model: torch.nn.Module,
    problem_cpu: Sequence[FrozenPolicyBatch],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """逐 chunk（逐 batch）把 CPU 上的冻结问题搬回 GPU 求残差并立即释放。"""
    p0_rows: List[torch.Tensor] = []
    pi_rows: List[torch.Tensor] = []
    p0n_rows: List[torch.Tensor] = []
    pin_rows: List[torch.Tensor] = []
    eta_rows: List[torch.Tensor] = []
    source_rows: List[torch.Tensor] = []
    finite_rows: List[torch.Tensor] = []
    with torch.no_grad():
        for frozen_cpu in problem_cpu:
            frozen = _frozen_batch_to(frozen_cpu, device)
            values = residual_tensors(model, frozen)
            row_finite: Optional[torch.Tensor] = None
            for key in FINITE_CHECK_KEYS:
                flat = torch.isfinite(values[key].reshape(-1))
                row_finite = flat if row_finite is None else (row_finite & flat)
            n_rows = int(frozen.parent.shape[0])
            p0_rows.append(values["p0"].reshape(-1).detach().cpu())
            pi_rows.append(values["pi"].reshape(-1).detach().cpu())
            p0n_rows.append(values["p0_normalized"].reshape(-1).detach().cpu())
            pin_rows.append(values["pi_normalized"].reshape(-1).detach().cpu())
            eta_rows.append(
                frozen.parent[:, 2].detach().reshape(-1).to(torch.float32).cpu()
            )
            if frozen.source_id is not None:
                source_rows.append(
                    frozen.source_id.detach().reshape(-1).to(torch.long).cpu()
                )
            else:
                source_rows.append(torch.full((n_rows,), -1, dtype=torch.long))
            finite_rows.append(row_finite.detach().cpu())
            del frozen, values
    return {
        "p0": torch.cat(p0_rows) if p0_rows else torch.empty(0),
        "pi": torch.cat(pi_rows) if pi_rows else torch.empty(0),
        "p0_norm": torch.cat(p0n_rows) if p0n_rows else torch.empty(0),
        "pi_norm": torch.cat(pin_rows) if pin_rows else torch.empty(0),
        "eta": torch.cat(eta_rows) if eta_rows else torch.empty(0),
        "source": (
            torch.cat(source_rows) if source_rows else torch.empty(0, dtype=torch.long)
        ),
        "row_finite": (
            torch.cat(finite_rows) if finite_rows else torch.empty(0, dtype=torch.bool)
        ),
    }


def _fresh_greedy_rows_chunked(
    episode: Any,
    model: torch.nn.Module,
    teacher: BPGridTeacher,
    subset_cpu: Sequence[Dict[str, Any]],
    device: torch.device,
    chunk_size: int,
    *,
    q_model: Optional[torch.nn.Module] = None,
) -> Dict[str, torch.Tensor]:
    """对 CPU 上的固定诊断子集分块执行 Fresh-Greedy 残差（同一子集可复用）。"""
    chunks = _chunk_batch_dicts(subset_cpu, chunk_size)
    r0: List[torch.Tensor] = []
    ri: List[torch.Tensor] = []
    eta: List[torch.Tensor] = []
    source: List[torch.Tensor] = []
    finite: List[torch.Tensor] = []
    for chunk in chunks:
        rows = _fresh_greedy_rows(
            episode,
            model,
            teacher,
            [_batch_dict_to_device(batch, device) for batch in chunk],
            q_model=q_model,
        )
        r0.append(rows["fg_p0_norm"])
        ri.append(rows["fg_pi_norm"])
        eta.append(rows["eta"])
        source.append(rows["source"])
        finite.append(rows["row_finite"])
    return {
        "fg_p0_norm": torch.cat(r0) if r0 else torch.empty(0),
        "fg_pi_norm": torch.cat(ri) if ri else torch.empty(0),
        "eta": torch.cat(eta) if eta else torch.empty(0),
        "source": torch.cat(source) if source else torch.empty(0, dtype=torch.long),
        "row_finite": torch.cat(finite) if finite else torch.empty(0, dtype=torch.bool),
    }


def _frozen_stats_from_rows(rows: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """与 ``evaluate_problem`` 同形状的统计（由行数据纯 CPU 聚合）。"""
    if rows["p0_norm"].numel() == 0:
        result: Dict[str, Any] = {
            "p0_norm_mae": float("nan"),
            "pi_norm_mae": float("nan"),
            "normalized_mae": float("nan"),
            "normalized_rmse": float("nan"),
            "finite": False,
            "n_total": 0,
            "n_finite": 0,
            "nonfinite_count": 0,
            "nonfinite_ratio": 0.0,
            "all_finite": False,
        }
        empty = torch.empty(0)
        for prefix in ("p0", "pi", "p0_norm", "pi_norm"):
            result.update(_eta_stratified_stats(empty, empty, prefix))
        return result

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

    p0n = rows["p0_norm"]
    pin = rows["pi_norm"]
    row_finite = rows["row_finite"]
    n_total = int(row_finite.numel())
    n_finite = int(row_finite.sum())
    result = {}
    result.update(_stats(rows["p0"], "p0"))
    result.update(_stats(rows["pi"], "pi"))
    result.update(_stats(p0n, "p0_norm"))
    result.update(_stats(pin, "pi_norm"))
    result.update(
        {
            "normalized_mae": 0.5 * (result["p0_norm_mae"] + result["pi_norm_mae"]),
            "normalized_rmse": math.sqrt(
                0.5 * (result["p0_norm_rmse"] ** 2 + result["pi_norm_rmse"] ** 2)
            ),
            "finite": bool(n_finite == n_total),
            "n_total": n_total,
            "n_finite": n_finite,
            "nonfinite_count": n_total - n_finite,
            "nonfinite_ratio": (
                float((n_total - n_finite) / n_total) if n_total else 0.0
            ),
            "all_finite": bool(n_finite == n_total) and n_total > 0,
        }
    )
    eta_all = rows["eta"]
    if eta_all.numel() == p0n.numel():
        result.update(_eta_stratified_stats(rows["p0"], eta_all, "p0"))
        result.update(_eta_stratified_stats(rows["pi"], eta_all, "pi"))
        result.update(_eta_stratified_stats(p0n, eta_all, "p0_norm"))
        result.update(_eta_stratified_stats(pin, eta_all, "pi_norm"))
    return result


def _mae_p90(values: torch.Tensor) -> Tuple[float, float]:
    flat = values.detach().reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return float("nan"), float("nan")
    return float(finite.abs().mean()), _quantile(finite.abs(), 0.9)


def _diag_view(
    label: str,
    mask: torch.Tensor,
    frozen_init: Dict[str, torch.Tensor],
    frozen_final: Dict[str, torch.Tensor],
    fresh_init: Dict[str, torch.Tensor],
    fresh_final: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """一种 Validation 分布的完整视图（Prompt §4.4 字段）。"""
    n_parents = int(mask.sum())
    if n_parents == 0:
        return {
            "source": label,
            "n_parents": 0,
            "eta0_count": 0,
            "eta1_count": 0,
            "eta1_share": float("nan"),
            "evaluable": False,
            "status": NOT_EVALUABLE,
            "reason": "empty_group",
        }
    eta = frozen_final["eta"][mask]
    combined_finite = (
        frozen_init["row_finite"][mask]
        & frozen_final["row_finite"][mask]
        & fresh_init["row_finite"][mask]
        & fresh_final["row_finite"][mask]
    )
    n_nonfinite = int((~combined_finite).sum())
    p0_frozen_mae, p0_frozen_p90 = _mae_p90(frozen_final["p0_norm"][mask])
    pi_frozen_mae, pi_frozen_p90 = _mae_p90(frozen_final["pi_norm"][mask])
    p0_fresh_mae, p0_fresh_p90 = _mae_p90(fresh_final["fg_p0_norm"][mask])
    pi_fresh_mae, pi_fresh_p90 = _mae_p90(fresh_final["fg_pi_norm"][mask])
    p0_frozen_i_mae, p0_frozen_i_p90 = _mae_p90(frozen_init["p0_norm"][mask])
    pi_frozen_i_mae, pi_frozen_i_p90 = _mae_p90(frozen_init["pi_norm"][mask])
    p0_fresh_i_mae, p0_fresh_i_p90 = _mae_p90(fresh_init["fg_p0_norm"][mask])
    pi_fresh_i_mae, pi_fresh_i_p90 = _mae_p90(fresh_init["fg_pi_norm"][mask])
    eta0_count = int((eta <= 0.5).sum())
    eta1_count = int((eta > 0.5).sum())
    return {
        "source": label,
        "n_parents": n_parents,
        "eta0_count": eta0_count,
        "eta1_count": eta1_count,
        "eta1_share": float(eta1_count / n_parents),
        "p0_frozen_mae": p0_frozen_mae,
        "p0_frozen_p90": p0_frozen_p90,
        "pi_frozen_mae": pi_frozen_mae,
        "pi_frozen_p90": pi_frozen_p90,
        "p0_fresh_greedy_mae": p0_fresh_mae,
        "p0_fresh_greedy_p90": p0_fresh_p90,
        "pi_fresh_greedy_mae": pi_fresh_mae,
        "pi_fresh_greedy_p90": pi_fresh_p90,
        "nonfinite_ratio": float(n_nonfinite / n_parents),
        "evaluable": bool(n_nonfinite == 0),
        "status": "completed",
        "frozen_initial": {
            "p0_mae": p0_frozen_i_mae,
            "p0_p90": p0_frozen_i_p90,
            "pi_mae": pi_frozen_i_mae,
            "pi_p90": pi_frozen_i_p90,
        },
        "frozen_final": {
            "p0_mae": p0_frozen_mae,
            "p0_p90": p0_frozen_p90,
            "pi_mae": pi_frozen_mae,
            "pi_p90": pi_frozen_p90,
        },
        "fresh_greedy_initial": {
            "p0_mae": p0_fresh_i_mae,
            "p0_p90": p0_fresh_i_p90,
            "pi_mae": pi_fresh_i_mae,
            "pi_p90": pi_fresh_i_p90,
        },
        "fresh_greedy_final": {
            "p0_mae": p0_fresh_mae,
            "p0_p90": p0_fresh_p90,
            "pi_mae": pi_fresh_mae,
            "pi_p90": pi_fresh_p90,
        },
    }


def _diag_improvement_flags(
    frozen_init: Dict[str, torch.Tensor],
    frozen_final: Dict[str, torch.Tensor],
    fresh_init: Dict[str, torch.Tensor],
    fresh_final: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Frozen / Fresh-Greedy 前后改善判断（三态；绝不把失败写成改善）。"""
    n = int(frozen_final["p0_norm"].numel())
    combined_finite = (
        n > 0
        and bool(frozen_init["row_finite"].all())
        and bool(frozen_final["row_finite"].all())
        and bool(fresh_init["row_finite"].all())
        and bool(fresh_final["row_finite"].all())
    )

    def _combined_frozen(rows: Dict[str, torch.Tensor]) -> float:
        p0, _ = _mae_p90(rows["p0_norm"])
        pi, _ = _mae_p90(rows["pi_norm"])
        return 0.5 * (p0 + pi)

    def _combined_fresh(rows: Dict[str, torch.Tensor]) -> float:
        p0, _ = _mae_p90(rows["fg_p0_norm"])
        pi, _ = _mae_p90(rows["fg_pi_norm"])
        return 0.5 * (p0 + pi)

    if not combined_finite:
        return {
            "natural_frozen_improved": NOT_EVALUABLE,
            "natural_frozen_worsened": NOT_EVALUABLE,
            "natural_fresh_greedy_improved": NOT_EVALUABLE,
            "natural_fresh_greedy_worsened": NOT_EVALUABLE,
            "natural_full_bellman_improvement": NOT_EVALUABLE,
        }
    frozen_before = _combined_frozen(frozen_init)
    frozen_after = _combined_frozen(frozen_final)
    fresh_before = _combined_fresh(fresh_init)
    fresh_after = _combined_fresh(fresh_final)
    frozen_improved = bool(frozen_after < frozen_before)
    frozen_worsened = bool(frozen_after > frozen_before)
    fresh_improved = bool(fresh_after < fresh_before)
    fresh_worsened = bool(fresh_after > fresh_before)
    return {
        "natural_frozen_improved": frozen_improved,
        "natural_frozen_worsened": frozen_worsened,
        "natural_fresh_greedy_improved": fresh_improved,
        "natural_fresh_greedy_worsened": fresh_worsened,
        "natural_full_bellman_improvement": bool(
            frozen_improved and fresh_improved
        ),
    }


def _assemble_natural_block(
    *,
    mode: str,
    natural_validation_reused: bool,
    calibers_identical: bool,
    sampling: Dict[str, Any],
    parity: Optional[Dict[str, Any]],
    frozen_init: Dict[str, torch.Tensor],
    frozen_final: Dict[str, torch.Tensor],
    fresh_init: Dict[str, torch.Tensor],
    fresh_final: Dict[str, torch.Tensor],
    validation_rows: Optional[Dict[str, torch.Tensor]] = None,
    frozen_initial_stats: Optional[Dict[str, Any]] = None,
    frozen_final_stats: Optional[Dict[str, Any]] = None,
    fresh_initial_stats: Optional[Dict[str, Any]] = None,
    fresh_final_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """组装 Natural Validation 块（复用与独立两种模式共用）。"""
    frozen_initial_stats = frozen_initial_stats or _frozen_stats_from_rows(frozen_init)
    frozen_final_stats = frozen_final_stats or _frozen_stats_from_rows(frozen_final)
    fresh_initial_stats = fresh_initial_stats or _fresh_stats_from_rows(fresh_init)
    fresh_final_stats = fresh_final_stats or _fresh_stats_from_rows(fresh_final)

    n = int(frozen_final["p0_norm"].numel())
    eta = frozen_final["eta"]
    source = frozen_final["source"]
    source_counts = {
        "source0": int((source == 0).sum()) if n else 0,
        "source1": int((source == 1).sum()) if n else 0,
        "unknown": int((source != 0).sum() - (source == 1).sum()) if n else 0,
    }
    rows_finite = (
        frozen_init["row_finite"]
        & frozen_final["row_finite"]
        & fresh_init["row_finite"]
        & fresh_final["row_finite"]
    )
    n_finite_rows = int(rows_finite.sum()) if n else 0
    if n == 0:
        diagnostic_status = NOT_EVALUABLE
        failure_kind = "empty_diagnostic_sample"
        failure_detail = "no evaluable parent rows in the diagnostic subset"
    elif n_finite_rows == 0:
        diagnostic_status = NOT_EVALUABLE
        failure_kind = "no_finite_rows"
        failure_detail = "all diagnostic rows nonfinite"
    else:
        diagnostic_status = "completed"
        failure_kind = None
        failure_detail = None

    all_mask = torch.ones(n, dtype=torch.bool)
    sim_mask = source == 0
    views: Dict[str, Any] = {}
    if validation_rows is not None:
        v_n = int(validation_rows["frozen_final"]["p0_norm"].numel())
        v_mask = torch.ones(v_n, dtype=torch.bool)
        v_label = (
            "natural_parent_eta"
            if (calibers_identical or natural_validation_reused)
            else "balanced_parent_eta"
        )
        views["balanced_validation"] = _diag_view(
            v_label,
            v_mask,
            validation_rows["frozen_init"],
            validation_rows["frozen_final"],
            validation_rows["fresh_init"],
            validation_rows["fresh_final"],
        )
    views["original_mixture"] = _diag_view(
        "natural_parent_eta_mixture",
        all_mask,
        frozen_init,
        frozen_final,
        fresh_init,
        fresh_final,
    )
    if int(sim_mask.sum()) > 0:
        views["simulated_only"] = _diag_view(
            "simulated_only_source0",
            sim_mask,
            frozen_init,
            frozen_final,
            fresh_init,
            fresh_final,
        )
    else:
        views["simulated_only"] = {
            "source": "simulated_only_source0",
            "n_parents": 0,
            "status": NOT_EVALUABLE,
            "evaluable": False,
            "reason": "no_simulated_parent_groups",
        }

    flags = _diag_improvement_flags(frozen_init, frozen_final, fresh_init, fresh_final)
    total_initial = float(frozen_initial_stats.get("normalized_mae", float("nan")))
    total_final = float(frozen_final_stats.get("normalized_mae", float("nan")))
    if flags["natural_frozen_worsened"] == NOT_EVALUABLE:
        regression_observed: Any = NOT_EVALUABLE
    else:
        regression_observed = bool(
            flags["natural_frozen_worsened"] is True
            or flags["natural_fresh_greedy_worsened"] is True
        )
    return {
        "diagnostic_status": diagnostic_status,
        "diagnostic_failure_kind": failure_kind,
        "diagnostic_failure_detail": failure_detail,
        "diagnostic_n_parents": int(n),
        "diagnostic_source_counts": source_counts,
        "diagnostic_finite_rows": n_finite_rows,
        "natural_validation_reused": bool(natural_validation_reused),
        "calibers_identical": bool(calibers_identical),
        "caliber": "natural_parent_eta",
        "p_eta_sampling_method": "natural_parent_eta",
        "diagnostic_mode": mode,
        "sampling": sampling,
        "n_parents": int(n),
        "eta0_count": int((eta <= 0.5).sum()) if n else 0,
        "eta1_count": int((eta > 0.5).sum()) if n else 0,
        "eta1_share": float((eta > 0.5).float().mean()) if n else 0.0,
        "total_residual_initial": total_initial,
        "total_residual_final": total_final,
        "frozen_improved": flags["natural_frozen_improved"],
        "natural_worsened": flags["natural_frozen_worsened"],
        "natural_validation_regression_observed": regression_observed,
        **flags,
        "frozen_policy_initial": frozen_initial_stats,
        "frozen_policy_final": frozen_final_stats,
        "fresh_greedy_initial": fresh_initial_stats,
        "fresh_greedy_final": fresh_final_stats,
        "views": views,
        "parity": parity,
    }


def natural_diag_prepare(
    episode: Any,
    model: torch.nn.Module,
    teacher: BPGridTeacher,
    batches: Sequence[Dict[str, Any]],
    *,
    q_model: Optional[torch.nn.Module] = None,
    max_parents: int = 4096,
    chunk_size: int = 512,
    seed: int = 97531,
) -> Dict[str, Any]:
    """独立 Natural 诊断第一步：固定子集 + 冻结问题（CPU 驻留）+ 初始评估。

    冻结问题在迭代起点构建一次，前后比较共用同一冻结政策问题（Prompt §5.2）。
    """
    device = next(model.parameters()).device
    subset, sampling = _select_diagnostic_batches(
        batches,
        max_parents=max_parents,
        seed=seed,
        episode_id=int(getattr(episode, "episode_id", 0)),
    )
    if not subset:
        return {
            "status": NOT_EVALUABLE,
            "failure_kind": "empty_diagnostic_subset",
            "failure_detail": "diagnostic subset selection produced no parent rows",
            "sampling": sampling,
        }
    chunks = _chunk_batch_dicts(subset, chunk_size)
    problem_cpu: List[FrozenPolicyBatch] = []
    parity_flags: List[bool] = []
    parity_errors: List[float] = []
    for chunk in chunks:
        problem_chunk, parity = build_problem(
            episode,
            model,
            teacher,
            [_batch_dict_to_device(batch, device) for batch in chunk],
            q_model=q_model,
        )
        parity_flags.append(bool(parity.get("parity_passed")))
        parity_errors.append(float(parity.get("parity_max_abs_error", 0.0)))
        problem_cpu.extend(
            _frozen_batch_to(frozen, torch.device("cpu")) for frozen in problem_chunk
        )
    frozen_init = _eval_frozen_rows_chunked(model, problem_cpu, device)
    fresh_init = _fresh_greedy_rows_chunked(
        episode, model, teacher, subset, device, chunk_size, q_model=q_model
    )
    return {
        "status": "ready",
        "problem_cpu": problem_cpu,
        "subset": subset,
        "sampling": sampling,
        "chunk_size": int(chunk_size),
        "parity": {
            "parity_passed": bool(all(parity_flags)) if parity_flags else False,
            "parity_max_abs_error": max(parity_errors) if parity_errors else 0.0,
        },
        "frozen_init": frozen_init,
        "fresh_init": fresh_init,
    }


def natural_diag_failure_block(
    *,
    status: str,
    failure_kind: str,
    failure_detail: Any,
    mode: str = "independent",
    sampling: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """诊断不可评估/失败时的诚实报告：绝不写成改善或无恶化。"""
    return {
        "diagnostic_status": status,
        "diagnostic_failure_kind": failure_kind,
        "diagnostic_failure_detail": failure_detail,
        "diagnostic_n_parents": 0,
        "diagnostic_source_counts": {"source0": 0, "source1": 0, "unknown": 0},
        "natural_validation_reused": mode == "reuse_validation",
        "caliber": "natural_parent_eta",
        "p_eta_sampling_method": "natural_parent_eta",
        "diagnostic_mode": mode,
        "sampling": sampling or {},
        "n_parents": 0,
        "eta0_count": 0,
        "eta1_count": 0,
        "eta1_share": 0.0,
        "natural_worsened": NOT_EVALUABLE,
        "frozen_improved": NOT_EVALUABLE,
        "natural_frozen_improved": NOT_EVALUABLE,
        "natural_fresh_greedy_improved": NOT_EVALUABLE,
        "natural_frozen_worsened": NOT_EVALUABLE,
        "natural_fresh_greedy_worsened": NOT_EVALUABLE,
        "natural_full_bellman_improvement": NOT_EVALUABLE,
        "natural_validation_regression_observed": NOT_EVALUABLE,
    }


def natural_diag_finalize(
    state: Dict[str, Any],
    episode: Any,
    model: torch.nn.Module,
    teacher: BPGridTeacher,
    *,
    q_model: Optional[torch.nn.Module] = None,
    validation_rows: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> Dict[str, Any]:
    """独立 Natural 诊断第二步：同一冻结问题终评 + Fresh-Greedy 终评。"""
    if state.get("status") != "ready":
        return natural_diag_failure_block(
            status=state.get("diagnostic_status", NOT_EVALUABLE),
            failure_kind=str(state.get("failure_kind", "prepare_failed")),
            failure_detail=state.get("failure_detail"),
            sampling=state.get("sampling", {}),
        )
    device = next(model.parameters()).device
    chunk_size = int(state.get("chunk_size", 512))
    frozen_final = _eval_frozen_rows_chunked(model, state["problem_cpu"], device)
    fresh_final = _fresh_greedy_rows_chunked(
        episode,
        model,
        teacher,
        state["subset"],
        device,
        chunk_size,
        q_model=q_model,
    )
    return _assemble_natural_block(
        mode="independent",
        natural_validation_reused=False,
        calibers_identical=False,
        sampling=state.get("sampling", {}),
        parity=state.get("parity"),
        frozen_init=state["frozen_init"],
        frozen_final=frozen_final,
        fresh_init=state["fresh_init"],
        fresh_final=fresh_final,
        validation_rows=validation_rows,
    )


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
    natural_validation_batches: Optional[Sequence[Dict[str, torch.Tensor]]] = None,
    natural_diag_mode: Optional[str] = None,
    natural_diag_config: Optional[Dict[str, Any]] = None,
    natural_calibers_identical: bool = False,
) -> Dict[str, Any]:
    """One frozen-policy P iteration: greedy refresh -> residual solve -> best state."""
    # 执行层性能开关（绝不改变训练/Acceptance/回滚语义）：非法取值立即失败。
    bp_drift_mode, bp_drift_subset_rows = _p_perf_switches(episode)
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
    phase_seconds: Dict[str, float] = {}
    _t_phase = time.perf_counter()
    train_problem, train_parity = build_problem(
        episode, model, teacher, train_batches, q_model=q_model
    )
    validation_problem, val_parity = build_problem(
        episode, model, teacher, validation_batches or train_batches, q_model=q_model
    )
    phase_seconds["problem_build"] = time.perf_counter() - _t_phase
    # Natural Validation 诊断模式（Prompt §1）：
    #   reuse_validation — 主 Validation 即 Natural 口径（η Balance 关闭，或
    #                      Validation 未被 η 平衡），直接复用其诊断，零额外
    #                      Grid Teacher 调用、零第二份问题构建；
    #   independent     — 存在第二种口径（Balanced Validation），对 Original
    #                      Validation 的固定预算子集做独立双时点诊断；
    #   not_evaluable   — 无独立 Validation 口径。绝不退化为整个训练池诊断。
    if natural_diag_mode is None:
        natural_diag_mode = (
            "independent" if natural_validation_batches else "reuse_validation"
        )
    if natural_diag_mode not in {
        "independent",
        "reuse_validation",
        "not_evaluable",
        "skipped",
    }:
        raise ValueError(f"Unknown natural_diag_mode: {natural_diag_mode!r}")
    if natural_diag_mode == "independent" and not natural_validation_batches:
        natural_diag_mode = "not_evaluable"
    natural_diag_state: Optional[Dict[str, Any]] = None
    natural_diag_failure: Optional[Dict[str, Any]] = None
    _t_phase = time.perf_counter()
    initial, validation_rows_initial = evaluate_problem_rows(
        model, validation_problem
    )
    phase_seconds["eval_initial"] = time.perf_counter() - _t_phase
    with torch.no_grad():
        start_p0, start_pi = model._value_outputs(validation_problem[0].parent)
    start_model_hash = _module_state_hash(model)
    start_p_hash = episode._params_content_hash(params)
    rng_cpu = torch.get_rng_state()
    rng_cuda = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )

    def _restore_rng() -> None:
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)

    def _reject(
        status: str,
        failure_kind: str,
        detail: Any,
        *,
        initial_legal: bool = True,
    ) -> Dict[str, Any]:
        """Roll back to the iteration-start state and report a non-accept."""
        _restore_state(model, start_state)
        _restore_rng()
        if _module_state_hash(model) != start_model_hash:
            raise RuntimeError("P rollback failed to restore the model hash")
        if episode._params_content_hash(params) != start_p_hash:
            raise RuntimeError("P rollback failed to restore the P-only hash")
        return {
            "inner_index": int(inner_index),
            "iteration_index": int(iteration_index),
            "status": status,
            "failure_kind": failure_kind,
            "failure_detail": detail,
            "optimizer_steps": 0,
            "adam_steps": 0,
            "lbfgs_closure_calls": 0,
            "lbfgs_iteration_budget": int(lbfgs_max_iter),
            "lbfgs_actual_iterations": 0,
            "stop_reason": failure_kind,
            "module_hashes": {
                "q_params": episode._params_content_hash(q_params),
                "sdf_fc1": sdf_hash_before,
                "model": start_model_hash,
                "p_root": start_p_hash,
            },
            "rollback": {
                "restored_iteration_start_state": True,
                # 初始状态本身非法时不得声称"恢复合法模型"；上层必须阻断
                # 本次 Firm Stage，而不是把非法起点当作可回退目标。
                "initial_state_was_legal": bool(initial_legal),
                "model_hash_verified": True,
                "p_root_hash_verified": True,
                "rng_restored": True,
                "block_firm_stage": not bool(initial_legal),
            },
        }

    # --- Parity 是硬性前置条件：任一失败不执行 P Optimizer、不标记 Accepted。
    if not (train_parity.get("parity_passed") and val_parity.get("parity_passed")):
        detail = {
            "train_parity": train_parity,
            "validation_parity": val_parity,
        }
        return _reject("rejected_numerical", "parity_failed", detail)

    # --- Initial Validation 硬有限 Gate（Prompt §2）：Parity 之后、任何
    # Optimizer Step 之前。初始状态非法时禁止 Adam/L-BFGS、禁止
    # accepted_reverted、禁止继续 Q。
    initial_all_finite = bool(initial.get("all_finite"))
    initial_n_total = int(initial.get("n_total", 0))
    initial_nonfinite = int(initial.get("nonfinite_count", 0))
    initial_loss_finite = math.isfinite(float(initial.get("normalized_mae", float("nan"))))
    params_finite = _params_finite(params)
    if not (
        initial_all_finite
        and initial_n_total > 0
        and initial_nonfinite == 0
        and initial_loss_finite
        and params_finite
    ):
        return _reject(
            "rejected_numerical",
            "initial_validation_nonfinite",
            {
                "initial_validation": {
                    "all_finite": initial_all_finite,
                    "n_total": initial_n_total,
                    "nonfinite_count": initial_nonfinite,
                    "normalized_mae": initial.get("normalized_mae"),
                    "p0_norm_mae": initial.get("p0_norm_mae"),
                    "pi_norm_mae": initial.get("pi_norm_mae"),
                },
                "root_loss_finite": initial_loss_finite,
                "p_root_params_finite": params_finite,
            },
            initial_legal=False,
        )

    # Fresh-greedy 基线（同一固定 Validation/CRN），用于如实记录 Frozen
    # Residual 改善而 Fresh Greedy 恶化的情况。
    _t_phase = time.perf_counter()
    fresh_rows_initial = _fresh_greedy_rows(
        episode, model, teacher, validation_batches or train_batches, q_model=q_model
    )
    phase_seconds["fresh_greedy_initial"] = time.perf_counter() - _t_phase
    fresh_initial = _fresh_stats_from_rows(fresh_rows_initial)
    if not fresh_initial.get("all_finite", False):
        return _reject(
            "rejected_numerical", "fresh_greedy_nonfinite_initial", fresh_initial
        )

    # --- 独立 Natural 诊断的迭代起点快照（Prompt §2/§5）：固定预算子集 + 冻结
    # 问题（CPU 驻留，优化期间不占 GPU）+ Frozen/Fresh-Greedy 初始评估。整个
    # 诊断与正式求解完全隔离：任何可恢复的诊断异常只记录诊断状态，绝不改变
    # 合法 P 更新的 Acceptance；CUDA 资源故障则记录并安全停止。
    if natural_diag_mode == "independent":
        rng_cpu_diag = torch.get_rng_state()
        rng_cuda_diag = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        diag_config = natural_diag_config or {}
        try:
            _t_phase = time.perf_counter()
            natural_diag_state = natural_diag_prepare(
                episode,
                model,
                teacher,
                natural_validation_batches,
                q_model=q_model,
                max_parents=int(diag_config.get("max_parents", 4096)),
                chunk_size=int(diag_config.get("chunk_size", 512)),
                seed=int(diag_config.get("seed", 97531)),
            )
            phase_seconds["natural_diag_prepare"] = time.perf_counter() - _t_phase
        except torch.cuda.OutOfMemoryError as exc:
            logger.error("Natural diagnostic CUDA OOM; safe-stopping: %s", exc)
            raise RuntimeError(
                f"natural_diagnostic_cuda_oom: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - 诊断异常必须与训练隔离
            natural_diag_failure = {
                "diagnostic_status": "failed_diagnostic",
                "diagnostic_failure_kind": type(exc).__name__,
                "diagnostic_failure_detail": str(exc),
            }
            logger.warning(
                "Natural diagnostic prepare failed (isolated): %s: %s",
                type(exc).__name__,
                exc,
            )
        finally:
            torch.set_rng_state(rng_cpu_diag)
            if rng_cuda_diag is not None:
                torch.cuda.set_rng_state_all(rng_cuda_diag)

    _t_phase = time.perf_counter()
    adam_history, adam_state, adam_best, adam_steps_done, adam_status = run_adam(
        model=model,
        params=params,
        train=train_problem,
        validation=validation_problem,
        steps=adam_steps,
        lr=adam_lr,
        clip=adam_clip,
        eval_every=eval_every,
    )
    phase_seconds["adam"] = time.perf_counter() - _t_phase
    if adam_status != "ok":
        return _reject("rejected_numerical", f"adam_{adam_status}", {
            "adam_history_tail": adam_history[-3:],
        })
    _restore_state(model, adam_state)
    if target_residual is not None and adam_best["normalized_mae"] <= target_residual:
        lbfgs_history: List[Dict[str, Any]] = []
        best_state, best = adam_state, adam_best
        stop_reason = "target_residual_reached_after_adam"
        lbfgs_counters = {
            "LBFGS_ITERATION_BUDGET": int(lbfgs_max_iter),
            "LBFGS_ACTUAL_ITERATIONS": 0,
            "LBFGS_CLOSURE_CALLS": 0,
            "LBFGS_NONFINITE_CLOSURES": 0,
        }
    else:
        _t_phase = time.perf_counter()
        lbfgs_history, best_state, best, stop_reason, lbfgs_counters = run_lbfgs_chunked(
            model=model,
            params=params,
            train=train_problem,
            validation=validation_problem,
            max_iter=lbfgs_max_iter,
            chunk_iterations=lbfgs_chunk_iterations,
            history_size=lbfgs_history_size,
            target_residual=target_residual,
        )
        phase_seconds["lbfgs"] = time.perf_counter() - _t_phase
    # --- L-BFGS Nonfinite Closure 保守拒绝（Prompt §1.2）：即使 Adam 曾取得
    # 合法改善，本次求解也不得报告为 accepted_improved，必须恢复完整的
    # Iteration Start，不向下一次 P Iteration 或 Q 传递。
    if (
        stop_reason == "lbfgs_nonfinite_closure"
        or int(lbfgs_counters.get("LBFGS_NONFINITE_CLOSURE_ABORT", 0)) > 0
    ):
        return _reject(
            "rejected_numerical",
            "lbfgs_nonfinite_closure",
            {
                "lbfgs_counters": dict(lbfgs_counters),
                "adam_steps_before_abort": int(adam_steps_done),
                "lbfgs_history_tail": lbfgs_history[-3:],
            },
        )
    _restore_state(model, best_state)
    _t_phase = time.perf_counter()
    final, validation_rows_final = evaluate_problem_rows(model, validation_problem)
    phase_seconds["eval_final"] = time.perf_counter() - _t_phase
    if not final.get("all_finite", False):
        return _reject(
            "rejected_numerical",
            "best_state_nonfinite_validation",
            {"final_validation": final},
        )

    # 有效优化步骤：Adam 成功步 + L-BFGS 真实迭代（closure calls 不算）。
    effective_steps = int(adam_steps_done) + int(
        lbfgs_counters.get("LBFGS_ACTUAL_ITERATIONS", 0)
    )
    if effective_steps <= 0:
        return _reject(
            "rejected_no_effective_update",
            "no_effective_optimizer_step",
            {
                "adam_steps": int(adam_steps_done),
                "lbfgs_counters": dict(lbfgs_counters),
            },
        )

    # 冻结策略残差 + 重新 Greedy 残差（独立固定 Validation/CRN）。
    _t_phase = time.perf_counter()
    fresh_rows_final = _fresh_greedy_rows(
        episode, model, teacher, validation_batches or train_batches, q_model=q_model
    )
    phase_seconds["fresh_greedy_final"] = time.perf_counter() - _t_phase
    fresh = _fresh_stats_from_rows(fresh_rows_final)
    if not fresh.get("all_finite", False):
        return _reject(
            "rejected_numerical", "fresh_greedy_nonfinite_final", fresh
        )

    # Greedy BP drift: frozen iteration policy vs fresh greedy at the accepted
    # state. 纯观测口径（全仓无消费方）：pv_bp_drift_mode 可降级为固定种子行
    # 子样本（subset）或完全跳过 _recompute_bp_star 重解（off）；廉价的
    # value_output_drift（两次 _value_outputs）在所有模式下照常记录。
    _t_phase = time.perf_counter()
    bp0_drift, bpi_drift = [], []
    bp_drift_rows = 0
    with torch.no_grad():
        if bp_drift_mode != "off":
            if bp_drift_mode == "subset":
                row_plan, bp_drift_rows = _bp_drift_row_plan(
                    train_problem, bp_drift_subset_rows
                )
            else:
                row_plan = [None] * len(train_problem)
                bp_drift_rows = int(
                    sum(
                        int(frozen_batch.parent.shape[0])
                        for frozen_batch in train_problem
                    )
                )
            for frozen_batch, source_batch, rows in zip(
                train_problem, train_batches, row_plan
            ):
                for branch_name, drift_sink in (("p0", bp0_drift), ("pi", bpi_drift)):
                    if rows is None:
                        fresh_bp = _recompute_bp_star(
                            episode, model, teacher, source_batch, branch_name, q_model
                        )
                        frozen_branch = getattr(frozen_batch, branch_name)
                    else:
                        if rows.numel() == 0:
                            continue
                        fresh_bp = _recompute_bp_star(
                            episode,
                            model,
                            teacher,
                            _batch_row_subset(source_batch, rows),
                            branch_name,
                            q_model,
                        )
                        frozen_branch = getattr(frozen_batch, branch_name)
                    frozen_bp = frozen_branch.bp_star
                    if rows is not None:
                        frozen_bp = frozen_bp.index_select(0, rows.to(frozen_bp.device))
                    drift_sink.append(
                        (frozen_bp - fresh_bp).abs().reshape(-1)
                    )
        end_p0, end_pi = model._value_outputs(validation_problem[0].parent)
        drift_p0 = float((end_p0 - start_p0).abs().mean())
        drift_pi = float((end_pi - start_pi).abs().mean())

    phase_seconds["bp_drift_recompute"] = time.perf_counter() - _t_phase
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

    # --- 接受判定（复用既有 best-checkpoint 语义：无既有退化门槛，不引入
    # 新的经济收敛阈值）。accepted_improved = 合法候选在固定 Validation 上
    # 取得有效改善；否则恢复迭代起点模型并标记 accepted_reverted。
    initial_mae = (
        float(initial["normalized_mae"]) if initial.get("all_finite") else float("nan")
    )
    final_mae = float(final["normalized_mae"])
    frozen_improved = (
        math.isfinite(initial_mae)
        and math.isfinite(final_mae)
        and final_mae < initial_mae
    )
    fresh_initial_combined = 0.5 * (
        float(fresh_initial.get("fresh_greedy_p0_norm_mae", float("nan")))
        + float(fresh_initial.get("fresh_greedy_pi_norm_mae", float("nan")))
    )
    fresh_final_combined = 0.5 * (
        float(fresh.get("fresh_greedy_p0_norm_mae", float("nan")))
        + float(fresh.get("fresh_greedy_pi_norm_mae", float("nan")))
    )
    fresh_worsened = (
        math.isfinite(fresh_initial_combined)
        and math.isfinite(fresh_final_combined)
        and fresh_final_combined > fresh_initial_combined
    )
    fresh_regression_observed = bool(frozen_improved and fresh_worsened)

    if frozen_improved:
        status = "accepted_improved"
        rollback = {
            "restored_iteration_start_state": False,
            "kept_best_validation_state": True,
            "rng_restored": False,
            "initial_state_was_legal": True,
        }
    else:
        # 候选未改善：恢复上一合法模型（迭代起点），不得计为成功 P 更新。
        # accepted_reverted 仅在初始状态合法（Initial Gate 已通过）、训练无
        # 未处理数值异常、且已严格恢复并校验哈希时成立。
        _restore_state(model, start_state)
        _restore_rng()
        if _module_state_hash(model) != start_model_hash:
            raise RuntimeError("P revert failed to restore the model hash")
        if episode._params_content_hash(params) != start_p_hash:
            raise RuntimeError("P revert failed to restore the P-only hash")
        status = "accepted_reverted"
        rollback = {
            "restored_iteration_start_state": True,
            "kept_best_validation_state": False,
            "rng_restored": True,
            "initial_state_was_legal": True,
            "model_hash_verified": True,
            "p_root_hash_verified": True,
        }

    elapsed = time.perf_counter() - started
    peak_memory = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))

    # Natural Validation 终评（模型已处于最终生效状态：accepted best 或
    # accepted_reverted 的迭代起点）。纯诊断：RNG 显式保护，异常与训练完全
    # 隔离，绝不参与 Acceptance / Revert 判定。
    validation_rows_bundle = {
        "frozen_init": validation_rows_initial,
        "frozen_final": validation_rows_final,
        "fresh_init": fresh_rows_initial,
        "fresh_final": fresh_rows_final,
    }
    if natural_diag_mode == "independent":
        if natural_diag_failure is not None:
            natural_validation_block = natural_diag_failure_block(
                status="failed_diagnostic",
                failure_kind=str(
                    natural_diag_failure.get("diagnostic_failure_kind")
                ),
                failure_detail=natural_diag_failure.get(
                    "diagnostic_failure_detail"
                ),
            )
        else:
            rng_cpu_diag = torch.get_rng_state()
            rng_cuda_diag = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
            try:
                natural_validation_block = natural_diag_finalize(
                    natural_diag_state,
                    episode,
                    model,
                    teacher,
                    q_model=q_model,
                    validation_rows=validation_rows_bundle,
                )
            except torch.cuda.OutOfMemoryError as exc:
                logger.error("Natural diagnostic CUDA OOM; safe-stopping: %s", exc)
                raise RuntimeError(
                    f"natural_diagnostic_cuda_oom: {exc}"
                ) from exc
            except Exception as exc:  # noqa: BLE001 - 诊断异常必须与训练隔离
                logger.warning(
                    "Natural diagnostic finalize failed (isolated): %s: %s",
                    type(exc).__name__,
                    exc,
                )
                natural_validation_block = natural_diag_failure_block(
                    status="failed_diagnostic",
                    failure_kind=type(exc).__name__,
                    failure_detail=str(exc),
                )
            finally:
                torch.set_rng_state(rng_cpu_diag)
                if rng_cuda_diag is not None:
                    torch.cuda.set_rng_state_all(rng_cuda_diag)
        natural_diag_state = None  # 诊断问题不再需要，立即释放
    elif natural_diag_mode == "reuse_validation":
        # 情况 A/C（Prompt §1.2）：主 Validation 即 Natural 口径 — 直接复用
        # 已算出的诊断，零重复 Grid Teacher、零第二份 Expanded Child States。
        n_val_rows = int(validation_rows_final["p0_norm"].numel())
        natural_validation_block = _assemble_natural_block(
            mode="reuse_validation",
            natural_validation_reused=True,
            calibers_identical=bool(natural_calibers_identical),
            sampling={
                "sampling_method": "reused_main_validation_diagnostics",
                "requested_parents": n_val_rows,
                "selected_parents": n_val_rows,
                "sampling_ratio": 1.0,
                "eta_rebalanced": False,
                "note": (
                    "no second frozen problem constructed; diagnostics "
                    "reused from the main validation evaluation"
                ),
            },
            parity=val_parity,
            frozen_init=validation_rows_initial,
            frozen_final=validation_rows_final,
            fresh_init=fresh_rows_initial,
            fresh_final=fresh_rows_final,
            frozen_initial_stats=initial,
            frozen_final_stats=final,
            fresh_initial_stats=fresh_initial,
            fresh_final_stats=fresh,
            validation_rows=validation_rows_bundle,
        )
    else:
        natural_validation_block = natural_diag_failure_block(
            status=NOT_EVALUABLE,
            failure_kind=(
                "skipped_by_frequency"
                if natural_diag_mode == "skipped"
                else "no_natural_validation_batches"
            ),
            failure_detail=(
                "natural diagnostic deferred by frequency control"
                if natural_diag_mode == "skipped"
                else (
                    "no independent validation caliber; falling back to the "
                    "full train pool is forbidden"
                )
            ),
            mode=natural_diag_mode,
        )
    natural_regression_observed = natural_validation_block[
        "natural_validation_regression_observed"
    ]

    summary: Dict[str, Any] = {
        "inner_index": int(inner_index),
        "iteration_index": int(iteration_index),
        "status": status,
        "effective_optimizer_steps": int(effective_steps),
        "optimizer_steps": int(adam_steps_done),
        "adam_steps": int(adam_steps_done),
        "lbfgs_closure_calls": int(lbfgs_counters.get("LBFGS_CLOSURE_CALLS", 0)),
        "lbfgs_iteration_budget": int(lbfgs_counters.get("LBFGS_ITERATION_BUDGET", 0)),
        "lbfgs_actual_iterations": int(
            lbfgs_counters.get("LBFGS_ACTUAL_ITERATIONS", 0)
        ),
        "lbfgs_nonfinite_closures": int(
            lbfgs_counters.get("LBFGS_NONFINITE_CLOSURES", 0)
        ),
        "lbfgs_optimizer_budget": int(lbfgs_max_iter),
        "stop_reason": stop_reason,
        "frozen_policy_p0_residual": {
            "mae": final["p0_norm_mae"],
            "rmse": final["p0_norm_rmse"],
            "p90": final["p0_norm_p90"],
            "eta0_mae": final.get("p0_norm_eta0_mae"),
            "eta1_mae": final.get("p0_norm_eta1_mae"),
            "eta0_p90": final.get("p0_norm_eta0_p90"),
            "eta1_p90": final.get("p0_norm_eta1_p90"),
            "eta0_n": final.get("p0_norm_eta0_n_total"),
            "eta1_n": final.get("p0_norm_eta1_n_total"),
            "eta0_nonfinite_ratio": final.get("p0_norm_eta0_nonfinite_ratio"),
            "eta1_nonfinite_ratio": final.get("p0_norm_eta1_nonfinite_ratio"),
        },
        "frozen_policy_pi_residual": {
            "mae": final["pi_norm_mae"],
            "rmse": final["pi_norm_rmse"],
            "p90": final["pi_norm_p90"],
            "eta0_mae": final.get("pi_norm_eta0_mae"),
            "eta1_mae": final.get("pi_norm_eta1_mae"),
            "eta0_p90": final.get("pi_norm_eta0_p90"),
            "eta1_p90": final.get("pi_norm_eta1_p90"),
            "eta0_n": final.get("pi_norm_eta0_n_total"),
            "eta1_n": final.get("pi_norm_eta1_n_total"),
            "eta0_nonfinite_ratio": final.get("pi_norm_eta0_nonfinite_ratio"),
            "eta1_nonfinite_ratio": final.get("pi_norm_eta1_nonfinite_ratio"),
        },
        "natural_validation": natural_validation_block,
        "natural_validation_regression_observed": natural_regression_observed,
        "initial_frozen_policy_residual": {
            "p0_mae": initial["p0_norm_mae"],
            "pi_mae": initial["pi_norm_mae"],
            "normalized_mae": initial_mae,
        },
        "validation_finiteness": {
            "initial_all_finite": bool(initial.get("all_finite")),
            "final_all_finite": bool(final.get("all_finite")),
            "final_nonfinite_count": int(final.get("nonfinite_count", 0)),
            "final_n_total": int(final.get("n_total", 0)),
        },
        "fresh_greedy_residual": fresh,
        "phase_seconds": dict(phase_seconds),
        "fresh_greedy_initial": fresh_initial,
        "frozen_residual_improved": bool(frozen_improved),
        "fresh_greedy_worsened": bool(fresh_worsened),
        "fresh_greedy_regression_observed": fresh_regression_observed,
        # Frozen 改善但 Fresh Greedy 恶化时不得宣称完整 Bellman 不动点改善。
        "full_bellman_fixed_point_improvement": bool(
            frozen_improved and not fresh_worsened
        ),
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
            "model": _module_state_hash(model),
            "p_root": episode._params_content_hash(params),
        },
        "rollback": rollback,
        "p_scope_modules": ["value_encoder", "v0_head", "vi_head"],
        "elapsed_seconds": float(elapsed),
        "peak_gpu_memory_bytes": peak_memory,
        "adam_history_len": len(adam_history),
        "lbfgs_history_len": len(lbfgs_history),
    }
    # 口径标注：仅非默认模式写入；默认（full/resolve）summary 形状逐位不变。
    drift_block = summary["greedy_bp_drift"]
    if bp_drift_mode == "subset":
        drift_block["mode"] = "subset"
        drift_block["n_rows"] = int(bp_drift_rows)
    elif bp_drift_mode == "off":
        for _drift_key in ("p0_mean_abs", "pi_mean_abs", "p0_p90", "pi_p90"):
            drift_block[_drift_key] = float("nan")
        drift_block["mode"] = "off"
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


# ---------------------------------------------------------------------------
# Execution-layer performance switches (observation-only; never economic)
# ---------------------------------------------------------------------------
#
# ``pv_bp_drift_mode`` only chooses how much of the greedy BP drift
# observation each P policy iteration performs.  Training,
# Acceptance (``frozen_improved`` reads validation residuals only) and rollback
# are identical in every mode; the defaults reproduce the legacy behaviour.

BP_DRIFT_MODES = ("full", "subset", "off")

# Fixed seed for the ``subset`` row sample (deterministic across runs).
_BP_DRIFT_SUBSET_SEED = 20261024


def _p_perf_switches(episode: Any) -> Tuple[str, int]:
    """Resolve ``(pv_bp_drift_mode, p_bp_drift_subset_rows)``.

    Fail fast with a clear ``ValueError`` on any illegal value.
    """
    hyperparams = episode.hyperparams
    bp_drift_mode = str(getattr(hyperparams, "pv_bp_drift_mode", "full")).lower()
    if bp_drift_mode not in BP_DRIFT_MODES:
        raise ValueError(
            "pv_bp_drift_mode must be one of "
            f"{list(BP_DRIFT_MODES)}, got {bp_drift_mode!r}"
        )
    subset_rows = int(getattr(hyperparams, "p_bp_drift_subset_rows", 4096))
    if subset_rows < 1:
        raise ValueError(
            f"p_bp_drift_subset_rows must be at least 1, got {subset_rows!r}"
        )
    return bp_drift_mode, subset_rows


def _bp_drift_row_plan(
    problem: Sequence[FrozenPolicyBatch], max_rows: int
) -> Tuple[List[torch.Tensor], int]:
    """Fixed-seed row subset plan for ``pv_bp_drift_mode='subset'``.

    Returns ``(per_batch_row_indices, n_rows)`` with ``n_rows <= max_rows``.
    """
    counts = [int(batch.parent.shape[0]) for batch in problem]
    total = int(sum(counts))
    if total <= int(max_rows):
        return [torch.arange(count, dtype=torch.long) for count in counts], total
    generator = torch.Generator().manual_seed(int(_BP_DRIFT_SUBSET_SEED))
    chosen = torch.randperm(total, generator=generator)[: int(max_rows)]
    plan: List[torch.Tensor] = []
    offset = 0
    for count in counts:
        mask = (chosen >= offset) & (chosen < offset + count)
        plan.append(torch.sort(chosen[mask] - offset).values.to(torch.long))
        offset += count
    return plan, int(max_rows)


def _batch_row_subset(
    batch: Dict[str, torch.Tensor], rows: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Row-slice the source-batch fields consumed by ``_recompute_bp_star``."""
    index = rows.to(device=batch["parent"].device, dtype=torch.long)
    subset: Dict[str, torch.Tensor] = {
        "parent": batch["parent"].index_select(0, index)
    }
    children = batch.get("children", [])
    if children:
        subset["children"] = [child.index_select(0, index) for child in children]
    if batch.get("child0") is not None:
        subset["child0"] = batch["child0"].index_select(0, index)
        subset["child1"] = batch["child1"].index_select(0, index)
    return subset
