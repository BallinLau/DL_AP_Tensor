"""
Target-grid teacher for firm leverage policy distillation.

The teacher evaluates economic Bellman RHS values over candidate bp values with
the frozen firm target network. It returns detached value targets for P0/PI and
detached bp labels for the policy heads. Simulation code never calls this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from config import Config
from utils.firm_transition import apply_refinancing_policy, normalize_child_weights


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GridChunkPlan:
    parent_batch_requested: int
    parent_chunk_effective: int
    candidate_chunk_effective: int
    n_children: int
    n_grid: int
    expanded_states_per_forward: int
    max_expanded_states: int


@dataclass(frozen=True)
class _EquityBlock:
    """Branch-independent candidate child equity block.

    ``p_child``/``bar_z_child``/``child_b_grid`` depend only on the frozen child
    states, the candidate bp grid, and the parent leverage. They never depend on
    the P0/PI branch, so the same block can be reused for every branch (and, when
    evaluated over the full Jmax child set, for every nested J prefix).
    """

    p_child: torch.Tensor
    bar_z_child: torch.Tensor
    child_b_grid: torch.Tensor


def resolve_grid_chunk_plan(
    *,
    n_parent: int,
    n_grid: int,
    n_children: int,
    configured_parent_chunk: int,
    configured_candidate_chunk: int,
    max_expanded_states: int,
) -> GridChunkPlan:
    """Jointly cap parent and candidate chunks by expanded child-state count."""
    n_parent = int(n_parent)
    n_grid = int(n_grid)
    n_children = int(n_children)
    max_expanded_states = int(max_expanded_states)
    if n_parent < 1 or n_grid < 1 or n_children < 1:
        raise ValueError("n_parent, n_grid, and n_children must all be positive")
    if max_expanded_states < n_children:
        raise ValueError(
            "bp_grid_max_expanded_states is smaller than one parent x one candidate "
            f"x all children: {max_expanded_states} < {n_children}"
        )
    requested_parent = (
        n_parent if int(configured_parent_chunk) <= 0
        else min(n_parent, int(configured_parent_chunk))
    )
    parent_cap = max_expanded_states // n_children
    parent_chunk = max(1, min(requested_parent, parent_cap))
    candidate_cap = max_expanded_states // (parent_chunk * n_children)
    requested_candidate = (
        n_grid if int(configured_candidate_chunk) <= 0
        else min(n_grid, int(configured_candidate_chunk))
    )
    candidate_chunk = max(1, min(requested_candidate, candidate_cap, n_grid))
    expanded = parent_chunk * candidate_chunk * n_children
    if expanded > max_expanded_states:
        raise AssertionError("resolved BP grid chunk plan exceeds max_expanded_states")
    return GridChunkPlan(
        parent_batch_requested=n_parent,
        parent_chunk_effective=parent_chunk,
        candidate_chunk_effective=candidate_chunk,
        n_children=n_children,
        n_grid=n_grid,
        expanded_states_per_forward=expanded,
        max_expanded_states=max_expanded_states,
    )


def _get_out(out: Any, name: str, idx: int) -> torch.Tensor:
    if isinstance(out, dict):
        return out[name]
    if hasattr(out, name):
        return getattr(out, name)
    return out[:, idx:idx + 1]


def _target_q_claim(
    model: Any,
    firm_state: torch.Tensor,
    *,
    equity_model: Optional[Any] = None,
) -> torch.Tensor:
    """Return the debt-claim price used by conditional-survival P objectives.

    ``equity_model`` remains in the signature for non-hybrid compatibility,
    but hybrid issuance prices deliberately ignore current/candidate Phat.
    Realized-default settlement is handled only by ``_q_effective_output``.
    """
    claim_fn = getattr(model, "_q_claim_output", None)
    if callable(claim_fn) and getattr(model, "q_parameterization", None) == "hybrid_regime":
        return claim_fn(firm_state)
    q_fn = getattr(model, "_q_output", None)
    if callable(q_fn):
        return q_fn(firm_state)
    return _get_out(model(firm_state), "Q", 0)


# Compatibility alias for diagnostics/tests importing the old private helper.
_target_q = _target_q_claim


def _target_equity(model: Any, firm_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    equity_fn = getattr(model, "forward_equity", None)
    if callable(equity_fn):
        out = equity_fn(firm_state)
    else:
        out = model(firm_state)
    return _get_out(out, "P", 7), _get_out(out, "bar_z", 6)


def _strip_extra(x: torch.Tensor) -> torch.Tensor:
    return x[:, :7] if x.shape[1] > 7 else x


def _stack_children(children: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    if isinstance(children, torch.Tensor):
        if children.ndim != 3:
            raise ValueError("children tensor must have shape [N,J,D]")
        return children
    if not children:
        raise ValueError("BP grid evaluation requires at least one child tensor.")
    return torch.stack(tuple(children), dim=1)


def _stack_m(m_values: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    if isinstance(m_values, torch.Tensor):
        if m_values.ndim != 3:
            raise ValueError("M tensor must have shape [N,J,1]")
        return m_values
    if not m_values:
        raise ValueError("BP grid evaluation requires at least one M tensor.")
    return torch.stack(tuple(m_values), dim=1)


def _slice_child_axis(
    values: Sequence[torch.Tensor] | torch.Tensor,
    start: int,
    stop: int,
) -> Sequence[torch.Tensor] | torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values[start:stop]
    return [value[start:stop] for value in values]


def _slice_prefix_weights(
    weights: Optional[torch.Tensor],
    count: int,
) -> Optional[torch.Tensor]:
    """Nested child-weight prefix; ``normalize_child_weights`` renormalizes it."""
    if weights is None:
        return None
    return weights[..., : int(count)]


def _expand_candidates(base: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """Repeat a (B,D) tensor over candidate dimension and flatten to (B*J,D)."""
    batch_size, n_grid = candidates.shape
    return (
        base.unsqueeze(1)
        .expand(batch_size, n_grid, base.shape[-1])
        .reshape(batch_size * n_grid, base.shape[-1])
        .clone()
    )


def _candidate_flat(candidates: torch.Tensor) -> torch.Tensor:
    return candidates.reshape(-1, 1)


def _expand_grid_children(
    children: Sequence[torch.Tensor] | torch.Tensor,
    bp_grid: torch.Tensor,
    b_parent: torch.Tensor,
    eta_current: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build candidate child states using the current parent refinancing state.

    Timing invariant:
        b_next = eta_current * bp_candidate + (1 - eta_current) * b_current

    ``eta_current`` is ``eta_t`` from the parent state. Child ``eta_next`` is
    still carried in the child states but never gates realized leverage, so
    ``child_b_grid`` is constant along the child axis:

    * parent ``eta_t = 1``: ``child_b = bp_candidate`` for every child;
    * parent ``eta_t = 0``: ``child_b = b_parent`` for every child.
    """
    children_t = _stack_children(children)
    child_state_raw = children_t[..., :7] if children_t.shape[-1] > 7 else children_t
    batch_size, n_children, state_dim = child_state_raw.shape
    n_grid = bp_grid.shape[1]

    child_states = (
        child_state_raw.unsqueeze(1)
        .expand(batch_size, n_grid, n_children, state_dim)
        .clone()
    )
    # Realized child leverage is constant along the child axis: it is driven by
    # the CURRENT parent eta_t, so the candidate axis is the only varying one.
    child_b_grid = apply_refinancing_policy(
        b_current=b_parent.reshape(batch_size, 1, 1),
        bp_candidate=bp_grid.unsqueeze(-1),
        eta_current=eta_current.reshape(batch_size, 1, 1),
    ).expand(-1, -1, n_children)
    child_states[..., 0] = child_b_grid
    return child_states, child_b_grid


def _forward_equity_grid_children(
    model: Any,
    children: Sequence[torch.Tensor] | torch.Tensor,
    bp_grid: torch.Tensor,
    b_parent: torch.Tensor,
    eta_current: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    child_states, child_b_grid = _expand_grid_children(
        children, bp_grid, b_parent, eta_current
    )
    batch_size, n_grid, n_children, state_dim = child_states.shape
    flat_states = child_states.reshape(batch_size * n_grid * n_children, state_dim)
    p_raw, bar_z_raw = _target_equity(model, flat_states)
    p_child = p_raw.reshape(batch_size, n_grid, n_children)
    bar_z_child = bar_z_raw.reshape(batch_size, n_grid, n_children).clamp(0.0, 1.0)
    return p_child, bar_z_child, child_b_grid


def _safe_top2_margin(value_grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if value_grid.shape[1] < 2:
        margin = torch.full((value_grid.shape[0], 1), float("inf"), device=value_grid.device, dtype=value_grid.dtype)
        idx = torch.zeros((value_grid.shape[0], 1), device=value_grid.device, dtype=torch.long)
        return margin, idx
    vals, idx = torch.topk(value_grid, k=2, dim=1)
    return (vals[:, 0:1] - vals[:, 1:2]), idx[:, 0:1]


def _gather_by_index(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return values.gather(1, index).reshape(-1, 1)


def _concat_chunk_outputs(chunks: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not chunks:
        return {}
    keys = chunks[0].keys()
    return {k: torch.cat([chunk[k] for chunk in chunks], dim=0) for k in keys}


def _quadratic_refine(
    bp_grid: torch.Tensor,
    value_grid: torch.Tensor,
    argmax_index: torch.Tensor,
    bp_star_grid: torch.Tensor,
    enabled: bool,
) -> torch.Tensor:
    """Optional local parabola vertex around the best grid point."""
    if not enabled or value_grid.shape[1] < 3:
        return bp_star_grid

    batch_size, n_grid = value_grid.shape
    idx = argmax_index.reshape(-1)
    valid = (idx > 0) & (idx < n_grid - 1)
    if not bool(valid.any()):
        return bp_star_grid

    rows = torch.arange(batch_size, device=value_grid.device)
    left = (idx - 1).clamp(0, n_grid - 1)
    right = (idx + 1).clamp(0, n_grid - 1)

    x_l = bp_grid[rows, left]
    x_c = bp_grid[rows, idx]
    x_r = bp_grid[rows, right]
    y_l = value_grid[rows, left]
    y_c = value_grid[rows, idx]
    y_r = value_grid[rows, right]

    # For a locally uniform grid, vertex offset from center is:
    # 0.5 * dx * (y_left - y_right) / (y_left - 2*y_center + y_right).
    dx = (x_r - x_l).abs().clamp_min(1e-8) / 2.0
    denom = y_l - 2.0 * y_c + y_r
    vertex = x_c + 0.5 * dx * (y_l - y_r) / denom.clamp(max=-1e-8)
    lo = torch.minimum(x_l, x_r)
    hi = torch.maximum(x_l, x_r)
    vertex = vertex.clamp(lo, hi)

    refined = bp_star_grid.reshape(-1).clone()
    refined = torch.where(valid & (denom < -1e-8), vertex, refined)
    return refined.reshape(-1, 1)


class BPGridTeacher:
    """Compute detached target-grid controls and value backups for P0/PI."""

    def __init__(
        self,
        target_model,
        p0_loss_fn,
        pi_loss_fn,
        *,
        q_target_model=None,
        grid_min: float = 0.0,
        grid_max: float = 1.0,
        coarse_size: int = 21,
        fine_size: int = 9,
        refine: bool = True,
        quadratic_refine: bool = False,
        parent_chunk_size: int = 2048,
        candidate_chunk_size: int = 0,
        max_expanded_states: int = 65536,
        margin_scale: float = 1e-3,
        confidence_relative: bool = True,
        confidence_min: float = 0.0,
        boundary_match_phat_eps: float = 1e-2,
    ):
        self.target_model = target_model
        self.q_target_model = q_target_model if q_target_model is not None else target_model
        self.p0_loss_fn = p0_loss_fn
        self.pi_loss_fn = pi_loss_fn
        self.grid_min = float(grid_min)
        self.grid_max = float(grid_max)
        self.coarse_size = max(2, int(coarse_size))
        self.fine_size = max(2, int(fine_size))
        self.refine = bool(refine)
        self.quadratic_refine = bool(quadratic_refine)
        self.parent_chunk_size = max(0, int(parent_chunk_size))
        self.candidate_chunk_size = max(0, int(candidate_chunk_size))
        self.max_expanded_states = max(1, int(max_expanded_states))
        self.margin_scale = max(float(margin_scale), 1e-12)
        self.confidence_relative = bool(confidence_relative)
        self.confidence_min = min(max(float(confidence_min), 0.0), 1.0)
        self.boundary_match_phat_eps = max(float(boundary_match_phat_eps), 0.0)
        self._logged_grid_chunk_plans: set[tuple[int, int]] = set()
        self._recorded_grid_chunk_plans: Dict[tuple[int, int], Dict[str, int]] = {}
        self._forward_stats: Dict[str, Any] = self._empty_forward_stats()

    def _empty_forward_stats(self) -> Dict[str, Any]:
        result = {
            "bp_model_forward_calls": 0,
            "bp_child_equity_forward_calls": 0,
            "bp_q_forward_calls": 0,
            "bp_parent_chunks": 0,
            "bp_candidate_chunks": 0,
            "bp_max_expanded_states": int(self.max_expanded_states),
            "bp_max_actual_expanded_states": 0,
            "bp_multi_j_reuse_enabled": False,
            "bp_branch_reuse_enabled": False,
        }
        return result

    def reset_forward_stats(self) -> None:
        self._forward_stats = self._empty_forward_stats()
        self._recorded_grid_chunk_plans = {}

    def forward_stats(self) -> Dict[str, Any]:
        """Return a copy of the BP hot-path instrumentation counters."""
        return dict(self._forward_stats)

    def grid_chunk_plans(self) -> List[Dict[str, int]]:
        """Return the effective chunk plans used since the last reset."""
        return [dict(value) for _, value in sorted(self._recorded_grid_chunk_plans.items())]

    def _record_q_forward(self) -> None:
        self._forward_stats["bp_q_forward_calls"] += 1
        self._forward_stats["bp_model_forward_calls"] += 1

    def _record_equity_forward(self, expanded_states: int) -> None:
        self._forward_stats["bp_child_equity_forward_calls"] += 1
        self._forward_stats["bp_model_forward_calls"] += 1
        self._forward_stats["bp_max_actual_expanded_states"] = max(
            int(self._forward_stats["bp_max_actual_expanded_states"]),
            int(expanded_states),
        )

    def _record_candidate_chunk(self, expanded_states: int) -> None:
        self._forward_stats["bp_candidate_chunks"] += 1
        self._forward_stats["bp_max_actual_expanded_states"] = max(
            int(self._forward_stats["bp_max_actual_expanded_states"]),
            int(expanded_states),
        )

    def _record_parent_chunk(self) -> None:
        self._forward_stats["bp_parent_chunks"] += 1

    def _log_grid_chunk_plan(self, plan: GridChunkPlan) -> None:
        log_key = (plan.n_grid, plan.n_children)
        self._recorded_grid_chunk_plans[log_key] = {
            "n_grid": int(plan.n_grid),
            "n_children": int(plan.n_children),
            "parent_chunk_effective": int(plan.parent_chunk_effective),
            "candidate_chunk_effective": int(plan.candidate_chunk_effective),
            "expanded_states_per_forward": int(plan.expanded_states_per_forward),
            "max_expanded_states": int(plan.max_expanded_states),
        }
        if log_key in self._logged_grid_chunk_plans:
            return
        logger.info(
            "BP grid chunk plan | parent_batch_requested=%d parent_chunk_effective=%d "
            "candidate_chunk_effective=%d n_children=%d n_grid=%d "
            "expanded_states_per_forward=%d max_expanded_states=%d",
            plan.parent_batch_requested,
            plan.parent_chunk_effective,
            plan.candidate_chunk_effective,
            plan.n_children,
            plan.n_grid,
            plan.expanded_states_per_forward,
            plan.max_expanded_states,
        )
        self._logged_grid_chunk_plans.add(log_key)

    @classmethod
    def from_hyperparams(
        cls,
        target_model,
        p0_loss_fn,
        pi_loss_fn,
        hyperparams,
        *,
        q_target_model=None,
        max_expanded_states_override: Optional[int] = None,
    ) -> "BPGridTeacher":
        max_expanded_states = int(getattr(hyperparams, "bp_grid_max_expanded_states", 65536))
        if max_expanded_states_override is not None:
            max_expanded_states = int(max_expanded_states_override)
        return cls(
            target_model,
            p0_loss_fn,
            pi_loss_fn,
            q_target_model=q_target_model,
            grid_min=float(getattr(hyperparams, "bp_grid_min", 0.0)),
            grid_max=float(getattr(hyperparams, "bp_grid_max", 1.0)),
            coarse_size=int(getattr(hyperparams, "bp_grid_coarse_size", 21)),
            fine_size=int(getattr(hyperparams, "bp_grid_fine_size", 9)),
            refine=bool(getattr(hyperparams, "bp_grid_refine_enabled", True)),
            quadratic_refine=bool(getattr(hyperparams, "bp_grid_quadratic_refine", False)),
            parent_chunk_size=int(getattr(hyperparams, "bp_grid_parent_chunk_size", 2048)),
            candidate_chunk_size=int(getattr(hyperparams, "bp_grid_candidate_chunk_size", 0)),
            max_expanded_states=max_expanded_states,
            margin_scale=float(getattr(hyperparams, "bp_grid_margin_scale", 1e-3)),
            confidence_relative=bool(getattr(hyperparams, "bp_grid_confidence_relative", True)),
            confidence_min=float(getattr(hyperparams, "bp_grid_confidence_min", 0.0)),
            boundary_match_phat_eps=float(
                getattr(hyperparams, "q_boundary_match_phat_eps", 1e-2)
            ),
        )

    def compute(
        self,
        parent_state: torch.Tensor,
        children: Sequence[torch.Tensor] | torch.Tensor,
        m_list: Sequence[torch.Tensor] | torch.Tensor,
        *,
        branch: str,
        bp_pred: Optional[torch.Tensor] = None,
        mix_weight: Optional[torch.Tensor] = None,
        child_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        branch = branch.lower()
        if branch not in {"p0", "pi", "mix"}:
            raise ValueError(f"Unknown branch: {branch}")
        if branch == "mix" and mix_weight is None:
            raise ValueError("mix_weight is required for branch='mix'")

        refinancing_active = self._refinancing_active_mask(parent_state)
        if bool(refinancing_active.all()):
            return self._compute_grid_batch(
                parent_state, children, m_list, branch=branch, bp_pred=bp_pred,
                mix_weight=mix_weight, child_weights=child_weights,
            )
        if not bool(refinancing_active.any()):
            return self._compute_forced_batch(
                parent_state, children, m_list, branch=branch, bp_pred=bp_pred,
                mix_weight=mix_weight, child_weights=child_weights,
            )
        return self._compute_mixed_batch(
            parent_state, children, m_list, branch=branch, bp_pred=bp_pred,
            mix_weight=mix_weight, child_weights=child_weights,
            refinancing_active=refinancing_active,
        )

    @staticmethod
    def _refinancing_active_mask(parent_state: torch.Tensor) -> torch.Tensor:
        """``eta_t = 1`` rows, i.e. the parents that actually have a bp choice."""
        return parent_state[:, 2:3].clamp(0.0, 1.0).reshape(-1) > 0.5

    @staticmethod
    def _select_rows(value: Any, mask: torch.Tensor, n_rows: int) -> Any:
        """Slice a per-row tensor/list by a boolean mask, leaving shared 1-D weights."""
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.ndim >= 1 and int(value.shape[0]) == n_rows:
                return value[mask]
            return value
        return [item[mask] for item in value]

    @staticmethod
    def _scatter_rows(
        active_out: Dict[str, torch.Tensor],
        active_index: torch.Tensor,
        forced_out: Dict[str, torch.Tensor],
        forced_index: torch.Tensor,
        n_rows: int,
    ) -> Dict[str, torch.Tensor]:
        """Merge eta-active and eta-inactive results back into original row order."""
        if set(active_out) != set(forced_out):
            raise RuntimeError(
                "refinancing-active and refinancing-inactive results expose different "
                f"keys: {sorted(set(active_out) ^ set(forced_out))}"
            )
        merged: Dict[str, torch.Tensor] = {}
        for key, active_value in active_out.items():
            forced_value = forced_out[key]
            # The two row groups have different sizes by construction, so only
            # the per-row tail shapes must agree.
            if active_value.shape[1:] != forced_value.shape[1:]:
                raise RuntimeError(
                    f"refinancing split produced mismatched shapes for {key}: "
                    f"{tuple(active_value.shape)} vs {tuple(forced_value.shape)}"
                )
            out = torch.empty(
                (n_rows, *active_value.shape[1:]),
                device=active_value.device,
                dtype=active_value.dtype,
            )
            out[active_index] = active_value
            out[forced_index] = forced_value
            merged[key] = out
        return merged

    def _compute_mixed_batch(
        self,
        parent_state: torch.Tensor,
        children: Sequence[torch.Tensor] | torch.Tensor,
        m_list: Sequence[torch.Tensor] | torch.Tensor,
        *,
        branch: str,
        refinancing_active: torch.Tensor,
        bp_pred: Optional[torch.Tensor] = None,
        mix_weight: Optional[torch.Tensor] = None,
        child_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Split a batch by parent eta_t: grid search for eta_t=1, forced for eta_t=0."""
        n_rows = int(parent_state.shape[0])
        active_index = refinancing_active.nonzero(as_tuple=True)[0]
        forced_index = (~refinancing_active).nonzero(as_tuple=True)[0]
        active_out = self._compute_grid_batch(
            parent_state[active_index],
            self._select_rows(children, refinancing_active, n_rows),
            self._select_rows(m_list, refinancing_active, n_rows),
            branch=branch,
            bp_pred=self._select_rows(bp_pred, refinancing_active, n_rows),
            mix_weight=self._select_rows(mix_weight, refinancing_active, n_rows),
            child_weights=self._select_rows(child_weights, refinancing_active, n_rows),
        )
        forced_out = self._compute_forced_batch(
            parent_state[forced_index],
            self._select_rows(children, ~refinancing_active, n_rows),
            self._select_rows(m_list, ~refinancing_active, n_rows),
            branch=branch,
            bp_pred=self._select_rows(bp_pred, ~refinancing_active, n_rows),
            mix_weight=self._select_rows(mix_weight, ~refinancing_active, n_rows),
            child_weights=self._select_rows(child_weights, ~refinancing_active, n_rows),
        )
        return self._scatter_rows(
            active_out, active_index, forced_out, forced_index, n_rows
        )

    def _compute_grid_batch(
        self,
        parent_state: torch.Tensor,
        children: Sequence[torch.Tensor] | torch.Tensor,
        m_list: Sequence[torch.Tensor] | torch.Tensor,
        *,
        branch: str,
        bp_pred: Optional[torch.Tensor] = None,
        mix_weight: Optional[torch.Tensor] = None,
        child_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full coarse/fine bp grid search for refinancing-active (eta_t=1) parents."""
        n_children = int(_stack_children(children).shape[1])
        grid_sizes = {self.coarse_size, self.fine_size} if self.refine else {self.coarse_size}
        plans = [
            resolve_grid_chunk_plan(
                n_parent=int(parent_state.shape[0]),
                n_grid=grid_size,
                n_children=n_children,
                configured_parent_chunk=self.parent_chunk_size,
                configured_candidate_chunk=self.candidate_chunk_size,
                max_expanded_states=self.max_expanded_states,
            )
            for grid_size in grid_sizes
        ]
        for plan in plans:
            self._log_grid_chunk_plan(plan)
        parent_chunk_size = min(plan.parent_chunk_effective for plan in plans)
        if parent_state.shape[0] > parent_chunk_size:
            chunks = []
            for start in range(0, parent_state.shape[0], parent_chunk_size):
                stop = min(start + parent_chunk_size, parent_state.shape[0])
                child_chunk = _slice_child_axis(children, start, stop)
                m_chunk = _slice_child_axis(m_list, start, stop)
                bp_chunk = bp_pred[start:stop] if bp_pred is not None else None
                mix_chunk = mix_weight[start:stop] if mix_weight is not None else None
                weight_chunk = (
                    child_weights
                    if child_weights is not None and child_weights.ndim == 1
                    else child_weights[start:stop] if child_weights is not None else None
                )
                chunks.append(
                    self._compute_no_parent_chunk(
                        parent_state[start:stop],
                        child_chunk,
                        m_chunk,
                        branch=branch,
                        bp_pred=bp_chunk,
                        mix_weight=mix_chunk,
                        child_weights=weight_chunk,
                    )
                )
            return _concat_chunk_outputs(chunks)
        return self._compute_no_parent_chunk(
            parent_state,
            children,
            m_list,
            branch=branch,
            bp_pred=bp_pred,
            mix_weight=mix_weight,
            child_weights=child_weights,
        )

    def _compute_forced_batch(
        self,
        parent_state: torch.Tensor,
        children: Sequence[torch.Tensor] | torch.Tensor,
        m_list: Sequence[torch.Tensor] | torch.Tensor,
        *,
        branch: str,
        bp_pred: Optional[torch.Tensor] = None,
        mix_weight: Optional[torch.Tensor] = None,
        child_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forced Bellman backup for refinancing-inactive (eta_t=0) parents.

        No bp grid is evaluated: leverage is forced to ``b_parent``, so the
        objective is flat in the candidate dimension and one evaluation point is
        exact.
        """
        n_children = int(_stack_children(children).shape[1])
        plan = resolve_grid_chunk_plan(
            n_parent=int(parent_state.shape[0]),
            n_grid=1,
            n_children=n_children,
            configured_parent_chunk=self.parent_chunk_size,
            configured_candidate_chunk=self.candidate_chunk_size,
            max_expanded_states=self.max_expanded_states,
        )
        self._log_grid_chunk_plan(plan)
        parent_chunk_size = plan.parent_chunk_effective
        if parent_state.shape[0] > parent_chunk_size:
            chunks = []
            for start in range(0, parent_state.shape[0], parent_chunk_size):
                stop = min(start + parent_chunk_size, parent_state.shape[0])
                chunks.append(
                    self._compute_forced_no_parent_chunk(
                        parent_state[start:stop],
                        _slice_child_axis(children, start, stop),
                        _slice_child_axis(m_list, start, stop),
                        branch=branch,
                        mix_weight=mix_weight[start:stop] if mix_weight is not None else None,
                        child_weights=(
                            child_weights
                            if child_weights is not None and child_weights.ndim == 1
                            else child_weights[start:stop] if child_weights is not None else None
                        ),
                    )
                )
            return _concat_chunk_outputs(chunks)
        return self._compute_forced_no_parent_chunk(
            parent_state,
            children,
            m_list,
            branch=branch,
            mix_weight=mix_weight,
            child_weights=child_weights,
        )

    def _compute_forced_no_parent_chunk(
        self,
        parent_state: torch.Tensor,
        children: Sequence[torch.Tensor] | torch.Tensor,
        m_list: Sequence[torch.Tensor] | torch.Tensor,
        *,
        branch: str,
        mix_weight: Optional[torch.Tensor] = None,
        child_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            self._record_parent_chunk()
            forced_grid = parent_state[:, 0:1].detach()
            forced = self._evaluate_grid(
                parent_state, children, m_list, forced_grid,
                branch=branch, mix_weight=mix_weight, child_weights=child_weights,
            )
            return self._finalize_forced_result(parent_state, forced, diagnostics=True)

    def _finalize_forced_result(
        self,
        parent_state: torch.Tensor,
        forced: Dict[str, torch.Tensor],
        *,
        diagnostics: bool,
    ) -> Dict[str, torch.Tensor]:
        """Build a grid-shaped result for eta_t=0 rows without running the grid.

        ``value_star`` is the exact Bellman value at the forced transition
        ``b_child = b_parent``. Because the eta_t=0 objective is flat in the
        candidate dimension, that single point is already the maximizer, so the
        bp grid adds nothing. Grid-only diagnostics are returned as NaN/sentinel
        so no candidate forward is wasted on them.
        """
        b_parent = parent_state[:, 0:1].detach()
        value_star = forced["value_grid"][:, 0:1].detach()
        if not diagnostics:
            return {
                "value_star": value_star,
                "bp_star": b_parent,
                "bp_star_grid": b_parent,
            }

        coarse_size = max(2, int(self.coarse_size))
        local_size = max(2, int(self.fine_size)) if self.refine else coarse_size
        n_rows = int(parent_state.shape[0])
        device = parent_state.device
        dtype = parent_state.dtype

        def _nan(rows: int, width: int) -> torch.Tensor:
            return torch.full((rows, width), float("nan"), device=device, dtype=dtype)

        def _broadcast(value: torch.Tensor, width: int) -> torch.Tensor:
            return value.expand(n_rows, width).clone()

        forced_value = value_star
        result = {
            "bp_grid": _broadcast(forced["bp_grid"], local_size),
            "value_grid": _nan(n_rows, local_size),
            "cashflow_grid_mean": _nan(n_rows, local_size),
            "continuation_grid_mean": _nan(n_rows, local_size),
            "argmax_index": torch.zeros((n_rows, 1), device=device, dtype=torch.long),
            "q_issue_grid": _nan(n_rows, local_size),
            "p_child_grid_mean": _nan(n_rows, local_size),
            "default_grid_mean": _nan(n_rows, local_size),
            # Row-level (not candidate-shaped) diagnostics keep the ``(n, 1)``
            # layout used by ``_finalize_grid_result`` so mixed batches can be
            # scattered back into a single tensor.
            "eta_next_active_share": forced["eta_next_active_share"][:, 0:1].detach(),
            "child_b_mean": _broadcast(forced["child_b_mean"], local_size),
            "child_b_eta0_mean": _broadcast(forced["child_b_eta0_mean"], local_size),
            "child_b_eta1_mean": _broadcast(forced["child_b_eta1_mean"], local_size),
            "bp_star": b_parent,
            "bp_star_grid": b_parent,
            "value_star": value_star,
            # No candidate comparison exists, so the margin/confidence are undefined.
            "top2_margin": _nan(n_rows, 1),
            "coarse_top2_margin": _nan(n_rows, 1),
            "fine_top2_margin": _nan(n_rows, 1),
            "confidence": torch.zeros((n_rows, 1), device=device, dtype=dtype),
            "boundary_low": torch.zeros((n_rows, 1), device=device, dtype=dtype),
            "boundary_high": torch.zeros((n_rows, 1), device=device, dtype=dtype),
            "refi_active": torch.zeros((n_rows, 1), device=device, dtype=dtype),
            "coarse_bp_grid": _broadcast(forced["bp_grid"], coarse_size),
            "coarse_value_grid": _nan(n_rows, coarse_size),
            "coarse_cashflow_grid_mean": _nan(n_rows, coarse_size),
            "coarse_continuation_grid_mean": _nan(n_rows, coarse_size),
            "coarse_q_issue_grid": _nan(n_rows, coarse_size),
            "coarse_p_child_grid_mean": _nan(n_rows, coarse_size),
            "coarse_default_grid_mean": _nan(n_rows, coarse_size),
            "coarse_eta_next_active_share": _broadcast(
                forced["eta_next_active_share"], coarse_size
            ),
            "coarse_child_b_mean": _broadcast(forced["child_b_mean"], coarse_size),
            "coarse_child_b_eta0_mean": _broadcast(
                forced["child_b_eta0_mean"], coarse_size
            ),
            "coarse_child_b_eta1_mean": _broadcast(
                forced["child_b_eta1_mean"], coarse_size
            ),
            "local_value_left": _nan(n_rows, 1),
            "local_value_right": _nan(n_rows, 1),
            "q_issue_at_star": forced["q_issue_grid"][:, 0:1].detach(),
            "p_child_at_star": forced["p_child_grid_mean"][:, 0:1].detach(),
            "default_at_star": forced["default_grid_mean"][:, 0:1].detach(),
            # Forced leverage is already the maximizer, so there is no regret.
            "value_pred": forced_value,
            "regret": torch.zeros_like(forced_value),
        }
        if "q_issue_unit_grid" in forced:
            for key in (
                "q_issue_claim_grid",
                "q_issue_unit_grid",
                "q_issue_realized_default_mask_grid",
                "q_issue_recovery_grid",
                "q_issue_candidate_phat_grid",
                "candidate_phat_gate_used_for_q_issue",
            ):
                result[key] = _nan(n_rows, local_size)
            for key in (
                "q_issue_max_abs_dq_db",
                "q_issue_candidate_default_share",
                "q_issue_boundary_gap_mean",
                "q_issue_unit_mean",
                "q_issue_unit_p50",
                "q_issue_unit_p95",
                "q_issue_unit_max",
            ):
                result[key] = _nan(n_rows, 1)
        return result

    def compute_value_target(
        self,
        parent_state: torch.Tensor,
        children: Sequence[torch.Tensor] | torch.Tensor,
        m_list: Sequence[torch.Tensor] | torch.Tensor,
        *,
        branch: str,
        child_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute only value targets needed by cached P/Q training."""
        branch = branch.lower()
        if branch not in {"p0", "pi"}:
            raise ValueError(f"compute_value_target only supports p0/pi, got {branch!r}")
        refinancing_active = self._refinancing_active_mask(parent_state)
        if bool(refinancing_active.all()):
            return self._compute_value_target_grid_batch(
                parent_state, children, m_list, branch=branch,
                child_weights=child_weights,
            )
        if not bool(refinancing_active.any()):
            return self._compute_value_target_forced_batch(
                parent_state, children, m_list, branch=branch,
                child_weights=child_weights,
            )
        n_rows = int(parent_state.shape[0])
        active_index = refinancing_active.nonzero(as_tuple=True)[0]
        forced_index = (~refinancing_active).nonzero(as_tuple=True)[0]
        active_out = self._compute_value_target_grid_batch(
            parent_state[active_index],
            self._select_rows(children, refinancing_active, n_rows),
            self._select_rows(m_list, refinancing_active, n_rows),
            branch=branch,
            child_weights=self._select_rows(child_weights, refinancing_active, n_rows),
        )
        forced_out = self._compute_value_target_forced_batch(
            parent_state[forced_index],
            self._select_rows(children, ~refinancing_active, n_rows),
            self._select_rows(m_list, ~refinancing_active, n_rows),
            branch=branch,
            child_weights=self._select_rows(child_weights, ~refinancing_active, n_rows),
        )
        return self._scatter_rows(
            active_out, active_index, forced_out, forced_index, n_rows
        )

    def _compute_value_target_grid_batch(
        self,
        parent_state: torch.Tensor,
        children: Sequence[torch.Tensor] | torch.Tensor,
        m_list: Sequence[torch.Tensor] | torch.Tensor,
        *,
        branch: str,
        child_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        n_children = int(_stack_children(children).shape[1])
        grid_sizes = {self.coarse_size, self.fine_size} if self.refine else {self.coarse_size}
        plans = [
            resolve_grid_chunk_plan(
                n_parent=int(parent_state.shape[0]),
                n_grid=grid_size,
                n_children=n_children,
                configured_parent_chunk=self.parent_chunk_size,
                configured_candidate_chunk=self.candidate_chunk_size,
                max_expanded_states=self.max_expanded_states,
            )
            for grid_size in grid_sizes
        ]
        for plan in plans:
            self._log_grid_chunk_plan(plan)
        parent_chunk_size = min(plan.parent_chunk_effective for plan in plans)
        if parent_state.shape[0] > parent_chunk_size:
            chunks = []
            for start in range(0, parent_state.shape[0], parent_chunk_size):
                stop = min(start + parent_chunk_size, parent_state.shape[0])
                chunks.append(
                    self._compute_value_target_no_parent_chunk(
                        parent_state[start:stop],
                        _slice_child_axis(children, start, stop),
                        _slice_child_axis(m_list, start, stop),
                        branch=branch,
                        child_weights=(
                            child_weights
                            if child_weights is not None and child_weights.ndim == 1
                            else child_weights[start:stop] if child_weights is not None else None
                        ),
                    )
                )
            return _concat_chunk_outputs(chunks)
        return self._compute_value_target_no_parent_chunk(
            parent_state,
            children,
            m_list,
            branch=branch,
            child_weights=child_weights,
        )

    def _compute_value_target_forced_batch(
        self,
        parent_state: torch.Tensor,
        children: Sequence[torch.Tensor] | torch.Tensor,
        m_list: Sequence[torch.Tensor] | torch.Tensor,
        *,
        branch: str,
        child_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forced Bellman value target for eta_t=0 parents (no bp grid)."""
        n_children = int(_stack_children(children).shape[1])
        plan = resolve_grid_chunk_plan(
            n_parent=int(parent_state.shape[0]),
            n_grid=1,
            n_children=n_children,
            configured_parent_chunk=self.parent_chunk_size,
            configured_candidate_chunk=self.candidate_chunk_size,
            max_expanded_states=self.max_expanded_states,
        )
        self._log_grid_chunk_plan(plan)
        parent_chunk_size = plan.parent_chunk_effective
        if parent_state.shape[0] > parent_chunk_size:
            chunks = []
            for start in range(0, parent_state.shape[0], parent_chunk_size):
                stop = min(start + parent_chunk_size, parent_state.shape[0])
                chunks.append(
                    self._compute_value_target_forced_no_parent_chunk(
                        parent_state[start:stop],
                        _slice_child_axis(children, start, stop),
                        _slice_child_axis(m_list, start, stop),
                        branch=branch,
                        child_weights=(
                            child_weights
                            if child_weights is not None and child_weights.ndim == 1
                            else child_weights[start:stop] if child_weights is not None else None
                        ),
                    )
                )
            return _concat_chunk_outputs(chunks)
        return self._compute_value_target_forced_no_parent_chunk(
            parent_state,
            children,
            m_list,
            branch=branch,
            child_weights=child_weights,
        )

    def _compute_value_target_forced_no_parent_chunk(
        self,
        parent_state: torch.Tensor,
        children: Sequence[torch.Tensor] | torch.Tensor,
        m_list: Sequence[torch.Tensor] | torch.Tensor,
        *,
        branch: str,
        child_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            self._record_parent_chunk()
            forced_grid = parent_state[:, 0:1].detach()
            forced = self._evaluate_grid(
                parent_state, children, m_list, forced_grid,
                branch=branch, child_weights=child_weights,
            )
            return self._finalize_forced_result(parent_state, forced, diagnostics=False)

    def compute_multi_j_branches(
        self,
        parent_states: Sequence[torch.Tensor],
        children: Sequence[torch.Tensor] | torch.Tensor,
        m_list: Sequence[torch.Tensor] | torch.Tensor,
        *,
        branches: Sequence[str],
        prefix_child_counts: Sequence[int],
        child_weights: Optional[torch.Tensor] = None,
        bp_preds: Optional[Sequence[Optional[torch.Tensor]]] = None,
        mix_weights: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ) -> List[Dict[int, Dict[str, torch.Tensor]]]:
        """Evaluate several branches over nested child prefixes sharing one forward.

        ``children``/``m_list`` must hold the largest (Jmax) child set. Every entry
        of ``prefix_child_counts`` is evaluated on its nested prefix along the child
        axis, which reproduces ``slice_frozen_transition_data`` semantics because the
        exact-eta expansion keeps continuous child ``j`` at expanded positions
        ``2j``/``2j+1``.

        Sharing rules (all of them are exact, not approximations):

        * the branch-independent candidate child-equity block
          (``P_child``/``bar_z_child``/``child_b_grid``) depends only on the children,
          the candidate bp grid and parent leverage, so it is computed once per
          coarse candidate chunk and reused by every branch *and* every prefix;
        * the coarse uniform candidate grid is identical for every branch;
        * ``q_current`` and the coarse ``q_issue`` grid are prefix independent, so
          they are computed once per branch;
        * fine-grid candidates stay branch- and prefix-specific because the coarse
          argmax (and therefore the local grid) differs between them.

        Returns one ``{prefix_child_count: result}`` mapping per requested branch, in
        the same order as ``branches``. Each result matches ``compute(...)`` on the
        sliced child prefix.
        """
        branch_labels = [str(branch).lower() for branch in branches]
        for branch in branch_labels:
            if branch not in {"p0", "pi", "mix"}:
                raise ValueError(f"Unknown branch: {branch}")
            if branch == "mix" and mix_weights is None:
                raise ValueError("mix_weights is required for branch='mix'")
        states = [
            state if isinstance(state, torch.Tensor) else torch.as_tensor(state)
            for state in parent_states
        ]
        if not states:
            raise ValueError("compute_multi_j_branches requires at least one parent state tensor")
        batch_size = int(states[0].shape[0])
        for state in states:
            if int(state.shape[0]) != batch_size:
                raise ValueError("all branch parent states must share the same batch size")
        reference_leverage = states[0][:, 0:1]
        for state in states[1:]:
            if not torch.equal(state[:, 0:1], reference_leverage):
                raise ValueError(
                    "compute_multi_j_branches requires identical parent leverage (column 0) "
                    "across branches so the child-equity block can be shared"
                )
        # ``bp_t`` is a control only for eta_t = 1 parents. A batch is either
        # fully refinancing-active (coarse/fine candidate grid) or fully inactive
        # (one forced candidate ``b_parent``, since the objective is flat in the
        # candidate dimension there). Mixing both in one shared-forward pass would
        # need two different candidate axes, so it is rejected explicitly.
        refinancing_active = self._refinancing_active_mask(states[0])
        refinancing_inactive = ~refinancing_active
        if bool(refinancing_active.any()) and bool(refinancing_inactive.any()):
            raise ValueError(
                "compute_multi_j_branches requires a uniform parent eta_t batch; got a "
                "mix of refinancing-active (eta_t = 1) and refinancing-inactive "
                "(eta_t = 0) parents"
            )
        children_tensor = _stack_children(children)
        m_tensor = _stack_m(m_list)
        n_children = int(children_tensor.shape[1])
        counts = sorted({int(value) for value in prefix_child_counts})
        if not counts:
            raise ValueError("compute_multi_j_branches requires at least one prefix child count")
        if counts[0] < 1 or counts[-1] > n_children:
            raise ValueError(
                f"prefix child counts must lie in [1, {n_children}], got {counts}"
            )
        if bp_preds is not None and len(bp_preds) != len(states):
            raise ValueError("bp_preds must provide one entry per branch")
        if mix_weights is not None and len(mix_weights) != len(states):
            raise ValueError("mix_weights must provide one entry per branch")

        grid_sizes = {self.coarse_size, self.fine_size} if self.refine else {self.coarse_size}
        plans = [
            resolve_grid_chunk_plan(
                n_parent=batch_size,
                n_grid=grid_size,
                n_children=n_children,
                configured_parent_chunk=self.parent_chunk_size,
                configured_candidate_chunk=self.candidate_chunk_size,
                max_expanded_states=self.max_expanded_states,
            )
            for grid_size in grid_sizes
        ]
        for plan in plans:
            self._log_grid_chunk_plan(plan)
        parent_chunk_size = min(plan.parent_chunk_effective for plan in plans)

        if len(counts) > 1:
            self._forward_stats["bp_multi_j_reuse_enabled"] = True
        if len(states) > 1:
            self._forward_stats["bp_branch_reuse_enabled"] = True

        collected: List[Dict[int, List[Dict[str, torch.Tensor]]]] = [
            {count: [] for count in counts} for _ in states
        ]
        with torch.no_grad():
            for start in range(0, batch_size, parent_chunk_size):
                stop = min(start + parent_chunk_size, batch_size)
                chunk_states = [state[start:stop] for state in states]
                child_chunk = children_tensor[start:stop]
                m_chunk = m_tensor[start:stop]
                weights_chunk = (
                    child_weights[start:stop]
                    if child_weights is not None and child_weights.ndim > 1
                    else child_weights
                )
                self._record_parent_chunk()
                if bool(refinancing_inactive.any()):
                    # eta_t = 0: one forced candidate, leverage is b_parent.
                    coarse_grid = chunk_states[0][:, 0:1].detach()
                else:
                    coarse_grid = self._uniform_grid(chunk_states[0], self.coarse_size)
                n_grid = int(coarse_grid.shape[1])
                candidate_plan = resolve_grid_chunk_plan(
                    n_parent=stop - start,
                    n_grid=n_grid,
                    n_children=n_children,
                    configured_parent_chunk=stop - start,
                    configured_candidate_chunk=self.candidate_chunk_size,
                    max_expanded_states=self.max_expanded_states,
                )
                if candidate_plan.parent_chunk_effective != stop - start:
                    raise RuntimeError(
                        "BP parent batch reached grid evaluation above the resolved hard cap; "
                        "compute_multi_j_branches() must apply the parent chunk plan first"
                    )
                self._log_grid_chunk_plan(candidate_plan)
                candidate_chunk = candidate_plan.candidate_chunk_effective
                q_current_by_branch: List[Optional[torch.Tensor]] = [
                    None for _ in states
                ]
                coarse_parts: List[Dict[int, List[Dict[str, torch.Tensor]]]] = [
                    {count: [] for count in counts} for _ in states
                ]
                for grid_start in range(0, n_grid, candidate_chunk):
                    grid_stop = min(grid_start + candidate_chunk, n_grid)
                    sub_grid = coarse_grid[:, grid_start:grid_stop]
                    equity = self._child_equity_block(
                        child_chunk,
                        sub_grid,
                        chunk_states[0][:, 0:1],
                        chunk_states[0][:, 2:3],
                    )
                    self._record_candidate_chunk(int(equity.p_child.numel()))
                    for index, branch in enumerate(branch_labels):
                        state_chunk = chunk_states[index]
                        if q_current_by_branch[index] is None:
                            q_current_by_branch[index] = _target_q_claim(
                                self.q_target_model,
                                state_chunk,
                                equity_model=self.target_model,
                            )
                            self._record_q_forward()
                        q_issue, q_diagnostics = self._q_issue_grid(state_chunk, sub_grid)
                        for count in counts:
                            coarse_parts[index][count].append(
                                self._branch_objective_grid(
                                    state_chunk,
                                    sub_grid,
                                    branch=branch,
                                    p_child=equity.p_child[:, :, :count],
                                    bar_z_child=equity.bar_z_child[:, :, :count],
                                    child_b_grid=equity.child_b_grid[:, :, :count],
                                    child_eta_next=child_chunk[:, :count, 2],
                                    m_tensor=m_chunk[:, :count, :],
                                    q_current=q_current_by_branch[index],
                                    q_issue=q_issue,
                                    q_diagnostics=q_diagnostics,
                                    mix_weight=(
                                        mix_weights[index][start:stop]
                                        if mix_weights is not None and mix_weights[index] is not None
                                        else None
                                    ),
                                    child_weights=_slice_prefix_weights(weights_chunk, count),
                                )
                            )
                for index, branch in enumerate(branch_labels):
                    for count in counts:
                        parts = coarse_parts[index][count]
                        coarse = {
                            "bp_grid": torch.cat([part["bp_grid"] for part in parts], dim=1),
                            "value_grid": torch.cat([part["value_grid"] for part in parts], dim=1),
                            "cashflow_grid_mean": torch.cat(
                                [part["cashflow_grid_mean"] for part in parts], dim=1
                            ),
                            "continuation_grid_mean": torch.cat(
                                [part["continuation_grid_mean"] for part in parts], dim=1
                            ),
                            "q_issue_grid": torch.cat(
                                [part["q_issue_grid"] for part in parts], dim=1
                            ),
                            "p_child_grid_mean": torch.cat(
                                [part["p_child_grid_mean"] for part in parts], dim=1
                            ),
                            "default_grid_mean": torch.cat(
                                [part["default_grid_mean"] for part in parts], dim=1
                            ),
                            "eta_next_active_share": torch.cat(
                                [part["eta_next_active_share"] for part in parts], dim=1
                            ),
                            "child_b_mean": torch.cat(
                                [part["child_b_mean"] for part in parts], dim=1
                            ),
                            "child_b_eta0_mean": torch.cat(
                                [part["child_b_eta0_mean"] for part in parts], dim=1
                            ),
                            "child_b_eta1_mean": torch.cat(
                                [part["child_b_eta1_mean"] for part in parts], dim=1
                            ),
                        }
                        coarse["argmax_index"] = coarse["value_grid"].argmax(
                            dim=1, keepdim=True
                        )
                        bp_pred = None
                        if bp_preds is not None and bp_preds[index] is not None:
                            bp_pred = bp_preds[index][start:stop]
                        if bool(refinancing_inactive.any()):
                            # No candidate comparison exists for eta_t = 0, so the
                            # grid-shaped diagnostics are NaN/sentinel and the star
                            # is the forced leverage.
                            result = self._finalize_forced_result(
                                chunk_states[index], coarse, diagnostics=True
                            )
                        else:
                            result = self._finalize_grid_result(
                                chunk_states[index],
                                child_chunk[:, :count],
                                m_chunk[:, :count],
                                coarse,
                                branch=branch,
                                mix_weight=(
                                    mix_weights[index][start:stop]
                                    if mix_weights is not None and mix_weights[index] is not None
                                    else None
                                ),
                                child_weights=_slice_prefix_weights(weights_chunk, count),
                                bp_pred=bp_pred,
                                diagnostics=True,
                            )
                        collected[index][count].append(result)

        results: List[Dict[int, Dict[str, torch.Tensor]]] = []
        for index in range(len(states)):
            results.append(
                {
                    count: _concat_chunk_outputs(collected[index][count])
                    for count in counts
                }
            )
        return results

    def _compute_value_target_no_parent_chunk(
        self,
        parent_state: torch.Tensor,
        children: Sequence[torch.Tensor] | torch.Tensor,
        m_list: Sequence[torch.Tensor] | torch.Tensor,
        *,
        branch: str,
        child_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            self._record_parent_chunk()
            coarse_grid = self._uniform_grid(parent_state, self.coarse_size)
            coarse = self._evaluate_grid(
                parent_state, children, m_list, coarse_grid,
                branch=branch, child_weights=child_weights,
            )
            return self._finalize_grid_result(
                parent_state, children, m_list, coarse, branch=branch,
                child_weights=child_weights, diagnostics=False,
            )

    def _compute_no_parent_chunk(
        self,
        parent_state: torch.Tensor,
        children: Sequence[torch.Tensor] | torch.Tensor,
        m_list: Sequence[torch.Tensor] | torch.Tensor,
        *,
        branch: str,
        bp_pred: Optional[torch.Tensor] = None,
        mix_weight: Optional[torch.Tensor] = None,
        child_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            self._record_parent_chunk()
            coarse_grid = self._uniform_grid(parent_state, self.coarse_size)
            coarse = self._evaluate_grid(
                parent_state, children, m_list, coarse_grid, branch=branch,
                mix_weight=mix_weight, child_weights=child_weights,
            )
            return self._finalize_grid_result(
                parent_state, children, m_list, coarse, branch=branch,
                mix_weight=mix_weight, child_weights=child_weights,
                bp_pred=bp_pred, diagnostics=True,
            )

    def _finalize_grid_result(
        self,
        parent_state: torch.Tensor,
        children: Sequence[torch.Tensor] | torch.Tensor,
        m_list: Sequence[torch.Tensor] | torch.Tensor,
        coarse: Dict[str, torch.Tensor],
        *,
        branch: str,
        mix_weight: Optional[torch.Tensor] = None,
        child_weights: Optional[torch.Tensor] = None,
        bp_pred: Optional[torch.Tensor] = None,
        diagnostics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Local refinement, star selection and diagnostics from a coarse grid result.

        ``coarse`` must contain the coarse candidate objective grids (either from
        ``_evaluate_grid`` or assembled from shared candidate chunks). Keeping this
        step separate lets the multi-J/multi-branch path reuse one coarse
        child-equity forward while still refining each (branch, J) locally.
        """
        if self.refine:
            fine_grid = self._local_fine_grid(coarse["bp_grid"], coarse["argmax_index"])
            result = self._evaluate_grid(
                parent_state, children, m_list, fine_grid, branch=branch,
                mix_weight=mix_weight, child_weights=child_weights,
            )
        else:
            result = coarse

        argmax_index = result["argmax_index"]
        bp_star_grid = _gather_by_index(result["bp_grid"], argmax_index)
        bp_star = _quadratic_refine(
            result["bp_grid"],
            result["value_grid"],
            argmax_index,
            bp_star_grid,
            self.quadratic_refine,
        ).clamp(self.grid_min, self.grid_max)
        if self.quadratic_refine:
            refined_eval = self._evaluate_grid(
                parent_state,
                children,
                m_list,
                bp_star,
                branch=branch,
                mix_weight=mix_weight,
                child_weights=child_weights,
            )
            value_star = refined_eval["value_grid"][:, 0:1]
            q_issue_at_star = refined_eval["q_issue_grid"][:, 0:1]
            p_child_at_star = refined_eval["p_child_grid_mean"][:, 0:1]
            default_at_star = refined_eval["default_grid_mean"][:, 0:1]
        else:
            value_star = _gather_by_index(result["value_grid"], argmax_index)
            q_issue_at_star = _gather_by_index(result["q_issue_grid"], argmax_index)
            p_child_at_star = _gather_by_index(result["p_child_grid_mean"], argmax_index)
            default_at_star = _gather_by_index(result["default_grid_mean"], argmax_index)

        if not diagnostics:
            return {
                "value_star": value_star.detach(),
                "bp_star": bp_star.detach(),
                "bp_star_grid": bp_star_grid.detach(),
            }

        coarse_top2_margin, _ = _safe_top2_margin(coarse["value_grid"])
        fine_top2_margin, _ = _safe_top2_margin(result["value_grid"])
        coarse_value_star = _gather_by_index(coarse["value_grid"], coarse["argmax_index"])
        if self.confidence_relative:
            value_scale = coarse_value_star.abs().clamp_min(1e-8)
            relative_margin = coarse_top2_margin / value_scale
            confidence = (relative_margin / self.margin_scale).clamp(self.confidence_min, 1.0)
        else:
            confidence = (coarse_top2_margin / self.margin_scale).clamp(self.confidence_min, 1.0)
        target_available = (
            self._refinancing_active_mask(parent_state)
            .to(parent_state.dtype)
            .reshape(-1, 1)
        )

        result.update(
            {
                "bp_star": bp_star.detach(),
                "bp_star_grid": bp_star_grid.detach(),
                "value_star": value_star.detach(),
                "top2_margin": coarse_top2_margin.detach(),
                "coarse_top2_margin": coarse_top2_margin.detach(),
                "fine_top2_margin": fine_top2_margin.detach(),
                "confidence": confidence.detach(),
                "boundary_low": (
                    bp_star_grid <= self.grid_min + 1e-8
                ).to(parent_state.dtype).detach(),
                "boundary_high": (
                    bp_star_grid >= self.grid_max - 1e-8
                ).to(parent_state.dtype).detach(),
                # ``bp_t`` is a real control only when eta_t = 1. This field is
                # the parent eta_t indicator, not a statement that a bp target
                # exists for every row.
                "refi_active": target_available.detach(),
                "eta_next_active_share": result["eta_next_active_share"][:, 0:1].detach(),
                "coarse_bp_grid": coarse["bp_grid"].detach(),
                "coarse_value_grid": coarse["value_grid"].detach(),
                "coarse_cashflow_grid_mean": coarse["cashflow_grid_mean"].detach(),
                "coarse_continuation_grid_mean": coarse["continuation_grid_mean"].detach(),
                "coarse_q_issue_grid": coarse["q_issue_grid"].detach(),
                "coarse_p_child_grid_mean": coarse["p_child_grid_mean"].detach(),
                "coarse_default_grid_mean": coarse["default_grid_mean"].detach(),
                "coarse_eta_next_active_share": coarse["eta_next_active_share"].detach(),
                "coarse_child_b_mean": coarse["child_b_mean"].detach(),
                "coarse_child_b_eta0_mean": coarse["child_b_eta0_mean"].detach(),
                "coarse_child_b_eta1_mean": coarse["child_b_eta1_mean"].detach(),
                "local_value_left": result["value_grid"][:, 0:1].detach(),
                "local_value_right": result["value_grid"][:, -1:].detach(),
                "q_issue_at_star": q_issue_at_star.detach(),
                "p_child_at_star": p_child_at_star.detach(),
                "default_at_star": default_at_star.detach(),
            }
        )

        if "q_issue_unit_grid" in result:
            q_claim = result["q_issue_claim_grid"]
            q_recovery = result["q_issue_recovery_grid"]
            q_phat = result["q_issue_candidate_phat_grid"]
            q_unit = result["q_issue_unit_grid"]
            if q_claim.shape[1] > 1:
                db = result["bp_grid"][:, 1:] - result["bp_grid"][:, :-1]
                dq = q_claim[:, 1:] - q_claim[:, :-1]
                max_abs_dq_db = (dq / db.clamp_min(1e-12)).abs().amax(dim=1, keepdim=True)
            else:
                max_abs_dq_db = torch.zeros_like(q_claim[:, :1])
            boundary_mask = q_phat.abs() <= self.boundary_match_phat_eps
            boundary_gap = (q_claim - q_recovery).abs()
            boundary_count = boundary_mask.sum(dim=1, keepdim=True)
            boundary_mean = (
                (boundary_gap * boundary_mask.to(boundary_gap.dtype)).sum(dim=1, keepdim=True)
                / boundary_count.clamp_min(1).to(boundary_gap.dtype)
            )
            boundary_mean = torch.where(
                boundary_count > 0,
                boundary_mean,
                torch.full_like(boundary_mean, float("nan")),
            )
            result.update({
                "q_current_claim": result["q_current_claim"].detach(),
                "coarse_q_issue_claim_grid": coarse["q_issue_claim_grid"].detach(),
                "candidate_phat_gate_used_for_q_issue": result[
                    "candidate_phat_gate_used_for_q_issue"
                ].detach(),
                "q_issue_max_abs_dq_db": max_abs_dq_db.detach(),
                "q_issue_candidate_default_share": result[
                    "q_issue_realized_default_mask_grid"
                ].mean(dim=1, keepdim=True).detach(),
                "q_issue_boundary_gap_mean": boundary_mean.detach(),
                "q_issue_unit_mean": q_unit.mean(dim=1, keepdim=True).detach(),
                "q_issue_unit_p50": torch.quantile(q_unit, 0.50, dim=1, keepdim=True).detach(),
                "q_issue_unit_p95": torch.quantile(q_unit, 0.95, dim=1, keepdim=True).detach(),
                "q_issue_unit_max": q_unit.max(dim=1, keepdim=True).values.detach(),
            })

        if bp_pred is not None:
            pred_grid = bp_pred.detach().clamp(self.grid_min, self.grid_max).reshape(-1, 1)
            pred_eval = self._evaluate_grid(
                parent_state, children, m_list, pred_grid, branch=branch,
                mix_weight=mix_weight, child_weights=child_weights,
            )
            value_pred = pred_eval["value_grid"][:, 0:1]
            result["value_pred"] = value_pred.detach()
            result["regret"] = (value_star - value_pred).clamp_min(0.0).detach()
        else:
            result["value_pred"] = torch.zeros_like(value_star)
            result["regret"] = torch.zeros_like(value_star)

        return {k: v.detach() for k, v in result.items()}

    def _uniform_grid(self, parent_state: torch.Tensor, size: int) -> torch.Tensor:
        grid = torch.linspace(
            self.grid_min,
            self.grid_max,
            steps=max(2, int(size)),
            device=parent_state.device,
            dtype=parent_state.dtype,
        )
        return grid.unsqueeze(0).expand(parent_state.shape[0], grid.numel())

    def _local_fine_grid(self, coarse_grid: torch.Tensor, argmax_index: torch.Tensor) -> torch.Tensor:
        batch_size, n_grid = coarse_grid.shape
        idx = argmax_index.reshape(-1)
        rows = torch.arange(batch_size, device=coarse_grid.device)
        left_idx = (idx - 1).clamp(0, n_grid - 1)
        right_idx = (idx + 1).clamp(0, n_grid - 1)
        left = coarse_grid[rows, left_idx]
        right = coarse_grid[rows, right_idx]
        step = (self.grid_max - self.grid_min) / max(n_grid - 1, 1)
        left = torch.where(idx <= 0, (coarse_grid[rows, idx] - step).clamp_min(self.grid_min), left)
        right = torch.where(idx >= n_grid - 1, (coarse_grid[rows, idx] + step).clamp_max(self.grid_max), right)
        alpha = torch.linspace(0.0, 1.0, steps=self.fine_size, device=coarse_grid.device, dtype=coarse_grid.dtype)
        return left.unsqueeze(1) + (right - left).unsqueeze(1) * alpha.unsqueeze(0)

    def _resolve_candidate_chunk_size(
        self,
        batch_size: int,
        n_grid: int,
        n_children: int,
    ) -> int:
        return resolve_grid_chunk_plan(
            n_parent=batch_size,
            n_grid=n_grid,
            n_children=n_children,
            configured_parent_chunk=batch_size,
            configured_candidate_chunk=self.candidate_chunk_size,
            max_expanded_states=self.max_expanded_states,
        ).candidate_chunk_effective

    def _evaluate_grid(
        self,
        parent_state: torch.Tensor,
        children: Sequence[torch.Tensor] | torch.Tensor,
        m_list: Sequence[torch.Tensor] | torch.Tensor,
        bp_grid: torch.Tensor,
        *,
        branch: str,
        mix_weight: Optional[torch.Tensor] = None,
        child_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, n_grid = bp_grid.shape
        children_tensor = _stack_children(children)
        m_tensor = _stack_m(m_list)
        n_children = int(children_tensor.shape[1])
        q_current = _target_q_claim(
            self.q_target_model,
            parent_state,
            equity_model=self.target_model,
        )
        plan = resolve_grid_chunk_plan(
            n_parent=batch_size,
            n_grid=n_grid,
            n_children=n_children,
            configured_parent_chunk=batch_size,
            configured_candidate_chunk=self.candidate_chunk_size,
            max_expanded_states=self.max_expanded_states,
        )
        if plan.parent_chunk_effective != batch_size:
            raise RuntimeError(
                "BP parent batch reached grid evaluation above the resolved hard cap; "
                "compute() must apply the parent chunk plan first"
            )
        chunk_size = plan.candidate_chunk_effective
        self._log_grid_chunk_plan(plan)
        if chunk_size >= n_grid:
            return self._evaluate_grid_chunk(
                parent_state, children_tensor, m_tensor, bp_grid, branch=branch,
                mix_weight=mix_weight, child_weights=child_weights, q_current=q_current,
            )

        chunks = []
        for start in range(0, n_grid, chunk_size):
            stop = min(start + chunk_size, n_grid)
            chunks.append(
                self._evaluate_grid_chunk(
                    parent_state,
                    children_tensor,
                    m_tensor,
                    bp_grid[:, start:stop],
                    branch=branch,
                    mix_weight=mix_weight,
                    child_weights=child_weights,
                    q_current=q_current,
                )
            )
        merged = {
            "bp_grid": torch.cat([c["bp_grid"] for c in chunks], dim=1),
            "value_grid": torch.cat([c["value_grid"] for c in chunks], dim=1),
            "cashflow_grid_mean": torch.cat([c["cashflow_grid_mean"] for c in chunks], dim=1),
            "continuation_grid_mean": torch.cat([c["continuation_grid_mean"] for c in chunks], dim=1),
            "q_issue_grid": torch.cat([c["q_issue_grid"] for c in chunks], dim=1),
            "p_child_grid_mean": torch.cat([c["p_child_grid_mean"] for c in chunks], dim=1),
            "default_grid_mean": torch.cat([c["default_grid_mean"] for c in chunks], dim=1),
            "eta_next_active_share": torch.cat([c["eta_next_active_share"] for c in chunks], dim=1),
            "child_b_mean": torch.cat([c["child_b_mean"] for c in chunks], dim=1),
            "child_b_eta0_mean": torch.cat([c["child_b_eta0_mean"] for c in chunks], dim=1),
            "child_b_eta1_mean": torch.cat([c["child_b_eta1_mean"] for c in chunks], dim=1),
            "argmax_index": torch.cat([c["value_grid"] for c in chunks], dim=1).argmax(dim=1, keepdim=True),
        }
        for key in (
            "q_issue_claim_grid",
            "q_issue_unit_grid",
            "q_issue_realized_default_mask_grid",
            "q_issue_recovery_grid",
            "q_issue_candidate_phat_grid",
            "candidate_phat_gate_used_for_q_issue",
        ):
            if key in chunks[0]:
                merged[key] = torch.cat([chunk[key] for chunk in chunks], dim=1)
        if "q_current_claim" in chunks[0]:
            merged["q_current_claim"] = chunks[0]["q_current_claim"]
        return merged

    def _child_equity_block(
        self,
        children: Sequence[torch.Tensor] | torch.Tensor,
        bp_grid: torch.Tensor,
        b_parent: torch.Tensor,
        eta_current: torch.Tensor,
    ) -> _EquityBlock:
        """Branch-independent child equity block for one candidate chunk."""
        children_tensor = _stack_children(children)
        p_child, bar_z_child, child_b_grid = _forward_equity_grid_children(
            self.target_model,
            children_tensor,
            bp_grid,
            b_parent,
            eta_current,
        )
        self._record_equity_forward(int(p_child.numel()))
        return _EquityBlock(
            p_child=p_child, bar_z_child=bar_z_child, child_b_grid=child_b_grid
        )

    def _q_issue_grid(
        self,
        parent_state: torch.Tensor,
        bp_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, n_grid = bp_grid.shape
        issue_state = _expand_candidates(parent_state, bp_grid)
        issue_state[:, 0:1] = _candidate_flat(bp_grid)
        if getattr(self.q_target_model, "q_parameterization", None) == "hybrid_regime":
            # P0/PI are conditional-survival value functions. Candidate Phat is
            # diagnostic only: issuance is priced as a live debt claim and is
            # never replaced by realized-default recovery at this point.
            equity_fn = getattr(self.target_model, "forward_equity", None)
            if callable(equity_fn):
                phat = equity_fn(issue_state)["Phat"]
            else:
                phat = _get_out(self.target_model(issue_state), "Phat", 8)
            q_unit = self.q_target_model._q_unit_output(issue_state)
            q_claim = self.q_target_model._q_claim_output(issue_state)
            recovery = self.q_target_model._q_recovery_output(issue_state)
            diagnostics = {
                "q_issue_claim_grid": q_claim.reshape(batch_size, n_grid),
                "q_issue_unit_grid": q_unit.reshape(batch_size, n_grid),
                "q_issue_realized_default_mask_grid": (
                    (phat <= 0.0).to(q_unit.dtype).reshape(batch_size, n_grid)
                ),
                "q_issue_recovery_grid": recovery.reshape(batch_size, n_grid),
                "q_issue_candidate_phat_grid": phat.reshape(batch_size, n_grid),
                "candidate_phat_gate_used_for_q_issue": torch.zeros(
                    (batch_size, n_grid), device=q_claim.device, dtype=q_claim.dtype
                ),
            }
            q_issue = diagnostics["q_issue_claim_grid"]
        else:
            q_issue = _target_q_claim(
                self.q_target_model,
                issue_state,
                equity_model=self.target_model,
            ).reshape(batch_size, n_grid)
            diagnostics = {}
        self._record_q_forward()
        return q_issue, diagnostics

    def _branch_objective_grid(
        self,
        parent_state: torch.Tensor,
        bp_grid: torch.Tensor,
        *,
        branch: str,
        p_child: torch.Tensor,
        bar_z_child: torch.Tensor,
        child_b_grid: torch.Tensor,
        child_eta_next: torch.Tensor,
        m_tensor: torch.Tensor,
        q_current: torch.Tensor,
        q_issue: torch.Tensor,
        q_diagnostics: Optional[Dict[str, torch.Tensor]] = None,
        mix_weight: Optional[torch.Tensor] = None,
        child_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Assemble branch-specific objectives from a precomputed equity block."""
        batch_size, n_grid = bp_grid.shape
        n_children = int(p_child.shape[2])
        mix_w = None
        if branch == "mix":
            if mix_weight is None:
                raise ValueError("mix_weight is required for branch='mix'")
            mix_w = mix_weight.clamp(0.0, 1.0).reshape(batch_size, 1, 1).expand(
                batch_size, n_grid, n_children
            )

        b_parent = parent_state[:, 0:1]
        eta_current = parent_state[:, 2:3].clamp(0.0, 1.0)
        x_parent = parent_state[:, 4:5]
        z_parent = parent_state[:, 1:2]
        i_parent = parent_state[:, 3:4]
        q_current_grid = q_current.expand(batch_size, n_grid)

        weights = normalize_child_weights(
            child_weights,
            n_parent=batch_size,
            n_child=n_children,
            device=parent_state.device,
            dtype=parent_state.dtype,
        )
        weights_grid = weights.unsqueeze(1)
        m_grid = m_tensor.reshape(batch_size, 1, n_children).expand(
            batch_size, n_grid, n_children
        )
        flat_shape = (batch_size * n_grid * n_children, 1)
        x_grid = x_parent.unsqueeze(1).expand(batch_size, n_grid, n_children).reshape(flat_shape)
        z_grid = z_parent.unsqueeze(1).expand(batch_size, n_grid, n_children).reshape(flat_shape)
        b_grid = b_parent.unsqueeze(1).expand(batch_size, n_grid, n_children).reshape(flat_shape)
        i_grid = i_parent.unsqueeze(1).expand(batch_size, n_grid, n_children).reshape(flat_shape)
        q_current_flat = (
            q_current_grid.unsqueeze(-1).expand(batch_size, n_grid, n_children).reshape(flat_shape)
        )
        q_issue_flat = (
            q_issue.unsqueeze(-1).expand(batch_size, n_grid, n_children).reshape(flat_shape)
        )
        eta_current_flat = (
            eta_current
            .reshape(batch_size, 1, 1)
            .expand(batch_size, n_grid, n_children)
            .reshape(flat_shape)
        )

        cf0 = self.p0_loss_fn.compute_cashflow_p0(
            x_grid,
            z_grid,
            b_grid,
            q_current_flat,
            q_issue_flat,
            eta_current_flat,
        ).reshape(batch_size, n_grid, n_children)
        continuation0 = m_grid * p_child
        value0 = cf0 + continuation0

        cfi = self.pi_loss_fn.compute_cashflow_pi(
            x_grid,
            z_grid,
            b_grid,
            i_grid,
            q_current_flat,
            q_issue_flat,
            eta_current_flat,
        ).reshape(batch_size, n_grid, n_children)
        continuationi = Config.G * m_grid * p_child
        valuei = cfi + continuationi

        if branch == "p0":
            branch_cashflow = cf0
            branch_continuation = continuation0
            branch_value = value0
        elif branch == "pi":
            branch_cashflow = cfi
            branch_continuation = continuationi
            branch_value = valuei
        else:
            branch_cashflow = (1.0 - mix_w) * cf0 + mix_w * cfi
            branch_continuation = (1.0 - mix_w) * continuation0 + mix_w * continuationi
            branch_value = (1.0 - mix_w) * value0 + mix_w * valuei

        value_grid = (weights_grid * branch_value).sum(dim=2)
        cashflow_grid_mean = (weights_grid * branch_cashflow).sum(dim=2)
        continuation_grid_mean = (weights_grid * branch_continuation).sum(dim=2)
        p_grid_mean = (weights_grid * p_child).sum(dim=2)
        default_grid_mean = (weights_grid * bar_z_child).sum(dim=2)
        eta_next = child_eta_next.clamp(0.0, 1.0)
        eta_next_active_share = (weights * eta_next).sum(dim=1, keepdim=True).expand(batch_size, n_grid)
        child_b_mean = (weights_grid * child_b_grid).sum(dim=2)

        def _conditional_child_b(mask: torch.Tensor) -> torch.Tensor:
            conditional_weights = weights * mask
            weight_grid = conditional_weights.unsqueeze(1).expand_as(child_b_grid)
            mass = weight_grid.sum(dim=2)
            mean = (child_b_grid * weight_grid).sum(dim=2) / mass.clamp_min(1e-12)
            return torch.where(mass > 0, mean, torch.full_like(mean, float("nan")))

        child_b_eta0_mean = _conditional_child_b((eta_next <= 0.5).to(child_b_grid.dtype))
        child_b_eta1_mean = _conditional_child_b((eta_next > 0.5).to(child_b_grid.dtype))
        argmax_index = value_grid.argmax(dim=1, keepdim=True)

        result = {
            "bp_grid": bp_grid,
            "value_grid": value_grid,
            "cashflow_grid_mean": cashflow_grid_mean,
            "continuation_grid_mean": continuation_grid_mean,
            "argmax_index": argmax_index,
            "q_issue_grid": q_issue,
            "p_child_grid_mean": p_grid_mean,
            "default_grid_mean": default_grid_mean,
            "eta_next_active_share": eta_next_active_share,
            "child_b_mean": child_b_mean,
            "child_b_eta0_mean": child_b_eta0_mean,
            "child_b_eta1_mean": child_b_eta1_mean,
        }
        if q_diagnostics:
            result["q_current_claim"] = q_current.detach()
            result.update(q_diagnostics)
        return result

    def _evaluate_grid_chunk(
        self,
        parent_state: torch.Tensor,
        children: Sequence[torch.Tensor] | torch.Tensor,
        m_list: Sequence[torch.Tensor] | torch.Tensor,
        bp_grid: torch.Tensor,
        *,
        branch: str,
        mix_weight: Optional[torch.Tensor] = None,
        child_weights: Optional[torch.Tensor] = None,
        q_current: Optional[torch.Tensor] = None,
        equity: Optional[_EquityBlock] = None,
    ) -> Dict[str, torch.Tensor]:
        children_tensor = _stack_children(children)
        m_tensor = _stack_m(m_list)
        if q_current is None:
            q_current = _target_q_claim(
                self.q_target_model,
                parent_state,
                equity_model=self.target_model,
            )
            self._record_q_forward()
        if equity is None:
            equity = self._child_equity_block(
                children_tensor, bp_grid, parent_state[:, 0:1], parent_state[:, 2:3]
            )
        self._record_candidate_chunk(int(equity.p_child.numel()))
        q_issue, q_diagnostics = self._q_issue_grid(parent_state, bp_grid)
        return self._branch_objective_grid(
            parent_state,
            bp_grid,
            branch=branch,
            p_child=equity.p_child,
            bar_z_child=equity.bar_z_child,
            child_b_grid=equity.child_b_grid,
            child_eta_next=children_tensor[..., 2],
            m_tensor=m_tensor,
            q_current=q_current,
            q_issue=q_issue,
            q_diagnostics=q_diagnostics,
            mix_weight=mix_weight,
            child_weights=child_weights,
        )

    def _attach_star_diagnostics(self, result: Dict[str, torch.Tensor], argmax_index: torch.Tensor) -> None:
        result["q_issue_at_star"] = _gather_by_index(result["q_issue_grid"], argmax_index)
        result["p_child_at_star"] = _gather_by_index(result["p_child_grid_mean"], argmax_index)
        result["default_at_star"] = _gather_by_index(result["default_grid_mean"], argmax_index)
