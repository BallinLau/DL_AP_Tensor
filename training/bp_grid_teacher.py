"""
Target-grid teacher for firm leverage policy distillation.

The teacher evaluates economic Bellman RHS values over candidate bp values with
the frozen firm target network. It returns detached value targets for P0/PI and
detached bp labels for the policy heads. Simulation code never calls this module.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from config import Config


logger = logging.getLogger(__name__)


def _get_out(out: Any, name: str, idx: int) -> torch.Tensor:
    if isinstance(out, dict):
        return out[name]
    if hasattr(out, name):
        return getattr(out, name)
    return out[:, idx:idx + 1]


def _target_q(model: Any, firm_state: torch.Tensor) -> torch.Tensor:
    q_fn = getattr(model, "_q_output", None)
    if callable(q_fn):
        return q_fn(firm_state)
    return _get_out(model(firm_state), "Q", 0)


def _target_equity(model: Any, firm_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    equity_fn = getattr(model, "forward_equity", None)
    if callable(equity_fn):
        out = equity_fn(firm_state)
    else:
        out = model(firm_state)
    return _get_out(out, "P", 7), _get_out(out, "bar_z", 6)


def _strip_extra(x: torch.Tensor) -> torch.Tensor:
    return x[:, :7] if x.shape[1] > 7 else x


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
        grid_min: float = 0.0,
        grid_max: float = 1.0,
        coarse_size: int = 21,
        fine_size: int = 9,
        refine: bool = True,
        quadratic_refine: bool = False,
        parent_chunk_size: int = 2048,
        candidate_chunk_size: int = 4,
        max_expanded_states: int = 65536,
        margin_scale: float = 1e-3,
        confidence_relative: bool = True,
        confidence_min: float = 0.0,
    ):
        self.target_model = target_model
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
        self._grid_chunk_logged = False

    @classmethod
    def from_hyperparams(cls, target_model, p0_loss_fn, pi_loss_fn, hyperparams) -> "BPGridTeacher":
        return cls(
            target_model,
            p0_loss_fn,
            pi_loss_fn,
            grid_min=float(getattr(hyperparams, "bp_grid_min", 0.0)),
            grid_max=float(getattr(hyperparams, "bp_grid_max", 1.0)),
            coarse_size=int(getattr(hyperparams, "bp_grid_coarse_size", 21)),
            fine_size=int(getattr(hyperparams, "bp_grid_fine_size", 9)),
            refine=bool(getattr(hyperparams, "bp_grid_refine_enabled", True)),
            quadratic_refine=bool(getattr(hyperparams, "bp_grid_quadratic_refine", False)),
            parent_chunk_size=int(getattr(hyperparams, "bp_grid_parent_chunk_size", 2048)),
            candidate_chunk_size=int(getattr(hyperparams, "bp_grid_candidate_chunk_size", 0)),
            max_expanded_states=int(getattr(hyperparams, "bp_grid_max_expanded_states", 65536)),
            margin_scale=float(getattr(hyperparams, "bp_grid_margin_scale", 1e-3)),
            confidence_relative=bool(getattr(hyperparams, "bp_grid_confidence_relative", True)),
            confidence_min=float(getattr(hyperparams, "bp_grid_confidence_min", 0.0)),
        )

    def compute(
        self,
        parent_state: torch.Tensor,
        children: List[torch.Tensor],
        m_list: List[torch.Tensor],
        *,
        branch: str,
        bp_pred: Optional[torch.Tensor] = None,
        mix_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        branch = branch.lower()
        if branch not in {"p0", "pi", "mix"}:
            raise ValueError(f"Unknown branch: {branch}")
        if branch == "mix" and mix_weight is None:
            raise ValueError("mix_weight is required for branch='mix'")

        if self.parent_chunk_size > 0 and parent_state.shape[0] > self.parent_chunk_size:
            chunks = []
            for start in range(0, parent_state.shape[0], self.parent_chunk_size):
                stop = min(start + self.parent_chunk_size, parent_state.shape[0])
                child_chunk = [child[start:stop] for child in children]
                m_chunk = [m[start:stop] for m in m_list]
                bp_chunk = bp_pred[start:stop] if bp_pred is not None else None
                mix_chunk = mix_weight[start:stop] if mix_weight is not None else None
                chunks.append(
                    self._compute_no_parent_chunk(
                        parent_state[start:stop],
                        child_chunk,
                        m_chunk,
                        branch=branch,
                        bp_pred=bp_chunk,
                        mix_weight=mix_chunk,
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
        )

    def _compute_no_parent_chunk(
        self,
        parent_state: torch.Tensor,
        children: List[torch.Tensor],
        m_list: List[torch.Tensor],
        *,
        branch: str,
        bp_pred: Optional[torch.Tensor] = None,
        mix_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            coarse_grid = self._uniform_grid(parent_state, self.coarse_size)
            coarse = self._evaluate_grid(parent_state, children, m_list, coarse_grid, branch=branch, mix_weight=mix_weight)
            if self.refine:
                fine_grid = self._local_fine_grid(coarse_grid, coarse["argmax_index"])
                result = self._evaluate_grid(parent_state, children, m_list, fine_grid, branch=branch, mix_weight=mix_weight)
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

            coarse_top2_margin, _ = _safe_top2_margin(coarse["value_grid"])
            fine_top2_margin, _ = _safe_top2_margin(result["value_grid"])
            coarse_value_star = _gather_by_index(coarse["value_grid"], coarse["argmax_index"])
            if self.confidence_relative:
                value_scale = coarse_value_star.abs().clamp_min(1e-8)
                relative_margin = coarse_top2_margin / value_scale
                confidence = (relative_margin / self.margin_scale).clamp(self.confidence_min, 1.0)
            else:
                confidence = (coarse_top2_margin / self.margin_scale).clamp(self.confidence_min, 1.0)

            result.update(
                {
                    "bp_star": bp_star.detach(),
                    "bp_star_grid": bp_star_grid.detach(),
                    "value_star": value_star.detach(),
                    "top2_margin": coarse_top2_margin.detach(),
                    "coarse_top2_margin": coarse_top2_margin.detach(),
                    "fine_top2_margin": fine_top2_margin.detach(),
                    "confidence": confidence.detach(),
                    "boundary_low": (bp_star_grid <= self.grid_min + 1e-8).to(parent_state.dtype).detach(),
                    "boundary_high": (bp_star_grid >= self.grid_max - 1e-8).to(parent_state.dtype).detach(),
                    "coarse_bp_grid": coarse["bp_grid"].detach(),
                    "coarse_value_grid": coarse["value_grid"].detach(),
                    "coarse_q_issue_grid": coarse["q_issue_grid"].detach(),
                    "coarse_p_child_grid_mean": coarse["p_child_grid_mean"].detach(),
                    "coarse_default_grid_mean": coarse["default_grid_mean"].detach(),
                    "local_value_left": result["value_grid"][:, 0:1].detach(),
                    "local_value_right": result["value_grid"][:, -1:].detach(),
                    "q_issue_at_star": q_issue_at_star.detach(),
                    "p_child_at_star": p_child_at_star.detach(),
                    "default_at_star": default_at_star.detach(),
                }
            )

            if bp_pred is not None:
                pred_grid = bp_pred.detach().clamp(self.grid_min, self.grid_max).reshape(-1, 1)
                pred_eval = self._evaluate_grid(parent_state, children, m_list, pred_grid, branch=branch, mix_weight=mix_weight)
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
    ) -> int:
        dynamic_chunk = max(1, self.max_expanded_states // max(batch_size, 1))
        requested_chunk = n_grid if self.candidate_chunk_size <= 0 else self.candidate_chunk_size
        return max(1, min(requested_chunk, dynamic_chunk, n_grid))

    def _evaluate_grid(
        self,
        parent_state: torch.Tensor,
        children: List[torch.Tensor],
        m_list: List[torch.Tensor],
        bp_grid: torch.Tensor,
        *,
        branch: str,
        mix_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, n_grid = bp_grid.shape
        q_current = _target_q(self.target_model, parent_state)
        dynamic_chunk = max(1, self.max_expanded_states // max(batch_size, 1))
        chunk_size = self._resolve_candidate_chunk_size(batch_size=batch_size, n_grid=n_grid)
        if not self._grid_chunk_logged:
            logger.info(
                "BP grid chunk plan | parent_batch=%d n_grid=%d candidate_chunk_cfg=%d "
                "dynamic_chunk=%d resolved_chunk=%d one_shot=%s expanded_states=%d "
                "max_expanded_states=%d",
                batch_size,
                n_grid,
                self.candidate_chunk_size,
                dynamic_chunk,
                chunk_size,
                str(chunk_size == n_grid),
                batch_size * chunk_size,
                self.max_expanded_states,
            )
            self._grid_chunk_logged = True
        if chunk_size >= n_grid:
            return self._evaluate_grid_chunk(parent_state, children, m_list, bp_grid, branch=branch, mix_weight=mix_weight, q_current=q_current)

        chunks = []
        for start in range(0, n_grid, chunk_size):
            stop = min(start + chunk_size, n_grid)
            chunks.append(
                self._evaluate_grid_chunk(
                    parent_state,
                    children,
                    m_list,
                    bp_grid[:, start:stop],
                    branch=branch,
                    mix_weight=mix_weight,
                    q_current=q_current,
                )
            )
        return {
            "bp_grid": torch.cat([c["bp_grid"] for c in chunks], dim=1),
            "value_grid": torch.cat([c["value_grid"] for c in chunks], dim=1),
            "q_issue_grid": torch.cat([c["q_issue_grid"] for c in chunks], dim=1),
            "p_child_grid_mean": torch.cat([c["p_child_grid_mean"] for c in chunks], dim=1),
            "default_grid_mean": torch.cat([c["default_grid_mean"] for c in chunks], dim=1),
            "argmax_index": torch.cat([c["value_grid"] for c in chunks], dim=1).argmax(dim=1, keepdim=True),
        }

    def _evaluate_grid_chunk(
        self,
        parent_state: torch.Tensor,
        children: List[torch.Tensor],
        m_list: List[torch.Tensor],
        bp_grid: torch.Tensor,
        *,
        branch: str,
        mix_weight: Optional[torch.Tensor] = None,
        q_current: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, n_grid = bp_grid.shape
        if q_current is None:
            q_current = _target_q(self.target_model, parent_state)

        issue_state = _expand_candidates(parent_state, bp_grid)
        issue_state[:, 0:1] = _candidate_flat(bp_grid)
        q_issue = _target_q(self.target_model, issue_state).reshape(batch_size, n_grid)

        value_grid = torch.zeros(batch_size, n_grid, device=parent_state.device, dtype=parent_state.dtype)
        p_grid_sum = torch.zeros_like(value_grid)
        default_grid_sum = torch.zeros_like(value_grid)
        mix_w = None
        if branch == "mix":
            if mix_weight is None:
                raise ValueError("mix_weight is required for branch='mix'")
            mix_w = mix_weight.clamp(0.0, 1.0).expand(batch_size, n_grid)

        b_parent = parent_state[:, 0:1]
        x_parent = parent_state[:, 4:5]
        z_parent = parent_state[:, 1:2]
        i_parent = parent_state[:, 3:4]
        q_current_grid = q_current.expand(batch_size, n_grid)

        for child, m in zip(children, m_list):
            child_state_raw = _strip_extra(child)
            eta_child = child[:, 2:3].clamp(0.0, 1.0)
            child_state = _expand_candidates(child_state_raw, bp_grid)
            eta_grid = eta_child.expand(batch_size, n_grid)
            b_grid = b_parent.expand(batch_size, n_grid)
            child_state[:, 0:1] = (eta_grid * bp_grid + (1.0 - eta_grid) * b_grid).reshape(-1, 1)

            p_child_raw, bar_z_child_raw = _target_equity(self.target_model, child_state)
            p_child = p_child_raw.reshape(batch_size, n_grid)
            bar_z_child = bar_z_child_raw.reshape(batch_size, n_grid).clamp(0.0, 1.0)

            eta_flat = eta_grid.reshape(-1, 1)
            cf0 = self.p0_loss_fn.compute_cashflow_p0(
                x_parent.expand(batch_size, n_grid).reshape(-1, 1),
                z_parent.expand(batch_size, n_grid).reshape(-1, 1),
                b_parent.expand(batch_size, n_grid).reshape(-1, 1),
                q_current_grid.reshape(-1, 1),
                q_issue.reshape(-1, 1),
                eta_flat,
            ).reshape(batch_size, n_grid)
            value0 = cf0 + m.expand(batch_size, n_grid) * p_child

            cfi = self.pi_loss_fn.compute_cashflow_pi(
                x_parent.expand(batch_size, n_grid).reshape(-1, 1),
                z_parent.expand(batch_size, n_grid).reshape(-1, 1),
                b_parent.expand(batch_size, n_grid).reshape(-1, 1),
                i_parent.expand(batch_size, n_grid).reshape(-1, 1),
                q_current_grid.reshape(-1, 1),
                q_issue.reshape(-1, 1),
                eta_flat,
            ).reshape(batch_size, n_grid)
            valuei = cfi + Config.G * m.expand(batch_size, n_grid) * p_child

            if branch == "p0":
                branch_value = value0
            elif branch == "pi":
                branch_value = valuei
            else:
                branch_value = (1.0 - mix_w) * value0 + mix_w * valuei

            value_grid = value_grid + branch_value
            p_grid_sum = p_grid_sum + p_child
            default_grid_sum = default_grid_sum + bar_z_child

        n_children = max(1, len(children))
        value_grid = value_grid / n_children
        p_grid_mean = p_grid_sum / n_children
        default_grid_mean = default_grid_sum / n_children
        argmax_index = value_grid.argmax(dim=1, keepdim=True)

        return {
            "bp_grid": bp_grid,
            "value_grid": value_grid,
            "argmax_index": argmax_index,
            "q_issue_grid": q_issue,
            "p_child_grid_mean": p_grid_mean,
            "default_grid_mean": default_grid_mean,
        }

    def _attach_star_diagnostics(self, result: Dict[str, torch.Tensor], argmax_index: torch.Tensor) -> None:
        result["q_issue_at_star"] = _gather_by_index(result["q_issue_grid"], argmax_index)
        result["p_child_at_star"] = _gather_by_index(result["p_child_grid_mean"], argmax_index)
        result["default_at_star"] = _gather_by_index(result["default_grid_mean"], argmax_index)
