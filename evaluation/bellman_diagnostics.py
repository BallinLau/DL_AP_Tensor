from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch

from analysis.economic_config import AnalysisEconomicConfig
from losses import P0Loss, PILoss
from utils.firm_transition import apply_refinancing_policy

from .bp_diagnostics import FrozenTransitionData
from .grids import FrozenFirmGrid


def _get(output: Any, name: str) -> torch.Tensor:
    if isinstance(output, dict):
        return output[name]
    return getattr(output, name)


def _forward_fields(
    model: torch.nn.Module,
    states: torch.Tensor,
    fields: Iterable[str],
    *,
    chunk_size: int,
) -> Dict[str, torch.Tensor]:
    result = {name: [] for name in fields}
    for start in range(0, states.shape[0], max(1, int(chunk_size))):
        output = model(states[start:start + max(1, int(chunk_size))])
        for name in fields:
            result[name].append(_get(output, name).detach())
    return {name: torch.cat(parts, dim=0) for name, parts in result.items()}


def _child_states(
    parent_states: torch.Tensor,
    children: List[torch.Tensor],
    bp: torch.Tensor,
) -> torch.Tensor:
    states = torch.stack([child[:, :7] for child in children], dim=1).clone()
    states[..., 0:1] = apply_refinancing_policy(
        b_current=parent_states[:, 0:1].unsqueeze(1),
        bp_candidate=bp.unsqueeze(1),
        eta_next=states[..., 2:3],
    )
    return states


def residual_statistics(values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {key: float("nan") for key in (
            "signed_mean", "abs_mean", "abs_median", "abs_p90", "abs_p99", "abs_max"
        )}
    absolute = np.abs(finite)
    return {
        "signed_mean": float(finite.mean()),
        "abs_mean": float(absolute.mean()),
        "abs_median": float(np.median(absolute)),
        "abs_p90": float(np.quantile(absolute, 0.90)),
        "abs_p99": float(np.quantile(absolute, 0.99)),
        "abs_max": float(absolute.max()),
    }


def evaluate_bellman_residuals(
    model: torch.nn.Module,
    grid: FrozenFirmGrid,
    transition: FrozenTransitionData,
    economic_config: AnalysisEconomicConfig,
    *,
    chunk_size: int = 8192,
) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Evaluate physical conditional-mean P0/PI Bellman residuals."""
    parent = grid.base_states
    p0_loss = P0Loss(
        delta=economic_config.DELTA, tau=economic_config.TAU,
        kappa_b=economic_config.KAPPA_B, kappa_e=economic_config.KAPPA_E,
        aio_weight=economic_config.AIO_WEIGHT, alpha_z=economic_config.ALPHA_Z,
        beta_z=economic_config.BETA_Z, z0=economic_config.Z0,
    )
    pi_loss = PILoss(
        delta=economic_config.DELTA, tau=economic_config.TAU, g=economic_config.G,
        kappa_b=economic_config.KAPPA_B, kappa_e=economic_config.KAPPA_E,
        aio_weight=economic_config.AIO_WEIGHT, alpha_z=economic_config.ALPHA_Z,
        beta_z=economic_config.BETA_Z, z0=economic_config.Z0,
        b_penalty_weight=0.0,
    )
    with torch.no_grad():
        parent_out = model(parent)
        bp0 = _get(parent_out, "bp0")
        bpi = _get(parent_out, "bpI")
        q = _get(parent_out, "Q")
        p0 = _get(parent_out, "P0")
        pi = _get(parent_out, "PI")

        issue_p0 = parent.clone()
        issue_p0[:, 0:1] = bp0
        issue_pi = parent.clone()
        issue_pi[:, 0:1] = bpi
        q_issue_p0 = _forward_fields(model, issue_p0, ("Q",), chunk_size=chunk_size)["Q"]
        q_issue_pi = _forward_fields(model, issue_pi, ("Q",), chunk_size=chunk_size)["Q"]
        eta_current = parent[:, 2:3].clamp(0.0, 1.0)
        cf0 = p0_loss.compute_cashflow_p0(
            parent[:, 4:5], parent[:, 1:2], parent[:, 0:1], q, q_issue_p0, eta_current
        )
        cfi = pi_loss.compute_cashflow_pi(
            parent[:, 4:5], parent[:, 1:2], parent[:, 0:1], parent[:, 3:4],
            q, q_issue_pi, eta_current,
        )

        child_p0 = _child_states(parent, transition.children, bp0)
        child_pi = _child_states(parent, transition.children, bpi)
        n_parent, n_child, state_dim = child_p0.shape
        p_child_p0 = _forward_fields(
            model, child_p0.reshape(-1, state_dim), ("P",), chunk_size=chunk_size
        )["P"].reshape(n_parent, n_child, 1)
        p_child_pi = _forward_fields(
            model, child_pi.reshape(-1, state_dim), ("P",), chunk_size=chunk_size
        )["P"].reshape(n_parent, n_child, 1)
        m_used = torch.stack(transition.m_used_list, dim=1)
        weights = transition.branch_weights.unsqueeze(-1)
        continuation0 = (weights * m_used * p_child_p0).sum(dim=1)
        continuationi = float(economic_config.G) * (weights * m_used * p_child_pi).sum(dim=1)
        r0 = p0 - cf0 - continuation0
        ri = pi - cfi - continuationi
        scale_fn = getattr(model, "equity_value_scale", None)
        scale = scale_fn(parent) if callable(scale_fn) else torch.ones_like(r0)

    def surface(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().reshape(grid.shape).numpy().astype(np.float64)

    surfaces = {
        "R0_signed": surface(r0),
        "abs_R0": surface(r0.abs()),
        "RI_signed": surface(ri),
        "abs_RI": surface(ri.abs()),
        "R0_scale_normalized": surface(r0 / scale.clamp_min(1e-12)),
        "RI_scale_normalized": surface(ri / scale.clamp_min(1e-12)),
        "CF0": surface(cf0),
        "CFI": surface(cfi),
        "continuation_P0": surface(continuation0),
        "continuation_PI": surface(continuationi),
    }
    summary: Dict[str, float] = {}
    for prefix, values in (("p0_residual", surfaces["R0_signed"]), ("pi_residual", surfaces["RI_signed"])):
        summary.update({f"{prefix}_{key}": value for key, value in residual_statistics(values).items()})
    return surfaces, summary


def representative_positions(grid: FrozenFirmGrid) -> Dict[str, int]:
    b_positions = {"b_low": 0, "b_mid": len(grid.b_values) // 2, "b_high": len(grid.b_values) - 1}
    z_positions = {"z_low": 0, "z_mid": len(grid.z_values) // 2, "z_high": len(grid.z_values) - 1}
    return {
        f"{b_name}_{z_name}": b_pos * len(grid.z_values) + z_pos
        for b_name, b_pos in b_positions.items()
        for z_name, z_pos in z_positions.items()
    }


def build_child_continuation_audit(
    model: torch.nn.Module,
    grid: FrozenFirmGrid,
    transition: FrozenTransitionData,
    economic_config: AnalysisEconomicConfig,
    *,
    candidate_bp: Sequence[float] = (0.2, 0.5, 0.8),
    chunk_size: int = 8192,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    positions = representative_positions(grid)
    for state_label, pos in positions.items():
        parent = grid.base_states[pos:pos + 1]
        for candidate in candidate_bp:
            child_rows = []
            for child_index, child in enumerate(transition.children):
                state = child[pos:pos + 1, :7].clone()
                eta_next = state[:, 2:3].clamp(0.0, 1.0)
                state[:, 0:1] = apply_refinancing_policy(
                    b_current=parent[:, 0:1],
                    bp_candidate=torch.full_like(parent[:, 0:1], float(candidate)),
                    eta_next=eta_next,
                )
                child_rows.append(state)
            states = torch.cat(child_rows, dim=0)
            with torch.no_grad():
                output = _forward_fields(
                    model, states, ("P", "Phat", "bar_z"), chunk_size=chunk_size
                )
            for child_index, state in enumerate(child_rows):
                eta_next = float(state[0, 2].item())
                child_b = float(state[0, 0].item())
                expected_b = float(candidate) if eta_next > 0.5 else float(parent[0, 0].item())
                m_raw = float(transition.m_raw_list[child_index][pos].item())
                m_used = float(transition.m_used_list[child_index][pos].item())
                p_child = float(output["P"][child_index].item())
                base = {
                    "state_label": state_label,
                    "parent_eta": float(parent[0, 2].item()),
                    "parent_b": float(parent[0, 0].item()),
                    "parent_z": float(parent[0, 1].item()),
                    "bp_candidate": float(candidate),
                    "child_index": int(child_index),
                    "branch_weight": float(transition.branch_weights[pos, child_index].item()),
                    "eta_next": eta_next,
                    "child_b": child_b,
                    "child_b_expected": expected_b,
                    "child_b_identity_error": abs(child_b - expected_b),
                    "child_z": float(state[0, 1].item()),
                    "child_i": float(state[0, 3].item()),
                    "child_x": float(state[0, 4].item()),
                    "M_raw": m_raw,
                    "M_used": m_used,
                    "P_child": p_child,
                    "Phat_child": float(output["Phat"][child_index].item()),
                    "bar_z_child": float(output["bar_z"][child_index].item()),
                    "M_times_P_child": m_used * p_child,
                }
                for branch, growth in (("p0", 1.0), ("pi", float(economic_config.G))):
                    raw_term = growth * m_used * p_child
                    weighted_contribution = base["branch_weight"] * raw_term
                    rows.append({
                        **base,
                        "branch": branch,
                        "raw_continuation_term": raw_term,
                        "weighted_continuation_contribution": weighted_contribution,
                        "continuation_contribution": weighted_contribution,
                    })
    return pd.DataFrame(rows)
