from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch

from analysis.economic_config import AnalysisEconomicConfig
from losses import P0Loss, PILoss
from losses.q_loss import (
    compute_parent_default_regime_weights,
    compute_q_survival_recovery_components,
    compute_recovery_target,
    resolve_q_parent_default_regime_mode,
    resolve_recovery_normalization_mode,
)
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
    children: torch.Tensor,
    bp: torch.Tensor,
) -> torch.Tensor:
    states = children[..., :7].clone()
    # Realized child leverage is gated by the parent eta_t, not eta_{t+1}.
    states[..., 0:1] = apply_refinancing_policy(
        b_current=parent_states[:, 0:1].unsqueeze(1),
        bp_candidate=bp.unsqueeze(1),
        eta_current=parent_states[:, 2:3].unsqueeze(1),
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
    recovery_normalization_mode: str | None = None,
    parent_default_regime_mode: str | None = None,
    parent_default_eps: float | None = None,
    parent_default_tau: float | None = None,
    boundary_low_threshold: float = 0.1,
    boundary_high_threshold: float = 0.9,
) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Evaluate physical conditional-mean P0/PI/Q Bellman residuals.

    ``recovery_normalization_mode`` / ``parent_default_regime_mode`` 决定 Q 违约回收
    口径与 parent default regime gating；默认沿用 ``Config``（见 config/constants.py）。

    ``boundary_low_threshold`` / ``boundary_high_threshold`` 只用于诊断 boundary penalty
    与 default regime 的重叠（``low_b_*_share`` / ``high_b_*_share``）。
    """
    recovery_mode = resolve_recovery_normalization_mode(recovery_normalization_mode)
    regime_mode = resolve_q_parent_default_regime_mode(parent_default_regime_mode)
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
        try:
            bar_i = _get(parent_out, "bar_i")
        except (AttributeError, KeyError):
            bar_i = torch.zeros_like(q)
        try:
            phat = _get(parent_out, "Phat")
        except (AttributeError, KeyError):
            phat = torch.zeros_like(q)

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

        children_tensor = transition.stacked_children()
        child_p0 = _child_states(parent, children_tensor, bp0)
        child_pi = _child_states(parent, children_tensor, bpi)
        n_parent, n_child, state_dim = child_p0.shape
        p_child_p0 = _forward_fields(
            model, child_p0.reshape(-1, state_dim), ("P",), chunk_size=chunk_size
        )["P"].reshape(n_parent, n_child, 1)
        p_child_pi = _forward_fields(
            model, child_pi.reshape(-1, state_dim), ("P",), chunk_size=chunk_size
        )["P"].reshape(n_parent, n_child, 1)
        weights = transition.branch_weights.unsqueeze(-1)
        # Match Episode._compute_q_bellman_signed_residuals exactly: Qsp is
        # evaluated at issue debt b / (bar_i * (G - 1) + 1), while child x/z,
        # default and the configured train-M semantics remain branch specific.
        multiplier = bar_i * (float(economic_config.G) - 1.0) + 1.0
        b_sp = parent[:, 0:1] / multiplier.clamp_min(1e-6)
        q_child_states = children_tensor[..., :7].clone()
        q_child_states[..., 0:1] = b_sp.unsqueeze(1)
        qsp = _forward_fields(
            model,
            q_child_states.reshape(-1, q_child_states.shape[-1]),
            ("Q",),
            chunk_size=chunk_size,
        )["Q"].reshape(n_parent, n_child, 1)
        try:
            bar_zsp = _forward_fields(
                model,
                q_child_states.reshape(-1, q_child_states.shape[-1]),
                ("bar_z",),
                chunk_size=chunk_size,
            )["bar_z"].reshape(n_parent, n_child, 1)
        except (AttributeError, KeyError):
            bar_zsp = torch.zeros_like(qsp)
        x_child = q_child_states[..., 4:5]
        z_child = q_child_states[..., 1:2]
        # Parent default regime gating 与当期回收（均与 M 无关，只依赖 parent 状态）。
        regime = compute_parent_default_regime_weights(
            phat, mode=regime_mode, eps=parent_default_eps, tau=parent_default_tau,
        )
        survival_w = regime["parent_survival_weight"]
        default_w = regime["parent_default_weight"]
        hard_default = regime["parent_hard_default"]
        recovery_current = compute_recovery_target(
            parent[:, 0:1], parent[:, 4:5], parent[:, 1:2],
            phi=float(economic_config.PHI), delta=float(economic_config.DELTA),
            recovery_normalization_mode=recovery_mode,
        )
        # q_unit is a derived reporting ratio Q/b, not the direct-Q network output.
        # The corresponding default-region ratio target is recovery / b.
        b_parent = parent[:, 0:1]
        q_unit_recovery_target = torch.where(
            b_parent > 0.0,
            recovery_current / b_parent.clamp_min(1e-12),
            torch.full_like(recovery_current, float("nan")),
        )
        # Derived predicted ratio (same reporting convention as firm_surfaces).
        q_unit_pred = torch.where(
            b_parent > 0.0,
            q / b_parent.clamp_min(1e-12),
            torch.full_like(q, float("nan")),
        )
        q_unit_minus_recovery_target = q_unit_pred - q_unit_recovery_target

        def residuals_for_m(m_values: torch.Tensor) -> Dict[str, torch.Tensor]:
            continuation0 = (weights * m_values * p_child_p0).sum(dim=1)
            continuationi = float(economic_config.G) * (
                weights * m_values * p_child_pi
            ).sum(dim=1)
            q_components = compute_q_survival_recovery_components(
                Q=q.unsqueeze(1),
                b=parent[:, 0:1].unsqueeze(1),
                bar_i=bar_i.unsqueeze(1),
                M=m_values,
                Qsp=qsp,
                bar_z=bar_zsp,
                x_child=x_child,
                z_child=z_child,
                g=float(economic_config.G),
                delta=float(economic_config.DELTA),
                phi=float(economic_config.PHI),
                recovery_normalization_mode=recovery_mode,
            )
            q_target_survival = (weights * q_components["q_target_survival"]).sum(dim=1)
            q_target_recovery = (weights * q_components["q_target_recovery"]).sum(dim=1)
            q_target = q_target_survival + q_target_recovery
            return {
                "r0": p0 - cf0 - continuation0,
                "ri": pi - cfi - continuationi,
                "rq": (weights * q_components["q_training_residual"]).sum(dim=1),
                "q_target": q_target,
                "q_target_survival": q_target_survival,
                "q_target_recovery": q_target_recovery,
                "q_target_used_for_training": (
                    survival_w * q_target + default_w * recovery_current
                ),
                "continuation0": continuation0,
                "continuationi": continuationi,
            }

        train_m = residuals_for_m(transition.stacked_m_used())
        raw_m = residuals_for_m(transition.stacked_m_raw())
        scale_fn = getattr(model, "equity_value_scale", None)
        scale = scale_fn(parent) if callable(scale_fn) else torch.ones_like(train_m["r0"])

    def surface(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().reshape(grid.shape).numpy().astype(np.float64)

    surfaces: Dict[str, np.ndarray] = {
        "CF0": surface(cf0),
        "CFI": surface(cfi),
        "continuation_P0": surface(train_m["continuation0"]),
        "continuation_PI": surface(train_m["continuationi"]),
    }
    for label, values in (("trainM", train_m), ("rawM", raw_m)):
        q_scale = torch.maximum(q.abs(), values["q_target"].abs()).clamp_min(1e-8)
        surfaces.update({
            f"R0_{label}_signed": surface(values["r0"]),
            f"abs_R0_{label}": surface(values["r0"].abs()),
            f"RI_{label}_signed": surface(values["ri"]),
            f"abs_RI_{label}": surface(values["ri"].abs()),
            f"R0_{label}_scale_normalized": surface(values["r0"] / scale.clamp_min(1e-12)),
            f"RI_{label}_scale_normalized": surface(values["ri"] / scale.clamp_min(1e-12)),
            f"RQ_{label}_signed": surface(values["rq"]),
            f"abs_RQ_{label}": surface(values["rq"].abs()),
            f"RQ_{label}_scale_normalized": surface(values["rq"] / q_scale),
            f"abs_RQ_{label}_scale_normalized": surface((values["rq"] / q_scale).abs()),
            f"Q_target_{label}": surface(values["q_target"]),
            f"Q_target_survival_{label}": surface(values["q_target_survival"]),
            f"Q_target_recovery_{label}": surface(values["q_target_recovery"]),
            f"Q_target_used_for_training_{label}": surface(
                values["q_target_used_for_training"]
            ),
            # 显式口径命名（不改动旧 Q_target 的经济含义）：
            #   Q_target_bellman        = continuation Bellman target
            #   Q_target_*_bellman      = 其 survival / recovery 分解
            #   Q_target_training       = w_survival * bellman + w_default * recovery
            #   RQ_bellman_signed       = Q_target_bellman - Q
            #   RQ_training_signed      = Q_target_training - Q
            f"Q_target_bellman_{label}": surface(values["q_target"]),
            f"Q_target_survival_bellman_{label}": surface(values["q_target_survival"]),
            f"Q_target_recovery_bellman_{label}": surface(values["q_target_recovery"]),
            f"Q_target_training_{label}": surface(values["q_target_used_for_training"]),
            f"RQ_bellman_signed_{label}": surface(values["rq"]),
            f"RQ_training_signed_{label}": surface(
                values["q_target_used_for_training"] - q
            ),
        })
    # Backward-compatible canonical files and fields retain training-M semantics.
    surfaces.update({
        "R0_signed": surfaces["R0_trainM_signed"],
        "abs_R0": surfaces["abs_R0_trainM"],
        "RI_signed": surfaces["RI_trainM_signed"],
        "abs_RI": surfaces["abs_RI_trainM"],
        "R0_scale_normalized": surfaces["R0_trainM_scale_normalized"],
        "RI_scale_normalized": surfaces["RI_trainM_scale_normalized"],
        "RQ_signed": surfaces["RQ_trainM_signed"],
        "abs_RQ": surfaces["abs_RQ_trainM"],
        "RQ_scale_normalized": surfaces["RQ_trainM_scale_normalized"],
        "abs_RQ_scale_normalized": surfaces["abs_RQ_trainM_scale_normalized"],
        "Q_target": surfaces["Q_target_trainM"],
        "Q_target_survival": surfaces["Q_target_survival_trainM"],
        "Q_target_recovery": surfaces["Q_target_recovery_trainM"],
        "Q_target_used_for_training": surfaces["Q_target_used_for_training_trainM"],
        "Q_target_bellman": surfaces["Q_target_bellman_trainM"],
        "Q_target_survival_bellman": surfaces["Q_target_survival_bellman_trainM"],
        "Q_target_recovery_bellman": surfaces["Q_target_recovery_bellman_trainM"],
        "Q_target_training": surfaces["Q_target_training_trainM"],
        "RQ_bellman_signed": surfaces["RQ_bellman_signed_trainM"],
        "RQ_training_signed": surfaces["RQ_training_signed_trainM"],
    })
    # Parent 违约 regime 与当期回收（只依赖 parent 状态，与 M 无关）。
    surfaces.update({
        "recovery_current": surface(recovery_current),
        "q_unit_pred": surface(q_unit_pred),
        "q_unit_recovery_target": surface(q_unit_recovery_target),
        "q_unit_minus_recovery_target": surface(q_unit_minus_recovery_target),
        "parent_hard_default": surface(hard_default),
        "parent_survival_weight": surface(survival_w),
        "parent_default_weight": surface(default_w),
        "Q_minus_recovery": surface(q - recovery_current),
        "Q_target_minus_recovery": surface(train_m["q_target"] - recovery_current),
        "Q_minus_Q_target": surface(q - train_m["q_target"]),
    })
    summary: Dict[str, float] = {}
    for prefix, values in (
        ("p0_trainM_residual", surfaces["R0_trainM_signed"]),
        ("pi_trainM_residual", surfaces["RI_trainM_signed"]),
        ("q_trainM_residual", surfaces["RQ_trainM_signed"]),
        ("p0_rawM_residual", surfaces["R0_rawM_signed"]),
        ("pi_rawM_residual", surfaces["RI_rawM_signed"]),
        ("q_rawM_residual", surfaces["RQ_rawM_signed"]),
        ("p0_trainM_residual_normalized", surfaces["R0_trainM_scale_normalized"]),
        ("pi_trainM_residual_normalized", surfaces["RI_trainM_scale_normalized"]),
        ("q_trainM_residual_normalized", surfaces["RQ_trainM_scale_normalized"]),
        ("p0_rawM_residual_normalized", surfaces["R0_rawM_scale_normalized"]),
        ("pi_rawM_residual_normalized", surfaces["RI_rawM_scale_normalized"]),
        ("q_rawM_residual_normalized", surfaces["RQ_rawM_scale_normalized"]),
    ):
        summary.update({f"{prefix}_{key}": value for key, value in residual_statistics(values).items()})
    for equation in ("p0", "pi", "q"):
        for key, value in residual_statistics(
            surfaces[{"p0": "R0_trainM_signed", "pi": "RI_trainM_signed", "q": "RQ_trainM_signed"}[equation]]
        ).items():
            summary[f"{equation}_residual_{key}"] = value
    for key, value in residual_statistics(surfaces["RQ_trainM_scale_normalized"]).items():
        summary[f"q_residual_normalized_{key}"] = value
    # Default-region derived-ratio statistics. The direct network predicts total
    # Q; q_unit=Q/b is retained only as a reporting ratio for b>0.
    # deep-default 样本稀缺，单独统计（仅 b > 0 且 hard-default 的点）。
    hard_default_flat = surfaces["parent_hard_default"].reshape(-1) > 0.5

    def _region_values(surface_name: str) -> np.ndarray:
        values = surfaces[surface_name].reshape(-1)[hard_default_flat]
        return values[np.isfinite(values)]

    q_unit_pred_default = _region_values("q_unit_pred")
    q_unit_target_default = _region_values("q_unit_recovery_target")
    q_unit_error_default = np.abs(_region_values("q_unit_minus_recovery_target"))

    def _write_stats(prefix: str, values: np.ndarray) -> None:
        summary[f"{prefix}_count"] = int(values.size)
        for key, value in (
            ("mean", float(values.mean()) if values.size else float("nan")),
            ("p90", float(np.quantile(values, 0.90)) if values.size else float("nan")),
            ("p99", float(np.quantile(values, 0.99)) if values.size else float("nan")),
            ("max", float(values.max()) if values.size else float("nan")),
        ):
            summary[f"{prefix}_{key}"] = value

    _write_stats("q_unit_pred_default", q_unit_pred_default)
    _write_stats("q_unit_recovery_target_default", q_unit_target_default)
    _write_stats("q_unit_error_default_abs", q_unit_error_default)
    # Legacy alias（旧名含糊，实为 recovery target 统计）：保留以兼容旧脚本。
    for key in ("count", "mean", "p90", "p99", "max"):
        summary[f"q_unit_default_region_{key}"] = summary[
            f"q_unit_recovery_target_default_{key}"
        ]

    # boundary penalty 与 default regime 的重叠诊断（判断两者是否互相冲突）。
    b_all = grid.b_values.repeat(len(grid.z_values))
    low_b_mask = b_all <= float(boundary_low_threshold)
    high_b_mask = b_all >= float(boundary_high_threshold)
    n_total = max(1, b_all.size)
    for name, mask in (
        ("low_b_default_share", low_b_mask & hard_default_flat),
        ("low_b_survival_share", low_b_mask & ~hard_default_flat),
        ("high_b_default_share", high_b_mask & hard_default_flat),
        ("high_b_survival_share", high_b_mask & ~hard_default_flat),
    ):
        summary[name] = float(mask.sum()) / float(n_total)
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
                # Realized child leverage is gated by the parent eta_t.
                state[:, 0:1] = apply_refinancing_policy(
                    b_current=parent[:, 0:1],
                    bp_candidate=torch.full_like(parent[:, 0:1], float(candidate)),
                    eta_current=parent[:, 2:3].clamp(0.0, 1.0),
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
                # Refinancing availability is the parent eta_t, not the child eta.
                expected_b = (
                    float(candidate)
                    if float(parent[0, 2].item()) > 0.5
                    else float(parent[0, 0].item())
                )
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
                        "continuation_raw": raw_term,
                        "continuation_weighted": weighted_contribution,
                        "raw_continuation_term": raw_term,
                        "weighted_continuation_contribution": weighted_contribution,
                        "continuation_contribution": weighted_contribution,
                    })
    return pd.DataFrame(rows)
