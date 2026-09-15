from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from analysis.economic_config import AnalysisEconomicConfig
from analysis.convergence_transition import (
    ConvergenceShockBank,
    MacroTransitionContext,
    build_child_exogenous_bundle,
)
from config import Config, HyperParams
from losses import P0Loss, PILoss
from training.bp_grid_teacher import BPGridTeacher
from utils.firm_transition import expand_children_exact_eta

from .grids import FrozenFirmGrid, ReferenceFirmState
from .plotting import plot_objective_slice


@dataclass(frozen=True)
class FrozenTransitionData:
    children: List[torch.Tensor]
    m_raw_list: List[torch.Tensor]
    m_used_list: List[torch.Tensor]
    branch_weights: torch.Tensor
    metadata: Dict[str, Any]


@contextmanager
def _checkpoint_economic_config(config: AnalysisEconomicConfig):
    saved = {name: getattr(Config, name) for name in config.field_names()}
    try:
        for name, value in config.to_dict().items():
            setattr(Config, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(Config, name, value)


def build_frozen_transition_data(
    sdf_fc1_model: torch.nn.Module,
    parent_states: torch.Tensor,
    reference: ReferenceFirmState,
    hyperparams: HyperParams,
    economic_config: AnalysisEconomicConfig,
    *,
    n_child_shocks: int,
    shock_seed: int,
    shock_bank_max_child_shocks: int | None = None,
) -> FrozenTransitionData:
    if int(n_child_shocks) < 2:
        raise ValueError("n_child_shocks must be at least 2")
    bank_child_shocks = int(shock_bank_max_child_shocks or n_child_shocks)
    if bank_child_shocks < int(n_child_shocks):
        raise ValueError(
            "shock_bank_max_child_shocks must be at least n_child_shocks"
        )
    device = parent_states.device
    dtype = parent_states.dtype
    base_bank = ConvergenceShockBank.create(
        1,
        bank_child_shocks,
        seed=int(shock_seed),
        device=device,
        dtype=dtype,
    )
    base_bank = ConvergenceShockBank(
        eps_x=base_bank.eps_x[:, :n_child_shocks],
        eps_z=base_bank.eps_z[:, :n_child_shocks],
        u_eta=base_bank.u_eta[:, :n_child_shocks],
        u_i=base_bank.u_i[:, :n_child_shocks],
        seed=base_bank.seed,
    )
    reference_index = torch.zeros(parent_states.shape[0], dtype=torch.long, device=device)
    shock_bank = base_bank.gather(reference_index)
    macro = MacroTransitionContext(
        hatc_cal=torch.full(
            (parent_states.shape[0], 1), reference.hatc_cal, device=device, dtype=dtype
        ),
        lnk_cal=torch.full(
            (parent_states.shape[0], 1), reference.lnk_cal, device=device, dtype=dtype
        ),
    )
    bundle = build_child_exogenous_bundle(
        sdf_fc1_model,
        parent_states,
        macro,
        shock_bank,
        economic_config=economic_config,
    )
    use_clipped_m = bool(getattr(hyperparams, "pv_use_clipped_m", True))
    m_lo = float(getattr(hyperparams, "pv_m_clamp_min", 0.7))
    m_hi = float(getattr(hyperparams, "pv_m_clamp_max", 1.3))
    continuous_children: List[torch.Tensor] = []
    continuous_m_raw: List[torch.Tensor] = []
    continuous_m_used: List[torch.Tensor] = []
    for child_pos in range(int(n_child_shocks)):
        continuous_children.append(
            torch.stack(
                [
                    parent_states[:, 0],
                    bundle.z_next[:, child_pos, 0],
                    torch.zeros_like(bundle.eta_next[:, child_pos, 0]),
                    bundle.i_next[:, child_pos, 0],
                    bundle.x_next[:, child_pos, 0],
                    bundle.hatcf_next[:, child_pos, 0],
                    bundle.lnkf_next[:, child_pos, 0],
                ],
                dim=1,
            )
        )
        m_raw = bundle.m_raw[:, child_pos, :]
        m = m_raw
        if use_clipped_m:
            m = m.clamp(m_lo, m_hi)
        continuous_m_raw.append(m_raw)
        continuous_m_used.append(m)
    eta_expansion = expand_children_exact_eta(
        continuous_children,
        zeta=float(economic_config.ZETA),
        child_weights=bundle.branch_weights,
    )
    m_raw_list = [continuous_m_raw[index] for index in eta_expansion.source_child_indices]
    m_used_list = [continuous_m_used[index] for index in eta_expansion.source_child_indices]
    eta_next = torch.stack([child[:, 2] for child in eta_expansion.children], dim=1)
    eta_probability_mass = (eta_expansion.branch_weights * eta_next).sum(dim=1)
    metadata = {
        "builder": "ConvergenceShockBank+build_child_exogenous_bundle",
        "bp_teacher_model": "policy_value",
        "shock_seed": int(shock_seed),
        "n_child_shocks": int(n_child_shocks),
        "eta_integration_mode": "exact",
        "eta_probability": float(economic_config.ZETA),
        "continuous_child_count": int(n_child_shocks),
        "expanded_child_count": 2 * int(n_child_shocks),
        "shock_bank_max_child_shocks": bank_child_shocks,
        "nested_prefix_from_max_J": bank_child_shocks > int(n_child_shocks),
        "common_shocks_across_frozen_grid": True,
        "macro_context": {
            "hatc_cal": float(reference.hatc_cal),
            "lnk_cal": float(reference.lnk_cal),
        },
        "m_source": "sdf_fc1.forward_step",
        "m_mode": "clipped_train_m" if use_clipped_m else "raw_sdf_m",
        "m_raw_mean": float(bundle.m_raw.detach().mean().item()),
        "m_raw_std": float(bundle.m_raw.detach().std(unbiased=False).item()),
        "eta_next_active_share": float(eta_probability_mass.mean().item()),
    }
    return FrozenTransitionData(
        children=eta_expansion.children,
        m_raw_list=m_raw_list,
        m_used_list=m_used_list,
        branch_weights=eta_expansion.branch_weights,
        metadata=metadata,
    )


def build_frozen_transition_children(
    sdf_fc1_model: torch.nn.Module,
    parent_states: torch.Tensor,
    reference: ReferenceFirmState,
    hyperparams: HyperParams,
    economic_config: AnalysisEconomicConfig,
    *,
    n_child_shocks: int,
    shock_seed: int,
    shock_bank_max_child_shocks: int | None = None,
) -> tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
    """Backward-compatible transition tuple used by existing evaluator callers."""
    data = build_frozen_transition_data(
        sdf_fc1_model,
        parent_states,
        reference,
        hyperparams,
        economic_config,
        n_child_shocks=n_child_shocks,
        shock_seed=shock_seed,
        shock_bank_max_child_shocks=shock_bank_max_child_shocks,
    )
    return data.children, data.m_used_list, data.metadata


def _losses(config: AnalysisEconomicConfig) -> tuple[P0Loss, PILoss]:
    p0 = P0Loss(
        delta=config.DELTA,
        tau=config.TAU,
        kappa_b=config.KAPPA_B,
        kappa_e=config.KAPPA_E,
        aio_weight=config.AIO_WEIGHT,
        alpha_z=config.ALPHA_Z,
        beta_z=config.BETA_Z,
        z0=config.Z0,
    )
    pi = PILoss(
        delta=config.DELTA,
        tau=config.TAU,
        g=config.G,
        kappa_b=config.KAPPA_B,
        kappa_e=config.KAPPA_E,
        aio_weight=config.AIO_WEIGHT,
        alpha_z=config.ALPHA_Z,
        beta_z=config.BETA_Z,
        z0=config.Z0,
        b_penalty_weight=0.0,
    )
    return p0, pi


def _gap_statistics(pred: np.ndarray, star: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    gap = np.abs(pred - star)
    finite = np.isfinite(gap) & mask.astype(bool)
    if not finite.any():
        return {key: np.nan for key in (
            "mae", "median_abs_gap", "p90_abs_gap",
            "predicted_low_boundary_share", "predicted_high_boundary_share",
            "grid_star_low_boundary_share", "grid_star_high_boundary_share",
        )}
    valid_gap = gap[finite]
    return {
        "mae": float(valid_gap.mean()),
        "median_abs_gap": float(np.median(valid_gap)),
        "p90_abs_gap": float(np.quantile(valid_gap, 0.90)),
        "predicted_low_boundary_share": float((pred[finite] < 0.05).mean()),
        "predicted_high_boundary_share": float((pred[finite] > 0.95).mean()),
        "grid_star_low_boundary_share": float((star[finite] < 0.05).mean()),
        "grid_star_high_boundary_share": float((star[finite] > 0.95).mean()),
    }


def _distribution_statistics(values: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    finite = np.isfinite(values) & mask.astype(bool)
    if not finite.any():
        return {key: np.nan for key in ("mean", "median", "p90", "p99", "max")}
    selected = values[finite]
    return {
        "mean": float(selected.mean()),
        "median": float(np.median(selected)),
        "p90": float(np.quantile(selected, 0.90)),
        "p99": float(np.quantile(selected, 0.99)),
        "max": float(selected.max()),
    }


def _summary(
    prefix: str,
    pred: np.ndarray,
    star: np.ndarray,
    survival_mask: np.ndarray,
    identified_mask: np.ndarray,
    regret: np.ndarray | None = None,
    top2_margin: np.ndarray | None = None,
    *,
    margin_tol: float,
) -> Dict[str, float]:
    primary_mask = survival_mask.astype(bool) & identified_mask.astype(bool)
    values = _gap_statistics(pred, star, primary_mask)
    survival_values = _gap_statistics(pred, star, survival_mask)
    raw_values = _gap_statistics(pred, star, np.ones_like(survival_mask, dtype=bool))
    values.update({f"survival_{key}": value for key, value in survival_values.items()})
    values.update({f"raw_{key}": value for key, value in raw_values.items()})
    values["survival_grid_share"] = float(survival_mask.astype(bool).mean())
    values["teacher_identified_grid_share"] = float(identified_mask.astype(bool).mean())
    values["survival_identified_grid_share"] = float(primary_mask.mean())
    survival_count = int(survival_mask.astype(bool).sum())
    values["teacher_identified_share"] = values["teacher_identified_grid_share"]
    values["teacher_identified_share_survival"] = (
        float(primary_mask.sum()) / float(survival_count) if survival_count else float("nan")
    )
    values["teacher_margin_tol"] = float(margin_tol)
    for metric_name, metric_values in (("bp_pred", pred), ("bp_grid_star", star)):
        values[f"{metric_name}_mean"] = _distribution_statistics(
            metric_values, primary_mask
        )["mean"]
        values[f"survival_{metric_name}_mean"] = _distribution_statistics(
            metric_values, survival_mask
        )["mean"]
        values[f"raw_{metric_name}_mean"] = _distribution_statistics(
            metric_values, np.ones_like(survival_mask, dtype=bool)
        )["mean"]
    if regret is not None:
        for key, value in _distribution_statistics(regret, primary_mask).items():
            values[f"regret_{key}"] = value
        for key, value in _distribution_statistics(regret, survival_mask).items():
            values[f"survival_regret_{key}"] = value
        for key, value in _distribution_statistics(
            regret, np.ones_like(survival_mask, dtype=bool)
        ).items():
            values[f"raw_regret_{key}"] = value
    if top2_margin is not None:
        margin_finite = np.isfinite(top2_margin)
        margin_values = top2_margin[margin_finite]
        values.update({
            "top2_margin_mean": float(margin_values.mean()) if margin_values.size else float("nan"),
            "top2_margin_p10": float(np.quantile(margin_values, 0.10)) if margin_values.size else float("nan"),
            "top2_margin_p50": float(np.quantile(margin_values, 0.50)) if margin_values.size else float("nan"),
            "top2_margin_p90": float(np.quantile(margin_values, 0.90)) if margin_values.size else float("nan"),
        })
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _objective_frame(result: Dict[str, torch.Tensor], pos: int) -> pd.DataFrame:
    mapping = {
        "bp_candidate": "coarse_bp_grid",
        "value": "coarse_value_grid",
        "cashflow": "coarse_cashflow_grid_mean",
        "continuation": "coarse_continuation_grid_mean",
        "q_issue": "coarse_q_issue_grid",
        "p_child_mean": "coarse_p_child_grid_mean",
        "default_mean": "coarse_default_grid_mean",
        "eta_next_active_share": "coarse_eta_next_active_share",
        "child_b_mean": "coarse_child_b_mean",
        "child_b_eta0_mean": "coarse_child_b_eta0_mean",
        "child_b_eta1_mean": "coarse_child_b_eta1_mean",
    }
    data = {
        name: result[key][pos].detach().cpu().numpy().astype(np.float64)
        for name, key in mapping.items()
    }
    n = len(data["bp_candidate"])
    data.update(
        {
            "bp_star": np.repeat(float(result["bp_star"][pos].item()), n),
            "value_star": np.repeat(float(result["value_star"][pos].item()), n),
            "regret": np.repeat(float(result["regret"][pos].item()), n),
        }
    )
    return pd.DataFrame(data)


def evaluate_bp_consistency(
    model: torch.nn.Module,
    sdf_fc1_model: torch.nn.Module,
    grid: FrozenFirmGrid,
    reference: ReferenceFirmState,
    hyperparams: HyperParams,
    economic_config: AnalysisEconomicConfig,
    *,
    output_dir: str | Path,
    n_child_shocks: int = 2,
    shock_seed: int = 12345,
    teacher_margin_tol: float = 1e-8,
    transition_data: FrozenTransitionData | None = None,
    write_objective_slices: bool = True,
) -> tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, Any]]:
    if float(teacher_margin_tol) < 0.0:
        raise ValueError("teacher_margin_tol must be non-negative")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    transition_data = transition_data or build_frozen_transition_data(
        sdf_fc1_model, grid.base_states, reference, hyperparams, economic_config,
        n_child_shocks=n_child_shocks, shock_seed=shock_seed,
    )
    children = transition_data.children
    m_list = transition_data.m_used_list
    transition_metadata = dict(transition_data.metadata)
    p0_loss, pi_loss = _losses(economic_config)
    teacher = BPGridTeacher.from_hyperparams(model, p0_loss, pi_loss, hyperparams)
    branch_specs = {
        "p0": ("p0", reference.i_mid),
        "pi_low": ("pi", reference.i_low),
        "pi_mid": ("pi", reference.i_mid),
        "pi_high": ("pi", reference.i_high),
    }
    surfaces: Dict[str, np.ndarray] = {}
    summary: Dict[str, float] = {}
    results: Dict[str, Dict[str, torch.Tensor]] = {}
    with _checkpoint_economic_config(economic_config), torch.no_grad():
        for label, (branch, i_value) in branch_specs.items():
            states = grid.base_states.clone()
            states[:, 3] = float(i_value)
            output_model = model(states)
            bp_pred = output_model.bp0 if branch == "p0" else output_model.bpI
            result = teacher.compute(
                states,
                children,
                m_list,
                branch=branch,
                bp_pred=bp_pred,
                child_weights=transition_data.branch_weights,
            )
            results[label] = result
            pred = bp_pred.detach().cpu().reshape(grid.shape).numpy().astype(np.float64)
            star = result["bp_star"].detach().cpu().reshape(grid.shape).numpy().astype(np.float64)
            phat = output_model.Phat.detach().cpu().reshape(grid.shape).numpy().astype(np.float64)
            survival_mask = np.isfinite(phat) & (phat > 0.0)
            top2_margin = (
                result["top2_margin"].detach().cpu().reshape(grid.shape).numpy().astype(np.float64)
            )
            confidence = (
                result["confidence"].detach().cpu().reshape(grid.shape).numpy().astype(np.float64)
            )
            regret = result["regret"].detach().cpu().reshape(grid.shape).numpy().astype(np.float64)
            identified_mask = np.isfinite(top2_margin) & (top2_margin > float(teacher_margin_tol))
            primary_mask = survival_mask & identified_mask
            gap = np.abs(pred - star)
            surfaces[f"{label}_bp_pred_raw"] = pred
            surfaces[f"{label}_bp_grid_star_raw"] = star
            surfaces[f"{label}_bp_abs_gap_raw"] = gap
            surfaces[f"{label}_teacher_top2_margin_raw"] = top2_margin
            surfaces[f"{label}_teacher_confidence_raw"] = confidence
            surfaces[f"{label}_teacher_identified_raw"] = identified_mask.astype(np.float64)
            surfaces[f"{label}_bp_regret_raw"] = regret
            surfaces[f"{label}_bp_pred_survival"] = np.where(survival_mask, pred, np.nan)
            surfaces[f"{label}_bp_grid_star_survival"] = np.where(survival_mask, star, np.nan)
            surfaces[f"{label}_bp_abs_gap_survival"] = np.where(survival_mask, gap, np.nan)
            surfaces[f"{label}_bp_regret_survival"] = np.where(survival_mask, regret, np.nan)
            surfaces[f"{label}_bp_pred_survival_identified"] = np.where(primary_mask, pred, np.nan)
            surfaces[f"{label}_bp_grid_star_survival_identified"] = np.where(primary_mask, star, np.nan)
            surfaces[f"{label}_bp_abs_gap_survival_identified"] = np.where(primary_mask, gap, np.nan)
            surfaces[f"{label}_bp_regret_survival_identified"] = np.where(primary_mask, regret, np.nan)
            summary.update(
                _summary(
                    label,
                    pred,
                    star,
                    survival_mask,
                    identified_mask,
                    regret=regret,
                    top2_margin=top2_margin,
                    margin_tol=teacher_margin_tol,
                )
            )
            for component, key in (
                ("cashflow", "coarse_cashflow_grid_mean"),
                ("continuation", "coarse_continuation_grid_mean"),
                ("value", "coarse_value_grid"),
                ("q_issue", "coarse_q_issue_grid"),
                ("p_child_mean", "coarse_p_child_grid_mean"),
                ("default_mean", "coarse_default_grid_mean"),
            ):
                candidate_values = result[key].detach().cpu().numpy().astype(np.float64)
                candidate_range = np.nanmax(candidate_values, axis=1) - np.nanmin(candidate_values, axis=1)
                summary[f"{label}_{component}_candidate_range_mean"] = float(
                    np.nanmean(candidate_range)
                )
            coarse_argmax = result["coarse_value_grid"].argmax(dim=1, keepdim=True)
            coarse_continuation_at_star = torch.gather(
                result["coarse_continuation_grid_mean"], 1, coarse_argmax
            )
            summary[f"{label}_coarse_continuation_at_star_mean"] = float(
                coarse_continuation_at_star.detach().float().mean().item()
            )
            summary[f"{label}_eta_next_active_share"] = float(
                result["eta_next_active_share"].detach().float().mean().item()
            )

    positions = {
        "b_low_z_low": 0,
        "b_low_z_mid": len(grid.z_values) // 2,
        "b_low_z_high": len(grid.z_values) - 1,
        "b_mid_z_low": (len(grid.b_values) // 2) * len(grid.z_values),
        "b_mid_z_mid": (len(grid.b_values) // 2) * len(grid.z_values) + len(grid.z_values) // 2,
        "b_mid_z_high": (len(grid.b_values) // 2 + 1) * len(grid.z_values) - 1,
        "b_high_z_low": (len(grid.b_values) - 1) * len(grid.z_values),
        "b_high_z_mid": (len(grid.b_values) - 1) * len(grid.z_values) + len(grid.z_values) // 2,
        "b_high_z_high": len(grid.b_values) * len(grid.z_values) - 1,
    }
    if write_objective_slices:
        for branch_label in ("p0", "pi_mid"):
            for state_label, pos in positions.items():
                frame = _objective_frame(results[branch_label], pos)
                stem = f"{branch_label}_{state_label}"
                frame.to_csv(output / f"{stem}.csv", index=False)
                plot_objective_slice(frame, output / f"{stem}.png", title=stem)

    transition_metadata["teacher_margin_tol"] = float(teacher_margin_tol)
    transition_metadata["primary_bp_mask"] = "finite Phat>0 and top2_margin>teacher_margin_tol"
    return surfaces, summary, transition_metadata
