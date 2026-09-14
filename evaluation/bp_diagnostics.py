from __future__ import annotations

from contextlib import contextmanager
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

from .grids import FrozenFirmGrid, ReferenceFirmState
from .plotting import plot_objective_slice


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


def build_frozen_transition_children(
    sdf_fc1_model: torch.nn.Module,
    parent_states: torch.Tensor,
    reference: ReferenceFirmState,
    hyperparams: HyperParams,
    economic_config: AnalysisEconomicConfig,
    *,
    n_child_shocks: int,
    shock_seed: int,
) -> tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
    if int(n_child_shocks) < 2:
        raise ValueError("n_child_shocks must be at least 2")
    device = parent_states.device
    dtype = parent_states.dtype
    base_bank = ConvergenceShockBank.create(
        1,
        int(n_child_shocks),
        seed=int(shock_seed),
        device=device,
        dtype=dtype,
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
    children: List[torch.Tensor] = []
    m_list: List[torch.Tensor] = []
    for child_pos in range(int(n_child_shocks)):
        children.append(
            torch.stack(
                [
                    parent_states[:, 0],
                    bundle.z_next[:, child_pos, 0],
                    bundle.eta_next[:, child_pos, 0],
                    bundle.i_next[:, child_pos, 0],
                    bundle.x_next[:, child_pos, 0],
                    bundle.hatcf_next[:, child_pos, 0],
                    bundle.lnkf_next[:, child_pos, 0],
                ],
                dim=1,
            )
        )
        m = bundle.m_raw[:, child_pos, :]
        if use_clipped_m:
            m = m.clamp(m_lo, m_hi)
        m_list.append(m)
    metadata = {
        "builder": "ConvergenceShockBank+build_child_exogenous_bundle",
        "shock_seed": int(shock_seed),
        "n_child_shocks": int(n_child_shocks),
        "common_shocks_across_frozen_grid": True,
        "macro_context": {
            "hatc_cal": float(reference.hatc_cal),
            "lnk_cal": float(reference.lnk_cal),
        },
        "m_source": "sdf_fc1.forward_step",
        "m_mode": "clipped_train_m" if use_clipped_m else "raw_sdf_m",
        "m_raw_mean": float(bundle.m_raw.detach().mean().item()),
        "m_raw_std": float(bundle.m_raw.detach().std(unbiased=False).item()),
    }
    return children, m_list, metadata


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


def _summary(
    prefix: str,
    pred: np.ndarray,
    star: np.ndarray,
    survival_mask: np.ndarray,
) -> Dict[str, float]:
    values = _gap_statistics(pred, star, survival_mask)
    raw_values = _gap_statistics(pred, star, np.ones_like(survival_mask, dtype=bool))
    values.update({f"raw_{key}": value for key, value in raw_values.items()})
    values["survival_grid_share"] = float(survival_mask.astype(bool).mean())
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
) -> tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, Any]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    children, m_list, transition_metadata = build_frozen_transition_children(
        sdf_fc1_model,
        grid.base_states,
        reference,
        hyperparams,
        economic_config,
        n_child_shocks=n_child_shocks,
        shock_seed=shock_seed,
    )
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
            )
            results[label] = result
            pred = bp_pred.detach().cpu().reshape(grid.shape).numpy().astype(np.float64)
            star = result["bp_star"].detach().cpu().reshape(grid.shape).numpy().astype(np.float64)
            phat = output_model.Phat.detach().cpu().reshape(grid.shape).numpy().astype(np.float64)
            survival_mask = np.isfinite(phat) & (phat > 0.0)
            gap = np.abs(pred - star)
            surfaces[f"{label}_bp_pred_raw"] = pred
            surfaces[f"{label}_bp_grid_star_raw"] = star
            surfaces[f"{label}_bp_abs_gap_raw"] = gap
            surfaces[f"{label}_bp_pred_survival"] = np.where(survival_mask, pred, np.nan)
            surfaces[f"{label}_bp_grid_star_survival"] = np.where(survival_mask, star, np.nan)
            surfaces[f"{label}_bp_abs_gap_survival"] = np.where(survival_mask, gap, np.nan)
            summary.update(_summary(label, pred, star, survival_mask))

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
    for branch_label in ("p0", "pi_mid"):
        for state_label, pos in positions.items():
            frame = _objective_frame(results[branch_label], pos)
            stem = f"{branch_label}_{state_label}"
            frame.to_csv(output / f"{stem}.csv", index=False)
            plot_objective_slice(frame, output / f"{stem}.png", title=stem)

    return surfaces, summary, transition_metadata
