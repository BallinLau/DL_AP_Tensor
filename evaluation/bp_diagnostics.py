from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch

from analysis.economic_config import AnalysisEconomicConfig
from config import Config, HyperParams
from losses import P0Loss, PILoss
from training.bp_grid_teacher import BPGridTeacher
from training.episode import Episode

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


def _cat_batches(batches: Iterable[Dict[str, Any]], key: str) -> torch.Tensor:
    values = [batch[key] for batch in batches if batch.get(key) is not None]
    if not values:
        raise RuntimeError(f"No values found for batch key {key!r}")
    return torch.cat(values, dim=0)


def build_reference_transition_bank(
    firm_df: pd.DataFrame,
    hyperparams: HyperParams,
    *,
    device: torch.device,
    n_branches: int = 2,
) -> Dict[str, Any]:
    episode = Episode.__new__(Episode)
    episode.device = device
    episode.hyperparams = hyperparams
    episode.models = {}
    batches = episode._create_firm_batches_from_df(
        firm_df,
        batch_size=max(1, len(firm_df)),
        n_branches=n_branches,
        eta_resample=False,
    )
    if not batches:
        raise RuntimeError("Reference firm dataframe contains no matched parent-child transitions")
    parent = _cat_batches(batches, "parent")
    children = [_cat_batches(batches, f"child{k}") for k in range(n_branches)]
    if parent.shape[1] < 7 or any(child.shape[1] < 8 for child in children):
        raise ValueError(
            "BP value-objective evaluation requires seven firm-state columns and observed child M"
        )
    active = parent[:, 2] > 0.5
    if bool(active.any()):
        parent = parent[active]
        children = [child[active] for child in children]
    transition_deltas = []
    m_values = []
    for child in children:
        transition_deltas.append((child[:, :7] - parent[:, :7]).median(dim=0).values)
        m_values.append(child[:, 7:8].median(dim=0).values.reshape(1, 1))
    return {
        "transition_deltas": transition_deltas,
        "m_values": m_values,
        "n_matched_parent_transitions": int(parent.shape[0]),
    }


def _expand_transition_bank(
    parent_states: torch.Tensor,
    bank: Dict[str, Any],
    hyperparams: HyperParams,
) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
    children: List[torch.Tensor] = []
    m_list: List[torch.Tensor] = []
    use_clipped_m = bool(getattr(hyperparams, "pv_use_clipped_m", True))
    m_lo = float(getattr(hyperparams, "pv_m_clamp_min", 0.7))
    m_hi = float(getattr(hyperparams, "pv_m_clamp_max", 1.3))
    for delta, m_value in zip(bank["transition_deltas"], bank["m_values"]):
        child = parent_states + delta.to(parent_states.device, parent_states.dtype)
        child = child.clone()
        child[:, 2] = child[:, 2].clamp(0.0, 1.0)
        children.append(child)
        m = m_value.to(parent_states.device, parent_states.dtype).expand(parent_states.shape[0], 1)
        if use_clipped_m:
            m = m.clamp(m_lo, m_hi)
        m_list.append(m)
    return children, m_list


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


def _summary(prefix: str, pred: np.ndarray, star: np.ndarray) -> Dict[str, float]:
    gap = np.abs(pred - star)
    finite = np.isfinite(gap)
    if not finite.any():
        values = {"mae": np.nan, "median_abs_gap": np.nan, "p90_abs_gap": np.nan}
    else:
        valid = gap[finite]
        values = {
            "mae": float(valid.mean()),
            "median_abs_gap": float(np.median(valid)),
            "p90_abs_gap": float(np.quantile(valid, 0.90)),
        }
    values.update(
        {
            "predicted_low_boundary_share": float((pred < 0.05).mean()),
            "predicted_high_boundary_share": float((pred > 0.95).mean()),
            "grid_star_low_boundary_share": float((star < 0.05).mean()),
            "grid_star_high_boundary_share": float((star > 0.95).mean()),
        }
    )
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
    grid: FrozenFirmGrid,
    reference: ReferenceFirmState,
    firm_df: pd.DataFrame,
    hyperparams: HyperParams,
    economic_config: AnalysisEconomicConfig,
    *,
    output_dir: str | Path,
    n_branches: int = 2,
) -> tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, Any]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bank = build_reference_transition_bank(
        firm_df,
        hyperparams,
        device=grid.base_states.device,
        n_branches=n_branches,
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
            children, m_list = _expand_transition_bank(states, bank, hyperparams)
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
            surfaces[f"{label}_bp_pred"] = pred
            surfaces[f"{label}_bp_grid_star"] = star
            surfaces[f"{label}_bp_abs_gap"] = np.abs(pred - star)
            summary.update(_summary(label, pred, star))

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

    bank_meta = {
        "n_matched_parent_transitions": bank["n_matched_parent_transitions"],
        "n_branches": len(bank["transition_deltas"]),
        "transition_deltas": [delta.detach().cpu().tolist() for delta in bank["transition_deltas"]],
        "observed_child_m": [float(value.item()) for value in bank["m_values"]],
        "m_mode": "clipped_train_m" if bool(getattr(hyperparams, "pv_use_clipped_m", True)) else "raw_observed_m",
    }
    return surfaces, summary, bank_meta
