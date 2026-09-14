from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import torch

from .boundaries import zero_crossings
from .grids import FrozenFirmGrid, ReferenceFirmState


OUTPUT_FIELDS = (
    "Q",
    "bp0",
    "bpI",
    "P0",
    "PI",
    "bar_i_cond",
    "bar_i_eff",
    "bar_z",
    "P",
    "Phat",
    "bp_cond",
    "bp",
    "survival_prob",
)


def _field(output, name: str) -> torch.Tensor:
    if isinstance(output, dict):
        return output[name]
    if not hasattr(output, name):
        raise AttributeError(f"PolicyValue output is missing required field {name!r}")
    return getattr(output, name)


def _forward_fields(
    model: torch.nn.Module,
    states: torch.Tensor,
    fields: Iterable[str],
    *,
    chunk_size: int,
) -> Dict[str, torch.Tensor]:
    step = int(chunk_size)
    if step <= 0:
        raise ValueError("chunk_size must be positive")
    collected: Dict[str, list[torch.Tensor]] = {name: [] for name in fields}
    with torch.no_grad():
        for start in range(0, states.shape[0], step):
            output = model(states[start:start + step])
            for name in fields:
                collected[name].append(_field(output, name).detach())
    return {name: torch.cat(parts, dim=0) for name, parts in collected.items()}


def _reshape(tensor: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    return tensor.detach().cpu().reshape(shape).numpy().astype(np.float64, copy=False)


def evaluate_firm_surfaces(
    model: torch.nn.Module,
    grid: FrozenFirmGrid,
    reference: ReferenceFirmState,
    *,
    chunk_size: int = 8192,
    q_unit_eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    model.eval()
    surfaces: Dict[str, np.ndarray] = {}
    i_slices = {
        "low": reference.i_low,
        "mid": reference.i_mid,
        "high": reference.i_high,
    }
    for label, i_value in i_slices.items():
        states = grid.base_states.clone()
        states[:, 3] = float(i_value)
        values = _forward_fields(model, states, OUTPUT_FIELDS, chunk_size=chunk_size)
        for name, tensor in values.items():
            surfaces[f"{name}_{label}"] = _reshape(tensor, grid.shape)

    for name in (
        "Q",
        "P0",
        "P",
        "Phat",
        "bar_i_cond",
        "bar_i_eff",
        "bar_z",
        "survival_prob",
        "bp0",
        "bpI",
        "bp_cond",
        "bp",
    ):
        surfaces[name] = surfaces[f"{name}_mid"]
    surfaces["V0"] = surfaces["P0"]
    for label in i_slices:
        surfaces[f"VI_{label}"] = surfaces[f"PI_{label}"]

    b = grid.mesh_b
    q = surfaces["Q"]
    q_unit = np.full_like(q, np.nan, dtype=np.float64)
    valid_b = b > float(q_unit_eps)
    q_unit[valid_b] = q[valid_b] / b[valid_b]
    surfaces["q_unit"] = q_unit
    surfaces["default_region"] = (surfaces["Phat"] <= 0.0).astype(np.float64)
    slice_masks = {
        label: np.isfinite(surfaces[f"Phat_{label}"]) & (surfaces[f"Phat_{label}"] > 0.0)
        for label in i_slices
    }
    survival_mask = slice_masks["mid"]
    surfaces["survival_mask"] = survival_mask.astype(np.float64)
    for label, mask in slice_masks.items():
        surfaces[f"survival_mask_{label}"] = mask.astype(np.float64)
    for name in ("bp0", "bpI", "bp_cond", "bp"):
        surfaces[f"{name}_raw"] = surfaces[name].copy()
        surfaces[f"{name}_survival"] = np.where(survival_mask, surfaces[name], np.nan)
        for label, mask in slice_masks.items():
            slice_name = f"{name}_{label}"
            surfaces[f"{slice_name}_raw"] = surfaces[slice_name].copy()
            surfaces[f"{slice_name}_survival"] = np.where(mask, surfaces[slice_name], np.nan)
    return surfaces


def finite_difference_summary(surfaces: Dict[str, np.ndarray]) -> Dict[str, float]:
    def share(values: np.ndarray, predicate) -> float:
        finite = np.isfinite(values)
        if not finite.any():
            return float("nan")
        return float(predicate(values[finite]).mean())

    p = surfaces["P"]
    v0 = surfaces["P0"]
    return {
        "P_z_positive_share": share(np.diff(p, axis=1), lambda x: x > 0.0),
        "P_b_negative_share": share(np.diff(p, axis=0), lambda x: x < 0.0),
        "V0_z_positive_share": share(np.diff(v0, axis=1), lambda x: x > 0.0),
        "V0_b_negative_share": share(np.diff(v0, axis=0), lambda x: x < 0.0),
    }


def evaluate_investment_cutoff(
    model: torch.nn.Module,
    grid: FrozenFirmGrid,
    reference: ReferenceFirmState,
    *,
    i_points: int = 101,
    i_min: float = 0.0,
    i_max: float = 0.5,
    chunk_size: int = 8192,
    survival_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    if i_points < 2:
        raise ValueError("i_points must be at least 2")
    i_min = float(i_min)
    i_max = float(i_max)
    if not i_min < i_max:
        raise ValueError("investment i_min must be below i_max")
    step = int(chunk_size)
    if step <= 0:
        raise ValueError("chunk_size must be positive")
    i_values = np.linspace(i_min, i_max, int(i_points), dtype=np.float64)
    n_state = grid.base_states.shape[0]
    expanded = (
        grid.base_states.unsqueeze(1)
        .expand(n_state, len(i_values), 7)
        .reshape(n_state * len(i_values), 7)
        .clone()
    )
    expanded[:, 3] = torch.tensor(
        np.tile(i_values, n_state),
        dtype=expanded.dtype,
        device=expanded.device,
    )
    deltas = []
    with torch.no_grad():
        for start in range(0, expanded.shape[0], step):
            components = model.forward_value_components(expanded[start:start + step])
            deltas.append((components["VI_physical"] - components["V0_physical"]).detach().cpu())
    delta = torch.cat(deltas, dim=0).reshape(n_state, len(i_values)).numpy().astype(np.float64)
    cutoff = np.full(n_state, np.nan, dtype=np.float64)
    crossing_count = np.zeros(n_state, dtype=np.int64)
    status = np.full(n_state, "nonfinite", dtype=object)
    for row in range(n_state):
        y = delta[row]
        finite = np.isfinite(y)
        if not finite.all():
            continue
        crossings = zero_crossings(i_values, y)
        crossing_count[row] = len(crossings)
        if len(crossings) == 1:
            cutoff[row] = float(crossings[0])
            status[row] = "single_crossing"
        elif len(crossings) > 1:
            status[row] = "multiple_crossings"
        elif np.all(y > 0.0):
            status[row] = "all_invest"
        elif np.all(y < 0.0):
            status[row] = "all_no_invest"
        else:
            status[row] = "nonfinite"
    cutoff = cutoff.reshape(grid.shape)
    cutoff = np.where(survival_mask.astype(bool), cutoff, np.nan)
    status = status.reshape(grid.shape)
    mid_pos = int(np.argmin(np.abs(i_values - reference.i_mid)))
    region = (delta[:, mid_pos].reshape(grid.shape) > 0.0) & survival_mask.astype(bool)
    return {
        "i_values": i_values,
        "delta_vi_v0": delta.reshape(grid.shape + (len(i_values),)),
        "i_star": cutoff,
        "investment_region_mid": region.astype(np.float64),
        "crossing_count": crossing_count.reshape(grid.shape),
        "investment_status": status,
        "survival_mask": survival_mask.astype(bool),
    }
