from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def zero_crossings(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    crossings = [float(grid[pos]) for pos in np.where(finite & (values == 0.0))[0]]
    for pos in range(len(values) - 1):
        if not finite[pos] or not finite[pos + 1]:
            continue
        y0, y1 = values[pos], values[pos + 1]
        if y0 * y1 < 0.0:
            crossings.append(
                float(grid[pos] - y0 * (grid[pos + 1] - grid[pos]) / (y1 - y0))
            )
    if not crossings:
        return np.empty(0, dtype=np.float64)
    crossings = sorted(crossings)
    unique = [crossings[0]]
    tolerance = max(float(np.ptp(grid)), 1.0) * 1e-12
    for value in crossings[1:]:
        if abs(value - unique[-1]) > tolerance:
            unique.append(value)
    return np.asarray(unique, dtype=np.float64)


def extract_phat_default_boundary(
    b_values: np.ndarray,
    z_values: np.ndarray,
    phat: np.ndarray,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    if phat.shape != (len(b_values), len(z_values)):
        raise ValueError("Phat surface shape does not match b/z grids")
    rows = []
    for b_pos, b_value in enumerate(b_values):
        y = np.asarray(phat[b_pos], dtype=np.float64)
        finite = np.isfinite(y)
        crossings = zero_crossings(z_values, y)
        if len(crossings) == 1:
            z_boundary = float(crossings[0])
            status = "single_crossing"
        elif len(crossings) > 1:
            z_boundary = float("nan")
            status = "multiple_crossings"
        else:
            z_boundary = float("nan")
            finite_y = y[finite]
            if finite_y.size == 0:
                status = "nonfinite"
            elif np.all(finite_y <= 0.0):
                status = "all_default"
            elif np.all(finite_y > 0.0):
                status = "all_survival"
            else:
                status = "no_identified_crossing"
        rows.append(
            {
                "b": float(b_value),
                "z_default": z_boundary,
                "boundary_status": status,
                "crossing_count": int(len(crossings)),
            }
        )
    frame = pd.DataFrame(rows)
    observed = (frame["boundary_status"] == "single_crossing").to_numpy()
    z_all = frame["z_default"].to_numpy(dtype=np.float64)
    adjacent = observed[:-1] & observed[1:]
    if adjacent.any():
        monotonic_share = float((np.diff(z_all)[adjacent] >= -1e-10).mean())
    else:
        monotonic_share = float("nan")
    triples = observed[:-2] & observed[1:-1] & observed[2:]
    if triples.any():
        roughness = float(np.mean(np.abs(np.diff(z_all, n=2)[triples])))
    else:
        roughness = float("nan")
    summary = {
        "default_boundary_observed_share": float(observed.mean()),
        "default_boundary_single_crossing_share": float(observed.mean()),
        "default_boundary_monotonic_share": monotonic_share,
        "default_boundary_roughness": roughness,
        "default_boundary_multiple_crossing_share": float(
            (frame["boundary_status"] == "multiple_crossings").mean()
        ),
    }
    return frame, summary


def compare_hard_soft_default_boundaries(
    b_values: np.ndarray,
    z_values: np.ndarray,
    phat: np.ndarray,
    bar_z: np.ndarray,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    """Compare the hard Phat=0 boundary with the soft bar_z=0.5 contour."""
    hard, _ = extract_phat_default_boundary(b_values, z_values, phat)
    soft, _ = extract_phat_default_boundary(b_values, z_values, 0.5 - bar_z)
    frame = hard.rename(columns={
        "z_default": "z_hard_phat0",
        "boundary_status": "hard_status",
        "crossing_count": "hard_crossing_count",
    }).merge(
        soft.rename(columns={
            "z_default": "z_soft_barz0p5",
            "boundary_status": "soft_status",
            "crossing_count": "soft_crossing_count",
        }),
        on="b",
        how="outer",
    )
    comparable = frame["hard_status"].eq("single_crossing") & frame["soft_status"].eq("single_crossing")
    gap = (
        frame.loc[comparable, "z_hard_phat0"]
        - frame.loc[comparable, "z_soft_barz0p5"]
    ).abs()
    summary = {
        "hard_default_share": float((np.asarray(phat) <= 0.0).mean()),
        "soft_default_mean": float(np.nanmean(np.asarray(bar_z, dtype=np.float64))),
        "hard_soft_boundary_comparable_share": float(comparable.mean()),
        "hard_soft_boundary_abs_gap_mean": float(gap.mean()) if len(gap) else float("nan"),
        "hard_soft_boundary_abs_gap_p90": float(gap.quantile(0.90)) if len(gap) else float("nan"),
    }
    return frame, summary
