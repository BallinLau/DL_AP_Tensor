from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


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
        crossing = np.where(
            finite[:-1]
            & finite[1:]
            & ((y[:-1] == 0.0) | (y[1:] == 0.0) | (y[:-1] * y[1:] < 0.0))
        )[0]
        if len(crossing):
            pos = int(crossing[0])
            y0, y1 = y[pos], y[pos + 1]
            if y0 == 0.0 or y1 == y0:
                z_boundary = float(z_values[pos])
            elif y1 == 0.0:
                z_boundary = float(z_values[pos + 1])
            else:
                z_boundary = float(
                    z_values[pos]
                    - y0 * (z_values[pos + 1] - z_values[pos]) / (y1 - y0)
                )
            status = "observed"
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
                "crossing_count": int(len(crossing)),
            }
        )
    frame = pd.DataFrame(rows)
    observed = (frame["boundary_status"] == "observed").to_numpy()
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
        "default_boundary_monotonic_share": monotonic_share,
        "default_boundary_roughness": roughness,
        "default_boundary_multiple_crossing_share": float((frame["crossing_count"] > 1).mean()),
    }
    return frame, summary
