from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .convergence_artifacts import SurfaceData, exact_surface_alignment


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def function_drift_metrics(current: SurfaceData, previous: SurfaceData) -> dict[str, float]:
    aligned, reason = exact_surface_alignment(previous, current)
    if not aligned:
        raise ValueError(reason)
    mask = np.isfinite(previous.values) & np.isfinite(current.values)
    if not bool(mask.any()):
        raise ValueError("no jointly finite surface observations")
    old = previous.values[mask]
    new = current.values[mask]
    absolute = np.abs(new - old)
    previous_abs = np.abs(old)
    eps = np.finfo(np.float64).eps
    return {
        "n_aligned": int(mask.sum()),
        "max_abs_diff": float(absolute.max()),
        "mean_abs_diff": float(absolute.mean()),
        "rmse_diff": float(np.sqrt(np.mean(np.square(new - old)))),
        "p90_abs_diff": float(np.quantile(absolute, 0.90)),
        "p99_abs_diff": float(np.quantile(absolute, 0.99)),
        "mean_abs_level_previous": float(previous_abs.mean()),
        "max_abs_level_previous": float(previous_abs.max()),
        "normalized_mean_abs_diff": float(absolute.mean() / (previous_abs.mean() + eps)),
        "normalized_max_abs_diff": float(absolute.max() / (previous_abs.max() + eps)),
    }


def residual_metrics(values: np.ndarray) -> dict[str, float]:
    finite = _finite(values)
    if finite.size == 0:
        return {
            key: float("nan")
            for key in (
                "n_finite", "mean_abs", "median_abs", "p90_abs", "p95_abs",
                "p99_abs", "max_abs", "mean_signed", "median_signed",
            )
        }
    absolute = np.abs(finite)
    return {
        "n_finite": int(finite.size),
        "mean_abs": float(absolute.mean()),
        "median_abs": float(np.median(absolute)),
        "p90_abs": float(np.quantile(absolute, 0.90)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "max_abs": float(absolute.max()),
        "mean_signed": float(finite.mean()),
        "median_signed": float(np.median(finite)),
    }


def compute_simulated_moments(frame: pd.DataFrame) -> tuple[dict[str, float], list[str]]:
    row: dict[str, float] = {"n_state_rows": int(len(frame))}
    unavailable: list[str] = []

    def numeric(name: str) -> np.ndarray | None:
        if name not in frame.columns:
            return None
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        return values[np.isfinite(values)]

    b = numeric("b")
    if b is None or b.size == 0:
        unavailable.append("leverage b")
    else:
        row.update({
            "mean_b": float(b.mean()),
            "std_b": float(b.std(ddof=0)),
            "median_b": float(np.median(b)),
            "p90_b": float(np.quantile(b, 0.90)),
            "p95_b": float(np.quantile(b, 0.95)),
            "p99_b": float(np.quantile(b, 0.99)),
        })

    bp = numeric("bp")
    if bp is None or bp.size == 0:
        unavailable.append("bp")
    else:
        row.update({
            "mean_bp": float(bp.mean()),
            "p90_bp": float(np.quantile(bp, 0.90)),
            "p99_bp": float(np.quantile(bp, 0.99)),
        })

    m = numeric("M")
    if m is not None and m.size:
        row.update({"mean_M": float(m.mean()), "std_M": float(m.std(ddof=0))})
    else:
        unavailable.append("M")

    if "alive" in frame.columns:
        alive = frame["alive"].astype(bool).to_numpy()
        row["survival_rate"] = float(alive.mean())
        row["default_rate"] = float(1.0 - alive.mean())
        row["alive_firm_count"] = int(alive.sum())
    elif "default" in frame.columns:
        default = frame["default"].astype(bool).to_numpy()
        row["default_rate"] = float(default.mean())
        row["survival_rate"] = float(1.0 - default.mean())
        row["alive_firm_count"] = int((~default).sum())
    else:
        # SimulateTS only writes extant current-node rows; this is a count, not
        # an inferred default rate.
        row["alive_firm_count"] = int(len(frame))
        unavailable.append("explicit default/alive indicator")

    if "investment" in frame.columns:
        investment = frame["investment"].astype(bool).to_numpy()
        row["investment_rate"] = float(investment.mean())
    else:
        unavailable.append("realized investment indicator")

    if "entry" in frame.columns:
        entry = pd.to_numeric(frame["entry"], errors="coerce").to_numpy(dtype=np.float64)
        row["entrant_count"] = int(np.nansum(entry > 0.5))
    else:
        unavailable.append("entry")
    if "exit" in frame.columns:
        exit_values = pd.to_numeric(frame["exit"], errors="coerce").to_numpy(dtype=np.float64)
        row["exit_count"] = int(np.nansum(exit_values > 0.5))
    else:
        unavailable.append("explicit exit indicator")
    return row, unavailable


def regression_metrics(
    calculated: np.ndarray,
    forecast: np.ndarray,
    *,
    calculated_name: str | None = None,
    forecast_name: str | None = None,
) -> dict[str, float]:
    raw_x = np.asarray(calculated, dtype=np.float64)
    raw_y = np.asarray(forecast, dtype=np.float64)
    if raw_x.shape != raw_y.shape:
        x_label = "calculated" if calculated_name is None else f"calculated ({calculated_name})"
        y_label = "forecast" if forecast_name is None else f"forecast ({forecast_name})"
        raise ValueError(
            "regression_metrics requires identically shaped inputs; got "
            f"{x_label} shape {raw_x.shape} vs {y_label} shape {raw_y.shape}. "
            "This usually means a duplicate column was selected from a pandas "
            "DataFrame (which returns a DataFrame, not a Series)."
        )
    x = raw_x.reshape(-1)
    y = raw_y.reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return {key: float("nan") for key in (
            "n", "r2", "pearson_corr", "slope", "intercept", "rmse", "mae"
        )}
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = intercept + slope * x
    residual = y - x
    ss_res = float(np.square(y - predicted).sum())
    ss_total = float(np.square(y - y.mean()).sum())
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return {
        "n": int(x.size),
        "r2": float(1.0 - ss_res / ss_total) if ss_total > 0 else float("nan"),
        "pearson_corr": corr,
        "slope": float(slope),
        "intercept": float(intercept),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
    }


def compute_shift_metrics(
    frame: pd.DataFrame,
    *,
    forecast_col: str,
    calculated_col: str,
    shifts: Sequence[int],
    path_col: str = "path",
    time_col: str = "t",
) -> pd.DataFrame:
    """Compare forecast_t with calculated_(t+shift) within each path."""
    required = {path_col, time_col, forecast_col, calculated_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"macro dataframe missing columns: {missing}")
    base = frame[[path_col, time_col, forecast_col, calculated_col]].copy()
    base[time_col] = pd.to_numeric(base[time_col], errors="coerce")
    base[forecast_col] = pd.to_numeric(base[forecast_col], errors="coerce")
    base[calculated_col] = pd.to_numeric(base[calculated_col], errors="coerce")
    rows: list[dict[str, float]] = []
    calculated = base[[path_col, time_col, calculated_col]].rename(
        columns={time_col: "calculated_t", calculated_col: "calculated_shifted"}
    )
    for shift in shifts:
        forecast = base[[path_col, time_col, forecast_col]].copy()
        forecast["calculated_t"] = forecast[time_col] + int(shift)
        aligned = forecast.merge(calculated, on=[path_col, "calculated_t"], how="inner")
        metrics = regression_metrics(
            aligned["calculated_shifted"].to_numpy(),
            aligned[forecast_col].to_numpy(),
        )
        rows.append({"shift": int(shift), **metrics})
    return pd.DataFrame(rows)


def choose_representative_episodes(episodes: Iterable[int]) -> list[int]:
    ordered = sorted(set(int(value) for value in episodes))
    if not ordered:
        return []
    return sorted(set([ordered[0], ordered[len(ordered) // 2], ordered[-1]]))
