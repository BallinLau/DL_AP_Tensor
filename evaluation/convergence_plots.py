from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_function_drift(frame: pd.DataFrame, metric: str, path: str | Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for surface, group in frame.groupby("surface", sort=False):
        ax.plot(group["episode"], group[metric], marker="o", label=surface)
    ax.set(xlabel="episode e (difference from e-1)", ylabel=metric, title=title)
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, path)


def plot_function_dashboard(frames: Sequence[pd.DataFrame], path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for frame in frames:
        if frame.empty:
            continue
        eta = int(frame["eta"].iloc[0])
        for surface, group in frame.groupby("surface", sort=False):
            axes[0].plot(
                group["episode"], group["mean_abs_diff"], marker="o",
                label=f"eta{eta} {surface}",
            )
            axes[1].plot(
                group["episode"], group["max_abs_diff"], marker="o",
                label=f"eta{eta} {surface}",
            )
    axes[0].set(title="Mean absolute function drift", xlabel="episode e", ylabel="mean |f_e-f_(e-1)|")
    axes[1].set(title="Maximum absolute function drift", xlabel="episode e", ylabel="max |f_e-f_(e-1)|")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    _save(fig, path)


def plot_bellman_metric(frame: pd.DataFrame, metric: str, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for (eta, equation), group in frame.groupby(["eta", "equation"], sort=False):
        ax.plot(group["episode"], group[metric], marker="o", label=f"eta{eta} {equation}")
    ax.set(xlabel="episode", ylabel=metric, title=f"Bellman residual {metric} by episode")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, path)


def plot_bellman_dashboard(frame: pd.DataFrame, path: str | Path) -> None:
    metrics = ["mean_abs", "p90_abs", "p99_abs", "max_abs"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, metric in zip(axes.reshape(-1), metrics):
        for (eta, equation), group in frame.groupby(["eta", "equation"], sort=False):
            ax.plot(group["episode"], group[metric], marker="o", label=f"eta{eta} {equation}")
        ax.set(title=metric, xlabel="episode", ylabel=metric)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Bellman residual convergence")
    _save(fig, path)


def plot_moment_lines(
    frame: pd.DataFrame,
    columns: Iterable[str],
    path: str | Path,
    *,
    title: str,
) -> bool:
    available = [name for name in columns if name in frame.columns and frame[name].notna().any()]
    if not available:
        return False
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in available:
        ax.plot(frame["episode"], frame[name], marker="o", label=name)
    ax.set(xlabel="episode", title=title)
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, path)
    return True


def plot_distribution(values: np.ndarray, path: str | Path, *, xlabel: str, title: str) -> None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(finite, bins=50, density=True, color="tab:blue", alpha=0.75)
    ax.set(xlabel=xlabel, ylabel="density", title=title)
    ax.grid(alpha=0.2)
    _save(fig, path)


def plot_joint_distribution(
    b: np.ndarray,
    z: np.ndarray,
    path: str | Path,
    *,
    title: str,
    boundaries: Mapping[str, pd.DataFrame] | None = None,
) -> None:
    b = np.asarray(b, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    mask = np.isfinite(b) & np.isfinite(z)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    hist = ax.hist2d(b[mask], z[mask], bins=60, cmap="viridis")
    fig.colorbar(hist[3], ax=ax, label="firm-state count")
    for label, frame in (boundaries or {}).items():
        if {"b", "z_default"}.issubset(frame.columns):
            observed = frame
            if "boundary_status" in frame.columns:
                observed = frame.loc[frame["boundary_status"] == "single_crossing"]
            ax.plot(observed["b"], observed["z_default"], linewidth=1.4, label=label)
    if boundaries:
        ax.legend()
    ax.set(xlabel="b", ylabel="z", title=title)
    _save(fig, path)


def plot_macro_scatter(
    calculated: np.ndarray,
    forecast: np.ndarray,
    path: str | Path,
    *,
    title: str,
    calculated_label: str,
    forecast_label: str,
) -> None:
    x = np.asarray(calculated, dtype=np.float64)
    y = np.asarray(forecast, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=9, alpha=0.35)
    if x.size:
        low, high = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        ax.plot([low, high], [low, high], color="black", linestyle="--", label="y=x")
    ax.set(xlabel=calculated_label, ylabel=forecast_label, title=title)
    ax.grid(alpha=0.2)
    ax.legend()
    _save(fig, path)


def plot_shift_metrics(frame: pd.DataFrame, path: str | Path, *, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, metric in zip(axes, ["pearson_corr", "rmse", "mae"]):
        ax.plot(frame["shift"], frame[metric], marker="o")
        ax.set(xlabel="k: forecast_t vs calculated_(t+k)", ylabel=metric, title=metric)
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    _save(fig, path)


def plot_macro_timeseries(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    time_col: str,
    calculated_col: str,
    forecast_col: str,
    title: str,
) -> None:
    ordered = frame.sort_values(time_col)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ordered[time_col], ordered[calculated_col], label=f"calculated: {calculated_col}")
    ax.plot(ordered[time_col], ordered[forecast_col], label=f"forecast: {forecast_col}")
    ax.set(xlabel="t", title=title)
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, path)


def plot_stage_dashboard(
    *,
    function_drift: pd.DataFrame,
    bellman: pd.DataFrame,
    moments: pd.DataFrame,
    macro_metrics: pd.DataFrame,
    path: str | Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    panels = axes.reshape(-1)

    for surfaces, ax, title in [
        (("Q", "P"), panels[0], "Q/P mean drift"),
        (("bar_z", "bp"), panels[1], "bar_z/bp mean drift"),
    ]:
        subset = (
            function_drift[function_drift["surface"].isin(surfaces)]
            if not function_drift.empty else function_drift
        )
        if not subset.empty:
            for (eta, surface), group in subset.groupby(["eta", "surface"], sort=False):
                ax.plot(group["episode"], group["mean_abs_diff"], marker="o", label=f"eta{eta} {surface}")
        ax.set(title=title, xlabel="episode")
        ax.grid(alpha=0.25)
        if not subset.empty:
            ax.legend(fontsize=8)

    ax = panels[2]
    if not bellman.empty:
        for (eta, equation), group in bellman.groupby(["eta", "equation"], sort=False):
            ax.plot(group["episode"], group["p90_abs"], marker="o", label=f"eta{eta} {equation}")
    ax.set(title="Bellman p90 absolute residual", xlabel="episode")
    ax.grid(alpha=0.25)
    if not bellman.empty:
        ax.legend(fontsize=8)

    ax = panels[3]
    moment_columns = [name for name in ("default_rate", "investment_rate", "mean_b") if name in moments.columns]
    for name in moment_columns:
        ax.plot(moments["episode"], moments[name], marker="o", label=name)
    ax.set(title="Simulated moments", xlabel="episode")
    ax.grid(alpha=0.25)
    if moment_columns:
        ax.legend(fontsize=8)

    for variable, ax in zip(("Hatc", "LnK"), panels[4:]):
        subset = macro_metrics[macro_metrics["variable"] == variable] if not macro_metrics.empty else macro_metrics
        if not subset.empty:
            row = subset.iloc[0]
            ax.axis("off")
            ax.text(
                0.05, 0.9,
                f"{variable} macro fit\nR2: {row['r2']:.4g}\nCorr: {row['pearson_corr']:.4g}\n"
                f"Slope: {row['slope']:.4g}\nRMSE: {row['rmse']:.4g}",
                va="top", fontsize=12,
            )
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{variable} macro fit\nN/A", ha="center", va="center")
    fig.suptitle("Stage convergence report")
    _save(fig, path)
