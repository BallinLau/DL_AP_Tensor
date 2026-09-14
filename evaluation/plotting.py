from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_heatmap(
    values: np.ndarray,
    b_values: np.ndarray,
    z_values: np.ndarray,
    path: str | Path,
    *,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.pcolormesh(b_values, z_values, values.T, shading="auto", cmap=cmap)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    ax.set(xlabel="b", ylabel="z", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_default_boundary(
    boundary: pd.DataFrame,
    path: str | Path,
) -> None:
    observed = boundary[boundary["boundary_status"] == "single_crossing"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(observed["b"], observed["z_default"], color="black", linewidth=1.5)
    ax.scatter(observed["b"], observed["z_default"], color="black", s=10)
    ax.set(xlabel="b", ylabel="z_default", title="Default boundary: Phat = 0")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_b_slices(
    surface: np.ndarray,
    b_values: np.ndarray,
    z_values: np.ndarray,
    path: str | Path,
    *,
    title: str,
    ylabel: str,
    z_positions: Sequence[int] | None = None,
) -> None:
    positions = list(z_positions or [0, len(z_values) // 2, len(z_values) - 1])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for pos in positions:
        ax.plot(b_values, surface[:, pos], label=f"z={z_values[pos]:.4g}")
    ax.set(xlabel="b", ylabel=ylabel, title=title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_i_star_slices(
    i_star: np.ndarray,
    b_values: np.ndarray,
    z_values: np.ndarray,
    path: str | Path,
) -> None:
    positions = [0, len(b_values) // 2, len(b_values) - 1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for pos in positions:
        ax.plot(z_values, i_star[pos], label=f"b={b_values[pos]:.4g}")
    ax.set(xlabel="z", ylabel="i_star", title="Value-implied investment cutoff")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_q_peak(peak: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(peak["z"], peak["b_peak"], color="tab:blue")
    ax.set(xlabel="z", ylabel="argmax_b Q(b,z)", title="Debt-value peak by z")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_objective_slice(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(frame["bp_candidate"], frame["value"], label="objective", linewidth=2)
    axes[0].plot(frame["bp_candidate"], frame["cashflow"], label="cash flow")
    axes[0].plot(frame["bp_candidate"], frame["continuation"], label="continuation")
    axes[0].axvline(float(frame["bp_star"].iloc[0]), color="black", linestyle="--", label="bp_star")
    axes[0].set_ylabel("value component")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(frame["bp_candidate"], frame["default_mean"], label="default")
    axes[1].plot(frame["bp_candidate"], frame["q_issue"], label="Q issue")
    axes[1].plot(frame["bp_candidate"], frame["p_child_mean"], label="P child")
    axes[1].set(xlabel="candidate bp", ylabel="diagnostic")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_surface_csvs(
    output_dir: str | Path,
    surfaces: Mapping[str, np.ndarray],
    b_values: np.ndarray,
    z_values: np.ndarray,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, values in surfaces.items():
        if not isinstance(values, np.ndarray) or values.shape != (len(b_values), len(z_values)):
            continue
        frame = pd.DataFrame(values, index=b_values, columns=z_values)
        frame.index.name = "b"
        frame.columns.name = "z"
        frame.to_csv(output / f"{name}.csv")
