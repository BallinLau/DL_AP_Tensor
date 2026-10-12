"""Cumulative entry/macro diagnostics generated during multi-episode training."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CORE_COLUMNS = (
    "path",
    "t",
    "branch",
    "n_firms",
    "K",
    "C",
    "Hatc",
    "LnK",
    "M",
)
ENTRY_COLUMNS = (
    "potential_count",
    "accepted_count",
    "acceptance_rate",
    "I_entry",
    "K_entry_gross",
    "K_entry_surviving",
    "K_entry_endpoint",
    "K_exit_old",
    "deltaK_incumbent",
    "K_decomposition_start",
    "K_decomposition_end",
    "capital_accounting_residual",
    "resource_accounting_residual",
)


def episode_from_path(path: Path) -> int:
    match = re.search(r"ep(\d+)_", path.name)
    if not match:
        raise ValueError(f"Cannot parse episode from {path}")
    return int(match.group(1))


def is_entry_macro_plot_milestone(
    episode_offset: int,
    n_episodes: int,
    every: int = 3,
) -> bool:
    """Return whether this completed episode should trigger a cumulative report."""
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    if every <= 0:
        raise ValueError("every must be positive")
    if not 0 <= episode_offset < n_episodes:
        raise ValueError("episode_offset must be within the current run")
    completed = episode_offset + 1
    return completed % every == 0 or completed == n_episodes


def _finite(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values[np.isfinite(values)]


def _quantile(series: pd.Series, q: float) -> float:
    values = _finite(series)
    return float(values.quantile(q)) if len(values) else math.nan


def _stats(series: pd.Series) -> dict[str, float]:
    values = _finite(series)
    if not len(values):
        return {"mean": math.nan, "p10": math.nan, "median": math.nan, "p90": math.nan}
    return {
        "mean": float(values.mean()),
        "p10": float(values.quantile(0.10)),
        "median": float(values.quantile(0.50)),
        "p90": float(values.quantile(0.90)),
    }


def load_realized_macro(path: Path) -> pd.DataFrame:
    """Load one episode and retain the unique state used to advance each path."""
    frame = pd.read_pickle(path)
    missing = sorted(set(CORE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    realized = pd.concat(
        [
            frame.loc[(frame["t"] == 0) & (frame["branch"] == -1)],
            frame.loc[(frame["t"] >= 1) & (frame["branch"] == 0)],
        ],
        ignore_index=True,
    )
    duplicated = realized.duplicated(["path", "t"], keep=False)
    if duplicated.any():
        sample = realized.loc[duplicated, ["path", "t", "branch"]].head().to_dict("records")
        raise ValueError(f"Duplicate realized macro states in {path}: {sample}")

    keep = list(CORE_COLUMNS)
    for column in ENTRY_COLUMNS:
        if column not in realized:
            realized[column] = np.nan
        keep.append(column)
    realized = realized.loc[:, keep].sort_values(["path", "t"]).reset_index(drop=True)
    realized["episode"] = episode_from_path(path)

    grouped = realized.groupby("path", sort=False)
    realized["firm_growth_abs"] = grouped["n_firms"].diff()
    realized["firm_log_growth"] = np.log(realized["n_firms"].clip(lower=1)) - np.log(
        grouped["n_firms"].shift(1).clip(lower=1)
    )
    realized["K_growth_rate"] = grouped["K"].pct_change(fill_method=None)
    realized["accepted_cumulative"] = grouped["accepted_count"].cumsum()
    realized["potential_cumulative"] = grouped["potential_count"].cumsum()
    return realized


def discover_macro_files(run_root: Path, latest_episode: int | None = None) -> list[Path]:
    files = sorted(
        run_root.glob("data/outputs/ep*_macro.pkl"),
        key=episode_from_path,
    )
    if latest_episode is not None:
        files = [path for path in files if episode_from_path(path) <= latest_episode]
    if not files:
        raise FileNotFoundError(f"No episode macro files found under {run_root}")
    return files


def summarize_time(realized: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    columns = (
        "n_firms",
        "firm_growth_abs",
        "firm_log_growth",
        "K",
        "K_growth_rate",
        "C",
        "Hatc",
        "LnK",
        "M",
        "potential_count",
        "accepted_count",
        "accepted_cumulative",
        "I_entry",
        "K_entry_gross",
        "K_entry_surviving",
        "K_entry_endpoint",
        "K_exit_old",
        "deltaK_incumbent",
        "resource_accounting_residual",
    )
    for (episode, time), group in realized.groupby(["episode", "t"], sort=True):
        row: dict[str, float | int] = {
            "episode": int(episode),
            "t": int(time),
            "n_paths": int(group["path"].nunique()),
        }
        for column in columns:
            values = _finite(group[column])
            row[f"{column}_mean"] = float(values.mean()) if len(values) else math.nan
            row[f"{column}_p10"] = _quantile(group[column], 0.10)
            row[f"{column}_median"] = _quantile(group[column], 0.50)
            row[f"{column}_p90"] = _quantile(group[column], 0.90)
            row[f"{column}_p99"] = _quantile(group[column], 0.99)
            row[f"{column}_max"] = float(values.max()) if len(values) else math.nan
        potential = pd.to_numeric(group["potential_count"], errors="coerce").fillna(0).sum()
        accepted = pd.to_numeric(group["accepted_count"], errors="coerce").fillna(0).sum()
        row["acceptance_rate_weighted"] = float(accepted / potential) if potential > 0 else math.nan
        row["potential_count_sum"] = float(potential)
        row["accepted_count_sum"] = float(accepted)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_episode(realized: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for episode, group in realized.groupby("episode", sort=True):
        first = group.loc[group["t"] == group["t"].min()]
        last = group.loc[group["t"] == group["t"].max()]
        path_totals = group.groupby("path", sort=False).agg(
            accepted_cumulative=("accepted_count", "sum"),
            potential_cumulative=("potential_count", "sum"),
        )
        total_potential = float(path_totals["potential_cumulative"].sum())
        total_accepted = float(path_totals["accepted_cumulative"].sum())
        initial_firms = _quantile(first["n_firms"], 0.50)
        initial_k = _quantile(first["K"], 0.50)
        final_firms = _quantile(last["n_firms"], 0.50)
        final_k = _quantile(last["K"], 0.50)
        rows.append(
            {
                "episode": int(episode),
                "t_max": int(group["t"].max()),
                "n_paths": int(group["path"].nunique()),
                "n_firms_initial_median": initial_firms,
                "n_firms_final_mean": float(last["n_firms"].mean()),
                "n_firms_final_median": final_firms,
                "n_firms_final_p90": _quantile(last["n_firms"], 0.90),
                "n_firms_final_p99": _quantile(last["n_firms"], 0.99),
                "n_firms_max": float(group["n_firms"].max()),
                "firm_count_median_ratio": float(final_firms / initial_firms),
                "K_initial_median": initial_k,
                "K_final_mean": float(last["K"].mean()),
                "K_final_median": final_k,
                "K_final_p90": _quantile(last["K"], 0.90),
                "K_median_ratio": float(final_k / initial_k),
                "C_final_median": _quantile(last["C"], 0.50),
                "Hatc_final_median": _quantile(last["Hatc"], 0.50),
                "LnK_final_median": _quantile(last["LnK"], 0.50),
                "M_final_median": _quantile(last["M"], 0.50),
                "accepted_per_path_mean": float(path_totals["accepted_cumulative"].mean()),
                "accepted_per_path_median": _quantile(path_totals["accepted_cumulative"], 0.50),
                "accepted_per_path_p90": _quantile(path_totals["accepted_cumulative"], 0.90),
                "potential_total": total_potential,
                "accepted_total": total_accepted,
                "acceptance_rate_weighted": (
                    float(total_accepted / total_potential) if total_potential else math.nan
                ),
                "resource_residual_abs_max": float(group["resource_accounting_residual"].abs().max()),
            }
        )
    return pd.DataFrame(rows)


def build_capital_decomposition(
    realized: pd.DataFrame,
    growth_start_time: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "episode",
        "path",
        "t",
        "deltaK_incumbent",
        "K_entry_endpoint",
        "K_exit_old",
        "K_decomposition_start",
        "K_decomposition_end",
        "capital_accounting_residual",
    }
    missing = sorted(required.difference(realized.columns))
    if missing:
        raise ValueError(f"Missing capital decomposition columns: {missing}")
    unavailable = [
        column
        for column in required.difference({"episode", "path", "t"})
        if not _finite(realized[column]).size
    ]
    if unavailable:
        raise ValueError(f"Capital decomposition columns contain no finite data: {sorted(unavailable)}")

    frame = realized.sort_values(["episode", "path", "t"]).copy()
    frame["deltaK_nonentry"] = frame["deltaK_incumbent"] - frame["K_exit_old"]
    frame["deltaK_accounted"] = (
        frame["deltaK_incumbent"] + frame["K_entry_endpoint"] - frame["K_exit_old"]
    )
    grouped = frame.groupby(["episode", "path"], sort=False)
    frame["cum_deltaK_nonentry"] = grouped["deltaK_nonentry"].cumsum()
    frame["cum_deltaK_incumbent"] = grouped["deltaK_incumbent"].cumsum()
    frame["cum_K_exit_old"] = grouped["K_exit_old"].cumsum()
    frame["cum_K_entry_endpoint"] = grouped["K_entry_endpoint"].cumsum()

    increment_rows = []
    for (episode, time), group in frame.groupby(["episode", "t"], sort=True):
        row = {"episode": int(episode), "t": int(time), "n_paths": int(group["path"].nunique())}
        for column in (
            "deltaK_nonentry",
            "deltaK_incumbent",
            "K_exit_old",
            "K_entry_endpoint",
            "deltaK_accounted",
            "cum_deltaK_nonentry",
        ):
            for key, value in _stats(group[column]).items():
                row[f"{column}_{key}"] = value
        increment_rows.append(row)
    increment_time = pd.DataFrame(increment_rows)

    increment_episode_rows = []
    for episode, group in frame.groupby("episode", sort=True):
        last = group.loc[group["t"] == group["t"].max()]
        increment_episode_rows.append(
            {
                "episode": int(episode),
                "cum_nonentry_mean": float(last["cum_deltaK_nonentry"].mean()),
                "cum_nonentry_median": _quantile(last["cum_deltaK_nonentry"], 0.50),
                "cum_incumbent_median": _quantile(last["cum_deltaK_incumbent"], 0.50),
                "cum_old_exit_median": _quantile(last["cum_K_exit_old"], 0.50),
                "cum_entry_median": _quantile(last["cum_K_entry_endpoint"], 0.50),
                "capital_residual_abs_max": float(group["capital_accounting_residual"].abs().max()),
            }
        )
    increment_episode = pd.DataFrame(increment_episode_rows)

    denominator = pd.to_numeric(frame["K_decomposition_start"], errors="coerce")
    valid = np.isfinite(denominator) & (denominator > 0)
    frame["gK_nonentry"] = np.where(
        valid,
        (frame["deltaK_incumbent"] - frame["K_exit_old"]) / denominator,
        np.nan,
    )
    frame["gK_entry_contribution"] = np.where(
        valid,
        frame["K_entry_endpoint"] / denominator,
        np.nan,
    )
    frame["gK_total"] = np.where(
        valid,
        (frame["K_decomposition_end"] - frame["K_decomposition_start"]) / denominator,
        np.nan,
    )
    frame["growth_identity_error"] = (
        frame["gK_total"] - frame["gK_nonentry"] - frame["gK_entry_contribution"]
    )
    growth_frame = frame.loc[frame["t"] >= int(growth_start_time)].copy()

    growth_rows = []
    for (episode, time), group in growth_frame.groupby(["episode", "t"], sort=True):
        row = {"episode": int(episode), "t": int(time)}
        for column in ("gK_nonentry", "gK_entry_contribution", "gK_total"):
            for key, value in _stats(group[column]).items():
                row[f"{column}_{key}"] = value
        row["valid_n"] = int(_finite(group["gK_nonentry"]).shape[0])
        growth_rows.append(row)
    growth_time = pd.DataFrame(growth_rows)

    growth_episode_rows = []
    for episode, group in growth_frame.groupby("episode", sort=True):
        row = {"episode": int(episode), "valid_n": int(_finite(group["gK_nonentry"]).shape[0])}
        for column in ("gK_nonentry", "gK_entry_contribution", "gK_total"):
            for key, value in _stats(group[column]).items():
                row[f"{column}_{key}"] = value
        identity = _finite(group["growth_identity_error"])
        row["growth_identity_abs_max"] = float(identity.abs().max()) if len(identity) else math.nan
        growth_episode_rows.append(row)
    growth_episode = pd.DataFrame(growth_episode_rows)
    return increment_time, increment_episode, growth_time, growth_episode


def _episode_colors(episodes: Iterable[int]) -> dict[int, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {episode: cmap(index % 10) for index, episode in enumerate(episodes)}


def _plot_firm_counts(time: pd.DataFrame, output: Path) -> None:
    episodes = sorted(time["episode"].unique())
    colors = _episode_colors(episodes)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for episode in episodes:
        group = time.loc[time["episode"] == episode]
        axes[0].plot(group["t"], group["n_firms_median"], marker="o", label=f"Episode {episode}", color=colors[episode])
        axes[0].fill_between(group["t"], group["n_firms_p10"], group["n_firms_p90"], alpha=0.16, color=colors[episode])
        axes[0].plot(group["t"], group["n_firms_max"], linestyle="--", alpha=0.65, color=colors[episode])
        axes[1].plot(group["t"], group["firm_log_growth_median"], marker="o", label=f"Episode {episode}", color=colors[episode])
        axes[1].fill_between(group["t"], group["firm_log_growth_p10"], group["firm_log_growth_p90"], alpha=0.16, color=colors[episode])
    axes[0].set_yscale("log")
    axes[0].set_title("Firm count along realized main branch")
    axes[0].set_ylabel("Number of firms (log scale)")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Firm-count log growth")
    axes[1].set_ylabel("log(N_t / N_{t-1})")
    for axis in axes:
        axis.set_xlabel("Simulation time")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_entry(time: pd.DataFrame, output: Path) -> None:
    episodes = sorted(time["episode"].unique())
    colors = _episode_colors(episodes)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for episode in episodes:
        group = time.loc[time["episode"] == episode]
        axes[0].plot(group["t"], group["potential_count_mean"], linestyle="--", color=colors[episode], label=f"Potential ep{episode}")
        axes[0].plot(group["t"], group["accepted_count_mean"], marker="o", color=colors[episode], label=f"Accepted ep{episode}")
        axes[1].plot(group["t"], group["acceptance_rate_weighted"], marker="o", color=colors[episode], label=f"Episode {episode}")
        axes[2].plot(group["t"], group["accepted_cumulative_median"], marker="o", color=colors[episode], label=f"Episode {episode}")
        axes[2].fill_between(group["t"], group["accepted_cumulative_p10"], group["accepted_cumulative_p90"], alpha=0.16, color=colors[episode])
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_title("Entrant candidates and accepted entrants")
    axes[0].set_ylabel("Mean count per path")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_title("Weighted entrant acceptance rate")
    axes[1].set_ylabel("accepted / potential")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(fontsize=8)
    axes[2].set_title("Cumulative accepted entrants per path")
    axes[2].set_ylabel("Median, P10-P90 band")
    axes[2].set_yscale("symlog", linthresh=1)
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel("Simulation time")
        axis.grid(alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_capital(time: pd.DataFrame, output: Path) -> None:
    episodes = sorted(time["episode"].unique())
    colors = _episode_colors(episodes)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for episode in episodes:
        group = time.loc[time["episode"] == episode]
        axes[0].plot(group["t"], group["K_median"], marker="o", color=colors[episode], label=f"Episode {episode}")
        axes[0].fill_between(group["t"], group["K_p10"], group["K_p90"], alpha=0.16, color=colors[episode])
        axes[1].plot(group["t"], 100 * group["K_growth_rate_median"], marker="o", color=colors[episode], label=f"Episode {episode}")
        axes[2].plot(group["t"], group["K_entry_gross_mean"], marker="o", color=colors[episode], label=f"Entry gross ep{episode}")
        axes[2].plot(group["t"], group["deltaK_incumbent_mean"], linestyle="--", color=colors[episode], label=f"Incumbent dK ep{episode}")
    axes[0].set_title("Aggregate capital K")
    axes[0].set_ylabel("Median, P10-P90 band")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Capital growth")
    axes[1].set_ylabel("Median percent growth")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("Capital flow decomposition")
    axes[2].set_ylabel("Mean across paths")
    for axis in axes:
        axis.set_xlabel("Simulation time")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_macro(time: pd.DataFrame, output: Path) -> None:
    episodes = sorted(time["episode"].unique())
    colors = _episode_colors(episodes)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, (column, title) in zip(
        axes.flat,
        (("C", "Consumption C"), ("Hatc", "Hatc"), ("LnK", "LnK"), ("M", "SDF M")),
    ):
        for episode in episodes:
            group = time.loc[time["episode"] == episode]
            axis.plot(group["t"], group[f"{column}_median"], marker="o", color=colors[episode], label=f"Episode {episode}")
            axis.fill_between(group["t"], group[f"{column}_p10"], group[f"{column}_p90"], alpha=0.16, color=colors[episode])
        axis.set_title(title)
        axis.set_xlabel("Simulation time")
        axis.set_ylabel("Median, P10-P90 band")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_heatmaps(time: pd.DataFrame, output: Path) -> None:
    episodes = sorted(time["episode"].unique())
    times = sorted(time["t"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
    specs = (
        ("n_firms_median", "log10 median firm count", True),
        ("K_median", "Median aggregate capital K", False),
        ("accepted_count_mean", "Mean accepted entrants", False),
    )
    for axis, (column, title, take_log) in zip(axes, specs):
        pivot = time.pivot(index="episode", columns="t", values=column).reindex(index=episodes, columns=times)
        values = pivot.to_numpy(dtype=float)
        if take_log:
            values = np.log10(np.clip(values, 1, None))
        image = axis.imshow(values, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("Simulation time")
        axis.set_ylabel("Episode")
        axis.set_xticks(range(len(times)), times, rotation=90, fontsize=7)
        axis.set_yticks(range(len(episodes)), episodes)
        fig.colorbar(image, ax=axis, shrink=0.85)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_episode_summary(summary: pd.DataFrame, output: Path) -> None:
    x = np.arange(len(summary))
    labels = [f"ep{episode}" for episode in summary["episode"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    axes[0].bar(x - 0.18, summary["n_firms_final_median"], width=0.36, label="Final median")
    axes[0].bar(x + 0.18, summary["n_firms_final_p90"], width=0.36, label="Final P90")
    axes[0].set_yscale("log")
    axes[0].set_title("Final firm count")
    axes[0].legend()
    axes[1].bar(x, summary["accepted_per_path_mean"])
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].set_title("Mean cumulative accepted entrants per path")
    axes[2].bar(x - 0.18, summary["K_initial_median"], width=0.36, label="Initial K")
    axes[2].bar(x + 0.18, summary["K_final_median"], width=0.36, label="Final K")
    axes[2].set_title("Median capital")
    axes[2].legend()
    for axis in axes:
        axis.set_xticks(x, labels, rotation=45)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_nonentry_increment(time: pd.DataFrame, episode: pd.DataFrame, output: Path) -> None:
    episodes = sorted(time["episode"].unique())
    colors = _episode_colors(episodes)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    for episode_id in episodes:
        group = time.loc[time["episode"] == episode_id]
        axes[0].plot(group["t"], group["deltaK_nonentry_median"], marker="o", ms=3, color=colors[episode_id], label=f"Episode {episode_id}")
        axes[0].fill_between(group["t"], group["deltaK_nonentry_p10"], group["deltaK_nonentry_p90"], color=colors[episode_id], alpha=0.10)
        axes[1].plot(group["t"], group["cum_deltaK_nonentry_median"], marker="o", ms=3, color=colors[episode_id], label=f"Episode {episode_id}")
    axes[0].set_title("Net capital change excluding entrants")
    axes[0].set_ylabel("Median dK incumbent - old-firm exits")
    axes[1].set_title("Cumulative non-entry capital change")
    axes[1].set_ylabel("Median cumulative non-entry dK")
    x = np.arange(len(episode))
    width = 0.25
    axes[2].bar(x - width, episode["cum_incumbent_median"], width, label="Incumbent intensive change")
    axes[2].bar(x, -episode["cum_old_exit_median"], width, label="Old-firm exits")
    axes[2].bar(x + width, episode["cum_entry_median"], width, label="Entrant capital (excluded)", color="0.65")
    axes[2].set_title("Cumulative capital-flow decomposition")
    axes[2].set_ylabel("Median cumulative contribution")
    axes[2].set_xticks(x, [f"ep{value}" for value in episode["episode"]], rotation=45)
    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Simulation time" if axis is not axes[2] else "Episode")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_nonentry_growth(
    time: pd.DataFrame,
    episode: pd.DataFrame,
    output: Path,
    growth_start_time: int,
) -> None:
    episodes = sorted(time["episode"].unique())
    colors = _episode_colors(episodes)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    for episode_id in episodes:
        group = time.loc[time["episode"] == episode_id]
        axes[0].plot(group["t"], 100 * group["gK_nonentry_median"], marker="o", ms=3, color=colors[episode_id], label=f"Episode {episode_id}")
        axes[0].fill_between(group["t"], 100 * group["gK_nonentry_p10"], 100 * group["gK_nonentry_p90"], color=colors[episode_id], alpha=0.10)
    axes[0].set_title(f"Non-entry capital growth rate (t >= {growth_start_time})")
    axes[0].set_ylabel("Median % of starting capital (P10-P90)")
    axes[0].set_yscale("symlog", linthresh=5)
    x = np.arange(len(episode))
    axes[1].bar(x - 0.18, 100 * episode["gK_nonentry_median"], 0.36, label="Median")
    axes[1].bar(x + 0.18, 100 * episode["gK_nonentry_mean"], 0.36, label="Mean")
    axes[1].set_title("Episode-level non-entry growth rate")
    axes[1].set_ylabel("Percent per transition")
    axes[1].set_yscale("symlog", linthresh=5)
    axes[2].bar(x, 100 * episode["gK_nonentry_mean"], label="Non-entry")
    axes[2].bar(x, 100 * episode["gK_entry_contribution_mean"], bottom=100 * episode["gK_nonentry_mean"], label="Entrant contribution", color="0.65")
    axes[2].plot(x, 100 * episode["gK_total_mean"], "ko--", label="Total growth")
    axes[2].set_title("Mean growth-rate decomposition")
    axes[2].set_ylabel("Percent per transition")
    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.8)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    axes[0].set_xlabel("Simulation time")
    for axis in axes[1:]:
        axis.set_xlabel("Episode")
        axis.set_xticks(x, [f"ep{value}" for value in episode["episode"]], rotation=45)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def generate_entry_macro_progress_report(
    run_root: Path,
    output_dir: Path,
    *,
    latest_episode: int | None = None,
    growth_start_time: int = 2,
) -> dict[str, object]:
    """Generate the fixed cumulative eight-figure entry/macro report."""
    run_root = Path(run_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    macro_files = discover_macro_files(run_root, latest_episode=latest_episode)
    realized = pd.concat([load_realized_macro(path) for path in macro_files], ignore_index=True)
    time_summary = summarize_time(realized)
    episode_summary = summarize_episode(realized)
    increment_time, increment_episode, growth_time, growth_episode = build_capital_decomposition(
        realized,
        growth_start_time,
    )

    time_summary.to_csv(output_dir / "episode_time_summary.csv", index=False)
    episode_summary.to_csv(output_dir / "episode_summary.csv", index=False)
    increment_time.to_csv(output_dir / "nonentry_K_increment_by_time.csv", index=False)
    increment_episode.to_csv(output_dir / "nonentry_K_increment_by_episode.csv", index=False)
    growth_time.to_csv(
        output_dir / f"nonentry_K_growth_rate_from_t{growth_start_time}_by_time.csv",
        index=False,
    )
    growth_episode.to_csv(
        output_dir / f"nonentry_K_growth_rate_from_t{growth_start_time}_by_episode.csv",
        index=False,
    )

    plot_paths = (
        output_dir / "01_firm_count_and_growth.png",
        output_dir / "02_entrant_flows.png",
        output_dir / "03_capital_level_and_growth.png",
        output_dir / "04_macro_levels.png",
        output_dir / "05_episode_time_heatmaps.png",
        output_dir / "06_episode_summary.png",
        output_dir / "07_nonentry_K_increment.png",
        output_dir / f"08_nonentry_K_growth_rate_from_t{growth_start_time}.png",
    )
    _plot_firm_counts(time_summary, plot_paths[0])
    _plot_entry(time_summary, plot_paths[1])
    _plot_capital(time_summary, plot_paths[2])
    _plot_macro(time_summary, plot_paths[3])
    _plot_heatmaps(time_summary, plot_paths[4])
    _plot_episode_summary(episode_summary, plot_paths[5])
    _plot_nonentry_increment(increment_time, increment_episode, plot_paths[6])
    _plot_nonentry_growth(growth_time, growth_episode, plot_paths[7], growth_start_time)

    episodes = [int(value) for value in sorted(realized["episode"].unique())]
    metadata: dict[str, object] = {
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "source_files": [str(path) for path in macro_files],
        "episodes": episodes,
        "latest_episode": episodes[-1],
        "selection": "t=0 branch=-1 plus t>=1 branch=0 (realized main branch)",
        "milestone_rule": "every N completed episodes and the final requested episode",
        "growth_start_time": int(growth_start_time),
        "nonentry_increment_definition": "deltaK_incumbent - K_exit_old",
        "nonentry_growth_definition": "(deltaK_incumbent - K_exit_old) / K_decomposition_start",
        "entry_growth_definition": "K_entry_endpoint / K_decomposition_start",
        "total_growth_definition": "(K_decomposition_end - K_decomposition_start) / K_decomposition_start",
        "plot_files": [path.name for path in plot_paths],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata
