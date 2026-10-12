from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.entry_macro_progress import (
    generate_entry_macro_progress_report,
    is_entry_macro_plot_milestone,
    load_realized_macro,
)


def _macro_frame(episode: int) -> pd.DataFrame:
    rows = []
    for path in range(2):
        for time in range(3):
            branches = (-1,) if time == 0 else (0, 1)
            for branch in branches:
                start_k = 100.0 + episode + path
                delta_incumbent = 10.0 + time
                old_exit = 2.0
                entry = 5.0
                end_k = start_k + delta_incumbent - old_exit + entry
                rows.append(
                    {
                        "path": path,
                        "t": time,
                        "branch": branch,
                        "n_firms": 100 + 10 * episode + time + (1000 if branch == 1 else 0),
                        "K": end_k,
                        "C": 20.0 + time,
                        "Hatc": 0.1 * time,
                        "LnK": np.log(end_k),
                        "M": 0.98,
                        "potential_count": 10.0,
                        "accepted_count": 2.0 + time,
                        "acceptance_rate": (2.0 + time) / 10.0,
                        "I_entry": 1.0,
                        "K_entry_gross": entry,
                        "K_entry_surviving": entry,
                        "K_entry_endpoint": entry,
                        "K_exit_old": old_exit,
                        "deltaK_incumbent": delta_incumbent,
                        "K_decomposition_start": start_k,
                        "K_decomposition_end": end_k,
                        "capital_accounting_residual": 0.0,
                        "resource_accounting_residual": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _write_macro(run_root: Path, episode: int) -> Path:
    output = run_root / "data" / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    mode = "mode0" if episode == 0 else "modeb"
    path = output / f"ep{episode}_stage_{mode}_macro.pkl"
    _macro_frame(episode).to_pickle(path)
    return path


def test_milestones_for_ten_episode_run() -> None:
    milestones = [
        offset
        for offset in range(10)
        if is_entry_macro_plot_milestone(offset, 10, every=3)
    ]
    assert milestones == [2, 5, 8, 9]


def test_realized_macro_uses_parent_then_main_branch(tmp_path: Path) -> None:
    path = _write_macro(tmp_path, 0)
    realized = load_realized_macro(path)

    assert len(realized) == 6
    assert not realized.duplicated(["path", "t"]).any()
    assert set(realized.loc[realized["t"] == 0, "branch"]) == {-1}
    assert set(realized.loc[realized["t"] >= 1, "branch"]) == {0}
    assert realized["n_firms"].max() < 1000


def test_cumulative_report_contains_all_episodes_and_fixed_outputs(tmp_path: Path) -> None:
    for episode in range(4):
        _write_macro(tmp_path, episode)
    output = tmp_path / "analysis" / "macro_growth_cumulative_ep2"

    metadata = generate_entry_macro_progress_report(
        tmp_path,
        output,
        latest_episode=2,
        growth_start_time=2,
    )

    assert metadata["episodes"] == [0, 1, 2]
    assert metadata["latest_episode"] == 2
    expected_plots = {
        "01_firm_count_and_growth.png",
        "02_entrant_flows.png",
        "03_capital_level_and_growth.png",
        "04_macro_levels.png",
        "05_episode_time_heatmaps.png",
        "06_episode_summary.png",
        "07_nonentry_K_increment.png",
        "08_nonentry_K_growth_rate_from_t2.png",
    }
    assert expected_plots == {path.name for path in output.glob("*.png")}
    assert all((output / name).stat().st_size > 0 for name in expected_plots)

    episode_summary = pd.read_csv(output / "episode_summary.csv")
    assert episode_summary["episode"].tolist() == [0, 1, 2]
    growth = pd.read_csv(output / "nonentry_K_growth_rate_from_t2_by_time.csv")
    assert growth["t"].min() == 2

    expected_growth = (12.0 - 2.0) / 100.0
    ep0_t2 = growth.loc[(growth["episode"] == 0) & (growth["t"] == 2)].iloc[0]
    assert ep0_t2["gK_nonentry_mean"] == pytest.approx(
        np.mean([(12.0 - 2.0) / 100.0, (12.0 - 2.0) / 101.0])
    )
    assert ep0_t2["gK_nonentry_median"] == pytest.approx(
        np.median([(12.0 - 2.0) / 100.0, (12.0 - 2.0) / 101.0])
    )
    assert expected_growth == pytest.approx(0.1)

    stored_metadata = json.loads((output / "metadata.json").read_text())
    assert stored_metadata["growth_start_time"] == 2
    assert stored_metadata["nonentry_growth_definition"] == (
        "(deltaK_incumbent - K_exit_old) / K_decomposition_start"
    )
