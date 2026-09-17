from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.convergence_artifacts import (
    SurfaceData,
    discover_episode_dirs,
    exact_surface_alignment,
    validate_grid_comparability,
)
from evaluation.convergence_metrics import compute_shift_metrics, function_drift_metrics


ROOT = Path(__file__).resolve().parents[1]


def _metadata(*, z_points: int = 3) -> dict:
    return {
        "grid": {
            "b_min": 0.0, "b_max": 1.0, "b_points": 2,
            "z_min": -1.0, "z_max": 1.0, "z_points": z_points,
            "i_min": 0.0, "i_max": 1.0, "i_points": 3, "eta": 0.0,
        },
        "reference_state": {
            "x": 0.0, "hatcf": -2.0, "lnkf": 4.0,
            "hatc_cal": -2.1, "lnk_cal": 4.1,
            "i_low": 0.0, "i_mid": 0.5, "i_high": 1.0, "eta": 0.0,
        },
        "m_mode": "train",
        "m_clamp_bounds": {"min": 0.7, "max": 1.3},
    }


def test_episode_discovery_uses_numeric_order(tmp_path):
    for episode in (10, 2, 1):
        directory = tmp_path / f"ep{episode}_eta_matrix_20260101"
        directory.mkdir()
        (directory / "metadata.json").write_text(json.dumps(_metadata()), encoding="utf-8")
    cases, warnings = discover_episode_dirs(tmp_path)
    assert [case.episode for case in cases] == [1, 2, 10]
    assert warnings == []


def test_function_drift_identity_is_zero():
    surface = SurfaceData(
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        b=np.array([0.0, 1.0]), z=np.array([-1.0, 1.0]), path=Path("same.csv"),
    )
    metrics = function_drift_metrics(surface, surface)
    assert metrics["max_abs_diff"] == 0.0
    assert metrics["mean_abs_diff"] == 0.0
    assert metrics["rmse_diff"] == 0.0


def test_grid_mismatch_is_detected():
    left = SurfaceData(
        values=np.zeros((2, 2)), b=np.array([0.0, 1.0]), z=np.array([-1.0, 1.0]),
        path=Path("left.csv"),
    )
    right = SurfaceData(
        values=np.zeros((2, 2)), b=np.array([0.0, 0.5]), z=np.array([-1.0, 1.0]),
        path=Path("right.csv"),
    )
    aligned, reason = exact_surface_alignment(left, right)
    assert aligned is False
    assert reason == "b-grid mismatch"
    with pytest.raises(ValueError, match="b-grid mismatch"):
        function_drift_metrics(left, right)

    comparable, differences = validate_grid_comparability(_metadata(), _metadata(z_points=4))
    assert comparable is False
    assert any("grid.z_points" in item for item in differences)


def test_shift_metric_recovers_forecast_t_equals_calculated_t_minus_1():
    rows = []
    for path in (0, 1):
        for t in range(8):
            rows.append({
                "path": path,
                "t": t,
                "calculated": float(10 * path + t),
                "forecast": float(10 * path + t - 1),
            })
    frame = pd.DataFrame(rows)
    metrics = compute_shift_metrics(
        frame, forecast_col="forecast", calculated_col="calculated", shifts=[-2, -1, 0, 1, 2]
    )
    best = metrics.loc[metrics["rmse"].idxmin()]
    assert int(best["shift"]) == -1
    assert best["rmse"] == pytest.approx(0.0)


def test_missing_simulation_and_evaluator_artifacts_do_not_abort_report(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    output = tmp_path / "report"
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "experiments" / "build_convergence_report.py"),
            "--run-root", str(run_root), "--output-dir", str(output),
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "README.md").is_file()
    assert (output / "missing_artifacts.md").is_file()
    missing = (output / "missing_artifacts.md").read_text(encoding="utf-8")
    assert "Episode-level firm simulation artifacts unavailable" in missing
    assert "final macro simulation absent" in missing
    assert (output / "stage_report_dashboard.png").is_file()


def _write_surface(path: Path, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        np.arange(6, dtype=np.float64).reshape(2, 3) + offset,
        index=[0.0, 1.0], columns=[-1.0, 0.0, 1.0],
    )
    frame.index.name = "b"
    frame.columns.name = "z"
    frame.to_csv(path)


def test_synthetic_cross_episode_report_writes_nonempty_csvs_and_plots(tmp_path):
    run_root = tmp_path / "run"
    evaluator_root = run_root / "data" / "outputs" / "firm_checkpoint_evaluator"
    for episode, offset in ((2, 0.0), (10, 0.25)):
        case = evaluator_root / f"ep{episode}_eta_matrix_20260101"
        case.mkdir(parents=True)
        (case / "metadata.json").write_text(json.dumps(_metadata()), encoding="utf-8")
        for eta in (0, 1):
            eta_dir = case / f"eta{eta}"
            eta_metadata = _metadata()
            eta_metadata["grid"]["eta"] = float(eta)
            eta_metadata["reference_state"]["eta"] = float(eta)
            eta_dir.mkdir()
            (eta_dir / "metadata.json").write_text(json.dumps(eta_metadata), encoding="utf-8")
            for relative in (
                "q/Q.csv", "value/P.csv", "default/bar_z.csv", "bp/bp_raw.csv",
                "bellman/R0_signed.csv", "bellman/RI_signed.csv",
            ):
                _write_surface(eta_dir / relative, offset)

        firm = pd.DataFrame({
            "path": [0, 0, 1], "t": [0, 1, 0], "branch": [0, 0, 0],
            "b": [0.2, 0.3, 0.4], "z": [-0.5, 0.0, 0.5], "bp": [0.25, 0.35, 0.45],
            "M": [1.0, 0.9, 1.1], "entry": [1, 0, 1],
        })
        firm.to_pickle(run_root / "data" / "outputs" / f"ep{episode}_stage_modeb.pkl")

    macro_rows = []
    for path_id in (0, 1):
        for t in range(4):
            macro_rows.append({
                "path": path_id, "t": t, "branch": -1,
                "Hatc": -2.0 + 0.1 * t, "hatcf": -2.05 + 0.1 * t,
                "LnK": 4.0 + 0.05 * t, "lnkf": 4.02 + 0.05 * t,
            })
    pd.DataFrame(macro_rows).to_pickle(
        run_root / "data" / "outputs" / "final_simulate_macro.pkl"
    )

    output = tmp_path / "report"
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "experiments" / "build_convergence_report.py"),
            "--run-root", str(run_root), "--output-dir", str(output), "--eta", "both",
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    drift = pd.read_csv(output / "function_drift" / "eta0_function_drift.csv")
    assert not drift.empty
    assert set(drift["surface"]) == {"Q", "P", "bar_z", "bp"}
    assert (output / "function_drift" / "function_drift_dashboard.png").is_file()
    bellman = pd.read_csv(output / "bellman_convergence" / "bellman_residual_by_episode.csv")
    assert not bellman.empty
    assert set(bellman["equation"]) == {"P0", "PI"}
    assert (output / "simulated_distribution" / "ep10_bz_joint.png").is_file()
    macro = pd.read_csv(output / "macro" / "macro_forecast_metrics.csv")
    assert not macro.empty
    assert (output / "macro" / "hatc_forecast_vs_cal.png").is_file()
    summary = pd.read_csv(output / "summary_metrics.csv")
    assert not summary.empty
