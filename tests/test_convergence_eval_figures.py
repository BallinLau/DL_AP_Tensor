"""Tests for the convergence-evaluator figure renderer.

Covers the Prompt figure contract for ``evaluation/convergence/figures.py``:

* every ``ok`` figure writes a non-empty PNG under ``figures/``;
* a figure whose data is missing is **skipped** (stable reason code) and writes
  **no** file -- no blank "successful" plot is ever fabricated;
* the manifest fields are complete (non-empty Chinese caption, ``n_samples``);
* the unified ``b``/``z`` support and colour limits are applied to every figure
  (checked through ``FigureRecord.meta`` and the recorded ranges);
* rendering the same input twice is byte-identical;
* 3D figures render headless on the Agg backend.

Everything uses hand-built synthetic result dataclasses (no checkpoint, no
network, no GPU).
"""

import hashlib
import os
import sys
import tempfile
from pathlib import Path

# MPLCONFIGDIR must be writable *before* matplotlib is imported (the slurm
# script does not set it).  ``setdefault`` keeps an operator-provided value.
os.environ.setdefault(
    "MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_conv_figures_")
)

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from evaluation.convergence.figures import (
    CANONICAL_3D_NAMES,
    CAPTIONS,
    FIGURE_FILENAMES,
    FIGURE_IDS,
    FigureRecord,
    FigureStyle,
    FiguresManifest,
    render_all_figures,
    render_f01,
    render_f11_waterfall,
    render_f12,
    render_m01,
    render_p04,
    render_v04,
    resolve_style,
)
from evaluation.convergence.firm_bellman import BellmanEvalResult
from evaluation.convergence.firm_policy import PolicyEvalResult
from evaluation.convergence.macro_sdf import MacroSdfEvalResult
from evaluation.convergence.sim_eval import SimEvalResult


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# synthetic geometry (fixed b/z grid -> shared coordinate support)
# ---------------------------------------------------------------------------

B_VALUES = np.linspace(0.0, 1.0, 9)
Z_VALUES = np.linspace(-1.0, 1.0, 7)


def _grid_shape():
    return (B_VALUES.size, Z_VALUES.size)


def _field(offset: float = 0.0, slope_b: float = -1.0, slope_z: float = 1.0):
    """Deterministic (b, z) surface used for every synthetic function."""
    mesh_b, mesh_z = np.meshgrid(B_VALUES, Z_VALUES, indexing="ij")
    return offset + slope_b * mesh_b + slope_z * mesh_z + 0.05 * mesh_b * mesh_z


def _default_boundary_frame():
    return pd.DataFrame(
        {
            "b": B_VALUES,
            "z_default": 0.2 * B_VALUES - 0.3,
            "boundary_status": ["single_crossing"] * B_VALUES.size,
            "crossing_count": np.ones(B_VALUES.size, dtype=np.int64),
            "crossings_json": ['[0.0]'] * B_VALUES.size,
            "crossing_intervals_json": ['[]'] * B_VALUES.size,
            "eta_view": np.ones(B_VALUES.size),
        }
    )


def _hard_soft_frame():
    return pd.DataFrame(
        {
            "b": B_VALUES,
            "z_hard_phat0": 0.2 * B_VALUES - 0.3,
            "hard_status": ["single_crossing"] * B_VALUES.size,
            "hard_crossing_count": np.ones(B_VALUES.size, dtype=np.int64),
            "z_soft_barz0p5": 0.2 * B_VALUES - 0.15,
            "soft_status": ["single_crossing"] * B_VALUES.size,
            "soft_crossing_count": np.ones(B_VALUES.size, dtype=np.int64),
        }
    )


def _investment_boundary_frame():
    rows = []
    for b_value in B_VALUES:
        for z_value in Z_VALUES:
            rows.append(
                {
                    "b": float(b_value),
                    "z": float(z_value),
                    "eta_view": 1.0,
                    "i_star": float(0.3 + 0.1 * z_value),
                    "investment_status": "single_crossing",
                    "investment_status_label": "single_crossing",
                    "crossing_count": 1,
                    "crossings_json": "[]",
                    "crossing_intervals_json": "[]",
                    "survival": True,
                }
            )
    return pd.DataFrame(rows)


def _policy_rows_frame(n_rows: int = 40, shift: float = 0.0):
    rng = np.random.default_rng(12345)
    b = rng.uniform(0.0, 1.0, n_rows)
    z = rng.uniform(-1.0, 1.0, n_rows)
    eta = (rng.uniform(0.0, 1.0, n_rows) > 0.5).astype(np.float64)
    bp_star = 0.4 + 0.2 * z + shift
    b_next = eta * bp_star + (1.0 - eta) * b
    phat = 0.5 - b + 0.4 * z
    return pd.DataFrame(
        {
            "row_index": np.arange(n_rows, dtype=np.int64),
            "b": b,
            "z": z,
            "eta": eta,
            "refi_active": eta,
            "bp0_star": bp_star,
            "bpi_star": bp_star + 0.1,
            "bp_star_mix": bp_star,
            "b_next_p0": b_next,
            "b_next_pi": b_next + 0.05,
            "b_next_mix": b_next,
            "bp0_head": bp_star + 0.01 * shift,
            "bpI_head": bp_star + 0.11,
            "bar_i_model": np.full(n_rows, 0.4),
            "bar_i_executed": np.full(n_rows, 0.3),
            "P0": 1.0 - b + 0.5 * z + shift,
            "PI": 1.1 - b + 0.5 * z + shift,
            "P": 1.05 - b + 0.5 * z + shift,
            "Phat": phat,
            "bar_z": 1.0 / (1.0 + np.exp(-phat / 0.2)),
            "q_issue_at_star_p0": 0.8 * b + 0.1 + shift,
            "conditional_default_prob": np.clip(0.5 - 0.3 * z, 0.0, 1.0),
        }
    )


def _objective_slices_frame():
    rows = []
    candidates = np.linspace(0.0, 1.0, 11)
    for row_index in (3, 17):
        for branch in ("p0", "pi"):
            for candidate in candidates:
                rows.append(
                    {
                        "row_index": row_index,
                        "state_label": f"row_{row_index}",
                        "branch": branch,
                        "eta": 1.0,
                        "refi_active": 1.0,
                        "bp_candidate": float(candidate),
                        "bellman_rhs": float(1.0 - (candidate - 0.45) ** 2),
                        "cashflow": float(0.3 - 0.1 * candidate),
                        "continuation": float(0.7 - (candidate - 0.45) ** 2),
                        "q_claim_issue": float(0.2 + 0.3 * candidate),
                        "p_child_mean": float(0.9 - 0.2 * candidate),
                        "default_share_mean": float(0.1 * candidate),
                    }
                )
    return pd.DataFrame(rows)


def make_policy_result(episode: int = 3, shift: float = 0.0) -> PolicyEvalResult:
    surfaces = {}
    for view, tilt in (("eta0", -0.05), ("eta1", 0.0)):
        surfaces[f"{view}_P0"] = _field(1.0 + tilt + shift)
        surfaces[f"{view}_PI"] = _field(1.2 + tilt + shift)
        surfaces[f"{view}_P"] = _field(1.1 + tilt + shift)
        surfaces[f"{view}_Q_claim"] = _field(0.6 + tilt + shift, slope_b=0.8, slope_z=0.2)
        surfaces[f"{view}_q_unit"] = _field(0.4 + tilt + shift, slope_b=-0.2, slope_z=0.1)
        surfaces[f"{view}_Q_effective"] = _field(0.5 + tilt + shift, slope_b=0.5, slope_z=0.1)
        surfaces[f"{view}_recovery"] = _field(0.3 + tilt + shift, slope_b=0.1, slope_z=0.3)
        surfaces[f"{view}_bp0"] = _field(0.5 + tilt + shift, slope_b=-0.4, slope_z=0.3)
        surfaces[f"{view}_bpI"] = _field(0.6 + tilt + shift, slope_b=-0.4, slope_z=0.3)
        surfaces[f"{view}_bar_z"] = np.clip(_field(0.5 + tilt, slope_b=1.0, slope_z=-1.0), 0.0, 1.0)
        surfaces[f"{view}_bar_i_cond"] = np.clip(_field(0.4 + tilt), 0.0, 1.0)
        surfaces[f"{view}_bar_i_eff"] = np.clip(_field(0.3 + tilt), 0.0, 1.0)
        surfaces[f"{view}_Phat"] = _field(0.8 + tilt + shift, slope_b=-1.0, slope_z=0.6)
        surfaces[f"{view}_investment_i_star"] = _field(0.3 + tilt, slope_b=0.0, slope_z=0.2)
        surfaces[f"{view}_default_region"] = (surfaces[f"{view}_Phat"] <= 0.0).astype(np.float64)
        surfaces[f"{view}_survival_prob"] = np.clip(_field(0.9 + tilt), 0.0, 1.0)
    return PolicyEvalResult(
        metrics={"objective_slice_eta0_flatness_max_abs": 1e-6},
        tables={
            "policy_rows": _policy_rows_frame(shift=shift),
            "investment_boundary": _investment_boundary_frame(),
            "default_boundary": _default_boundary_frame(),
            "hard_soft_boundary": _hard_soft_frame(),
            "objective_slices": _objective_slices_frame(),
        },
        surfaces=surfaces,
        diagnostics={
            "episode": episode,
            "policy_flip_tol": 1e-2,
            "bank_id": "bank_c_synthetic",
            "bank_kind": "synthetic-stress",
            "reference_i": 0.25,
        },
    )


def make_bellman_result(episode: int = 3, level: float = 1e-2) -> BellmanEvalResult:
    metrics = {}
    for prefix in (
        "fresh_p0_normalized",
        "fresh_pi_normalized",
        "frozen_p0_normalized",
        "frozen_pi_normalized",
        "fresh_p0_physical",
        "fresh_pi_physical",
    ):
        metrics[f"{prefix}_mae"] = level
        metrics[f"{prefix}_p90"] = level * 2.0
        metrics[f"{prefix}_p99"] = level * 3.0
        metrics[f"{prefix}_max"] = level * 5.0
        metrics[f"{prefix}_n"] = 40.0
        metrics[f"{prefix}_n_finite"] = 40.0
    return BellmanEvalResult(
        metrics=metrics,
        tables={},
        surfaces={},
        diagnostics={"episode": episode, "bank_id": "bank_c_synthetic"},
    )


def make_macro_result(episode: int = 3) -> MacroSdfEvalResult:
    n_rows = 24
    rng = np.random.default_rng(7)
    path = np.repeat(np.arange(4), 6)
    t = np.tile(np.arange(6), 4)
    hatc_t = 0.1 * rng.normal(size=n_rows)
    lnk_t = 1.0 + 0.1 * rng.normal(size=n_rows)
    pairs = pd.DataFrame(
        {
            "path": path,
            "t": t.astype(np.float64),
            "x_t": 0.05 * rng.normal(size=n_rows),
            "x_t1": 0.05 * rng.normal(size=n_rows),
            "hatcf_t": hatc_t + 0.01 * rng.normal(size=n_rows),
            "lnkf_t": lnk_t + 0.01 * rng.normal(size=n_rows),
            "hatc_t": hatc_t,
            "lnk_t": lnk_t,
            "hatc_t1": hatc_t * 0.98,
            "lnk_t1": lnk_t * 1.01,
        }
    )
    anchor = pd.DataFrame(
        {
            "step": np.arange(11, dtype=np.float64),
            "x": np.zeros(11),
            "hatc_pred": 0.1 + 0.002 * np.arange(11),
            "lnk_pred": 1.0 - 0.003 * np.arange(11),
            "hatc_in_training_range": np.ones(11),
            "lnk_in_training_range": np.ones(11),
        }
    )
    recursion_rows = []
    for caliber in ("production_one_step", "autonomous_recursive"):
        for variable in ("hatc", "lnk"):
            for horizon in (1, 2, 5, 10, 20, 50):
                base = 0.01 * horizon * (1.0 if caliber == "autonomous_recursive" else 0.3)
                recursion_rows.append(
                    {
                        "caliber": caliber,
                        "variable": variable,
                        "horizon": float(horizon),
                        "n": 100.0,
                        "mae": base + 0.01,
                        "rmse": base * 1.5 + 0.01,
                        "bias": 0.002 * horizon,
                    }
                )
    sdf_summary = pd.DataFrame(
        {
            "macro_state": ["recession", "normal", "expansion"],
            "n_parents": [30.0, 40.0, 30.0],
            "mean_residual": [-0.01, 0.0, 0.01],
            "se_residual": [0.002, 0.002, 0.002],
            "M_mean": [0.98, 1.0, 1.02],
            "M_mean_se": [0.003, 0.003, 0.003],
        }
    )
    metrics = {
        "fc1_one_step_hatc_mae": 0.02,
        "fc1_one_step_lnk_mae": 0.03,
        "sdf_M_mean": 1.0,
        "sdf_M_mean_se": 0.002,
        "sdf_M_var": 0.001,
        "sdf_M_std": 0.03,
        "sdf_signed_aio_estimate": -0.004,
        "sdf_signed_aio_se": 0.001,
        "sdf_signed_aio_negative_share": 0.45,
        "sdf_population_target_estimate": 0.001,
        "sdf_population_target_se": 0.0005,
        "sdf_population_target_t_stat": 2.0,
        "sdf_g_mean_low": -0.01,
        "sdf_g_mean_high": -0.02,
        "sdf_g_var_high": -0.03,
        "sdf_g_max": 0.0,
        "sdf_nonfinite_ratio": 0.0,
        "sdf_log_R_clip_share": 0.001,
        "sdf_valid_parent_ratio": 1.0,
    }
    metrics["fc1_rec_autonomous_recursive_hatc_drift_slope"] = 0.002
    metrics["fc1_rec_autonomous_recursive_hatc_out_of_training_range_share"] = 0.0
    return MacroSdfEvalResult(
        metrics=metrics,
        tables={
            "fc1_one_step_pairs": pairs,
            "fc1_anchor_recursion_normal": anchor,
            "fc1_recursion_errors": pd.DataFrame(recursion_rows),
            "sdf_macro_state_summary": sdf_summary,
        },
        surfaces={},
        diagnostics={"episode": episode, "bank_id": "bank_d_synthetic"},
    )


def make_sim_result(episode: int = 3) -> SimEvalResult:
    waterfall_rows = []
    totals_rows = []
    for t_from, t_to in ((0, 1), (1, 2), (2, 3)):
        row = {
            "path": 0,
            "t_from": float(t_from),
            "t_to": float(t_to),
            "initial_capital": 100.0 + 5.0 * t_from,
            "incumbent_dk": 3.0,
            "exiting_incumbent_capital": 4.0,
            "entrant_capital": 6.0,
            "same_node_entrant_exit": 0.5,
            "ending_capital": 105.0 + 5.0 * t_from,
            "accounting_residual": 0.0,
            "recorded_accounting_residual": 0.0,
            "n_firms_from": 50.0,
            "n_firms_to": 48.0,
            "default_rate": 0.04,
            "survival_rate": 0.96,
            "active_firms": 48.0,
            "active_paths": 1.0,
            "investment_rate": 0.3,
            "entry_rate": 0.1,
        }
        waterfall_rows.append(row)
        totals_rows.append(
            {
                "t_from": float(t_from),
                "t_to": float(t_to),
                "initial_capital": row["initial_capital"],
                "incumbent_dk": row["incumbent_dk"],
                "exiting_incumbent_capital": row["exiting_incumbent_capital"],
                "entrant_capital": row["entrant_capital"],
                "same_node_entrant_exit": row["same_node_entrant_exit"],
                "ending_capital": row["ending_capital"],
                "accounting_residual": row["accounting_residual"],
                "default_rate": row["default_rate"],
                "survival_rate": row["survival_rate"],
                "investment_rate": row["investment_rate"],
                "entry_rate": row["entry_rate"],
                "active_firms": row["active_firms"],
                "active_paths": row["active_paths"],
            }
        )
    waterfall = pd.DataFrame(waterfall_rows)
    period_counts = pd.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 3.0],
            "active_paths": [8.0, 7.0, 6.0, 5.0],
            "extinct_paths": [0.0, 1.0, 2.0, 3.0],
            "extinct_path_ids": ["", "3", "3,5", "3,5,7"],
            "n_alive_firms": [80.0, 76.0, 70.0, 64.0],
            "n_entrants": [0.0, 4.0, 3.0, 2.0],
            "n_defaults": [0.0, 4.0, 5.0, 6.0],
            "n_investors": [0.0, 20.0, 18.0, 15.0],
        }
    )
    path_status = pd.DataFrame(
        {
            "path": np.arange(8),
            "first_t": np.zeros(8, dtype=np.int64),
            "last_t": [3, 3, 3, 0, 3, 1, 3, 2],
            "n_rows": [4, 4, 4, 1, 4, 2, 4, 3],
            "extinct": [False, False, False, True, False, True, False, True],
            "stop_time": [None, None, None, 1, None, 2, None, 3],
        }
    )
    rng = np.random.default_rng(99)
    firm_panel = pd.DataFrame(
        {
            "path": np.repeat(np.arange(4), 40),
            "t": np.tile(np.repeat([0.0, 1.0, 2.0, 3.0], 10), 4),
            "branch": -np.ones(160),
            "b": rng.uniform(0.0, 1.0, 160),
            "z": rng.uniform(-1.0, 1.0, 160),
            "x": rng.uniform(-0.2, 0.2, 160),
            "K": rng.uniform(1.0, 3.0, 160),
            "Y": rng.uniform(0.5, 1.5, 160),
            "I": rng.uniform(0.05, 0.3, 160),
            "C": rng.uniform(0.3, 0.9, 160),
        }
    )
    macro_panel = pd.DataFrame(
        {
            "path": np.repeat(np.arange(4), 4),
            "t": np.tile([0.0, 1.0, 2.0, 3.0], 4),
            "branch": -np.ones(16),
            "K": 100.0 + 5.0 * np.tile([0.0, 1.0, 2.0, 3.0], 4),
            "C": 30.0 + 1.0 * np.tile([0.0, 1.0, 2.0, 3.0], 4),
            "Hatc": np.log(0.3) + 0.01 * np.tile([0.0, 1.0, 2.0, 3.0], 4),
            "LnK": np.log(100.0) + 0.02 * np.tile([0.0, 1.0, 2.0, 3.0], 4),
            "I_total": 8.0 + 0.5 * np.tile([0.0, 1.0, 2.0, 3.0], 4),
        }
    )
    marginals = []
    surfaces = {}
    for t_slice in (0, 1, 2, 3):
        for caliber in ("count", "capital"):
            hist = np.abs(_field(0.1, slope_b=-0.3, slope_z=-0.2)) + 0.01
            hist = hist / hist.sum()
            surfaces[f"bz_hist_t{t_slice}_{caliber}"] = hist
            for axis, values, edges in (
                ("b", hist.sum(axis=1), B_VALUES),
                ("z", hist.sum(axis=0), Z_VALUES),
            ):
                for center, weight in zip(
                    0.5 * (edges[:-1] + edges[1:]), np.asarray(values, dtype=np.float64)
                ):
                    marginals.append(
                        {
                            "t": float(t_slice),
                            "caliber": caliber,
                            "axis": axis,
                            "bin_center": float(center),
                            "weight": float(weight),
                        }
                    )
    return SimEvalResult(
        metrics={
            "sim_baseline_capital_change_t0_t1": -4.0,
            "sim_baseline_initial_cohort_default_share_t1": 0.05,
        },
        tables={
            "capital_waterfall": waterfall,
            "capital_waterfall_totals": pd.DataFrame(totals_rows),
            "period_counts": period_counts,
            "path_status": path_status,
            "distribution_marginals": pd.DataFrame(marginals),
            "firm_panel": firm_panel,
            "macro_panel": macro_panel,
        },
        surfaces=surfaces,
        diagnostics={"episode": episode, "bank_id": "bank_e_synthetic"},
    )


def _all_inputs(n_episodes: int = 2):
    bellman = {
        episode: make_bellman_result(episode, level=1e-2 / (episode + 1))
        for episode in range(1, n_episodes + 1)
    }
    policy = {
        episode: make_policy_result(episode, shift=0.02 * episode)
        for episode in range(1, n_episodes + 1)
    }
    macro = {episode: make_macro_result(episode) for episode in range(1, n_episodes + 1)}
    sim = {episode: make_sim_result(episode) for episode in range(1, n_episodes + 1)}
    bank = {
        "C": {
            "bank_id": "bank_c_synthetic",
            "bank_kind": "synthetic-stress",
            "macro_anchor": {
                "Recession": {"quantile": 0.1},
                "Normal": {"quantile": 0.5},
                "Expansion": {"quantile": 0.9},
            },
        }
    }
    return bellman, policy, macro, sim, bank


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# full-set rendering
# ---------------------------------------------------------------------------


def test_render_all_figures_produces_every_figure_file(tmp_path):
    bellman, policy, macro, sim, bank = _all_inputs(n_episodes=2)
    output_dir = tmp_path / "figures"
    manifest = render_all_figures(
        bellman_results=bellman,
        policy_results=policy,
        macro_results=macro,
        sim_results=sim,
        bank=bank,
        output_dir=output_dir,
    )

    assert isinstance(manifest, FiguresManifest)
    assert set(manifest.figures) == set(FIGURE_IDS)

    for figure_id, record in manifest.figures.items():
        assert record.status == "ok", (
            f"{figure_id} skipped: {record.missing_reason} "
            f"panels={record.meta.get('panel_errors')}"
        )
        assert record.missing_reason is None
        path = output_dir / FIGURE_FILENAMES[figure_id]
        assert path.exists(), f"{figure_id} did not write {path}"
        assert path.stat().st_size > 0
        assert Path(record.path) == path
        assert path.read_bytes()[:8] == PNG_MAGIC


def test_skip_without_any_results_writes_no_files(tmp_path):
    output_dir = tmp_path / "figures"
    manifest = render_all_figures(output_dir=output_dir)

    assert set(manifest.figures) == set(FIGURE_IDS)
    for figure_id, record in manifest.figures.items():
        assert record.status == "skipped"
        assert record.missing_reason
        assert record.path is None
        assert not (output_dir / FIGURE_FILENAMES[figure_id]).exists()
    assert not list(output_dir.iterdir())


def test_missing_data_figures_are_skipped_with_reason_and_no_file(tmp_path):
    _, policy, _, _, bank = _all_inputs(n_episodes=1)
    output_dir = tmp_path / "figures"
    manifest = render_all_figures(policy_results=policy, bank=bank, output_dir=output_dir)

    # firm-side figures have data
    for figure_id in ("F01", "F02", "F03", "F04", "V01", "V04", "P01", "P04"):
        assert manifest[figure_id].status == "ok", manifest[figure_id].missing_reason

    expected_reasons = {
        "F05": "MISSING_BELLMAN_METRICS",
        "F07": "MISSING_MACRO_PAIRS",
        "F08": "MISSING_MACRO_ANCHOR_PATHS",
        "F09": "MISSING_SIM_PANEL",
        "F10": "MISSING_JOINT_HISTOGRAM",
        "F11": "MISSING_CAPITAL_WATERFALL",
        "M01": "MISSING_MACRO_PAIRS",
        "M04": "MISSING_FC1_RECURSION_TABLE",
        "M05": "MISSING_SDF_TABLE",
        "M06": "MISSING_MACRO_PANEL",
    }
    for figure_id, reason in expected_reasons.items():
        record = manifest[figure_id]
        assert record.status == "skipped", f"{figure_id} unexpectedly ok"
        assert record.missing_reason == reason
        assert record.path is None
        assert not (output_dir / FIGURE_FILENAMES[figure_id]).exists()
        assert record.caption == CAPTIONS[figure_id]

    # single-episode input -> no cross-episode drift evidence
    assert manifest["F06"].status == "skipped"
    assert manifest["F06"].missing_reason == "INSUFFICIENT_EPISODES_FOR_DRIFT"
    assert manifest["V06"].status == "skipped"
    # one result block only -> the dashboard refuses to fabricate a synthesis
    assert manifest["F12"].status == "skipped"
    assert manifest["F12"].missing_reason == "INSUFFICIENT_RESULT_BLOCKS"


def test_manifest_fields_are_complete(tmp_path):
    bellman, policy, macro, sim, bank = _all_inputs(n_episodes=2)
    output_dir = tmp_path / "figures"
    manifest = render_all_figures(
        bellman_results=bellman,
        policy_results=policy,
        macro_results=macro,
        sim_results=sim,
        bank=bank,
        output_dir=output_dir,
    )

    for figure_id, record in manifest.figures.items():
        payload = record.to_dict()
        for key in (
            "figure_id",
            "path",
            "caption",
            "status",
            "missing_reason",
            "n_samples",
            "meta",
        ):
            assert key in payload
        assert payload["figure_id"] == figure_id
        assert record.caption.strip(), f"{figure_id} has an empty caption"
        assert record.caption == CAPTIONS[figure_id]
        assert record.status in ("ok", "skipped")
        # every ok record carries a sample count in the meta block
        assert "n_samples" in record.meta
        assert record.meta["n_samples"] == record.n_samples
        if record.status == "ok":
            assert isinstance(record.meta["n_samples"], int)
            assert record.meta["n_samples"] > 0

    serialisable = manifest.to_dict()
    assert set(serialisable["figures"]) == set(FIGURE_IDS)
    assert serialisable["output_dir"] == str(output_dir)


def test_unified_scale_and_axis_ranges_are_applied(tmp_path):
    bellman, policy, macro, sim, bank = _all_inputs(n_episodes=2)
    output_dir = tmp_path / "figures"
    manifest = render_all_figures(
        bellman_results=bellman,
        policy_results=policy,
        macro_results=macro,
        sim_results=sim,
        bank=bank,
        output_dir=output_dir,
    )

    b_range = [float(B_VALUES.min()), float(B_VALUES.max())]
    z_range = [float(Z_VALUES.min()), float(Z_VALUES.max())]
    reference_limits = None
    for figure_id, record in manifest.figures.items():
        assert record.meta["unified_scale"] is True
        assert record.meta["b_range"] == b_range, figure_id
        assert record.meta["z_range"] == z_range, figure_id
        limits = record.meta["value_limits"]
        if reference_limits is None:
            reference_limits = limits
        assert limits == reference_limits, f"{figure_id} colour limits diverge"
    assert reference_limits, "global colour limits were not recorded"
    for family in ("value", "q", "bp", "prob"):
        low, high = reference_limits[family]
        assert low < high, family

    # resolve_style spans the union of all episodes' grids
    style = resolve_style(None, list(policy.values()))
    assert style.b_range == (float(B_VALUES.min()), float(B_VALUES.max()))
    assert style.z_range == (float(Z_VALUES.min()), float(Z_VALUES.max()))
    assert style.value_limits

    # a caller-provided range is never overridden
    forced = resolve_style(FigureStyle(b_range=(-5.0, 5.0)), list(policy.values()))
    assert forced.b_range == (-5.0, 5.0)


def test_rendering_twice_is_byte_stable(tmp_path):
    """Same input -> identical PNG bytes.

    matplotlib writes a fixed ``Software`` text chunk (we pin it explicitly in
    ``_save``) and no creation timestamp, so a strict SHA-256 comparison of the
    rendered PNGs is the right stability check here -- no fallback to "compare
    the arrays" is needed.
    """
    bellman, policy, macro, sim, bank = _all_inputs(n_episodes=2)
    first = tmp_path / "run_a"
    second = tmp_path / "run_b"
    manifest_a = render_all_figures(
        bellman_results=bellman,
        policy_results=policy,
        macro_results=macro,
        sim_results=sim,
        bank=bank,
        output_dir=first,
    )
    manifest_b = render_all_figures(
        bellman_results=bellman,
        policy_results=policy,
        macro_results=macro,
        sim_results=sim,
        bank=bank,
        output_dir=second,
    )

    for figure_id in FIGURE_IDS:
        record_a = manifest_a[figure_id]
        record_b = manifest_b[figure_id]
        assert record_a.status == record_b.status
        assert record_a.meta == record_b.meta
        if record_a.status != "ok":
            continue
        path_a = first / FIGURE_FILENAMES[figure_id]
        path_b = second / FIGURE_FILENAMES[figure_id]
        assert _sha256(path_a) == _sha256(path_b), f"{figure_id} is not byte-stable"


def test_3d_figures_render_headless(tmp_path):
    _, policy, _, _, _ = _all_inputs(n_episodes=1)
    output_dir = tmp_path / "figures"
    manifest = render_all_figures(policy_results=policy, output_dir=output_dir)

    three_d = ("F01", "F03", "F04", "V01", "V02", "V03", "P01", "P02")
    for figure_id in three_d:
        record = manifest[figure_id]
        assert record.status == "ok", record.missing_reason
        assert (output_dir / FIGURE_FILENAMES[figure_id]).read_bytes()[:8] == PNG_MAGIC

    # the canonical 3D set is reported and rendered on the 3D figures
    for figure_id in ("F01", "F03", "F04"):
        assert manifest[figure_id].meta["canonical_3d"] == list(CANONICAL_3D_NAMES)
        panel_titles = manifest[figure_id].meta["panels"]
        for name in CANONICAL_3D_NAMES:
            assert any(name in title and title.startswith("3D") for title in panel_titles), (
                f"{figure_id} is missing the canonical 3D surface {name}"
            )
        assert not manifest[figure_id].meta["dropped_panels"]


def test_single_figure_entries_are_independently_callable(tmp_path):
    _, policy, macro, _, _ = _all_inputs(n_episodes=2)
    output_dir = tmp_path / "single"

    record = render_f01(policy_results=policy, output_dir=output_dir)
    assert isinstance(record, FigureRecord)
    assert record.status == "ok"
    assert (output_dir / FIGURE_FILENAMES["F01"]).exists()

    waterfall = render_f11_waterfall(
        sim_results=make_sim_result(), output_dir=output_dir
    )
    assert waterfall.status == "ok"
    assert waterfall.meta["survival_panel_source"] in ("period_counts", "path_status")
    assert (output_dir / FIGURE_FILENAMES["F11"]).exists()

    scatter = render_m01(macro_results=make_macro_result(), output_dir=output_dir)
    assert scatter.status == "ok"
    assert scatter.n_samples == 24

    q_calibers = render_v04(policy_results=policy, output_dir=output_dir)
    assert q_calibers.status == "ok"
    assert q_calibers.meta["q_calibers"] == [
        "Q_claim",
        "q_unit",
        "Q_effective",
        "recovery",
    ]

    objective = render_p04(policy_results=policy, output_dir=output_dir)
    assert objective.status == "ok"
    assert objective.n_samples == 44

    dashboard = render_f12(
        bellman_results=make_bellman_result(),
        policy_results=policy,
        macro_results=macro,
        output_dir=output_dir,
    )
    assert dashboard.status == "ok"
    assert set(dashboard.meta["result_blocks"]) == {"bellman", "policy", "macro"}


def test_skipped_single_entry_does_not_create_a_file(tmp_path):
    output_dir = tmp_path / "single"
    record = render_f01(policy_results=None, output_dir=output_dir)
    assert record.status == "skipped"
    assert record.missing_reason == "MISSING_POLICY_SURFACES"
    assert record.path is None
    assert not (output_dir / FIGURE_FILENAMES["F01"]).exists()

    record = render_f11_waterfall(sim_results=make_sim_result(), output_dir=output_dir)
    assert record.status == "ok"
    empty = SimEvalResult(metrics={}, tables={}, surfaces={}, diagnostics={})
    record = render_f11_waterfall(sim_results=empty, output_dir=output_dir)
    assert record.status == "skipped"
    assert record.missing_reason == "MISSING_CAPITAL_WATERFALL"


def test_episode_axis_is_unified_across_episodes(tmp_path):
    bellman, policy, macro, sim, bank = _all_inputs(n_episodes=3)
    output_dir = tmp_path / "figures"
    manifest = render_all_figures(
        bellman_results=bellman,
        policy_results=policy,
        macro_results=macro,
        sim_results=sim,
        bank=bank,
        output_dir=output_dir,
    )
    assert manifest["F05"].meta["episodes"] == [1, 2, 3]
    assert manifest["F06"].meta["episodes"] == [1, 2, 3]
    assert manifest["V06"].meta["episodes"] == [1, 2, 3]
    assert manifest["F05"].meta["episode"] == "1..3"
    assert manifest["F06"].n_samples == 40 * 2
