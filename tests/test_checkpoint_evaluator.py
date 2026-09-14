from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.checkpoint_loader import load_analysis_checkpoint
from analysis.economic_config import AnalysisEconomicConfig
from config import HyperParams
from evaluation.boundaries import extract_phat_default_boundary
from evaluation.bp_diagnostics import _summary, build_frozen_transition_children
from evaluation.firm_surfaces import evaluate_investment_cutoff
from evaluation.grids import ReferenceFirmState, build_frozen_grid, load_reference_state
from experiments.run_utils import build_models
from models import PolicyValueModel


def _small_model() -> PolicyValueModel:
    return PolicyValueModel(
        share_hidden_dims=[8],
        share_output_dim=8,
        q_head_dims=[4],
        p0_head_dims=[4],
        pi_head_dims=[4],
        bp0_head_dims=[4],
        bpi_head_dims=[4],
        barz_hidden_dims=[4],
        bari_hidden_dims=[4],
        i_grid_size=5,
        dropout=0.0,
        value_scale_mode="none",
    )


def _write_combined_policy_checkpoint(path: Path) -> None:
    torch.manual_seed(77)
    model = _small_model()
    sdf_fc1 = build_models(torch.device("cpu"))["sdf_fc1"]
    hp = HyperParams()
    hp.bp_grid_coarse_size = 5
    hp.bp_grid_fine_size = 3
    hp.bp_grid_refine_enabled = False
    hp.bp_grid_parent_chunk_size = 16
    hp.bp_grid_candidate_chunk_size = 5
    hp.bp_grid_max_expanded_states = 4096
    hp.pv_use_clipped_m = True
    hp.pv_m_clamp_min = 0.7
    hp.pv_m_clamp_max = 1.3
    hp.pv_value_scale_mode = "none"
    hp.pv_value_scale_log_max = 20.0
    hp.pv_bellman_normalize_by_value_scale = False
    torch.save(
        {
            "models": {
                "policy_value": model.state_dict(),
                "sdf_fc1": sdf_fc1.state_dict(),
            },
            "hyperparams": hp.__dict__,
            "config_snapshot": AnalysisEconomicConfig.from_current_config().to_dict(),
            "policy_value_model_spec": model.model_spec(),
            "value_parameterization": {
                "mode": "none",
                "scale_formula": "1",
                "bellman_normalization": False,
                "log_max": 20.0,
            },
        },
        path,
    )


def _write_reference_firm(path: Path) -> None:
    rows = []
    for idx in range(6):
        parent = {
            "path": idx,
            "ID": idx,
            "t": 0,
            "branch": -1,
            "b": 0.1 + 0.1 * idx,
            "z": -0.5 + 0.2 * idx,
            "ETA": 1.0,
            "i": 0.05 + 0.05 * idx,
            "x": -1.9 + 0.01 * idx,
            "Hatcf": -2.2 + 0.02 * idx,
            "LnKF": 4.0 + 0.01 * idx,
            "Hatc": -2.1 + 0.02 * idx,
            "LnK": 4.1 + 0.01 * idx,
            "M": 0.98,
        }
        rows.append(parent)
        for branch, z_delta, m_value in ((0, -0.08, 0.96), (1, 0.08, 1.00)):
            rows.append(
                {
                    **parent,
                    "t": 1,
                    "branch": branch,
                    "b": parent["b"] + 0.02,
                    "z": parent["z"] + z_delta,
                    "x": parent["x"] + (branch * 2 - 1) * 0.01,
                    "M": m_value,
                }
            )
    pd.DataFrame(rows).to_pickle(path)


def _run_evaluator(checkpoint: Path, firm_data: Path, output: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [
            sys.executable,
            "experiments/evaluate_checkpoints.py",
            "--checkpoint", str(checkpoint),
            "--firm-data", str(firm_data),
            "--output-dir", str(output),
            "--device", "cpu",
            "--b-points", "5",
            "--z-points", "5",
            "--i-points", "5",
            "--forward-chunk-size", "17",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )


def test_policy_only_combined_checkpoint_does_not_require_sdf(tmp_path):
    checkpoint = tmp_path / "policy_combined.pt"
    _write_combined_policy_checkpoint(checkpoint)
    loaded = load_analysis_checkpoint(checkpoint, device="cpu", m_source="none")
    assert loaded.metadata["loaded_model_keys"] == ["policy_value"]
    assert loaded.metadata["sdf_state_hash"] is None


def test_reference_state_loads_calculated_macro_from_sibling_file(tmp_path):
    firm_path = tmp_path / "ep2_stage_modeb.pkl"
    macro_path = tmp_path / "ep2_stage_modeb_macro.pkl"
    firm = pd.DataFrame(
        {
            "path": [0], "t": [0], "branch": [-1], "b": [0.2], "z": [0.1],
            "ETA": [1.0], "i": [0.2], "x": [-2.0], "Hatcf": [-2.2], "LnKF": [4.0],
        }
    )
    macro = pd.DataFrame(
        {"path": [0], "t": [0], "branch": [0], "Hatc": [-2.1], "LnK": [4.1]}
    )
    firm.to_pickle(firm_path)
    macro.to_pickle(macro_path)
    _, reference = load_reference_state(firm_path)
    assert reference.hatc_cal == -2.1
    assert reference.lnk_cal == 4.1
    assert reference.macro_source == str(macro_path)


def test_final_simulate_macro_auto_discovery_and_explicit_override(tmp_path):
    firm_path = tmp_path / "final_simulate_firm.pkl"
    auto_macro_path = tmp_path / "final_simulate_macro.pkl"
    explicit_macro_path = tmp_path / "chosen_macro.pkl"
    firm = pd.DataFrame(
        {
            "path": [0], "t": [0], "branch": [-1], "b": [0.2], "z": [0.1],
            "ETA": [1.0], "i": [0.2], "x": [-2.0], "Hatcf": [-2.2], "LnKF": [4.0],
        }
    )
    firm.to_pickle(firm_path)
    pd.DataFrame(
        {"path": [0], "t": [0], "Hatc": [-2.1], "LnK": [4.1]}
    ).to_pickle(auto_macro_path)
    pd.DataFrame(
        {"path": [0], "t": [0], "Hatc": [-3.1], "LnK": [5.1]}
    ).to_pickle(explicit_macro_path)

    _, automatic = load_reference_state(firm_path)
    _, explicit = load_reference_state(firm_path, macro_path=explicit_macro_path)

    assert automatic.macro_source == str(auto_macro_path)
    assert automatic.hatc_cal == -2.1
    assert explicit.macro_source == str(explicit_macro_path)
    assert explicit.hatc_cal == -3.1
    assert explicit.lnk_cal == 5.1


def test_phat_boundary_preserves_missing_crossings_as_nan():
    b = np.array([0.0, 0.5, 1.0])
    z = np.array([-1.0, 0.0, 1.0])
    phat = np.array([[-1.0, 0.0, 1.0], [-2.0, -1.0, -0.5], [0.5, 1.0, 2.0]])
    boundary, summary = extract_phat_default_boundary(b, z, phat)
    assert boundary.loc[0, "z_default"] == 0.0
    assert np.isnan(boundary.loc[1, "z_default"])
    assert np.isnan(boundary.loc[2, "z_default"])
    assert boundary.loc[1, "boundary_status"] == "all_default"
    assert boundary.loc[2, "boundary_status"] == "all_survival"
    assert summary["default_boundary_observed_share"] == 1.0 / 3.0


def test_phat_boundary_recognizes_zero_at_grid_endpoint():
    b = np.array([0.0, 1.0])
    z = np.array([-1.0, 0.0, 1.0])
    phat = np.array([[-2.0, -1.0, 0.0], [0.0, 1.0, 2.0]])
    boundary, _ = extract_phat_default_boundary(b, z, phat)
    assert boundary.loc[0, "z_default"] == 1.0
    assert boundary.loc[1, "z_default"] == -1.0
    assert (boundary["boundary_status"] == "single_crossing").all()


def test_phat_multiple_crossings_are_excluded_from_main_boundary():
    b = np.array([0.0, 0.5, 1.0])
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    phat = np.array(
        [
            [-1.0, -0.5, 0.5, 1.0, 2.0],
            [-1.0, 1.0, -1.0, 1.0, -1.0],
            [-2.0, -1.0, -0.5, 0.5, 1.0],
        ]
    )
    boundary, summary = extract_phat_default_boundary(b, z, phat)
    assert boundary.loc[1, "boundary_status"] == "multiple_crossings"
    assert boundary.loc[1, "crossing_count"] == 4
    assert np.isnan(boundary.loc[1, "z_default"])
    assert summary["default_boundary_multiple_crossing_share"] == 1.0 / 3.0
    assert np.isnan(summary["default_boundary_monotonic_share"])


def test_frozen_transition_children_use_state_dependent_ar1_and_sdf_m():
    class FakeSDF(torch.nn.Module):
        def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical=True):
            m = 1.0 + 0.1 * x_curr
            return (
                torch.ones_like(x_prev),
                torch.ones_like(x_curr),
                m,
                hatcf_prev.unsqueeze(1).expand_as(x_curr) + 0.01 * x_curr,
                lnkf_prev.unsqueeze(1).expand_as(x_curr) + 0.01 * x_curr,
            )

    reference = ReferenceFirmState(
        eta=1.0,
        i_low=0.1,
        i_mid=0.2,
        i_high=0.3,
        x=-2.0,
        hatcf=-2.2,
        lnkf=4.0,
        hatc_cal=-2.1,
        lnk_cal=4.1,
        n_parent_rows=2,
        source="fixture",
        macro_source="fixture",
    )
    parents = torch.tensor(
        [
            [0.2, -4.0, 1.0, 0.2, -2.0, -2.2, 4.0],
            [0.2, 4.0, 1.0, 0.2, -2.0, -2.2, 4.0],
        ]
    )
    hp = HyperParams()
    hp.pv_use_clipped_m = False
    config = AnalysisEconomicConfig.from_current_config()
    children, m_list, metadata = build_frozen_transition_children(
        FakeSDF(),
        parents,
        reference,
        hp,
        config,
        n_child_shocks=2,
        shock_seed=9,
    )
    expected_z_difference = float(config.RHO_Z) * 8.0
    assert children[0][1, 1] - children[0][0, 1] == pytest.approx(expected_z_difference)
    assert expected_z_difference != pytest.approx(8.0)
    assert not torch.allclose(m_list[0], m_list[1])
    assert metadata["builder"] == "ConvergenceShockBank+build_child_exogenous_bundle"
    assert 0.0 <= metadata["eta_next_active_share"] <= 1.0


def test_bp_consistency_primary_statistics_use_survival_mask():
    pred = np.array([[0.99, 0.20]])
    star = np.array([[0.01, 0.20]])
    mask = np.array([[False, True]])
    summary = _summary("p0", pred, star, mask)
    assert summary["p0_mae"] == 0.0
    assert summary["p0_raw_mae"] == pytest.approx(0.49)
    assert summary["p0_predicted_high_boundary_share"] == 0.0
    assert summary["p0_raw_predicted_high_boundary_share"] == 0.5


def test_investment_status_requires_exactly_one_crossing():
    class InvestmentModel(torch.nn.Module):
        def forward_value_components(self, states):
            b = states[:, 0:1]
            i = states[:, 3:4]
            delta = torch.where(
                b < 0.2,
                torch.ones_like(i),
                torch.where(
                    b < 0.5,
                    -torch.ones_like(i),
                    torch.where(b < 0.8, i - 1.0, (i - 0.25) * (i - 0.75)),
                ),
            )
            return {"V0_physical": torch.zeros_like(delta), "VI_physical": delta}

    reference = ReferenceFirmState(
        eta=1.0, i_low=0.0, i_mid=0.5, i_high=1.0,
        x=-2.0, hatcf=-2.1, lnkf=4.0, hatc_cal=-2.0, lnk_cal=4.1,
        n_parent_rows=1, source="fixture", macro_source="fixture",
    )
    grid = build_frozen_grid(
        reference,
        b_min=0.0,
        b_max=1.0,
        b_points=4,
        z_min=-1.0,
        z_max=1.0,
        z_points=2,
        device=torch.device("cpu"),
    )
    result = evaluate_investment_cutoff(
        InvestmentModel(),
        grid,
        reference,
        i_points=5,
        i_min=0.0,
        i_max=1.0,
        chunk_size=32,
        survival_mask=np.ones(grid.shape, dtype=bool),
    )
    assert set(result["investment_status"][:, 0]) == {
        "all_invest", "all_no_invest", "single_crossing", "multiple_crossings"
    }
    assert np.all(result["i_star"][2] == 1.0)
    assert np.isnan(result["i_star"][3]).all()


def test_firm_checkpoint_evaluator_smoke_is_read_only_and_deterministic(tmp_path):
    checkpoint = tmp_path / "policy_combined.pt"
    firm_data = tmp_path / "firm.pkl"
    out_a = tmp_path / "eval_a"
    out_b = tmp_path / "eval_b"
    _write_combined_policy_checkpoint(checkpoint)
    _write_reference_firm(firm_data)

    first = _run_evaluator(checkpoint, firm_data, out_a)
    second = _run_evaluator(checkpoint, firm_data, out_b)
    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr

    required = [
        "metadata.json",
        "summary.csv",
        "value/P0.png",
        "value/PI_low.png",
        "value/PI_mid.png",
        "value/PI_high.png",
        "default/default_boundary.csv",
        "investment/i_star.png",
        "investment/bar_i_cond.csv",
        "investment/bar_i_eff.csv",
        "q/Q_b_slices.png",
        "q/q_unit.csv",
        "bp/bp_raw.csv",
        "bp/bp_survival.csv",
        "bp/p0_bp_abs_gap_survival.png",
        "bp/pi_mid_bp_grid_star_survival.csv",
        "objective_slices/p0_b_mid_z_mid.csv",
        "objective_slices/pi_mid_b_mid_z_mid.png",
    ]
    for relative in required:
        assert (out_a / relative).exists(), relative
    assert len(list((out_a / "objective_slices").glob("*.csv"))) == 18

    metadata = json.loads((out_a / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["model_state_unchanged"] is True
    assert metadata["model_state_hash_before"] == metadata["model_state_hash_after"]
    assert set(metadata["model_state_hash_before"]) == {"policy_value", "sdf_fc1"}
    assert metadata["grid"]["eta"] == 1.0
    assert metadata["reference_state"]["n_parent_rows"] == 6
    assert metadata["reference_transition_bank"]["m_source"] == "sdf_fc1.forward_step"
    assert 0.0 <= metadata["reference_transition_bank"]["eta_next_active_share"] <= 1.0
    assert metadata["bp_teacher_model"] == "policy_value"
    assert metadata["reference_transition_bank"]["bp_teacher_model"] == "policy_value"

    q_unit = pd.read_csv(out_a / "q" / "q_unit.csv", index_col=0)
    assert q_unit.iloc[0].isna().all()
    pdt.assert_frame_equal(
        pd.read_csv(out_a / "summary.csv"),
        pd.read_csv(out_b / "summary.csv"),
        check_exact=True,
    )
    pdt.assert_frame_equal(
        pd.read_csv(out_a / "objective_slices" / "p0_b_mid_z_mid.csv"),
        pd.read_csv(out_b / "objective_slices" / "p0_b_mid_z_mid.csv"),
        check_exact=True,
    )
    objective = pd.read_csv(out_a / "objective_slices" / "p0_b_mid_z_mid.csv")
    assert {
        "eta_next_active_share",
        "child_b_mean",
        "child_b_eta0_mean",
        "child_b_eta1_mean",
    }.issubset(objective.columns)
