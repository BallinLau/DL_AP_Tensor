from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.checkpoint_loader import load_analysis_checkpoint
from analysis.economic_config import AnalysisEconomicConfig
from config import HyperParams
from evaluation.boundaries import extract_phat_default_boundary
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
            "models": {"policy_value": model.state_dict()},
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
    assert (boundary["boundary_status"] == "observed").all()


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
        "bp/bp.csv",
        "bp/bp_cond.csv",
        "bp/p0_bp_abs_gap.png",
        "bp/pi_mid_bp_grid_star.csv",
        "objective_slices/p0_b_mid_z_mid.csv",
        "objective_slices/pi_mid_b_mid_z_mid.png",
    ]
    for relative in required:
        assert (out_a / relative).exists(), relative
    assert len(list((out_a / "objective_slices").glob("*.csv"))) == 18

    metadata = json.loads((out_a / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["model_state_unchanged"] is True
    assert metadata["model_state_hash_before"] == metadata["model_state_hash_after"]
    assert metadata["grid"]["eta"] == 1.0
    assert metadata["reference_state"]["n_parent_rows"] == 6

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
