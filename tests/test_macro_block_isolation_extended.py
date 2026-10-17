from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.run_macro_block_isolation import TwoCycleDiagnosticsConfig
from analysis.run_macro_block_isolation_extended import (
    _assert_prohibitions,
    add_transition_metrics,
    checkpoint_for_outer,
    ensure_new_output,
    finite_movement_metrics,
    partial_resume_outer,
    sorted_macro_vectors,
    used_m_trace,
    validate_extended_limits,
)


def test_strict_cycle_requires_two_consecutive_candidates():
    config = TwoCycleDiagnosticsConfig(consecutive_required=2)
    counts = {}
    first = {}
    add_transition_metrics(
        first,
        {"function_m": [torch.ones(8), -torch.ones(8), torch.ones(8)]},
        counts,
        config,
    )
    assert first["function_m_two_cycle_candidate"]
    assert not first["possible_two_cycle_strict"]
    second = {}
    add_transition_metrics(
        second,
        {"function_m": [-torch.ones(8), torch.ones(8), -torch.ones(8)]},
        counts,
        config,
    )
    assert second["possible_two_cycle_strict"]


def test_damped_oscillation_is_negative_but_not_strict_cycle():
    config = TwoCycleDiagnosticsConfig()
    row = {}
    add_transition_metrics(
        row,
        {
            "function_m": [
                torch.ones(8),
                torch.full((8,), -0.5),
                torch.full((8,), 0.25),
            ]
        },
        {},
        config,
    )
    assert row["negative_directional_movement"]
    assert row["damped_oscillation_flag"]
    assert not row["possible_two_cycle_strict"]


def test_macro_movements_align_dynamic_survivors_by_path_time_branch():
    first = pd.DataFrame(
        {
            "path": [0, 0, 1],
            "t": [0, 1, 0],
            "branch": [-1, -1, -1],
            "Hatc": [1.0, 2.0, 10.0],
            "LnK": [3.0, 4.0, 30.0],
            "C": [5.0, 6.0, 50.0],
            "K": [7.0, 8.0, 70.0],
        }
    )
    second = pd.DataFrame(
        {
            "path": [0, 1, 1],
            "t": [0, 0, 1],
            "branch": [-1, -1, -1],
            "Hatc": [1.0, 10.0, 20.0],
            "LnK": [3.0, 30.0, 40.0],
            "C": [5.0, 50.0, 60.0],
            "K": [7.0, 70.0, 80.0],
        }
    )
    left = sorted_macro_vectors(
        first, expected_paths=2, expected_horizon=1, branch_num=0
    )["target_hatc"]
    right = sorted_macro_vectors(
        second, expected_paths=2, expected_horizon=1, branch_num=0
    )["target_hatc"]
    metrics = finite_movement_metrics(left, right, "target_hatc")
    assert left.shape == right.shape == (4,)
    assert metrics["target_hatc_n_common"] == 2
    assert metrics["target_hatc_overlap_share"] == pytest.approx(0.5)
    assert metrics["target_hatc_nmae"] == pytest.approx(0.0)


def test_resume_checkpoint_outer_indices_are_exact(tmp_path: Path):
    assert checkpoint_for_outer(tmp_path, "fc1_only", 3) == (
        tmp_path / "l1/fc1_only/outer_002_fc1_solve/best.pt"
    )
    assert checkpoint_for_outer(tmp_path, "joint", 8) == (
        tmp_path / "l1/joint/outer_007_sdf_solve/checkpoint_chunk_004.pt"
    )


def test_extended_outer_limits_are_strict():
    valid = SimpleNamespace(
        fc1_resume_outer=3,
        fc1_final_outer=8,
        sdf_final_outer=8,
        joint_resume_outer=8,
        joint_final_outer=14,
    )
    validate_extended_limits(valid)
    with pytest.raises(ValueError):
        validate_extended_limits(
            SimpleNamespace(**{**vars(valid), "fc1_final_outer": 9})
        )
    with pytest.raises(ValueError):
        validate_extended_limits(
            SimpleNamespace(**{**vars(valid), "joint_final_outer": 15})
        )


def test_old_output_is_never_overwritten(tmp_path: Path):
    old = tmp_path / "sdf_only"
    old.mkdir()
    dependency = old / "dependency_audit.json"
    dependency.write_text("original", encoding="utf-8")
    ensure_new_output(old, allowed_existing=("dependency_audit.json",))
    assert dependency.read_text(encoding="utf-8") == "original"
    (old / "metrics.csv").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ensure_new_output(old, allowed_existing=("dependency_audit.json",))


def test_partial_resume_requires_one_durable_next_outer(tmp_path: Path):
    root = tmp_path / "fc1_only_extended"
    (root / "checkpoints").mkdir(parents=True)
    (root / "outer_003_fc1_solve").mkdir()
    (root / "checkpoints" / "outer_004.pt").touch()
    (root / "functions_outer_004_start.pt").touch()
    (root / "outer_003_fc1_solve" / "summary.json").write_text("{}", encoding="utf-8")
    assert (
        partial_resume_outer(root, baseline_outer=3, final_outer=8, enabled=True) == 4
    )
    (root / "metrics.csv").touch()
    with pytest.raises(FileExistsError):
        partial_resume_outer(root, baseline_outer=3, final_outer=8, enabled=True)


def test_raw_to_used_m_trace_matches_production_clamps():
    args = SimpleNamespace(
        pv_m_clamp_min=0.7,
        pv_m_clamp_max=1.3,
        q_m_clamp_min=0.5,
        q_m_clamp_max=1.5,
    )
    raw = torch.tensor([0.1, 1.0, 10.0, 1000.0])
    traced = used_m_trace(raw, args)
    assert torch.equal(traced["raw_M"], raw)
    assert torch.equal(traced["post_transform_M"], raw)
    assert traced["used_P_M"].tolist() == pytest.approx([0.7, 1.0, 1.3, 1.3])
    assert traced["used_Q_M"].tolist() == pytest.approx([0.5, 1.0, 1.5, 1.5])


def test_extended_runner_contains_no_p_q_or_policy_solver_calls():
    _assert_prohibitions()
