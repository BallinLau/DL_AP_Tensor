from __future__ import annotations

import copy
import inspect
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluation.pq_fixed_point import (
    BATCH_BANK_FORMAT,
    absolute_gap_summary,
    choose_verdict,
    fixed_rms_scale,
    make_frozen_batch_bank_payload,
    normalized_rms_error,
    function_drift,
    object_sha256,
    seed_fixed_mapping,
    update_cosine,
    validate_cycle_teacher_hash,
    validate_checkpoint_bank_provenance,
    validate_frozen_batch_bank,
    verify_production_pq_method_fingerprints,
)
from experiments.frozen_env_pq_fixed_point import _assert_stage_isolation
from training.episode import Episode


def _batch(offset: float = 0.0) -> dict:
    parent = torch.tensor([[offset, 0.0], [offset + 1.0, 0.0]])
    return {
        "parent": parent,
        "children": [parent + 0.1, parent + 0.2],
    }


def _payload() -> dict:
    return {
        "format": BATCH_BANK_FORMAT,
        "provenance": {
            "episode": 2,
            "panel_stage": "post_sdf_refresh_pre_pv",
            "simulation_reused_without_rerun": True,
            "batch_composition_frozen": True,
            "batch_order_frozen": True,
            "validation_split_frozen": True,
            "source_commit": "4b236eaacf47b2ac7cb506508e54b21a199e89b0",
            "source_run_commit": "capture-commit",
            "source_run_identity": "/run",
            "run_root": "/run",
            "seed": 12345,
            "policy_value_checkpoint_hash_pre_pv": "policy-pre-pv",
            "sdf_fc1_hash": "sdf",
            "economic_config_hash": "econ",
            "hyperparameter_fingerprint": "hp",
            "pv_training_flow": "staged",
            "q_target_refresh_mode": "phase",
            "simulation_bp_action_source": "grid",
            "pv_bp_head_training_enabled": False,
        },
        "train_batches": [_batch()],
        "validation_batches": [_batch(2.0)],
    }


def test_exact_batch_bank_provenance_and_hash_are_deterministic() -> None:
    first = validate_frozen_batch_bank(_payload())
    second = validate_frozen_batch_bank(copy.deepcopy(_payload()))
    assert first["dataset_sha256"] == second["dataset_sha256"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("panel_stage", "post_pv_resimulation"),
        ("simulation_reused_without_rerun", False),
        ("batch_order_frozen", False),
    ],
)
def test_batch_bank_rejects_inexact_environment(key: str, value: object) -> None:
    payload = _payload()
    payload["provenance"][key] = value
    with pytest.raises(ValueError, match="provenance"):
        validate_frozen_batch_bank(payload)


def test_batch_bank_explicitly_rejects_post_pv_output() -> None:
    payload = _payload()
    payload["provenance"]["source_is_post_pv_output"] = True
    with pytest.raises(ValueError, match="post-PV"):
        validate_frozen_batch_bank(payload)


def test_batch_order_changes_dataset_hash() -> None:
    payload = _payload()
    payload["train_batches"] = [_batch(0.0), _batch(4.0)]
    reversed_payload = copy.deepcopy(payload)
    reversed_payload["train_batches"].reverse()
    assert object_sha256(payload["train_batches"]) != object_sha256(
        reversed_payload["train_batches"]
    )


def test_fixed_mapping_seed_is_reproducible() -> None:
    seed_fixed_mapping(24681357)
    first = (random.random(), np.random.rand(), torch.rand(4))
    seed_fixed_mapping(24681357)
    second = (random.random(), np.random.rand(), torch.rand(4))
    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2], rtol=0.0, atol=0.0)


def test_function_drift_uses_fixed_cycle_zero_scales() -> None:
    previous = {"P": np.array([1.0, 1.0]), "Q": np.array([2.0, 2.0])}
    current = {"P": np.array([2.0, 2.0]), "Q": np.array([4.0, 4.0])}
    result = function_drift(previous, current, p_scale=1.0, q_scale=2.0)
    assert result.d_p == pytest.approx(1.0)
    assert result.d_q == pytest.approx(1.0)
    assert result.d_joint == pytest.approx(np.sqrt(2.0))


def test_update_cosine_detects_reversal() -> None:
    assert update_cosine(np.array([1.0, -2.0]), np.array([-1.0, 2.0])) == pytest.approx(-1.0)


def test_absolute_gap_summary_uses_global_quantiles() -> None:
    summary = absolute_gap_summary(np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0, 100.0]))
    assert summary["max"] == 100.0
    assert summary["p90"] == pytest.approx(80.4)


def _convergent_rows() -> list[dict]:
    distances = (1.0, 0.8, 0.6, 0.3, 0.2, 0.1)
    residuals = (1.0, 0.8, 0.6, 0.3, 0.2, 0.1)
    boundary = (0.10, 0.08, 0.06, 0.02, 0.01, 0.001)
    rows = []
    for index, (distance, residual, switch) in enumerate(
        zip(distances, residuals, boundary), start=1
    ):
        rows.append({
            "cycle": index,
            "P_stage_fit_on_norm": 0.01,
            "Q_stage_fit_on_norm": 0.01,
            "P_stage_fit_canonical_norm": 0.02,
            "Q_stage_fit_canonical_norm": 0.02,
            "d_joint": distance,
            "rho": 0.5,
            "cos_theta": 0.8,
            "two_step_distance": distance,
            "P_fixed_point_residual_mean": residual,
            "Q_fixed_point_residual_mean": residual,
            "boundary_switch_share": switch,
            "bar_z_mean_abs_drift": switch,
        })
    return rows


def _verdict(rows: list[dict]) -> dict:
    return choose_verdict(
        rows,
        fit_normalized_tolerance=0.05,
        distance_tolerance=0.001,
        residual_improvement_ratio=0.8,
        boundary_drift_tolerance=0.005,
        two_cycle_ratio_threshold=0.5,
    )


def test_verdict_prioritizes_fitted_update_failure() -> None:
    rows = _convergent_rows()
    rows[-1]["P_stage_fit_on_norm"] = 0.5
    assert _verdict(rows)["primary"] == "B"


def test_verdict_residual_not_improving_cannot_be_a() -> None:
    rows = _convergent_rows()
    for row in rows:
        row["P_fixed_point_residual_mean"] = 1.0
        row["Q_fixed_point_residual_mean"] = 1.0
    assert _verdict(rows)["primary"] != "A"


def test_verdict_boundary_not_stabilizing_cannot_be_a() -> None:
    rows = _convergent_rows()
    for row in rows:
        row["boundary_switch_share"] = 0.1
        row["bar_z_mean_abs_drift"] = 0.1
    assert _verdict(rows)["primary"] != "A"


def test_verdict_detects_controlled_convergence() -> None:
    verdict = _verdict(_convergent_rows())
    assert verdict["primary"] == "A"
    assert all(verdict["conditions"][key] for key in (
        "stage_fit_controlled",
        "joint_distance_shrinking",
        "p_residual_improving",
        "q_residual_improving",
        "no_two_cycle",
        "boundary_stabilizing",
    ))


def test_verdict_detects_two_cycle_oscillation() -> None:
    rows = _convergent_rows()
    for row in rows[-3:]:
        row["cos_theta"] = -0.9
        row["two_step_distance"] = row["d_joint"] * 0.1
    assert _verdict(rows)["primary"] == "D"


def test_normalized_fit_uses_fixed_cycle_zero_scale() -> None:
    scale = fixed_rms_scale(np.array([2.0, 2.0]))
    first = normalized_rms_error(
        np.array([3.0, 3.0]), np.array([2.0, 2.0]), fixed_scale=scale
    )
    second = normalized_rms_error(
        np.array([101.0, 101.0]), np.array([100.0, 100.0]), fixed_scale=scale
    )
    assert first == pytest.approx(0.5)
    assert second == pytest.approx(0.5)


def test_checkpoint_bank_provenance_rejects_episode_and_sdf_mismatch() -> None:
    common = {
        "episode": 2,
        "source_run_commit": "commit",
        "source_run_identity": "/run",
        "sdf_fc1_hash": "sdf",
        "economic_config_hash": "econ",
        "hyperparameter_fingerprint": "hp",
        "pv_training_flow": "staged",
        "q_target_refresh_mode": "phase",
        "simulation_bp_action_source": "grid",
        "pv_bp_head_training_enabled": False,
    }
    validate_checkpoint_bank_provenance(common, common)
    bad_episode = dict(common, episode=3)
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_checkpoint_bank_provenance(bad_episode, common)
    bad_sdf = dict(common, sdf_fc1_hash="other")
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_checkpoint_bank_provenance(bad_sdf, common)


def test_capture_is_before_staged_optimizer_and_preserves_exact_objects(tmp_path) -> None:
    episode = Episode.__new__(Episode)
    episode.episode_id = 2
    episode.models = {
        "policy_value": torch.nn.Linear(2, 1),
        "sdf_fc1": torch.nn.Linear(2, 1),
    }
    episode.hyperparams = SimpleNamespace(
        save_frozen_pq_batch_bank=True,
        pv_training_flow="staged",
        q_target_refresh_mode="phase",
        simulation_bp_action_source="grid",
        pv_bp_head_training_enabled=False,
    )
    episode.frozen_pq_capture_context = {
        "run_root": str(tmp_path),
        "seed": 12345,
        "source_run_commit": "capture-commit",
    }
    train = [_batch()]
    validation = [_batch(3.0)]
    summary = episode._capture_frozen_pq_batch_bank_if_enabled(train, validation)
    assert summary is not None
    payload = torch.load(summary["path"], map_location="cpu")
    metadata = validate_frozen_batch_bank(payload)
    assert metadata["train_batches_sha256"] == object_sha256(train)
    assert metadata["validation_batches_sha256"] == object_sha256(validation)
    assert payload["provenance"]["p_optimizer_steps_in_current_stage"] == 0

    source = inspect.getsource(Episode.run_episode)
    assert source.index("_capture_frozen_pq_batch_bank_if_enabled") < source.index(
        "_run_policy_value_staged", source.index("_capture_frozen_pq_batch_bank_if_enabled")
    )


def test_captured_payload_is_cpu_detached_copy() -> None:
    train = [_batch()]
    validation = [_batch(2.0)]
    payload = make_frozen_batch_bank_payload(
        train_batches=train,
        validation_batches=validation,
        provenance=_payload()["provenance"],
    )
    train[0]["parent"].add_(100.0)
    assert not torch.equal(payload["train_batches"][0]["parent"], train[0]["parent"])


def test_stage_isolation_rejects_forbidden_component_change() -> None:
    before = {"p": "p0", "q": "q0", "bp": "bp0", "sdf_fc1": "s0"}
    after = dict(before)
    after["q"] = "q1"
    with pytest.raises(RuntimeError, match="forbidden component q"):
        _assert_stage_isolation(before, after, allowed="p", stage="P stage")


def test_stage_isolation_allows_only_requested_component() -> None:
    before = {"p": "p0", "q": "q0", "bp": "bp0", "sdf_fc1": "s0"}
    after = dict(before)
    after["p"] = "p1"
    _assert_stage_isolation(before, after, allowed="p", stage="P stage")


def test_teacher_hash_tracks_each_changed_cycle_start() -> None:
    validate_cycle_teacher_hash(cycle_start_hash="x0", teacher_hash="x0")
    validate_cycle_teacher_hash(
        cycle_start_hash="x1",
        teacher_hash="x1",
        previous_teacher_hash="x0",
        previous_cycle_end_hash="x1",
    )


def test_teacher_hash_rejects_pinned_initial_teacher() -> None:
    with pytest.raises(RuntimeError, match="older X snapshot"):
        validate_cycle_teacher_hash(
            cycle_start_hash="x1",
            teacher_hash="x0",
            previous_teacher_hash="x0",
            previous_cycle_end_hash="x1",
        )


def test_production_pq_methods_match_audited_grid_mapping() -> None:
    fingerprints = verify_production_pq_method_fingerprints(Episode)
    assert len(fingerprints) == 5
