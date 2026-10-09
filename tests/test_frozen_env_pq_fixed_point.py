from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

from evaluation.pq_fixed_point import (
    BATCH_BANK_FORMAT,
    absolute_gap_summary,
    choose_verdict,
    function_drift,
    object_sha256,
    seed_fixed_mapping,
    update_cosine,
    validate_cycle_teacher_hash,
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


def test_verdict_prioritizes_fitted_update_failure() -> None:
    rows = [{
        "P_stage_fit_on_mean": 0.5,
        "Q_stage_fit_on_mean": 0.5,
        "P_stage_fit_canonical_mean": 0.5,
        "Q_stage_fit_canonical_mean": 0.5,
        "d_joint": 0.0,
        "rho": 0.0,
        "cos_theta": 1.0,
        "two_step_distance": 0.0,
    }]
    verdict, _ = choose_verdict(rows, fit_tolerance=0.01, distance_tolerance=0.001)
    assert verdict == "B"


def test_verdict_detects_controlled_convergence() -> None:
    rows = []
    for cycle, distance in enumerate((1.0, 0.4, 0.1, 0.0005), start=1):
        rows.append({
            "cycle": cycle,
            "P_stage_fit_on_mean": 0.001,
            "Q_stage_fit_on_mean": 0.001,
            "P_stage_fit_canonical_mean": 0.002,
            "Q_stage_fit_canonical_mean": 0.002,
            "d_joint": distance,
            "rho": 0.4,
            "cos_theta": 0.8,
            "two_step_distance": distance,
        })
    verdict, _ = choose_verdict(rows, fit_tolerance=0.01, distance_tolerance=0.001)
    assert verdict == "A"


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
