from __future__ import annotations

from types import SimpleNamespace
import inspect
from dataclasses import replace

import pandas as pd
import pytest
import torch

import analysis.macro_parent_state_bank as bank_module
from analysis.macro_parent_state_bank import (
    B_DISTRIBUTION_STRATA,
    MACRO_CANDIDATE_STATUS_VALID,
    build_macro_candidate_schedule,
    build_one_step_macro_dataset,
    boundary_macro_rows,
    lock_validation_candidate_ids,
    make_boundary_candidate_initial_state,
)
from analysis.run_full_nested_equilibrium import (
    FIRM_FIRST_STAGE_ORDER,
    _candidate_schedule_overlap,
    _record_firm_first_stage,
    main,
    run_firm_first_orchestration_mock,
)
from data.tensor_data import TensorTable


class _FakeEpisode:
    def __init__(self) -> None:
        self.config = SimpleNamespace(SIM_B_INIT_MIN=0.0, SIM_B_INIT_MAX=1.0)
        self.hyperparams = SimpleNamespace(
            sdf_true_target_batches=2,
            sdf_min_parent_groups_per_batch=1,
        )
        self.train_mode = "1time"
        self.add_FC1loss = False

    def _build_sdf_pairs_from_macro_tensor(self, macro: TensorTable) -> TensorTable:
        frame = pd.DataFrame(macro.data.cpu().numpy(), columns=macro.columns)
        rows = []
        for candidate_id, group in frame.groupby("path"):
            parent = group[group["branch"] == -1].iloc[0]
            for _, child in group[group["branch"] >= 0].iterrows():
                rows.append([
                    candidate_id, child.t, child.branch, parent.x, child.x,
                    parent.hatcf, parent.lnkf, parent.Hatc, parent.LnK,
                    child.Hatc, child.LnK, child.lineage_id,
                ])
        return TensorTable(
            data=torch.tensor(rows, dtype=torch.float32),
            columns=[
                "path", "t", "branch", "x_t", "x_t1", "Hatcf_t",
                "LnKF_t", "Hatc_t", "LnK_t", "Hatc_t1", "LnK_t1",
                "lineage_id",
            ],
        )

    def _create_sdf_batches_from_macro_tensor(self, table, **_kwargs):
        return [{"rows": table.data}] if len(table) else []


def _fake_candidate_evaluator(*, specs, **_kwargs):
    rows, records, manifests = [], [], []
    for spec in specs:
        # Required 32-candidate mock: 23 valid, 4 parent, 3 child-0, 2 child-1 failures.
        position = spec.distribution_parameter_id
        if position in {1, 4, 7, 10}:
            status = "PARENT_CLEARING_FAILED"
        elif position in {13, 16, 19}:
            status = "CHILD_0_CLEARING_FAILED"
        elif position in {22, 25}:
            status = "CHILD_1_CLEARING_FAILED"
        else:
            status = MACRO_CANDIDATE_STATUS_VALID
        failed = status != MACRO_CANDIDATE_STATUS_VALID
        records.append({
            **spec.to_dict(),
            "status": status,
            "failure_reason": status if failed else "",
            "accepted_for_training": False,
            "parent_status": "INVALID" if status == "PARENT_CLEARING_FAILED" else "VALID",
            "child_0_status": "INVALID" if status == "CHILD_0_CLEARING_FAILED" else "VALID",
            "child_1_status": "INVALID" if status == "CHILD_1_CLEARING_FAILED" else "VALID",
        })
        manifests.append({**spec.to_dict(), "candidate_status": status})
        if failed:
            continue
        for t, branch in ((0, -1), (1, 0), (1, 1)):
            rows.append({
                "path": spec.candidate_id,
                "candidate_id": spec.candidate_id,
                "lineage_id": spec.lineage_id,
                "t": t,
                "branch": branch,
                "x": 0.0 + 0.01 * branch,
                "hatcf": -2.0,
                "lnkf": 4.0,
                "Hatc": -1.9 + 0.01 * branch,
                "LnK": 4.1 + 0.01 * branch,
                "M": 1.0,
                "C": 1.0,
                "K": 1.0,
                "I_oper": 0.1,
                "I_entry": 0.0,
                "I_maintenance": 0.0,
                "n_firms": 10.0,
                "default_mass": 0.0,
                "investment_mass": 1.0,
                "mean_b": 0.5,
            })
    return rows, records, manifests


def _schedule(split: str, offset: int = 0):
    return build_macro_candidate_schedule(
        split=split,
        namespace=split,
        target_valid=23,
        initial_candidate_budget=32,
        max_candidate_budget=32,
        minimum_partial_valid=16,
        firms_per_parent=16,
        seed=17 + offset,
        candidate_id_offset=offset * 1_000_000,
    )


def test_b_only_schedule_is_reproducible_diverse_and_disjoint():
    train = _schedule("train", 0)
    repeat = _schedule("train", 0)
    validation = _schedule("validation", 1)
    assert train.fingerprint == repeat.fingerprint
    assert len({(spec.alpha, spec.beta) for spec in train.candidate_specs}) == 32
    assert len({spec.distribution_family for spec in train.candidate_specs}) == len(
        B_DISTRIBUTION_STRATA
    )
    assert _candidate_schedule_overlap(train, validation)["disjoint"] is True


def test_canonical_entry_cannot_call_legacy_rollout_parent_harvest():
    source = inspect.getsource(main)
    assert "_build_or_refresh_macro_candidate_context" in source
    assert "_build_or_refresh_macro_parent_context" not in source
    assert "prepare_macro_parent_schedule" not in source
    assert "harvest_available_macro_parent_schedule" not in source


def test_b_only_initializer_changes_only_b_and_preserves_mass_capital_laws():
    schedule = _schedule("train")
    specs = schedule.candidate_specs[:4]
    changed_specs = tuple(
        replace(spec, alpha=spec.alpha + 0.4, beta=spec.beta + 0.2)
        for spec in specs
    )
    config = SimpleNamespace(
        XBAR=0.0, RHO_X=0.9, SIGMA_X=0.01,
        RHO_Z=0.85, SIGMA_Z=0.1, ZBAR=0.0, ZETA=0.25,
        SIM_B_INIT_MIN=0.05, SIM_B_INIT_MAX=0.95,
    )
    sim = SimpleNamespace(n_paths=4, device=torch.device("cpu"), config=config)
    state, manifest, preservation = make_boundary_candidate_initial_state(
        sim, specs, firms=16
    )
    changed, _, _ = make_boundary_candidate_initial_state(
        sim, changed_specs, firms=16
    )
    assert not torch.equal(state["b"], changed["b"])
    for name in (
        "x", "hatcf", "lnkf", "K", "component_mass", "component_mask",
        "z_conditional_mean", "z_conditional_std", "eta",
    ):
        torch.testing.assert_close(state[name], changed[name], rtol=0, atol=0)
    assert bool(((state["b"] >= 0.05) & (state["b"] <= 0.95)).all())
    torch.testing.assert_close(
        state["component_mass"].sum(dim=1), torch.full((4,), 16.0)
    )
    torch.testing.assert_close(
        (state["component_mass"] * state["K"]).sum(dim=1),
        torch.exp(state["lnkf"]), rtol=1e-5, atol=1e-5,
    )
    assert len(manifest) == 4
    assert preservation["only_b_replaced"] is True

    state["hatc_belief"] = state["hatcf"].clone()
    state["hatc_cleared"] = state["hatcf"] + 0.25
    state["hatc_cal"] = state["hatc_cleared"].clone()
    state["hatcf"] = state["hatc_cleared"].clone()
    state["lnk_cal"] = state["lnkf"] + 0.5
    ones = torch.ones(4)
    zeros = torch.zeros(4)
    state.update({
        "accounting_C_raw": ones,
        "accounting_K_total": ones,
        "accounting_I_oper": zeros,
        "accounting_I_entry": zeros,
        "accounting_I_maintenance": zeros,
        "market_root_residual": zeros,
        "market_root_residual_normalized": zeros,
    })
    rows = boundary_macro_rows(state, t=0, branch=-1, split="train")
    assert rows[0]["hatcf"] == rows[0]["Hatc"]
    assert rows[0]["lnkf"] == rows[0]["LnK"]
    assert rows[0]["lnk_belief"] != rows[0]["lnkf"]


def test_partial_failure_isolation_23_valid_9_failed_and_chunk_invariant(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        bank_module,
        "_evaluate_boundary_candidates_with_isolation",
        _fake_candidate_evaluator,
    )
    episode = _FakeEpisode()
    schedule = _schedule("train")
    small_chunks = build_one_step_macro_dataset(
        models={}, episode=episode, candidate_schedule=schedule,
        parent_batch_size=4, macro_batch_size=64, device=torch.device("cpu"),
        output_dir=tmp_path / "small_chunks",
    )
    large_chunks = build_one_step_macro_dataset(
        models={}, episode=episode, candidate_schedule=schedule,
        parent_batch_size=16, macro_batch_size=64, device=torch.device("cpu"),
        output_dir=tmp_path / "large_chunks",
    )
    for result in (small_chunks, large_chunks):
        coverage = result.pairing_audit["candidate_coverage"]
        assert result.dataset_status == "TARGET_VALID_DATASET"
        assert coverage["valid_complete_pair_count"] == 23
        assert coverage["accepted_training_pair_count"] == 23
        assert coverage["failure_status_counts"] == {
            "PARENT_CLEARING_FAILED": 4,
            "CHILD_0_CLEARING_FAILED": 3,
            "CHILD_1_CLEARING_FAILED": 2,
        }
        assert result.pairing_audit["each_parent_has_exact_children"] is True
    assert (
        small_chunks.pairing_audit["candidate_coverage"]["accepted_candidate_ids"]
        == large_chunks.pairing_audit["candidate_coverage"]["accepted_candidate_ids"]
    )


def test_fixed_validation_identity_has_no_replacement(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bank_module,
        "_evaluate_boundary_candidates_with_isolation",
        _fake_candidate_evaluator,
    )
    episode = _FakeEpisode()
    schedule = _schedule("validation", 2)
    first = build_one_step_macro_dataset(
        models={}, episode=episode, candidate_schedule=schedule,
        parent_batch_size=8, macro_batch_size=64, device=torch.device("cpu"),
        output_dir=tmp_path / "first",
    )
    ids = first.pairing_audit["candidate_coverage"]["accepted_candidate_ids"]
    locked = lock_validation_candidate_ids(schedule, ids)
    repeat = build_one_step_macro_dataset(
        models={}, episode=episode, candidate_schedule=locked,
        parent_batch_size=5, macro_batch_size=64, device=torch.device("cpu"),
        output_dir=tmp_path / "repeat", candidate_ids=ids,
        allow_replenishment=False,
    )
    assert repeat.pairing_audit["candidate_coverage"]["accepted_candidate_ids"] == ids
    assert repeat.pairing_audit["candidate_coverage"]["attempted_candidate_count"] == len(ids)


def test_budget_exhaustion_returns_audited_partial_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bank_module,
        "_evaluate_boundary_candidates_with_isolation",
        _fake_candidate_evaluator,
    )
    base = _schedule("train")
    schedule = bank_module.MacroCandidateSchedule(
        split=base.split,
        namespace=base.namespace,
        target_valid=30,
        initial_candidate_budget=30,
        max_candidate_budget=32,
        minimum_partial_valid=16,
        firms_per_parent=base.firms_per_parent,
        seed=base.seed,
        candidate_specs=base.candidate_specs,
    )
    result = build_one_step_macro_dataset(
        models={}, episode=_FakeEpisode(), candidate_schedule=schedule,
        parent_batch_size=7, macro_batch_size=64, device=torch.device("cpu"),
        output_dir=tmp_path / "partial",
    )
    coverage = result.pairing_audit["candidate_coverage"]
    assert result.dataset_status == "PARTIAL_BOOTSTRAP_TRAINING"
    assert coverage["candidate_budget_exhausted"] is True
    assert coverage["accepted_training_pair_count"] == 23
    assert coverage["replenishment_attempted"] == 2
    assert result.candidate_audit["candidate_id"].is_unique


def test_engineering_failure_is_fail_fast(monkeypatch):
    def broken(**_kwargs):
        raise RuntimeError("schema mismatch")

    monkeypatch.setattr(
        bank_module, "_evaluate_boundary_candidates_with_isolation", broken
    )
    with pytest.raises(RuntimeError, match="schema mismatch"):
        build_one_step_macro_dataset(
            models={}, episode=_FakeEpisode(), candidate_schedule=_schedule("train"),
            parent_batch_size=8, macro_batch_size=64,
            device=torch.device("cpu"),
        )


def test_two_outer_firm_first_control_flow_and_resume():
    fresh = run_firm_first_orchestration_mock(outer_count=2)
    resumed = run_firm_first_orchestration_mock(outer_count=2, resume_after=7)
    expected = list(FIRM_FIRST_STAGE_ORDER) * 2
    assert fresh["full_trace"] == expected
    assert resumed["full_trace"] == expected
    assert resumed["executed_trace"] == expected[7:]


def test_production_stage_recorder_two_outers_and_idempotent_resume(tmp_path):
    for outer in range(2):
        for stage in FIRM_FIRST_STAGE_ORDER:
            _record_firm_first_stage(
                tmp_path, equilibrium_outer=outer, stage=stage
            )
            # Re-entering an already committed stage after resume is a no-op.
            _record_firm_first_stage(
                tmp_path, equilibrium_outer=outer, stage=stage
            )
    payload = __import__("json").loads(
        (tmp_path / "metadata" / "firm_first_stage_trace.json").read_text()
    )
    assert payload["completed_outer_count"] == 2
    assert [entry["stage"] for entry in payload["entries"]] == (
        list(FIRM_FIRST_STAGE_ORDER) * 2
    )

