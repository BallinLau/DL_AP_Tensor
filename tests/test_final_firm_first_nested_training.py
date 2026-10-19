from __future__ import annotations

from types import SimpleNamespace
import inspect
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

import analysis.macro_parent_state_bank as bank_module
import analysis.run_full_nested_equilibrium as runner
from analysis.macro_parent_state_bank import (
    B_DISTRIBUTION_STRATA,
    DEFAULT_MAX_INITIAL_WEIGHTED_B_MEAN,
    RESTRICTED_BETA_FULL_OUTER_V1,
    MACRO_CANDIDATE_STATUS_VALID,
    OneStepMacroDataset,
    build_macro_candidate_schedule,
    build_one_step_macro_dataset,
    boundary_macro_rows,
    load_committed_one_step_macro_dataset,
    lock_validation_candidate_ids,
    make_boundary_candidate_initial_state,
    midpoint_beta_mean,
    one_step_macro_dataset_fingerprint,
    save_committed_one_step_macro_dataset,
    _macro_rows_to_table,
)
from analysis.run_full_nested_equilibrium import (
    FIRM_FIRST_STAGE_ORDER,
    _args_fingerprint,
    _build_or_load_fixed_banks,
    _candidate_target_movements,
    _candidate_schedule_overlap,
    _record_firm_first_stage,
    _load_committed_macro_inner_inputs,
    _resume_with_recorded_config,
    main,
    parse_args,
    run_firm_first_orchestration_mock,
)
from analysis.nested_equilibrium_utils import (
    load_checkpoint,
    make_checkpoint_payload,
    publish_checkpoint,
    stable_json_hash,
)
from data.tensor_data import TensorTable
from training.episode import Episode
from training.inner_fixed_point import InnerFPChildBank, InnerFPCollocationBank


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
            metadata=dict(macro.metadata),
        )

    def _create_sdf_batches_from_macro_tensor(self, table, **_kwargs):
        return [{"rows": table.data}] if len(table) else []


def _production_episode_for_pairing() -> Episode:
    episode = Episode.__new__(Episode)
    episode.device = torch.device("cpu")
    episode.train_mode = "2time"
    episode.add_FC1loss = True
    episode.hyperparams = SimpleNamespace(
        fc1_rollout_horizon=0,
        fc1_recursive_aux_training_enabled=False,
        fc1_rollout_weight=0.0,
        fc1_rollout_diagnostic_enabled=False,
        sdf_true_target_batches=4,
        sdf_min_parent_groups_per_batch=1,
    )
    return episode


def _identity_rows(candidate_ids):
    rows = []
    for offset, candidate_id in enumerate(candidate_ids):
        for t, branch in ((0, -1), (1, 0), (1, 1)):
            rows.append({
                "path": candidate_id,
                "candidate_id": candidate_id,
                "lineage_id": candidate_id,
                "t": t,
                "branch": branch,
                "x": 0.01 * offset + 0.001 * branch,
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
    return rows


@pytest.mark.parametrize(
    "candidate_ids",
    [
        [1_000_000_000, 1_000_000_001, 1_000_000_002,
         1_000_000_063, 1_000_000_064, 1_000_000_065],
        [20_000_000, 20_000_001, 20_000_002],
        list(range(1_000_000_000, 1_000_000_512)),
    ],
)
def test_production_episode_pairing_preserves_large_canonical_candidate_ids(
    candidate_ids,
):
    episode = _production_episode_for_pairing()
    macro = _macro_rows_to_table(_identity_rows(candidate_ids), episode.device)
    pairs = Episode._build_sdf_pairs_from_macro_tensor(episode, macro)
    audit = bank_module.audit_candidate_pairing(pairs, candidate_ids)
    batches = Episode._create_sdf_batches_from_macro_tensor(
        episode, pairs, batch_size=64, n_branches=2
    )

    assert audit["each_parent_has_exact_children"] is True
    assert set(map(int, audit["canonical_to_local"])) == set(candidate_ids)
    assert len(set(audit["canonical_to_local"].values())) == len(candidate_ids)
    assert sum(batch["parent"].shape[0] for batch in batches) == len(candidate_ids)
    local_path = pairs.data[:, pairs.columns.index("path")].to(torch.int64)
    assert int(torch.unique(local_path).numel()) == len(candidate_ids)
    assert int(local_path.max()) < 16_777_216


def test_candidate_target_movement_aligns_canonical_not_dataset_local_ids():
    episode = _production_episode_for_pairing()
    candidate_ids = [1_000_000_000, 1_000_000_001]
    old_macro = _macro_rows_to_table(
        _identity_rows(candidate_ids), episode.device
    )
    new_macro = _macro_rows_to_table(
        _identity_rows(list(reversed(candidate_ids))), episode.device
    )
    old_pairs = Episode._build_sdf_pairs_from_macro_tensor(episode, old_macro)
    new_pairs = Episode._build_sdf_pairs_from_macro_tensor(episode, new_macro)
    movement = _candidate_target_movements(old_pairs, new_pairs)
    assert movement["common_candidate_count"] == 2
    assert movement["Hatc_target_move"] == pytest.approx(0.0)
    assert movement["LnK_target_move"] == pytest.approx(0.0)


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
    assert all(
        midpoint_beta_mean(spec.alpha, spec.beta, train.firms_per_parent)
        <= DEFAULT_MAX_INITIAL_WEIGHTED_B_MEAN + 1.0e-12
        for spec in train.candidate_specs
    )
    assert len({spec.distribution_family for spec in train.candidate_specs}) >= 3
    assert _candidate_schedule_overlap(train, validation)["disjoint"] is True


def test_b_only_schedule_can_explicitly_replay_uncapped_legacy_support():
    legacy = build_macro_candidate_schedule(
        split="legacy", namespace="legacy", target_valid=23,
        initial_candidate_budget=32, max_candidate_budget=32,
        minimum_partial_valid=16, firms_per_parent=16, seed=17,
        candidate_id_offset=90_000_000, max_initial_weighted_b_mean=None,
    )
    assert len({spec.distribution_family for spec in legacy.candidate_specs}) == len(
        B_DISTRIBUTION_STRATA
    )
    assert any(
        midpoint_beta_mean(spec.alpha, spec.beta, legacy.firms_per_parent)
        > DEFAULT_MAX_INITIAL_WEIGHTED_B_MEAN
        for spec in legacy.candidate_specs
    )


def test_restricted_full_outer_schedule_uses_declared_ranges_and_exact_mean_cap():
    schedule = build_macro_candidate_schedule(
        split="train", namespace="restricted", target_valid=64,
        initial_candidate_budget=128, max_candidate_budget=256,
        minimum_partial_valid=32, firms_per_parent=200, seed=20291018,
        candidate_id_offset=70_000_000,
        max_initial_weighted_b_mean=0.30,
        distribution_design=RESTRICTED_BETA_FULL_OUTER_V1,
    )
    assert all(
        spec.distribution_family == RESTRICTED_BETA_FULL_OUTER_V1
        for spec in schedule.candidate_specs
    )
    assert all(1.8 <= spec.alpha <= 3.2 for spec in schedule.candidate_specs)
    assert all(4.8 <= spec.beta <= 7.8 for spec in schedule.candidate_specs)
    assert all(
        midpoint_beta_mean(spec.alpha, spec.beta, 200) <= 0.30 + 1.0e-12
        for spec in schedule.candidate_specs
    )


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

    restricted, restricted_manifest, restricted_preservation = (
        make_boundary_candidate_initial_state(
            sim,
            specs,
            firms=16,
            initial_total_k_min=45.0,
            initial_total_k_max=85.0,
        )
    )
    restricted_total_k = torch.exp(restricted["lnkf"])
    assert bool(((restricted_total_k >= 45.0) & (restricted_total_k <= 85.0)).all())
    assert restricted_manifest["K_total"].between(45.0, 85.0).all()
    assert restricted_preservation["initial_K_unchanged"] is False
    assert restricted_preservation["initial_macro_state_distribution_restricted"] is True

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


def test_candidate_resume_loads_committed_shards_without_re_evaluation(
    monkeypatch, tmp_path
):
    schedule = _schedule("train")
    output = tmp_path / "resume"
    monkeypatch.setattr(
        bank_module,
        "_evaluate_boundary_candidates_with_isolation",
        _fake_candidate_evaluator,
    )
    first = build_one_step_macro_dataset(
        models={}, episode=_FakeEpisode(), candidate_schedule=schedule,
        parent_batch_size=4, macro_batch_size=64, device=torch.device("cpu"),
        output_dir=output,
    )

    def must_not_run(**_kwargs):
        raise AssertionError("committed candidate shard was re-evaluated")

    monkeypatch.setattr(
        bank_module,
        "_evaluate_boundary_candidates_with_isolation",
        must_not_run,
    )
    resumed = build_one_step_macro_dataset(
        models={}, episode=_FakeEpisode(), candidate_schedule=schedule,
        parent_batch_size=4, macro_batch_size=64, device=torch.device("cpu"),
        output_dir=output, require_cached_shards=True,
    )
    assert (
        resumed.pairing_audit["candidate_coverage"]["accepted_candidate_ids"]
        == first.pairing_audit["candidate_coverage"]["accepted_candidate_ids"]
    )


def test_candidate_cache_rejects_model_version_mismatch(monkeypatch, tmp_path):
    schedule = _schedule("train")
    output = tmp_path / "model_version"
    monkeypatch.setattr(
        bank_module,
        "_evaluate_boundary_candidates_with_isolation",
        _fake_candidate_evaluator,
    )
    model = torch.nn.Linear(2, 1)
    build_one_step_macro_dataset(
        models={"policy_value": model},
        episode=_FakeEpisode(),
        candidate_schedule=schedule,
        parent_batch_size=4,
        macro_batch_size=64,
        device=torch.device("cpu"),
        output_dir=output,
    )
    with torch.no_grad():
        model.weight.add_(1.0)
    with pytest.raises(
        RuntimeError,
        match="candidate dataset resume provenance mismatch: model_hashes_at_generation",
    ):
        build_one_step_macro_dataset(
            models={"policy_value": model},
            episode=_FakeEpisode(),
            candidate_schedule=schedule,
            parent_batch_size=4,
            macro_batch_size=64,
            device=torch.device("cpu"),
            output_dir=output,
            require_cached_shards=True,
        )


def test_two_outer_firm_first_control_flow_and_resume():
    fresh = run_firm_first_orchestration_mock(outer_count=2)
    resumed = run_firm_first_orchestration_mock(outer_count=2, resume_after=7)
    expected = list(FIRM_FIRST_STAGE_ORDER) * 2
    assert fresh["full_trace"] == expected
    assert resumed["full_trace"] == expected
    assert resumed["executed_trace"] == expected[7:]


def test_resume_restores_original_immutable_configuration_and_rejects_conflict(
    tmp_path,
):
    checkpoint_source = tmp_path / "custom_checkpoint.pt"
    checkpoint_source.write_bytes(b"checkpoint provenance marker")
    run_root = tmp_path / "run"
    (run_root / "metadata").mkdir(parents=True)
    checkpoint = run_root / "stage.pt"
    checkpoint.write_bytes(b"stage")
    original = parse_args([
        "--cold-start",
        "--device", "cpu",
        "--equilibrium-outer-max", "2",
        "--macro-inner-max", "3",
        "--macro-train-target-valid", "2048",
        "--macro-validation-target-valid", "512",
        "--initial-checkpoint", str(checkpoint_source),
        "--seed", "314159",
    ])
    metadata = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(original).items()
        },
        "hyperparameter_fingerprint": _args_fingerprint(original),
    }
    (run_root / "metadata" / "run.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (run_root / "latest_checkpoint.json").write_text(
        json.dumps({"latest": str(checkpoint)}), encoding="utf-8"
    )

    resumed = parse_args(["--resume", str(run_root), "--device", "cpu"])
    resumed, observed_root, observed_checkpoint = _resume_with_recorded_config(
        resumed, ["--resume", str(run_root), "--device", "cpu"]
    )
    assert observed_root == run_root.resolve()
    assert observed_checkpoint == checkpoint.resolve()
    assert resumed.equilibrium_outer_max == 2
    assert resumed.macro_inner_max == 3
    assert resumed.seed == 314159
    assert resumed.initial_checkpoint == checkpoint_source
    assert _args_fingerprint(resumed) == _args_fingerprint(original)

    conflicting = parse_args([
        "--resume", str(run_root), "--device", "cpu", "--seed", "7"
    ])
    with pytest.raises(RuntimeError, match="immutable argument conflict"):
        _resume_with_recorded_config(
            conflicting,
            ["--resume", str(run_root), "--device", "cpu", "--seed", "7"],
        )


def test_committed_macro_dataset_roundtrip_preserves_identity_and_fingerprint(
    tmp_path,
):
    candidate_ids = [1_000_000_000, 1_000_000_001, 1_000_000_064]
    episode = _production_episode_for_pairing()
    macro = _macro_rows_to_table(_identity_rows(candidate_ids), episode.device)
    pairs = Episode._build_sdf_pairs_from_macro_tensor(episode, macro)
    pairing = bank_module.audit_candidate_pairing(pairs, candidate_ids)
    pairing["candidate_coverage"] = {
        "accepted_candidate_ids": candidate_ids,
        "accepted_training_pair_count": len(candidate_ids),
    }
    dataset = OneStepMacroDataset(
        macro=macro,
        pairs=pairs,
        batches={"fc1": [], "sdf": []},
        pairing_audit=pairing,
        used_parent_batch_size=2,
    )
    path = tmp_path / "committed_dataset.pt"
    expected = save_committed_one_step_macro_dataset(path, dataset)
    rng_before = torch.get_rng_state().clone()
    loaded = load_committed_one_step_macro_dataset(
        path,
        episode=episode,
        macro_batch_size=2,
        n_branches=2,
        device=torch.device("cpu"),
    )
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert one_step_macro_dataset_fingerprint(loaded) == expected
    assert loaded.pairing_audit == pairing
    assert torch.equal(loaded.macro.data, dataset.macro.data)
    assert torch.equal(loaded.pairs.data, dataset.pairs.data)


def test_uninterrupted_and_resumed_macro_inner_use_same_committed_output(
    tmp_path,
):
    candidate_ids = [20_000_000, 20_000_001, 20_000_002]
    episode = _production_episode_for_pairing()
    macro = _macro_rows_to_table(_identity_rows(candidate_ids), episode.device)
    pairs = Episode._build_sdf_pairs_from_macro_tensor(episode, macro)
    pairing = bank_module.audit_candidate_pairing(pairs, candidate_ids)
    pairing["candidate_coverage"] = {
        "accepted_candidate_ids": candidate_ids,
        "accepted_training_pair_count": len(candidate_ids),
    }
    output = OneStepMacroDataset(
        macro=macro,
        pairs=pairs,
        batches={"fc1": [], "sdf": []},
        pairing_audit=pairing,
        used_parent_batch_size=2,
    )
    macro_root = tmp_path / "macro"
    inner0 = macro_root / "macro_inner_000"
    train_hash = save_committed_one_step_macro_dataset(
        inner0 / "post_fc1_train" / "committed_dataset.pt", output
    )
    validation_hash = save_committed_one_step_macro_dataset(
        inner0 / "post_fc1_validation" / "committed_dataset.pt", output
    )
    (inner0 / "committed_dataset_manifest.json").write_text(
        json.dumps({
            "committed_stage": "macro_inner_complete",
            "output_dataset_fingerprints": {
                "train": train_hash,
                "validation": validation_hash,
            },
        }),
        encoding="utf-8",
    )
    uninterrupted = {
        "train": one_step_macro_dataset_fingerprint(output),
        "validation": one_step_macro_dataset_fingerprint(output),
    }
    rng_before = torch.get_rng_state().clone()
    resumed_train, resumed_validation, manifest = (
        _load_committed_macro_inner_inputs(
            macro_root=macro_root,
            macro_inner=1,
            episode=episode,
            macro_batch_size=2,
            branch_num=2,
            device=torch.device("cpu"),
        )
    )
    resumed = {
        "train": one_step_macro_dataset_fingerprint(resumed_train),
        "validation": one_step_macro_dataset_fingerprint(resumed_validation),
    }
    assert resumed == uninterrupted
    assert manifest["committed_stage"] == "macro_inner_complete"
    assert torch.equal(torch.get_rng_state(), rng_before)


def _fixed_banks_for_test():
    parent = InnerFPCollocationBank(
        parent_states=torch.zeros((4, 7)),
        source_kind=torch.zeros(4, dtype=torch.long),
        seed=1,
        split="train",
    )
    child = InnerFPChildBank(
        children=torch.zeros((4, 2, 8)),
        eps_x=torch.zeros((4, 2, 1)),
        eps_z=torch.zeros((4, 2, 1)),
        u_eta=torch.zeros((4, 2, 1)),
        u_i=torch.zeros((4, 2, 1)),
        seed=2,
    )
    return parent, child


def test_market_cleared_firm_bootstrap_is_bounded_to_one_step(
    monkeypatch, tmp_path,
):
    observed_horizons = []
    parent, child = _fixed_banks_for_test()
    args = parse_args([
        "--cold-start", "--smoke", "--device", "cpu",
        "--output-dir", str(tmp_path),
    ])
    monkeypatch.setattr(
        runner,
        "make_initial_state",
        lambda **kwargs: {"x": torch.zeros(kwargs["paths"])},
    )

    def one_step_only(**kwargs):
        observed_horizons.append(int(kwargs["horizon"]))
        assert int(kwargs["horizon"]) == 1
        return SimpleNamespace(meta={"horizon": 1})

    monkeypatch.setattr(runner, "run_simulation", one_step_only)
    monkeypatch.setattr(
        runner,
        "build_market_cleared_transition_bank",
        lambda _output, **kwargs: (
            replace(parent, split=kwargs["split"], seed=kwargs["seed"]),
            replace(child, seed=kwargs["seed"]),
            {"split": kwargs["split"], "selected_transitions": kwargs["n_parents"]},
        ),
    )
    _build_or_load_fixed_banks(
        args=args,
        run_root=tmp_path,
        episode=SimpleNamespace(),
        models={},
        device=torch.device("cpu"),
    )
    assert observed_horizons == [1, 1]


def test_real_main_reaches_firm_before_any_unapproved_long_rollout(
    monkeypatch, tmp_path,
):
    class FirmReached(RuntimeError):
        pass

    parent, child = _fixed_banks_for_test()
    policy_value = torch.nn.Linear(7, 3)
    sdf_fc1 = torch.nn.Linear(9, 3)
    episode = SimpleNamespace(
        config=SimpleNamespace(),
        hyperparams=SimpleNamespace(),
        models={"policy_value": policy_value, "sdf_fc1": sdf_fc1},
    )
    monkeypatch.setattr(
        runner, "_setup_fresh",
        lambda _args, _device: (
            episode, SimpleNamespace(), episode.models
        ),
    )
    monkeypatch.setattr(
        runner, "_build_or_load_fixed_banks",
        lambda **_kwargs: (parent, replace(parent, split="validation"), child, child),
    )
    monkeypatch.setattr(
        runner, "make_initial_state",
        lambda **kwargs: {
            "x": torch.arange(kwargs["paths"], dtype=torch.float32)
            + float(kwargs["seed"]),
        },
    )
    monkeypatch.setattr(
        runner, "_build_or_load_macro_probe",
        lambda **_kwargs: {
            "parents": torch.zeros((2, 9)),
            "children": torch.zeros((2, 2, 9)),
        },
    )
    monkeypatch.setattr(
        runner, "_refresh_children",
        lambda **_kwargs: (
            child.children,
            child.children,
            {"parity": {"max_abs_m_difference": 0.0}},
        ),
    )
    monkeypatch.setattr(runner, "_write_metadata", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_git_head", lambda: "test-head")
    monkeypatch.setattr(runner, "_git_branch", lambda: "test-branch")
    monkeypatch.setattr(
        runner,
        "run_simulation",
        lambda **kwargs: (
            (_ for _ in ()).throw(AssertionError("Long rollout forbidden"))
            if int(kwargs["horizon"]) > 1
            else None
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_problem",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FirmReached()),
    )
    with pytest.raises(FirmReached):
        main([
            "--cold-start", "--smoke", "--device", "cpu",
            "--output-dir", str(tmp_path / "run"),
            "--no-run-long-diagnostics",
        ])


def test_post_fc1_train_uses_bounded_replenishment_but_validation_does_not(
    monkeypatch, tmp_path,
):
    schedule = build_macro_candidate_schedule(
        split="train",
        namespace="train",
        target_valid=16,
        initial_candidate_budget=16,
        max_candidate_budget=24,
        minimum_partial_valid=8,
        firms_per_parent=16,
        seed=77,
        candidate_id_offset=20_000_000,
    )

    def evaluator_with_failures(failed_positions):
        def evaluate(*, specs, **_kwargs):
            rows, records, manifests = [], [], []
            for spec in specs:
                failed = spec.distribution_parameter_id in failed_positions
                status = (
                    "PARENT_CLEARING_FAILED"
                    if failed else MACRO_CANDIDATE_STATUS_VALID
                )
                records.append({
                    **spec.to_dict(),
                    "status": status,
                    "failure_reason": status if failed else "",
                    "accepted_for_training": False,
                })
                manifests.append({**spec.to_dict(), "candidate_status": status})
                if not failed:
                    rows.extend(_identity_rows([spec.candidate_id]))
            return rows, records, manifests
        return evaluate

    monkeypatch.setattr(
        bank_module,
        "_evaluate_boundary_candidates_with_isolation",
        evaluator_with_failures(set()),
    )
    pre = build_one_step_macro_dataset(
        models={}, episode=_FakeEpisode(), candidate_schedule=schedule,
        parent_batch_size=8, macro_batch_size=32, device=torch.device("cpu"),
        output_dir=tmp_path / "pre",
    )
    pre_ids = pre.pairing_audit["candidate_coverage"]["accepted_candidate_ids"]
    assert len(pre_ids) == 16

    monkeypatch.setattr(
        bank_module,
        "_evaluate_boundary_candidates_with_isolation",
        evaluator_with_failures({0, 1, 2, 3}),
    )
    replenished = build_one_step_macro_dataset(
        models={}, episode=_FakeEpisode(), candidate_schedule=schedule,
        parent_batch_size=8, macro_batch_size=32, device=torch.device("cpu"),
        output_dir=tmp_path / "post_train", candidate_ids=pre_ids,
        allow_replenishment=True,
    )
    fixed = build_one_step_macro_dataset(
        models={}, episode=_FakeEpisode(), candidate_schedule=schedule,
        parent_batch_size=8, macro_batch_size=32, device=torch.device("cpu"),
        output_dir=tmp_path / "post_validation", candidate_ids=pre_ids,
        allow_replenishment=False,
    )
    replenished_ids = replenished.pairing_audit["candidate_coverage"][
        "accepted_candidate_ids"
    ]
    fixed_ids = fixed.pairing_audit["candidate_coverage"]["accepted_candidate_ids"]
    assert len(replenished_ids) == 16
    assert len(set(replenished_ids)) == 16
    assert set(pre_ids[:4]).isdisjoint(replenished_ids)
    assert set(replenished_ids).difference(pre_ids) == {
        20_000_016, 20_000_017, 20_000_018, 20_000_019
    }
    assert fixed_ids == pre_ids[4:]


def test_resume_restores_optimizer_state_without_restart(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = model(torch.ones((4, 2))).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    expected_optimizer_hash = stable_json_hash(optimizer.state_dict())
    payload = make_checkpoint_payload(
        models={"model": model},
        optimizers={"model": optimizer},
        state={"macro_inner": 1},
        git_head="review-fix",
        hyperparameter_fingerprint="immutable-config",
    )
    path = tmp_path / "stage.pt"
    publish_checkpoint(path=path, payload=payload, run_root=tmp_path)

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.5)
    loaded = load_checkpoint(
        path,
        models={"model": restored_model},
        optimizers={"model": restored_optimizer},
        git_head="review-fix",
        hyperparameter_fingerprint="immutable-config",
        allow_code_mismatch=False,
    )
    assert loaded["optimizer_state_restored"] is True
    assert stable_json_hash(restored_optimizer.state_dict()) == expected_optimizer_hash


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

