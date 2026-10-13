from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from config import Config, HyperParams
from data import simulate_ts_parallel
from data.entry import compute_entry_reference
from data.simulate_ts import SimulateTS
from data.state_bank import (
    StateBank,
    StateBankCapture,
    StateBankRolloutChunk,
    StateBankRolloutPlan,
    clone_state,
    summarize_parent_transition_budget,
    summarize_state_bank_rollout_diagnostics,
)
from data.tensor_data import TensorSimulationOutput, TensorTable
from experiments.run_multi_episode_job import (
    build_and_commit_state_bank,
    configure_hyperparams,
    evaluate_state_bank_acceptance,
    parse_args,
)
import experiments.run_multi_episode_job as job_module
from training.pv_mixture import PVParentGroupPool, split_pool_by_lineage
from training.episode import Episode


def _compatibility(*, branch_num: int = 2, marker: str = "baseline") -> dict:
    return {
        "format_version": "state_bank_v1",
        "branch_num": branch_num,
        "main_branch": 0,
        "marker": marker,
    }


def _simulator(
    *,
    n_paths: int = 2,
    group_size: int = 2,
    horizon: int = 2,
    initial_state=None,
    capture=None,
) -> SimulateTS:
    return SimulateTS(
        models={"policy_value": None, "sdf_fc1": None, "fc2": None},
        config=Config,
        n_paths=n_paths,
        group_size=group_size,
        horizon=horizon,
        branch_num=2,
        main_branch=0,
        enable_entry=False,
        enable_exit=False,
        device=torch.device("cpu"),
        transition_rng_mode="stable_firm_identity",
        common_transition_seed=24680,
        initial_state=initial_state,
        state_bank_capture=capture,
    )


def _fresh_state(*, n_paths: int = 2, group_size: int = 2):
    torch.manual_seed(123)
    sim = _simulator(n_paths=n_paths, group_size=group_size, horizon=1)
    return sim, simulate_ts_parallel._initialize_batched_state(sim, group_size)


def _assert_tensor_equal(left: torch.Tensor, right: torch.Tensor) -> None:
    torch.testing.assert_close(left, right, rtol=0.0, atol=0.0, equal_nan=True)


def test_state_bank_round_trip_captures_complete_state_and_cpu_storage(tmp_path):
    _, state = _fresh_state()
    bank = StateBank(compatibility=_compatibility(), source_episode=1)

    assert bank.capture(state, capture_time=3) == 2
    assert set(bank.field_schema) == set(state)
    assert all(
        tensor.device.type == "cpu"
        for entry in bank.entries
        for tensor in entry["fields"].values()
    )

    restored = bank.stack_entries(bank.entry_ids, device="cpu")
    for name, expected in state.items():
        _assert_tensor_equal(restored[name], expected.cpu())

    path = bank.save(tmp_path / "bank_ep1.pt")
    loaded = StateBank.load(path, expected_compatibility=_compatibility())
    assert loaded.entry_ids == bank.entry_ids
    assert loaded.field_schema == bank.field_schema
    for name, expected in restored.items():
        _assert_tensor_equal(
            loaded.stack_entries(loaded.entry_ids)[name], expected
        )


def test_state_bank_stacks_variable_firm_capacity_with_field_specific_padding():
    sim, state_small = _fresh_state(n_paths=1, group_size=2)
    bank = StateBank(compatibility=_compatibility(), source_episode=1)
    bank.capture(state_small, capture_time=2)

    state_large = simulate_ts_parallel._expand_firm_capacity(
        sim,
        clone_state(state_small),
        required_capacity=4,
    )
    state_large["alive"][0, 2] = True
    state_large["firm_id"][0, 2] = 99
    state_large["b"][0, 2] = 0.75
    bank.capture(state_large, capture_time=3)

    stacked = bank.stack_entries(bank.entry_ids)
    assert stacked["b"].shape == (2, 4)
    assert stacked["alive"].shape == (2, 4)
    assert not bool(stacked["alive"][0, 2:].any())
    assert stacked["firm_id"][0, 2:].tolist() == [-1, -1]
    assert torch.isnan(stacked["entry_P0"][0, 2:]).all()
    assert stacked["b"][1, 2].item() == pytest.approx(0.75)


def test_state_bank_rejects_compatibility_fingerprint_mismatch(tmp_path):
    _, state = _fresh_state(n_paths=1)
    bank = StateBank(compatibility=_compatibility(), source_episode=1)
    bank.capture(state, capture_time=2)
    path = bank.save(tmp_path / "bank.pt")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        StateBank.load(
            path,
            expected_compatibility=_compatibility(marker="changed"),
        )


def test_sampling_is_reproducible_episode_specific_and_rng_neutral():
    _, state = _fresh_state(n_paths=2)
    bank = StateBank(compatibility=_compatibility(), source_episode=1)
    bank.capture(state, capture_time=2)
    bank.capture(state, capture_time=3)

    torch.manual_seed(999)
    rng_before = torch.get_rng_state().clone()
    first = bank.sample_entry_ids(3, seed=17, episode_id=2)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert first == bank.sample_entry_ids(3, seed=17, episode_id=2)
    assert first != bank.sample_entry_ids(3, seed=18, episode_id=2)
    assert first != bank.sample_entry_ids(3, seed=17, episode_id=3)


def test_initial_state_none_preserves_fresh_initializer(monkeypatch):
    state = {
        "b": torch.ones(1, 1),
        "K": torch.ones(1, 1),
        "alive": torch.ones(1, 1, dtype=torch.bool),
    }
    calls = {"initialize": 0}

    def fake_initialize(_sim, _capacity):
        calls["initialize"] += 1
        return clone_state(state)

    def fake_process(_sim, _state, _t, branch_k):
        return torch.zeros(1, 1), torch.zeros(1, 1)

    monkeypatch.setattr(
        simulate_ts_parallel, "_initialize_batched_state", fake_initialize
    )
    monkeypatch.setattr(simulate_ts_parallel, "_process_node_batched", fake_process)
    monkeypatch.setattr(
        simulate_ts_parallel,
        "_expand_branches_batched",
        lambda _sim, current, child_t: [clone_state(current)],
    )
    monkeypatch.setattr(
        simulate_ts_parallel,
        "_annotate_parent_next_leverage_rows",
        lambda *_args, **_kwargs: None,
    )
    sim = SimpleNamespace(
        device=torch.device("cpu"),
        n_paths=1,
        group_size=1,
        horizon=1,
        branch_num=1,
        main_branch=0,
        enable_entry=False,
        enable_exit=False,
        entry_mode="legacy",
        FIRM_COLUMNS=["value"],
        MACRO_COLUMNS=["value"],
        _max_firms_observed=1,
        initial_state=None,
        state_bank_capture=None,
    )

    simulate_ts_parallel.simulate_tensor_parallel(sim)
    assert calls["initialize"] == 1


def test_capture_is_parent_ready_before_node_processing(monkeypatch):
    initial = {
        "b": torch.tensor([[0.25]]),
        "K": torch.ones(1, 1),
        "alive": torch.ones(1, 1, dtype=torch.bool),
    }
    seen = []

    class CaptureSpy:
        def capture_parent_ready(self, state, *, time_index):
            seen.append((time_index, float(state["b"][0, 0])))

    def fake_process(_sim, state, _t, branch_k):
        state["b"][0, 0] = 0.9
        return torch.zeros(1, 1), torch.zeros(1, 1)

    monkeypatch.setattr(
        simulate_ts_parallel,
        "_initialize_batched_state",
        lambda *_args: pytest.fail("fresh initializer must not run"),
    )
    monkeypatch.setattr(simulate_ts_parallel, "_process_node_batched", fake_process)
    monkeypatch.setattr(
        simulate_ts_parallel,
        "_expand_branches_batched",
        lambda _sim, current, child_t: [clone_state(current)],
    )
    monkeypatch.setattr(
        simulate_ts_parallel,
        "_annotate_parent_next_leverage_rows",
        lambda *_args, **_kwargs: None,
    )
    sim = SimpleNamespace(
        device=torch.device("cpu"),
        n_paths=1,
        group_size=1,
        horizon=1,
        branch_num=1,
        main_branch=0,
        enable_entry=False,
        enable_exit=False,
        entry_mode="legacy",
        FIRM_COLUMNS=["value"],
        MACRO_COLUMNS=["value"],
        _max_firms_observed=1,
        initial_state=clone_state(initial),
        state_bank_capture=CaptureSpy(),
    )

    simulate_ts_parallel.simulate_tensor_parallel(sim)
    assert seen == [(0, 0.25)]


def test_episode1_bank_warm_starts_episode2_and_matched_resume_is_exact():
    bank1 = StateBank(compatibility=_compatibility(), source_episode=1)
    capture1 = StateBankCapture(bank1, burn_in=1, capture_stride=1)
    output1 = _simulator(horizon=2, capture=capture1).simulate_tensor()
    assert output1.meta["initial_state_source"] == "fresh"
    assert len(bank1) == 2

    sampled_ids = bank1.sample_entry_ids(2, seed=77, episode_id=2)
    initial = bank1.stack_entries(sampled_ids)
    initial_snapshot = clone_state(initial)
    first = _simulator(initial_state=initial, horizon=2).simulate_tensor()
    second = _simulator(initial_state=initial, horizon=2).simulate_tensor()

    assert first.meta["initial_state_source"] == "state_bank"
    _assert_tensor_equal(first.firm.data, second.firm.data)
    _assert_tensor_equal(first.macro.data, second.macro.data)
    for name, expected in initial_snapshot.items():
        _assert_tensor_equal(initial[name], expected)


def test_failed_bank_refresh_does_not_replace_previous_bank(tmp_path):
    _, state = _fresh_state(n_paths=1)
    previous = StateBank(compatibility=_compatibility(), source_episode=1)
    previous.capture(state, capture_time=2)
    path = previous.save(tmp_path / "bank_ep2.pt")
    before = path.read_bytes()

    class FailingEpisode:
        def build_state_bank_after_success(self, **_kwargs):
            raise RuntimeError("refresh failed")

    with pytest.raises(RuntimeError, match="refresh failed"):
        build_and_commit_state_bank(
            FailingEpisode(),
            output_path=path,
            n_paths=1,
            group_size=1,
            n_branches=2,
            simulate_kwargs={},
        )

    assert path.read_bytes() == before
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_state_bank_cli_is_opt_in_and_preserves_entry_scale(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_multi_episode_job.py",
            "--state-bank-enabled",
            "--state-bank-short-horizon",
            "7",
            "--state-bank-burn-in",
            "2",
            "--state-bank-capture-stride",
            "2",
            "--state-bank-max-entries",
            "11",
            "--state-bank-sample-seed",
            "101",
            "--state-bank-refresh-seed",
            "202",
        ],
    )
    hp = configure_hyperparams(parse_args())

    assert hp.state_bank_enabled is True
    assert hp.state_bank_short_horizon == 7
    assert hp.state_bank_burn_in == 2
    assert hp.state_bank_capture_stride == 2
    assert hp.state_bank_max_entries == 11
    assert hp.state_bank_sample_seed == 101
    assert hp.state_bank_refresh_seed == 202
    assert hp.entry_size_ratio == pytest.approx(1.0)


def test_bootstrap_and_refresh_capture_schedules_are_distinct():
    _, state = _fresh_state(n_paths=1)
    bootstrap_bank = StateBank(compatibility=_compatibility(), source_episode=1)
    bootstrap = StateBankCapture(
        bootstrap_bank, burn_in=4, capture_stride=2
    )
    refresh_bank = StateBank(compatibility=_compatibility(), source_episode=2)
    refresh = StateBankCapture(refresh_bank, burn_in=1, capture_stride=1)
    for time_index in range(7):
        state["economic_time"].fill_(time_index)
        bootstrap.capture_parent_ready(state, time_index=time_index)
        refresh.capture_parent_ready(state, time_index=time_index)
    assert [entry["capture_time"] for entry in bootstrap_bank.entries] == [4, 6]
    assert [entry["capture_time"] for entry in refresh_bank.entries] == list(range(1, 7))


def test_rollout_plan_meets_budget_in_chunks_without_path_namespace_collision():
    _, state = _fresh_state(n_paths=4)
    bank = StateBank(compatibility=_compatibility(), source_episode=1)
    bank.capture(
        state,
        capture_time=3,
        lineage_ids=[f"lineage-{index}" for index in range(4)],
    )
    plan = StateBankRolloutPlan.build(
        bank,
        episode_id=2,
        n_paths_per_chunk=3,
        short_horizon=2,
        parent_transition_budget=13,
        sample_seed=17,
        sampling_mode="uniform_lineage",
        transition_seed=29,
    )
    assert len(plan.chunks) == 3
    assert sum(chunk.n_paths for chunk in plan.chunks) * plan.short_horizon >= 13
    namespaces = [
        range(chunk.path_offset, chunk.path_offset + chunk.n_paths)
        for chunk in plan.chunks
    ]
    flat = [path for namespace in namespaces for path in namespace]
    assert len(flat) == len(set(flat))
    assert plan.summary()["requested_parent_budget"] == 13


def test_episode_multichunk_concat_and_matched_replay_use_same_plan(monkeypatch):
    class FakeSimulator:
        FIRM_COLUMNS = ["path", "t", "branch"]
        MACRO_COLUMNS = ["path", "t", "branch", "LnK"]

        def __init__(self, *, n_paths, horizon, **_kwargs):
            self.n_paths = int(n_paths)
            self.horizon = int(horizon)

        def simulate_tensor(self):
            rows = []
            for time_index in range(self.horizon):
                for path_index in range(self.n_paths):
                    rows.append([path_index, time_index, -1, float(time_index)])
                    rows.append([path_index, time_index + 1, 0, float(time_index + 1)])
                    rows.append([path_index, time_index + 1, 1, float(time_index + 1)])
            table = torch.tensor(rows, dtype=torch.float32)
            return TensorSimulationOutput(
                firm=TensorTable(table[:, :3].clone(), self.FIRM_COLUMNS),
                macro=TensorTable(table.clone(), self.MACRO_COLUMNS),
                meta={"fake": True},
            )

    monkeypatch.setattr("training.episode.SimulateTS", FakeSimulator)
    chunks = (
        StateBankRolloutChunk(
            chunk_id=0,
            path_offset=0,
            entry_ids=("a", "b"),
            lineage_ids=("la", "lb"),
            initial_state_cpu={"b": torch.zeros(2, 1)},
            transition_seed=3,
        ),
        StateBankRolloutChunk(
            chunk_id=1,
            path_offset=2,
            entry_ids=("c",),
            lineage_ids=("lc",),
            initial_state_cpu={"b": torch.zeros(1, 1)},
            transition_seed=3,
        ),
    )
    plan = StateBankRolloutPlan(
        episode_id=2,
        short_horizon=2,
        requested_parent_transitions=6,
        sampling_mode="uniform_lineage",
        chunks=chunks,
    )
    episode = Episode.__new__(Episode)
    episode.models = {}
    episode.config = Config
    episode._prepare_simulation_kwargs = lambda kwargs: dict(kwargs)
    episode._state_bank_rollout_replay_fingerprints = []
    episode._simulate_state_bank_rollout_plan(
        plan,
        group_size=1,
        n_branches=2,
        simulate_kwargs={},
    )
    first_firm = episode.tensor_firm.data.clone()
    paths = torch.unique(
        episode.tensor_macro.data[
            :, list(episode.tensor_macro.columns).index("path")
        ]
    )
    assert paths.tolist() == [0.0, 1.0, 2.0]
    assert episode._last_simulation_meta["actual_parent_transitions"] == 6
    episode._simulate_state_bank_rollout_plan(
        plan,
        group_size=1,
        n_branches=2,
        simulate_kwargs={},
    )
    _assert_tensor_equal(episode.tensor_firm.data, first_firm)
    assert episode._last_simulation_meta["rollout_replay_count"] == 2
    assert episode._last_simulation_meta["matched_entry_ids_equal"] is True


def test_uniform_lineage_sampling_does_not_weight_long_lineage_by_entry_count():
    _, state = _fresh_state(n_paths=1)
    bank = StateBank(compatibility=_compatibility(), source_episode=1)
    for capture_time in range(20):
        bank.capture(
            state,
            capture_time=capture_time,
            lineage_ids=["long"],
            entry_namespace=f"long-{capture_time}",
        )
    bank.capture(
        state,
        capture_time=100,
        lineage_ids=["short"],
        entry_namespace="short",
    )
    sampled = bank.sample_entry_ids(
        2000, seed=7, episode_id=2, mode="uniform_lineage"
    )
    lineages = bank.lineage_ids_for_entries(sampled)
    short_share = lineages.count("short") / len(lineages)
    assert short_share == pytest.approx(0.5, abs=0.05)


def test_lineage_id_persists_when_descendant_bank_is_captured():
    _, state = _fresh_state(n_paths=2)
    parent = StateBank(compatibility=_compatibility(), source_episode=1)
    parent.capture(
        state,
        capture_time=20,
        lineage_ids=["root:path:0", "root:path:1"],
    )
    sampled_ids = parent.sample_entry_ids(
        2,
        seed=41,
        episode_id=2,
        mode="uniform_lineage",
    )
    inherited_lineages = parent.lineage_ids_for_entries(sampled_ids)
    descendant = StateBank(compatibility=_compatibility(), source_episode=2)
    capture = StateBankCapture(descendant, burn_in=1, capture_stride=1)
    capture.set_context(
        lineage_ids=inherited_lineages,
        entry_namespace="refresh-chunk-0",
    )
    restored = parent.stack_entries(sampled_ids)
    capture.capture_parent_ready(restored, time_index=1)

    assert [entry["lineage_id"] for entry in descendant.entries] == inherited_lineages
    assert all(entry["source_episode"] == 2 for entry in descendant.entries)
    assert all(entry["capture_time"] == 1 for entry in descendant.entries)


def test_reservoir_cap_is_deterministic_rng_neutral_and_not_prefix_biased():
    _, state = _fresh_state(n_paths=1)

    def build(seed):
        bank = StateBank(compatibility=_compatibility(), source_episode=1)
        capture = StateBankCapture(
            bank,
            burn_in=1,
            capture_stride=1,
            max_entries=5,
            reservoir_seed=seed,
        )
        for time_index in range(1, 31):
            state["economic_time"].fill_(time_index)
            capture.capture_parent_ready(state, time_index=time_index)
        return bank

    torch.manual_seed(999)
    before = torch.get_rng_state().clone()
    first = build(123)
    assert torch.equal(torch.get_rng_state(), before)
    second = build(123)
    assert first.entry_ids == second.entry_ids
    capture_times = [entry["capture_time"] for entry in first.entries]
    assert len(first) == 5
    assert max(capture_times) > 5


def test_lineage_split_has_zero_overlap_with_multiple_states_per_lineage():
    parent = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    pool = PVParentGroupPool(
        parent=parent,
        children=[parent.clone(), parent.clone()],
        source_id=torch.zeros(6, dtype=torch.long),
        source_index=torch.arange(6),
        lineage_id=torch.tensor([0, 0, 1, 1, 2, 2]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(10)
    train, val, summary = split_pool_by_lineage(pool, 1 / 3, generator)
    assert summary["train_val_lineage_overlap"] == 0
    assert set(train.lineage_id.tolist()).isdisjoint(set(val.lineage_id.tolist()))
    assert len(train) + len(val) == len(pool)


def test_compatibility_mismatch_reports_changed_field(tmp_path):
    _, state = _fresh_state(n_paths=1)
    bank = StateBank(
        compatibility={"format_version": "state_bank_v2", "G": 1.12},
        source_episode=1,
    )
    bank.capture(state, capture_time=2)
    path = bank.save(tmp_path / "bank.pt")
    with pytest.raises(ValueError, match='"G".*1.12.*1.14'):
        StateBank.load(
            path,
            expected_compatibility={"format_version": "state_bank_v2", "G": 1.14},
        )


def test_transition_only_continuous_and_saved_resume_match_exactly(tmp_path):
    continuous_bank = StateBank(compatibility=_compatibility(), source_episode=1)
    continuous_capture = StateBankCapture(
        continuous_bank, burn_in=2, capture_stride=2
    )
    _simulator(horizon=5, capture=continuous_capture).simulate_tensor()
    checkpoint_id = next(
        entry["entry_id"]
        for entry in continuous_bank.entries
        if entry["capture_time"] == 2
    )
    expected_id = next(
        entry["entry_id"]
        for entry in continuous_bank.entries
        if entry["capture_time"] == 4
    )
    checkpoint = continuous_bank.stack_entries([checkpoint_id])
    saved = continuous_bank.save(tmp_path / "resume.pt")
    loaded = StateBank.load(saved)
    resumed_bank = StateBank(compatibility=_compatibility(), source_episode=2)
    resumed_capture = StateBankCapture(resumed_bank, burn_in=2, capture_stride=1)
    _simulator(
        n_paths=1,
        horizon=3,
        initial_state=loaded.stack_entries([checkpoint_id]),
        capture=resumed_capture,
    ).simulate_tensor()
    expected = continuous_bank._entry_index[expected_id]["fields"]
    actual = resumed_bank.entries[0]["fields"]
    for name in expected:
        _assert_tensor_equal(actual[name], expected[name])
    for name, value in checkpoint.items():
        _assert_tensor_equal(value, loaded.stack_entries([checkpoint_id])[name])


def test_entry_identity_draws_survive_bank_save_reload_without_global_rng_use(tmp_path):
    _, state = _fresh_state(n_paths=2)
    sim = SimpleNamespace(
        config=Config,
        entry_rng_seed=86420,
        effective_entry_cost_max=1.0,
        entry_capital_ratio=0.10,
        entry_size_ratio=1.0,
    )
    reference = compute_entry_reference(
        state["K"],
        state["alive"],
        entry_capital_ratio=sim.entry_capital_ratio,
        entry_size_ratio=sim.entry_size_ratio,
    )
    before = torch.get_rng_state().clone()
    expected = simulate_ts_parallel._stable_entry_candidates(sim, state, reference)
    assert torch.equal(torch.get_rng_state(), before)
    bank = StateBank(compatibility=_compatibility(), source_episode=1)
    bank.capture(state, capture_time=2)
    loaded = StateBank.load(bank.save(tmp_path / "entry_resume.pt"))
    resumed = loaded.stack_entries(loaded.entry_ids)
    resumed_reference = compute_entry_reference(
        resumed["K"],
        resumed["alive"],
        entry_capital_ratio=sim.entry_capital_ratio,
        entry_size_ratio=sim.entry_size_ratio,
    )
    actual = simulate_ts_parallel._stable_entry_candidates(
        sim, resumed, resumed_reference
    )
    for name in (
        "counts", "mask", "z", "entry_cost", "K_birth",
        "potential_capital_nominal", "candidate_capital_realized",
    ):
        _assert_tensor_equal(getattr(actual, name), getattr(expected, name))


def _strict_acceptance_summary(
    *,
    bellman_passed,
    formal_sdf_active: bool = False,
    safe_to_continue: bool = True,
    stage_progress: bool = True,
    converged: bool = False,
    strict_passed: bool = False,
):
    modules = {
        "sdf_formal_stage_schedule": {
            "formal_sdf_active": formal_sdf_active,
        },
        "policy_value": {
            "metadata": {
                "policy_value_stage_status": "accepted",
                "q_stage_required_gate_passed": True,
                "bp_distillation_stage": {"status": "accepted"},
            }
        },
    }
    if formal_sdf_active:
        modules["sdf_fc1_post_refresh_gate"] = {
            "strict_passed": strict_passed,
            "sdf_safe_to_continue": safe_to_continue,
            "sdf_stage_progress": stage_progress,
            "sdf_converged": converged,
        }
    return {
        "module_summaries": modules,
        "convergence": {
            "bellman_passed": bellman_passed,
            "policy_passed": True,
            "passed": None if bellman_passed is None else bool(bellman_passed),
        },
    }


def _strict_acceptance_hyperparams(*, sdf_publication_gate_enabled: bool = True):
    return SimpleNamespace(
        state_bank_acceptance_mode="strict",
        state_bank_sdf_publication_gate_enabled=sdf_publication_gate_enabled,
        state_bank_min_unique_lineages=1,
        bellman_conditional_mean_thresh=None,
        bellman_conditional_p90_thresh=None,
    )


def _finite_candidate_bank():
    _, state = _fresh_state(n_paths=1)
    state["accounting_valid"].fill_(True)
    bank = StateBank(compatibility=_compatibility(), source_episode=2)
    bank.capture(state, capture_time=3, lineage_ids=["root:0"])
    return bank


def test_strict_acceptance_treats_unconfigured_bellman_as_diagnostic_only():
    hp = SimpleNamespace(
        state_bank_acceptance_mode="strict",
        state_bank_min_unique_lineages=1,
        bellman_conditional_mean_thresh=None,
        bellman_conditional_p90_thresh=None,
    )
    result = evaluate_state_bank_acceptance(
        episode_summary=_strict_acceptance_summary(bellman_passed=None),
        candidate_bank=_finite_candidate_bank(),
        hyperparams=hp,
    )
    assert result["episode_acceptance_passed"] is True
    assert result["bellman_gate"] == {
        "configured": False,
        "required": False,
        "status": "diagnostic_only",
        "mean_threshold": None,
        "p90_threshold": None,
        "bellman_passed": None,
    }
    assert (
        result["strict_checks"]["bellman_fixed_point_gate"]
        == "not_required_unconfigured"
    )


def test_strict_acceptance_requires_configured_bellman_gate_to_pass():
    hp = SimpleNamespace(
        state_bank_acceptance_mode="strict",
        state_bank_min_unique_lineages=1,
        bellman_conditional_mean_thresh=0.01,
        bellman_conditional_p90_thresh=0.02,
    )
    result = evaluate_state_bank_acceptance(
        episode_summary=_strict_acceptance_summary(bellman_passed=False),
        candidate_bank=_finite_candidate_bank(),
        hyperparams=hp,
    )
    assert result["episode_acceptance_passed"] is False
    assert result["bellman_gate"]["status"] == "failed"
    assert "bellman_required_gate_failed" in result["reasons"]


@pytest.mark.parametrize(
    (
        "safe_to_continue",
        "stage_progress",
        "converged",
        "expected_passed",
        "expected_reason",
    ),
    [
        (True, True, False, True, "safe_and_progressing_not_yet_converged"),
        (True, False, False, False, "sdf_safe_but_no_progress"),
        (True, False, True, True, "sdf_converged"),
        (False, True, False, False, "sdf_unsafe_for_continuation"),
    ],
)
def test_strict_acceptance_uses_sdf_continuation_quality_for_publication(
    safe_to_continue,
    stage_progress,
    converged,
    expected_passed,
    expected_reason,
):
    result = evaluate_state_bank_acceptance(
        episode_summary=_strict_acceptance_summary(
            bellman_passed=None,
            formal_sdf_active=True,
            safe_to_continue=safe_to_continue,
            stage_progress=stage_progress,
            converged=converged,
        ),
        candidate_bank=_finite_candidate_bank(),
        hyperparams=_strict_acceptance_hyperparams(),
    )
    assert result["episode_acceptance_passed"] is expected_passed
    assert result["sdf_publication_gate"] == {
        "formal_sdf_active": True,
        "required": True,
        "safe_to_continue": safe_to_continue,
        "stage_progress": stage_progress,
        "converged": converged,
        "publication_passed": expected_passed,
        "reason": expected_reason,
    }
    if not expected_passed:
        assert expected_reason in result["reasons"]


def test_strict_acceptance_keeps_disabled_sdf_gate_diagnostic_only():
    result = evaluate_state_bank_acceptance(
        episode_summary=_strict_acceptance_summary(
            bellman_passed=None,
            formal_sdf_active=True,
            safe_to_continue=True,
            stage_progress=False,
            converged=False,
        ),
        candidate_bank=_finite_candidate_bank(),
        hyperparams=_strict_acceptance_hyperparams(
            sdf_publication_gate_enabled=False
        ),
    )
    assert result["episode_acceptance_passed"] is True
    assert result["sdf_publication_gate"] == {
        "formal_sdf_active": True,
        "required": False,
        "safe_to_continue": True,
        "stage_progress": False,
        "converged": False,
        "publication_passed": False,
        "reason": "sdf_safe_but_no_progress",
    }
    assert "sdf_publication_gate_passed" not in result["strict_checks"]
    assert "sdf_safe_but_no_progress" not in result["reasons"]


def test_strict_acceptance_ignores_final_strict_failure_when_sdf_is_progressing():
    result = evaluate_state_bank_acceptance(
        episode_summary=_strict_acceptance_summary(
            bellman_passed=None,
            formal_sdf_active=True,
            safe_to_continue=True,
            stage_progress=True,
            converged=False,
            strict_passed=False,
        ),
        candidate_bank=_finite_candidate_bank(),
        hyperparams=_strict_acceptance_hyperparams(),
    )
    assert result["episode_acceptance_passed"] is True
    assert result["strict_checks"]["sdf_publication_gate_passed"] is True
    assert "formal_sdf_strict_gate_missing_or_failed" not in result["reasons"]


def test_calibration_episode_does_not_require_sdf_publication_gate():
    result = evaluate_state_bank_acceptance(
        episode_summary=_strict_acceptance_summary(
            bellman_passed=None,
            formal_sdf_active=False,
        ),
        candidate_bank=_finite_candidate_bank(),
        hyperparams=_strict_acceptance_hyperparams(),
    )
    assert result["episode_acceptance_passed"] is True
    assert result["sdf_publication_gate"] == {
        "formal_sdf_active": False,
        "required": False,
        "safe_to_continue": None,
        "stage_progress": None,
        "converged": None,
        "publication_passed": True,
        "reason": "formal_sdf_inactive_not_required",
    }


@pytest.mark.parametrize(
    ("mean_threshold", "p90_threshold"),
    [(0.01, None), (None, 0.02)],
)
def test_partial_bellman_threshold_configuration_fails_fast(
    monkeypatch,
    mean_threshold,
    p90_threshold,
):
    hp = HyperParams()
    hp.bellman_conditional_mean_thresh = mean_threshold
    hp.bellman_conditional_p90_thresh = p90_threshold
    monkeypatch.setattr(job_module, "build_hyperparams", lambda: hp)
    monkeypatch.setattr("sys.argv", ["run_multi_episode_job.py"])
    with pytest.raises(
        ValueError,
        match="Bellman conditional thresholds must be both configured or both None",
    ):
        configure_hyperparams(parse_args())


def test_bank_capital_summary_separates_forecast_calculated_and_direct_rebuild():
    _, state = _fresh_state(n_paths=1, group_size=2)
    state["lnkf"].fill_(100.0)
    state["lnk_cal"] = torch.full_like(state["lnkf"], 5.0)
    state["hatcf"].fill_(20.0)
    state["hatc_cal"] = torch.full_like(state["hatcf"], 3.0)
    alive_count = int(state["alive"].sum().item())
    state["K"][state["alive"]] = math.exp(6.0) / alive_count
    bank = StateBank(compatibility=_compatibility(), source_episode=1)
    bank.capture(state, capture_time=20, lineage_ids=["root:0"])
    summary = bank.distribution_summary()

    assert summary["lnkf_forecast"]["mean"] == pytest.approx(100.0)
    assert summary["lnk_cal"]["mean"] == pytest.approx(5.0)
    assert summary["log_K_sum"]["mean"] == pytest.approx(6.0, abs=1e-6)
    assert summary["hatcf_forecast"]["mean"] == pytest.approx(20.0)
    assert summary["hatc_calculated"]["mean"] == pytest.approx(3.0)
    assert summary["deprecated_aliases"]["LnK"] == "lnkf_forecast"


def test_bank_snapshots_do_not_report_transition_growth():
    _, state = _fresh_state(n_paths=1)
    state["lnk_cal"] = torch.zeros_like(state["lnkf"])
    bank = StateBank(compatibility=_compatibility(), source_episode=1)
    for capture_time, economic_time, lnk_cal in ((20, 20, 4.0), (35, 35, 9.0)):
        state["economic_time"].fill_(economic_time)
        state["lnk_cal"].fill_(lnk_cal)
        bank.capture(
            state,
            capture_time=capture_time,
            lineage_ids=["same-lineage"],
            entry_namespace=f"t{capture_time}",
        )
    summary = bank.distribution_summary()
    assert "mean_dLnK" not in summary
    assert "snapshot_spacing_difference_not_transition_growth" not in summary


def test_realized_dlnk_uses_matched_parent_main_child_transitions():
    macro = TensorTable(
        data=torch.tensor(
            [
                [0, 0, -1, 4.0],
                [1, 3, -1, 2.0],
                [0, 1, 0, 4.1],
                [1, 4, 0, 1.8],
                [0, 1, 1, 99.0],
            ],
            dtype=torch.float32,
        ),
        columns=["path", "t", "branch", "LnK"],
    )
    result = summarize_state_bank_rollout_diagnostics(
        macro,
        requested_parent_transitions=2,
        main_branch=0,
    )
    realized = result["realized_dlnk_cal"]
    assert realized["n"] == 2
    assert realized["mean"] == pytest.approx(-0.05, abs=1e-6)
    assert realized["p10"] == pytest.approx(-0.17, abs=1e-6)
    assert realized["p90"] == pytest.approx(0.07, abs=1e-6)


def test_parent_transition_budget_ratio_and_shortfall():
    below = summarize_parent_transition_budget(100, 98)
    assert below["parent_transition_budget_ratio"] == pytest.approx(0.98)
    assert below["parent_transition_shortfall"] == 2
    above = summarize_parent_transition_budget(100, 103)
    assert above["parent_transition_budget_ratio"] == pytest.approx(1.03)
    assert above["parent_transition_shortfall"] == 0


def test_strict_acceptance_rejects_missing_gates_and_preserves_bank_file(tmp_path):
    _, state = _fresh_state(n_paths=1)
    previous = StateBank(compatibility=_compatibility(), source_episode=1)
    previous.capture(state, capture_time=2)
    output = previous.save(tmp_path / "bank_ep2.pt")
    before = output.read_bytes()
    candidate = StateBank(compatibility=_compatibility(), source_episode=2)
    candidate.capture(state, capture_time=3, lineage_ids=["root:0"])
    hp = SimpleNamespace(
        state_bank_acceptance_mode="strict",
        state_bank_min_unique_lineages=1,
    )
    summary = {"module_summaries": {}}
    fake_episode = SimpleNamespace(
        hyperparams=hp,
        build_state_bank_after_success=lambda **_kwargs: candidate,
    )
    with pytest.raises(RuntimeError, match="State Bank candidate rejected"):
        build_and_commit_state_bank(
            fake_episode,
            output_path=output,
            n_paths=1,
            group_size=1,
            n_branches=2,
            simulate_kwargs={},
            episode_summary=summary,
            previous_bank=previous,
        )
    result = summary["module_summaries"]["state_bank"]["acceptance"]
    assert result["episode_acceptance_passed"] is False
    assert any("policy_value_stage_passed" in reason for reason in result["reasons"])
    assert output.read_bytes() == before


def test_state_bank_slurm_pins_formal_v2_parameters():
    text = Path(
        "slurm/run_normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_"
        "investment_compare_state_bank.slurm"
    ).read_text(encoding="utf-8")
    for expected in (
        "STATE_BANK_ENABLED=1",
        "STATE_BANK_PARENT_TRANSITION_BUDGET:-40000",
        "STATE_BANK_MAX_ENTRIES:-5000",
        "STATE_BANK_SAMPLING_MODE:-uniform_lineage",
        "STATE_BANK_ACCEPTANCE_MODE:-strict",
        "STATE_BANK_SDF_PUBLICATION_GATE_ENABLED=0",
        "TRANSITION_RNG_MODE=stable_firm_identity",
        "ENTRY_RNG_MODE=stable_economic_identity",
        "ENTRY_SIZE_RATIO=1.0",
    ):
        assert expected in text
