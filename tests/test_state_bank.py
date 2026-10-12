from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from config import Config
from data import simulate_ts_parallel
from data.simulate_ts import SimulateTS
from data.state_bank import StateBank, StateBankCapture, clone_state
from experiments.run_multi_episode_job import (
    build_and_commit_state_bank,
    configure_hyperparams,
    parse_args,
)


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
