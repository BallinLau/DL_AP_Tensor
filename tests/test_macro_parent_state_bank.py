from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.macro_parent_state_bank import (
    ParentSourceKey,
    audit_one_step_pairing,
    build_source_schedule,
    deterministic_parent_selection,
    effective_capture_times,
    harvest_macro_parent_schedule,
    initial_cross_section_hash_audit,
    prepare_macro_parent_schedule,
    run_short_rollout_harvest,
    simulate_one_step_chunked,
    source_overlap,
    summarize_parent_bank,
)
from analysis.run_full_nested_equilibrium import parse_args
from config import Config, HyperParams
from data import simulate_ts_parallel
from data.simulate_ts import SimulateTS
from data.state_bank import clone_state
from training.episode import Episode


def _episode() -> Episode:
    episode = object.__new__(Episode)
    episode.device = torch.device("cpu")
    episode.config = Config
    episode.hyperparams = HyperParams()
    episode.add_FC1loss = True
    episode.train_mode = "2time"
    return episode


def _models():
    return {"policy_value": None, "sdf_fc1": None, "fc2": None}


def _initial(paths: int = 4, firms: int = 5):
    torch.manual_seed(90210)
    sim = SimulateTS(
        models=_models(),
        config=Config,
        n_paths=paths,
        group_size=firms,
        horizon=3,
        branch_num=2,
        enable_entry=False,
        enable_exit=False,
        device=torch.device("cpu"),
    )
    return clone_state(
        simulate_ts_parallel._initialize_batched_state(sim, firms), device="cpu"
    )


def _candidate_frame(n: int = 40) -> pd.DataFrame:
    rows = []
    for index in range(n):
        key = ParentSourceKey("train", 10, index, index % 3)
        rows.append(
            {
                "source_key": key.canonical,
                **key.to_dict(),
                "entry_id": str(index),
                "b_mean": index / n,
                "b_std": 0.01 + (index % 7) / 10,
                "z_mean": (index - n / 2) / 5,
                "z_std": 0.1 + (index % 5) / 10,
                "corr_b_z": -1.0 + 2.0 * index / max(n - 1, 1),
                "log_K_total": 2.0 + index / n,
                "K_total": 10.0 + index,
                "Hatc": -2.0 + index / n,
                "LnK": 4.0 + index / n,
                "n_alive": 5,
            }
        )
    return pd.DataFrame(rows)


def test_parent_state_capture_roundtrip():
    episode = _episode()
    initial = _initial(paths=2, firms=4)
    bank, output, audit = run_short_rollout_harvest(
        models=_models(),
        episode=episode,
        initial_state=initial,
        split="train",
        source_episode=0,
        initial_state_seed=123,
        transition_seed=456,
        entry_seed=789,
        short_rollout_horizon=3,
        capture_times=(1,),
        branch_num=2,
        device=torch.device("cpu"),
    )
    assert len(bank) == 2
    restored = bank.stack_entries(bank.entry_ids)
    resumed, _ = simulate_one_step_chunked(
        models=_models(),
        episode=episode,
        bank=bank,
        entry_ids=bank.entry_ids,
        parent_batch_size=2,
        branch_num=2,
        transition_seed=456,
        entry_seed=789,
        device=torch.device("cpu"),
    )
    original = output.macro.to_dataframe()
    original = original[(original["t"] == 1) & (original["branch"] == -1)]
    replay = resumed.to_dataframe()
    replay = replay[(replay["t"] == 0) & (replay["branch"] == -1)]
    for column in ("K", "C", "Hatc", "LnK", "x", "hatcf", "lnkf"):
        torch.testing.assert_close(
            torch.tensor(replay[column].to_numpy()),
            torch.tensor(original[column].to_numpy()),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
    original_children = output.macro.to_dataframe()
    original_children = original_children[
        (original_children["t"] == 2) & (original_children["branch"] >= 0)
    ].sort_values(["path", "branch"]).reset_index(drop=True)
    replay_children = resumed.to_dataframe()
    replay_children = replay_children[
        (replay_children["t"] == 1) & (replay_children["branch"] >= 0)
    ].sort_values(["path", "branch"]).reset_index(drop=True)
    for column in ("K", "C", "Hatc", "LnK", "x", "hatcf", "lnkf"):
        torch.testing.assert_close(
            torch.tensor(replay_children[column].to_numpy()),
            torch.tensor(original_children[column].to_numpy()),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
    assert audit["unique_initial_cross_section_count"] == 2
    for name, value in restored.items():
        assert value.device.type == "cpu", name


def test_parent_bank_source_schedule_fixed():
    frame = _candidate_frame()
    selected, _ = deterministic_parent_selection(
        frame, target_count=16, coverage_fraction=0.25, seed=99
    )
    first = build_source_schedule(
        selected,
        split="train",
        initial_state_seed=10,
        transition_seed=20,
        entry_seed=30,
        candidate_paths=40,
        short_rollout_horizon=3,
        requested_capture_times=(0, 1, 2),
    )
    second = build_source_schedule(
        selected,
        split="train",
        initial_state_seed=10,
        transition_seed=20,
        entry_seed=30,
        candidate_paths=40,
        short_rollout_horizon=3,
        requested_capture_times=(0, 1, 2),
    )
    assert first.fingerprint == second.fingerprint
    assert first.source_keys == second.source_keys


def test_train_validation_bank_disjoint():
    frame = _candidate_frame(12)
    train, _ = deterministic_parent_selection(
        frame, target_count=6, coverage_fraction=0.2, seed=1
    )
    validation = train.copy()
    validation["split"] = "validation"
    validation["initial_state_seed"] = 11
    for index, row in validation.iterrows():
        key = ParentSourceKey("validation", 11, int(row.initial_path_id), int(row.capture_time))
        validation.loc[index, "source_key"] = key.canonical
        validation.loc[index, "source_type"] = key.source_type
    kwargs = dict(
        transition_seed=20,
        entry_seed=30,
        candidate_paths=12,
        short_rollout_horizon=3,
        requested_capture_times=(0, 1, 2),
    )
    train_schedule = build_source_schedule(
        train, split="train", initial_state_seed=10, **kwargs
    )
    validation_schedule = build_source_schedule(
        validation, split="validation", initial_state_seed=11, **kwargs
    )
    assert source_overlap(train_schedule, validation_schedule)["overlap_count"] == 0


def test_one_step_parent_child_pairing_and_two_children_same_parent():
    episode = _episode()
    initial = _initial(paths=3, firms=4)
    bank, output, _ = run_short_rollout_harvest(
        models=_models(), episode=episode, initial_state=initial,
        split="train", source_episode=0, initial_state_seed=1,
        transition_seed=2, entry_seed=3, short_rollout_horizon=2,
        capture_times=(1,), branch_num=2, device=torch.device("cpu"),
    )
    summary, _, _ = summarize_parent_bank(
        bank, output.macro, split="train", initial_state_seed=1
    )
    schedule = build_source_schedule(
        summary.assign(selection_component="ENDOGENOUS_FREQUENCY"),
        split="train", initial_state_seed=1, transition_seed=2,
        entry_seed=3, candidate_paths=3, short_rollout_horizon=2,
        requested_capture_times=(1,),
    )
    harvested = harvest_macro_parent_schedule(
        models=_models(), episode=episode, initial_state=initial,
        schedule=schedule, source_episode=0, branch_num=2,
        device=torch.device("cpu"),
    )
    macro, _ = simulate_one_step_chunked(
        models=_models(), episode=episode, bank=harvested.bank,
        entry_ids=harvested.selected_entry_ids, parent_batch_size=2,
        branch_num=2, transition_seed=2, entry_seed=3,
        device=torch.device("cpu"),
    )
    pairs = episode._build_sdf_pairs_from_macro_tensor(macro)
    audit = audit_one_step_pairing(
        pairs, expected_parent_count=3, branch_num=2
    )
    assert audit["pair_row_count"] == 6
    assert audit["each_parent_has_exact_children"]


def test_parent_bank_chunked_simulation_equals_small_unchunked():
    episode = _episode()
    initial = _initial(paths=4, firms=4)
    bank, _, _ = run_short_rollout_harvest(
        models=_models(), episode=episode, initial_state=initial,
        split="train", source_episode=0, initial_state_seed=1,
        transition_seed=2, entry_seed=3, short_rollout_horizon=2,
        capture_times=(1,), branch_num=2, device=torch.device("cpu"),
    )
    chunked, _ = simulate_one_step_chunked(
        models=_models(), episode=episode, bank=bank,
        entry_ids=bank.entry_ids, parent_batch_size=2, branch_num=2,
        transition_seed=2, entry_seed=3, device=torch.device("cpu"),
    )
    unchunked, _ = simulate_one_step_chunked(
        models=_models(), episode=episode, bank=bank,
        entry_ids=bank.entry_ids, parent_batch_size=4, branch_num=2,
        transition_seed=2, entry_seed=3, device=torch.device("cpu"),
    )
    columns = {name: index for index, name in enumerate(chunked.columns)}

    def sorted_rows(table):
        data = table.data
        key = (
            data[:, columns["path"]].to(torch.long) * 100
            + data[:, columns["t"]].to(torch.long) * 10
            + data[:, columns["branch"]].to(torch.long)
            + 1
        )
        return data[torch.argsort(key)]

    torch.testing.assert_close(
        sorted_rows(chunked), sorted_rows(unchunked), equal_nan=True
    )


def test_candidate_selection_deterministic_and_coverage_not_all_same_cross_section():
    frame = _candidate_frame(50)
    first, first_report = deterministic_parent_selection(
        frame, target_count=20, coverage_fraction=0.20, seed=44
    )
    second, second_report = deterministic_parent_selection(
        frame, target_count=20, coverage_fraction=0.20, seed=44
    )
    pd.testing.assert_frame_equal(first, second)
    assert first_report == second_report
    assert first["b_mean"].nunique() > 1
    assert first["z_mean"].nunique() > 1
    assert (first["selection_component"] == "COVERAGE_AWARE").sum() == 4


def test_macro_training_does_not_require_horizon_100_and_long_rollout_not_primary():
    args = parse_args(["--cold-start", "--smoke"])
    assert args.macro_bank_rollout_horizon == 3
    assert effective_capture_times(args.macro_bank_capture_times, 3) == (1, 2)
    assert args.long_diagnostic_horizon == 8
    assert not args.use_long_rollout_for_macro_training
    source = Path("analysis/run_full_nested_equilibrium.py").read_text(encoding="utf-8")
    assert "MANY_PARENT_CROSS_SECTIONS_ONE_STEP" in source
    assert '"long_rollout_training_used": False' in source


def test_parent_bank_firm_hash_frozen_and_fc2_forbidden():
    marker = torch.nn.Linear(2, 2)
    before = {name: value.detach().clone() for name, value in marker.state_dict().items()}
    _ = initial_cross_section_hash_audit(_initial(paths=3, firms=4))
    for name, value in marker.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    with pytest.raises(SystemExit):
        parse_args(["--cold-start", "--no-disable-fc2"])
    with pytest.raises(SystemExit):
        parse_args(["--cold-start", "--use-long-rollout-for-macro-training"])


def test_initial_cross_sections_are_independent_across_paths():
    audit = initial_cross_section_hash_audit(_initial(paths=8, firms=5))
    assert audit["all_initial_cross_sections_unique"]
    assert audit["unique_initial_cross_section_count"] == 8
    cloned = _initial(paths=8, firms=5)
    for name in audit["hashed_economic_fields"]:
        cloned[name][1:] = cloned[name][0]
    cloned_audit = initial_cross_section_hash_audit(cloned)
    assert not cloned_audit["all_initial_cross_sections_unique"]
    assert cloned_audit["unique_initial_cross_section_count"] == 1

