from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.nested_equilibrium_utils import (
    atomic_write_json,
    capture_episode_runtime_state,
    capture_training_transaction,
    load_checkpoint,
    make_checkpoint_payload,
    publish_checkpoint,
    restore_training_transaction,
    run_with_oom_fallback,
    stable_json_hash,
)
from analysis.run_full_nested_equilibrium import (
    _configure_hyperparams,
    _one_step_target_movements,
    parse_args,
)
from data.tensor_data import TensorTable


def _models() -> dict[str, torch.nn.Module]:
    return {
        "policy_value": torch.nn.Linear(2, 2),
        "sdf_fc1": torch.nn.Linear(2, 1),
    }


class _Scheduler:
    def __init__(self):
        self.step_count = 0

    def step(self):
        self.step_count += 1

    def state_dict(self):
        return {"step_count": self.step_count}

    def load_state_dict(self, state):
        self.step_count = int(state["step_count"])


class _RuntimeEpisode:
    def __init__(self):
        self._sdf_al_state = {
            "lambda_mean_low": 0.1,
            "lambda_mean_high": 0.2,
            "lambda_var_high": 0.3,
            "rho": 2.0,
        }
        self.lr_schedulers = {"model": _Scheduler()}
        self.weight_scheduler = _Scheduler()
        self.sdf_training_phase = "sdf_true_only"
        self._checkpoint_continuation_counters = {
            "accepted_primal_epochs_since_dual_update": 2,
            "last_dual_update_epoch": 7,
        }

    def _get_sdf_al_state(self):
        return dict(self._sdf_al_state)

    def set_sdf_training_phase(self, phase):
        self.sdf_training_phase = phase


def test_production_defaults_match_formal_protocol():
    args = parse_args(["--cold-start"])
    assert args.seed == 20261018
    assert args.equilibrium_outer_max == 10
    assert args.firm_inner_max == 8
    assert args.p_batch_size == 1024
    assert args.q_batch_size == 2048
    assert args.macro_batch_size == 4096
    assert args.macro_train_target_valid == 2048
    assert args.macro_validation_target_valid == 512
    assert args.macro_train_initial_candidate_budget == 4096
    assert args.macro_train_max_candidate_budget == 8192
    assert args.macro_validation_initial_candidate_budget == 1024
    assert args.macro_validation_max_candidate_budget == 2048
    assert args.macro_minimum_partial_train == 1024
    assert args.macro_minimum_partial_validation == 256
    assert args.macro_parent_batch_size == 256
    assert args.long_diagnostic_paths == 50
    assert args.long_diagnostic_horizon == 100
    assert not args.use_long_rollout_for_macro_training
    assert not args.run_long_diagnostics
    assert args.market_clearing_mode == "market_cleared_experimental_v1"
    assert args.aggregation_mode == "continuum_mass_consistent_v1"
    assert args.firms_per_path == 200
    assert args.branch_num == 2
    assert args.q_solver == "picard"
    assert args.sdf_chunks == 4
    assert args.sdf_epochs_per_chunk == 100
    assert args.sdf_recursive_epochs == 0
    assert args.disable_fc2


def test_configured_macro_clearing_solver_reaches_simulation_hyperparams():
    args = parse_args([
        "--cold-start",
        "--macro-clearing-solver-mode",
        "fc1_bounded_improvement_pilot",
    ])
    hp = _configure_hyperparams(args)
    assert hp.macro_clearing_solver_mode == "fc1_bounded_improvement_pilot"


def test_smoke_overrides_only_scale_and_budget():
    args = parse_args(["--cold-start", "--smoke"])
    assert args.equilibrium_outer_max == 1
    assert args.firm_inner_max == 1
    assert args.p_policy_outer_max == 1
    assert args.q_picard_outer_max == 1
    assert args.macro_inner_max == 1
    assert (args.simulation_paths, args.validation_paths) == (4, 4)
    assert (args.firms_per_path, args.simulation_horizon) == (32, 8)
    assert args.macro_train_target_valid == 8
    assert args.macro_validation_target_valid == 4
    assert args.macro_train_initial_candidate_budget == 16
    assert args.macro_train_max_candidate_budget == 32
    assert args.macro_validation_initial_candidate_budget == 8
    assert args.macro_validation_max_candidate_budget == 16
    assert args.macro_parent_batch_size == 4
    assert args.sdf_recursive_epochs == 0
    assert args.disable_fc2


def test_recursive_sdf_and_fc2_are_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--cold-start", "--sdf-recursive-epochs", "1"])
    with pytest.raises(SystemExit):
        parse_args(["--cold-start", "--no-disable-fc2"])


def test_market_clearing_locks_audited_production_economic_configuration():
    args = parse_args([
        "--cold-start",
        "--market-clearing-mode",
        "market_cleared_experimental_v1",
    ])
    hp = _configure_hyperparams(args)

    assert hp.entry_mode == "investment_compare"
    assert hp.entry_spec_version == "investment_compare_v1"
    assert hp.entry_capital_ratio == pytest.approx(0.05)
    assert hp.entry_size_ratio == pytest.approx(1.0)
    assert hp.entry_cost_max == pytest.approx(0.5)
    assert hp.entry_dummy_i == pytest.approx(0.0)
    assert hp.entry_rng_mode == "stable_economic_identity"
    assert hp.transition_rng_mode == "stable_firm_identity"
    assert hp.consumption_aggregation_mode == "raw"
    assert hp.consumption_floor == pytest.approx(0.0)
    assert hp.node_accounting_mode == "economic_node_ledger"
    assert hp.simulation_bp_action_source == "grid"
    assert hp.simulation_bp_grid_n_child_shocks == 2


def test_one_step_target_movement_aligns_common_sources_after_censoring():
    columns = ["path", "t", "branch", "Hatc_t1", "LnK_t1"]

    def table(path_start: int, offset: float) -> TensorTable:
        rows = []
        for local_path in range(3):
            for branch in range(2):
                rows.append([
                    path_start + local_path,
                    1,
                    branch,
                    10.0 + local_path + branch + offset,
                    20.0 + local_path + branch + offset,
                ])
        return TensorTable(data=torch.tensor(rows), columns=columns)

    old_summary = pd.DataFrame({"source_key": ["a", "b", "c"]})
    new_summary = pd.DataFrame({"source_key": ["b", "c", "d"]})
    movement = _one_step_target_movements(
        table(100, 0.0),
        table(200, 0.25),
        old_summary,
        new_summary,
    )

    assert movement["old_source_count"] == 3
    assert movement["new_source_count"] == 3
    assert movement["common_source_count"] == 2
    assert movement["common_pair_row_count"] == 4
    assert movement["movement_alignment"] == "COMMON_FIXED_SOURCE_KEYS_ONLY"
    assert movement["Hatc_target_move"] > 0.0


def test_a800_launcher_enforces_single_gpu_and_formal_modes():
    source = (ROOT / "slurm" / "run_full_nested_equilibrium_a800.slurm").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH -p a01" in source
    assert "#SBATCH --gres=gpu:a800:1" in source
    assert "export CUDA_VISIBLE_DEVICES=0" in source
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in source
    assert "--sdf-recursive-epochs 0" in source
    assert "--disable-fc2" in source
    assert "--q-solver picard" in source
    assert "MACRO_PRIMARY_TRAINING=MANY_PARENT_ONE_STEP" in source
    assert "--macro-bank-candidate-paths 1024" in source
    assert "--macro-train-parent-count 4096" in source
    assert "--long-diagnostic-horizon 100" in source
    assert "--no-use-long-rollout-for-macro-training" in source
    assert 'REQUIRED_COMMIT="${REQUIRED_COMMIT:?' in source
    assert 'RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"' in source
    assert 'MODE_ARGS=(--resume "$RESUME_CHECKPOINT")' in source
    assert "TEE_ARGS=(-a)" in source
    assert "--allow-code-mismatch" in source


def test_atomic_json_has_no_partial_generation(tmp_path: Path):
    path = tmp_path / "state.json"
    atomic_write_json(path, {"generation": 1})
    atomic_write_json(path, {"generation": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"generation": 2}
    assert not path.with_name("state.json.tmp").exists()


def test_checkpoint_roundtrip_latest_previous_and_hashes(tmp_path: Path):
    models = _models()
    first = make_checkpoint_payload(
        models=models, state={"equilibrium_outer": 0}, git_head="abc",
        hyperparameter_fingerprint="fp", extra={"bank_hashes": {"fixed": "one"}},
    )
    first_path = tmp_path / "outer0.pt"
    publish_checkpoint(path=first_path, payload=first, run_root=tmp_path)
    with torch.no_grad():
        models["policy_value"].weight.add_(1.0)
    second = make_checkpoint_payload(
        models=models, state={"equilibrium_outer": 1}, git_head="abc",
        hyperparameter_fingerprint="fp", extra={"bank_hashes": {"fixed": "one"}},
    )
    second_path = tmp_path / "outer1.pt"
    publish_checkpoint(path=second_path, payload=second, run_root=tmp_path)
    pointer = json.loads((tmp_path / "latest_checkpoint.json").read_text(encoding="utf-8"))
    assert pointer["latest"] == str(second_path.resolve())
    assert pointer["previous"] == str(first_path.resolve())
    restored = _models()
    loaded = load_checkpoint(
        second_path, models=restored, git_head="abc",
        hyperparameter_fingerprint="fp", allow_code_mismatch=False,
    )
    assert loaded["state"]["equilibrium_outer"] == 1
    assert torch.equal(restored["policy_value"].weight, models["policy_value"].weight)


def test_resume_rejects_provenance_mismatch(tmp_path: Path):
    models = _models()
    payload = make_checkpoint_payload(
        models=models, state={}, git_head="old", hyperparameter_fingerprint="old-fp",
    )
    path = tmp_path / "checkpoint.pt"
    publish_checkpoint(path=path, payload=payload, run_root=tmp_path)
    with pytest.raises(RuntimeError, match="resume provenance mismatch"):
        load_checkpoint(
            path, models=_models(), git_head="new",
            hyperparameter_fingerprint="new-fp", allow_code_mismatch=False,
        )


def test_oom_fallback_reloads_and_retries_once(tmp_path: Path):
    calls: list[int] = []
    reloads: list[bool] = []
    events: list[dict[str, object]] = []

    def execute(batch_size: int) -> str:
        calls.append(batch_size)
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return "ok"

    result, used = run_with_oom_fallback(
        block="P", initial_batch_size=1024, fallback_batch_size=512,
        execute=execute, reload_block_start=lambda: reloads.append(True),
        events=events, events_path=tmp_path / "oom_events.json",
        device=torch.device("cpu"),
        indices={"equilibrium_outer": 1, "firm_inner": 2, "macro_inner": -1},
    )
    assert result == "ok"
    assert used == 512
    assert calls == [1024, 512]
    assert reloads == [True]
    assert events[0]["block"] == "P"
    assert events[0]["old_batch_size"] == 1024
    assert events[0]["new_batch_size"] == 512


def test_oom_second_failure_hard_stops(tmp_path: Path):
    def execute(_batch_size: int):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="persisted after one fallback"):
        run_with_oom_fallback(
            block="MACRO", initial_batch_size=4096, fallback_batch_size=2048,
            execute=execute, reload_block_start=lambda: None, events=[],
            events_path=tmp_path / "oom_events.json", device=torch.device("cpu"),
            indices={},
        )


def test_checkpoint_restores_complete_episode_al_scheduler_and_counters(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    episode = _RuntimeEpisode()
    episode.lr_schedulers["model"].step()
    episode.weight_scheduler.step()
    episode._sdf_al_state["lambda_mean_low"] = 0.75
    loss = model(torch.ones((4, 2))).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    payload = make_checkpoint_payload(
        models={"model": model},
        optimizers={"model": optimizer},
        schedulers=episode.lr_schedulers,
        episode_runtime_state=capture_episode_runtime_state(episode),
        state={"sdf_chunk": 3, "al_active_phase": "sdf_true_only"},
        git_head="resume-test",
        hyperparameter_fingerprint="fixed",
    )
    path = tmp_path / "complete.pt"
    publish_checkpoint(path=path, payload=payload, run_root=tmp_path)

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.5)
    restored_episode = _RuntimeEpisode()
    restored_episode._sdf_al_state = {
        "lambda_mean_low": 9.0,
        "lambda_mean_high": 9.0,
        "lambda_var_high": 9.0,
        "rho": 9.0,
    }
    loaded = load_checkpoint(
        path,
        models={"model": restored_model},
        optimizers={"model": restored_optimizer},
        schedulers=restored_episode.lr_schedulers,
        episode=restored_episode,
        git_head="resume-test",
        hyperparameter_fingerprint="fixed",
        allow_code_mismatch=False,
    )
    assert loaded["state"]["sdf_chunk"] == 3
    assert restored_episode._get_sdf_al_state() == episode._get_sdf_al_state()
    assert restored_episode.lr_schedulers["model"].state_dict() == {"step_count": 1}
    assert restored_episode.weight_scheduler.state_dict() == {"step_count": 1}
    assert restored_episode._checkpoint_continuation_counters == (
        episode._checkpoint_continuation_counters
    )
    assert stable_json_hash(restored_optimizer.state_dict()) == stable_json_hash(
        optimizer.state_dict()
    )


def test_oom_transaction_rolls_back_model_optimizer_al_scheduler_and_rng(tmp_path):
    torch.manual_seed(1234)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    episode = _RuntimeEpisode()
    snapshot = capture_training_transaction(
        models={"model": model},
        optimizers={"model": optimizer},
        episode=episode,
        stage_progress={"stage": "sdf"},
        dataset_provenance={"fingerprint": "immutable"},
    )
    calls = 0
    cleanup_calls = 0

    def one_step(target_model, target_optimizer, target_episode):
        x = torch.randn((8, 3))
        target_optimizer.zero_grad(set_to_none=True)
        loss = target_model(x).square().mean()
        loss.backward()
        target_optimizer.step()
        target_episode._sdf_al_state["lambda_mean_low"] += 0.1
        target_episode.lr_schedulers["model"].step()
        return float(loss.detach())

    def execute(_batch_size):
        nonlocal calls
        calls += 1
        value = one_step(model, optimizer, episode)
        if calls == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return value

    def cleanup():
        nonlocal cleanup_calls
        cleanup_calls += 1

    run_with_oom_fallback(
        block="SDF",
        initial_batch_size=64,
        fallback_batch_size=32,
        execute=execute,
        reload_block_start=lambda: restore_training_transaction(
            snapshot,
            models={"model": model},
            optimizers={"model": optimizer},
            episode=episode,
        ),
        cleanup_failed_attempt=cleanup,
        events=[],
        events_path=tmp_path / "oom.json",
        device=torch.device("cpu"),
        indices={"macro_inner": 0},
    )
    actual_model = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    actual_optimizer_hash = stable_json_hash(optimizer.state_dict())
    actual_runtime = capture_episode_runtime_state(episode)
    actual_rng = torch.get_rng_state().clone()

    reference_model = torch.nn.Linear(3, 1)
    reference_optimizer = torch.optim.Adam(reference_model.parameters(), lr=0.01)
    reference_episode = _RuntimeEpisode()
    restore_training_transaction(
        snapshot,
        models={"model": reference_model},
        optimizers={"model": reference_optimizer},
        episode=reference_episode,
    )
    one_step(reference_model, reference_optimizer, reference_episode)

    assert calls == 2
    assert cleanup_calls == 1
    for key, value in reference_model.state_dict().items():
        torch.testing.assert_close(actual_model[key], value, rtol=0.0, atol=0.0)
    assert actual_optimizer_hash == stable_json_hash(reference_optimizer.state_dict())
    assert actual_runtime == capture_episode_runtime_state(reference_episode)
    torch.testing.assert_close(actual_rng, torch.get_rng_state(), rtol=0.0, atol=0.0)
