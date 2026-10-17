from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.nested_equilibrium_utils import (
    atomic_write_json,
    load_checkpoint,
    make_checkpoint_payload,
    publish_checkpoint,
    run_with_oom_fallback,
)
from analysis.run_full_nested_equilibrium import parse_args


def _models() -> dict[str, torch.nn.Module]:
    return {
        "policy_value": torch.nn.Linear(2, 2),
        "sdf_fc1": torch.nn.Linear(2, 1),
    }


def test_production_defaults_match_formal_protocol():
    args = parse_args(["--cold-start"])
    assert args.seed == 20261018
    assert args.equilibrium_outer_max == 10
    assert args.firm_inner_max == 8
    assert args.p_batch_size == 1024
    assert args.q_batch_size == 2048
    assert args.macro_batch_size == 4096
    assert args.simulation_paths == args.validation_paths == 400
    assert args.firms_per_path == 200
    assert args.simulation_horizon == 100
    assert args.branch_num == 2
    assert args.q_solver == "picard"
    assert args.sdf_chunks == 4
    assert args.sdf_epochs_per_chunk == 100
    assert args.sdf_recursive_epochs == 0
    assert args.disable_fc2


def test_smoke_overrides_only_scale_and_budget():
    args = parse_args(["--cold-start", "--smoke"])
    assert args.equilibrium_outer_max == 1
    assert args.firm_inner_max == 1
    assert args.p_policy_outer_max == 1
    assert args.q_picard_outer_max == 1
    assert args.macro_inner_max == 1
    assert (args.simulation_paths, args.validation_paths) == (4, 4)
    assert (args.firms_per_path, args.simulation_horizon) == (32, 8)
    assert args.sdf_recursive_epochs == 0
    assert args.disable_fc2


def test_recursive_sdf_and_fc2_are_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--cold-start", "--sdf-recursive-epochs", "1"])
    with pytest.raises(SystemExit):
        parse_args(["--cold-start", "--no-disable-fc2"])


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
