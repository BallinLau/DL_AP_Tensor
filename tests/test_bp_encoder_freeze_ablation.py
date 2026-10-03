from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.hyperparams import HyperParams
from evaluation.full_run_diagnostics import model_state_hash
from experiments.run_bp_encoder_freeze_ablation import (
    ARM_PREFIXES,
    active_cache_indices,
    build_step_schedule,
    cache_hash,
    clone_paired_students,
    configure_trainable_parameters,
    evaluate_cache,
    execute_after_initial_reproduction_gate,
    make_loss_adapter,
    optimizer_parameter_names,
    resolve_checkpoint,
    run_arm,
    validate_initial_evaluator_reproduction,
    validate_stage_checkpoint_metadata,
    validate_cache_teacher,
    validate_cache,
)
from models.policy_value import PolicyValueModel
from training.episode import Episode


def _hyperparams() -> HyperParams:
    hp = HyperParams()
    hp.bp_grid_policy_loss_space = "logit"
    hp.bp_grid_logit_target_eps = 1e-4
    hp.bp_grid_logit_huber_delta = 1.0
    hp.bp_grid_policy_huber_delta = 0.05
    hp.bp_grid_policy_weight = 1.0
    hp.bp_grid_mix_policy_weight = 1.0
    hp.bp_distill_grad_clip_norm = 10.0
    hp.pv_grad_hard_threshold = 1000.0
    hp.policy_weight_decay = 0.0
    return hp


def _model(seed: int = 7) -> PolicyValueModel:
    torch.manual_seed(seed)
    return PolicyValueModel(
        share_hidden_dims=[8],
        share_output_dim=8,
        q_head_dims=[4],
        p0_head_dims=[4],
        pi_head_dims=[4],
        bp0_head_dims=[4],
        bpi_head_dims=[4],
        barz_hidden_dims=[4],
        bari_hidden_dims=[4],
        i_grid_size=3,
        q_parameterization="direct",
    )


def _cache(model: PolicyValueModel) -> list[dict[str, torch.Tensor]]:
    parent = torch.tensor(
        [
            [0.10, -0.2, 1.0, 0.1, 0.0, -1.0, 4.0],
            [0.30, 0.1, 0.0, 0.3, 0.1, -1.1, 4.1],
            [0.50, 0.4, 1.0, 0.5, -0.1, -0.9, 3.9],
            [0.70, 0.8, 1.0, 0.7, 0.2, -1.2, 4.2],
        ],
        dtype=torch.float32,
    )
    eta = parent[:, 2:3]
    with torch.no_grad():
        output = model(parent)
        bp0 = output.bp0
        bpi = output.bpI
        mix = output.bp_cond
    return [
        {
            "batch_id": 0,
            "parent": parent.clone(),
            "source_index": torch.arange(len(parent)),
            "eta_current": eta.clone(),
            "bp0_target": (bp0 + 0.15).clamp(0.01, 0.99),
            "bpi_target": (bpi - 0.15).clamp(0.01, 0.99),
            "mix_target": (mix + 0.10).clamp(0.01, 0.99),
            "bp0_confidence": eta.clone(),
            "bpi_confidence": eta.clone(),
            "mix_confidence": eta.clone(),
            "mix_sample_weight": torch.ones_like(eta),
            "eta_next_active": torch.ones_like(eta, dtype=torch.bool),
            "teacher_snapshot_hash": "fixed-teacher",
        }
    ]


def _loaded(model: PolicyValueModel, hp: HyperParams):
    economic = SimpleNamespace(to_dict=lambda: {"synthetic": True})
    return SimpleNamespace(
        hyperparams=hp,
        economic_config=economic,
        models={"sdf_fc1": torch.nn.Linear(1, 1)},
        metadata={
            "policy_value_model_spec": model.model_spec(),
            "value_parameterization": {
                "checkpoint": {
                    "mode": "none",
                    "scale_formula": "1",
                    "bellman_normalization": False,
                    "log_max": 20.0,
                }
            },
        },
    )


def test_paired_students_start_equal_without_shared_storage():
    baseline = _model()
    students = clone_paired_students(baseline, torch.device("cpu"))
    state = _cache(baseline)[0]["parent"]
    with torch.no_grad():
        left = torch.cat(students["frozen_encoder"].forward_policy(state), dim=1)
        right = torch.cat(students["trainable_encoder"].forward_policy(state), dim=1)
    assert torch.equal(left, right)
    assert next(students["frozen_encoder"].parameters()).data_ptr() != next(
        students["trainable_encoder"].parameters()
    ).data_ptr()


@pytest.mark.parametrize("arm", ["frozen_encoder", "trainable_encoder"])
def test_optimizer_scope_matches_explicit_whitelist(arm: str):
    model = _model()
    names = configure_trainable_parameters(model, arm)
    optimizer = torch.optim.AdamW([dict(model.named_parameters())[name] for name in names])
    assert optimizer_parameter_names(model, optimizer) == names
    assert all(name.startswith(ARM_PREFIXES[arm]) for name in names)
    encoder_names = [name for name, _ in model.named_parameters() if name.startswith("policy_encoder.")]
    assert bool(set(encoder_names) & set(names)) is (arm == "trainable_encoder")


def test_loss_adapter_matches_production_value_and_gradient():
    hp = _hyperparams()
    first = _model()
    second = copy.deepcopy(first)
    item = _cache(first)[0]
    adapter = make_loss_adapter(first, hp, torch.device("cpu"))
    production = Episode.__new__(Episode)
    production.models = {"policy_value": second}
    production.hyperparams = hp
    production.device = torch.device("cpu")

    first_loss, _ = adapter._compute_bp_cache_loss(item)
    second_loss, _ = production._compute_bp_cache_loss(item)
    assert torch.allclose(first_loss, second_loss, atol=0.0, rtol=0.0)
    first_loss.backward()
    second_loss.backward()
    for (left_name, left), (right_name, right) in zip(
        first.named_parameters(), second.named_parameters()
    ):
        assert left_name == right_name
        if left.grad is None or right.grad is None:
            assert left.grad is None and right.grad is None
        else:
            assert torch.allclose(left.grad, right.grad, atol=0.0, rtol=0.0)


def test_no_active_batch_is_skipped_and_preserved_in_shared_schedule():
    model = _model()
    active = _cache(model)[0]
    inactive = copy.deepcopy(active)
    inactive["eta_current"].zero_()
    inactive["parent"][:, 2].zero_()
    inactive["bp0_confidence"].zero_()
    inactive["bpi_confidence"].zero_()
    inactive["mix_confidence"].zero_()
    cache = [inactive, active]
    assert active_cache_indices(cache) == [1]
    schedule = build_step_schedule(cache, steps=4, seed=12345)
    assert sum(index == 1 for index in schedule) == 4
    assert 0 in schedule
    with pytest.raises(ValueError, match="no active BP supervision"):
        build_step_schedule([inactive], steps=1, seed=12345)


def test_arm_training_updates_exact_modules_and_preserves_teacher_cache(tmp_path: Path):
    hp = _hyperparams()
    baseline = _model()
    teacher = copy.deepcopy(baseline).eval().requires_grad_(False)
    train_cache = _cache(baseline)
    val_cache = copy.deepcopy(train_cache)
    train_hash = cache_hash(train_cache)
    val_hash = cache_hash(val_cache)
    students = clone_paired_students(baseline, torch.device("cpu"))
    schedule = build_step_schedule(train_cache, steps=2, seed=12345)
    loaded = _loaded(baseline, hp)

    frozen = run_arm(
        arm="frozen_encoder",
        model=students["frozen_encoder"],
        hyperparams=hp,
        train_cache=train_cache,
        val_cache=val_cache,
        schedule=schedule,
        record_steps=[0, 1, 2],
        evaluation_max_batch_rows=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        output_dir=tmp_path,
        loaded=loaded,
        source_checkpoint=tmp_path / "source.pt",
        teacher_hash_before=model_state_hash(teacher),
        teacher_model=teacher,
        train_cache_hash_before=train_hash,
        val_cache_hash_before=val_hash,
    )
    trainable = run_arm(
        arm="trainable_encoder",
        model=students["trainable_encoder"],
        hyperparams=hp,
        train_cache=train_cache,
        val_cache=val_cache,
        schedule=schedule,
        record_steps=[0, 1, 2],
        evaluation_max_batch_rows=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        output_dir=tmp_path,
        loaded=loaded,
        source_checkpoint=tmp_path / "source.pt",
        teacher_hash_before=model_state_hash(teacher),
        teacher_model=teacher,
        train_cache_hash_before=train_hash,
        val_cache_hash_before=val_hash,
    )
    assert frozen.checks["module_changes"]["policy_encoder_parameter_change_norm"] == 0.0
    assert trainable.checks["gradient_seen"]["policy_encoder"] is True
    assert trainable.checks["module_changes"]["policy_encoder_parameter_change_norm"] > 0.0
    assert frozen.checks["buffers_unchanged"] is True
    assert trainable.checks["buffers_unchanged"] is True
    assert cache_hash(train_cache) == train_hash
    assert (tmp_path / "frozen_encoder" / "last.pt").is_file()
    payload = torch.load(tmp_path / "trainable_encoder" / "last.pt", map_location="cpu")
    reloaded = _model(seed=99)
    reloaded.load_state_dict(payload["models"]["policy_value"], strict=True)


def test_cache_schema_and_checkpoint_ambiguity_fail_clearly(tmp_path: Path):
    model = _model()
    invalid = _cache(model)
    del invalid[0]["mix_target"]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_cache(invalid, "bad cache")

    valid = _cache(model)
    with pytest.raises(ValueError, match="teacher provenance mismatch"):
        validate_cache_teacher(valid, copy.deepcopy(valid), "different-teacher")

    run_root = tmp_path / "run"
    stage = run_root / "episode_diagnostics" / "ep_002" / "post_bp.pt"
    stage.parent.mkdir(parents=True)
    torch.save({}, stage)
    first = run_root / "checkpoints" / "ep2_combined.pt"
    second = run_root / "checkpoints_analysis" / "ep2_combined.pt"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    torch.save({}, first)
    torch.save({}, second)
    with pytest.raises(RuntimeError, match="exactly one"):
        resolve_checkpoint(run_root, 2, None)


def test_stage_checkpoint_episode_and_stage_mismatch_fail_clearly(tmp_path: Path):
    checkpoint = tmp_path / "stage.pt"
    torch.save(
        {
            "episode": 1,
            "stage": "post_p",
            "models": {"policy_value": _model().state_dict()},
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_stage_checkpoint_metadata(
            checkpoint,
            episode=2,
            expected_stage="post_bp",
            label="student initialization checkpoint",
        )

    payload = torch.load(checkpoint, map_location="cpu")
    payload["episode"] = 2
    payload["stage"] = "post_bp"
    torch.save(payload, checkpoint)
    validate_stage_checkpoint_metadata(
        checkpoint,
        episode=2,
        expected_stage="post_bp",
        label="student initialization checkpoint",
    )


def _identity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "branch": ["p0", "p0", "pi_low"],
            "bank_row": [0, 1, 2],
            "source_index": [10, 11, 12],
            "refi_active": [True, False, True],
            "primary_mask": [True, False, True],
            "survival": [True, False, True],
            "numerical_identified": [True, False, True],
            "bp_pred": [0.2, 0.3, 0.4],
            "bp_star": [0.21, 0.31, 0.39],
            "regret": [0.01, 0.02, 0.03],
            "top2_margin": [0.4, float("nan"), 0.7],
        }
    )


def test_initial_reproduction_accepts_matching_structural_eta0_margin_nan():
    saved = _identity_frame()
    reproduced = saved.copy(deep=True)
    errors = validate_initial_evaluator_reproduction(saved, reproduced)
    assert errors == {
        "bp_pred": 0.0,
        "bp_star": 0.0,
        "regret": 0.0,
        "top2_margin": 0.0,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.__setitem__("top2_margin", [0.4, 0.0, 0.7]), "structural NaN"),
        (lambda frame: frame.__setitem__("top2_margin", [float("nan"), float("nan"), 0.7]), "refi_active"),
        (lambda frame: frame.__setitem__("refi_active", [True, True, True]), "mask 'refi_active'"),
        (lambda frame: frame.__setitem__("bp_pred", [0.2, 0.3, 0.5]), "does not reproduce"),
        (lambda frame: frame.__setitem__("top2_margin", [0.4, float("nan"), float("inf")]), "contains Inf"),
        (lambda frame: frame.__setitem__("source_index", [10, 11, 99]), "different row count"),
    ],
)
def test_initial_reproduction_rejects_invalid_nan_mask_or_value(mutation, message):
    saved = _identity_frame()
    reproduced = saved.copy(deep=True)
    mutation(reproduced)
    with pytest.raises(RuntimeError, match=message):
        validate_initial_evaluator_reproduction(saved, reproduced)


def _merge_cache(cache: list[dict]) -> dict:
    merged = {}
    n_rows = [int(item["parent"].shape[0]) for item in cache]
    for key in cache[0]:
        values = [item[key] for item in cache]
        if all(
            torch.is_tensor(value)
            and value.ndim > 0
            and int(value.shape[0]) == rows
            for value, rows in zip(values, n_rows)
        ):
            merged[key] = torch.cat(values, dim=0)
        else:
            merged[key] = values[0]
    return merged


def test_evaluate_cache_chunked_global_aggregation_matches_one_shot_reference():
    hp = _hyperparams()
    model = _model()
    full = _cache(model)[0]
    first = {
        key: value[:3] if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 4 else value
        for key, value in full.items()
    }
    second = {
        key: value[3:] if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 4 else value
        for key, value in full.items()
    }
    cache = [first, second]
    adapter = make_loss_adapter(model, hp, torch.device("cpu"))
    observed_batch_sizes = []

    def record_encoder_batch(_module, inputs):
        observed_batch_sizes.append(int(inputs[0].shape[0]))

    handle = model.policy_encoder.register_forward_pre_hook(record_encoder_batch)
    try:
        chunked = evaluate_cache(
            adapter,
            cache,
            split="validation",
            arm="chunked",
            step=0,
            max_batch_rows=2,
        )
    finally:
        handle.remove()
    reference = evaluate_cache(
        adapter,
        [_merge_cache(cache)],
        split="validation",
        arm="reference",
        step=0,
        max_batch_rows=100,
    )
    assert observed_batch_sizes
    assert max(observed_batch_sizes) <= 2
    for left, right in zip(chunked, reference):
        assert left["branch"] == right["branch"]
        for key in (
            "total_loss",
            "branch_loss",
            "unweighted_mae",
            "signed_bias",
            "p90_absolute_action_gap",
            "confidence_weighted_mae",
            "confidence_weight_sum",
            "active_n",
            "loss_active_count",
        ):
            assert left[key] == pytest.approx(right[key], abs=1e-7, rel=1e-7)


def test_initial_reproduction_failure_prevents_any_training_update():
    calls = {"training": 0}

    def fail_reproduction():
        raise RuntimeError("baseline reproduction failed")

    def would_train():
        calls["training"] += 1
        return "trained"

    with pytest.raises(RuntimeError, match="baseline reproduction failed"):
        execute_after_initial_reproduction_gate(fail_reproduction, would_train)
    assert calls["training"] == 0
