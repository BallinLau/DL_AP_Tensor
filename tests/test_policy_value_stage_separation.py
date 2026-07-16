from pathlib import Path
from dataclasses import replace
import sys
import random

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from config.hyperparams import HyperParams  # noqa: E402
from models.policy_value import PolicyValueModel  # noqa: E402
from training.bp_grid_teacher import BPGridTeacher  # noqa: E402
from training.episode import Episode  # noqa: E402


def _small_hyperparams() -> HyperParams:
    hp = HyperParams()
    hp.pv_bp_training_mode = "target_grid"
    hp.pv_training_flow = "staged"
    hp.firm_target_update = "stage_hard"
    hp.bp_grid_coarse_size = 3
    hp.bp_grid_refine_enabled = False
    hp.bp_grid_candidate_chunk_size = 2
    hp.bp_grid_parent_chunk_size = 2
    hp.bp_grid_max_expanded_states = 16
    hp.bp_grid_mix_policy_weight = 1.0
    hp.bp_grid_margin_scale = 1e-4
    hp.bp_grid_policy_loss_space = "logit"
    hp.bp_grid_logit_target_eps = 1e-4
    hp.pv_use_clipped_m = False
    hp.pv_target_grid_val_fraction = 0.0
    hp.bp_distill_epochs = 1
    hp.bp_distill_patience = 1
    hp.pv_eval_epochs = 1
    hp.policy_lr = 1e-3
    return hp


def _batch(device: torch.device):
    parent_state = torch.tensor(
        [
            [0.10, 0.20, 1.00, 0.20, 0.10, -2.00, 4.00],
            [0.30, 0.10, 0.00, 0.40, 0.00, -1.80, 4.20],
        ],
        device=device,
        dtype=torch.float32,
    )
    child0 = parent_state.clone()
    child1 = parent_state.clone()
    child0[:, 1:2] = child0[:, 1:2] + 0.05
    child1[:, 1:2] = child1[:, 1:2] - 0.03
    child0[:, 2:3] = torch.tensor([[1.0], [0.0]], device=device)
    child1[:, 2:3] = torch.tensor([[0.0], [1.0]], device=device)
    m0 = torch.full((2, 1), 0.98, device=device)
    m1 = torch.full((2, 1), 1.02, device=device)
    return {
        "parent": torch.cat([parent_state, torch.ones(2, 1, device=device)], dim=1),
        "children": [
            torch.cat([child0, m0], dim=1),
            torch.cat([child1, m1], dim=1),
        ],
    }


def _episode() -> Episode:
    device = torch.device("cpu")
    Config.DEVICE = device
    torch.manual_seed(123)
    online = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8).to(device)
    target = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.AdamW(online.parameters(), lr=1e-3)
    episode = Episode(
        models={"policy_value": online},
        optimizers={"policy_value": optimizer},
        config=Config,
        hyperparams=_small_hyperparams(),
        device=device,
        firm_target=target,
    )
    episode._prepare_sdf_shock_bank_for_epoch = lambda *args, **kwargs: None
    episode.evaluate_bellman_convergence = lambda *args, **kwargs: {"passed": True}
    return episode


def test_target_grid_component_decomposition_matches_joint_loss():
    episode = _episode()
    batch = _batch(episode.device)

    episode._target_grid_loss_component_mode = "joint"
    p0_joint = episode._compute_p0_loss(batch)
    pi_joint = episode._compute_pi_loss(batch)
    episode._target_grid_loss_component_mode = "value"
    p0_value = episode._compute_p0_loss(batch)
    pi_value = episode._compute_pi_loss(batch)
    episode._target_grid_loss_component_mode = "policy"
    p0_policy = episode._compute_p0_loss(batch)
    pi_policy = episode._compute_pi_loss(batch)
    episode._target_grid_loss_component_mode = "joint"

    assert torch.allclose(p0_joint, p0_value + p0_policy, atol=1e-7)
    assert torch.allclose(pi_joint, pi_value + pi_policy, atol=1e-7)


def test_policy_value_stage_optimizer_scopes_are_disjoint_and_unique():
    episode = _episode()
    pq_params = episode._policy_value_stage_params("pq")
    bp_params = episode._policy_value_stage_params("bp")
    pq_ids = [id(param) for param in pq_params]
    bp_ids = [id(param) for param in bp_params]

    assert len(pq_ids) == len(set(pq_ids))
    assert len(bp_ids) == len(set(bp_ids))
    assert set(pq_ids).isdisjoint(bp_ids)


def test_pq_stage_keeps_bp_heads_fixed():
    episode = _episode()
    batch = _batch(episode.device)
    bp_params = episode._policy_value_stage_params("bp")
    pq_params = episode._policy_value_stage_params("pq")
    bp_snapshot = episode._snapshot_params(bp_params)
    pq_snapshot = episode._snapshot_params(pq_params)
    teacher = episode.firm_target

    summary = episode._run_policy_value_evaluation_stage([batch], [batch], teacher, n_epochs=1)

    assert summary["status"] == "accepted"
    assert episode._param_max_change_from_snapshot(bp_params, bp_snapshot) == 0.0
    assert episode._param_max_change_from_snapshot(pq_params, pq_snapshot) > 0.0
    assert summary["teacher_hash_before"] == summary["teacher_hash_after"]
    assert summary["optimizer_steps"] > 0
    assert summary["accepted_epochs"] == 1
    assert summary["policy_value_eval_step_count"] > 0


def test_bp_stage_uses_fixed_cache_and_keeps_non_bp_params_fixed():
    episode = _episode()
    batch = _batch(episode.device)
    teacher = episode.firm_target
    cache = episode._build_bp_target_cache([batch], teacher)
    cache_hash_before = episode._bp_cache_hash(cache)
    bp_params = episode._policy_value_stage_params("bp")
    _, all_modules = episode._policy_value_stage_modules("bp")
    non_bp = [
        param
        for module in all_modules
        if module is not None
        for param in module.parameters()
        if id(param) not in {id(p) for p in bp_params}
    ]
    non_bp_snapshot = episode._snapshot_params(non_bp)
    bp_snapshot = episode._snapshot_params(bp_params)

    summary = episode._run_bp_distillation_stage(cache, cache, teacher, n_epochs=1)

    assert summary["status"] == "accepted"
    assert summary["train_cache_hash"] == cache_hash_before
    assert summary["train_cache_hash_after"] == cache_hash_before
    assert episode._param_max_change_from_snapshot(non_bp, non_bp_snapshot) == 0.0
    assert episode._param_max_change_from_snapshot(bp_params, bp_snapshot) > 0.0
    assert summary["optimizer_steps"] > 0
    assert summary["accepted_epochs"] == 1
    assert summary["bp_distill_step_count"] > 0
    assert summary["validation_source"] == "holdout"
    assert summary["validation_informative"] is True
    assert summary["train_active_counts"]["total_active_supervision_entries"] >= summary["train_active_counts"]["unique_parent_active_count"]


def test_pq_value_cache_matches_old_teacher_value_star():
    episode = _episode()
    batch = _batch(episode.device)
    cache, summary = episode._build_pq_value_target_cache([batch], episode.firm_target)
    teacher = BPGridTeacher.from_hyperparams(
        episode.firm_target,
        episode.loss_fns["p0"],
        episode.loss_fns["pi"],
        episode.hyperparams,
    )
    parent_state, children, m_list, *_ = episode._policy_batch_hash_components(batch)
    p0_old = teacher.compute(parent_state, children, m_list, branch="p0")
    pi_old = teacher.compute(parent_state, children, m_list, branch="pi")

    assert summary["cache_batches"] == 1
    assert torch.allclose(cache[0].p0_value_target, p0_old["value_star"].cpu(), atol=1e-6, rtol=1e-5)
    assert torch.allclose(cache[0].pi_value_target, pi_old["value_star"].cpu(), atol=1e-6, rtol=1e-5)


def test_cached_pq_value_loss_matches_legacy_value_mode():
    episode = _episode()
    batch = _batch(episode.device)
    cache, _ = episode._build_pq_value_target_cache([batch], episode.firm_target)
    previous_teacher = getattr(episode, "_policy_value_stage_target_model", None)
    episode._policy_value_stage_target_model = episode.firm_target
    try:
        legacy_total, legacy_losses = episode._compute_policy_value_component_loss(
            batch,
            component_mode="value",
            policy_loss_terms=["p0", "pi"],
        )
        cached_total, cached_losses = episode._compute_cached_pq_loss(
            batch,
            cache[0],
            q_create_graph=False,
            include_q=False,
        )
    finally:
        episode._policy_value_stage_target_model = previous_teacher

    assert torch.allclose(cached_total, legacy_total, atol=1e-6, rtol=1e-5)
    assert cached_losses["q"] == 0.0
    assert cached_losses["p0"] == pytest.approx(legacy_losses["p0"], abs=1e-6, rel=1e-5)
    assert cached_losses["pi"] == pytest.approx(legacy_losses["pi"], abs=1e-6, rel=1e-5)


def test_pq_cache_builds_once_and_skips_mix_grid(monkeypatch):
    episode = _episode()
    batch = _batch(episode.device)
    calls = {"p0": 0, "pi": 0, "mix": 0}
    original_value_target = BPGridTeacher.compute_value_target
    original_compute = BPGridTeacher.compute

    def _counting_value_target(self, *args, branch, **kwargs):
        calls[branch] += 1
        return original_value_target(self, *args, branch=branch, **kwargs)

    def _fail_on_mix(self, *args, branch, **kwargs):
        if branch == "mix":
            calls["mix"] += 1
            raise AssertionError("P/Q cache must not build mix grid")
        return original_compute(self, *args, branch=branch, **kwargs)

    monkeypatch.setattr(BPGridTeacher, "compute_value_target", _counting_value_target)
    monkeypatch.setattr(BPGridTeacher, "compute", _fail_on_mix)
    summary = episode._run_policy_value_evaluation_stage(
        [batch],
        [batch],
        episode.firm_target,
        n_epochs=2,
    )

    assert summary["status"] == "accepted"
    assert calls["p0"] == 2
    assert calls["pi"] == 2
    assert calls["mix"] == 0
    assert summary["pq_train_cache_hash"] == summary["pq_train_cache_hash_after"]
    assert summary["pq_validation_cache_hash"] == summary["pq_validation_cache_hash_after"]


def test_pq_train_fallback_reuses_train_cache_without_rebuild(monkeypatch):
    episode = _episode()
    batch = _batch(episode.device)
    calls = {"p0": 0, "pi": 0}
    original_value_target = BPGridTeacher.compute_value_target

    def _counting_value_target(self, *args, branch, **kwargs):
        calls[branch] += 1
        return original_value_target(self, *args, branch=branch, **kwargs)

    monkeypatch.setattr(BPGridTeacher, "compute_value_target", _counting_value_target)
    summary = episode._run_policy_value_evaluation_stage(
        [batch],
        [],
        episode.firm_target,
        n_epochs=2,
    )

    assert summary["status"] == "accepted"
    assert calls["p0"] == 1
    assert calls["pi"] == 1
    assert summary["pq_validation_source"] == "train_fallback"
    assert summary["pq_validation_cache_summary"]["reused_train_cache"] is True
    assert summary["pq_train_cache_hash"] == summary["pq_validation_cache_hash"]


def test_pq_cache_validation_rejects_length_mismatch():
    episode = _episode()
    batch = _batch(episode.device)

    with pytest.raises(RuntimeError, match="length mismatch"):
        episode._evaluate_cached_pq_score([batch], [])


def test_pq_cache_validation_rejects_source_metadata_presence_mismatch():
    episode = _episode()
    batch = _batch(episode.device)
    batch_with_source = {
        **batch,
        "source_id": torch.tensor([[11], [12]], dtype=torch.long),
        "source_index": torch.tensor([[0], [1]], dtype=torch.long),
    }
    cache, _ = episode._build_pq_value_target_cache([batch_with_source], episode.firm_target)

    with pytest.raises(RuntimeError, match="source_id presence mismatch"):
        episode._validate_pq_cache_item(
            batch,
            cache[0],
            batch_id=0,
            current_teacher_hash=cache[0].teacher_hash,
            current_grid_hash=cache[0].grid_config_hash,
            integrity_mode="metadata",
        )


def test_pq_cache_hash_includes_metadata():
    episode = _episode()
    batch = _batch(episode.device)
    cache, _ = episode._build_pq_value_target_cache([batch], episode.firm_target)
    base_hash = episode._pq_value_cache_hash(cache)

    assert episode._pq_value_cache_hash([replace(cache[0], batch_id=99)]) != base_hash
    assert episode._pq_value_cache_hash([replace(cache[0], teacher_hash="different")]) != base_hash
    assert episode._pq_value_cache_hash([replace(cache[0], parent_hash="different")]) != base_hash
    assert episode._pq_value_cache_hash([
        replace(cache[0], source_id=torch.tensor([[1], [2]], dtype=torch.long))
    ]) != base_hash


def test_pq_cache_metadata_validation_skips_full_data_rehash(monkeypatch):
    episode = _episode()
    batch = _batch(episode.device)
    cache, _ = episode._build_pq_value_target_cache([batch], episode.firm_target)

    def _fail_rehash(_batch):
        raise AssertionError("metadata validation must not recompute parent/child/M hashes")

    monkeypatch.setattr(episode, "_policy_batch_hash_components", _fail_rehash)
    episode._validate_pq_cache_once(
        [batch],
        cache,
        current_teacher_hash=cache[0].teacher_hash,
        current_grid_hash=cache[0].grid_config_hash,
        integrity_mode="metadata",
        label="train",
    )


def test_pq_cache_full_validation_recomputes_data_hash(monkeypatch):
    episode = _episode()
    batch = _batch(episode.device)
    cache, _ = episode._build_pq_value_target_cache([batch], episode.firm_target)

    def _fail_rehash(_batch):
        raise AssertionError("full validation recomputes parent/child/M hashes")

    monkeypatch.setattr(episode, "_policy_batch_hash_components", _fail_rehash)
    with pytest.raises(AssertionError, match="full validation"):
        episode._validate_pq_cache_once(
            [batch],
            cache,
            current_teacher_hash=cache[0].teacher_hash,
            current_grid_hash=cache[0].grid_config_hash,
            integrity_mode="full",
            label="train",
        )


def test_pq_cache_targets_stay_on_episode_device():
    episode = _episode()
    batch = _batch(episode.device)
    cache, _ = episode._build_pq_value_target_cache([batch], episode.firm_target)

    assert cache[0].p0_value_target.device == episode.device
    assert cache[0].pi_value_target.device == episode.device


def test_pq_stage_restores_target_pointer_after_cache_build_error(monkeypatch):
    episode = _episode()
    batch = _batch(episode.device)
    original_teacher = object()
    episode._policy_value_stage_target_model = original_teacher

    def _raise_cache_error(*args, **kwargs):
        raise RuntimeError("cache build failed")

    monkeypatch.setattr(episode, "_build_pq_value_target_cache", _raise_cache_error)

    with pytest.raises(RuntimeError, match="cache build failed"):
        episode._run_policy_value_evaluation_stage(
            [batch],
            [batch],
            episode.firm_target,
            n_epochs=1,
        )

    assert episode._policy_value_stage_target_model is original_teacher


def test_pq_stage_teacher_hash_not_recomputed_per_batch(monkeypatch):
    episode = _episode()
    batch_a = _batch(episode.device)
    batch_b = _batch(episode.device)
    calls = {"teacher": 0}
    original_hash = episode._state_dict_hash

    def _counting_hash(model):
        if model is episode.firm_target:
            calls["teacher"] += 1
        return original_hash(model)

    monkeypatch.setattr(episode, "_state_dict_hash", _counting_hash)
    summary = episode._run_policy_value_evaluation_stage(
        [batch_a, batch_b],
        [batch_a, batch_b],
        episode.firm_target,
        n_epochs=1,
    )

    assert summary["status"] == "accepted"
    assert calls["teacher"] == 2


def test_cached_pq_validation_uses_first_order_q_graph(monkeypatch):
    episode = _episode()
    batch = _batch(episode.device)
    cache, _ = episode._build_pq_value_target_cache([batch], episode.firm_target)
    flags = []
    original_grad = torch.autograd.grad

    def _recording_grad(*args, **kwargs):
        flags.append(bool(kwargs.get("create_graph", False)))
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", _recording_grad)
    episode._evaluate_cached_pq_score([batch], cache)

    assert flags
    assert all(flag is False for flag in flags)

    flags.clear()
    episode._compute_cached_pq_loss(batch, cache[0], q_create_graph=True)
    assert any(flag is True for flag in flags)


def test_cached_pq_validation_value_path_uses_no_grad(monkeypatch):
    episode = _episode()
    batch = _batch(episode.device)
    cache, _ = episode._build_pq_value_target_cache([batch], episode.firm_target)
    model = episode.models["policy_value"]
    original_value_outputs = model._value_outputs
    grad_enabled = []

    def _wrapped_value_outputs(parent_state):
        grad_enabled.append(torch.is_grad_enabled())
        return original_value_outputs(parent_state)

    monkeypatch.setattr(model, "_value_outputs", _wrapped_value_outputs)
    episode._evaluate_cached_pq_score([batch], cache)

    assert grad_enabled
    assert all(flag is False for flag in grad_enabled)


def test_staged_zero_epoch_statuses():
    episode = _episode()
    batch = _batch(episode.device)
    teacher = episode.firm_target
    cache = episode._build_bp_target_cache([batch], teacher)

    pq = episode._run_policy_value_evaluation_stage([batch], [batch], teacher, n_epochs=0)
    bp = episode._run_bp_distillation_stage(cache, cache, teacher, n_epochs=0)

    assert pq["status"] == "skipped_no_epochs"
    assert pq["accepted_epochs"] == 0
    assert pq["optimizer_steps"] == 0
    assert bp["status"] == "skipped_no_epochs"
    assert bp["optimizer_steps"] == 0


def _zero_bp_cache_activity(cache):
    for item in cache:
        item["bp0_confidence"] = torch.zeros_like(item["bp0_confidence"])
        item["bpi_confidence"] = torch.zeros_like(item["bpi_confidence"])
        item["mix_confidence"] = torch.zeros_like(item["mix_confidence"])


def test_bp_validation_without_active_states_uses_train_fallback():
    episode = _episode()
    batch = _batch(episode.device)
    teacher = episode.firm_target
    train_cache = episode._build_bp_target_cache([batch], teacher)
    val_cache = episode._build_bp_target_cache([batch], teacher)
    _zero_bp_cache_activity(val_cache)

    summary = episode._run_bp_distillation_stage(
        train_cache,
        val_cache,
        teacher,
        n_epochs=1,
    )

    assert summary["status"] == "accepted"
    assert summary["train_active_count"] > 0
    assert summary["validation_active_count"] == 0
    assert summary["validation_source"] == "train_fallback"
    assert summary["validation_informative"] is False
    assert summary["optimizer_steps"] > 0


def test_bp_train_without_active_states_skips_stage():
    episode = _episode()
    batch = _batch(episode.device)
    teacher = episode.firm_target
    train_cache = episode._build_bp_target_cache([batch], teacher)
    val_cache = episode._build_bp_target_cache([batch], teacher)
    _zero_bp_cache_activity(train_cache)

    summary = episode._run_bp_distillation_stage(
        train_cache,
        val_cache,
        teacher,
        n_epochs=1,
    )

    assert summary["status"] == "skipped_no_active_refinancing"
    assert summary["train_active_count"] == 0


def test_reduce_signed_branch_residuals_cancellation_and_realized_abs():
    signed = torch.tensor([[0.3, -0.3]], dtype=torch.float32)
    reduced = Episode._reduce_signed_branch_residuals(signed)

    assert reduced["conditional_signed"].item() == pytest.approx(0.0, abs=1e-7)
    assert reduced["conditional_abs"].item() == pytest.approx(0.0, abs=1e-7)
    assert reduced["realized_abs"].item() == pytest.approx(0.3, abs=1e-7)
    assert reduced["child_std"].item() == pytest.approx(0.3, abs=1e-7)


def test_reduce_signed_branch_residuals_non_cancelling_equal_weights():
    signed = torch.tensor([[0.3, 0.1]], dtype=torch.float32)
    reduced = Episode._reduce_signed_branch_residuals(signed)

    assert reduced["conditional_signed"].item() == pytest.approx(0.2, abs=1e-7)
    assert reduced["conditional_abs"].item() == pytest.approx(0.2, abs=1e-7)
    assert reduced["realized_abs"].item() == pytest.approx(0.2, abs=1e-7)
    assert reduced["child_std"].item() == pytest.approx(0.1, abs=1e-7)


def test_reduce_signed_branch_residuals_weighted_branches():
    signed = torch.tensor([[0.3, -0.3]], dtype=torch.float32)
    weights = torch.tensor([0.75, 0.25], dtype=torch.float32)
    reduced = Episode._reduce_signed_branch_residuals(signed, weights)

    assert reduced["conditional_signed"].item() == pytest.approx(0.15, abs=1e-7)
    assert reduced["conditional_abs"].item() == pytest.approx(0.15, abs=1e-7)


@pytest.mark.parametrize("branch_count", [1, 2, 4])
def test_reduce_signed_branch_residuals_supports_branch_counts(branch_count):
    signed = torch.arange(1, branch_count + 1, dtype=torch.float32).reshape(1, branch_count)
    reduced = Episode._reduce_signed_branch_residuals(signed)

    assert reduced["conditional_signed"].shape == (1,)
    assert reduced["realized_abs"].shape == (1,)


def test_reduce_signed_branch_residuals_rejects_bad_weights():
    signed = torch.tensor([[0.3, -0.3]], dtype=torch.float32)

    with pytest.raises(ValueError, match="nonnegative"):
        Episode._reduce_signed_branch_residuals(signed, torch.tensor([1.1, -0.1]))

    with pytest.raises(ValueError, match="sum to one"):
        Episode._reduce_signed_branch_residuals(signed, torch.tensor([0.5, 0.4]))


def _batch_with_child_m(batch, m_values):
    children = []
    for child, m in zip(batch["children"], m_values):
        updated = child.clone()
        updated[:, 7:8] = float(m)
        children.append(updated)
    return {**batch, "children": children}


class _ConstantPolicyValue(torch.nn.Module):
    def forward(self, x):
        n = x.shape[0]
        device = x.device
        dtype = x.dtype
        return {
            "Q": torch.full((n, 1), 0.7, device=device, dtype=dtype),
            "bp0": torch.full((n, 1), 0.2, device=device, dtype=dtype),
            "bpI": torch.full((n, 1), 0.4, device=device, dtype=dtype),
            "P0": torch.full((n, 1), 1.1, device=device, dtype=dtype),
            "PI": torch.full((n, 1), 1.3, device=device, dtype=dtype),
            "bar_i": torch.full((n, 1), 0.5, device=device, dtype=dtype),
            "bar_z": torch.full((n, 1), 0.1, device=device, dtype=dtype),
            "P": torch.full((n, 1), 2.0, device=device, dtype=dtype),
            "bp": torch.full((n, 1), 0.3, device=device, dtype=dtype),
        }


def test_p0_signed_residual_train_m_uses_pv_clamp_and_raw_does_not():
    episode = _episode()
    episode.models["policy_value"] = _ConstantPolicyValue()
    episode.hyperparams.pv_use_clipped_m = True
    episode.hyperparams.pv_m_clamp_min = 0.7
    episode.hyperparams.pv_m_clamp_max = 1.3
    batch_raw = _batch_with_child_m(_batch(episode.device), [0.1, 9.0])
    batch_clamped = _batch_with_child_m(_batch(episode.device), [0.7, 1.3])

    with torch.no_grad():
        train = episode._compute_p0_bellman_signed_residuals(batch_raw, m_mode="train")
        raw = episode._compute_p0_bellman_signed_residuals(batch_raw, m_mode="raw")
        raw_clamped = episode._compute_p0_bellman_signed_residuals(batch_clamped, m_mode="raw")

    assert torch.allclose(train, raw_clamped, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(raw, raw_clamped, atol=1e-6, rtol=1e-5)


def test_pi_signed_residual_train_m_uses_pv_clamp():
    episode = _episode()
    episode.models["policy_value"] = _ConstantPolicyValue()
    episode.hyperparams.pv_use_clipped_m = True
    episode.hyperparams.pv_m_clamp_min = 0.7
    episode.hyperparams.pv_m_clamp_max = 1.3
    batch_raw = _batch_with_child_m(_batch(episode.device), [0.1, 9.0])
    batch_clamped = _batch_with_child_m(_batch(episode.device), [0.7, 1.3])

    with torch.no_grad():
        train = episode._compute_pi_bellman_signed_residuals(batch_raw, m_mode="train")
        raw_clamped = episode._compute_pi_bellman_signed_residuals(batch_clamped, m_mode="raw")

    assert torch.allclose(train, raw_clamped, atol=1e-6, rtol=1e-5)


def test_q_signed_residual_train_m_matches_q_clamp_semantics():
    episode = _episode()
    episode.models["policy_value"] = _ConstantPolicyValue()
    episode.hyperparams.q_use_detached_m = True
    episode.hyperparams.q_m_clamp_min = 0.5
    episode.hyperparams.q_m_clamp_max = 1.5
    batch_raw = _batch_with_child_m(_batch(episode.device), [0.1, 9.0])
    batch_clamped = _batch_with_child_m(_batch(episode.device), [0.5, 1.5])

    with torch.no_grad():
        train = episode._compute_q_bellman_signed_residuals(batch_raw, m_mode="train")
        raw = episode._compute_q_bellman_signed_residuals(batch_raw, m_mode="raw")
        raw_clamped = episode._compute_q_bellman_signed_residuals(batch_clamped, m_mode="raw")

    assert torch.allclose(train, raw_clamped, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(raw, raw_clamped, atol=1e-6, rtol=1e-5)


def test_bellman_convergence_primary_uses_conditional_not_legacy_abs():
    episode = _episode()
    episode.evaluate_bellman_convergence = Episode.evaluate_bellman_convergence.__get__(episode, Episode)
    episode.hyperparams.bellman_conditional_mean_thresh = 1e-3
    episode.hyperparams.bellman_conditional_p90_thresh = 1e-3
    episode.hyperparams.bellman_conv_mean_thresh = 1e-3
    episode.hyperparams.bellman_conv_p90_thresh = 1e-3
    batch = _batch(episode.device)
    signed = torch.tensor([[0.3, -0.3], [0.2, -0.2]], dtype=torch.float32)

    episode._compute_p0_bellman_signed_residuals = lambda _batch, *, m_mode: signed
    episode._compute_pi_bellman_signed_residuals = lambda _batch, *, m_mode: signed
    episode._compute_q_bellman_signed_residuals = lambda _batch, *, m_mode: signed
    episode.evaluate_target_grid_policy_convergence = lambda _batches: {
        "enabled": True,
        "informative": True,
        "all_skipped": False,
        "passed": True,
        "policies": {},
    }

    result = episode.evaluate_bellman_convergence([batch], validation_batches=[batch])

    assert result["definition_version"] == "conditional_mean_v1"
    assert result["primary_metric"] == "conditional_train_m"
    assert result["bellman_passed"] is True
    assert result["passed"] is True
    assert result["equations"]["p0"]["conditional_train_m"]["mean"] == pytest.approx(0.0, abs=1e-7)
    assert result["equations"]["p0"]["realized_abs_train_m"]["mean"] > 0.0
    assert result["legacy"]["equations"]["p0"]["passed"] is False


def test_bellman_convergence_none_thresholds_report_without_pass_fail():
    episode = _episode()
    episode.evaluate_bellman_convergence = Episode.evaluate_bellman_convergence.__get__(episode, Episode)
    episode.hyperparams.bellman_conditional_mean_thresh = None
    episode.hyperparams.bellman_conditional_p90_thresh = None
    batch = _batch(episode.device)
    signed = torch.tensor([[0.1, -0.1]], dtype=torch.float32).expand(2, 2)

    episode._compute_p0_bellman_signed_residuals = lambda _batch, *, m_mode: signed
    episode._compute_pi_bellman_signed_residuals = lambda _batch, *, m_mode: signed
    episode._compute_q_bellman_signed_residuals = lambda _batch, *, m_mode: signed
    episode.evaluate_target_grid_policy_convergence = lambda _batches: {
        "enabled": True,
        "informative": True,
        "all_skipped": False,
        "passed": True,
        "policies": {},
    }

    result = episode.evaluate_bellman_convergence([batch], validation_batches=[batch])

    assert result["bellman_passed"] is None
    assert result["passed"] is None
    assert result["equations"]["p0"]["conditional_train_m"]["passed"] is None
    assert result["legacy"]["definition"] == "flattened_child_absolute"


def test_bp_cache_active_counts_reports_entries_and_unique_parents():
    episode = _episode()
    parent = torch.zeros(2, 7)
    cache = [{
        "parent": parent,
        "bp0_target": torch.zeros(2, 1),
        "bpi_target": torch.zeros(2, 1),
        "mix_target": torch.zeros(2, 1),
        "bp0_confidence": torch.tensor([[1.0], [0.0]]),
        "bpi_confidence": torch.tensor([[1.0], [0.0]]),
        "mix_confidence": torch.tensor([[1.0], [0.0]]),
        "mix_sample_weight": torch.tensor([[1.0], [0.0]]),
    }]

    counts = episode._bp_cache_active_counts(cache)

    assert counts["bp0_active_count"] == 1
    assert counts["bpi_active_count"] == 1
    assert counts["mix_active_count"] == 1
    assert counts["total_active_supervision_entries"] == 3
    assert counts["total_active_count"] == 3
    assert counts["unique_parent_active_count"] == 1


def test_bp_validation_score_uses_global_weighted_mae_not_batch_average():
    episode = _episode()
    records = iter([
        {
            "total": 0.0,
            "bp0_abs_error_sum": 100.0,
            "bp0_weight_sum": 100.0,
            "bp0_active_n": 100.0,
            "bpi_abs_error_sum": 0.0,
            "bpi_weight_sum": 0.0,
            "bpi_active_n": 0.0,
            "mix_abs_error_sum": 0.0,
            "mix_weight_sum": 0.0,
            "mix_active_n": 0.0,
        },
        {
            "total": 0.0,
            "bp0_abs_error_sum": 0.0,
            "bp0_weight_sum": 1.0,
            "bp0_active_n": 1.0,
            "bpi_abs_error_sum": 0.0,
            "bpi_weight_sum": 0.0,
            "bpi_active_n": 0.0,
            "mix_abs_error_sum": 0.0,
            "mix_weight_sum": 0.0,
            "mix_active_n": 0.0,
        },
    ])
    episode._compute_bp_cache_loss = lambda _item: (torch.tensor(0.0), next(records))

    score, summary = episode._evaluate_bp_cache_score([{"batch": 0}, {"batch": 1}])

    expected = 100.0 / 101.0
    assert score == expected
    assert summary["bp0_active_mae"] == expected
    assert summary["bp0_global_weighted_mae"] == expected
    assert summary["bp0_active_n"] == 101.0


def test_staged_training_restores_model_mode_after_eval_start():
    episode = _episode()
    batch = _batch(episode.device)
    model = episode.models["policy_value"]
    model.eval()

    summary = episode._run_policy_value_evaluation_stage(
        [batch],
        [batch],
        episode.firm_target,
        n_epochs=1,
    )

    assert summary["status"] == "accepted"
    assert model.training is False


def test_pq_stage_zero_optimizer_steps_is_not_accepted():
    episode = _episode()
    batch = _batch(episode.device)
    episode.hyperparams.pv_grad_hard_threshold = 0.0

    summary = episode._run_policy_value_evaluation_stage(
        [batch],
        [batch],
        episode.firm_target,
        n_epochs=1,
    )

    assert summary["status"] == "failed_no_finite_update"
    assert summary["optimizer_steps"] == 0
    assert summary["accepted_epochs"] == 0


def test_pq_integrity_mode_invalid_value_fails_before_cache_build(monkeypatch):
    episode = _episode()
    batch = _batch(episode.device)
    episode.hyperparams.pq_cache_integrity_check = "bad_mode"
    called = {"build_cache": False}

    def _unexpected_cache_build(*args, **kwargs):
        called["build_cache"] = True
        raise AssertionError("cache build should not run for invalid integrity mode")

    monkeypatch.setattr(episode, "_build_pq_value_target_cache", _unexpected_cache_build)

    with pytest.raises(ValueError, match="pq_cache_integrity_check"):
        episode._run_policy_value_evaluation_stage(
            [batch],
            [batch],
            episode.firm_target,
            n_epochs=1,
        )

    assert called["build_cache"] is False


def test_staged_failure_does_not_update_firm_target():
    episode = _episode()
    batch = _batch(episode.device)
    episode.hyperparams.pv_grad_hard_threshold = 0.0
    before = episode._state_dict_hash(episode.firm_target)

    result = episode._run_policy_value_staged(
        pv_train_batches=[batch],
        validation_batches=[batch],
        n_epochs=1,
    )

    assert result["metadata"]["policy_value_stage_status"] == "failed_pq"
    assert episode._state_dict_hash(episode.firm_target) == before


def _patch_stage_loss(episode: Episode, param: torch.nn.Parameter):
    def _loss(batch, cache_item=None, *, q_create_graph=True, include_q=True):
        scale = float(batch.get("scale", 1.0))
        total = param.sum() * scale
        return total, {"total": float(total.detach().item())}

    episode._compute_cached_pq_loss = _loss


def test_pq_rejected_epoch_rolls_back_model_state():
    episode = _episode()
    param = episode._policy_value_stage_params("pq")[0]
    _patch_stage_loss(episode, param)
    episode._evaluate_policy_value_component_score = lambda *args, **kwargs: (1.0, {"total": 1.0})
    episode._evaluate_cached_pq_score = lambda *args, **kwargs: (1.0, {"total": 1.0})
    episode.hyperparams.pv_grad_hard_threshold = 100.0
    episode.hyperparams.pv_epoch_max_skip_ratio = 0.4
    before = episode._state_dict_hash(episode.models["policy_value"])
    batch_ok = _batch(episode.device)
    batch_hard = _batch(episode.device)
    batch_ok["scale"] = 1.0
    batch_hard["scale"] = 1_000.0

    summary = episode._run_policy_value_evaluation_stage(
        [batch_ok, batch_hard],
        [batch_ok],
        episode.firm_target,
        n_epochs=1,
    )

    assert summary["status"] == "failed_no_valid_checkpoint"
    assert summary["optimizer_steps"] == 0
    assert summary["attempted_optimizer_steps"] == 1
    assert summary["accepted_epochs"] == 0
    assert episode._state_dict_hash(episode.models["policy_value"]) == before


def test_pq_stage_failure_rolls_back_online_model():
    episode = _episode()
    param = episode._policy_value_stage_params("pq")[0]
    _patch_stage_loss(episode, param)
    episode._evaluate_policy_value_component_score = lambda *args, **kwargs: (float("nan"), {})
    episode._evaluate_cached_pq_score = lambda *args, **kwargs: (float("nan"), {})
    before = episode._state_dict_hash(episode.models["policy_value"])
    batch = _batch(episode.device)
    batch["scale"] = 1.0

    summary = episode._run_policy_value_evaluation_stage(
        [batch],
        [batch],
        episode.firm_target,
        n_epochs=1,
    )

    assert summary["status"] == "failed_no_valid_checkpoint"
    assert summary["optimizer_steps"] == 0
    assert summary["attempted_optimizer_steps"] == 1
    assert episode._state_dict_hash(episode.models["policy_value"]) == before


def test_pq_best_checkpoint_restore_aligns_counters_and_records():
    episode = _episode()
    param = episode._policy_value_stage_params("pq")[0]
    _patch_stage_loss(episode, param)
    score_iter = iter([1.0, 2.0])
    best_snapshot = {}

    def _score(*args, **kwargs):
        score = next(score_iter)
        if score == 1.0:
            best_snapshot["model_hash"] = episode._state_dict_hash(
                episode.models["policy_value"]
            )
            best_snapshot["step_count"] = int(episode.step_count)
            best_snapshot["pv_eval_step_count"] = int(
                episode.policy_value_eval_step_count
            )
        return score, {"total": score}

    episode._evaluate_cached_pq_score = _score
    before_step_count = int(episode.step_count)
    before_pv_eval_steps = int(episode.policy_value_eval_step_count)
    batch = _batch(episode.device)
    batch["scale"] = 1.0

    summary = episode._run_policy_value_evaluation_stage(
        [batch],
        [batch],
        episode.firm_target,
        n_epochs=2,
    )

    assert summary["status"] == "accepted"
    assert summary["best_epoch"] == 1
    assert summary["attempted_optimizer_steps"] == 2
    assert summary["accepted_optimizer_steps_total"] == 2
    assert summary["optimizer_steps"] == 1
    assert summary["best_checkpoint_optimizer_steps"] == 1
    assert summary["accepted_epochs_total"] == 2
    assert summary["best_checkpoint_accepted_epochs"] == 1
    assert episode.step_count == before_step_count + 1
    assert episode.policy_value_eval_step_count == before_pv_eval_steps + 1
    assert episode.step_count == best_snapshot["step_count"]
    assert episode.policy_value_eval_step_count == best_snapshot["pv_eval_step_count"]
    assert episode._state_dict_hash(episode.models["policy_value"]) == best_snapshot["model_hash"]


def test_pq_best_checkpoint_restore_aligns_rng_state():
    episode = _episode()
    param = episode._policy_value_stage_params("pq")[0]
    _patch_stage_loss(episode, param)
    score_iter = iter([1.0, 2.0])
    expected_next = {}

    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)

    def _score(*args, **kwargs):
        score = next(score_iter)
        if score == 1.0:
            state = episode._capture_rng_state()
            expected_next["python"] = random.random()
            expected_next["numpy"] = float(np.random.rand())
            expected_next["torch"] = float(torch.rand(1).item())
            episode._restore_rng_state(state)
        else:
            random.random()
            np.random.rand()
            torch.rand(1)
        return score, {"total": score}

    episode._evaluate_cached_pq_score = _score
    batch = _batch(episode.device)
    batch["scale"] = 1.0

    summary = episode._run_policy_value_evaluation_stage(
        [batch],
        [batch],
        episode.firm_target,
        n_epochs=2,
    )

    assert summary["status"] == "accepted"
    assert summary["best_epoch"] == 1
    assert random.random() == expected_next["python"]
    assert float(np.random.rand()) == expected_next["numpy"]
    assert float(torch.rand(1).item()) == expected_next["torch"]


def test_pq_rejected_epoch_restores_rng_state():
    episode = _episode()
    param = episode._policy_value_stage_params("pq")[0]
    _patch_stage_loss(episode, param)
    episode._evaluate_cached_pq_score = lambda *args, **kwargs: (1.0, {"total": 1.0})
    episode.hyperparams.pv_grad_hard_threshold = 100.0
    episode.hyperparams.pv_epoch_max_skip_ratio = 0.4

    random.seed(3030)
    np.random.seed(3030)
    torch.manual_seed(3030)
    expected_state = episode._capture_rng_state()
    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = float(torch.rand(1).item())
    episode._restore_rng_state(expected_state)
    norm_iter = iter([1.0, 1_000.0])

    def _advance_rng_after_step(params, max_norm):
        random.random()
        np.random.rand()
        torch.rand(1)
        value = next(norm_iter)
        return value, min(value, max_norm)

    episode._clip_params_with_raw_norm = _advance_rng_after_step
    batch_ok = _batch(episode.device)
    batch_hard = _batch(episode.device)
    batch_ok["scale"] = 1.0
    batch_hard["scale"] = 1_000.0

    summary = episode._run_policy_value_evaluation_stage(
        [batch_ok, batch_hard],
        [batch_ok],
        episode.firm_target,
        n_epochs=1,
    )

    assert summary["status"] == "failed_no_valid_checkpoint"
    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    assert float(torch.rand(1).item()) == expected_torch


def test_bp_rejected_epoch_rolls_back_bp_heads():
    episode = _episode()
    bp_params = episode._policy_value_stage_params("bp")
    param = bp_params[0]
    calls = iter([1.0, 1_000.0])

    def _loss(_item):
        scale = next(calls)
        total = param.sum() * scale
        return total, {"total": float(total.detach().item())}

    episode._compute_bp_cache_loss = _loss
    episode._evaluate_bp_cache_score = lambda *args, **kwargs: (1.0, {"total": 1.0})
    episode._bp_cache_hash = lambda _cache: "fixed-cache"
    episode._bp_cache_active_counts = lambda _cache: {
        "bp0_active_count": 1.0,
        "bpi_active_count": 1.0,
        "mix_active_count": 1.0,
        "total_active_count": 3.0,
    }
    episode.hyperparams.pv_grad_hard_threshold = 100.0
    episode.hyperparams.pv_epoch_max_skip_ratio = 0.4
    before = episode._state_dict_hash(episode.models["policy_value"])
    before_step_count = int(episode.step_count)
    before_bp_steps = int(episode.bp_distill_step_count)

    summary = episode._run_bp_distillation_stage(
        [{"batch": 0}, {"batch": 1}],
        [{"batch": 2}],
        episode.firm_target,
        n_epochs=1,
    )

    assert summary["status"] == "failed_no_valid_checkpoint"
    assert summary["optimizer_steps"] == 0
    assert summary["attempted_optimizer_steps"] == 1
    assert summary["accepted_epochs"] == 0
    assert episode.step_count == before_step_count
    assert episode.bp_distill_step_count == before_bp_steps
    assert episode._state_dict_hash(episode.models["policy_value"]) == before


def test_bp_best_checkpoint_restore_aligns_counters_records_and_rng():
    episode = _episode()
    bp_params = episode._policy_value_stage_params("bp")
    param = bp_params[0]
    score_iter = iter([1.0, 2.0])
    best_snapshot = {}
    random.seed(4040)
    np.random.seed(4040)
    torch.manual_seed(4040)

    def _loss(_item):
        total = param.sum()
        return total, {"total": float(total.detach().item())}

    def _score(_cache):
        score = next(score_iter)
        if score == 1.0:
            best_snapshot["model_hash"] = episode._state_dict_hash(
                episode.models["policy_value"]
            )
            best_snapshot["step_count"] = int(episode.step_count)
            best_snapshot["bp_steps"] = int(episode.bp_distill_step_count)
            state = episode._capture_rng_state()
            best_snapshot["next_python"] = random.random()
            best_snapshot["next_numpy"] = float(np.random.rand())
            best_snapshot["next_torch"] = float(torch.rand(1).item())
            episode._restore_rng_state(state)
        else:
            random.random()
            np.random.rand()
            torch.rand(1)
        return score, {"total": score}

    episode._compute_bp_cache_loss = _loss
    episode._evaluate_bp_cache_score = _score
    episode._bp_cache_hash = lambda _cache: "fixed-cache"
    episode._bp_cache_active_counts = lambda _cache: {
        "bp0_active_count": 1.0,
        "bpi_active_count": 1.0,
        "mix_active_count": 1.0,
        "total_active_count": 3.0,
    }

    summary = episode._run_bp_distillation_stage(
        [{"batch": 0}],
        [{"batch": 1}],
        episode.firm_target,
        n_epochs=2,
    )

    assert summary["status"] == "accepted"
    assert summary["best_epoch"] == 1
    assert summary["attempted_optimizer_steps"] == 2
    assert summary["accepted_optimizer_steps_total"] == 2
    assert summary["optimizer_steps"] == 1
    assert summary["best_checkpoint_optimizer_steps"] == 1
    assert summary["accepted_epochs_total"] == 2
    assert summary["best_checkpoint_accepted_epochs"] == 1
    assert episode.step_count == best_snapshot["step_count"]
    assert episode.bp_distill_step_count == best_snapshot["bp_steps"]
    assert episode._state_dict_hash(episode.models["policy_value"]) == best_snapshot["model_hash"]
    assert random.random() == best_snapshot["next_python"]
    assert float(np.random.rand()) == best_snapshot["next_numpy"]
    assert float(torch.rand(1).item()) == best_snapshot["next_torch"]


def test_bp_exception_restores_stage_start_runtime():
    episode = _episode()
    bp_params = episode._policy_value_stage_params("bp")
    param = bp_params[0]
    first_call = {"done": False}
    random.seed(4545)
    np.random.seed(4545)
    torch.manual_seed(4545)
    start_rng = episode._capture_rng_state()
    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = float(torch.rand(1).item())
    episode._restore_rng_state(start_rng)
    before = episode._state_dict_hash(episode.models["policy_value"])
    before_step_count = int(episode.step_count)
    before_bp_steps = int(episode.bp_distill_step_count)

    def _loss(_item):
        if not first_call["done"]:
            first_call["done"] = True
            total = param.sum()
            return total, {"total": float(total.detach().item())}
        random.random()
        np.random.rand()
        torch.rand(1)
        raise RuntimeError("bp loss exploded")

    episode._compute_bp_cache_loss = _loss
    episode._bp_cache_hash = lambda _cache: "fixed-cache"
    episode._bp_cache_active_counts = lambda _cache: {
        "bp0_active_count": 1.0,
        "bpi_active_count": 1.0,
        "mix_active_count": 1.0,
        "total_active_count": 3.0,
    }

    with pytest.raises(RuntimeError, match="bp loss exploded"):
        episode._run_bp_distillation_stage(
            [{"batch": 0}, {"batch": 1}],
            [{"batch": 2}],
            episode.firm_target,
            n_epochs=1,
        )

    assert episode._state_dict_hash(episode.models["policy_value"]) == before
    assert episode.step_count == before_step_count
    assert episode.bp_distill_step_count == before_bp_steps
    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    assert float(torch.rand(1).item()) == expected_torch


def test_staged_bp_failure_rolls_back_full_online_model_and_target():
    episode = _episode()
    batch = _batch(episode.device)
    episode.step_count = 17
    episode.policy_value_eval_step_count = 5
    episode.bp_distill_step_count = 3
    random.seed(5050)
    np.random.seed(5050)
    torch.manual_seed(5050)
    start_rng = episode._capture_rng_state()
    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = float(torch.rand(1).item())
    episode._restore_rng_state(start_rng)
    before_online = episode._state_dict_hash(episode.models["policy_value"])
    before_target = episode._state_dict_hash(episode.firm_target)

    def _failed_bp(*args, **kwargs):
        # Mutate online model to prove staged-flow failure restores atomically.
        with torch.no_grad():
            next(episode.models["policy_value"].parameters()).add_(1.0)
        episode.step_count += 11
        episode.policy_value_eval_step_count += 7
        episode.bp_distill_step_count += 13
        random.random()
        np.random.rand()
        torch.rand(1)
        return {"status": "failed_no_valid_checkpoint", "optimizer_steps": 1}

    episode._run_bp_distillation_stage = _failed_bp

    result = episode._run_policy_value_staged(
        pv_train_batches=[batch],
        validation_batches=[batch],
        n_epochs=1,
    )

    assert result["metadata"]["policy_value_stage_status"] == "failed_bp"
    assert result["metadata"]["firm_target_stage_update"]["firm_target_update_count"] == 0
    assert episode._state_dict_hash(episode.models["policy_value"]) == before_online
    assert episode._state_dict_hash(episode.firm_target) == before_target
    assert episode.step_count == 17
    assert episode.policy_value_eval_step_count == 5
    assert episode.bp_distill_step_count == 3
    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    assert float(torch.rand(1).item()) == expected_torch


def test_staged_bp_failure_skips_convergence_after_restore():
    episode = _episode()
    batch = _batch(episode.device)
    called = {"convergence": False}

    episode._run_bp_distillation_stage = lambda *args, **kwargs: {
        "status": "failed_no_valid_checkpoint",
        "optimizer_steps": 1,
    }

    def _fail_if_called(*args, **kwargs):
        called["convergence"] = True
        raise AssertionError("convergence should be skipped after staged failure")

    episode.evaluate_bellman_convergence = _fail_if_called

    result = episode._run_policy_value_staged(
        pv_train_batches=[batch],
        validation_batches=[batch],
        n_epochs=1,
    )

    assert result["metadata"]["policy_value_stage_status"] == "failed_bp"
    assert result["convergence"]["enabled"] is False
    assert result["convergence"]["skip_reason"] == "staged_training_failed"
    assert called["convergence"] is False


def test_staged_exception_restores_full_runtime_state():
    episode = _episode()
    batch = _batch(episode.device)
    episode.step_count = 21
    episode.policy_value_eval_step_count = 8
    episode.bp_distill_step_count = 4
    random.seed(6060)
    np.random.seed(6060)
    torch.manual_seed(6060)
    start_rng = episode._capture_rng_state()
    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = float(torch.rand(1).item())
    episode._restore_rng_state(start_rng)
    before_online = episode._state_dict_hash(episode.models["policy_value"])
    before_target = episode._state_dict_hash(episode.firm_target)

    def _raise_cache_error(*args, **kwargs):
        with torch.no_grad():
            next(episode.models["policy_value"].parameters()).add_(1.0)
        episode.step_count += 5
        episode.policy_value_eval_step_count += 6
        random.random()
        np.random.rand()
        torch.rand(1)
        raise RuntimeError("bp cache build failed")

    episode._build_bp_target_cache = _raise_cache_error

    with pytest.raises(RuntimeError, match="bp cache build failed"):
        episode._run_policy_value_staged(
            pv_train_batches=[batch],
            validation_batches=[batch],
            n_epochs=1,
        )

    assert episode._state_dict_hash(episode.models["policy_value"]) == before_online
    assert episode._state_dict_hash(episode.firm_target) == before_target
    assert episode.step_count == 21
    assert episode.policy_value_eval_step_count == 8
    assert episode.bp_distill_step_count == 4
    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    assert float(torch.rand(1).item()) == expected_torch


def test_staged_firm_target_update_exception_restores_full_state():
    episode = _episode()
    batch = _batch(episode.device)
    episode.step_count = 31
    episode.policy_value_eval_step_count = 9
    episode.bp_distill_step_count = 6
    random.seed(7070)
    np.random.seed(7070)
    torch.manual_seed(7070)
    start_rng = episode._capture_rng_state()
    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = float(torch.rand(1).item())
    episode._restore_rng_state(start_rng)
    before_online = episode._state_dict_hash(episode.models["policy_value"])
    before_target = episode._state_dict_hash(episode.firm_target)

    def _raise_update_error(*args, **kwargs):
        with torch.no_grad():
            next(episode.firm_target.parameters()).add_(1.0)
        episode.step_count += 5
        episode.policy_value_eval_step_count += 6
        episode.bp_distill_step_count += 7
        random.random()
        np.random.rand()
        torch.rand(1)
        raise RuntimeError("firm target update failed")

    episode._update_firm_target_now = _raise_update_error

    with pytest.raises(RuntimeError, match="firm target update failed"):
        episode._run_policy_value_staged(
            pv_train_batches=[batch],
            validation_batches=[batch],
            n_epochs=1,
        )

    assert episode._state_dict_hash(episode.models["policy_value"]) == before_online
    assert episode._state_dict_hash(episode.firm_target) == before_target
    assert episode.step_count == 31
    assert episode.policy_value_eval_step_count == 9
    assert episode.bp_distill_step_count == 6
    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    assert float(torch.rand(1).item()) == expected_torch


def test_staged_convergence_exception_restores_after_firm_update():
    episode = _episode()
    batch = _batch(episode.device)
    episode.step_count = 41
    episode.policy_value_eval_step_count = 10
    episode.bp_distill_step_count = 7
    random.seed(8080)
    np.random.seed(8080)
    torch.manual_seed(8080)
    start_rng = episode._capture_rng_state()
    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = float(torch.rand(1).item())
    episode._restore_rng_state(start_rng)
    before_online = episode._state_dict_hash(episode.models["policy_value"])
    before_target = episode._state_dict_hash(episode.firm_target)

    def _raise_convergence(*args, **kwargs):
        random.random()
        np.random.rand()
        torch.rand(1)
        raise RuntimeError("convergence failed")

    episode.evaluate_bellman_convergence = _raise_convergence

    with pytest.raises(RuntimeError, match="convergence failed"):
        episode._run_policy_value_staged(
            pv_train_batches=[batch],
            validation_batches=[batch],
            n_epochs=1,
        )

    assert episode._state_dict_hash(episode.models["policy_value"]) == before_online
    assert episode._state_dict_hash(episode.firm_target) == before_target
    assert episode.step_count == 41
    assert episode.policy_value_eval_step_count == 10
    assert episode.bp_distill_step_count == 7
    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    assert float(torch.rand(1).item()) == expected_torch


def test_staged_success_convergence_is_rng_neutral():
    episode = _episode()
    batch = _batch(episode.device)
    random.seed(9090)
    np.random.seed(9090)
    torch.manual_seed(9090)
    expected_next = {}

    def _convergence(*args, **kwargs):
        state = episode._capture_rng_state()
        expected_next["python"] = random.random()
        expected_next["numpy"] = float(np.random.rand())
        expected_next["torch"] = float(torch.rand(1).item())
        episode._restore_rng_state(state)
        random.random()
        np.random.rand()
        torch.rand(1)
        return {"passed": True, "diagnostic": "rng_neutral"}

    episode.evaluate_bellman_convergence = _convergence
    before_target = episode._state_dict_hash(episode.firm_target)

    result = episode._run_policy_value_staged(
        pv_train_batches=[batch],
        validation_batches=[batch],
        n_epochs=1,
    )

    assert result["metadata"]["policy_value_stage_status"] == "accepted"
    assert result["metadata"]["firm_target_stage_update"]["firm_target_update_count"] == 1
    assert episode._state_dict_hash(episode.firm_target) != before_target
    assert result["convergence"]["passed"] is True
    assert random.random() == expected_next["python"]
    assert float(np.random.rand()) == expected_next["numpy"]
    assert float(torch.rand(1).item()) == expected_next["torch"]


def _patch_simple_bellman_residuals(episode: Episode):
    episode._compute_p0_bellman_abs_residual = lambda _batch: torch.tensor([0.0])
    episode._compute_pi_bellman_abs_residual = lambda _batch: torch.tensor([0.0])
    episode._compute_q_bellman_abs_residual = lambda _batch: torch.tensor([0.0])
    signed = torch.zeros((2, 1), dtype=torch.float32)
    episode._compute_p0_bellman_signed_residuals = lambda _batch, *, m_mode: signed
    episode._compute_pi_bellman_signed_residuals = lambda _batch, *, m_mode: signed
    episode._compute_q_bellman_signed_residuals = lambda _batch, *, m_mode: signed


def test_bellman_convergence_keeps_policy_diagnostics_in_eval_mode():
    episode = _episode()
    episode.evaluate_bellman_convergence = Episode.evaluate_bellman_convergence.__get__(episode, Episode)
    batch = _batch(episode.device)
    model = episode.models["policy_value"]
    model.train()
    _patch_simple_bellman_residuals(episode)
    calls = {"n": 0}

    def _policy_convergence(_batches):
        calls["n"] += 1
        assert model.training is False
        return {
            "enabled": True,
            "informative": True,
            "all_skipped": False,
            "passed": True,
            "policies": {},
        }

    episode.evaluate_target_grid_policy_convergence = _policy_convergence

    result = episode.evaluate_bellman_convergence([batch], validation_batches=[batch])

    assert result["enabled"] is True
    assert calls["n"] == 2
    assert model.training is True


def test_bellman_convergence_exception_restores_train_mode():
    episode = _episode()
    episode.evaluate_bellman_convergence = Episode.evaluate_bellman_convergence.__get__(episode, Episode)
    batch = _batch(episode.device)
    model = episode.models["policy_value"]
    model.train()
    _patch_simple_bellman_residuals(episode)

    def _raise_policy_convergence(_batches):
        assert model.training is False
        raise RuntimeError("policy convergence failed")

    episode.evaluate_target_grid_policy_convergence = _raise_policy_convergence

    with pytest.raises(RuntimeError, match="policy convergence failed"):
        episode.evaluate_bellman_convergence([batch], validation_batches=[batch])

    assert model.training is True


def test_bellman_convergence_exception_preserves_eval_mode():
    episode = _episode()
    episode.evaluate_bellman_convergence = Episode.evaluate_bellman_convergence.__get__(episode, Episode)
    batch = _batch(episode.device)
    model = episode.models["policy_value"]
    model.eval()
    _patch_simple_bellman_residuals(episode)

    def _raise_policy_convergence(_batches):
        assert model.training is False
        raise RuntimeError("policy convergence failed")

    episode.evaluate_target_grid_policy_convergence = _raise_policy_convergence

    with pytest.raises(RuntimeError, match="policy convergence failed"):
        episode.evaluate_bellman_convergence([batch], validation_batches=[batch])

    assert model.training is False


def test_staged_exception_restores_online_model_mode():
    episode = _episode()
    batch = _batch(episode.device)
    model = episode.models["policy_value"]
    model.train()
    before_online = episode._state_dict_hash(model)
    before_target = episode._state_dict_hash(episode.firm_target)

    def _raise_convergence(*args, **kwargs):
        model.eval()
        raise RuntimeError("convergence mode failure")

    episode.evaluate_bellman_convergence = _raise_convergence

    with pytest.raises(RuntimeError, match="convergence mode failure"):
        episode._run_policy_value_staged(
            pv_train_batches=[batch],
            validation_batches=[batch],
            n_epochs=1,
        )

    assert episode._state_dict_hash(model) == before_online
    assert episode._state_dict_hash(episode.firm_target) == before_target
    assert model.training is True
