from pathlib import Path
import sys

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
    assert summary["validation_active_count"] > 0
    assert summary["optimizer_steps"] == 0


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
    assert summary["optimizer_steps"] == 1
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
    assert summary["optimizer_steps"] == 1
    assert episode._state_dict_hash(episode.models["policy_value"]) == before


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

    summary = episode._run_bp_distillation_stage(
        [{"batch": 0}, {"batch": 1}],
        [{"batch": 2}],
        episode.firm_target,
        n_epochs=1,
    )

    assert summary["status"] == "failed_no_valid_checkpoint"
    assert summary["optimizer_steps"] == 1
    assert summary["accepted_epochs"] == 0
    assert episode._state_dict_hash(episode.models["policy_value"]) == before


def test_staged_bp_failure_rolls_back_full_online_model_and_target():
    episode = _episode()
    batch = _batch(episode.device)
    before_online = episode._state_dict_hash(episode.models["policy_value"])
    before_target = episode._state_dict_hash(episode.firm_target)

    def _failed_bp(*args, **kwargs):
        # Mutate online model to prove staged-flow failure restores atomically.
        with torch.no_grad():
            next(episode.models["policy_value"].parameters()).add_(1.0)
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
