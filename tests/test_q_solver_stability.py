from copy import deepcopy
import types

import numpy as np
import pytest
import torch

from config import Config
from config.hyperparams import HyperParams
from losses.q_loss import normalize_q_bellman_residuals
from models.policy_value import PolicyValueModel
from training.episode import Episode


def _model() -> PolicyValueModel:
    return PolicyValueModel(
        q_parameterization="hybrid_regime",
        share_hidden_dims=[4],
        share_output_dim=4,
        q_head_dims=[4],
        p0_head_dims=[4],
        pi_head_dims=[4],
        bp0_head_dims=[4],
        bpi_head_dims=[4],
        barz_hidden_dims=[4],
        bari_hidden_dims=[4],
        i_grid_size=2,
    )


def _episode() -> Episode:
    hp = HyperParams()
    hp.q_parameterization = "hybrid_regime"
    hp.policy_lr = 1e-3
    hp.policy_weight_decay = 0.0
    hp.q_zero_boundary_epochs = 1
    hp.q_default_pretrain_epochs = 1
    hp.q_survival_aio_epochs = 1
    hp.q_mixed_polish_epochs = 1
    hp.q_require_zero_phase = True
    hp.q_require_default_phase = True
    hp.q_require_survival_phase = True
    model = _model()
    target = deepcopy(model)
    return Episode(
        models={"policy_value": model},
        optimizers={"policy_value": torch.optim.AdamW(model.parameters(), lr=1e-3)},
        config=Config,
        hyperparams=hp,
        device=torch.device("cpu"),
        firm_target=target,
    )


def _batch() -> dict[str, object]:
    parent = torch.tensor(
        [
            [0.2, 0.1, 1.0, 0.2, 0.0, -2.0, 4.0, 1.0],
            [0.4, 0.2, 1.0, 0.3, 0.1, -2.0, 4.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return {"parent": parent, "children": [parent.clone(), parent.clone()]}


class _AlwaysSurvivalP(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = deepcopy(base)

    def forward(self, state):
        return self.base(state)

    def forward_equity(self, state):
        phat = torch.ones_like(state[:, 0:1])
        return {
            "Phat": phat,
            "P": phat,
            "bar_z": torch.zeros_like(phat),
            "survival_prob": torch.ones_like(phat),
        }


def _install_controlled_phase(
    episode: Episode,
    monkeypatch: pytest.MonkeyPatch,
    *,
    validation_target: float,
    training_target: float,
) -> torch.nn.Parameter:
    parameter = next(episode.models["policy_value"].q_head.parameters())
    start = float(parameter.detach().mean().item())
    def fixed_batch(batch, *_args, **kwargs):
        return (batch, {}) if kwargs.get("return_diagnostics") else batch

    monkeypatch.setattr(episode, "_build_q_survival_batch", fixed_batch)

    def train_loss(*_args, **_kwargs):
        value = parameter.mean()
        episode._latest_q_terms = {
            "q_regime_claim_n": 1.0,
            "q_claim_bellman_abs_mean_raw": float(abs(value.detach().item() - validation_target)),
            "q_claim_bellman_abs_mean_normalized": float(abs(value.detach().item() - validation_target)),
            "q_unit_mean": float(value.detach().item()),
            "q_unit_p95": float(value.detach().item()),
            "q_unit_max": float(value.detach().item()),
            "q_claim_value_mean": float(value.detach().item()),
            "q_claim_value_p95": float(value.detach().item()),
            "q_claim_value_max": float(value.detach().item()),
            "q_recursion_gain_mean": 0.5,
            "q_recursion_gain_p95": 0.5,
            "q_recursion_gain_gt1_share": 0.0,
        }
        return (value - training_target).pow(2)

    def evaluate(*_args, **_kwargs):
        score = abs(float(parameter.detach().mean().item()) - validation_target)
        return {
            "score_primary": score,
            "score_raw_abs": score,
            "score_normalized_abs": score,
            "sample_count": 1,
            "batch_count": 1,
            "metrics": dict(episode._latest_q_terms),
        }

    monkeypatch.setattr(episode, "_compute_q_survival_bellman_loss", train_loss)
    monkeypatch.setattr(episode, "_evaluate_q_validation_bank", evaluate)
    episode.hyperparams.q_epoch_validation_enabled = True
    episode.hyperparams.q_validation_min_delta = 0.0
    episode.hyperparams.q_restore_best_checkpoint = True
    episode._controlled_start = start
    return parameter


def test_parent_common_q_normalization_and_fixed_point():
    residuals = [torch.tensor([[2.0]], requires_grad=True), torch.tensor([[-2.0]], requires_grad=True)]
    targets = [torch.tensor([[12.0]], requires_grad=True), torch.tensor([[8.0]], requires_grad=True)]
    normalized, scale = normalize_q_bellman_residuals(residuals, targets)
    torch.testing.assert_close(scale, torch.tensor([[11.0]]))
    torch.testing.assert_close(normalized[0], torch.tensor([[2.0 / 11.0]]))
    torch.testing.assert_close(normalized[1], torch.tensor([[-2.0 / 11.0]]))
    assert scale.requires_grad is False
    zeros, _ = normalize_q_bellman_residuals(
        [torch.zeros(1, 1), torch.zeros(1, 1)], targets
    )
    assert all(value.count_nonzero() == 0 for value in zeros)
    low_b_residual = torch.tensor([[1.0]])
    low_b_target = torch.tensor([[0.005]])
    normalized_low_b, low_b_scale = normalize_q_bellman_residuals(
        [low_b_residual], [low_b_target]
    )
    torch.testing.assert_close(low_b_scale, torch.tensor([[1.005]]))
    torch.testing.assert_close(normalized_low_b[0], low_b_residual / 1.005)


def test_q_optimizer_lr_override_and_policy_fallback():
    episode = _episode()
    params = episode._policy_value_stage_params("q")
    episode.hyperparams.policy_lr = 3e-3
    episode.hyperparams.q_stage_lr = None
    assert episode._make_q_regime_optimizer(params).param_groups[0]["lr"] == pytest.approx(3e-3)
    episode.hyperparams.q_stage_lr = 1e-4
    assert episode._make_q_regime_optimizer(params).param_groups[0]["lr"] == pytest.approx(1e-4)
    assert episode._make_policy_value_stage_optimizer(params).param_groups[0]["lr"] == pytest.approx(3e-3)


@pytest.mark.parametrize("phase", ["survival", "polish"])
def test_validation_improvement_keeps_best(monkeypatch, phase):
    episode = _episode()
    parameter = next(episode.models["policy_value"].q_head.parameters())
    start = float(parameter.detach().mean().item())
    _install_controlled_phase(
        episode, monkeypatch, validation_target=start + 1.0, training_target=start + 1.0
    )
    summary = episode._run_q_regime_phase(
        phase=phase,
        batches=[_batch()],
        frozen_p_model=deepcopy(episode.firm_target),
        q_target_model=deepcopy(episode.firm_target),
        epochs=1,
        validation_bank=[_batch()],
    )
    assert summary["status"] == "accepted_improved"
    assert float(parameter.detach().mean().item()) != pytest.approx(start)
    assert summary["validation_best_epoch"] == 1


@pytest.mark.parametrize("phase", ["survival", "polish"])
def test_no_improvement_restores_only_phase_start(monkeypatch, phase):
    episode = _episode()
    parameter = next(episode.models["policy_value"].q_head.parameters())
    start_tensor = parameter.detach().clone()
    start = float(parameter.detach().mean().item())
    _install_controlled_phase(
        episode, monkeypatch, validation_target=start, training_target=start + 1.0
    )
    summary = episode._run_q_regime_phase(
        phase=phase,
        batches=[_batch()],
        frozen_p_model=deepcopy(episode.firm_target),
        q_target_model=deepcopy(episode.firm_target),
        epochs=1,
        validation_bank=[_batch()],
    )
    assert summary["status"] == "accepted_reverted"
    torch.testing.assert_close(parameter, start_tensor, rtol=0.0, atol=0.0)
    assert summary["validation_after_restore"]["score_primary"] == pytest.approx(0.0)


def test_nonfinite_q_loss_is_hard_failure(monkeypatch):
    episode = _episode()
    def fixed_batch(batch, *_args, **kwargs):
        return (batch, {}) if kwargs.get("return_diagnostics") else batch

    monkeypatch.setattr(episode, "_build_q_survival_batch", fixed_batch)
    monkeypatch.setattr(
        episode,
        "_compute_q_survival_bellman_loss",
        lambda *_args, **_kwargs: torch.tensor(float("nan")),
    )
    summary = episode._run_q_regime_phase(
        phase="survival",
        batches=[_batch()],
        frozen_p_model=deepcopy(episode.firm_target),
        q_target_model=deepcopy(episode.firm_target),
        epochs=1,
    )
    assert summary["status"] == "rejected_numerical"
    assert summary["rollback_reason"] == "nonfinite_loss"


def test_nonfinite_validation_is_hard_failure(monkeypatch):
    episode = _episode()
    episode.hyperparams.q_epoch_validation_enabled = True
    monkeypatch.setattr(
        episode,
        "_evaluate_q_validation_bank",
        lambda *_args, **_kwargs: {
            "score_primary": float("inf"),
            "score_raw_abs": float("inf"),
            "score_normalized_abs": float("inf"),
            "sample_count": 1,
            "batch_count": 1,
            "metrics": {},
        },
    )
    summary = episode._run_q_regime_phase(
        phase="survival",
        batches=[_batch()],
        frozen_p_model=deepcopy(episode.firm_target),
        q_target_model=deepcopy(episode.firm_target),
        epochs=1,
        validation_bank=[_batch()],
    )
    assert summary["status"] == "rejected_numerical"
    assert summary["rollback_reason"] == "nonfinite_validation_start"


@pytest.mark.parametrize("refresh_mode,same_target", [("stage", True), ("phase", False)])
def test_q_target_refresh_hash_semantics(monkeypatch, refresh_mode, same_target):
    episode = _episode()
    episode.hyperparams.q_target_refresh_mode = refresh_mode
    episode.hyperparams.q_epoch_validation_enabled = False
    seen = {}

    def fake_phase(self, *, phase, q_target_model, **_kwargs):
        seen[phase] = self._state_dict_hash(q_target_model)
        if phase == "survival":
            with torch.no_grad():
                next(self.models["policy_value"].q_head.parameters()).add_(0.5)
        if phase == "zero":
            return {"phase": phase, "status": "structural_verified", "optimizer_steps": 0, "metrics": {"q_structural_abs_max": 0.0}, "coverage": {}}
        if phase == "default":
            return {"phase": phase, "status": "no_realized_default_observed", "optimizer_steps": 0, "metrics": {}, "coverage": {"realized_parent_default_count": 0}}
        return {
            "phase": phase,
            "status": "accepted",
            "optimizer_steps": 1,
            "metrics": {},
            "coverage": {"claim_total_sample_count": 2, "realized_parent_survival_count": 2},
            "q_target_hash": seen[phase],
            "q_online_hash_before": "before",
            "q_online_hash_after": "after",
        }

    episode._run_q_regime_phase = types.MethodType(fake_phase, episode)
    frozen_p = deepcopy(episode.firm_target)
    before = episode._state_dict_hash(frozen_p)
    summary = episode._run_q_regime_training([_batch()], frozen_p)
    assert (seen["survival"] == seen["polish"]) is same_target
    assert summary["frozen_p_hash_before"] == before
    assert summary["frozen_p_hash_after"] == before


def test_structural_failure_is_hard_rejection(monkeypatch):
    episode = _episode()

    def fake_phase(self, *, phase, **_kwargs):
        if phase == "zero":
            return {"phase": phase, "status": "structural_verified", "optimizer_steps": 0, "metrics": {"q_structural_abs_max": 1.0}, "coverage": {}}
        if phase == "default":
            return {"phase": phase, "status": "no_realized_default_observed", "optimizer_steps": 0, "metrics": {}, "coverage": {"realized_parent_default_count": 0}}
        return {"phase": phase, "status": "accepted", "optimizer_steps": 1, "metrics": {}, "coverage": {"claim_total_sample_count": 2, "realized_parent_survival_count": 2}}

    episode._run_q_regime_phase = types.MethodType(fake_phase, episode)
    summary = episode._run_q_regime_training([_batch()], deepcopy(episode.firm_target))
    assert summary["status"] == "rejected_structural"
    assert summary["q_stage_rejection_reason"] == "rejected_structural_zero_identity"


def test_two_episode_hybrid_q_stage_tiny_smoke():
    episode = _episode()
    episode.hyperparams.q_target_refresh_mode = "stage"
    episode.hyperparams.q_bellman_normalize_by_target_scale = True
    episode.hyperparams.q_stage_lr = 1e-4
    episode.hyperparams.q_epoch_validation_enabled = True
    episode.hyperparams.q_validation_max_batches = 1
    episode.hyperparams.q_claim_coverage_enabled = False
    episode.hyperparams.q_min_survival_samples = 1
    episode.hyperparams.q_mixed_polish_epochs = 1
    frozen_p = _AlwaysSurvivalP(episode.models["policy_value"])
    batch = _batch()
    results = []
    for episode_id in (0, 1):
        episode.episode_id = episode_id
        summary = episode._run_q_regime_training(
            [batch], frozen_p, validation_batches=[batch]
        )
        results.append(summary)
        assert summary["status"] in {"accepted_improved", "accepted_reverted"}
        assert summary["q_target_hash_survival"] == summary["q_target_hash_polish"]
        assert summary["q_validation_history"]
        assert summary["q_survival_status"] in {
            "accepted_improved", "accepted_reverted"
        }
        assert summary["q_polish_status"] in {
            "accepted_improved", "accepted_reverted"
        }
        final_phase = summary["phases"][-1]
        metrics = final_phase["metrics"]
        for key in (
            "q_unit_p95",
            "q_unit_max",
            "q_claim_value_p95",
            "q_claim_value_max",
            "q_claim_bellman_abs_mean_raw",
            "q_claim_bellman_abs_mean_normalized",
            "q_recursion_gain_p95",
            "q_recursion_gain_gt1_share",
        ):
            assert np.isfinite(metrics[key])
    assert len(results) == 2
