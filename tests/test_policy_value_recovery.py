from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from config.hyperparams import HyperParams  # noqa: E402
from training.episode import Episode  # noqa: E402


class ScalarPolicyValue(torch.nn.Module):
    def __init__(self, value: float = 1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([value], dtype=torch.float32))


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, lr: float):
        super().__init__(params, lr=lr)
        self.step_calls = 0

    def step(self, closure=None):  # noqa: D401
        self.step_calls += 1
        return super().step(closure=closure)


def _episode(
    scale_fn,
    *,
    lr: float = 0.1,
    weight: float = 1.0,
) -> Episode:
    device = torch.device("cpu")
    Config.DEVICE = device
    model = ScalarPolicyValue(weight).to(device)
    optimizer = CountingSGD(model.parameters(), lr=lr)
    hp = HyperParams()
    hp.firm_target_update = "epoch_hard"
    hp.pv_grad_clip_norm = 10.0
    hp.pv_grad_soft_threshold = 100.0
    hp.pv_grad_hard_threshold = 1000.0
    hp.pv_loss_hard_threshold = 1000.0
    hp.pv_epoch_max_retries = 1
    hp.pv_retry_lr_decay = 0.3
    hp.pv_epoch_max_skip_ratio = 0.05
    hp.bp_refine_steps_per_epoch = 0
    hp.q_pretrain_epochs = 1
    hp.q_warmstart_epochs = 0
    episode = Episode(
        models={"policy_value": model},
        optimizers={"policy_value": optimizer},
        config=Config,
        hyperparams=hp,
        device=device,
        firm_target=ScalarPolicyValue(weight).to(device),
    )
    episode._prepare_sdf_shock_bank_for_epoch = lambda *args, **kwargs: None
    episode.evaluate_bellman_convergence = lambda *args, **kwargs: {"passed": True}
    episode._set_policy_q_only_freeze = lambda *args, **kwargs: None
    episode._set_policy_bp_only_freeze = lambda *args, **kwargs: None
    episode._latest_p0_terms = {}
    episode._latest_pi_terms = {}
    episode._latest_q_terms = {}

    def _loss(_batch):
        scale = scale_fn()
        if isinstance(scale, str) and scale == "nan":
            return model.weight.sum() * torch.tensor(float("nan"))
        return model.weight.sum() * float(scale)

    episode._compute_p0_loss = _loss
    episode._compute_pi_loss = _loss
    episode._compute_q_loss = _loss
    return episode


def test_policy_value_soft_spike_clips_and_steps():
    episode = _episode(lambda: 125.0)
    before = episode.models["policy_value"].weight.detach().clone()

    losses = episode.train_step({}, ["policy_value"], policy_loss_terms=["p0"])

    assert losses["pv_batch_action"] == "soft_clip_step"
    assert losses["pv_optimizer_step_executed"] == 1.0
    assert losses["pv_raw_grad_norm"] > episode.hyperparams.pv_grad_soft_threshold
    assert episode.optimizers["policy_value"].step_calls == 1
    assert torch.allclose(
        episode.models["policy_value"].weight.detach(),
        before - torch.tensor([episode.hyperparams.pv_grad_clip_norm * 0.1]),
        atol=1e-6,
    )


def test_policy_value_hard_spike_skips_without_optimizer_step():
    episode = _episode(lambda: 1500.0)
    before = episode.models["policy_value"].weight.detach().clone()

    losses = episode.train_step({}, ["policy_value"], policy_loss_terms=["p0"])

    assert losses["pv_batch_action"] == "skip_batch"
    assert losses["pv_batch_skipped"] == 1.0
    assert losses["pv_optimizer_step_executed"] == 0.0
    assert episode.optimizers["policy_value"].step_calls == 0
    assert torch.allclose(episode.models["policy_value"].weight.detach(), before)


def test_policy_value_nonfinite_requests_epoch_rollback_without_step():
    episode = _episode(lambda: "nan")
    before = episode.models["policy_value"].weight.detach().clone()

    losses = episode.train_step({}, ["policy_value"], policy_loss_terms=["p0"])

    assert losses["pv_batch_action"] == "rollback_epoch"
    assert losses["pv_requires_epoch_rollback"] == 1.0
    assert losses["pv_optimizer_step_executed"] == 0.0
    assert episode.optimizers["policy_value"].step_calls == 0
    assert torch.allclose(episode.models["policy_value"].weight.detach(), before)


def test_policy_value_epoch_retry_accepts_after_rollback_and_decays_lr():
    calls = iter(["nan", 1.0])
    episode = _episode(lambda: next(calls), lr=0.1)

    result = episode._run_batches(
        [{}],
        n_epochs=1,
        log_interval=1,
        train_modules=["policy_value"],
    )

    metadata = result["metadata"]
    assert metadata["policy_value_stage_status"] == "accepted"
    assert metadata["policy_value_epoch_retries"] == 1
    assert metadata["policy_value_accepted_epochs"] == 1
    assert episode.optimizers["policy_value"].step_calls == 1
    assert episode.optimizers["policy_value"].param_groups[0]["lr"] == pytest.approx(0.03)


def test_policy_value_retry_failure_restores_last_good_and_continues():
    episode = _episode(lambda: "nan", lr=0.1)
    model = episode.models["policy_value"]
    target = episode.firm_target
    before_model = model.weight.detach().clone()
    before_target = target.weight.detach().clone()

    result = episode._run_batches(
        [{}],
        n_epochs=1,
        log_interval=1,
        train_modules=["policy_value"],
    )

    metadata = result["metadata"]
    assert metadata["policy_value_stage_status"] == "degraded_recovery"
    assert metadata["policy_value_accepted_epochs"] == 0
    assert episode.optimizers["policy_value"].step_calls == 0
    assert torch.allclose(model.weight.detach(), before_model)
    assert torch.allclose(target.weight.detach(), before_target)
