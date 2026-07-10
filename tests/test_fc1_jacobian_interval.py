from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from config.hyperparams import HyperParams  # noqa: E402
from losses.sdf_loss import SDFLoss  # noqa: E402
from training.episode import Episode, SDFTrainingPhase  # noqa: E402


class _ToySdfFc1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_hat = torch.nn.Parameter(torch.tensor(0.8))
        self.c_lnk = torch.nn.Parameter(torch.tensor(0.2))
        self.k_hat = torch.nn.Parameter(torch.tensor(-0.1))
        self.k_lnk = torch.nn.Parameter(torch.tensor(0.7))

    def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical=True):
        n_children = x_curr.shape[1]
        hat = hatcf_prev.unsqueeze(1)
        lnk = lnkf_prev.unsqueeze(1)
        c_children = self.c_hat * hat + self.c_lnk * lnk + 0.01 * x_curr
        k_children = self.k_hat * hat + self.k_lnk * lnk + 0.02 * x_curr
        c_parent = hatcf_prev
        w_parent = torch.exp(c_parent).clamp_max(10.0) + 2.0
        w_children = torch.exp(c_children).clamp_max(10.0) + 2.0
        m = torch.ones(x_curr.shape[0], n_children, 1, device=x_curr.device, dtype=x_curr.dtype)
        return w_parent, w_children, m, c_children, k_children


def _episode_with_interval(interval: int) -> Episode:
    hp = HyperParams()
    hp.sdf_fresh_pair_enabled = False
    hp.fc1_forecast_recon_weight = 0.1
    hp.fc1_rollout_weight = 0.0
    hp.fc1_jacobian_penalty_weight = 1.0
    hp.fc1_jacobian_penalty_interval = interval
    hp.fc1_use_true_macro_state_in_stage2 = True
    ep = Episode.__new__(Episode)
    ep.models = {"sdf_fc1": _ToySdfFc1()}
    ep.loss_fns = {"sdf": SDFLoss(wealth_loss_mode="legacy_abs_log1p")}
    ep.hyperparams = hp
    ep.config = Config
    ep.device = torch.device("cpu")
    ep.add_FC1loss = True
    ep._fc1_teacher_forcing_stage = False
    ep.sdf_training_phase = SDFTrainingPhase.FC1_ONLY
    ep._current_epoch_idx = 0
    ep.step_count = 0
    ep.sdf_fc1_step_count = 0
    ep._latest_sdf_terms = {}
    ep._latest_sdf_diag = {}
    return ep


def _batch():
    parent = torch.tensor(
        [
            [0.2, 0.1, 1.0, 0.0, -1.0, -0.20, 4.50, -0.10, 4.60],
            [0.4, 0.2, 0.0, 0.0, -0.8, -0.15, 4.55, -0.12, 4.62],
        ],
        dtype=torch.float32,
    )
    child0 = torch.tensor(
        [
            [0.1, 0.2, 1.0, 0.0, -0.9, -0.18, 4.52, -0.11, 4.61],
            [0.3, 0.1, 0.0, 0.0, -0.7, -0.16, 4.57, -0.13, 4.63],
        ],
        dtype=torch.float32,
    )
    child1 = torch.tensor(
        [
            [0.1, 0.3, 0.0, 0.0, -1.1, -0.19, 4.51, -0.12, 4.60],
            [0.3, 0.2, 1.0, 0.0, -0.6, -0.14, 4.58, -0.14, 4.64],
        ],
        dtype=torch.float32,
    )
    return {"parent": parent, "children": [child0, child1]}


def test_fc1_jacobian_penalty_uses_interval_gate():
    ep = _episode_with_interval(interval=2)
    batch = _batch()

    ep.sdf_fc1_step_count = 0
    ep._compute_sdf_loss(batch)
    assert ep._latest_sdf_terms["sdf_jacobian_penalty_active"] == 1.0
    assert ep._latest_sdf_terms["sdf_jacobian_penalty"] > 0.0

    ep.sdf_fc1_step_count = 1
    ep._compute_sdf_loss(batch)
    assert ep._latest_sdf_terms["sdf_jacobian_penalty_active"] == 0.0
    assert ep._latest_sdf_terms["sdf_jacobian_penalty"] == 0.0


def test_fc1_jacobian_penalty_uses_sdf_counter_not_global_counter():
    ep = _episode_with_interval(interval=2)
    ep.step_count = 999
    ep.sdf_fc1_step_count = 0

    ep._compute_sdf_loss(_batch())

    assert ep._latest_sdf_terms["sdf_jacobian_penalty_active"] == 1.0
    assert ep._latest_sdf_terms["sdf_fc1_step_count"] == 0.0


def test_fc1_jacobian_interval_zero_disables_penalty():
    ep = _episode_with_interval(interval=0)
    ep.sdf_fc1_step_count = 0

    ep._compute_sdf_loss(_batch())

    assert ep._latest_sdf_terms["sdf_jacobian_penalty_active"] == 0.0
    assert ep._latest_sdf_terms["sdf_jacobian_penalty"] == 0.0


def test_fc1_jacobian_interval_one_computes_every_sdf_step():
    ep = _episode_with_interval(interval=1)
    batch = _batch()

    for sdf_step in (0, 1, 2):
        ep.sdf_fc1_step_count = sdf_step
        ep._compute_sdf_loss(batch)
        assert ep._latest_sdf_terms["sdf_jacobian_penalty_active"] == 1.0
        assert ep._latest_sdf_terms["sdf_jacobian_penalty"] > 0.0


def test_fc1_rollout_enabled_requires_sequence_tensors():
    ep = _episode_with_interval(interval=0)
    ep.hyperparams.fc1_rollout_weight = 0.5

    try:
        ep._compute_sdf_loss(_batch())
    except RuntimeError as exc:
        assert "FC1 rollout enabled but batch is missing" in str(exc)
    else:
        raise AssertionError("Expected missing rollout tensors to fail fast.")


def test_fc1_jacobian_active_batch_backward_has_finite_grads():
    ep = _episode_with_interval(interval=1)
    loss = ep._compute_sdf_loss(_batch())

    loss.backward()

    grads = [p.grad for p in ep.models["sdf_fc1"].parameters() if p.grad is not None]
    assert grads
    for grad in grads:
        assert torch.isfinite(grad).all()


if __name__ == "__main__":
    test_fc1_jacobian_penalty_uses_interval_gate()
    test_fc1_jacobian_penalty_uses_sdf_counter_not_global_counter()
    test_fc1_jacobian_interval_zero_disables_penalty()
    test_fc1_jacobian_interval_one_computes_every_sdf_step()
    test_fc1_rollout_enabled_requires_sequence_tensors()
    test_fc1_jacobian_active_batch_backward_has_finite_grads()
