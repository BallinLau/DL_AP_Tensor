from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from config.hyperparams import HyperParams  # noqa: E402
from models.policy_value import PolicyValueModel  # noqa: E402
from training.episode import Episode  # noqa: E402


def _small_hyperparams() -> HyperParams:
    hp = HyperParams()
    hp.pv_bp_training_mode = "target_grid"
    hp.bp_grid_coarse_size = 3
    hp.bp_grid_refine_enabled = False
    hp.bp_grid_candidate_chunk_size = 2
    hp.bp_grid_parent_chunk_size = 2
    hp.bp_grid_max_expanded_states = 16
    hp.bp_grid_mix_policy_weight = 1.0
    hp.bp_grid_margin_scale = 1e-4
    hp.bp_grid_confidence_relative = True
    hp.pv_use_clipped_m = False
    hp.firm_target_update = "epoch_hard"
    hp.epochs = 1
    hp.batch_size = 2
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


def test_target_grid_loss_logs_mix_regret_and_freezes_target_gradients():
    device = torch.device("cpu")
    Config.DEVICE = device
    online = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8).to(device)
    target = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8).to(device)
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
    episode = Episode(
        models={"policy_value": online},
        optimizers={"policy_value": optimizer},
        config=Config,
        hyperparams=_small_hyperparams(),
        device=device,
        firm_target=target,
    )

    batch = _batch(device)
    loss = episode._compute_p0_loss(batch) + episode._compute_pi_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()

    assert "mix_grid_bp_mae" in episode._latest_pi_terms
    assert "mix_grid_regret_p90" in episode._latest_pi_terms
    assert episode._latest_pi_terms["mix_grid_policy_weight"] == 1.0

    online_grad = [
        p.grad.detach().abs().sum().item()
        for p in online.parameters()
        if p.grad is not None
    ]
    assert online_grad and sum(online_grad) > 0.0
    assert all(p.grad is None for p in episode.firm_target.parameters())


def test_epoch_hard_target_is_fixed_within_epoch_and_updates_at_epoch_end():
    online = torch.nn.Linear(2, 1)
    target = torch.nn.Linear(2, 1)
    episode = Episode.__new__(Episode)
    episode.models = {"policy_value": online}
    episode.firm_target = target
    episode.hyperparams = HyperParams()
    episode.hyperparams.firm_target_update = "epoch_hard"
    episode.step_count = 0

    with torch.no_grad():
        target.weight.fill_(0.0)
        target.bias.fill_(0.0)
        online.weight.fill_(2.0)
        online.bias.fill_(3.0)

    episode._maybe_update_firm_target(["policy_value"])
    assert torch.allclose(target.weight, torch.zeros_like(target.weight))
    assert torch.allclose(target.bias, torch.zeros_like(target.bias))

    episode._maybe_update_firm_target_epoch(["policy_value"])
    assert torch.allclose(target.weight, online.weight)
    assert torch.allclose(target.bias, online.bias)
    assert not target.training
    assert all(not p.requires_grad for p in target.parameters())


if __name__ == "__main__":
    test_target_grid_loss_logs_mix_regret_and_freezes_target_gradients()
    test_epoch_hard_target_is_fixed_within_epoch_and_updates_at_epoch_end()
