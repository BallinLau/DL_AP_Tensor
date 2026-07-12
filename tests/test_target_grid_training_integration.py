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
    required_active_diag_keys = [
        "mix_grid_refi_active_n",
        "mix_grid_refi_active_available",
        "mix_grid_bp_mae_active",
        "mix_grid_bp_err_p90_active",
        "mix_grid_regret_mean_active",
        "mix_grid_regret_p90_active",
        "mix_grid_bp_star_p50_active",
        "mix_grid_bp_star_p90_active",
        "mix_grid_bp_candidate_p50_active",
        "mix_grid_bp_candidate_p90_active",
        "mix_grid_bp_candidate_low_share_active",
        "mix_grid_default_at_star_mean_active",
        "mix_grid_p_child_at_star_mean_active",
        "mix_grid_q_issue_at_star_mean_active",
    ]
    for key in required_active_diag_keys:
        assert key in episode._latest_pi_terms
        assert torch.isfinite(torch.tensor(episode._latest_pi_terms[key]))
    assert episode._latest_pi_terms["mix_grid_refi_active_n"] == 1.0
    assert episode._latest_pi_terms["mix_grid_refi_active_available"] == 1.0
    assert episode._latest_pi_terms["mix_grid_refi_active_share"] == 0.5

    online_grad = [
        p.grad.detach().abs().sum().item()
        for p in online.parameters()
        if p.grad is not None
    ]
    assert online_grad and sum(online_grad) > 0.0
    assert all(p.grad is None for p in episode.firm_target.parameters())


def test_masked_grid_diagnostics_ignore_inactive_observations():
    value = torch.tensor([[1.0], [100.0]])
    active = torch.tensor([[True], [False]])

    assert Episode._masked_diag_mean(value, active) == 1.0
    assert Episode._masked_diag_quantile(value, active, 0.90) == 1.0


def test_masked_grid_diagnostics_are_finite_without_active_observations():
    value = torch.tensor([[1.0], [2.0]])
    active = torch.tensor([[False], [False]])

    assert Episode._masked_diag_mean(value, active) == 0.0
    assert Episode._masked_diag_quantile(value, active, 0.90) == 0.0


def test_bp_grid_boundary_low_threshold_is_explicit_hyperparameter():
    hp = HyperParams()

    assert hasattr(hp, "bp_grid_boundary_low_threshold")
    assert hp.bp_grid_boundary_low_threshold == 0.05


def test_target_grid_policy_convergence_skips_without_active_refinancing():
    device = torch.device("cpu")
    Config.DEVICE = device
    online = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8).to(device)
    target = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8).to(device)
    episode = Episode(
        models={"policy_value": online},
        optimizers={"policy_value": torch.optim.Adam(online.parameters(), lr=1e-3)},
        config=Config,
        hyperparams=_small_hyperparams(),
        device=device,
        firm_target=target,
    )

    batch = _batch(device)
    batch["parent"][:, 2:3] = 0.0
    result = episode.evaluate_target_grid_policy_convergence([batch])

    assert result["enabled"] is True
    assert result["passed"] is True
    assert result["informative"] is False
    assert result["all_skipped"] is True
    assert result["skip_reason"] == "no_active_refinancing_states"
    assert result["informative_policies"] == []
    assert result["skipped_policies"] == ["bp0", "bpI", "mix"]
    assert result["policies"]["bp0"]["skip_reason"] == "no_active_refinancing_states"
    assert result["policies"]["bpI"]["skip_reason"] == "no_active_refinancing_states"


def test_policy_convergence_selector_prefers_informative_validation():
    train = {"enabled": True, "informative": True, "passed": False}
    val = {"enabled": True, "informative": True, "passed": True}

    selected, source = Episode._select_policy_convergence_result(train, val)

    assert selected is val
    assert source == "validation"


def test_policy_convergence_selector_falls_back_to_training_when_validation_skipped():
    train = {"enabled": True, "informative": True, "passed": True}
    val = {
        "enabled": True,
        "informative": False,
        "all_skipped": True,
        "passed": True,
        "skip_reason": "no_active_refinancing_states",
    }

    selected, source = Episode._select_policy_convergence_result(train, val)

    assert selected is train
    assert source == "training_fallback"


def test_policy_convergence_selector_skips_only_when_train_and_validation_uninformative():
    train = {
        "enabled": True,
        "informative": False,
        "all_skipped": True,
        "passed": True,
        "skip_reason": "no_active_refinancing_states",
    }
    val = {
        "enabled": True,
        "informative": False,
        "all_skipped": True,
        "passed": True,
        "skip_reason": "no_active_refinancing_states",
    }

    selected, source = Episode._select_policy_convergence_result(train, val)

    assert source == "skipped"
    assert selected["passed"] is True
    assert selected["informative"] is False
    assert selected["all_skipped"] is True


def test_policy_convergence_selector_marks_unavailable_errors_as_failed():
    train = {"enabled": False, "informative": False, "passed": False, "policies": {}}
    val = {
        "enabled": True,
        "informative": False,
        "all_skipped": True,
        "passed": True,
        "skip_reason": "no_active_refinancing_states",
    }

    selected, source = Episode._select_policy_convergence_result(train, val)

    assert source == "unavailable"
    assert selected["passed"] is False
    assert selected["skip_reason"] == "policy_convergence_unavailable"


def _module_grad_sum(module: torch.nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().abs().sum().item())
    return total


def test_mixed_policy_conditional_loss_does_not_update_value_heads():
    device = torch.device("cpu")
    Config.DEVICE = device
    online = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8).to(device)
    episode = Episode(
        models={"policy_value": online},
        optimizers={"policy_value": torch.optim.Adam(online.parameters(), lr=1e-3)},
        config=Config,
        hyperparams=_small_hyperparams(),
        device=device,
    )

    parent_state = _batch(device)["parent"][:, :7]
    output = online(parent_state)
    online.zero_grad(set_to_none=True)
    bp_cond_pred = episode._mixed_policy_conditional_bp(
        output,
        output.bp0,
        output.bpI,
        parent_state[:, 0:1],
        fallback_bar_i=output.bar_i_cond,
    )
    loss = (bp_cond_pred - torch.full_like(bp_cond_pred, 0.7)).pow(2).mean()
    loss.backward()

    assert _module_grad_sum(online.policy_encoder) > 0.0
    assert _module_grad_sum(online.bp0_head) > 0.0
    assert _module_grad_sum(online.bpi_head) > 0.0
    assert _module_grad_sum(online.value_encoder) == 0.0
    assert _module_grad_sum(online.v0_head) == 0.0
    assert _module_grad_sum(online.vi_head) == 0.0


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
    test_mixed_policy_conditional_loss_does_not_update_value_heads()
    test_epoch_hard_target_is_fixed_within_epoch_and_updates_at_epoch_end()
