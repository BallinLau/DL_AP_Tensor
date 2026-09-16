from __future__ import annotations

from pathlib import Path
import copy
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis.bp_eta_imbalance import (  # noqa: E402
    bp_teacher_cache_sha256,
    branch_distillation_loss,
    configure_bp_heads_only,
    eta_switch_panel_sensitivity,
    gradient_contribution_audit,
    parameter_change_summary,
    sample_current_eta_rows,
    state_dict_sha256,
    training_equivalent_total_objective_audit,
)
from config import Config  # noqa: E402
from config.hyperparams import HyperParams  # noqa: E402
from models.policy_value import PolicyValueModel  # noqa: E402


def _cache(n_eta0: int = 12, n_eta1: int = 4) -> dict[str, torch.Tensor]:
    n = n_eta0 + n_eta1
    torch.manual_seed(17)
    parent = torch.randn(n, 7) * 0.1
    parent[:, 0] = torch.linspace(0.05, 0.8, n)
    parent[:, 2] = torch.cat([torch.zeros(n_eta0), torch.ones(n_eta1)])
    parent[:, 3] = 0.4
    target0 = torch.cat([torch.full((n_eta0, 1), 0.02), torch.full((n_eta1, 1), 0.70)])
    targeti = torch.cat([torch.full((n_eta0, 1), 0.04), torch.full((n_eta1, 1), 0.65)])
    return {
        "parent": parent,
        "bp0_target": target0,
        "bpi_target": targeti,
        "mix_target": 0.5 * (target0 + targeti),
        "bp0_confidence": torch.ones(n, 1),
        "bpi_confidence": torch.ones(n, 1),
        "mix_confidence": torch.ones(n, 1),
        "mix_sample_weight": torch.ones(n, 1),
        "eta_current": parent[:, 2:3].to(torch.long),
    }


def _model_and_hp() -> tuple[PolicyValueModel, HyperParams]:
    torch.manual_seed(123)
    model = PolicyValueModel(
        share_hidden_dims=[8],
        share_output_dim=8,
        bp0_head_dims=[4],
        bpi_head_dims=[4],
        dropout=0.0,
    )
    hp = HyperParams()
    hp.bp_grid_policy_loss_space = "logit"
    hp.bp_grid_logit_target_eps = 1e-4
    hp.bp_grid_logit_huber_delta = 1.0
    hp.bp_grid_policy_weight = 1.0
    return model, hp


def test_natural_sampler_matches_cache_eta_ratio_with_replacement():
    cache = _cache(n_eta0=97, n_eta1=3)
    selected = sample_current_eta_rows(
        cache["eta_current"],
        batch_size=10000,
        eta1_share=None,
        generator=torch.Generator().manual_seed(1),
    )
    sampled_share = float(cache["eta_current"][selected].float().mean().item())
    assert selected.numel() == 10000
    assert selected.unique().numel() < selected.numel()
    assert sampled_share == pytest.approx(0.03, abs=0.005)


@pytest.mark.parametrize("target_share", [0.25, 0.50])
def test_stratified_sampler_hits_requested_current_eta_share(target_share: float):
    cache = _cache(n_eta0=97, n_eta1=3)
    selected = sample_current_eta_rows(
        cache["eta_current"],
        batch_size=100,
        eta1_share=target_share,
        generator=torch.Generator().manual_seed(2),
    )
    sampled_share = float(cache["eta_current"][selected].float().mean().item())
    assert sampled_share == pytest.approx(target_share)


def test_diagnostic_sampler_does_not_change_future_eta_configuration():
    cache = _cache()
    hp = HyperParams()
    zeta_before = float(Config.ZETA)
    exact_before = bool(hp.pv_exact_eta_integration_enabled)
    for share in (None, 0.25, 0.50):
        sample_current_eta_rows(
            cache["eta_current"],
            batch_size=8,
            eta1_share=share,
            generator=torch.Generator().manual_seed(3),
        )
    assert float(Config.ZETA) == zeta_before
    assert bool(hp.pv_exact_eta_integration_enabled) == exact_before


def test_head_only_step_changes_only_bp_heads():
    model, hp = _model_and_hp()
    cache = _cache()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    trainable = configure_bp_heads_only(model)
    optimizer = torch.optim.AdamW(trainable, lr=1e-2)
    optimizer.zero_grad(set_to_none=True)
    loss = branch_distillation_loss(model, cache, hp, branch="bp0")
    loss = loss + branch_distillation_loss(model, cache, hp, branch="bpi")
    loss.backward()
    optimizer.step()

    changes = parameter_change_summary(before, model)
    assert changes["bp_head_parameter_max_change"] > 0.0
    assert changes["non_bp_parameter_max_change"] == 0.0
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == name.startswith(("bp0_head.", "bpi_head."))


def test_experiment_copies_share_identical_initial_bp_head_state():
    model, _ = _model_and_hp()
    copy_a = copy.deepcopy(model)
    copy_b = copy.deepcopy(model)
    prefixes = ("bp0_head.", "bpi_head.")
    assert state_dict_sha256(copy_a, prefixes) == state_dict_sha256(copy_b, prefixes)
    assert state_dict_sha256(copy_a, prefixes) == state_dict_sha256(model, prefixes)


def test_gradient_audit_does_not_update_parameters_or_populate_gradients():
    model, hp = _model_and_hp()
    cache = _cache()
    before = state_dict_sha256(model)
    assert all(parameter.grad is None for parameter in model.parameters())

    audit = gradient_contribution_audit(model, cache, hp)

    assert state_dict_sha256(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())
    assert set(audit) == {"bp0", "bpi"}
    for branch in audit.values():
        assert branch["eta0_gradient_norm"] > 0.0
        assert branch["eta1_gradient_norm"] > 0.0
        assert np_is_finite(branch["gradient_cosine_similarity"])


def test_training_equivalent_total_audit_includes_mix_and_takes_no_step():
    model, _ = _model_and_hp()
    cache = _cache()
    before = state_dict_sha256(model)

    def _full_loss(item):
        parent = item["parent"]
        bp0, bpi = model.forward_policy(parent)
        mix = 0.5 * (bp0 + bpi)
        bp0_loss = (bp0 - item["bp0_target"]).square().mean()
        bpi_loss = (bpi - item["bpi_target"]).square().mean()
        mix_loss = (mix - item["mix_target"]).square().mean()
        total = bp0_loss + bpi_loss + mix_loss
        return total, {
            "bp0_loss": float(bp0_loss.detach().item()),
            "bpi_loss": float(bpi_loss.detach().item()),
            "mix_loss": float(mix_loss.detach().item()),
        }

    audit = training_equivalent_total_objective_audit(model, cache, _full_loss)
    eta1_index = torch.where(cache["eta_current"].reshape(-1).bool())[0]
    expected_eta1, expected_parts = _full_loss(
        {key: value[eta1_index] for key, value in cache.items()}
    )

    assert audit["objective"] == "bp0_total + bpi_total + mix_total"
    assert audit["eta1_conditional_loss"] == pytest.approx(float(expected_eta1.detach().item()))
    assert audit["conditional_loss_components"]["eta1"]["mix_loss"] == pytest.approx(
        expected_parts["mix_loss"]
    )
    assert audit["eta0_bp0_head_gradient_norm"] > 0.0
    assert audit["eta1_bpi_head_gradient_norm"] > 0.0
    assert state_dict_sha256(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_teacher_cache_hash_covers_targets_and_is_stable():
    cache = _cache()
    cache.update({
        "bp0_regret": torch.zeros_like(cache["bp0_target"]),
        "bpi_regret": torch.zeros_like(cache["bpi_target"]),
        "mix_regret": torch.zeros_like(cache["mix_target"]),
    })
    first = bp_teacher_cache_sha256(cache)
    clone = {key: value.clone() for key, value in cache.items()}
    assert bp_teacher_cache_sha256(clone) == first
    clone["bpi_target"][0, 0] += 0.01
    assert bp_teacher_cache_sha256(clone) != first


def test_eta_switch_panel_uses_actual_parent_rows():
    model, _ = _model_and_hp()
    cache = _cache()
    summary = eta_switch_panel_sensitivity(model, cache["parent"], max_rows=8)
    assert summary["n_actual_parent_states"] == 8
    assert summary["selection"] == "evenly_spaced"
    for branch in ("delta_bp0_eta", "delta_bpi_eta"):
        assert set(summary[branch]) == {"mean", "median", "p10", "p90"}
        assert all(np_is_finite(value) for value in summary[branch].values())


def np_is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
