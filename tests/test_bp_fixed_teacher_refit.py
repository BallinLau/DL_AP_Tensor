from __future__ import annotations

import copy

import torch

from experiments.run_bp_fixed_teacher_refit_probe import (
    assert_only_allowed_changed,
    configure_refit_trainable_parameters,
    forward_policy_bundle,
    refit_loss,
)
from experiments.run_utils import build_hyperparams
from models.policy_value import PolicyValueModel


def _grad_norm(module: torch.nn.Module) -> float:
    return float(sum(
        p.grad.detach().pow(2).sum().item()
        for p in module.parameters()
        if p.grad is not None
    ) ** 0.5)


def test_fixed_teacher_refit_decreases_loss_and_preserves_original_model():
    torch.manual_seed(123)
    hp = build_hyperparams()
    original = PolicyValueModel()
    probe = copy.deepcopy(original)
    original_before = {name: p.detach().clone() for name, p in original.named_parameters()}
    probe_before = {name: p.detach().clone() for name, p in probe.named_parameters()}
    trainable = configure_refit_trainable_parameters(probe)
    assert trainable
    assert all(
        name.startswith(("policy_encoder.", "bp0_head.", "bpi_head."))
        for name in trainable
    )

    parent_state = torch.rand(16, 7)
    target = torch.full((16, 1), 0.4)
    confidence = torch.ones(16, 1)
    targets = {
        "bp0": target,
        "bp0_confidence": confidence,
        "bpI": target,
        "bpI_confidence": confidence,
        "bp_mix": target,
        "bp_mix_confidence": confidence,
        "bp_mix_survival_weight": confidence,
    }
    idx = torch.arange(parent_state.shape[0])
    opt = torch.optim.AdamW([p for p in probe.parameters() if p.requires_grad], lr=1e-2)

    initial_bundle = forward_policy_bundle(probe, parent_state)
    initial_loss, _ = refit_loss(initial_bundle, targets, idx, hp)
    saw_bp_grad = False
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        bundle = forward_policy_bundle(probe, parent_state)
        loss, _ = refit_loss(bundle, targets, idx, hp)
        loss.backward()
        saw_bp_grad = saw_bp_grad or _grad_norm(probe.bp0_head) > 0.0 or _grad_norm(probe.bpi_head) > 0.0
        opt.step()

    final_bundle = forward_policy_bundle(probe, parent_state)
    final_loss, _ = refit_loss(final_bundle, targets, idx, hp)
    assert final_loss.item() < initial_loss.item()
    assert saw_bp_grad

    for name, param in original.named_parameters():
        torch.testing.assert_close(param, original_before[name])
    assert_only_allowed_changed(probe_before, probe)

    changed_allowed = [
        name
        for name, param in probe.named_parameters()
        if name.startswith(("policy_encoder.", "bp0_head.", "bpi_head."))
        and not torch.equal(probe_before[name], param.detach())
    ]
    assert changed_allowed
