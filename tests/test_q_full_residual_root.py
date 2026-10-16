from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn

from analysis.run_q_root_solve_fixed_macro import (
    QSurvivalBatch,
    _non_q_hash,
    _q_scope,
    _used_m,
    initial_q_target_parity,
    q_dynamic_target_components,
    q_dynamic_targets,
    q_residual_tensors,
    q_root_loss,
    run_q_adam,
)
from losses.q_loss import QLoss, classify_q_parent_regimes


class TinyQModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(31)
        self.q_encoder = nn.Linear(7, 5)
        self.q_head = nn.Linear(5, 1)
        self.value_encoder = nn.Linear(7, 4)
        self.v0_head = nn.Linear(4, 1)
        self.vi_head = nn.Linear(4, 1)
        self.barz_model = nn.Linear(4, 1)
        self.bari_model = nn.Linear(4, 1)
        self.policy_encoder = nn.Linear(7, 3)
        self.bp0_head = nn.Linear(3, 1)
        self.bpi_head = nn.Linear(3, 1)

    def _q_claim_output(self, state: torch.Tensor) -> torch.Tensor:
        return self.q_head(torch.tanh(self.q_encoder(state)))


class TinyQEpisode:
    def __init__(self, model: TinyQModel) -> None:
        self.models = {"policy_value": model}
        self.loss_fns = {
            "q": QLoss(delta=0.1, phi=0.4, g=1.14,
                       recovery_normalization_mode="asset_only")
        }
        self.hyperparams = SimpleNamespace(q_m_clamp_min=0.5, q_m_clamp_max=1.5)

    def _policy_value_stage_params(self, stage: str):
        assert stage == "q"
        model = self.models["policy_value"]
        return [*model.q_encoder.parameters(), *model.q_head.parameters()]

    @staticmethod
    def _collapse_policy_eta_pairs(values):
        return values


def _fixture() -> tuple[TinyQModel, TinyQEpisode, QSurvivalBatch]:
    model = TinyQModel()
    episode = TinyQEpisode(model)
    parent = torch.tensor([
        [0.3, -0.2, 1.0, 0.1, -1.8, -2.0, 4.0],
        [0.7, 0.1, 1.0, 0.2, -2.0, -1.9, 4.2],
        [0.5, 0.2, 1.0, 0.3, -1.9, -2.1, 3.9],
    ])
    children = torch.stack((parent.clone(), parent.clone()), dim=1)
    children[:, 0, 1] += 0.07
    children[:, 1, 1] -= 0.05
    bar_i = torch.tensor([[0.1], [0.2], [0.15]])
    bar_z = torch.tensor([
        [[0.05], [0.20]],
        [[0.15], [0.10]],
        [[0.30], [0.25]],
    ])
    shell = QSurvivalBatch(
        parent=parent,
        child_states=children,
        m=torch.tensor([
            [[0.8], [1.1]],
            [[1.0], [1.2]],
            [[0.7], [1.3]],
        ]),
        bar_i=bar_i,
        bar_z=bar_z,
        production_targets=torch.empty(0),
        production_scales=torch.empty(0),
        m_source="CORRECTED_CHILD_CLAMPED_0.5_1.5",
        regime=torch.full((3,), 2, dtype=torch.long),
    )
    with torch.no_grad():
        targets, scales, _ = q_dynamic_target_components(
            model, episode, shell, detach_child=True
        )
    batch = QSurvivalBatch(**{
        **shell.__dict__,
        "production_targets": targets,
        "production_scales": scales,
    })
    return model, episode, batch


def test_initial_q_target_matches_frozen_production_target() -> None:
    model, episode, batch = _fixture()
    with torch.no_grad():
        target = q_dynamic_targets(model, episode, batch, detach_child=True)
        components = episode.loss_fns["q"].compute_main_components(
            model._q_claim_output(batch.parent), batch.parent[:, 0:1], batch.bar_i,
            list(batch.m.unbind(dim=1)),
            list(model._q_claim_output(batch.child_states.reshape(-1, 7)).reshape(3, 2, 1).unbind(dim=1)),
            list(batch.bar_z.unbind(dim=1)),
            list(batch.child_states[:, :, 4:5].unbind(dim=1)),
            list(batch.child_states[:, :, 1:2].unbind(dim=1)),
        )
        branches = torch.stack([item["q_target_total"] for item in components], dim=1)
        scale = 1.0 + branches.abs().mean(dim=1)
    parity = initial_q_target_parity(model, episode, batch, target, scale)
    assert parity["passed"]
    assert parity["max_abs_error"] == 0.0


def test_full_residual_target_has_child_q_gradient() -> None:
    model, episode, batch = _fixture()
    params, _ = _q_scope(episode, model)
    targets = q_dynamic_targets(model, episode, batch, detach_child=False)
    grads = torch.autograd.grad(targets.sum(), params, allow_unused=True)
    assert any(grad is not None and bool((grad.abs() > 0).any()) for grad in grads)


def test_detached_child_target_is_a_true_negative_control() -> None:
    model, episode, batch = _fixture()
    target = q_dynamic_targets(model, episode, batch, detach_child=True)
    assert not target.requires_grad
    assert q_residual_tensors(model, episode, batch)["target"].requires_grad


def test_full_residual_gradient_differs_from_frozen_target_fit() -> None:
    model, episode, batch = _fixture()
    full = q_root_loss(model, episode, batch)
    full_grad = torch.autograd.grad(full, model.q_encoder.weight, retain_graph=True)[0]
    frozen = q_root_loss(model, episode, batch, frozen_target=True)
    frozen_grad = torch.autograd.grad(frozen, model.q_encoder.weight)[0]
    assert not torch.allclose(full_grad, frozen_grad)


def test_q_optimizer_changes_only_q_parameters() -> None:
    model, episode, batch = _fixture()
    non_q_before = _non_q_hash(model)
    q_before = deepcopy({name: value for name, value in model.state_dict().items()
                         if name.startswith(("q_encoder", "q_head"))})
    history = run_q_adam(
        model=model, episode=episode, train=[batch], validation=[batch],
        steps=2, lr=1e-3, frozen_target=False, eval_every=1, phase="test",
    )
    assert _non_q_hash(model) == non_q_before
    assert any(not torch.equal(value, model.state_dict()[name])
               for name, value in q_before.items())
    assert all(row["clipped_grad_norm"] <= row["raw_grad_norm"] + 1e-12 for row in history)
    assert all(row["parameter_norm"] > 0.0 for row in history)


def test_q_scope_is_exactly_q_encoder_and_head() -> None:
    model, episode, _ = _fixture()
    _, names = _q_scope(episode, model)
    assert names
    assert all(name.startswith(("q_encoder", "q_head")) for name in names)
    assert all(
        parameter.requires_grad == name.startswith(("q_encoder", "q_head"))
        for name, parameter in model.named_parameters()
    )


def test_corrected_m_is_clamped_once_and_reused_in_problem() -> None:
    model, episode, batch = _fixture()
    raw_children = [
        torch.cat((batch.child_states[:, index], torch.tensor([[0.1], [1.0], [2.2]])), dim=1)
        for index in range(batch.child_states.shape[1])
    ]
    used, source = _used_m(episode, raw_children, "corrected_child")
    expected = torch.tensor([[0.5], [1.0], [1.5]])
    assert source == "CORRECTED_CHILD_CLAMPED_0.5_1.5"
    assert all(torch.equal(value, expected) for value in used)
    assert batch.digest() == deepcopy(batch).digest()


def test_parent_regimes_are_disjoint_and_complete_for_valid_support() -> None:
    b = torch.tensor([[0.0], [0.2], [0.4]])
    phat = torch.tensor([[2.0], [-0.1], [0.3]])
    masks = classify_q_parent_regimes(b, phat)
    assigned = masks["zero"].int() + masks["default"].int() + masks["survival"].int()
    assert torch.equal(assigned, torch.ones_like(assigned))
    assert [int(masks[key].sum()) for key in ("zero", "default", "survival")] == [1, 1, 1]
    assert int(masks["unclassified"].sum()) == 0


def test_full_regime_targets_apply_zero_and_default_boundaries() -> None:
    model, episode, batch = _fixture()
    batch = QSurvivalBatch(**{
        **batch.__dict__,
        "regime": torch.tensor([0, 1, 2]),
    })
    target, scale, _ = q_dynamic_target_components(
        model, episode, batch, detach_child=False
    )
    recovery = episode.loss_fns["q"].compute_current_recovery(
        batch.parent[1:2, 0:1], batch.parent[1:2, 4:5], batch.parent[1:2, 1:2]
    )
    assert target[0].item() == 0.0
    assert scale[0].item() == 1.0
    assert torch.allclose(target[1:2], recovery)
    assert torch.allclose(scale[1:2], 1.0 + recovery.abs())
    assert target[2:].requires_grad
