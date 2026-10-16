from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.test_frozen_policy_root_solve import (
    BP_PREFIXES,
    Q_PREFIXES,
    FrozenBranchPolicy,
    FrozenPolicyBatch,
    _expand_i_grid_states,
    _state_hash,
    assert_legacy_m1,
    current_frozen_child_equity,
    current_frozen_child_equity_loop,
    current_frozen_child_equity_vectorized,
    coalesce_problem,
    dynamic_rhs,
    evaluate_problem,
    initial_parity,
    residual_tensors,
    root_loss,
    run_adam,
    run_lbfgs,
)


class TinyPolicyValue(nn.Module):
    i_threshold = 0.5
    i_grid_size = 3

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.value_encoder = nn.Linear(6, 5)
        self.v0_head = nn.Linear(5, 1)
        self.vi_head = nn.Linear(6, 1)
        self.q_encoder = nn.Linear(6, 2)
        self.q_head = nn.Linear(2, 1)
        self.policy_encoder = nn.Linear(6, 2)
        self.bp0_head = nn.Linear(2, 1)
        self.bpi_head = nn.Linear(2, 1)
        self.barz_model = nn.Identity()
        self.bari_model = nn.Identity()

    def _value_outputs(self, states: torch.Tensor):
        base = torch.cat((states[:, :3], states[:, 4:]), dim=1)
        hidden = torch.tanh(self.value_encoder(base))
        p0 = self.v0_head(hidden)
        pi = self.vi_head(torch.cat((hidden, states[:, 3:4]), dim=1))
        return p0, pi

    def equity_value_scale(self, states: torch.Tensor):
        return 1.0 + torch.exp((states[:, 4:5] + states[:, 1:2]).clamp(max=20))


class FakeEpisode:
    def __init__(self, model: nn.Module):
        self.models = {"policy_value": model}

    def _policy_value_stage_params(self, stage: str):
        assert stage == "p"
        model = self.models["policy_value"]
        modules = (model.value_encoder, model.v0_head, model.vi_head,
                   model.barz_model, model.bari_model)
        return [param for module in modules for param in module.parameters()]


def _branch(model: TinyPolicyValue, states: torch.Tensor, weights: torch.Tensor,
            name: str, factor: float) -> FrozenBranchPolicy:
    flat = states.reshape(-1, 7)
    p0_rows, pi_rows = [], []
    for i_value in torch.linspace(0.0, model.i_threshold, model.i_grid_size):
        value = flat.clone(); value[:, 3] = i_value
        p0, pi = model._value_outputs(value)
        p0_rows.append(p0.reshape(states.shape[:2])); pi_rows.append(pi.reshape(states.shape[:2]))
    p0_grid, pi_grid = torch.stack(p0_rows), torch.stack(pi_rows)
    selector = pi_grid > p0_grid
    phat = torch.where(selector, pi_grid, p0_grid).mean(0)
    survival = (phat > 0).float()
    cashflow = torch.full((states.shape[0], 1), 0.2 if name == "p0" else 0.1)
    continuation = (weights * survival * phat).sum(1, keepdim=True)
    return FrozenBranchPolicy(
        name=name, child_states=states.detach(), child_weights=weights.detach(),
        selector_pi=selector.detach(), survival=survival.detach(), cashflow=cashflow,
        bp_star=torch.full((states.shape[0], 1), 0.3),
        production_target=(cashflow + factor * continuation).detach(),
        continuation_factor=factor,
        expanded_child_states=_expand_i_grid_states(model, states),
    )


def _fixture():
    model = TinyPolicyValue()
    parent = torch.tensor([
        [0.2, 0.1, 1.0, 0.1, -1.8, -2.0, 4.0],
        [0.4, -0.1, 0.0, 0.3, -2.0, -1.9, 4.2],
        [0.1, 0.2, 1.0, 0.2, -1.9, -2.1, 3.9],
    ])
    children = torch.stack((parent.clone(), parent.clone()), dim=1)
    children[:, 0, 1] += 0.05; children[:, 1, 1] -= 0.04
    weights = torch.full((3, 2), 0.5)
    batch = FrozenPolicyBatch(
        parent=parent, scale=model.equity_value_scale(parent).detach(),
        p0=_branch(model, children, weights, "p0", 1.0),
        pi=_branch(model, children, weights, "pi", 1.14),
    )
    return model, FakeEpisode(model), batch


def test_initial_frozen_rhs_matches_stored_production_target():
    model, _, batch = _fixture()
    rows = initial_parity([batch], model)
    assert all(row["passed"] for row in rows)
    assert max(row["max_abs_error"] for row in rows) < 1e-7


def test_legacy_m1_invariant_accepts_only_one():
    assert assert_legacy_m1(torch.ones(3, 4)) == 0.0
    with pytest.raises(RuntimeError, match="legacy M=1"):
        assert_legacy_m1(torch.tensor([1.0, 0.99]))


def test_q_hash_unchanged_after_root_step():
    model, episode, batch = _fixture(); before = _state_hash(model, Q_PREFIXES)
    run_adam(model=model, episode=episode, train=[batch], validation=[batch],
             steps=1, lr=1e-4, clip=10.0, eval_every=1)
    assert _state_hash(model, Q_PREFIXES) == before


def test_bp_policy_hash_unchanged_after_root_step():
    model, episode, batch = _fixture(); before = _state_hash(model, BP_PREFIXES)
    run_adam(model=model, episode=episode, train=[batch], validation=[batch],
             steps=1, lr=1e-4, clip=10.0, eval_every=1)
    assert _state_hash(model, BP_PREFIXES) == before


def test_frozen_policy_hash_unchanged_through_root_step():
    model, episode, batch = _fixture(); before = batch.digest()
    run_adam(model=model, episode=episode, train=[batch], validation=[batch],
             steps=2, lr=1e-4, clip=10.0, eval_every=1)
    assert batch.digest() == before


def test_child_continuation_has_nonzero_gradient():
    model, _, batch = _fixture()
    rhs = dynamic_rhs(model, batch.p0)
    grads = torch.autograd.grad(rhs.sum(), list(model.value_encoder.parameters()), allow_unused=True)
    assert any(grad is not None and grad.abs().max() > 0 for grad in grads)


def test_detached_child_negative_control_removes_rhs_gradient():
    model, _, batch = _fixture()
    rhs = dynamic_rhs(model, batch.p0, detach_child=True)
    assert not rhs.requires_grad
    full = residual_tensors(model, batch)
    assert full["p0_rhs"].requires_grad


def test_train_validation_bank_digest_is_deterministic():
    _, _, batch = _fixture()
    assert batch.digest() == deepcopy(batch).digest()


def test_smoke_residuals_are_finite():
    model, _, batch = _fixture()
    metrics = evaluate_problem(model, [batch])
    assert metrics["finite"]
    assert torch.isfinite(root_loss(model, batch))


def test_optimizer_does_not_call_bp_grid_search(monkeypatch):
    model, episode, batch = _fixture()
    import training.bp_grid_teacher as module
    monkeypatch.setattr(module.BPGridTeacher, "compute_value_target",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("grid refresh")))
    run_adam(model=model, episode=episode, train=[batch], validation=[batch],
             steps=1, lr=1e-4, clip=10.0, eval_every=1)


def test_optimizer_does_not_refresh_selector_or_default():
    model, episode, batch = _fixture()
    selector = batch.p0.selector_pi.clone(); survival = batch.p0.survival.clone()
    run_adam(model=model, episode=episode, train=[batch], validation=[batch],
             steps=2, lr=1e-3, clip=10.0, eval_every=1)
    assert torch.equal(batch.p0.selector_pi, selector)
    assert torch.equal(batch.p0.survival, survival)


def test_full_residual_gradient_differs_from_semigradient_control():
    model, _, batch = _fixture()
    full = root_loss(model, batch)
    full_grad = torch.autograd.grad(full, model.value_encoder.weight, retain_graph=True)[0]
    detached = residual_tensors(model, batch, detach_child=True)
    semi = detached["p0_normalized"].square().mean() + detached["pi_normalized"].square().mean()
    semi_grad = torch.autograd.grad(semi, model.value_encoder.weight)[0]
    assert not torch.allclose(full_grad, semi_grad)


def test_current_child_equity_uses_frozen_active_sets():
    model, _, batch = _fixture()
    before_selector = batch.pi.selector_pi.clone(); before_survival = batch.pi.survival.clone()
    with torch.no_grad():
        model.vi_head.bias.add_(10.0)
    value = current_frozen_child_equity(model, batch.pi)
    assert torch.isfinite(value).all()
    assert torch.equal(batch.pi.selector_pi, before_selector)
    assert torch.equal(batch.pi.survival, before_survival)


def test_vectorized_child_equity_matches_legacy_loop():
    model, _, batch = _fixture()
    for branch in (batch.p0, batch.pi):
        old = current_frozen_child_equity_loop(model, branch)
        new = current_frozen_child_equity_vectorized(model, branch)
        torch.testing.assert_close(new, old, rtol=1e-6, atol=1e-6)


def test_vectorized_residual_and_loss_match_legacy_loop():
    model, _, batch = _fixture()
    old = residual_tensors(model, batch, implementation="legacy")
    new = residual_tensors(model, batch, implementation="fast")
    for name in ("p0", "pi", "p0_normalized", "pi_normalized", "p0_rhs", "pi_rhs"):
        torch.testing.assert_close(new[name], old[name], rtol=1e-6, atol=1e-6)
    old_loss = root_loss(model, batch, implementation="legacy")
    new_loss = root_loss(model, batch, implementation="fast")
    torch.testing.assert_close(new_loss, old_loss, rtol=1e-6, atol=1e-7)


def test_vectorized_gradient_matches_legacy_loop():
    model, _, batch = _fixture()
    old_model = deepcopy(model)
    new_model = deepcopy(model)
    old_loss = root_loss(old_model, batch, implementation="legacy")
    new_loss = root_loss(new_model, batch, implementation="fast")
    old_grads = torch.autograd.grad(old_loss, tuple(old_model.parameters()), allow_unused=True)
    new_grads = torch.autograd.grad(new_loss, tuple(new_model.parameters()), allow_unused=True)
    for old_grad, new_grad in zip(old_grads, new_grads):
        assert (old_grad is None) == (new_grad is None)
        if old_grad is not None:
            torch.testing.assert_close(new_grad, old_grad, rtol=2e-5, atol=2e-6)


def test_cached_and_noncached_vectorized_paths_match():
    model, _, batch = _fixture()
    cached = residual_tensors(model, batch, implementation="fast", use_cache=True)
    uncached = residual_tensors(model, batch, implementation="fast", use_cache=False)
    for name in ("p0", "pi", "p0_normalized", "pi_normalized"):
        torch.testing.assert_close(cached[name], uncached[name], rtol=0.0, atol=0.0)


def test_fast_residual_reduces_value_forward_calls(monkeypatch):
    model, _, batch = _fixture()
    original = model._value_outputs
    calls = 0

    def counted(states):
        nonlocal calls
        calls += 1
        return original(states)

    monkeypatch.setattr(model, "_value_outputs", counted)
    residual_tensors(model, batch, implementation="legacy")
    legacy_calls = calls
    calls = 0
    residual_tensors(model, batch, implementation="fast")
    fast_calls = calls
    assert legacy_calls == 1 + 2 * model.i_grid_size
    assert fast_calls == 2


def test_coalesced_static_snapshots_preserve_residuals():
    model, _, batch = _fixture()
    merged = coalesce_problem([batch, deepcopy(batch)], batch_size=6)
    assert len(merged) == 1
    individual = residual_tensors(model, batch, implementation="fast")
    combined = residual_tensors(model, merged[0], implementation="fast")
    for name in ("p0", "pi", "p0_normalized", "pi_normalized"):
        expected = torch.cat([individual[name], individual[name]], dim=0)
        torch.testing.assert_close(combined[name], expected, rtol=1e-6, atol=1e-6)


def test_lbfgs_full_bank_closure_runs_with_dynamic_rhs():
    model, episode, batch = _fixture()
    history, _, metrics = run_lbfgs(
        model=model, episode=episode, train=[batch], validation=[batch],
        max_iter=1, history_size=2,
    )
    assert history
    assert metrics["finite"]
