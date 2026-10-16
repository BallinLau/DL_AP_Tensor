from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from analysis.run_neural_policy_iteration import (
    PolicyView,
    assert_outer_invariants,
    detect_two_cycle,
    function_transition_metrics,
    policy_transition_metrics,
    run_boundary_only_loop,
)
from analysis.test_frozen_policy_root_solve import (
    FrozenBranchPolicy,
    assert_legacy_m1,
    dynamic_rhs,
)


def _view(*, value: float = 0.0, active: bool = False, survive: bool = True, bp: float = 0.0) -> PolicyView:
    parent = torch.zeros(4, 7)
    parent[:, 0] = torch.arange(4)
    selector = torch.full((4, 3), active)
    return PolicyView(
        parent=parent,
        scale=torch.ones(4, 1),
        p0=torch.full((4, 1), value),
        pi=torch.full((4, 1), value + 1.0),
        investment_selector=selector,
        survival=torch.full((4,), survive),
        bp0=torch.full((4, 1), bp),
        bpI=torch.full((4, 1), bp),
    )


def test_outer_warm_start_uses_previous_root():
    seen = []

    def build(model):
        return {"model": model}

    def solve(model, policy, outer):
        seen.append((outer, model, policy["model"]))
        return model + 1

    run_boundary_only_loop(0, 3, build, solve, lambda policy: str(policy["model"]))
    assert seen == [(0, 0, 0), (1, 1, 1), (2, 2, 2)]


def test_each_outer_refreshes_policy_from_current_and_root_models():
    calls = []

    def build(model):
        calls.append(model)
        return (model,)

    run_boundary_only_loop(0, 2, build, lambda model, policy, outer: model + 1, repr)
    assert calls == [0, 1, 1, 2]


def test_root_optimizer_must_not_mutate_policy_hash():
    def build(model):
        return {"value": model}

    def solve(model, policy, outer):
        policy["value"] = 99
        return model + 1

    with pytest.raises(RuntimeError, match="mutated the frozen policy"):
        run_boundary_only_loop(0, 1, build, solve, lambda policy: str(policy["value"]))


def test_outer_boundary_allows_policy_hash_to_change():
    records = run_boundary_only_loop(
        0, 2, lambda model: (model,), lambda model, policy, outer: model + 1, repr
    )
    assert records[0][0] != records[0][1]
    assert records[1][0] != records[1][1]


def test_q_hash_is_invariant():
    assert_outer_invariants(
        q_before="q", q_after="q", bp_before="bp", bp_after="bp",
        parent_bank_before="p", parent_bank_after="p",
        child_bank_before="c", child_bank_after="c", legacy_m1=True,
    )
    with pytest.raises(RuntimeError, match="Q hash"):
        assert_outer_invariants(
            q_before="q0", q_after="q1", bp_before="bp", bp_after="bp",
            parent_bank_before="p", parent_bank_after="p",
            child_bank_before="c", child_bank_after="c", legacy_m1=True,
        )


def test_parent_and_child_bank_hashes_are_invariant():
    with pytest.raises(RuntimeError, match="parent bank hash"):
        assert_outer_invariants(
            q_before="q", q_after="q", bp_before="bp", bp_after="bp",
            parent_bank_before="p0", parent_bank_after="p1",
            child_bank_before="c", child_bank_after="c", legacy_m1=True,
        )
    with pytest.raises(RuntimeError, match="child bank hash"):
        assert_outer_invariants(
            q_before="q", q_after="q", bp_before="bp", bp_after="bp",
            parent_bank_before="p", parent_bank_after="p",
            child_bank_before="c0", child_bank_after="c1", legacy_m1=True,
        )


def test_legacy_m1_is_enforced():
    assert assert_legacy_m1(torch.ones(3, 2)) == 0.0
    with pytest.raises(RuntimeError, match="legacy M=1"):
        assert_legacy_m1(torch.tensor([[1.0, 0.9]]))
    with pytest.raises(RuntimeError, match="legacy M=1"):
        assert_outer_invariants(
            q_before="q", q_after="q", bp_before="bp", bp_after="bp",
            parent_bank_before="p", parent_bank_after="p",
            child_bank_before="c", child_bank_after="c", legacy_m1=False,
        )


class _ToyValue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.i_threshold = 1.0
        self.i_grid_size = 2

    def _value_outputs(self, states):
        value = self.weight * states[:, 0:1]
        return value, value + 0.1


def test_child_continuation_is_not_detached():
    model = _ToyValue()
    child = torch.zeros(2, 3, 7)
    child[..., 0] = 1.0
    branch = FrozenBranchPolicy(
        name="p0", child_states=child, child_weights=torch.full((2, 3), 1 / 3),
        selector_pi=torch.zeros(2, 2, 3, dtype=torch.bool), survival=torch.ones(2, 3),
        cashflow=torch.zeros(2, 1), bp_star=torch.zeros(2, 1),
        production_target=torch.zeros(2, 1), continuation_factor=1.0,
    )
    dynamic_rhs(model, branch).sum().backward()
    assert model.weight.grad is not None
    assert float(model.weight.grad.abs()) > 0


def test_no_policy_search_occurs_inside_solver_callback():
    build_calls = 0

    def build(model):
        nonlocal build_calls
        build_calls += 1
        return (model,)

    def solve(model, policy, outer):
        before = build_calls
        for _ in range(20):
            _ = model * model
        assert build_calls == before
        return model + 1

    run_boundary_only_loop(0, 3, build, solve, repr)


def test_policy_refresh_only_occurs_at_outer_boundaries():
    calls = []
    run_boundary_only_loop(
        0, 4, lambda model: calls.append(model) or (model,),
        lambda model, policy, outer: model + 1, repr,
    )
    assert len(calls) == 8
    assert calls == [0, 1, 1, 2, 2, 3, 3, 4]


def test_transition_metrics_match_known_synthetic_case():
    old = _view(value=0.0, active=False, survive=True, bp=0.0)
    new = _view(value=2.0, active=True, survive=False, bp=0.5)
    policy = policy_transition_metrics(old, new)
    function = function_transition_metrics(old, new)
    assert policy["investment_selector_flip_share"] == 1.0
    assert policy["default_survival_flip_share"] == 1.0
    assert policy["bp0_mean"] == pytest.approx(0.5)
    assert policy["bpI_p90"] == pytest.approx(0.5)
    assert function["p0_function_change_mean"] == pytest.approx(2.0)
    assert function["pi_function_change_p90"] == pytest.approx(2.0)


def test_two_cycle_detector_finds_a_to_b_to_a():
    a = _view(active=False, survive=True, bp=0.0)
    b = _view(active=True, survive=False, bp=0.5)
    a_again = replace(a, p0=a.p0.clone(), pi=a.pi.clone())
    result = detect_two_cycle(a_again, b, a)
    assert result["possible_two_cycle"] is True
    assert result["two_step_investment_flip"] == 0.0
    assert result["one_step_investment_flip"] == 1.0
