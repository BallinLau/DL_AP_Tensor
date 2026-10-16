from __future__ import annotations

import pytest
import torch

from analysis.run_coupled_fixed_macro import (
    CoupledView,
    assert_block_invariants,
    bounded_coupled_loop,
    detect_coupled_two_cycle,
)
from analysis.run_neural_policy_iteration import PolicyView


def _view(value: float, *, active: bool, survive: bool, bp: float) -> CoupledView:
    parent = torch.zeros(8, 7)
    parent[:, 0] = torch.linspace(0.0, 1.0, 8)
    policy = PolicyView(
        parent=parent,
        scale=torch.ones(8, 1),
        p0=torch.full((8, 1), value),
        pi=torch.full((8, 1), value + 0.1),
        investment_selector=torch.full((8, 3), active),
        survival=torch.full((8,), survive),
        bp0=torch.full((8, 1), bp),
        bpI=torch.full((8, 1), bp),
    )
    return CoupledView(
        policy=policy,
        q=torch.full((8, 1), value),
        phat=torch.full((8, 1), value + 0.05),
    )


def test_p_block_freezes_q_and_bp_hashes() -> None:
    assert_block_invariants(
        block="P", frozen_before="q:bp", frozen_after="q:bp",
        macro_before="m", macro_after="m", parent_before="p", parent_after="p",
        child_before="c", child_after="c",
    )
    with pytest.raises(RuntimeError, match="P frozen-parameter hash"):
        assert_block_invariants(
            block="P", frozen_before="q0:bp", frozen_after="q1:bp",
            macro_before="m", macro_after="m", parent_before="p", parent_after="p",
            child_before="c", child_after="c",
        )


def test_q_block_freezes_p_and_bp_hashes() -> None:
    assert_block_invariants(
        block="Q", frozen_before="p:bp", frozen_after="p:bp",
        macro_before="m", macro_after="m", parent_before="p", parent_after="p",
        child_before="c", child_after="c",
    )
    with pytest.raises(RuntimeError, match="Q frozen-parameter hash"):
        assert_block_invariants(
            block="Q", frozen_before="p0:bp", frozen_after="p1:bp",
            macro_before="m", macro_after="m", parent_before="p", parent_after="p",
            child_before="c", child_after="c",
        )


@pytest.mark.parametrize(
    "field,match",
    [
        ("macro_after", "macro hash"),
        ("parent_after", "parent bank hash"),
        ("child_after", "child bank hash"),
    ],
)
def test_macro_and_banks_must_remain_constant(field: str, match: str) -> None:
    values = dict(
        block="Q", frozen_before="x", frozen_after="x",
        macro_before="m", macro_after="m",
        parent_before="p", parent_after="p",
        child_before="c", child_after="c",
    )
    values[field] = "changed"
    with pytest.raises(RuntimeError, match=match):
        assert_block_invariants(**values)


def test_outer_warm_start_and_max_are_respected() -> None:
    seen = []

    def p_step(state, outer):
        seen.append(("p", outer, state))
        return state + 10

    def q_step(state, outer):
        seen.append(("q", outer, state))
        return state + 1

    states = bounded_coupled_loop(0, 3, p_step, q_step)
    assert states == [0, 11, 22, 33]
    assert seen == [
        ("p", 1, 0), ("q", 1, 10),
        ("p", 2, 11), ("q", 2, 21),
        ("p", 3, 22), ("q", 3, 32),
    ]
    with pytest.raises(ValueError, match=r"\[1, 5\]"):
        bounded_coupled_loop(0, 6, p_step, q_step)


def test_two_cycle_detector_includes_p_q_and_policy() -> None:
    two_back = _view(0.0, active=False, survive=True, bp=0.0)
    previous = _view(1.0, active=True, survive=False, bp=0.5)
    current = _view(0.0, active=False, survive=True, bp=0.0)
    result = detect_coupled_two_cycle(current, previous, two_back)
    assert result["possible_two_cycle"]
    assert result["q_two_cycle"]
    assert result["p_function_two_cycle"]
