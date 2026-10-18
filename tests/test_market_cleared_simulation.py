from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data.market_clearing import (
    MARKET_CLEARING_V1,
    MarketClearingConfig,
    MarketClearingController,
    MarketClearingError,
)
from data.simulate_ts import SimulateTS
from data.simulate_ts_parallel import _initialize_batched_state
from analysis.market_cleared_banks import build_market_cleared_transition_bank


class _Config:
    DEVICE = torch.device("cpu")
    RHO_X = 0.0
    SIGMA_X = 0.0
    XBAR = 0.0
    RHO_Z = 0.0
    SIGMA_Z = 0.0
    ZBAR = 0.0
    DELTA = 0.0
    PHI = 1.0
    G = 1.0
    I_THRESHOLD = 0.5
    ZETA = 0.0
    SIM_B_INIT_MIN = 0.0
    SIM_B_INIT_MAX = 1.0
    ENTRY_B_MIN = 0.0
    ENTRY_B_MAX = 1.0


class _StaticPolicy(nn.Module):
    """A policy with C_oper=K and entry accepted at every candidate h."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.i_threshold = _Config.I_THRESHOLD

    def forward_value_components(self, state: torch.Tensor):
        one = torch.ones_like(state[:, :1]) + 0.0 * self.anchor
        return {"V0_physical": one, "VI_physical": one + 1.0}

    def forward_simulation(self, state: torch.Tensor):
        n = state.shape[0]
        zero = state.new_zeros((n, 1)) + 0.0 * self.anchor
        one = torch.ones_like(zero)
        return SimpleNamespace(
            Q=zero,
            P0=one,
            PI=one + 1.0,
            bar_i=zero,
            bar_z=zero,
            P=one,
            bp0=zero,
            bpI=zero,
            bp=zero,
        )


def _sim(**overrides) -> SimulateTS:
    kwargs = dict(
        models={"policy_value": _StaticPolicy(), "sdf_fc1": None},
        config=_Config,
        n_paths=2,
        group_size=3,
        horizon=1,
        branch_num=2,
        enable_entry=True,
        enable_exit=False,
        device=torch.device("cpu"),
        entry_mode="investment_compare",
        entry_spec_version="investment_compare_v1",
        entry_capital_ratio=1.0,
        entry_size_ratio=1.0,
        entry_cost_max=_Config.I_THRESHOLD,
        entry_rng_seed=17,
        entry_rng_mode="stable_economic_identity",
        common_transition_seed=23,
        transition_rng_mode="stable_firm_identity",
        consumption_aggregation_mode="raw",
        consumption_floor=0.0,
        node_accounting_mode="economic_node_ledger",
        market_clearing_mode=MARKET_CLEARING_V1,
        market_clearing_residual_tolerance=2.0e-5,
    )
    kwargs.update(overrides)
    return SimulateTS(**kwargs)


def test_market_cleared_simulation_clears_parent_and_both_children():
    sim = _sim()
    output = sim.simulate_tensor()
    columns = {name: i for i, name in enumerate(output.macro.columns)}
    macro = output.macro.data

    assert macro.shape[0] == 2 * 3
    assert set(macro[:, columns["branch"]].to(torch.long).tolist()) == {-1, 0, 1}
    assert torch.all(macro[:, columns["market_root_found"]] == 1.0)
    assert torch.all(macro[:, columns["market_root_count"]] >= 1.0)
    assert torch.max(
        macro[:, columns["market_committed_residual_normalized"]].abs()
    ) <= 2.0e-5
    assert torch.allclose(
        macro[:, columns["Hatc_cleared"]],
        macro[:, columns["Hatc_firm_input"]],
        atol=0.0,
        rtol=0.0,
    )
    assert output.meta["market_clearing"]["VECTORIZED_EVALUATOR"] is True
    assert output.meta["market_clearing"]["node_count"] == 6
    for name in (
        "market_root_branch_id",
        "market_root_branch_switch",
        "market_investment_flip_vs_belief",
        "market_default_flip_vs_belief",
        "market_entry_flip_vs_belief",
    ):
        assert torch.isfinite(macro[:, columns[name]]).all()


def test_market_clearing_reuses_exact_entry_candidates_for_commit():
    torch.manual_seed(101)
    first = _sim().simulate_tensor()
    torch.manual_seed(101)
    second = _sim().simulate_tensor()
    names = (
        "accepted_count",
        "I_entry",
        "C_raw",
        "K",
        "Hatc_cleared",
        "market_root_residual_normalized",
    )
    for name in names:
        index = first.macro.columns.index(name)
        assert torch.equal(first.macro.data[:, index], second.macro.data[:, index])


def test_market_clearing_strictly_rejects_support_without_root():
    sim = _sim(
        n_paths=1,
        group_size=1,
        enable_entry=False,
        market_clearing_support_lower=-8.0,
        market_clearing_support_upper=-1.0,
        market_clearing_support_padding=0.25,
    )
    with pytest.raises(MarketClearingError, match="no valid root"):
        sim.simulate_tensor()


def test_disabled_mode_retains_nan_market_diagnostics():
    sim = _sim(market_clearing_mode="disabled")
    output = sim.simulate_tensor()
    columns = {name: i for i, name in enumerate(output.macro.columns)}
    assert torch.all(output.macro.data[:, columns["market_root_found"]] == 0.0)
    assert torch.isnan(
        output.macro.data[:, columns["market_root_residual_normalized"]]
    ).all()
    assert output.meta["market_clearing"] == {"mode": "disabled"}


def test_firm_bank_uses_two_matched_root_cleared_children():
    torch.manual_seed(303)
    output = _sim().simulate_tensor()
    parents, children, audit = build_market_cleared_transition_bank(
        output,
        n_parents=4,
        seed=19,
        split="train",
        root_tolerance=2.0e-5,
    )
    assert parents.parent_states.shape == (4, 7)
    assert children.children.shape == (4, 2, 8)
    assert children.metadata["definition"] == (
        "two_independently_root_cleared_production_children"
    )
    assert children.metadata["artificial_b_only_used"] is False
    assert audit["child_branches"] == [0, 1]


def test_root_candidate_admission_is_stricter_than_public_commit_gate():
    controller = MarketClearingController(
        MarketClearingConfig(residual_tolerance=1.0e-5)
    )
    assert controller._root_acceptance_tolerance == pytest.approx(5.0e-6)
    assert (
        controller.summary()["root_candidate_acceptance_tolerance"]
        == pytest.approx(5.0e-6)
    )
    assert controller.config.residual_tolerance == pytest.approx(1.0e-5)


def test_committed_parity_checks_cleared_node_even_after_all_firms_exit():
    controller = MarketClearingController(
        MarketClearingConfig(residual_tolerance=1.0e-5)
    )
    state = {
        "accounting_K_total": torch.tensor([1.0]),
        "accounting_C_raw": torch.tensor([2.0]),
        "hatc_cleared": torch.tensor([0.0]),
        "alive": torch.zeros((1, 2), dtype=torch.bool),
        "market_root_found": torch.tensor([True]),
    }
    with pytest.raises(MarketClearingError, match="commit parity failed"):
        controller.validate_committed(state, t=0, branch=-1)


def test_market_clearing_skips_extinct_padding_path_and_clears_survivor():
    seed_sim = _sim(market_clearing_mode="disabled")
    initial_state = _initialize_batched_state(seed_sim, max_firms=3)
    initial_state["alive"][0] = False
    initial_state["initial_cohort"][0] = False

    sim = _sim(initial_state=initial_state)
    output = sim.simulate_tensor()
    columns = {name: i for i, name in enumerate(output.macro.columns)}
    macro = output.macro.data

    assert macro.shape[0] == 3
    assert set(macro[:, columns["path"]].to(torch.long).tolist()) == {1}
    assert set(macro[:, columns["branch"]].to(torch.long).tolist()) == {-1, 0, 1}
    assert torch.all(macro[:, columns["market_root_found"]] == 1.0)
    assert torch.max(
        macro[:, columns["market_committed_residual_normalized"]].abs()
    ) <= 2.0e-5
    assert output.meta["market_clearing"]["node_count"] == 3
    assert output.meta["market_clearing"]["failure_count"] == 0


def test_active_path_merge_does_not_replace_production_state_tensors():
    state = {
        "b": torch.zeros((2, 1)),
        "hatcf": torch.tensor([-2.0, -1.0]),
        "entry": torch.zeros((2, 1)),
        "alive": torch.tensor([[False], [True]]),
    }
    entry_tensor = state["entry"]
    active_index = torch.tensor([1], dtype=torch.long)
    cleared = {
        "b": torch.ones((1, 1)),
        "entry": torch.ones((1, 1)),
        "hatcf": torch.tensor([-1.4]),
        "hatc_belief": torch.tensor([-1.0]),
        "hatc_cleared": torch.tensor([-1.4]),
        "market_previous_root": torch.tensor([-1.4]),
        "market_previous_branch_id": torch.tensor([0]),
        "market_root_branch_id": torch.tensor([0]),
        "market_root_branch_switch": torch.tensor([False]),
        "market_root_found": torch.tensor([True]),
        "market_root_count": torch.tensor([1]),
        "market_root_residual": torch.tensor([0.0]),
        "market_root_residual_normalized": torch.tensor([0.0]),
        "market_root_C": torch.tensor([1.0]),
        "market_root_K": torch.tensor([1.0]),
        "market_root_refinement_rounds": torch.tensor([1]),
        "market_root_policy_jump": torch.tensor([False]),
        "market_investment_flip_vs_belief": torch.tensor([0.0]),
        "market_default_flip_vs_belief": torch.tensor([0.0]),
        "market_entry_flip_vs_belief": torch.tensor([0.0]),
        "market_root_bracket_count": torch.tensor([1]),
        "market_root_operator_evaluations": torch.tensor([31]),
    }

    merged = MarketClearingController._merge_active_paths(
        state, cleared, active_index
    )

    assert merged["entry"] is entry_tensor
    assert torch.equal(merged["entry"], torch.zeros((2, 1)))
    assert torch.equal(merged["b"], torch.zeros((2, 1)))
    assert torch.equal(merged["market_root_found"], torch.tensor([False, True]))
