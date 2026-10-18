from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from data.consistent_aggregation import (
    NodeDecisionRecord,
    aggregate_mass_ledger,
    assert_candidate_commit_parity,
    continuum_entry_reference,
    equal_probability_normal_quadrature,
    integrate_hard_investment_region,
    uniform_midpoint_quadrature,
)
from data.market_clearing import MARKET_CLEARING_V1
from data.simulate_ts import SimulateTS


class _ContinuumConfig:
    DEVICE = torch.device("cpu")
    RHO_X = 0.0
    SIGMA_X = 0.0
    XBAR = 0.0
    RHO_Z = 0.5
    SIGMA_Z = 0.1
    ZBAR = 0.0
    DELTA = 0.0
    PHI = 1.0
    G = 1.1
    I_THRESHOLD = 0.5
    ZETA = 0.2
    SIM_B_INIT_MIN = 0.0
    SIM_B_INIT_MAX = 1.0
    ENTRY_B_MIN = 0.0
    ENTRY_B_MAX = 1.0


class _HardPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward_value_components(self, state):
        p0 = torch.ones_like(state[:, :1]) + 0.0 * self.anchor
        pi = p0 + (0.25 - state[:, 3:4])
        return {"V0_physical": p0, "VI_physical": pi}

    def forward_simulation(self, state):
        p0 = torch.ones_like(state[:, :1]) + 0.0 * self.anchor
        pi = p0 + (0.25 - state[:, 3:4])
        zero = torch.zeros_like(p0)
        one = torch.ones_like(p0)
        hard_i = (pi >= p0).to(p0.dtype)
        return SimpleNamespace(
            Q=zero,
            P0=p0,
            PI=pi,
            V0=p0,
            VI=pi,
            P=one,
            Phat=one,
            bar_i=hard_i,
            bar_i_cond=hard_i,
            bar_i_eff=hard_i,
            bar_z=zero,
            survival_prob=one,
            bp0=zero,
            bpI=zero,
            bp=zero,
            bp_cond=zero,
        )


def test_uniform_investment_integral_matches_analytic_threshold():
    i, weights = uniform_midpoint_quadrature(
        10_000, 0.0, 0.5, device=torch.device("cpu"), dtype=torch.float64
    )
    threshold = 0.2
    gap = threshold - i
    result = integrate_hard_investment_region(
        gap[None, :], i[None, :], weights, torch.tensor([3.0])
    )
    assert result["investment_probability"].item() == pytest.approx(0.4, abs=1e-4)
    # k * integral_0^i* i di / i_max
    expected_cost = 3.0 * threshold ** 2 / (2.0 * 0.5)
    assert result["expected_project_cost"].item() == pytest.approx(
        expected_cost, abs=1e-5
    )


@pytest.mark.parametrize("gap, expected", [(-1.0, 0.0), (1.0, 1.0)])
def test_investment_region_boundaries(gap, expected):
    i, weights = uniform_midpoint_quadrature(
        17, 0.0, 0.5, device=torch.device("cpu"), dtype=torch.float64
    )
    result = integrate_hard_investment_region(
        torch.full_like(i[None, :], gap),
        i[None, :],
        weights,
        torch.tensor([1.0]),
    )
    assert result["investment_probability"].item() == pytest.approx(expected)


def test_entry_reference_uses_mass_not_slot_count():
    k = torch.tensor([[2.0, 10.0]], dtype=torch.float64)
    mass = torch.tensor([[4.0, 0.5]], dtype=torch.float64)
    alive = torch.tensor([[True, True]])
    result = continuum_entry_reference(
        k,
        mass,
        alive,
        entry_capital_ratio=0.05,
        entry_size_ratio=1.0,
    )
    assert result["reference_mass"].item() == pytest.approx(4.5)
    assert result["reference_capital"].item() == pytest.approx(13.0)
    assert result["mean_capital_per_unit"].item() == pytest.approx(13.0 / 4.5)
    assert result["potential_entry_mass"].item() == pytest.approx(0.225)


def test_mass_ledger_uses_same_hard_action_for_cost_and_default_transfer():
    mass = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    capital = torch.tensor([[2.0, 2.0]], dtype=torch.float64)
    ledger = aggregate_mass_ledger(
        mass=mass,
        capital_per_unit=capital,
        b=torch.tensor([[0.2, 0.4]], dtype=torch.float64),
        z=torch.zeros((1, 2), dtype=torch.float64),
        x=torch.zeros(1, dtype=torch.float64),
        project_investment=torch.tensor([[0.4, 0.0]], dtype=torch.float64),
        maintenance=torch.tensor([[0.04, 0.04]], dtype=torch.float64),
        default=torch.tensor([[False, True]]),
        phi=0.4,
    )
    assert ledger["firm_mass"].item() == pytest.approx(1.0)
    assert ledger["K_total"].item() == pytest.approx(2.0)
    assert ledger["default_mass"].item() == pytest.approx(0.75)
    # Debt is mass * b * k, not b summed over numerical slots.
    assert ledger["aggregate_debt"].item() == pytest.approx(0.7)


def test_normal_quadrature_preserves_probability_and_symmetry():
    nodes, weights = equal_probability_normal_quadrature(
        101, device=torch.device("cpu"), dtype=torch.float64
    )
    assert weights.sum().item() == pytest.approx(1.0)
    assert (nodes * weights).sum().item() == pytest.approx(0.0, abs=1e-12)


def test_candidate_commit_parity_is_strict():
    one = torch.tensor([1.0], dtype=torch.float64)
    record = NodeDecisionRecord(
        snapshot_hash="abc",
        candidate_h=torch.tensor([0.0]),
        bar_i=torch.tensor([[True]]),
        bar_z=torch.tensor([[False]]),
        bp=torch.tensor([[0.2]]),
        accepted_entry=torch.empty((1, 0), dtype=torch.bool),
        entry_bar_i=torch.empty((1, 0), dtype=torch.bool),
        entry_bar_z=torch.empty((1, 0), dtype=torch.bool),
        firm_mass=one[:, None],
        entry_mass=torch.empty((1, 0)),
        ledger={"C_raw": one, "K_total": one},
        provenance={},
    )
    assert_candidate_commit_parity(record, {"C_raw": one, "K_total": one})
    with pytest.raises(AssertionError):
        assert_candidate_commit_parity(
            record, {"C_raw": one + 1e-3, "K_total": one}
        )


def test_continuum_two_step_simulation_tracks_mass_separately_from_slots():
    sim = SimulateTS(
        models={"policy_value": _HardPolicy(), "sdf_fc1": None},
        config=_ContinuumConfig,
        n_paths=1,
        group_size=2,
        horizon=2,
        branch_num=2,
        enable_entry=False,
        enable_exit=True,
        device=torch.device("cpu"),
        entry_mode="investment_compare",
        entry_spec_version="investment_compare_v1",
        consumption_aggregation_mode="raw",
        node_accounting_mode="economic_node_ledger",
        entry_rng_mode="stable_economic_identity",
        common_transition_seed=11,
        transition_rng_mode="stable_firm_identity",
        aggregation_mode="continuum_mass_consistent_v1",
        continuum_i_points=3,
        continuum_z_points=3,
        market_clearing_mode=MARKET_CLEARING_V1,
        market_clearing_support_lower=-3.0,
        market_clearing_support_upper=1.0,
        market_clearing_residual_tolerance=2.0e-5,
    )
    output = sim.simulate_tensor()
    macro = output.macro.to_dataframe()
    assert len(macro) == 6
    assert (macro["slot_count"] >= macro["firm_mass"]).all()
    assert (macro["firm_mass"] > 0.0).all()
    assert macro["distribution_projection_error"].max() < 1.0e-5
    assert output.meta["aggregation"]["AGGREGATION_SEMANTICS_CHANGED"] is True
    assert (
        output.meta["market_clearing"]["candidate_commit_parity_pass_count"]
        == output.meta["market_clearing"]["node_count"]
    )


def test_continuum_entry_uses_fractional_mass_consistently():
    sim = SimulateTS(
        models={"policy_value": _HardPolicy(), "sdf_fc1": None},
        config=_ContinuumConfig,
        n_paths=1,
        group_size=2,
        horizon=1,
        branch_num=2,
        enable_entry=True,
        enable_exit=True,
        device=torch.device("cpu"),
        entry_mode="investment_compare",
        entry_spec_version="investment_compare_v1",
        entry_capital_ratio=0.05,
        entry_size_ratio=1.0,
        entry_cost_max=_ContinuumConfig.I_THRESHOLD,
        entry_rng_mode="stable_economic_identity",
        common_transition_seed=13,
        transition_rng_mode="stable_firm_identity",
        consumption_aggregation_mode="raw",
        node_accounting_mode="economic_node_ledger",
        aggregation_mode="continuum_mass_consistent_v1",
        continuum_i_points=4,
        continuum_z_points=3,
        market_clearing_mode=MARKET_CLEARING_V1,
        market_clearing_support_lower=-3.0,
        market_clearing_support_upper=1.0,
        market_clearing_residual_tolerance=2.0e-5,
    )
    output = sim.simulate_tensor()
    macro = output.macro.to_dataframe()
    children = macro.loc[macro["branch"].isin([0.0, 1.0])]
    assert (children["potential_count"] > 0.0).all()
    assert (children["accepted_count"] > 0.0).all()
    assert (children["accepted_count"] < 1.0).all()
    assert (children["I_entry"] > 0.0).all()
    assert (children["market_committed_residual_normalized"].abs() <= 2e-5).all()

