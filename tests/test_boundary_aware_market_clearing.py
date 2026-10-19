from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from data.boundary_aware_market_clearing import (
    BoundaryAwareMarketClearingController,
    BoundaryIntegrationConfig,
)
from data.market_clearing import MarketClearingConfig, empty_entry_candidates
from data.entry import EntryCandidates
from data.boundary_aware_distribution import propagate_ar1_to_predecision


class _BoundaryPolicy(torch.nn.Module):
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
        phat = state[:, 1:2]
        zero = torch.zeros_like(p0)
        invest = (pi >= p0).to(p0.dtype)
        default = (phat <= 0.0).to(p0.dtype)
        return SimpleNamespace(
            Q=zero,
            P0=p0,
            PI=pi,
            V0=p0,
            VI=pi,
            P=torch.maximum(p0, pi),
            Phat=phat,
            bar_i=invest,
            bar_i_cond=invest,
            bar_i_eff=invest,
            bar_z=default,
            survival_prob=1.0 - default,
            bp0=zero,
            bpI=zero,
            bp=zero,
            bp_cond=zero,
        )


class _Config:
    I_THRESHOLD = 0.5
    PHI = 0.4
    RHO_Z = 0.8
    SIGMA_Z = 0.1
    ZBAR = 0.0


def _state():
    return {
        "component_mass": torch.tensor([[100.0]]),
        "firm_mass": torch.tensor([[100.0]]),
        "b": torch.tensor([[0.4]]),
        "K": torch.tensor([[1.0]]),
        "eta": torch.tensor([[0.0]]),
        "component_mask": torch.tensor([[True]]),
        "alive": torch.tensor([[True]]),
        "z": torch.tensor([[0.0]]),
        "z_conditional_mean": torch.tensor([[0.0]]),
        "z_conditional_std": torch.tensor([[0.1]]),
        "maintenance_eligibility": torch.tensor([[False]]),
        "x": torch.tensor([0.0]),
        "hatcf": torch.tensor([0.0]),
        "lnkf": torch.tensor([0.0]),
    }


def test_candidate_evaluation_integrates_default_and_investment_boundaries():
    controller = BoundaryAwareMarketClearingController(
        MarketClearingConfig(candidate_chunk_size=100_000),
        BoundaryIntegrationConfig(
            z_scan_points=33,
            z_quadrature_points=12,
            i_scan_points=17,
            i_quadrature_points=12,
            boundary_refinement_rounds=18,
        ),
    )
    sim = SimpleNamespace(
        models={"policy_value": _BoundaryPolicy()},
        config=_Config,
        delta=0.0,
    )
    state = _state()
    result = controller.evaluate(
        sim,
        state,
        empty_entry_candidates(state),
        torch.tensor([[0.0]]),
        detail=True,
    )
    assert result["default_mass"].item() == pytest.approx(50.0, abs=2e-4)
    assert result["investment_mass"].item() == pytest.approx(25.0, abs=2e-4)
    assert result["firm_count"].item() == pytest.approx(100.0, abs=2e-4)
    assert result["K_total"].item() == pytest.approx(100.0, abs=2e-4)
    assert result["bar_z_firm"].item() == pytest.approx(0.5, abs=2e-6)
    assert result["bar_i_firm"].item() == pytest.approx(0.25, abs=2e-6)
    assert result["transport_atom_weight"].sum().item() == pytest.approx(50.0, abs=2e-4)
    assert controller.default_boundary_count == 1
    assert controller.investment_boundary_count > 0


def test_entry_is_integrated_over_continuous_z_and_cost_boundaries():
    controller = BoundaryAwareMarketClearingController(
        MarketClearingConfig(candidate_chunk_size=100_000),
        BoundaryIntegrationConfig(
            z_scan_points=33,
            z_quadrature_points=12,
            i_scan_points=17,
            i_quadrature_points=12,
            boundary_refinement_rounds=18,
        ),
    )
    sim = SimpleNamespace(
        models={"policy_value": _BoundaryPolicy()},
        config=_Config,
        delta=0.0,
        effective_entry_cost_max=0.5,
    )
    state = _state()
    candidates = EntryCandidates(
        counts=torch.tensor([1]),
        mask=torch.tensor([[True]]),
        # These sampled coordinates are intentionally extreme.  The boundary-
        # aware operator must ignore them and integrate the declared densities.
        z=torch.tensor([[9.0]]),
        entry_cost=torch.tensor([[0.49]]),
        K_birth=torch.tensor([[1.0]]),
        potential_capital_nominal=torch.tensor([10.0]),
        candidate_capital_realized=torch.tensor([1.0]),
    )
    result = controller.evaluate(
        sim, state, candidates, torch.tensor([[0.0]]), detail=True
    )
    assert result["entry_mass"].item() == pytest.approx(5.0, abs=3e-4)
    assert result["I_entry_total"].item() == pytest.approx(0.625, abs=3e-4)
    assert result["K_total"].item() == pytest.approx(105.0, abs=4e-4)
    assert result["default_mass"].item() == pytest.approx(52.5, abs=4e-4)
    assert result["entry_transport_atom_weight"].sum().item() == pytest.approx(
        2.5, abs=4e-4
    )


def test_candidate_evaluation_has_no_distribution_side_effect():
    controller = BoundaryAwareMarketClearingController(
        MarketClearingConfig(candidate_chunk_size=100_000),
        BoundaryIntegrationConfig(z_scan_points=17, z_quadrature_points=4, i_scan_points=9, i_quadrature_points=4),
    )
    sim = SimpleNamespace(models={"policy_value": _BoundaryPolicy()}, config=_Config, delta=0.0)
    state = _state()
    before = {name: value.clone() for name, value in state.items() if torch.is_tensor(value)}
    candidates = empty_entry_candidates(state)
    first = controller.evaluate(sim, state, candidates, torch.tensor([[-0.2, 0.2]]))
    second = controller.evaluate(sim, state, candidates, torch.tensor([[-0.2, 0.2]]))
    for name, value in before.items():
        torch.testing.assert_close(state[name], value)
    for name in ("G", "C_raw", "K_total", "default_mass", "investment_mass"):
        torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)


def test_selected_candidate_commits_same_ledger_and_transport():
    controller = BoundaryAwareMarketClearingController(
        MarketClearingConfig(candidate_chunk_size=100_000),
        BoundaryIntegrationConfig(
            z_scan_points=33,
            z_quadrature_points=12,
            i_scan_points=17,
            i_quadrature_points=12,
            boundary_refinement_rounds=18,
        ),
    )
    sim = SimpleNamespace(
        models={"policy_value": _BoundaryPolicy()},
        config=_Config,
        delta=0.0,
        g=1.14,
    )
    state = _state()
    candidates = empty_entry_candidates(state)
    selected_h = torch.tensor([[0.0]])
    detail = controller.evaluate(sim, state, candidates, selected_h, detail=True)
    controller._commit_selected_decision_record(
        sim,
        state,
        candidates,
        selected_h[:, 0],
        detail,
        t=1,
        branch=0,
    )
    record = state["_boundary_decision_record"]
    record.assert_ledger_parity(record.ledger)
    assert record.transport_atoms.total_mass().item() == pytest.approx(50.0, abs=2e-4)
    assert record.transport_atoms.total_capital().item() == pytest.approx(53.5, abs=4e-4)
    t2 = propagate_ar1_to_predecision(
        record.transport_atoms, rho_z=0.8, sigma_z=0.1, z_bar=0.0
    )
    assert t2.total_mass().item() == pytest.approx(50.0, abs=2e-4)
    assert t2.total_capital().item() == pytest.approx(53.5, abs=4e-4)

