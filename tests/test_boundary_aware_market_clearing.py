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
from data.boundary_aware_distribution import (
    BoundaryAwareDecisionRecord,
    PostDecisionTransportAtoms,
)
from data.market_clearing import (
    ECONOMIC_NODE_ACTIVE,
    ECONOMIC_NODE_EXTINCT_BEFORE_NODE,
)


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


class _BoundaryPolicyWithEquity(_BoundaryPolicy):
    def __init__(self):
        super().__init__()
        self.equity_calls = 0
        self.simulation_calls = 0

    def forward_equity(self, state):
        self.equity_calls += 1
        return {"Phat": state[:, 1:2]}

    def forward_simulation(self, state):
        self.simulation_calls += 1
        return super().forward_simulation(state)


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


def test_candidate_h_microbatching_preserves_boundary_operator_values():
    sim = SimpleNamespace(
        models={"policy_value": _BoundaryPolicy()},
        config=_Config,
        delta=0.0,
    )
    state = _state()
    h = torch.linspace(-0.3, 0.3, 8).reshape(1, 8)
    results = []
    for chunk in (1, 4, 8):
        controller = BoundaryAwareMarketClearingController(
            MarketClearingConfig(candidate_chunk_size=100_000),
            BoundaryIntegrationConfig(
                z_scan_points=17,
                z_quadrature_points=6,
                i_scan_points=9,
                i_quadrature_points=6,
                boundary_refinement_rounds=8,
                candidate_h_microbatch_size=chunk,
            ),
        )
        results.append(
            controller.evaluate(
                sim, state, empty_entry_candidates(state), h, detail=False
            )
        )
    for name in (
        "G",
        "G_normalized",
        "C_raw",
        "K_total",
        "investment_mass",
        "default_mass",
        "entry_mass",
    ):
        torch.testing.assert_close(results[0][name], results[1][name])
        torch.testing.assert_close(results[0][name], results[2][name])


def test_default_boundary_scan_uses_chunked_equity_only_phat_path():
    model = _BoundaryPolicyWithEquity()
    controller = BoundaryAwareMarketClearingController(
        MarketClearingConfig(candidate_chunk_size=3),
        BoundaryIntegrationConfig(
            z_scan_points=17,
            z_quadrature_points=6,
            i_scan_points=9,
            i_quadrature_points=6,
            boundary_refinement_rounds=8,
            candidate_h_microbatch_size=1,
        ),
    )
    states = torch.tensor(
        [
            [0.4, -0.2, 0.0, 0.0, 0.0, -0.1, 0.0],
            [0.4, -0.1, 0.0, 0.0, 0.0, -0.1, 0.0],
            [0.4, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0],
            [0.4, 0.1, 0.0, 0.0, 0.0, -0.1, 0.0],
            [0.4, 0.2, 0.0, 0.0, 0.0, -0.1, 0.0],
        ]
    )
    phat = controller._firm_phat(model, states)
    torch.testing.assert_close(phat, states[:, 1])
    assert model.equity_calls == 2
    assert model.simulation_calls == 0


def test_chunked_policy_and_entry_outputs_match_single_chunk():
    states = torch.tensor(
        [
            [0.4, -0.2, 0.0, 0.05, 0.0, -0.1, 0.0],
            [0.4, -0.1, 0.0, 0.10, 0.0, -0.1, 0.0],
            [0.4, 0.0, 0.0, 0.15, 0.0, -0.1, 0.0],
            [0.4, 0.1, 0.0, 0.20, 0.0, -0.1, 0.0],
            [0.4, 0.2, 0.0, 0.25, 0.0, -0.1, 0.0],
        ]
    )
    model = _BoundaryPolicy()
    single = BoundaryAwareMarketClearingController(
        MarketClearingConfig(candidate_chunk_size=states.shape[0]),
        BoundaryIntegrationConfig(),
    )
    chunked = BoundaryAwareMarketClearingController(
        MarketClearingConfig(candidate_chunk_size=2),
        BoundaryIntegrationConfig(),
    )
    full_policy = single._firm_policy(model, states)
    chunked_policy = chunked._firm_policy(model, states)
    for name in full_policy:
        torch.testing.assert_close(chunked_policy[name], full_policy[name])
    full_entry = single._entry_values(model, states)
    chunked_entry = chunked._entry_values(model, states)
    torch.testing.assert_close(chunked_entry[0], full_entry[0])
    torch.testing.assert_close(chunked_entry[1], full_entry[1])


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


def test_entry_only_node_does_not_require_a_fake_incumbent():
    controller = BoundaryAwareMarketClearingController(
        MarketClearingConfig(candidate_chunk_size=100_000),
        BoundaryIntegrationConfig(
            z_scan_points=33,
            z_quadrature_points=8,
            i_scan_points=17,
            i_quadrature_points=8,
        ),
    )
    sim = SimpleNamespace(
        models={"policy_value": _BoundaryPolicy()},
        config=_Config,
        delta=0.0,
        effective_entry_cost_max=0.5,
    )
    state = _state()
    state["component_mass"].zero_()
    state["firm_mass"].zero_()
    state["component_mask"].zero_()
    state["alive"].zero_()
    candidates = EntryCandidates(
        counts=torch.tensor([1]),
        mask=torch.tensor([[True]]),
        z=torch.zeros((1, 1)),
        entry_cost=torch.zeros((1, 1)),
        K_birth=torch.ones((1, 1)),
        potential_capital_nominal=torch.tensor([10.0]),
        candidate_capital_realized=torch.tensor([1.0]),
    )
    result = controller.evaluate(
        sim, state, candidates, torch.tensor([[0.0]]), detail=True
    )
    assert result["entry_mass"].item() == pytest.approx(5.0, abs=5e-4)
    assert result["K_total"].item() == pytest.approx(5.0, abs=5e-4)
    assert result["firm_count"].item() == pytest.approx(5.0, abs=5e-4)
    assert result["entry_transport_atom_weight"].sum().item() == pytest.approx(
        2.5, abs=5e-4
    )
    assert result["transport_atom_weight"].numel() == 0


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


def _synthetic_transport(batch_size: int) -> PostDecisionTransportAtoms:
    shape = (batch_size, 1)
    weight = torch.arange(1, batch_size + 1, dtype=torch.float64)[:, None]
    zeros = torch.zeros(shape)
    ids = torch.arange(batch_size, dtype=torch.long)[:, None]
    return PostDecisionTransportAtoms(
        transport_weight=weight,
        source_z_current=zeros.clone(),
        b_next=zeros.clone(),
        capital_next_per_unit=torch.ones(shape),
        atom_mask=torch.ones(shape, dtype=torch.bool),
        source_component_id=ids.clone(),
        survival_region_id=torch.zeros(shape, dtype=torch.long),
        investment_region_id=torch.zeros(shape, dtype=torch.long),
        policy_branch_id=torch.zeros(shape, dtype=torch.long),
        economic_node_id=ids.clone(),
        source_time=torch.ones(shape, dtype=torch.long),
        destination_time=torch.full(shape, 2, dtype=torch.long),
        cohort=torch.zeros(shape, dtype=torch.long),
        birth_time=torch.full(shape, -1, dtype=torch.long),
        maintenance_eligibility=torch.ones(shape, dtype=torch.bool),
        origin_kind=torch.zeros(shape, dtype=torch.long),
        origin_source_id=ids.clone(),
    ).validate()


def test_partial_batch_merge_scatter_all_commit_outputs_and_transport_in_order():
    controller = BoundaryAwareMarketClearingController(
        MarketClearingConfig(), BoundaryIntegrationConfig()
    )
    batch_size = 8
    active_size = 6
    active_index = torch.arange(active_size, dtype=torch.long)
    state = {
        "b": torch.zeros((batch_size, 1)),
        "alive": torch.tensor([[True]] * active_size + [[False], [False]]),
        "hatcf": torch.arange(batch_size, dtype=torch.float32),
        "component_mass": torch.tensor([[1.0]] * active_size + [[0.0], [0.0]]),
        "component_mask": torch.tensor([[True]] * active_size + [[False], [False]]),
        "K": torch.ones((batch_size, 1)),
        "z_conditional_mean": torch.zeros((batch_size, 1)),
        "z_conditional_std": torch.ones((batch_size, 1)),
        "economic_node_id_component": torch.arange(batch_size)[:, None],
        "component_id": torch.arange(batch_size)[:, None],
        "origin_kind": torch.zeros((batch_size, 1), dtype=torch.long),
        "origin_source_id": torch.arange(batch_size)[:, None],
        "origin_time": torch.zeros((batch_size, 1), dtype=torch.long),
        "evaluation_time": torch.zeros((batch_size, 1), dtype=torch.long),
        "cohort": torch.zeros((batch_size, 1), dtype=torch.long),
        "birth_time": torch.full((batch_size, 1), -1, dtype=torch.long),
        "maintenance_eligibility": torch.ones((batch_size, 1), dtype=torch.bool),
        "eta": torch.zeros((batch_size, 1)),
    }
    h = torch.linspace(-1.0, -0.5, active_size)
    accounting = torch.arange(1, active_size + 1, dtype=torch.float32)
    ledger = {
        "C_raw": accounting.to(torch.float64),
        "C_oper": accounting.to(torch.float64),
        "K_total": accounting.to(torch.float64),
        "Y_total": accounting.to(torch.float64),
        "I_oper_total": accounting.to(torch.float64),
        "I_entry_total": torch.zeros(active_size, dtype=torch.float64),
        "Liquidation_total": torch.zeros(active_size, dtype=torch.float64),
        "I_maintenance_total": torch.zeros(active_size, dtype=torch.float64),
    }
    record = BoundaryAwareDecisionRecord(
        predecision_fingerprint="six-path-reference",
        candidate_h=h,
        ledger=ledger,
        transport_atoms=_synthetic_transport(active_size),
        provenance={"test": True},
    )
    cleared = {
        "hatcf": h.clone(),
        "hatc_belief": h + 0.1,
        "hatc_cleared": h.clone(),
        "hatc_cal": h.clone(),
        "lnk_cal": torch.log(accounting),
        "market_previous_root": h.clone(),
        "market_previous_branch_id": torch.zeros(active_size, dtype=torch.long),
        "market_root_branch_id": torch.zeros(active_size, dtype=torch.long),
        "market_root_branch_switch": torch.zeros(active_size, dtype=torch.bool),
        "market_root_found": torch.ones(active_size, dtype=torch.bool),
        "market_root_count": torch.ones(active_size, dtype=torch.long),
        "market_root_residual": torch.zeros(active_size),
        "market_root_residual_normalized": torch.zeros(active_size),
        "market_root_C": accounting.clone(),
        "market_root_K": accounting.clone(),
        "market_root_refinement_rounds": torch.ones(active_size, dtype=torch.long),
        "market_root_policy_jump": torch.zeros(active_size, dtype=torch.bool),
        "market_investment_flip_vs_belief": torch.zeros(active_size),
        "market_default_flip_vs_belief": torch.zeros(active_size),
        "market_entry_flip_vs_belief": torch.zeros(active_size),
        "market_root_bracket_count": torch.ones(active_size, dtype=torch.long),
        "market_root_operator_evaluations": torch.full(
            (active_size,), 30, dtype=torch.long
        ),
        "accounting_C_raw": accounting.clone(),
        "accounting_C_oper": accounting.clone(),
        "accounting_K_total": accounting.clone(),
        "accounting_Y_total": accounting.clone(),
        "accounting_I_oper": accounting.clone(),
        "accounting_I_entry": torch.zeros(active_size),
        "accounting_Liquidation_total": torch.zeros(active_size),
        "accounting_I_maintenance": torch.zeros(active_size),
        "accounting_valid": torch.ones(active_size, dtype=torch.bool),
        "economic_node_status": torch.full(
            (active_size,), ECONOMIC_NODE_ACTIVE, dtype=torch.long
        ),
        "economic_state_defined": torch.ones(active_size, dtype=torch.bool),
        "_boundary_decision_record": record,
    }

    merged = controller._merge_active_paths(state, cleared, active_index)

    torch.testing.assert_close(merged["accounting_C_raw"][:active_size], accounting)
    assert torch.equal(merged["accounting_C_raw"][active_size:], torch.zeros(2))
    torch.testing.assert_close(merged["hatc_cleared"][:active_size], h)
    assert torch.isnan(merged["hatc_cleared"][active_size:]).all()
    assert torch.equal(
        merged["economic_node_status"],
        torch.tensor(
            [ECONOMIC_NODE_ACTIVE] * active_size
            + [ECONOMIC_NODE_EXTINCT_BEFORE_NODE] * 2
        ),
    )
    atoms = merged["_selected_transport_atoms"]
    torch.testing.assert_close(
        atoms.transport_weight[:active_size, 0],
        torch.arange(1, active_size + 1, dtype=torch.float64),
    )
    assert not bool(atoms.atom_mask[active_size:].any())
    merged["_boundary_decision_record"].assert_ledger_parity(
        merged["_boundary_decision_record"].ledger
    )


def test_fully_extinct_batch_returns_explicit_terminal_without_fake_macro_state():
    controller = BoundaryAwareMarketClearingController(
        MarketClearingConfig(), BoundaryIntegrationConfig()
    )
    state = _state()
    state["component_mass"].zero_()
    state["firm_mass"].zero_()
    state["component_mask"].zero_()
    state["alive"].zero_()
    cleared = controller.clear_state(
        SimpleNamespace(), state, empty_entry_candidates(state), t=4, branch=0
    )
    controller.validate_committed(cleared, t=4, branch=0)
    assert int(cleared["economic_node_status"][0]) == ECONOMIC_NODE_EXTINCT_BEFORE_NODE
    assert not bool(cleared["economic_state_defined"][0])
    assert torch.isnan(cleared["hatc_cleared"][0])
    assert torch.isnan(cleared["lnk_cal"][0])
    assert float(cleared["accounting_C_raw"][0]) == 0.0
    assert not bool(cleared["_selected_transport_atoms"].atom_mask.any())

