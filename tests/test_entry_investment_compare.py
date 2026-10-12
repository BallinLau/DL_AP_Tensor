from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from data.entry import (
    ALL_FIRM_MAINTENANCE_VERSION,
    EntryCandidates,
    INVESTMENT_COMPARE_SPEC_VERSION,
    INVESTMENT_COMPARE_MAINTENANCE_VERSION,
    EntryAccountingError,
    aggregate_node_resources,
    evaluate_investment_compare_entry,
    identify_birth_at_node,
    scan_investment_compare_hatc,
    slice_entry_candidates,
    validate_node_resource_account,
)
from data.simulate_ts import SimulateTS
from data.simulate_ts_parallel import _write_entry_hatc_failure_scan
from experiments.run_multi_episode_job import configure_hyperparams, parse_args


class _BranchValuePolicy(nn.Module):
    def __init__(self, *, i_threshold: float = 0.5, nonfinite: bool = False):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.i_threshold = float(i_threshold)
        self.nonfinite = bool(nonfinite)
        self.seen_states: list[torch.Tensor] = []

    def forward_value_components(self, state: torch.Tensor):
        self.seen_states.append(state.detach().clone())
        v0 = torch.ones_like(state[:, 0:1]) + self.anchor * 0.0
        vi = v0 + state[:, 1:2]
        if self.nonfinite:
            vi = vi.clone()
            vi[0] = float("nan")
        return {
            "V0_physical": v0,
            "VI_physical": vi,
            "V0_normalized": 100.0 * v0,
            "VI_normalized": -100.0 * vi,
        }

    def forward_simulation(self, state: torch.Tensor):
        raise AssertionError("investment entry must not use aggregate P")


class _AlwaysInvestPolicy(_BranchValuePolicy):
    def forward_value_components(self, state: torch.Tensor):
        self.seen_states.append(state.detach().clone())
        v0 = torch.ones_like(state[:, 0:1]) + self.anchor * 0.0
        return {"V0_physical": v0, "VI_physical": v0 + 1.0}

    def forward_simulation(self, state: torch.Tensor):
        n = state.shape[0]
        zeros = state.new_zeros((n, 1)) + self.anchor * 0.0
        ones = torch.ones_like(zeros)
        bp = zeros + 0.4
        return SimpleNamespace(
            Q=zeros,
            P0=ones,
            PI=ones + 1.0,
            bar_i=ones,
            bar_z=zeros,
            P=ones,
            bp0=bp,
            bpI=bp,
            bp=bp,
        )


class _HatcResponsivePolicy(_BranchValuePolicy):
    def forward_value_components(self, state: torch.Tensor):
        v0 = torch.ones_like(state[:, 0:1]) + self.anchor * 0.0
        return {
            "V0_physical": v0,
            "VI_physical": v0 + state[:, 1:2] + state[:, 5:6],
        }

    def forward_simulation(self, state: torch.Tensor):
        n = state.shape[0]
        zeros = state.new_zeros((n, 1)) + self.anchor * 0.0
        ones = torch.ones_like(zeros)
        bar_i = torch.sigmoid(state[:, 5:6])
        return SimpleNamespace(
            Q=zeros,
            P0=ones,
            PI=ones,
            bar_i=bar_i,
            bar_z=zeros,
            P=ones,
            bp0=zeros,
            bpI=zeros,
            bp=zeros,
        )


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
    G = 1.5
    I_THRESHOLD = 0.5
    ZETA = 0.25
    SIM_B_INIT_MIN = 0.0
    SIM_B_INIT_MAX = 1.0
    ENTRY_B_MIN = 0.0
    ENTRY_B_MAX = 1.0


class _PositiveDeltaConfig(_Config):
    DELTA = 0.1


class _PositiveDeltaExitConfig(_PositiveDeltaConfig):
    PHI = 0.5


def _candidates() -> EntryCandidates:
    return EntryCandidates(
        counts=torch.tensor([3]),
        mask=torch.tensor([[True, True, True, False]]),
        z=torch.tensor([[-1.0, 0.0, 1.0, 99.0]]),
        entry_cost=torch.tensor([[0.4, 0.4, 0.4, 0.0]]),
        K_birth=torch.tensor([[2.0, 2.0, 2.0, 0.0]]),
        potential_capital_nominal=torch.tensor([6.0]),
        candidate_capital_realized=torch.tensor([6.0]),
    )


def test_hatc_node_scan_freezes_candidates_and_recomputes_only_node_decisions():
    model = _HatcResponsivePolicy().train()
    simulator = SimulateTS(
        models={"policy_value": model},
        config=_Config,
        n_paths=1,
        group_size=1,
        horizon=1,
        enable_entry=True,
        enable_exit=True,
        entry_mode="investment_compare",
        entry_spec_version=INVESTMENT_COMPARE_SPEC_VERSION,
        entry_cost_max=_Config.I_THRESHOLD,
        consumption_aggregation_mode="raw",
        node_accounting_mode="economic_node_ledger",
    )
    model.train()
    candidates = slice_entry_candidates(_candidates(), 0)
    candidate_before = {
        name: getattr(candidates, name).clone()
        for name in candidates.__dataclass_fields__
    }
    parameter_before = model.anchor.detach().clone()

    kwargs = dict(
        policy_value_model=model,
        candidates=candidates,
        hatc_values=[-1.0, 0.0, 1.0],
        incumbent_b=torch.tensor([0.2, 0.4]),
        incumbent_z=torch.tensor([0.0, 0.1]),
        incumbent_eta=torch.tensor([0.0, 1.0]),
        incumbent_i=torch.tensor([0.2, 0.3]),
        incumbent_K=torch.tensor([2.0, 3.0]),
        incumbent_birth_at_node=torch.tensor([False, False]),
        x=torch.tensor(0.0),
        lnkf=torch.tensor(0.0),
        resource_accounting=simulator._resource_accounting,
        chunk_size=2,
    )
    first = scan_investment_compare_hatc(**kwargs)
    second = scan_investment_compare_hatc(**kwargs)

    assert set(first) >= {
        "hatc_trial",
        "consumption_implied",
        "aggregate_resource_balance",
        "clearing_residual",
        "ordinary_investment_spending",
        "creation_spending",
        "accepted_entrant_count",
        "aggregate_capital",
    }
    torch.testing.assert_close(first["hatc_trial"], torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64))
    assert first["accepted_entrant_count"].tolist() == [1.0, 2.0, 3.0]
    assert first["aggregate_capital"].tolist() == [7.0, 9.0, 11.0]
    for name in first:
        torch.testing.assert_close(first[name], second[name])
    for name, value in candidate_before.items():
        torch.testing.assert_close(getattr(candidates, name), value)
    torch.testing.assert_close(model.anchor, parameter_before)
    assert model.training


def test_hatc_node_scan_uses_resource_identity_for_clearing_residual():
    model = _HatcResponsivePolicy()
    simulator = SimulateTS(
        models={"policy_value": model},
        config=_Config,
        n_paths=1,
        group_size=1,
        horizon=1,
        entry_mode="investment_compare",
        entry_spec_version=INVESTMENT_COMPARE_SPEC_VERSION,
        entry_cost_max=_Config.I_THRESHOLD,
        consumption_aggregation_mode="raw",
        node_accounting_mode="economic_node_ledger",
    )
    result = scan_investment_compare_hatc(
        model,
        slice_entry_candidates(_candidates(), 0),
        hatc_values=[0.0],
        incumbent_b=torch.tensor([0.2]),
        incumbent_z=torch.tensor([0.0]),
        incumbent_eta=torch.tensor([0.0]),
        incumbent_i=torch.tensor([0.2]),
        incumbent_K=torch.tensor([2.0]),
        incumbent_birth_at_node=torch.tensor([False]),
        x=torch.tensor(0.0),
        lnkf=torch.tensor(0.0),
        resource_accounting=simulator._resource_accounting,
    )
    torch.testing.assert_close(
        result["clearing_residual"],
        result["aggregate_resource_balance"] - result["consumption_implied"],
    )
    assert result["accepted_entrant_count"].item() == 2.0
    assert result["creation_spending"].item() == pytest.approx(1.6)


def test_hatc_failure_scan_writes_lightweight_node_artifacts(tmp_path):
    model = _HatcResponsivePolicy()
    simulator = SimulateTS(
        models={"policy_value": model},
        config=_Config,
        n_paths=1,
        group_size=2,
        horizon=1,
        entry_mode="investment_compare",
        entry_spec_version=INVESTMENT_COMPARE_SPEC_VERSION,
        entry_cost_max=_Config.I_THRESHOLD,
        consumption_aggregation_mode="raw",
        node_accounting_mode="economic_node_ledger",
        entry_hatc_failure_scan_enabled=True,
        entry_hatc_failure_scan_output_dir=str(tmp_path),
        entry_hatc_failure_scan_points=5,
        entry_hatc_failure_scan_half_width=1.0,
        entry_hatc_failure_episode_id=2,
    )
    state = {
        "b": torch.tensor([[0.2, 0.4]]),
        "z": torch.tensor([[0.0, 0.1]]),
        "eta": torch.tensor([[0.0, 1.0]]),
        "i": torch.tensor([[0.2, 0.3]]),
        "K": torch.tensor([[2.0, 3.0]]),
        "alive": torch.tensor([[True, True]]),
        "entry": torch.tensor([[0.0, 0.0]]),
        "initial_cohort": torch.tensor([[1.0, 1.0]]),
        "birth_time": torch.tensor([[0.0, 0.0]]),
        "entry_cost": torch.tensor([[0.0, 0.0]]),
        "K_birth": torch.tensor([[2.0, 3.0]]),
        "firm_id": torch.tensor([[0, 1]]),
        "x": torch.tensor([0.0]),
        "hatcf": torch.tensor([0.0]),
        "lnkf": torch.tensor([0.0]),
        "economic_time": torch.tensor([4]),
    }
    validation = {
        "I_entry_reported": torch.tensor([1.0]),
        "I_entry_rebuilt": torch.tensor([1.1]),
        "entry_spend_residual": torch.tensor([-0.1]),
        "entry_spend_tolerance": torch.tensor([1e-5]),
        "C_reported": torch.tensor([2.0]),
        "C_rebuilt": torch.tensor([2.1]),
        "resource_accounting_residual": torch.tensor([-0.1]),
        "resource_accounting_tolerance": torch.tensor([2e-5]),
        "I_maintenance_reported": torch.tensor([0.0]),
        "I_maintenance_rebuilt": torch.tensor([0.0]),
        "maintenance_accounting_residual": torch.tensor([0.0]),
        "maintenance_accounting_tolerance": torch.tensor([1e-6]),
        "I_oper_reported": torch.tensor([0.2]),
        "I_oper_rebuilt": torch.tensor([0.2]),
        "I_oper_accounting_residual": torch.tensor([0.0]),
        "I_oper_accounting_tolerance": torch.tensor([3e-6]),
        "finite_totals": torch.tensor([True]),
    }
    summary = _write_entry_hatc_failure_scan(
        simulator,
        state,
        _candidates(),
        validation,
        path=0,
        t=4,
        branch_k=1,
    )
    assert (tmp_path / "ep2_t4_branch1_path0_hatc_scan.csv").is_file()
    assert (tmp_path / "ep2_t4_branch1_path0_hatc_scan_summary.json").is_file()
    assert (tmp_path / "ep2_t4_branch1_path0_node_snapshot.pt").is_file()
    assert summary["hatc_points"] == 5
    assert summary["fixed_candidate_count"] == 3
    assert summary["original_accounting_failure"]["finite_totals"] is True


def test_investment_compare_uses_physical_pi_ge_p0_at_b0_eta0_i_equals_e():
    model = _BranchValuePolicy().train()
    result = evaluate_investment_compare_entry(
        model,
        _candidates(),
        x=torch.zeros(1),
        hatcf=torch.zeros(1),
        lnkf=torch.zeros(1),
        chunk_size=2,
    )

    assert result["accepted"].tolist() == [[False, True, True, False]]
    torch.testing.assert_close(
        result["entry_value_gap"][0, :3], torch.tensor([-1.0, 0.0, 1.0])
    )
    seen = torch.cat(model.seen_states)
    assert torch.equal(seen[:, 0], torch.zeros(3))
    assert torch.equal(seen[:, 2], torch.zeros(3))
    torch.testing.assert_close(seen[:, 3], torch.full((3,), 0.4))
    assert model.training


def test_investment_compare_does_not_subtract_entry_cost_twice_and_rejects_nonfinite():
    candidates = _candidates()
    candidates.z[0, :3] = torch.tensor([0.1, 0.1, 0.1])
    candidates.entry_cost[0, :3] = 0.4
    result = evaluate_investment_compare_entry(
        _BranchValuePolicy(), candidates,
        x=torch.zeros(1), hatcf=torch.zeros(1), lnkf=torch.zeros(1),
    )
    assert result["accepted"][0, :3].all()

    class _NegativeBranches(_BranchValuePolicy):
        def forward_value_components(self, state: torch.Tensor):
            v0 = torch.full_like(state[:, 0:1], -2.0)
            return {"V0_physical": v0, "VI_physical": v0 + 0.1}

    negative = evaluate_investment_compare_entry(
        _NegativeBranches(), candidates,
        x=torch.zeros(1), hatcf=torch.zeros(1), lnkf=torch.zeros(1),
    )
    assert negative["accepted"][0, :3].all()

    with pytest.raises(FloatingPointError, match="non-finite"):
        evaluate_investment_compare_entry(
            _BranchValuePolicy(nonfinite=True), candidates,
            x=torch.zeros(1), hatcf=torch.zeros(1), lnkf=torch.zeros(1),
        )
    candidates.entry_cost[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="candidate state or cost"):
        evaluate_investment_compare_entry(
            _BranchValuePolicy(), candidates,
            x=torch.zeros(1), hatcf=torch.zeros(1), lnkf=torch.zeros(1),
        )


def _investment_sim(*, horizon: int = 3, config=_Config) -> SimulateTS:
    return SimulateTS(
        models={"policy_value": _AlwaysInvestPolicy(i_threshold=config.I_THRESHOLD)},
        config=config,
        n_paths=1,
        group_size=1,
        horizon=horizon,
        branch_num=1,
        enable_entry=True,
        enable_exit=True,
        device=torch.device("cpu"),
        entry_mode="investment_compare",
        entry_spec_version=INVESTMENT_COMPARE_SPEC_VERSION,
        entry_capital_ratio=1.0,
        entry_size_ratio=1.0,
        entry_cost_max=config.I_THRESHOLD,
        entry_rng_seed=17,
        consumption_aggregation_mode="raw",
        node_accounting_mode="economic_node_ledger",
    )


def test_incumbent_only_maintenance_fixed_node_identity():
    simulator = _investment_sim(horizon=1, config=_PositiveDeltaConfig)
    K = torch.tensor([10.0, 2.0])
    zeros = torch.zeros_like(K)
    eligible = torch.tensor([True, False])

    Y, I_new, Phi, C_new_firm, maintenance = simulator._resource_accounting(
        K,
        zeros,
        0.0,
        zeros,
        zeros,
        zeros,
        maintenance_eligible=eligible,
        return_maintenance=True,
    )
    new = aggregate_node_resources(
        C_new_firm,
        torch.tensor([0, 0]),
        n_paths=1,
        I_entry=torch.tensor([0.6]),
        mode="raw",
    )
    _, I_old, Phi_old, C_old_firm, maintenance_old = (
        simulator._resource_accounting(
            K,
            zeros,
            0.0,
            zeros,
            zeros,
            zeros,
            return_maintenance=True,
        )
    )
    old = aggregate_node_resources(
        C_old_firm,
        torch.tensor([0, 0]),
        n_paths=1,
        I_entry=torch.tensor([0.6]),
        mode="raw",
    )

    assert Y.sum().item() == pytest.approx(12.0)
    assert Phi.sum().item() == pytest.approx(0.0)
    assert torch.equal(Phi, Phi_old)
    assert maintenance.sum().item() == pytest.approx(1.0)
    assert maintenance_old.sum().item() == pytest.approx(1.2)
    assert I_new.sum().item() == pytest.approx(1.0)
    assert I_old.sum().item() == pytest.approx(1.2)
    assert new["I_entry"].item() == pytest.approx(0.6)
    assert old["I_entry"].item() == pytest.approx(0.6)
    assert new["C_raw"].item() == pytest.approx(10.4)
    assert old["C_raw"].item() == pytest.approx(10.2)
    assert (
        (I_new.sum() + new["I_entry"])
        - (I_old.sum() + old["I_entry"])
    ).item() == pytest.approx(-0.2)
    assert (new["C_raw"] - old["C_raw"]).item() == pytest.approx(0.2)


def test_legacy_and_value_cost_keep_all_participating_firm_maintenance():
    for entry_mode, aggregation, node_mode in (
        ("legacy", "legacy_per_firm_clamp", "legacy_recompute"),
        ("value_cost", "raw", "economic_node_ledger"),
    ):
        simulator = SimulateTS(
            models={"policy_value": _AlwaysInvestPolicy()},
            config=_PositiveDeltaConfig,
            n_paths=1,
            group_size=1,
            horizon=1,
            branch_num=1,
            enable_entry=False,
            enable_exit=False,
            device=torch.device("cpu"),
            entry_mode=entry_mode,
            consumption_aggregation_mode=aggregation,
            node_accounting_mode=node_mode,
        )
        K = torch.tensor([10.0, 2.0])
        zeros = torch.zeros_like(K)
        _, _, _, _, maintenance = simulator._resource_accounting(
            K,
            zeros,
            0.0,
            zeros,
            zeros,
            zeros,
            return_maintenance=True,
        )
        torch.testing.assert_close(maintenance, torch.tensor([1.0, 0.2]))
        assert simulator.entry_configuration["resource_accounting_version"] == (
            ALL_FIRM_MAINTENANCE_VERSION
        )
        assert simulator.entry_configuration["birth_maintenance_exempt"] is False


def test_birth_identity_ignores_entry_flag_and_initial_cohort_pays_maintenance():
    path_index = torch.tensor([0, 0, 0])
    economic_time = torch.tensor([3])
    initial_cohort = torch.tensor([1.0, 0.0, 0.0])
    birth_time = torch.tensor([-1.0, 3.0, 2.0])
    first = identify_birth_at_node(
        initial_cohort, birth_time, economic_time, path_index
    )
    second = identify_birth_at_node(
        initial_cohort, birth_time, economic_time, path_index
    )
    assert first.tolist() == [False, True, False]
    assert torch.equal(first, second)

    simulator = _investment_sim(horizon=1, config=_PositiveDeltaConfig)
    K = torch.tensor([10.0, 2.0, 3.0])
    zeros = torch.zeros_like(K)
    _, _, _, _, maintenance = simulator._resource_accounting(
        K,
        zeros,
        0.0,
        zeros,
        zeros,
        zeros,
        maintenance_eligible=~first,
        return_maintenance=True,
    )
    torch.testing.assert_close(maintenance, torch.tensor([1.0, 0.0, 0.3]))


def test_independent_maintenance_rebuild_detects_mask_and_total_corruption():
    common = dict(
        Y=torch.tensor([10.0, 2.0]),
        Phi=torch.zeros(2),
        path_index=torch.tensor([0, 0]),
        birth_mask=torch.tensor([False, True]),
        entry_cost=torch.tensor([0.0, 0.3]),
        K_birth=torch.tensor([0.0, 2.0]),
        K_current=torch.tensor([10.0, 2.0]),
        n_paths=1,
        I_entry_ledger=torch.tensor([0.6]),
        mode="raw",
        bar_i_executed=torch.zeros(2),
        bar_z=torch.zeros(2),
        investment_cost=torch.zeros(2),
        initial_cohort=torch.tensor([1.0, 0.0]),
        birth_time=torch.tensor([-1.0, 1.0]),
        economic_time=torch.tensor([1]),
        delta=0.1,
        birth_maintenance_exempt=True,
    )
    valid = validate_node_resource_account(
        **common,
        I_oper=torch.tensor([1.0, 0.0]),
        maintenance_reported=torch.tensor([1.0, 0.0]),
        C_reported=torch.tensor([10.4]),
        strict=True,
    )
    assert valid["K_maintenance_base"].item() == pytest.approx(10.0)
    assert valid["K_birth_current_node"].item() == pytest.approx(2.0)
    assert valid["I_maintenance_rebuilt"].item() == pytest.approx(1.0)
    assert valid["maintenance_accounting_residual"].item() == pytest.approx(0.0)

    corruptions = (
        {
            "I_oper": torch.tensor([1.0, 0.2]),
            "maintenance_reported": torch.tensor([1.0, 0.2]),
            "C_reported": torch.tensor([10.2]),
        },
        {
            "I_oper": torch.tensor([0.0, 0.0]),
            "maintenance_reported": torch.tensor([0.0, 0.0]),
            "C_reported": torch.tensor([11.4]),
        },
        {
            "I_oper": torch.tensor([1.0, 0.0]),
            "maintenance_reported": torch.tensor([0.9, 0.0]),
            "C_reported": torch.tensor([10.4]),
        },
    )
    for corruption in corruptions:
        with pytest.raises(EntryAccountingError):
            validate_node_resource_account(
                **common,
                **corruption,
                strict=True,
            )


def test_i_oper_validation_uses_gross_scale_under_strong_cancellation():
    n_firms = 1024
    K = torch.ones(n_firms, dtype=torch.float32)
    bar_z = torch.cat(
        [
            torch.zeros(n_firms // 2, dtype=torch.float32),
            torch.full((n_firms // 2,), 2.0, dtype=torch.float32),
        ]
    )
    maintenance = torch.ones_like(K)
    I_oper_rebuilt = maintenance - bar_z
    I_oper_reported = I_oper_rebuilt.clone()
    I_oper_reported[0] += 2.5e-6
    Y = torch.full_like(K, 2.0)

    result = validate_node_resource_account(
        Y=Y,
        I_oper=I_oper_reported,
        Phi=torch.zeros_like(K),
        path_index=torch.zeros(n_firms, dtype=torch.long),
        birth_mask=torch.zeros(n_firms, dtype=torch.bool),
        entry_cost=torch.zeros_like(K),
        K_birth=torch.zeros_like(K),
        K_current=K,
        n_paths=1,
        I_entry_ledger=torch.zeros(1, dtype=torch.float32),
        C_reported=(Y - I_oper_reported).sum().reshape(1),
        mode="raw",
        bar_i_executed=torch.zeros_like(K),
        bar_z=bar_z,
        investment_cost=torch.zeros_like(K),
        initial_cohort=torch.ones_like(K),
        birth_time=torch.full_like(K, -1.0),
        economic_time=torch.tensor([1]),
        delta=1.0,
        maintenance_reported=maintenance,
        birth_maintenance_exempt=True,
        strict=True,
    )

    residual = result["I_oper_accounting_residual"].abs().item()
    net_scale = max(
        result["I_oper_reported"].abs().item(),
        result["I_oper_rebuilt"].abs().item(),
    )
    old_net_tolerance = 1e-6 + 1e-5 * net_scale
    assert residual > old_net_tolerance
    assert result["I_oper_gross_scale"].item() == pytest.approx(2048.0)
    assert residual <= result["I_oper_accounting_tolerance"].item()
    assert bool(result["accounting_valid"].item())


def test_investment_compare_birth_state_and_extra_expansion_timing():
    with (
        patch(
            "data.simulate_ts_parallel.sample_bernoulli",
            side_effect=lambda n, probability, device: torch.ones(n, device=device),
        ),
        patch(
            "data.simulate_ts_parallel.sample_uniform",
            side_effect=lambda n, low, high, device: torch.full(
                (n,), 0.2, device=device
            ),
        ),
    ):
        output = _investment_sim().simulate_tensor()
    firm = output.firm.to_dataframe()
    macro = output.macro.to_dataframe()
    births = firm[(firm["initial_cohort"] < 0.5) & (firm["birth_time"] == firm["t"])]
    assert not births.empty
    assert (births["b"] == 0.0).all()
    assert (births["ETA"] == 0.0).all()
    assert (births["i"] == births["entry_cost"]).all()
    assert births["entry_cost"].between(0.0, _Config.I_THRESHOLD).all()
    assert (births["K"] == births["K_birth"]).all()
    assert (births["Bar_i_model"] == 1.0).all()
    assert (births["Bar_i_executed"] == 0.0).all()
    assert (births["I"] == 0.0).all()
    assert (births["entry_PI"] >= births["entry_P0"]).all()
    assert (births["entry_value_gap"] >= 0.0).all()

    first = births.iloc[0]
    same_node = firm[
        (firm["ID"] == first["ID"])
        & (firm["t"] == first["t"])
    ]
    assert set(same_node["branch"].tolist()) == {-1, 0}
    assert (same_node["K"] == first["K_birth"]).all()
    assert (same_node["b"] == first["b"]).all()
    assert (same_node["ETA"] == first["ETA"]).all()
    assert (same_node["i"] == first["i"]).all()
    assert (same_node["entry_cost"] == first["entry_cost"]).all()
    assert (same_node["Bar_i_executed"] == 0.0).all()

    birth_child = births[
        (births["path"] == first["path"])
        & (births["t"] == first["t"])
        & (births["branch"] == 0)
    ]
    birth_macro = macro[
        (macro["path"] == first["path"])
        & (macro["t"] == first["t"])
        & (macro["branch"] == 0)
    ]
    assert len(birth_macro) == 1
    expected_entry_spend = (birth_child["entry_cost"] * birth_child["K_birth"]).sum()
    assert birth_macro.iloc[0]["I_entry"] == pytest.approx(expected_entry_spend)
    assert birth_macro.iloc[0]["I_entry_rebuilt"] == pytest.approx(expected_entry_spend)
    assert birth_macro.iloc[0]["entry_spend_residual"] == pytest.approx(0.0, abs=1e-6)

    next_period = firm[
        (firm["ID"] == first["ID"])
        & (firm["t"] == first["t"] + 1)
        & (firm["branch"] == 0)
    ]
    assert len(next_period) == 1
    assert next_period.iloc[0]["K"] == pytest.approx(first["K_birth"])
    assert next_period.iloc[0]["Bar_i_executed"] == pytest.approx(1.0)
    assert next_period.iloc[0]["entry_cost"] == pytest.approx(first["entry_cost"])
    assert next_period.iloc[0]["i"] == pytest.approx(0.2)
    assert next_period.iloc[0]["ETA"] == pytest.approx(1.0)

    after_investment = firm[
        (firm["ID"] == first["ID"])
        & (firm["t"] == first["t"] + 2)
        & (firm["branch"] == 0)
    ]
    assert len(after_investment) == 1
    assert after_investment.iloc[0]["K"] == pytest.approx(
        first["K_birth"] * _Config.G
    )

    initial = firm[(firm["initial_cohort"] > 0.5) & (firm["t"] == 1) & (firm["branch"] == 0)]
    assert len(initial) == 1
    assert initial.iloc[0]["Bar_i_executed"] == pytest.approx(1.0)


def test_investment_compare_maintenance_timing_and_authoritative_node_ledger():
    with (
        patch(
            "data.simulate_ts_parallel.sample_bernoulli",
            side_effect=lambda n, probability, device: torch.ones(n, device=device),
        ),
        patch(
            "data.simulate_ts_parallel.sample_uniform",
            side_effect=lambda n, low, high, device: torch.full(
                (n,), 0.2, device=device
            ),
        ),
    ):
        output = _investment_sim(
            horizon=3, config=_PositiveDeltaConfig
        ).simulate_tensor()
    firm = output.firm.to_dataframe()
    macro = output.macro.to_dataframe()
    births = firm[
        (firm["initial_cohort"] < 0.5)
        & (firm["birth_time"] == firm["t"])
        & (firm["branch"] == 0)
    ]
    assert not births.empty
    first = births.iloc[0]

    same_node = firm[
        (firm["ID"] == first["ID"])
        & (firm["t"] == first["t"])
    ]
    assert set(same_node["branch"].tolist()) == {-1, 0}
    assert (same_node["birth_at_node"] == 1.0).all()
    assert (same_node["maintenance_eligible"] == 0.0).all()
    assert (same_node["I_maintenance"] == 0.0).all()

    child_macro = macro[
        (macro["path"] == first["path"])
        & (macro["t"] == first["t"])
        & (macro["branch"] == 0)
    ].iloc[0]
    parent_macro = macro[
        (macro["path"] == first["path"])
        & (macro["t"] == first["t"])
        & (macro["branch"] == -1)
    ].iloc[0]
    for field in (
        "K_maintenance_base",
        "K_birth_current_node",
        "I_maintenance",
        "I_maintenance_rebuilt",
        "maintenance_accounting_residual",
        "birth_maintenance_exemption",
        "I_entry",
        "C_raw",
    ):
        assert parent_macro[field] == pytest.approx(child_macro[field])
    assert child_macro["K"] == pytest.approx(
        child_macro["K_maintenance_base"]
        + child_macro["K_birth_current_node"]
    )
    assert child_macro["I_maintenance"] == pytest.approx(
        _PositiveDeltaConfig.DELTA * child_macro["K_maintenance_base"]
    )
    assert child_macro["I_maintenance_rebuilt"] == pytest.approx(
        child_macro["I_maintenance"]
    )
    assert child_macro["maintenance_accounting_residual"] == pytest.approx(
        0.0, abs=1e-6
    )
    assert child_macro["birth_maintenance_exemption"] == pytest.approx(
        _PositiveDeltaConfig.DELTA * child_macro["K_birth_current_node"]
    )

    node_firms = firm[
        (firm["path"] == first["path"])
        & (firm["t"] == first["t"])
        & (firm["branch"] == 0)
    ].copy()
    birth_mask = (
        (node_firms["initial_cohort"] < 0.5)
        & (node_firms["birth_time"] == node_firms["t"])
    )
    maintenance_expected = (
        _PositiveDeltaConfig.DELTA
        * node_firms["K"]
        * (~birth_mask).astype(float)
    )
    i_oper_expected = (
        node_firms["Bar_i_executed"] * node_firms["K"] * node_firms["i"]
        - node_firms["Bar_z"] * node_firms["K"]
        + maintenance_expected
    )
    i_entry_expected = (
        node_firms.loc[birth_mask, "entry_cost"]
        * node_firms.loc[birth_mask, "K_birth"]
    ).sum()
    c_raw_expected = (
        node_firms["Y"] - i_oper_expected - node_firms["Phi"]
    ).sum() - i_entry_expected
    assert node_firms["I_maintenance"].to_numpy() == pytest.approx(
        maintenance_expected.to_numpy()
    )
    assert node_firms["I"].to_numpy() == pytest.approx(
        i_oper_expected.to_numpy()
    )
    assert child_macro["I_oper"] == pytest.approx(i_oper_expected.sum())
    assert child_macro["I_entry"] == pytest.approx(i_entry_expected)
    assert child_macro["C_raw"] == pytest.approx(c_raw_expected)

    next_period = firm[
        (firm["ID"] == first["ID"])
        & (firm["t"] == first["t"] + 1)
        & (firm["branch"] == 0)
    ].iloc[0]
    assert next_period["birth_at_node"] == 0.0
    assert next_period["maintenance_eligible"] == 1.0
    assert next_period["I_maintenance"] == pytest.approx(
        _PositiveDeltaConfig.DELTA * next_period["K"]
    )

    grown_period = firm[
        (firm["ID"] == first["ID"])
        & (firm["t"] == first["t"] + 2)
        & (firm["branch"] == 0)
    ].iloc[0]
    assert grown_period["K"] == pytest.approx(
        first["K_birth"] * _PositiveDeltaConfig.G
    )
    assert grown_period["I_maintenance"] == pytest.approx(
        _PositiveDeltaConfig.DELTA * grown_period["K"]
    )
    assert output.meta["entry"]["resource_accounting_version"] == (
        INVESTMENT_COMPARE_MAINTENANCE_VERSION
    )
    assert output.meta["entry"]["maintenance_basis"] == (
        "incumbent_capital_only"
    )
    assert output.meta["entry"]["birth_maintenance_exempt"] is True


def test_birth_node_exit_keeps_creation_phi_and_liquidation_but_no_maintenance():
    class _ExitAllPolicy(_AlwaysInvestPolicy):
        def forward_simulation(self, state: torch.Tensor):
            output = super().forward_simulation(state)
            output.bar_z = torch.ones_like(output.bar_z)
            return output

    simulator = _investment_sim(horizon=1, config=_PositiveDeltaExitConfig)
    simulator.models["policy_value"] = _ExitAllPolicy(
        i_threshold=_PositiveDeltaExitConfig.I_THRESHOLD
    )
    with (
        patch(
            "data.simulate_ts_parallel.sample_bernoulli",
            side_effect=lambda n, probability, device: torch.ones(n, device=device),
        ),
        patch(
            "data.simulate_ts_parallel.sample_uniform",
            side_effect=lambda n, low, high, device: torch.full(
                (n,), 0.2, device=device
            ),
        ),
    ):
        output = simulator.simulate_tensor()
    firm = output.firm.to_dataframe()
    macro = output.macro.to_dataframe()
    births = firm[
        (firm["initial_cohort"] < 0.5)
        & (firm["birth_time"] == firm["t"])
        & (firm["branch"] == 0)
    ]
    assert not births.empty
    assert (births["Bar_z"] == 1.0).all()
    assert (births["I_maintenance"] == 0.0).all()
    assert births["Phi"].to_numpy() == pytest.approx(
        (
            (1.0 - _PositiveDeltaExitConfig.PHI)
            * (1.0 + torch.exp(torch.tensor(0.0)).item())
            * births["K"]
        ).to_numpy()
    )
    node = macro[
        (macro["path"] == births.iloc[0]["path"])
        & (macro["t"] == births.iloc[0]["t"])
        & (macro["branch"] == 0)
    ].iloc[0]
    assert node["same_node_entrant_exit_count"] == pytest.approx(
        node["accepted_count"]
    )
    assert node["K_entry_same_node_exit"] == pytest.approx(node["K_entry_gross"])
    assert node["I_entry"] > 0.0
    assert node["birth_maintenance_exemption"] == pytest.approx(
        _PositiveDeltaExitConfig.DELTA * node["K_birth_current_node"]
    )


def test_investment_compare_cost_configuration_is_ordinary_i_support():
    simulator = _investment_sim(horizon=1)
    assert simulator.effective_entry_cost_max == pytest.approx(_Config.I_THRESHOLD)
    assert simulator.entry_configuration["cost_distribution_source"] == "ordinary_investment_cost"
    assert simulator.entry_configuration["birth_i_equals_entry_cost"] is True
    assert simulator.entry_configuration["birth_extra_expansion"] is False

    with pytest.raises(ValueError, match="entry_cost_max"):
        SimulateTS(
            models={"policy_value": _AlwaysInvestPolicy()}, config=_Config,
            n_paths=1, group_size=1, horizon=1, branch_num=1,
            entry_mode="investment_compare",
            entry_spec_version=INVESTMENT_COMPARE_SPEC_VERSION,
            entry_cost_max=0.25,
            consumption_aggregation_mode="raw",
            node_accounting_mode="economic_node_ledger",
        )

    with pytest.raises(ValueError, match="i_threshold"):
        SimulateTS(
            models={"policy_value": _AlwaysInvestPolicy(i_threshold=0.25)}, config=_Config,
            n_paths=1, group_size=1, horizon=1, branch_num=1,
            entry_mode="investment_compare",
            entry_spec_version=INVESTMENT_COMPARE_SPEC_VERSION,
            entry_cost_max=_Config.I_THRESHOLD,
            consumption_aggregation_mode="raw",
            node_accounting_mode="economic_node_ledger",
        )


def test_runner_resolves_investment_compare_version_and_cost_support():
    argv = [
        "run_multi_episode_job.py",
        "--entry-mode", "investment_compare",
        "--consumption-aggregation-mode", "raw",
        "--node-accounting-mode", "economic_node_ledger",
    ]
    with patch("sys.argv", argv):
        hp = configure_hyperparams(parse_args())
    assert hp.entry_spec_version == INVESTMENT_COMPARE_SPEC_VERSION
    assert hp.resource_accounting_version == INVESTMENT_COMPARE_MAINTENANCE_VERSION
    assert hp.entry_cost_max == pytest.approx(_Config.I_THRESHOLD)

    with patch("sys.argv", argv + ["--entry-cost-max", "0.25"]):
        with pytest.raises(ValueError, match="entry-cost-max"):
            configure_hyperparams(parse_args())
