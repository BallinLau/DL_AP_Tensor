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
    validate_node_resource_account,
)
from data.simulate_ts import SimulateTS
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
