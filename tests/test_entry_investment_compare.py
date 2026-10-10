from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from data.entry import (
    EntryCandidates,
    INVESTMENT_COMPARE_SPEC_VERSION,
    evaluate_investment_compare_entry,
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


def _investment_sim(*, horizon: int = 3) -> SimulateTS:
    return SimulateTS(
        models={"policy_value": _AlwaysInvestPolicy(i_threshold=_Config.I_THRESHOLD)},
        config=_Config,
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
        entry_cost_max=_Config.I_THRESHOLD,
        entry_rng_seed=17,
        consumption_aggregation_mode="raw",
        node_accounting_mode="economic_node_ledger",
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
    assert hp.entry_cost_max == pytest.approx(_Config.I_THRESHOLD)

    with patch("sys.argv", argv + ["--entry-cost-max", "0.25"]):
        with pytest.raises(ValueError, match="entry-cost-max"):
            configure_hyperparams(parse_args())
