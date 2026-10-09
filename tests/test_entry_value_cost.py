from __future__ import annotations

import math
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from data.entry import (
    aggregate_node_resources,
    capital_growth_decomposition,
    compute_entry_reference,
    draw_value_cost_candidates,
    entry_configuration_fingerprint,
    entry_configuration_snapshot,
    evaluate_entry_cutoff,
    make_entry_generator,
    stochastic_round_counts,
)
from data.simulate_ts import SimulateTS
from data.sample import Sample
from experiments.fill_fullN_entrants import fill_df_to_fullN
from experiments.run_multi_episode_job import configure_hyperparams, parse_args
from training.episode import Episode


class _Output:
    def __init__(self, p):
        self.P = p


class _MockPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("seen", torch.tensor(0.0))

    def forward_simulation(self, state):
        # Economic P depends on eta and z, but deliberately not dummy i.
        return _Output((1.0 + state[:, 1:2] + 2.0 * state[:, 2:3]) + self.anchor)


class _SimulationPolicy(nn.Module):
    def __init__(self, *, p=2.0, bar_z=0.0, default_zero_debt=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.p = float(p)
        self.default = float(bar_z)
        self.default_zero_debt = bool(default_zero_debt)

    def forward_simulation(self, state):
        n = state.shape[0]
        zeros = state.new_zeros((n, 1)) + self.anchor * 0.0
        p = zeros + self.p
        bp = zeros + 0.4
        bar_z = zeros + self.default
        if self.default_zero_debt:
            bar_z = (state[:, 0:1] == 0.0).to(state.dtype)
        return SimpleNamespace(
            Q=zeros,
            P0=p,
            PI=p,
            bar_i=zeros,
            bar_z=bar_z,
            P=p,
            bp0=bp,
            bpI=bp,
            bp=bp,
        )


class _TestConfig:
    DEVICE = torch.device("cpu")
    RHO_X = 0.0
    SIGMA_X = 0.0
    XBAR = 0.0
    RHO_Z = 0.0
    SIGMA_Z = 0.0
    ZBAR = 0.0
    DELTA = 0.0
    PHI = 1.0
    G = 1.14
    I_THRESHOLD = 0.5
    ZETA = 0.25
    SIM_B_INIT_MIN = 0.0
    SIM_B_INIT_MAX = 1.0
    ENTRY_B_MIN = 0.0
    ENTRY_B_MAX = 1.0


def test_reference_scale_and_scale_invariance():
    K = torch.tensor([[1.0, 3.0, 99.0]])
    alive = torch.tensor([[True, True, False]])
    ref = compute_entry_reference(K, alive, entry_capital_ratio=0.3, entry_size_ratio=0.5)
    assert ref.reference_K.item() == 4.0
    assert ref.reference_N.item() == 2
    assert ref.mean_K_ref.item() == 2.0
    assert ref.K_per_entrant.item() == 1.0
    assert ref.potential_capital_nominal.item() == pytest.approx(1.2)
    assert ref.n_star.item() == pytest.approx(1.2)
    scaled = compute_entry_reference(10 * K, alive, entry_capital_ratio=0.3, entry_size_ratio=0.5)
    assert scaled.K_per_entrant.item() == 10.0
    assert scaled.n_star.item() == pytest.approx(ref.n_star.item())


def test_zero_ratio_and_extinct_path_make_no_candidates():
    K = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    alive = torch.tensor([[True, True], [False, False]])
    ref = compute_entry_reference(K, alive, entry_capital_ratio=0.0, entry_size_ratio=0.5)
    candidates = draw_value_cost_candidates(
        ref, rho_z=0.9, sigma_z=0.1, zbar=0.0, entry_cost_max=1.0,
        generator=make_entry_generator(1),
    )
    assert candidates.counts.tolist() == [0, 0]
    assert candidates.mask.numel() == 0


def test_stochastic_rounding_has_correct_long_run_mean_and_private_rng():
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    generator = make_entry_generator(9)
    values = torch.full((20000,), 2.25)
    rounded = stochastic_round_counts(values, generator=generator)
    assert rounded.float().mean().item() == pytest.approx(2.25, abs=0.02)
    assert torch.equal(before, torch.random.get_rng_state())


def test_value_screen_uses_zeta_weighted_P_and_not_dummy_i():
    ref = compute_entry_reference(
        torch.tensor([[2.0]]), torch.tensor([[True]]),
        entry_capital_ratio=1.0, entry_size_ratio=1.0,
    )
    candidates = draw_value_cost_candidates(
        ref, rho_z=0.0, sigma_z=0.0, zbar=0.0, entry_cost_max=1.0,
        generator=make_entry_generator(2),
    )
    candidates.entry_cost.fill_(1.4)
    model = _MockPolicy()
    args = dict(
        policy_value_model=model, candidates=candidates,
        x=torch.zeros(1), hatcf=torch.zeros(1), lnkf=torch.zeros(1), zeta=0.25,
    )
    low = evaluate_entry_cutoff(**args, dummy_i=0.0)
    high = evaluate_entry_cutoff(**args, dummy_i=999.0)
    assert low["entry_cutoff"][0, 0].item() == pytest.approx(1.5)
    assert low["accepted"][0, 0]
    assert torch.equal(low["entry_cutoff"], high["entry_cutoff"])
    candidates.entry_cost.fill_(1.5)
    rejected = evaluate_entry_cutoff(**args)
    assert not rejected["accepted"][0, 0]


def test_entry_forward_is_read_only_and_rejects_nonfinite():
    ref = compute_entry_reference(
        torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool),
        entry_capital_ratio=1.0, entry_size_ratio=1.0,
    )
    candidates = draw_value_cost_candidates(
        ref, rho_z=0.0, sigma_z=0.0, zbar=0.0, entry_cost_max=1.0,
        generator=make_entry_generator(3),
    )
    model = _MockPolicy().train()
    evaluate_entry_cutoff(
        model, candidates, x=torch.zeros(1), hatcf=torch.zeros(1),
        lnkf=torch.zeros(1), zeta=0.03,
    )
    assert model.training
    assert model.seen.item() == 0.0

    class Bad(_MockPolicy):
        def forward_simulation(self, state):
            return _Output(torch.full((state.shape[0], 1), float("nan")))

    with pytest.raises(FloatingPointError):
        evaluate_entry_cutoff(
            Bad(), candidates, x=torch.zeros(1), hatcf=torch.zeros(1),
            lnkf=torch.zeros(1), zeta=0.03,
        )


def test_raw_resource_accounting_preserves_negative_firm_contribution():
    result = aggregate_node_resources(
        torch.tensor([2.0, -0.5]), torch.tensor([0, 0]),
        n_paths=1, I_entry=torch.tensor([0.2]), mode="raw",
    )
    assert result["C_oper"].item() == pytest.approx(1.5)
    assert result["C_raw"].item() == pytest.approx(1.3)
    assert result["resource_accounting_residual"].item() == 0.0
    legacy = aggregate_node_resources(
        torch.tensor([2.0, -0.5]), torch.tensor([0, 0]),
        n_paths=1, I_entry=torch.tensor([0.2]), mode="legacy_per_firm_clamp",
    )
    assert legacy["C_raw"].item() == pytest.approx(1.8)


@pytest.mark.parametrize(
    "previous_ids,previous_K,next_ids,next_K",
    [
        ([1], [2.0], [1], [2.0]),
        ([1], [2.0], [1], [3.0]),
        ([1], [2.0], [1, 2], [2.0, 0.5]),
        ([1, 2], [2.0, 0.5], [1], [2.0]),
        ([1, 2], [2.0, 0.5], [1, 3], [3.0, 0.7]),
    ],
)
def test_capital_growth_decomposition_identity(previous_ids, previous_K, next_ids, next_K):
    result = capital_growth_decomposition(
        torch.tensor(previous_ids), torch.tensor(previous_K),
        torch.tensor(next_ids), torch.tensor(next_K),
    )
    assert abs(result["capital_accounting_residual"].item()) < 1e-7


def _value_cost_sim(*, horizon=2, capital_ratio=1.0, bp_source="head"):
    model = _SimulationPolicy()
    kwargs = {}
    if bp_source == "grid":
        kwargs.update(
            bp_action_source="grid",
            bp_grid_policy=lambda **context: torch.full_like(context["bp_head"], 0.6),
        )
    simulator = SimulateTS(
        models={"policy_value": model},
        config=_TestConfig,
        n_paths=1,
        group_size=2,
        horizon=horizon,
        branch_num=1,
        enable_entry=True,
        enable_exit=True,
        device=torch.device("cpu"),
        entry_mode="value_cost",
        entry_capital_ratio=capital_ratio,
        entry_size_ratio=0.25,
        entry_cost_max=0.01,
        entry_rng_seed=17,
        consumption_aggregation_mode="raw",
        **kwargs,
    )
    return simulator, model


@pytest.mark.parametrize("bp_source", ["head", "grid"])
def test_formal_value_cost_simulation_is_accounted_idempotent_and_dynamic(bp_source):
    simulator, model = _value_cost_sim(bp_source=bp_source)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    output = simulator.simulate_tensor()
    firm = output.firm.to_dataframe()
    macro = output.macro.to_dataframe()

    entrants = firm[firm["entry"] > 0.5]
    assert not entrants.empty
    assert (entrants["b"] == 0.0).all()
    assert (entrants["initial_cohort"] == 0.0).all()
    assert (entrants["birth_time"] >= 1.0).all()
    assert (entrants["entry_cost"] >= 0.0).all()
    assert (entrants["K_birth"] > 0.0).all()
    assert (entrants["K_birth"] == entrants["K"]).all()
    assert output.meta["max_firms"] > simulator.group_size

    child = macro[(macro["t"] == 1.0) & (macro["branch"] == 0.0)].iloc[0]
    parent = macro[(macro["t"] == 1.0) & (macro["branch"] == -1.0)].iloc[0]
    assert child["economic_node_id"] == parent["economic_node_id"]
    for field in ("I_entry", "C_raw", "Hatc", "LnK", "K"):
        assert parent[field] == pytest.approx(child[field])
    assert child["accounting_stage"] == 0.0
    assert parent["accounting_stage"] == 1.0
    assert abs(child["resource_accounting_residual"]) < 1e-7
    assert abs(child["capital_accounting_residual"]) < 1e-6
    assert child["entry_cutoff_mean"] == pytest.approx(2.0)
    assert 0.0 <= child["entry_cost_mean"] <= 0.01
    assert output.meta["entry"]["entry_configuration_fingerprint"]
    for name, value in model.state_dict().items():
        assert torch.equal(before[name], value)


def test_zero_entry_ratio_produces_no_birth_or_creation_spending():
    simulator, _ = _value_cost_sim(horizon=1, capital_ratio=0.0)
    output = simulator.simulate_tensor()
    firm = output.firm.to_dataframe()
    macro = output.macro.to_dataframe()
    assert not (firm["entry"] > 0.5).any()
    assert (macro["potential_count"] == 0.0).all()
    assert (macro["accepted_count"] == 0.0).all()
    assert (macro["I_entry"] == 0.0).all()


def test_entry_fingerprint_is_configuration_sensitive():
    common = dict(
        entry_mode="value_cost",
        entry_spec_version="value_cost_v1",
        entry_capital_ratio=0.1,
        entry_size_ratio=0.1,
        entry_cost_max=1.0,
        entry_dummy_i=0.0,
        entry_inference_chunk_size=64,
        entry_rng_seed=7,
        consumption_aggregation_mode="raw",
        economic_config=_TestConfig,
    )
    first = entry_configuration_fingerprint(entry_configuration_snapshot(**common))
    second = entry_configuration_fingerprint(
        entry_configuration_snapshot(**{**common, "entry_cost_max": 2.0})
    )
    assert first != second

    episode = Episode.__new__(Episode)
    episode.config = _TestConfig
    episode.hyperparams = SimpleNamespace(
        entry_mode="value_cost",
        entry_spec_version="value_cost_v1",
        entry_capital_ratio=0.1,
        entry_size_ratio=0.1,
        entry_cost_max=1.0,
        entry_dummy_i=0.0,
        entry_rng_seed=7,
        consumption_aggregation_mode="raw",
    )
    cache_hash_before = episode._pq_grid_config_hash()
    episode.hyperparams.entry_cost_max = 2.0
    assert episode._pq_grid_config_hash() != cache_hash_before


def test_unsupported_legacy_entry_paths_reject_value_cost():
    simulator, _ = _value_cost_sim(horizon=1)
    with pytest.raises(RuntimeError, match="formal tensor-path-parallel"):
        simulator._simulate_path_tensor(0)


def test_same_node_entrant_exit_keeps_creation_cost_and_ledger_on_parent_view():
    simulator, _ = _value_cost_sim(horizon=2)
    simulator.models["policy_value"] = _SimulationPolicy(default_zero_debt=True)
    output = simulator.simulate_tensor()
    macro = output.macro.to_dataframe()
    child = macro[(macro["t"] == 1.0) & (macro["branch"] == 0.0)].iloc[0]
    parent = macro[(macro["t"] == 1.0) & (macro["branch"] == -1.0)].iloc[0]
    assert child["same_node_entrant_exit_count"] == child["accepted_count"]
    assert child["K_entry_same_node_exit"] == pytest.approx(child["K_entry_gross"])
    assert child["K_entry_surviving"] == 0.0
    assert child["I_entry"] > 0.0
    for field in (
        "I_entry",
        "C_raw",
        "same_node_entrant_exit_count",
        "K_entry_same_node_exit",
        "K_entry_surviving",
    ):
        assert parent[field] == pytest.approx(child[field])


def test_raw_infeasible_node_fails_without_positive_floor():
    class InfeasibleConfig(_TestConfig):
        DELTA = 2.0

    simulator = SimulateTS(
        models={"policy_value": _SimulationPolicy()},
        config=InfeasibleConfig,
        n_paths=1,
        group_size=2,
        horizon=1,
        branch_num=1,
        device=torch.device("cpu"),
        entry_mode="value_cost",
        entry_capital_ratio=0.0,
        entry_size_ratio=0.25,
        entry_cost_max=0.01,
        consumption_aggregation_mode="raw",
    )
    with pytest.raises(RuntimeError, match="C_raw<=0"):
        simulator.simulate_tensor()


def test_runner_entry_configuration_reaches_effective_hyperparams():
    argv = [
        "run_multi_episode_job.py",
        "--entry-mode", "value_cost",
        "--entry-capital-ratio", "0.2",
        "--entry-size-ratio", "0.05",
        "--entry-cost-max", "1.25",
        "--entry-dummy-i", "0.1",
        "--entry-inference-chunk-size", "1234",
        "--entry-rng-seed", "99",
        "--consumption-aggregation-mode", "raw",
    ]
    with patch.object(sys, "argv", argv):
        hp = configure_hyperparams(parse_args())
    assert hp.entry_mode == "value_cost"
    assert hp.entry_capital_ratio == pytest.approx(0.2)
    assert hp.entry_size_ratio == pytest.approx(0.05)
    assert hp.entry_cost_max == pytest.approx(1.25)
    assert hp.entry_dummy_i == pytest.approx(0.1)
    assert hp.entry_inference_chunk_size == 1234
    assert hp.entry_rng_seed == 99
    assert hp.consumption_aggregation_mode == "raw"


def test_non_economic_padding_paths_reject_value_cost_mode():
    with pytest.raises(RuntimeError, match="bootstrap/coverage"):
        Sample(
            models={},
            n_samples=2,
            n_paths=1,
            data_mode="simulate",
            enable_entry=True,
            entry_mode="value_cost",
            device=torch.device("cpu"),
        )
    with pytest.raises(RuntimeError, match="numerical coverage"):
        fill_df_to_fullN(
            __import__("pandas").DataFrame(),
            full_N=2,
            entry_mode="value_cost",
        )


def test_matched_smoke_controls_keep_macro_and_incumbent_shocks_common():
    common = dict(
        models={"policy_value": _SimulationPolicy()},
        config=_TestConfig,
        n_paths=1,
        group_size=2,
        horizon=2,
        branch_num=1,
        enable_entry=True,
        enable_exit=True,
        device=torch.device("cpu"),
        entry_capital_ratio=1.0,
        entry_size_ratio=0.25,
        entry_cost_max=0.01,
        entry_rng_seed=17,
        preserve_global_rng_around_entry=True,
        common_transition_seed=321,
    )
    torch.manual_seed(77)
    legacy = SimulateTS(
        **common,
        entry_mode="legacy",
        consumption_aggregation_mode="legacy_per_firm_clamp",
    ).simulate_tensor()
    torch.manual_seed(77)
    value_cost = SimulateTS(
        **common,
        entry_mode="value_cost",
        consumption_aggregation_mode="raw",
    ).simulate_tensor()
    legacy_macro = legacy.macro.to_dataframe().sort_values(["path", "t", "branch"])
    value_macro = value_cost.macro.to_dataframe().sort_values(["path", "t", "branch"])
    torch.testing.assert_close(
        torch.tensor(legacy_macro["x"].to_numpy()),
        torch.tensor(value_macro["x"].to_numpy()),
    )
    legacy_firm = legacy.firm.to_dataframe()
    value_firm = value_cost.firm.to_dataframe()
    keys = ["path", "t", "branch", "ID"]
    left = legacy_firm[legacy_firm["initial_cohort"] > 0.5][keys + ["z"]]
    right = value_firm[value_firm["initial_cohort"] > 0.5][keys + ["z"]]
    merged = left.merge(right, on=keys, suffixes=("_legacy", "_value"))
    assert not merged.empty
    assert (merged["z_legacy"] == merged["z_value"]).all()
