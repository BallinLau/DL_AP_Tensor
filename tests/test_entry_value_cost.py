from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from data.entry import (
    EntryEconomicInfeasibilityError,
    EntryAccountingError,
    aggregate_node_resources,
    capital_growth_decomposition,
    compute_entry_reference,
    draw_value_cost_candidates,
    entry_configuration_fingerprint,
    entry_configuration_snapshot,
    evaluate_entry_cutoff,
    make_entry_generator,
    validate_node_resource_account,
    stochastic_round_counts,
)
from data.simulate_ts import SimulateTS
from data.simulate_ts_parallel import _expand_branches_batched
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


class _RecordingSDF(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.inputs = []

    def forward_step(self, *, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical):
        self.inputs.append((hatcf_prev.detach().clone(), lnkf_prev.detach().clone()))
        zeros = x_prev.new_zeros(x_prev.shape) + self.anchor * 0.0
        ones = torch.ones_like(zeros)
        return zeros, zeros, ones, hatcf_prev, lnkf_prev


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
    assert "resource_accounting_residual" not in result
    legacy = aggregate_node_resources(
        torch.tensor([2.0, -0.5]), torch.tensor([0, 0]),
        n_paths=1, I_entry=torch.tensor([0.2]), mode="legacy_per_firm_clamp",
    )
    assert legacy["C_raw"].item() == pytest.approx(1.8)


def test_independent_resource_rebuild_detects_ledger_and_detail_corruption():
    common = dict(
        Y=torch.tensor([2.0, 0.0]),
        I_oper=torch.tensor([0.0, 0.5]),
        Phi=torch.tensor([0.0, 0.0]),
        path_index=torch.tensor([0, 0]),
        birth_mask=torch.tensor([True, False]),
        entry_cost=torch.tensor([0.2, 0.0]),
        K_birth=torch.tensor([1.0, 0.0]),
        K_current=torch.tensor([1.0, 1.0]),
        n_paths=1,
        mode="raw",
    )
    valid = validate_node_resource_account(
        **common,
        I_entry_ledger=torch.tensor([0.2]),
        C_reported=torch.tensor([1.3]),
    )
    assert valid["I_entry_rebuilt"].item() == pytest.approx(0.2)
    assert valid["C_rebuilt"].item() == pytest.approx(1.3)
    assert valid["entry_spend_residual"].item() == pytest.approx(0.0)
    assert valid["resource_accounting_residual"].item() == pytest.approx(0.0)
    assert bool(valid["accounting_valid"].item())

    corruptions = [
        {"I_entry_ledger": torch.tensor([0.3]), "C_reported": torch.tensor([1.3])},
        {"I_entry_ledger": torch.tensor([0.2]), "C_reported": torch.tensor([1.2])},
        {
            "I_entry_ledger": torch.tensor([0.2]),
            "C_reported": torch.tensor([1.3]),
            "birth_mask": torch.tensor([True, True]),
            "entry_cost": torch.tensor([0.2, 0.2]),
            "K_birth": torch.tensor([1.0, 1.0]),
        },
        {
            "I_entry_ledger": torch.tensor([0.2]),
            "C_reported": torch.tensor([1.3]),
            "birth_mask": torch.tensor([False, False]),
        },
        {"I_entry_ledger": torch.tensor([float("inf")]), "C_reported": torch.tensor([1.3])},
    ]
    for overrides in corruptions:
        with pytest.raises(EntryAccountingError):
            validate_node_resource_account(**{**common, **overrides}, strict=True)


def test_resource_account_reconstruction_accepts_float32_gpu_reduction_noise():
    # Captured from a formal A800 rollout. Separate, mathematically equivalent
    # float32 index_add_ reductions differed only in atomic accumulation order.
    C_rebuilt = torch.tensor([2.2625603675842285], dtype=torch.float32)
    C_reported = torch.tensor([2.262557029724121], dtype=torch.float32)
    result = validate_node_resource_account(
        Y=C_rebuilt.clone(),
        I_oper=torch.zeros(1, dtype=torch.float32),
        Phi=torch.zeros(1, dtype=torch.float32),
        path_index=torch.tensor([0]),
        birth_mask=torch.tensor([False]),
        entry_cost=torch.zeros(1, dtype=torch.float32),
        K_birth=torch.zeros(1, dtype=torch.float32),
        K_current=torch.ones(1, dtype=torch.float32),
        n_paths=1,
        I_entry_ledger=torch.zeros(1, dtype=torch.float32),
        C_reported=C_reported,
        mode="raw",
        strict=True,
    )

    residual = result["resource_accounting_residual"].abs().item()
    old_tolerance = 1e-6 + 1e-6 * max(C_reported.abs().item(), C_rebuilt.abs().item())
    assert residual > old_tolerance
    assert residual == pytest.approx(3.337860107421875e-6)
    assert bool(result["accounting_valid"].item())


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


def test_capital_decomposition_reports_actual_endpoints_and_rejects_duplicate_ids():
    result = capital_growth_decomposition(
        torch.tensor([10, 11]), torch.tensor([2.0, 1.0]),
        torch.tensor([10]), torch.tensor([2.0]),
    )
    assert result["K_decomposition_start"].item() == pytest.approx(3.0)
    assert result["K_decomposition_end"].item() == pytest.approx(2.0)
    assert result["K_exit_old"].item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unique"):
        capital_growth_decomposition(
            torch.tensor([10, 10]), torch.tensor([1.0, 1.0]),
            torch.tensor([10]), torch.tensor([1.0]),
        )


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
        node_accounting_mode="economic_node_ledger",
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
    assert abs(child["entry_spend_residual"]) < 1e-7
    assert abs(child["capital_accounting_residual"]) < 1e-6
    assert child["K_node_pre_exit"] >= child["K_endpoint_post_exit"]
    assert child["K_decomposition_end"] == pytest.approx(child["K_endpoint_post_exit"])
    assert child["K_entry_gross"] == pytest.approx(
        child["K_entry_surviving"] + child["K_entry_same_node_exit"]
    )
    assert child["entry_capital_event_residual"] == pytest.approx(0.0, abs=1e-7)
    assert child["entry_cutoff_mean"] == pytest.approx(2.0)
    assert 0.0 <= child["entry_cost_mean"] <= 0.01
    assert output.meta["entry"]["entry_configuration_fingerprint"]
    for name, value in model.state_dict().items():
        assert torch.equal(before[name], value)


def test_exported_firm_details_independently_rebuild_entry_spend_and_consumption():
    simulator, _ = _value_cost_sim(horizon=2)
    output = simulator.simulate_tensor()
    firm = output.firm.to_dataframe()
    macro = output.macro.to_dataframe()
    child_macro = macro[(macro["t"] == 1.0) & (macro["branch"] == 0.0)].iloc[0]
    node_firms = firm[
        (firm["path"] == child_macro["path"])
        & (firm["economic_node_id"] == child_macro["economic_node_id"])
        & (firm["accounting_stage"] == 0.0)
    ]
    assert not node_firms.empty
    births = (
        (node_firms["entry"] > 0.5)
        & (node_firms["initial_cohort"] < 0.5)
        & (node_firms["birth_time"] == child_macro["t"])
    )
    rebuilt_entry_spend = (
        node_firms.loc[births, "entry_cost"]
        * node_firms.loc[births, "K_birth"]
    ).sum()
    rebuilt_consumption = node_firms["C"].sum() - rebuilt_entry_spend
    assert rebuilt_entry_spend == pytest.approx(child_macro["I_entry"], abs=1e-6)
    assert rebuilt_consumption == pytest.approx(child_macro["C_raw"], abs=1e-5)


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
        node_accounting_mode="economic_node_ledger",
        transition_rng_mode="legacy_position",
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
        "I_entry_rebuilt",
        "entry_spend_residual",
    ):
        assert parent[field] == pytest.approx(child[field])


def test_exit_toggle_controls_endpoint_capital_and_exported_decomposition():
    class ExitSecondFirm(_SimulationPolicy):
        def forward_simulation(self, state):
            out = super().forward_simulation(state)
            out.bar_z = torch.ones_like(state[:, 0:1])
            return out

    common = dict(
        models={"policy_value": ExitSecondFirm()},
        config=_TestConfig,
        n_paths=1,
        group_size=2,
        horizon=1,
        branch_num=1,
        enable_entry=False,
        device=torch.device("cpu"),
        entry_mode="legacy",
        consumption_aggregation_mode="legacy_per_firm_clamp",
        node_accounting_mode="legacy_recompute",
    )
    torch.manual_seed(41)
    with_exit = SimulateTS(**common, enable_exit=True).simulate()
    torch.manual_seed(41)
    no_exit = SimulateTS(**common, enable_exit=False).simulate()
    macro_exit = with_exit[1].query("t == 1 and branch == 0").iloc[0]
    macro_no_exit = no_exit[1].query("t == 1 and branch == 0").iloc[0]
    assert macro_exit["K_node_pre_exit"] >= macro_exit["K_endpoint_post_exit"]
    assert macro_exit["K_exit_old"] >= 0.0
    assert macro_no_exit["K_node_pre_exit"] == pytest.approx(macro_no_exit["K_endpoint_post_exit"])
    assert macro_no_exit["K_exit_old"] == pytest.approx(0.0)


def test_legacy_recompute_matches_4b236ea_exit_fixture_and_fc1_parent_macro():
    class HistoricalExitPolicy(_SimulationPolicy):
        def forward_simulation(self, state):
            out = super().forward_simulation(state)
            out.bar_z = (state[:, 0:1] > 0.2).to(state.dtype)
            return out

    sdf = _RecordingSDF()
    torch.manual_seed(123)
    simulator = SimulateTS(
        models={"policy_value": HistoricalExitPolicy(), "sdf_fc1": sdf},
        config=_TestConfig,
        n_paths=1,
        group_size=4,
        horizon=2,
        branch_num=1,
        enable_entry=False,
        enable_exit=True,
        device=torch.device("cpu"),
        entry_mode="legacy",
        consumption_aggregation_mode="legacy_per_firm_clamp",
        node_accounting_mode="legacy_recompute",
        transition_rng_mode="legacy_position",
    )
    _, macro = simulator.simulate()
    parent = macro.query("t == 1 and branch == -1").iloc[0]
    # Frozen fixture generated from 4b236ea with the same seed/model/config.
    assert parent["K"] == pytest.approx(21.430923, abs=2e-5)
    assert parent["C"] == pytest.approx(21.430923, abs=2e-5)
    assert parent["LnK"] == pytest.approx(3.064835, abs=2e-5)
    assert parent["Hatc"] == pytest.approx(0.000010, abs=2e-6)
    assert parent["n_firms"] == 1
    assert len(sdf.inputs) == 2
    hatc_parent, lnk_parent = sdf.inputs[-1]
    assert hatc_parent.item() == pytest.approx(parent["Hatc"], abs=2e-6)
    assert lnk_parent.item() == pytest.approx(parent["LnK"], abs=2e-5)


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
        "--node-accounting-mode", "economic_node_ledger",
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
    assert hp.node_accounting_mode == "economic_node_ledger"


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


def test_identity_keyed_crn_survives_capacity_padding_and_slot_reordering():
    class NondegenerateConfig(_TestConfig):
        RHO_X = 0.3
        SIGMA_X = 0.2
        RHO_Z = 0.4
        SIGMA_Z = 0.3

    class ExitByZPolicy(_SimulationPolicy):
        def forward_simulation(self, state):
            out = super().forward_simulation(state)
            out.bar_z = (state[:, 1:2] > 0.2).to(state.dtype)
            return out

    common = dict(
        models={"policy_value": ExitByZPolicy()},
        config=NondegenerateConfig,
        n_paths=2,
        group_size=3,
        horizon=3,
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
        transition_rng_mode="stable_firm_identity",
        node_accounting_mode="economic_node_ledger",
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
    macro_keys = ["path", "t", "branch", "economic_node_id"]
    macro_merged = legacy_macro[macro_keys + ["eps_x_transition"]].merge(
        value_macro[macro_keys + ["eps_x_transition"]],
        on=macro_keys,
        suffixes=("_legacy", "_value"),
    )
    macro_valid = (
        macro_merged["eps_x_transition_legacy"].notna()
        & macro_merged["eps_x_transition_value"].notna()
    )
    assert macro_valid.any()
    assert (
        macro_merged.loc[macro_valid, "eps_x_transition_legacy"]
        == macro_merged.loc[macro_valid, "eps_x_transition_value"]
    ).all()
    legacy_firm = legacy.firm.to_dataframe()
    value_firm = value_cost.firm.to_dataframe()
    keys = ["path", "t", "branch", "ID"]
    shock_fields = ["eps_z_transition", "u_eta_transition", "u_i_transition"]
    left = legacy_firm[legacy_firm["initial_cohort"] > 0.5][keys + shock_fields]
    right = value_firm[value_firm["initial_cohort"] > 0.5][keys + shock_fields]
    merged = left.merge(right, on=keys, suffixes=("_legacy", "_value"))
    assert not merged.empty
    for field in shock_fields:
        finite = merged[f"{field}_legacy"].notna() & merged[f"{field}_value"].notna()
        assert finite.any()
        assert (merged.loc[finite, f"{field}_legacy"] == merged.loc[finite, f"{field}_value"]).all()
    assert (legacy_firm["Bar_z"] >= 0.5).any()
    assert (value_firm["Bar_z"] >= 0.5).any()
    assert legacy.meta["max_firms"] != value_cost.meta["max_firms"]


def test_identity_keyed_crn_is_slot_order_invariant():
    class NondegenerateConfig(_TestConfig):
        RHO_X = 0.2
        SIGMA_X = 0.1
        RHO_Z = 0.5
        SIGMA_Z = 0.2

    sim = SimulateTS(
        models={"policy_value": _SimulationPolicy()}, config=NondegenerateConfig,
        n_paths=2, group_size=3, horizon=1, branch_num=1,
        enable_entry=False, enable_exit=True, device=torch.device("cpu"),
        transition_rng_mode="stable_firm_identity", common_transition_seed=91,
    )
    from data.simulate_ts_parallel import _initialize_batched_state
    state = _initialize_batched_state(sim, 5)
    state["hatc_cal"] = torch.zeros(2)
    state["lnk_cal"] = torch.zeros(2)
    original = _expand_branches_batched(sim, state, child_t=1)[0]
    permutation = torch.tensor([2, 0, 1, 3, 4])
    reordered = {
        key: value[:, permutation].clone()
        if torch.is_tensor(value) and value.ndim == 2 and value.shape[1] == 5
        else value.clone() if torch.is_tensor(value) else value
        for key, value in state.items()
    }
    permuted = _expand_branches_batched(sim, reordered, child_t=1)[0]
    for path in range(2):
        for firm_id in range(3):
            lhs = original["firm_id"][path] == firm_id
            rhs = permuted["firm_id"][path] == firm_id
            for field in ("transition_eps_z", "transition_u_eta", "transition_u_i"):
                assert original[field][path][lhs].item() == pytest.approx(
                    permuted[field][path][rhs].item()
                )


def test_smoke_main_keeps_namespace_paths_and_runs_all_arms(tmp_path):
    from experiments import run_entry_mechanism_smoke as smoke

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"mock")
    output_dir = tmp_path / "out"
    args = SimpleNamespace(
        checkpoint=checkpoint, output_dir=output_dir, device="cpu", seed=7,
        n_paths=1, group_size=2, horizon=1, branch_num=1,
        entry_capital_ratio=0.0, entry_size_ratio=0.25, entry_cost_max=0.1,
        entry_dummy_i=0.0, entry_rng_seed=4, entry_inference_chunk_size=32,
        simulation_bp_action_source="head",
    )
    loaded = SimpleNamespace(
        models={"policy_value": _SimulationPolicy()},
        economic_config=SimpleNamespace(to_dict=lambda: {
            name: getattr(_TestConfig, name)
            for name in (
                "RHO_X", "SIGMA_X", "XBAR", "RHO_Z", "SIGMA_Z", "ZBAR",
                "ZETA", "I_THRESHOLD", "G", "DELTA", "PHI",
            )
        }),
        hyperparams=SimpleNamespace(
            simulation_bp_action_source="head", pv_bp_head_training_enabled=True,
        ),
        metadata={"hyperparameter_recorded_fields": [
            "simulation_bp_action_source", "pv_bp_head_training_enabled"
        ]},
    )
    with patch.object(smoke, "parse_args", return_value=args), patch.object(
        smoke, "load_analysis_checkpoint", return_value=loaded
    ):
        report = smoke.main()
    assert isinstance(args.checkpoint, Path)
    assert isinstance(args.output_dir, Path)
    assert set(report["arms"]) == {
        "A_legacy_entry_legacy_aggregation",
        "B_legacy_entry_raw_aggregation",
        "C_value_cost_entry_raw_aggregation",
    }
    assert all(arm["status"] == "completed" for arm in report["arms"].values())
    assert (output_dir / "summary.json").is_file()


def test_smoke_bp_source_resolution_requires_checkpoint_provenance_and_honors_override():
    from experiments.run_entry_mechanism_smoke import resolve_simulation_bp_source

    loaded = SimpleNamespace(
        hyperparams=SimpleNamespace(
            simulation_bp_action_source="grid", pv_bp_head_training_enabled=False,
        ),
        metadata={"hyperparameter_recorded_fields": [
            "simulation_bp_action_source", "pv_bp_head_training_enabled"
        ]},
    )
    resolved = resolve_simulation_bp_source(loaded, "checkpoint")
    assert resolved["resolved_source"] == "grid"
    explicit = resolve_simulation_bp_source(loaded, "head")
    assert explicit["resolved_source"] == "head"
    assert explicit["warning"]
    loaded.metadata["hyperparameter_recorded_fields"] = []
    with pytest.raises(ValueError, match="does not record"):
        resolve_simulation_bp_source(loaded, "checkpoint")

    loaded.metadata["hyperparameter_recorded_fields"] = [
        "simulation_bp_action_source", "pv_bp_head_training_enabled"
    ]
    loaded.hyperparams.simulation_bp_action_source = "head"
    with pytest.raises(ValueError, match="head training disabled"):
        resolve_simulation_bp_source(loaded, "checkpoint")


def test_smoke_all_economic_failures_write_report_and_return_failure(tmp_path):
    from experiments import run_entry_mechanism_smoke as smoke

    class AlwaysInfeasibleSimulation:
        def __init__(self, **kwargs):
            pass

        def simulate(self):
            raise EntryEconomicInfeasibilityError("deliberate infeasible node")

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"mock")
    output_dir = tmp_path / "out"
    args = SimpleNamespace(
        checkpoint=checkpoint, output_dir=output_dir, device="cpu", seed=7,
        n_paths=1, group_size=2, horizon=1, branch_num=1,
        entry_capital_ratio=0.0, entry_size_ratio=0.25, entry_cost_max=0.1,
        entry_dummy_i=0.0, entry_rng_seed=4, entry_inference_chunk_size=32,
        simulation_bp_action_source="head",
    )
    loaded = SimpleNamespace(
        models={"policy_value": _SimulationPolicy()},
        economic_config=SimpleNamespace(to_dict=lambda: {
            name: getattr(_TestConfig, name)
            for name in (
                "RHO_X", "SIGMA_X", "XBAR", "RHO_Z", "SIGMA_Z", "ZBAR",
                "ZETA", "I_THRESHOLD", "G", "DELTA", "PHI",
            )
        }),
        hyperparams=SimpleNamespace(
            simulation_bp_action_source="head", pv_bp_head_training_enabled=True,
        ),
        metadata={"hyperparameter_recorded_fields": [
            "simulation_bp_action_source", "pv_bp_head_training_enabled"
        ]},
    )
    with patch.object(smoke, "parse_args", return_value=args), patch.object(
        smoke, "load_analysis_checkpoint", return_value=loaded
    ), patch.object(smoke, "SimulateTS", AlwaysInfeasibleSimulation):
        with pytest.raises(RuntimeError, match="all entry smoke arms failed"):
            smoke.main()
    report = __import__("json").loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert report["model_hash_invariant"] is True
    assert len(report["arms"]) == 3
    assert all(
        arm["status"] == "economic_infeasible"
        for arm in report["arms"].values()
    )


def test_smoke_grid_source_constructs_and_calls_grid_resolver(tmp_path):
    from experiments import run_entry_mechanism_smoke as smoke

    calls = {"constructed": 0, "called": 0}

    class SpyGridPolicy:
        def __init__(self, **kwargs):
            calls["constructed"] += 1

        def __call__(self, **context):
            calls["called"] += 1
            return torch.full_like(context["bp_head"], 0.55)

        def verify_immutable(self):
            return None

        def instrumentation(self):
            return {"simulation_bp_source": "grid", "calls": calls["called"]}

    economic_values = {
        name: getattr(_TestConfig, name, value)
        for name, value in {
            "RHO_X": 0.0, "SIGMA_X": 0.0, "XBAR": 0.0,
            "RHO_Z": 0.0, "SIGMA_Z": 0.0, "ZBAR": 0.0,
            "ZETA": 0.25, "I_THRESHOLD": 0.5, "G": 1.14,
            "DELTA": 0.0, "PHI": 1.0, "TAU": 0.2,
            "KAPPA_B": 0.1, "KAPPA_E": 0.1, "AIO_WEIGHT": 1.0,
            "ALPHA_Z": 1.0, "BETA_Z": 1.0, "Z0": 0.0,
        }.items()
    }
    economic = SimpleNamespace(**economic_values, to_dict=lambda: economic_values)
    loaded = SimpleNamespace(
        models={
            "policy_value": _SimulationPolicy(),
            "firm_target": _SimulationPolicy(),
            "sdf_fc1": _RecordingSDF(),
        },
        economic_config=economic,
        hyperparams=SimpleNamespace(
            simulation_bp_action_source="grid",
            pv_bp_head_training_enabled=False,
            simulation_bp_grid_n_child_shocks=2,
            simulation_bp_grid_shock_seed=17,
        ),
        metadata={"hyperparameter_recorded_fields": [
            "simulation_bp_action_source", "pv_bp_head_training_enabled"
        ]},
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"mock")
    with patch.object(smoke, "load_analysis_checkpoint", return_value=loaded), patch.object(
        smoke, "GridBPSimulationPolicy", SpyGridPolicy
    ):
        report = smoke.main([
            "--checkpoint", str(checkpoint),
            "--output-dir", str(tmp_path / "out"),
            "--simulation-bp-action-source", "checkpoint",
            "--n-paths", "1", "--group-size", "2", "--horizon", "1",
            "--branch-num", "1", "--entry-capital-ratio", "0",
        ])
    assert calls["constructed"] == 3
    assert calls["called"] >= 3
    assert report["bp_action_resolution"]["resolved_source"] == "grid"
    assert all(
        arm["bp_grid_target_model"] == "firm_target"
        for arm in report["arms"].values()
    )
