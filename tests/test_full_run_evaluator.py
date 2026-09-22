from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from analysis.economic_config import AnalysisEconomicConfig
from evaluation.bellman_diagnostics import evaluate_bellman_residuals
from analysis.convergence_transition import ConvergenceShockBank
from evaluation.bp_diagnostics import (
    FrozenTransitionData,
    _shock_bank_hash,
    build_frozen_transition_data,
    slice_frozen_transition_data,
)
from evaluation.full_run_diagnostics import (
    build_config_invariant_snapshot,
    compare_config_invariant_snapshots,
    conditional_residual_summary,
    evaluate_fc1_checkpoint,
    evaluate_sdf_heldout,
    evaluate_sdf_heldout_multi_k,
    model_state_hash,
    namespace_sdf_summary,
    parse_sdf_validation_log_blocks,
    parse_training_log,
    select_primary_sdf_validation_blocks,
    stable_config_hash,
    summarize_episode_statuses,
)
from evaluation.grids import FrozenFirmGrid, ReferenceFirmState, build_frozen_grid
from experiments.evaluate_full_run import (
    HEADLINE_COLUMNS,
    _discover_checkpoints,
    _prepare_output_directory,
    _sample_parent_tensors,
    aggregate_episode_peak_memory,
    select_episode_child_counts,
)
import experiments.evaluate_full_run as full_run_module


def _reference() -> ReferenceFirmState:
    return ReferenceFirmState(
        eta=1.0, i_low=0.1, i_mid=0.2, i_high=0.3, x=-2.0,
        hatcf=-2.1, lnkf=4.0, hatc_cal=-2.0, lnk_cal=4.1,
        n_parent_rows=2, source="fixture", macro_source="fixture",
    )


def _transition(grid, *, m=1.0, raw_m=None) -> FrozenTransitionData:
    children = [grid.base_states.clone(), grid.base_states.clone()]
    used_values = [torch.full((len(grid.base_states), 1), m) for _ in children]
    raw_values = [
        torch.full((len(grid.base_states), 1), raw_m if raw_m is not None else m)
        for _ in children
    ]
    return FrozenTransitionData(
        children=children, m_raw_list=raw_values, m_used_list=used_values,
        branch_weights=torch.full((len(grid.base_states), 2), 0.5), metadata={},
    )


class QDiagnosticModel(torch.nn.Module):
    def __init__(self, q_value: float):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(q_value))
        self.value_scale_mode = "none"
        self.value_scale_log_max = 20.0

    def forward(self, states):
        q = torch.ones_like(states[:, :1]) * self.anchor
        zeros = torch.zeros_like(q)
        return SimpleNamespace(
            Q=q, bp0=zeros, bpI=zeros, P0=zeros, PI=zeros, P=zeros,
            bar_i=zeros, bar_z=zeros,
        )

    def equity_value_scale(self, states):
        return torch.ones_like(states[:, :1])


def test_formal_q_residual_zero_and_known_bias():
    state = torch.tensor([[0.0, 0.0, 1.0, 0.2, -2.0, -2.1, 4.0]])
    grid = FrozenFirmGrid(
        b_values=np.array([0.0]), z_values=np.array([0.0]),
        mesh_b=np.array([[0.0]]), mesh_z=np.array([[0.0]]), base_states=state,
    )
    economic = AnalysisEconomicConfig.from_current_config()
    zero_surfaces, _ = evaluate_bellman_residuals(
        QDiagnosticModel(0.0), grid, _transition(grid), economic
    )
    np.testing.assert_allclose(zero_surfaces["RQ_signed"], 0.0)

    biased_surfaces, _ = evaluate_bellman_residuals(
        QDiagnosticModel(0.25), grid, _transition(grid), economic
    )
    # With b=0 and no default, target is M*Qsp=0.25 and parent Q=0.25.
    np.testing.assert_allclose(biased_surfaces["RQ_signed"], 0.0)
    # Force a known target-parent bias through M.
    biased_surfaces, _ = evaluate_bellman_residuals(
        QDiagnosticModel(0.25), grid, _transition(grid, m=0.5), economic
    )
    np.testing.assert_allclose(biased_surfaces["RQ_signed"], -0.125)
    np.testing.assert_allclose(biased_surfaces["abs_RQ"], 0.125)


def test_raw_m_and_train_m_residuals_are_distinct_and_aliases_use_train_m():
    state = torch.tensor([[0.0, 0.0, 1.0, 0.2, -2.0, -2.1, 4.0]])
    grid = FrozenFirmGrid(
        b_values=np.array([0.0]), z_values=np.array([0.0]),
        mesh_b=np.array([[0.0]]), mesh_z=np.array([[0.0]]), base_states=state,
    )
    surfaces, summary = evaluate_bellman_residuals(
        QDiagnosticModel(0.25), grid, _transition(grid, m=0.5, raw_m=1.0),
        AnalysisEconomicConfig.from_current_config(),
    )
    np.testing.assert_allclose(surfaces["RQ_trainM_signed"], -0.125)
    np.testing.assert_allclose(surfaces["RQ_rawM_signed"], 0.0)
    np.testing.assert_allclose(surfaces["RQ_signed"], surfaces["RQ_trainM_signed"])
    assert summary["q_residual_abs_mean"] == summary["q_trainM_residual_abs_mean"]


def test_conditional_moment_summary_matches_manual_u_statistic():
    residuals = torch.tensor([[1.0, -1.0], [2.0, 4.0]])
    result = conditional_residual_summary(residuals, prefix="x")
    expected_u = np.mean([-1.0, 8.0])
    assert result["x_u_stat"] == pytest.approx(expected_u)
    assert result["x_cm_mse"] == pytest.approx((0.0 ** 2 + 3.0 ** 2) / 2.0)
    assert result["x_conditional_p95_abs"] == pytest.approx(np.quantile([0.0, 3.0], 0.95))


class StableSDF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.sdf_model = SimpleNamespace(gamma=2.0, kappa=-1.0, sigma=1.0, beta=0.98)

    def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical=True):
        batch, children, _ = x_curr.shape
        w_parent = torch.full((batch, 1), 10.0, device=x_curr.device)
        w_children = torch.full((batch, children, 1), 10.0, device=x_curr.device)
        hatc = hatcf_prev.unsqueeze(1).expand(batch, children, 1)
        lnk = lnkf_prev.unsqueeze(1).expand(batch, children, 1)
        m = torch.full_like(w_children, 0.98)
        return w_parent, w_children, m, hatc, lnk


class CountingStableSDF(StableSDF):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward_step(self, *args, **kwargs):
        self.forward_calls += 1
        return super().forward_step(*args, **kwargs)


def test_sdf_heldout_is_nested_and_read_only():
    model = StableSDF()
    parents = torch.tensor([
        [0.2, -0.2, 1.0, 0.2, -2.0, -2.1, 4.0],
        [0.4, 0.2, 0.0, 0.3, -1.9, -2.0, 4.1],
    ])
    hatc = torch.full((2, 1), -2.0)
    lnk = torch.full((2, 1), 4.1)
    before = model_state_hash(model)
    result2, meta2 = evaluate_sdf_heldout(
        model, parents, hatc_cal=hatc, lnk_cal=lnk,
        economic_config=AnalysisEconomicConfig.from_current_config(),
        n_children=2, seed=7, shock_bank_max_children=4,
    )
    result4, meta4 = evaluate_sdf_heldout(
        model, parents, hatc_cal=hatc, lnk_cal=lnk,
        economic_config=AnalysisEconomicConfig.from_current_config(),
        n_children=4, seed=7, shock_bank_max_children=4,
    )
    assert result2["sdf_normalized_n_children"] == 2
    assert result4["sdf_normalized_n_children"] == 4
    assert meta2["shock_bank_max_children"] == meta4["shock_bank_max_children"] == 4
    assert meta2["shock_bank_sha256"] == meta4["shock_bank_sha256"]
    assert model_state_hash(model) == before


def test_sdf_multi_k_matches_single_k_and_uses_one_forward():
    parents = torch.tensor([
        [0.2, -0.2, 1.0, 0.2, -2.0, -2.1, 4.0],
        [0.4, 0.2, 0.0, 0.3, -1.9, -2.0, 4.1],
    ])
    hatc = torch.full((2, 1), -2.0)
    lnk = torch.full((2, 1), 4.1)
    model = CountingStableSDF()
    multi, metadata = evaluate_sdf_heldout_multi_k(
        model, parents, hatc_cal=hatc, lnk_cal=lnk,
        economic_config=AnalysisEconomicConfig.from_current_config(),
        child_counts=[2, 4, 8], seed=7, shock_bank_max_children=8,
    )
    assert model.forward_calls == 1
    assert metadata["sdf_forward_calls"] == 1
    for child_count in (2, 4, 8):
        single, _ = evaluate_sdf_heldout(
            StableSDF(), parents, hatc_cal=hatc, lnk_cal=lnk,
            economic_config=AnalysisEconomicConfig.from_current_config(),
            n_children=child_count, seed=7, shock_bank_max_children=8,
        )
        assert multi[child_count].keys() == single.keys()
        for key in single:
            assert multi[child_count][key] == pytest.approx(single[key], nan_ok=True)


class NonfiniteSDF(StableSDF):
    def forward_step(self, *args, **kwargs):
        result = list(super().forward_step(*args, **kwargs))
        result[1][0, 0, 0] = float("nan")
        return tuple(result)


def test_sdf_valid_parent_ratio_reports_filtered_nonfinite_parent():
    parents = torch.tensor([
        [0.2, -0.2, 1.0, 0.2, -2.0, -2.1, 4.0],
        [0.4, 0.2, 0.0, 0.3, -1.9, -2.0, 4.1],
    ])
    summary, _ = evaluate_sdf_heldout(
        NonfiniteSDF(), parents, hatc_cal=torch.full((2, 1), -2.0),
        lnk_cal=torch.full((2, 1), 4.1),
        economic_config=AnalysisEconomicConfig.from_current_config(),
        n_children=2, seed=7,
    )
    assert summary["sdf_n_parents_requested"] == 2
    assert summary["sdf_n_parents_valid"] == 1
    assert summary["sdf_valid_parent_ratio"] == pytest.approx(0.5)
    scoped = namespace_sdf_summary(summary, scope="common")
    assert scoped["sdf_common_valid_parent_ratio"] == pytest.approx(0.5)


class ExactFC1(torch.nn.Module):
    def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical=True):
        hatc = hatcf_prev + (x_curr - x_prev)
        lnk = lnkf_prev + 0.5 * (x_curr - x_prev)
        ones = torch.ones_like(hatc)
        return ones, ones, ones, hatc, lnk


def test_fc1_skill_and_timing_alignment():
    rows = []
    for path in range(2):
        for t in range(6):
            x = -2.0 + 0.1 * t
            rows.append({
                "path": path, "t": t, "x": x,
                "Hatc": -1.0 + 0.1 * t, "LnK": 4.0 + 0.05 * t,
            })
    summary, timing, rollout = evaluate_fc1_checkpoint(
        ExactFC1(), pd.DataFrame(rows), device=torch.device("cpu"),
        rollout_horizons=(1, 5), shifts=(-1, 0, 1),
    )
    assert summary["fc1_hatc_rmse"] == pytest.approx(0.0, abs=1e-6)
    assert summary["fc1_lnk_rmse"] == pytest.approx(0.0, abs=1e-6)
    assert summary["fc1_hatc_best_timing_shift"] == 0
    assert set(timing["shift"]) == {-1, 0, 1}
    assert set(rollout["horizon"]) == {1, 5}


def test_training_log_parser_handles_episode_and_key_value_variants(tmp_path):
    log = tmp_path / "train.out"
    log.write_text(
        "Episode 0 start\nloss=1.25 sdf/aio_t=-3.0\n"
        "Episode 1 (modeb)\naccepted_optimizer_steps_total = 500\n",
        encoding="utf-8",
    )
    frame, metadata = parse_training_log(log)
    assert list(frame["episode"]) == [0, 1]
    assert frame.loc[0, "loss"] == pytest.approx(1.25)
    assert frame.loc[0, "sdf_aio_t"] == pytest.approx(-3.0)
    assert frame.loc[1, "accepted_optimizer_steps_total"] == 500
    assert metadata["source"] == "explicit_training_log"


def test_training_log_parser_handles_repeated_scientific_boolean_and_nonfinite(tmp_path):
    log = tmp_path / "train.out"
    log.write_text(
        "Episode 3 start\n"
        "metric=1e-3 safe_to_continue=True missing_only_here=4\n"
        "metric=2.5E+1 bad_nan=nan bad_inf=-inf\n"
        "Episode 4 start\nother=False\n",
        encoding="utf-8",
    )
    frame, metadata = parse_training_log(log)
    ep3 = frame.loc[frame["episode"] == 3].iloc[0]
    ep4 = frame.loc[frame["episode"] == 4].iloc[0]
    assert ep3["metric"] == pytest.approx(25.0)
    assert bool(ep3["safe_to_continue"]) is True
    assert np.isnan(ep3["bad_nan"])
    assert np.isnan(ep3["bad_inf"])
    assert np.isnan(ep4["missing_only_here"])
    assert bool(ep4["other"]) is False
    assert metadata["repeated_key_occurrences"] == 1
    assert metadata["nonfinite_values_recorded_as_nan"] == 2



def test_sdf_validation_parser_preserves_arrow_values_and_repeated_blocks(tmp_path):
    log = tmp_path / "train.out"
    log.write_text(
        "Episode 2 SDF_TRUE_ONLY\n"
        "SDF validation | safe_to_continue=True stage_progress=True converged=False "
        "normalized_mean=1.2e-2->4.0e-3 normalized_t=30->12 "
        "max_constraint_violation=3e-2->1e-2 result=CONTINUE\n"
        "SDF validation | safe_to_continue=True stage_progress=False converged=False "
        "normalized_mean=4e-3->3e-3 normalized_t=12->10 "
        "max_constraint_violation=1e-2->9e-3 result=RETRY\n",
        encoding="utf-8",
    )
    blocks, metadata = parse_sdf_validation_log_blocks(log)
    assert len(blocks) == 2
    assert blocks.iloc[0]["normalized_mean_before"] == pytest.approx(0.012)
    assert blocks.iloc[0]["normalized_mean_after"] == pytest.approx(0.004)
    assert blocks.iloc[0]["constraint_after"] == pytest.approx(0.01)
    assert list(blocks["occurrence"]) == [1, 2]
    primary, selection = select_primary_sdf_validation_blocks(blocks)
    assert primary.empty
    assert selection["ambiguous_episodes"] == [2]
    assert metadata["n_blocks"] == 2


def test_sdf_validation_parser_prefers_explicit_stage_and_rejects_duplicates(tmp_path):
    log = tmp_path / "train.out"
    log.write_text(
        "Episode 3 SDF recursive context\n"
        "SDF validation | stage=sdf_true_only safe_to_continue=True "
        "stage_progress=True converged=False normalized_mean=1->0.5 "
        "normalized_t=4->3 max_constraint_violation=0.2->0.1 result=CONTINUE\n"
        "Episode 4\n"
        "SDF validation | stage=sdf_true_only safe_to_continue=True "
        "stage_progress=True converged=False normalized_mean=1->0.5 "
        "normalized_t=4->3 max_constraint_violation=0.2->0.1 result=CONTINUE\n"
        "SDF validation | stage=sdf_true_only safe_to_continue=True "
        "stage_progress=True converged=True normalized_mean=0.5->0.1 "
        "normalized_t=3->1 max_constraint_violation=0.1->0 result=PASS\n",
        encoding="utf-8",
    )
    blocks, _ = parse_sdf_validation_log_blocks(log)
    assert list(blocks["stage_source"]) == ["explicit", "explicit", "explicit"]
    primary, selection = select_primary_sdf_validation_blocks(blocks)
    assert list(primary["episode"]) == [3]
    assert primary.iloc[0]["sdf_validation_stage_source"] == "explicit"
    assert selection["ambiguous_episodes"] == [4]


def test_config_invariant_hash_and_comparison_are_stable():
    hp = SimpleNamespace(
        sdf_wealth_loss_mode="signed_aio", sdf_normalized_logr_clip=20.0,
        pv_use_clipped_m=True, pv_m_clamp_min=0.7, pv_m_clamp_max=1.3,
        pv_bellman_normalize_by_value_scale=True,
        pv_exact_eta_integration_enabled=True,
    )
    loaded = SimpleNamespace(
        economic_config=AnalysisEconomicConfig.from_current_config(), hp=hp,
        hyperparams=hp,
        models={"policy_value": QDiagnosticModel(0.0), "sdf_fc1": StableSDF()},
    )
    snapshot, config_hash = build_config_invariant_snapshot(loaded)
    assert stable_config_hash(snapshot) == config_hash
    reordered = {key: snapshot[key] for key in reversed(snapshot)}
    assert stable_config_hash(reordered) == config_hash
    changed = {**snapshot, "policy_value": {**snapshot["policy_value"], "pv_m_clamp_max": 1.4}}
    frame, metadata = compare_config_invariant_snapshots([
        (0, snapshot, config_hash), (1, reordered, stable_config_hash(reordered)),
        (2, changed, stable_config_hash(changed)),
    ])
    assert metadata["all_comparable"] is False
    assert metadata["mismatched_episodes"] == [2]
    assert frame.loc[frame["episode"] == 2, "diff_fields"].iloc[0] == "policy_value.pv_m_clamp_max"


def test_status_summary_distinguishes_partial_error_and_missing():
    summary = summarize_episode_statuses(pd.DataFrame({
        "status": ["ok", "partial", "error", "missing"]
    }))
    assert summary == {
        "n_requested": 4, "n_ok": 1, "n_partial": 1, "n_error": 1,
        "n_missing": 1, "n_not_fully_ok": 3, "n_error_or_missing": 2,
    }


def test_full_run_schema_has_no_fc2_and_output_overwrite_is_explicit(tmp_path):
    assert not any("fc2" in name.lower() for name in HEADLINE_COLUMNS)
    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "stale.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-empty"):
        _prepare_output_directory(output, overwrite=False)
    _prepare_output_directory(output, overwrite=True)
    assert list(output.iterdir()) == []


def test_common_parent_bank_is_deterministic_and_ondist_bank_can_differ(tmp_path):
    def write_firm(path, offset):
        pd.DataFrame([
            {
                "path": 0, "t": index, "branch": -1, "b": 0.1 * index + offset,
                "z": -0.2 + index, "ETA": index % 2, "i": 0.1,
                "x": -2.0, "Hatcf": -2.1, "LnKF": 4.0,
                "hatc_cal": -2.0, "lnk_cal": 4.1,
            }
            for index in range(4)
        ]).to_pickle(path)

    common_path = tmp_path / "common.pkl"
    ondist_path = tmp_path / "ondist.pkl"
    write_firm(common_path, 0.0)
    write_firm(ondist_path, 0.3)
    common_a = _sample_parent_tensors(
        common_path, None, device=torch.device("cpu"), max_parents=3
    )[-1]
    common_b = _sample_parent_tensors(
        common_path, None, device=torch.device("cpu"), max_parents=3
    )[-1]
    ondist = _sample_parent_tensors(
        ondist_path, None, device=torch.device("cpu"), max_parents=3
    )[-1]
    assert common_a["parent_bank_sha256"] == common_b["parent_bank_sha256"]
    assert common_a["selected_row_indices"] == common_b["selected_row_indices"]
    assert common_a["parent_bank_sha256"] != ondist["parent_bank_sha256"]


def test_actual_shock_bank_hash_uses_tensor_contents():
    bank = ConvergenceShockBank.create(
        3, 4, seed=19, device=torch.device("cpu"), dtype=torch.float32
    )
    same = ConvergenceShockBank.create(
        3, 4, seed=19, device=torch.device("cpu"), dtype=torch.float32
    )
    different = ConvergenceShockBank.create(
        3, 4, seed=20, device=torch.device("cpu"), dtype=torch.float32
    )
    assert _shock_bank_hash(bank) == _shock_bank_hash(same)
    assert _shock_bank_hash(bank) != _shock_bank_hash(different)
    parents = torch.tensor([
        [0.2, -0.2, 1.0, 0.2, -2.0, -2.1, 4.0],
        [0.4, 0.2, 0.0, 0.3, -1.9, -2.0, 4.1],
    ])
    transition = build_frozen_transition_data(
        StableSDF(), parents, _reference(),
        SimpleNamespace(
            pv_use_clipped_m=True, pv_m_clamp_min=0.7, pv_m_clamp_max=1.3,
            pv_exact_eta_integration_enabled=True,
        ),
        AnalysisEconomicConfig.from_current_config(), n_child_shocks=2,
        shock_seed=19, shock_bank_max_child_shocks=4,
    )
    actual_builder_bank = ConvergenceShockBank.create(
        1, 4, seed=19, device=torch.device("cpu"), dtype=parents.dtype
    )
    assert transition.metadata["shock_bank_sha256"] == _shock_bank_hash(actual_builder_bank)
    prefix = slice_frozen_transition_data(transition, n_continuous_children=2)
    assert prefix.children_tensor.shape == (2, 4, 7)
    assert prefix.m_raw_tensor.shape == (2, 4, 1)
    torch.testing.assert_close(prefix.branch_weights.sum(dim=1), torch.ones(2))
    eta_mass = (prefix.branch_weights * prefix.children_tensor[..., 2]).sum(dim=1)
    torch.testing.assert_close(
        eta_mass, torch.full_like(eta_mass, AnalysisEconomicConfig.from_current_config().ZETA)
    )
    torch.testing.assert_close(prefix.children_tensor[:, 0::2, 2], torch.zeros(2, 2))
    torch.testing.assert_close(prefix.children_tensor[:, 1::2, 2], torch.ones(2, 2))
    torch.testing.assert_close(
        prefix.children_tensor[:, 0::2][:, :, [0, 1, 3, 4, 5, 6]],
        prefix.children_tensor[:, 1::2][:, :, [0, 1, 3, 4, 5, 6]],
    )
    assert prefix.metadata["source_max_J"] == 4
    assert prefix.metadata["requested_J"] == 2
    assert prefix.metadata["max_bank_sha256"] == _shock_bank_hash(actual_builder_bank)
    expected_prefix_bank = ConvergenceShockBank(
        eps_x=actual_builder_bank.eps_x[:, :2],
        eps_z=actual_builder_bank.eps_z[:, :2],
        u_eta=actual_builder_bank.u_eta[:, :2],
        u_i=actual_builder_bank.u_i[:, :2],
        seed=actual_builder_bank.seed,
    )
    assert prefix.metadata["prefix_sha256"] == _shock_bank_hash(expected_prefix_bank)
    assert prefix.metadata["prefix_sha256"] != prefix.metadata["max_bank_sha256"]


def test_robustness_scope_and_episode_peak_aggregation():
    representatives = {0, 4, 8}
    for episode in range(9):
        expected = [32, 64, 128] if episode in representatives else [64]
        assert select_episode_child_counts(
            episode=episode,
            representative_episodes=representatives,
            primary_child_shocks=64,
            robustness_child_shocks=[32, 64, 128],
            robustness_scope="representative",
        ) == expected
        assert select_episode_child_counts(
            episode=episode,
            representative_episodes=representatives,
            primary_child_shocks=64,
            robustness_child_shocks=[32, 64, 128],
            robustness_scope="all",
        ) == [32, 64, 128]
        assert select_episode_child_counts(
            episode=episode,
            representative_episodes=representatives,
            primary_child_shocks=64,
            robustness_child_shocks=[32, 64, 128],
            robustness_scope="none",
        ) == [64]
    assert aggregate_episode_peak_memory([100.0, 250.0, 180.0]) == 250.0
    assert aggregate_episode_peak_memory([None, float("nan")]) is None


def test_checkpoint_without_episode_firm_still_runs_structural_and_common_sdf(
    tmp_path, monkeypatch,
):
    run_root = tmp_path / "run"
    checkpoint_dir = run_root / "checkpoints_analysis"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "ep0_combined.pt"
    checkpoint.write_bytes(b"checkpoint")
    reference = tmp_path / "reference.pkl"
    pd.DataFrame([
        {
            "path": 0, "t": index, "branch": -1, "b": 0.1 * index,
            "z": -0.2 + index, "ETA": index % 2, "i": 0.1,
            "x": -2.0, "Hatcf": -2.1, "LnKF": 4.0,
            "hatc_cal": -2.0, "lnk_cal": 4.1,
        }
        for index in range(3)
    ]).to_pickle(reference)
    output = tmp_path / "evaluation"

    load_calls = []

    def fake_matrix(args):
        assert args.loaded_checkpoint is loaded
        assert args.defer_model_state_hash_to_outer is True
        assert args.manage_cuda_peak_stats is False
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for eta in (0, 1):
            eta_dir = args.output_dir / f"eta{eta}"
            eta_dir.mkdir()
            (eta_dir / "metadata.json").write_text("{}", encoding="utf-8")
        return pd.DataFrame([{
            "eta_parent": 1.0, "n_child_shocks": 2,
            "p0_residual_abs_mean": 0.1, "pi_residual_abs_mean": 0.2,
            "q_residual_abs_mean": 0.3,
        }]), {
            "reference_transition_bank": {"shock_bank_sha256": "actual"},
            "timing": {"bellman_seconds": 0.1, "bp_seconds": 0.1, "investment_seconds": 0.1},
        }

    loaded = SimpleNamespace(
        models={"policy_value": QDiagnosticModel(0.25), "sdf_fc1": StableSDF()},
        metadata={"loaded_model_keys": ["policy_value", "sdf_fc1"]},
        hyperparams=SimpleNamespace(
            sdf_normalized_logr_clip=20.0, sdf_wealth_loss_mode="signed_aio",
            pv_use_clipped_m=True, pv_m_clamp_min=0.7, pv_m_clamp_max=1.3,
            pv_bellman_normalize_by_value_scale=False,
            pv_exact_eta_integration_enabled=True,
        ),
        economic_config=AnalysisEconomicConfig.from_current_config(),
    )
    monkeypatch.setattr(full_run_module, "evaluate_matrix", fake_matrix)
    def fake_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        return loaded

    monkeypatch.setattr(full_run_module, "load_analysis_checkpoint", fake_load)
    monkeypatch.setattr(full_run_module, "discover_episode_firm_data", lambda root: ({}, []))
    monkeypatch.setattr(full_run_module, "_git", lambda args: "")
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_full_run.py", "--run-root", str(run_root),
            "--reference-firm-data", str(reference), "--output-dir", str(output),
            "--device", "cpu", "--episodes", "0", "--n-child-shocks", "2",
            "--robustness-child-shocks", "2", "--max-sdf-parents", "2",
            "--b-points", "2", "--z-points", "2", "--i-points", "2",
        ],
    )
    full_run_module.main()
    headline = pd.read_csv(output / "headline_metrics.csv")
    assert headline.loc[0, "status"] == "partial"
    assert headline.loc[0, "p0_residual_abs_mean"] == pytest.approx(0.1)
    assert np.isfinite(headline.loc[0, "sdf_common_conditional_abs_mean"])
    assert not (output / "episodes" / "ep0" / "fc2").exists()
    run_summary = pd.read_json(output / "run_summary.json", typ="series")
    assert run_summary["n_partial"] == 1
    assert run_summary["n_error_or_missing"] == 0
    metadata = pd.read_json(output / "metadata.json", typ="series")
    assert bool(metadata["cross_episode_comparable"])
    assert len(load_calls) == 1
    timing = json.loads((output / "evaluation_timing.json").read_text(encoding="utf-8"))
    assert timing["checkpoint_load_count"] == 1
    assert (output / "evaluation_timing.json").is_file()
    assert (output / "tables" / "cross_episode_config_invariants.csv").is_file()


def test_checkpoint_discovery_preserves_missing_episode(tmp_path):
    directory = tmp_path / "checkpoints_analysis"
    directory.mkdir()
    (directory / "ep0_combined.pt").write_bytes(b"x")
    (directory / "ep2_combined.pt").write_bytes(b"x")
    assert sorted(_discover_checkpoints(tmp_path)) == [0, 2]
    requested = {0, 1, 2}
    assert sorted(requested - set(_discover_checkpoints(tmp_path))) == [1]
