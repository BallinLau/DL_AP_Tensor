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
    evaluate_bp_consistency,
    evaluate_bp_consistency_multi_j,
    slice_frozen_transition_data,
)
from config import HyperParams
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
from evaluation.convergence_metrics import regression_metrics
from evaluation.grids import FrozenFirmGrid, ReferenceFirmState, build_frozen_grid
from experiments.evaluate_full_run import (
    HEADLINE_COLUMNS,
    _discover_checkpoints,
    _prepare_output_directory,
    _sample_parent_tensors,
    aggregate_episode_peak_memory,
    canonical_bank_max_children,
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


def test_regression_metrics_rejects_duplicate_column_selection():
    frame = pd.DataFrame({"target": [1.0, 2.0, 3.0], "forecast": [1.0, 2.0, 3.0]})
    duplicated = pd.concat([frame, frame[["target"]]], axis=1)
    assert isinstance(duplicated["target"], pd.DataFrame)
    assert isinstance(duplicated["forecast"], pd.Series)
    with pytest.raises(ValueError, match="identically shaped"):
        regression_metrics(
            duplicated["target"], duplicated["forecast"],
            calculated_name="target", forecast_name="forecast",
        )


def test_fc1_checkpoint_handles_artifact_hatcf_lnkf_columns():
    """Regression: real macro frames already carry Hatcf/LnKF.

    The old evaluator renamed its own forecast columns to ``Hatcf``/``LnKF``,
    producing duplicate columns whose selection returned a DataFrame instead of a
    Series, which crashed the FC1 timing-alignment regression.
    """
    rows = []
    for path in range(2):
        for t in range(6):
            x = -2.0 + 0.1 * t
            rows.append({
                "path": path, "t": t, "x": x,
                "Hatc": -1.0 + 0.1 * t, "LnK": 4.0 + 0.05 * t,
                "Hatcf": -1.0 + 0.1 * t, "LnKF": 4.0 + 0.05 * t,
            })
    frame = pd.DataFrame(rows)
    assert frame.columns.duplicated().sum() == 0
    summary, timing, rollout = evaluate_fc1_checkpoint(
        ExactFC1(), frame, device=torch.device("cpu"),
        rollout_horizons=(1, 5), shifts=(-1, 0, 1),
    )
    assert summary["fc1_hatc_rmse"] == pytest.approx(0.0, abs=1e-6)
    assert summary["fc1_lnk_rmse"] == pytest.approx(0.0, abs=1e-6)
    assert summary["fc1_hatc_best_timing_shift"] == 0
    assert set(timing["variable"]) == {"hatc", "lnk"}
    assert len(timing) == 6
    assert set(rollout["horizon"]) == {1, 5}


class ShockSensitiveSDF(torch.nn.Module):
    """SDF stub whose held-out residuals genuinely depend on the shock draws."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.sdf_model = SimpleNamespace(gamma=2.0, kappa=-1.0, sigma=1.0, beta=0.98)

    def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical=True):
        batch, children, _ = x_curr.shape
        w_parent = 10.0 + x_prev
        w_children = 10.0 + x_curr
        hatc = hatcf_prev.unsqueeze(1).expand(batch, children, 1) + x_curr
        lnk = lnkf_prev.unsqueeze(1).expand(batch, children, 1) + 0.5 * x_curr
        m = torch.full((batch, children, 1), 0.98, device=x_curr.device)
        return w_parent, w_children, m, hatc, lnk


def test_canonical_bank_max_children_spans_primary_and_robustness():
    assert canonical_bank_max_children(64, [32, 64, 128]) == 128
    assert canonical_bank_max_children(64, [32, 64]) == 64
    assert canonical_bank_max_children(2, []) == 2


def test_transition_metadata_separates_canonical_bank_from_evaluated_prefix():
    """The transition bank max J and the evaluated J prefix must stay distinct."""
    reference = _reference()
    grid = build_frozen_grid(
        reference, b_min=0.2, b_max=0.4, b_points=2,
        z_min=-0.4, z_max=0.4, z_points=2, device=torch.device("cpu"),
    )
    hyperparams = HyperParams()
    economic = AnalysisEconomicConfig.from_current_config()
    transition_max = build_frozen_transition_data(
        StableSDF(), grid.base_states, reference, hyperparams, economic,
        n_child_shocks=8, shock_seed=12345, shock_bank_max_child_shocks=8,
    )
    assert transition_max.metadata["shock_bank_max_child_shocks"] == 8
    assert transition_max.metadata["source_max_J"] == 8

    # A canonical bank of 8 with only the J=2 prefix evaluated.
    sliced = slice_frozen_transition_data(transition_max, n_continuous_children=2)
    assert sliced.metadata["source_max_J"] == 8
    assert sliced.metadata["requested_J"] == 2
    assert sliced.metadata["shock_bank_max_child_shocks"] == 8
    assert sliced.metadata["nested_prefix_from_max_J"] is True
    assert sliced.metadata["max_bank_sha256"] == transition_max.metadata["shock_bank_sha256"]

    # The prefix hash must be the hash of the requested prefix, not of the full bank.
    prefix_bank = ConvergenceShockBank(
        eps_x=transition_max.continuous_shock_bank.eps_x[:, :2],
        eps_z=transition_max.continuous_shock_bank.eps_z[:, :2],
        u_eta=transition_max.continuous_shock_bank.u_eta[:, :2],
        u_i=transition_max.continuous_shock_bank.u_i[:, :2],
        seed=transition_max.continuous_shock_bank.seed,
    )
    assert sliced.metadata["prefix_sha256"] == _shock_bank_hash(prefix_bank)
    assert sliced.metadata["prefix_sha256"] != sliced.metadata["max_bank_sha256"]


def test_transition_children_depend_on_the_canonical_bank_not_the_evaluated_j():
    """Shrinking the bank to the episode's own max J silently breaks cross-episode CRN."""
    reference = _reference()
    grid = build_frozen_grid(
        reference, b_min=0.2, b_max=0.4, b_points=2,
        z_min=-0.4, z_max=0.4, z_points=2, device=torch.device("cpu"),
    )
    hyperparams = HyperParams()
    economic = AnalysisEconomicConfig.from_current_config()

    def build(bank_max: int):
        transition = build_frozen_transition_data(
            StableSDF(), grid.base_states, reference, hyperparams, economic,
            n_child_shocks=4, shock_seed=12345, shock_bank_max_child_shocks=bank_max,
        )
        return slice_frozen_transition_data(transition, n_continuous_children=4)

    canonical_a = build(8)
    canonical_b = build(8)
    shrunken = build(4)
    torch.testing.assert_close(
        canonical_a.stacked_children(), canonical_b.stacked_children(), rtol=0, atol=0
    )
    assert canonical_a.metadata["prefix_sha256"] == canonical_b.metadata["prefix_sha256"]
    assert canonical_a.metadata["source_max_J"] == 8
    # The per-episode bank is *not* equivalent: the four shock tensors are drawn
    # sequentially from one generator, so a smaller bank shifts every tensor
    # after the first.
    assert shrunken.metadata["source_max_J"] == 4
    assert shrunken.metadata["prefix_sha256"] != canonical_a.metadata["prefix_sha256"]
    assert not torch.equal(shrunken.stacked_children(), canonical_a.stacked_children())


def test_compact_transition_matches_jmax_build_then_slice():
    """A J=4 transition built from an 8-child canonical bank must equal an
    J=8 transition sliced to 4, element for element."""
    reference = _reference()
    grid = build_frozen_grid(
        reference, b_min=0.2, b_max=0.4, b_points=2,
        z_min=-0.4, z_max=0.4, z_points=2, device=torch.device("cpu"),
    )
    hyperparams = HyperParams()
    economic = AnalysisEconomicConfig.from_current_config()

    def build(n_child_shocks: int):
        return build_frozen_transition_data(
            StableSDF(), grid.base_states, reference, hyperparams, economic,
            n_child_shocks=n_child_shocks, shock_seed=12345,
            shock_bank_max_child_shocks=8,
        )

    compact = build(4)
    wide = build(8)
    sliced = slice_frozen_transition_data(wide, n_continuous_children=4)

    assert compact.metadata["continuous_child_count"] == 4
    assert compact.metadata["expanded_child_count"] == 8
    assert compact.metadata["source_max_J"] == 8
    assert compact.metadata["requested_J"] == 4
    assert wide.metadata["continuous_child_count"] == 8
    assert wide.metadata["expanded_child_count"] == 16

    torch.testing.assert_close(
        compact.stacked_children(), sliced.stacked_children(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        compact.stacked_m_raw(), sliced.stacked_m_raw(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        compact.stacked_m_used(), sliced.stacked_m_used(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        compact.branch_weights, sliced.branch_weights, rtol=1e-6, atol=1e-9
    )
    assert compact.metadata["prefix_sha256"] == sliced.metadata["prefix_sha256"]
    assert compact.metadata["max_bank_sha256"] == sliced.metadata["max_bank_sha256"]


def test_compact_ordinary_transition_keeps_canonical_crn_prefix():
    """Sizing the transition by the evaluated J must not break cross-episode CRN."""
    reference = _reference()
    grid = build_frozen_grid(
        reference, b_min=0.2, b_max=0.4, b_points=2,
        z_min=-0.4, z_max=0.4, z_points=2, device=torch.device("cpu"),
    )
    hyperparams = HyperParams()
    economic = AnalysisEconomicConfig.from_current_config()

    def build(n_child_shocks: int):
        return build_frozen_transition_data(
            StableSDF(), grid.base_states, reference, hyperparams, economic,
            n_child_shocks=n_child_shocks, shock_seed=12345,
            shock_bank_max_child_shocks=8,
        )

    # Ordinary episode: canonical bank 8, only J=4 evaluated.
    ordinary = slice_frozen_transition_data(build(4), n_continuous_children=4)
    # Representative episode: canonical bank 8, J in {2, 4, 8} evaluated.
    representative = slice_frozen_transition_data(build(8), n_continuous_children=4)
    assert ordinary.metadata["source_max_J"] == 8
    assert representative.metadata["source_max_J"] == 8
    assert ordinary.metadata["prefix_sha256"] == representative.metadata["prefix_sha256"]
    torch.testing.assert_close(
        ordinary.stacked_children(), representative.stacked_children(), rtol=0, atol=0
    )


def test_multi_j_bp_evaluator_accepts_compact_transition_equally(tmp_path):
    """The compact transition must give the same BP numbers with less work.

    Path A: a J=4 transition built from the 8-child canonical bank.
    Path B: the J=8 transition sliced down to J=4 inside the evaluator.
    """
    reference, grid, hyperparams, economic, transition_max = _bp_parity_setup()
    compact = build_frozen_transition_data(
        _BpToySDF(), grid.base_states, reference, hyperparams, economic,
        n_child_shocks=4, shock_seed=12345, shock_bank_max_child_shocks=8,
    )
    assert compact.metadata["expanded_child_count"] == 8
    assert compact.metadata["source_max_J"] == 8
    assert transition_max.metadata["expanded_child_count"] == 16

    def run(transition, name):
        results, stats = evaluate_bp_consistency_multi_j(
            _BpToyPolicy(), _BpToySDF(), grid, reference, hyperparams, economic,
            output_dir=tmp_path / name, j_values=[4], primary_j=4,
            transition_max=transition, shock_seed=12345,
            write_objective_slices=False,
        )
        return results[4], stats

    compact_entry, compact_stats = run(compact, "compact")
    wide_entry, wide_stats = run(transition_max, "wide")

    for key, expected in wide_entry["surfaces"].items():
        assert key in compact_entry["surfaces"], f"missing surface {key}"
        np.testing.assert_allclose(
            compact_entry["surfaces"][key], expected, rtol=1e-5, atol=1e-6,
            equal_nan=True,
        )
    for key, expected in wide_entry["summary"].items():
        assert key in compact_entry["summary"], f"missing summary {key}"
        actual = compact_entry["summary"][key]
        if isinstance(expected, float) and np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert actual == pytest.approx(expected, rel=1e-5, abs=1e-6)

    # The chunk planner must only ever see the evaluated children. The wide path
    # still forwards the full Jmax child block on the coarse grid (then slices the
    # objective), which is exactly the wasted work the compact transition removes.
    assert {plan["n_children"] for plan in compact_stats["bp_grid_chunk_plans"]} == {8}
    assert max(plan["n_children"] for plan in wide_stats["bp_grid_chunk_plans"]) == 16
    assert (
        compact_stats["bp_max_actual_expanded_states"]
        < wide_stats["bp_max_actual_expanded_states"]
    )
    assert (
        compact_stats["bp_child_equity_forward_calls"]
        <= wide_stats["bp_child_equity_forward_calls"]
    )


def test_cross_episode_crn_requires_one_canonical_shock_bank():
    """A J=64 episode must draw the same shocks as a J=128 representative episode."""
    parents = torch.tensor([
        [0.2, -0.2, 1.0, 0.2, -2.0, -2.1, 4.0],
        [0.4, 0.2, 0.0, 0.3, -1.9, -2.0, 4.1],
    ])
    hatc = torch.full((2, 1), -2.0)
    lnk = torch.full((2, 1), 4.1)
    economic = AnalysisEconomicConfig.from_current_config()
    canonical = canonical_bank_max_children(64, [32, 64, 128])
    assert canonical == 128
    shared = dict(
        hatc_cal=hatc, lnk_cal=lnk, economic_config=economic, seed=12345,
        shock_bank_max_children=canonical,
    )
    representative, rep_meta = evaluate_sdf_heldout_multi_k(
        ShockSensitiveSDF(), parents, child_counts=[32, 64, 128], **shared
    )
    plain, plain_meta = evaluate_sdf_heldout_multi_k(
        ShockSensitiveSDF(), parents, child_counts=[64], **shared
    )
    assert plain_meta["shock_bank_max_children"] == 128
    assert plain_meta["shock_bank_sha256"] == rep_meta["shock_bank_sha256"]
    for key, value in plain[64].items():
        assert value == pytest.approx(representative[64][key], nan_ok=True)
    # Shrinking the bank to the episode's own max child count shifts the per-parent
    # shock draws, so the shared cross-episode CRN prefix silently breaks.
    drifted, drifted_meta = evaluate_sdf_heldout_multi_k(
        ShockSensitiveSDF(), parents, child_counts=[64], hatc_cal=hatc, lnk_cal=lnk,
        economic_config=economic, seed=12345, shock_bank_max_children=64,
    )
    assert drifted_meta["shock_bank_sha256"] != rep_meta["shock_bank_sha256"]
    changed = [
        key for key in representative[64]
        if drifted[64][key] != representative[64][key]
    ]
    assert "sdf_normalized_conditional_mean_abs" in changed


def test_component_isolation_keeps_firm_and_sdf_when_fc1_raises(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    checkpoint_dir = run_root / "checkpoints_analysis"
    checkpoint_dir.mkdir(parents=True)
    for episode in (0, 1):
        (checkpoint_dir / f"ep{episode}_combined.pt").write_bytes(b"checkpoint")
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
    macro_path = tmp_path / "ep1_macro.pkl"
    macro_path.write_bytes(b"macro")
    output = tmp_path / "evaluation"

    def fake_matrix(args):
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
    monkeypatch.setattr(full_run_module, "load_analysis_checkpoint", lambda *a, **k: loaded)
    monkeypatch.setattr(full_run_module, "discover_episode_firm_data", lambda root: ({}, []))
    monkeypatch.setattr(full_run_module, "_git", lambda args: "")
    monkeypatch.setattr(full_run_module, "_discover_episode_macro", lambda root, episode: macro_path)
    monkeypatch.setattr(
        full_run_module, "read_dataframe",
        lambda path: pd.DataFrame({"path": [0], "t": [0], "x": [-2.0]}),
    )

    def failing_fc1(*args, **kwargs):
        raise RuntimeError("fc1 boom")

    monkeypatch.setattr(full_run_module, "evaluate_fc1_checkpoint", failing_fc1)
    drift_calls = {}

    def fake_drift(cases, eta, out, missing):
        drift_calls.setdefault("episodes", set()).update(case.episode for case in cases)
        return pd.DataFrame(), {}

    monkeypatch.setattr(full_run_module, "compute_function_drift", fake_drift)
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_full_run.py", "--run-root", str(run_root),
            "--reference-firm-data", str(reference), "--output-dir", str(output),
            "--device", "cpu", "--episodes", "0,1", "--n-child-shocks", "2",
            "--robustness-child-shocks", "2", "--max-sdf-parents", "2",
            "--b-points", "2", "--z-points", "2", "--i-points", "2",
        ],
    )
    full_run_module.main()
    headline = pd.read_csv(output / "headline_metrics.csv").set_index("episode")
    assert set(headline["status"]) == {"partial"}
    assert set(headline["firm_status"]) == {"ok"}
    assert set(headline["sdf_common_status"]) == {"ok"}
    assert set(headline["fc1_status"]) == {"error"}
    assert set(headline["sdf_ondist_status"]) == {"missing"}
    assert headline["p0_residual_abs_mean"].notna().all()
    assert headline["sdf_common_conditional_abs_mean"].notna().all()
    errors = pd.read_csv(output / "errors.csv")
    assert set(errors.loc[errors["stage"] == "fc1", "episode"]) == {0, 1}
    assert not any("firm_structural" in str(message) for message in errors["stage"])
    assert drift_calls["episodes"] == {0, 1}
    metadata = pd.read_json(output / "metadata.json", typ="series")
    assert metadata["common_shock_bank"]["canonical_shock_bank_max_children"] == 2
    episode_meta = pd.read_json(output / "episodes" / "ep1" / "metadata.json", typ="series")
    assert episode_meta["component_status"]["fc1"] == "error"
    assert episode_meta["component_status"]["firm_structural"] == "ok"
    assert not (output / "episodes" / "ep1" / "fc2").exists()


class _BpToySDF(torch.nn.Module):
    """Deterministic SDF stub whose child draws depend on the parent state."""

    def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical=True):
        batch, children, _ = x_curr.shape
        w_parent = 10.0 + x_prev
        w_children = 10.0 + x_curr
        m = 0.95 + 0.02 * torch.tanh(x_curr)
        hatc = hatcf_prev.unsqueeze(1).expand(batch, children, 1) + x_curr
        lnk = lnkf_prev.unsqueeze(1).expand(batch, children, 1) + 0.5 * x_curr
        return w_parent, w_children, m, hatc, lnk


class _BpToyPolicy(torch.nn.Module):
    """Toy policy/value model: every branch sees a different ``i`` column."""

    def forward(self, states):
        b = states[:, 0:1]
        z = states[:, 1:2]
        i = states[:, 3:4]
        phat = torch.sigmoid(1.5 * b - 0.5 * z + 0.8 * i)
        bp0 = (0.30 + 0.15 * b + 0.10 * i).clamp(0.02, 0.95)
        bp_i = (0.65 + 0.10 * b - 0.08 * i).clamp(0.02, 0.95)
        return SimpleNamespace(Phat=phat, bp0=bp0, bpI=bp_i)

    def _q_output(self, state):
        return (
            0.70
            + 0.05 * state[:, 0:1]
            + 0.60 * state[:, 3:4]
            + 0.02 * state[:, 4:5]
        )

    def forward_equity(self, state):
        b = state[:, 0:1]
        z = state[:, 1:2]
        i = state[:, 3:4]
        p = 1.20 + (0.8 + 4.0 * i) * b - 12.0 * b * b - 0.15 * z
        bar_z = torch.sigmoid(0.40 * z - 0.30 * b + 3.0 * (i - 0.2))
        return {"P": p, "bar_z": bar_z, "Q": self._q_output(state)}


BP_PARITY_J_VALUES = (2, 4, 8)


def _bp_parity_setup():
    reference = ReferenceFirmState(
        eta=1.0, i_low=0.1, i_mid=0.2, i_high=0.3, x=-2.0,
        hatcf=-2.1, lnkf=4.0, hatc_cal=-2.0, lnk_cal=4.1,
        n_parent_rows=2, source="fixture", macro_source="fixture",
    )
    grid = build_frozen_grid(
        reference, b_min=0.2, b_max=0.6, b_points=3,
        z_min=-0.4, z_max=0.4, z_points=3, device=torch.device("cpu"),
    )
    hyperparams = HyperParams()
    hyperparams.pv_use_clipped_m = True
    hyperparams.pv_m_clamp_min = 0.7
    hyperparams.pv_m_clamp_max = 1.3
    economic = AnalysisEconomicConfig.from_current_config()
    transition_max = build_frozen_transition_data(
        _BpToySDF(), grid.base_states, reference, hyperparams, economic,
        n_child_shocks=max(BP_PARITY_J_VALUES), shock_seed=12345,
    )
    return reference, grid, hyperparams, economic, transition_max


def _legacy_bp_by_j(tmp_path):
    reference, grid, hyperparams, economic, transition_max = _bp_parity_setup()
    legacy = {}
    for j_value in BP_PARITY_J_VALUES:
        sliced = slice_frozen_transition_data(
            transition_max, n_continuous_children=int(j_value)
        )
        surfaces, summary, _ = evaluate_bp_consistency(
            _BpToyPolicy(), _BpToySDF(), grid, reference, hyperparams, economic,
            output_dir=tmp_path / f"legacy_j{j_value}",
            n_child_shocks=int(j_value),
            shock_seed=12345,
            transition_data=sliced,
            write_objective_slices=False,
        )
        legacy[int(j_value)] = (surfaces, summary)
    return legacy, (reference, grid, hyperparams, economic, transition_max)


def _optimized_bp_by_j(tmp_path, setup):
    reference, grid, hyperparams, economic, transition_max = setup
    results_by_j, forward_stats = evaluate_bp_consistency_multi_j(
        _BpToyPolicy(), _BpToySDF(), grid, reference, hyperparams, economic,
        output_dir=tmp_path / "optimized",
        j_values=list(BP_PARITY_J_VALUES),
        primary_j=4,
        transition_max=transition_max,
        shock_seed=12345,
        write_objective_slices=False,
    )
    return results_by_j, forward_stats


def test_multi_j_bp_evaluator_matches_legacy_per_j_evaluator(tmp_path):
    """End-to-end parity: the shared multi-J pass must equal per-J legacy runs."""
    legacy, setup = _legacy_bp_by_j(tmp_path)
    results_by_j, forward_stats = _optimized_bp_by_j(tmp_path, setup)
    assert set(results_by_j) == set(BP_PARITY_J_VALUES)

    compared_surfaces = 0
    compared_summaries = 0
    for j_value, (legacy_surfaces, legacy_summary) in legacy.items():
        optimized = results_by_j[j_value]
        for key, expected in legacy_surfaces.items():
            assert key in optimized["surfaces"], f"missing surface {key} at J={j_value}"
            np.testing.assert_allclose(
                optimized["surfaces"][key], expected, rtol=1e-5, atol=1e-6, equal_nan=True
            )
            compared_surfaces += 1
        for key, expected in legacy_summary.items():
            assert key in optimized["summary"], f"missing summary {key} at J={j_value}"
            actual = optimized["summary"][key]
            if isinstance(expected, float) and np.isnan(expected):
                assert np.isnan(actual)
            else:
                assert actual == pytest.approx(expected, rel=1e-5, abs=1e-6)
            compared_summaries += 1
    assert compared_surfaces > 0 and compared_summaries > 0
    assert forward_stats["bp_multi_j_reuse_enabled"] is True
    assert forward_stats["bp_branch_reuse_enabled"] is True


def test_multi_j_bp_evaluator_does_not_alias_branches(tmp_path):
    """Regression: every branch must read its own bundle, never a shared one.

    ``bp_regret``/``teacher_confidence``/``top2_margin`` are derived from the
    branch's own value grid, so if the J index leaked into the branch axis they
    would become identical across ``p0``/``pi_low``/``pi_mid``/``pi_high``.
    """
    legacy, setup = _legacy_bp_by_j(tmp_path)
    results_by_j, _ = _optimized_bp_by_j(tmp_path, setup)
    labels = ("p0", "pi_low", "pi_mid", "pi_high")
    branch_pairs = tuple(zip(labels, labels[1:]))
    for j_value in BP_PARITY_J_VALUES:
        optimized = results_by_j[j_value]
        surfaces = optimized["surfaces"]
        summary = optimized["summary"]
        for left, right in branch_pairs:
            for surface in (
                "bp_regret_raw",
                "teacher_confidence_raw",
                "teacher_top2_margin_raw",
            ):
                assert not np.allclose(
                    surfaces[f"{left}_{surface}"],
                    surfaces[f"{right}_{surface}"],
                    equal_nan=True,
                ), f"{left} aliased {right} on {surface} at J={j_value}"
            assert summary[f"{left}_regret_mean"] != pytest.approx(
                summary[f"{right}_regret_mean"], rel=1e-6, abs=1e-12
            ), f"{left} aliased {right} on regret_mean at J={j_value}"
        # The legacy reference must show the same branch separation, so the toy
        # problem itself cannot be degenerate.
        legacy_surfaces = legacy[j_value][0]
        for left, right in branch_pairs:
            assert not np.allclose(
                legacy_surfaces[f"{left}_bp_regret_raw"],
                legacy_surfaces[f"{right}_bp_regret_raw"],
                equal_nan=True,
            )


def test_multi_j_bp_evaluator_rejects_malformed_branch_bundles(tmp_path, monkeypatch):
    """A branch-axis/J-axis mix-up must fail loudly instead of silently aliasing."""
    _legacy_bp_by_j(tmp_path)
    reference, grid, hyperparams, economic, transition_max = _bp_parity_setup()
    import evaluation.bp_diagnostics as bp_module

    original = bp_module.BPGridTeacher.compute_multi_j_branches

    def collapsed(self, parent_states, children, m_list, **kwargs):
        bundles = original(self, parent_states, children, m_list, **kwargs)
        # Simulate the historical bug: every branch shares branch 0's bundle and
        # loses the higher prefix counts.
        smallest = min(bundles[0])
        return [{smallest: bundle[smallest]} for bundle in bundles]

    monkeypatch.setattr(
        bp_module.BPGridTeacher, "compute_multi_j_branches", collapsed
    )
    with pytest.raises(RuntimeError, match="exposes"):
        evaluate_bp_consistency_multi_j(
            _BpToyPolicy(), _BpToySDF(), grid, reference, hyperparams, economic,
            output_dir=tmp_path / "aliased",
            j_values=[2, 4],
            primary_j=4,
            transition_max=transition_max,
            shock_seed=12345,
            write_objective_slices=False,
        )
