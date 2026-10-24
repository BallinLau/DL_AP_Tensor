"""CPU tests for the cross-episode comparator and the convergence gates.

Covers (Prompt section 14):
* Test 9 -- Outer Drift: two evaluations of the same checkpoint on the same
  fixed EvalBank drift by zero within numerical tolerance (both with synthetic
  results and with a scaled-down ``evaluate_firm_policy`` / ``evaluate_firm_bellman``
  run), while a known parameter perturbation (a small bias added to the P head)
  produces an explainable non-zero drift whose direction and magnitude match the
  injected shift exactly.
* Test 12 -- Convergence Decision: no ``Converged Candidate`` is ever emitted
  when thresholds are unconfigured, when fewer than three episodes are
  available, when the independent test error worsens, when the macro block is
  unstable, when the simulation loses paths to mass extinction or when the
  episodes are semantically incomparable; only three consecutive comparable
  episodes passing every necessary gate yield ``Converged Candidate`` and a
  failure on the third episode resets the consecutive counter.
* extras -- path-clustered bootstrap shape/coverage on known distributions,
  ``NOT_COMPARABLE`` reason codes, absolute errors and sample sizes retained in
  the metric tables, threshold calibration without invented thresholds, and the
  ``save_comparison`` artifact contract.

Everything is CPU-only and synthetic: no cachedir, no training episode, no GPU.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.economic_config import AnalysisEconomicConfig
from config import HyperParams
from evaluation.convergence.checkpoint_io import EvalCheckpoint, compute_param_hashes
from evaluation.convergence.comparator import (
    CONVERGED_CANDIDATE,
    GATE_FAIL,
    GATE_INSUFFICIENT,
    GATE_NOT_CONFIGURED,
    GATE_PASS,
    GATE_SPECS,
    INSUFFICIENT_EVIDENCE,
    NOT_CONVERGED,
    REASON_FIXED_STATE_MISMATCH,
    REASON_GRID_MISMATCH,
    REASON_MISSING_BELLMAN_RESULT,
    REASON_NO_PREVIOUS_EPISODE,
    REASON_NON_CONSECUTIVE_EPISODE_GAP,
    REASON_SEMANTIC_MISMATCH,
    STATUS_NOT_COMPARABLE,
    STATUS_NOT_EVALUATED,
    STATUS_OK,
    ComparisonResult,
    EpisodeEvalRecord,
    ThresholdConfig,
    build_threshold_calibration,
    compare_episodes,
    path_clustered_bootstrap,
    save_comparison,
)
from evaluation.convergence.eval_bank import build_bank_c
from evaluation.convergence.firm_bellman import (
    BellmanEvalConfig,
    BellmanEvalResult,
    evaluate_firm_bellman,
)
from evaluation.convergence.firm_policy import (
    PolicyEvalConfig,
    PolicyEvalResult,
    evaluate_firm_policy,
)
from evaluation.convergence.macro_sdf import MacroSdfEvalResult
from evaluation.convergence.sim_eval import SimEvalResult
from models import PolicyValueModel


# ---------------------------------------------------------------------------
# synthetic semantic record / bank reference
# ---------------------------------------------------------------------------


def _semantic(**overrides):
    record = {
        "label": "synthetic",
        "checkpoint_path": "synthetic://test",
        "checkpoint_sha256": None,
        "episode": None,
        "git_commit": None,
        "checkpoint_format": "combined",
        "loaded_model_keys": ["policy_value", "sdf_fc1"],
        "q_parameterization": "hybrid_regime",
        "q_parameterization_source": "checkpoint_spec",
        "q_parameterization_assumed_legacy": False,
        "q_semantics_recorded_fields": ["q_parameterization", "q_recovery_normalization_mode"],
        "q_recovery_normalization_mode": "asset_only",
        "q_recovery_normalization_mode_source": "checkpoint_hyperparams",
        "q_recovery_normalization_mode_assumed_legacy": False,
        "value_parameterization": {
            "mode": "none",
            "scale_formula": "1",
            "bellman_normalization": False,
            "log_max": 20.0,
        },
        "policy_value_model_spec": {"base_state_dim": 6, "q_parameterization": "hybrid_regime"},
        "economic_config": {"ZETA": 0.5, "I_THRESHOLD": 0.5},
        "param_hashes": {},
        "optimizer_steps": 0,
        "evaluation_only": True,
        "missing_optional_fields": [],
    }
    record.update(overrides)
    return record


_GRID_METADATA = {
    "grid": {
        "b_min": 0.0,
        "b_max": 1.0,
        "b_points": 5,
        "z_min": -1.0,
        "z_max": 1.0,
        "z_points": 5,
        "i_min": 0.0,
        "i_max": 0.5,
        "i_points": 5,
        "eta": [0.0, 1.0],
    },
    "reference_state": {
        "x": -3.0,
        "hatcf": -3.0,
        "lnkf": 4.0,
        "hatc_cal": -3.0,
        "lnk_cal": 4.0,
        "i_low": 0.0,
        "i_mid": 0.25,
        "i_high": 0.5,
        "eta": 1.0,
    },
    "m_mode": "clipped_train_m",
    "m_clamp_bounds": [0.7, 1.3],
}


def _bank_ref(**overrides):
    ref = {
        "bank_id": "C",
        "bank_version": "evaluation_bank_v1",
        "bank_file_sha256": "bank-file-hash",
        "bank_content_sha256": "bank-content-hash",
        "parent_states_sha256": "parent-states-hash",
        "grid_metadata": json.loads(json.dumps(_GRID_METADATA)),
    }
    ref.update(overrides)
    return ref


# ---------------------------------------------------------------------------
# synthetic evaluator results
# ---------------------------------------------------------------------------


def _rows_frame(*, shift_p0: float = 0.0, n_rows: int = 6):
    """Fixed-state rows of one Bellman evaluation (the value-drift input)."""
    sources = ["simulated", "coverage", "stress"]
    rows = []
    for index in range(n_rows):
        p0 = 1.0 + 0.05 * index + float(shift_p0)
        pi = 1.2 + 0.03 * index
        rows.append(
            {
                "row_index": index,
                "b": 0.2 + 0.1 * index,
                "z": -0.3 + 0.1 * index,
                "eta": float(index % 2),
                "i": 0.1,
                "x": -3.0,
                "source_label": sources[index % 3],
                "p0_physical": p0,
                "pi_physical": pi,
                "p0_normalized": p0,
                "pi_normalized": pi,
                "P": p0,
                "Phat": p0,
                "Q_claim": 0.5 + 0.02 * index,
            }
        )
    return pd.DataFrame(rows)


def _bellman_result(rows: pd.DataFrame, *, metrics=None, state_hash="parent-states-hash"):
    default_metrics = {
        "fresh_p0_normalized_mae": 1.0e-4,
        "fresh_p0_normalized_n": float(len(rows)),
        "fresh_pi_normalized_mae": 2.0e-4,
        "fresh_pi_normalized_n": float(len(rows)),
        "q_qs_claim_bellman_trainM_mae": 3.0e-4,
        "q_qs_claim_bellman_trainM_n": float(len(rows)),
        "frozen_p0_physical_mae": 1.5e-4,
        "frozen_nonfinite_ratio": 0.0,
    }
    default_metrics.update(metrics or {})
    surfaces = {
        "P0_physical": rows["p0_physical"].to_numpy(dtype=np.float64),
        "PI_physical": rows["pi_physical"].to_numpy(dtype=np.float64),
        "P0_normalized": rows["p0_normalized"].to_numpy(dtype=np.float64),
        "PI_normalized": rows["pi_normalized"].to_numpy(dtype=np.float64),
        "P": rows["P"].to_numpy(dtype=np.float64),
        "Q_claim": rows["Q_claim"].to_numpy(dtype=np.float64),
    }
    return BellmanEvalResult(
        metrics=default_metrics,
        tables={"rows": rows.copy()},
        surfaces=surfaces,
        diagnostics={"parent_states_sha256": state_hash, "episode": None},
    )


def _policy_rows_frame(*, shift_p0: float = 0.0, shift_bp: float = 0.0, n_rows: int = 6):
    rows = []
    for index in range(n_rows):
        p0 = 1.0 + 0.05 * index + float(shift_p0)
        rows.append(
            {
                "row_index": index,
                "b": 0.2 + 0.1 * index,
                "eta": float(index % 2),
                "bp0_star": 0.4 + 0.01 * index + float(shift_bp),
                "bpi_star": 0.5 + 0.01 * index + float(shift_bp),
                "bp_star_mix": 0.45 + 0.01 * index + float(shift_bp),
                "b_next_mix": 0.45 + 0.01 * index + float(shift_bp),
                "Phat": 0.2 - 0.15 * index,
                "P0": p0,
                "PI": p0 + 0.1,
                "bar_i_model": 0.5,
                "bar_i_executed": 1.0 if index % 3 == 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _default_boundary_frame(n_rows: int = 6):
    return pd.DataFrame(
        {
            "eta_view": [0.0] * n_rows + [1.0] * n_rows,
            "b": [0.2 + 0.1 * i for i in range(n_rows)] * 2,
            "z_hard_phat0": [-0.5 + 0.05 * i for i in range(n_rows)] * 2,
        }
    )


def _policy_result(
    policy_rows: pd.DataFrame,
    *,
    metrics=None,
    state_hash="parent-states-hash",
    episode=None,
):
    default_metrics = {"policy_rows_n": float(len(policy_rows))}
    default_metrics.update(metrics or {})
    return PolicyEvalResult(
        metrics=default_metrics,
        tables={
            "policy_rows": policy_rows.copy(),
            "default_boundary": _default_boundary_frame(len(policy_rows)),
        },
        surfaces={},
        diagnostics={
            "parent_states_sha256": state_hash,
            "episode": episode,
            "policy_flip_tol": 1.0e-2,
        },
    )


def _macro_result(*, level: float = 1.0e-3, nonfinite: float = 0.0):
    return MacroSdfEvalResult(
        metrics={
            "fc1_one_step_hatc_rmse": level,
            "fc1_one_step_hatc_mae": level,
            "fc1_one_step_lnk_rmse": level,
            "fc1_rec_autonomous_recursive_hatc_h10_rmse": 2.0 * level,
            "fc1_rec_autonomous_recursive_hatc_drift_at_max_h": 2.0 * level,
            "fc1_rec_production_one_step_lnk_h10_rmse": 2.0 * level,
            "sdf_g_max": level,
            "sdf_population_target_abs": level,
            "sdf_nonfinite_ratio": nonfinite,
        },
        tables={},
        surfaces={},
        diagnostics={},
    )


def _marginals_frame(*, weight_shift: float = 0.0):
    return pd.DataFrame(
        {
            "t": [0.0, 0.0, 1.0, 1.0],
            "caliber": ["count", "count", "count", "count"],
            "axis": ["b", "b", "b", "b"],
            "bin_center": [0.25, 0.75, 0.25, 0.75],
            "weight": [0.5 + weight_shift, 0.5 - weight_shift, 0.5, 0.5],
        }
    )


def _sim_result(
    *,
    resource_residual: float = 1.0e-7,
    nonfinite: float = 0.0,
    extinct_paths: float = 0.0,
    capital_drop: float = 0.1,
    feasible_share: float = 1.0,
    completed_horizon: float = 50.0,
    weight_shift: float = 0.0,
):
    return SimEvalResult(
        metrics={
            "waterfall_identity_residual_abs_max": resource_residual,
            "waterfall_node_id_continuity_violations": 0.0,
            "waterfall_end_node_id_mismatch_count": 0.0,
            "waterfall_side_survival_rate_mean": 0.9,
            "waterfall_side_default_rate_mean": 0.1,
            "waterfall_side_investment_rate_mean": 0.3,
            "waterfall_side_entry_rate_mean": 0.2,
            "sim_resource_feasible_share_min": feasible_share,
            "sim_completed_horizon_min": completed_horizon,
            "sim_n_extinct_paths_max": extinct_paths,
            "sim_nonfinite_share_max": nonfinite,
            "sim_baseline_capital_drop_t0_t1": capital_drop,
            "sim_baseline_capital_change_t0_t1": -capital_drop,
            "sim_baseline_k0_total": 100.0,
            "sim_baseline_k_t1_total": 90.0,
            "sim_baseline_dist_extinct_paths_total": extinct_paths,
            "sim_baseline_dist_w1_b_0_to_1_count": 0.01,
        },
        tables={"distribution_marginals": _marginals_frame(weight_shift=weight_shift)},
        surfaces={},
        diagnostics={},
    )


def _record(
    episode: int,
    *,
    semantic=None,
    bank_ref=None,
    bellman=True,
    policy=True,
    macro=True,
    simulation=True,
    seed=None,
):
    return EpisodeEvalRecord(
        episode=int(episode),
        semantic=semantic if semantic is not None else _semantic(),
        bellman=_bellman_result(_rows_frame()) if bellman is True else bellman,
        policy=_policy_result(_policy_rows_frame(), episode=episode) if policy is True else policy,
        macro=_macro_result() if macro is True else macro,
        simulation=_sim_result() if simulation is True else simulation,
        bank_ref=bank_ref if bank_ref is not None else _bank_ref(),
        checkpoint=None,
        seed=seed,
    )


def _thresholds(**overrides):
    base = {
        "hard_resource_ledger_residual": 1.0e-4,
        "hard_capital_transition_validity": 0.0,
        "hard_consumption_feasibility": 0.99,
        "hard_nonfinite": 0.0,
        "hard_q_semantic_consistency": 0.0,
        "hard_sim_path_completeness": 10.0,
        "firm_p_bellman_residual": 1.0e-3,
        "firm_q_bellman_residual": 1.0e-3,
        "firm_policy_stability": 0.1,
        "firm_value_drift": 1.0e-2,
        "firm_bp_star_drift": 1.0e-2,
        "macro_fc1_one_step": 1.0e-2,
        "macro_fc1_recursive": 1.0e-2,
        "macro_sdf_euler_moments": 1.0e-2,
        "outer_fixed_shock_rollout": 1.0e-2,
        "outer_macro_trajectory_drift": 1.0e-2,
        "outer_distribution_drift": 1.0e-2,
        "outer_extinct_paths": 0.0,
    }
    base.update(overrides)
    return ThresholdConfig.from_dict(base)


def _gate_entry(status, episode, gate_id):
    for entry in status["episode_status"]:
        if entry["episode"] == episode:
            spec = GATE_SPECS[gate_id]
            return entry["gates"][spec.group][gate_id]
    raise AssertionError(f"episode {episode} not found")


# ---------------------------------------------------------------------------
# Test 9 -- Outer Drift (synthetic results with hand-checkable numbers)
# ---------------------------------------------------------------------------


def test_9_identical_evaluations_drift_by_zero():
    records = [_record(3), _record(4)]
    result = compare_episodes(records, threshold_config=_thresholds())

    value_drift = result.metrics["value_drift"]
    first = value_drift.loc[value_drift["episode"] == 3].iloc[0]
    assert first["status"] == STATUS_NOT_EVALUATED
    assert first["reason_code"] == REASON_NO_PREVIOUS_EPISODE

    row = value_drift.loc[value_drift["episode"] == 4].iloc[0]
    assert row["status"] == STATUS_OK
    for quantity in (
        "p0_physical",
        "pi_physical",
        "p0_normalized",
        "pi_normalized",
        "P",
        "Q_claim",
    ):
        assert row[f"{quantity}_mae"] == pytest.approx(0.0, abs=1e-12)
        assert row[f"{quantity}_rmse"] == pytest.approx(0.0, abs=1e-12)
        assert row[f"{quantity}_max"] == pytest.approx(0.0, abs=1e-12)
        assert row[f"{quantity}_relative_mae"] == pytest.approx(0.0, abs=1e-12)
        assert row[f"{quantity}_n"] == 6.0
        assert row[f"{quantity}_n_finite"] == 6.0

    policy_drift = result.metrics["policy_drift"]
    policy_row = policy_drift.loc[policy_drift["episode"] == 4].iloc[0]
    assert policy_row["status"] == STATUS_OK
    assert policy_row["bp_star_drift_max"] == pytest.approx(0.0, abs=1e-12)
    assert policy_row["bp_star_drift_mean"] == pytest.approx(0.0, abs=1e-12)
    assert policy_row["policy_flip_rate"] == pytest.approx(0.0, abs=1e-12)
    assert policy_row["default_flip_rate"] == pytest.approx(0.0, abs=1e-12)
    assert policy_row["investment_flip_rate"] == pytest.approx(0.0, abs=1e-12)
    assert policy_row["bar_i_executed_drift_mean_abs"] == pytest.approx(0.0, abs=1e-12)
    assert policy_row["default_boundary_z_drift_mean_abs_eta1"] == pytest.approx(0.0, abs=1e-12)


def test_9_known_perturbation_drift_direction_and_magnitude():
    epsilon = 0.05
    previous_rows = _rows_frame()
    current_rows = _rows_frame(shift_p0=epsilon)
    previous = _record(
        3,
        bellman=_bellman_result(previous_rows),
        policy=_policy_result(_policy_rows_frame(), episode=3),
    )
    current = _record(
        4,
        bellman=_bellman_result(current_rows),
        policy=_policy_result(_policy_rows_frame(shift_p0=epsilon), episode=4),
    )
    result = compare_episodes([previous, current], threshold_config=_thresholds())

    row = result.metrics["value_drift"].loc[lambda frame: frame["episode"] == 4].iloc[0]
    expected_relative = epsilon / float(np.mean(1.0 + np.abs(current_rows["p0_physical"].to_numpy())))
    # the drift of the perturbed quantity is exactly the injected shift
    assert row["p0_physical_mae"] == pytest.approx(epsilon, abs=1e-12)
    assert row["p0_physical_max"] == pytest.approx(epsilon, abs=1e-12)
    assert row["p0_physical_mean_signed"] == pytest.approx(epsilon, abs=1e-12)
    assert row["p0_physical_relative_mae"] == pytest.approx(expected_relative, rel=1e-12)
    # untouched quantities stay at zero
    assert row["pi_physical_mae"] == pytest.approx(0.0, abs=1e-12)
    assert row["Q_claim_mae"] == pytest.approx(0.0, abs=1e-12)
    assert row["P_mae"] == pytest.approx(epsilon, abs=1e-12)
    # absolute errors and sample sizes are retained, with eta / source strata
    assert row["p0_physical_n"] == 6.0
    assert row["p0_physical_n_finite"] == 6.0
    assert row["p0_physical_eta0_mae"] == pytest.approx(epsilon, abs=1e-12)
    assert row["p0_physical_eta1_mae"] == pytest.approx(epsilon, abs=1e-12)
    assert row["p0_physical_simulated_mae"] == pytest.approx(epsilon, abs=1e-12)
    assert row["p0_physical_coverage_mae"] == pytest.approx(epsilon, abs=1e-12)
    assert row["p0_physical_stress_mae"] == pytest.approx(epsilon, abs=1e-12)
    # the surface-level drift mirrors the row-level one
    assert row["fdrift_p0_physical_mean_abs_diff"] == pytest.approx(epsilon, abs=1e-12)
    assert row["fdrift_pi_physical_mean_abs_diff"] == pytest.approx(0.0, abs=1e-12)

    policy_row = result.metrics["policy_drift"].loc[lambda frame: frame["episode"] == 4].iloc[0]
    assert policy_row["p0_drift_mean_abs"] == pytest.approx(epsilon, abs=1e-12)
    assert policy_row["bar_i_executed_drift_mean_abs"] == pytest.approx(0.0, abs=1e-12)
    assert 0.0 <= policy_row["policy_flip_rate"] <= 1.0
    assert 0.0 <= policy_row["default_flip_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Test 9 -- Outer Drift (scaled-down real evaluator runs)
# ---------------------------------------------------------------------------


def _hyperparams(**overrides) -> HyperParams:
    hp = HyperParams()
    hp.bp_grid_min = 0.0
    hp.bp_grid_max = 1.0
    hp.bp_grid_coarse_size = 5
    hp.bp_grid_fine_size = 3
    hp.bp_grid_refine_enabled = True
    hp.bp_grid_quadratic_refine = False
    hp.bp_grid_parent_chunk_size = 64
    hp.bp_grid_candidate_chunk_size = 0
    hp.bp_grid_max_expanded_states = 4096
    hp.pv_use_clipped_m = True
    hp.pv_m_clamp_min = 0.7
    hp.pv_m_clamp_max = 1.3
    hp.pv_value_scale_mode = "none"
    hp.pv_value_scale_log_max = 20.0
    hp.pv_bellman_normalize_by_value_scale = False
    hp.q_recovery_normalization_mode = "asset_only"
    for name, value in overrides.items():
        setattr(hp, name, value)
    return hp


def _small_model(*, seed: int = 11) -> PolicyValueModel:
    torch.manual_seed(seed)
    return PolicyValueModel(
        share_hidden_dims=[8],
        share_output_dim=8,
        q_head_dims=[4],
        p0_head_dims=[4],
        pi_head_dims=[4],
        bp0_head_dims=[4],
        bpi_head_dims=[4],
        barz_hidden_dims=[4],
        bari_hidden_dims=[4],
        i_grid_size=5,
        dropout=0.0,
        value_scale_mode="none",
        q_parameterization="hybrid_regime",
    )


class FakeSDF(torch.nn.Module):
    """Minimal FC1/SDF stand-in for ``build_transition_for_checkpoint``."""

    def __init__(self, m_value: float = 0.9):
        super().__init__()
        self.m_value = float(m_value)

    def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical=True):
        n_rows, n_children, _ = x_curr.shape
        m = torch.full((n_rows, n_children, 1), self.m_value, dtype=x_curr.dtype)
        hatcf = torch.full((n_rows, n_children, 1), -3.0, dtype=x_curr.dtype)
        lnkf = torch.full((n_rows, n_children, 1), 4.0, dtype=x_curr.dtype)
        return m, m, m, hatcf, lnkf


def _bank():
    return build_bank_c(
        seed=7,
        macro_anchors={
            "Normal": {
                "label": "Normal",
                "quantile": 0.5,
                "x": -3.0,
                "hatcf": -3.0,
                "lnkf": 4.0,
                "hatc_cal": -3.0,
                "lnk_cal": 4.0,
            }
        },
        b_values=(0.7, 1.0),
        z_values=(-0.3, 0.3),
        i_values=(0.1, 0.4),
        eta_values=(0.0, 1.0),
    )


def _checkpoint(model, sdf, economic_config, hyperparams, *, episode: int = 0) -> EvalCheckpoint:
    models = {"policy_value": model, "sdf_fc1": sdf}
    metadata = {
        "label": "synthetic",
        "q_parameterization": getattr(model, "q_parameterization", None),
        "q_semantics_recorded_fields": ["q_recovery_normalization_mode"],
    }
    value_metadata = getattr(model, "value_parameterization_metadata", None)
    if callable(value_metadata):
        metadata["value_parameterization"] = {"current": value_metadata()}
    return EvalCheckpoint(
        models=models,
        hyperparams=hyperparams,
        economic_config=economic_config,
        metadata=metadata,
        episode=episode,
        checkpoint_path="synthetic://test",
        param_hashes=compute_param_hashes(models),
    )


def _shift_head_bias(model: PolicyValueModel, head: str, bias: float) -> None:
    module = getattr(model, head)
    linear_layers = [item for item in module.modules() if isinstance(item, torch.nn.Linear)]
    with torch.no_grad():
        linear_layers[-1].bias.add_(float(bias))


def test_9_outer_drift_on_scaled_down_evaluator_runs():
    epsilon = 0.05
    economic = AnalysisEconomicConfig.from_current_config()
    hyperparams = _hyperparams()
    bank = _bank()
    policy_config = PolicyEvalConfig(
        b_points=5, z_points=5, i_points=5, objective_slice_points=5
    )
    bellman_config = BellmanEvalConfig(
        parent_chunk_size=4, dense_grid_size=5, chunk_size=64
    )

    def _run(episode: int, *, perturb: float = 0.0) -> EpisodeEvalRecord:
        model = _small_model(seed=5)
        if perturb:
            _shift_head_bias(model, "v0_head", perturb)
        cp = _checkpoint(model, FakeSDF(), economic, hyperparams, episode=episode)
        policy = evaluate_firm_policy(cp, bank, config=policy_config)
        bellman = evaluate_firm_bellman(cp, bank, config=bellman_config)
        state_hash = policy.diagnostics["parent_states_sha256"]
        return EpisodeEvalRecord(
            episode=episode,
            semantic=_semantic(),
            bellman=bellman,
            policy=policy,
            macro=_macro_result(),
            simulation=_sim_result(),
            bank_ref=_bank_ref(parent_states_sha256=state_hash),
        )

    first = _run(0)
    repeated = _run(1)
    perturbed = _run(1, perturb=epsilon)

    zero = compare_episodes([first, repeated], threshold_config=_thresholds())
    zero_row = zero.metrics["value_drift"].loc[lambda frame: frame["episode"] == 1].iloc[0]
    assert zero_row["p0_physical_mae"] == pytest.approx(0.0, abs=1e-8)
    assert zero_row["pi_physical_mae"] == pytest.approx(0.0, abs=1e-8)
    assert zero_row["Q_claim_mae"] == pytest.approx(0.0, abs=1e-8)
    zero_policy = zero.metrics["policy_drift"].loc[lambda frame: frame["episode"] == 1].iloc[0]
    assert zero_policy["bp_star_drift_max"] == pytest.approx(0.0, abs=1e-8)
    assert zero_policy["policy_flip_rate"] == pytest.approx(0.0, abs=1e-12)
    assert zero_policy["default_flip_rate"] == pytest.approx(0.0, abs=1e-12)
    assert zero_policy["investment_flip_rate"] == pytest.approx(0.0, abs=1e-12)

    drift = compare_episodes([first, perturbed], threshold_config=_thresholds())
    row = drift.metrics["value_drift"].loc[lambda frame: frame["episode"] == 1].iloc[0]
    # the injected +epsilon bias on the v0 head shifts P0 by exactly epsilon
    assert row["p0_physical_mae"] == pytest.approx(epsilon, abs=1e-6)
    assert row["p0_physical_mean_signed"] == pytest.approx(epsilon, abs=1e-6)
    # heads that were not touched do not drift
    assert row["pi_physical_mae"] == pytest.approx(0.0, abs=1e-6)
    assert row["Q_claim_mae"] == pytest.approx(0.0, abs=1e-6)
    policy_row = drift.metrics["policy_drift"].loc[lambda frame: frame["episode"] == 1].iloc[0]
    assert policy_row["p0_drift_mean_abs"] == pytest.approx(epsilon, abs=1e-6)
    assert policy_row["bp_star_drift_max"] >= 0.0
    assert 0.0 <= policy_row["policy_flip_rate"] <= 1.0


# ---------------------------------------------------------------------------
# path-clustered bootstrap
# ---------------------------------------------------------------------------


def test_path_clustered_bootstrap_shape_on_known_distribution():
    rng = np.random.default_rng(7)
    n_paths, per_path = 40, 5
    path_means = 2.0 + rng.normal(0.0, 1.0, size=n_paths)
    values = path_means.repeat(per_path)
    path_ids = np.arange(n_paths).repeat(per_path)
    out = path_clustered_bootstrap(values, path_ids, n_boot=500, seed=11, stat="mean")
    assert out["n_paths"] == n_paths
    assert out["n_obs"] == n_paths * per_path
    assert out["estimate"] == pytest.approx(float(values.mean()), abs=1e-12)
    assert out["ci_low"] <= out["estimate"] <= out["ci_high"]
    assert out["ci_low"] < out["ci_high"]
    assert out["reason_code"] == "OK"

    median = path_clustered_bootstrap(values, path_ids, n_boot=200, seed=3, stat="median")
    assert median["estimate"] == pytest.approx(float(np.median(values)), abs=1e-12)

    empty = path_clustered_bootstrap([], [], n_boot=10, seed=0)
    assert empty["reason_code"] == "NO_FINITE_OBSERVATIONS"
    assert math.isnan(empty["estimate"])

    with pytest.raises(ValueError):
        path_clustered_bootstrap([1.0, 2.0], [0], n_boot=10, seed=0)


def test_path_clustered_bootstrap_ci_coverage_on_known_mean():
    true_mean = 2.0
    trials, covered = 200, 0
    for trial in range(trials):
        rng = np.random.default_rng(1000 + trial)
        path_means = true_mean + rng.normal(0.0, 1.0, size=30)
        values = path_means.repeat(4)
        path_ids = np.arange(30).repeat(4)
        out = path_clustered_bootstrap(values, path_ids, n_boot=200, seed=trial, stat="mean")
        if out["ci_low"] <= true_mean <= out["ci_high"]:
            covered += 1
    coverage = covered / trials
    assert 0.85 <= coverage <= 1.0


# ---------------------------------------------------------------------------
# Test 12 -- Convergence Decision (no false Converged Candidate)
# ---------------------------------------------------------------------------


def test_12_unconfigured_thresholds_are_never_passed():
    records = [_record(3), _record(4), _record(5)]
    result = compare_episodes(records)  # no ThresholdConfig at all
    status = result.convergence_status
    assert status["convergence_status"] == INSUFFICIENT_EVIDENCE
    assert status["converged_candidate"] is False
    assert status["max_consecutive_satisfied_count"] == 0
    assert status["threshold_config_complete"] is False
    assert "firm_value_drift" in status["missing_required_thresholds"]
    for entry in status["episode_status"]:
        for group in entry["gates"].values():
            for gate in group.values():
                assert gate["status"] != GATE_PASS
                if gate["applicable"]:
                    assert gate["status"] == GATE_NOT_CONFIGURED
                    assert gate["reason_code"] == "THRESHOLD_NOT_CONFIGURED"
                else:
                    assert gate["status"] == GATE_INSUFFICIENT
                    assert gate["reason_code"] == "NO_PREVIOUS_EPISODE_FOR_DRIFT_GATE"


def test_12_fewer_than_three_episodes_never_converges():
    result = compare_episodes([_record(3), _record(4)], threshold_config=_thresholds())
    status = result.convergence_status
    assert status["convergence_status"] != CONVERGED_CANDIDATE
    assert status["converged_candidate"] is False
    assert status["max_consecutive_satisfied_count"] == 2
    assert status["reason_code"] == "INSUFFICIENT_EVIDENCE_FOR_CONVERGENCE"


def test_12_worsening_independent_test_error_blocks_convergence():
    bad = _record(
        5,
        bellman=_bellman_result(_rows_frame(), metrics={"fresh_p0_normalized_mae": 0.99}),
    )
    records = [_record(3), _record(4), bad]
    result = compare_episodes(records, threshold_config=_thresholds())
    status = result.convergence_status
    assert status["convergence_status"] == NOT_CONVERGED
    assert status["converged_candidate"] is False
    gate = _gate_entry(status, 5, "firm_p_bellman_residual")
    assert gate["status"] == GATE_FAIL
    assert gate["reason_code"] == "FAIL_ABOVE_THRESHOLD"
    assert gate["value"] == pytest.approx(0.99)
    counts = [entry["consecutive_satisfied_count"] for entry in status["episode_status"]]
    assert counts == [1, 2, 0]
    assert status["episode_status"][2]["satisfied"] is False
    assert status["episode_status"][2]["streak_reset_reason"] == "GATE_FAIL"


def test_12_macro_instability_blocks_convergence():
    unstable = _record(5, macro=_macro_result(level=5.0))
    result = compare_episodes([_record(3), _record(4), unstable], threshold_config=_thresholds())
    status = result.convergence_status
    assert status["convergence_status"] == NOT_CONVERGED
    failing = [
        gate
        for group in status["episode_status"][2]["gates"].values()
        for gate in group.values()
        if gate["status"] == GATE_FAIL
    ]
    assert any(gate["group"] == "macro" for gate in failing)
    assert status["episode_status"][2]["consecutive_satisfied_count"] == 0


def test_12_simulation_mass_extinction_blocks_convergence():
    extinct = _record(5, simulation=_sim_result(extinct_paths=7.0))
    result = compare_episodes([_record(3), _record(4), extinct], threshold_config=_thresholds())
    status = result.convergence_status
    assert status["convergence_status"] == NOT_CONVERGED
    gate = _gate_entry(status, 5, "outer_extinct_paths")
    assert gate["status"] == GATE_FAIL
    assert gate["value"] == pytest.approx(7.0)


def test_12_semantically_incomparable_episodes_never_converge():
    # Q-caliber semantics changed across episodes: the semantic pre-check vetoes
    # the comparison and the hard Q-semantic-consistency gate fails outright.
    records = [
        _record(3),
        _record(4),
        _record(5, semantic=_semantic(q_parameterization="b_times_unit")),
    ]
    result = compare_episodes(records, threshold_config=_thresholds())
    status = result.convergence_status
    assert status["convergence_status"] == NOT_CONVERGED
    assert status["converged_candidate"] is False
    gate = _gate_entry(status, 5, "hard_q_semantic_consistency")
    assert gate["status"] == GATE_FAIL
    value_row = result.metrics["value_drift"].loc[
        result.metrics["value_drift"]["episode"] == 5
    ].iloc[0]
    assert value_row["status"] == STATUS_NOT_COMPARABLE
    assert REASON_SEMANTIC_MISMATCH in value_row["reason_code"]

    # Non-Q semantic mismatch: the drift groups become NOT_COMPARABLE and the
    # episode cannot be judged (INSUFFICIENT_EVIDENCE) -- never a Converged
    # Candidate, and the affected gate surfaces GROUP_NOT_COMPARABLE.
    other_records = [
        _record(3),
        _record(4),
        _record(5, semantic=_semantic(economic_config={"ZETA": 0.7, "I_THRESHOLD": 0.5})),
    ]
    result2 = compare_episodes(other_records, threshold_config=_thresholds())
    status2 = result2.convergence_status
    assert status2["convergence_status"] == INSUFFICIENT_EVIDENCE
    assert status2["converged_candidate"] is False
    last = status2["episode_status"][2]
    assert last["satisfied"] is None
    assert last["streak_reset_reason"] == "NOT_COMPARABLE"
    assert REASON_SEMANTIC_MISMATCH in last["not_comparable_reason_code"]
    policy_row = result2.metrics["policy_drift"].loc[
        result2.metrics["policy_drift"]["episode"] == 5
    ].iloc[0]
    assert policy_row["status"] == STATUS_NOT_COMPARABLE
    gate2 = _gate_entry(status2, 5, "firm_value_drift")
    assert gate2["status"] == GATE_INSUFFICIENT
    assert gate2["reason_code"] == "GROUP_NOT_COMPARABLE"
    assert status2["comparable_episode_pairs"] == [[3, 4]]


def test_12_three_consecutive_passing_episodes_yield_converged_candidate():
    records = [_record(3), _record(4), _record(5)]
    result = compare_episodes(records, threshold_config=_thresholds())
    status = result.convergence_status
    assert status["convergence_status"] == CONVERGED_CANDIDATE
    assert status["converged_candidate"] is True
    assert status["reason_code"] == "THREE_CONSECUTIVE_COMPARABLE_EPISODES_SATISFIED"
    assert status["required_consecutive_comparable_episodes"] == 3
    assert status["max_consecutive_satisfied_count"] == 3
    assert status["final_consecutive_satisfied_count"] == 3
    assert status["participating_episodes"] == [3, 4, 5]
    assert status["comparable_episode_pairs"] == [[3, 4], [4, 5]]
    assert [entry["consecutive_satisfied_count"] for entry in status["episode_status"]] == [1, 2, 3]
    for entry in status["episode_status"]:
        assert entry["satisfied"] is True
    for entry in status["episode_status"][1:]:
        for gate in entry["gates"]["firm"].values():
            assert gate["status"] == GATE_PASS


def test_12_third_episode_failure_resets_consecutive_counter():
    failing = _record(
        5,
        bellman=_bellman_result(_rows_frame(), metrics={"fresh_p0_normalized_mae": 0.99}),
    )
    short_records = [_record(3), _record(4), failing, _record(6), _record(7)]
    result = compare_episodes(short_records, threshold_config=_thresholds())
    counts = [entry["consecutive_satisfied_count"] for entry in result.convergence_status["episode_status"]]
    assert counts == [1, 2, 0, 1, 2]
    assert result.convergence_status["max_consecutive_satisfied_count"] == 2
    assert result.convergence_status["convergence_status"] == NOT_CONVERGED

    long_records = short_records + [_record(8)]
    result_long = compare_episodes(long_records, threshold_config=_thresholds())
    counts_long = [
        entry["consecutive_satisfied_count"] for entry in result_long.convergence_status["episode_status"]
    ]
    assert counts_long == [1, 2, 0, 1, 2, 3]
    assert result_long.convergence_status["convergence_status"] == CONVERGED_CANDIDATE
    assert result_long.convergence_status["episode_status"][2]["streak_reset_reason"] == "GATE_FAIL"


# ---------------------------------------------------------------------------
# comparability reason codes / missing evidence
# ---------------------------------------------------------------------------


def test_not_comparable_reason_codes_are_explicit():
    base = _record(3)
    scenarios = [
        ({"bank_ref": _bank_ref(bank_file_sha256="other-hash")}, "BANK_FILE_HASH_MISMATCH"),
        ({"bank_ref": _bank_ref(bank_content_sha256="other-hash")}, "BANK_CONTENT_HASH_MISMATCH"),
        ({"bank_ref": _bank_ref(parent_states_sha256="other-hash")}, REASON_FIXED_STATE_MISMATCH),
        (
            {"bank_ref": _bank_ref(grid_metadata={**_GRID_METADATA, "m_mode": "raw_sdf_m"})},
            REASON_GRID_MISMATCH,
        ),
        (
            {
                "semantic": _semantic(
                    value_parameterization={
                        "mode": "exp_xz",
                        "scale_formula": "1+exp(clamp(x+z))",
                        "bellman_normalization": True,
                        "log_max": 20.0,
                    }
                )
            },
            REASON_SEMANTIC_MISMATCH,
        ),
    ]
    for overrides, expected in scenarios:
        other = _record(4, **overrides)
        result = compare_episodes([base, other], threshold_config=_thresholds())
        row = result.metrics["value_drift"].loc[
            result.metrics["value_drift"]["episode"] == 4
        ].iloc[0]
        assert row["status"] == STATUS_NOT_COMPARABLE
        assert expected in row["reason_code"]
        assert result.convergence_status["converged_candidate"] is False


def test_episode_gaps_are_flagged_and_break_streaks():
    records = [_record(3), _record(5), _record(7)]
    result = compare_episodes(records, threshold_config=_thresholds())
    status = result.convergence_status
    row = result.metrics["value_drift"].loc[result.metrics["value_drift"]["episode"] == 5].iloc[0]
    assert row["status"] == STATUS_NOT_COMPARABLE
    assert REASON_NON_CONSECUTIVE_EPISODE_GAP in row["reason_code"]
    assert status["comparable_episode_pairs"] == []
    assert status["convergence_status"] != CONVERGED_CANDIDATE


def test_missing_results_carry_explicit_reason_codes():
    records = [
        _record(3),
        _record(4, bellman=None),
        _record(5, macro=None, simulation=None),
    ]
    result = compare_episodes(records, threshold_config=_thresholds())
    firm_row = result.metrics["firm_bellman"].loc[
        result.metrics["firm_bellman"]["episode"] == 4
    ].iloc[0]
    assert firm_row["status"] == "MISSING"
    assert firm_row["reason_code"] == REASON_MISSING_BELLMAN_RESULT
    value_row = result.metrics["value_drift"].loc[
        result.metrics["value_drift"]["episode"] == 4
    ].iloc[0]
    assert value_row["status"] == "MISSING"
    assert value_row["reason_code"] == REASON_MISSING_BELLMAN_RESULT
    macro_row = result.metrics["macro"].loc[result.metrics["macro"]["episode"] == 5].iloc[0]
    assert macro_row["status"] == "MISSING"
    assert macro_row["reason_code"] == "MISSING_MACRO_RESULT"
    simulation_row = result.metrics["simulation"].loc[
        result.metrics["simulation"]["episode"] == 5
    ].iloc[0]
    assert simulation_row["status"] == "MISSING"
    assert simulation_row["reason_code"] == "MISSING_SIMULATION_RESULT"
    distribution_row = result.metrics["distribution"].loc[
        result.metrics["distribution"]["episode"] == 5
    ].iloc[0]
    assert distribution_row["status"] == "MISSING"


def test_metric_tables_keep_absolute_errors_and_sample_sizes():
    result = compare_episodes([_record(3), _record(4)], threshold_config=_thresholds())
    value_drift = result.metrics["value_drift"]
    for quantity in ("p0_physical", "pi_physical", "P", "Q_claim"):
        assert f"{quantity}_mae" in value_drift.columns
        assert f"{quantity}_rmse" in value_drift.columns
        assert f"{quantity}_max" in value_drift.columns
        assert f"{quantity}_n" in value_drift.columns
        assert f"{quantity}_n_finite" in value_drift.columns
        assert f"{quantity}_relative_mae" in value_drift.columns
    firm_bellman = result.metrics["firm_bellman"]
    assert "fresh_p0_normalized_mae" in firm_bellman.columns
    assert "fresh_p0_normalized_n" in firm_bellman.columns
    assert "q_qs_claim_bellman_trainM_mae" in firm_bellman.columns


def test_duplicate_episodes_are_rejected():
    with pytest.raises(ValueError):
        compare_episodes([_record(3), _record(3)])


# ---------------------------------------------------------------------------
# thresholds / calibration / persistence
# ---------------------------------------------------------------------------


def test_threshold_config_round_trip_and_validation():
    config = ThresholdConfig.from_dict({"firm_value_drift": 0.1, "outer_extinct_paths": 0})
    round_trip = json.loads(json.dumps(config.to_dict()))
    assert round_trip["firm_value_drift"] == 0.1
    assert round_trip["outer_extinct_paths"] == 0.0
    assert round_trip["macro_fc1_one_step"] is None
    assert config.is_complete() is False
    assert "firm_bp_star_drift" in config.missing_required_gate_ids()
    assert ThresholdConfig().is_complete() is False
    with pytest.raises(ValueError):
        ThresholdConfig.from_dict({"no_such_gate": 1.0})
    with pytest.raises(ValueError):
        ThresholdConfig.from_dict({"firm_value_drift": float("inf")})


def test_build_threshold_calibration_never_picks_thresholds():
    records = [
        _record(3, seed=1),
        _record(4, seed=1),
        _record(5, seed=2),
        _record(6, seed=2),
    ]
    report = build_threshold_calibration(
        records,
        gate_ids=["firm_p_bellman_residual", "firm_value_drift", "hard_nonfinite"],
        n_boot=200,
        seed=1,
    )
    assert report["schema_version"] == "threshold_calibration_v1"
    assert report["bootstrap_config"]["cluster_unit"] == "episode / simulation path"

    value_drift = report["candidates"]["firm_value_drift"]
    assert value_drift["recommended_threshold"] is None
    assert value_drift["suggested_interval"] is not None
    low, high = value_drift["suggested_interval"]
    assert low <= high
    assert value_drift["path_clustered_bootstrap"]["n_obs"] == 2
    assert value_drift["multi_seed_monte_carlo"]["available"] is True
    assert value_drift["multi_seed_monte_carlo"]["n_seeds"] == 2
    assert "absolute_error_companion" in value_drift
    assert set(value_drift["basis"]) >= {"historical_residual_distribution_on_fixed_evalbank"}

    nonfinite = report["candidates"]["hard_nonfinite"]
    assert nonfinite["economic_feasibility"]["value"] == 0.0
    assert nonfinite["recommended_threshold"] is None
    assert nonfinite["historical_residual_distribution"]["n"] == 4

    ledger = report["candidates"]["firm_p_bellman_residual"]
    assert ledger["historical_residual_distribution"]["n"] == 4
    assert ledger["suggested_interval_reason_code"] in {
        "BOOTSTRAP_CI_OF_OBSERVED_DISTRIBUTION",
        "INSUFFICIENT_OBSERVATIONS",
    }
    with pytest.raises(ValueError):
        build_threshold_calibration(records, gate_ids=["no_such_gate"])


def test_save_comparison_writes_artifacts(tmp_path):
    result = compare_episodes([_record(3), _record(4), _record(5)], threshold_config=_thresholds())
    assert isinstance(result, ComparisonResult)
    written = save_comparison(result, tmp_path)
    payload = json.loads(Path(written["convergence_status"]).read_text(encoding="utf-8"))
    assert payload["schema_version"] == "convergence_status_v1"
    assert payload["convergence_status"] == CONVERGED_CANDIDATE
    assert payload["converged_candidate"] is True
    assert len(payload["episode_status"]) == 3
    for group in (
        "firm_bellman",
        "value_drift",
        "policy_drift",
        "macro",
        "simulation",
        "distribution",
        "comparability",
    ):
        assert (tmp_path / f"metrics_{group}.csv").is_file()
    long_frame = pd.read_csv(tmp_path / "metrics.csv")
    assert {"episode", "group", "metric", "value", "status", "reason_code"}.issubset(
        long_frame.columns
    )
    assert (tmp_path / "comparison_diagnostics.json").is_file()


