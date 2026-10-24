"""CPU tests for the macro/SDF and simulation/distribution evaluators.

Covers Prompt Test 7 (distribution accounting + capital Waterfall identity),
Test 8 (extinct paths retained and annotated), plus the shift=0 belief-vs-realized
alignment, the production-vs-autonomous FC1 recursion calibres, Signed-AiO
negative handling and the Wasserstein / fixed-grid 2D histogram distance
identities.  Everything runs on CPU with synthetic data and scaled-down
simulation parameters -- no cachedir checkpoint is required.
"""

import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from analysis.economic_config import AnalysisEconomicConfig
from config import HyperParams
from evaluation.convergence.checkpoint_io import EvalCheckpoint, compute_param_hashes
from evaluation.convergence.macro_sdf import (
    MacroSdfEvalConfig,
    evaluate_macro_sdf,
    fc1_one_step_belief_metrics,
    fc1_recursive_stability,
    fc1_shift_alignment,
)
from evaluation.convergence.sim_eval import (
    SimEvalConfig,
    compare_fixed_shock_rollouts,
    evaluate_distributions,
    evaluate_simulation,
    evaluate_waterfall,
    joint_histogram,
    joint_histogram_distance,
    wasserstein_1,
)
from evaluation.convergence.macro_sdf import bank_d_frame, evaluate_sdf_block
from losses.sdf_loss import SDFLoss


# ---------------------------------------------------------------------------
# synthetic macro panels (hand-checkable numbers)
# ---------------------------------------------------------------------------


def _firm_row(**kwargs):
    row = {
        "path": 0,
        "t": 0,
        "branch": -1,
        "ID": "A",
        "entry": 0.0,
        "b": 0.10,
        "z": 0.0,
        "ETA": 1.0,
        "i": 0.10,
        "x": 0.0,
        "Hatcf": -1.0,
        "LnKF": 1.0,
        "K": 1.0,
        "Bar_i": 0.0,
        "Bar_i_model": 0.5,
        "Bar_z": 0.0,
        "economic_node_id": 0,
        "node_view": 0,
        "accounting_stage": 0.0,
        "initial_cohort": 1.0,
    }
    row.update(kwargs)
    return row


def _macro_row(**kwargs):
    row = {
        "path": 0,
        "t": 0,
        "branch": -1,
        "K": 1.0,
        "C": 0.5,
        "LnK": 0.0,
        "Hatc": -1.0,
        "n_firms": 1,
        "M": 1.0,
        "x": 0.0,
        "hatcf": -1.0,
        "lnkf": 0.0,
        "economic_node_id": 0,
        "node_view": 0,
        "accounting_stage": 0.0,
        "accepted_count": 0.0,
        "potential_count": 0.0,
        "same_node_entrant_exit_count": 0.0,
        "K_entry_gross": 0.0,
        "K_entry_surviving": 0.0,
        "K_entry_same_node_exit": 0.0,
        "entry_capital_event_residual": 0.0,
        "deltaK_incumbent": 0.0,
        "K_entry_endpoint": 0.0,
        "K_exit_old": 0.0,
        "capital_accounting_residual": 0.0,
        "K_decomposition_start": 0.0,
        "K_decomposition_end": 0.0,
        "decomposition_start_stage": "none",
        "decomposition_end_stage": "none",
        "decomposition_start_node_id": -1.0,
        "decomposition_end_node_id": -1.0,
        "decomposition_transition_valid": 0.0,
        "resource_feasible": 1.0,
    }
    row.update(kwargs)
    return row


def _test7_panel():
    """Small hand-checkable economy: 3 firms, 2 transitions, entry and exit.

    t=0  parents           A(1.0) B(2.0) C(3.0)                  -> K_start = 6.0
    t0->t1 incumbents      A +0.2, B -0.5, C exits (old K 3.0)
                           D enters (0.8, survives), E enters (0.4, exits same node)
    t=1  endpoint          A(1.2) B(1.5) D(0.8)                  -> K_end   = 3.5
    t1->t2 incumbents      A +0.0, B +0.5, D exits (old K 0.8)
                           F enters (1.0, survives)
    t=2  endpoint          A(1.2) B(2.0) F(1.0)                  -> K_end   = 4.2

    Identity: 3.5 - 6.0 = -0.3 + 0.8 - 3.0 + 0.0 and
              4.2 - 3.5 = +0.5 + 1.0 - 0.8 + 0.0.
    """
    firm_rows = [
        # t=0 parent view only (production exports no child view at t=0)
        _firm_row(t=0, branch=-1, ID="A", K=1.0, b=0.10, z=0.0),
        _firm_row(t=0, branch=-1, ID="B", K=2.0, b=0.20, z=0.5),
        _firm_row(t=0, branch=-1, ID="C", K=3.0, b=0.30, z=-0.5),
        # t=1 child view (pre-exit): C still visible, same-node exiting entrant E visible
        _firm_row(t=1, branch=0, ID="A", K=1.2, b=0.10, z=0.1, node_view=1),
        _firm_row(t=1, branch=0, ID="B", K=1.5, b=0.20, z=0.4, node_view=1),
        _firm_row(t=1, branch=0, ID="C", K=3.0, b=0.30, z=-0.6, node_view=1),
        _firm_row(t=1, branch=0, ID="D", K=0.8, b=0.40, z=0.2, entry=1.0, initial_cohort=0.0, node_view=1),
        _firm_row(t=1, branch=0, ID="E", K=0.4, b=0.50, z=-1.0, entry=1.0, initial_cohort=0.0, node_view=1),
        # t=1 parent view (post-exit): C and E are gone
        _firm_row(t=1, branch=-1, ID="A", K=1.2, b=0.10, z=0.1, node_view=0),
        _firm_row(t=1, branch=-1, ID="B", K=1.5, b=0.20, z=0.4, node_view=0),
        _firm_row(t=1, branch=-1, ID="D", K=0.8, b=0.40, z=0.2, entry=1.0, initial_cohort=0.0, node_view=0),
        # t=2 child view (pre-exit): D still visible
        _firm_row(t=2, branch=0, ID="A", K=1.2, b=0.10, z=0.2, node_view=1),
        _firm_row(t=2, branch=0, ID="B", K=2.0, b=0.25, z=0.6, node_view=1),
        _firm_row(t=2, branch=0, ID="D", K=0.8, b=0.40, z=0.0, entry=1.0, initial_cohort=0.0, node_view=1),
        _firm_row(t=2, branch=0, ID="F", K=1.0, b=0.60, z=0.3, entry=1.0, initial_cohort=0.0, node_view=1),
        # t=2 parent view (post-exit)
        _firm_row(t=2, branch=-1, ID="A", K=1.2, b=0.10, z=0.2, node_view=0),
        _firm_row(t=2, branch=-1, ID="B", K=2.0, b=0.25, z=0.6, node_view=0),
        _firm_row(t=2, branch=-1, ID="F", K=1.0, b=0.60, z=0.3, entry=1.0, initial_cohort=0.0, node_view=0),
    ]
    macro_rows = [
        _macro_row(t=0, branch=-1, K=6.0, n_firms=3, LnK=float(np.log(6.0)), lnkf=float(np.log(6.0)) + 0.05),
        # transition 0->1 on the child view (complete event accounting)
        _macro_row(
            t=1, branch=0, K=6.9, n_firms=5, node_view=1, economic_node_id=1,
            deltaK_incumbent=-0.3, K_entry_endpoint=0.8, K_exit_old=3.0,
            capital_accounting_residual=0.0, K_decomposition_start=6.0, K_decomposition_end=3.5,
            decomposition_start_stage="parent_pre_transition_alive",
            decomposition_end_stage="child_post_exit_endpoint",
            decomposition_start_node_id=0.0, decomposition_end_node_id=1.0,
            decomposition_transition_valid=1.0,
            K_entry_gross=1.2, K_entry_surviving=0.8, K_entry_same_node_exit=0.4,
            same_node_entrant_exit_count=1.0, entry_capital_event_residual=0.0,
            accepted_count=2.0,
        ),
        # dual parent view of the same transition: same-node exits already applied
        # (recompute hazard -> zeroed event columns); must NOT be selected.
        _macro_row(
            t=1, branch=-1, K=3.5, n_firms=3, node_view=0, economic_node_id=1,
            deltaK_incumbent=-0.3, K_entry_endpoint=0.8, K_exit_old=3.0,
            capital_accounting_residual=0.0, K_decomposition_start=6.0, K_decomposition_end=3.5,
            decomposition_start_stage="parent_pre_transition_alive",
            decomposition_end_stage="child_post_exit_endpoint",
            decomposition_start_node_id=0.0, decomposition_end_node_id=1.0,
            decomposition_transition_valid=1.0,
            K_entry_gross=1.2, K_entry_surviving=1.2, K_entry_same_node_exit=0.0,
            same_node_entrant_exit_count=0.0, entry_capital_event_residual=0.0,
            accepted_count=2.0,
        ),
        # transition 1->2 on the child view
        _macro_row(
            t=2, branch=0, K=5.0, n_firms=4, node_view=1, economic_node_id=2,
            deltaK_incumbent=0.5, K_entry_endpoint=1.0, K_exit_old=0.8,
            capital_accounting_residual=0.0, K_decomposition_start=3.5, K_decomposition_end=4.2,
            decomposition_start_stage="parent_pre_transition_alive",
            decomposition_end_stage="child_post_exit_endpoint",
            decomposition_start_node_id=1.0, decomposition_end_node_id=2.0,
            decomposition_transition_valid=1.0,
            K_entry_gross=1.0, K_entry_surviving=1.0, K_entry_same_node_exit=0.0,
            same_node_entrant_exit_count=0.0, entry_capital_event_residual=0.0,
            accepted_count=1.0,
        ),
        _macro_row(
            t=2, branch=-1, K=4.2, n_firms=3, node_view=0, economic_node_id=2,
            deltaK_incumbent=0.5, K_entry_endpoint=1.0, K_exit_old=0.8,
            capital_accounting_residual=0.0, K_decomposition_start=3.5, K_decomposition_end=4.2,
            decomposition_start_stage="parent_pre_transition_alive",
            decomposition_end_stage="child_post_exit_endpoint",
            decomposition_start_node_id=1.0, decomposition_end_node_id=2.0,
            decomposition_transition_valid=1.0,
            K_entry_gross=1.0, K_entry_surviving=1.0, K_entry_same_node_exit=0.0,
            same_node_entrant_exit_count=0.0, entry_capital_event_residual=0.0,
            accepted_count=1.0,
        ),
    ]
    return pd.DataFrame(firm_rows), pd.DataFrame(macro_rows)


# ---------------------------------------------------------------------------
# Test 7 -- distribution accounting + capital Waterfall identity
# ---------------------------------------------------------------------------


def test_7_waterfall_identity_hand_checked_accounting():
    firm_df, macro_df = _test7_panel()
    result = evaluate_waterfall(firm_df, macro_df, config=SimEvalConfig())
    table = result.tables["capital_waterfall"].sort_values(["path", "t_from"]).reset_index(drop=True)

    assert len(table) == 2
    first = table.iloc[0]
    assert (first["t_from"], first["t_to"]) == (0, 1)
    assert first["initial_capital"] == pytest.approx(6.0)
    assert first["incumbent_dk"] == pytest.approx(-0.3)
    assert first["exiting_incumbent_capital"] == pytest.approx(3.0)
    assert first["entrant_capital"] == pytest.approx(0.8)
    assert first["same_node_entrant_exit"] == pytest.approx(0.4)
    assert first["ending_capital"] == pytest.approx(3.5)
    assert first["accounting_residual"] == pytest.approx(0.0, abs=1e-9)
    # Ending = Initial + Incumbent dK + Entrant - Exiting + Residual exactly.
    rebuilt = (
        first["initial_capital"]
        + first["incumbent_dk"]
        + first["entrant_capital"]
        - first["exiting_incumbent_capital"]
        + first["accounting_residual"]
    )
    assert rebuilt == pytest.approx(first["ending_capital"], abs=1e-9)

    second = table.iloc[1]
    assert (second["t_from"], second["t_to"]) == (1, 2)
    assert second["initial_capital"] == pytest.approx(3.5)
    assert second["incumbent_dk"] == pytest.approx(0.5)
    assert second["exiting_incumbent_capital"] == pytest.approx(0.8)
    assert second["entrant_capital"] == pytest.approx(1.0)
    assert second["same_node_entrant_exit"] == pytest.approx(0.0)
    assert second["ending_capital"] == pytest.approx(4.2)
    assert second["accounting_residual"] == pytest.approx(0.0, abs=1e-9)

    metrics = result.metrics
    assert metrics["waterfall_identity_residual_abs_max"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["waterfall_identity_matches_recorded_abs_max"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["waterfall_identity_within_tolerance_share"] == pytest.approx(1.0)
    # Start / end cross-checked against the firm panel capital sums.
    assert metrics["waterfall_firm_start_crosscheck_abs_max"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["waterfall_firm_end_crosscheck_abs_max"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["waterfall_node_id_continuity_violations"] == 0
    assert metrics["waterfall_end_node_id_mismatch_count"] == 0
    assert metrics["waterfall_same_node_entrant_exit_total"] == pytest.approx(0.4)
    assert metrics["waterfall_entrant_event_residual_abs_max"] == pytest.approx(0.0, abs=1e-9)


def test_7_waterfall_dual_views_and_same_node_exit_not_double_counted():
    firm_df, macro_df = _test7_panel()
    result = evaluate_waterfall(firm_df, macro_df, config=SimEvalConfig())
    table = result.tables["capital_waterfall"].sort_values(["path", "t_from"]).reset_index(drop=True)

    # The dual-view nodes (t=1 and t=2 each exported twice) must be counted once:
    # no doubled initial capital and no duplicate transition rows.
    assert table["initial_capital"].tolist() == pytest.approx([6.0, 3.5])
    assert result.metrics["waterfall_transitions_count"] == 2
    assert result.metrics["waterfall_dual_view_rows_excluded"] >= 2

    # The same-node entrant exit (E, K=0.4) is a memo line: it is in neither
    # endpoint set, so it must not enter the identity sum.
    first = table.iloc[0]
    identity_without_memo = (
        first["initial_capital"]
        + first["incumbent_dk"]
        + first["entrant_capital"]
        - first["exiting_incumbent_capital"]
    )
    assert identity_without_memo == pytest.approx(first["ending_capital"])
    assert identity_without_memo + first["same_node_entrant_exit"] != pytest.approx(
        first["ending_capital"]
    )
    # The child view (pre-exit) carries the complete event accounting; the
    # zeroed parent-view copy must not be selected.
    assert first["same_node_entrant_exit"] == pytest.approx(0.4)
    assert first["entrant_capital_gross"] == pytest.approx(1.2)
    assert first["entrant_capital_surviving"] == pytest.approx(0.8)

    # Cross-section statistics use the parent view only: 3 living firms at t=1.
    cross_status = result.tables["path_status"]
    assert int(cross_status.loc[cross_status["path"] == 0, "n_rows"].iloc[0]) == 9
    distribution = evaluate_distributions(firm_df, macro_df, config=SimEvalConfig())
    period = distribution.tables["period_counts"].set_index("t")
    assert period.loc[1, "n_alive_firms"] == 3
    assert period.loc[1, "n_defaults_incumbent"] == 1
    assert period.loc[1, "n_same_node_entrant_exits"] == 1
    assert period.loc[1, "n_defaults"] == 2


def test_7_distribution_accounting_hand_checked():
    firm_df, macro_df = _test7_panel()
    result = evaluate_distributions(firm_df, macro_df, config=SimEvalConfig())

    period = result.tables["period_counts"].set_index("t")
    assert sorted(result.tables["period_counts"]["t"].tolist()) == [0, 1, 2]
    assert period.loc[0, "n_alive_firms"] == 3
    assert period.loc[0, "active_paths"] == 1
    assert period.loc[0, "extinct_paths"] == 0
    assert period.loc[1, "n_alive_firms"] == 3
    assert period.loc[1, "n_entrants"] == 1  # parent view: D only
    assert period.loc[1, "n_entrants_event"] == 2  # event account: D and E
    assert period.loc[2, "n_alive_firms"] == 3
    assert period.loc[2, "n_entrants_event"] == 1

    slices = result.tables["distribution_slices"]
    t0_count = slices.loc[
        (slices["t"] == 0) & (slices["caliber"] == "count") & (slices["macro_state"] == "all")
    ].iloc[0]
    assert t0_count["n_firms"] == 3
    # b = {0.1, 0.2, 0.3}: count-weighted mean 0.2, variance 0.02/3.
    assert t0_count["b_mean"] == pytest.approx(0.2)
    assert t0_count["b_var"] == pytest.approx(np.var([0.1, 0.2, 0.3]))
    # z = {0.0, 0.5, -0.5}: mean 0, variance 0.5^2*2/3
    assert t0_count["z_mean"] == pytest.approx(0.0)
    assert t0_count["z_var"] == pytest.approx(np.var([0.0, 0.5, -0.5]))
    # Cov(b, z) hand value: mean(bz) - mean(b)mean(z) = (0 + 0.1 - 0.15)/3
    assert t0_count["cov_bz"] == pytest.approx((0.0 * 0.1 + 0.5 * 0.2 + -0.5 * 0.3) / 3.0)

    t0_capital = slices.loc[
        (slices["t"] == 0) & (slices["caliber"] == "capital") & (slices["macro_state"] == "all")
    ].iloc[0]
    assert t0_capital["capital_weighted_mean_b"] == pytest.approx((0.1 * 1.0 + 0.2 * 2.0 + 0.3 * 3.0) / 6.0)
    assert t0_capital["b_mean"] == pytest.approx((0.1 * 1.0 + 0.2 * 2.0 + 0.3 * 3.0) / 6.0)

    # Histograms are normalized probabilities on the fixed grid.
    for caliber in ("count", "capital"):
        hist = result.surfaces[f"bz_hist_t0_{caliber}"]
        assert hist.sum() == pytest.approx(1.0)
        assert hist.shape == (21, 21)
    marginals = result.tables["distribution_marginals"]
    for caliber in ("count", "capital"):
        total = marginals.loc[
            (marginals["t"] == 0) & (marginals["caliber"] == caliber) & (marginals["axis"] == "b"),
            "weight",
        ].sum()
        assert total == pytest.approx(1.0)

    distances = result.tables["distribution_distances"]
    assert not distances.empty
    assert {"w1_b", "w1_z", "hist2d_l1", "hist2d_tv"}.issubset(distances.columns)
    zero = distances.loc[(distances["t_from"] == 0) & (distances["t_to"] == 0)]
    assert zero.empty  # distances are between distinct slices only


# ---------------------------------------------------------------------------
# Test 8 -- extinct paths are retained and annotated
# ---------------------------------------------------------------------------


def _test8_panel():
    """Path 1 dies at t=1 (rows stop); path 0 keeps running to t=2."""
    firm_rows = [
        _firm_row(t=0, branch=-1, ID="P0A", K=1.0, b=0.1, z=0.0),
        _firm_row(t=0, branch=-1, ID="P0B", K=1.5, b=0.2, z=0.1),
        _firm_row(t=0, branch=-1, path=1, ID="P1A", K=2.0, b=0.3, z=-0.2),
        _firm_row(t=1, branch=-1, ID="P0A", K=1.1, b=0.1, z=0.0),
        _firm_row(t=1, branch=-1, ID="P0B", K=1.4, b=0.2, z=0.1),
        _firm_row(t=2, branch=-1, ID="P0A", K=1.2, b=0.1, z=0.0),
    ]
    macro_rows = [
        _macro_row(t=0, branch=-1, K=2.5, n_firms=2, accepted_count=0.0),
        _macro_row(t=0, branch=-1, path=1, K=2.0, n_firms=1, economic_node_id=0),
        _macro_row(
            t=1, branch=-1, K=2.5, n_firms=2, economic_node_id=1,
            deltaK_incumbent=0.0, K_entry_endpoint=0.0, K_exit_old=0.0,
            capital_accounting_residual=0.0, K_decomposition_start=2.5, K_decomposition_end=2.5,
            decomposition_start_stage="parent_pre_transition_alive",
            decomposition_end_stage="child_post_exit_endpoint",
            decomposition_start_node_id=0.0, decomposition_end_node_id=1.0,
            decomposition_transition_valid=1.0,
        ),
        _macro_row(
            t=2, branch=-1, K=1.2, n_firms=1, economic_node_id=2,
            deltaK_incumbent=0.1, K_entry_endpoint=0.0, K_exit_old=1.4,
            capital_accounting_residual=0.0, K_decomposition_start=2.5, K_decomposition_end=1.2,
            decomposition_start_stage="parent_pre_transition_alive",
            decomposition_end_stage="child_post_exit_endpoint",
            decomposition_start_node_id=1.0, decomposition_end_node_id=2.0,
            decomposition_transition_valid=1.0,
        ),
    ]
    return pd.DataFrame(firm_rows), pd.DataFrame(macro_rows)


def test_8_extinct_path_is_retained_and_annotated():
    firm_df, macro_df = _test8_panel()
    result = evaluate_distributions(firm_df, macro_df, config=SimEvalConfig())

    period = result.tables["period_counts"].set_index("t")
    assert period.loc[0, "active_paths"] == 2
    assert period.loc[1, "active_paths"] == 1
    assert period.loc[1, "extinct_paths"] == 1
    assert period.loc[1, "extinct_path_ids"] == "1"
    assert period.loc[2, "extinct_paths"] == 1
    assert period.loc[2, "extinct_path_ids"] == "1"

    status = result.tables["path_status"].set_index("path")
    assert bool(status.loc[1, "extinct"]) is True
    assert int(status.loc[1, "stop_time"]) == 1
    assert int(status.loc[1, "last_t"]) == 0
    assert bool(status.loc[0, "extinct"]) is False
    assert result.metrics["dist_extinct_paths_total"] == 1

    # Distribution statistics at t>=1 are computed over the surviving path but
    # the extinct path is explicitly annotated, never silently dropped.
    slices = result.tables["distribution_slices"]
    slice_t2 = slices.loc[
        (slices["t"] == 2) & (slices["caliber"] == "count") & (slices["macro_state"] == "all")
    ].iloc[0]
    assert slice_t2["n_firms"] == 1
    assert slice_t2["extinct_paths"] == 1
    assert "1" in str(slice_t2["extinct_path_ids"])

    # The waterfall keeps the surviving transitions and reports the missing
    # ones through the per-path status table instead of fabricating rows.
    waterfall = evaluate_waterfall(firm_df, macro_df, config=SimEvalConfig())
    table = waterfall.tables["capital_waterfall"]
    assert sorted(table["path"].unique().tolist()) == [0]
    waterfall_status = waterfall.tables["path_status"].set_index("path")
    assert bool(waterfall_status.loc[1, "extinct"]) is True
    assert int(waterfall_status.loc[1, "stop_time"]) == 1


# ---------------------------------------------------------------------------
# synthetic FC1/SDF stand-in with a fully controlled forecast rule
# ---------------------------------------------------------------------------


class _FakeSdfParams(torch.nn.Module):
    def __init__(self, gamma: float, kappa: float, sigma: float, beta: float):
        super().__init__()
        self.gamma = float(gamma)
        self.kappa = float(kappa)
        self.sigma = float(sigma)
        self.beta = float(beta)
        self.register_buffer("probe", torch.zeros(1))


class _FakeSDFFC1(torch.nn.Module):
    """Deterministic FC1/SDF stand-in.

    Forecast rule ``hatc_next = hatc_slope * hatcf_prev + hatc_step`` and
    ``lnk_next = lnk_slope * lnkf_prev + lnk_step`` (plus an optional x-change
    gain), so the production vs autonomous recursion calibres are analytically
    distinguishable; ``w_child_gain`` alternates child wealth values so the
    Signed AiO of the first two children is negative by construction.
    """

    def __init__(
        self,
        *,
        hatc_slope: float = 0.0,
        hatc_step: float = 0.0,
        lnk_slope: float = 0.0,
        lnk_step: float = 0.0,
        x_gain: float = 0.0,
        w_base: float = 1.5,
        w_child_gain: float = 0.0,
        m_base: float = 1.0,
        gamma: float = 2.0,
        kappa: float = 1.0,
        sigma: float = 1.0,
        beta: float = 0.96,
    ):
        super().__init__()
        self.sdf_model = _FakeSdfParams(gamma, kappa, sigma, beta)
        self.hatc_slope = float(hatc_slope)
        self.hatc_step = float(hatc_step)
        self.lnk_slope = float(lnk_slope)
        self.lnk_step = float(lnk_step)
        self.x_gain = float(x_gain)
        self.w_base = float(w_base)
        self.w_child_gain = float(w_child_gain)
        self.m_base = float(m_base)
        self.register_buffer("probe", torch.zeros(1))

    @staticmethod
    def _child_pattern(n_children: int, batch: int, device, dtype) -> torch.Tensor:
        pattern = torch.arange(n_children, device=device, dtype=dtype)
        pattern = torch.where(pattern % 2 == 0, torch.ones_like(pattern), -torch.ones_like(pattern))
        return pattern.view(1, n_children, 1).expand(batch, n_children, 1)

    def _predict(self, x_prev, x_curr, hatcf_prev, lnkf_prev):
        hatc_next = self.hatc_slope * hatcf_prev + self.hatc_step + self.x_gain * (x_curr - x_prev)
        lnk_next = self.lnk_slope * lnkf_prev + self.lnk_step + self.x_gain * (x_curr - x_prev)
        return hatc_next, lnk_next

    def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical: bool = True):
        def _col(value):
            return value if value.dim() > 1 else value.unsqueeze(-1)

        x_prev = _col(x_prev)
        x_curr = _col(x_curr)
        hatcf_prev = _col(hatcf_prev)
        lnkf_prev = _col(lnkf_prev)

        if x_curr.dim() == 3:
            batch, n_children, _ = x_curr.shape
            hp = hatcf_prev[:, 0, :] if hatcf_prev.dim() == 3 else hatcf_prev
            lp = lnkf_prev[:, 0, :] if lnkf_prev.dim() == 3 else lnkf_prev
            xp = x_prev[:, 0, :] if x_prev.dim() == 3 else x_prev
            hatc_next, lnk_next = self._predict(
                xp.unsqueeze(1), x_curr, hp.unsqueeze(1), lp.unsqueeze(1)
            )
            w_parent = self.w_base + 0.1 * hp
            w_children = (
                self.w_base
                + 0.1 * hatc_next
                + self.w_child_gain
                * self._child_pattern(n_children, batch, x_curr.device, x_curr.dtype)
            )
            m_raw = torch.full_like(hatc_next, self.m_base)
            return w_parent, w_children, m_raw, hatc_next, lnk_next

        hatc_next, lnk_next = self._predict(x_prev, x_curr, hatcf_prev, lnkf_prev)
        w_parent = self.w_base + 0.1 * hatcf_prev
        w_children = self.w_base + 0.1 * hatc_next
        m_raw = torch.full_like(hatc_next, self.m_base)
        return w_parent, w_children, m_raw, hatc_next, lnk_next


def _fake_checkpoint(model: torch.nn.Module) -> EvalCheckpoint:
    return EvalCheckpoint(
        models={"sdf_fc1": model},
        hyperparams=HyperParams(),
        economic_config=AnalysisEconomicConfig.from_current_config(),
        metadata={},
        episode=0,
        checkpoint_path="<synthetic>",
        param_hashes={},
    )


def _bank_d_payload(path_series: Sequence[dict]) -> dict:
    """Bank D rows from per-path node series ``{x, hatc, lnk, hatcf, lnkf}``."""
    columns = {name: [] for name in ("x_t", "x_t1", "hatcf_t", "lnkf_t", "hatc_t", "lnk_t", "hatc_t1", "lnk_t1")}
    keys = {name: [] for name in ("path", "t", "branch", "economic_node_id", "t1", "economic_node_id_child")}
    node_id = 0
    for path, series in enumerate(path_series):
        n_nodes = len(series["x"])
        for t in range(n_nodes - 1):
            columns["x_t"].append(float(series["x"][t]))
            columns["x_t1"].append(float(series["x"][t + 1]))
            columns["hatcf_t"].append(float(series["hatcf"][t]))
            columns["lnkf_t"].append(float(series["lnkf"][t]))
            columns["hatc_t"].append(float(series["hatc"][t]))
            columns["lnk_t"].append(float(series["lnk"][t]))
            columns["hatc_t1"].append(float(series["hatc"][t + 1]))
            columns["lnk_t1"].append(float(series["lnk"][t + 1]))
            keys["path"].append(path)
            keys["t"].append(t)
            keys["branch"].append(0)
            keys["economic_node_id"].append(node_id)
            keys["t1"].append(t + 1)
            keys["economic_node_id_child"].append(node_id + 1)
            node_id += 1
    payload = dict(columns)
    payload["pairing_keys"] = keys
    return payload


def _constant_series(n_nodes: int, value: float = 0.0) -> dict:
    return {
        "x": [0.0] * n_nodes,
        "hatc": [float(value)] * n_nodes,
        "lnk": [float(value)] * n_nodes,
        "hatcf": [float(value)] * n_nodes,
        "lnkf": [float(value)] * n_nodes,
    }


# ---------------------------------------------------------------------------
# FC1 one-step alignment (shift = 0) and key-based pairing
# ---------------------------------------------------------------------------


def test_fc1_shift0_alignment_is_key_based_not_row_adjacent():
    n_nodes = 9
    hatc = [float(np.sin(0.7 * t)) for t in range(n_nodes)]
    lnk = [float(np.cos(0.4 * t)) for t in range(n_nodes)]
    series = {
        "x": [0.1 * t for t in range(n_nodes)],
        "hatc": hatc,
        "lnk": lnk,
        "hatcf": [value + 0.1 for value in hatc],  # belief over-predicts by 0.1
        "lnkf": [value - 0.05 for value in lnk],
    }
    payload = _bank_d_payload([series])
    frame = bank_d_frame(payload)
    config = MacroSdfEvalConfig()

    metrics, _, diagnostics = fc1_one_step_belief_metrics(frame, config=config)
    assert metrics["fc1_one_step_hatc_bias"] == pytest.approx(0.1, abs=1e-9)
    assert metrics["fc1_one_step_hatc_mae"] == pytest.approx(0.1, abs=1e-9)
    assert metrics["fc1_one_step_lnk_bias"] == pytest.approx(-0.05, abs=1e-9)
    assert diagnostics["fc1_pairing"]["shift"] == 0

    shift_metrics, shift_tables = fc1_shift_alignment(frame, config=config)
    assert shift_metrics["fc1_shift_hatc_best_shift"] == 0
    assert shift_metrics["fc1_shift_hatc_shift0_rmse"] == pytest.approx(0.1, abs=1e-9)

    # Pairing is by key, not by row adjacency: reversing the row order of the
    # bank must leave every metric unchanged.
    reversed_payload = {
        key: list(values)[::-1]
        for key, values in payload.items()
        if key != "pairing_keys"
    }
    reversed_payload["pairing_keys"] = {
        key: list(values)[::-1] for key, values in payload["pairing_keys"].items()
    }
    reversed_frame = bank_d_frame(
        reversed_payload
    )
    reversed_metrics, _, _ = fc1_one_step_belief_metrics(reversed_frame, config=config)
    for key in ("fc1_one_step_hatc_bias", "fc1_one_step_hatc_mae", "fc1_one_step_lnk_bias"):
        assert reversed_metrics[key] == pytest.approx(metrics[key], abs=1e-12)


# ---------------------------------------------------------------------------
# FC1 recursion: production_one_step vs autonomous_recursive
# ---------------------------------------------------------------------------


def test_fc1_production_and_autonomous_recursion_are_distinguishable():
    n_nodes = 11
    series = _constant_series(n_nodes, value=0.0)
    payload = _bank_d_payload([series])
    frame = bank_d_frame(payload)
    model = _FakeSDFFC1(hatc_slope=1.0, hatc_step=1.0, lnk_slope=1.0, lnk_step=1.0)
    cp = _fake_checkpoint(model)
    config = MacroSdfEvalConfig(
        recursion_horizons=(1, 2, 5, 10),
        recursion_path_horizon=5,
    )
    anchors = {
        "Recession": {"x": 0.0, "hatc_cal": 0.0, "lnk_cal": 0.0},
        "Normal": {"x": 0.0, "hatc_cal": 0.0, "lnk_cal": 0.0},
        "Expansion": {"x": 0.0, "hatc_cal": 0.0, "lnk_cal": 0.0},
    }

    metrics, tables, surfaces, diagnostics = fc1_recursive_stability(
        cp, frame, anchors, config=config
    )

    # production_one_step: every call is fed the realized parent state, so the
    # horizon error is the constant one-step error at every horizon.
    for horizon in (1, 2, 5, 10):
        assert metrics[f"fc1_rec_production_one_step_hatc_h{horizon}_bias"] == pytest.approx(
            1.0, abs=1e-6
        )
        assert metrics[f"fc1_rec_production_one_step_lnk_h{horizon}_bias"] == pytest.approx(
            1.0, abs=1e-6
        )
    assert metrics["fc1_rec_production_one_step_hatc_h1_rmse"] == pytest.approx(
        metrics["fc1_rec_production_one_step_hatc_h10_rmse"], abs=1e-9
    )

    # autonomous_recursive: the model's own forecast is fed back, so the error
    # compounds linearly with the horizon.
    assert metrics["fc1_rec_autonomous_recursive_hatc_h1_bias"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["fc1_rec_autonomous_recursive_hatc_h10_bias"] == pytest.approx(10.0, abs=1e-5)
    assert metrics["fc1_rec_autonomous_recursive_lnk_h10_bias"] == pytest.approx(10.0, abs=1e-5)
    assert (
        metrics["fc1_rec_autonomous_recursive_hatc_h10_rmse"]
        > metrics["fc1_rec_autonomous_recursive_hatc_h1_rmse"]
    )
    assert metrics["fc1_rec_autonomous_recursive_hatc_out_of_training_range_share"] > 0.0

    # Anchor recursion paths under different macro initial states.
    assert set(tables.keys()) >= {
        "fc1_anchor_recursion_recession",
        "fc1_anchor_recursion_normal",
        "fc1_anchor_recursion_expansion",
    }
    assert metrics["fc1_anchor_recession_lnk_drift_per_step"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["fc1_anchor_recession_diverged"] == 0
    assert metrics["fc1_anchor_recession_degenerate_constant"] == 0
    assert metrics["fc1_anchor_recession_out_of_training_range_share"] > 0
    assert surfaces["fc1_anchor_recursion_recession"].shape == (6, 2)
    assert diagnostics["fc1_recursion_calibres"]["pooled"] is False


# ---------------------------------------------------------------------------
# SDF: Signed AiO negative values are reported with standard errors, never
# mis-stated as a negative population conditional moment
# ---------------------------------------------------------------------------


def test_sdf_signed_aio_negative_reports_estimate_and_se():
    series_a = {
        "x": [0.0, 0.1, 0.2],
        "hatc": [-1.0, -1.0, -1.0],
        "lnk": [1.0, 1.0, 1.0],
        "hatcf": [-1.0, -1.0, -1.0],
        "lnkf": [1.0, 1.0, 1.0],
    }
    series_b = dict(series_a)
    series_b["hatc"] = [-1.2, -1.2, -1.2]
    series_b["hatcf"] = [-1.2, -1.2, -1.2]
    frame = bank_d_frame(_bank_d_payload([series_a, series_b]))
    model = _FakeSDFFC1(
        hatc_slope=0.0, hatc_step=-1.0, lnk_slope=1.0, lnk_step=0.0,
        w_base=1.5, w_child_gain=5.0,
    )
    cp = _fake_checkpoint(model)
    config = MacroSdfEvalConfig(sdf_child_counts=(2, 4), sdf_shock_bank_max_children=4)

    metrics, tables, diagnostics = evaluate_sdf_block(cp, frame, config=config)

    # The sample mean of the Signed AiO is negative here and must be reported as
    # a sample estimate together with its standard error.
    assert metrics["sdf_signed_aio_estimate"] < 0.0
    assert math.isfinite(metrics["sdf_signed_aio_se"])
    assert metrics["sdf_signed_aio_se"] > 0.0
    assert metrics["sdf_signed_aio_n"] == float(len(frame))
    assert metrics["sdf_signed_aio_estimate"] == pytest.approx(
        metrics["sdf_signed_aio_loss_helper_estimate"], abs=1e-4
    )
    assert metrics["sdf_signed_aio_negative_share"] > 0.0

    # ... never as a negative population conditional moment.
    guard = diagnostics["sdf_signed_aio_interpretation_guard"]
    assert guard["negative_sample_mean_is_not_negative_population_moment"] is True
    assert guard["reported_estimate"] == (
        "sdf_signed_aio_estimate (sample mean of r_1 * r_2 across parents)"
    )
    assert guard["reported_standard_error"] == "sdf_signed_aio_se"

    # Population target estimate + standard error, numerical safety constraints
    # and the moment-constraint deviations are reported as separate blocks.
    assert math.isfinite(metrics["sdf_population_target_estimate"])
    assert math.isfinite(metrics["sdf_population_target_se"])
    assert metrics["sdf_population_target_se"] > 0.0
    for key in (
        "sdf_nonfinite_ratio",
        "sdf_nonfinite_parent_ratio",
        "sdf_log_R_clip_share",
        "sdf_valid_parent_ratio",
    ):
        assert key in metrics
        assert math.isfinite(metrics[key])
    assert metrics["sdf_nonfinite_ratio"] == pytest.approx(0.0, abs=1e-12)
    for key in ("sdf_g_mean_low", "sdf_g_mean_high", "sdf_g_var_high", "sdf_g_max"):
        assert key in metrics
    assert math.isfinite(metrics["sdf_M_mean"])
    assert math.isfinite(metrics["sdf_M_second_moment"])
    assert math.isfinite(metrics["sdf_M_mean_se"])
    # Conditional moment error and per-macro-state conditional distributions.
    assert "sdf_normalized_cm_mse" in metrics
    assert "sdf_normalized_u_stat" in metrics
    assert "sdf_macro_state_summary" in tables
    assert {"sdf_k2_sdf_M_mean", "sdf_k4_sdf_M_mean"}.issubset(set(metrics.keys()))


# ---------------------------------------------------------------------------
# distribution distances on known shifts
# ---------------------------------------------------------------------------


def test_distribution_distances_on_known_shifts():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.0, 1.0, size=200)
    b = a + 0.5
    assert wasserstein_1(a, b) == pytest.approx(0.5, abs=1e-12)
    assert wasserstein_1(a, a) == pytest.approx(0.0, abs=1e-12)

    z = rng.normal(0.0, 1.0, size=200)
    assert wasserstein_1(z, z - 0.25) == pytest.approx(0.25, abs=1e-12)

    # weighted two-point distributions: {0,1} vs {0,2} -> |1-2| / 2 = 0.5
    assert wasserstein_1(
        [0.0, 1.0], [0.0, 2.0], weights_a=[1.0, 1.0], weights_b=[1.0, 1.0]
    ) == pytest.approx(0.5, abs=1e-12)

    b_edges = np.linspace(0.0, 2.0, 21)
    z_edges = np.linspace(-1.0, 1.0, 21)
    hist_a = joint_histogram(np.zeros(50), np.zeros(50), b_edges=b_edges, z_edges=z_edges)
    hist_b = joint_histogram(np.ones(50), np.zeros(50), b_edges=b_edges, z_edges=z_edges)
    distance = joint_histogram_distance(hist_a, hist_b)
    assert distance["hist2d_l1"] == pytest.approx(2.0)
    assert distance["hist2d_tv"] == pytest.approx(1.0)
    assert distance["hist2d_overlap"] == pytest.approx(0.0)
    same = joint_histogram_distance(hist_a, hist_a)
    assert same["hist2d_l1"] == pytest.approx(0.0)
    assert same["hist2d_tv"] == pytest.approx(0.0)
    assert same["hist2d_overlap"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# integration: full macro/SDF block on a synthetic Bank D
# ---------------------------------------------------------------------------


def test_evaluate_macro_sdf_end_to_end_synthetic():
    series_a = {
        "x": [0.0, 0.1, -0.1, 0.2, 0.0, 0.1],
        "hatc": [-1.0 + 0.05 * t for t in range(6)],
        "lnk": [1.0 + 0.02 * t for t in range(6)],
        "hatcf": [-1.0 + 0.05 * t + 0.02 for t in range(6)],
        "lnkf": [1.0 + 0.02 * t + 0.01 for t in range(6)],
    }
    series_b = dict(series_a)
    series_b["hatc"] = [-1.3 + 0.03 * t for t in range(6)]
    series_b["hatcf"] = [-1.3 + 0.03 * t - 0.01 for t in range(6)]
    payload = _bank_d_payload([series_a, series_b])
    model = _FakeSDFFC1(
        hatc_slope=0.5, hatc_step=-0.2, lnk_slope=0.9, lnk_step=0.05, w_child_gain=2.0
    )
    cp = _fake_checkpoint(model)
    anchors = {
        "Recession": {"x": -0.5, "hatc_cal": -1.5, "lnk_cal": 0.5},
        "Normal": {"x": 0.0, "hatc_cal": -1.0, "lnk_cal": 1.0},
        "Expansion": {"x": 0.5, "hatc_cal": -0.5, "lnk_cal": 1.5},
    }
    config = MacroSdfEvalConfig(
        sdf_child_counts=(2, 4),
        sdf_shock_bank_max_children=4,
        recursion_horizons=(1, 2, 5),
        recursion_path_horizon=4,
    )

    result = evaluate_macro_sdf(cp, payload, anchors, config=config)
    metrics = result.metrics

    for key in (
        "fc1_one_step_hatc_mae",
        "fc1_one_step_lnk_bias",
        "fc1_one_step_hatc_mae_recession",
        "fc1_one_step_hatc_mae_cap_small",
        "fc1_model_one_step_hatc_mae",
        "fc1_eval_fc1_hatc_rmse",
        "fc1_shift_hatc_best_shift",
        "fc1_rec_production_one_step_hatc_h5_bias",
        "fc1_rec_autonomous_recursive_hatc_h5_bias",
        "fc1_anchor_normal_lnk_final",
        "sdf_signed_aio_estimate",
        "sdf_signed_aio_se",
        "sdf_population_target_se",
        "sdf_M_mean",
        "sdf_g_max",
        "sdf_nonfinite_ratio",
    ):
        assert key in metrics, key

    assert result.diagnostics["optimizer_steps"] == 0
    assert result.diagnostics["evaluation_only"] is True
    assert result.diagnostics["belief_vs_realized"]["never_mixed"] is True
    assert "sdf_signed_aio_interpretation_guard" in result.diagnostics
    assert "fc1_recursion_calibres" in result.diagnostics
    assert "fc1_anchor_recursion_normal" in result.surfaces
    assert "fc1_one_step_pairs" in result.tables


# ---------------------------------------------------------------------------
# scaled-down CPU smoke of the fixed-rollout simulation driver
# ---------------------------------------------------------------------------


def _bias_last_linear(module: torch.nn.Module, bias: float) -> None:
    """Force a head's output to a constant (zero out its last layer weights)."""
    last = None
    for child in module.modules():
        if isinstance(child, torch.nn.Linear):
            last = child
    assert last is not None
    with torch.no_grad():
        last.weight.mul_(0.0)
        last.bias.fill_(float(bias))


def test_simulation_fixed_rollout_smoke():
    from evaluation.convergence.eval_bank import build_bank_e
    from experiments.run_utils import build_models

    device = torch.device("cpu")
    models = build_models(device)
    policy = models["policy_value"]
    # Healthy-economy micro model: survive (Phat > 0 -> bar_z ~ 0) and do not
    # invest (VI < V0 -> bar_i ~ 0), so the scaled-down rollout stays feasible.
    _bias_last_linear(policy.v0_head, 3.0)
    _bias_last_linear(policy.vi_head, -3.0)

    cp = EvalCheckpoint(
        models=models,
        hyperparams=HyperParams(),
        economic_config=AnalysisEconomicConfig.from_current_config(),
        metadata={},
        episode=0,
        checkpoint_path="<synthetic>",
        param_hashes={},
    )
    bank_e = build_bank_e(
        seed=7,
        n_firms=4,
        n_paths=2,
        horizon=3,
        economic_config=cp.economic_config,
        k0_total=1.0,
    )
    for variant in bank_e["variants"].values():
        n_rows = int(torch.as_tensor(variant["x0"]).numel())
        variant["x0"] = torch.full((n_rows,), 1.0)
        variant["z0"] = torch.zeros(n_rows)

    before = compute_param_hashes(cp.models)
    config = SimEvalConfig(
        horizon=3,
        variants=("Baseline",),
        capacity_extra_per_step=20,
        calibrate_initial_hstar=True,
        node_accounting_mode="economic_node_ledger",
    )
    result = evaluate_simulation(cp, bank_e, config=config)
    after = compute_param_hashes(cp.models)

    # Evaluation-Only: no parameter update, zero optimizer steps.
    assert before == after
    assert result.diagnostics["optimizer_steps"] == 0

    rollout = result.diagnostics["rollouts"]["Baseline"]
    assert rollout["completed_horizon"] == 3
    assert rollout["stopped_early"] is False
    assert isinstance(rollout["extinct_paths"], list)
    assert isinstance(rollout["path_stop_times"], list)
    assert rollout["shock_spec"]["rng_mode"] == "stable_firm_identity"
    assert len(rollout["initial_hstar"]["selected_hstar_per_path"]) == 2
    assert math.isfinite(float(rollout["resource_feasible_share"]))
    assert math.isfinite(float(rollout["nonfinite_share"]))

    # The capital Waterfall identity holds on the produced panel.
    assert result.metrics["sim_baseline_waterfall_identity_residual_abs_max"] < 1e-3
    waterfall = result.tables["capital_waterfall_baseline"]
    assert not waterfall.empty
    assert set(["initial_capital", "incumbent_dk", "exiting_incumbent_capital",
                "entrant_capital", "same_node_entrant_exit", "ending_capital",
                "accounting_residual"]).issubset(set(waterfall.columns))
    assert result.metrics["sim_baseline_dist_n_slices"] >= 1

    fixed = result.tables["fixed_shock_metrics"]
    assert len(fixed) == 1
    assert float(fixed.iloc[0]["k0_total"]) == pytest.approx(1.0, abs=1e-5)
    assert math.isfinite(float(fixed.iloc[0]["fc1_lnk_bias"]))
    assert math.isfinite(float(fixed.iloc[0]["capital_change_t0_t1"]))

    # Cross-checkpoint comparison on one fixed bank (same result twice -> all
    # deltas against the reference checkpoint must be exactly zero).
    comparison = compare_fixed_shock_rollouts({"EP3": result, "EP4": result})
    assert len(comparison) == 2
    delta_columns = [name for name in comparison.columns if name.endswith("_delta_vs_reference")]
    assert delta_columns
    values = comparison[delta_columns].to_numpy(dtype=np.float64)
    assert np.nanmax(np.abs(values)) == pytest.approx(0.0, abs=1e-12)
