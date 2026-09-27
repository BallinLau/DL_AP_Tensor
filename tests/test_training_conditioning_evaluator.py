"""Regression tests for experiments/evaluate_training_conditioning.py (§25).

TEST 1-11 follow the evaluator spec; TEST 12/13 cover the two additional
pieces of new evaluator logic (the "Episode N done" payload parser and the
§22 per-eta metrics flattening).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from analysis.economic_config import AnalysisEconomicConfig
from config import HyperParams
from evaluation.bp_diagnostics import build_frozen_transition_data
from evaluation.firm_surfaces import evaluate_firm_surfaces
from evaluation.full_run_diagnostics import model_state_hash
from evaluation.grids import ReferenceFirmState, build_frozen_grid
from experiments.evaluate_training_conditioning import (
    DEFAULT_ANCHOR_B,
    add_eta_ratio_columns,
    build_low_b_values,
    confusion_metrics,
    finite_difference_stats,
    flatten_conditioning_row,
    parse_conditioning_training_log,
    teacher_surfaces_over_i_grid,
    value_scale_grid,
)
from losses import P0Loss, PILoss
from models.policy_value import PolicyValueModel
from training.bp_grid_teacher import BPGridTeacher


def _reference() -> ReferenceFirmState:
    return ReferenceFirmState(
        eta=1.0, i_low=0.1, i_mid=0.2, i_high=0.3, x=-2.0,
        hatcf=-2.1, lnkf=4.0, hatc_cal=-2.0, lnk_cal=4.1,
        n_parent_rows=2, source="fixture", macro_source="fixture",
    )


def _state(b_values=(0.0, 0.2, 0.8)) -> torch.Tensor:
    rows = []
    for b in b_values:
        rows.append([b, 0.1, 1.0, 0.2, 0.0, -2.0, 4.0])
    return torch.tensor(rows, dtype=torch.float32)


def _small_model(mode: str) -> PolicyValueModel:
    return PolicyValueModel(
        q_parameterization=mode,
        share_hidden_dims=[4],
        share_output_dim=4,
        q_head_dims=[4],
        p0_head_dims=[4],
        pi_head_dims=[4],
        bp0_head_dims=[4],
        bpi_head_dims=[4],
        barz_hidden_dims=[4],
        bari_hidden_dims=[4],
        i_grid_size=2,
    )


def _small_grid() -> "tuple[ReferenceFirmState, object]":
    reference = _reference()
    grid = build_frozen_grid(
        reference, b_min=0.0, b_max=0.5, b_points=4,
        z_min=-0.5, z_max=0.5, z_points=3, device=torch.device("cpu"),
    )
    return reference, grid


class _ToySDF(torch.nn.Module):
    """Deterministic SDF stub: child draws depend on the parent state only."""

    def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical=True):
        batch, children, _ = x_curr.shape
        w_parent = 10.0 + x_prev
        w_children = 10.0 + x_curr
        m = 0.95 + 0.02 * torch.tanh(x_curr)
        hatc = hatcf_prev.unsqueeze(1).expand(batch, children, 1) + x_curr
        lnk = lnkf_prev.unsqueeze(1).expand(batch, children, 1) + 0.5 * x_curr
        return w_parent, w_children, m, hatc, lnk


# ---------------------------------------------------------------------------
# TEST 1 — hybrid Q surface exposes the five required fields
# ---------------------------------------------------------------------------


def test_hybrid_surfaces_expose_q_components_q_effective_recovery_and_phat():
    model = _small_model("hybrid_regime")
    reference, grid = _small_grid()

    surfaces = evaluate_firm_surfaces(model, grid, reference)

    for name in ("q_unit", "Q_claim", "Q_effective", "recovery", "Phat"):
        assert name in surfaces, name
        assert surfaces[name].shape == grid.shape
    # Q_effective mirrors the raw Q surface; hybrid parameterization keeps
    # Q_claim = b * q_unit on every grid row (including the b=0 row).
    assert np.array_equal(surfaces["Q_effective"], surfaces["Q"])
    np.testing.assert_allclose(
        surfaces["Q_claim"], grid.mesh_b * surfaces["q_unit"], rtol=1e-5, atol=1e-7
    )


# ---------------------------------------------------------------------------
# TEST 2 — low-b grid always contains the conditioning anchors
# ---------------------------------------------------------------------------


def test_low_b_grid_contains_required_conditioning_anchors():
    for anchor in (0.0, 0.005, 0.01, 0.025, 0.05):
        assert anchor in DEFAULT_ANCHOR_B, anchor
    values = build_low_b_values(0.05, 5, (0.0, 0.005, 0.01, 0.025, 0.05))
    for anchor in (0.0, 0.005, 0.01, 0.025, 0.05):
        assert bool(np.isclose(values, anchor).any()), anchor
    assert np.all(np.diff(values) > 0), "low-b grid must be sorted and unique"
    assert values[0] == 0.0
    assert values[-1] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# TEST 3 — structural zero: Q_claim(b=0) == 0 without requiring q_unit == 0
# ---------------------------------------------------------------------------


def test_structural_zero_q_claim_at_b0_does_not_require_zero_q_unit():
    model = _small_model("hybrid_regime")
    with torch.no_grad():
        for parameter in model.q_head.parameters():
            parameter.fill_(0.37)  # generic nonzero q_unit head
    state = _state((0.0, 0.2, 0.8))

    unit = model._q_unit_output(state)
    claim = model._q_claim_output(state)

    assert torch.isfinite(unit).all()
    assert claim[0].item() == 0.0
    torch.testing.assert_close(claim[1:], state[1:, 0:1] * unit[1:])


# ---------------------------------------------------------------------------
# TEST 4 — normalized P residual uses the production equity_value_scale helper
# ---------------------------------------------------------------------------


def test_value_scale_uses_production_equity_value_scale_helper():
    model = _small_model("hybrid_regime")
    _, grid = _small_grid()

    scale = value_scale_grid(model, grid)
    expected = (
        model.equity_value_scale(grid.base_states).detach()
        .reshape(grid.shape).cpu().numpy().astype(np.float64)
    )
    np.testing.assert_allclose(scale, expected, rtol=1e-6)

    class _MarkerScale(torch.nn.Module):
        def equity_value_scale(self, states):
            return torch.full((states.shape[0],), 7.25)

    marker = value_scale_grid(_MarkerScale(), grid)
    assert marker.shape == grid.shape
    assert np.allclose(marker, 7.25)

    # The per-episode normalization broadcasts the (b, z) scale over the
    # leading i axis of the residual stack (r_norm = r_phys / scale[None]).
    residual = np.ones((2, *grid.shape))
    normalized = residual / scale[None, :, :]
    assert normalized.shape == residual.shape
    np.testing.assert_allclose(normalized[:, 0, 0], 1.0 / scale[0, 0])


# ---------------------------------------------------------------------------
# TEST 5 — children depend on parent (x, z) context, never on parent b/eta/i
# ---------------------------------------------------------------------------


def test_transition_children_do_not_depend_on_parent_b_eta_or_i():
    hyperparams = HyperParams()
    hyperparams.pv_use_clipped_m = True
    economic = AnalysisEconomicConfig.from_current_config()
    parents = torch.tensor(
        [
            # b, z, ETA, i, x, Hatcf, LnKF
            [0.2, 0.1, 0.0, 0.15, -2.0, -2.1, 4.0],
            [0.7, 0.1, 1.0, 0.90, -2.0, -2.1, 4.0],
            [0.2, 0.1, 0.0, 0.15, -1.0, -2.1, 4.0],
        ],
        dtype=torch.float32,
    )

    transition = build_frozen_transition_data(
        _ToySDF(), parents, _reference(), hyperparams, economic,
        n_child_shocks=4, shock_seed=77,
    )

    children = transition.stacked_children()
    # Column 0 (child b) intentionally inherits the parent leverage; every
    # exogenous feature column must be independent of parent b/eta/i.
    torch.testing.assert_close(children[0, :, 1:], children[1, :, 1:])
    assert not torch.allclose(children[0, :, 1:], children[2, :, 1:])


# ---------------------------------------------------------------------------
# TEST 6 — eta_t = 0 teacher uses forced semantics (single candidate b_parent)
# ---------------------------------------------------------------------------


def test_eta0_teacher_uses_forced_semantics():
    model = _small_model("hybrid_regime")
    teacher = BPGridTeacher(
        model, P0Loss(), PILoss(), q_target_model=model, coarse_size=3, refine=False
    )
    parent = _state((0.2, 0.4, 0.6))
    parent[:, 2] = 0.0
    child = parent.clone()
    children = torch.stack([child, child], dim=1)
    m = torch.ones(3, 2, 1)

    results = teacher.compute_multi_j_branches(
        [parent], children, m, branches=["p0"], prefix_child_counts=[2]
    )
    result = results[0][2]

    assert result["refi_active"].reshape(-1).tolist() == [0.0, 0.0, 0.0]
    assert torch.all(result["argmax_index"] == 0)
    torch.testing.assert_close(result["bp_star"], parent[:, 0:1])
    torch.testing.assert_close(result["regret"], torch.zeros_like(result["regret"]))
    # Grid-shaped hybrid diagnostics are NaN/sentinel on the forced path.
    assert torch.isnan(result["coarse_q_issue_claim_grid"]).all()
    assert torch.isnan(result["candidate_phat_gate_used_for_q_issue"]).all()
    assert "value_star" in result  # the i-grid teacher reads value_star


# ---------------------------------------------------------------------------
# TEST 7 — eta_t = 1 teacher uses the candidate bp grid
# ---------------------------------------------------------------------------


def test_eta1_teacher_uses_candidate_grid():
    model = _small_model("hybrid_regime")
    teacher = BPGridTeacher(
        model, P0Loss(), PILoss(), q_target_model=model, coarse_size=3, refine=False
    )
    parent = _state((0.2, 0.4, 0.6))
    parent[:, 2] = 1.0
    child = parent.clone()
    children = torch.stack([child, child], dim=1)
    m = torch.ones(3, 2, 1)

    results = teacher.compute_multi_j_branches(
        [parent], children, m, branches=["pi"], prefix_child_counts=[2]
    )
    result = results[0][2]

    assert result["refi_active"].reshape(-1).tolist() == [1.0, 1.0, 1.0]
    assert result["bp_grid"].shape == (3, 3)  # coarse candidate grid (no refine)
    for key in (
        "q_issue_claim_grid",
        "q_issue_unit_grid",
        "q_issue_realized_default_mask_grid",
        "q_issue_recovery_grid",
        "q_issue_candidate_phat_grid",
        "candidate_phat_gate_used_for_q_issue",
    ):
        assert key in result, key
    torch.testing.assert_close(result["q_current_claim"], model._q_claim_output(parent))
    coarse_claim = result["coarse_q_issue_claim_grid"]
    assert coarse_claim.shape == (3, 3)
    assert bool(torch.isfinite(coarse_claim).any())
    assert "value_star" in result


# ---------------------------------------------------------------------------
# TEST 8 — Phat_teacher over the i-grid is mean_i max(T0, TI)
# ---------------------------------------------------------------------------


class _StubTeacher:
    """Records calls and returns known value_star per branch."""

    def __init__(self):
        self.calls = []

    def compute_multi_j_branches(
        self, branch_states, children, m_list, *, branches, prefix_child_counts, **unused
    ):
        self.calls.append({
            "branches": list(branches),
            "counts": [int(value) for value in prefix_child_counts],
            "n_rows": int(branch_states[0].shape[0]),
        })
        bundles = []
        for index, branch in enumerate(branches):
            states = branch_states[index]
            base = 0.5 if branch == "p0" else 0.2
            bundles.append({
                int(prefix_child_counts[0]): {"value_star": base + states[:, 3].clone()}
            })
        return bundles


def test_phat_teacher_is_mean_over_i_grid_of_max_t0_ti():
    _, grid = _small_grid()
    grids_by_eta = {0.0: grid, 1.0: grid}
    transition = SimpleNamespace(
        stacked_children=lambda: torch.zeros(1, 4, 7),
        stacked_m_used=lambda: torch.ones(1, 4, 1),
        branch_weights=torch.full((1, 4), 0.25),
    )
    teacher = _StubTeacher()
    i_values = np.array([0.0, 0.5, 1.0])

    output = teacher_surfaces_over_i_grid(
        teacher, grids_by_eta, transition,
        eta_values=[0.0, 1.0], i_values=i_values, prefix_child_counts=[2],
    )

    assert len(teacher.calls) == 3  # one shared forward per i slice
    assert teacher.calls[0]["branches"] == ["p0", "pi", "p0", "pi"]
    assert teacher.calls[0]["counts"] == [2]
    for eta in (0.0, 1.0):
        assert output[eta]["T0"].shape == (3, *grid.shape)
        np.testing.assert_allclose(output[eta]["T0"][:, 0, 0], 0.5 + i_values)
        np.testing.assert_allclose(output[eta]["TI"][:, 1, 2], 0.2 + i_values)
        phat_teacher = np.mean(np.maximum(output[eta]["T0"], output[eta]["TI"]), axis=0)
        # max(0.5 + i, 0.2 + i) = 0.5 + i; mean_i over {0.0, 0.5, 1.0} = 1.0
        np.testing.assert_allclose(phat_teacher, np.full(grid.shape, 1.0))


# ---------------------------------------------------------------------------
# TEST 9 — default confusion metrics match hand-computed counts
# ---------------------------------------------------------------------------


def test_confusion_metrics_match_hand_computed_shares():
    d_pred = np.array([[1.0, 0.0], [0.0, 0.0]])
    d_teacher = np.array([[1.0, 1.0], [0.0, 0.0]])

    metrics = confusion_metrics(d_pred, d_teacher)

    assert metrics["pred_default_share"] == pytest.approx(0.25)
    assert metrics["teacher_default_share"] == pytest.approx(0.50)
    assert metrics["agreement_share"] == pytest.approx(0.75)
    assert metrics["false_survival_share"] == pytest.approx(0.25)
    assert metrics["false_default_share"] == pytest.approx(0.0)

    empty = confusion_metrics(np.array([]), np.array([]))
    assert all(np.isnan(value) for value in empty.values())


# ---------------------------------------------------------------------------
# TEST 10 — hybrid multi-J merge keeps the full q_issue diagnostics schema
# (regression: coarse merge used to drop q_issue_claim_grid and
#  _finalize_grid_result raised KeyError for eta_t = 1 multi-J batches)
# ---------------------------------------------------------------------------


def test_multi_j_merge_keeps_hybrid_q_issue_diagnostics_schema():
    model = _small_model("hybrid_regime")
    teacher = BPGridTeacher(
        model, P0Loss(), PILoss(), q_target_model=model, coarse_size=3, refine=False
    )
    teacher.candidate_chunk_size = 1  # force a genuine multi-part coarse merge
    parent = _state((0.2, 0.4, 0.6))
    parent[:, 2] = 1.0
    child = parent.clone()
    children = torch.stack([child, child], dim=1)
    m = torch.ones(3, 2, 1)

    results = teacher.compute_multi_j_branches(
        [parent, parent], children, m, branches=["p0", "pi"], prefix_child_counts=[1, 2]
    )

    expected_keys = (
        "q_issue_claim_grid",
        "q_issue_unit_grid",
        "q_issue_realized_default_mask_grid",
        "q_issue_recovery_grid",
        "q_issue_candidate_phat_grid",
        "candidate_phat_gate_used_for_q_issue",
    )
    for bundle in results:
        for count, result in bundle.items():
            for key in expected_keys:
                assert key in result, (count, key)
            assert result["q_issue_claim_grid"].shape == (3, 3)
            assert torch.isfinite(result["q_current_claim"]).all()


# ---------------------------------------------------------------------------
# TEST 11 — the evaluator forward path never mutates model state
# ---------------------------------------------------------------------------


def test_evaluator_forward_path_leaves_model_state_unchanged():
    model = _small_model("hybrid_regime")
    reference, grid = _small_grid()

    before = model_state_hash(model)
    surfaces = evaluate_firm_surfaces(model, grid, reference)
    value_scale_grid(model, grid)
    finite_difference_stats(surfaces["Q_claim"], axis=0, coordinates=grid.b_values)
    after = model_state_hash(model)

    assert before == after

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.5)
    assert model_state_hash(model) != after


# ---------------------------------------------------------------------------
# TEST 12 — "Episode N done: {...}" training-log parser (new evaluator logic)
# ---------------------------------------------------------------------------


def test_conditioning_training_log_parser_reads_episode_done_dicts(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "warmup lines are ignored\n"
        "Episode 3 done: {'p_train_eta1_share_before': 0.03, 'p_train_eta1_share_after': 0.5, "
        "'p_validation_eta1_share_after': 0.52, 'q_train_current_eta1_share': 0.31, "
        "'claim_coverage_fraction_anchors_occupied': 0.75, 'claim_coverage_low_b_sample_share': 0.2}\n"
        "Episode 4 done: {'bad': }\n",
        encoding="utf-8",
    )

    frame, status = parse_conditioning_training_log(log)

    assert status["status"] == "ok"
    assert status["parse_errors"] == 1
    assert list(frame["episode"]) == [3]
    row = frame.iloc[0]
    assert row["p_train_eta1_share_after"] == pytest.approx(0.5)
    assert row["p_validation_eta1_share_after"] == pytest.approx(0.52)
    assert row["q_batches_eta1_share"] == pytest.approx(0.31)
    assert row["claim_coverage_fraction_anchors_occupied"] == pytest.approx(0.75)
    assert pd.isna(row["bp_batches_eta1_share"])  # NaN when absent from the payload

    empty_frame, empty_status = parse_conditioning_training_log(None)
    assert empty_frame.empty and empty_status["status"] == "missing"


# ---------------------------------------------------------------------------
# TEST 13 — §22 flattening emits eta-suffixed columns and eta1/eta0 ratios
# ---------------------------------------------------------------------------


def test_metrics_flatten_emits_eta_suffixed_columns_and_ratio_columns():
    row = {"episode": 3, "model_state_unchanged": True, "seconds": 1.0}
    row["_blocks"] = {
        "q_low_b": {
            0.0: {"q_unit_low_b_p95": 1.2, "qclaim_low_b_d1_d2_abs_p95": 9.7},
            1.0: {"q_unit_low_b_p95": 2.2, "qclaim_low_b_d1_d2_abs_p95": 11.1},
        },
        "p_residual": {
            0.0: {"P0_norm_mean": 0.02, "PI_norm_mean": 0.03},
            1.0: {"P0_norm_mean": 0.05, "PI_norm_mean": 0.06},
        },
        "zbin": {0.0: pd.DataFrame({"a": [1]}), 1.0: pd.DataFrame({"a": [2]})},
        "confusion": {
            1.0: {"pred_default_share": 0.2, "false_survival_share": 0.19},
        },
        "boundary": {
            1.0: {
                "z_default_median": -2.0,
                "teacher_z_default_median": -2.1,
                "boundary_z_abs_gap_p90": 0.09,
            },
        },
    }

    frame = add_eta_ratio_columns(pd.DataFrame([flatten_conditioning_row(row)]))

    for name in (
        "q_unit_low_b_p95_eta0", "q_unit_low_b_p95_eta1",
        "q_unit_low_b_p95_0.0", "q_unit_low_b_p95_1.0",
        "qclaim_low_b_d2_p95_eta1", "qclaim_low_b_d2_p95_1.0",
        "P0_norm_mean_eta1", "P0_norm_mean_eta0",
        "pred_default_share_eta1", "false_survival_share_eta1",
        "pred_zdefault_median_eta1", "teacher_zdefault_median_eta1",
        "boundary_gap_p90_eta1",
        "P0_norm_eta1_eta0_ratio", "PI_norm_eta1_eta0_ratio",
    ):
        assert name in frame.columns, name
    assert "a" not in frame.columns  # zbin frames never leak into the metrics row
    assert frame.loc[0, "P0_norm_eta1_eta0_ratio"] == pytest.approx(2.5)
    assert frame.loc[0, "PI_norm_eta1_eta0_ratio"] == pytest.approx(2.0)
