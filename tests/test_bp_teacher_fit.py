import numpy as np
import pandas as pd
import pytest
import torch

from evaluation.full_run_diagnostics import model_state_hash
from experiments.evaluate_bp_teacher_fit import (
    B_BIN_EDGES,
    Z_BIN_EDGES,
    build_binned_summary,
    build_grid_bank,
    build_masks,
    build_summary,
    deterministic_sample,
    normalized_regrets,
    pearson_correlation,
    rank_worst_states,
    relative_margins,
    reshape_dense_surface,
    separated_margin,
    spearman_correlation,
    summarize_sample,
    validate_tiny_smoke_mae,
)
from evaluation.grids import ReferenceFirmState
from experiments.evaluate_p_teacher_drift import atomic_write_csv, atomic_write_json


def test_pearson_toy_example():
    assert pearson_correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)


def test_spearman_toy_example():
    assert spearman_correlation([10, 20, 30], [3, 1, 2]) == pytest.approx(-0.5)


def test_constant_prediction_correlation_is_nan():
    assert np.isnan(pearson_correlation([1, 1, 1], [0, 1, 2]))
    assert np.isnan(spearman_correlation([1, 1, 1], [0, 1, 2]))


def test_primary_mask_combines_refi_survival_and_identification():
    masks = build_masks(
        [True, True, True, False], [1.0, -1.0, 1.0, 1.0],
        [1e-3, 1e-3, 1e-10, 1e-3], teacher_margin_tol=1e-8,
    )
    np.testing.assert_array_equal(masks["primary"], [True, False, False, False])


def test_eta0_refi_inactive_is_excluded_from_primary():
    masks = build_masks([False], [2.0], [1.0], teacher_margin_tol=1e-8)
    assert not bool(masks["primary"][0])


def _frame():
    regret = np.array([0.1, 0.2])
    value = np.array([2.0, 4.0])
    scale = np.array([10.0, 20.0])
    relative, scaled = normalized_regrets(regret, value, scale)
    return pd.DataFrame({
        "bp_pred": [0.1, 0.3], "bp_star": [0.2, 0.7], "bp_gap": [0.1, 0.4],
        "regret": regret, "regret_relative": relative, "regret_scaled": scaled,
        "top2_margin": [0.02, 0.04], "relative_margin": [0.01, 0.01],
        "numerical_identified": [True, True],
        "separated_margin_rel_0p05": [0.02, 0.03],
        "separated_margin_rel_0p10": [0.01, 0.02],
    })


def test_regret_and_relative_regret_summary():
    summary = summarize_sample(
        _frame(), large_gap_threshold=0.2,
        high_regret_relative_threshold=0.04, weak_margin_relative_threshold=0.001,
    )
    assert summary["regret_mean"] == pytest.approx(0.15)
    assert summary["regret_p90"] == pytest.approx(0.19)
    assert summary["regret_relative_mean"] == pytest.approx(0.05)


def test_value_scale_regret_uses_given_scale():
    relative, scaled = normalized_regrets([2.0], [4.0], [8.0])
    assert relative.item() == pytest.approx(0.5)
    assert scaled.item() == pytest.approx(0.25)


def test_relative_margin_and_weak_shares():
    frame = _frame()
    frame["relative_margin"] = [1e-5, 5e-3]
    summary = summarize_sample(
        frame, large_gap_threshold=0.2,
        high_regret_relative_threshold=0.04, weak_margin_relative_threshold=1e-3,
    )
    assert summary["weak_rel_margin_1e4"] == pytest.approx(0.5)
    assert summary["weak_rel_margin_1e2"] == pytest.approx(1.0)


def test_relative_margin_uses_absolute_value_star_plus_epsilon():
    actual = relative_margins([0.2, 0.4], [-2.0, 4.0])
    expected = np.array([0.2 / (2.0 + 1e-8), 0.4 / (4.0 + 1e-8)])
    np.testing.assert_allclose(actual, expected)


def test_separated_margin_toy_objective():
    result = separated_margin(
        np.array([[0.0, 0.5, 1.0]]), np.array([[1.0, 3.0, 2.0]]),
        np.array([0.5]), np.array([3.0]), delta=0.4,
    )
    assert result.item() == pytest.approx(1.0)


def test_separated_margin_without_eligible_candidate_is_nan():
    result = separated_margin(
        np.array([[0.45, 0.5, 0.55]]), np.array([[1.0, 2.0, 1.0]]),
        np.array([0.5]), np.array([2.0]), delta=0.10,
    )
    assert np.isnan(result.item())


def test_nan_is_preserved_not_replaced():
    relative, scaled = normalized_regrets([np.nan, 1.0], [2.0, 2.0], [1.0, np.nan])
    assert np.isnan(relative[0])
    assert np.isnan(scaled[0])
    assert np.isnan(scaled[1])


def test_std_ratio_is_prediction_std_over_teacher_std():
    frame = _frame()
    summary = summarize_sample(
        frame, large_gap_threshold=0.2,
        high_regret_relative_threshold=0.04, weak_margin_relative_threshold=0.001,
    )
    expected = np.std(frame.bp_pred) / np.std(frame.bp_star)
    assert summary["std_ratio"] == pytest.approx(expected)


def test_read_only_forward_preserves_model_hash():
    model = torch.nn.Linear(2, 1)
    before = model_state_hash(model)
    model.eval()
    with torch.no_grad():
        model(torch.ones(3, 2))
    assert model_state_hash(model) == before


def test_large_gap_high_and_low_regret_shares_are_separate():
    summary = summarize_sample(
        _frame(), large_gap_threshold=0.2,
        high_regret_relative_threshold=0.04, weak_margin_relative_threshold=0.02,
    )
    assert summary["large_gap_share"] == pytest.approx(0.5)
    assert summary["large_gap_high_regret_share"] == pytest.approx(0.5)
    assert summary["large_gap_low_regret_share"] == pytest.approx(0.0)
    assert summary["large_gap_weak_margin_share"] == pytest.approx(0.5)


def test_empty_sample_summary_returns_nan_not_zero():
    empty = _frame().iloc[0:0]
    summary = summarize_sample(
        empty, large_gap_threshold=0.2,
        high_regret_relative_threshold=0.04, weak_margin_relative_threshold=0.001,
    )
    assert summary["n"] == 0
    assert np.isnan(summary["bp_mae"])
    assert np.isnan(summary["pearson_r"])


def test_correlations_use_pairwise_finite_values_without_filling_nan():
    assert pearson_correlation([1.0, np.nan, 3.0], [2.0, 99.0, 6.0]) == pytest.approx(1.0)
    assert spearman_correlation([1.0, np.nan, 3.0], [2.0, 99.0, 6.0]) == pytest.approx(1.0)


def _bank_frame(bank="dense_grid", branch="pi_mid"):
    frame = _frame()
    frame["bank"] = bank
    frame["branch"] = branch
    frame["b"] = [0.05, 0.5]
    frame["z"] = [-3.0, 2.5]
    frame["refi_active"] = True
    frame["survival"] = True
    frame["primary_mask"] = True
    return frame


def _reference():
    return ReferenceFirmState(
        eta=1.0, i_low=0.1, i_mid=0.5, i_high=0.9,
        x=0.0, hatcf=0.0, lnkf=0.0, hatc_cal=0.0, lnk_cal=0.0,
        n_parent_rows=10, source="toy", macro_source="toy_macro",
    )


def test_dense_grid_reshape_preserves_b_rows_and_z_columns():
    matrix = reshape_dense_surface(np.arange(6), (2, 3))
    np.testing.assert_array_equal(matrix, [[0, 1, 2], [3, 4, 5]])


def test_b_and_z_bin_summaries_assign_expected_intervals():
    frame = _bank_frame()
    by_z = build_binned_summary(frame, column="z", edges=Z_BIN_EDGES, bin_column="z_bin")
    by_b = build_binned_summary(frame, column="b", edges=B_BIN_EDGES, bin_column="b_bin")
    assert set(by_z.z_bin.astype(str)) == {"[-4,-2)", "[2,3)"}
    assert set(by_b.b_bin.astype(str)) == {"[0,0.1)", "[0.5,0.7)"}


def test_bank_summary_isolation_does_not_mix_state_banks():
    frame = pd.concat([_bank_frame("tiny_smoke"), _bank_frame("dense_grid")], ignore_index=True)
    summary = build_summary(
        frame, large_gap_threshold=0.2,
        high_regret_relative_threshold=0.04,
        weak_margin_relative_threshold=0.001,
    )
    primary = summary[summary.mask_type == "primary"]
    assert set(primary.bank) == {"tiny_smoke", "dense_grid"}
    assert set(primary.n) == {2}


def test_on_distribution_sampling_is_deterministic_and_not_stratified():
    frame = pd.DataFrame({"value": np.arange(100), "ETA": np.tile([0, 1], 50)})
    first = deterministic_sample(frame, 12, 12345)
    second = deterministic_sample(frame, 12, 12345)
    other = deterministic_sample(frame, 12, 12346)
    pd.testing.assert_frame_equal(first, second)
    assert not first.source_index.equals(other.source_index)


def test_compression_ratio_is_prediction_std_over_teacher_std():
    summary = summarize_sample(
        _frame(), large_gap_threshold=0.2,
        high_regret_relative_threshold=0.04,
        weak_margin_relative_threshold=0.001,
    )
    assert summary["std_ratio"] == pytest.approx(np.std([0.1, 0.3]) / np.std([0.2, 0.7]))


def test_gap_threshold_shares_are_reported_separately():
    summary = summarize_sample(
        _frame(), large_gap_threshold=0.2,
        high_regret_relative_threshold=0.04,
        weak_margin_relative_threshold=0.001,
    )
    assert summary["gap_gt_0p05_share"] == pytest.approx(1.0)
    assert summary["gap_gt_0p20_share"] == pytest.approx(0.5)
    assert summary["gap_gt_0p40_share"] == pytest.approx(0.0)


def test_worst_state_ranking_keeps_three_ranking_categories():
    ranked = rank_worst_states(_bank_frame(), top_n=1)
    assert set(ranked.ranking_category) == {
        "largest_gap", "largest_regret_scaled", "largest_gap_x_regret",
    }
    assert len(ranked) == 3


def test_nan_mask_is_preserved_by_dense_reshape():
    matrix = reshape_dense_surface([1.0, np.nan, 2.0, 3.0], (2, 2))
    assert np.isnan(matrix[0, 1])


def test_incremental_atomic_artifact_writers_replace_complete_files(tmp_path):
    csv_path = tmp_path / "tables" / "bank.csv"
    json_path = tmp_path / "metadata" / "bank.json"
    atomic_write_csv(csv_path, pd.DataFrame({"x": [1, 2]}))
    atomic_write_json(json_path, {"complete": True, "n": 2})
    assert pd.read_csv(csv_path).x.tolist() == [1, 2]
    assert json_path.read_text().strip().endswith("}")
    assert not list(tmp_path.rglob("*.tmp"))


def test_state_bank_coverage_metadata_records_order_and_grid_shape():
    bank = build_grid_bank(
        "dense_grid", _reference(), [0.0, 1.0], [-4.0, 0.0, 4.0],
        eta=1.0, device=torch.device("cpu"),
    )
    assert bank.grid_shape == (2, 3)
    assert bank.metadata["ordering"] == "b-major_z-minor"
    assert bank.metadata["n_states"] == 6


def test_tiny_smoke_regression_accepts_tolerance_and_hard_fails_outside_it():
    validate_tiny_smoke_mae(0.00624, 0.0062404, 1e-5)
    with pytest.raises(RuntimeError, match="tiny smoke regression failed"):
        validate_tiny_smoke_mae(0.02, 0.0062404, 1e-4)
