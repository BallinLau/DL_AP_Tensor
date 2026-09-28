import json

import numpy as np
import pandas as pd
import torch

from evaluation.full_run_diagnostics import model_state_hash
from experiments.evaluate_p_teacher_drift import (
    aggregate_selected_components,
    atomic_write_csv,
    atomic_write_json,
    classify_mechanism,
    compute_cumulative_drift,
    compute_episode_drift,
    reconstruct_p0,
    reconstruct_pi,
    select_branch_values,
    write_figures,
)


def _by_i_frame(*, episode=0, eta=1.0, nan=False):
    value = float("nan") if nan else 1.0
    rows = []
    for i_index, (t0, ti) in enumerate(((1.0, 2.0), (3.0, 2.0))):
        selected, mask = select_branch_values(np.array([t0]), np.array([ti]))
        use_pi = bool(mask[0])
        production = value
        investment = 0.2 if use_pi else 0.0
        financing = 0.4 if use_pi else 0.3
        equity = 0.1
        continuation = selected[0] - production + investment - financing + equity
        rows.append({
            "episode": episode,
            "eta": eta,
            "b": 0.05,
            "z": 2.0,
            "i_index": i_index,
            "i_value": 0.2,
            "production": production,
            "T_selected": selected[0] if not nan else value,
            "selected_is_pi": int(use_pi),
            "financing_selected": financing,
            "equity_cost_selected": equity,
            "continuation_selected": continuation if not nan else value,
            "investment_selected": investment,
            "bp_selected": 0.2 + i_index * 0.1,
            "q_issue_selected": 0.5,
            "q_current_claim": 0.25,
            "selected_finite": not nan,
            "Phat_pred": 2.4,
            "P_pred": 2.4,
            "bar_z_pred": 0.0,
        })
    return pd.DataFrame(rows)


def test_branch_selection_uses_strict_ti_greater_than_t0():
    selected, is_pi = select_branch_values(
        np.array([1.0, 3.0]), np.array([2.0, 2.0])
    )
    np.testing.assert_array_equal(selected, np.array([2.0, 3.0]))
    np.testing.assert_array_equal(is_pi, np.array([True, False]))
    assert is_pi.mean() == 0.5


def test_p0_identity():
    reconstructed = reconstruct_p0(
        np.array([2.0]), np.array([0.4]), np.array([0.1]), np.array([1.2])
    )
    np.testing.assert_allclose(reconstructed, [3.5])


def test_pi_identity():
    reconstructed = reconstruct_pi(
        np.array([2.0]), np.array([0.3]), np.array([0.5]),
        np.array([0.2]), np.array([1.4]),
    )
    np.testing.assert_allclose(reconstructed, [3.4])


def test_phat_selected_component_identity():
    aggregated = aggregate_selected_components(_by_i_frame())
    assert len(aggregated) == 1
    assert abs(float(aggregated.loc[0, "phat_identity_error"])) < 1e-12
    assert float(aggregated.loc[0, "pi_selected_share"]) == 0.5


def test_drift_identity():
    first = aggregate_selected_components(_by_i_frame(episode=1))
    second_raw = _by_i_frame(episode=2)
    second_raw["continuation_selected"] += 0.5
    second_raw["T_selected"] += 0.5
    second = aggregate_selected_components(second_raw)
    drift = compute_episode_drift(pd.concat([first, second], ignore_index=True))
    assert len(drift) == 1
    assert abs(float(drift.loc[0, "drift_identity_error"])) < 1e-12
    assert np.isclose(float(drift.loc[0, "delta_continuation"]), 0.5)


def test_eta0_financing_is_zero_by_definition():
    eta = np.zeros(3)
    q_current = np.array([0.2, 0.3, 0.4])
    q_issue = np.array([0.6, 0.7, 0.8])
    financing = eta * ((1.0 - 0.1) * q_issue - q_current)
    np.testing.assert_array_equal(financing, np.zeros(3))


def test_pi_continuation_is_not_multiplied_by_g_twice():
    value_star = np.array([4.0])
    cashflow = np.array([1.5])
    continuation_pi = value_star - cashflow
    g = 1.14
    raw_expected_mp = continuation_pi / g
    np.testing.assert_allclose(raw_expected_mp * g, continuation_pi)
    assert not np.allclose(continuation_pi * g, continuation_pi)


def test_nan_is_preserved_in_aggregation_and_drift():
    aggregated = aggregate_selected_components(_by_i_frame(nan=True))
    assert np.isnan(aggregated.loc[0, "Phat_teacher"])
    assert np.isnan(aggregated.loc[0, "production_component"])
    assert aggregated.loc[0, "selected_finite_share"] == 0.0


def test_contribution_ratios_keep_sign_and_can_exceed_one():
    first = aggregate_selected_components(_by_i_frame(episode=1))
    last = first.copy()
    last["episode"] = 4
    last["Phat_teacher"] += 1.0
    last["financing_component"] += 2.0
    last["continuation_component"] -= 1.0
    cumulative = compute_cumulative_drift(pd.concat([first, last], ignore_index=True))
    assert cumulative.loc[0, "financing_ratio"] == 2.0
    assert cumulative.loc[0, "continuation_ratio"] == -1.0
    assert classify_mechanism(2.0, -1.0) == "FINANCING_DOMINANT"


def test_read_only_model_state_hash_is_unchanged():
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
    before = model_state_hash(model)
    model.eval()
    with torch.no_grad():
        _ = model(torch.ones(3, 2))
    assert model_state_hash(model) == before


def test_incremental_episode_artifact_writer(tmp_path):
    episode_dir = tmp_path / "episodes" / "ep_000"
    frame = _by_i_frame(episode=0)
    atomic_write_json(episode_dir / "metadata.json", {"episode": 0, "status": "ok"})
    atomic_write_json(episode_dir / "finite_audit.json", {"selected_finite_share": 1.0})
    atomic_write_csv(episode_dir / "p_teacher_components_by_i.csv", frame)
    atomic_write_csv(
        episode_dir / "p_teacher_selected_branch.csv",
        aggregate_selected_components(frame),
    )
    assert json.loads((episode_dir / "metadata.json").read_text())["episode"] == 0
    assert (episode_dir / "p_teacher_components_by_i.csv").is_file()
    assert (episode_dir / "p_teacher_selected_branch.csv").is_file()
    assert not (tmp_path / "episodes" / "ep_001").exists()


def test_single_episode_figures_allow_empty_cumulative_table(tmp_path):
    levels = aggregate_selected_components(_by_i_frame(episode=0))
    write_figures(levels, pd.DataFrame(), tmp_path)
    expected = {
        "b005_phat_teacher_by_episode.png",
        "b005_financing_component_by_episode.png",
        "b005_continuation_component_by_episode.png",
        "b005_pi_selected_share_by_episode.png",
        "b005_ep1_ep4_cumulative_contributions.png",
    }
    assert expected == {path.name for path in tmp_path.glob("*.png")}
