from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from analysis.economic_config import AnalysisEconomicConfig
from evaluation.bellman_diagnostics import evaluate_bellman_residuals
from evaluation.bp_diagnostics import FrozenTransitionData
from evaluation.full_run_diagnostics import (
    conditional_residual_summary,
    evaluate_fc1_checkpoint,
    evaluate_fc2_checkpoint,
    evaluate_sdf_heldout,
    model_state_hash,
    parse_training_log,
)
from evaluation.grids import FrozenFirmGrid, ReferenceFirmState, build_frozen_grid
from experiments.evaluate_full_run import _discover_checkpoints


def _reference() -> ReferenceFirmState:
    return ReferenceFirmState(
        eta=1.0, i_low=0.1, i_mid=0.2, i_high=0.3, x=-2.0,
        hatcf=-2.1, lnkf=4.0, hatc_cal=-2.0, lnk_cal=4.1,
        n_parent_rows=2, source="fixture", macro_source="fixture",
    )


def _transition(grid, *, m=1.0) -> FrozenTransitionData:
    children = [grid.base_states.clone(), grid.base_states.clone()]
    values = [torch.full((len(grid.base_states), 1), m) for _ in children]
    return FrozenTransitionData(
        children=children, m_raw_list=values, m_used_list=values,
        branch_weights=torch.full((len(grid.base_states), 2), 0.5), metadata={},
    )


class QDiagnosticModel(torch.nn.Module):
    def __init__(self, q_value: float):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(q_value))

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
    assert model_state_hash(model) == before


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


class ConstantFC2(torch.nn.Module):
    quantile_num = 2

    def forward(self, phi):
        value = torch.zeros((phi.shape[0], 1), device=phi.device)
        return {"hatc": value, "lnk": value}


class ConstantPolicy(torch.nn.Module):
    def forward(self, states):
        zeros = torch.zeros_like(states[:, :1])
        return {"bar_i": zeros, "bar_z": zeros}


def test_fc2_checkpoint_uses_formal_resource_accounting():
    rows = []
    for branch in (-1, 0, 1):
        for position in range(4):
            rows.append({
                "path": 0, "t": 0, "branch": branch,
                "b": 0.1 * (position + 1), "z": -0.2 + 0.1 * position,
                "ETA": float(position % 2), "i": 0.1 * (position + 1),
                "x": -2.0, "Hatcf": -2.0, "LnKF": 4.0, "K": 1.0,
            })
    firm = pd.DataFrame(rows)
    summary, nodes = evaluate_fc2_checkpoint(
        ConstantFC2(), ConstantPolicy(), firm,
        pd.DataFrame({"path": [0], "t": [0], "Hatc": [-2.0], "LnK": [4.0]}),
        device=torch.device("cpu"),
        economic_config=AnalysisEconomicConfig.from_current_config(),
    )
    assert summary["fc2_n_nodes"] == 3
    assert bool(summary["fc2_transition_consistency_available"]) is True
    assert summary["fc2_resource_residual_abs_max"] == pytest.approx(0.0, abs=1e-7)
    assert summary["fc2_resource_residual_relative_abs_mean"] == pytest.approx(0.0, abs=1e-7)
    assert nodes.loc[0, "consumption_finite_ratio"] == pytest.approx(1.0)


def test_checkpoint_discovery_preserves_missing_episode(tmp_path):
    directory = tmp_path / "checkpoints_analysis"
    directory.mkdir()
    (directory / "ep0_combined.pt").write_bytes(b"x")
    (directory / "ep2_combined.pt").write_bytes(b"x")
    assert sorted(_discover_checkpoints(tmp_path)) == [0, 2]
    requested = {0, 1, 2}
    assert sorted(requested - set(_discover_checkpoints(tmp_path))) == [1]
