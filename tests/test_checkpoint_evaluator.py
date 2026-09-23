from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.checkpoint_loader import load_analysis_checkpoint
from analysis.economic_config import AnalysisEconomicConfig
from config import HyperParams
from evaluation.boundaries import extract_phat_default_boundary
from evaluation.bellman_diagnostics import (
    build_child_continuation_audit,
    evaluate_bellman_residuals,
)
from evaluation.bp_diagnostics import (
    FrozenTransitionData,
    _summary,
    build_frozen_transition_data,
    build_frozen_transition_children,
)
from evaluation.firm_surfaces import (
    evaluate_firm_surfaces,
    evaluate_investment_cutoff,
    investment_margin_diagnostics,
)
from evaluation.grids import ReferenceFirmState, build_frozen_grid, load_reference_state
from experiments.run_utils import build_models
from models import PolicyValueModel
from utils.firm_transition import expand_children_exact_eta
import experiments.evaluate_checkpoints as evaluator_module


def _small_model() -> PolicyValueModel:
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
    )


def _write_combined_policy_checkpoint(path: Path) -> None:
    torch.manual_seed(77)
    model = _small_model()
    sdf_fc1 = build_models(torch.device("cpu"))["sdf_fc1"]
    hp = HyperParams()
    hp.bp_grid_coarse_size = 5
    hp.bp_grid_fine_size = 3
    hp.bp_grid_refine_enabled = False
    hp.bp_grid_parent_chunk_size = 16
    hp.bp_grid_candidate_chunk_size = 5
    hp.bp_grid_max_expanded_states = 4096
    hp.pv_use_clipped_m = True
    hp.pv_m_clamp_min = 0.7
    hp.pv_m_clamp_max = 1.3
    hp.pv_value_scale_mode = "none"
    hp.pv_value_scale_log_max = 20.0
    hp.pv_bellman_normalize_by_value_scale = False
    torch.save(
        {
            "models": {
                "policy_value": model.state_dict(),
                "sdf_fc1": sdf_fc1.state_dict(),
            },
            "hyperparams": hp.__dict__,
            "config_snapshot": AnalysisEconomicConfig.from_current_config().to_dict(),
            "policy_value_model_spec": model.model_spec(),
            "value_parameterization": {
                "mode": "none",
                "scale_formula": "1",
                "bellman_normalization": False,
                "log_max": 20.0,
            },
        },
        path,
    )


def _write_reference_firm(path: Path) -> None:
    rows = []
    for idx in range(6):
        parent = {
            "path": idx,
            "ID": idx,
            "t": 0,
            "branch": -1,
            "b": 0.1 + 0.1 * idx,
            "z": -0.5 + 0.2 * idx,
            "ETA": 1.0,
            "i": 0.05 + 0.05 * idx,
            "x": -1.9 + 0.01 * idx,
            "Hatcf": -2.2 + 0.02 * idx,
            "LnKF": 4.0 + 0.01 * idx,
            "Hatc": -2.1 + 0.02 * idx,
            "LnK": 4.1 + 0.01 * idx,
            "M": 0.98,
        }
        rows.append(parent)
        for branch, z_delta, m_value in ((0, -0.08, 0.96), (1, 0.08, 1.00)):
            rows.append(
                {
                    **parent,
                    "t": 1,
                    "branch": branch,
                    "b": parent["b"] + 0.02,
                    "z": parent["z"] + z_delta,
                    "x": parent["x"] + (branch * 2 - 1) * 0.01,
                    "M": m_value,
                }
            )
    pd.DataFrame(rows).to_pickle(path)


def _run_evaluator(
    checkpoint: Path,
    firm_data: Path,
    output: Path,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [
            sys.executable,
            "experiments/evaluate_checkpoints.py",
            "--checkpoint", str(checkpoint),
            "--firm-data", str(firm_data),
            "--output-dir", str(output),
            "--device", "cpu",
            "--b-points", "5",
            "--z-points", "5",
            "--i-points", "5",
            "--forward-chunk-size", "17",
        ] + list(extra_args or []),
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )


def test_policy_only_combined_checkpoint_does_not_require_sdf(tmp_path):
    checkpoint = tmp_path / "policy_combined.pt"
    _write_combined_policy_checkpoint(checkpoint)
    loaded = load_analysis_checkpoint(checkpoint, device="cpu", m_source="none")
    assert loaded.metadata["loaded_model_keys"] == ["policy_value"]
    assert loaded.metadata["sdf_state_hash"] is None


def test_reference_state_loads_calculated_macro_from_sibling_file(tmp_path):
    firm_path = tmp_path / "ep2_stage_modeb.pkl"
    macro_path = tmp_path / "ep2_stage_modeb_macro.pkl"
    firm = pd.DataFrame(
        {
            "path": [0], "t": [0], "branch": [-1], "b": [0.2], "z": [0.1],
            "ETA": [1.0], "i": [0.2], "x": [-2.0], "Hatcf": [-2.2], "LnKF": [4.0],
        }
    )
    macro = pd.DataFrame(
        {"path": [0], "t": [0], "branch": [0], "Hatc": [-2.1], "LnK": [4.1]}
    )
    firm.to_pickle(firm_path)
    macro.to_pickle(macro_path)
    _, reference = load_reference_state(firm_path)
    assert reference.hatc_cal == -2.1
    assert reference.lnk_cal == 4.1
    assert reference.macro_source == str(macro_path)


def test_final_simulate_macro_auto_discovery_and_explicit_override(tmp_path):
    firm_path = tmp_path / "final_simulate_firm.pkl"
    auto_macro_path = tmp_path / "final_simulate_macro.pkl"
    explicit_macro_path = tmp_path / "chosen_macro.pkl"
    firm = pd.DataFrame(
        {
            "path": [0], "t": [0], "branch": [-1], "b": [0.2], "z": [0.1],
            "ETA": [1.0], "i": [0.2], "x": [-2.0], "Hatcf": [-2.2], "LnKF": [4.0],
        }
    )
    firm.to_pickle(firm_path)
    pd.DataFrame(
        {"path": [0], "t": [0], "Hatc": [-2.1], "LnK": [4.1]}
    ).to_pickle(auto_macro_path)
    pd.DataFrame(
        {"path": [0], "t": [0], "Hatc": [-3.1], "LnK": [5.1]}
    ).to_pickle(explicit_macro_path)

    _, automatic = load_reference_state(firm_path)
    _, explicit = load_reference_state(firm_path, macro_path=explicit_macro_path)

    assert automatic.macro_source == str(auto_macro_path)
    assert automatic.hatc_cal == -2.1
    assert explicit.macro_source == str(explicit_macro_path)
    assert explicit.hatc_cal == -3.1
    assert explicit.lnk_cal == 5.1


def test_phat_boundary_preserves_missing_crossings_as_nan():
    b = np.array([0.0, 0.5, 1.0])
    z = np.array([-1.0, 0.0, 1.0])
    phat = np.array([[-1.0, 0.0, 1.0], [-2.0, -1.0, -0.5], [0.5, 1.0, 2.0]])
    boundary, summary = extract_phat_default_boundary(b, z, phat)
    assert boundary.loc[0, "z_default"] == 0.0
    assert np.isnan(boundary.loc[1, "z_default"])
    assert np.isnan(boundary.loc[2, "z_default"])
    assert boundary.loc[1, "boundary_status"] == "all_default"
    assert boundary.loc[2, "boundary_status"] == "all_survival"
    assert summary["default_boundary_observed_share"] == 1.0 / 3.0


def test_phat_boundary_recognizes_zero_at_grid_endpoint():
    b = np.array([0.0, 1.0])
    z = np.array([-1.0, 0.0, 1.0])
    phat = np.array([[-2.0, -1.0, 0.0], [0.0, 1.0, 2.0]])
    boundary, _ = extract_phat_default_boundary(b, z, phat)
    assert boundary.loc[0, "z_default"] == 1.0
    assert boundary.loc[1, "z_default"] == -1.0
    assert (boundary["boundary_status"] == "single_crossing").all()


def test_phat_multiple_crossings_are_excluded_from_main_boundary():
    b = np.array([0.0, 0.5, 1.0])
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    phat = np.array(
        [
            [-1.0, -0.5, 0.5, 1.0, 2.0],
            [-1.0, 1.0, -1.0, 1.0, -1.0],
            [-2.0, -1.0, -0.5, 0.5, 1.0],
        ]
    )
    boundary, summary = extract_phat_default_boundary(b, z, phat)
    assert boundary.loc[1, "boundary_status"] == "multiple_crossings"
    assert boundary.loc[1, "crossing_count"] == 4
    assert np.isnan(boundary.loc[1, "z_default"])
    assert summary["default_boundary_multiple_crossing_share"] == 1.0 / 3.0
    assert np.isnan(summary["default_boundary_monotonic_share"])


def test_frozen_transition_children_use_state_dependent_ar1_and_sdf_m():
    class FakeSDF(torch.nn.Module):
        def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical=True):
            m = 1.0 + 0.1 * x_curr
            return (
                torch.ones_like(x_prev),
                torch.ones_like(x_curr),
                m,
                hatcf_prev.unsqueeze(1).expand_as(x_curr) + 0.01 * x_curr,
                lnkf_prev.unsqueeze(1).expand_as(x_curr) + 0.01 * x_curr,
            )

    reference = ReferenceFirmState(
        eta=1.0,
        i_low=0.1,
        i_mid=0.2,
        i_high=0.3,
        x=-2.0,
        hatcf=-2.2,
        lnkf=4.0,
        hatc_cal=-2.1,
        lnk_cal=4.1,
        n_parent_rows=2,
        source="fixture",
        macro_source="fixture",
    )
    parents = torch.tensor(
        [
            [0.2, -4.0, 1.0, 0.2, -2.0, -2.2, 4.0],
            [0.2, 4.0, 1.0, 0.2, -2.0, -2.2, 4.0],
        ]
    )
    hp = HyperParams()
    hp.pv_use_clipped_m = False
    config = AnalysisEconomicConfig.from_current_config()
    children, m_list, metadata = build_frozen_transition_children(
        FakeSDF(),
        parents,
        reference,
        hp,
        config,
        n_child_shocks=2,
        shock_seed=9,
    )
    expected_z_difference = float(config.RHO_Z) * 8.0
    assert children[0][1, 1] - children[0][0, 1] == pytest.approx(expected_z_difference)
    assert expected_z_difference != pytest.approx(8.0)
    assert len(children) == 4
    assert len(m_list) == 4
    torch.testing.assert_close(m_list[0], m_list[1])
    torch.testing.assert_close(m_list[2], m_list[3])
    assert not torch.allclose(m_list[0], m_list[2])
    assert metadata["builder"] == "ConvergenceShockBank+build_child_exogenous_bundle"
    assert metadata["eta_integration_mode"] == "exact"
    assert metadata["continuous_child_count"] == 2
    assert metadata["expanded_child_count"] == 4
    assert metadata["eta_next_active_share"] == pytest.approx(float(config.ZETA))

    nested = build_frozen_transition_data(
        FakeSDF(), parents, reference, hp, config,
        n_child_shocks=2, shock_seed=12345, shock_bank_max_child_shocks=32,
    )
    full = build_frozen_transition_data(
        FakeSDF(), parents, reference, hp, config,
        n_child_shocks=32, shock_seed=12345, shock_bank_max_child_shocks=32,
    )
    assert len(nested.children) == 2 * nested.metadata["continuous_child_count"]
    for child_index in range(len(nested.children)):
        torch.testing.assert_close(nested.children[child_index], full.children[child_index])
        torch.testing.assert_close(nested.m_raw_list[child_index], full.m_raw_list[child_index])
        torch.testing.assert_close(nested.m_used_list[child_index], full.m_used_list[child_index])
    for pair_start in range(0, len(nested.children), 2):
        assert nested.children[pair_start][:, 2].eq(0.0).all()
        assert nested.children[pair_start + 1][:, 2].eq(1.0).all()
    assert nested.metadata["nested_prefix_from_max_J"] is True
    assert full.metadata["nested_prefix_from_max_J"] is False


@pytest.mark.parametrize("zeta", [0.0, 0.03, 1.0])
def test_exact_eta_expansion_preserves_probability_and_pair_identity(zeta):
    child0 = torch.tensor(
        [[0.2, -0.5, 1.0, 0.1, -2.0, -2.2, 4.0]], dtype=torch.float64
    )
    child1 = torch.tensor(
        [[0.2, 0.5, 0.0, 0.3, -1.8, -2.0, 4.2]], dtype=torch.float64
    )
    expanded = expand_children_exact_eta(
        [child0, child1],
        zeta=zeta,
        child_weights=torch.tensor([[0.25, 0.75]], dtype=torch.float64),
    )

    assert len(expanded.children) == 4
    torch.testing.assert_close(
        expanded.branch_weights.sum(dim=1), torch.ones(1, dtype=torch.float64)
    )
    eta = torch.stack([child[:, 2] for child in expanded.children], dim=1)
    torch.testing.assert_close(
        (expanded.branch_weights * eta).sum(dim=1),
        torch.tensor([zeta], dtype=torch.float64),
    )
    for pair_start in (0, 2):
        eta0 = expanded.children[pair_start]
        eta1 = expanded.children[pair_start + 1]
        assert eta0[0, 2].item() == 0.0
        assert eta1[0, 2].item() == 1.0
        torch.testing.assert_close(
            eta0[:, [0, 1, 3, 4, 5, 6]], eta1[:, [0, 1, 3, 4, 5, 6]]
        )


def test_exact_eta_child_leverage_expectation():
    b_parent = torch.tensor([[0.2], [0.7]])
    bp_candidate = torch.tensor([[0.9], [0.1]])
    zeta = 0.03
    continuous = [
        torch.cat(
            [b_parent, torch.zeros(2, 1), torch.zeros(2, 1), torch.zeros(2, 4)],
            dim=1,
        )
        for _ in range(2)
    ]
    expanded = expand_children_exact_eta(continuous, zeta=zeta)
    child_b = []
    for child in expanded.children:
        eta = child[:, 2:3]
        child_b.append(eta * bp_candidate + (1.0 - eta) * b_parent)
    weighted_b = (
        expanded.branch_weights.unsqueeze(-1) * torch.stack(child_b, dim=1)
    ).sum(dim=1)
    torch.testing.assert_close(
        weighted_b, (1.0 - zeta) * b_parent + zeta * bp_candidate
    )


def test_exact_eta_probability_mass_is_seed_invariant():
    class FakeSDF(torch.nn.Module):
        def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical=True):
            return (
                torch.ones_like(x_prev),
                torch.ones_like(x_curr),
                torch.ones_like(x_curr),
                hatcf_prev.unsqueeze(1).expand_as(x_curr),
                lnkf_prev.unsqueeze(1).expand_as(x_curr),
            )

    reference = ReferenceFirmState(
        eta=1.0, i_low=0.1, i_mid=0.2, i_high=0.3, x=-2.0,
        hatcf=-2.2, lnkf=4.0, hatc_cal=-2.1, lnk_cal=4.1,
        n_parent_rows=1, source="fixture", macro_source="fixture",
    )
    parents = torch.tensor([[0.2, 0.0, 1.0, 0.2, -2.0, -2.2, 4.0]])
    hp = HyperParams()
    hp.pv_use_clipped_m = False
    config = AnalysisEconomicConfig.from_current_config()
    for seed in (1, 7, 12345):
        transition = build_frozen_transition_data(
            FakeSDF(), parents, reference, hp, config,
            n_child_shocks=2, shock_seed=seed,
        )
        eta = torch.stack([child[:, 2] for child in transition.children], dim=1)
        eta_mass = (transition.branch_weights * eta).sum(dim=1)
        torch.testing.assert_close(
            eta_mass, torch.full_like(eta_mass, float(config.ZETA))
        )


def test_formal_evaluator_exact_eta_is_independent_of_training_ablation():
    class FakeSDF(torch.nn.Module):
        def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical=True):
            return (
                torch.ones_like(x_prev), torch.ones_like(x_curr), torch.ones_like(x_curr),
                hatcf_prev.unsqueeze(1).expand_as(x_curr),
                lnkf_prev.unsqueeze(1).expand_as(x_curr),
            )

    reference = ReferenceFirmState(
        eta=1.0, i_low=0.1, i_mid=0.2, i_high=0.3, x=-2.0,
        hatcf=-2.2, lnkf=4.0, hatc_cal=-2.1, lnk_cal=4.1,
        n_parent_rows=1, source="fixture", macro_source="fixture",
    )
    hp = HyperParams()
    hp.pv_exact_eta_integration_enabled = False
    config = AnalysisEconomicConfig.from_current_config()
    transition = build_frozen_transition_data(
        FakeSDF(), torch.tensor([[0.2, 0.0, 1.0, 0.2, -2.0, -2.2, 4.0]]),
        reference, hp, config, n_child_shocks=2, shock_seed=3,
    )

    assert transition.metadata["eta_integration_mode"] == "exact"
    assert transition.metadata["formal_eta_integration_independent_of_training_ablation"] is True
    assert transition.metadata["training_eta_integration_mode"] == "legacy_sampled_ablation"
    assert transition.metadata["eta_next_active_share"] == pytest.approx(float(config.ZETA))
    assert transition.metadata["expanded_child_count"] == 2 * transition.metadata["continuous_child_count"]


def test_bp_consistency_primary_statistics_require_survival_and_identification():
    pred = np.array([[0.99, 0.80, 0.20]])
    star = np.array([[0.01, 0.20, 0.20]])
    survival = np.array([[False, True, True]])
    identified = np.array([[True, False, True]])
    summary = _summary(
        "p0",
        pred,
        star,
        survival,
        identified,
        margin_tol=1e-8,
    )
    assert summary["p0_mae"] == 0.0
    assert summary["p0_survival_mae"] == pytest.approx(0.3)
    assert summary["p0_raw_mae"] == pytest.approx((0.98 + 0.60) / 3.0)
    assert summary["p0_predicted_high_boundary_share"] == 0.0
    assert summary["p0_survival_predicted_high_boundary_share"] == 0.0
    assert summary["p0_raw_predicted_high_boundary_share"] == pytest.approx(1.0 / 3.0)
    assert summary["p0_survival_identified_grid_share"] == pytest.approx(1.0 / 3.0)
    assert summary["p0_bp_grid_star_mean"] == pytest.approx(0.2)
    assert summary["p0_survival_bp_grid_star_mean"] == pytest.approx(0.2)
    assert summary["p0_raw_bp_grid_star_mean"] == pytest.approx((0.01 + 0.20 + 0.20) / 3.0)


def test_flat_bp_objective_is_excluded_from_identified_metric():
    summary = _summary(
        "p0",
        np.array([[0.9]]),
        np.array([[0.1]]),
        np.array([[True]]),
        np.array([[False]]),
        regret=np.array([[0.0]]),
        top2_margin=np.array([[0.0]]),
        margin_tol=1e-8,
    )
    assert np.isnan(summary["p0_mae"])
    assert summary["p0_survival_mae"] == pytest.approx(0.8)
    assert summary["p0_teacher_identified_share"] == 0.0
    assert summary["p0_regret_mean"] != summary["p0_regret_mean"]


def test_bp_regret_is_zero_when_prediction_equals_teacher_star():
    summary = _summary(
        "p0",
        np.array([[0.2, 0.8]]),
        np.array([[0.2, 0.8]]),
        np.array([[True, True]]),
        np.array([[True, True]]),
        regret=np.zeros((1, 2)),
        margin_tol=1e-8,
    )
    assert summary["p0_mae"] == 0.0
    assert summary["p0_regret_mean"] == 0.0
    assert summary["p0_regret_max"] == 0.0


def _transition_fixture(grid, eta_values=(0.0, 1.0)):
    children = []
    m_raw = []
    m_used = []
    for eta in eta_values:
        child = grid.base_states.clone()
        child[:, 2] = float(eta)
        children.append(child)
        m_raw.append(torch.ones(len(child), 1))
        m_used.append(torch.ones(len(child), 1))
    return FrozenTransitionData(
        children=children,
        m_raw_list=m_raw,
        m_used_list=m_used,
        branch_weights=torch.full((len(grid.base_states), len(children)), 1.0 / len(children)),
        metadata={"m_mode": "raw_sdf_m"},
    )


def test_child_audit_uses_eta_next_and_preserves_candidate_bp_for_eta1():
    class ChildModel(torch.nn.Module):
        def forward(self, states):
            b = states[:, 0:1]
            return SimpleNamespace(P=b, Phat=torch.ones_like(b), bar_z=torch.zeros_like(b))

    reference = ReferenceFirmState(
        eta=0.0, i_low=0.1, i_mid=0.2, i_high=0.3,
        x=-2.0, hatcf=-2.1, lnkf=4.0, hatc_cal=-2.0, lnk_cal=4.1,
        n_parent_rows=1, source="fixture", macro_source="fixture",
    )
    grid = build_frozen_grid(
        reference, b_min=0.4, b_max=0.5, b_points=2,
        z_min=-1.0, z_max=1.0, z_points=2, device=torch.device("cpu"),
    )
    audit = build_child_continuation_audit(
        ChildModel(), grid, _transition_fixture(grid),
        AnalysisEconomicConfig.from_current_config(), candidate_bp=(0.2, 0.5, 0.8),
    )
    rows = audit[(audit["state_label"] == "b_low_z_low") & (audit["branch"] == "p0")]
    assert rows["child_b_identity_error"].max() == pytest.approx(0.0, abs=1e-7)
    for candidate in (0.2, 0.5, 0.8):
        selected = rows[rows["bp_candidate"] == candidate]
        assert selected.loc[selected["eta_next"] == 0.0, "child_b"].iloc[0] == pytest.approx(0.4)
        assert selected.loc[selected["eta_next"] == 1.0, "child_b"].iloc[0] == pytest.approx(candidate)
    continuation = rows.groupby("bp_candidate")["M_times_P_child"].mean()
    assert continuation.loc[0.2] != pytest.approx(continuation.loc[0.8])
    np.testing.assert_allclose(
        rows["continuation_contribution"],
        rows["branch_weight"] * rows["raw_continuation_term"],
    )
    np.testing.assert_allclose(
        rows["weighted_continuation_contribution"],
        rows["continuation_contribution"],
    )
    from losses import P0Loss
    loss = P0Loss()
    zeros = torch.zeros(1, 1)
    cf_low = loss.compute_cashflow_p0(
        torch.tensor([[-2.0]]), torch.tensor([[-1.0]]), torch.tensor([[0.4]]),
        zeros, torch.tensor([[0.2]]), zeros,
    )
    cf_high = loss.compute_cashflow_p0(
        torch.tensor([[-2.0]]), torch.tensor([[-1.0]]), torch.tensor([[0.4]]),
        zeros, torch.tensor([[0.8]]), zeros,
    )
    torch.testing.assert_close(cf_low, cf_high)


def test_bellman_physical_residual_is_zero_for_exact_mock_equation():
    economic = AnalysisEconomicConfig.from_current_config()

    class ExactModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            from losses import P0Loss, PILoss
            self.p0_loss = P0Loss(
                delta=economic.DELTA, tau=economic.TAU,
                kappa_b=economic.KAPPA_B, kappa_e=economic.KAPPA_E,
            )
            self.pi_loss = PILoss(
                delta=economic.DELTA, tau=economic.TAU, g=economic.G,
                kappa_b=economic.KAPPA_B, kappa_e=economic.KAPPA_E,
                b_penalty_weight=0.0,
            )

        def forward(self, states):
            zeros = torch.zeros_like(states[:, 0:1])
            p_child = torch.full_like(zeros, 2.0)
            cf0 = self.p0_loss.compute_cashflow_p0(
                states[:, 4:5], states[:, 1:2], states[:, 0:1], zeros, zeros,
                states[:, 2:3],
            )
            cfi = self.pi_loss.compute_cashflow_pi(
                states[:, 4:5], states[:, 1:2], states[:, 0:1], states[:, 3:4],
                zeros, zeros, states[:, 2:3],
            )
            return SimpleNamespace(
                Q=zeros, bp0=torch.full_like(zeros, 0.3), bpI=torch.full_like(zeros, 0.7),
                P0=cf0 + 2.0, PI=cfi + float(economic.G) * 2.0, P=p_child,
            )

        def equity_value_scale(self, states):
            return torch.ones_like(states[:, 0:1])

    reference = ReferenceFirmState(
        eta=0.0, i_low=0.1, i_mid=0.2, i_high=0.3,
        x=-2.0, hatcf=-2.1, lnkf=4.0, hatc_cal=-2.0, lnk_cal=4.1,
        n_parent_rows=1, source="fixture", macro_source="fixture",
    )
    grid = build_frozen_grid(
        reference, b_min=0.2, b_max=0.4, b_points=2,
        z_min=-0.2, z_max=0.2, z_points=2, device=torch.device("cpu"),
    )
    surfaces, summary = evaluate_bellman_residuals(
        ExactModel(), grid, _transition_fixture(grid), economic,
    )
    assert np.max(np.abs(surfaces["R0_signed"])) < 1e-6
    assert np.max(np.abs(surfaces["RI_signed"])) < 1e-6
    assert summary["p0_residual_abs_mean"] < 1e-6
    assert summary["pi_residual_abs_mean"] < 1e-6


def test_q_unit_masks_zero_debt_without_epsilon_division():
    class QModel(torch.nn.Module):
        def forward(self, states):
            one = torch.ones_like(states[:, 0:1])
            q = 2.0 * states[:, 0:1]
            return SimpleNamespace(
                Q=q, bp0=0.2 * one, bpI=0.8 * one, P0=one, PI=one,
                bar_i_cond=0.5 * one, bar_i_eff=0.5 * one, bar_z=torch.zeros_like(one),
                P=one, Phat=one, bp_cond=0.5 * one, bp=0.5 * one,
                survival_prob=one,
            )

    reference = ReferenceFirmState(
        eta=1.0, i_low=0.1, i_mid=0.2, i_high=0.3,
        x=-2.0, hatcf=-2.1, lnkf=4.0, hatc_cal=-2.0, lnk_cal=4.1,
        n_parent_rows=1, source="fixture", macro_source="fixture",
    )
    grid = build_frozen_grid(
        reference, b_min=0.0, b_max=1.0, b_points=2,
        z_min=-1.0, z_max=1.0, z_points=2, device=torch.device("cpu"),
    )
    surfaces = evaluate_firm_surfaces(QModel(), grid, reference)
    assert np.isnan(surfaces["q_unit"][0]).all()
    assert np.allclose(surfaces["q_unit"][1], 2.0)


def test_investment_monotonicity_reports_survival_headline_separately():
    surfaces = {
        "P0": np.zeros((2, 2)),
        "PI_low": np.array([[2.0, 2.0], [2.0, 2.0]]),
        "PI_mid": np.array([[3.0, 1.0], [3.0, 1.0]]),
        "PI_high": np.array([[4.0, 0.0], [4.0, 0.0]]),
    }
    investment = {
        "survival_mask": np.array([[False, True], [False, True]]),
        "investment_status": np.full((2, 2), "single_crossing", dtype=object),
    }
    _, summary = investment_margin_diagnostics(surfaces, investment)
    assert summary["investment_i_monotonicity_violation_share_raw"] == 0.5
    assert summary["investment_i_monotonicity_violation_share"] == 0.5
    assert summary["investment_i_monotonicity_violation_share_survival"] == 0.0


def test_eta0_eta1_grids_differ_only_in_parent_eta():
    reference = ReferenceFirmState(
        eta=0.0, i_low=0.1, i_mid=0.2, i_high=0.3,
        x=-2.0, hatcf=-2.1, lnkf=4.0, hatc_cal=-2.0, lnk_cal=4.1,
        n_parent_rows=1, source="fixture", macro_source="fixture",
    )
    kwargs = dict(
        b_min=0.0, b_max=1.0, b_points=3, z_min=-2.0, z_max=2.0,
        z_points=3, device=torch.device("cpu"),
    )
    eta0 = build_frozen_grid(reference, **kwargs).base_states
    eta1 = build_frozen_grid(replace(reference, eta=1.0), **kwargs).base_states
    torch.testing.assert_close(eta0[:, [0, 1, 3, 4, 5, 6]], eta1[:, [0, 1, 3, 4, 5, 6]])
    assert torch.all(eta0[:, 2] == 0.0)
    assert torch.all(eta1[:, 2] == 1.0)


def test_investment_status_requires_exactly_one_crossing():
    class InvestmentModel(torch.nn.Module):
        def forward_value_components(self, states):
            b = states[:, 0:1]
            i = states[:, 3:4]
            delta = torch.where(
                b < 0.2,
                torch.ones_like(i),
                torch.where(
                    b < 0.5,
                    -torch.ones_like(i),
                    torch.where(b < 0.8, i - 1.0, (i - 0.25) * (i - 0.75)),
                ),
            )
            return {"V0_physical": torch.zeros_like(delta), "VI_physical": delta}

    reference = ReferenceFirmState(
        eta=1.0, i_low=0.0, i_mid=0.5, i_high=1.0,
        x=-2.0, hatcf=-2.1, lnkf=4.0, hatc_cal=-2.0, lnk_cal=4.1,
        n_parent_rows=1, source="fixture", macro_source="fixture",
    )
    grid = build_frozen_grid(
        reference,
        b_min=0.0,
        b_max=1.0,
        b_points=4,
        z_min=-1.0,
        z_max=1.0,
        z_points=2,
        device=torch.device("cpu"),
    )
    result = evaluate_investment_cutoff(
        InvestmentModel(),
        grid,
        reference,
        i_points=5,
        i_min=0.0,
        i_max=1.0,
        chunk_size=32,
        survival_mask=np.ones(grid.shape, dtype=bool),
    )
    assert set(result["investment_status"][:, 0]) == {
        "all_invest", "all_no_invest", "single_crossing", "multiple_crossings"
    }
    assert np.all(result["i_star"][2] == 1.0)
    assert np.isnan(result["i_star"][3]).all()


def test_firm_checkpoint_evaluator_smoke_is_read_only_and_deterministic(tmp_path):
    checkpoint = tmp_path / "policy_combined.pt"
    firm_data = tmp_path / "firm.pkl"
    out_a = tmp_path / "eval_a"
    out_b = tmp_path / "eval_b"
    _write_combined_policy_checkpoint(checkpoint)
    _write_reference_firm(firm_data)

    first = _run_evaluator(checkpoint, firm_data, out_a)
    second = _run_evaluator(checkpoint, firm_data, out_b)
    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr

    required = [
        "metadata.json",
        "summary.csv",
        "value/P0.png",
        "value/PI_low.png",
        "value/PI_mid.png",
        "value/PI_high.png",
        "default/default_boundary.csv",
        "default/boundary_comparison.csv",
        "default/boundary_comparison.png",
        "investment/i_star.png",
        "investment/D_VI_minus_V0_mid.csv",
        "investment/bar_i_cond.csv",
        "investment/bar_i_eff.csv",
        "q/Q_b_slices.png",
        "q/q_unit.csv",
        "bp/bp_raw.csv",
        "bp/bp_survival.csv",
        "bp/p0_bp_abs_gap_survival.png",
        "bp/p0_bp_abs_gap_survival_identified.png",
        "bp/p0_bp_regret_survival_identified.csv",
        "bp/p0_teacher_top2_margin_raw.csv",
        "bp/pi_mid_bp_grid_star_survival.csv",
        "objective_slices/p0_b_mid_z_mid.csv",
        "objective_slices/pi_mid_b_mid_z_mid.png",
        "bellman/R0_signed.csv",
        "bellman/RI_signed.png",
        "audits/child_continuation_audit.csv",
    ]
    for relative in required:
        assert (out_a / relative).exists(), relative
    assert len(list((out_a / "objective_slices").glob("*.csv"))) == 18

    metadata = json.loads((out_a / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["model_state_unchanged"] is True
    assert metadata["model_state_hash_before"] == metadata["model_state_hash_after"]
    assert set(metadata["model_state_hash_before"]) == {"policy_value", "sdf_fc1"}
    assert metadata["grid"]["eta"] == 1.0
    assert metadata["reference_state"]["n_parent_rows"] == 6
    assert metadata["reference_transition_bank"]["m_source"] == "sdf_fc1.forward_step"
    assert metadata["reference_transition_bank"]["eta_integration_mode"] == "exact"
    assert metadata["formal_evaluator_eta_integration_mode"] == "exact"
    assert metadata["formal_eta_integration_independent_of_training_ablation"] is True
    assert metadata["reference_transition_bank"]["eta_next_active_share"] == pytest.approx(0.03)
    assert metadata["bp_teacher_model"] == "policy_value"
    assert metadata["reference_transition_bank"]["bp_teacher_model"] == "policy_value"
    assert metadata["reference_transition_bank"]["teacher_margin_tol"] == 1e-8
    assert metadata["reference_transition_bank"]["primary_bp_mask"] == (
        "finite Phat>0 and top2_margin>teacher_margin_tol"
    )
    assert metadata["semantics"]["child_leverage_timing"] == (
        "b_next = eta_next * bp_current + (1-eta_next) * b_current"
    )
    assert metadata["semantics"]["current_financing_eta"] == "eta_current"

    summary = pd.read_csv(out_a / "summary.csv")
    assert {
        "p0_residual_signed_mean", "p0_residual_abs_p90",
        "pi_residual_signed_mean", "pi_residual_abs_p90",
        "bp_mae_raw", "bp_mae_survival_identified", "bp_regret_mean",
        "bp_grid_star_mean", "bp_continuation_at_coarse_star_mean",
        "bp_regret_median", "bp_regret_p90", "bp_regret_p99", "bp_regret_max",
        "teacher_identified_share", "investment_i_monotonicity_violation_share",
        "investment_i_monotonicity_violation_share_survival",
        "hard_default_share", "soft_default_mean", "q_unit_mean_survival",
    }.issubset(summary.columns)

    q_unit = pd.read_csv(out_a / "q" / "q_unit.csv", index_col=0)
    assert q_unit.iloc[0].isna().all()
    pdt.assert_frame_equal(
        pd.read_csv(out_a / "summary.csv"),
        pd.read_csv(out_b / "summary.csv"),
        check_exact=True,
    )
    pdt.assert_frame_equal(
        pd.read_csv(out_a / "objective_slices" / "p0_b_mid_z_mid.csv"),
        pd.read_csv(out_b / "objective_slices" / "p0_b_mid_z_mid.csv"),
        check_exact=True,
    )
    objective = pd.read_csv(out_a / "objective_slices" / "p0_b_mid_z_mid.csv")
    assert {
        "eta_next_active_share",
        "child_b_mean",
        "child_b_eta0_mean",
        "child_b_eta1_mean",
    }.issubset(objective.columns)
    audit = pd.read_csv(out_a / "audits" / "child_continuation_audit.csv")
    assert audit["child_b_identity_error"].max() < 1e-6
    assert {
        "parent_eta", "eta_next", "child_b", "M_raw", "M_used", "P_child",
        "Phat_child", "bar_z_child", "M_times_P_child", "continuation_raw",
        "continuation_weighted", "raw_continuation_term",
        "weighted_continuation_contribution", "continuation_contribution",
    }.issubset(audit.columns)
    grouped = audit.groupby(["state_label", "bp_candidate", "branch"], sort=False)
    for (_, _, branch), rows in grouped:
        growth = 1.0 if branch == "p0" else float(AnalysisEconomicConfig.from_current_config().G)
        expected = (rows["branch_weight"] * growth * rows["M_used"] * rows["P_child"]).sum()
        assert rows["continuation_weighted"].sum() == pytest.approx(expected)


def test_transition_is_built_at_evaluated_j_not_canonical_bank(tmp_path):
    """An ordinary episode must not build a J=Jcanonical transition."""
    checkpoint = tmp_path / "policy_combined.pt"
    firm_data = tmp_path / "firm.pkl"
    output = tmp_path / "matrix"
    _write_combined_policy_checkpoint(checkpoint)
    _write_reference_firm(firm_data)

    result = _run_evaluator(
        checkpoint,
        firm_data,
        output,
        [
            "--eta-values", "1",
            "--n-child-shocks", "4",
            "--robustness-child-shocks", "4",
            "--shock-bank-max-child-shocks", "8",
            "--b-points", "3",
            "--z-points", "3",
            "--i-points", "3",
        ],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["canonical_shock_bank_max_child_shocks"] == 8
    assert metadata["max_evaluated_child_shocks"] == 4
    assert metadata["evaluated_child_shocks"] == [4]
    assert metadata["shock_bank_max_child_shocks"] == 8
    case = metadata["case_metadata_by_eta_j"]["eta1_J4"]
    assert case["canonical_shock_bank_max_child_shocks"] == 8
    assert case["max_evaluated_child_shocks"] == 4
    assert case["evaluated_child_shocks"] == [4]
    # Built transition is J=4 (8 expanded children), drawn from the 8-child bank.
    assert case["transition_continuous_child_count"] == 4
    assert case["transition_expanded_child_count"] == 8
    assert case["reference_transition_bank"]["source_max_J"] == 8
    assert case["reference_transition_bank"]["requested_J"] == 4
    assert case["reference_transition_bank"]["continuous_child_count"] == 4
    assert case["reference_transition_bank"]["expanded_child_count"] == 8
    # The chunk planner must see 8 expanded children, never 16.
    plans = case["bp_forward_stats"]["bp_grid_chunk_plans"]
    assert {plan["n_children"] for plan in plans} == {8}
    assert max(plan["n_children"] for plan in plans) == 8


def test_objective_slices_are_written_per_eta(tmp_path):
    """Each eta must own its objective slices; eta1 must not overwrite eta0."""
    checkpoint = tmp_path / "policy_combined.pt"
    firm_data = tmp_path / "firm.pkl"
    output = tmp_path / "matrix"
    _write_combined_policy_checkpoint(checkpoint)
    _write_reference_firm(firm_data)

    result = _run_evaluator(
        checkpoint,
        firm_data,
        output,
        [
            "--eta-values", "0", "1",
            "--n-child-shocks", "4",
            "--robustness-child-shocks", "2", "4",
            "--b-points", "3",
            "--z-points", "3",
            "--i-points", "3",
        ],
    )
    assert result.returncode == 0, result.stdout + result.stderr

    eta_dirs = {eta: output / eta / "objective_slices" for eta in ("eta0", "eta1")}
    for directory in eta_dirs.values():
        assert directory.is_dir(), f"missing objective slice dir {directory}"
        for label in ("p0", "pi_mid"):
            assert list(directory.glob(f"{label}_*.csv")), f"missing {label} csv"
            assert list(directory.glob(f"{label}_*.png")), f"missing {label} png"

    names = sorted(path.name for path in eta_dirs["eta0"].glob("*.csv"))
    assert names == sorted(path.name for path in eta_dirs["eta1"].glob("*.csv"))
    assert names, "no objective slices were written"
    assert any(
        not pd.read_csv(eta_dirs["eta0"] / name).equals(
            pd.read_csv(eta_dirs["eta1"] / name)
        )
        for name in names
    ), "eta0 and eta1 objective slices are identical: they were overwritten"
    # Only the primary J carries detailed slices; robustness cases must not.
    assert not list((output / "robustness").glob("**/objective_slices"))


def test_eta_and_child_shock_matrix_writes_full_and_compact_cases(tmp_path):
    checkpoint = tmp_path / "policy_combined.pt"
    firm_data = tmp_path / "firm.pkl"
    output = tmp_path / "matrix"
    _write_combined_policy_checkpoint(checkpoint)
    _write_reference_firm(firm_data)

    result = _run_evaluator(
        checkpoint,
        firm_data,
        output,
        [
            "--eta-values", "0", "1",
            "--n-child-shocks", "2",
            "--robustness-child-shocks", "2", "3",
            "--b-points", "3",
            "--z-points", "3",
            "--i-points", "3",
        ],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = pd.read_csv(output / "summary.csv")
    assert set(zip(summary["eta_parent"], summary["n_child_shocks"])) == {
        (0.0, 2), (0.0, 3), (1.0, 2), (1.0, 3)
    }
    for eta_label in ("eta0", "eta1"):
        assert (output / eta_label / "bellman" / "R0_signed.csv").is_file()
        assert (output / eta_label / "audits" / "child_continuation_audit.csv").is_file()
        assert (output / "robustness" / f"{eta_label}_J3" / "summary.csv").is_file()
        assert not (output / "robustness" / f"{eta_label}_J3" / "bellman").exists()
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["eta_values"] == [0.0, 1.0]
    assert metadata["robustness_n_child_shocks"] == [2, 3]
    assert metadata["common_random_numbers_scope"] == (
        "within_each_eta_grid_and_nested_prefix_across_J"
    )
    assert metadata["nested_shock_prefix_across_J"] is True
    assert metadata["shock_bank_max_child_shocks"] == 3
    assert metadata["matrix_reuse"] == {
        "checkpoint_loaded_once": True,
        "static_surfaces_once_per_eta": True,
        "transition_built_at_Jmax_once_per_eta": True,
    }
    assert metadata["timing"]["checkpoint_load_count"] == 1
    assert metadata["timing"]["transition_build_count"] == 2


def test_matrix_loads_once_hashes_only_at_boundary_and_uses_primary_metadata(
    tmp_path, monkeypatch,
):
    loaded = SimpleNamespace(
        models={
            "policy_value": torch.nn.Linear(1, 1),
            "sdf_fc1": torch.nn.Linear(1, 1),
        },
        metadata={"loaded_model_keys": ["policy_value", "sdf_fc1"]},
    )
    load_calls = []
    hash_calls = []

    def fake_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        return loaded

    def fake_hash(model):
        hash_calls.append(model)
        return f"hash-{id(model)}"

    def fake_evaluate(case_args):
        assert case_args.loaded_checkpoint is loaded
        assert case_args.defer_model_state_hash is True
        assert case_args.manage_cuda_peak_stats is False
        case_args.output_dir.mkdir(parents=True, exist_ok=True)
        marker = f"eta{case_args.eta:g}-J{case_args.n_child_shocks}"
        metadata = {
            "training_eta_integration_mode": "exact",
            "grid": {"marker": marker},
            "reference_state": {"marker": marker},
            "reference_transition_bank": {"marker": marker},
            "m_mode": marker,
            "m_clamp_bounds": None,
            "timing": {
                "firm_static_seconds": 1.0,
                "investment_seconds": 2.0,
                "bellman_seconds": 3.0,
                "bp_seconds": 4.0,
            },
        }
        return pd.DataFrame([{
            "eta_parent": case_args.eta,
            "n_child_shocks": case_args.n_child_shocks,
        }]), metadata

    monkeypatch.setattr(evaluator_module, "load_analysis_checkpoint", fake_load)
    monkeypatch.setattr(evaluator_module, "_state_hash", fake_hash)
    monkeypatch.setattr(evaluator_module, "evaluate", fake_evaluate)
    args = SimpleNamespace(
        output_dir=tmp_path / "matrix",
        eta_values=[0.0, 1.0],
        eta=1.0,
        n_child_shocks=64,
        robustness_child_shocks=[32, 64, 128],
        device="cpu",
        checkpoint=tmp_path / "checkpoint.pt",
        pv_ckpt=None,
        sdf_ckpt=None,
        hyperparams_json=None,
        config_json=None,
        model_spec_json=None,
        allow_default_hyperparams=False,
        allow_current_config=False,
        bp_teacher_margin_tol=1e-8,
        summary_only_all=True,
        shock_seed=12345,
        robustness_scope="all",
    )
    summary, metadata = evaluator_module.evaluate_matrix(args)

    assert len(summary) == 6
    assert len(load_calls) == 1
    assert len(hash_calls) == 4
    assert metadata["reference_transition_bank"] == {"marker": "eta1-J64"}
    assert metadata["primary_eta"] == 1.0
    assert metadata["primary_n_child_shocks"] == 64
    assert metadata["robustness_scope"] == "all"
    assert metadata["model_state_unchanged"] is True
    assert metadata["timing"]["checkpoint_load_count"] == 1
    assert len(metadata["case_metadata_by_eta_j"]) == 6


def _matrix_metadata_args(tmp_path, **overrides):
    args = SimpleNamespace(
        output_dir=tmp_path / "matrix",
        eta_values=[1.0],
        eta=1.0,
        n_child_shocks=64,
        robustness_child_shocks=None,
        shock_bank_max_child_shocks=None,
        device="cpu",
        checkpoint=tmp_path / "checkpoint.pt",
        pv_ckpt=None,
        sdf_ckpt=None,
        hyperparams_json=None,
        config_json=None,
        model_spec_json=None,
        allow_default_hyperparams=False,
        allow_current_config=False,
        bp_teacher_margin_tol=1e-8,
        summary_only_all=True,
        shock_seed=12345,
        robustness_scope="representative",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _install_fake_matrix(monkeypatch):
    loaded = SimpleNamespace(
        models={
            "policy_value": torch.nn.Linear(1, 1),
            "sdf_fc1": torch.nn.Linear(1, 1),
        },
        metadata={"loaded_model_keys": ["policy_value", "sdf_fc1"]},
    )
    monkeypatch.setattr(evaluator_module, "load_analysis_checkpoint", lambda *a, **k: loaded)
    monkeypatch.setattr(evaluator_module, "_state_hash", lambda model: f"hash-{id(model)}")

    def fake_evaluate(case_args):
        case_args.output_dir.mkdir(parents=True, exist_ok=True)
        return pd.DataFrame([{
            "eta_parent": case_args.eta,
            "n_child_shocks": case_args.n_child_shocks,
        }]), {
            "training_eta_integration_mode": "exact",
            "reference_transition_bank": {
                "shock_bank_max_child_shocks": int(case_args.shock_bank_max_child_shocks),
                "requested_J": int(case_args.n_child_shocks),
            },
            "timing": {"bp_seconds": 0.1},
        }

    monkeypatch.setattr(evaluator_module, "evaluate", fake_evaluate)


def test_matrix_metadata_separates_canonical_and_evaluated_child_shocks(tmp_path, monkeypatch):
    """Canonical bank max must not be confused with the largest evaluated J."""
    _install_fake_matrix(monkeypatch)

    # Case A: normal episode, canonical 128 but only J=64 evaluated.
    args = _matrix_metadata_args(
        tmp_path / "case_a", robustness_child_shocks=None, shock_bank_max_child_shocks=128,
    )
    _, metadata = evaluator_module.evaluate_matrix(args)
    assert metadata["canonical_shock_bank_max_child_shocks"] == 128
    assert metadata["max_evaluated_child_shocks"] == 64
    assert metadata["evaluated_child_shocks"] == [64]
    assert metadata["shock_bank_max_child_shocks"] == 128
    assert metadata["robustness_n_child_shocks"] == [64]
    case_metadata = next(iter(metadata["case_metadata_by_eta_j"].values()))
    assert case_metadata["reference_transition_bank"]["shock_bank_max_child_shocks"] == 128

    # Case B: representative episode, canonical 128 with J=32/64/128 evaluated.
    args = _matrix_metadata_args(
        tmp_path / "case_b",
        robustness_child_shocks=[32, 64, 128],
        shock_bank_max_child_shocks=128,
    )
    _, metadata = evaluator_module.evaluate_matrix(args)
    assert metadata["canonical_shock_bank_max_child_shocks"] == 128
    assert metadata["max_evaluated_child_shocks"] == 128
    assert metadata["evaluated_child_shocks"] == [32, 64, 128]
    assert metadata["shock_bank_max_child_shocks"] == 128

    # Fallback: no explicit canonical value means the largest evaluated J.
    args = _matrix_metadata_args(
        tmp_path / "case_c",
        robustness_child_shocks=[32, 64, 128],
        shock_bank_max_child_shocks=None,
    )
    _, metadata = evaluator_module.evaluate_matrix(args)
    assert metadata["canonical_shock_bank_max_child_shocks"] == 128
    assert metadata["max_evaluated_child_shocks"] == 128


def test_matrix_rejects_canonical_bank_smaller_than_largest_evaluated_j(tmp_path, monkeypatch):
    _install_fake_matrix(monkeypatch)
    args = _matrix_metadata_args(
        tmp_path / "case_bad",
        robustness_child_shocks=[32, 64, 128],
        shock_bank_max_child_shocks=64,
    )
    with pytest.raises(ValueError, match="must cover the largest evaluated J"):
        evaluator_module.evaluate_matrix(args)
