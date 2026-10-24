"""Tests for the firm-side equilibrium convergence evaluators.

Covers (Prompt section 14):
* Test 4 -- the evaluator's formal BP policy is the real Grid-BP teacher
  solution (``BPGridTeacher``), eta-gated: eta=0 rows force ``b_next =
  b_current`` and eta=1 rows execute ``b_next = bp_star``; bp head outputs are
  never used as the formal policy.
* Test 5 -- Q four-caliber semantics (``q_unit`` / ``Q_claim`` /
  ``Q_effective`` / ``recovery``) for hybrid and non-hybrid parameterizations;
  issuance pricing only ever uses the claim caliber.
* Test 6 -- investment and default boundary solving (no crossing /
  multiple crossings / grid-endpoint zero / NaN) and Q region bucketing.
* extra -- frozen-policy and fresh-greedy residuals vanish on a model that
  satisfies the Bellman equation exactly, under ``param_hash_guard`` (evaluation
  only, zero optimizer steps).

Everything is CPU-only and synthetic: no cachedir, no training episode, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.economic_config import AnalysisEconomicConfig
from config import HyperParams
from evaluation.convergence.checkpoint_io import (
    EvalCheckpoint,
    compute_param_hashes,
    param_hash_guard,
)
from evaluation.convergence.eval_bank import build_bank_c, build_transition_for_checkpoint
from evaluation.convergence.firm_bellman import (
    BellmanEvalConfig,
    evaluate_firm_bellman,
    q_caliber_tensors,
    q_issue_claim_at_candidates,
    q_region_masks,
    residual_stat_block,
    split_transition,
)
from evaluation.convergence.firm_policy import (
    PolicyEvalConfig,
    build_policy_grid,
    build_policy_reference,
    compare_policy_across_episodes,
    evaluate_firm_policy,
    objective_slices,
    solve_default_boundary,
    solve_investment_boundary,
)
from evaluation.grids import ReferenceFirmState, build_frozen_grid
from losses import P0Loss, PILoss
from models import PolicyValueModel
from training.bp_grid_teacher import BPGridTeacher


# ---------------------------------------------------------------------------
# synthetic fixtures (no cachedir dependency)
# ---------------------------------------------------------------------------


def _economic_config(**overrides) -> AnalysisEconomicConfig:
    values = {
        "RHO_X": 0.0,
        "SIGMA_X": 0.0,
        "XBAR": -3.0,
        "RHO_Z": 0.0,
        "SIGMA_Z": 0.0,
        "ZBAR": 0.0,
        "ZETA": 0.5,
        "I_THRESHOLD": 0.5,
        "G": 1.14,
        "DELTA": 0.02,
        "PHI": 0.4,
        "TAU": 0.2,
        "KAPPA_B": 0.004,
        "KAPPA_E": 0.025,
        "AIO_WEIGHT": 0.5,
        "ALPHA_Z": 1.0,
        "BETA_Z": 5.0,
        "Z0": 1.0,
    }
    values.update(overrides)
    return AnalysisEconomicConfig.from_dict(values)


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


def _small_model(
    *,
    q_parameterization: str = "hybrid_regime",
    value_scale_mode: str = "none",
    seed: int = 11,
) -> PolicyValueModel:
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
        value_scale_mode=value_scale_mode,
        q_parameterization=q_parameterization,
    )


class FakeSDF(torch.nn.Module):
    """Minimal FC1/SDF stand-in for ``build_transition_for_checkpoint``."""

    def __init__(self, m_value: float = 1.0, hatcf_value: float = -3.0, lnkf_value: float = 4.0):
        super().__init__()
        self.m_value = float(m_value)
        self.hatcf_value = float(hatcf_value)
        self.lnkf_value = float(lnkf_value)

    def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical=True):
        n_rows, n_children, _ = x_curr.shape
        m = torch.full((n_rows, n_children, 1), self.m_value, dtype=x_curr.dtype)
        hatcf = torch.full((n_rows, n_children, 1), self.hatcf_value, dtype=x_curr.dtype)
        lnkf = torch.full((n_rows, n_children, 1), self.lnkf_value, dtype=x_curr.dtype)
        return m, m, m, hatcf, lnkf


def _bank(
    *,
    seed: int = 7,
    n_child_shocks: int = 2,
    x: float = -3.0,
    b_values=(0.7, 1.0),
    z_values=(-0.3, 0.3),
    i_values=(0.1, 0.4),
    eta_values=(0.0, 1.0),
):
    anchors = {
        "Normal": {
            "label": "Normal",
            "quantile": 0.5,
            "x": x,
            "hatcf": x,
            "lnkf": 4.0,
            "hatc_cal": x,
            "lnk_cal": 4.0,
        }
    }
    return build_bank_c(
        seed=seed,
        macro_anchors=anchors,
        b_values=b_values,
        z_values=z_values,
        i_values=i_values,
        eta_values=eta_values,
        n_child_shocks=n_child_shocks,
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


def _losses(economic_config):
    p0_loss = P0Loss(
        delta=economic_config.DELTA,
        tau=economic_config.TAU,
        kappa_b=economic_config.KAPPA_B,
        kappa_e=economic_config.KAPPA_E,
        aio_weight=economic_config.AIO_WEIGHT,
        alpha_z=economic_config.ALPHA_Z,
        beta_z=economic_config.BETA_Z,
        z0=economic_config.Z0,
    )
    pi_loss = PILoss(
        delta=economic_config.DELTA,
        tau=economic_config.TAU,
        g=economic_config.G,
        kappa_b=economic_config.KAPPA_B,
        kappa_e=economic_config.KAPPA_E,
        aio_weight=economic_config.AIO_WEIGHT,
        alpha_z=economic_config.ALPHA_Z,
        beta_z=economic_config.BETA_Z,
        z0=economic_config.Z0,
        b_penalty_weight=0.0,
    )
    return p0_loss, pi_loss


def _independent_teacher(model, economic_config, hyperparams) -> BPGridTeacher:
    p0_loss, pi_loss = _losses(economic_config)
    return BPGridTeacher(
        model,
        p0_loss,
        pi_loss,
        grid_min=float(hyperparams.bp_grid_min),
        grid_max=float(hyperparams.bp_grid_max),
        coarse_size=int(hyperparams.bp_grid_coarse_size),
        fine_size=int(hyperparams.bp_grid_fine_size),
        refine=bool(hyperparams.bp_grid_refine_enabled),
        quadratic_refine=bool(hyperparams.bp_grid_quadratic_refine),
        parent_chunk_size=int(hyperparams.bp_grid_parent_chunk_size),
        candidate_chunk_size=int(hyperparams.bp_grid_candidate_chunk_size),
        max_expanded_states=int(hyperparams.bp_grid_max_expanded_states),
        margin_scale=float(hyperparams.bp_grid_margin_scale),
        confidence_relative=bool(hyperparams.bp_grid_confidence_relative),
        confidence_min=float(hyperparams.bp_grid_confidence_min),
    )


class ExactBellmanMock(torch.nn.Module):
    """Mock that satisfies the P0/PI Bellman equations *exactly*.

    Q is identically zero, so the cashflow does not depend on the refinancing
    candidate; the value outputs are exactly the cashflow functions and are
    negative on the test support, so child equity (``survival * ...`` with
    ``survival = 1{Phat > 0}``) is zero and the continuation term vanishes.
    Both the frozen-policy and the fresh-greedy residuals must then be zero.
    """

    def __init__(self, economic_config: AnalysisEconomicConfig):
        super().__init__()
        self.q_parameterization = "hybrid_regime"
        self.i_grid_size = 5
        self.i_threshold = float(economic_config.I_THRESHOLD)
        self.tau_z = 0.1
        self._delta = float(economic_config.DELTA)
        self._phi = float(economic_config.PHI)
        self._p0_loss, self._pi_loss = _losses(economic_config)

    def _cf0(self, states):
        zeros = torch.zeros_like(states[:, 0:1])
        return self._p0_loss.compute_cashflow_p0(
            states[:, 4:5], states[:, 1:2], states[:, 0:1], zeros, zeros, states[:, 2:3]
        )

    def _cfi(self, states):
        zeros = torch.zeros_like(states[:, 0:1])
        return self._pi_loss.compute_cashflow_pi(
            states[:, 4:5],
            states[:, 1:2],
            states[:, 0:1],
            states[:, 3:4],
            zeros,
            zeros,
            states[:, 2:3],
        )

    def _value_outputs(self, states):
        return self._cf0(states), self._cfi(states)

    def equity_value_scale(self, states):
        return torch.ones_like(states[:, 0:1])

    def forward_value_components(self, states):
        v0, vi = self._value_outputs(states)
        return {
            "V0_physical": v0,
            "VI_physical": vi,
            "V0_normalized": v0,
            "VI_normalized": vi,
            "value_scale": torch.ones_like(v0),
        }

    def forward_equity(self, states):
        i_grid = torch.linspace(
            0.0, self.i_threshold, steps=max(self.i_grid_size, 2),
            device=states.device, dtype=states.dtype,
        )
        v0_list = []
        vi_list = []
        for i_value in i_grid:
            state_i = states.clone()
            state_i[:, 3] = i_value
            v0, vi = self._value_outputs(state_i)
            v0_list.append(v0)
            vi_list.append(vi)
        v0_stack = torch.stack(v0_list, dim=0)
        vi_stack = torch.stack(vi_list, dim=0)
        phat = torch.maximum(v0_stack, vi_stack).mean(dim=0)
        survival_prob = torch.sigmoid(phat / self.tau_z)
        return {
            "Phat": phat,
            "P": torch.clamp_min(phat, 0.0),
            "bar_z": 1.0 - survival_prob,
            "survival_prob": survival_prob,
        }

    def _q_unit_output(self, states):
        return torch.zeros_like(states[:, 0:1])

    def _q_claim_output(self, states):
        return torch.zeros_like(states[:, 0:1])

    def _q_output(self, states, *, phat=None, default_mask=None):
        return torch.zeros_like(states[:, 0:1])

    def _q_recovery_output(self, states):
        return self._phi * (
            1.0 - self._delta + torch.exp(states[:, 4:5] + states[:, 1:2])
        )

    def _q_effective_output(self, states, *, q_claim=None, phat=None, default_mask=None):
        return torch.zeros_like(states[:, 0:1])

    def forward(self, states):
        p0, pi = self._value_outputs(states)
        equity = self.forward_equity(states)
        zeros = torch.zeros_like(p0)
        return SimpleNamespace(
            Q=zeros,
            bp0=zeros,
            bpI=zeros,
            P0=p0,
            PI=pi,
            bar_i=zeros,
            bar_i_cond=zeros,
            bar_i_eff=zeros,
            bar_z=equity["bar_z"],
            P=equity["P"],
            Phat=equity["Phat"],
            bp_cond=zeros,
            bp=zeros,
            survival_prob=equity["survival_prob"],
        )


def test_frozen_and_fresh_residuals_are_zero_for_exact_bellman_model():
    economic = _economic_config()
    model = ExactBellmanMock(economic)
    hp = _hyperparams(bp_grid_min=0.6, bp_grid_max=1.0)
    cp = _checkpoint(model, FakeSDF(m_value=1.0), economic, hp)
    bank = _bank(x=-3.0)

    with param_hash_guard(cp):
        result = evaluate_firm_bellman(
            cp,
            bank,
            config=BellmanEvalConfig(parent_chunk_size=8, dense_grid_size=5, chunk_size=256),
        )

    rows = result.tables["rows"]
    assert len(rows) > 0
    for column in (
        "frozen_r0_physical",
        "frozen_r0_normalized",
        "frozen_ri_physical",
        "frozen_ri_normalized",
        "fresh_r0_physical",
        "fresh_r0_normalized",
        "fresh_ri_physical",
        "fresh_ri_normalized",
    ):
        finite = np.isfinite(rows[column].to_numpy())
        assert finite.all(), f"{column} has non-finite entries"
        assert np.max(np.abs(rows[column].to_numpy())) < 1e-5, column
    assert result.metrics["parity_passed"] == 1.0
    assert result.metrics["parity_max_abs_error"] < 1e-5
    assert result.metrics["optimizer_steps"] == 0.0
    assert result.diagnostics["optimizer_steps"] == 0
    assert result.diagnostics["evaluation_only"] is True


# ---------------------------------------------------------------------------
# Test 4 -- the formal policy is the real Grid-BP teacher solution
# ---------------------------------------------------------------------------


def test_grid_bp_matches_teacher_and_enforces_eta_financing_gate():
    economic = AnalysisEconomicConfig.from_current_config()
    model = _small_model()
    hp = _hyperparams()
    cp = _checkpoint(model, FakeSDF(m_value=0.9), economic, hp)
    bank = _bank()

    result = evaluate_firm_policy(
        cp,
        bank,
        config=PolicyEvalConfig(
            b_points=5, z_points=5, i_points=5, objective_slice_points=5,
            representative_count=2, parent_chunk_size=8,
        ),
    )
    rows = result.tables["policy_rows"]

    # The evaluator's bp_star must equal an independently built BPGridTeacher.
    teacher = _independent_teacher(model, economic, hp)
    transition = build_transition_for_checkpoint(bank, cp.models["sdf_fc1"], hp, economic)
    children_list, m_used, _m_raw, weights = split_transition(transition)
    for branch, column in (("p0", "bp0_star"), ("pi", "bpi_star")):
        expected = teacher.compute_value_target(
            parent_state=transition.parent_states,
            children=children_list,
            m_list=m_used,
            branch=branch,
            child_weights=weights,
        )
        np.testing.assert_allclose(
            rows[column].to_numpy(),
            expected["bp_star"].reshape(-1).detach().numpy(),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            rows[f"value_star_{branch if branch == 'p0' else 'pi'}"].to_numpy(),
            expected["value_star"].reshape(-1).detach().numpy(),
            atol=1e-6,
        )

    # Financing gate: eta=0 forces b_next = b_current, eta=1 executes bp_star.
    eta = rows["eta"].to_numpy()
    b_values = rows["b"].to_numpy()
    for branch, star_column, next_column in (
        ("p0", "bp0_star", "b_next_p0"),
        ("pi", "bpi_star", "b_next_pi"),
    ):
        star = rows[star_column].to_numpy()
        nxt = rows[next_column].to_numpy()
        np.testing.assert_allclose(nxt[eta <= 0.5], b_values[eta <= 0.5], atol=1e-7)
        np.testing.assert_allclose(nxt[eta > 0.5], star[eta > 0.5], atol=1e-7)
    assert result.metrics["eta0_financing_gate_max_error_mix"] < 1e-7
    assert result.metrics["eta1_financing_gate_max_error_mix"] < 1e-7

    # eta = 0 rows carry no BP choice: grid diagnostics are NaN sentinels there.
    assert np.isnan(rows.loc[eta <= 0.5, "margin_p0"].to_numpy()).all()
    assert np.isfinite(rows.loc[eta > 0.5, "margin_p0"].to_numpy()).all()


def test_bp_head_outputs_are_not_used_as_the_formal_policy():
    economic = AnalysisEconomicConfig.from_current_config()
    model = _small_model(seed=23)
    hp = _hyperparams()
    cp = _checkpoint(model, FakeSDF(m_value=0.9), economic, hp)
    bank = _bank()

    result = evaluate_firm_policy(
        cp,
        bank,
        config=PolicyEvalConfig(b_points=5, z_points=5, i_points=5, objective_slice_points=5),
    )
    rows = result.tables["policy_rows"]
    eta = rows["eta"].to_numpy()
    head_gap = np.abs(rows["bp0_head"].to_numpy() - rows["bp0_star"].to_numpy())
    assert (head_gap > 1e-6).any(), "head and grid policy should differ somewhere"

    # The executed leverage follows the grid star, never the head, wherever they
    # differ; the head is recorded only as a compat column with its regret.
    differing = head_gap > 1e-6
    np.testing.assert_allclose(
        rows.loc[differing & (eta > 0.5), "b_next_p0"].to_numpy(),
        rows.loc[differing & (eta > 0.5), "bp0_star"].to_numpy(),
        atol=1e-7,
    )
    assert result.diagnostics["formal_policy"].startswith("grid_bp")
    assert "compatibility columns only" in result.diagnostics["bp_head_semantics"]
    assert result.metrics["bp_head_regret_p0_eta1_max"] >= 0.0
    assert 0.0 <= result.metrics["bp_head_match_rate_p0_eta1"] <= 1.0


# ---------------------------------------------------------------------------
# Test 5 -- Q four-caliber semantics
# ---------------------------------------------------------------------------


class QCaliberMock(torch.nn.Module):
    """Known-Q mock: constant unit bond price, controlled Phat envelope."""

    def __init__(
        self,
        economic_config: AnalysisEconomicConfig,
        *,
        q_parameterization: str,
        q_unit_value: float = 0.5,
    ):
        super().__init__()
        self.q_parameterization = str(q_parameterization)
        self.q_unit_value = float(q_unit_value)
        self.i_grid_size = 3
        self.i_threshold = float(economic_config.I_THRESHOLD)
        self.tau_z = 0.1
        self._delta = float(economic_config.DELTA)
        self._phi = float(economic_config.PHI)

    # Phat is positive exactly when z > b: a plain, checkable envelope.
    def _phat(self, states):
        return (states[:, 1:2] - states[:, 0:1]) + 0.25

    def forward_equity(self, states):
        phat = self._phat(states)
        survival = torch.sigmoid(phat / self.tau_z)
        return {
            "Phat": phat,
            "P": torch.clamp_min(phat, 0.0),
            "bar_z": 1.0 - survival,
            "survival_prob": survival,
        }

    def _q_unit_output(self, states):
        return torch.full_like(states[:, 0:1], self.q_unit_value)

    def _q_claim_output(self, states):
        return torch.clamp(states[:, 0:1], min=0.0) * self._q_unit_output(states)

    def _q_output(self, states, *, phat=None, default_mask=None):
        return self._q_claim_output(states)

    def _q_recovery_output(self, states):
        return self._phi * (
            1.0 - self._delta + torch.exp(states[:, 4:5] + states[:, 1:2])
        )

    def _q_effective_output(self, states, *, q_claim=None, phat=None, default_mask=None):
        claim = self._q_claim_output(states) if q_claim is None else q_claim
        if self.q_parameterization != "hybrid_regime":
            return claim
        if phat is None:
            phat = self._phat(states)
        default_mask = phat <= 0.0
        recovery = self._q_recovery_output(states)
        zero_mask = states[:, 0:1] <= 0.0
        return torch.where(
            zero_mask,
            torch.zeros_like(claim),
            torch.where(default_mask, recovery, claim),
        )

    def forward(self, states):
        equity = self.forward_equity(states)
        zeros = torch.zeros_like(states[:, 0:1])
        return SimpleNamespace(
            Q=self._q_effective_output(states, phat=equity["Phat"]),
            bp0=zeros,
            bpI=zeros,
            P0=zeros,
            PI=zeros,
            bar_i=zeros,
            bar_i_cond=zeros,
            bar_i_eff=zeros,
            bar_z=equity["bar_z"],
            P=equity["P"],
            Phat=equity["Phat"],
            bp_cond=zeros,
            bp=zeros,
            survival_prob=equity["survival_prob"],
        )


def _q_states():
    # (b, z, eta, i, x, hatcf, lnkf) rows covering b<=0, default and survival.
    return torch.tensor(
        [
            [0.00, 0.50, 1.0, 0.1, -3.0, -3.0, 4.0],   # b = 0
            [-0.2, 0.50, 1.0, 0.1, -3.0, -3.0, 4.0],   # b < 0
            [1.00, 0.10, 1.0, 0.1, -3.0, -3.0, 4.0],   # default: z - b + 0.25 < 0
            [0.20, 1.00, 1.0, 0.1, -3.0, -3.0, 4.0],   # survival
            [1.50, -1.0, 0.0, 0.2, -3.0, -3.0, 4.0],   # deep default
        ],
        dtype=torch.float32,
    )


def test_q_caliber_relations_hybrid_and_non_hybrid():
    economic = _economic_config()
    states = _q_states()
    b = states[:, 0:1]

    for parameterization in ("hybrid_regime", "b_times_unit"):
        model = QCaliberMock(economic, q_parameterization=parameterization)
        calibers = q_caliber_tensors(
            model,
            states,
            economic_config=economic,
            recovery_normalization_mode="asset_only",
            chunk_size=2,
        )
        q_unit = calibers["q_unit"]
        claim = calibers["Q_claim"]
        effective = calibers["Q_effective"]
        recovery = calibers["recovery"]
        phat = calibers["Phat"]

        # Four calibers: unit price / claim / recovery.
        torch.testing.assert_close(q_unit, torch.full_like(q_unit, 0.5))
        torch.testing.assert_close(claim, torch.clamp(b, min=0.0) * 0.5)
        torch.testing.assert_close(
            recovery,
            economic.PHI
            * (1.0 - economic.DELTA + torch.exp(states[:, 4:5] + states[:, 1:2])),
        )

        default_mask = (phat <= 0.0).reshape(-1)
        zero_mask = (b.reshape(-1) <= 0.0)
        if parameterization == "hybrid_regime":
            # Settlement hard gate: b <= 0 -> 0, default -> recovery, else claim.
            assert torch.all(effective[zero_mask] == 0.0)
            non_zero_default = default_mask & ~zero_mask
            torch.testing.assert_close(effective[non_zero_default], recovery[non_zero_default])
            survivors = ~default_mask & ~zero_mask
            torch.testing.assert_close(effective[survivors], claim[survivors])
        else:
            # Non-hybrid: effective is the claim caliber everywhere.
            torch.testing.assert_close(effective, claim)


def test_issuance_pricing_uses_claim_only():
    economic = _economic_config()
    states = _q_states()
    model = QCaliberMock(economic, q_parameterization="hybrid_regime")
    candidates = torch.tensor(
        [[0.3], [0.4], [1.0], [0.1], [1.5]], dtype=torch.float32
    )
    issued = q_issue_claim_at_candidates(model, states, candidates)
    expected = torch.clamp(candidates, min=0.0) * 0.5
    torch.testing.assert_close(issued, expected)

    # Even at candidates whose Phat is deep in the default region the issuance
    # price stays the claim; recovery never substitutes here.
    issue_states = states.clone()
    issue_states[:, 0:1] = candidates
    phat_issue = model._phat(issue_states).reshape(-1)
    recovery_issue = model._q_recovery_output(issue_states).reshape(-1)
    deep_default = phat_issue <= 0.0
    assert deep_default.any()
    assert torch.all(issued.reshape(-1)[deep_default] != recovery_issue[deep_default])
    assert torch.allclose(
        issued.reshape(-1)[deep_default],
        (torch.clamp(candidates, min=0.0) * 0.5).reshape(-1)[deep_default],
    )


# ---------------------------------------------------------------------------
# Test 6 -- investment / default boundary solving and Q region bucketing
# ---------------------------------------------------------------------------


class InvestmentShapeMock(torch.nn.Module):
    """``VI - V0`` shape is encoded in the b column of each grid row."""

    CASE_BY_B = {0.1: 1, 0.2: 2, 0.3: 3, 0.4: 4, 0.5: 5, 0.6: 6}

    def __init__(self, i_threshold: float = 0.5):
        super().__init__()
        self.i_threshold = float(i_threshold)
        self.q_parameterization = "direct"
        self.i_grid_size = 3

    def forward_value_components(self, states):
        i_value = states[:, 3:4]
        case = torch.round(states[:, 0:1] * 10.0).to(torch.long)
        delta = torch.zeros_like(i_value)
        delta = torch.where(case == 1, torch.ones_like(i_value), delta)              # always invest
        delta = torch.where(case == 2, -torch.ones_like(i_value), delta)             # never invest
        delta = torch.where(case == 3, i_value - 0.15, delta)                        # one crossing
        delta = torch.where(case == 4, (i_value - 0.10) * (i_value - 0.30), delta)   # two crossings
        delta = torch.where(case == 5, i_value, delta)                               # zero at i = 0
        delta = torch.where(case == 6, torch.full_like(i_value, float("nan")), delta)
        v0 = torch.zeros_like(delta)
        return {
            "V0_physical": v0,
            "VI_physical": delta,
            "V0_normalized": v0,
            "VI_normalized": delta,
            "value_scale": torch.ones_like(delta),
        }


def _boundary_reference(eta: float) -> ReferenceFirmState:
    return ReferenceFirmState(
        eta=eta, i_low=0.1, i_mid=0.25, i_high=0.4,
        x=-3.0, hatcf=-3.0, lnkf=4.0, hatc_cal=-3.0, lnk_cal=4.0,
        n_parent_rows=1, source="fixture", macro_source="fixture",
    )


def test_investment_boundary_covers_uniqueness_failures():
    reference = _boundary_reference(1.0)
    grid = build_frozen_grid(
        reference,
        b_min=0.1, b_max=0.6, b_points=6,
        z_min=-0.5, z_max=0.5, z_points=2,
        device=torch.device("cpu"),
    )
    model = InvestmentShapeMock()
    survival_mask = np.ones(grid.shape, dtype=bool)
    frame, metrics, arrays = solve_investment_boundary(
        model,
        grid,
        reference,
        config=PolicyEvalConfig(i_points=51),
        survival_mask=survival_mask,
        i_max=0.5,
    )

    def _for_b(value):
        return frame[np.isclose(frame["b"], value)].iloc[0]

    always = _for_b(0.1)
    assert always["investment_status_label"] == "always_invest"
    assert np.isnan(always["i_star"])
    never = _for_b(0.2)
    assert never["investment_status_label"] == "never_invest"
    single = _for_b(0.3)
    assert single["investment_status_label"] == "single_crossing"
    assert single["i_star"] == pytest.approx(0.15, abs=2e-3)
    multi = _for_b(0.4)
    assert multi["investment_status_label"] == "multiple_crossings"
    assert multi["crossing_count"] == 2
    np.testing.assert_allclose(
        np.asarray(json.loads(multi["crossings_json"]), dtype=np.float64),
        [0.1, 0.3],
        atol=1e-6,
    )
    assert len(json.loads(multi["crossing_intervals_json"])) == 1
    endpoint = _for_b(0.5)
    assert endpoint["investment_status_label"] == "single_crossing"
    assert endpoint["i_star"] == pytest.approx(0.0, abs=1e-9)
    nonfinite = _for_b(0.6)
    assert nonfinite["investment_status_label"] == "nonfinite"

    assert metrics["investment_always_invest_share_raw"] == pytest.approx(1.0 / 6.0)
    assert metrics["investment_never_invest_share_raw"] == pytest.approx(1.0 / 6.0)
    assert metrics["investment_multiple_crossings_share_raw"] == pytest.approx(1.0 / 6.0)
    assert metrics["investment_multi_crossing_rows"] == 2.0
    assert arrays["delta_vi_v0"].shape == (6, 2, 51)


def test_default_boundary_classifies_and_keeps_all_crossings():
    reference = _boundary_reference(1.0)
    grid = build_frozen_grid(
        reference,
        b_min=0.1, b_max=0.5, b_points=5,
        z_min=-2.0, z_max=2.0, z_points=5,
        device=torch.device("cpu"),
    )
    z_values = np.asarray(grid.z_values, dtype=np.float64)
    phat = np.vstack(
        [
            np.ones_like(z_values),            # all survival
            -np.ones_like(z_values),           # all default
            z_values.copy(),                   # single crossing at z = 0
            z_values ** 2 - 1.0,               # crossings at z = -1 and z = 1
            np.full_like(z_values, np.nan),    # nonfinite
        ]
    )
    bar_z = 1.0 - 1.0 / (1.0 + np.exp(np.nan_to_num(phat, nan=0.0) / 0.1))
    bar_z = np.where(np.isfinite(phat), bar_z, np.nan)

    frame, metrics, arrays = solve_default_boundary(grid, phat, bar_z)
    statuses = list(frame["boundary_status"])
    assert statuses == [
        "all_survival",
        "all_default",
        "single_crossing",
        "multiple_crossings",
        "nonfinite",
    ]
    z_default = frame["z_default"].to_numpy(dtype=np.float64)
    assert np.isnan(z_default[[0, 1, 3, 4]]).all()
    assert z_default[2] == pytest.approx(0.0, abs=1e-9)
    multi_crossings = json.loads(frame.loc[3, "crossings_json"])
    np.testing.assert_allclose(sorted(multi_crossings), [-1.0, 1.0], atol=1e-9)
    # multiple crossings never collapse into the single z_default column
    assert np.isnan(arrays["z_default"][3])

    for status in (
        "single_crossing",
        "multiple_crossings",
        "all_default",
        "all_survival",
        "nonfinite",
    ):
        assert metrics[f"default_{status}_share"] == pytest.approx(0.2)
    assert "hard_soft_boundary_abs_gap_mean" in metrics
    assert set(["z_hard_phat0", "z_soft_barz0p5"]).issubset(arrays)


def test_q_region_masks_bucket_states_and_stat_block_stratifies():
    b = np.array([0.0, 0.5, 0.5, 0.5, 0.5])
    phat = np.array([-1.0, -0.5, 0.5, 0.005, 0.5])
    bar_z = np.array([0.5, 0.8, 0.2, 0.5, 0.95])
    masks = q_region_masks(b, phat, bar_z, phat_eps=1e-2, bar_z_low=0.1, bar_z_high=0.9)
    np.testing.assert_array_equal(masks["q0"], [True, False, False, False, False])
    np.testing.assert_array_equal(masks["qd"], [False, True, False, False, False])
    np.testing.assert_array_equal(masks["qs"], [False, False, True, True, True])
    np.testing.assert_array_equal(masks["qboundary"], [True, True, True, True, False])
    # q0/qd/qs form a partition; the boundary band overlaps by design.
    partition = masks["q0"].astype(int) + masks["qd"].astype(int) + masks["qs"].astype(int)
    np.testing.assert_array_equal(partition, np.ones(5, dtype=int))

    values = np.array([1.0, -2.0, 3.0, np.nan])
    eta = np.array([0.0, 1.0, 0.0, 1.0])
    source = ["simulated", "coverage", "stress", "simulated"]
    block = residual_stat_block(
        values,
        prefix="demo",
        level=np.array([1.0, 1.0, 1.0, 1.0]),
        eta=eta,
        source=source,
    )
    assert block["demo_n"] == 4.0
    assert block["demo_n_finite"] == 3.0
    assert block["demo_nonfinite_ratio"] == pytest.approx(0.25)
    assert block["demo_mae"] == pytest.approx(2.0)
    assert block["demo_eta0_n"] == 2.0
    assert block["demo_eta1_mae"] == pytest.approx(2.0)
    assert block["demo_simulated_n"] == 2.0
    assert block["demo_relative_mae"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# cross-episode drift and evaluation-only smoke
# ---------------------------------------------------------------------------


def test_cross_episode_policy_drift_and_flip_rates():
    economic = AnalysisEconomicConfig.from_current_config()
    hp = _hyperparams()
    bank = _bank()
    config = PolicyEvalConfig(
        b_points=5, z_points=5, i_points=5, objective_slice_points=5, policy_flip_tol=1e-2
    )

    def _run(episode, seed):
        model = _small_model(seed=seed)
        cp = _checkpoint(model, FakeSDF(m_value=0.9), economic, hp, episode=episode)
        return evaluate_firm_policy(cp, bank, config=config)

    first = _run(0, 5)
    repeated = _run(0, 5)
    changed = _run(1, 31)

    identical = compare_policy_across_episodes([first, repeated])
    assert len(identical) == 1
    assert identical.loc[0, "bp_star_drift_max"] == pytest.approx(0.0, abs=1e-12)
    assert identical.loc[0, "policy_flip_rate"] == 0.0
    assert identical.loc[0, "default_flip_rate"] == 0.0
    assert identical.loc[0, "investment_flip_rate"] == 0.0

    drift = compare_policy_across_episodes([first, changed])
    assert len(drift) == 1
    assert drift.loc[0, "bp_star_drift_max"] >= 0.0
    for key in ("policy_flip_rate", "default_flip_rate", "investment_flip_rate"):
        assert 0.0 <= drift.loc[0, key] <= 1.0
    assert drift.loc[0, "n_eta1_rows"] == int(
        (first.tables["policy_rows"]["eta"].to_numpy() > 0.5).sum()
    )

    # Refuse to compare results computed on different fixed bank states.
    other_bank = _bank(b_values=(0.3, 0.9))
    other = _run(0, 5)
    other_cp = _checkpoint(_small_model(seed=5), FakeSDF(m_value=0.9), economic, hp)
    other = evaluate_firm_policy(other_cp, other_bank, config=config)
    with pytest.raises(ValueError):
        compare_policy_across_episodes([first, other])
    with pytest.raises(ValueError):
        compare_policy_across_episodes([first])


def test_evaluate_firm_bellman_smoke_is_evaluation_only_and_dual_caliber():
    economic = AnalysisEconomicConfig.from_current_config()
    model = _small_model(value_scale_mode="exp_xz", seed=13)
    hp = _hyperparams(
        pv_value_scale_mode="exp_xz",
        pv_value_scale_log_max=20.0,
        pv_bellman_normalize_by_value_scale=True,
    )
    cp = _checkpoint(model, FakeSDF(m_value=0.9), economic, hp)
    bank = _bank(i_values=(0.1,), eta_values=(1.0,))

    with param_hash_guard(cp):
        result = evaluate_firm_bellman(
            cp,
            bank,
            config=BellmanEvalConfig(parent_chunk_size=3, dense_grid_size=9, chunk_size=64),
        )

    assert set(result.tables) == {
        "rows",
        "residual_stats",
        "q_region_stats",
        "bp_discretization",
        "parity",
    }
    assert len(result.tables["rows"]) == 4
    assert {
        "frozen_p0_physical_mae",
        "frozen_p0_normalized_mae",
        "fresh_pi_normalized_mae",
        "frozen_p0_physical_eta1_mae",
        "frozen_p0_physical_simulated_n",
        "q_qs_claim_bellman_trainM_mae",
        "parity_max_abs_error",
        "bp_grid_discretization_bp_gap_mean",
        "optimizer_steps",
    }.issubset(set(result.metrics))
    assert result.metrics["optimizer_steps"] == 0.0
    assert result.diagnostics["optimizer_steps"] == 0

    # Physical and normalized calibers are both reported and annotated.
    assert result.diagnostics["value_parameterization"]["mode"] == "exp_xz"
    assert "normalized" in result.diagnostics["caliber_note"]
    assert "physical" in result.diagnostics["caliber_note"]
    np.testing.assert_allclose(
        result.surfaces["P0_normalized"] * result.tables["rows"]["value_scale"].to_numpy().reshape(2, 2),
        result.surfaces["P0_physical"],
        rtol=1e-6,
    )
    # Grid-shaped banks are reshaped to (b, z) surfaces for downstream figures.
    assert result.surfaces["Q_claim"].shape == (2, 2)
    assert result.surfaces["R0_frozen_physical"].shape == (2, 2)
    assert set(["trainM", "rawM"]) == {
        result.tables["q_region_stats"]["m_mode"].unique()[0],
        result.tables["q_region_stats"]["m_mode"].unique()[1],
    }


def test_objective_slices_report_claim_overlay_and_eta0_flatness():
    economic = AnalysisEconomicConfig.from_current_config()
    model = _small_model(seed=29)
    hp = _hyperparams()
    cp = _checkpoint(model, FakeSDF(m_value=0.9), economic, hp)
    bank = _bank()
    transition = build_transition_for_checkpoint(bank, cp.models["sdf_fc1"], hp, economic)

    frame, metrics = objective_slices(
        cp,
        transition,
        config=PolicyEvalConfig(objective_slice_points=7),
        model=model,
        representative=[0, 1],
    )
    assert set(["bellman_rhs", "cashflow", "continuation", "q_claim_issue"]).issubset(
        set(frame.columns)
    )
    assert set(frame["branch"]) == {"p0", "pi"}
    # Bank rows 0/1 are one eta = 0 and one eta = 1 state.
    assert (frame["eta"] <= 0.5).any() and (frame["eta"] > 0.5).any()
    # eta = 0: the financing gate forces b_next = b_current, so the Bellman
    # objective must be flat in the candidate bp.
    assert metrics["objective_slice_eta0_flatness_max_abs"] < 1e-6
    # the overlay is the issuance claim caliber, never the effective caliber
    assert np.isfinite(frame["q_claim_issue"].to_numpy()).all()
