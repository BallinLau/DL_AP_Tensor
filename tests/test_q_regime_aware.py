"""Firm-side Q regime / recovery 口径的单元测试（TEST 1-8）。

锁定三件事：
1. Q 违约回收的经济单位（asset recovery，不含债务面值 b）；
2. parent default regime gating（surviving vs deep default vs transition band）；
3. child default pricing 与 parent gating 是两件事，不能相互删掉；
4. evaluator 的 Q_target / recovery / regime 分解契约。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from analysis.economic_config import AnalysisEconomicConfig
from config import Config
from evaluation.bellman_diagnostics import evaluate_bellman_residuals
from evaluation.bp_diagnostics import FrozenTransitionData
from evaluation.firm_surfaces import evaluate_firm_surfaces
from evaluation.grids import ReferenceFirmState, build_frozen_grid
from losses.q_loss import (
    QLoss,
    build_q_polish_coverage_weights,
    compute_parent_default_regime_weights,
    compute_q_survival_recovery_components,
    compute_recovery_target,
)
from models import PolicyValueModel
from training.episode import Episode
from losses.utils import compute_aio_residual


# ---------------------------------------------------------------------------
# TEST 1: recovery normalization
# ---------------------------------------------------------------------------

def test_recovery_normalization_asset_only_ignores_leverage():
    """同一 (x, z) 下，asset_only 回收不随 b 变化；legacy 口径按 b 线性放大。"""
    b = torch.tensor([[0.1], [0.9]], dtype=torch.float64)
    x = torch.tensor([[0.2], [0.2]], dtype=torch.float64)
    z = torch.tensor([[-0.3], [-0.3]], dtype=torch.float64)

    asset_only = compute_recovery_target(
        b, x, z, phi=Config.PHI, delta=Config.DELTA,
        recovery_normalization_mode="asset_only",
    )
    unit = Config.PHI * (1.0 - Config.DELTA + torch.exp(x[0:1] + z[0:1]))
    # 与 main_4.tex:344 / :968-969 一致：回收只来自资产，不含 b。
    torch.testing.assert_close(asset_only[0:1], unit)
    torch.testing.assert_close(asset_only[1:2], unit)

    legacy = compute_recovery_target(
        b, x, z, phi=Config.PHI, delta=Config.DELTA,
        recovery_normalization_mode="legacy_b_times_unit",
    )
    torch.testing.assert_close(legacy, b * unit)
    assert not torch.allclose(asset_only, legacy)


def test_default_config_recovery_mode_is_asset_only():
    assert Config.RECOVERY_NORMALIZATION_MODE == "asset_only"
    assert QLoss().recovery_normalization_mode == "asset_only"


# ---------------------------------------------------------------------------
# TEST 2: surviving parent -> Bellman only
# ---------------------------------------------------------------------------

def test_surviving_parent_uses_bellman_only():
    weights = compute_parent_default_regime_weights(
        torch.tensor([[0.5]], dtype=torch.float64), mode="hard"
    )
    assert weights["parent_survival_weight"].item() == 1.0
    assert weights["parent_default_weight"].item() == 0.0

    loss_fn = QLoss(parent_default_regime_mode="hard")
    Q = torch.tensor([[0.4]], dtype=torch.float64)
    b = torch.tensor([[0.3]], dtype=torch.float64)
    x = torch.tensor([[0.2]], dtype=torch.float64)
    z = torch.tensor([[0.1]], dtype=torch.float64)
    terms = loss_fn.compute_regime_aware_objective(
        residuals=loss_fn.compute_main_residual(
            Q, b, torch.zeros_like(b),
            M_list=[torch.tensor([[0.9]], dtype=torch.float64)],
            Qsp_children=[torch.tensor([[0.5]], dtype=torch.float64)],
            bar_z_children=[torch.tensor([[0.2]], dtype=torch.float64)],
            x_children=[x], z_children=[z],
        ),
        Q=Q, b=b, x=x, z=z,
        phat=torch.tensor([[0.5]], dtype=torch.float64),
    )
    assert terms["recovery_loss"].item() == 0.0
    torch.testing.assert_close(terms["bellman_loss"], terms["aio_residual"].mean())


# ---------------------------------------------------------------------------
# TEST 3: deep default parent -> current recovery only
# ---------------------------------------------------------------------------

def test_deep_default_parent_uses_current_recovery_only():
    weights = compute_parent_default_regime_weights(
        torch.tensor([[-1.0]], dtype=torch.float64), mode="hard"
    )
    assert weights["parent_survival_weight"].item() == 0.0
    assert weights["parent_default_weight"].item() == 1.0

    loss_fn = QLoss(parent_default_regime_mode="hard")
    Q = torch.tensor([[0.9]], dtype=torch.float64)
    b = torch.tensor([[0.3]], dtype=torch.float64)
    x = torch.tensor([[0.2]], dtype=torch.float64)
    z = torch.tensor([[-1.0]], dtype=torch.float64)
    terms = loss_fn.compute_regime_aware_objective(
        residuals=loss_fn.compute_main_residual(
            Q, b, torch.zeros_like(b),
            M_list=[torch.tensor([[0.9]], dtype=torch.float64)],
            Qsp_children=[torch.tensor([[0.5]], dtype=torch.float64)],
            bar_z_children=[torch.tensor([[0.8]], dtype=torch.float64)],
            x_children=[x], z_children=[z],
        ),
        Q=Q, b=b, x=x, z=z,
        phat=torch.tensor([[-1.0]], dtype=torch.float64),
    )
    # Bellman continuation 不再影响 Q loss。
    assert terms["bellman_loss"].item() == 0.0
    recovery = loss_fn.compute_current_recovery(b, x, z)
    torch.testing.assert_close(terms["recovery_loss"], (recovery - Q).pow(2).mean())

    target = loss_fn.compute_regime_aware_q_target(
        q_target_bellman=torch.tensor([[5.0]], dtype=torch.float64),
        recovery_current=recovery,
        phat=torch.tensor([[-1.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(target["q_target_used_for_training"], recovery)


# ---------------------------------------------------------------------------
# TEST 4: transition band
# ---------------------------------------------------------------------------

def test_transition_band_weights_smooth_inside_and_collapse_outside():
    inside = compute_parent_default_regime_weights(
        torch.tensor([[0.0]], dtype=torch.float64),
        mode="transition_band", eps=1e-2, tau=1e-2,
    )
    survival = inside["parent_survival_weight"].item()
    default = inside["parent_default_weight"].item()
    assert 0.0 < survival < 1.0
    assert 0.0 < default < 1.0
    assert abs(survival + default - 1.0) < 1e-12

    deep = compute_parent_default_regime_weights(
        torch.tensor([[-1.0]], dtype=torch.float64),
        mode="transition_band", eps=1e-2, tau=1e-2,
    )
    assert deep["parent_survival_weight"].item() == 0.0
    assert deep["parent_default_weight"].item() == 1.0

    survivor = compute_parent_default_regime_weights(
        torch.tensor([[1.0]], dtype=torch.float64),
        mode="transition_band", eps=1e-2, tau=1e-2,
    )
    assert survivor["parent_survival_weight"].item() == 1.0


# ---------------------------------------------------------------------------
# TEST 5: child default semantics survive parent gating
# ---------------------------------------------------------------------------

def test_child_default_pricing_is_kept_for_surviving_parent():
    components = compute_q_survival_recovery_components(
        Q=torch.tensor([[0.4]], dtype=torch.float64),
        b=torch.tensor([[0.3]], dtype=torch.float64),
        bar_i=torch.zeros(1, 1, dtype=torch.float64),
        M=torch.tensor([[0.9]], dtype=torch.float64),
        Qsp=torch.tensor([[0.5]], dtype=torch.float64),
        bar_z=torch.tensor([[0.8]], dtype=torch.float64),
        x_child=torch.tensor([[0.2]], dtype=torch.float64),
        z_child=torch.tensor([[-0.5]], dtype=torch.float64),
        g=Config.G, delta=Config.DELTA, phi=Config.PHI,
    )
    # 即使 child default 概率很高，Bellman target 仍含 child recovery 分量。
    assert components["q_target_recovery"].item() > 0.0
    torch.testing.assert_close(
        components["q_target_total"],
        components["q_target_survival"] + components["q_target_recovery"],
    )

    loss_fn = QLoss(parent_default_regime_mode="hard")
    Q = torch.tensor([[0.4]], dtype=torch.float64)
    b = torch.tensor([[0.3]], dtype=torch.float64)
    x = torch.tensor([[0.2]], dtype=torch.float64)
    z = torch.tensor([[0.1]], dtype=torch.float64)
    q_bellman = torch.tensor([[0.37]], dtype=torch.float64)
    target = loss_fn.compute_regime_aware_q_target(
        q_target_bellman=q_bellman,
        recovery_current=loss_fn.compute_current_recovery(b, x, z),
        phat=torch.tensor([[0.5]], dtype=torch.float64),
    )
    # surviving parent 的 target 完全等于 Bellman（含 child default 项）。
    torch.testing.assert_close(target["q_target_used_for_training"], q_bellman)


# ---------------------------------------------------------------------------
# evaluator-level fixtures (TEST 6-8)
# ---------------------------------------------------------------------------

class _QRegimeModel(torch.nn.Module):
    """Q = 2b 的确定性 mock；Phat / bar_z 由构造参数固定。"""

    def __init__(self, phat_value: float, bar_z_value: float):
        super().__init__()
        self.phat_value = float(phat_value)
        self.bar_z_value = float(bar_z_value)

    def forward(self, states):
        b = states[:, 0:1]
        ones = torch.ones_like(b)
        return SimpleNamespace(
            Q=2.0 * b, bp0=0.2 * ones, bpI=0.8 * ones, P0=ones, PI=ones,
            bar_i=0.5 * ones, bar_i_cond=0.5 * ones, bar_i_eff=0.5 * ones,
            bar_z=torch.full_like(b, self.bar_z_value),
            P=ones, Phat=torch.full_like(b, self.phat_value),
            bp_cond=0.5 * ones, bp=0.5 * ones, survival_prob=ones,
        )

    def equity_value_scale(self, states):
        return torch.ones_like(states[:, 0:1])


def _reference() -> ReferenceFirmState:
    return ReferenceFirmState(
        eta=1.0, i_low=0.1, i_mid=0.2, i_high=0.3,
        x=-2.0, hatcf=-2.2, lnkf=4.0, hatc_cal=-2.1, lnk_cal=4.1,
        n_parent_rows=1, source="fixture", macro_source="fixture",
    )


def _grid():
    return build_frozen_grid(
        _reference(), b_min=0.2, b_max=0.8, b_points=3,
        z_min=-1.0, z_max=1.0, z_points=3, device=torch.device("cpu"),
    )


def _transition(grid) -> FrozenTransitionData:
    children, m_raw, m_used = [], [], []
    for eta in (0.0, 1.0):
        child = grid.base_states.clone()
        child[:, 2] = float(eta)
        children.append(child)
        m_raw.append(torch.ones(len(child), 1))
        m_used.append(torch.ones(len(child), 1))
    return FrozenTransitionData(
        children=children, m_raw_list=m_raw, m_used_list=m_used,
        branch_weights=torch.full((len(grid.base_states), len(children)), 0.5),
        metadata={"m_mode": "raw_sdf_m"},
    )


# ---------------------------------------------------------------------------
# TEST 6: evaluator decomposition identity
# ---------------------------------------------------------------------------

def test_evaluator_q_target_decomposition_identity():
    grid = _grid()
    surfaces, _ = evaluate_bellman_residuals(
        _QRegimeModel(phat_value=0.5, bar_z_value=0.3),
        grid, _transition(grid), AnalysisEconomicConfig.from_current_config(),
    )
    for label in ("trainM", "rawM"):
        error = np.nanmax(np.abs(
            surfaces[f"Q_target_{label}"]
            - (surfaces[f"Q_target_survival_{label}"] + surfaces[f"Q_target_recovery_{label}"])
        ))
        assert error < 1e-6
    canonical_error = np.nanmax(np.abs(
        surfaces["Q_target"]
        - (surfaces["Q_target_survival"] + surfaces["Q_target_recovery"])
    ))
    assert canonical_error < 1e-6


# ---------------------------------------------------------------------------
# TEST 7: RQ_signed = Q_target - Q
# ---------------------------------------------------------------------------

def test_evaluator_signed_q_residual_identity():
    grid = _grid()
    model = _QRegimeModel(phat_value=0.5, bar_z_value=0.3)
    surfaces, _ = evaluate_bellman_residuals(
        model, grid, _transition(grid), AnalysisEconomicConfig.from_current_config(),
    )
    firm = evaluate_firm_surfaces(model, grid, _reference())
    q = firm["Q"]
    error = np.nanmax(np.abs(surfaces["RQ_trainM_signed"] - (surfaces["Q_target_trainM"] - q)))
    assert error < 1e-5


# ---------------------------------------------------------------------------
# TEST 8: deep default fixture -> training target == current recovery
# ---------------------------------------------------------------------------

def test_evaluator_deep_default_training_target_is_current_recovery():
    grid = _grid()
    surfaces, _ = evaluate_bellman_residuals(
        _QRegimeModel(phat_value=-1.0, bar_z_value=0.9),
        grid, _transition(grid), AnalysisEconomicConfig.from_current_config(),
        parent_default_regime_mode="hard", parent_default_eps=1e-2,
    )
    np.testing.assert_allclose(surfaces["parent_survival_weight"], 0.0, atol=1e-6)
    np.testing.assert_allclose(surfaces["parent_default_weight"], 1.0, atol=1e-6)
    np.testing.assert_allclose(surfaces["parent_hard_default"], 1.0, atol=1e-6)
    np.testing.assert_allclose(
        surfaces["Q_target_used_for_training_trainM"],
        surfaces["recovery_current"],
        atol=1e-5,
    )


def test_evaluator_recovery_current_follows_normalization_mode():
    grid = _grid()
    asset_only, _ = evaluate_bellman_residuals(
        _QRegimeModel(phat_value=-1.0, bar_z_value=0.9),
        grid, _transition(grid), AnalysisEconomicConfig.from_current_config(),
        recovery_normalization_mode="asset_only",
    )
    legacy, _ = evaluate_bellman_residuals(
        _QRegimeModel(phat_value=-1.0, bar_z_value=0.9),
        grid, _transition(grid), AnalysisEconomicConfig.from_current_config(),
        recovery_normalization_mode="legacy_b_times_unit",
    )
    b = np.asarray(grid.mesh_b)
    np.testing.assert_allclose(
        legacy["recovery_current"], b * asset_only["recovery_current"], rtol=1e-5,
    )
    assert not np.allclose(asset_only["recovery_current"], legacy["recovery_current"])


def test_legacy_regime_reproduces_uniform_survival_weighting():
    grid = _grid()
    surfaces, _ = evaluate_bellman_residuals(
        _QRegimeModel(phat_value=-1.0, bar_z_value=0.9),
        grid, _transition(grid), AnalysisEconomicConfig.from_current_config(),
        parent_default_regime_mode="legacy_soft_penalty",
    )
    np.testing.assert_allclose(surfaces["parent_survival_weight"], 1.0, atol=1e-6)
    np.testing.assert_allclose(surfaces["parent_default_weight"], 0.0, atol=1e-6)
    # legacy 下 training target 退化为纯 Bellman target。
    np.testing.assert_allclose(
        surfaces["Q_target_used_for_training_trainM"],
        surfaces["Q_target_trainM"],
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Phase E: Q-only polishing 支持
# ---------------------------------------------------------------------------

def test_q_polish_coverage_weights_rebalance_three_groups():
    phat = torch.tensor([[-1.0], [-0.005], [0.5], [0.6], [0.7]], dtype=torch.float64)
    out = build_q_polish_coverage_weights(
        phat, sim_share=0.5, boundary_share=0.25, default_share=0.25, eps_boundary=1e-2,
    )
    assert out["default_mask"].flatten().tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert out["boundary_mask"].flatten().tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert out["sim_mask"].flatten().tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]
    weights = out["coverage_weight"].flatten()
    # 稀疏的 default / boundary 组被抬高到与 sim 组同量级（share * N / n_g）。
    assert weights[0].item() == pytest.approx(0.25 * 5 / 1)
    assert weights[1].item() == pytest.approx(0.25 * 5 / 1)
    assert weights[2].item() == pytest.approx(0.5 * 5 / 3)
    assert weights.sum().item() == pytest.approx(5.0)


def test_q_polishing_freeze_keeps_only_q_encoder_and_head_trainable():
    model = PolicyValueModel(
        share_hidden_dims=[8], share_output_dim=8, q_head_dims=[4],
        p0_head_dims=[4], pi_head_dims=[4], bp0_head_dims=[4], bpi_head_dims=[4],
        barz_hidden_dims=[4], bari_hidden_dims=[4], i_grid_size=5,
        dropout=0.0, value_scale_mode="none",
    )
    sdf_fc1 = torch.nn.Linear(2, 2)
    stub = SimpleNamespace(
        models={"policy_value": model, "sdf_fc1": sdf_fc1},
        _q_polishing_active=False,
        _q_polishing_grad_backup={},
    )
    Episode._set_q_polishing_freeze(stub, True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable
    assert all(n.startswith(("q_encoder.", "q_head.")) for n in trainable)
    assert not any(p.requires_grad for p in sdf_fc1.parameters())

    Episode._set_q_polishing_freeze(stub, False)
    assert all(p.requires_grad for p in sdf_fc1.parameters())
    assert any(
        n.startswith("value_encoder.") and p.requires_grad
        for n, p in model.named_parameters()
    )


# ---------------------------------------------------------------------------
# Targeted review 2: regime weight 必须对 Phat 梯度隔离
# ---------------------------------------------------------------------------

def test_regime_weight_isolates_gradient_from_phat():
    """transition band 内 sigmoid(Phat/tau) 不得把梯度传回 Phat。"""
    phat = torch.tensor([[0.0]], dtype=torch.float64, requires_grad=True)
    weights = compute_parent_default_regime_weights(
        phat, mode="transition_band", eps=1e-2, tau=1e-2,
    )
    # detach 后权重不再是计算图的一部分，Phat 的梯度路径被彻底切断。
    assert not weights["parent_survival_weight"].requires_grad
    assert not weights["parent_default_weight"].requires_grad
    assert weights["parent_survival_weight"].grad_fn is None
    assert weights["parent_default_weight"].grad_fn is None
    assert phat.grad is None


def test_q_loss_regime_weight_does_not_backprop_to_phat():
    """backward Q loss 后，regime-weight 路径对 Phat 的梯度为 0。"""
    loss_fn = QLoss(
        parent_default_regime_mode="transition_band",
        parent_default_eps=1e-2, parent_default_tau=1e-2,
    )
    Q = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[0.3]], dtype=torch.float64)
    x = torch.tensor([[0.2]], dtype=torch.float64)
    z = torch.tensor([[0.0]], dtype=torch.float64)
    phat = torch.tensor([[0.0]], dtype=torch.float64, requires_grad=True)
    terms = loss_fn.compute_regime_aware_objective(
        residuals=loss_fn.compute_main_residual(
            Q, b, torch.zeros_like(b),
            M_list=[torch.tensor([[0.9]], dtype=torch.float64)],
            Qsp_children=[torch.tensor([[0.5]], dtype=torch.float64)],
            bar_z_children=[torch.tensor([[0.2]], dtype=torch.float64)],
            x_children=[x], z_children=[z],
        ),
        Q=Q, b=b, x=x, z=z, phat=phat,
    )
    (terms["bellman_loss"] + terms["recovery_loss"]).backward()
    assert phat.grad is None or bool((phat.grad == 0).all())
    assert Q.grad is not None


# ---------------------------------------------------------------------------
# Targeted review 3: w_survival 必须在 AiO 之后施加（不是 w^2）
# ---------------------------------------------------------------------------

def test_regime_aware_objective_weights_aio_after_not_before():
    """L_parent = mean(w_survival ⊙ AiO(r_1..r_N))，而非 w_survival² ⊙ AiO。"""
    loss_fn = QLoss(
        parent_default_regime_mode="transition_band",
        parent_default_eps=1e-2, parent_default_tau=1e-2,
    )
    Q = torch.tensor([[0.4], [0.4]], dtype=torch.float64)
    b = torch.tensor([[0.3], [0.3]], dtype=torch.float64)
    x = torch.tensor([[0.2], [0.2]], dtype=torch.float64)
    z = torch.tensor([[0.1], [0.1]], dtype=torch.float64)
    residuals = [
        torch.tensor([[0.1], [0.05]], dtype=torch.float64),
        torch.tensor([[-0.2], [0.03]], dtype=torch.float64),
    ]
    terms = loss_fn.compute_regime_aware_objective(
        residuals=residuals, Q=Q, b=b, x=x, z=z,
        phat=torch.tensor([[0.0], [0.002]], dtype=torch.float64),
    )
    aio = compute_aio_residual(residuals, loss_fn.aio_weight)
    w = terms["parent_survival_weight"]
    assert bool(((w > 0) & (w < 1)).all())
    torch.testing.assert_close(terms["bellman_loss"], (w * aio).mean())
    assert not torch.isclose(terms["bellman_loss"], (w.pow(2) * aio).mean())


# ---------------------------------------------------------------------------
# Targeted review 4: evaluator 显式口径命名与分解恒等式
# ---------------------------------------------------------------------------

def test_evaluator_bellman_decomposition_identity_explicit_names():
    grid = _grid()
    surfaces, _ = evaluate_bellman_residuals(
        _QRegimeModel(phat_value=0.5, bar_z_value=0.3),
        grid, _transition(grid), AnalysisEconomicConfig.from_current_config(),
        parent_default_regime_mode="hard",
    )
    for label in ("trainM", "rawM"):
        error = np.nanmax(np.abs(
            surfaces[f"Q_target_bellman_{label}"]
            - (
                surfaces[f"Q_target_survival_bellman_{label}"]
                + surfaces[f"Q_target_recovery_bellman_{label}"]
            )
        ))
        assert error < 1e-6
    canonical_error = np.nanmax(np.abs(
        surfaces["Q_target_bellman"]
        - (surfaces["Q_target_survival_bellman"] + surfaces["Q_target_recovery_bellman"])
    ))
    assert canonical_error < 1e-6
    # 旧 Q_target 经济含义未被覆盖：仍等于 Bellman target。
    np.testing.assert_allclose(surfaces["Q_target"], surfaces["Q_target_bellman"], rtol=0, atol=0)


def test_evaluator_rq_bellman_and_training_signed_identities():
    grid = _grid()
    model = _QRegimeModel(phat_value=-0.5, bar_z_value=0.9)
    surfaces, _ = evaluate_bellman_residuals(
        model, grid, _transition(grid), AnalysisEconomicConfig.from_current_config(),
        parent_default_regime_mode="hard",
    )
    q = evaluate_firm_surfaces(model, grid, _reference())["Q"]
    np.testing.assert_allclose(
        surfaces["RQ_bellman_signed"], surfaces["Q_target_bellman"] - q, atol=1e-5,
    )
    np.testing.assert_allclose(
        surfaces["RQ_training_signed"], surfaces["Q_target_training"] - q, atol=1e-5,
    )
    # hard regime + 全 default fixture：training target 退化为当期回收。
    np.testing.assert_allclose(
        surfaces["Q_target_training"], surfaces["recovery_current"], atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Targeted review 5: default-region q_unit = recovery / b 统计
# ---------------------------------------------------------------------------

def test_evaluator_default_region_q_unit_statistics():
    grid = _grid()
    surfaces, summary = evaluate_bellman_residuals(
        _QRegimeModel(phat_value=-1.0, bar_z_value=0.9),
        grid, _transition(grid), AnalysisEconomicConfig.from_current_config(),
        parent_default_regime_mode="hard",
    )
    b = np.asarray(grid.mesh_b)
    expected = surfaces["recovery_current"] / b
    np.testing.assert_allclose(
        surfaces["q_unit_recovery_target"], expected, rtol=1e-6, atol=1e-12,
    )
    sel = expected[np.isfinite(expected)]
    assert summary["q_unit_default_region_count"] == sel.size
    np.testing.assert_allclose(summary["q_unit_default_region_mean"], sel.mean(), rtol=1e-6)
    np.testing.assert_allclose(summary["q_unit_default_region_p90"], np.quantile(sel, 0.90), rtol=1e-6)
    np.testing.assert_allclose(summary["q_unit_default_region_p99"], np.quantile(sel, 0.99), rtol=1e-6)
    np.testing.assert_allclose(summary["q_unit_default_region_max"], sel.max(), rtol=1e-6)

