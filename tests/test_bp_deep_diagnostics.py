from __future__ import annotations

import torch

from config import Config
from experiments.export_bp_deep_diagnostics import stable_logit_with_censoring
from experiments.export_target_grid_decomposition import make_summary_rows
from losses.utils import compute_cashflow
from models.policy_value import PolicyValueModel
from training.bp_policy_loss import (
    compute_target_grid_policy_distillation_loss,
    huber_element,
)


def test_stable_logit_does_not_clip_small_positive_to_1e_minus_12():
    bp = torch.tensor([[1e-14]], dtype=torch.float64)
    logit, censored_low, censored_high = stable_logit_with_censoring(bp)
    expected = torch.log(bp) - torch.log1p(-bp)
    torch.testing.assert_close(logit, expected)
    assert not bool(censored_low.item())
    assert not bool(censored_high.item())
    assert logit.item() < -30.0


def test_stable_logit_censors_exact_zero_and_one():
    bp = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    logit, censored_low, censored_high = stable_logit_with_censoring(bp)
    assert torch.isfinite(logit).all()
    assert bool(censored_low[0].item())
    assert not bool(censored_low[1].item())
    assert not bool(censored_high[0].item())
    assert bool(censored_high[1].item())


def test_policy_model_eval_mode_still_supports_backward():
    torch.manual_seed(123)
    model = PolicyValueModel()
    model.eval()
    state = torch.rand(4, 7)
    with torch.enable_grad():
        out = model(state)
        loss = out.bp0.mean() + out.bpI.mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = sum(
        float(p.grad.detach().abs().sum().item())
        for p in model.policy_encoder.parameters()
        if p.grad is not None
    )
    assert grad_norm > 0.0


def test_training_equivalent_policy_loss_matches_previous_formula():
    pred = torch.tensor([[0.1], [0.4], [0.9]], dtype=torch.float64)
    target = torch.tensor([[0.2], [0.7], [0.1]], dtype=torch.float64)
    confidence = torch.tensor([[1.0], [0.5], [0.0]], dtype=torch.float64)
    sample_weight = torch.tensor([[1.0], [0.2], [0.5]], dtype=torch.float64)
    delta = 0.05
    branch_weight = 1.7

    total, unweighted, elem = compute_target_grid_policy_distillation_loss(
        pred,
        target,
        confidence,
        huber_delta=delta,
        branch_weight=branch_weight,
        sample_weight=sample_weight,
    )

    manual_elem = huber_element(pred, target, delta)
    manual_unweighted = (confidence * sample_weight.clamp(0.0, 1.0) * manual_elem).mean()
    torch.testing.assert_close(elem, manual_elem)
    torch.testing.assert_close(unweighted, manual_unweighted)
    torch.testing.assert_close(total, branch_weight * manual_unweighted)


def test_decomposition_summary_margin_and_confidence_fields_are_finite():
    result = {
        "coarse_bp_grid": torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float64),
        "coarse_value_grid": torch.tensor([[1.0, 2.0, 1.5]], dtype=torch.float64),
        "coarse_cashflow_grid_mean": torch.tensor([[0.2, 0.7, 0.4]], dtype=torch.float64),
        "coarse_continuation_grid_mean": torch.tensor([[0.8, 1.3, 1.1]], dtype=torch.float64),
        "coarse_q_issue_grid": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64),
        "coarse_p_child_grid_mean": torch.tensor([[1.1, 1.2, 1.3]], dtype=torch.float64),
        "coarse_default_grid_mean": torch.tensor([[0.0, 0.1, 0.2]], dtype=torch.float64),
        "bp_star": torch.tensor([[0.5]], dtype=torch.float64),
        "value_star": torch.tensor([[2.0]], dtype=torch.float64),
        "regret": torch.tensor([[0.1]], dtype=torch.float64),
        "q_issue_at_star": torch.tensor([[0.2]], dtype=torch.float64),
        "p_child_at_star": torch.tensor([[1.2]], dtype=torch.float64),
        "default_at_star": torch.tensor([[0.1]], dtype=torch.float64),
        "confidence": torch.tensor([[0.8]], dtype=torch.float64),
        "refi_active": torch.tensor([[1.0]], dtype=torch.float64),
    }
    rows = make_summary_rows(
        result=result,
        parent_state=torch.tensor([[0.3, 0.2, 1.0, 0.1, 0.4, 0.5, 0.6]], dtype=torch.float64),
        source_index=torch.tensor([42]),
        branch="p0",
        bp_pred=torch.tensor([[0.4]], dtype=torch.float64),
        mix_survival_weight=torch.tensor([[1.0]], dtype=torch.float64),
    )
    row = rows[0]
    assert row["value_best"] == 2.0
    assert row["value_second_best"] == 1.5
    assert row["value_margin"] >= -1e-7
    assert 0.0 <= row["confidence_weight"] <= 1.0
    assert row["mix_survival_weight"] == 1.0
    assert torch.isfinite(torch.tensor(row["relative_value_margin"]))


def test_mix_training_equivalent_loss_uses_survival_sample_weight():
    pred = torch.tensor([[0.1], [0.9]], dtype=torch.float64)
    target = torch.tensor([[0.3], [0.2]], dtype=torch.float64)
    confidence = torch.ones_like(pred)
    survival = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    loss, _, elem = compute_target_grid_policy_distillation_loss(
        pred,
        target,
        confidence,
        huber_delta=0.05,
        branch_weight=2.0,
        sample_weight=survival,
    )
    expected = 2.0 * (confidence * survival * elem).mean()
    torch.testing.assert_close(loss, expected)


def test_cashflow_identity_formula_is_small():
    x = torch.tensor([[0.1], [0.2]], dtype=torch.float64)
    z = torch.tensor([[0.3], [0.4]], dtype=torch.float64)
    b = torch.tensor([[0.2], [0.5]], dtype=torch.float64)
    q_current = torch.tensor([[0.1], [0.2]], dtype=torch.float64)
    q_issue = torch.tensor([[0.3], [0.4]], dtype=torch.float64)
    eta = torch.tensor([[1.0], [0.5]], dtype=torch.float64)

    production = compute_cashflow(x, z, b, Config.DELTA, Config.TAU)
    debt_adjustment = ((1.0 - Config.KAPPA_B) * q_issue - q_current) * eta
    raw = production + debt_adjustment
    equity_cost = Config.KAPPA_E * torch.relu(-raw)
    reconstructed = production + debt_adjustment - equity_cost
    exported = raw - equity_cost
    assert float((exported - reconstructed).abs().max().item()) < 1e-5
