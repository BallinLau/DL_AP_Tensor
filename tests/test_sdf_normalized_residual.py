from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from losses.sdf_loss import SDFLoss  # noqa: E402


def _loss_fn() -> SDFLoss:
    return SDFLoss(gamma=2.0, kappa=-6.0, sigma=2.0, beta=0.95)


def _base_inputs():
    w_parent = torch.tensor([5.0, 6.0], dtype=torch.float32)
    w_children = torch.tensor([[4.0, 4.5], [5.0, 5.5]], dtype=torch.float32)
    k_parent = torch.tensor([1.0, 1.2], dtype=torch.float32)
    k_children = torch.tensor([[1.1, 0.9], [1.3, 1.1]], dtype=torch.float32)
    c_parent = torch.tensor([0.2, 0.3], dtype=torch.float32)
    c_children = torch.tensor([[0.3, 0.1], [0.4, 0.2]], dtype=torch.float32)
    return w_parent, w_children, k_parent, k_children, c_parent, c_children


def test_raw_and_normalized_residuals_match_algebra_without_clipping():
    loss_fn = _loss_fn()
    pack = loss_fn.compute_wealth_residuals(*_base_inputs(), normalized_logr_clip=100.0)

    expected_raw = pack["surplus_parent"].pow(loss_fn.kappa).unsqueeze(-1) * pack["normalized"]

    assert torch.allclose(pack["raw"], expected_raw, rtol=1e-5, atol=1e-6)
    assert float(pack["log_R_clip_share"].item()) == 0.0


def test_raw_and_normalized_residuals_have_same_zero():
    loss_fn = _loss_fn()
    w_parent, _, k_parent, k_children, c_parent, c_children = _base_inputs()
    surplus = (w_parent - torch.exp(c_parent)).clamp_min(1e-8)
    log_a = (
        loss_fn.kappa * torch.log(torch.tensor(loss_fn.beta))
        + (1.0 - loss_fn.gamma) * (k_children - k_parent.unsqueeze(-1))
        + loss_fn.kappa / loss_fn.sigma * (c_children - c_parent.unsqueeze(-1))
    )
    w_children = surplus.unsqueeze(-1) * torch.exp(-log_a / loss_fn.kappa)

    pack = loss_fn.compute_wealth_residuals(
        w_parent, w_children, k_parent, k_children, c_parent, c_children, normalized_logr_clip=100.0
    )

    assert pack["normalized"].abs().max() < 1e-5
    assert pack["raw"].abs().max() < 1e-5


def test_normalized_residual_is_scale_invariant_and_raw_scales_by_kappa():
    loss_fn = _loss_fn()
    w_parent, w_children, k_parent, k_children, c_parent, c_children = _base_inputs()
    original = loss_fn.compute_wealth_residuals(
        w_parent, w_children, k_parent, k_children, c_parent, c_children, normalized_logr_clip=100.0
    )

    scale = 3.0
    surplus = (w_parent - torch.exp(c_parent)).clamp_min(1e-8)
    w_parent_scaled = scale * surplus + torch.exp(c_parent)
    w_children_scaled = scale * w_children
    scaled = loss_fn.compute_wealth_residuals(
        w_parent_scaled,
        w_children_scaled,
        k_parent,
        k_children,
        c_parent,
        c_children,
        normalized_logr_clip=100.0,
    )

    assert torch.allclose(original["normalized"], scaled["normalized"], rtol=1e-5, atol=1e-6)
    assert torch.allclose(scaled["raw"], original["raw"] * (scale ** loss_fn.kappa), rtol=1e-5, atol=1e-6)


def test_normalized_residual_gradients_are_finite():
    loss_fn = _loss_fn()
    w_parent, w_children, k_parent, k_children, c_parent, c_children = _base_inputs()
    w_parent = w_parent.clone().requires_grad_(True)
    w_children = w_children.clone().requires_grad_(True)

    pack = loss_fn.compute_wealth_residuals(
        w_parent, w_children, k_parent, k_children, c_parent, c_children, normalized_logr_clip=100.0
    )
    loss = (pack["normalized"][:, 0] * pack["normalized"][:, 1]).mean()
    loss.backward()

    assert torch.isfinite(w_parent.grad).all()
    assert torch.isfinite(w_children.grad).all()


def test_normalized_residual_clipping_reports_share_and_keeps_finite_values():
    loss_fn = _loss_fn()
    w_parent, w_children, k_parent, k_children, c_parent, c_children = _base_inputs()
    w_children = torch.full_like(w_children, 1e-8)

    pack = loss_fn.compute_wealth_residuals(
        w_parent, w_children, k_parent, k_children, c_parent, c_children, normalized_logr_clip=1.0
    )

    assert float(pack["log_R_clip_share"].item()) > 0.0
    assert torch.isfinite(pack["normalized"]).all()


if __name__ == "__main__":
    test_raw_and_normalized_residuals_match_algebra_without_clipping()
    test_raw_and_normalized_residuals_have_same_zero()
    test_normalized_residual_is_scale_invariant_and_raw_scales_by_kappa()
    test_normalized_residual_gradients_are_finite()
    test_normalized_residual_clipping_reports_share_and_keeps_finite_values()
