from types import SimpleNamespace

import pytest
import torch

from losses.q_loss import QLoss
from losses.utils import (
    compute_aio_residual,
    compute_pairwise_aio_components,
    compute_pairwise_aio_residual,
)
from training.episode import Episode


@pytest.mark.parametrize("weight", [0.0, 0.2, 0.5, 1.0])
def test_pairwise_aio_j2_matches_legacy_two_child(weight):
    generator = torch.Generator().manual_seed(20261014)
    for _ in range(8):
        residuals = [torch.randn(13, 1, generator=generator) for _ in range(2)]
        old = compute_aio_residual(residuals, weight)
        new = compute_pairwise_aio_residual(residuals, weight)
        assert torch.allclose(new, old, atol=1e-7, rtol=1e-6)


def test_pairwise_aio_j3_matches_explicit_formula():
    residuals = [
        torch.tensor([[1.0], [-2.0]]),
        torch.tensor([[3.0], [4.0]]),
        torch.tensor([[-5.0], [6.0]]),
    ]
    components = compute_pairwise_aio_components(residuals, 0.5)
    explicit = (
        residuals[0] * residuals[1]
        + residuals[0] * residuals[2]
        + residuals[1] * residuals[2]
    ) / 3.0
    assert torch.allclose(components["u_signed"], explicit)


def test_pairwise_aio_j16_matches_explicit_unordered_pair_mean():
    generator = torch.Generator().manual_seed(16)
    residuals = [torch.randn(5, 1, generator=generator) for _ in range(16)]
    fast = compute_pairwise_aio_components(residuals, 0.5)["u_signed"]
    naive = torch.stack(
        [residuals[j] * residuals[k] for j in range(16) for k in range(j + 1, 16)]
    ).mean(dim=0)
    assert torch.allclose(fast, naive, atol=2e-7, rtol=2e-6)


def test_pairwise_aio_excludes_diagonal_terms():
    residuals = [
        torch.tensor([[2.0]]),
        torch.tensor([[3.0]]),
        torch.tensor([[5.0]]),
    ]
    result = compute_pairwise_aio_components(residuals, 1.0)["u_signed"]
    pair_only = torch.tensor([[(2.0 * 3.0 + 2.0 * 5.0 + 3.0 * 5.0) / 3.0]])
    with_diagonal = torch.tensor([[((2.0 + 3.0 + 5.0) ** 2) / 9.0]])
    assert torch.equal(result, pair_only)
    assert not torch.equal(result, with_diagonal)


def test_pairwise_aio_j16_has_finite_gradients():
    generator = torch.Generator().manual_seed(17)
    residuals = [
        torch.randn(7, 1, generator=generator, requires_grad=True) for _ in range(16)
    ]
    loss = compute_pairwise_aio_residual(residuals, 0.5).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in residuals)


@pytest.mark.parametrize("value, expected, legacy_product", [(0.1, 0.01, 1e-16), (2.0, 4.0, 65536.0)])
def test_pairwise_aio_scaling_is_quadratic_not_jth_power(value, expected, legacy_product):
    residuals = [torch.full((3, 1), value, dtype=torch.float64) for _ in range(16)]
    pairwise = compute_pairwise_aio_components(residuals, 1.0)["u_abs"]
    product = torch.stack(residuals).prod(dim=0).abs()
    assert torch.allclose(pairwise, torch.full_like(pairwise, expected))
    assert torch.allclose(product, torch.full_like(product, legacy_product))


def test_exact_eta_is_collapsed_before_pairwise_aio():
    episode = object.__new__(Episode)
    episode.hyperparams = SimpleNamespace(
        pv_exact_eta_integration_enabled=True,
    )
    episode.config = SimpleNamespace(ZETA=0.25)
    r10 = torch.tensor([[1.0]])
    r11 = torch.tensor([[5.0]])
    r20 = torch.tensor([[-2.0]])
    r21 = torch.tensor([[6.0]])
    collapsed = episode._collapse_policy_eta_pairs([r10, r11, r20, r21])
    assert len(collapsed) == 2
    expected_r1 = 0.75 * r10 + 0.25 * r11
    expected_r2 = 0.75 * r20 + 0.25 * r21
    pair = compute_pairwise_aio_components(collapsed, 1.0)["u_signed"]
    assert torch.equal(collapsed[0], expected_r1)
    assert torch.equal(collapsed[1], expected_r2)
    assert torch.equal(pair, expected_r1 * expected_r2)


def test_q_survival_objective_reports_pairwise_components_and_q_main_identity():
    loss_fn = QLoss(aio_weight=0.35)
    residuals = [torch.tensor([[j - 2.0], [0.1 * j]]) for j in range(4)]
    q = torch.ones(2, 1)
    result = loss_fn.compute_survival_parent_objective(
        residuals=residuals, Q=q, nonnegative_weight=1.0
    )
    assert torch.equal(result["aio_residual"], result["per_sample"])
    assert torch.equal(result["bellman_loss"], result["aio_residual"].mean())
    assert torch.equal(result["aio_u_abs"], result["aio_u_signed"].abs())
