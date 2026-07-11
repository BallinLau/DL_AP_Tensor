import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from losses.sdf_loss import SDFLoss  # noqa: E402
from models.sdf_fc1 import compute_sdf  # noqa: E402
from utils.metrics import conditional_moment_metrics  # noqa: E402


class SDFWealthLossModeTest(unittest.TestCase):
    def test_legacy_abs_log1p_matches_existing_objective(self):
        residuals = torch.tensor(
            [
                [2.0, -3.0],
                [0.5, 4.0],
            ],
            dtype=torch.float32,
        )
        loss_fn = SDFLoss(wealth_loss_mode="legacy_abs_log1p")

        loss, details = loss_fn.compute_wealth_main_loss(residuals)

        expected = torch.log1p(residuals.prod(dim=1).abs()).mean()
        self.assertTrue(torch.allclose(loss, expected))
        self.assertTrue(torch.allclose(details["legacy_abs_log1p"], expected))
        self.assertIn("signed_aio", details)

    def test_signed_aio_is_signed_product_mean_without_abs_or_clamp(self):
        residuals = torch.tensor(
            [
                [2.0, -3.0],
                [0.5, -4.0],
            ],
            dtype=torch.float32,
        )
        loss_fn = SDFLoss(wealth_loss_mode="signed_aio")

        loss, details = loss_fn.compute_wealth_main_loss(residuals)

        expected = (residuals[:, 0] * residuals[:, 1]).mean()
        self.assertLess(float(expected.item()), 0.0)
        self.assertTrue(torch.allclose(loss, expected))
        self.assertTrue(torch.allclose(details["signed_aio"], expected))
        self.assertTrue(torch.allclose(details["product_signed_mean"], expected))
        self.assertGreater(float(details["legacy_abs_log1p"].item()), 0.0)

    def test_signed_aio_requires_two_branches(self):
        residuals = torch.ones(3, 3)
        loss_fn = SDFLoss(wealth_loss_mode="signed_aio")

        with self.assertRaisesRegex(ValueError, "requires exactly two"):
            loss_fn.compute_wealth_main_loss(residuals)

    def test_conditional_moment_metrics_use_parent_mean_residual(self):
        residuals = torch.tensor(
            [
                [1.0, -1.0, 1.0, -1.0],
                [2.0, 2.0, 2.0, 2.0],
            ],
            dtype=torch.float32,
        )

        metrics = conditional_moment_metrics(residuals)

        mean_r = residuals.to(torch.float64).mean(dim=1)
        expected_cm = mean_r.pow(2).mean()
        self.assertAlmostEqual(metrics["heldout_cm_mse"], float(expected_cm.item()), places=12)
        self.assertEqual(metrics["heldout_n_children"], 4.0)

    def test_wealth_residual_consumption_growth_uses_minus_kappa_over_sigma(self):
        loss_fn = SDFLoss(gamma=2.0, kappa=-6.0, sigma=2.0, beta=0.9)
        w_parent = torch.tensor([5.0])
        w_children = torch.tensor([[4.0, 4.5]])
        k_parent = torch.tensor([1.0])
        k_children = torch.tensor([[1.1, 0.9]])
        c_parent = torch.tensor([0.2])
        c_children = torch.tensor([[0.3, 0.1]])

        residual_tensor = loss_fn.compute_euler_residuals(
            w_parent, w_children, k_parent, k_children, c_parent, c_children
        )
        residual_list = loss_fn.compute_euler_residuals(
            w_parent,
            [w_children[:, 0], w_children[:, 1]],
            k_parent,
            [k_children[:, 0], k_children[:, 1]],
            c_parent,
            [c_children[:, 0], c_children[:, 1]],
        )

        expected_exp = torch.exp(
            (k_children - k_parent.unsqueeze(-1)) * (1.0 - loss_fn.gamma)
            - loss_fn.kappa / loss_fn.sigma * (c_children - c_parent.unsqueeze(-1))
        ) * loss_fn.tmp
        expected = (
            expected_exp * torch.pow(w_children, loss_fn.kappa)
            - torch.pow((w_parent - torch.exp(c_parent)).clamp_min(1e-8), loss_fn.kappa).unsqueeze(-1)
        )

        self.assertTrue(torch.allclose(residual_tensor, expected))
        self.assertTrue(torch.allclose(torch.stack(residual_list, dim=1), expected))

    def test_compute_sdf_consumption_growth_uses_minus_kappa_over_sigma(self):
        beta = 0.9
        gamma = 2.0
        kappa = -6.0
        sigma = 2.0
        w_parent = torch.tensor([5.0])
        w_children = torch.tensor([[4.0, 4.5]])
        k_parent = torch.tensor([1.0])
        k_children = torch.tensor([[1.1, 0.9]])
        c_parent = torch.tensor([0.2])
        c_children = torch.tensor([[0.3, 0.1]])

        m_tensor = compute_sdf(
            w_parent,
            w_children,
            k_parent,
            k_children,
            c_parent,
            c_children,
            beta=beta,
            gamma=gamma,
            kappa=kappa,
            sigma=sigma,
            exponent_clip=None,
        )
        m_list = compute_sdf(
            w_parent,
            [w_children[:, 0], w_children[:, 1]],
            k_parent,
            [k_children[:, 0], k_children[:, 1]],
            c_parent,
            [c_children[:, 0], c_children[:, 1]],
            beta=beta,
            gamma=gamma,
            kappa=kappa,
            sigma=sigma,
            exponent_clip=None,
        )

        denom = (w_parent - torch.exp(c_parent)).clamp_min(1e-3).unsqueeze(-1)
        ratio = (w_children / denom).clamp_min(1e-3)
        expected = (
            torch.exp(
                (k_children - k_parent.unsqueeze(-1)) * (-gamma)
                - kappa / sigma * (c_children - c_parent.unsqueeze(-1))
            )
            * torch.pow(ratio, kappa - 1)
            * (beta ** kappa)
        )

        self.assertTrue(torch.allclose(m_tensor, expected))
        self.assertTrue(torch.allclose(torch.stack(m_list, dim=1), expected))


if __name__ == "__main__":
    unittest.main()
