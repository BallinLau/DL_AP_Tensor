import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from losses.sdf_loss import SDFLoss  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
