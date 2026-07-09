import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from training.sdf_shock_bank import (
    SDFShockBank,
    _make_generator,
    shock_pair_diagnostics,
    shocks_to_x_children,
)


class SDFShockBankTest(unittest.TestCase):
    def test_sample_pair_never_collides_and_uses_parent_rows(self):
        bank = SDFShockBank.create(
            n_parents=8,
            bank_size=4,
            device=torch.device("cpu"),
            base_seed=11,
        )
        parent_index = torch.tensor([0, 3, 7, 1, 3], dtype=torch.long)
        gen = _make_generator(torch.device("cpu"))
        gen.manual_seed(99)

        eps1, eps2, j1, j2 = bank.sample_pair(parent_index, gen)

        self.assertTrue(torch.all(j1 != j2))
        self.assertTrue(torch.allclose(eps1, bank.eps[parent_index, j1]))
        self.assertTrue(torch.allclose(eps2, bank.eps[parent_index, j2]))

    def test_large_cross_parent_pairs_are_nearly_uncorrelated(self):
        n_parents = 100_000
        bank = SDFShockBank.create(
            n_parents=n_parents,
            bank_size=16,
            device=torch.device("cpu"),
            base_seed=123,
        )
        parent_index = torch.arange(n_parents, dtype=torch.long)
        gen = _make_generator(torch.device("cpu"))
        gen.manual_seed(456)

        eps1, eps2, j1, j2 = bank.sample_pair(parent_index, gen)
        diag = shock_pair_diagnostics(eps1, eps2, j1, j2, bank.bank_size)

        self.assertEqual(diag["sdf_pair_collision_rate"], 0.0)
        self.assertLess(abs(diag["sdf_eps_cross_corr"]), 0.02)

    def test_refresh_changes_bank_and_is_reproducible_for_same_seed(self):
        bank_a = SDFShockBank.create(
            n_parents=32,
            bank_size=8,
            device=torch.device("cpu"),
            base_seed=7,
        )
        original = bank_a.eps.clone()
        bank_a.refresh_(seed=8)

        bank_b = SDFShockBank.create(
            n_parents=32,
            bank_size=8,
            device=torch.device("cpu"),
            base_seed=7,
        )
        bank_b.refresh_(seed=8)

        self.assertFalse(torch.allclose(original, bank_a.eps))
        self.assertTrue(torch.allclose(bank_a.eps, bank_b.eps))

    def test_shocks_to_x_children_uses_parent_conditional_mean(self):
        x_parent = torch.tensor([[-2.0], [-1.0]], dtype=torch.float32)
        eps1 = torch.zeros_like(x_parent)
        eps2 = torch.ones_like(x_parent)

        x_children = shocks_to_x_children(
            x_parent=x_parent,
            eps1=eps1,
            eps2=eps2,
            rho_x=0.95,
            sigma_x=0.012,
            xbar=-2.0,
        )

        conditional_mean = 0.05 * -2.0 + 0.95 * x_parent
        self.assertEqual(tuple(x_children.shape), (2, 2, 1))
        self.assertTrue(torch.allclose(x_children[:, 0], conditional_mean))
        self.assertTrue(torch.allclose(x_children[:, 1], conditional_mean + 0.012))


if __name__ == "__main__":
    unittest.main()
