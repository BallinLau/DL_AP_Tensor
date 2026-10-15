from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.analyze_pi_continuation_gain import (
    FunctionOutputInterpolation,
    local_gain_components,
    realized_gain,
)


def test_local_gain_closure_and_structural_factor():
    p0 = torch.tensor([
        [[1.0, 2.0], [-1.0, 1.0]],
        [[1.0, 0.0], [-1.0, 1.0]],
    ])
    pi = torch.tensor([
        [[2.0, 1.0], [-2.0, 2.0]],
        [[0.0, 2.0], [-2.0, 2.0]],
    ])
    m = torch.tensor([[[1.2], [0.8]], [[1.0], [1.0]]])
    weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]])
    g = 1.14
    out = local_gain_components(p0, pi, m, weights, g)
    torch.testing.assert_close(out["a_total"], out["a_from_p0"] + out["a_from_pi"])
    torch.testing.assert_close(out["a_from_pi"], g * out["a_pi_g1"])
    assert float(out["alpha_child"].min()) >= 0.0
    assert float(out["alpha_child"].max()) <= 1.0
    # Parent 1, child 0 has a negative envelope and is hard-defaulted.
    assert out["survival_child"][1, 0].item() == 0.0


def test_m1_counterfactual_and_conditional_m_are_exact():
    p0 = torch.zeros(2, 1, 2)
    pi = torch.ones_like(p0)
    m = torch.tensor([[[1.2], [0.8]]])
    weights = torch.tensor([[0.25, 0.75]])
    out = local_gain_components(p0, pi, m, weights, 1.14)
    torch.testing.assert_close(out["a_pi_m1"], torch.tensor([1.14]))
    torch.testing.assert_close(out["m_mean"], torch.tensor([0.9]))
    torch.testing.assert_close(out["m_survival_pi"], torch.tensor([0.9]))


def test_realized_gain_reports_direction_not_only_norm():
    channel = torch.tensor([2.0, 0.0])
    movement = torch.tensor([1.0, 1.0])
    scale = torch.ones(2)
    out = realized_gain(channel, movement, scale)
    assert abs(out["gain_l2_normalized"] - 2 ** 0.5) < 1e-12
    assert abs(out["lambda_dir_normalized"] - 1.0) < 1e-12
    assert abs(out["cosine_normalized"] - 2 ** -0.5) < 1e-12


def test_function_interpolation_is_on_outputs_and_preserves_hard_equity():
    class Derived:
        @staticmethod
        def survival(phat):
            return torch.sigmoid(phat)

    class Toy:
        i_threshold = 1.0
        i_grid_size = 2
        derived = Derived()

        def __init__(self, shift):
            self.shift = shift

        def _value_outputs(self, state):
            p0 = state[:, 3:4] - 0.75 + self.shift
            pi = 0.25 - state[:, 3:4] + self.shift
            return p0, pi

    state = torch.zeros(1, 7)
    out = FunctionOutputInterpolation(Toy(0.0), Toy(2.0), 0.25).forward_equity(state)
    # Old envelope mean is 0.25; output interpolation adds 0.5 everywhere.
    torch.testing.assert_close(out["Phat"], torch.tensor([[0.75]]))
    torch.testing.assert_close(out["P"], torch.tensor([[0.75]]))
