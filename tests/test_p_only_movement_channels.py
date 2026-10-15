from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.analyze_p_only_movement_channels import (
    child_equity_decomposition,
    projection_movement_decomposition,
)


def test_child_phat_and_hard_equity_decomposition_closes():
    p0_old = torch.tensor([[[-0.3, 0.2], [0.4, 0.1]], [[0.1, 0.5], [0.2, -0.2]]])
    pi_old = torch.tensor([[[-0.2, 0.1], [0.3, 0.2]], [[0.0, 0.4], [0.1, -0.1]]])
    p0_new = p0_old + torch.tensor(0.15)
    pi_new = pi_old + torch.tensor([[[0.4, -0.1], [-0.2, 0.3]], [[0.1, 0.2], [0.5, -0.4]]])
    out = child_equity_decomposition(p0_old, pi_old, p0_new, pi_new)
    raw = out["c_p0_raw"] + out["c_pi_raw"] + out["c_inv_raw"]
    hard = out["c_p0"] + out["c_pi"] + out["c_inv"] + out["c_default"]
    torch.testing.assert_close(out["phat_new"] - out["phat_old"], raw)
    torch.testing.assert_close(out["p_new"] - out["p_old"], hard)
    assert torch.all(out["c_inv_raw"] >= -1e-7)


def test_projection_movement_index_alignment_closes():
    p_prev = torch.tensor([[1.0], [2.0]])
    p_current = torch.tensor([[1.4], [1.7]])
    y_prev = torch.tensor([[0.8], [1.9]])
    y_current = torch.tensor([[1.1], [2.2]])
    out = projection_movement_decomposition(p_prev, p_current, y_prev, y_current)
    torch.testing.assert_close(out["delta_p"], out["delta_target_prev"] + out["delta_e"])
    torch.testing.assert_close(out["closure"], torch.zeros_like(out["closure"]))


def test_ties_deterministically_choose_p0_without_value_error():
    p0_old = torch.zeros(3, 2, 4)
    pi_old = torch.zeros_like(p0_old)
    p0_new = torch.ones_like(p0_old)
    pi_new = torch.ones_like(p0_old)
    out = child_equity_decomposition(p0_old, pi_old, p0_new, pi_new)
    assert not bool(out["old_i"].any())
    assert not bool(out["new_i"].any())
    torch.testing.assert_close(out["p_new"] - out["p_old"], torch.ones_like(out["p_old"]))
