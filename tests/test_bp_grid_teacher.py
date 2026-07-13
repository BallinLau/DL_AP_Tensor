from types import SimpleNamespace
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from losses import P0Loss, PILoss
from training.bp_grid_teacher import BPGridTeacher


class ParabolicTarget:
    def _q_output(self, firm_state):
        return torch.zeros(firm_state.shape[0], 1, device=firm_state.device, dtype=firm_state.dtype)

    def forward_equity(self, firm_state):
        b = firm_state[:, 0:1]
        p = 1.0 - (b - 0.32).pow(2)
        p = p.clamp_min(0.0)
        return {
            "P": p,
            "Phat": p,
            "bar_z": (p <= 0.0).to(p.dtype),
        }

    def __call__(self, firm_state):
        equity = self.forward_equity(firm_state)
        return SimpleNamespace(
            Q=self._q_output(firm_state),
            P=equity["P"],
            Phat=equity["Phat"],
            bar_z=equity["bar_z"],
        )


class RecordingTarget(ParabolicTarget):
    def __init__(self):
        self.q_b = []
        self.equity_b = []

    def _q_output(self, firm_state):
        self.q_b.append(firm_state[:, 0:1].detach().clone())
        return firm_state[:, 0:1].clamp_min(0.0)

    def forward_equity(self, firm_state):
        self.equity_b.append(firm_state[:, 0:1].detach().clone())
        return super().forward_equity(firm_state)


class HighLeverageTarget(ParabolicTarget):
    def forward_equity(self, firm_state):
        b = firm_state[:, 0:1].clamp(0.0, 1.0)
        return {
            "P": b,
            "Phat": b,
            "bar_z": torch.zeros_like(b),
        }


class LargeScaleLinearTarget(ParabolicTarget):
    def forward_equity(self, firm_state):
        b = firm_state[:, 0:1].clamp(0.0, 1.0)
        p = 100.0 - 0.2 * b
        return {
            "P": p,
            "Phat": p,
            "bar_z": torch.zeros_like(b),
        }


def test_bp_grid_teacher_selects_value_maximizing_bp():
    target = ParabolicTarget()
    teacher = BPGridTeacher(
        target,
        P0Loss(),
        PILoss(),
        coarse_size=11,
        refine=False,
        margin_scale=1e-4,
    )
    parent = torch.tensor(
        [
            [0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    child = parent.clone()
    child[:, 2:3] = 1.0
    m_list = [torch.ones(parent.shape[0], 1)]

    out = teacher.compute(parent, [child], m_list, branch="p0", bp_pred=torch.full((2, 1), 0.9))

    assert out["bp_star"].shape == (2, 1)
    assert torch.allclose(out["bp_star"], torch.full((2, 1), 0.3), atol=1e-6)
    assert torch.all(out["regret"] > 0)
    assert torch.all(out["confidence"] > 0)
    assert out["boundary_high"].sum().item() == 0
    assert not out["bp_star"].requires_grad


def test_grid_teacher_debt_state_semantics():
    target = RecordingTarget()
    teacher = BPGridTeacher(
        target,
        P0Loss(),
        PILoss(),
        coarse_size=3,
        refine=False,
        candidate_chunk_size=3,
        margin_scale=1e-4,
    )
    parent = torch.tensor([[0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    child = parent.clone()
    child[:, 2:3] = 0.25

    out = teacher.compute(parent, [child], [torch.ones(1, 1)], branch="p0")

    assert out["coarse_bp_grid"].shape == (1, 3)
    issue_b = [x.reshape(-1) for x in target.q_b if x.numel() == 3][0]
    expected_issue_b = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(issue_b, expected_issue_b, atol=1e-6)

    child_b = target.equity_b[0].reshape(-1)
    assert torch.allclose(child_b, expected_issue_b, atol=1e-6)

    inactive_target = RecordingTarget()
    inactive_teacher = BPGridTeacher(
        inactive_target,
        P0Loss(),
        PILoss(),
        coarse_size=3,
        refine=False,
        candidate_chunk_size=3,
        margin_scale=1e-4,
    )
    inactive_parent = parent.clone()
    inactive_parent[:, 2:3] = 0.0
    inactive_child = child.clone()
    inactive_child[:, 2:3] = 1.0
    inactive_teacher.compute(inactive_parent, [inactive_child], [torch.ones(1, 1)], branch="p0")
    inactive_issue_b = [x.reshape(-1) for x in inactive_target.q_b if x.numel() == 3][0]
    inactive_child_b = inactive_target.equity_b[0].reshape(-1)
    expected_inactive_b = torch.full((3,), 0.2)
    assert torch.allclose(inactive_issue_b, expected_inactive_b, atol=1e-6)
    assert torch.allclose(inactive_child_b, expected_inactive_b, atol=1e-6)


def test_grid_teacher_mix_branch_and_coarse_confidence():
    target = ParabolicTarget()
    teacher = BPGridTeacher(
        target,
        P0Loss(),
        PILoss(),
        coarse_size=5,
        fine_size=5,
        refine=True,
        candidate_chunk_size=2,
        margin_scale=1e-4,
    )
    parent = torch.tensor(
        [
            [0.1, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0],
            [0.2, 0.0, 1.0, 0.3, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    child0 = parent.clone()
    child1 = parent.clone()
    child0[:, 2:3] = 1.0
    child1[:, 2:3] = 0.0
    mix_weight = torch.tensor([[0.25], [0.75]], dtype=torch.float32)

    out = teacher.compute(
        parent,
        [child0, child1],
        [torch.ones(2, 1), torch.ones(2, 1)],
        branch="mix",
        bp_pred=torch.full((2, 1), 0.9),
        mix_weight=mix_weight,
    )

    assert out["bp_star"].shape == (2, 1)
    assert out["coarse_value_grid"].shape == (2, 5)
    assert out["value_grid"].shape == (2, 5)
    torch.testing.assert_close(
        out["value_grid"],
        out["cashflow_grid_mean"] + out["continuation_grid_mean"],
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        out["coarse_value_grid"],
        out["coarse_cashflow_grid_mean"] + out["coarse_continuation_grid_mean"],
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.all(out["coarse_top2_margin"] >= out["fine_top2_margin"] - 1e-6)
    assert torch.all(out["confidence"] > 0)
    assert torch.all(out["regret"] >= 0)


def test_pi_grid_target_multi_child_boundary_and_refine():
    target = HighLeverageTarget()
    teacher = BPGridTeacher(
        target,
        P0Loss(),
        PILoss(),
        coarse_size=5,
        fine_size=5,
        refine=True,
        candidate_chunk_size=2,
        margin_scale=1e-4,
    )
    parent = torch.tensor(
        [
            [0.1, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0],
            [0.3, 0.0, 1.0, 0.4, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    child_eta1 = parent.clone()
    child_eta0 = parent.clone()
    child_eta1[:, 2:3] = 1.0
    child_eta0[:, 2:3] = 0.0

    out = teacher.compute(
        parent,
        [child_eta1, child_eta0],
        [torch.ones(2, 1), torch.ones(2, 1)],
        branch="pi",
        bp_pred=torch.zeros(2, 1),
    )

    assert torch.allclose(out["bp_star"], torch.ones(2, 1), atol=1e-6)
    assert torch.all(out["boundary_high"] == 1.0)
    assert torch.all(out["boundary_low"] == 0.0)
    assert out["coarse_value_grid"].shape == (2, 5)
    assert out["value_grid"].shape == (2, 5)
    assert torch.all(out["regret"] > 0)


def test_relative_confidence_scales_margin_by_config_threshold():
    target = LargeScaleLinearTarget()
    teacher = BPGridTeacher(
        target,
        P0Loss(),
        PILoss(),
        coarse_size=3,
        refine=False,
        margin_scale=1e-3,
        confidence_relative=True,
    )
    parent = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    child = parent.clone()
    child[:, 2:3] = 1.0

    out = teacher.compute(parent, [child], [torch.ones(1, 1)], branch="p0")

    assert out["coarse_top2_margin"].item() > 0.0
    assert out["confidence"].item() > 0.9


def test_candidate_chunk_zero_matches_chunked_grid_outputs():
    target = ParabolicTarget()
    parent = torch.tensor(
        [
            [0.1, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
            [0.3, 0.0, 1.0, 0.4, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    child0 = parent.clone()
    child1 = parent.clone()
    child0[:, 2:3] = 1.0
    child1[:, 2:3] = 0.0
    m_list = [torch.ones(3, 1), torch.full((3, 1), 0.95)]

    teacher_chunked = BPGridTeacher(
        target,
        P0Loss(),
        PILoss(),
        coarse_size=7,
        refine=False,
        candidate_chunk_size=4,
        max_expanded_states=65536,
    )
    teacher_full = BPGridTeacher(
        target,
        P0Loss(),
        PILoss(),
        coarse_size=7,
        refine=False,
        candidate_chunk_size=0,
        max_expanded_states=65536,
    )

    out_chunked = teacher_chunked.compute(parent, [child0, child1], m_list, branch="p0")
    out_full = teacher_full.compute(parent, [child0, child1], m_list, branch="p0")

    for key in [
        "bp_star",
        "value_star",
        "value_grid",
        "cashflow_grid_mean",
        "continuation_grid_mean",
        "q_issue_grid",
        "p_child_grid_mean",
        "default_grid_mean",
        "coarse_cashflow_grid_mean",
        "coarse_continuation_grid_mean",
    ]:
        torch.testing.assert_close(out_chunked[key], out_full[key], rtol=1e-5, atol=1e-6)
    assert torch.equal(out_chunked["argmax_index"], out_full["argmax_index"])


if __name__ == "__main__":
    test_bp_grid_teacher_selects_value_maximizing_bp()
    test_grid_teacher_debt_state_semantics()
    test_grid_teacher_mix_branch_and_coarse_confidence()
    test_pi_grid_target_multi_child_boundary_and_refine()
    test_relative_confidence_scales_margin_by_config_threshold()
    test_candidate_chunk_zero_matches_chunked_grid_outputs()
