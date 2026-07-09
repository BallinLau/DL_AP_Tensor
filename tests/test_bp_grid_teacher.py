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

    def __call__(self, firm_state):
        b = firm_state[:, 0:1]
        p = 1.0 - (b - 0.32).pow(2)
        p = p.clamp_min(0.0)
        return SimpleNamespace(
            Q=self._q_output(firm_state),
            P=p,
            Phat=p,
            bar_z=(p <= 0.0).to(p.dtype),
        )


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


if __name__ == "__main__":
    test_bp_grid_teacher_selects_value_maximizing_bp()
