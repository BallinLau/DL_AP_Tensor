from __future__ import annotations

import torch

from config import Config
from experiments.export_continuation_surface import make_surface_states


def test_make_surface_states_varies_only_child_b_and_z():
    child = torch.tensor([[0.2, 0.1, 1.0, 0.3, 0.4, 0.5, 0.6]], dtype=torch.float64)
    states, b, z = make_surface_states(child, b_grid_size=3, z_min=-1.0, z_max=1.0, z_grid_size=5)
    assert states.shape == (15, 7)
    assert b.shape == (15, 1)
    assert z.shape == (15, 1)
    torch.testing.assert_close(states[:, 0:1], b)
    torch.testing.assert_close(states[:, 1:2], z)
    torch.testing.assert_close(states[:, 2:], child.expand(15, -1)[:, 2:])


def test_q_survival_recovery_decomposition_identity():
    m = torch.tensor([[0.9], [1.1]], dtype=torch.float64)
    b = torch.tensor([[0.2], [0.5]], dtype=torch.float64)
    q_child = torch.tensor([[0.1], [0.2]], dtype=torch.float64)
    default = torch.tensor([[0.0], [0.7]], dtype=torch.float64)
    x = torch.tensor([[0.3], [0.4]], dtype=torch.float64)
    z = torch.tensor([[0.1], [0.2]], dtype=torch.float64)
    multiplier = Config.G
    q_issue = torch.tensor([[0.25], [0.35]], dtype=torch.float64)
    recovery = b * Config.PHI * (1.0 - Config.DELTA + torch.exp(x + z))
    survival_part = m * (b + multiplier * q_child) * (1.0 - default)
    recovery_part = m * recovery * multiplier * default
    total = survival_part + recovery_part
    residual = q_issue - total
    torch.testing.assert_close(q_issue, total + residual)
    recovery_share = recovery_part / total.clamp_min(1e-12)
    assert torch.isfinite(recovery_share).all()
