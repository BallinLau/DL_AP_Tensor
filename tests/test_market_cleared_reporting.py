from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.market_cleared_reporting import _plot_firm_probe_surfaces


def test_firm_surface_uses_parent_decision_not_flattened_i_grid(tmp_path: Path):
    count = 8
    parents = torch.zeros((count, 7), dtype=torch.float32)
    parents[:, 0] = torch.linspace(0.05, 0.95, count)
    parents[:, 1] = torch.linspace(-1.0, 1.0, count)
    parents[:, 2] = torch.tensor([0, 1] * (count // 2), dtype=torch.float32)
    p0 = torch.linspace(0.1, 0.8, count).reshape(-1, 1)
    pi = p0 + torch.tensor(
        [-0.2, 0.2, -0.1, 0.1, -0.3, 0.3, -0.4, 0.4]
    ).reshape(-1, 1)
    payload = {
        "parent_states": parents,
        "P0": p0,
        "PI": pi,
        "Q": torch.ones((count, 1)),
        "bp0": torch.zeros((count, 1)),
        "bpI": torch.ones((count, 1)),
        # This is the production shape that exposed the report-only bug.
        "investment": torch.zeros((count, 11), dtype=torch.bool),
        "survival": torch.ones(count, dtype=torch.bool),
    }
    outer = tmp_path / "equilibrium_outer_000"
    outer.mkdir(parents=True)
    torch.save(payload, outer / "equilibrium_probe.pt")

    _plot_firm_probe_surfaces(tmp_path)

    figures = outer / "figures" / "firm_surfaces"
    assert (figures / "outer_000_investment_boundary_eta0.png").is_file()
    assert (figures / "outer_000_investment_boundary_eta1.png").is_file()
    assert len(list(figures.glob("*.png"))) == 14
