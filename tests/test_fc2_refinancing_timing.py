"""FC2 leverage-timing regression tests.

The realized next-period leverage is

    b_{t+1} = eta_t * bp_t + (1 - eta_t) * b_t

where ``eta_t`` is the CURRENT parent refinancing realization. The child's
``eta_{t+1}`` is a state shock only: it must never gate realized leverage, and
both children of a parent share the same realized ``b_next``.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from losses.FC2losspipe import FC2Pipeline  # noqa: E402

B_PARENT = 0.7
BP_TARGET = 0.2


class _StubFC2Model(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        n = x.shape[0]
        dtype = x.dtype
        return {
            "hatc": torch.zeros(n, 1, dtype=dtype),
            "lnk": torch.zeros(n, 1, dtype=dtype),
        }


class _StubPolicyValueModel(torch.nn.Module):
    """Minimal stand-in for the PV net: constant survival, zero investment, fixed bp."""

    def __init__(self, bp: float):
        super().__init__()
        self._bp = float(bp)

    def forward(self, x: torch.Tensor):
        n = x.shape[0]
        dtype = x.dtype
        return {
            "bar_z": torch.ones(n, 1, dtype=dtype),
            "bar_i": torch.zeros(n, 1, dtype=dtype),
            "bp": torch.full((n, 1), self._bp, dtype=dtype),
        }


def _fc2_dataframe(
    *,
    eta_parent: float,
    eta_child1: float,
    eta_child2: float,
    b_parent: float = B_PARENT,
) -> pd.DataFrame:
    """One path, one parent, two children with independently chosen eta shocks."""
    rows = [
        dict(
            path=0, ID="p0", t="t", branch=0,
            b=b_parent, z=0.1, ETA=eta_parent, i=0.1, x=0.0,
            Hatcf=0.0, LnKF=0.0, M=1.0, K=1.0, Entry=0,
        ),
        dict(
            path=0, ID="p0", t="t+1_0", branch=1,
            b=0.0, z=0.05, ETA=eta_child1, i=0.1, x=0.0,
            Hatcf=0.0, LnKF=0.0, M=1.0, K=1.0, Entry=0,
        ),
        dict(
            path=0, ID="p0", t="t+1_1", branch=2,
            b=0.0, z=-0.05, ETA=eta_child2, i=0.1, x=0.0,
            Hatcf=0.0, LnKF=0.0, M=1.0, K=1.0, Entry=0,
        ),
    ]
    return pd.DataFrame(rows)


def _run_fc2(*, eta_parent: float, eta_child1: float, eta_child2: float) -> torch.Tensor:
    pipe = FC2Pipeline(
        df=_fc2_dataframe(
            eta_parent=eta_parent,
            eta_child1=eta_child1,
            eta_child2=eta_child2,
        ),
        full_N=1,
        device="cpu",
    )
    outputs = pipe.forward(_StubFC2Model(), _StubPolicyValueModel(BP_TARGET))
    return outputs["children_s_full"]


def test_fc2_child_leverage_uses_current_parent_eta_not_child_eta():
    """TEST 1: the same parent eta_t forces an identical b_next for both children.

    Child eta_{t+1} realizations deliberately differ (0 and 1) so that any
    regression back to ``child_eta * bp + (1 - child_eta) * b`` would make the
    two children disagree and fail here.
    """
    # Case A: parent eta_t = 0 -> no refinancing, realized leverage stays at b_t.
    children = _run_fc2(eta_parent=0.0, eta_child1=0.0, eta_child2=1.0)
    child_b = children[0, 0, :, 0]
    assert float(child_b[0]) == pytest.approx(B_PARENT, abs=1e-6)
    assert float(child_b[1]) == pytest.approx(B_PARENT, abs=1e-6)
    # The differing child eta_{t+1} shocks are preserved untouched.
    child_eta = children[0, 0, :, 2]
    assert float(child_eta[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(child_eta[1]) == pytest.approx(1.0, abs=1e-6)

    # Case B: parent eta_t = 1 -> refinancing triggers, realized leverage is bp_t.
    children = _run_fc2(eta_parent=1.0, eta_child1=0.0, eta_child2=1.0)
    child_b = children[0, 0, :, 0]
    assert float(child_b[0]) == pytest.approx(BP_TARGET, abs=1e-6)
    assert float(child_b[1]) == pytest.approx(BP_TARGET, abs=1e-6)
    child_eta = children[0, 0, :, 2]
    assert float(child_eta[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(child_eta[1]) == pytest.approx(1.0, abs=1e-6)
