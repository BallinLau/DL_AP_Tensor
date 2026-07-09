"""
Policy/value forward helpers for simulation rollouts.

Simulation should use an explicit rollout-facing model entry point.  The
target-grid teacher is a training-only object and is never called here.
"""

from __future__ import annotations

from typing import Any

import torch


def forward_policy_value_for_simulation(pv_model: Any, firm_state: torch.Tensor) -> Any:
    """
    Run the policy/value model through its simulation-specific entry point.

    Models that do not yet implement ``forward_simulation`` fall back to the
    regular callable interface for backward compatibility.
    """
    forward_simulation = getattr(pv_model, "forward_simulation", None)
    if callable(forward_simulation):
        return forward_simulation(firm_state)
    return pv_model(firm_state)
