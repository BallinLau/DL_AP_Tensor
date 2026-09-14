"""Firm state transition helpers."""

from __future__ import annotations

import torch


def apply_refinancing_policy(
    b_current: torch.Tensor,
    bp_candidate: torch.Tensor,
    eta_next: torch.Tensor,
) -> torch.Tensor:
    """Apply the child refinancing realization to next-period leverage.

    Timing invariant:
        b_{t+1} = eta_{t+1} * bp_t + (1 - eta_{t+1}) * b_t

    ``eta_next`` belongs to the child state. Current ``eta_t`` remains relevant
    to current-period financing cash flow, but never determines child leverage.
    """
    eta = eta_next.to(
        device=bp_candidate.device,
        dtype=bp_candidate.dtype,
    ).clamp(0.0, 1.0)
    b = b_current.to(
        device=bp_candidate.device,
        dtype=bp_candidate.dtype,
    )
    return eta * bp_candidate + (1.0 - eta) * b
