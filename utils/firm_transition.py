"""Firm state transition helpers."""

from __future__ import annotations

import torch


def apply_refinancing_policy(
    b_current: torch.Tensor,
    bp_candidate: torch.Tensor,
    eta_current: torch.Tensor,
) -> torch.Tensor:
    """Apply the current-period refinancing shock to next leverage.

    GS timing:
        b_{t+1} = eta_t * bp_t + (1 - eta_t) * b_t

    ``eta_current`` is observed at the beginning of period t. The next-period
    refinancing shock eta_{t+1} must not enter this transition.
    """
    eta = eta_current.to(
        device=bp_candidate.device,
        dtype=bp_candidate.dtype,
    ).clamp(0.0, 1.0)
    b = b_current.to(
        device=bp_candidate.device,
        dtype=bp_candidate.dtype,
    )
    return eta * bp_candidate + (1.0 - eta) * b
