"""
Deterministic firm-side derived objects.

This module keeps default, survival, investment, and total-equity semantics out
of trainable heads. Value heads produce branch values; this layer derives the
objects used by Bellman and simulation code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FirmDerivedObjects(nn.Module):
    """Parameter-free transforms from branch values to firm decisions."""

    def __init__(self, tau_i: float = 0.1, tau_z: float = 0.1):
        super().__init__()
        self.tau_i = max(float(tau_i), 1e-8)
        self.tau_z = max(float(tau_z), 1e-8)

    def investment_conditional(self, v0: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        """Investment probability conditional on current survival."""
        return torch.sigmoid((vi - v0) / self.tau_i)

    def survival(self, phat: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """Survival probability implied by the equity envelope."""
        if hard:
            return (phat > 0).to(phat.dtype)
        return torch.sigmoid(phat / self.tau_z)

    def default(self, phat: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """Default probability implied by the equity envelope."""
        return 1.0 - self.survival(phat, hard=hard)

    def total_equity(self, phat: torch.Tensor, hard: bool = True) -> torch.Tensor:
        """Limited-liability total equity value."""
        if hard:
            return torch.clamp_min(phat, 0.0)
        return self.tau_z * F.softplus(phat / self.tau_z)
