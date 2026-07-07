"""
Target-network utilities for firm policy/value training.
"""

import torch
import torch.nn as nn


@torch.no_grad()
def hard_update(target: nn.Module, online: nn.Module) -> None:
    """Copy online parameters and buffers into target."""
    target.load_state_dict(online.state_dict())


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """Polyak update target parameters and buffers from online."""
    tau = float(tau)
    if tau <= 0.0:
        return
    if tau >= 1.0:
        hard_update(target, online)
        return

    target_state = target.state_dict()
    online_state = online.state_dict()
    for name, target_value in target_state.items():
        online_value = online_state[name]
        if torch.is_floating_point(target_value):
            target_value.mul_(1.0 - tau).add_(online_value, alpha=tau)
        else:
            target_value.copy_(online_value)
