"""Firm state transition helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch


@dataclass(frozen=True)
class ExactEtaChildExpansion:
    """Eta-enumerated children for exact Bernoulli marginalization.

    Each continuous child contributes an adjacent ``eta=0, eta=1`` pair. The
    expanded states are an expectation representation, not independent AiO
    branches.
    """

    children: List[torch.Tensor]
    branch_weights: torch.Tensor
    source_child_indices: tuple[int, ...]
    continuous_child_count: int


def normalize_child_weights(
    weights: Optional[torch.Tensor],
    *,
    n_parent: int,
    n_child: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if weights is None:
        return torch.full(
            (n_parent, n_child), 1.0 / float(n_child), device=device, dtype=dtype
        )
    normalized = weights.to(device=device, dtype=dtype)
    if normalized.ndim == 1:
        if normalized.numel() != n_child:
            raise ValueError(f"child weights must have {n_child} entries")
        normalized = normalized.unsqueeze(0).expand(n_parent, n_child)
    if normalized.shape != (n_parent, n_child):
        raise ValueError(
            f"child weights must have shape {(n_parent, n_child)}, got {tuple(normalized.shape)}"
        )
    if torch.any(normalized < 0):
        raise ValueError("child weights must be nonnegative")
    row_sum = normalized.sum(dim=1, keepdim=True)
    if torch.any(row_sum <= 0):
        raise ValueError("child weight rows must have positive mass")
    return normalized / row_sum


def expand_children_exact_eta(
    children: Sequence[torch.Tensor],
    *,
    zeta: float,
    child_weights: Optional[torch.Tensor] = None,
) -> ExactEtaChildExpansion:
    """Enumerate eta=0/1 for each independent continuous child.

    The returned weights are ``w_j*(1-zeta)`` and ``w_j*zeta``. All state
    components except eta are copied exactly within each pair.
    """
    if not children:
        raise ValueError("exact eta expansion requires at least one child")
    probability = float(zeta)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"zeta must be in [0, 1], got {probability}")
    first = children[0]
    n_parent = int(first.shape[0])
    for child in children:
        if child.ndim != 2 or child.shape[0] != n_parent or child.shape[1] < 3:
            raise ValueError("all child tensors must have shape [B,D] with D>=3")
    base_weights = normalize_child_weights(
        child_weights,
        n_parent=n_parent,
        n_child=len(children),
        device=first.device,
        dtype=first.dtype,
    )
    expanded: List[torch.Tensor] = []
    expanded_weights = []
    source_indices = []
    for child_index, child in enumerate(children):
        eta0 = child.clone()
        eta1 = child.clone()
        eta0[:, 2] = 0.0
        eta1[:, 2] = 1.0
        expanded.extend((eta0, eta1))
        expanded_weights.extend(
            (
                base_weights[:, child_index] * (1.0 - probability),
                base_weights[:, child_index] * probability,
            )
        )
        source_indices.extend((child_index, child_index))
    return ExactEtaChildExpansion(
        children=expanded,
        branch_weights=torch.stack(expanded_weights, dim=1),
        source_child_indices=tuple(source_indices),
        continuous_child_count=len(children),
    )


def exact_eta_pair_expectation(values: torch.Tensor, *, zeta: float) -> torch.Tensor:
    """Collapse adjacent eta pairs without changing the independent-child axis."""
    if values.ndim < 2 or values.shape[1] % 2:
        raise ValueError("values must have shape [B,2J,...] with adjacent eta pairs")
    probability = float(zeta)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"zeta must be in [0, 1], got {probability}")
    paired = values.reshape(values.shape[0], values.shape[1] // 2, 2, *values.shape[2:])
    return (1.0 - probability) * paired[:, :, 0] + probability * paired[:, :, 1]


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
