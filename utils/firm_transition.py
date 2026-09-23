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
    children_tensor: Optional[torch.Tensor] = None


def expand_children_exact_eta_tensor(
    children: torch.Tensor,
    *,
    zeta: float,
    child_weights: Optional[torch.Tensor] = None,
) -> ExactEtaChildExpansion:
    """Tensorized exact Bernoulli eta expansion for children shaped ``[B,J,D]``."""
    if children.ndim != 3 or children.shape[1] < 1 or children.shape[2] < 3:
        raise ValueError("children must have shape [B,J,D] with J>=1 and D>=3")
    probability = float(zeta)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"zeta must be in [0, 1], got {probability}")
    n_parent, n_child, _ = children.shape
    base_weights = normalize_child_weights(
        child_weights,
        n_parent=n_parent,
        n_child=n_child,
        device=children.device,
        dtype=children.dtype,
    )
    eta_pair = children.unsqueeze(2).expand(-1, -1, 2, -1).clone()
    eta_pair[:, :, 0, 2] = 0.0
    eta_pair[:, :, 1, 2] = 1.0
    children_tensor = eta_pair.reshape(n_parent, 2 * n_child, children.shape[2])
    eta_probabilities = torch.as_tensor(
        [1.0 - probability, probability],
        device=children.device,
        dtype=children.dtype,
    )
    expanded_weights = (
        base_weights.unsqueeze(-1) * eta_probabilities.reshape(1, 1, 2)
    ).reshape(n_parent, 2 * n_child)
    source_indices = tuple(index for index in range(n_child) for _ in range(2))
    return ExactEtaChildExpansion(
        children=list(children_tensor.unbind(dim=1)),
        branch_weights=expanded_weights,
        source_child_indices=source_indices,
        continuous_child_count=n_child,
        children_tensor=children_tensor,
    )


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
    first = children[0]
    n_parent = int(first.shape[0])
    for child in children:
        if child.ndim != 2 or child.shape[0] != n_parent or child.shape[1] < 3:
            raise ValueError("all child tensors must have shape [B,D] with D>=3")
    return expand_children_exact_eta_tensor(
        torch.stack(tuple(children), dim=1),
        zeta=zeta,
        child_weights=child_weights,
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
    eta_current: torch.Tensor,
) -> torch.Tensor:
    """Apply the current refinancing realization to next-period leverage.

    Timing invariant:
        b_{t+1} = eta_t * bp_t + (1 - eta_t) * b_t

    ``eta_current`` is ``eta_t``, the refinancing realization that is already part
    of the current parent state. It is the only eta that determines realized next
    leverage:

    * ``eta_t = 1``: refinancing happens, so ``b_{t+1} = bp_t``.
    * ``eta_t = 0``: no refinancing choice exists, so ``b_{t+1} = b_t``.

    ``eta_{t+1}`` remains a child state shock and is still enumerated in the
    expectation over children, but it never gates ``b_t -> b_{t+1}``.
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
