from __future__ import annotations

import torch


def huber_element(prediction: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
    diff = (prediction - target).abs()
    beta = float(beta)
    if beta <= 0:
        return diff
    return torch.where(diff < beta, 0.5 * diff.pow(2) / beta, diff - 0.5 * beta)


def compute_target_grid_policy_distillation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence_weight: torch.Tensor,
    *,
    huber_delta: float,
    branch_weight: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted target-grid policy loss with training-equivalent semantics.

    This mirrors the policy distillation term used by Episode target-grid PV
    training: elementwise Huber loss, multiplied by the teacher confidence and
    any optional branch/sample gate, then reduced by a plain mean. The returned
    total includes the branch-level scalar weight.
    """
    elem = huber_element(prediction, target, huber_delta)
    weight = confidence_weight
    if sample_weight is not None:
        weight = weight * sample_weight.detach().clamp(0.0, 1.0)
    unweighted_branch_loss = (weight * elem).mean()
    weighted_branch_loss = float(branch_weight) * unweighted_branch_loss
    return weighted_branch_loss, unweighted_branch_loss, elem
