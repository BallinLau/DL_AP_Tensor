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


def stable_logit_target(target_probability: torch.Tensor, eps: float) -> torch.Tensor:
    eps = float(eps)
    if not 0.0 < eps < 0.5:
        raise ValueError(f"eps must be in (0, 0.5), got {eps}")
    return torch.logit(target_probability.detach().clamp(eps, 1.0 - eps))


def compute_target_grid_policy_logit_distillation_loss(
    prediction_logit: torch.Tensor,
    target_probability: torch.Tensor,
    confidence_weight: torch.Tensor,
    *,
    target_eps: float,
    huber_delta: float,
    branch_weight: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted target-grid policy loss in BP-logit space.

    Weighting and reduction intentionally match
    compute_target_grid_policy_distillation_loss: confidence/sample weights
    multiply elementwise Huber losses, then a plain mean is taken, then the
    branch-level scalar is applied.
    """
    target_logit = stable_logit_target(target_probability, target_eps)
    elem = huber_element(prediction_logit, target_logit, huber_delta)
    weight = confidence_weight
    if sample_weight is not None:
        weight = weight * sample_weight.detach().clamp(0.0, 1.0)
    unweighted_branch_loss = (weight * elem).mean()
    weighted_branch_loss = float(branch_weight) * unweighted_branch_loss
    return weighted_branch_loss, unweighted_branch_loss, elem, target_logit
