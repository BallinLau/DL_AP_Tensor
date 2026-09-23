from __future__ import annotations

import torch


def huber_element(prediction: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
    diff = (prediction - target).abs()
    beta = float(beta)
    if beta <= 0:
        return diff
    return torch.where(diff < beta, 0.5 * diff.pow(2) / beta, diff - 0.5 * beta)


def reduce_active_weighted(
    weighted_element: torch.Tensor,
    active_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Reduce ``weight * elem`` over the active refinancing rows only.

    ``active_mask`` is the ``eta_t = 1`` indicator. Rows with ``eta_t = 0`` have
    no refinancing choice, so they must contribute neither to the numerator nor
    to the denominator: the objective is the conditional
    ``E[confidence * elem | eta_t = 1]`` and must not be diluted by the
    refinancing frequency.

    ``confidence`` stays a teacher-reliability multiplier inside the numerator
    and is deliberately **not** renormalized by ``sum(confidence)``.

    When no row is active, an exact zero is returned that still carries a valid
    autograd graph so a caller can safely call ``backward()`` on it.
    """
    if active_mask is None:
        return weighted_element.mean()
    active = active_mask.to(weighted_element.dtype)
    while active.ndim < weighted_element.ndim:
        active = active.unsqueeze(-1)
    active = active.expand_as(weighted_element) > 0.5
    if bool(active.any()):
        return weighted_element[active].mean()
    return weighted_element.sum() * 0.0


def compute_target_grid_policy_distillation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence_weight: torch.Tensor,
    *,
    huber_delta: float,
    branch_weight: float = 1.0,
    sample_weight: torch.Tensor | None = None,
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted target-grid policy loss with training-equivalent semantics.

    This mirrors the policy distillation term used by Episode target-grid PV
    training: elementwise Huber loss, multiplied by the teacher confidence and
    any optional branch/sample gate, then reduced over the ``eta_t = 1`` rows
    only. The returned total includes the branch-level scalar weight.

    ``active_mask`` is required to be the parent ``eta_t`` indicator (or the
    teacher's ``refi_active``); eta_t = 0 rows are excluded from both numerator
    and denominator. Passing ``None`` restores the legacy full-batch mean and is
    reserved for callers that already restricted the batch to active rows.
    """
    elem = huber_element(prediction, target, huber_delta)
    weight = confidence_weight
    if sample_weight is not None:
        weight = weight * sample_weight.detach().clamp(0.0, 1.0)
    unweighted_branch_loss = reduce_active_weighted(weight * elem, active_mask)
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
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted target-grid policy loss in BP-logit space.

    Weighting and reduction intentionally match
    compute_target_grid_policy_distillation_loss: confidence/sample weights
    multiply elementwise Huber losses, the mean is taken over the ``eta_t = 1``
    rows only, then the branch-level scalar is applied.
    """
    target_logit = stable_logit_target(target_probability, target_eps)
    elem = huber_element(prediction_logit, target_logit, huber_delta)
    weight = confidence_weight
    if sample_weight is not None:
        weight = weight * sample_weight.detach().clamp(0.0, 1.0)
    unweighted_branch_loss = reduce_active_weighted(weight * elem, active_mask)
    weighted_branch_loss = float(branch_weight) * unweighted_branch_loss
    return weighted_branch_loss, unweighted_branch_loss, elem, target_logit
