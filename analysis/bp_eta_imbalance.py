from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Iterable, Mapping, Optional

import torch

from training.bp_policy_loss import (
    compute_target_grid_policy_distillation_loss,
    compute_target_grid_policy_logit_distillation_loss,
)


BP_HEAD_PREFIXES = ("bp0_head.", "bpi_head.")


def flatten_bp_target_cache(cache: Iterable[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
    """Concatenate the tensor portion of a fixed BP teacher cache."""
    items = list(cache)
    if not items:
        raise ValueError("BP target cache is empty")
    required = (
        "parent",
        "bp0_target",
        "bpi_target",
        "mix_target",
        "bp0_confidence",
        "bpi_confidence",
        "mix_confidence",
        "mix_sample_weight",
    )
    result: Dict[str, torch.Tensor] = {}
    for key in required:
        result[key] = torch.cat([item[key].detach().cpu() for item in items], dim=0)
    for key in ("bp0_regret", "bpi_regret", "mix_regret", "source_id", "source_index"):
        present = [isinstance(item.get(key), torch.Tensor) for item in items]
        if any(present) and not all(present):
            raise ValueError(f"BP cache has inconsistent optional field {key!r}")
        if all(present):
            result[key] = torch.cat([item[key].detach().cpu() for item in items], dim=0)
    result["eta_current"] = (result["parent"][:, 2:3] > 0.5).to(torch.long)
    return result


def select_bp_cache_rows(
    cache: Mapping[str, torch.Tensor],
    row_index: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    index = row_index.detach().cpu().to(torch.long).reshape(-1)
    n_rows = int(cache["parent"].shape[0])
    return {
        key: value[index].clone()
        for key, value in cache.items()
        if key != "eta_current" and value.shape[0] == n_rows
    }


def sample_current_eta_rows(
    eta_current: torch.Tensor,
    *,
    batch_size: int,
    eta1_share: Optional[float],
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample cache rows without changing future-eta transition probabilities."""
    eta = eta_current.detach().cpu().reshape(-1).bool()
    n_rows = int(eta.numel())
    if n_rows == 0:
        raise ValueError("Cannot sample an empty BP cache")
    requested = max(1, min(int(batch_size), n_rows))
    if eta1_share is None:
        # A full-cache natural batch preserves the empirical ratio exactly.
        if requested == n_rows:
            return torch.randperm(n_rows, generator=generator)
        return torch.randperm(n_rows, generator=generator)[:requested]

    share = float(eta1_share)
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"eta1_share must be in [0, 1], got {share}")
    eta1_index = torch.where(eta)[0]
    eta0_index = torch.where(~eta)[0]
    if eta0_index.numel() == 0 or eta1_index.numel() == 0:
        raise ValueError("Stratified current-eta sampling requires both eta strata")
    n_eta1 = int(round(requested * share))
    n_eta1 = min(max(n_eta1, 0), requested)
    n_eta0 = requested - n_eta1

    def _draw(pool: torch.Tensor, count: int) -> torch.Tensor:
        if count == 0:
            return torch.empty(0, dtype=torch.long)
        positions = torch.randint(pool.numel(), (count,), generator=generator)
        return pool[positions]

    selected = torch.cat([_draw(eta0_index, n_eta0), _draw(eta1_index, n_eta1)])
    return selected[torch.randperm(selected.numel(), generator=generator)]


def configure_bp_heads_only(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Freeze the full policy/value model except the two BP output heads."""
    trainable: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith(BP_HEAD_PREFIXES)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(parameter)
    if not trainable:
        raise ValueError("Model has no bp0_head/bpi_head parameters")
    return trainable


def state_dict_sha256(model: torch.nn.Module, prefixes: Optional[tuple[str, ...]] = None) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if prefixes is not None and not name.startswith(prefixes):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def parameter_change_summary(
    before: Mapping[str, torch.Tensor],
    model: torch.nn.Module,
) -> Dict[str, float]:
    head_max = 0.0
    non_head_max = 0.0
    for name, parameter in model.named_parameters():
        delta = float((parameter.detach().cpu() - before[name]).abs().max().item())
        if name.startswith(BP_HEAD_PREFIXES):
            head_max = max(head_max, delta)
        else:
            non_head_max = max(non_head_max, delta)
    return {"bp_head_parameter_max_change": head_max, "non_bp_parameter_max_change": non_head_max}


def _branch_tensors(
    model: torch.nn.Module,
    cache: Mapping[str, torch.Tensor],
    branch: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    parent = cache["parent"].to(device)
    bp0_logit, bpi_logit = model.forward_policy_logits(parent)
    if branch == "bp0":
        return bp0_logit, torch.sigmoid(bp0_logit), cache["bp0_target"].to(device), cache["bp0_confidence"].to(device)
    if branch == "bpi":
        return bpi_logit, torch.sigmoid(bpi_logit), cache["bpi_target"].to(device), cache["bpi_confidence"].to(device)
    raise ValueError(f"Unsupported direct BP branch: {branch!r}")


def branch_distillation_loss(
    model: torch.nn.Module,
    cache: Mapping[str, torch.Tensor],
    hyperparams: Any,
    *,
    branch: str,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = next(model.parameters()).device
    logit, prediction, target, confidence = _branch_tensors(model, cache, branch, device)
    if mask is not None:
        selected = mask.to(device).reshape(-1).bool()
        logit = logit[selected]
        prediction = prediction[selected]
        target = target[selected]
        confidence = confidence[selected]
    if prediction.numel() == 0:
        raise ValueError(f"No rows available for {branch} conditional loss")
    branch_weight = float(getattr(hyperparams, "bp_grid_policy_weight", 1.0))
    loss_space = str(getattr(hyperparams, "bp_grid_policy_loss_space", "output")).lower()
    if loss_space == "logit":
        total, _, _, _ = compute_target_grid_policy_logit_distillation_loss(
            logit,
            target,
            confidence,
            target_eps=float(getattr(hyperparams, "bp_grid_logit_target_eps", 1e-4)),
            huber_delta=float(getattr(hyperparams, "bp_grid_logit_huber_delta", 1.0)),
            branch_weight=branch_weight,
        )
        return total
    total, _, _ = compute_target_grid_policy_distillation_loss(
        prediction,
        target,
        confidence,
        huber_delta=float(getattr(hyperparams, "bp_grid_policy_huber_delta", 0.05)),
        branch_weight=branch_weight,
    )
    return total


def loss_contribution_audit(
    model: torch.nn.Module,
    cache: Mapping[str, torch.Tensor],
    hyperparams: Any,
) -> Dict[str, Dict[str, float]]:
    eta = cache["eta_current"].reshape(-1).bool()
    n = int(eta.numel())
    shares = {0: float((~eta).sum().item() / n), 1: float(eta.sum().item() / n)}
    result: Dict[str, Dict[str, float]] = {}
    for branch in ("bp0", "bpi"):
        l0 = float(branch_distillation_loss(model, cache, hyperparams, branch=branch, mask=~eta).detach().item())
        l1 = float(branch_distillation_loss(model, cache, hyperparams, branch=branch, mask=eta).detach().item())
        c0 = shares[0] * l0
        c1 = shares[1] * l1
        result[branch] = {
            "eta0_conditional_loss": l0,
            "eta1_conditional_loss": l1,
            "eta0_natural_contribution": c0,
            "eta1_natural_contribution": c1,
            "natural_contribution_ratio_eta0_to_eta1": c0 / (c1 + 1e-12),
        }
    return result


def _flat_gradient(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    chunks = [
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.detach().reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ]
    return torch.cat(chunks) if chunks else torch.empty(0, device=loss.device)


def gradient_contribution_audit(
    model: torch.nn.Module,
    cache: Mapping[str, torch.Tensor],
    hyperparams: Any,
) -> Dict[str, Dict[str, float]]:
    """Measure conditional gradients without populating .grad or taking a step."""
    eta = cache["eta_current"].reshape(-1).bool()
    n = int(eta.numel())
    share0 = float((~eta).sum().item() / n)
    share1 = float(eta.sum().item() / n)
    result: Dict[str, Dict[str, float]] = {}
    for branch, head_name in (("bp0", "bp0_head"), ("bpi", "bpi_head")):
        parameters = list(getattr(model, head_name).parameters())
        g0 = _flat_gradient(
            branch_distillation_loss(model, cache, hyperparams, branch=branch, mask=~eta),
            parameters,
        )
        g1 = _flat_gradient(
            branch_distillation_loss(model, cache, hyperparams, branch=branch, mask=eta),
            parameters,
        )
        norm0 = float(torch.linalg.vector_norm(g0).item())
        norm1 = float(torch.linalg.vector_norm(g1).item())
        weighted0 = share0 * norm0
        weighted1 = share1 * norm1
        denom = norm0 * norm1
        cosine = float(torch.dot(g0, g1).item() / denom) if denom > 0.0 else float("nan")
        result[branch] = {
            "eta0_gradient_norm": norm0,
            "eta1_gradient_norm": norm1,
            "eta0_natural_weighted_gradient_norm": weighted0,
            "eta1_natural_weighted_gradient_norm": weighted1,
            "natural_weighted_gradient_norm_ratio_eta0_to_eta1": weighted0 / (weighted1 + 1e-12),
            "gradient_cosine_similarity": cosine,
        }
    return result


def _finite_stats(values: torch.Tensor) -> Dict[str, float]:
    finite = values.detach().cpu().reshape(-1)
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        return {"mean": float("nan"), "std": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
        "p10": float(torch.quantile(finite, 0.10).item()),
        "p50": float(torch.quantile(finite, 0.50).item()),
        "p90": float(torch.quantile(finite, 0.90).item()),
    }


@torch.no_grad()
def audit_bp_cache_by_current_eta(
    model: torch.nn.Module,
    cache: Mapping[str, torch.Tensor],
    *,
    regrets: Optional[Mapping[str, torch.Tensor]] = None,
) -> list[Dict[str, float | int | str]]:
    device = next(model.parameters()).device
    parent = cache["parent"].to(device)
    output = model(parent)
    bp0_logit, bpi_logit = model.forward_policy_logits(parent)
    predictions = {"bp0": torch.sigmoid(bp0_logit), "bpi": torch.sigmoid(bpi_logit)}
    mix_pred = getattr(output, "bp_cond", None)
    if mix_pred is not None:
        predictions["mix"] = mix_pred
    eta = cache["eta_current"].reshape(-1).bool()
    n_total = int(eta.numel())
    rows: list[Dict[str, float | int | str]] = []
    for branch in predictions:
        target_key = f"{branch}_target"
        confidence_key = f"{branch}_confidence"
        target = cache[target_key].to(device).reshape(-1)
        prediction = predictions[branch].reshape(-1)
        confidence = cache[confidence_key].to(device).reshape(-1)
        if branch == "mix":
            confidence = confidence * cache["mix_sample_weight"].to(device).reshape(-1)
        branch_regret = None
        if regrets is not None:
            if branch in regrets:
                branch_regret = regrets[branch].to(device).reshape(-1)
        elif f"{branch}_regret" in cache:
            branch_regret = cache[f"{branch}_regret"].to(device).reshape(-1)
        for eta_value, mask in ((0, ~eta), (1, eta)):
            mask_device = mask.to(device)
            count = int(mask.sum().item())
            target_group = target[mask_device]
            pred_group = prediction[mask_device]
            confidence_group = confidence[mask_device]
            error = pred_group - target_group
            target_stats = _finite_stats(target_group)
            pred_stats = _finite_stats(pred_group)
            regret_group = branch_regret[mask_device] if branch_regret is not None else torch.empty(0, device=device)
            regret_stats = _finite_stats(regret_group)
            rows.append({
                "branch": branch,
                "eta_current": eta_value,
                "count": count,
                "share": count / n_total,
                "bp_star_mean": target_stats["mean"],
                "bp_star_std": target_stats["std"],
                "bp_star_p10": target_stats["p10"],
                "bp_star_p50": target_stats["p50"],
                "bp_star_p90": target_stats["p90"],
                "bp_pred_mean": pred_stats["mean"],
                "bp_pred_std": pred_stats["std"],
                "mae": float(error.abs().mean().item()) if error.numel() else float("nan"),
                "mean_abs_error": float(error.abs().mean().item()) if error.numel() else float("nan"),
                "mse": float(error.square().mean().item()) if error.numel() else float("nan"),
                "mean_regret": regret_stats["mean"],
                "median_regret": regret_stats["p50"],
                "p90_regret": regret_stats["p90"],
                "teacher_confidence_mean": float(confidence_group.mean().item()) if confidence_group.numel() else float("nan"),
                "teacher_identified_share": float((confidence_group > 0).float().mean().item()) if confidence_group.numel() else float("nan"),
                "target_low_boundary_share": float((target_group < 0.05).float().mean().item()) if target_group.numel() else float("nan"),
                "target_high_boundary_share": float((target_group > 0.95).float().mean().item()) if target_group.numel() else float("nan"),
            })
    return rows


@torch.no_grad()
def eta_switch_sensitivity(model: torch.nn.Module, parent: torch.Tensor) -> Dict[str, float]:
    device = next(model.parameters()).device
    reference = parent.detach().to(device).reshape(1, -1).clone()
    eta0 = reference.clone()
    eta1 = reference.clone()
    eta0[:, 2] = 0.0
    eta1[:, 2] = 1.0
    bp0_eta0, bpi_eta0 = model.forward_policy(eta0)
    bp0_eta1, bpi_eta1 = model.forward_policy(eta1)
    return {
        "bp0_eta0": float(bp0_eta0.item()),
        "bp0_eta1": float(bp0_eta1.item()),
        "delta_bp0_eta": float((bp0_eta1 - bp0_eta0).item()),
        "bpi_eta0": float(bpi_eta0.item()),
        "bpi_eta1": float(bpi_eta1.item()),
        "delta_bpi_eta": float((bpi_eta1 - bpi_eta0).item()),
    }


def eta_count_summary(cache: Mapping[str, torch.Tensor]) -> Dict[str, float | int]:
    eta = cache["eta_current"].reshape(-1).bool()
    eta1 = int(eta.sum().item())
    eta0 = int((~eta).sum().item())
    total = eta0 + eta1
    return {
        "eta0_count": eta0,
        "eta1_count": eta1,
        "eta0_share": eta0 / total,
        "eta1_share": eta1 / total,
        "eta0_to_eta1_count_ratio": eta0 / eta1 if eta1 else math.inf,
    }
