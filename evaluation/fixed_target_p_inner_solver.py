"""Utilities for the fixed-target P inner-solver diagnostic.

This module is experiment-only.  It deliberately delegates target construction,
value scaling, optimizer construction, clipping, and parameter routing to the
production ``Episode`` implementation while keeping the Bellman target cache
immutable across independent training budgets.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch


SELECTED_REPORTED_CYCLES = (1, 5, 10)


def cycle_start_index(reported_cycle: int) -> int:
    """Map reported fitted-update cycle c to its X^(c-1) start checkpoint."""
    cycle = int(reported_cycle)
    if cycle <= 0:
        raise ValueError("reported_cycle must be positive")
    return cycle - 1


def tensor_payload_sha256(payload: Any) -> str:
    """Stable content hash for nested tensors and metadata."""
    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(b"ndarray")
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(str(array.shape).encode("utf-8"))
            digest.update(array.tobytes())
        elif isinstance(value, Mapping):
            digest.update(b"mapping")
            for key in sorted(value, key=str):
                update(str(key))
                update(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode("utf-8"))
            for item in value:
                update(item)
        elif value is None:
            digest.update(b"none")
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))

    update(payload)
    return digest.hexdigest()


def parameter_state_sha256(model: torch.nn.Module) -> str:
    return tensor_payload_sha256(model.state_dict())


def parameter_max_abs_difference(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
) -> float:
    if set(first) != set(second):
        raise ValueError("parameter state keys differ")
    maximum = 0.0
    for key in first:
        left = first[key].detach().cpu()
        right = second[key].detach().cpu()
        maximum = max(maximum, float((left - right).abs().max().item()))
    return maximum


def snapshot_named_parameters(
    model: torch.nn.Module,
    prefixes: Sequence[str],
) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    }


def parameter_update_norms(
    model: torch.nn.Module,
    before: Mapping[str, torch.Tensor],
) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        prefix = name.split(".", 1)[0]
        delta = parameter.detach().cpu() - before[name]
        sums[prefix] = sums.get(prefix, 0.0) + float(delta.square().sum().item())
    return {f"{prefix}_update_norm": math.sqrt(value) for prefix, value in sums.items()}


def _finite_vector(tensors: Iterable[torch.Tensor]) -> np.ndarray:
    arrays = [tensor.detach().cpu().reshape(-1).numpy() for tensor in tensors]
    values = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("fixed-target diagnostic encountered non-finite values")
    return values.astype(np.float64, copy=False)


def _error_summary(predicted: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    error = predicted - target
    absolute = np.abs(error)
    return {
        "mae": float(absolute.mean()),
        "rms": float(np.sqrt(np.mean(np.square(error)))),
        "p50_abs": float(np.quantile(absolute, 0.50)),
        "p90_abs": float(np.quantile(absolute, 0.90)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "max_abs": float(absolute.max()),
    }


@dataclass(frozen=True)
class FitEvaluation:
    metrics: Dict[str, float]
    objective: Dict[str, float]


def evaluate_fixed_cache(
    episode: Any,
    batches: Sequence[Mapping[str, Any]],
    cache: Sequence[Any],
) -> FitEvaluation:
    """Evaluate global train/validation fit without averaging batch quantiles."""
    if len(batches) != len(cache) or not batches:
        raise ValueError("batches/cache must be non-empty and aligned")
    model = episode.models["policy_value"]
    was_training = model.training
    p0_predicted: list[torch.Tensor] = []
    p0_targets: list[torch.Tensor] = []
    pi_predicted: list[torch.Tensor] = []
    pi_targets: list[torch.Tensor] = []
    p0_normalized: list[torch.Tensor] = []
    pi_normalized: list[torch.Tensor] = []
    p0_loss_space: list[torch.Tensor] = []
    pi_loss_space: list[torch.Tensor] = []
    objective_sums: Dict[str, float] = {}
    objective_weight = 0
    try:
        model.eval()
        with torch.no_grad():
            for batch, item in zip(batches, cache):
                parent = batch["parent"]
                state = parent[:, :7] if parent.shape[1] > 7 else parent
                p0_pred, pi_pred = model._value_outputs(state)
                p0_target = item.p0_value_target.to(p0_pred.device)
                pi_target = item.pi_value_target.to(pi_pred.device)
                scale = episode._pv_value_scale(model, state).detach().clamp_min(1e-8)
                normalize = bool(
                    getattr(episode.hyperparams, "pv_bellman_normalize_by_value_scale", False)
                )
                loss_scale = scale if normalize else torch.ones_like(scale)
                p0_predicted.append(p0_pred)
                pi_predicted.append(pi_pred)
                p0_targets.append(p0_target)
                pi_targets.append(pi_target)
                p0_normalized.append((p0_pred - p0_target) / scale)
                pi_normalized.append((pi_pred - pi_target) / scale)
                p0_loss_space.append((p0_pred - p0_target) / loss_scale)
                pi_loss_space.append((pi_pred - pi_target) / loss_scale)
                _, terms = episode._compute_cached_value_loss(batch, item)
                rows = int(state.shape[0])
                objective_weight += rows
                for key, value in terms.items():
                    if isinstance(value, (int, float, bool)):
                        objective_sums[key] = objective_sums.get(key, 0.0) + float(value) * rows
    finally:
        model.train(was_training)

    p0_pred = _finite_vector(p0_predicted)
    pi_pred = _finite_vector(pi_predicted)
    p0_target = _finite_vector(p0_targets)
    pi_target = _finite_vector(pi_targets)
    p0 = _error_summary(p0_pred, p0_target)
    pi = _error_summary(pi_pred, pi_target)
    combined = _error_summary(
        np.concatenate([p0_pred, pi_pred]),
        np.concatenate([p0_target, pi_target]),
    )
    metrics: Dict[str, float] = {}
    for prefix, summary in (("p0", p0), ("pi", pi), ("combined", combined)):
        metrics.update({f"{prefix}_physical_{key}": value for key, value in summary.items()})
    for prefix, values in (
        ("p0_normalized", _finite_vector(p0_normalized)),
        ("pi_normalized", _finite_vector(pi_normalized)),
        ("p0_loss_space", _finite_vector(p0_loss_space)),
        ("pi_loss_space", _finite_vector(pi_loss_space)),
    ):
        metrics[f"{prefix}_mae"] = float(np.abs(values).mean())
        metrics[f"{prefix}_rms"] = float(np.sqrt(np.mean(np.square(values))))
    combined_normalized = np.concatenate(
        [_finite_vector(p0_normalized), _finite_vector(pi_normalized)]
    )
    combined_loss_space = np.concatenate(
        [_finite_vector(p0_loss_space), _finite_vector(pi_loss_space)]
    )
    metrics["combined_normalized_mae"] = float(np.abs(combined_normalized).mean())
    metrics["combined_normalized_rms"] = float(
        np.sqrt(np.mean(np.square(combined_normalized)))
    )
    metrics["combined_loss_space_mae"] = float(np.abs(combined_loss_space).mean())
    metrics["combined_loss_space_rms"] = float(
        np.sqrt(np.mean(np.square(combined_loss_space)))
    )
    objective = {
        key: value / max(objective_weight, 1)
        for key, value in objective_sums.items()
    }
    return FitEvaluation(metrics=metrics, objective=objective)


def pure_value_regression_loss(
    episode: Any,
    batch: Mapping[str, Any],
    cache_item: Any,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Production Huber/value-scale semantics without z/b auxiliary weighting."""
    parent = batch["parent"]
    state = parent[:, :7] if parent.shape[1] > 7 else parent
    model = episode.models["policy_value"]
    p0_pred, pi_pred = model._value_outputs(state)
    p0_target = cache_item.p0_value_target.to(p0_pred.device)
    pi_target = cache_item.pi_value_target.to(pi_pred.device)
    scale = episode._pv_value_scale(model, state).detach().clamp_min(1e-8)
    normalize = bool(
        getattr(episode.hyperparams, "pv_bellman_normalize_by_value_scale", False)
    )
    loss_scale = scale if normalize else torch.ones_like(scale)
    delta = float(getattr(episode.hyperparams, "bp_grid_value_huber_delta", 1.0))
    p0_loss = episode._huber_element(
        p0_pred / loss_scale, p0_target / loss_scale, delta
    ).mean()
    pi_loss = episode._huber_element(
        pi_pred / loss_scale, pi_target / loss_scale, delta
    ).mean()
    total = (
        episode.weight_scheduler["p0"] * p0_loss
        + episode.weight_scheduler["pi"] * pi_loss
    )
    return total, {
        "p0_cached_value_loss": float(p0_loss.detach().item()),
        "pi_cached_value_loss": float(pi_loss.detach().item()),
        "p0_cached_penalty_z": 0.0,
        "pi_cached_penalty_z": 0.0,
        "pi_cached_penalty_b": 0.0,
        "total": float(total.detach().item()),
    }


def surface_complexity(
    values: np.ndarray,
    *,
    jump_threshold: float | None = None,
) -> Dict[str, float]:
    """Summarize level, first differences, curvature, and total variation."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 3:
        raise ValueError("surface must be a 2-D array with at least 3 points per axis")
    if not np.isfinite(array).all():
        raise ValueError("surface contains non-finite values")
    diff_b = np.diff(array, axis=0)
    diff_z = np.diff(array, axis=1)
    curv_b = np.diff(array, n=2, axis=0)
    curv_z = np.diff(array, n=2, axis=1)
    flat = array.reshape(-1)
    result = {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "rms": float(np.sqrt(np.mean(np.square(flat)))),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "p01": float(np.quantile(flat, 0.01)),
        "p50": float(np.quantile(flat, 0.50)),
        "p99": float(np.quantile(flat, 0.99)),
        "dynamic_range": float(flat.max() - flat.min()),
        "roughness_b": float(np.sqrt(np.mean(np.square(diff_b)))),
        "roughness_z": float(np.sqrt(np.mean(np.square(diff_z)))),
        "tv_b": float(np.abs(diff_b).mean()),
        "tv_z": float(np.abs(diff_z).mean()),
        "curvature_b": float(np.abs(curv_b).mean()),
        "curvature_z": float(np.abs(curv_z).mean()),
        "local_max_gradient": float(max(np.abs(diff_b).max(), np.abs(diff_z).max())),
        "local_max_curvature": float(max(np.abs(curv_b).max(), np.abs(curv_z).max())),
        "total_variation": float(np.abs(diff_b).sum() + np.abs(diff_z).sum()),
    }
    if jump_threshold is not None:
        all_diffs = np.concatenate([np.abs(diff_b).reshape(-1), np.abs(diff_z).reshape(-1)])
        result["finite_difference_mean"] = float(all_diffs.mean())
        result["finite_difference_p90"] = float(np.quantile(all_diffs, 0.90))
        result["finite_difference_max"] = float(all_diffs.max())
        result["argmax_jump_share"] = float((all_diffs > float(jump_threshold)).mean())
    return result


def classify_diagnosis(
    *,
    production: Mapping[int, Mapping[int, Mapping[str, float]]],
    pure: Mapping[int, Mapping[int, Mapping[str, float]]],
    memorization: Mapping[str, Mapping[str, float]],
    complexity: Mapping[int, Mapping[str, Mapping[str, float]]],
    fit_tolerance: float = 0.05,
    memorization_tolerance: float = 0.01,
) -> Dict[str, Any]:
    """Apply the pre-registered A-G decision tree to completed diagnostics."""
    max_budget = max(next(iter(production.values())))
    prod = {cycle: production[cycle][max_budget] for cycle in production}
    pure_best = {cycle: pure[cycle][max_budget] for cycle in pure}
    prod_controlled = all(
        value["validation_combined_normalized_rms"] <= fit_tolerance
        for value in prod.values()
    )
    if prod_controlled:
        primary = "A"
        reason = "Longer production-objective solves reach the fixed fit tolerance."
    else:
        pure_controlled = all(
            value["validation_combined_normalized_rms"] <= fit_tolerance
            for value in pure_best.values()
        )
        if pure_controlled:
            primary = "B"
            reason = "Pure target regression fits but the production objective does not."
        elif memorization.get("1_batch", {}).get(
            "train_combined_normalized_rms", float("inf")
        ) > memorization_tolerance:
            primary = "C"
            reason = "One-batch pure regression cannot memorize the fixed target."
        elif pure_best[max(pure_best)]["train_combined_normalized_rms"] > fit_tolerance:
            primary = "D"
            reason = "Small subsets memorize but full-bank pure regression remains inaccurate."
        elif any(
            value["train_combined_normalized_rms"] <= fit_tolerance
            and value["validation_combined_normalized_rms"] > fit_tolerance
            for value in pure_best.values()
        ):
            primary = "E"
            reason = "Train fit is controlled while validation fit is not."
        else:
            first = min(complexity)
            last = max(complexity)
            keys = ("curvature_b", "curvature_z", "total_variation")
            escalated = any(
                complexity[last]["p0_value_target"][key]
                > 1.5 * max(complexity[first]["p0_value_target"][key], 1e-12)
                or complexity[last]["pi_value_target"][key]
                > 1.5 * max(complexity[first]["pi_value_target"][key], 1e-12)
                for key in keys
            )
            if escalated:
                primary = "F"
                reason = "Later Bellman targets are materially rougher or more curved."
            else:
                primary = "G"
                reason = "The completed evidence does not uniquely identify A-F."
    labels = {
        "A": "Under-trained inner solve",
        "B": "Production-objective conflict",
        "C": "Optimizer / implementation failure",
        "D": "Capacity / projection limitation",
        "E": "Coverage / generalization",
        "F": "Target-complexity escalation",
        "G": "Inconclusive",
    }
    return {"primary": primary, "label": labels[primary], "reason": reason}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
