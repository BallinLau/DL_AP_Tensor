"""Paired fixed-teacher BP refit with a frozen versus trainable policy encoder."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from evaluation.bp_diagnostics import (  # noqa: E402
    _checkpoint_economic_config,
    build_frozen_transition_data,
)
from evaluation.full_run_diagnostics import model_state_hash  # noqa: E402
from evaluation.grids import ReferenceFirmState, load_reference_state  # noqa: E402
from experiments.evaluate_bp_teacher_fit import (  # noqa: E402
    B_BIN_EDGES,
    BRANCH_ORDER,
    StateBank,
    Z_BIN_EDGES,
    _branch_specs,
    _checkpoint_payload,
    _state_rows,
    build_on_distribution_bank,
    build_binned_summary,
    build_summary,
)
from experiments.evaluate_p_teacher_drift import _losses  # noqa: E402
from training.bp_grid_teacher import BPGridTeacher  # noqa: E402
from training.episode import Episode  # noqa: E402


ARM_PREFIXES = {
    "frozen_encoder": ("bp0_head.", "bpi_head."),
    "trainable_encoder": ("policy_encoder.", "bp0_head.", "bpi_head."),
}
REQUIRED_CACHE_FIELDS = {
    "parent",
    "bp0_target",
    "bpi_target",
    "mix_target",
    "bp0_confidence",
    "bpi_confidence",
    "mix_confidence",
    "mix_sample_weight",
    "eta_next_active",
    "teacher_snapshot_hash",
}
DEFAULT_RECORD_STEPS = (0, 50, 100, 200, 500)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unavailable"


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def cache_hash(cache: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for batch_index, item in enumerate(cache):
        digest.update(str(batch_index).encode())
        for key in sorted(item):
            value = item[key]
            digest.update(key.encode())
            if torch.is_tensor(value):
                digest.update(tensor_hash(value).encode())
            elif value is not None:
                digest.update(repr(value).encode())
    return digest.hexdigest()


def state_dict_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor_hash(tensor).encode())
    return digest.hexdigest()


def state_subset_hash(model: torch.nn.Module, prefixes: Sequence[str], *, invert: bool = False) -> str:
    state = {
        name: value
        for name, value in model.state_dict().items()
        if (name.startswith(tuple(prefixes))) != invert
    }
    return state_dict_hash(state)


def validate_cache(cache: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(cache, list) or not cache:
        raise ValueError(f"{label} must be a non-empty list of cache batches")
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(cache):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label}[{index}] must be a mapping")
        missing = sorted(REQUIRED_CACHE_FIELDS - set(raw))
        if missing:
            raise ValueError(f"{label}[{index}] is missing required fields: {missing}")
        item = dict(raw)
        parent = item["parent"]
        if not torch.is_tensor(parent) or parent.ndim != 2 or parent.shape[1] < 7:
            raise ValueError(f"{label}[{index}].parent must have shape (N, >=7)")
        n_rows = int(parent.shape[0])
        for key in REQUIRED_CACHE_FIELDS - {"parent", "teacher_snapshot_hash"}:
            value = item[key]
            if not torch.is_tensor(value) or int(value.shape[0]) != n_rows:
                raise ValueError(
                    f"{label}[{index}].{key} must be a tensor with first dimension {n_rows}"
                )
        if "eta_current" not in item:
            item["eta_current"] = parent[:, 2:3].detach().clone()
        teacher_hash = item["teacher_snapshot_hash"]
        if not isinstance(teacher_hash, str) or not teacher_hash:
            raise ValueError(
                f"{label}[{index}].teacher_snapshot_hash must be a non-empty string"
            )
        validated.append(item)
    return validated


def load_cache(path: Path, label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    manifest: dict[str, Any] = {}
    if isinstance(payload, Mapping) and "cache" in payload:
        cache = payload["cache"]
        manifest = dict(payload.get("manifest") or {})
    else:
        cache = payload
    return validate_cache(cache, label), manifest


def save_cache(path: Path, cache: list[dict[str, Any]], manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cache": cache, "manifest": dict(manifest)}, path)


def validate_cache_teacher(
    train_cache: Sequence[Mapping[str, Any]],
    val_cache: Sequence[Mapping[str, Any]],
    expected_hash: str,
) -> None:
    observed = {
        str(item["teacher_snapshot_hash"])
        for item in [*train_cache, *val_cache]
    }
    if observed != {expected_hash}:
        raise ValueError(
            "BP cache teacher provenance mismatch: "
            f"expected {expected_hash}, observed {sorted(observed)}"
        )


def clone_paired_students(
    baseline: torch.nn.Module, device: torch.device
) -> dict[str, torch.nn.Module]:
    students = {
        arm: copy.deepcopy(baseline).to(device)
        for arm in ARM_PREFIXES
    }
    if model_state_hash(students["frozen_encoder"]) != model_state_hash(baseline):
        raise RuntimeError("frozen_encoder step-0 state does not match baseline")
    if model_state_hash(students["trainable_encoder"]) != model_state_hash(baseline):
        raise RuntimeError("trainable_encoder step-0 state does not match baseline")
    left_ptrs = {parameter.data_ptr() for parameter in students["frozen_encoder"].parameters()}
    right_ptrs = {parameter.data_ptr() for parameter in students["trainable_encoder"].parameters()}
    teacher_ptrs = {parameter.data_ptr() for parameter in baseline.parameters()}
    if left_ptrs & right_ptrs or left_ptrs & teacher_ptrs or right_ptrs & teacher_ptrs:
        raise RuntimeError("paired students or baseline share mutable parameter storage")
    return students


def configure_trainable_parameters(model: torch.nn.Module, arm: str) -> list[str]:
    if arm not in ARM_PREFIXES:
        raise ValueError(f"unknown ablation arm: {arm}")
    allowed = ARM_PREFIXES[arm]
    trainable = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith(allowed)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    if not trainable:
        raise RuntimeError(f"{arm} resolved no trainable parameters")
    unexpected = [name for name in trainable if not name.startswith(allowed)]
    if unexpected:
        raise RuntimeError(f"{arm} optimizer whitelist violation: {unexpected}")
    return trainable


def optimizer_parameter_names(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> list[str]:
    lookup = {id(parameter): name for name, parameter in model.named_parameters()}
    names = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in lookup:
                raise RuntimeError("optimizer contains a parameter outside the student model")
            names.append(lookup[id(parameter)])
    return names


def make_loss_adapter(model: torch.nn.Module, hyperparams: Any, device: torch.device) -> Episode:
    adapter = Episode.__new__(Episode)
    adapter.models = {"policy_value": model}
    adapter.hyperparams = hyperparams
    adapter.device = device
    return adapter


def active_cache_indices(cache: Sequence[Mapping[str, Any]]) -> list[int]:
    return [
        index
        for index, item in enumerate(cache)
        if Episode._bp_cache_item_has_active_supervision(dict(item))
    ]


def build_step_schedule(cache: Sequence[Mapping[str, Any]], steps: int, seed: int) -> list[int]:
    active = active_cache_indices(cache)
    if not active:
        raise ValueError("training cache has no active BP supervision")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    schedule: list[int] = []
    successful = 0
    while successful < int(steps):
        order = torch.randperm(len(cache), generator=generator).tolist()
        for index in order:
            schedule.append(index)
            successful += int(index in active)
            if successful >= int(steps):
                break
    return schedule


def _policy_predictions(adapter: Episode, item: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    model = adapter.models["policy_value"]
    parent = item["parent"].to(adapter.device)
    output = model(parent)
    bp0_logit, bpi_logit = model.forward_policy_logits(parent)
    bp0 = torch.sigmoid(bp0_logit)
    bpi = torch.sigmoid(bpi_logit)
    bp_mix = adapter._mixed_policy_conditional_bp(
        output,
        bp0,
        bpi,
        parent[:, 0:1],
        fallback_bar_i=getattr(output, "bar_i_cond", getattr(output, "bar_i")),
    )
    return {"bp0": bp0, "bpi": bpi, "mix": bp_mix}


def evaluate_cache(
    adapter: Episode,
    cache: Sequence[Mapping[str, Any]],
    *,
    split: str,
    arm: str,
    step: int,
    module_diagnostics: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    model = adapter.models["policy_value"]
    was_training = model.training
    loss_item: dict[str, Any]
    branch_values: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {key: [] for key in ("pred", "target", "confidence", "active")}
        for name in ("bp0", "bpi", "mix")
    }
    try:
        model.eval()
        with torch.no_grad():
            merged = {
                key: torch.cat([item[key].detach().cpu() for item in cache], dim=0)
                for key in REQUIRED_CACHE_FIELDS
                if key != "teacher_snapshot_hash"
            }
            merged["teacher_snapshot_hash"] = cache[0]["teacher_snapshot_hash"]
            _, loss_item = adapter._compute_bp_cache_loss(merged)
            for item in cache:
                predictions = _policy_predictions(adapter, item)
                active = item.get("eta_current", item["parent"][:, 2:3]).detach().cpu() > 0.5
                for branch, target_key, confidence_key in (
                    ("bp0", "bp0_target", "bp0_confidence"),
                    ("bpi", "bpi_target", "bpi_confidence"),
                    ("mix", "mix_target", "mix_confidence"),
                ):
                    confidence = item[confidence_key].detach().cpu().clamp_min(0.0)
                    if branch == "mix":
                        confidence = confidence * item["mix_sample_weight"].detach().cpu().clamp(0.0, 1.0)
                    branch_values[branch]["pred"].append(predictions[branch].detach().cpu())
                    branch_values[branch]["target"].append(item[target_key].detach().cpu())
                    branch_values[branch]["confidence"].append(confidence)
                    branch_values[branch]["active"].append(active)
    finally:
        model.train(was_training)

    rows = []
    total_loss = float(loss_item["total"])
    loss_by_branch = {
        branch: float(loss_item[f"{branch}_loss"])
        for branch in ("bp0", "bpi", "mix")
    }
    for branch, values in branch_values.items():
        pred = torch.cat(values["pred"]).reshape(-1)
        target = torch.cat(values["target"]).reshape(-1)
        confidence = torch.cat(values["confidence"]).reshape(-1)
        active = torch.cat(values["active"]).reshape(-1)
        valid = active & (confidence > 0) & torch.isfinite(pred) & torch.isfinite(target)
        error = pred[valid] - target[valid]
        weighted = confidence[valid]
        weighted_denom = float(weighted.sum().item())
        row = {
            "arm": arm,
            "step": int(step),
            "split": split,
            "branch": branch,
            "total_loss": total_loss,
            "branch_loss": loss_by_branch[branch],
            "n_rows": int(pred.numel()),
            "active_n": int(valid.sum().item()),
            "unweighted_mae": float(error.abs().mean().item()) if error.numel() else float("nan"),
            "signed_bias": float(error.mean().item()) if error.numel() else float("nan"),
            "p90_absolute_action_gap": (
                float(torch.quantile(error.abs(), 0.90).item()) if error.numel() else float("nan")
            ),
            "confidence_weighted_mae": (
                float((error.abs() * weighted).sum().item() / weighted_denom)
                if weighted_denom > 0.0 else float("nan")
            ),
            "confidence_weight_sum": weighted_denom,
            **dict(module_diagnostics or {}),
        }
        rows.append(row)
    return rows


def _module_norms(model: torch.nn.Module, *, gradients: bool) -> dict[str, float]:
    result = {}
    for module_name in ("policy_encoder", "bp0_head", "bpi_head"):
        module = getattr(model, module_name)
        total = 0.0
        for parameter in module.parameters():
            value = parameter.grad if gradients else parameter.detach()
            if value is not None:
                total += float(value.detach().pow(2).sum().item())
        result[f"{module_name}_{'grad' if gradients else 'param'}_norm"] = math.sqrt(total)
    return result


def _module_changes(
    model: torch.nn.Module, baseline_state: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    result = {}
    current = model.state_dict()
    for module_name in ("policy_encoder", "bp0_head", "bpi_head"):
        prefix = f"{module_name}."
        total = sum(
            float((current[name].detach().cpu() - baseline_state[name]).pow(2).sum().item())
            for name in current
            if name.startswith(prefix)
        )
        result[f"{module_name}_parameter_change_norm"] = math.sqrt(total)
    return result


def _clip_gradients(parameters: Sequence[torch.nn.Parameter], max_norm: float) -> tuple[float, float]:
    raw_sq = sum(
        float(parameter.grad.detach().pow(2).sum().item())
        for parameter in parameters
        if parameter.grad is not None
    )
    raw = math.sqrt(raw_sq)
    torch.nn.utils.clip_grad_norm_(parameters, float(max_norm))
    clipped_sq = sum(
        float(parameter.grad.detach().pow(2).sum().item())
        for parameter in parameters
        if parameter.grad is not None
    )
    return raw, math.sqrt(clipped_sq)


@dataclass
class ArmResult:
    model: torch.nn.Module
    metrics: list[dict[str, Any]]
    checks: dict[str, Any]
    best_step: int


def save_ablation_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    loaded: Any,
    source_checkpoint: Path,
    arm: str,
    step: int,
    validation_score: float,
    episode: int = 2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "models": {
                "policy_value": {
                    name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
                },
                "sdf_fc1": {
                    name: tensor.detach().cpu()
                    for name, tensor in loaded.models["sdf_fc1"].state_dict().items()
                },
            },
            "hyperparams": vars(loaded.hyperparams),
            "config_snapshot": loaded.economic_config.to_dict(),
            "policy_value_model_spec": loaded.metadata["policy_value_model_spec"],
            "value_parameterization": loaded.metadata["value_parameterization"]["checkpoint"],
            "episode": int(episode),
            "stage": "bp_encoder_freeze_ablation",
            "ablation": {
                "arm": arm,
                "source_checkpoint": str(source_checkpoint),
                "successful_optimizer_steps": int(step),
                "validation_score": float(validation_score),
                "trainable_prefixes": list(ARM_PREFIXES[arm]),
                "not_formal_training_checkpoint": True,
            },
        },
        path,
    )


def run_arm(
    *,
    arm: str,
    model: torch.nn.Module,
    hyperparams: Any,
    train_cache: list[dict[str, Any]],
    val_cache: list[dict[str, Any]],
    schedule: Sequence[int],
    record_steps: Sequence[int],
    learning_rate: float,
    weight_decay: float,
    output_dir: Path,
    loaded: Any,
    source_checkpoint: Path,
    teacher_hash_before: str,
    teacher_model: torch.nn.Module,
    train_cache_hash_before: str,
    val_cache_hash_before: str,
    episode: int = 2,
) -> ArmResult:
    trainable_names = configure_trainable_parameters(model, arm)
    named_parameters = dict(model.named_parameters())
    parameters = [named_parameters[name] for name in trainable_names]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    optimizer_names = optimizer_parameter_names(model, optimizer)
    if optimizer_names != trainable_names:
        raise RuntimeError(f"{arm} optimizer parameters do not match the whitelist")
    adapter = make_loss_adapter(model, hyperparams, next(model.parameters()).device)
    baseline_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    nontrainable_hash_before = state_subset_hash(model, ARM_PREFIXES[arm], invert=True)
    buffers_before = {name: tensor.detach().cpu().clone() for name, tensor in model.named_buffers()}
    records: list[dict[str, Any]] = []
    record_set = set(int(step) for step in record_steps)
    best_score = float("inf")
    best_step = 0
    attempted = successful = empty = nonfinite = 0
    gradient_seen = {name: False for name in ("policy_encoder", "bp0_head", "bpi_head")}
    last_diag: dict[str, float] = {}
    hard_threshold = float(getattr(hyperparams, "pv_grad_hard_threshold", 1000.0))
    clip_norm = float(getattr(hyperparams, "bp_distill_grad_clip_norm", 10.0))

    def record(step: int) -> None:
        nonlocal best_score, best_step
        changes = _module_changes(model, baseline_state)
        diagnostics = {
            **last_diag,
            **changes,
            "attempted_optimizer_steps": attempted,
            "successful_optimizer_steps": successful,
            "empty_active_batches": empty,
            "nonfinite_failures": nonfinite,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        records.extend(evaluate_cache(adapter, train_cache, split="train", arm=arm, step=step, module_diagnostics=diagnostics))
        val_rows = evaluate_cache(adapter, val_cache, split="validation", arm=arm, step=step, module_diagnostics=diagnostics)
        records.extend(val_rows)
        score = max(
            float(row["confidence_weighted_mae"])
            for row in val_rows
            if math.isfinite(float(row["confidence_weighted_mae"]))
        )
        if score < best_score:
            best_score = score
            best_step = int(step)
            save_ablation_checkpoint(
                output_dir / arm / "best.pt",
                model=model,
                loaded=loaded,
                source_checkpoint=source_checkpoint,
                arm=arm,
                step=step,
                validation_score=score,
                episode=episode,
            )

    record(0)
    model.train()
    for batch_index in schedule:
        item = train_cache[int(batch_index)]
        if not Episode._bp_cache_item_has_active_supervision(item):
            empty += 1
            continue
        attempted += 1
        optimizer.zero_grad(set_to_none=True)
        total, _losses = adapter._compute_bp_cache_loss(item)
        if not torch.isfinite(total):
            nonfinite += 1
            raise FloatingPointError(f"{arm}: non-finite BP loss at attempted step {attempted}")
        total.backward()
        for module_name in gradient_seen:
            module = getattr(model, module_name)
            gradient_seen[module_name] |= any(
                parameter.grad is not None and bool(torch.any(parameter.grad != 0))
                for parameter in module.parameters()
            )
        raw_module_norms = _module_norms(model, gradients=True)
        raw_norm, clipped_norm = _clip_gradients(parameters, clip_norm)
        clipped_module_norms = _module_norms(model, gradients=True)
        if not math.isfinite(raw_norm) or raw_norm > hard_threshold:
            nonfinite += int(not math.isfinite(raw_norm))
            raise FloatingPointError(
                f"{arm}: invalid/hard gradient norm {raw_norm} at attempted step {attempted}"
            )
        optimizer.step()
        successful += 1
        last_diag = {
            "raw_gradient_norm": raw_norm,
            "clipped_gradient_norm": clipped_norm,
            "gradient_clipping_triggered": bool(clipped_norm + 1e-12 < raw_norm),
            **{f"raw_{key}": value for key, value in raw_module_norms.items()},
            **{f"clipped_{key}": value for key, value in clipped_module_norms.items()},
        }
        if successful in record_set:
            record(successful)
    expected_successful = sum(
        int(Episode._bp_cache_item_has_active_supervision(train_cache[index]))
        for index in schedule
    )
    if successful != expected_successful:
        raise RuntimeError(
            f"{arm}: expected {expected_successful} successful steps, got {successful}"
        )
    if successful not in record_set:
        record(successful)

    changes = _module_changes(model, baseline_state)
    expected_encoder_change = arm == "trainable_encoder"
    checks = {
        "arm": arm,
        "trainable_parameter_names": trainable_names,
        "optimizer_parameter_names": optimizer_names,
        "gradient_seen": gradient_seen,
        "module_changes": changes,
        "policy_encoder_change_expected": expected_encoder_change,
        "nontrainable_state_hash_before": nontrainable_hash_before,
        "nontrainable_state_hash_after": state_subset_hash(model, ARM_PREFIXES[arm], invert=True),
        "buffers_unchanged": all(
            torch.equal(buffers_before[name], value.detach().cpu())
            for name, value in model.named_buffers()
        ),
        "teacher_hash_before": teacher_hash_before,
        "teacher_hash_after": model_state_hash(teacher_model),
        "train_cache_hash_before": train_cache_hash_before,
        "train_cache_hash_after": cache_hash(train_cache),
        "val_cache_hash_before": val_cache_hash_before,
        "val_cache_hash_after": cache_hash(val_cache),
        "attempted_optimizer_steps": attempted,
        "successful_optimizer_steps": successful,
        "empty_active_batches": empty,
        "nonfinite_failures": nonfinite,
        "best_step": best_step,
        "best_validation_score": best_score,
    }
    if checks["nontrainable_state_hash_before"] != checks["nontrainable_state_hash_after"]:
        raise RuntimeError(f"{arm}: non-whitelisted model state changed")
    if not checks["buffers_unchanged"]:
        raise RuntimeError(f"{arm}: model buffers changed")
    if checks["teacher_hash_before"] != checks["teacher_hash_after"]:
        raise RuntimeError(f"{arm}: frozen teacher changed")
    if checks["train_cache_hash_before"] != checks["train_cache_hash_after"] or checks["val_cache_hash_before"] != checks["val_cache_hash_after"]:
        raise RuntimeError(f"{arm}: fixed cache changed")
    if changes["bp0_head_parameter_change_norm"] <= 0.0 or changes["bpi_head_parameter_change_norm"] <= 0.0:
        raise RuntimeError(f"{arm}: both BP heads must update")
    encoder_change = changes["policy_encoder_parameter_change_norm"]
    if expected_encoder_change and (not gradient_seen["policy_encoder"] or encoder_change <= 0.0):
        raise RuntimeError("trainable_encoder did not receive gradient and update")
    if not expected_encoder_change and encoder_change != 0.0:
        raise RuntimeError("frozen_encoder changed policy_encoder")
    save_ablation_checkpoint(
        output_dir / arm / "last.pt",
        model=model,
        loaded=loaded,
        source_checkpoint=source_checkpoint,
        arm=arm,
        step=successful,
        validation_score=max(
            row["confidence_weighted_mae"]
            for row in records
            if row["split"] == "validation" and row["step"] == successful
        ),
        episode=episode,
    )
    return ArmResult(model=model, metrics=records, checks=checks, best_step=best_step)


def resolve_record_steps(steps: int, requested: Sequence[int]) -> list[int]:
    if steps < 0:
        raise ValueError("--steps must be non-negative")
    return sorted({0, int(steps), *(int(value) for value in requested if 0 <= int(value) <= steps)})


def resolve_checkpoint(
    run_root: Path,
    episode: int,
    explicit: Path | None,
    explicit_combined: Path | None = None,
) -> tuple[Path, Path]:
    stage = explicit.expanduser().resolve() if explicit is not None else (
        run_root / "episode_diagnostics" / f"ep_{episode:03d}" / "post_bp.pt"
    ).resolve()
    if not stage.is_file():
        raise FileNotFoundError(f"EP{episode} post_bp checkpoint not found: {stage}")
    if explicit_combined is not None:
        combined = explicit_combined.expanduser().resolve()
        if not combined.is_file():
            raise FileNotFoundError(f"combined checkpoint not found: {combined}")
        return stage, combined
    combined_candidates = [
        run_root / "checkpoints_analysis" / f"ep{episode}_combined.pt",
        run_root / "checkpoints" / f"ep{episode}_combined.pt",
    ]
    found = [path.resolve() for path in combined_candidates if path.is_file()]
    if len(found) != 1:
        raise RuntimeError(
            f"expected exactly one EP{episode} combined checkpoint, found {found}; "
            "pass a run root with unambiguous metadata"
        )
    return stage, found[0]


def validate_stage_checkpoint_metadata(
    path: Path,
    *,
    episode: int,
    expected_stage: str,
    label: str,
) -> None:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a checkpoint mapping: {path}")
    observed_episode = payload.get("episode")
    observed_stage = payload.get("stage")
    if observed_episode != int(episode) or observed_stage != expected_stage:
        raise ValueError(
            f"{label} metadata mismatch: expected episode={episode}, "
            f"stage={expected_stage!r}; observed episode={observed_episode!r}, "
            f"stage={observed_stage!r}"
        )
    policy_state = payload.get("models", {}).get("policy_value")
    if not isinstance(policy_state, Mapping) or not policy_state:
        raise ValueError(f"{label} is missing models.policy_value: {path}")


def _batch_split(batches: list[dict[str, Any]], fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(batches) < 2:
        raise ValueError("cache reconstruction requires at least two production batches")
    fraction = min(max(float(fraction), 0.0), 0.5)
    n_val = max(1, int(round(len(batches) * fraction)))
    n_val = min(n_val, len(batches) - 1)
    return batches[:-n_val], batches[-n_val:]


def reconstruct_caches(
    *,
    firm_data: Path,
    teacher_model: torch.nn.Module,
    loaded: Any,
    episode: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    frame = pd.read_pickle(firm_data)
    builder = Episode.__new__(Episode)
    builder.device = device
    builder.hyperparams = loaded.hyperparams
    builder.models = loaded.models
    builder.config = Config
    builder.episode_id = int(episode)
    p0_loss, pi_loss = _losses(loaded)
    builder.loss_fns = {"p0": p0_loss, "pi": pi_loss}
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        batches = builder._create_firm_batches_from_df(
            frame, batch_size=int(batch_size), n_branches=2, eta_resample=False
        )
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    train_batches, val_batches = _batch_split(
        batches, float(getattr(loaded.hyperparams, "pv_target_grid_val_fraction", 0.10))
    )
    with _checkpoint_economic_config(loaded.economic_config):
        train_cache = builder._build_bp_target_cache(train_batches, teacher_model)
        val_cache = builder._build_bp_target_cache(val_batches, teacher_model)
    manifest = {
        "cache_origin": "reconstructed",
        "historical_training_cache_exact": False,
        "reason": "rebuilt once from saved EP2 parent/child dataframe with production builder",
        "firm_data": str(firm_data),
        "batch_size": int(batch_size),
        "split_rule": "production batch-tail pv_target_grid_val_fraction",
        "split_fraction": float(getattr(loaded.hyperparams, "pv_target_grid_val_fraction", 0.10)),
        "seed": int(seed),
    }
    return validate_cache(train_cache, "reconstructed train cache"), validate_cache(val_cache, "reconstructed val cache"), manifest


def discover_saved_cache_pair(run_root: Path, episode: int) -> tuple[Path, Path] | None:
    pairs = [
        (
            run_root / "caches" / f"ep{episode}_bp_train.pt",
            run_root / "caches" / f"ep{episode}_bp_validation.pt",
        ),
        (
            run_root / "bp_cache" / f"ep{episode}_train.pt",
            run_root / "bp_cache" / f"ep{episode}_validation.pt",
        ),
        (
            run_root / "caches" / "bp" / f"ep{episode}_train.pt",
            run_root / "caches" / "bp" / f"ep{episode}_validation.pt",
        ),
    ]
    found = [(train.resolve(), val.resolve()) for train, val in pairs if train.is_file() and val.is_file()]
    if len(found) > 1:
        raise RuntimeError(f"ambiguous saved BP caches: {found}")
    return found[0] if found else None


def prepare_caches(
    *,
    args: argparse.Namespace,
    teacher_model: torch.nn.Module,
    loaded: Any,
    output_dir: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if args.train_cache is None and args.cache_mode == "auto":
        found = discover_saved_cache_pair(args.source_run_root, args.episode)
        if found is not None:
            args.train_cache, args.val_cache = found
    if (args.train_cache is None) != (args.val_cache is None):
        raise ValueError("--train-cache and --val-cache must be provided together")
    if args.train_cache is not None:
        train, train_meta = load_cache(args.train_cache.resolve(), "train cache")
        val, val_meta = load_cache(args.val_cache.resolve(), "validation cache")
        manifest = {
            "cache_origin": "saved",
            "train_cache": str(args.train_cache.resolve()),
            "val_cache": str(args.val_cache.resolve()),
            "train_manifest": train_meta,
            "val_manifest": val_meta,
        }
    else:
        if args.cache_mode == "saved":
            raise ValueError("CACHE_MODE=saved requires both --train-cache and --val-cache")
        if args.firm_data is None:
            raise ValueError(
                "cache reconstruction requires --firm-data; evaluator Primary states are not a training-cache substitute"
            )
        firm_data = args.firm_data.expanduser().resolve()
        if not firm_data.is_file():
            raise FileNotFoundError(f"cache reconstruction firm data not found: {firm_data}")
        train, val, manifest = reconstruct_caches(
            firm_data=firm_data,
            teacher_model=teacher_model,
            loaded=loaded,
            episode=args.episode,
            batch_size=args.resolved_batch_size,
            seed=args.seed,
            device=device,
        )
        save_cache(output_dir / "shared_cache" / "train.pt", train, manifest)
        save_cache(output_dir / "shared_cache" / "validation.pt", val, manifest)
    manifest.update({
        "train_hash": cache_hash(train),
        "validation_hash": cache_hash(val),
        "train_batches": len(train),
        "validation_batches": len(val),
        "train_rows": int(sum(item["parent"].shape[0] for item in train)),
        "validation_rows": int(sum(item["parent"].shape[0] for item in val)),
    })
    validate_cache_teacher(
        train,
        val,
        Episode._state_dict_hash(teacher_model),
    )
    return train, val, manifest


def _fixed_external_bank(
    baseline_dir: Path,
    *,
    device: torch.device,
) -> tuple[StateBank, pd.DataFrame, dict[str, Any], Path, Path]:
    metadata_path = baseline_dir / "metadata.json"
    state_path = baseline_dir / "tables" / "bp_fit_state_level_on_distribution.csv"
    if not metadata_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(
            f"baseline evaluator must contain metadata.json and {state_path.relative_to(baseline_dir)}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    baseline = pd.read_csv(state_path)
    bank_meta = metadata.get("banks", {}).get("on_distribution")
    if not isinstance(bank_meta, Mapping):
        raise ValueError("baseline metadata is missing banks.on_distribution")
    firm = Path(bank_meta["source_artifact"]).expanduser().resolve()
    macro_raw = bank_meta.get("macro_artifact")
    macro = Path(macro_raw).expanduser().resolve() if macro_raw else None
    if not firm.is_file() or macro is None or not macro.is_file():
        raise FileNotFoundError(f"baseline on-distribution artifacts not found: {firm}, {macro}")
    frame, _ondist_reference = load_reference_state(firm, macro_path=macro)
    reference_payload = metadata.get("reference_state")
    if not isinstance(reference_payload, Mapping):
        raise ValueError("baseline metadata is missing reference_state")
    reference = ReferenceFirmState(**{
        field: reference_payload[field]
        for field in ReferenceFirmState.__dataclass_fields__
    })
    bank = build_on_distribution_bank(
        frame,
        reference=reference,
        source=firm,
        macro_source=macro,
        max_states=int(bank_meta["sampled_states"]),
        seed=int(bank_meta["sample_seed"]),
        device=device,
    )
    saved_ids = (
        baseline[["bank_row", "source_index"]]
        .drop_duplicates()
        .sort_values("bank_row")
        .reset_index(drop=True)
    )
    rebuilt_ids = bank.context[["bank_row", "source_index"]].reset_index(drop=True)
    if not saved_ids.equals(rebuilt_ids):
        raise RuntimeError("rebuilt on-distribution state bank does not match saved source_index order")
    bank_identity = hashlib.sha256(
        pd.util.hash_pandas_object(saved_ids, index=False).to_numpy().tobytes()
    ).hexdigest()
    metadata["ablation_rebuilt_bank_identity_sha256"] = bank_identity
    return bank, baseline, metadata, firm, macro


def evaluate_external_fixed_teacher(
    *,
    label: str,
    student: torch.nn.Module,
    teacher_model: torch.nn.Module,
    sdf_fc1: torch.nn.Module,
    loaded: Any,
    bank: StateBank,
    n_child_shocks: int,
    shock_seed: int,
    teacher_margin_tol: float,
    large_gap_threshold: float,
    high_regret_relative_threshold: float,
    weak_margin_relative_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    specs = _branch_specs(teacher_model, bank.reference)
    teacher = BPGridTeacher.from_hyperparams(
        teacher_model, *_losses(loaded), loaded.hyperparams
    )
    n_rows = int(bank.base_states.shape[0])
    eta_values = bank.base_states[:, 2].detach().cpu().numpy()
    parts = []
    transition_meta = []
    for eta_value in sorted(np.unique(eta_values).tolist()):
        positions = np.flatnonzero(np.isclose(eta_values, eta_value))
        index = torch.as_tensor(positions, dtype=torch.long, device=bank.base_states.device)
        transition_states = bank.base_states.index_select(0, index).clone()
        transition_states[:, 3] = float(bank.reference.i_mid)
        transition = build_frozen_transition_data(
            sdf_fc1,
            transition_states,
            bank.reference,
            loaded.hyperparams,
            loaded.economic_config,
            n_child_shocks=int(n_child_shocks),
            shock_seed=int(shock_seed),
            shock_bank_max_child_shocks=int(n_child_shocks),
            hatc_cal_values=bank.hatc_cal.index_select(0, index),
            lnk_cal_values=bank.lnk_cal.index_select(0, index),
        )
        states = []
        predictions = []
        teacher_outputs = []
        with torch.no_grad():
            for branch_label in BRANCH_ORDER:
                branch, i_value = specs[branch_label]
                state = bank.base_states.index_select(0, index).clone()
                state[:, 3] = float(i_value)
                states.append(state)
                student_out = student(state)
                predictions.append(student_out.bp0 if branch == "p0" else student_out.bpI)
                teacher_outputs.append(teacher_model(state))
        child_count = int(transition.stacked_children().shape[1])
        with _checkpoint_economic_config(loaded.economic_config), torch.no_grad():
            bundles = teacher.compute_multi_j_branches(
                states,
                transition.stacked_children(),
                transition.stacked_m_used(),
                branches=[specs[name][0] for name in BRANCH_ORDER],
                prefix_child_counts=[child_count],
                child_weights=transition.branch_weights,
                bp_preds=predictions,
            )
        transition_meta.append({"eta": float(eta_value), **transition.metadata})
        for branch_index, branch_label in enumerate(BRANCH_ORDER):
            fixed_output = teacher_outputs[branch_index]
            with torch.no_grad():
                fixed_scale = teacher_model.equity_value_scale(states[branch_index])
            part, _ = _state_rows(
                episode=2,
                checkpoint_stage=label,
                label=branch_label,
                state=states[branch_index],
                prediction=predictions[branch_index],
                phat=fixed_output.Phat,
                value_scale=fixed_scale,
                result=bundles[branch_index][child_count],
                teacher_margin_tol=float(teacher_margin_tol),
            )
            part.insert(0, "bank", "on_distribution")
            part.insert(1, "bank_row", positions)
            context = bank.context.iloc[positions].reset_index(drop=True)
            for column in context:
                if column not in part:
                    part[column] = context[column].to_numpy()
            parts.append(part)
    state_level = pd.concat(parts, ignore_index=True).sort_values(
        ["branch", "bank_row"], kind="stable"
    ).reset_index(drop=True)
    summary = build_summary(
        state_level,
        large_gap_threshold=float(large_gap_threshold),
        high_regret_relative_threshold=float(high_regret_relative_threshold),
        weak_margin_relative_threshold=float(weak_margin_relative_threshold),
    )
    shock_hashes = sorted({item["shock_bank_sha256"] for item in transition_meta})
    return state_level, summary, {
        "student_label": label,
        "n_rows": n_rows,
        "n_child_shocks": int(n_child_shocks),
        "shock_seed": int(shock_seed),
        "shock_bank_sha256": shock_hashes[0] if len(shock_hashes) == 1 else shock_hashes,
        "teacher_model_hash": model_state_hash(teacher_model),
        "mask_and_scale_model": "fixed_teacher",
    }


def run_external_evaluation(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    initial: torch.nn.Module,
    arms: Mapping[str, ArmResult],
    teacher: torch.nn.Module,
    loaded: Any,
    train_cache: Sequence[Mapping[str, Any]],
    val_cache: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    bank, baseline_rows, baseline_meta, _firm, _macro = _fixed_external_bank(
        args.baseline_diag_dir.resolve(), device=next(initial.parameters()).device
    )
    baseline_bank_meta = baseline_meta["banks"]["on_distribution"]
    expected_policy_hash = baseline_meta.get("hashes", {}).get("policy_value_before")
    expected_sdf_hash = baseline_meta.get("hashes", {}).get("sdf_fc1_before")
    actual_policy_hash = model_state_hash(initial)
    actual_sdf_hash = model_state_hash(loaded.models["sdf_fc1"])
    if expected_policy_hash is not None and actual_policy_hash != expected_policy_hash:
        raise RuntimeError(
            "initial post_bp policy checkpoint does not match baseline evaluator provenance"
        )
    if expected_sdf_hash is not None and actual_sdf_hash != expected_sdf_hash:
        raise RuntimeError(
            "SDF/FC1 checkpoint does not match baseline evaluator provenance"
        )
    expected_shock_hash = baseline_bank_meta.get("shock_bank_sha256")
    thresholds = baseline_meta.get("thresholds", {})
    for key in ("large_gap", "high_regret_relative", "weak_margin_relative"):
        if key not in thresholds:
            raise ValueError(f"baseline metadata is missing thresholds.{key}")
    external_root = output_dir / "external_eval"
    external_root.mkdir(parents=True, exist_ok=True)
    models = {"initial": initial, **{f"{arm}_last": result.model for arm, result in arms.items()}}
    summaries = []
    state_frames = {}
    metadata = {}
    for label, model in models.items():
        state, summary, meta = evaluate_external_fixed_teacher(
            label=label,
            student=model,
            teacher_model=teacher,
            sdf_fc1=loaded.models["sdf_fc1"],
            loaded=loaded,
            bank=bank,
            n_child_shocks=int(baseline_bank_meta["n_child_shocks"]),
            shock_seed=int(baseline_bank_meta["shock_seed"]),
            teacher_margin_tol=float(baseline_meta["teacher_margin_tol"]),
            large_gap_threshold=float(thresholds["large_gap"]),
            high_regret_relative_threshold=float(thresholds["high_regret_relative"]),
            weak_margin_relative_threshold=float(thresholds["weak_margin_relative"]),
        )
        if expected_shock_hash is not None and meta["shock_bank_sha256"] != expected_shock_hash:
            raise RuntimeError(
                f"{label}: evaluator shock bank hash differs from saved baseline"
            )
        state.insert(0, "student", label)
        summary.insert(0, "student", label)
        state.to_csv(external_root / f"{label}_state_level.csv", index=False)
        summary.to_csv(external_root / f"{label}_summary.csv", index=False)
        build_binned_summary(
            state, column="b", edges=B_BIN_EDGES, bin_column="b_bin"
        ).to_csv(external_root / f"{label}_by_b_bin.csv", index=False)
        build_binned_summary(
            state, column="z", edges=Z_BIN_EDGES, bin_column="z_bin"
        ).to_csv(external_root / f"{label}_by_z_bin.csv", index=False)
        state_frames[label] = state
        summaries.append(summary)
        metadata[label] = meta
    identity_columns = ("bp_pred", "bp_star", "regret", "top2_margin")
    saved_initial = baseline_rows[["branch", "bank_row", *identity_columns]].copy()
    reproduced_initial = state_frames["initial"][["branch", "bank_row", *identity_columns]].copy()
    identity = saved_initial.merge(
        reproduced_initial,
        on=["branch", "bank_row"],
        suffixes=("_saved", "_reproduced"),
        validate="one_to_one",
    )
    if len(identity) != len(saved_initial):
        raise RuntimeError("initial evaluator reproduction did not cover every saved state row")
    identity_errors = {}
    for column in identity_columns:
        saved = identity[f"{column}_saved"].to_numpy(dtype=np.float64)
        reproduced = identity[f"{column}_reproduced"].to_numpy(dtype=np.float64)
        if not np.isfinite(saved).all() or not np.isfinite(reproduced).all():
            raise RuntimeError(f"initial evaluator identity column {column} is non-finite")
        identity_errors[column] = float(np.max(np.abs(saved - reproduced)))
    if any(value > 1e-5 for value in identity_errors.values()):
        raise RuntimeError(
            f"initial fixed evaluator does not reproduce saved baseline: {identity_errors}"
        )
    metadata["initial_reproduction_max_abs_error"] = identity_errors
    metadata["provenance"] = {
        "policy_hash_expected": expected_policy_hash,
        "policy_hash_actual": actual_policy_hash,
        "sdf_hash_expected": expected_sdf_hash,
        "sdf_hash_actual": actual_sdf_hash,
        "rebuilt_bank_identity_sha256": baseline_meta[
            "ablation_rebuilt_bank_identity_sha256"
        ],
    }
    fixed_tail = baseline_rows[
        baseline_rows["primary_mask"].astype(bool)
        & (baseline_rows["bp_gap"] >= float(baseline_meta["thresholds"]["large_gap"]))
    ][["branch", "bank_row", "source_index"]].drop_duplicates()
    fixed_tail.to_csv(external_root / "fixed_initial_tail_ids.csv", index=False)
    tail_frames = []
    for label, state in state_frames.items():
        selected = state.merge(fixed_tail, on=["branch", "bank_row", "source_index"], how="inner")
        selected.to_csv(external_root / f"{label}_fixed_tail.csv", index=False)
        tail_frames.append(selected)
    pd.concat(tail_frames, ignore_index=True).to_csv(
        external_root / "fixed_tail_comparison.csv", index=False
    )
    training_ids = set()
    for item in [*train_cache, *val_cache]:
        source_index = item.get("source_index")
        if torch.is_tensor(source_index):
            training_ids.update(int(value) for value in source_index.detach().cpu().reshape(-1).tolist())
    eval_ids = set(int(value) for value in bank.context["source_index"].tolist())
    metadata["overlap"] = {
        "training_source_ids_available": bool(training_ids),
        "n_training_ids": len(training_ids),
        "n_evaluator_ids": len(eval_ids),
        "n_overlap": len(training_ids & eval_ids),
        "independent_test_set_claimed": False,
        "independence_note": (
            "source_index overlap is reported, but independence is not claimed because "
            "saved/reconstructed cache and evaluator IDs may use different namespaces"
        ),
    }
    write_json(external_root / "metadata.json", metadata)
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(external_root / "comparison.csv", index=False)
    return combined


def write_training_plot(metrics: pd.DataFrame, output_dir: Path) -> None:
    subset = metrics[(metrics["split"] == "validation") & (metrics["branch"] == "bp0")]
    if subset.empty:
        return
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for arm, group in subset.groupby("arm"):
        axis.plot(group["step"], group["total_loss"], marker="o", label=arm)
    axis.set(xlabel="successful optimizer step", ylabel="validation total loss", title="BP encoder freeze ablation")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "validation_loss.png", dpi=150)
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--baseline-diag-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--combined-checkpoint", type=Path, default=None)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--train-cache", type=Path, default=None)
    parser.add_argument("--val-cache", type=Path, default=None)
    parser.add_argument("--cache-mode", choices=("auto", "saved", "reconstruct"), default="auto")
    parser.add_argument("--firm-data", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--record-steps", type=int, nargs="+", default=list(DEFAULT_RECORD_STEPS))
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-external-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def validate_preflight_inputs(args: argparse.Namespace) -> None:
    if (args.train_cache is None) != (args.val_cache is None):
        raise ValueError("--train-cache and --val-cache must be provided together")
    if args.train_cache is not None:
        for label, path in (("train cache", args.train_cache), ("validation cache", args.val_cache)):
            if not path.expanduser().resolve().is_file():
                raise FileNotFoundError(f"{label} not found: {path}")
    else:
        saved = discover_saved_cache_pair(args.source_run_root, args.episode)
        if args.cache_mode == "saved" and saved is None:
            raise FileNotFoundError("CACHE_MODE=saved but no explicit or recognized saved cache pair exists")
        if saved is None:
            if args.firm_data is None or not args.firm_data.expanduser().resolve().is_file():
                raise FileNotFoundError(
                    "no saved cache pair was found and cache reconstruction firm data is missing"
                )
    if not args.skip_external_eval:
        required = (
            args.baseline_diag_dir / "metadata.json",
            args.baseline_diag_dir / "tables" / "bp_fit_state_level_on_distribution.csv",
        )
        missing = [str(path) for path in required if not path.expanduser().resolve().is_file()]
        if missing:
            raise FileNotFoundError(f"baseline external-evaluator inputs are missing: {missing}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_root = args.source_run_root.expanduser().resolve()
        args.source_run_root = run_root
        if not run_root.is_dir():
            raise FileNotFoundError(f"source run root not found: {run_root}")
        stage_checkpoint, combined_checkpoint = resolve_checkpoint(
            run_root, args.episode, args.checkpoint, args.combined_checkpoint
        )
        validate_stage_checkpoint_metadata(
            stage_checkpoint,
            episode=args.episode,
            expected_stage="post_bp",
            label="student initialization checkpoint",
        )
        device = torch.device(args.device)
        loaded = _checkpoint_payload(stage_checkpoint, combined_checkpoint, device)
        baseline = loaded.models["policy_value"]
        baseline.eval()
        teacher_checkpoint = (
            args.teacher_checkpoint.expanduser().resolve()
            if args.teacher_checkpoint is not None
            else (
                run_root
                / "episode_diagnostics"
                / f"ep_{args.episode:03d}"
                / "post_q_final.pt"
            ).resolve()
        )
        if not teacher_checkpoint.is_file():
            raise FileNotFoundError(
                "fixed BP cache teacher checkpoint not found; pass --teacher-checkpoint "
                f"explicitly: {teacher_checkpoint}"
            )
        validate_stage_checkpoint_metadata(
            teacher_checkpoint,
            episode=args.episode,
            expected_stage="post_q_final",
            label="fixed cache-teacher checkpoint",
        )
        teacher_loaded = _checkpoint_payload(teacher_checkpoint, combined_checkpoint, device)
        cache_teacher = teacher_loaded.models["policy_value"].eval()
        cache_teacher.requires_grad_(False)
        external_teacher = copy.deepcopy(baseline).to(device).eval()
        external_teacher.requires_grad_(False)
        resolved_lr = (
            float(args.learning_rate)
            if args.learning_rate is not None
            else float(getattr(loaded.hyperparams, "policy_lr", 0.0))
        )
        if resolved_lr <= 0.0:
            raise ValueError("learning rate is missing/non-positive; pass --learning-rate explicitly")
        resolved_batch_size = (
            int(args.batch_size)
            if args.batch_size is not None
            else int(getattr(loaded.hyperparams, "pv_batch_size", 0) or getattr(loaded.hyperparams, "batch_size", 0))
        )
        if resolved_batch_size <= 0:
            raise ValueError("batch size is missing/non-positive; pass --batch-size explicitly")
        args.resolved_batch_size = resolved_batch_size
        record_steps = resolve_record_steps(args.steps, args.record_steps)
        resolved = {
            "source_run_root": run_root,
            "baseline_diag_dir": args.baseline_diag_dir.resolve(),
            "stage_checkpoint": stage_checkpoint,
            "combined_checkpoint": combined_checkpoint,
            "cache_teacher_checkpoint": teacher_checkpoint,
            "episode": int(args.episode),
            "steps": int(args.steps),
            "record_steps": record_steps,
            "learning_rate": resolved_lr,
            "weight_decay": float(getattr(loaded.hyperparams, "policy_weight_decay", 0.0)),
            "batch_size": resolved_batch_size,
            "seed": int(args.seed),
            "device": str(device),
            "loss_space": str(getattr(loaded.hyperparams, "bp_grid_policy_loss_space", "output")),
            "logit_target_eps": float(getattr(loaded.hyperparams, "bp_grid_logit_target_eps", 1e-4)),
            "logit_huber_delta": float(getattr(loaded.hyperparams, "bp_grid_logit_huber_delta", 1.0)),
            "branch_weight": float(getattr(loaded.hyperparams, "bp_grid_policy_weight", 1.0)),
            "mix_weight": float(getattr(loaded.hyperparams, "bp_grid_mix_policy_weight", 1.0)),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_branch": git_value("branch", "--show-current"),
            "git_status_short": git_value("status", "--short"),
            "external_eval_enabled": not args.skip_external_eval,
            "student_initialization": "EP2 post_bp",
            "cache_teacher": "EP2 post_q_final",
            "external_teacher": "saved evaluator EP2 post_bp semantics",
        }
        validate_preflight_inputs(args)
        write_json(output_dir / "resolved_config.json", resolved)
        print(json.dumps(_json_value(resolved), indent=2), flush=True)
        if args.dry_run:
            print("Dry-run preflight passed; cache construction and training were not executed.")
            return

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        train_cache, val_cache, cache_manifest = prepare_caches(
            args=args,
            teacher_model=cache_teacher,
            loaded=loaded,
            output_dir=output_dir,
            device=device,
        )
        cache_manifest.update({
            "teacher_hash": model_state_hash(cache_teacher),
            "teacher_snapshot_hash_in_cache": Episode._state_dict_hash(cache_teacher),
            "teacher_source": str(teacher_checkpoint),
            "teacher_semantics": "fixed EP2 post_q_final policy_value snapshot",
        })
        write_json(output_dir / "cache_manifest.json", cache_manifest)
        train_hash = cache_hash(train_cache)
        val_hash = cache_hash(val_cache)
        teacher_hash = model_state_hash(cache_teacher)
        students = clone_paired_students(baseline, device)
        with torch.no_grad():
            first_parent = train_cache[active_cache_indices(train_cache)[0]]["parent"].to(device)
            first_parent = first_parent[:, :7]
            initial_predictions = {
                arm: torch.cat(students[arm].forward_policy(first_parent), dim=1).cpu()
                for arm in students
            }
        if not torch.equal(initial_predictions["frozen_encoder"], initial_predictions["trainable_encoder"]):
            raise RuntimeError("paired students have different step-0 predictions")
        schedule = build_step_schedule(train_cache, args.steps, args.seed)
        results = {}
        all_metrics = []
        checks = {
            "step0_state_hash_equal": True,
            "step0_prediction_equal": True,
            "schedule_sha256": tensor_hash(torch.tensor(schedule, dtype=torch.long)),
            "schedule_shared": True,
        }
        paired_cpu_rng = torch.random.get_rng_state()
        paired_cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        for arm in ("frozen_encoder", "trainable_encoder"):
            torch.random.set_rng_state(paired_cpu_rng)
            if paired_cuda_rng is not None:
                torch.cuda.set_rng_state_all(paired_cuda_rng)
            result = run_arm(
                arm=arm,
                model=students[arm],
                hyperparams=loaded.hyperparams,
                train_cache=train_cache,
                val_cache=val_cache,
                schedule=schedule,
                record_steps=record_steps,
                learning_rate=resolved_lr,
                weight_decay=float(getattr(loaded.hyperparams, "policy_weight_decay", 0.0)),
                output_dir=output_dir,
                loaded=loaded,
                source_checkpoint=stage_checkpoint,
                teacher_hash_before=teacher_hash,
                teacher_model=cache_teacher,
                train_cache_hash_before=train_hash,
                val_cache_hash_before=val_hash,
                episode=args.episode,
            )
            results[arm] = result
            all_metrics.extend(result.metrics)
            checks[arm] = result.checks
        metrics = pd.DataFrame(all_metrics)
        metrics.to_csv(output_dir / "training_metrics.csv", index=False)
        write_training_plot(metrics, output_dir)
        write_json(output_dir / "parameter_checks.json", checks)

        last_metrics = metrics[
            (metrics["split"] == "validation") & (metrics["step"] == int(args.steps))
        ].copy()
        comparison = last_metrics.pivot(
            index="branch", columns="arm", values=[
                "total_loss", "unweighted_mae", "signed_bias",
                "p90_absolute_action_gap", "confidence_weighted_mae",
            ]
        )
        comparison.columns = [f"{metric}_{arm}" for metric, arm in comparison.columns]
        comparison.reset_index().to_csv(output_dir / "comparison.csv", index=False)

        external = None
        if not args.skip_external_eval:
            external = run_external_evaluation(
                args=args,
                output_dir=output_dir,
                initial=baseline,
                arms=results,
                teacher=external_teacher,
                loaded=loaded,
                train_cache=train_cache,
                val_cache=val_cache,
            )
        summary_lines = [
            "# BP Encoder Freeze Ablation",
            "",
            f"- Source checkpoint: `{stage_checkpoint}`",
            f"- Cache origin: `{cache_manifest['cache_origin']}`",
            f"- Main comparison: both arms at `{args.steps}` successful optimizer steps",
            "- Only experimental variable: whether `policy_encoder` is trainable.",
            "- Formal training/equilibrium status: diagnostic only.",
            "",
            "## Files",
            "",
            "- `training_metrics.csv`: fixed train/validation monitoring.",
            "- `comparison.csv`: A/B last-step validation comparison.",
            "- `parameter_checks.json`: whitelist, gradient, mutation and hash checks.",
            "- `external_eval/`: fixed on-distribution evaluator outputs." if external is not None else "- External evaluator was skipped explicitly.",
        ]
        (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print(f"Wrote paired BP ablation to {output_dir}", flush=True)
    except Exception as exc:
        write_json(output_dir / "failure_report.json", {
            "stage": "bp_encoder_freeze_ablation",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "paired_comparison_valid": False,
        })
        raise


if __name__ == "__main__":
    main()
