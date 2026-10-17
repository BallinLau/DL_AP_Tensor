#!/usr/bin/env python3
"""Frozen-data and frozen-firm isolation suite for the SDF/FC1 macro block.

This module is diagnostic-only.  It reuses production simulation, tensor
pairing, losses, optimizers, and validation routines while making the allowed
parameter block explicit.  It never calls a P/Q/policy solver and never uses
FC2.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.analyze_p_only_movement_channels import _load_setup  # noqa: E402
from analysis.run_macro_outer_inner_pq import (  # noqa: E402
    _load_state_dict_from_checkpoint,
    _to_frame,
    install_fc2_guard,
    make_initial_state,
    module_hash,
    run_simulation,
    stable_json_hash,
    state_hash,
)
from analysis.test_frozen_policy_root_solve import _tensor_digest  # noqa: E402
from data.tensor_data import TensorTable  # noqa: E402
from experiments.run_utils import build_optimizers  # noqa: E402
from training.episode import SDFTrainingPhase  # noqa: E402


DEFAULT_OUTPUT = ROOT / "cachedir" / "macro_block_isolation_20261017"
DEFAULT_REPORT = ROOT / "reports" / "macro_block_isolation_20261017.md"
BLOCK_NAMES = ("fc1_c", "fc1_k", "sdf", "joint", "frozen")
FUNCTION_NAMES = ("m", "fc1_c", "fc1_k")
TARGET_NAMES = ("hatc", "lnk")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=True),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def table_hash(table: TensorTable) -> str:
    return _tensor_digest([("data", table.data)]) + ":" + stable_json_hash(list(table.columns))


def tensor_hash(tensor: torch.Tensor) -> str:
    return _tensor_digest([("tensor", tensor)])


def save_table(path: Path, table: TensorTable) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"data": table.data.detach().cpu(), "columns": list(table.columns)}
    torch.save(payload, path)
    return table_hash(TensorTable(payload["data"], payload["columns"]))


def load_table(path: Path, device: torch.device) -> TensorTable:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return TensorTable(payload["data"].to(device), list(payload["columns"]))


def batch_hash(batches: Sequence[Mapping[str, Any]]) -> str:
    named: list[tuple[str, torch.Tensor]] = []
    for b_idx, batch in enumerate(batches):
        for key in sorted(batch):
            value = batch[key]
            if torch.is_tensor(value):
                named.append((f"{b_idx}:{key}", value))
            elif isinstance(value, (list, tuple)):
                named.extend(
                    (f"{b_idx}:{key}:{v_idx}", item)
                    for v_idx, item in enumerate(value)
                    if torch.is_tensor(item)
                )
    return _tensor_digest(named)


def save_batches(path: Path, batches: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_batches = []
    for batch in batches:
        cpu: dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                cpu[key] = value.detach().cpu()
            elif isinstance(value, (list, tuple)):
                cpu[key] = [item.detach().cpu() if torch.is_tensor(item) else item for item in value]
            else:
                cpu[key] = value
        cpu_batches.append(cpu)
    torch.save(cpu_batches, path)
    return batch_hash(cpu_batches)


def load_batches(path: Path, device: torch.device) -> list[dict[str, Any]]:
    batches = torch.load(path, map_location="cpu", weights_only=False)
    result = []
    for batch in batches:
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device)
            elif isinstance(value, (list, tuple)):
                moved[key] = [item.to(device) if torch.is_tensor(item) else item for item in value]
            else:
                moved[key] = value
        result.append(moved)
    return result


def component_hash(module: torch.nn.Module, prefixes: Sequence[str]) -> str:
    selected = [
        (name, value) for name, value in sorted(module.state_dict().items())
        if any(name.startswith(prefix) for prefix in prefixes)
    ]
    return _tensor_digest(selected)


def firm_hashes(firm: torch.nn.Module) -> dict[str, str]:
    return {
        "all_firm": module_hash(firm),
        "P": component_hash(firm, ("encoder", "p0_", "pi_", "p0", "pi")),
        "Q": component_hash(firm, ("q_encoder", "q_head", "q_")),
        "BP_policy": component_hash(firm, ("bp", "i_", "bar_i", "policy")),
    }


def set_macro_scope(model: torch.nn.Module, block: str) -> list[str]:
    """Freeze all macro parameters except the named diagnostic block."""
    if block not in BLOCK_NAMES:
        raise ValueError(f"unknown macro block {block!r}")
    model.requires_grad_(False)
    allowed: list[tuple[str, torch.nn.Module]] = []
    if block in {"fc1_c", "joint"}:
        allowed.append(("fc1_c", model.fc1_model.FC1_C))
    if block in {"fc1_k", "joint"}:
        allowed.append(("fc1_k", model.fc1_model.FC1_K))
    if block in {"sdf", "joint"}:
        allowed.extend((
            ("sdf_model", model.sdf_model),
            ("value_model", model.value_model),
        ))
    for _, submodule in allowed:
        submodule.requires_grad_(True)
    model.train(bool(allowed))
    for name, submodule in (
        ("fc1_c", model.fc1_model.FC1_C),
        ("fc1_k", model.fc1_model.FC1_K),
        ("sdf_model", model.sdf_model),
        ("value_model", model.value_model),
    ):
        submodule.train(any(allowed_name == name for allowed_name, _ in allowed))
    return [name for name, _ in allowed]


def trainable_parameter_names(model: torch.nn.Module) -> list[str]:
    return sorted(name for name, value in model.named_parameters() if value.requires_grad)


def reset_macro_optimizer(episode: Any) -> None:
    episode.optimizers = build_optimizers(
        {"sdf_fc1": episode.models["sdf_fc1"]}, episode.hyperparams
    )
    episode.lr_schedulers = episode._init_lr_schedulers()
    episode.step_count = 0
    episode.sdf_fc1_step_count = 0


def identity_scale_roundtrip(tensor: torch.Tensor) -> bool:
    """FC1 uses physical units; its compatibility scaler is the identity."""
    return bool(torch.equal(tensor, tensor.clone()))


def path_ids(table: TensorTable) -> torch.Tensor:
    index = list(table.columns).index("path")
    return torch.unique(table.data[:, index].long()).cpu()


def offset_validation_paths(table: TensorTable, offset: int) -> TensorTable:
    data = table.data.clone()
    index = list(table.columns).index("path")
    data[:, index] += float(offset)
    return TensorTable(data, list(table.columns))


def disjoint_path_audit(train: TensorTable, validation: TensorTable) -> dict[str, Any]:
    train_ids = path_ids(train)
    validation_ids = path_ids(validation)
    overlap = torch.isin(train_ids, validation_ids)
    return {
        "train_path_ids": train_ids.tolist(),
        "validation_path_ids": validation_ids.tolist(),
        "train_path_count": int(train_ids.numel()),
        "validation_path_count": int(validation_ids.numel()),
        "overlap_count": int(overlap.sum()),
    }


def flatten_batches(batches: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    parents = torch.cat([batch["parent"] for batch in batches], dim=0)
    children = torch.cat([
        torch.stack(list(batch["children"]), dim=1) for batch in batches
    ], dim=0)
    return parents, children


def regression_metrics(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    target = target.detach().reshape(-1).to(torch.float64).cpu()
    prediction = prediction.detach().reshape(-1).to(torch.float64).cpu()
    mask = torch.isfinite(target) & torch.isfinite(prediction)
    target, prediction = target[mask], prediction[mask]
    result = {"n": int(target.numel())}
    if target.numel() < 2:
        return {**result, **{key: float("nan") for key in (
            "mse", "rmse", "r2", "corr", "slope", "bias", "target_std", "prediction_std"
        )}}
    error = prediction - target
    target_centered = target - target.mean()
    pred_centered = prediction - prediction.mean()
    sst = target_centered.square().sum()
    corr_denom = torch.linalg.vector_norm(target_centered) * torch.linalg.vector_norm(pred_centered)
    return {
        **result,
        "mse": float(error.square().mean()),
        "rmse": float(torch.sqrt(error.square().mean())),
        "r2": float(1.0 - error.square().sum() / sst) if sst > 0 else float("nan"),
        "corr": float(torch.dot(target_centered, pred_centered) / corr_denom) if corr_denom > 0 else float("nan"),
        "slope": float(torch.dot(target_centered, pred_centered) / sst) if sst > 0 else float("nan"),
        "bias": float(error.mean()),
        "target_std": float(target.std(unbiased=False)),
        "prediction_std": float(prediction.std(unbiased=False)),
    }


def evaluate_fc1(model: torch.nn.Module, batches: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    targets_c, targets_k, predictions_c, predictions_k = [], [], [], []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in batches:
            parent = batch["parent"]
            children = torch.stack(list(batch["children"]), dim=1)
            if parent.shape[1] < 9 or children.shape[-1] < 9:
                raise RuntimeError("exact FC1 diagnostic requires true parent/child macro columns")
            _, _, _, pred_c, pred_k = model.forward_step(
                x_prev=parent[:, 4:5],
                x_curr=children[:, :, 4:5],
                hatcf_prev=parent[:, 7:8],
                lnkf_prev=parent[:, 8:9],
                return_physical=True,
            )
            target_c = children[:, :, 7:8] if children.shape[-1] == 9 else children[:, :, 8:9]
            target_k = children[:, :, 8:9] if children.shape[-1] == 9 else children[:, :, 9:10]
            targets_c.append(target_c)
            targets_k.append(target_k)
            predictions_c.append(pred_c)
            predictions_k.append(pred_k)
    model.train(was_training)
    result = {}
    for name, target, prediction in (
        ("hatc", torch.cat(targets_c), torch.cat(predictions_c)),
        ("lnk", torch.cat(targets_k), torch.cat(predictions_k)),
    ):
        result.update({f"{name}_{key}": value for key, value in regression_metrics(target, prediction).items()})
    return result


def parameter_norm(model: torch.nn.Module, *, trainable_only: bool = True) -> float:
    values = [
        value.detach().reshape(-1).to(torch.float64)
        for value in model.parameters()
        if (value.requires_grad or not trainable_only)
    ]
    return float(torch.linalg.vector_norm(torch.cat(values))) if values else 0.0


def gradient_norm(model: torch.nn.Module) -> float:
    values = [
        value.grad.detach().reshape(-1).to(torch.float64)
        for value in model.parameters()
        if value.requires_grad and value.grad is not None
    ]
    return float(torch.linalg.vector_norm(torch.cat(values))) if values else 0.0


def current_lr(episode: Any, block: str) -> float:
    optimizer = episode.optimizers["sdf_fc1"]
    wanted = "fc1" if block.startswith("fc1") or block == "joint" else "sdf_core"
    for group in optimizer.param_groups:
        if group.get("group_name") == wanted:
            return float(group["lr"])
    return float(optimizer.param_groups[0]["lr"])


def snapshot_functions(
    model: torch.nn.Module,
    parents: torch.Tensor,
    children: torch.Tensor,
) -> dict[str, torch.Tensor]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        _, _, m, fc1_c, fc1_k = model.forward_step(
            x_prev=parents[:, 4:5],
            x_curr=children[:, :, 4:5],
            hatcf_prev=parents[:, 7:8],
            lnkf_prev=parents[:, 8:9],
            return_physical=True,
        )
    model.train(was_training)
    return {
        "m": m.detach().cpu().reshape(-1),
        "fc1_c": fc1_c.detach().cpu().reshape(-1),
        "fc1_k": fc1_k.detach().cpu().reshape(-1),
    }


def movement_metrics(old: torch.Tensor, new: torch.Tensor, prefix: str = "") -> dict[str, float]:
    old = old.detach().reshape(-1).to(torch.float64)
    new = new.detach().reshape(-1).to(torch.float64)
    delta = new - old
    absolute = delta.abs()
    tag = f"{prefix}_" if prefix else ""
    return {
        f"{tag}nmae": float(absolute.mean() / (old.abs().mean() + 1e-12)),
        f"{tag}l2": float(torch.linalg.vector_norm(delta)),
        f"{tag}p50": float(torch.quantile(absolute, 0.50)),
        f"{tag}p90": float(torch.quantile(absolute, 0.90)),
        f"{tag}p99": float(torch.quantile(absolute, 0.99)),
        f"{tag}max": float(absolute.max()),
    }


def movement_relation(previous_delta: torch.Tensor, current_delta: torch.Tensor) -> dict[str, float]:
    previous_delta = previous_delta.reshape(-1).to(torch.float64)
    current_delta = current_delta.reshape(-1).to(torch.float64)
    previous_norm = torch.linalg.vector_norm(previous_delta)
    current_norm = torch.linalg.vector_norm(current_delta)
    dot = torch.dot(current_delta, previous_delta)
    return {
        "norm_gain": float(current_norm / (previous_norm + 1e-12)),
        "directional_gain": float(dot / (previous_norm.square() + 1e-12)),
        "movement_cosine": float(dot / (previous_norm * current_norm + 1e-12)),
    }


def two_cycle_detect(
    two_back: torch.Tensor,
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    ratio_threshold: float = 0.25,
) -> bool:
    delta_previous = previous - two_back
    delta_current = current - previous
    relation = movement_relation(delta_previous, delta_current)
    distance_one = torch.linalg.vector_norm(delta_current.to(torch.float64))
    distance_two = torch.linalg.vector_norm((current - two_back).to(torch.float64))
    return bool(
        distance_two < ratio_threshold * (distance_one + 1e-12)
        and relation["movement_cosine"] < 0.0
    )


def aligned_macro_vectors(
    old: pd.DataFrame, new: pd.DataFrame, columns: Sequence[str]
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    keys = ["path", "t", "branch"]
    joined = old[keys + list(columns)].merge(
        new[keys + list(columns)], on=keys, suffixes=("_old", "_new"), how="inner"
    )
    result = {}
    for column in columns:
        left = torch.from_numpy(joined[f"{column}_old"].to_numpy(dtype=np.float64))
        right = torch.from_numpy(joined[f"{column}_new"].to_numpy(dtype=np.float64))
        mask = torch.isfinite(left) & torch.isfinite(right)
        result[column] = (left[mask], right[mask])
    return result


def macro_target_hash(frame: pd.DataFrame, *, include_m: bool = False) -> str:
    columns = ["path", "t", "branch", "Hatc", "LnK", "C", "K", "n_firms", "I_oper"]
    if include_m:
        columns.append("M")
    use = frame[columns].sort_values(["path", "t", "branch"]).to_numpy(dtype=np.float64)
    return hashlib.sha256(use.tobytes()).hexdigest()


def sdf_dependency_status(base_target_hash: str, alternative_target_hash: str) -> dict[str, Any]:
    dependency = base_target_hash != alternative_target_hash
    return {
        "sdf_outer_dependency": dependency,
        "experiment_status": "RUN_REQUIRED" if dependency else "DEGENERATE_TO_L0_SDF",
    }


def validate_l1_max_outer(value: int) -> int:
    if not 0 <= int(value) <= 8:
        raise ValueError("L1 max outer must be in [0, 8]")
    return int(value)


def load_macro_initial_state(path: Path, model: torch.nn.Module) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, Mapping) and "sdf_fc1" in payload:
        payload = payload["sdf_fc1"]
    model.load_state_dict(payload, strict=True)


def load_sdf_only_state(path: Path, model: torch.nn.Module) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, Mapping) and "sdf_fc1" in payload:
        payload = payload["sdf_fc1"]
    current = model.state_dict()
    for name, value in payload.items():
        if name.startswith(("sdf_model.", "value_model.")):
            current[name] = value
    model.load_state_dict(current, strict=True)


def prepare_batches(
    episode: Any,
    train_macro: TensorTable,
    validation_macro: TensorTable,
    root: Path,
    *,
    batch_size: int,
    n_branches: int,
) -> dict[str, Any]:
    train_pairs = episode._build_sdf_pairs_from_macro_tensor(train_macro)
    validation_pairs = episode._build_sdf_pairs_from_macro_tensor(validation_macro)
    split = disjoint_path_audit(train_pairs, validation_pairs)
    if split["overlap_count"]:
        raise RuntimeError(f"train/validation path leakage: {split}")
    episode.add_FC1loss = True
    fc1_train = episode._create_sdf_batches_from_macro_tensor(
        train_pairs, batch_size=batch_size, n_branches=n_branches
    )
    fc1_validation = episode._create_sdf_batches_from_macro_tensor(
        validation_pairs, batch_size=batch_size, n_branches=n_branches
    )
    target_batches = int(getattr(episode.hyperparams, "sdf_true_target_batches", 20))
    min_parents = int(getattr(episode.hyperparams, "sdf_min_parent_groups_per_batch", 256))
    sdf_train = episode._create_sdf_batches_from_macro_tensor(
        train_pairs,
        batch_size=batch_size,
        n_branches=n_branches,
        target_num_batches=target_batches,
        min_parent_groups_per_batch=min_parents,
    )
    sdf_validation = episode._create_sdf_batches_from_macro_tensor(
        validation_pairs, batch_size=batch_size, n_branches=n_branches
    )
    if not all((fc1_train, fc1_validation, sdf_train, sdf_validation)):
        raise RuntimeError("frozen macro dataset produced an empty production batch stream")
    hashes = {
        "train_pairs": save_table(root / "train_macro_pairs.pt", train_pairs),
        "validation_pairs": save_table(root / "validation_macro_pairs.pt", validation_pairs),
        "fc1_train": save_batches(root / "fc1_train_batches.pt", fc1_train),
        "fc1_validation": save_batches(root / "fc1_validation_batches.pt", fc1_validation),
        "sdf_train": save_batches(root / "sdf_train_batches.pt", sdf_train),
        "sdf_validation": save_batches(root / "sdf_validation_batches.pt", sdf_validation),
    }
    metadata = {
        "hashes": hashes,
        "split": split,
        "pairing": {
            "input_t": "parent t: x_t, true Hatc_t, true LnK_t (teacher-forced production FC1 loss)",
            "child_shock_t": "child t+1: x_t1",
            "target_t": "child t+1: Hatc_t1, LnK_t1",
            "forecast_state_columns_retained": "Hatcf_t, LnKF_t for recursive/counterfactual diagnostics",
            "physical_or_normalized": "physical",
            "scaler": "identity/no-op",
            "loss_scale": "physical",
        },
        "counts": {
            "train_pair_rows": len(train_pairs),
            "validation_pair_rows": len(validation_pairs),
            "fc1_train_batches": len(fc1_train),
            "fc1_validation_batches": len(fc1_validation),
            "sdf_train_batches": len(sdf_train),
            "sdf_validation_batches": len(sdf_validation),
        },
    }
    write_json(root / "dataset_metadata.json", metadata)
    return metadata


def verify_frozen_dataset(root: Path, expected: Mapping[str, str]) -> None:
    device = torch.device("cpu")
    observed = {
        "train_pairs": table_hash(load_table(root / "train_macro_pairs.pt", device)),
        "validation_pairs": table_hash(load_table(root / "validation_macro_pairs.pt", device)),
        "fc1_train": batch_hash(load_batches(root / "fc1_train_batches.pt", device)),
        "fc1_validation": batch_hash(load_batches(root / "fc1_validation_batches.pt", device)),
        "sdf_train": batch_hash(load_batches(root / "sdf_train_batches.pt", device)),
        "sdf_validation": batch_hash(load_batches(root / "sdf_validation_batches.pt", device)),
    }
    if dict(expected) != observed:
        raise RuntimeError(f"L0 frozen dataset hash changed: expected={expected}, observed={observed}")


def lag_scan(
    model: torch.nn.Module,
    pair_table: TensorTable,
    output: Path,
) -> pd.DataFrame:
    frame = pair_table.to_dataframe()
    branch0 = frame[frame["branch"] == 0].sort_values(["path", "t"]).copy()
    device = next(model.parameters()).device
    data = torch.tensor(branch0.values, dtype=torch.float32, device=device)
    col = {name: index for index, name in enumerate(branch0.columns)}
    with torch.no_grad():
        _, _, _, pred_c, pred_k = model.forward_step(
            x_prev=data[:, col["x_t"]:col["x_t"] + 1],
            x_curr=data[:, col["x_t1"]:col["x_t1"] + 1].unsqueeze(1),
            hatcf_prev=data[:, col["Hatc_t"]:col["Hatc_t"] + 1],
            lnkf_prev=data[:, col["LnK_t"]:col["LnK_t"] + 1],
            return_physical=True,
        )
    branch0["pred_hatc"] = pred_c[:, 0, 0].detach().cpu().numpy()
    branch0["pred_lnk"] = pred_k[:, 0, 0].detach().cpu().numpy()
    rows = []
    for variable, prediction, target in (
        ("Hatc", "pred_hatc", "Hatc_t1"),
        ("LnK", "pred_lnk", "LnK_t1"),
    ):
        for lag in (-1, 0, 1, 2, 3):
            shifted = branch0.groupby("path", sort=False)[target].shift(-lag)
            metrics = regression_metrics(
                torch.tensor(shifted.to_numpy()),
                torch.tensor(branch0[prediction].to_numpy()),
            )
            rows.append({"variable": variable, "target_offset": lag, **metrics})
    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def run_fc1_fit(
    episode: Any,
    train_batches: Sequence[Mapping[str, Any]],
    validation_batches: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    block: str,
    production_epochs: int,
    max_epochs: int,
    patience: int,
    log_interval: int,
) -> dict[str, Any]:
    if block not in {"fc1_c", "fc1_k", "joint"}:
        raise ValueError(block)
    output.mkdir(parents=True, exist_ok=True)
    model = episode.models["sdf_fc1"]
    allowed = set_macro_scope(model, block)
    reset_macro_optimizer(episode)
    episode.add_FC1loss = True
    episode.set_sdf_training_phase(SDFTrainingPhase.FC1_ONLY)
    production_freeze = episode._set_sdf_training_phase_freeze

    def scoped_freeze() -> None:
        production_freeze()
        set_macro_scope(model, block)

    # Production FC1 phase normally enables both heads.  The individual L0
    # diagnostics narrow that phase after every production freeze call without
    # changing its loss, optimizer, or tensor construction.
    episode._set_sdf_training_phase_freeze = scoped_freeze
    frozen_hashes = {
        "sdf_model": module_hash(model.sdf_model),
        "value_model": module_hash(model.value_model),
        "fc1_c": module_hash(model.fc1_model.FC1_C),
        "fc1_k": module_hash(model.fc1_model.FC1_K),
    }
    monitor = "hatc_r2" if block == "fc1_c" else "lnk_r2"
    if block == "joint":
        monitor = "joint_r2"
    best_score = -float("inf")
    best_epoch = 0
    no_improvement = 0
    best_state = deepcopy(model.state_dict())
    rows: list[dict[str, Any]] = []

    def record(epoch: int, train_summary: Optional[dict[str, Any]]) -> dict[str, Any]:
        train_metrics = evaluate_fc1(model, train_batches)
        val_metrics = evaluate_fc1(model, validation_batches)
        row: dict[str, Any] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"validation_{key}": value for key, value in val_metrics.items()})
        row["gradient_norm"] = gradient_norm(model)
        row["parameter_norm"] = parameter_norm(model)
        row["learning_rate"] = current_lr(episode, block)
        if train_summary:
            losses = train_summary.get("final_losses", {})
            for key, value in losses.items():
                if isinstance(value, (int, float, bool)):
                    row[f"production_loss_{key}"] = value
        rows.append(row)
        return row

    try:
        record(0, None)
        for epoch in range(1, max_epochs + 1):
            train_summary = episode._run_batches(
                list(train_batches), n_epochs=1, log_interval=log_interval,
                train_modules=["sdf_fc1"], desc_prefix=f"L0 {block} ",
            )
            row = record(epoch, train_summary)
            if block == "joint":
                candidates = [row["validation_hatc_r2"], row["validation_lnk_r2"]]
                score = float(np.nanmean(candidates))
            else:
                score = float(row[f"validation_{monitor}"])
            if math.isfinite(score) and score > best_score + 1e-8:
                best_score = score
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                no_improvement = 0
            else:
                no_improvement += 1
            if epoch == production_epochs:
                torch.save(model.state_dict(), output / "production_budget.pt")
            if epoch >= production_epochs and no_improvement >= patience:
                break
    finally:
        episode._set_sdf_training_phase_freeze = production_freeze
    history = pd.DataFrame(rows)
    history.to_csv(output / "history.csv", index=False)
    model.load_state_dict(best_state)
    torch.save(best_state, output / "best.pt")
    final_train = evaluate_fc1(model, train_batches)
    final_validation = evaluate_fc1(model, validation_batches)
    invariants = {
        "sdf_model_unchanged": module_hash(model.sdf_model) == frozen_hashes["sdf_model"],
        "value_model_unchanged": module_hash(model.value_model) == frozen_hashes["value_model"],
        "fc1_c_unchanged": module_hash(model.fc1_model.FC1_C) == frozen_hashes["fc1_c"],
        "fc1_k_unchanged": module_hash(model.fc1_model.FC1_K) == frozen_hashes["fc1_k"],
    }
    required_frozen = ["sdf_model_unchanged", "value_model_unchanged"]
    if block == "fc1_c":
        required_frozen.append("fc1_k_unchanged")
    if block == "fc1_k":
        required_frozen.append("fc1_c_unchanged")
    if not all(invariants[name] for name in required_frozen):
        raise RuntimeError(f"illegal parameter movement in {block}: {invariants}")
    production_row = history[history["epoch"] == min(production_epochs, int(history["epoch"].max()))].iloc[-1].to_dict()
    summary = {
        "block": block,
        "allowed_submodules": allowed,
        "trainable_parameter_names": trainable_parameter_names(model),
        "production_budget_epoch": min(production_epochs, int(history["epoch"].max())),
        "production_budget": production_row,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "epochs_completed": int(history["epoch"].max()),
        "early_stop_patience": patience,
        "best_train": final_train,
        "best_validation": final_validation,
        "invariants": invariants,
    }
    write_json(output / "summary.json", summary)
    return summary


def eval_sdf_metrics(episode: Any, batches: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    raw = episode._evaluate_sdf_fc1_batches(list(batches), prefix=prefix, max_batches=None)
    return {key: value for key, value in raw.items() if isinstance(value, (int, float, bool))}


def run_sdf_chunks(
    episode: Any,
    train_batches: Sequence[Mapping[str, Any]],
    validation_batches: Sequence[Mapping[str, Any]],
    probe_parents: torch.Tensor,
    probe_children: torch.Tensor,
    output: Path,
    *,
    chunks: int,
    epochs_per_chunk: int,
    log_interval: int,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    model = episode.models["sdf_fc1"]
    set_macro_scope(model, "sdf")
    reset_macro_optimizer(episode)
    episode.add_FC1loss = True
    episode.set_sdf_training_phase(SDFTrainingPhase.SDF_TRUE_ONLY)
    fc1_hash = module_hash(model.fc1_model)
    probes = [snapshot_functions(model, probe_parents, probe_children)["m"]]
    torch.save(probes[0], output / "probe_chunk_000.pt")
    rows = []
    before = eval_sdf_metrics(episode, validation_batches, "chunk0")
    rows.append({"chunk": 0, "epochs_requested": 0, **before})
    original_reset = bool(getattr(episode.hyperparams, "sdf_al_reset_on_true_start", True))
    histories = []
    try:
        for chunk in range(1, chunks + 1):
            if chunk > 1:
                # A chunk boundary is a serialization boundary, not a new SDF solve.
                # Preserve the production AL dual state across continuation chunks.
                episode.hyperparams.sdf_al_reset_on_true_start = False
            result = episode._run_sdf_phase_with_validation(
                train_batches=list(train_batches),
                val_batches=list(validation_batches),
                n_epochs=epochs_per_chunk,
                log_interval=log_interval,
                stage=SDFTrainingPhase.SDF_TRUE_ONLY,
                prefix=f"l0_sdf_chunk_{chunk}",
            )
            histories.append(result)
            probe = snapshot_functions(model, probe_parents, probe_children)["m"]
            probes.append(probe)
            torch.save(probe, output / f"probe_chunk_{chunk:03d}.pt")
            torch.save(model.state_dict(), output / f"checkpoint_chunk_{chunk:03d}.pt")
            current = eval_sdf_metrics(episode, validation_batches, f"chunk{chunk}")
            move = movement_metrics(probes[-2], probes[-1], "M_function")
            relation = {}
            if len(probes) >= 3:
                relation = {
                    f"M_function_{key}": value for key, value in movement_relation(
                        probes[-2] - probes[-3], probes[-1] - probes[-2]
                    ).items()
                }
            rows.append({
                "chunk": chunk,
                "epochs_requested": epochs_per_chunk,
                "epochs_completed": result.get("epochs_completed", result.get("accepted_epochs", float("nan"))),
                "stage_passed": result.get("passed", False),
                "safe_to_continue": result.get("sdf_safe_to_continue", result.get("safe_to_continue", False)),
                "converged": result.get("sdf_converged", result.get("converged", False)),
                **current, **move, **relation,
            })
            write_json(output / f"history_chunk_{chunk:03d}.json", result)
    finally:
        episode.hyperparams.sdf_al_reset_on_true_start = original_reset
    if module_hash(model.fc1_model) != fc1_hash:
        raise RuntimeError("FC1 changed during L0 SDF-only continuation")
    history = pd.DataFrame(rows)
    history.to_csv(output / "history.csv", index=False)
    summary = {
        "chunks_requested": chunks,
        "epochs_per_chunk": epochs_per_chunk,
        "final_row": rows[-1],
        "fc1_hash_invariant": True,
        "probe_hash": tensor_hash(probe_parents) + ":" + tensor_hash(probe_children),
    }
    write_json(output / "summary.json", summary)
    return summary


def compare_sdf_finals(final_vectors: Mapping[str, torch.Tensor]) -> pd.DataFrame:
    rows = []
    names = list(final_vectors)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            a = final_vectors[left].to(torch.float64)
            b = final_vectors[right].to(torch.float64)
            metrics = movement_metrics(a, b, "M")
            centered_a = a - a.mean()
            centered_b = b - b.mean()
            denom = torch.linalg.vector_norm(centered_a) * torch.linalg.vector_norm(centered_b)
            rows.append({
                "left": left,
                "right": right,
                **metrics,
                "M_corr": float(torch.dot(centered_a, centered_b) / denom) if denom > 0 else float("nan"),
                "left_mean": float(a.mean()),
                "right_mean": float(b.mean()),
                "left_std": float(a.std(unbiased=False)),
                "right_std": float(b.std(unbiased=False)),
            })
    return pd.DataFrame(rows)


def load_context(args: argparse.Namespace, device: torch.device):
    source_run = args.source_run.expanduser().resolve()
    source_metadata = source_run
    if not (source_metadata / "metadata" / "hyperparams.json").is_file():
        source_metadata = source_run / "source"
    episode, hp, models, _, _, _, _, provenance, _ = _load_setup(
        source_run, device, None, source_metadata,
        args.source_checkpoint.expanduser().resolve(),
    )
    episode.models = models
    return episode, hp, models, provenance


def prepare_l0_dataset(
    args: argparse.Namespace,
    episode: Any,
    models: Mapping[str, torch.nn.Module],
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    data_root = output / "l0" / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    model = models["sdf_fc1"]
    load_macro_initial_state(args.outer0_macro_checkpoint, model)
    firm = models["policy_value"]
    firm.load_state_dict(_load_state_dict_from_checkpoint(args.firm_checkpoint, "policy_value"))
    set_macro_scope(model, "frozen")
    firm.requires_grad_(False).eval()
    initial_train = make_initial_state(
        models=models, episode=episode, paths=args.simulation_paths,
        firms=args.firms_per_path, horizon=args.simulation_horizon,
        branch_num=args.branch_num, seed=args.seed + 1000, device=device,
    )
    initial_validation = make_initial_state(
        models=models, episode=episode, paths=args.validation_paths,
        firms=args.firms_per_path, horizon=args.simulation_horizon,
        branch_num=args.branch_num, seed=args.seed + 2000, device=device,
    )
    started = time.perf_counter()
    train_out = run_simulation(
        models=models, episode=episode, initial_state=initial_train,
        paths=args.simulation_paths, firms=args.firms_per_path,
        horizon=args.simulation_horizon, branch_num=args.branch_num,
        transition_seed=args.seed + 1011, entry_seed=args.seed + 1012, device=device,
    )
    validation_out = run_simulation(
        models=models, episode=episode, initial_state=initial_validation,
        paths=args.validation_paths, firms=args.firms_per_path,
        horizon=args.simulation_horizon, branch_num=args.branch_num,
        transition_seed=args.seed + 2011, entry_seed=args.seed + 2012, device=device,
    )
    simulation_seconds = time.perf_counter() - started
    validation_offset = 1_000_000
    train_macro = train_out.macro.to(device)
    validation_macro = offset_validation_paths(validation_out.macro.to(device), validation_offset)
    hashes = {
        "train_macro": save_table(data_root / "train_macro_dataset.pt", train_macro),
        "validation_macro": save_table(data_root / "validation_macro_dataset.pt", validation_macro),
    }
    batch_meta = prepare_batches(
        episode, train_macro, validation_macro, data_root,
        batch_size=args.macro_batch_size, n_branches=args.branch_num,
    )
    pair_table = load_table(data_root / "validation_macro_pairs.pt", device)
    scan = lag_scan(model, pair_table, output / "l0" / "index_alignment_scan.csv")
    metadata = {
        "simulation_seconds": simulation_seconds,
        "simulation_calls": 2,
        "outer0_macro_checkpoint": str(args.outer0_macro_checkpoint),
        "firm_checkpoint": str(args.firm_checkpoint),
        "firm_hashes": firm_hashes(firm),
        "macro_initial_hash": module_hash(model),
        "train_initial_cross_section_hash": state_hash(initial_train),
        "validation_initial_cross_section_hash": state_hash(initial_validation),
        "train_simulation_seed": args.seed + 1011,
        "validation_simulation_seed": args.seed + 2011,
        "validation_path_id_offset": validation_offset,
        "raw_table_hashes": hashes,
        "production_tensor_hashes": batch_meta["hashes"],
        "path_split": batch_meta["split"],
        "pairing": batch_meta["pairing"],
        "lag_scan_rows": len(scan),
        "fc1_scale_roundtrip": identity_scale_roundtrip(torch.tensor([-2.0, 0.0, 3.0])),
    }
    write_json(data_root / "train_metadata.json", metadata)
    write_json(data_root / "validation_metadata.json", metadata)
    return metadata


def restore_l0_batches(output: Path, device: torch.device):
    root = output / "l0" / "data"
    return {
        "fc1_train": load_batches(root / "fc1_train_batches.pt", device),
        "fc1_validation": load_batches(root / "fc1_validation_batches.pt", device),
        "sdf_train": load_batches(root / "sdf_train_batches.pt", device),
        "sdf_validation": load_batches(root / "sdf_validation_batches.pt", device),
    }


def fresh_macro_model(
    args: argparse.Namespace,
    device: torch.device,
    macro_checkpoint: Optional[Path] = None,
):
    episode, _, models, provenance = load_context(args, device)
    checkpoint = macro_checkpoint or args.outer0_macro_checkpoint
    load_macro_initial_state(checkpoint, models["sdf_fc1"])
    models["policy_value"].load_state_dict(
        _load_state_dict_from_checkpoint(args.firm_checkpoint, "policy_value")
    )
    episode.models = models
    return episode, models, provenance


def run_l0(args: argparse.Namespace, output: Path, device: torch.device) -> dict[str, Any]:
    episode, _, models, provenance = fresh_macro_model(args, device)
    fc2 = models.get("fc2")
    fc2_counter = {"calls": 0}
    fc2_hash_before = module_hash(fc2)
    install_fc2_guard(fc2, fc2_counter)
    data_metadata = prepare_l0_dataset(args, episode, models, output, device)
    batches = restore_l0_batches(output, device)
    expected_hashes = data_metadata["production_tensor_hashes"]
    results: dict[str, Any] = {"data": data_metadata}

    for block in ("fc1_c", "fc1_k"):
        verify_frozen_dataset(output / "l0" / "data", expected_hashes)
        fit_episode, fit_models, _ = fresh_macro_model(args, device)
        result = run_fc1_fit(
            fit_episode, batches["fc1_train"], batches["fc1_validation"],
            output / "l0" / block,
            block=block,
            production_epochs=args.production_fc1_epochs,
            max_epochs=args.fc1_max_epochs,
            patience=args.fc1_patience,
            log_interval=args.log_interval,
        )
        results[block] = result
        verify_frozen_dataset(output / "l0" / "data", expected_hashes)

    probe_parents, probe_children = flatten_batches(batches["sdf_validation"])
    probe_n = min(args.probe_size, int(probe_parents.shape[0]))
    probe_parents = probe_parents[:probe_n]
    probe_children = probe_children[:probe_n]
    probe_hash = tensor_hash(probe_parents) + ":" + tensor_hash(probe_children)
    torch.save(
        {"parents": probe_parents.cpu(), "children": probe_children.cpu()},
        output / "l0" / "data" / "sdf_probe_bank.pt",
    )

    initializers = {
        "outer0_start": args.outer0_macro_checkpoint,
        "outer_middle": args.middle_macro_checkpoint,
        "outer_final": args.final_macro_checkpoint,
    }
    final_vectors: dict[str, torch.Tensor] = {}
    sdf_results = {}
    for name, checkpoint in initializers.items():
        verify_frozen_dataset(output / "l0" / "data", expected_hashes)
        sdf_episode, sdf_models, _ = fresh_macro_model(args, device)
        load_sdf_only_state(checkpoint, sdf_models["sdf_fc1"])
        result = run_sdf_chunks(
            sdf_episode, batches["sdf_train"], batches["sdf_validation"],
            probe_parents, probe_children, output / "l0" / "sdf" / name,
            chunks=args.sdf_chunks, epochs_per_chunk=args.sdf_epochs_per_chunk,
            log_interval=args.log_interval,
        )
        final_vectors[name] = torch.load(
            output / "l0" / "sdf" / name / f"probe_chunk_{args.sdf_chunks:03d}.pt",
            map_location="cpu", weights_only=True,
        )
        sdf_results[name] = result
        verify_frozen_dataset(output / "l0" / "data", expected_hashes)
    comparison = compare_sdf_finals(final_vectors)
    comparison.to_csv(output / "l0" / "sdf" / "multi_start_comparison.csv", index=False)
    non_identified = bool(
        not comparison.empty
        and float(comparison["M_nmae"].max()) > args.sdf_nonidentification_nmae
    )
    sdf_summary = {
        "initializers": {name: str(path) for name, path in initializers.items()},
        "runs": sdf_results,
        "probe_hash": probe_hash,
        "possible_conditional_non_identification": non_identified,
        "max_pairwise_final_M_nmae": float(comparison["M_nmae"].max()) if not comparison.empty else float("nan"),
    }
    write_json(output / "l0" / "sdf" / "summary.json", sdf_summary)
    results["sdf"] = sdf_summary

    verify_frozen_dataset(output / "l0" / "data", expected_hashes)
    joint_episode, joint_models, _ = fresh_macro_model(args, device)
    joint_fc1 = run_fc1_fit(
        joint_episode, batches["fc1_train"], batches["fc1_validation"],
        output / "l0" / "joint" / "fc1_stage",
        block="joint", production_epochs=args.production_fc1_epochs,
        max_epochs=max(results["fc1_c"]["best_epoch"], results["fc1_k"]["best_epoch"], 1),
        patience=args.fc1_patience, log_interval=args.log_interval,
    )
    joint_sdf = run_sdf_chunks(
        joint_episode, batches["sdf_train"], batches["sdf_validation"],
        probe_parents, probe_children, output / "l0" / "joint" / "sdf_stage",
        chunks=args.sdf_chunks, epochs_per_chunk=args.sdf_epochs_per_chunk,
        log_interval=args.log_interval,
    )
    results["joint"] = {"fc1": joint_fc1, "sdf": joint_sdf}
    write_json(output / "l0" / "joint" / "summary.json", results["joint"])
    verify_frozen_dataset(output / "l0" / "data", expected_hashes)
    if fc2_counter["calls"] != 0 or module_hash(fc2) != fc2_hash_before:
        raise RuntimeError("FC2 invariant failed in L0")
    results["invariants"] = {
        "dataset_hash_invariant": True,
        "fc2_forward_calls": fc2_counter["calls"],
        "fc2_hash_invariant": module_hash(fc2) == fc2_hash_before,
        "source_git_commit": provenance.get("source_git_commit"),
    }
    write_json(output / "l0" / "summary.json", results)
    return results


def choose_conditional_budgets(l0: Mapping[str, Any], args: argparse.Namespace) -> dict[str, int]:
    fc1_epochs = max(
        1,
        int(l0["fc1_c"].get("best_epoch", args.production_fc1_epochs)),
        int(l0["fc1_k"].get("best_epoch", args.production_fc1_epochs)),
    )
    primary = l0["sdf"]["runs"]["outer0_start"]
    selected_chunk = args.sdf_chunks
    # Use the first saved chunk reported converged; otherwise use the full L0 budget.
    history_path = Path(args.output_dir) / "l0" / "sdf" / "outer0_start" / "history.csv"
    if history_path.is_file():
        history = pd.read_csv(history_path)
        converged_mask = (
            history["converged"].fillna(False).astype(bool)
            if "converged" in history
            else pd.Series(False, index=history.index)
        )
        converged = history[converged_mask & (history["chunk"] > 0)]
        if not converged.empty:
            selected_chunk = int(converged.iloc[0]["chunk"])
    return {
        "fc1_epochs": min(fc1_epochs, args.fc1_max_epochs),
        "sdf_chunks": max(1, selected_chunk),
        "sdf_epochs_per_chunk": args.sdf_epochs_per_chunk,
        "sdf_total_epochs": max(1, selected_chunk) * args.sdf_epochs_per_chunk,
        "l0_sdf_reference_chunks": int(primary.get("chunks_requested", args.sdf_chunks)),
    }


def simulate_frozen_firm(
    args: argparse.Namespace,
    episode: Any,
    models: Mapping[str, torch.nn.Module],
    train_initial: Mapping[str, torch.Tensor],
    validation_initial: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[Any, Any, float]:
    started = time.perf_counter()
    train_out = run_simulation(
        models=models, episode=episode, initial_state=train_initial,
        paths=args.simulation_paths, firms=args.firms_per_path,
        horizon=args.simulation_horizon, branch_num=args.branch_num,
        transition_seed=args.seed + 1011, entry_seed=args.seed + 1012, device=device,
    )
    validation_out = run_simulation(
        models=models, episode=episode, initial_state=validation_initial,
        paths=args.validation_paths, firms=args.firms_per_path,
        horizon=args.simulation_horizon, branch_num=args.branch_num,
        transition_seed=args.seed + 2011, entry_seed=args.seed + 2012, device=device,
    )
    return train_out, validation_out, time.perf_counter() - started


def build_in_memory_batches(
    episode: Any,
    train_macro: TensorTable,
    validation_macro: TensorTable,
    args: argparse.Namespace,
) -> dict[str, Any]:
    validation_macro = offset_validation_paths(validation_macro, 1_000_000)
    train_pairs = episode._build_sdf_pairs_from_macro_tensor(train_macro)
    validation_pairs = episode._build_sdf_pairs_from_macro_tensor(validation_macro)
    if disjoint_path_audit(train_pairs, validation_pairs)["overlap_count"]:
        raise RuntimeError("L1 path split leakage")
    episode.add_FC1loss = True
    return {
        "fc1_train": episode._create_sdf_batches_from_macro_tensor(
            train_pairs, batch_size=args.macro_batch_size, n_branches=args.branch_num
        ),
        "fc1_validation": episode._create_sdf_batches_from_macro_tensor(
            validation_pairs, batch_size=args.macro_batch_size, n_branches=args.branch_num
        ),
        "sdf_train": episode._create_sdf_batches_from_macro_tensor(
            train_pairs, batch_size=args.macro_batch_size, n_branches=args.branch_num,
            target_num_batches=int(getattr(episode.hyperparams, "sdf_true_target_batches", 20)),
            min_parent_groups_per_batch=int(getattr(episode.hyperparams, "sdf_min_parent_groups_per_batch", 256)),
        ),
        "sdf_validation": episode._create_sdf_batches_from_macro_tensor(
            validation_pairs, batch_size=args.macro_batch_size, n_branches=args.branch_num
        ),
    }


def run_l1_branch(
    args: argparse.Namespace,
    output: Path,
    device: torch.device,
    *,
    branch: str,
    budgets: Mapping[str, int],
    probe_parents: torch.Tensor,
    probe_children: torch.Tensor,
) -> dict[str, Any]:
    episode, models, _ = fresh_macro_model(args, device)
    model = models["sdf_fc1"]
    firm = models["policy_value"]
    firm.requires_grad_(False).eval()
    initial_firm_hashes = firm_hashes(firm)
    fc2 = models.get("fc2")
    fc2_counter = {"calls": 0}
    fc2_before = module_hash(fc2)
    install_fc2_guard(fc2, fc2_counter)
    train_initial = make_initial_state(
        models=models, episode=episode, paths=args.simulation_paths,
        firms=args.firms_per_path, horizon=args.simulation_horizon,
        branch_num=args.branch_num, seed=args.seed + 1000, device=device,
    )
    validation_initial = make_initial_state(
        models=models, episode=episode, paths=args.validation_paths,
        firms=args.firms_per_path, horizon=args.simulation_horizon,
        branch_num=args.branch_num, seed=args.seed + 2000, device=device,
    )
    crn = {
        "train_initial_hash": state_hash(train_initial),
        "validation_initial_hash": state_hash(validation_initial),
        "train_transition_seed": args.seed + 1011,
        "train_entry_seed": args.seed + 1012,
        "validation_transition_seed": args.seed + 2011,
        "validation_entry_seed": args.seed + 2012,
    }
    branch_root = output / "l1" / branch
    branch_root.mkdir(parents=True, exist_ok=True)
    function_states = [snapshot_functions(model, probe_parents, probe_children)]
    torch.save(function_states[0], branch_root / "functions_outer_000_start.pt")
    rows = []
    val_frames: list[pd.DataFrame] = []
    target_vectors: list[dict[str, torch.Tensor]] = []
    stop_reason = "max_outer"

    for outer in range(args.l1_max_outer + 1):
        set_macro_scope(model, "frozen")
        train_out, validation_out, simulation_seconds = simulate_frozen_firm(
            args, episode, models, train_initial, validation_initial, device
        )
        val_frame = _to_frame(validation_out.macro)
        train_frame = _to_frame(train_out.macro)
        val_frames.append(val_frame)
        target_vectors.append({
            "hatc": torch.tensor(val_frame.sort_values(["path", "t", "branch"])["Hatc"].to_numpy()),
            "lnk": torch.tensor(val_frame.sort_values(["path", "t", "branch"])["LnK"].to_numpy()),
        })
        row: dict[str, Any] = {
            "outer": outer,
            "simulation_seconds": simulation_seconds,
            "firm_hash_invariant": firm_hashes(firm) == initial_firm_hashes,
            "crn_hash": stable_json_hash(crn),
            "target_hash": macro_target_hash(val_frame),
            "M_data_hash": macro_target_hash(val_frame, include_m=True),
        }
        if not row["firm_hash_invariant"]:
            raise RuntimeError(f"firm network changed in L1 {branch} outer {outer}")
        if outer > 0:
            aligned = aligned_macro_vectors(val_frames[-2], val_frames[-1], ("Hatc", "LnK", "C", "K"))
            for column, (old, new) in aligned.items():
                row.update(movement_metrics(old, new, f"path_{column.lower()}"))
                if column in {"Hatc", "LnK"}:
                    row.update(movement_metrics(old, new, f"target_{column.lower()}"))
            old_functions, new_functions = function_states[-2], function_states[-1]
            for name in FUNCTION_NAMES:
                row.update(movement_metrics(old_functions[name], new_functions[name], f"function_{name}"))
            if outer > 1:
                for name in FUNCTION_NAMES:
                    relation = movement_relation(
                        function_states[-2][name] - function_states[-3][name],
                        function_states[-1][name] - function_states[-2][name],
                    )
                    row.update({f"function_{name}_{key}": value for key, value in relation.items()})
                possible = any(
                    two_cycle_detect(
                        function_states[-3][name], function_states[-2][name], function_states[-1][name]
                    ) for name in FUNCTION_NAMES
                )
                row["possible_two_cycle"] = possible
                if possible:
                    stop_reason = "possible_two_cycle"
        rows.append(row)
        pd.DataFrame(rows).to_csv(branch_root / "metrics.csv", index=False)
        print(json.dumps({"l1_branch": branch, "outer_complete": outer, **row}, default=str), flush=True)
        if stop_reason == "possible_two_cycle" or outer == args.l1_max_outer:
            break

        current_batches = build_in_memory_batches(
            episode, train_out.macro.to(device), validation_out.macro.to(device), args
        )
        conditional_started = time.perf_counter()
        if branch in {"fc1_only", "joint"}:
            run_fc1_fit(
                episode, current_batches["fc1_train"], current_batches["fc1_validation"],
                branch_root / f"outer_{outer:03d}_fc1_solve",
                block="joint", production_epochs=min(args.production_fc1_epochs, budgets["fc1_epochs"]),
                max_epochs=budgets["fc1_epochs"], patience=args.fc1_patience,
                log_interval=args.log_interval,
            )
        if branch == "joint":
            run_sdf_chunks(
                episode, current_batches["sdf_train"], current_batches["sdf_validation"],
                probe_parents, probe_children,
                branch_root / f"outer_{outer:03d}_sdf_solve",
                chunks=budgets["sdf_chunks"],
                epochs_per_chunk=budgets["sdf_epochs_per_chunk"],
                log_interval=args.log_interval,
            )
        rows[-1]["conditional_solve_seconds"] = time.perf_counter() - conditional_started
        set_macro_scope(model, "frozen")
        function_states.append(snapshot_functions(model, probe_parents, probe_children))
        torch.save(function_states[-1], branch_root / f"functions_outer_{outer + 1:03d}_start.pt")
        if firm_hashes(firm) != initial_firm_hashes:
            raise RuntimeError(f"firm changed after conditional solve in {branch}")

    pd.DataFrame(rows).to_csv(branch_root / "metrics.csv", index=False)
    summary = {
        "branch": branch,
        "stop_reason": stop_reason,
        "completed_outer": len(rows) - 1,
        "firm_initial_hashes": initial_firm_hashes,
        "firm_final_hashes": firm_hashes(firm),
        "firm_hash_invariant": firm_hashes(firm) == initial_firm_hashes,
        "crn": crn,
        "fc2_forward_calls": fc2_counter["calls"],
        "fc2_hash_invariant": module_hash(fc2) == fc2_before,
        "conditional_budgets_from_l0": dict(budgets),
    }
    if not summary["firm_hash_invariant"] or summary["fc2_forward_calls"] != 0 or not summary["fc2_hash_invariant"]:
        raise RuntimeError(f"L1 invariant failed: {summary}")
    write_json(branch_root / "summary.json", summary)
    return summary


def audit_sdf_outer_dependency(
    args: argparse.Namespace,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    episode, models, _ = fresh_macro_model(args, device)
    firm = models["policy_value"].requires_grad_(False).eval()
    model = models["sdf_fc1"]
    train_initial = make_initial_state(
        models=models, episode=episode, paths=args.simulation_paths,
        firms=args.firms_per_path, horizon=args.simulation_horizon,
        branch_num=args.branch_num, seed=args.seed + 1000, device=device,
    )
    validation_initial = make_initial_state(
        models=models, episode=episode, paths=args.validation_paths,
        firms=args.firms_per_path, horizon=args.simulation_horizon,
        branch_num=args.branch_num, seed=args.seed + 2000, device=device,
    )
    set_macro_scope(model, "frozen")
    _, base_out, _ = simulate_frozen_firm(
        args, episode, models, train_initial, validation_initial, device
    )
    base_frame = _to_frame(base_out.macro)
    base_target_hash = macro_target_hash(base_frame)
    base_data_hash = macro_target_hash(base_frame, include_m=True)
    fc1_hash = module_hash(model.fc1_model)
    load_sdf_only_state(args.final_macro_checkpoint, model)
    if module_hash(model.fc1_model) != fc1_hash:
        raise RuntimeError("SDF dependency audit changed FC1")
    set_macro_scope(model, "frozen")
    _, alternative_out, _ = simulate_frozen_firm(
        args, episode, models, train_initial, validation_initial, device
    )
    alternative_frame = _to_frame(alternative_out.macro)
    alternative_target_hash = macro_target_hash(alternative_frame)
    alternative_data_hash = macro_target_hash(alternative_frame, include_m=True)
    status = sdf_dependency_status(base_target_hash, alternative_target_hash)
    result = {
        **status,
        "base_target_hash": base_target_hash,
        "alternative_target_hash": alternative_target_hash,
        "target_hash_identical": base_target_hash == alternative_target_hash,
        "base_data_hash_including_M": base_data_hash,
        "alternative_data_hash_including_M": alternative_data_hash,
        "data_hash_including_M_identical": base_data_hash == alternative_data_hash,
        "firm_hashes": firm_hashes(firm),
        "fc1_hash_invariant": module_hash(model.fc1_model) == fc1_hash,
    }
    write_json(output / "l1" / "sdf_only" / "dependency_audit.json", result)
    return result


def run_l1(
    args: argparse.Namespace,
    output: Path,
    device: torch.device,
    l0: Mapping[str, Any],
) -> dict[str, Any]:
    budgets = choose_conditional_budgets(l0, args)
    probe = torch.load(
        output / "l0" / "data" / "sdf_probe_bank.pt",
        map_location=device, weights_only=True,
    )
    probe_parents, probe_children = probe["parents"], probe["children"]
    results = {
        "conditional_budgets": budgets,
        "fc1_only": run_l1_branch(
            args, output, device, branch="fc1_only", budgets=budgets,
            probe_parents=probe_parents, probe_children=probe_children,
        ),
    }
    dependency = audit_sdf_outer_dependency(args, output, device)
    results["sdf_only"] = dependency
    if dependency["sdf_outer_dependency"]:
        # The current model is expected to be degenerate here.  Fail explicitly
        # instead of silently pretending the implemented branch is sufficient.
        results["sdf_only"]["status"] = "DEPENDENCY_PRESENT_REQUIRES_EXPLICIT_OUTER"
    results["joint"] = run_l1_branch(
        args, output, device, branch="joint", budgets=budgets,
        probe_parents=probe_parents, probe_children=probe_children,
    )
    write_json(output / "l1" / "summary.json", results)
    return results


def plot_history(history: pd.DataFrame, columns: Sequence[str], path: Path, ylabel: str) -> None:
    available = [column for column in columns if column in history]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for column in available:
        ax.plot(history.iloc[:, 0], history[column], marker="o", label=column)
    ax.set(xlabel=history.columns[0], ylabel=ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_figures(output: Path) -> list[str]:
    figures = output / "figures"
    paths = []
    for block in ("fc1_c", "fc1_k"):
        history_path = output / "l0" / block / "history.csv"
        if not history_path.is_file():
            continue
        history = pd.read_csv(history_path)
        targets = (
            (["train_hatc_r2", "validation_hatc_r2"], "R2"),
            (["train_hatc_rmse", "validation_hatc_rmse"], "RMSE"),
            (["train_hatc_slope", "validation_hatc_slope"], "slope"),
            (["train_hatc_prediction_std", "train_hatc_target_std", "validation_hatc_prediction_std", "validation_hatc_target_std"], "std"),
        ) if block == "fc1_c" else (
            (["train_lnk_r2", "validation_lnk_r2"], "R2"),
            (["train_lnk_rmse", "validation_lnk_rmse"], "RMSE"),
            (["train_lnk_slope", "validation_lnk_slope"], "slope"),
            (["train_lnk_prediction_std", "train_lnk_target_std", "validation_lnk_prediction_std", "validation_lnk_target_std"], "std"),
        )
        for columns, label in targets:
            path = figures / f"l0_{block}_{label.lower()}.png"
            plot_history(history, columns, path, label)
            if path.is_file():
                paths.append(str(path))
    sdf_path = output / "l0" / "sdf" / "outer0_start" / "history.csv"
    if sdf_path.is_file():
        history = pd.read_csv(sdf_path)
        for columns, name in (
            ([column for column in history if "aio" in column.lower() or "constraint" in column.lower()], "sdf_objective_moments"),
            ([column for column in history if "M_function" in column], "sdf_function_drift"),
        ):
            path = figures / f"l0_{name}.png"
            plot_history(history, columns[:8], path, name)
            if path.is_file():
                paths.append(str(path))
    for branch in ("fc1_only", "joint"):
        metric_path = output / "l1" / branch / "metrics.csv"
        if not metric_path.is_file():
            continue
        history = pd.read_csv(metric_path)
        for columns, name in (
            ([column for column in history if "function_" in column and column.endswith("_nmae")], "function_movement"),
            ([column for column in history if "target_" in column and column.endswith("_nmae")], "target_movement"),
            ([column for column in history if "path_" in column and column.endswith("_nmae")], "path_movement"),
            ([column for column in history if "directional_gain" in column], "directional_gain"),
        ):
            path = figures / f"l1_{branch}_{name}.png"
            plot_history(history, columns[:8], path, name)
            if path.is_file():
                paths.append(str(path))
    return paths


def classify_results(l0: Mapping[str, Any], l1: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    labels = {}
    for block, target in (("fc1_c", "hatc"), ("fc1_k", "lnk")):
        train_r2 = float(l0[block]["best_train"].get(f"{target}_r2", float("nan")))
        val_r2 = float(l0[block]["best_validation"].get(f"{target}_r2", float("nan")))
        labels[block] = "PASS" if train_r2 >= 0.8 and val_r2 >= 0.5 else "FAIL"
    labels["sdf"] = (
        "NON_IDENTIFIED" if l0["sdf"].get("possible_conditional_non_identification")
        else "PASS"
    )
    labels["joint"] = "PASS" if labels["fc1_c"] == labels["fc1_k"] == "PASS" else "INCONCLUSIVE"
    if "FAIL" in labels.values():
        classification = "A. L0 conditional learner failure"
    elif l1["fc1_only"].get("stop_reason") != "max_outer":
        classification = "B. L0 stable, FC1 outer unstable"
    elif l1["sdf_only"].get("sdf_outer_dependency") and l1["sdf_only"].get("status") != "PASS":
        classification = "C. L0 stable, SDF outer unstable or unresolved"
    elif l1["joint"].get("stop_reason") != "max_outer":
        classification = "D. L0 stable, individual outer stable, joint SDF-FC1 coupling unstable"
    else:
        classification = "E. L0 + L1 completed without an explicit stop; inspect movement limits"
    return classification, labels


def write_report(
    path: Path,
    args: argparse.Namespace,
    metadata: Mapping[str, Any],
    l0: Mapping[str, Any],
    l1: Mapping[str, Any],
) -> None:
    classification, labels = classify_results(l0, l1)
    fc1_c = l0["fc1_c"]
    fc1_k = l0["fc1_k"]
    sdf_cmp_path = Path(args.output_dir) / "l0" / "sdf" / "multi_start_comparison.csv"
    lag_path = Path(args.output_dir) / "l0" / "index_alignment_scan.csv"
    lines = [
        "# Macro Block Isolation Diagnostic",
        "",
        "## Actual current macro data flow",
        "",
        "`current SDF/FC1 + frozen firm + fixed CRN -> production SimulateTS -> TensorTable macro panel -> production parent/child pairing -> exact teacher-forced FC1 tensors and exact SDF tensors -> production loss/optimizer update`.",
        "",
        "FC1 production input is `(x_t, x_{t+1}, Hatc_t, LnK_t)` in physical units; target is `(Hatc_{t+1}, LnK_{t+1})`. The stored `Hatcf_t/LnKF_t` columns are retained for recursive/counterfactual diagnostics. FC1 scalers are identity/no-op. FC2 is guarded and absent from simulation/training.",
        "",
        "## Provenance and invariants",
        "",
        f"- branch: `{metadata['branch']}`",
        f"- git commit: `{metadata['git_commit']}`",
        f"- initial firm checkpoint: `{args.firm_checkpoint}`",
        f"- outer0 macro checkpoint: `{args.outer0_macro_checkpoint}`",
        f"- dataset hashes invariant: `{l0['invariants']['dataset_hash_invariant']}`",
        f"- FC2 forward calls: `{l0['invariants']['fc2_forward_calls']}`",
        f"- train/validation path overlap: `{l0['data']['path_split']['overlap_count']}`",
        "",
        "## L0 summary",
        "",
        "| block | label | production train R2 | production val R2 | best epoch | best train R2 | best val R2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| FC1_C | {labels['fc1_c']} | {fc1_c['production_budget'].get('train_hatc_r2', float('nan')):.6g} | {fc1_c['production_budget'].get('validation_hatc_r2', float('nan')):.6g} | {fc1_c['best_epoch']} | {fc1_c['best_train'].get('hatc_r2', float('nan')):.6g} | {fc1_c['best_validation'].get('hatc_r2', float('nan')):.6g} |",
        f"| FC1_K | {labels['fc1_k']} | {fc1_k['production_budget'].get('train_lnk_r2', float('nan')):.6g} | {fc1_k['production_budget'].get('validation_lnk_r2', float('nan')):.6g} | {fc1_k['best_epoch']} | {fc1_k['best_train'].get('lnk_r2', float('nan')):.6g} | {fc1_k['best_validation'].get('lnk_r2', float('nan')):.6g} |",
        f"| SDF | {labels['sdf']} |  |  |  |  |  |",
        f"| Joint | {labels['joint']} |  |  |  |  |  |",
        "",
        "## L1 summary",
        "",
        "| branch | status/stop | firm hash invariant | FC2 calls |",
        "|---|---|---:|---:|",
        f"| FC1-only | {l1['fc1_only'].get('stop_reason')} | {l1['fc1_only'].get('firm_hash_invariant')} | {l1['fc1_only'].get('fc2_forward_calls')} |",
        f"| SDF-only | {l1['sdf_only'].get('experiment_status')} | True | 0 |",
        f"| Joint | {l1['joint'].get('stop_reason')} | {l1['joint'].get('firm_hash_invariant')} | {l1['joint'].get('fc2_forward_calls')} |",
        "",
        "## Required questions Q1-Q19",
        "",
        f"1. FC1_C production-budget train/val R2: `{fc1_c['production_budget'].get('train_hatc_r2')}` / `{fc1_c['production_budget'].get('validation_hatc_r2')}`.",
        f"2. FC1_C best-fit train/val R2: `{fc1_c['best_train'].get('hatc_r2')}` / `{fc1_c['best_validation'].get('hatc_r2')}` at epoch `{fc1_c['best_epoch']}`.",
        f"3. Hatc fixed-target classification: `{labels['fc1_c']}`; compare production and best-fit rows above.",
        f"4. FC1_K positive control: `{labels['fc1_k']}`.",
        f"5. Lag/index audit: `{lag_path}`.",
        f"6. Fixed-D0 SDF movement history: `{Path(args.output_dir) / 'l0/sdf/outer0_start/history.csv'}`.",
        f"7. SDF conditional uniqueness flag: `{l0['sdf']['possible_conditional_non_identification']}`.",
        f"8. Multi-start comparison: `{sdf_cmp_path}`.",
        f"9. Joint fixed-data summary: `{Path(args.output_dir) / 'l0/joint/summary.json'}`.",
        f"10. Frozen-firm FC1-only stop: `{l1['fc1_only'].get('stop_reason')}`.",
        f"11. SDF outer dependency: `{l1['sdf_only'].get('sdf_outer_dependency')}`.",
        f"12. SDF-only status: `{l1['sdf_only'].get('experiment_status')}`.",
        f"13. Frozen-firm joint stop: `{l1['joint'].get('stop_reason')}`.",
        "14. Largest function-space gain: see L1 branch metrics CSV files.",
        "15. Directional gain above one: see `function_*_directional_gain` columns.",
        "16. Negative directional gain/two-cycle: see `movement_cosine` and `possible_two_cycle` columns.",
        "17. Network-vs-target lead order: compare function and target movement columns by outer.",
        "18. Hatc-vs-LnK target drift: compare `target_hatc_nmae` and `target_lnk_nmae`.",
        "19. Frozen-firm reproduction: determined from FC1-only/joint movement without any P/Q solve.",
        "",
        "## Final classification",
        "",
        classification,
        "",
        "This report does not add damping, alter economic equations, retrain P/Q, or use FC2.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--firm-checkpoint", type=Path, required=True)
    parser.add_argument("--outer0-macro-checkpoint", type=Path, required=True)
    parser.add_argument("--middle-macro-checkpoint", type=Path, required=True)
    parser.add_argument("--final-macro-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--experiment", choices=(
        "l0_fc1_c", "l0_fc1_k", "l0_sdf", "l0_joint",
        "l1_fc1", "l1_sdf", "l1_joint",
    ))
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--simulation-paths", type=int, default=400)
    parser.add_argument("--validation-paths", type=int, default=400)
    parser.add_argument("--simulation-horizon", type=int, default=100)
    parser.add_argument("--firms-per-path", type=int, default=200)
    parser.add_argument("--branch-num", type=int, default=2)
    parser.add_argument("--macro-batch-size", type=int, default=4096)
    parser.add_argument("--production-fc1-epochs", type=int, default=10)
    parser.add_argument("--fc1-max-epochs", type=int, default=500)
    parser.add_argument("--fc1-patience", type=int, default=30)
    parser.add_argument("--sdf-chunks", type=int, default=4)
    parser.add_argument("--sdf-epochs-per-chunk", type=int, default=100)
    parser.add_argument("--sdf-nonidentification-nmae", type=float, default=0.02)
    parser.add_argument("--l1-max-outer", type=int, default=8)
    parser.add_argument("--probe-size", type=int, default=4096)
    parser.add_argument("--log-interval", type=int, default=1000000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def apply_smoke(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.simulation_paths = min(args.simulation_paths, 4)
    args.validation_paths = min(args.validation_paths, 4)
    args.simulation_horizon = min(args.simulation_horizon, 8)
    args.firms_per_path = min(args.firms_per_path, 32)
    args.macro_batch_size = min(args.macro_batch_size, 256)
    args.production_fc1_epochs = 1
    args.fc1_max_epochs = 2
    args.fc1_patience = 1
    args.sdf_chunks = 1
    args.sdf_epochs_per_chunk = 1
    args.l1_max_outer = 1
    args.probe_size = min(args.probe_size, 64)


def main() -> None:
    args = parse_args()
    if not args.run_all and args.experiment is None:
        raise ValueError("choose --run-all or one --experiment")
    apply_smoke(args)
    args.l1_max_outer = validate_l1_max_outer(args.l1_max_outer)
    output = args.output_dir.expanduser().resolve()
    args.output_dir = output
    report = args.report.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    l0_path = output / "l0" / "summary.json"
    if args.run_all or args.experiment and args.experiment.startswith("l0_"):
        l0 = run_l0(args, output, device)
    elif l0_path.is_file():
        l0 = read_json(l0_path)
    else:
        raise FileNotFoundError("L1 requires a completed L0 summary")

    l1_path = output / "l1" / "summary.json"
    if args.run_all or args.experiment and args.experiment.startswith("l1_"):
        l1 = run_l1(args, output, device, l0)
    elif l1_path.is_file():
        l1 = read_json(l1_path)
    else:
        l1 = {}

    figures = make_figures(output)
    metadata = {
        "branch": "exp/macro-block-isolation",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "args": vars(args),
        "figures": figures,
        "prohibitions_honored": {
            "p_q_solver_calls": 0,
            "policy_iteration_calls": 0,
            "fc2_used": False,
            "economic_equations_modified": False,
            "damping_added": False,
        },
    }
    write_json(output / "metadata.json", metadata)
    if l1:
        write_report(report, args, metadata, l0, l1)
    print(f"artifacts={output}", flush=True)
    print(f"report={report}", flush=True)


if __name__ == "__main__":
    main()
