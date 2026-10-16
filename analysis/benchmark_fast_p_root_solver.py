#!/usr/bin/env python3
"""Benchmark the legacy and vectorized frozen-policy P residual engines."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence
from unittest.mock import patch

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.analyze_p_only_movement_channels import (  # noqa: E402
    _load_policy_state,
    _load_setup,
)
from analysis.test_frozen_policy_root_solve import (  # noqa: E402
    FrozenPolicyBatch,
    _clone_state,
    _p_scope,
    build_problem,
    evaluate_problem,
    residual_tensors,
    root_loss,
    run_lbfgs_chunked,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _gpu_utilization() -> float:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return float(raw.strip().splitlines()[0])
    except Exception:
        return float("nan")


def _forward_calls(
    model: torch.nn.Module,
    batch: FrozenPolicyBatch,
    implementation: str,
) -> int:
    original = model._value_outputs
    count = 0

    def counted(states: torch.Tensor):
        nonlocal count
        count += 1
        return original(states)

    with patch.object(model, "_value_outputs", side_effect=counted):
        root_loss(model, batch, implementation=implementation)
    return count


def _profile_engine(
    *,
    model: torch.nn.Module,
    episode: Any,
    train: Sequence[FrozenPolicyBatch],
    validation: Sequence[FrozenPolicyBatch],
    implementation: str,
    repeats: int,
    batch_size: int,
) -> dict[str, Any]:
    device = train[0].parent.device
    params, _ = _p_scope(episode, model)
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        value = root_loss(model, train[0], implementation=implementation)
        value.backward()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    forward_times: list[float] = []
    backward_times: list[float] = []
    for _ in range(repeats):
        model.zero_grad(set_to_none=True)
        _sync(device)
        started = time.perf_counter()
        value = root_loss(model, train[0], implementation=implementation)
        _sync(device)
        forward_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        value.backward()
        _sync(device)
        backward_times.append(time.perf_counter() - started)

    model.zero_grad(set_to_none=True)
    util_before = _gpu_utilization()
    _sync(device)
    started = time.perf_counter()
    closure_loss = torch.zeros((), device=device)
    for batch in train:
        value = root_loss(model, batch, implementation=implementation) / len(train)
        value.backward()
        closure_loss = closure_loss + value.detach()
    _sync(device)
    closure_seconds = time.perf_counter() - started
    util_after = _gpu_utilization()

    _sync(device)
    started = time.perf_counter()
    validation_metrics = evaluate_problem(model, validation)
    _sync(device)
    validation_seconds = time.perf_counter() - started
    peak = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
        if device.type == "cuda" else float("nan")
    )
    return {
        "implementation": implementation,
        "batch_size": batch_size,
        "train_batches": len(train),
        "validation_batches": len(validation),
        "value_forward_calls_per_batch": _forward_calls(model, train[0], implementation),
        "forward_residual_seconds": sum(forward_times) / len(forward_times),
        "backward_seconds": sum(backward_times) / len(backward_times),
        "full_bank_closure_seconds": closure_seconds,
        "validation_seconds": validation_seconds,
        "peak_vram_mib": peak,
        "gpu_utilization_before": util_before,
        "gpu_utilization_after": util_after,
        "closure_loss": float(closure_loss),
        "validation_residual": validation_metrics["normalized_mae"],
        "parameter_count": sum(param.numel() for param in params),
    }


def _parity(
    model: torch.nn.Module,
    episode: Any,
    batch: FrozenPolicyBatch,
) -> dict[str, Any]:
    old_model = deepcopy(model)
    new_model = deepcopy(model)
    old_params, old_names = _p_scope(episode, old_model)
    new_params, new_names = _p_scope(episode, new_model)
    if old_names != new_names:
        raise RuntimeError("P trainable parameter names differ across parity models")
    old = residual_tensors(old_model, batch, implementation="legacy")
    new = residual_tensors(new_model, batch, implementation="fast")
    old_loss = root_loss(old_model, batch, implementation="legacy")
    new_loss = root_loss(new_model, batch, implementation="fast")
    old_loss.backward()
    new_loss.backward()
    grad_max = 0.0
    grad_l2_sq = 0.0
    for old_param, new_param in zip(old_params, new_params):
        if old_param.grad is None and new_param.grad is None:
            continue
        if old_param.grad is None or new_param.grad is None:
            raise RuntimeError("Gradient support differs between residual engines")
        difference = old_param.grad - new_param.grad
        grad_max = max(grad_max, float(difference.abs().max()))
        grad_l2_sq += float(difference.square().sum())
    cached = residual_tensors(model, batch, implementation="fast", use_cache=True)
    uncached = residual_tensors(model, batch, implementation="fast", use_cache=False)
    return {
        "p0_residual_max_abs_error": float((old["p0"] - new["p0"]).abs().max()),
        "pi_residual_max_abs_error": float((old["pi"] - new["pi"]).abs().max()),
        "root_loss_abs_difference": float((old_loss - new_loss).abs()),
        "gradient_max_abs_difference": grad_max,
        "gradient_l2_difference": math.sqrt(grad_l2_sq),
        "cached_noncached_p0_max_abs_error": float(
            (cached["p0"] - uncached["p0"]).abs().max()
        ),
        "cached_noncached_pi_max_abs_error": float(
            (cached["pi"] - uncached["pi"]).abs().max()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--sweep", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-limit", type=int, default=8192)
    parser.add_argument("--validation-limit", type=int, default=2048)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--profile-repeats", type=int, default=5)
    parser.add_argument("--lbfgs-iterations", type=int, default=20)
    parser.add_argument("--lbfgs-history-size", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(20261016)
    device = torch.device(args.device)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    run = args.run.expanduser().resolve()
    checkpoint = run / "checkpoints" / f"sweep_{args.sweep:03d}.pt"
    episode, _, models, _, _, _, _, provenance, _ = _load_setup(
        run, device, None, args.source_run, args.source_checkpoint
    )
    model = models["policy_value"]
    payload = _load_policy_state(model, checkpoint, device)
    initial_state = _clone_state(model)
    train_payload = torch.load(
        run / "fixed_point/collocation_train.pt", map_location="cpu", weights_only=False
    )
    validation_payload = torch.load(
        run / "fixed_point/collocation_val.pt", map_location="cpu", weights_only=False
    )
    child_payload = torch.load(
        run / "fixed_point/child_shock_bank.pt", map_location="cpu", weights_only=False
    )
    n_train = min(args.train_limit, train_payload["parent_states"].shape[0])
    n_validation = min(
        args.validation_limit, validation_payload["parent_states"].shape[0]
    )
    train_parent = train_payload["parent_states"][:n_train].to(device)
    validation_parent = validation_payload["parent_states"][:n_validation].to(device)
    train_child = child_payload["train"]["children"][:n_train].to(device)
    validation_child = child_payload["validation"]["children"][:n_validation].to(device)

    profile_rows: list[dict[str, Any]] = []
    lbfgs_rows: list[dict[str, Any]] = []
    parity_row: dict[str, Any] | None = None
    for batch_size in args.batch_sizes:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            train = build_problem(
                episode,
                model,
                train_parent,
                train_child,
                batch_size,
                snapshot_batch_size=512,
            )
            validation = build_problem(
                episode,
                model,
                validation_parent,
                validation_child,
                batch_size,
                snapshot_batch_size=512,
            )
            if parity_row is None:
                parity_row = _parity(model, episode, train[0])
            implementations = ["fast"]
            if batch_size == args.batch_sizes[0]:
                implementations.insert(0, "legacy")
            for implementation in implementations:
                profile_model = deepcopy(model).to(device)
                profile_rows.append(_profile_engine(
                    model=profile_model,
                    episode=episode,
                    train=train,
                    validation=validation,
                    implementation=implementation,
                    repeats=args.profile_repeats,
                    batch_size=batch_size,
                ))
                solve_model = deepcopy(model).to(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                initial = evaluate_problem(solve_model, validation)
                history, best_state, best, stop_reason = run_lbfgs_chunked(
                    model=solve_model,
                    episode=episode,
                    train=train,
                    validation=validation,
                    max_iter=args.lbfgs_iterations,
                    chunk_iterations=args.lbfgs_iterations,
                    history_size=args.lbfgs_history_size,
                    implementation=implementation,
                )
                elapsed = sum(float(row["seconds"]) for row in history)
                closures = sum(int(row["block_closure_calls"]) for row in history)
                peak = (
                    float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
                    if device.type == "cuda" else float("nan")
                )
                lbfgs_rows.append({
                    "implementation": implementation,
                    "batch_size": batch_size,
                    "optimizer_iterations_budget": args.lbfgs_iterations,
                    "closure_calls": closures,
                    "seconds": elapsed,
                    "seconds_per_closure": elapsed / max(closures, 1),
                    "peak_vram_mib": peak,
                    "initial_residual": initial["normalized_mae"],
                    "best_residual": best["normalized_mae"],
                    "residual_reduction": initial["normalized_mae"] - best["normalized_mae"],
                    "residual_reduction_per_minute": (
                        (initial["normalized_mae"] - best["normalized_mae"])
                        / max(elapsed / 60.0, 1e-12)
                    ),
                    "stop_reason": stop_reason,
                })
                del solve_model, profile_model, best_state
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            profile_rows.append({
                "implementation": "fast",
                "batch_size": batch_size,
                "status": "oom",
                "error": str(exc),
            })

    if parity_row is None:
        raise RuntimeError("No batch size completed; parity was not evaluated")
    pd.DataFrame([parity_row]).to_csv(output / "p_numerical_parity.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(output / "p_speed_profile.csv", index=False)
    pd.DataFrame(lbfgs_rows).to_csv(output / "p_lbfgs_benchmark.csv", index=False)
    metadata = {
        "args": vars(args),
        "checkpoint": str(checkpoint),
        "checkpoint_stage": payload.get("stage"),
        "source_git_commit": provenance.get("source_git_commit"),
        "initial_state_loaded": bool(initial_state),
        "train_rows": n_train,
        "validation_rows": n_validation,
        "train_parent_bank_fingerprint": train_payload.get("fingerprint"),
        "validation_parent_bank_fingerprint": validation_payload.get("fingerprint"),
        "train_child_bank_fingerprint": child_payload["train"].get("fingerprint"),
        "validation_child_bank_fingerprint": child_payload["validation"].get("fingerprint"),
    }
    (output / "p_speed_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str, sort_keys=True), encoding="utf-8"
    )
    print(pd.DataFrame([parity_row]).to_string(index=False))
    print(pd.DataFrame(profile_rows).to_string(index=False))
    print(pd.DataFrame(lbfgs_rows).to_string(index=False))
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
