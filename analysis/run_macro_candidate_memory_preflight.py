#!/usr/bin/env python3
"""Run one formal-shape Macro candidate batch without optimizer updates.

This is a GPU-memory regression for the boundary-aware parent/two-child
operator.  It loads a committed checkpoint, evaluates one complete candidate
batch with the production dataset builder, writes auditable coverage/memory
metadata, and performs no training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.macro_parent_state_bank import (  # noqa: E402
    RESTRICTED_BETA_FULL_OUTER_V1,
    build_macro_candidate_schedule,
    build_one_step_macro_dataset,
)
from analysis.run_full_nested_equilibrium import (  # noqa: E402
    _candidate_operator_configs,
    _setup_fresh,
    parse_args as parse_full_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--parent-batch-size", type=int, default=32)
    parser.add_argument("--firms-per-parent", type=int, default=200)
    parser.add_argument("--candidate-h-microbatch-size", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.parent_batch_size != args.candidate_count:
        raise ValueError("preflight must evaluate the complete candidate set in one batch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("memory preflight requires CUDA")

    full_args = parse_full_args(
        [
            "--cold-start",
            "--initial-checkpoint", str(checkpoint),
            "--device", str(device),
            "--output-dir", str(output / "runner_scratch"),
            "--seed", str(args.seed),
            "--firms-per-path", str(args.firms_per_parent),
            "--macro-parent-batch-size", str(args.parent_batch_size),
            "--macro-clearing-solver-mode", "fc1_bounded_improvement_pilot",
            "--boundary-candidate-h-microbatch-size",
            str(args.candidate_h_microbatch_size),
            "--aggregation-mode", "continuum_mass_consistent_v1",
            "--disable-fc2",
            "--no-run-long-diagnostics",
            "--no-use-long-rollout-for-macro-training",
        ]
    )
    episode, _, models = _setup_fresh(full_args, device)
    integration, root = _candidate_operator_configs(full_args)
    schedule = build_macro_candidate_schedule(
        split="gpu_memory_preflight",
        namespace="formal_shape_gpu_memory_preflight_v1",
        target_valid=1,
        initial_candidate_budget=args.candidate_count,
        max_candidate_budget=args.candidate_count,
        minimum_partial_valid=1,
        firms_per_parent=args.firms_per_parent,
        seed=args.seed + 30000,
        candidate_id_offset=2_000_000_000,
        max_initial_weighted_b_mean=0.30,
        distribution_design=RESTRICTED_BETA_FULL_OUTER_V1,
    )

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    dataset = build_one_step_macro_dataset(
        models=models,
        episode=episode,
        candidate_schedule=schedule,
        parent_batch_size=args.parent_batch_size,
        macro_batch_size=4096,
        device=device,
        output_dir=output / "candidate_data",
        allow_replenishment=False,
        integration=integration,
        root_config=root,
        source_checkpoint_provenance={
            "policy": str(checkpoint),
            "fc1_sdf": str(checkpoint),
            "purpose": "GPU_MEMORY_PREFLIGHT_NO_TRAINING",
        },
        initial_total_k_min=45.0,
        initial_total_k_max=85.0,
    )
    elapsed = time.perf_counter() - started
    summary = {
        "status": "PASS",
        "training_performed": False,
        "checkpoint": str(checkpoint),
        "candidate_count": args.candidate_count,
        "parent_batch_size": args.parent_batch_size,
        "firms_per_parent": args.firms_per_parent,
        "candidate_h_microbatch_size": args.candidate_h_microbatch_size,
        "dataset_status": dataset.dataset_status,
        "accepted_training_pair_count": int(
            dataset.pairing_audit.get("candidate_coverage", {}).get(
                "accepted_training_pair_count", 0
            )
        ),
        "elapsed_seconds": elapsed,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
