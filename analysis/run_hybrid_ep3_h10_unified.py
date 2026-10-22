"""Plan-only EP3-H10 unified two-Outer runner.

This file wires the reviewed experiment and refuses production execution until
an explicit content-bound approval is supplied.  It intentionally contains no
optimizer call in plan mode.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.hybrid_ep3_h10_unified import (
    EXPERIMENT_ID,
    HYBRID_OUTER_COUNT,
    SCHEMA,
    HybridFresh100Contract,
    HybridMacroMixtureContract,
    hybrid_stage_plan,
    stage_id,
    training_plan,
)
from analysis.nested_equilibrium_utils import atomic_write_json, stable_json_hash
from data.finite_direct_sampling import file_hash


def approval_binding(args: argparse.Namespace, commit: str) -> dict:
    """Content identity for a future explicit user approval.

    No approved file is created by this implementation.
    """

    if args.sampling_dir is None or args.initial_checkpoint is None:
        raise ValueError("sampling directory and converted EP3 checkpoint are required")
    signature_path = args.sampling_dir / "sampling_signature.json"
    signature = json.loads(signature_path.read_text())
    required = (
        "sampling_signature.json",
        "train_initial_bank.pt",
        "validation_initial_bank.pt",
    )
    return {
        "schema_name": SCHEMA,
        "schema_version": 1,
        "approved": True,
        "source_commit": commit,
        "experiment_id": EXPERIMENT_ID,
        "plan_fingerprint": stable_json_hash(training_plan()),
        "fresh100_contract": asdict(HybridFresh100Contract()),
        "macro_mixture_contract": asdict(HybridMacroMixtureContract()),
        "sampling_signature_fingerprint": signature["fingerprint"],
        "sampling_signature_sha256": file_hash(signature_path),
        "input_artifact_sha256": {
            name: file_hash(args.sampling_dir / name) for name in required
        },
        "initial_checkpoint_sha256": file_hash(args.initial_checkpoint),
        "initial_checkpoint_format": "finite_direct_outer4_initial_weights_v1",
        "initialization": "EP3_WEIGHTS_ONLY_NEW_OPTIMIZERS",
        "operator": "finite_direct_parent_exit_before_child_v2",
        "outer_count": HYBRID_OUTER_COUNT,
    }


def require_approval(args: argparse.Namespace, commit: str) -> bool:
    if not args.execute_approved:
        return False
    if args.approval is None:
        raise ValueError("explicit hybrid approval file required")
    supplied = json.loads(args.approval.read_text())
    expected = approval_binding(args, commit)
    if any(supplied.get(key) != value for key, value in expected.items()):
        raise ValueError("hybrid approval is not bound to the exact source/plan/artifacts")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise ValueError("clean source checkout required")
    return True


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--execute-approved", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parent-batch-size", type=int, default=128)
    parser.add_argument("--fit-parent-batch-size", type=int, default=128)
    parser.add_argument("--firm-batch-size", type=int, default=512)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if not require_approval(args, commit):
        args.output.mkdir(parents=True, exist_ok=True)
        plan = training_plan()
        atomic_write_json(args.output / "training_plan.json", plan)
        print(
            json.dumps(
                {
                    "CURRENT_PHASE": "WAITING_FOR_USER_REVIEW",
                    "FORMAL_TRAINING_STARTED": False,
                    "SLURM_JOB_SUBMITTED": False,
                    "APPROVAL_CREATED": False,
                    "plan_fingerprint": stable_json_hash(plan),
                },
                indent=2,
            )
        )
        return 0

    # Production execution is intentionally not implemented in this review
    # commit.  An approved execution must be wired only after the user reviews
    # the exact data/Q/BP contracts and tests.
    raise RuntimeError(
        "HYBRID_EXECUTION_NOT_IMPLEMENTED; this review commit is plan-only"
    )


if __name__ == "__main__":
    raise SystemExit(main())
