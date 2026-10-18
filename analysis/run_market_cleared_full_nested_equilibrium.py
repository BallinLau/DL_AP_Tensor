#!/usr/bin/env python3
"""Run Tiny -> Small -> Formal market-cleared nested-equilibrium stages.

Each stage executes the real Firm, root-cleared simulation, FC1, and SDF blocks
through ``run_full_nested_equilibrium.py``.  The chain advances only after a
structural/economic stage gate succeeds.  No Git operation is performed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.market_cleared_reporting import (
    generate_chain_report,
    generate_stage_report,
)

NESTED_RUNNER = ROOT / "analysis" / "run_full_nested_equilibrium.py"
EXPERIMENT_NAME = "market_cleared_full_nested_equilibrium_20261019"
STAGES = ("tiny", "small", "formal")


def _common() -> list[str]:
    return [
        "--market-clearing-mode", "market_cleared_experimental_v1",
        "--market-clearing-coarse-points", "10",
        "--market-clearing-refinement-points", "20",
        "--market-clearing-max-refinement-rounds", "4",
        "--market-clearing-support-lower", "-8.0",
        "--market-clearing-support-upper", "2.0",
        "--market-clearing-residual-tolerance", "1e-5",
        "--market-clearing-width-tolerance", "1e-5",
        "--market-clearing-candidate-chunk-size", "262144",
        "--branch-num", "2",
        "--firms-per-path", "200",
        "--disable-fc2",
        "--sdf-recursive-epochs", "0",
    ]


def _stage_args(stage: str) -> list[str]:
    if stage == "tiny":
        return [
            "--equilibrium-outer-max", "1",
            "--firm-inner-max", "1",
            "--p-policy-outer-max", "1",
            "--q-picard-outer-max", "1",
            "--macro-inner-max", "1",
            "--bootstrap-paths", "8",
            "--bootstrap-validation-paths", "4",
            "--bootstrap-horizon", "5",
            "--train-limit", "1024",
            "--validation-limit", "512",
            "--child-shocks", "2",
            "--macro-bank-candidate-paths", "8",
            "--macro-bank-validation-paths", "4",
            "--macro-bank-rollout-horizon", "5",
            "--macro-bank-capture-times", "0,1,2,3,4",
            "--macro-train-parent-count", "16",
            "--macro-validation-parent-count", "8",
            "--macro-parent-batch-size", "8",
            "--macro-bank-shard-size", "8",
            "--long-diagnostic-paths", "4",
            "--long-diagnostic-horizon", "5",
            "--macro-probe-size", "32",
            "--p-batch-size", "1024",
            "--q-batch-size", "1024",
            "--macro-batch-size", "32",
            "--p-adam-steps", "2",
            "--p-eval-every", "1",
            "--p-lbfgs-max-iter", "2",
            "--p-lbfgs-chunk-iterations", "1",
            "--q-fit-steps", "2",
            "--fc1-production-epochs", "1",
            "--fc1-max-epochs", "1",
            "--fc1-patience", "1",
            "--sdf-chunks", "1",
            "--sdf-epochs-per-chunk", "1",
            "--policy-distill-max-epochs", "1",
            "--policy-distill-max-steps", "2",
            "--policy-distill-patience", "1",
        ]
    if stage == "small":
        return [
            "--equilibrium-outer-max", "2",
            "--firm-inner-max", "3",
            "--p-policy-outer-max", "3",
            "--q-picard-outer-max", "3",
            "--macro-inner-max", "2",
            "--bootstrap-paths", "128",
            "--bootstrap-validation-paths", "32",
            "--bootstrap-horizon", "10",
            "--train-limit", "4096",
            "--validation-limit", "1024",
            "--child-shocks", "2",
            "--macro-bank-candidate-paths", "128",
            "--macro-bank-validation-paths", "32",
            "--macro-bank-rollout-horizon", "10",
            "--macro-bank-capture-times", "0,2,4,6,8",
            "--macro-train-parent-count", "512",
            "--macro-validation-parent-count", "128",
            "--macro-parent-batch-size", "64",
            "--macro-bank-shard-size", "64",
            "--long-diagnostic-paths", "32",
            "--long-diagnostic-horizon", "10",
            "--macro-probe-size", "512",
            "--p-batch-size", "4096",
            "--q-batch-size", "4096",
            "--macro-batch-size", "1024",
            "--p-adam-steps", "20",
            "--p-eval-every", "5",
            "--p-lbfgs-max-iter", "20",
            "--p-lbfgs-chunk-iterations", "5",
            "--q-fit-steps", "20",
            "--fc1-production-epochs", "5",
            "--fc1-max-epochs", "10",
            "--fc1-patience", "4",
            "--sdf-chunks", "2",
            "--sdf-epochs-per-chunk", "10",
            "--policy-distill-max-epochs", "100",
            "--policy-distill-max-steps", "100",
            "--policy-distill-patience", "10",
        ]
    if stage == "formal":
        return [
            "--equilibrium-outer-max", "6",
            "--firm-inner-max", "6",
            "--p-policy-outer-max", "6",
            "--q-picard-outer-max", "6",
            "--macro-inner-max", "4",
            "--bootstrap-paths", "400",
            "--bootstrap-validation-paths", "100",
            "--bootstrap-horizon", "20",
            "--train-limit", "8192",
            "--validation-limit", "2048",
            "--child-shocks", "2",
            "--macro-bank-candidate-paths", "400",
            "--macro-bank-validation-paths", "100",
            "--macro-bank-rollout-horizon", "20",
            "--macro-bank-capture-times", "0,4,8,12,16",
            "--macro-train-parent-count", "4096",
            "--macro-validation-parent-count", "1024",
            "--macro-parent-batch-size", "256",
            "--macro-bank-shard-size", "128",
            "--long-diagnostic-paths", "100",
            "--long-diagnostic-horizon", "20",
            "--macro-probe-size", "2048",
            "--p-batch-size", "8192",
            "--q-batch-size", "8192",
            "--macro-batch-size", "4096",
            "--p-adam-steps", "100",
            "--p-eval-every", "10",
            "--p-lbfgs-max-iter", "100",
            "--p-lbfgs-chunk-iterations", "20",
            "--q-fit-steps", "100",
            "--fc1-production-epochs", "10",
            "--fc1-max-epochs", "100",
            "--fc1-patience", "20",
            "--sdf-chunks", "4",
            "--sdf-epochs-per-chunk", "50",
            "--policy-distill-max-epochs", "1000",
            "--policy-distill-max-steps", "1000",
            "--policy-distill-patience", "20",
        ]
    raise ValueError(stage)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_gate(stage_root: Path, stage: str) -> dict[str, Any]:
    reasons: list[str] = []
    final_summary_path = stage_root / "reports" / "final_summary.json"
    final_checkpoint = stage_root / "checkpoints" / "final.pt"
    fixed_metadata_path = stage_root / "fixed_point" / "metadata.json"
    summary_path = stage_root / "summary.csv"
    if not final_summary_path.is_file():
        reasons.append("missing final_summary.json")
    if not final_checkpoint.is_file():
        reasons.append("missing final checkpoint")
    if not fixed_metadata_path.is_file():
        reasons.append("missing fixed Firm bank metadata")
    required_stages = {"firm_inner", "macro_inner", "equilibrium_outer"}
    observed_stages: set[str] = set()
    if summary_path.is_file():
        summary = pd.read_csv(summary_path)
        if "stage" in summary:
            observed_stages = set(summary["stage"].dropna().astype(str))
        missing_stages = required_stages - observed_stages
        if missing_stages:
            reasons.append(f"missing executed stages {sorted(missing_stages)}")
    else:
        reasons.append("missing summary.csv")

    final_summary = _read_json(final_summary_path) if final_summary_path.is_file() else {}
    if final_summary and (
        int(final_summary.get("fc2_forward_calls", -1)) != 0
        or not bool(final_summary.get("fc2_hash_invariant", False))
    ):
        reasons.append("FC2 invariant failed")
    fixed_metadata = _read_json(fixed_metadata_path) if fixed_metadata_path.is_file() else {}
    if fixed_metadata and fixed_metadata.get("source") != (
        "root_cleared_full_economy_matched_transitions"
    ):
        reasons.append("Firm bank is not from matched root-cleared transitions")
    root_summaries = []
    if fixed_metadata:
        for split in ("train_simulation", "validation_simulation"):
            market = fixed_metadata.get(split, {}).get("market_clearing", {})
            root_summaries.append(market)
            if market.get("mode") != "market_cleared_experimental_v1":
                reasons.append(f"{split} did not use market clearing")
            if int(market.get("failure_count", -1)) != 0:
                reasons.append(f"{split} contains root failures")
            if int(market.get("node_count", 0)) < 1:
                reasons.append(f"{split} contains no cleared node")
    gate = {
        "stage": stage,
        "passed": not reasons,
        "reasons": reasons,
        "observed_stages": sorted(observed_stages),
        "final_summary": final_summary,
        "root_summaries": root_summaries,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    (stage_root / "reports").mkdir(parents=True, exist_ok=True)
    (stage_root / "reports" / "stage_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
    )
    return gate


def _profile(root: Path) -> dict[str, Any]:
    stages = {}
    for stage in ("tiny", "small"):
        summary = _read_json(root / stage / "reports" / "final_summary.json")
        frame = pd.read_csv(root / stage / "summary.csv")
        stages[stage] = {
            "total_elapsed": float(summary["total_elapsed"]),
            "peak_gpu_memory": float(
                pd.to_numeric(frame.get("GPU_max_memory"), errors="coerce").max()
            ),
            "rows": int(len(frame)),
        }
    disk = shutil.disk_usage(root)
    estimate_ratio = (
        stages["small"]["total_elapsed"] / max(stages["tiny"]["total_elapsed"], 1.0)
    )
    profile = {
        "stages": stages,
        "small_to_tiny_runtime_ratio": estimate_ratio,
        "formal_runtime_lower_bound_seconds": stages["small"]["total_elapsed"] * 3.0,
        "disk_free_bytes": int(disk.free),
        "disk_total_bytes": int(disk.total),
        "formal_allowed": bool(disk.free >= 10 * 1024**3),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (root / "profiling_before_formal.json").write_text(
        json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8"
    )
    return profile


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "cachedir" / EXPERIMENT_NAME,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--through", choices=STAGES, default="formal")
    parser.add_argument("--start-at", choices=STAGES, default="tiny")
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    selected = STAGES[STAGES.index(args.start_at) : STAGES.index(args.through) + 1]
    warm_start = (
        args.initial_checkpoint.expanduser().resolve()
        if args.initial_checkpoint is not None
        else None
    )
    chain = {
        "experiment": EXPERIMENT_NAME,
        "source_root": str(ROOT),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "stages": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    for stage in selected:
        stage_root = output / stage
        stage_root.mkdir(parents=True, exist_ok=True)
        final_summary = stage_root / "reports" / "final_summary.json"
        resume_pointer = stage_root / "latest_checkpoint.json"
        command = [sys.executable, str(NESTED_RUNNER)]
        if final_summary.is_file():
            command = []
        elif resume_pointer.is_file():
            command += ["--resume", str(resume_pointer)]
        else:
            command += ["--cold-start"]
        if command:
            command += [
                "--output-dir", str(stage_root),
                "--device", args.device,
                "--seed", str(args.seed),
                *_common(),
                *_stage_args(stage),
            ]
            if warm_start is not None:
                command += ["--initial-checkpoint", str(warm_start)]
        command_path = output / "logs" / f"{stage}_command.json"
        command_path.write_text(
            json.dumps(command, indent=2), encoding="utf-8"
        )
        if args.dry_run:
            chain["stages"].append({"stage": stage, "command": command})
            continue
        started = time.time()
        if command:
            log_path = output / "logs" / f"{stage}.log"
            with log_path.open("ab") as log:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            return_code = int(completed.returncode)
        else:
            return_code = 0
        gate = _stage_gate(stage_root, stage) if return_code == 0 else {
            "stage": stage,
            "passed": False,
            "reasons": [f"runner return code {return_code}"],
        }
        chain["stages"].append(
            {
                "stage": stage,
                "return_code": return_code,
                "elapsed_seconds": time.time() - started,
                "gate": gate,
            }
        )
        (output / "chain_status.json").write_text(
            json.dumps(chain, indent=2, sort_keys=True), encoding="utf-8"
        )
        if return_code != 0 or not gate["passed"]:
            print(
                f"MARKET_CLEARED_CHAIN_BLOCKED stage={stage} "
                f"reasons={gate['reasons']}",
                flush=True,
            )
            return 2
        try:
            report_metrics = generate_stage_report(stage_root)
        except Exception as error:
            chain["stages"][-1]["report_error"] = (
                f"{type(error).__name__}: {error}"
            )
            (output / "chain_status.json").write_text(
                json.dumps(chain, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(
                f"MARKET_CLEARED_CHAIN_BLOCKED stage={stage}_report "
                f"reasons={[chain['stages'][-1]['report_error']]}",
                flush=True,
            )
            return 4
        chain["stages"][-1]["report_metrics"] = report_metrics
        warm_start = stage_root / "checkpoints" / "final.pt"
        if stage == "small" and "formal" in selected:
            profile = _profile(output)
            if not profile["formal_allowed"]:
                print(
                    "MARKET_CLEARED_CHAIN_BLOCKED stage=profiling "
                    "reasons=['less than 10 GiB free']",
                    flush=True,
                )
                return 3
    chain["completed_at"] = datetime.now(timezone.utc).isoformat()
    chain["status"] = "COMPLETED"
    generate_chain_report(output)
    (output / "chain_status.json").write_text(
        json.dumps(chain, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"MARKET_CLEARED_CHAIN_ROOT={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
