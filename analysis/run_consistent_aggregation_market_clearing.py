#!/usr/bin/env python3
"""Gate and run the continuum Tiny -> Small equilibrium experiment.

The chain never starts Formal.  Every stage is a cold experiment with an
explicit weights-only warm start, fresh banks/targets, and reset optimizer/AL
state.  Existing stage checkpoints are used only for crash-safe same-stage
resume after provenance validation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC = ROOT / "analysis" / "run_consistent_aggregation_diagnostics.py"
NESTED = ROOT / "analysis" / "run_full_nested_equilibrium.py"


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
        "--aggregation-mode", "continuum_mass_consistent_v1",
        "--continuum-i-points", "9",
        "--continuum-z-points", "7",
        "--continuum-projection-tolerance", "1e-6",
        "--bp-actual-use-trace",
        "--branch-num", "2",
        "--firms-per-path", "200",
        "--disable-fc2",
        "--sdf-recursive-epochs", "0",
    ]


def _stage_args(stage: str) -> list[str]:
    if stage == "tiny":
        return [
            "--equilibrium-outer-max", "1", "--firm-inner-max", "1",
            "--p-policy-outer-max", "1", "--q-picard-outer-max", "1",
            "--macro-inner-max", "1", "--bootstrap-paths", "8",
            "--bootstrap-validation-paths", "4", "--bootstrap-horizon", "5",
            "--train-limit", "1024", "--validation-limit", "512",
            "--child-shocks", "2", "--macro-bank-candidate-paths", "8",
            "--macro-bank-validation-paths", "4", "--macro-bank-rollout-horizon", "5",
            "--macro-bank-capture-times", "0,1,2,3,4",
            "--macro-train-parent-count", "16", "--macro-validation-parent-count", "8",
            "--macro-parent-batch-size", "8", "--macro-bank-shard-size", "8",
            "--long-diagnostic-paths", "4", "--long-diagnostic-horizon", "5",
            "--macro-probe-size", "32", "--p-batch-size", "1024",
            "--q-batch-size", "1024", "--macro-batch-size", "32",
            "--p-adam-steps", "2", "--p-eval-every", "1",
            "--p-lbfgs-max-iter", "2", "--p-lbfgs-chunk-iterations", "1",
            "--q-fit-steps", "2", "--fc1-production-epochs", "1",
            "--fc1-max-epochs", "1", "--fc1-patience", "1",
            "--sdf-chunks", "1", "--sdf-epochs-per-chunk", "1",
            "--policy-distill-max-epochs", "1", "--policy-distill-max-steps", "2",
            "--policy-distill-patience", "1",
        ]
    if stage == "small":
        return [
            "--equilibrium-outer-max", "2", "--firm-inner-max", "3",
            "--p-policy-outer-max", "3", "--q-picard-outer-max", "3",
            "--macro-inner-max", "2", "--bootstrap-paths", "128",
            "--bootstrap-validation-paths", "32", "--bootstrap-horizon", "10",
            "--train-limit", "4096", "--validation-limit", "1024",
            "--child-shocks", "2", "--macro-bank-candidate-paths", "128",
            "--macro-bank-validation-paths", "32", "--macro-bank-rollout-horizon", "10",
            "--macro-bank-capture-times", "0,2,4,6,8",
            "--macro-train-parent-count", "512", "--macro-validation-parent-count", "128",
            "--macro-parent-batch-size", "64", "--macro-bank-shard-size", "64",
            "--long-diagnostic-paths", "32", "--long-diagnostic-horizon", "10",
            "--macro-probe-size", "512", "--p-batch-size", "4096",
            "--q-batch-size", "4096", "--macro-batch-size", "1024",
            "--p-adam-steps", "20", "--p-eval-every", "5",
            "--p-lbfgs-max-iter", "20", "--p-lbfgs-chunk-iterations", "5",
            "--q-fit-steps", "20", "--fc1-production-epochs", "5",
            "--fc1-max-epochs", "10", "--fc1-patience", "4",
            "--sdf-chunks", "2", "--sdf-epochs-per-chunk", "10",
            "--policy-distill-max-epochs", "100", "--policy-distill-max-steps", "100",
            "--policy-distill-patience", "10",
        ]
    raise ValueError(stage)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    return int(completed.returncode)


def _stage_gate(stage_root: Path, stage: str) -> dict[str, Any]:
    reasons: list[str] = []
    final_summary_path = stage_root / "reports" / "final_summary.json"
    final_checkpoint = stage_root / "checkpoints" / "final.pt"
    fixed_metadata_path = stage_root / "fixed_point" / "metadata.json"
    summary_path = stage_root / "summary.csv"
    for path, label in (
        (final_summary_path, "final_summary.json"),
        (final_checkpoint, "final checkpoint"),
        (fixed_metadata_path, "fixed Firm bank metadata"),
        (summary_path, "summary.csv"),
    ):
        if not path.is_file():
            reasons.append(f"missing {label}")
    observed: set[str] = set()
    if summary_path.is_file():
        frame = pd.read_csv(summary_path)
        if "stage" in frame:
            observed = set(frame["stage"].dropna().astype(str))
        missing = {"firm_inner", "macro_inner", "equilibrium_outer"} - observed
        if missing:
            reasons.append(f"missing executed stages {sorted(missing)}")
    final_summary = _read_json(final_summary_path) if final_summary_path.is_file() else {}
    if final_summary and (
        int(final_summary.get("fc2_forward_calls", -1)) != 0
        or not bool(final_summary.get("fc2_hash_invariant", False))
    ):
        reasons.append("FC2 invariant failed")
    fixed = _read_json(fixed_metadata_path) if fixed_metadata_path.is_file() else {}
    if fixed and fixed.get("source") != "root_cleared_full_economy_matched_transitions":
        reasons.append("Firm bank is not from matched root-cleared transitions")
    market_summaries = []
    for split in ("train_simulation", "validation_simulation"):
        simulation = fixed.get(split, {})
        market = simulation.get("market_clearing", {})
        aggregation = simulation.get("aggregation", {})
        if market:
            market_summaries.append(market)
        if market.get("mode") != "market_cleared_experimental_v1":
            reasons.append(f"{split} did not use market clearing")
        if int(market.get("failure_count", -1)) != 0:
            reasons.append(f"{split} contains market-clearing failures")
        if int(market.get("candidate_commit_parity_pass_count", -1)) != int(market.get("node_count", -2)):
            reasons.append(f"{split} candidate/commit parity incomplete")
        if aggregation.get("mode") != "continuum_mass_consistent_v1":
            reasons.append(f"{split} did not use continuum aggregation")
        if bool(aggregation.get("ARTIFICIAL_SMOOTHING_ADDED", True)):
            reasons.append(f"{split} reports artificial smoothing")
        if bool(aggregation.get("CONSUMPTION_FLOOR_ADDED", True)):
            reasons.append(f"{split} reports a consumption floor")
        if bool(aggregation.get("FAILED_CHILDREN_DROPPED", True)):
            reasons.append(f"{split} reports failed-child filtering")
    gate = {
        "stage": stage,
        "passed": not reasons,
        "reasons": reasons,
        "observed_stages": sorted(observed),
        "final_summary": final_summary,
        "market_summaries": market_summaries,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(stage_root / "reports" / "stage_gate.json", gate)
    return gate


def _report(root: Path, chain: Mapping[str, Any]) -> None:
    diagnostic = _read_json(root / "summary.json")
    lines = [
        "# Consistent Aggregation Market Clearing — 2026-10-19",
        "",
        "## Scope and provenance",
        "",
        f"- Source commit: `{diagnostic.get('gate', {}).get('git_head', diagnostic.get('git_head', 'see metadata/source_manifest.json'))}`",
        "- Formal started: **FALSE**",
        "- Economic calibration changed: **FALSE**",
        "- Aggregation semantics changed: **TRUE**",
        "- Numerical state representation changed: **TRUE**",
        "- Artificial smoothing added: **FALSE**",
        "- Consumption floor added: **FALSE**",
        "- Failed children dropped: **FALSE**",
        "",
        "## Diagnostic gate",
        "",
        "```json",
        json.dumps(diagnostic.get("gate", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Small chain",
        "",
        "```json",
        json.dumps(chain, indent=2, sort_keys=True, default=str),
        "```",
        "",
        "The legacy node's corrected terminal label is expected to differ from the old label while its economic operator values remain the replay-parity object.",
    ]
    report = root.parent.parent / "reports" / "consistent_aggregation_market_clearing_20261019.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--old-command", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--start-at", choices=("diagnostic", "tiny", "small"), default="diagnostic")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = args.output_dir.expanduser().resolve()
    (root / "logs").mkdir(parents=True, exist_ok=True)
    initial = args.initial_checkpoint.expanduser().resolve()
    chain: dict[str, Any] = {
        "experiment": "consistent_aggregation_market_clearing_20261019",
        "source_root": str(ROOT),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "optimizer_and_al_state_restored": False,
        "old_banks_targets_and_metrics_reused": False,
        "FORMAL_STARTED": False,
        "stages": [],
    }

    if args.start_at == "diagnostic":
        diagnostic_command = [
            sys.executable, str(DIAGNOSTIC),
            "--old-command", str(args.old_command.resolve()),
            "--checkpoint", str(initial),
            "--replay-dir", str(args.replay_dir.resolve()),
            "--output-dir", str(root),
            "--device", args.device,
            "--seed", str(args.seed),
        ]
        _write_json(root / "logs" / "diagnostic_command.json", diagnostic_command)
        started = time.time()
        code = _run(diagnostic_command, root / "logs" / "diagnostic.log")
        gate = _read_json(root / "diagnostic_gate.json") if (root / "diagnostic_gate.json").is_file() else {"passed": False}
        chain["stages"].append({"stage": "diagnostic", "return_code": code, "elapsed_seconds": time.time() - started, "gate": gate})
        _write_json(root / "chain_status.json", chain)
        if code != 0 or not gate.get("passed"):
            print(f"CONSISTENT_AGGREGATION_CHAIN_BLOCKED stage=diagnostic gate={gate}", flush=True)
            return 2

    stages = ("tiny", "small")
    start_index = 0 if args.start_at in {"diagnostic", "tiny"} else 1
    for stage in stages[start_index:]:
        stage_root = root / stage
        stage_root.mkdir(parents=True, exist_ok=True)
        final_summary = stage_root / "reports" / "final_summary.json"
        resume_pointer = stage_root / "latest_checkpoint.json"
        if final_summary.is_file():
            command: list[str] = []
        elif resume_pointer.is_file():
            command = [sys.executable, str(NESTED), "--resume", str(resume_pointer)]
        else:
            warm = initial if stage == "tiny" else root / "tiny" / "checkpoints" / "final.pt"
            command = [
                sys.executable, str(NESTED), "--cold-start",
                "--initial-checkpoint", str(warm),
                "--output-dir", str(stage_root),
                "--device", args.device,
                "--seed", str(args.seed),
                *_common(),
                *_stage_args(stage),
            ]
        _write_json(root / "logs" / f"{stage}_command.json", command)
        started = time.time()
        code = _run(command, root / "logs" / f"{stage}.log") if command else 0
        gate = _stage_gate(stage_root, stage) if code == 0 else {
            "stage": stage, "passed": False, "reasons": [f"runner return code {code}"]
        }
        chain["stages"].append({
            "stage": stage,
            "return_code": code,
            "elapsed_seconds": time.time() - started,
            "gate": gate,
        })
        _write_json(root / "chain_status.json", chain)
        if code != 0 or not gate["passed"]:
            _report(root, chain)
            print(f"CONSISTENT_AGGREGATION_CHAIN_BLOCKED stage={stage} reasons={gate['reasons']}", flush=True)
            return 2

    chain["completed_at"] = datetime.now(timezone.utc).isoformat()
    chain["SMALL_OUTERS_COMPLETED"] = 2
    chain["READY_FOR_FORMAL"] = False
    chain["ready_for_formal_reason"] = "requires explicit post-Small scientific review; Formal is outside this run"
    _write_json(root / "chain_status.json", chain)
    _report(root, chain)
    print("CONSISTENT_AGGREGATION_SMALL_CHAIN_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
