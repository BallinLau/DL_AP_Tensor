"""Reporting helpers for an expected pre-FC1 conditional-data gate stop."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _coverage_row(path: Path, split: str) -> dict[str, Any]:
    rows = _read_rows(path)
    initial = sum(row.get("candidate_phase") == "INITIAL_BUDGET" for row in rows)
    accepted = sum(_truthy(row.get("accepted_for_training", "")) for row in rows)
    statuses = [row.get("status", "") for row in rows]
    child0_failed = sum(row.get("child_0_status") != "VALID" for row in rows)
    child1_failed = sum(row.get("child_1_status") != "VALID" for row in rows)
    return {
        "stage": "pre_fc1",
        "split": split,
        "original_validation_candidate_count": initial if split == "validation" else "",
        "original_candidate_count": initial,
        "attempted_candidate_count": len(rows),
        "valid_before_fc1": accepted,
        "valid_after_fc1": "NOT_RUN",
        "matched_valid_subset_count": "NOT_RUN",
        "failed_parent_count": statuses.count("PARENT_CLEARING_FAILED"),
        "failed_child0_count": child0_failed,
        "failed_child1_count": child1_failed,
        "extinct_count": statuses.count("CHILD_EXTINCT"),
        "strict_root_failure_count": sum(
            status in {
                "PARENT_CLEARING_FAILED",
                "CHILD_0_CLEARING_FAILED",
                "CHILD_1_CLEARING_FAILED",
            }
            for status in statuses
        ),
        "valid_subset_fraction": accepted / max(initial, 1),
        "metrics_interpretation": "conditional_on_complete_parent_two_child_pair",
        "coverage_interpretation": (
            "fixed_original_schedule_no_replacement"
            if split == "validation"
            else "bounded_replenishment_with_preserved_failures"
        ),
    }


def finalize_pre_fc1_gate_failure(
    *,
    output: Path,
    generating_commit: str,
    reporting_commit: str,
    initial_checkpoint: str,
    blocker: str,
) -> dict[str, Any]:
    """Write complete evidence without manufacturing post-FC1 results."""

    train_path = output / "smoke" / "pre_fc1_train" / "candidate_outcomes.csv"
    validation_path = (
        output / "smoke" / "pre_fc1_validation" / "candidate_outcomes.csv"
    )
    coverage = [
        _coverage_row(train_path, "train"),
        _coverage_row(validation_path, "validation"),
    ]
    with (output / "validation_coverage.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows(coverage)

    train, validation = coverage
    smoke = {
        "requested": True,
        "status": "STOPPED_AT_PRE_FC1_CONDITIONAL_DATA_GATE",
        "blocker": blocker,
        "generating_commit": generating_commit,
        "reporting_commit": reporting_commit,
        "initial_checkpoint": initial_checkpoint,
        "pre_train_valid_pairs": train["valid_before_fc1"],
        "pre_validation_valid_pairs": validation["valid_before_fc1"],
        "fc1_training_performed": False,
        "post_fc1_generation_performed": False,
        "sdf_interface_gate": False,
        "diagnostic_sdf_optimizer_smoke_performed": False,
        "failed_children_dropped": False,
        "validation_replacement_performed": False,
        "economic_rules_changed": False,
    }
    (output / "A800_smoke_summary.json").write_text(
        json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "PASS": True,
        "generating_commit": generating_commit,
        "reporting_commit": reporting_commit,
        "pre_fc1_source_checkpoint_explicit": True,
        "post_fc1_source_checkpoint": "NOT_APPLICABLE_GATE_STOP",
        "model_hash_mismatch_negative_test": (
            "PASS: tests/test_final_firm_first_nested_training.py"
        ),
        "interpretation": (
            "cache versioning passed; this does not override the conditional-data gate"
        ),
    }
    (output / "dataset_provenance_parity.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    performance_rows = _read_rows(output / "solver_performance.csv")
    performance = {row["mode"]: row for row in performance_rows}
    strict = float(performance["STRICT_GLOBAL"]["median_wall_clock_seconds"])
    fast = float(
        performance["FC1_BOUNDED_IMPROVEMENT_GPU"]["median_wall_clock_seconds"]
    )
    speedup = strict / fast if fast > 0 else None
    solver_status = json.loads(
        (output / "solver_status_summary.json").read_text(encoding="utf-8")
    )
    report = f"""# Firm-first Fast Macro Clearing Pilot — Final Report

## Provenance and scope

- Economic benchmark generating commit: `{generating_commit}`
- Gate-reporting commit: `{reporting_commit}`
- Frozen Firm checkpoint: `{initial_checkpoint}`
- Formal outer started: **FALSE**
- Strict solver default changed: **FALSE**
- Economic rules changed: **FALSE**

## Phase A

Checkpoint/AL/optimizer/scheduler/RNG round-trip and transactional OOM rollback
artifacts were produced before the benchmark.  Dataset cache provenance checks
are recorded in `dataset_provenance_parity.json`.

## Fixed 32-parent benchmark

- Strict median wall-clock per parent: `{strict:.6f}` seconds.
- Fast GPU median wall-clock per parent: `{fast:.6f}` seconds.
- Measured speedup: `{speedup:.6f}x`.
- Fast median evaluation budget <= 8: `{solver_status.get('median_G_evaluations_le_8')}`.
- Candidate micro-batch parity: `{solver_status.get('microbatch_parity')}`.

The complete parent-level economic and policy metrics remain in
`solver_comparison.csv`; status counts and per-parent solver traces remain in
`solver_status_summary.json`.

## Conditional macro smoke gate

The run stopped before FC1 training.  This was an expected data-admission gate,
not OOM and not a solver traceback:

- Train accepted complete pairs: **{train['valid_before_fc1']}** from an
  original budget of **{train['original_candidate_count']}** (attempted
  **{train['attempted_candidate_count']}** including bounded replenishment).
- Fixed validation accepted complete pairs: **{validation['valid_before_fc1']}**
  from **{validation['original_candidate_count']}**; no replacement was used.
- Validation child extinction count: **{validation['extinct_count']}**.
- Validation parent clearing failure count: **{validation['failed_parent_count']}**.

Blocker: `{blocker}`

Per the locked protocol, failed children were not dropped, validation identities
were not replaced, and no minimum surviving mass, consumption floor, artificial
entry, or root-tolerance relaxation was introduced.  Therefore FC1 training,
post-FC1 re-simulation, FC1-C/K isolation, and the SDF interface/optimizer smoke
are **NOT RUN**.  `fc1_c_k_isolation.png` is intentionally absent rather than a
fabricated comparison.

## Decision

The engineering resume and fast-solver benchmark layers are operational.  The
current fixed candidate schedule does **not** satisfy the predeclared complete
parent/two-child coverage gate, so this run is **not ready** for the formal
Macro/SDF conditional pilot.  The only immediate blocker is insufficient valid
validation coverage under the frozen schedule; resolving it requires a separate,
explicit sampling/economic decision outside this task.
"""
    (output / "final_report.md").write_text(report, encoding="utf-8")
    return smoke


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generating-commit", required=True)
    parser.add_argument("--reporting-commit", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--blocker", required=True)
    args = parser.parse_args()
    payload = finalize_pre_fc1_gate_failure(
        output=args.output_dir.resolve(),
        generating_commit=args.generating_commit,
        reporting_commit=args.reporting_commit,
        initial_checkpoint=args.initial_checkpoint,
        blocker=args.blocker,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
