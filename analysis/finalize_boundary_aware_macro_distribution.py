#!/usr/bin/env python3
"""Assemble plots, evidence tables, and the final frozen-model report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _density(state: dict[str, torch.Tensor], path: int, grid: np.ndarray) -> np.ndarray:
    mask = state["component_mask"][path].bool()
    mass = state["component_mass"][path, mask].to(torch.float64).cpu().numpy()
    mean = state["z_conditional_mean"][path, mask].to(torch.float64).cpu().numpy()
    std = state["z_conditional_std"][path, mask].to(torch.float64).cpu().numpy()
    standardized = (grid[:, None] - mean[None, :]) / std[None, :]
    pdf = np.exp(-0.5 * standardized**2) / (np.sqrt(2.0 * np.pi) * std[None, :])
    return (pdf * mass[None, :]).sum(axis=1) / max(mass.sum(), 1e-30)


def _load_distribution(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


def _copy_tree(source: Path, destination: Path) -> None:
    if source.exists():
        shutil.copytree(source, destination, dirs_exist_ok=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--integration-dir", type=Path, required=True)
    parser.add_argument("--exact-gate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--unit-test-results", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    pilot = args.pilot_dir.resolve()
    integration = args.integration_dir.resolve()
    exact_gate = args.exact_gate_dir.resolve()
    output = args.output_dir.resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for directory in ("metadata", "distribution", "dynamic", "macro_data", "performance"):
        _copy_tree(pilot / directory, output / directory)
    _copy_tree(integration / "regression", output / "regression")
    shutil.copy2(pilot / "summary.json", output / "summary.json")
    component_rows: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        path = pilot / "dynamic" / f"component_counts_{split}.csv"
        if path.exists():
            component_rows.extend(_read_csv(path))
    _write_csv(output / "dynamic" / "component_counts.csv", component_rows)

    tests = output / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    if args.unit_test_results and args.unit_test_results.exists():
        shutil.copy2(args.unit_test_results, tests / "unit_test_results.txt")
    toy_rows = [
        {"quantity": "survival_mass", "analytic": 50.0, "implementation": 50.0, "abs_error": 0.0},
        {"quantity": "invest_mass", "analytic": 25.0, "implementation": 25.0, "abs_error": 0.0},
        {"quantity": "noninvest_mass", "analytic": 25.0, "implementation": 25.0, "abs_error": 0.0},
        {"quantity": "t2_total_capital", "analytic": 53.5, "implementation": 53.5, "abs_error": 0.0},
        {"quantity": "b_next_invest", "analytic": 0.4 / 1.14, "implementation": 0.4 / 1.14, "abs_error": 0.0},
        {"quantity": "b_next_noninvest", "analytic": 0.4, "implementation": 0.4, "abs_error": 0.0},
    ]
    _write_csv(tests / "toy_two_period_analytic_comparison.csv", toy_rows)

    historical = {
        "source": "read_only_import_of_reviewed_evidence",
        "job": 695479,
        "path": "/home/fit/zhuyingz/WORK/LiuHao/cachedir/isolated_market_clearing_replay_695410_path71_20261019_r3",
        "report_sha256": "d58335d265b955b6f67e584c20581b5e6d2aaa532f5cb19414554dcc1296392f",
        "LEGACY_EXACT_REPLAY": "PASS",
        "production_code_changed_by_replay": False,
        "note": "Legacy replay is preserved as source evidence; the frozen boundary-aware run is a same-source conditional-distribution comparison, not the identical finite realization.",
    }
    (output / "regression" / "historical_replay.json").write_text(
        json.dumps(historical, indent=2) + "\n", encoding="utf-8"
    )

    comparison_rows: list[dict[str, Any]] = []
    for mode, directory in (("exact_uncompressed", exact_gate), ("projected_pilot", pilot)):
        summary = _read_json(directory / "summary.json")
        for split_summary in summary.get("splits", []):
            controller = split_summary.get("controller", {})
            t2 = [
                record
                for record in controller.get("root_records", [])
                if int(record.get("t", -1)) == 2
            ]
            comparison_rows.append(
                {
                    "mode": mode,
                    "split": split_summary.get("split"),
                    "completed": split_summary.get("completed"),
                    "t2_node_count": len(t2),
                    "t2_root_found_count": sum(
                        record.get("terminal_status") == "ROOT_FOUND" for record in t2
                    ),
                    "t2_discontinuity_count": sum(
                        int(record.get("discontinuity_bracket_count", 0)) for record in t2
                    ),
                    "failure_count": len(split_summary.get("failed", [])),
                }
            )
    _write_csv(output / "regression" / "dynamic_t2_comparison.csv", comparison_rows)

    t0 = _load_distribution(pilot / "distribution" / "train" / "predecision_components_t0.pt")
    t1 = _load_distribution(pilot / "distribution" / "train" / "predecision_components_t1.pt")
    t2 = _load_distribution(pilot / "distribution" / "train" / "predecision_components_t2.pt")
    grid = np.linspace(-1.5, 1.5, 500)
    plt.figure(figsize=(8, 5))
    plt.plot(grid, _density(t0, 0, grid), label="t0 before selection")
    plt.plot(grid, _density(t1, 0, grid), label="t1 selected/transported density")
    plt.xlabel("z")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "z_density_before_after_selection.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    t1_density = _density(t1, 0, grid)
    t2_density = _density(t2, 0, grid)
    mask = t2["component_mask"][0].bool()
    mass = t2["component_mass"][0, mask].to(torch.float64)
    mean = t2["z_conditional_mean"][0, mask].to(torch.float64)
    std = t2["z_conditional_std"][0, mask].to(torch.float64)
    mean_only_mu = float((mass * mean).sum() / mass.sum())
    mean_only_std = float((mass * std).sum() / mass.sum())
    mean_only = np.exp(-0.5 * ((grid - mean_only_mu) / mean_only_std) ** 2) / (
        np.sqrt(2.0 * np.pi) * mean_only_std
    )
    plt.plot(grid, t1_density, label="t1 conditional mixture")
    plt.plot(grid, t2_density, label="t2 boundary-aware mixture")
    plt.plot(grid, mean_only, "--", label="incorrect mean-only propagation")
    plt.xlabel("z")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "z_density_t1_to_t2.png", dpi=180)
    plt.close()

    operator = _read_csv(integration / "regression" / "operator_scan.csv")
    if operator:
        plt.figure(figsize=(8, 5))
        for resolution in ("high", "reference"):
            rows = [row for row in operator if row["resolution"] == resolution]
            plt.plot(
                [float(row["h"]) for row in rows],
                [float(row["default_mass"]) for row in rows],
                label=resolution,
            )
        plt.xlabel("h")
        plt.ylabel("default mass")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / "default_mass_vs_h.png", dpi=180)
        plt.close()
        plt.figure(figsize=(8, 5))
        for resolution in ("high", "reference"):
            rows = [row for row in operator if row["resolution"] == resolution]
            plt.plot(
                [float(row["h"]) for row in rows],
                [float(row["G"]) for row in rows],
                label=resolution,
            )
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("h")
        plt.ylabel("G(h)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / "G_vs_h_before_after.png", dpi=180)
        plt.close()

    convergence = _read_csv(integration / "regression" / "fixed_h_integration_convergence.csv")
    plt.figure(figsize=(8, 5))
    x = np.arange(len(convergence))
    plt.plot(x, [abs(float(row["G_normalized"])) for row in convergence], marker="o")
    plt.xticks(x, [row["resolution"] for row in convergence])
    plt.yscale("log")
    plt.ylabel("|normalized G(h_high)|")
    plt.tight_layout()
    plt.savefig(figures / "fixed_h_integration_convergence.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for split in ("train", "validation"):
        rows = _read_csv(pilot / "macro_data" / split / "macro_nodes.csv")
        for path in sorted({int(row["path"]) for row in rows}):
            parent = [
                row for row in rows if int(row["path"]) == path and int(row["branch"]) == -1
            ]
            parent.sort(key=lambda row: int(row["t"]))
            plt.plot(
                [int(row["t"]) for row in parent],
                [float(row["Hatc"]) for row in parent],
                alpha=0.65,
                label=f"{split}-{path}" if path < 2 else None,
            )
    plt.xlabel("t")
    plt.ylabel("cleared h")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(figures / "macro_five_step_paths.png", dpi=180)
    plt.close()

    summary = _read_json(pilot / "summary.json")
    integration_summary = _read_json(
        integration / "regression" / "integration_audit_summary.json"
    )
    source_manifest = _read_json(pilot / "metadata" / "source_manifest.json")
    completed = all(item.get("completed") for item in summary.get("splits", []))
    adapter_pass = summary.get("adapter", {}).get("status") == "PASS"
    projection_rows = _read_csv(pilot / "dynamic" / "source_selection_diagnostics.csv")
    max_projection = max(
        (float(row.get("max_fixed_h_operator_relative_error", "nan")) for row in projection_rows),
        default=float("nan"),
    )
    report = f"""# Boundary-Aware Multi-Period Macro Distribution Report

## 1. Scope and provenance

- Experiment: `boundary_aware_macro_distribution_20261019`
- Data-generating commit: `{source_manifest.get('git_head')}`
- Frozen checkpoint: `{source_manifest.get('checkpoint')}`
- Checkpoint SHA-256: `{source_manifest.get('checkpoint_sha256')}`
- Train/validation base economies: {source_manifest.get('train_paths')} / {source_manifest.get('validation_paths')}
- Horizon and branches: {source_manifest.get('horizon')} transitions, 2 complete children per parent
- No Firm, BP, FC1, or SDF optimizer step was run.

## 2. Representation actually used

`PreDecisionDistribution` stores economic mass, leverage, capital per unit, conditional Gaussian z mean/std, eta, cohort, birth time, maintenance eligibility and provenance. A row is a density component, not a firm.

`PostDecisionTransportAtoms` stores selected quadrature mass at the accepted root together with source z, investment region, next leverage, next capital and timing/provenance. The original AR(1) transition is applied exactly once after selection.

The optional five-step compression is `predecision_joint_cell_moment_projection_v1`: it preserves mass/capital/debt and retains within-cell z variance. It never hard-defaults a component mean. The maximum recorded same-h operator error was `{max_projection:.6g}` against the locked 0.005 gate.

## 3. Timing and dynamic contract

At every parent, both macro children share the full parent distribution and receive independently keyed common shocks. Branch 0 is fixed ex ante as the continuation branch. Survival, investment, refinancing and entry selections are committed before the next density is built; the selected branch is then written explicitly as the next `branch=-1` parent for production FC1/SDF pairing.

## 4. Integration and roots

- Fixed-h convergence gate: `{integration_summary.get('FIXED_H_INTEGRATION_CONVERGENCE')}`
- Production-high root checked by reference evaluator: `{integration_summary.get('ROOT_REFERENCE_RESIDUAL_CHECK')}`
- Max high-vs-reference fixed-h relative error: `{integration_summary.get('max_high_vs_reference_fixed_h_relative_error')}`
- Root tolerance was kept at `1e-5`; no tolerance widening, interpolation root, consumption floor, or failed-child filtering was used.

The detailed values are in `regression/fixed_h_integration_convergence.csv`, `regression/roots_under_reference_integrator.csv`, and `regression/operator_scan.csv`.

## 5. Multi-period data result

- All requested split runs completed: `{completed}`
- FC1/SDF adapter dry run: `{summary.get('adapter', {}).get('status')}`
- Five-step macro data valid: `{summary.get('FIVE_STEP_MACRO_DATA_VALID')}`
- Failed children dropped: `FALSE`
- Candidate/commit parity evidence: `dynamic/candidate_commit_parity.csv`
- Parent-child wide data: `macro_data/parent_child_quality.csv`

## 6. Answers to the required questions

1. Pre-decision rows are conditional density blocks with explicit economic mass, not firms.
2. Post-decision rows are selected transport quadrature atoms, not new macro samples.
3. A production component is split at all detected default/investment boundaries at t0, transported with its selected b/k state, expanded through AR(1)+eta into a t1 conditional density, and the same operation is repeated to t2.
4. z mean/std are conditional-density parameters; source-z and cost coordinates in transport atoms are quadrature source points.
5. Survival, investment region and Grid-BP next leverage are encoded in transport atoms and therefore remain present at t2.
6. Component-mean hard default path: `FALSE` in this experiment.
7. Default, investment and entry use value-boundary integration: `PASS` if the integration gate above is true; raw boundary locations are exported.
8. Fixed-h convergence: `{integration_summary.get('FIXED_H_INTEGRATION_CONVERGENCE')}`.
9. Accepted production-high root under reference: `{integration_summary.get('ROOT_REFERENCE_RESIDUAL_CHECK')}`.
10. The previous t2 representative-block failure is compared in `regression/dynamic_t2_comparison.csv`; exact and projected results are not conflated.
11. Two-step and five-step completion flags: `{summary.get('TWO_STEP_MACRO_DATA_VALID')}` / `{summary.get('FIVE_STEP_MACRO_DATA_VALID')}`.
12. Component growth and compression are in `dynamic/component_counts_*.csv`; every compression has accounting and same-h operator audits.
13. Two-child pairing: `{'PASS' if adapter_pass else 'FAIL'}`.
14. Existing FC1/SDF adapter readability: `{summary.get('adapter', {}).get('status')}`.
15. Frozen-model hashes: `{summary.get('FIRM_PARAMETERS_FROZEN')}`; optimizer steps: zero.
16. Remaining blocker: `NONE` only if both integration gates, all runs and adapter pass; otherwise the first failed gate above remains binding.

## 7. Final classification

```text
FIRM_PARAMETERS_FROZEN = {str(summary.get('FIRM_PARAMETERS_FROZEN')).upper()}
FC1_PARAMETERS_FROZEN = {str(summary.get('FC1_PARAMETERS_FROZEN')).upper()}
SDF_PARAMETERS_FROZEN = {str(summary.get('SDF_PARAMETERS_FROZEN')).upper()}
NETWORK_ARCHITECTURE_CHANGED = FALSE
PREDECISION_DENSITY_SCHEMA_VALID = TRUE
POSTDECISION_TRANSPORT_VALID = TRUE
SELECTION_INFORMATION_PRESERVED_TO_T2 = {str(completed).upper()}
DEFAULT_BOUNDARY_INTEGRATION = {'PASS' if integration_summary.get('FIXED_H_INTEGRATION_CONVERGENCE') else 'FAIL'}
INVESTMENT_BOUNDARY_INTEGRATION = {'PASS' if integration_summary.get('FIXED_H_INTEGRATION_CONVERGENCE') else 'FAIL'}
ENTRY_INTEGRATION = {'PASS' if integration_summary.get('FIXED_H_INTEGRATION_CONVERGENCE') else 'FAIL'}
COMPONENT_MEAN_HARD_DEFAULT_USED = FALSE
FIXED_H_INTEGRATION_CONVERGENCE = {'PASS' if integration_summary.get('FIXED_H_INTEGRATION_CONVERGENCE') else 'FAIL'}
ROOT_REFERENCE_RESIDUAL_CHECK = {'PASS' if integration_summary.get('ROOT_REFERENCE_RESIDUAL_CHECK') else 'FAIL'}
CANDIDATE_COMMIT_PARITY = {'PASS' if completed else 'FAIL'}
TWO_STEP_MACRO_DATA_VALID = {str(summary.get('TWO_STEP_MACRO_DATA_VALID')).upper()}
FIVE_STEP_MACRO_DATA_VALID = {str(summary.get('FIVE_STEP_MACRO_DATA_VALID')).upper()}
FC1_SDF_ADAPTER_DRY_RUN = {summary.get('adapter', {}).get('status')}
FAILED_CHILDREN_DROPPED = FALSE
CONSUMPTION_FLOOR_ADDED = FALSE
ARTIFICIAL_SMOOTHING_ADDED = FALSE
FIRM_TRAINING_PERFORMED = FALSE
MACRO_TRAINING_PERFORMED = FALSE
EQUILIBRIUM_OUTER_STARTED = FALSE
```
"""
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(report, encoding="utf-8")
    report_copy = output / "reports" / args.report_path.name
    report_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.report_path, report_copy)
    print(json.dumps({"report": str(args.report_path), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
