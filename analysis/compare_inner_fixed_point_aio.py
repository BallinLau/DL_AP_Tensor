#!/usr/bin/env python3
"""Build a sweep-aligned old-product versus pairwise-U inner-FP comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


SWEEPS = (1, 2, 3, 5, 10)


def _get(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _q_survival_metrics(row: Mapping[str, Any]) -> Mapping[str, Any]:
    phases = _get(row, "training", "q", "phases") or []
    for phase in phases:
        if phase.get("phase") == "survival":
            return phase.get("metrics", {})
    return {}


def _extract(row: Mapping[str, Any]) -> Dict[str, Any]:
    q = _q_survival_metrics(row)
    p_best = _get(row, "training", "p", "best_validation_metrics") or {}
    result: Dict[str, Any] = {
        "sweep": row.get("sweep"),
        "fp_mean": _get(row, "operator", "empirical_fixed_point_distance_mean"),
        "fp_p90": _get(row, "operator", "empirical_fixed_point_distance_p90"),
    }
    for equation in ("p0", "pi", "q"):
        result[f"operator_{equation}_normalized_mean"] = _get(
            row, "operator", "equations", equation, "normalized", "mean"
        )
        result[f"operator_{equation}_normalized_p90"] = _get(
            row, "operator", "equations", equation, "normalized", "p90"
        )
        result[f"change_{equation}_normalized_mean"] = _get(
            row, "function_change", equation, "normalized_mean"
        )
        result[f"change_{equation}_normalized_p90"] = _get(
            row, "function_change", equation, "normalized_p90"
        )
    for key in (
        "q_main",
        "q_aio_stable_mean",
        "q_aio_u_signed_mean",
        "q_aio_u_abs_mean",
        "q_aio_total_mean",
        "q_raw_grad_norm",
        "q_clipped_grad_norm",
        "q_recursion_gain_mean",
        "q_recursion_gain_p90",
        "q_recursion_gain_gt1_share",
    ):
        result[key] = q.get(key)
    for key in (
        "default_area_share",
        "default_flip_share",
        "default_cutoff_drift_p90",
    ):
        result[key] = _get(row, "default_boundary", key)
    result["q_regime_default"] = _get(row, "operator", "q_regime_counts", "default")
    result["q_regime_survival"] = _get(row, "operator", "q_regime_counts", "survival")
    result["p_best_p0_normalized_abs_residual_mean"] = p_best.get(
        "p0_normalized_abs_residual_mean"
    )
    result["p_best_pi_normalized_abs_residual_mean"] = p_best.get(
        "pi_normalized_abs_residual_mean"
    )
    return result


def _load(run_root: Path) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    fixed = run_root / "fixed_point"
    metrics = json.loads((fixed / "metrics.json").read_text(encoding="utf-8"))
    summary_path = fixed / "final_summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file() else {}
    )
    return metrics.get("sweeps", []), summary


def _format(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_comparison(old_run: Path, new_run: Path, output_dir: Path) -> tuple[Path, Path]:
    old_rows, old_summary = _load(old_run)
    new_rows, new_summary = _load(new_run)
    selected = set(SWEEPS)
    flat = []
    for label, rows, mode in (
        ("old", old_rows, "legacy_j_way_product"),
        ("new", new_rows, "pairwise_u_statistic"),
    ):
        for row in rows:
            if int(row.get("sweep", -1)) in selected:
                flat.append({"run": label, "q_aio_mode": mode, **_extract(row)})

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "old_vs_pairwise_aio_metrics.csv"
    fieldnames = list(flat[0]) if flat else ["run", "q_aio_mode", "sweep"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)

    by_key = {(row["run"], int(row["sweep"])): row for row in flat}
    common_sweeps = sorted(
        sweep for sweep in selected
        if ("old", sweep) in by_key and ("new", sweep) in by_key
    )
    headline = (
        "fp_mean", "fp_p90", "q_main", "q_aio_stable_mean",
        "q_aio_u_abs_mean", "q_raw_grad_norm", "q_clipped_grad_norm",
        "default_area_share", "default_flip_share",
    )
    lines = [
        "# Legacy J-way product vs pairwise-U AiO",
        "",
        f"- Old run: `{old_run}`",
        f"- New run: `{new_run}`",
        f"- Old numerical failure: `{old_summary.get('numerical_failure')}`",
        f"- New numerical failure: `{new_summary.get('numerical_failure')}`",
        "- `N/A` for old pairwise-only diagnostics means the legacy run did not record that quantity.",
        "",
        "| sweep | metric | old product | new pairwise-U |",
        "|---:|---|---:|---:|",
    ]
    for sweep in common_sweeps:
        for metric in headline:
            lines.append(
                f"| {sweep} | {metric} | {_format(by_key[('old', sweep)].get(metric))} "
                f"| {_format(by_key[('new', sweep)].get(metric))} |"
            )
    lines.extend([
        "",
        "The CSV beside this report contains the complete requested operator, function-change, "
        "Q-recursion, default-region, regime-count, and P-fit fields.",
        "",
    ])
    md_path = output_dir / "old_vs_pairwise_aio_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in build_comparison(args.old_run, args.new_run, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
