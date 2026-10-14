#!/usr/bin/env python3
"""Combine three independent inner-FP block runs and render the causal audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODES = ("p_only", "q_only", "coupled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-only", type=Path, required=True)
    parser.add_argument("--q-only", type=Path, required=True)
    parser.add_argument("--coupled", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--date", default="20261014")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run(path: Path, expected_mode: str) -> Tuple[pd.DataFrame, dict, dict]:
    frame_path = path / "fixed_point" / "block_isolation_metrics.csv"
    summary_path = path / "fixed_point" / "final_summary.json"
    metrics_path = path / "fixed_point" / "metrics.json"
    if not frame_path.is_file() or not summary_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Incomplete block-isolation run: {path}")
    frame = pd.read_csv(frame_path)
    summary = _load_json(summary_path)
    metrics = _load_json(metrics_path)
    if summary.get("mode") != expected_mode or set(frame["mode"]) != {expected_mode}:
        raise ValueError(f"Mode mismatch for {path}: expected {expected_mode}")
    if int(summary.get("completed_sweeps", -1)) != 10 or len(frame) != 10:
        raise ValueError(f"Expected exactly 10 sweeps for {expected_mode}")
    if summary.get("numerical_failure") is not None:
        raise ValueError(f"Numerical failure in {expected_mode}: {summary['numerical_failure']}")
    return frame, summary, metrics


def _trend(values: Iterable[float]) -> Tuple[str, float, float]:
    series = np.asarray(list(values), dtype=float)
    tail = series[-5:]
    slope = float(np.polyfit(np.arange(tail.size), tail, 1)[0])
    change = float(series[-1] - series[0])
    if change < 0.0 and slope < 0.0:
        label = "improving"
    elif change > 0.0 and slope > 0.0:
        label = "diverging"
    else:
        label = "flat/stalled or mixed"
    return label, change, slope


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plots(frame: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"p_only": "#1f77b4", "q_only": "#ff7f0e", "coupled": "#d62728"}

    fig, ax = plt.subplots(figsize=(7, 4))
    for mode, group in frame.groupby("mode"):
        if mode == "p_only":
            values = group[["p0_residual_p90", "pi_residual_p90"]].max(axis=1)
        elif mode == "q_only":
            values = group["q_residual_p90"]
        else:
            values = group[["p0_residual_p90", "pi_residual_p90", "q_residual_p90"]].max(axis=1)
        ax.plot(group["sweep"], values, marker="o", label=mode, color=colors[mode])
    ax.set(xlabel="sweep", ylabel="primary normalized residual p90")
    ax.legend()
    _save(fig, figure_dir / "01_primary_residual_comparison.png")

    p = frame[frame["mode"] == "p_only"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(p.sweep, p.p0_residual_p90, marker="o", label="P0")
    ax.plot(p.sweep, p.pi_residual_p90, marker="o", label="PI")
    ax.set(xlabel="sweep", ylabel="normalized residual p90")
    ax.legend()
    _save(fig, figure_dir / "02_p_only_residuals.png")

    q = frame[frame["mode"] == "q_only"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(q.sweep, q.q_residual_p90, marker="o")
    ax.set(xlabel="sweep", ylabel="Q normalized residual p90")
    _save(fig, figure_dir / "03_q_only_residual.png")

    c = frame[frame["mode"] == "coupled"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for column, label in (
        ("start_fp_residual", "start"),
        ("post_p_fp_residual", "post-P"),
        ("post_q_fp_residual", "post-Q"),
    ):
        ax.plot(c.sweep, c[column], marker="o", label=label)
    ax.set(xlabel="sweep", ylabel="normalized FP residual p90")
    ax.legend()
    _save(fig, figure_dir / "04_coupled_stage_residuals.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for column, label in (
        ("p_to_q_effect_normalized_p90", "P→Q"),
        ("q_to_p_p0_effect_normalized_p90", "Q→P0"),
        ("q_to_p_pi_effect_normalized_p90", "Q→PI"),
    ):
        ax.plot(c.sweep, c[column], marker="o", label=label)
    ax.set(xlabel="sweep", ylabel="normalized target movement p90")
    ax.legend()
    _save(fig, figure_dir / "05_cross_coupling.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for mode, group in frame.groupby("mode"):
        ax.plot(group.sweep, group.default_area, marker="o", label=mode, color=colors[mode])
    ax.set(xlabel="sweep", ylabel="default area share")
    ax.legend()
    _save(fig, figure_dir / "06_default_area.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for mode in ("q_only", "coupled"):
        group = frame[frame["mode"] == mode]
        ax.plot(group.sweep, group.q_recursion_gain_p90, marker="o", label=mode)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set(xlabel="sweep", ylabel="Q recursion gain p90")
    ax.legend()
    _save(fig, figure_dir / "07_q_recursion_gain.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(p.sweep, p.p_fit_p90, marker="o", label="frozen-target fit")
    axes[0].plot(
        p.sweep, p[["p0_residual_p90", "pi_residual_p90"]].max(axis=1),
        marker="o", label="fresh self-consistency",
    )
    axes[0].set(title="P-only", xlabel="sweep", ylabel="normalized p90")
    axes[0].legend()
    axes[1].plot(q.sweep, q.q_fit_p90, marker="o", label="frozen-target fit")
    axes[1].plot(q.sweep, q.q_residual_p90, marker="o", label="fresh self-consistency")
    axes[1].set(title="Q-only", xlabel="sweep", ylabel="normalized p90")
    axes[1].legend()
    _save(fig, figure_dir / "08_fit_vs_self_consistency.png")


def main() -> int:
    args = parse_args()
    paths: Dict[str, Path] = {
        "p_only": args.p_only.resolve(),
        "q_only": args.q_only.resolve(),
        "coupled": args.coupled.resolve(),
    }
    loaded = {mode: _load_run(paths[mode], mode) for mode in MODES}
    frames = [loaded[mode][0] for mode in MODES]
    summaries = {mode: loaded[mode][1] for mode in MODES}

    bank_hashes = [summaries[mode]["bank_hashes"] for mode in MODES]
    source_hashes = [
        summaries[mode]["source_provenance"]["source_checkpoint_sha256"] for mode in MODES
    ]
    if not all(item == bank_hashes[0] for item in bank_hashes[1:]):
        raise ValueError("The three runs do not share identical bank fingerprints")
    if len(set(source_hashes)) != 1:
        raise ValueError("The three runs do not share the same source checkpoint")
    for mode in MODES:
        thresholds = summaries[mode]["thresholds"]
        if float(thresholds["damping"]) != 1.0:
            raise ValueError(f"{mode} did not use damping=1")
        invariants = summaries[mode].get("final_invariants") or {}
        if any(value is False for value in invariants.values()):
            raise ValueError(f"{mode} has a failed block invariant: {invariants}")

    frame = pd.concat(frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "block_isolation_metrics.csv"
    frame.to_csv(csv_path, index=False)
    figure_dir = args.output_dir / "figures"
    _plots(frame, figure_dir)

    p = frame[frame["mode"] == "p_only"].copy()
    p["primary"] = p[["p0_residual_p90", "pi_residual_p90"]].max(axis=1)
    q = frame[frame["mode"] == "q_only"].copy()
    q["primary"] = q["q_residual_p90"]
    c = frame[frame["mode"] == "coupled"].copy()
    c["primary"] = c[["p0_residual_p90", "pi_residual_p90", "q_residual_p90"]].max(axis=1)
    trends = {name: _trend(group.primary) for name, group in (("p_only", p), ("q_only", q), ("coupled", c))}
    coupling_supported = (
        trends["p_only"][0] == "improving"
        and trends["q_only"][0] == "improving"
        and trends["coupled"][0] == "diverging"
    )
    default_changes = {
        mode: float(group.default_area.iloc[-1] - group.default_area.iloc[0])
        for mode, group in frame.groupby("mode")
    }
    p_fit_ratio = float(p.p_fit_p90.iloc[-1] / max(p.primary.iloc[-1], 1e-12))
    q_fit_ratio = float(q.q_fit_p90.iloc[-1] / max(q.primary.iloc[-1], 1e-12))

    lines = [
        f"# Inner fixed-point block isolation ({args.date})",
        "",
        "## Design and invariant checks",
        "",
        "The three branches independently start from the same Episode-1 checkpoint and use identical fixed parent/child banks. Damping is 1, J is inherited from the saved bank, outer simulation is disabled, and the coupled branch retains P-then-Q Gauss-Seidel timing.",
        "",
        f"- Source checkpoint SHA-256: `{source_hashes[0]}`",
        f"- Shared validation-parent fingerprint: `{bank_hashes[0]['validation_parent']}`",
        f"- Shared validation-child fingerprint: `{bank_hashes[0]['validation_children']}`",
        f"- Run roots: P-only `{paths['p_only']}`; Q-only `{paths['q_only']}`; coupled `{paths['coupled']}`",
        "- All recorded block/hash/geometry invariants passed.",
        "",
        "## Primary trajectories",
        "",
        "| mode | sweep 1 | sweep 10 | change | last-5 slope | classification |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for mode, group in (("p_only", p), ("q_only", q), ("coupled", c)):
        label, change, slope = trends[mode]
        lines.append(
            f"| {mode} | {group.primary.iloc[0]:.6g} | {group.primary.iloc[-1]:.6g} | {change:+.6g} | {slope:+.6g} | {label} |"
        )
    lines.extend([
        "",
        "These labels describe the observed ten-sweep path only; they are not a theorem or an invented pass/fail threshold.",
        "",
        "## Fit error versus fresh self-consistency",
        "",
        f"- P-only final frozen-target fit p90 / fresh P residual p90 = `{p_fit_ratio:.4g}`.",
        f"- Q-only final frozen-target fit p90 / fresh Q residual p90 = `{q_fit_ratio:.4g}`.",
        "",
        "## Coupling diagnostics",
        "",
        "| metric | sweep 1 | sweep 10 |",
        "|---|---:|---:|",
        f"| P→Q normalized p90 | {c.p_to_q_effect_normalized_p90.iloc[0]:.6g} | {c.p_to_q_effect_normalized_p90.iloc[-1]:.6g} |",
        f"| Q→P0 normalized p90 | {c.q_to_p_p0_effect_normalized_p90.iloc[0]:.6g} | {c.q_to_p_p0_effect_normalized_p90.iloc[-1]:.6g} |",
        f"| Q→PI normalized p90 | {c.q_to_p_pi_effect_normalized_p90.iloc[0]:.6g} | {c.q_to_p_pi_effect_normalized_p90.iloc[-1]:.6g} |",
        f"| RQ start p90 | {c.start_q_residual_p90.iloc[0]:.6g} | {c.start_q_residual_p90.iloc[-1]:.6g} |",
        f"| RQ post-P p90 | {c.post_p_q_residual_p90.iloc[0]:.6g} | {c.post_p_q_residual_p90.iloc[-1]:.6g} |",
        "",
        "## Default geometry and recursion",
        "",
        f"- Default-area sweep-1→10 changes: P-only `{default_changes['p_only']:+.6g}`, Q-only `{default_changes['q_only']:+.6g}`, coupled `{default_changes['coupled']:+.6g}`.",
        f"- Q-only recursion-gain >1 share: sweep 1 `{q.q_recursion_gain_gt1_share.iloc[0]:.6g}`, sweep 10 `{q.q_recursion_gain_gt1_share.iloc[-1]:.6g}`.",
        f"- Coupled recursion-gain >1 share: sweep 1 `{c.q_recursion_gain_gt1_share.iloc[0]:.6g}`, sweep 10 `{c.q_recursion_gain_gt1_share.iloc[-1]:.6g}`.",
        "",
        "## Causal conclusion",
        "",
    ])
    if coupling_supported:
        lines.append(
            "The observed pattern supports the coupling hypothesis: both isolated blocks improve while the coupled iteration diverges. A P×m→Q×n block-solver experiment is therefore warranted, but it was not enabled here."
        )
    else:
        failed = [mode for mode in ("p_only", "q_only") if trends[mode][0] != "improving"]
        lines.append(
            "The strict coupling hypothesis is not supported because the following isolated block(s) did not show an improving ten-sweep trajectory: "
            + ", ".join(failed)
            + ". The failure must first be attributed to those conditional block dynamics and separated into fit versus operator movement using the ratios above. A block solver is not yet causally justified."
        )
    lines.extend([
        "",
        f"Combined metrics: `{csv_path}`",
        f"Figures: `{figure_dir}`",
        "",
    ])
    report_path = args.output_dir / f"inner_fixed_point_block_isolation_{args.date}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"BLOCK_ISOLATION_METRICS={csv_path}")
    print(f"BLOCK_ISOLATION_REPORT={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
