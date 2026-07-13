from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


PROBES = [
    "baseline_output_loss",
    "branch_only_output_loss",
    "bias_recenter_output_loss",
    "original_init_logit_loss",
]

SUMMARY_FIELDS = [
    "final_holdout_mae",
    "final_holdout_mae_p90",
    "final_weighted_holdout_mae",
    "final_holdout_pred_target_corr",
    "final_holdout_pred_target_spearman",
    "final_holdout_bp_pred_std",
    "holdout_constant_policy_flag",
    "initial_bp_head_grad_norm",
    "final_bp_head_grad_norm",
]


def parse_int_list(value: str) -> List[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one integer.")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize BP recovery probe runs across seeds.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=parse_int_list, required=True)
    parser.add_argument("--seeds", type=parse_int_list, required=True)
    return parser.parse_args()


def expected_summary_path(input_root: Path, episode: int, seed: int) -> Path:
    return input_root / f"ep{episode}" / f"seed{seed}" / f"ep{episode}_bp_recovery_probe_summary.csv"


def expected_history_path(input_root: Path, episode: int, seed: int) -> Path:
    return input_root / f"ep{episode}" / f"seed{seed}" / f"ep{episode}_bp_recovery_probe_history.csv"


def require_finite(df: pd.DataFrame, label: str) -> None:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad = numeric.columns[~np.isfinite(values).all(axis=0)].tolist()
        raise RuntimeError(f"{label} contains non-finite numeric columns: {bad}")


def load_one_run(input_root: Path, episode: int, seed: int) -> pd.DataFrame:
    summary_path = expected_summary_path(input_root, episode, seed)
    history_path = expected_history_path(input_root, episode, seed)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV for episode={episode}, seed={seed}: {summary_path}")
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history CSV for episode={episode}, seed={seed}: {history_path}")
    df = pd.read_csv(summary_path)
    if df.empty:
        raise RuntimeError(f"Empty summary CSV: {summary_path}")
    probes = sorted(df["probe"].unique().tolist())
    if probes != sorted(PROBES):
        raise RuntimeError(f"Unexpected probes in {summary_path}: {probes}")
    missing = [field for field in SUMMARY_FIELDS if field not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required summary fields in {summary_path}: {missing}")
    if "final_pred_target_corr" in df.columns and "final_holdout_pred_target_corr" not in df.columns:
        raise RuntimeError("Summary must not substitute train correlation for holdout correlation.")
    df = df.copy()
    if set(df["episode"].astype(int).unique()) != {int(episode)}:
        raise RuntimeError(f"Summary episode mismatch in {summary_path}")
    if set(df["seed"].astype(int).unique()) != {int(seed)}:
        raise RuntimeError(f"Summary seed mismatch in {summary_path}")
    df["episode"] = int(episode)
    df["seed"] = int(seed)
    require_finite(df, str(summary_path))
    return df


def load_all_runs(input_root: Path, episodes: Iterable[int], seeds: Iterable[int]) -> pd.DataFrame:
    rows = [load_one_run(input_root, episode, seed) for episode in episodes for seed in seeds]
    all_runs = pd.concat(rows, ignore_index=True)
    expected = len(list(episodes)) * len(list(seeds)) * len(PROBES)
    if len(all_runs) != expected:
        raise RuntimeError(f"Expected {expected} summary rows, found {len(all_runs)}")
    return all_runs


def multiseed_summary(all_runs: pd.DataFrame) -> pd.DataFrame:
    df = all_runs.copy()
    df["holdout_constant_policy_flag"] = df["holdout_constant_policy_flag"].astype(float)
    grouped = df.groupby(["episode", "probe"], sort=True)[SUMMARY_FIELDS]
    summary = grouped.agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    return summary.reset_index()


def probe_value(df: pd.DataFrame, probe: str, field: str) -> float:
    rows = df.loc[df["probe"] == probe, field]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row for probe={probe}, field={field}; found {len(rows)}")
    return float(rows.iloc[0])


def pairwise_comparisons(all_runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (episode, seed), group in all_runs.groupby(["episode", "seed"], sort=True):
        a_full = probe_value(group, "baseline_output_loss", "final_holdout_mae")
        a_branch = probe_value(group, "branch_only_output_loss", "final_holdout_mae")
        b_full = probe_value(group, "bias_recenter_output_loss", "final_holdout_mae")
        c_branch = probe_value(group, "original_init_logit_loss", "final_holdout_mae")
        rows.extend(
            [
                {
                    "episode": int(episode),
                    "seed": int(seed),
                    "comparison": "H002_bias_recenter_minus_baseline",
                    "left_probe": "bias_recenter_output_loss",
                    "right_probe": "baseline_output_loss",
                    "delta_holdout_mae": b_full - a_full,
                    "improved": bool(b_full < a_full),
                },
                {
                    "episode": int(episode),
                    "seed": int(seed),
                    "comparison": "H003_logit_minus_branch_output",
                    "left_probe": "original_init_logit_loss",
                    "right_probe": "branch_only_output_loss",
                    "delta_holdout_mae": c_branch - a_branch,
                    "improved": bool(c_branch < a_branch),
                },
                {
                    "episode": int(episode),
                    "seed": int(seed),
                    "comparison": "mix_branch_minus_full_output",
                    "left_probe": "branch_only_output_loss",
                    "right_probe": "baseline_output_loss",
                    "delta_holdout_mae": a_branch - a_full,
                    "improved": bool(a_branch < a_full),
                },
            ]
        )
    out = pd.DataFrame(rows)
    require_finite(out, "pairwise comparisons")
    return out


def no_holdout_constant(group: pd.DataFrame, probe: str) -> bool:
    rows = group.loc[group["probe"] == probe, "holdout_constant_policy_flag"].astype(bool)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one holdout constant flag for probe={probe}")
    return not bool(rows.iloc[0])


def decision_table(all_runs: pd.DataFrame, comparisons: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for episode, episode_runs in all_runs.groupby("episode", sort=True):
        episode_comps = comparisons.loc[comparisons["episode"] == episode]
        h002 = episode_comps.loc[episode_comps["comparison"] == "H002_bias_recenter_minus_baseline"]
        h003 = episode_comps.loc[episode_comps["comparison"] == "H003_logit_minus_branch_output"]
        mix = episode_comps.loc[episode_comps["comparison"] == "mix_branch_minus_full_output"]
        b_no_constant = all(
            no_holdout_constant(group, "bias_recenter_output_loss")
            for _, group in episode_runs.groupby("seed", sort=True)
        )
        c_no_constant = all(
            no_holdout_constant(group, "original_init_logit_loss")
            for _, group in episode_runs.groupby("seed", sort=True)
        )
        rows.extend(
            [
                {
                    "episode": int(episode),
                    "hypothesis": "H002_supported",
                    "comparison": "bias_recenter_output_loss - baseline_output_loss",
                    "all_seeds_improve": bool(h002["improved"].all()),
                    "mean_delta_holdout_mae": float(h002["delta_holdout_mae"].mean()),
                    "no_holdout_constant_policy": bool(b_no_constant),
                    "supported": bool(h002["improved"].all() and h002["delta_holdout_mae"].mean() < 0.0 and b_no_constant),
                    "caveat": "Evidence table only; Bellman regret is not included.",
                },
                {
                    "episode": int(episode),
                    "hypothesis": "H003_supported",
                    "comparison": "original_init_logit_loss - branch_only_output_loss",
                    "all_seeds_improve": bool(h003["improved"].all()),
                    "mean_delta_holdout_mae": float(h003["delta_holdout_mae"].mean()),
                    "no_holdout_constant_policy": bool(c_no_constant),
                    "supported": bool(h003["improved"].all() and h003["delta_holdout_mae"].mean() < 0.0 and c_no_constant),
                    "caveat": "Evidence table only; Bellman regret is not included.",
                },
                {
                    "episode": int(episode),
                    "hypothesis": "mix_objective_material",
                    "comparison": "branch_only_output_loss - baseline_output_loss",
                    "all_seeds_improve": bool(mix["improved"].all()),
                    "mean_delta_holdout_mae": float(mix["delta_holdout_mae"].mean()),
                    "no_holdout_constant_policy": True,
                    "supported": bool(abs(float(mix["delta_holdout_mae"].mean())) > 1e-3),
                    "caveat": "Materiality threshold is absolute mean holdout MAE delta > 1e-3.",
                },
            ]
        )
    out = pd.DataFrame(rows)
    require_finite(out, "decision table")
    return out


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs = load_all_runs(input_root, args.episodes, args.seeds)
    multi = multiseed_summary(all_runs)
    comparisons = pairwise_comparisons(all_runs)
    decisions = decision_table(all_runs, comparisons)

    all_runs.to_csv(output_dir / "bp_recovery_all_runs.csv", index=False)
    multi.to_csv(output_dir / "bp_recovery_multiseed_summary.csv", index=False)
    comparisons.to_csv(output_dir / "bp_recovery_pairwise_comparisons.csv", index=False)
    decisions.to_csv(output_dir / "bp_recovery_decision_table.csv", index=False)
    print(f"Wrote BP recovery summaries to {output_dir}")
    print(decisions.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
