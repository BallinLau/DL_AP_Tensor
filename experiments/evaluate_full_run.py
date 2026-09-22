from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.checkpoint_loader import load_analysis_checkpoint  # noqa: E402
from evaluation.convergence_artifacts import (  # noqa: E402
    EpisodeEvaluation,
    discover_episode_firm_data,
    parse_episode_selection,
    read_dataframe,
    select_simulated_state_rows,
)
from evaluation.convergence_metrics import (  # noqa: E402
    choose_representative_episodes,
    compute_simulated_moments,
)
from evaluation.full_run_diagnostics import (  # noqa: E402
    build_config_invariant_snapshot,
    compare_config_invariant_snapshots,
    evaluate_fc1_checkpoint,
    evaluate_sdf_heldout_multi_k,
    model_state_hash,
    namespace_sdf_summary,
    parse_sdf_validation_log_blocks,
    parse_training_log,
    select_primary_sdf_validation_blocks,
    summarize_episode_statuses,
    write_json,
)
from evaluation.bellman_diagnostics import evaluate_bellman_residuals  # noqa: E402
from evaluation.bp_diagnostics import build_frozen_transition_data  # noqa: E402
from evaluation.grids import FrozenFirmGrid, load_reference_state, select_parent_rows  # noqa: E402
from experiments.evaluate_checkpoints import evaluate_matrix  # noqa: E402
from experiments.build_convergence_report import (  # noqa: E402
    MissingArtifacts,
    compute_function_drift,
)


HEADLINE_COLUMNS = [
    "episode", "status", "p0_residual_abs_mean", "p0_residual_abs_p90",
    "p0_residual_abs_p99", "pi_residual_abs_mean", "pi_residual_abs_p90",
    "pi_residual_abs_p99", "q_residual_abs_mean", "q_residual_abs_p90",
    "q_residual_abs_p99", "q_residual_normalized_abs_mean",
    "p0_trainM_residual_abs_mean", "p0_rawM_residual_abs_mean",
    "pi_trainM_residual_abs_mean", "pi_rawM_residual_abs_mean",
    "q_trainM_residual_abs_mean", "q_rawM_residual_abs_mean",
    "sdf_common_conditional_abs_mean", "sdf_common_conditional_abs_p90",
    "sdf_common_conditional_abs_p99", "sdf_common_u_stat", "sdf_common_g_max",
    "sdf_common_M_mean", "sdf_common_M_std", "sdf_common_valid_parent_ratio",
    "sdf_ondist_conditional_abs_mean", "sdf_ondist_conditional_abs_p90",
    "sdf_ondist_conditional_abs_p99", "sdf_ondist_u_stat", "sdf_ondist_g_max",
    "sdf_ondist_M_mean", "sdf_ondist_M_std", "sdf_ondist_valid_parent_ratio",
    "bp_mae_survival_identified", "bp_regret_mean", "bp_regret_p90", "bp_regret_p99",
    "teacher_identified_share", "fc1_hatc_rmse", "fc1_hatc_skill",
    "fc1_hatc_persistence_skill", "fc1_lnk_rmse", "fc1_lnk_skill",
    "fc1_lnk_persistence_skill", "fc1_hatc_best_shift", "fc1_lnk_best_shift",
    "Q_mean_abs_diff", "P0_mean_abs_diff", "PI_mean_abs_diff", "P_mean_abs_diff",
    "bp_mean_abs_diff", "bar_z_mean_abs_diff",
    "mean_b", "p90_b", "p99_b", "mean_bp", "p99_bp",
    "default_rate", "survival_rate", "investment_rate",
    "sdf_before_aio_mean", "sdf_after_aio_mean", "sdf_before_aio_t",
    "sdf_after_aio_t", "sdf_safe_to_continue", "sdf_stage_progress", "sdf_converged",
    "checkpoint", "firm_data", "macro_data", "error",
]


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def aggregate_episode_peak_memory(peaks: list[float | None]) -> float | None:
    finite = [float(value) for value in peaks if value is not None and np.isfinite(value)]
    return max(finite) if finite else None


def select_episode_child_counts(
    *,
    episode: int,
    representative_episodes: set[int],
    primary_child_shocks: int,
    robustness_child_shocks: list[int],
    robustness_scope: str,
) -> list[int]:
    if robustness_scope not in {"representative", "all", "none"}:
        raise ValueError(f"Unsupported robustness_scope={robustness_scope!r}")
    include_robustness = (
        robustness_scope == "all"
        or (robustness_scope == "representative" and episode in representative_episodes)
    )
    selected = robustness_child_shocks if include_robustness else []
    return sorted(set([int(primary_child_shocks), *(int(value) for value in selected)]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only full-run checkpoint evaluator across all episodes."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--episodes", default=None, help="Comma/range selection, e.g. 0,2:5")
    parser.add_argument("--training-log", type=Path)
    parser.add_argument("--reference-firm-data", type=Path)
    parser.add_argument("--reference-macro-data", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-child-shocks", type=int, default=64)
    parser.add_argument("--robustness-child-shocks", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument(
        "--robustness-scope",
        choices=("representative", "all", "none"),
        default="representative",
        help="Run extra J robustness cases on representative episodes, all episodes, or none.",
    )
    parser.add_argument("--shock-seed", type=int, default=12345)
    parser.add_argument("--max-sdf-parents", type=int, default=512)
    parser.add_argument("--eta-values", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--b-min", type=float, default=0.0)
    parser.add_argument("--b-max", type=float, default=1.0)
    parser.add_argument("--b-points", type=int, default=101)
    parser.add_argument("--z-min", type=float, default=-2.0)
    parser.add_argument("--z-max", type=float, default=2.0)
    parser.add_argument("--z-points", type=int, default=101)
    parser.add_argument("--i-points", type=int, default=101)
    parser.add_argument("--forward-chunk-size", type=int, default=8192)
    parser.add_argument("--bp-teacher-margin-tol", type=float, default=1e-8)
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _git(args: list[str]) -> str:
    import subprocess
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_checkpoints(run_root: Path) -> Dict[int, Path]:
    import re
    pattern = re.compile(r"^ep(?P<episode>\d+)_combined\.pt$")
    found: Dict[int, Path] = {}
    for directory in (run_root / "checkpoints_analysis", run_root / "checkpoints"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("ep*_combined.pt")):
            match = pattern.match(path.name)
            if match:
                found.setdefault(int(match.group("episode")), path.resolve())
    return dict(sorted(found.items()))


def _macro_for_firm(firm: Path) -> Path | None:
    candidates = [firm.with_name(f"{firm.stem}_macro{firm.suffix}")]
    if firm.stem.endswith("_firm"):
        candidates.insert(0, firm.with_name(f"{firm.stem.removesuffix('_firm')}_macro{firm.suffix}"))
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _choose_reference(run_root: Path, firms: Dict[int, Path], args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.reference_firm_data is not None:
        firm = args.reference_firm_data.resolve()
        macro = args.reference_macro_data.resolve() if args.reference_macro_data else _macro_for_firm(firm)
        return firm, macro
    final_firm = run_root / "data" / "outputs" / "final_simulate_firm.pkl"
    if final_firm.is_file():
        return final_firm.resolve(), _macro_for_firm(final_firm)
    if not firms:
        raise FileNotFoundError("No firm data available for a common reference state")
    firm = firms[max(firms)]
    return firm, _macro_for_firm(firm)


def _sample_parent_tensors(
    firm_path: Path,
    macro_path: Path | None,
    *,
    device: torch.device,
    max_parents: int,
) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    merged, _ = load_reference_state(firm_path, macro_path=macro_path)
    parents = select_parent_rows(merged)
    columns = ["b", "z", "ETA", "i", "x", "Hatcf", "LnKF", "hatc_cal", "lnk_cal"]
    numeric = parents[columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[np.isfinite(numeric.to_numpy()).all(axis=1)]
    if numeric.empty:
        raise ValueError("No finite parent states for held-out diagnostics")
    source_positions = numeric.index.to_numpy()
    if len(numeric) > max_parents:
        positions = np.linspace(0, len(numeric) - 1, max_parents).round().astype(int)
        numeric = numeric.iloc[positions]
        source_positions = source_positions[positions]
    states = torch.as_tensor(
        numeric[["b", "z", "ETA", "i", "x", "Hatcf", "LnKF"]].to_numpy(np.float32),
        device=device,
    )
    hatc = torch.as_tensor(numeric["hatc_cal"].to_numpy(np.float32), device=device).reshape(-1, 1)
    lnk = torch.as_tensor(numeric["lnk_cal"].to_numpy(np.float32), device=device).reshape(-1, 1)
    digest = hashlib.sha256()
    for name, value in (("states", states), ("hatc_cal", hatc), ("lnk_cal", lnk)):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(array.tobytes())
    metadata = {
        "source_artifact": str(firm_path.resolve()),
        "macro_context_source": str(macro_path.resolve()) if macro_path else None,
        "n_parent": int(len(numeric)),
        "selected_row_indices": [str(value) for value in source_positions.tolist()],
        "columns": columns,
        "parent_bank_sha256": digest.hexdigest(),
        "selection": "finite_parent_rows_then_evenly_spaced_positions",
    }
    return merged, states, hatc, lnk, metadata


def _firm_eval_args(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    reference_firm: Path,
    reference_macro: Path | None,
    output: Path,
    representative: bool,
    child_counts: list[int],
    loaded_checkpoint: Any,
) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=checkpoint, pv_ckpt=None, sdf_ckpt=None, hyperparams_json=None,
        config_json=None, model_spec_json=None, allow_default_hyperparams=False,
        allow_current_config=False, firm_data=reference_firm, macro_data=reference_macro,
        output_dir=output, device=str(args.device), eta=1.0, eta_values=args.eta_values,
        b_min=args.b_min, b_max=args.b_max, b_points=args.b_points,
        z_min=args.z_min, z_max=args.z_max, z_points=args.z_points,
        i_points=args.i_points, forward_chunk_size=args.forward_chunk_size,
        n_child_shocks=args.n_child_shocks, shock_seed=args.shock_seed,
        shock_bank_max_child_shocks=max(child_counts),
        robustness_child_shocks=[
            value for value in child_counts if value != int(args.n_child_shocks)
        ],
        robustness_scope=args.robustness_scope,
        bp_teacher_margin_tol=args.bp_teacher_margin_tol,
        summary_only_all=not representative,
        loaded_checkpoint=loaded_checkpoint,
        checkpoint_load_count=1,
        defer_model_state_hash_to_outer=True,
        manage_cuda_peak_stats=False,
    )


def _write_dashboard(headline: pd.DataFrame, output: Path) -> None:
    metrics = [
        "p0_rawM_residual_abs_mean", "pi_rawM_residual_abs_mean", "q_rawM_residual_abs_mean",
        "sdf_common_conditional_abs_mean", "fc1_hatc_rmse", "fc1_lnk_rmse",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, metric in zip(axes.reshape(-1), metrics):
        values = pd.to_numeric(headline.get(metric), errors="coerce")
        axis.plot(headline["episode"], values, marker="o")
        axis.set_title(metric)
        axis.set_xlabel("episode")
        axis.grid(alpha=0.25)
    fig.savefig(output / "convergence_dashboard.png", dpi=160)
    plt.close(fig)


def _plot_metric_dashboard(
    frame: pd.DataFrame,
    metrics: list[str],
    output: Path,
    *,
    title: str,
) -> None:
    available = [name for name in metrics if name in frame.columns]
    if not available or frame.empty:
        return
    ncols = min(3, len(available))
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), squeeze=False)
    for axis, metric in zip(axes.reshape(-1), available):
        values = pd.to_numeric(frame[metric], errors="coerce")
        axis.plot(frame["episode"], values, marker="o")
        axis.set_title(metric)
        axis.set_xlabel("episode")
        axis.grid(alpha=0.25)
    for axis in axes.reshape(-1)[len(available):]:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _merge_training_log(headline: pd.DataFrame, parsed: pd.DataFrame) -> pd.DataFrame:
    if parsed.empty:
        return headline
    selected = parsed.copy()
    extra = [name for name in selected.columns if name != "episode" and name not in {
        "sdf_before_aio_mean", "sdf_after_aio_mean", "sdf_before_aio_t",
        "sdf_after_aio_t", "sdf_constraint_before", "sdf_constraint_after",
        "sdf_safe_to_continue", "sdf_stage_progress", "sdf_converged",
        "sdf_validation_result", "sdf_validation_stage", "sdf_validation_occurrence",
    }]
    selected = selected.rename(columns={name: f"training_log_{name}" for name in extra})
    existing = [name for name in selected.columns if name != "episode" and name in headline.columns]
    if existing:
        headline = headline.drop(columns=existing)
    return headline.merge(selected, on="episode", how="left")


def _add_function_drift(
    headline: pd.DataFrame,
    output: Path,
    eta_values: list[float],
    missing: MissingArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = headline.loc[headline["status"].isin(["ok", "partial"]), "episode"].astype(int).tolist()
    cases = [EpisodeEvaluation(ep, output / "episodes" / f"ep{ep}" / "firm") for ep in valid]
    frames = []
    for eta_value in eta_values:
        if not float(eta_value).is_integer():
            missing.add(f"Function drift supports integer eta directories; skipped eta={eta_value:g}")
            continue
        frame, _ = compute_function_drift(
            cases, int(eta_value), output / "cross_episode" / "function_drift", missing
        )
        if not frame.empty:
            frames.append(frame)
    drift = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if drift.empty:
        return headline, drift
    preferred_eta = 1 if 1 in set(drift["eta"].astype(int)) else int(drift["eta"].iloc[0])
    primary = drift[drift["eta"].astype(int) == preferred_eta]
    wide = primary.pivot(index="episode", columns="surface", values="mean_abs_diff")
    wide = wide.rename(columns={name: f"{name}_mean_abs_diff" for name in wide.columns}).reset_index()
    existing = [name for name in wide.columns if name != "episode" and name in headline.columns]
    if existing:
        headline = headline.drop(columns=existing)
    return headline.merge(wide, on="episode", how="left"), drift


def _write_error_vs_drift(headline: pd.DataFrame, output: Path) -> pd.DataFrame:
    specifications = (
        ("rawM", "P0", "p0_rawM_residual_abs_mean", "P0_mean_abs_diff"),
        ("rawM", "PI", "pi_rawM_residual_abs_mean", "PI_mean_abs_diff"),
        ("rawM", "Q", "q_rawM_residual_abs_mean", "Q_mean_abs_diff"),
        ("rawM", "BP", "bp_regret_mean", "bp_mean_abs_diff"),
        ("trainM", "P0", "p0_trainM_residual_abs_mean", "P0_mean_abs_diff"),
        ("trainM", "PI", "pi_trainM_residual_abs_mean", "PI_mean_abs_diff"),
        ("trainM", "Q", "q_trainM_residual_abs_mean", "Q_mean_abs_diff"),
        ("trainM", "BP", "bp_regret_mean", "bp_mean_abs_diff"),
    )
    rows = []
    for semantics, module, error_name, drift_name in specifications:
        if error_name not in headline or drift_name not in headline:
            continue
        for _, item in headline.iterrows():
            rows.append({
                "episode": int(item["episode"]), "semantics": semantics, "module": module,
                "equation_error_metric": error_name,
                "equation_error": item[error_name],
                "function_drift_metric": drift_name,
                "function_drift": item[drift_name],
            })
    frame = pd.DataFrame(rows, columns=[
        "episode", "semantics", "module", "equation_error_metric",
        "equation_error", "function_drift_metric", "function_drift",
    ])
    destination = output / "cross_episode" / "equation_error_vs_drift"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "equation_error_vs_drift.csv", index=False)
    for semantics in ("rawM", "trainM"):
        subset = frame[frame["semantics"] == semantics]
        finite = subset.assign(
            equation_error=pd.to_numeric(subset.get("equation_error"), errors="coerce"),
            function_drift=pd.to_numeric(subset.get("function_drift"), errors="coerce"),
        ).dropna(subset=["equation_error", "function_drift"])
        if finite.empty:
            continue
        fig, axis = plt.subplots(figsize=(8, 6))
        for module, group in finite.groupby("module"):
            axis.plot(group["function_drift"], group["equation_error"], marker="o", label=module)
            for _, item in group.iterrows():
                axis.annotate(f"ep{int(item['episode'])}", (item["function_drift"], item["equation_error"]), fontsize=7)
        axis.set_xlabel("mean absolute function drift")
        axis.set_ylabel(f"equation error / BP regret ({semantics})")
        axis.set_title(f"Equation error vs drift: {semantics}")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(destination / f"equation_error_vs_drift_{semantics}.png", dpi=160)
        plt.close(fig)
    return frame


def _write_output_schema(
    headline: pd.DataFrame,
    parsed_log: pd.DataFrame,
    drift: pd.DataFrame,
    structural_long: pd.DataFrame,
    sdf_long: pd.DataFrame,
    sdf_blocks: pd.DataFrame,
    output: Path,
) -> None:
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    headline.to_csv(tables / "headline_metrics_by_episode.csv", index=False)
    table_groups = {
        "equation_metrics_by_episode.csv": ("p0_", "pi_", "q_", "ondist_p0_", "ondist_pi_", "ondist_q_"),
        "macro_metrics_by_episode.csv": ("fc1_",),
        "sdf_metrics_by_episode.csv": ("sdf_",),
        "simulated_moments_by_episode.csv": (
            "mean_b", "std_b", "median_b", "p90_b", "p95_b", "p99_b",
            "mean_bp", "p90_bp", "p99_bp", "default_rate", "survival_rate",
            "investment_rate", "alive_firm_count", "entrant_count", "exit_count",
        ),
    }
    for filename, prefixes in table_groups.items():
        columns = ["episode"] + [
            name for name in headline.columns
            if name != "episode" and any(name == prefix or name.startswith(prefix) for prefix in prefixes)
        ]
        headline[columns].to_csv(tables / filename, index=False)
    parsed_log.to_csv(tables / "training_log_metrics_by_episode.csv", index=False)
    structural_long.to_csv(tables / "structural_metrics_by_episode_eta.csv", index=False)
    sdf_long.to_csv(tables / "sdf_metrics_long.csv", index=False)
    sdf_blocks.to_csv(tables / "sdf_validation_log_blocks.csv", index=False)

    cross = output / "cross_episode"
    cross.mkdir(parents=True, exist_ok=True)
    mappings = {
        "equation_residuals": tables / "equation_metrics_by_episode.csv",
        "macro": tables / "macro_metrics_by_episode.csv",
        "sdf": tables / "sdf_metrics_by_episode.csv",
        "simulated_moments": tables / "simulated_moments_by_episode.csv",
        "log_diagnostics": tables / "training_log_metrics_by_episode.csv",
    }
    for name, source in mappings.items():
        destination = cross / name
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination / source.name)
    if not drift.empty:
        drift.to_csv(cross / "function_drift" / "all_eta_function_drift.csv", index=False)

    _plot_metric_dashboard(
        headline,
        ["p0_trainM_residual_abs_mean", "p0_rawM_residual_abs_mean",
         "pi_trainM_residual_abs_mean", "pi_rawM_residual_abs_mean",
         "q_trainM_residual_abs_mean", "q_rawM_residual_abs_mean"],
        figures / "equation_error_by_episode.png", title="Equation error by episode",
    )
    _plot_metric_dashboard(
        headline,
        ["p0_residual_abs_p99", "pi_residual_abs_p99", "q_residual_abs_p99"],
        figures / "equation_tail_error_by_episode.png", title="Equation tail error by episode",
    )
    _plot_metric_dashboard(
        headline,
        ["Q_mean_abs_diff", "P0_mean_abs_diff", "PI_mean_abs_diff",
         "P_mean_abs_diff", "bar_z_mean_abs_diff", "bp_mean_abs_diff"],
        figures / "function_drift_dashboard.png", title="Function drift",
    )
    _plot_metric_dashboard(
        headline,
        ["sdf_common_conditional_abs_mean", "sdf_ondist_conditional_abs_mean",
         "sdf_common_u_stat", "sdf_ondist_g_max", "sdf_ondist_M_mean",
         "sdf_ondist_M_std"],
        figures / "sdf_convergence_dashboard.png", title="SDF held-out convergence",
    )
    _plot_metric_dashboard(
        headline,
        ["fc1_hatc_rmse", "fc1_lnk_rmse", "fc1_hatc_skill", "fc1_lnk_skill"],
        figures / "fc1_convergence_dashboard.png", title="FC1 convergence",
    )
    _plot_metric_dashboard(
        headline,
        ["mean_b", "p90_b", "mean_bp", "default_rate", "survival_rate", "investment_rate"],
        figures / "simulated_moments_dashboard.png", title="Simulated moments",
    )


def _prepare_output_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Output directory is non-empty: {path}. Pass --overwrite to replace evaluator output."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _discover_episode_macro(run_root: Path, episode: int) -> Path | None:
    output = run_root / "data" / "outputs"
    candidates = sorted(output.glob(f"ep{episode}_stage_*_macro.pkl")) if output.is_dir() else []
    return candidates[-1].resolve() if candidates else None


def main() -> None:
    args = parse_args()
    total_started = time.perf_counter()
    run_root = args.run_root.expanduser().resolve()
    output = (args.output_dir or run_root / "data" / "outputs" / "full_run_evaluation").resolve()
    args.device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)
    dirty = _git(["status", "--porcelain"])
    if dirty and not args.allow_dirty_worktree:
        raise RuntimeError("Working tree is dirty; use --allow-dirty-worktree only for intentional local diagnostics")
    _prepare_output_directory(output, overwrite=bool(args.overwrite))

    checkpoints = _discover_checkpoints(run_root)
    firms, discovery_warnings = discover_episode_firm_data(run_root)
    requested = parse_episode_selection(args.episodes)
    available = sorted(set(checkpoints) | set(firms))
    episodes = available if requested is None else sorted(requested)
    if not episodes:
        raise RuntimeError("No episodes discovered or selected")
    reference_firm, reference_macro = _choose_reference(run_root, firms, args)
    _, reference_state = load_reference_state(reference_firm, macro_path=reference_macro)
    _, common_parent_states, common_hatc, common_lnk, common_parent_meta = _sample_parent_tensors(
        reference_firm, reference_macro, device=device, max_parents=args.max_sdf_parents
    )
    representative = set(choose_representative_episodes(episodes))
    child_counts_by_episode = {
        int(episode): select_episode_child_counts(
            episode=int(episode),
            representative_episodes=representative,
            primary_child_shocks=args.n_child_shocks,
            robustness_child_shocks=args.robustness_child_shocks,
            robustness_scope=args.robustness_scope,
        )
        for episode in episodes
    }

    headline_rows: list[Dict[str, Any]] = []
    error_rows: list[Dict[str, Any]] = []
    episode_metadata: list[Dict[str, Any]] = []
    availability_notes: list[str] = []
    structural_rows: list[Dict[str, Any]] = []
    sdf_rows_all: list[Dict[str, Any]] = []
    timing_rows: list[Dict[str, Any]] = []
    config_records: list[tuple[int, Dict[str, Any], str]] = []
    for episode in episodes:
        episode_started = time.perf_counter()
        episode_timing: Dict[str, Any] = {
            "episode": int(episode),
            "checkpoint_load_count": 0,
            "firm_matrix_wall_seconds": float("nan"),
            "firm_structural_seconds": float("nan"),
            "firm_static_seconds": float("nan"),
            "sdf_common_seconds": float("nan"),
            "sdf_ondist_seconds": float("nan"),
            "bellman_seconds": float("nan"),
            "bp_seconds": float("nan"),
            "investment_seconds": float("nan"),
            "fc1_seconds": float("nan"),
            "cuda_peak_memory_mb": None,
        }
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        episode_root = output / "episodes" / f"ep{episode}"
        for name in ("firm", "sdf", "fc1", "simulation", "training_log"):
            (episode_root / name).mkdir(parents=True, exist_ok=True)
        row: Dict[str, Any] = {name: np.nan for name in HEADLINE_COLUMNS}
        row.update({"episode": episode, "status": "missing"})
        checkpoint = checkpoints.get(episode)
        firm_path = firms.get(episode)
        if checkpoint is None:
            row["error"] = "missing checkpoint"
            availability_notes.append(f"Episode {episode}: {row['error']}")
            headline_rows.append(row)
            error_rows.append({"episode": episode, "stage": "discovery", "error": row["error"]})
            episode_timing["total_seconds"] = time.perf_counter() - episode_started
            timing_rows.append(episode_timing)
            continue
        macro_path = _macro_for_firm(firm_path) if firm_path else _discover_episode_macro(run_root, episode)
        row.update({
            "checkpoint": str(checkpoint),
            "firm_data": str(firm_path) if firm_path else np.nan,
            "macro_data": str(macro_path) if macro_path else np.nan,
        })
        try:
            selected_child_counts = child_counts_by_episode[int(episode)]
            max_j = max(selected_child_counts)
            loaded = load_analysis_checkpoint(checkpoint, device=device)
            episode_timing["checkpoint_load_count"] = 1
            for loaded_model in loaded.models.values():
                loaded_model.eval()
            active_model_names = loaded.metadata.get("loaded_model_keys", [])
            before = {
                name: model_state_hash(loaded.models[name])
                for name in active_model_names
            }
            config_snapshot, config_hash = build_config_invariant_snapshot(loaded)
            config_records.append((episode, config_snapshot, config_hash))
            _sync_cuda(device)
            firm_started = time.perf_counter()
            firm_summary, firm_meta = evaluate_matrix(_firm_eval_args(
                args, checkpoint=checkpoint, reference_firm=reference_firm,
                reference_macro=reference_macro, output=episode_root / "firm",
                representative=episode in representative,
                child_counts=selected_child_counts,
                loaded_checkpoint=loaded,
            ))
            _sync_cuda(device)
            episode_timing["firm_matrix_wall_seconds"] = time.perf_counter() - firm_started
            firm_timing = firm_meta.get("timing", {})
            for timing_name in (
                "firm_static_seconds", "bellman_seconds", "bp_seconds", "investment_seconds"
            ):
                episode_timing[timing_name] = float(firm_timing.get(timing_name, float("nan")))
            episode_timing["firm_structural_seconds"] = sum(
                float(episode_timing[name])
                for name in (
                    "firm_static_seconds", "investment_seconds", "bellman_seconds", "bp_seconds"
                )
                if np.isfinite(episode_timing[name])
            )
            structural_rows.extend(
                {"episode": episode, **item}
                for item in firm_summary.to_dict(orient="records")
            )
            primary = firm_summary[
                (firm_summary["n_child_shocks"] == args.n_child_shocks)
                & (firm_summary["eta_parent"] == 1.0)
            ]
            if primary.empty:
                primary = firm_summary.iloc[[0]]
            structural_values = primary.iloc[0].to_dict()
            row.update(structural_values)
            row.update({f"structural_{key}": value for key, value in structural_values.items()})

            episode_sdf_rows: list[Dict[str, Any]] = []
            common_primary_meta: Dict[str, Any] | None = None
            sdf_child_counts = selected_child_counts
            _sync_cuda(device)
            sdf_common_started = time.perf_counter()
            common_sdf_summaries, common_sdf_meta = evaluate_sdf_heldout_multi_k(
                loaded.models["sdf_fc1"], common_parent_states, hatc_cal=common_hatc,
                lnk_cal=common_lnk, economic_config=loaded.economic_config,
                child_counts=sdf_child_counts, seed=args.shock_seed,
                shock_bank_max_children=max_j,
                normalized_logr_clip=float(getattr(loaded.hyperparams, "sdf_normalized_logr_clip", 20.0)),
            )
            _sync_cuda(device)
            episode_timing["sdf_common_seconds"] = time.perf_counter() - sdf_common_started
            for child_count, sdf_summary in common_sdf_summaries.items():
                namespaced = namespace_sdf_summary(sdf_summary, scope="common")
                item = {"episode": episode, "scope": "common", "n_children": child_count, **namespaced}
                episode_sdf_rows.append(item)
                sdf_rows_all.append(item)
                if child_count == args.n_child_shocks:
                    row.update(namespaced)
                    common_primary_meta = {**common_sdf_meta, "parent_bank": common_parent_meta}
                    if float(namespaced["sdf_common_valid_parent_ratio"]) < 1.0:
                        availability_notes.append(
                            f"Episode {episode}: common SDF valid_parent_ratio="
                            f"{namespaced['sdf_common_valid_parent_ratio']:.6g}"
                        )

            firm_frame = pd.DataFrame()
            ondist_parent_meta: Dict[str, Any] | None = None
            ondist_primary_meta: Dict[str, Any] | None = None
            if firm_path is not None:
                firm_frame, parent_states, hatc_cal, lnk_cal, ondist_parent_meta = _sample_parent_tensors(
                    firm_path, macro_path, device=device, max_parents=args.max_sdf_parents
                )
                ondist_transition = build_frozen_transition_data(
                    loaded.models["sdf_fc1"], parent_states, reference_state,
                    loaded.hyperparams, loaded.economic_config,
                    n_child_shocks=args.n_child_shocks, shock_seed=args.shock_seed,
                    shock_bank_max_child_shocks=max_j,
                    hatc_cal_values=hatc_cal, lnk_cal_values=lnk_cal,
                )
                parent_b = parent_states[:, 0].detach().cpu().numpy().astype(np.float64)
                parent_z = parent_states[:, 1].detach().cpu().numpy().astype(np.float64)
                ondist_grid = FrozenFirmGrid(
                    b_values=parent_b, z_values=np.asarray([0.0]),
                    mesh_b=parent_b.reshape(-1, 1), mesh_z=parent_z.reshape(-1, 1),
                    base_states=parent_states,
                )
                _, ondist_summary = evaluate_bellman_residuals(
                    loaded.models["policy_value"], ondist_grid, ondist_transition,
                    loaded.economic_config, chunk_size=args.forward_chunk_size,
                )
                ondist_summary = {f"ondist_{key}": value for key, value in ondist_summary.items()}
                row.update(ondist_summary)
                pd.DataFrame([ondist_summary]).to_csv(
                    episode_root / "firm" / "on_distribution_metrics.csv", index=False
                )
                _sync_cuda(device)
                sdf_ondist_started = time.perf_counter()
                ondist_sdf_summaries, ondist_sdf_meta = evaluate_sdf_heldout_multi_k(
                    loaded.models["sdf_fc1"], parent_states, hatc_cal=hatc_cal,
                    lnk_cal=lnk_cal, economic_config=loaded.economic_config,
                    child_counts=sdf_child_counts, seed=args.shock_seed,
                    shock_bank_max_children=max_j,
                    normalized_logr_clip=float(getattr(loaded.hyperparams, "sdf_normalized_logr_clip", 20.0)),
                )
                _sync_cuda(device)
                episode_timing["sdf_ondist_seconds"] = time.perf_counter() - sdf_ondist_started
                for child_count, sdf_summary in ondist_sdf_summaries.items():
                    namespaced = namespace_sdf_summary(sdf_summary, scope="ondist")
                    item = {"episode": episode, "scope": "ondist", "n_children": child_count, **namespaced}
                    episode_sdf_rows.append(item)
                    sdf_rows_all.append(item)
                    if child_count == args.n_child_shocks:
                        row.update(namespaced)
                        ondist_primary_meta = {**ondist_sdf_meta, "parent_bank": ondist_parent_meta}
                        if float(namespaced["sdf_ondist_valid_parent_ratio"]) < 1.0:
                            availability_notes.append(
                                f"Episode {episode}: on-distribution SDF valid_parent_ratio="
                                f"{namespaced['sdf_ondist_valid_parent_ratio']:.6g}"
                            )
                moments, unavailable = compute_simulated_moments(select_simulated_state_rows(firm_frame))
                row.update(moments)
                pd.DataFrame([moments]).to_csv(episode_root / "simulation" / "moments.csv", index=False)
                write_json(episode_root / "simulation" / "metadata.json", {"unavailable": unavailable})
            else:
                write_json(episode_root / "simulation" / "missing.json", {"reason": "episode firm artifact absent"})
                availability_notes.append(
                    f"Episode {episode}: episode firm artifact absent; on-distribution and simulation metrics unavailable"
                )
            pd.DataFrame(episode_sdf_rows).to_csv(episode_root / "sdf" / "metrics.csv", index=False)
            write_json(episode_root / "sdf" / "metadata.json", {
                "common": common_primary_meta,
                "ondist": ondist_primary_meta,
                "common_parent_bank_sha256": common_parent_meta["parent_bank_sha256"],
            })

            macro_frame = read_dataframe(macro_path) if macro_path else pd.DataFrame()
            if not macro_frame.empty:
                _sync_cuda(device)
                fc1_started = time.perf_counter()
                fc1_summary, timing, rollout = evaluate_fc1_checkpoint(
                    loaded.models["sdf_fc1"], macro_frame, device=device
                )
                _sync_cuda(device)
                episode_timing["fc1_seconds"] = time.perf_counter() - fc1_started
                row.update(fc1_summary)
                row.update({
                    "fc1_hatc_skill": fc1_summary.get("fc1_hatc_persistence_skill"),
                    "fc1_lnk_skill": fc1_summary.get("fc1_lnk_persistence_skill"),
                    "fc1_hatc_best_shift": fc1_summary.get("fc1_hatc_best_timing_shift"),
                    "fc1_lnk_best_shift": fc1_summary.get("fc1_lnk_best_timing_shift"),
                })
                pd.DataFrame([fc1_summary]).to_csv(episode_root / "fc1" / "metrics.csv", index=False)
                timing.to_csv(episode_root / "fc1" / "timing_alignment.csv", index=False)
                rollout.to_csv(episode_root / "fc1" / "rollout.csv", index=False)
            else:
                write_json(episode_root / "fc1" / "missing.json", {"reason": "macro artifact absent"})
                availability_notes.append(f"Episode {episode}: FC1 unavailable because macro artifact is absent")
            after = {
                name: model_state_hash(loaded.models[name])
                for name in active_model_names
            }
            if before != after:
                raise RuntimeError("Full-run evaluator changed checkpoint model parameters")
            missing_components = []
            if firm_path is None:
                missing_components.append("firm_data")
            if macro_path is None:
                missing_components.append("macro_data")
            row.update({
                "status": "partial" if missing_components else "ok",
                "error": f"unavailable: {', '.join(missing_components)}" if missing_components else "",
            })
            episode_metadata.append({
                "episode": episode, "checkpoint_sha256": _file_hash(checkpoint),
                "model_hashes_before": before, "model_hashes_after": after,
                "model_state_unchanged": True, "firm_metadata": firm_meta,
                "checkpoint_load_count": 1,
                "robustness_scope": args.robustness_scope,
                "selected_child_shocks": selected_child_counts,
                "common_sdf_parent_bank": common_parent_meta,
                "ondist_sdf_parent_bank": ondist_parent_meta,
                "common_sdf_metadata": common_primary_meta,
                "ondist_sdf_metadata": ondist_primary_meta,
                "config_invariant_snapshot": config_snapshot,
                "config_hash": config_hash,
            })
        except Exception as exc:
            row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            error_rows.append({
                "episode": episode, "stage": "evaluation",
                "error_type": type(exc).__name__, "error": str(exc),
            })
        _sync_cuda(device)
        if device.type == "cuda":
            episode_timing["cuda_peak_memory_mb"] = (
                float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
            )
        write_json(episode_root / "metadata.json", {
            "episode": episode,
            "status": row["status"],
            "checkpoint": row.get("checkpoint"),
            "firm_data": row.get("firm_data"),
            "macro_data": row.get("macro_data"),
            "representative_detailed_visuals": episode in representative,
            "robustness_scope": args.robustness_scope,
            "selected_child_shocks": child_counts_by_episode[int(episode)],
            "error": row.get("error", ""),
        })
        pd.DataFrame([row]).to_csv(episode_root / "summary.csv", index=False)
        write_json(episode_root / "summary.json", row)
        headline_rows.append(row)
        episode_timing["total_seconds"] = time.perf_counter() - episode_started
        timing_rows.append(episode_timing)

    log_meta: Dict[str, Any] = {"source": "unavailable", "reason": "no explicit or unambiguous training log"}
    log_path = args.training_log.resolve() if args.training_log else None
    if log_path is None:
        candidates = sorted((run_root / "logs").glob("*")) if (run_root / "logs").is_dir() else []
        candidates = [path for path in candidates if path.is_file()]
        if len(candidates) == 1:
            log_path = candidates[0]
    if log_path is not None and log_path.is_file():
        parsed_log, log_meta = parse_training_log(log_path)
        sdf_log_blocks, sdf_block_meta = parse_sdf_validation_log_blocks(log_path)
        primary_sdf_log, primary_sdf_meta = select_primary_sdf_validation_blocks(sdf_log_blocks)
        log_meta.update({
            "sdf_validation_blocks": sdf_block_meta,
            "sdf_primary_selection": primary_sdf_meta,
        })
    else:
        parsed_log = pd.DataFrame(columns=["episode"])
        sdf_log_blocks = pd.DataFrame()
        primary_sdf_log = pd.DataFrame(columns=["episode"])
        availability_notes.append("Training log diagnostics unavailable: no explicit or unambiguous log path")

    if not parsed_log.empty:
        for _, log_row in parsed_log.iterrows():
            episode = int(log_row["episode"])
            episode_log_dir = output / "episodes" / f"ep{episode}" / "training_log"
            if episode_log_dir.is_dir():
                pd.DataFrame([log_row]).to_csv(episode_log_dir / "metrics.csv", index=False)

    headline = _merge_training_log(pd.DataFrame(headline_rows), primary_sdf_log)
    if not parsed_log.empty:
        generic = parsed_log.rename(
            columns={name: f"training_log_{name}" for name in parsed_log if name != "episode"}
        )
        headline = headline.merge(generic, on="episode", how="left")
    missing = MissingArtifacts()
    for episode in log_meta.get("sdf_primary_selection", {}).get("ambiguous_episodes", []):
        missing.add(
            f"Episode {episode}: SDF validation primary block is ambiguous; headline log fields are NaN"
        )
    for warning in discovery_warnings:
        missing.add(warning)
    for item in error_rows:
        missing.add(
            f"Episode {item.get('episode')} {item.get('stage')} failure: "
            f"{item.get('error', item.get('error_type', 'unknown error'))}"
        )
    for note in availability_notes:
        missing.add(note)
    headline, drift = _add_function_drift(
        headline, output, list(args.eta_values), missing
    )
    for name in HEADLINE_COLUMNS:
        if name not in headline:
            headline[name] = np.nan
    headline = headline[
        HEADLINE_COLUMNS + [c for c in headline.columns if c not in HEADLINE_COLUMNS]
    ].sort_values("episode")
    headline.to_csv(output / "headline_metrics.csv", index=False)
    status_summary = summarize_episode_statuses(headline, n_requested=len(episodes))
    write_json(output / "run_summary.json", {
        "episodes": headline.to_dict(orient="records"),
        **status_summary,
        "deprecated_fields": {
            "n_error_or_missing": "alias for n_error + n_missing; partial is excluded"
        },
    })
    headline.to_csv(output / "run_summary.csv", index=False)
    pd.DataFrame(error_rows).to_csv(output / "errors.csv", index=False)
    parsed_log.to_csv(output / "training_log_metrics.csv", index=False)
    equation_cols = [
        c for c in headline
        if c.startswith(("p0_residual_", "pi_residual_", "q_residual_",
                         "p0_trainM_", "p0_rawM_", "pi_trainM_", "pi_rawM_",
                         "q_trainM_", "q_rawM_"))
    ]
    headline[["episode", *equation_cols]].to_csv(
        output / "equation_residual_trajectories.csv", index=False
    )

    _write_dashboard(headline, output)
    _write_error_vs_drift(headline, output)
    _write_output_schema(
        headline, parsed_log, drift, pd.DataFrame(structural_rows),
        pd.DataFrame(sdf_rows_all), sdf_log_blocks, output,
    )
    config_table, config_comparability = compare_config_invariant_snapshots(config_records)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    config_table.to_csv(output / "tables" / "cross_episode_config_invariants.csv", index=False)
    (output / "cross_episode").mkdir(parents=True, exist_ok=True)
    write_json(
        output / "cross_episode" / "config_comparability.json",
        config_comparability,
    )
    for episode in config_comparability["mismatched_episodes"]:
        fields = config_table.loc[
            config_table["episode"] == episode, "diff_fields"
        ].iloc[0]
        missing.add(
            f"Episode {episode}: config invariant mismatch versus episode "
            f"{config_comparability['reference_episode']}: {fields}"
        )
    shutil.copyfile(
        output / "convergence_dashboard.png",
        output / "figures" / "equilibrium_convergence_dashboard.png",
    )
    error_vs_drift_plot = output / "cross_episode" / "equation_error_vs_drift" / "equation_error_vs_drift_rawM.png"
    if error_vs_drift_plot.is_file():
        shutil.copyfile(error_vs_drift_plot, output / "figures" / "equation_error_vs_drift_rawM.png")
    missing.write(output / "missing_artifacts.md")

    metric_sources = {}
    for name in headline.columns:
        if name.startswith("training_log_") or name in {
            "sdf_before_aio_mean", "sdf_after_aio_mean", "sdf_before_aio_t",
            "sdf_after_aio_t", "sdf_safe_to_continue", "sdf_stage_progress",
            "sdf_converged",
        }:
            metric_sources[name] = "parsed_explicit_training_log"
        elif name.endswith("_mean_abs_diff"):
            metric_sources[name] = "derived_adjacent_checkpoint_function_drift"
        elif name.startswith(("mean_b", "std_b", "median_b", "p90_b", "p95_b", "p99_b", "mean_bp", "p90_bp", "p99_bp")) or name in {
            "default_rate", "survival_rate", "investment_rate", "alive_firm_count",
            "entrant_count", "exit_count",
        }:
            metric_sources[name] = "structured_simulation_artifact"
        elif name not in {"episode", "status", "checkpoint", "firm_data", "macro_data", "error"}:
            metric_sources[name] = "formal_read_only_checkpoint_evaluator"
    _sync_cuda(device)
    timing_frame = pd.DataFrame(timing_rows)
    timing_payload = {
        "total_seconds": time.perf_counter() - total_started,
        "checkpoint_load_count": int(pd.to_numeric(
            timing_frame.get("checkpoint_load_count"), errors="coerce"
        ).sum()),
        "firm_matrix_wall_seconds": float(pd.to_numeric(
            timing_frame.get("firm_matrix_wall_seconds"), errors="coerce"
        ).sum()),
        "firm_structural_seconds": float(pd.to_numeric(
            timing_frame.get("firm_structural_seconds"), errors="coerce"
        ).sum()),
        "sdf_common_seconds": float(pd.to_numeric(
            timing_frame.get("sdf_common_seconds"), errors="coerce"
        ).sum()),
        "sdf_ondist_seconds": float(pd.to_numeric(
            timing_frame.get("sdf_ondist_seconds"), errors="coerce"
        ).sum()),
        "bellman_seconds": float(pd.to_numeric(
            timing_frame.get("bellman_seconds"), errors="coerce"
        ).sum()),
        "bp_seconds": float(pd.to_numeric(
            timing_frame.get("bp_seconds"), errors="coerce"
        ).sum()),
        "investment_seconds": float(pd.to_numeric(
            timing_frame.get("investment_seconds"), errors="coerce"
        ).sum()),
        "fc1_seconds": float(pd.to_numeric(
            timing_frame.get("fc1_seconds"), errors="coerce"
        ).sum()),
        "cuda_peak_memory_mb": aggregate_episode_peak_memory(
            [item.get("cuda_peak_memory_mb") for item in timing_rows]
        ),
        "timing_semantics": {
            "firm_matrix_wall_seconds": "wall time for evaluate_matrix",
            "firm_structural_seconds": (
                "derived sum of firm_static, investment, bellman, and bp component times"
            ),
            "total_seconds": "full evaluator wall time; component fields are not re-summed into it",
        },
        "episodes": timing_rows,
    }
    write_json(output / "evaluation_timing.json", timing_payload)
    metadata = {
        "evaluator": "full_run_checkpoint_evaluator_v2",
        "run_root": str(run_root), "git_commit": _git(["rev-parse", "HEAD"]),
        "episodes": episodes, "representative_visual_episodes": sorted(representative),
        "robustness_scope": args.robustness_scope,
        "primary_n_child_shocks": args.n_child_shocks,
        "robustness_child_shocks": args.robustness_child_shocks,
        "selected_child_shocks_by_episode": child_counts_by_episode,
        "reference_firm_data": str(reference_firm),
        "reference_macro_data": str(reference_macro) if reference_macro else None,
        "reference_state": reference_state.to_dict(),
        "common_grid": {
            "b": [args.b_min, args.b_max, args.b_points],
            "z": [args.z_min, args.z_max, args.z_points], "eta_values": args.eta_values,
        },
        "common_shock_bank": {
            "seed": args.shock_seed,
            "max_children": max(max(values) for values in child_counts_by_episode.values()),
            "robustness_children": args.robustness_child_shocks,
            "robustness_scope": args.robustness_scope,
            "selected_children_by_episode": child_counts_by_episode,
            "nested_prefix": True,
            "sdf_common_parent_bank_sha256": common_parent_meta["parent_bank_sha256"],
            "sdf_shock_bank_sha256": next((
                item.get("common_sdf_metadata", {}).get("shock_bank_sha256")
                for item in episode_metadata
                if item.get("common_sdf_metadata")
            ), None),
            "structural_actual_shock_bank_sha256_by_episode": {
                str(item["episode"]): (
                    item.get("firm_metadata", {}).get("reference_transition_bank", {}) or {}
                ).get("shock_bank_sha256")
                for item in episode_metadata
            },
        },
        "reference_firm_sha256": _file_hash(reference_firm),
        "reference_macro_sha256": _file_hash(reference_macro) if reference_macro else None,
        "training_log": log_meta, "discovery_warnings": discovery_warnings,
        "episode_metadata": episode_metadata,
        "cross_episode_comparable": bool(config_comparability["all_comparable"]),
        "config_invariants": config_comparability,
        "bellman_residual_semantics": {
            "trainM": (
                "current-policy fixed-point residual using M_used/clipped-M semantics; "
                "not historical target-network training residual"
            ),
            "rawM": "current-policy fixed-point residual using raw SDF M",
        },
        "timing": timing_payload,
        "metric_source_priority": "formal evaluator > structured artifact > parsed log",
        "metric_sources": metric_sources,
        "primary_structural_eta": 1.0,
        "residual_alias_semantics": {
            "p0_residual_*": "p0_trainM_residual_*",
            "pi_residual_*": "pi_trainM_residual_*",
            "q_residual_*": "q_trainM_residual_*",
        },
        "equation_error_vs_drift": {
            "primary": "rawM",
            "supplementary": "trainM",
            "p0_drift_surface": "P0",
            "pi_drift_surface": "PI_mid",
        },
        "sdf_scopes": {
            "common": "fixed parent bank from reference artifact for model convergence",
            "ondist": "episode-specific parent bank for visited-distribution fit",
        },
    }
    write_json(output / "metadata.json", metadata)
    (output / "README.md").write_text(
        "# Full-run checkpoint evaluation\n\n"
        f"Episodes: {episodes}\n\n"
        "Formal evaluator metrics are primary. Missing checkpoint/model/data fields remain NaN "
        "and are listed in `missing_artifacts.md`, `errors.csv`, or component `missing.json` files.\n\n"
        "Cross-episode function drift uses the same frozen reference state, exact grid, and "
        "nested common-random-number shock-bank definition for every checkpoint.\n",
        encoding="utf-8",
    )
    print(f"Full-run evaluation written to: {output}")
    print(headline[["episode", "status", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
