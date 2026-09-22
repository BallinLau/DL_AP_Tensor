from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.convergence_artifacts import (  # noqa: E402
    EpisodeEvaluation,
    discover_episode_dirs,
    discover_episode_firm_data,
    load_metadata,
    load_surface_csv,
    parse_episode_selection,
    read_dataframe,
    select_simulated_state_rows,
    validate_grid_comparability,
)
from evaluation.convergence_metrics import (  # noqa: E402
    choose_representative_episodes,
    compute_shift_metrics,
    compute_simulated_moments,
    function_drift_metrics,
    regression_metrics,
    residual_metrics,
)
from evaluation.convergence_plots import (  # noqa: E402
    plot_bellman_dashboard,
    plot_bellman_metric,
    plot_distribution,
    plot_function_dashboard,
    plot_function_drift,
    plot_joint_distribution,
    plot_macro_scatter,
    plot_macro_timeseries,
    plot_moment_lines,
    plot_shift_metrics,
    plot_stage_dashboard,
)


SURFACE_PATHS = {
    "Q": Path("q/Q.csv"),
    "P": Path("value/P.csv"),
    "P0": Path("value/P0.csv"),
    "PI": Path("value/PI_mid.csv"),
    "bar_z": Path("default/bar_z.csv"),
    "bp": Path("bp/bp_raw.csv"),
}

BELLMAN_PATHS = {
    "P0": (Path("bellman/R0_signed.csv"), Path("bellman/abs_R0.csv")),
    "PI": (Path("bellman/RI_signed.csv"), Path("bellman/abs_RI.csv")),
    # Current firm evaluator does not emit Q residual. These canonical names
    # allow a future artifact to be consumed without interpreting Q.csv as one.
    "Q": (Path("bellman/RQ_signed.csv"), Path("bellman/abs_RQ.csv")),
}


class MissingArtifacts:
    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, message: str) -> None:
        if message not in self.items:
            self.items.append(message)
        print(f"WARNING: {message}")

    def write(self, path: Path) -> None:
        lines = ["# Missing Artifacts", ""]
        if self.items:
            lines.extend(f"- {item}" for item in self.items)
        else:
            lines.append("- None")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a read-only cross-episode convergence report from existing artifacts."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eta", choices=("0", "1", "both"), default="both")
    parser.add_argument("--episodes", default=None, help="Episode list/range, e.g. 0:14 or 0,2,10")
    parser.add_argument("--timing-max-shift", type=int, default=2)
    parser.add_argument("--macro-path", type=int, default=None)
    return parser.parse_args()


def _eta_values(spec: str) -> list[int]:
    return [0, 1] if spec == "both" else [int(spec)]


def _episode_case_metadata(case: EpisodeEvaluation, eta: int) -> dict[str, Any]:
    case_path = case.eta_dir(eta) / "metadata.json"
    if case_path.is_file():
        return load_metadata(case_path)
    return load_metadata(case.root / "metadata.json")


def compute_function_drift(
    cases: list[EpisodeEvaluation],
    eta: int,
    output_dir: Path,
    missing: MissingArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for previous, current in zip(cases, cases[1:]):
        previous_eta = previous.eta_dir(eta)
        current_eta = current.eta_dir(eta)
        if not previous_eta.is_dir() or not current_eta.is_dir():
            reason = f"eta{eta} output absent for episode pair {previous.episode}->{current.episode}"
            missing.add(reason)
            comparisons.append({
                "eta": eta, "previous_episode": previous.episode, "episode": current.episode,
                "comparable": False, "reason": reason,
            })
            continue
        metadata_ok, metadata_diffs = validate_grid_comparability(
            _episode_case_metadata(previous, eta), _episode_case_metadata(current, eta)
        )
        if not metadata_ok:
            reason = "; ".join(metadata_diffs)
            comparisons.append({
                "eta": eta, "previous_episode": previous.episode, "episode": current.episode,
                "comparable": False, "reason": reason,
            })
            missing.add(
                f"Skipped eta{eta} function drift {previous.episode}->{current.episode}: {reason}"
            )
            continue
        pair_ok = True
        for surface, relative_path in SURFACE_PATHS.items():
            old_path, new_path = previous_eta / relative_path, current_eta / relative_path
            if not old_path.is_file() or not new_path.is_file():
                pair_ok = False
                missing.add(
                    f"Missing {surface} surface for eta{eta} episode pair "
                    f"{previous.episode}->{current.episode}"
                )
                continue
            try:
                metrics = function_drift_metrics(
                    load_surface_csv(new_path), load_surface_csv(old_path)
                )
            except ValueError as exc:
                pair_ok = False
                missing.add(
                    f"Skipped {surface} eta{eta} drift {previous.episode}->{current.episode}: {exc}"
                )
                continue
            rows.append({
                "eta": eta,
                "previous_episode": previous.episode,
                "episode": current.episode,
                "surface": surface,
                **metrics,
            })
        comparisons.append({
            "eta": eta, "previous_episode": previous.episode, "episode": current.episode,
            "comparable": pair_ok, "reason": "exact metadata and grid alignment" if pair_ok else "surface missing/misaligned",
        })

    frame = pd.DataFrame(rows)
    comparison_frame = pd.DataFrame(comparisons)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"eta{eta}_function_drift.csv", index=False)
    comparison_frame.to_csv(output_dir / f"eta{eta}_grid_comparability.csv", index=False)
    if not frame.empty:
        for metric in ("max_abs_diff", "mean_abs_diff", "rmse_diff"):
            plot_function_drift(
                frame, metric, output_dir / f"eta{eta}_function_drift_{metric.split('_')[0]}.png",
                title=f"eta{eta} function drift: {metric}",
            )
    return frame, comparison_frame


def compute_bellman_trajectory(
    cases: list[EpisodeEvaluation],
    eta_values: Iterable[int],
    output_dir: Path,
    missing: MissingArtifacts,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for eta in eta_values:
            eta_dir = case.eta_dir(eta)
            for equation, (signed_relative, abs_relative) in BELLMAN_PATHS.items():
                signed_path, abs_path = eta_dir / signed_relative, eta_dir / abs_relative
                if signed_path.is_file():
                    surface = load_surface_csv(signed_path)
                    metrics = residual_metrics(surface.values)
                    source = signed_path
                elif abs_path.is_file():
                    surface = load_surface_csv(abs_path)
                    metrics = residual_metrics(surface.values)
                    metrics["mean_signed"] = float("nan")
                    metrics["median_signed"] = float("nan")
                    source = abs_path
                else:
                    missing.add(
                        f"{equation} Bellman residual absent for episode {case.episode}, eta{eta}; "
                        f"expected {signed_relative} or {abs_relative}"
                    )
                    continue
                rows.append({
                    "episode": case.episode, "eta": eta, "equation": equation,
                    "source": str(source), **metrics,
                })
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "bellman_residual_by_episode.csv", index=False)
    if not frame.empty:
        for metric, name in (
            ("mean_abs", "bellman_mean_by_episode.png"),
            ("p90_abs", "bellman_p90_by_episode.png"),
            ("p99_abs", "bellman_p99_by_episode.png"),
            ("max_abs", "bellman_max_by_episode.png"),
        ):
            plot_bellman_metric(frame, metric, output_dir / name)
        plot_bellman_dashboard(frame, output_dir / "bellman_convergence_dashboard.png")
    return frame


def _load_firm_states(
    run_root: Path,
    cases: list[EpisodeEvaluation],
    missing: MissingArtifacts,
) -> tuple[dict[int, pd.DataFrame], dict[int, Path]]:
    paths, warnings = discover_episode_firm_data(run_root)
    for warning in warnings:
        missing.add(warning)
    states: dict[int, pd.DataFrame] = {}
    for episode, path in paths.items():
        try:
            states[episode] = select_simulated_state_rows(read_dataframe(path))
        except (OSError, TypeError, ValueError) as exc:
            missing.add(f"Could not load episode {episode} firm simulation {path}: {exc}")
    if not states:
        final_path = run_root / "data" / "outputs" / "final_simulate_firm.pkl"
        if final_path.is_file():
            episode = max((case.episode for case in cases), default=-1)
            states[episode] = select_simulated_state_rows(read_dataframe(final_path))
            paths[episode] = final_path
            missing.add("Episode-level firm simulation artifacts unavailable; using final_simulate_firm.pkl only.")
        else:
            missing.add("Episode-level firm simulation artifacts unavailable, and final_simulate_firm.pkl is absent.")
    return states, paths


def compute_simulation_outputs(
    run_root: Path,
    cases: list[EpisodeEvaluation],
    output_root: Path,
    missing: MissingArtifacts,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    states, paths = _load_firm_states(run_root, cases, missing)
    rows: list[dict[str, Any]] = []
    for episode, frame in states.items():
        moments, unavailable = compute_simulated_moments(frame)
        rows.append({"episode": episode, "source": str(paths[episode]), **moments})
        if unavailable:
            missing.add(f"Episode {episode} simulated moments unavailable: {', '.join(unavailable)}")
    moments_frame = pd.DataFrame(rows).sort_values("episode") if rows else pd.DataFrame()
    moments_dir = output_root / "simulated_moments"
    moments_dir.mkdir(parents=True, exist_ok=True)
    moments_frame.to_csv(moments_dir / "simulated_moments_by_episode.csv", index=False)
    if not moments_frame.empty:
        plot_moment_lines(
            moments_frame, ["mean_b", "std_b", "median_b", "p90_b", "p95_b", "p99_b"],
            moments_dir / "leverage_moments.png", title="Simulated leverage moments",
        )
        plot_moment_lines(
            moments_frame, ["default_rate", "survival_rate", "investment_rate"],
            moments_dir / "default_investment_rate.png", title="Default and investment rates",
        )
        plot_moment_lines(
            moments_frame, ["alive_firm_count", "entrant_count", "exit_count"],
            moments_dir / "firm_count.png", title="Firm counts",
        )
        plot_moment_lines(
            moments_frame, ["mean_M", "std_M"],
            moments_dir / "sdf_moments.png", title="SDF moments",
        )

    distribution_dir = output_root / "simulated_distribution"
    distribution_dir.mkdir(parents=True, exist_ok=True)
    eval_by_episode = {case.episode: case for case in cases}
    for episode in choose_representative_episodes(states):
        frame = states[episode]
        if "b" in frame.columns:
            plot_distribution(frame["b"].to_numpy(), distribution_dir / f"ep{episode}_b_distribution.png", xlabel="b", title=f"Episode {episode}: simulated leverage")
        if "z" in frame.columns:
            plot_distribution(frame["z"].to_numpy(), distribution_dir / f"ep{episode}_z_distribution.png", xlabel="z", title=f"Episode {episode}: simulated productivity")
        if {"b", "z"}.issubset(frame.columns):
            boundaries: dict[str, pd.DataFrame] = {}
            case = eval_by_episode.get(episode)
            if case is not None:
                for eta in (0, 1):
                    boundary_path = case.eta_dir(eta) / "default" / "default_boundary.csv"
                    if boundary_path.is_file():
                        boundaries[f"eta{eta} Phat=0"] = pd.read_csv(boundary_path)
            plot_joint_distribution(
                frame["b"].to_numpy(), frame["z"].to_numpy(),
                distribution_dir / f"ep{episode}_bz_joint.png",
                title=f"Episode {episode}: actual simulated (b,z)", boundaries=boundaries,
            )
    return moments_frame, states


def _macro_branch_groups(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if "branch" not in frame.columns:
        return [("all", frame)]
    branch = pd.to_numeric(frame["branch"], errors="coerce")
    return [(f"branch_{int(value)}", frame.loc[branch == value].copy()) for value in sorted(branch.dropna().unique())]


def compute_macro_outputs(
    run_root: Path,
    output_root: Path,
    *,
    timing_max_shift: int,
    requested_path: int | None,
    missing: MissingArtifacts,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    path = run_root / "data" / "outputs" / "final_simulate_macro.pkl"
    macro_dir = output_root / "macro"
    macro_dir.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        missing.add(f"final macro simulation absent: {path}")
        return pd.DataFrame(), {}, {}
    frame = read_dataframe(path)
    info = {
        "path": str(path), "type": type(frame).__name__, "shape": list(frame.shape),
        "columns": list(frame.columns), "index_type": type(frame.index).__name__,
    }
    print("Macro artifact inspection:")
    print(json.dumps(info, indent=2, default=str))
    (macro_dir / "macro_artifact_structure.json").write_text(
        json.dumps(info, indent=2, default=str), encoding="utf-8"
    )
    mappings = {
        "Hatc": ("Hatc", "hatcf"),
        "LnK": ("LnK", "lnkf"),
    }
    metric_rows: list[dict[str, Any]] = []
    groups = _macro_branch_groups(frame)
    for group_name, group in groups:
        for variable, (calculated_col, forecast_col) in mappings.items():
            if calculated_col not in group.columns or forecast_col not in group.columns:
                missing.add(
                    f"Macro {variable} comparison missing columns {calculated_col}/{forecast_col}"
                )
                continue
            metrics = regression_metrics(group[calculated_col].to_numpy(), group[forecast_col].to_numpy())
            metric_rows.append({
                "variable": variable, "branch_group": group_name,
                "x": calculated_col, "y": forecast_col, **metrics,
            })
            plot_macro_scatter(
                group[calculated_col].to_numpy(), group[forecast_col].to_numpy(),
                macro_dir / f"{variable.lower()}_forecast_vs_cal_{group_name}.png",
                title=f"{variable}: forecast vs calculated ({group_name})",
                calculated_label=f"calculated {calculated_col}", forecast_label=f"forecast {forecast_col}",
            )
            if group_name == "branch_-1" or (group_name == "all" and len(groups) == 1):
                plot_macro_scatter(
                    group[calculated_col].to_numpy(), group[forecast_col].to_numpy(),
                    macro_dir / f"{variable.lower()}_forecast_vs_cal.png",
                    title=f"{variable}: forecast vs calculated ({group_name})",
                    calculated_label=f"calculated {calculated_col}",
                    forecast_label=f"forecast {forecast_col}",
                )
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(macro_dir / "macro_forecast_metrics.csv", index=False)

    if "branch" in frame.columns and bool((pd.to_numeric(frame["branch"], errors="coerce") == -1).any()):
        time_frame = frame.loc[pd.to_numeric(frame["branch"], errors="coerce") == -1].copy()
        time_branch = "branch_-1_parent"
    elif "branch" in frame.columns and bool((pd.to_numeric(frame["branch"], errors="coerce") == 0).any()):
        time_frame = frame.loc[pd.to_numeric(frame["branch"], errors="coerce") == 0].copy()
        time_branch = "branch_0"
    else:
        time_frame, time_branch = frame.copy(), "all"

    shift_frames: dict[str, pd.DataFrame] = {}
    if {"path", "t"}.issubset(time_frame.columns):
        shifts = list(range(-int(timing_max_shift), int(timing_max_shift) + 1))
        for variable, (calculated_col, forecast_col) in mappings.items():
            if calculated_col not in time_frame.columns or forecast_col not in time_frame.columns:
                continue
            shift = compute_shift_metrics(
                time_frame, forecast_col=forecast_col, calculated_col=calculated_col, shifts=shifts,
            )
            shift.insert(0, "variable", variable)
            shift.insert(1, "branch_group", time_branch)
            shift.to_csv(macro_dir / f"{variable.lower()}_shift_metrics.csv", index=False)
            plot_shift_metrics(
                shift, macro_dir / f"{variable.lower()}_shift_diagnostic.png",
                title=f"{variable} timing alignment ({time_branch})",
            )
            shift_frames[variable] = shift

        available_paths = sorted(pd.to_numeric(time_frame["path"], errors="coerce").dropna().astype(int).unique())
        chosen = []
        if available_paths:
            chosen.extend([available_paths[0], available_paths[len(available_paths) // 2]])
            if requested_path is not None and requested_path in available_paths:
                chosen.append(requested_path)
            elif requested_path is not None:
                missing.add(f"Requested macro path {requested_path} is unavailable")
        for path_id in sorted(set(chosen)):
            path_frame = time_frame.loc[pd.to_numeric(time_frame["path"], errors="coerce") == path_id]
            for variable, (calculated_col, forecast_col) in mappings.items():
                if calculated_col in path_frame.columns and forecast_col in path_frame.columns:
                    plot_macro_timeseries(
                        path_frame, macro_dir / f"{variable.lower()}_timeseries_overlay_path{path_id}.png",
                        time_col="t", calculated_col=calculated_col, forecast_col=forecast_col,
                        title=f"{variable} path {path_id}: current-node calculated and forecast",
                    )
    else:
        missing.add("Macro timing diagnostics require path and t columns")
    return metrics_frame, shift_frames, info


def _latest_surface_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    for (eta, surface), group in frame.groupby(["eta", "surface"]):
        row = group.sort_values("episode").iloc[-1]
        for metric in ("max_abs_diff", "mean_abs_diff", "rmse_diff", "p90_abs_diff", "p99_abs_diff"):
            rows.append({"module": "function_drift", "eta": eta, "episode": row["episode"], "object": surface, "metric": metric, "value": row[metric]})
    return rows


def build_summary_metrics(
    function_frame: pd.DataFrame,
    bellman: pd.DataFrame,
    moments: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    rows = _latest_surface_rows(function_frame)
    if not bellman.empty:
        for (eta, equation), group in bellman.groupby(["eta", "equation"]):
            row = group.sort_values("episode").iloc[-1]
            for metric in ("mean_abs", "p90_abs", "max_abs", "mean_signed"):
                rows.append({"module": "bellman", "eta": eta, "episode": row["episode"], "object": equation, "metric": metric, "value": row[metric]})
    if not moments.empty:
        row = moments.sort_values("episode").iloc[-1]
        for metric in ("default_rate", "investment_rate", "mean_b", "p90_b"):
            if metric in row.index:
                rows.append({"module": "simulation", "eta": np.nan, "episode": row["episode"], "object": "firm", "metric": metric, "value": row[metric]})
    if not macro.empty:
        preferred = macro.loc[macro["branch_group"] == "branch_-1"]
        if preferred.empty:
            preferred = macro
        for _, row in preferred.iterrows():
            for metric in ("r2", "pearson_corr", "slope", "rmse", "mae"):
                rows.append({"module": "macro", "eta": np.nan, "episode": np.nan, "object": row["variable"], "metric": metric, "value": row[metric]})
    return pd.DataFrame(rows, columns=["module", "eta", "episode", "object", "metric", "value"])


def _metric_text(frame: pd.DataFrame, object_name: str, metrics: Iterable[str]) -> str:
    subset = frame[frame["object"] == object_name]
    if subset.empty:
        return "N/A"
    values = {row["metric"]: row["value"] for _, row in subset.iterrows()}
    return ", ".join(f"{metric}={values.get(metric, float('nan')):.6g}" for metric in metrics)


def write_readme(
    path: Path,
    *,
    run_root: Path,
    cases: list[EpisodeEvaluation],
    eta_values: list[int],
    summary: pd.DataFrame,
    comparability: pd.DataFrame,
    shift_frames: dict[str, pd.DataFrame],
    missing: MissingArtifacts,
) -> None:
    lines = [
        "# Convergence Report", "", "## Run", "",
        f"- Run path: `{run_root}`",
        f"- Episodes found: `{[case.episode for case in cases]}`",
        f"- Eta modes: `{eta_values}`",
        f"- Generated UTC: `{datetime.now(timezone.utc).isoformat()}`", "",
        "## Grid Comparability", "",
    ]
    if comparability.empty:
        lines.append("No adjacent episode pair was available for comparison.")
    else:
        for _, row in comparability.iterrows():
            lines.append(
                f"- eta{int(row['eta'])} ep{int(row['previous_episode'])}->ep{int(row['episode'])}: "
                f"{'comparable' if bool(row['comparable']) else 'not comparable'}; {row['reason']}"
            )
    lines.extend(["", "## Function Drift", ""])
    function = summary[summary["module"] == "function_drift"]
    for name in ("Q", "P", "bar_z", "bp"):
        lines.append(f"- {name}: {_metric_text(function, name, ['mean_abs_diff', 'max_abs_diff'])}")
    lines.extend(["", "## Bellman Residual", ""])
    bellman = summary[summary["module"] == "bellman"]
    for name in ("P0", "PI", "Q"):
        lines.append(f"- {name}: {_metric_text(bellman, name, ['mean_abs', 'p90_abs', 'max_abs'])}")
    lines.extend(["", "## Simulated Moments", ""])
    simulation = summary[summary["module"] == "simulation"]
    for metric in ("default_rate", "investment_rate", "mean_b", "p90_b"):
        subset = simulation[simulation["metric"] == metric]
        value = float(subset["value"].iloc[-1]) if not subset.empty else float("nan")
        lines.append(f"- {metric}: {value:.6g}")
    lines.extend(["", "## Macro Fit", ""])
    macro = summary[summary["module"] == "macro"]
    for name in ("Hatc", "LnK"):
        lines.append(f"- {name}: {_metric_text(macro, name, ['r2', 'pearson_corr', 'slope', 'rmse'])}")
    lines.extend(["", "## Timing Alignment", ""])
    for name in ("Hatc", "LnK"):
        shift = shift_frames.get(name)
        if shift is None or shift.empty:
            lines.append(f"- {name}: N/A")
            continue
        corr_rows = shift[np.isfinite(shift["pearson_corr"])]
        rmse_rows = shift[np.isfinite(shift["rmse"])]
        best_corr = int(corr_rows.loc[corr_rows["pearson_corr"].idxmax(), "shift"]) if not corr_rows.empty else None
        best_rmse = int(rmse_rows.loc[rmse_rows["rmse"].idxmin(), "shift"]) if not rmse_rows.empty else None
        lines.append(
            f"- {name}: highest correlation at shift={best_corr}; minimum RMSE at shift={best_rmse}. "
            "Here shift=+1 means forecast_t is compared with calculated_(t+1)."
        )
    lines.extend(["", "## Missing Artifacts", ""])
    lines.extend(f"- {item}" for item in missing.items) if missing.items else lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    missing = MissingArtifacts()
    selection = parse_episode_selection(args.episodes)
    cases, discovery_warnings = discover_episode_dirs(run_root, episodes=selection)
    for warning in discovery_warnings:
        missing.add(warning)
    if not cases:
        missing.add(f"No ep*_eta_matrix_* evaluator directories found under {run_root}")
    eta_values = _eta_values(args.eta)

    function_frames: list[pd.DataFrame] = []
    comparison_frames: list[pd.DataFrame] = []
    function_dir = output_root / "function_drift"
    for eta in eta_values:
        frame, comparisons = compute_function_drift(cases, eta, function_dir, missing)
        function_frames.append(frame)
        comparison_frames.append(comparisons)
    function_frame = pd.concat(function_frames, ignore_index=True) if function_frames else pd.DataFrame()
    comparison_frame = pd.concat(comparison_frames, ignore_index=True) if comparison_frames else pd.DataFrame()
    if not function_frame.empty:
        plot_function_dashboard(function_frames, function_dir / "function_drift_dashboard.png")

    bellman = compute_bellman_trajectory(
        cases, eta_values, output_root / "bellman_convergence", missing
    )
    moments, _ = compute_simulation_outputs(run_root, cases, output_root, missing)
    macro, shifts, macro_info = compute_macro_outputs(
        run_root, output_root, timing_max_shift=args.timing_max_shift,
        requested_path=args.macro_path, missing=missing,
    )
    summary = build_summary_metrics(function_frame, bellman, moments, macro)
    summary.to_csv(output_root / "summary_metrics.csv", index=False)
    missing.write(output_root / "missing_artifacts.md")
    write_readme(
        output_root / "README.md", run_root=run_root, cases=cases, eta_values=eta_values,
        summary=summary, comparability=comparison_frame, shift_frames=shifts, missing=missing,
    )
    plot_stage_dashboard(
        function_drift=function_frame, bellman=bellman, moments=moments,
        macro_metrics=macro, path=output_root / "stage_report_dashboard.png",
    )
    manifest = {
        "run_root": str(run_root), "output_dir": str(output_root),
        "episodes": [case.episode for case in cases], "eta_values": eta_values,
        "timing_shift_definition": "shift k compares forecast_t with calculated_(t+k)",
        "macro_artifact": macro_info,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_root / "report_metadata.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(f"Convergence report written to: {output_root}")


if __name__ == "__main__":
    main()
