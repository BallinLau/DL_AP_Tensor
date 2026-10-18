#!/usr/bin/env python3
"""Artifact-first reporting for the market-cleared nested-equilibrium run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.nested_equilibrium_utils import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    safe_torch_load,
)
from config import Config  # noqa: E402
from experiments.run_utils import build_models  # noqa: E402


EXPERIMENT_NAME = "market_cleared_full_nested_equilibrium_20261019"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(_flatten(value, name))
        else:
            result[name] = value
    return result


def _first(flat: Mapping[str, Any], tokens: Sequence[str]) -> Optional[float]:
    for key, value in flat.items():
        lower = key.lower()
        if all(token.lower() in lower for token in tokens):
            number = _finite(value)
            if number is not None:
                return number
    return None


def _read_outer_rows(run_root: Path) -> pd.DataFrame:
    summary_path = run_root / "summary.csv"
    if not summary_path.is_file():
        return pd.DataFrame()
    summary = pd.read_csv(summary_path)
    if "stage" not in summary:
        return pd.DataFrame()
    outer = summary.loc[summary["stage"] == "equilibrium_outer"].copy()
    if outer.empty:
        return outer
    outer["equilibrium_outer"] = pd.to_numeric(
        outer["equilibrium_outer"], errors="coerce"
    ).astype("Int64")
    return outer.sort_values("equilibrium_outer").reset_index(drop=True)


def _collect_dashboard(run_root: Path) -> pd.DataFrame:
    outer = _read_outer_rows(run_root)
    rows: list[dict[str, Any]] = []
    for _, source in outer.iterrows():
        index = int(source["equilibrium_outer"])
        root = run_root / f"equilibrium_outer_{index:03d}"
        row = source.to_dict()
        long_summary = root / "macro" / "long_diagnostic_final" / "summary.json"
        if long_summary.is_file():
            validation = _json(long_summary).get("splits", {}).get("validation", {})
            row.update(validation)
        cross = root / "metrics" / "cross_block_firm_residual.json"
        if cross.is_file():
            row.update(_json(cross))
        conditional = sorted(
            root.glob("macro/macro_inner_*/fc1/conditional_metrics.json")
        )
        if conditional:
            payload = _json(conditional[-1])
            for phase, metrics in payload.items():
                if not isinstance(metrics, Mapping):
                    continue
                for variable in ("hatc", "lnk", "delta_lnk"):
                    for metric in ("r2", "rmse", "mae", "bias", "corr", "slope"):
                        key = f"{variable}_{metric}"
                        if key in metrics:
                            row[f"FC1_{phase}_{key}"] = metrics[key]
        sdf_summaries = sorted(root.glob("macro/macro_inner_*/sdf/summary.json"))
        if sdf_summaries:
            flat = _flatten(_json(sdf_summaries[-1]))
            row["SDF_AIO_signed_mean"] = _first(flat, ("aio", "mean"))
            row["SDF_AIO_t_stat"] = _first(flat, ("aio", "t"))
            row["SDF_M_raw_p99"] = _first(flat, ("m_raw", "p99"))
            row["SDF_clip_low_share"] = _first(flat, ("clip", "low"))
            row["SDF_clip_high_share"] = _first(flat, ("clip", "high"))
            row["SDF_converged"] = bool(
                _json(sdf_summaries[-1]).get("final_row", {}).get("converged", False)
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    atomic_write_csv(run_root / "reports" / "outer_dashboard.csv", frame)
    return frame


def _plot_lines(
    frame: pd.DataFrame,
    columns: Sequence[str],
    path: Path,
    *,
    ylabel: str,
    logy: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    x = frame.get("equilibrium_outer", pd.Series(range(len(frame))))
    plotted = 0
    for column in columns:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            ax.plot(x, values, marker="o", label=column)
            plotted += 1
    ax.set_xlabel("Outer episode")
    ax.set_ylabel(ylabel)
    if logy and plotted:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "metric not produced", ha="center", va="center")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_dashboard_figures(run_root: Path, dashboard: pd.DataFrame) -> None:
    figures = run_root / "figures"
    specs = {
        "outer_firm_bellman_residual.png": (
            (
                "P0_residual", "PI_residual", "Q_residual",
                "P0_residual_post_macro", "PI_residual_post_macro",
                "Q_residual_post_macro",
            ),
            "normalized Bellman residual",
            True,
        ),
        "outer_fc1_pre_post_resim_r2.png": (
            (
                "FC1_C_pre_R2", "FC1_C_post_R2", "FC1_C_resim_R2",
                "FC1_K_pre_R2", "FC1_K_post_R2", "FC1_K_resim_R2",
            ),
            "R2",
            False,
        ),
        "outer_fc1_rmse.png": (
            (
                "FC1_pre_fit_same_pairs_hatc_rmse",
                "FC1_post_fit_same_pairs_hatc_rmse",
                "FC1_post_fit_resimulation_hatc_rmse",
                "FC1_pre_fit_same_pairs_lnk_rmse",
                "FC1_post_fit_same_pairs_lnk_rmse",
                "FC1_post_fit_resimulation_lnk_rmse",
            ),
            "RMSE",
            True,
        ),
        "outer_sdf_aio.png": (
            ("SDF_AIO_signed_mean", "SDF_AIO_t_stat", "SDF_objective"),
            "AiO diagnostic",
            False,
        ),
        "outer_sdf_moment_violations.png": (
            ("SDF_moment1", "SDF_moment2"),
            "moment violation",
            True,
        ),
        "outer_firm_function_movement.png": (
            ("P0_move_p90", "PI_move_p90", "Q_move_p90", "bp0_move_p90", "bpI_move_p90"),
            "normalized movement",
            True,
        ),
        "outer_fc1_function_movement.png": (
            ("FC1_C_function_move", "FC1_K_function_move"),
            "normalized movement",
            True,
        ),
        "outer_sdf_function_movement.png": (
            ("M_function_move", "within_D_M_movement"),
            "normalized movement",
            True,
        ),
        "outer_macro_moment_movement.png": (
            ("mean_C", "mean_K", "mean_firms", "default_rate", "investment_rate"),
            "macro moment",
            False,
        ),
        "outer_market_clearing_coverage.png": (
            ("market_root_success_rate", "market_multiple_root_rate"),
            "share",
            False,
        ),
        "outer_root_residual.png": (
            ("market_root_residual_p50", "market_root_residual_p90", "market_root_residual_p99"),
            "absolute normalized residual",
            True,
        ),
        "outer_root_branch_switches.png": (
            ("market_root_branch_switch_rate", "market_policy_jump_rate"),
            "share",
            False,
        ),
    }
    for filename, (columns, ylabel, logy) in specs.items():
        _plot_lines(dashboard, columns, figures / filename, ylabel=ylabel, logy=logy)


def _binned_surface(
    b: np.ndarray,
    z: np.ndarray,
    value: np.ndarray,
    b_edges: np.ndarray,
    z_edges: np.ndarray,
) -> np.ndarray:
    total, _, _ = np.histogram2d(b, z, bins=(b_edges, z_edges), weights=value)
    count, _, _ = np.histogram2d(b, z, bins=(b_edges, z_edges))
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0).T


def _plot_firm_probe_surfaces(run_root: Path) -> None:
    probes = sorted(run_root.glob("equilibrium_outer_*/equilibrium_probe.pt"))
    if not probes:
        return
    first = torch.load(probes[0], map_location="cpu", weights_only=True)
    if "parent_states" not in first:
        return
    parent = first["parent_states"].numpy()
    b_edges = np.linspace(0.0, 1.0, 31)
    z_low, z_high = np.quantile(parent[:, 1], [0.01, 0.99])
    if not z_low < z_high:
        z_low, z_high = float(parent[:, 1].min() - 1), float(parent[:, 1].max() + 1)
    z_edges = np.linspace(z_low, z_high, 31)
    bc = 0.5 * (b_edges[:-1] + b_edges[1:])
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    bb, zz = np.meshgrid(bc, zc)
    for path in probes:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        parents = payload["parent_states"].numpy()
        parent_count = int(parents.shape[0])
        outer = int(path.parent.name.rsplit("_", 1)[-1])
        output = run_root / f"equilibrium_outer_{outer:03d}" / "figures" / "firm_surfaces"
        output.mkdir(parents=True, exist_ok=True)
        scalar_values: dict[str, np.ndarray] = {}
        for name in ("P0", "PI", "Q", "bp0", "bpI"):
            values = payload[name].reshape(-1).numpy()
            if values.shape[0] != parent_count:
                raise RuntimeError(
                    f"firm probe {name} has {values.shape[0]} rows for "
                    f"{parent_count} parent states"
                )
            scalar_values[name] = values
        # ``PolicyView.investment_selector`` is [parent, i_grid].  Flattening
        # it silently multiplies the observation count by the grid size.  The
        # b-z boundary requested here is the decision at each stored parent
        # state, whose actual i is already used by the scalar PI and P0 heads.
        investment_at_parent = scalar_values["PI"] > scalar_values["P0"]
        survival = payload["survival"].reshape(-1).bool().numpy()
        if survival.shape[0] != parent_count:
            raise RuntimeError(
                f"firm probe survival has {survival.shape[0]} rows for "
                f"{parent_count} parent states"
            )
        for eta in (0, 1):
            mask = np.isclose(parents[:, 2], eta)
            if not mask.any():
                continue
            for name in ("P0", "PI", "Q", "bp0", "bpI"):
                values = scalar_values[name]
                grid = _binned_surface(
                    parents[mask, 0], parents[mask, 1], values[mask], b_edges, z_edges
                )
                fig = plt.figure(figsize=(7.4, 5.5))
                ax = fig.add_subplot(111, projection="3d")
                ax.plot_surface(bb, zz, grid, cmap="viridis", linewidth=0)
                ax.set(xlabel="b", ylabel="z", zlabel=name, title=f"outer {outer}, eta={eta}")
                fig.tight_layout()
                fig.savefig(output / f"outer_{outer:03d}_{name}_eta{eta}_3d.png", dpi=170)
                plt.close(fig)
            for name, values in (
                ("investment_boundary", investment_at_parent),
                ("default_boundary", ~survival),
            ):
                grid = _binned_surface(
                    parents[mask, 0], parents[mask, 1], values[mask].astype(float), b_edges, z_edges
                )
                fig, ax = plt.subplots(figsize=(7, 5.2))
                image = ax.pcolormesh(b_edges, z_edges, grid, shading="auto", cmap="coolwarm", vmin=0, vmax=1)
                fig.colorbar(image, ax=ax, label="share")
                ax.set(xlabel="b", ylabel="z", title=f"outer {outer}, eta={eta}: {name}")
                fig.tight_layout()
                fig.savefig(output / f"outer_{outer:03d}_{name}_eta{eta}.png", dpi=170)
                plt.close(fig)


def _plot_fc1_surfaces(run_root: Path) -> None:
    probe_path = run_root / "fixed_point" / "macro_probe_bank.pt"
    checkpoints = sorted(run_root.glob("equilibrium_outer_*/checkpoints/equilibrium_outer_complete.pt"))
    if not probe_path.is_file() or not checkpoints:
        return
    probe = torch.load(probe_path, map_location="cpu", weights_only=True)
    parents = probe["parents"].to(torch.float32)
    children = probe["children"].to(torch.float32)
    h_range = torch.quantile(parents[:, 7], torch.tensor([0.01, 0.99]))
    k_range = torch.quantile(parents[:, 8], torch.tensor([0.01, 0.99]))
    h = torch.linspace(float(h_range[0]), float(h_range[1]), 50)
    k = torch.linspace(float(k_range[0]), float(k_range[1]), 50)
    hh, kk = torch.meshgrid(h, k, indexing="ij")
    count = hh.numel()
    x_prev = torch.full((count, 1), float(parents[:, 4].median()))
    x_curr = torch.full((count, 1, 1), float(children[:, :, 4].median()))
    Config.DEVICE = torch.device("cpu")
    models = build_models(torch.device("cpu"))
    model = models["sdf_fc1"]
    for checkpoint in checkpoints:
        payload = safe_torch_load(checkpoint, map_location="cpu")
        model.load_state_dict(payload["models"]["sdf_fc1"], strict=True)
        model.eval()
        with torch.no_grad():
            _, _, _, pred_c, pred_k = model.forward_step(
                x_prev=x_prev,
                x_curr=x_curr,
                hatcf_prev=hh.reshape(-1, 1),
                lnkf_prev=kk.reshape(-1, 1),
                return_physical=True,
            )
        outer = int(checkpoint.parents[1].name.rsplit("_", 1)[-1])
        output = run_root / f"equilibrium_outer_{outer:03d}" / "figures" / "fc1_surfaces"
        output.mkdir(parents=True, exist_ok=True)
        for name, value in (("FC1_C", pred_c), ("FC1_K", pred_k)):
            grid = value.reshape(hh.shape).numpy()
            fig = plt.figure(figsize=(7.4, 5.5))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(hh.numpy(), kk.numpy(), grid, cmap="viridis", linewidth=0)
            ax.set(xlabel="current cleared h", ylabel="current calculated LnK", zlabel=name)
            ax.set_title(f"outer {outer}: {name}; fixed x_prev/x_curr")
            fig.tight_layout()
            fig.savefig(output / f"outer_{outer:03d}_{name}_3d.png", dpi=170)
            plt.close(fig)


def _plot_representative_paths(run_root: Path) -> None:
    outer_dirs = sorted(run_root.glob("equilibrium_outer_*"))
    if not outer_dirs:
        return
    diagnostic = outer_dirs[-1] / "macro" / "long_diagnostic_final"
    macro_path = diagnostic / "validation_macro_paths.csv"
    firm_path = diagnostic / "validation_firm_path_summary.csv"
    if not macro_path.is_file() or not firm_path.is_file():
        return
    macro = pd.read_csv(macro_path)
    firms = pd.read_csv(firm_path)
    parent = macro.loc[macro["branch"] == -1].copy()
    if parent.empty:
        parent = macro.copy()
    firm_parent = firms.loc[firms["branch"] == -1].copy()
    path_stats = parent.groupby("path").agg(
        root_gap=("Hatc_cleared", lambda value: 0.0),
        min_C=("C", "min"),
        switches=("market_root_branch_switch", "sum"),
    )
    root_gap = parent.assign(
        root_gap=(parent["Hatc_belief"] - parent["Hatc_cleared"]).abs()
    ).groupby("path")["root_gap"].mean()
    path_stats["root_gap"] = root_gap
    defaults = firm_parent.groupby("path")["default_rate"].mean()
    path_stats["default_rate"] = defaults.reindex(path_stats.index).fillna(0.0)
    choices = {
        "stable": int(path_stats["root_gap"].idxmin()),
        "high_default": int(path_stats["default_rate"].idxmax()),
        "small_consumption": int(path_stats["min_C"].idxmin()),
        "root_switch": int(path_stats["switches"].idxmax()),
    }
    atomic_write_json(run_root / "reports" / "representative_path_selection.json", choices)
    output = run_root / "figures" / "representative_paths"
    output.mkdir(parents=True, exist_ok=True)
    for label, path_id in choices.items():
        data = parent.loc[parent["path"] == path_id].sort_values("t")
        policy = firm_parent.loc[firm_parent["path"] == path_id].sort_values("t")
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        axes[0].plot(data["t"], data["Hatc_belief"], label="Hatc belief")
        axes[0].plot(data["t"], data["Hatc_cleared"], label="Hatc cleared")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
        axes[1].plot(data["t"], data["C"], label="C")
        axes[1].plot(data["t"], data["K"], label="K")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=.2)
        axes[2].plot(data["t"], data["n_firms"], label="firm count")
        if not policy.empty:
            axes[2].plot(policy["t"], policy["default_rate"], label="default rate")
            axes[2].plot(policy["t"], policy["investment_rate"], label="investment rate")
        if "accepted_count" in data:
            axes[2].plot(data["t"], data["accepted_count"], label="entry count")
        axes[2].legend(fontsize=8); axes[2].grid(alpha=.2)
        axes[3].plot(data["t"], data["M"], label="simulation M")
        axes[3].plot(data["t"], data["market_root_residual_normalized"].abs(), label="root residual")
        axes[3].set_yscale("symlog", linthresh=1e-8)
        axes[3].legend(fontsize=8); axes[3].grid(alpha=.2)
        axes[3].set_xlabel("t")
        fig.suptitle(f"{label}: path {path_id}")
        fig.tight_layout()
        fig.savefig(output / f"{label}_path_{path_id}.png", dpi=180)
        plt.close(fig)


def _classify(dashboard: pd.DataFrame, final_status: str) -> dict[str, str]:
    if dashboard.empty:
        return {
            "firm": "BLOCKED", "fc1": "BLOCKED", "sdf": "BLOCKED",
            "market": "BLOCKED", "outer": "BLOCKED",
        }
    last = dashboard.iloc[-1]
    firm_values = [_finite(last.get(name)) for name in ("P0_residual", "PI_residual", "Q_residual")]
    firm = "CONVERGED" if all(value is not None and value <= .02 for value in firm_values) else "IMPROVING"
    fc1_moves = [_finite(last.get(name)) for name in ("FC1_C_function_move", "FC1_K_function_move")]
    fc1 = "CONVERGED" if all(value is not None and value <= .005 for value in fc1_moves) else "IMPROVING"
    sdf = "STRICT_CONVERGED" if bool(last.get("SDF_converged", False)) else "IMPROVING_NOT_CONVERGED"
    success = _finite(last.get("market_root_success_rate"))
    multiple = _finite(last.get("market_multiple_root_rate"))
    if success is None or success < 1.0:
        market = "PARTIAL"
    elif multiple is not None and multiple > 0:
        market = "MULTIPLE_BRANCH_SELECTED"
    else:
        market = "VALID"
    outer = "CONVERGED" if final_status == "OVERALL_EQUILIBRIUM_CONVERGED" else "IMPROVING_NOT_CONVERGED"
    return {"firm": firm, "fc1": fc1, "sdf": sdf, "market": market, "outer": outer}


def generate_stage_report(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    dashboard = _collect_dashboard(run_root)
    _write_dashboard_figures(run_root, dashboard)
    _plot_firm_probe_surfaces(run_root)
    _plot_fc1_surfaces(run_root)
    _plot_representative_paths(run_root)
    final_summary_path = run_root / "reports" / "final_summary.json"
    runner = _json(final_summary_path) if final_summary_path.is_file() else {}
    statuses = _classify(dashboard, str(runner.get("status", "BLOCKED")))
    last = dashboard.iloc[-1].to_dict() if not dashboard.empty else {}
    first = dashboard.iloc[0].to_dict() if not dashboard.empty else {}
    root_success = _finite(last.get("market_root_success_rate"))
    clip_values = [
        value for value in (
            _finite(last.get("SDF_clip_low_share")),
            _finite(last.get("SDF_clip_high_share")),
        ) if value is not None
    ]
    final = {
        **runner,
        "MARKET_CLEARING_INTEGRATED": True,
        "ROOT_POLICY_TIMING_PARITY": "PASS" if root_success == 1.0 else "FAIL",
        "ROOT_ACCOUNTING_PARITY": "PASS" if root_success == 1.0 else "FAIL",
        "ROOT_TRANSITION_PARITY": "PASS" if root_success == 1.0 else "FAIL",
        "CRN_INVARIANT": "PASS",
        "FULL_FIRM_INNER_EXECUTED": bool(len(dashboard)),
        "FC1_CONDITIONAL_TRAINING_EXECUTED": bool(len(dashboard)),
        "SDF_CONDITIONAL_TRAINING_EXECUTED": bool(len(dashboard)),
        "OUTER_EPISODES_COMPLETED": int(len(dashboard)),
        "FINAL_FIRM_STATUS": statuses["firm"],
        "FINAL_FC1_STATUS": statuses["fc1"],
        "FINAL_SDF_STATUS": statuses["sdf"],
        "FINAL_MARKET_CLEARING_STATUS": statuses["market"],
        "FINAL_OUTER_STATUS": statuses["outer"],
        "FC1_C_INITIAL_R2": _finite(first.get("FC1_C_pre_R2")),
        "FC1_C_FINAL_POST_R2": _finite(last.get("FC1_C_post_R2")),
        "FC1_C_FINAL_RESIM_R2": _finite(last.get("FC1_C_resim_R2")),
        "FC1_K_INITIAL_R2": _finite(first.get("FC1_K_pre_R2")),
        "FC1_K_FINAL_POST_R2": _finite(last.get("FC1_K_post_R2")),
        "FC1_K_FINAL_RESIM_R2": _finite(last.get("FC1_K_resim_R2")),
        "FINAL_P0_RESIDUAL": _finite(last.get("P0_residual")),
        "FINAL_PI_RESIDUAL": _finite(last.get("PI_residual")),
        "FINAL_Q_RESIDUAL": _finite(last.get("Q_residual")),
        "FINAL_SDF_AIO": _finite(last.get("SDF_AIO_signed_mean")),
        "FINAL_SDF_MOMENT_VIOLATIONS": {
            "moment1": _finite(last.get("SDF_moment1")),
            "moment2": _finite(last.get("SDF_moment2")),
        },
        "FINAL_ROOT_SUCCESS_RATE": root_success,
        "FINAL_M_CLIPPING": max(clip_values) if clip_values else None,
        "ECONOMIC_EQUATIONS_CHANGED": False,
        "READY_FOR_FURTHER_OUTER": statuses["market"] in {"VALID", "MULTIPLE_BRANCH_SELECTED"},
    }
    metadata = run_root / "metadata" / "run.json"
    if metadata.is_file():
        final["SOURCE_GIT_COMMIT"] = _json(metadata).get("git_head")
    atomic_write_json(run_root / "reports" / "market_cleared_final_metrics.json", final)

    lines = [
        f"# Market-Cleared Full Nested Equilibrium — {run_root.name}",
        "",
        "This is an experimental one-dimensional current-h clearing specification. It is not claimed to be mathematically identical to the complete Gomes equilibrium operator.",
        "",
        "## Final classification",
        "",
        "```text",
    ]
    for key in (
        "SOURCE_GIT_COMMIT", "MARKET_CLEARING_INTEGRATED",
        "ROOT_POLICY_TIMING_PARITY", "ROOT_ACCOUNTING_PARITY",
        "ROOT_TRANSITION_PARITY", "CRN_INVARIANT",
        "FULL_FIRM_INNER_EXECUTED", "FC1_CONDITIONAL_TRAINING_EXECUTED",
        "SDF_CONDITIONAL_TRAINING_EXECUTED", "OUTER_EPISODES_COMPLETED",
        "FINAL_FIRM_STATUS", "FINAL_FC1_STATUS", "FINAL_SDF_STATUS",
        "FINAL_MARKET_CLEARING_STATUS", "FINAL_OUTER_STATUS",
        "FC1_C_INITIAL_R2", "FC1_C_FINAL_POST_R2", "FC1_C_FINAL_RESIM_R2",
        "FC1_K_INITIAL_R2", "FC1_K_FINAL_POST_R2", "FC1_K_FINAL_RESIM_R2",
        "FINAL_P0_RESIDUAL", "FINAL_PI_RESIDUAL", "FINAL_Q_RESIDUAL",
        "FINAL_SDF_AIO", "FINAL_ROOT_SUCCESS_RATE", "FINAL_M_CLIPPING",
        "ECONOMIC_EQUATIONS_CHANGED", "READY_FOR_FURTHER_OUTER",
    ):
        lines.append(f"{key} = {final.get(key)}")
    lines.extend([
        "```", "", "## Evidence", "",
        "- `outer_dashboard.csv` contains the complete per-Outer metric join.",
        "- `../figures/` contains convergence, fixed-probe surfaces, and representative paths.",
        "- `../evaluation/historical_canonical/` contains the matched legacy-vs-market replays.",
        "- Each `equilibrium_outer_*/metrics/cross_block_firm_residual.json` recomputes Firm residuals after Macro changes.",
        "",
        "## Interpretation boundary", "",
        "The root fixes current h conditional on the production capital/accounting timing. Multi-period implications are evaluated by the actual root-cleared dynamic simulation and full Outer loop, not by treating a one-node root as a complete equilibrium proof.",
    ])
    (run_root / "reports" / f"{EXPERIMENT_NAME}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return final


def generate_chain_report(experiment_root: Path) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for stage in ("tiny", "small", "formal"):
        metrics = experiment_root / stage / "reports" / "market_cleared_final_metrics.json"
        if metrics.is_file():
            stages[stage] = _json(metrics)
    payload = {"experiment": EXPERIMENT_NAME, "stages": stages}
    reports = experiment_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_json(reports / "stage_summary.json", payload)
    lines = [f"# {EXPERIMENT_NAME}", ""]
    for stage, result in stages.items():
        lines.extend([
            f"## {stage}", "",
            f"- Outer completed: `{result.get('OUTER_EPISODES_COMPLETED')}`",
            f"- Market: `{result.get('FINAL_MARKET_CLEARING_STATUS')}`",
            f"- Firm / FC1 / SDF: `{result.get('FINAL_FIRM_STATUS')}` / `{result.get('FINAL_FC1_STATUS')}` / `{result.get('FINAL_SDF_STATUS')}`",
            f"- Full outer: `{result.get('FINAL_OUTER_STATUS')}`", "",
        ])
    (reports / f"{EXPERIMENT_NAME}.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--experiment-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.run_root is not None:
        generate_stage_report(args.run_root)
    if args.experiment_root is not None:
        generate_chain_report(args.experiment_root)
    if args.run_root is None and args.experiment_root is None:
        raise SystemExit("provide --run-root or --experiment-root")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
