from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.checkpoint_loader import load_analysis_checkpoint  # noqa: E402
from config import Config  # noqa: E402
from evaluation.boundaries import (  # noqa: E402
    compare_hard_soft_default_boundaries,
    extract_phat_default_boundary,
)
from evaluation.bellman_diagnostics import (  # noqa: E402
    build_child_continuation_audit,
    evaluate_bellman_residuals,
)
from evaluation.bp_diagnostics import (  # noqa: E402
    build_frozen_transition_data,
    evaluate_bp_consistency,
)
from evaluation.firm_surfaces import (  # noqa: E402
    evaluate_firm_surfaces,
    evaluate_investment_cutoff,
    finite_difference_summary,
    investment_margin_diagnostics,
    q_unit_summary,
)
from evaluation.grids import build_frozen_grid, load_reference_state  # noqa: E402
from evaluation.plotting import (  # noqa: E402
    plot_b_slices,
    plot_default_boundary,
    plot_default_boundary_comparison,
    plot_heatmap,
    plot_i_star_slices,
    plot_q_peak,
    save_surface_csvs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only firm-side PolicyValue checkpoint evaluator."
    )
    checkpoints = parser.add_mutually_exclusive_group(required=True)
    checkpoints.add_argument("--checkpoint", type=Path, help="Combined analysis/trainer checkpoint")
    checkpoints.add_argument("--pv-ckpt", type=Path, help="Raw PolicyValue state_dict")
    parser.add_argument("--sdf-ckpt", type=Path, help="Raw SDF/FC1 state_dict paired with --pv-ckpt")
    parser.add_argument("--hyperparams-json", type=Path)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--model-spec-json", type=Path)
    parser.add_argument("--allow-default-hyperparams", action="store_true")
    parser.add_argument("--allow-current-config", action="store_true")
    parser.add_argument("--firm-data", type=Path, required=True)
    parser.add_argument(
        "--macro-data",
        type=Path,
        help="Explicit calculated macro dataframe; takes priority over sibling auto-discovery",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument(
        "--eta-values",
        type=float,
        nargs="+",
        default=None,
        help="Optional parent-eta matrix; e.g. --eta-values 0 1.",
    )
    parser.add_argument("--b-min", type=float, default=None)
    parser.add_argument("--b-max", type=float, default=None)
    parser.add_argument("--b-points", type=int, default=101)
    parser.add_argument("--z-min", type=float, default=-4.0)
    parser.add_argument("--z-max", type=float, default=4.0)
    parser.add_argument("--z-points", type=int, default=101)
    parser.add_argument("--i-points", type=int, default=101)
    parser.add_argument("--forward-chunk-size", type=int, default=8192)
    parser.add_argument("--n-child-shocks", "--n-branches", dest="n_child_shocks", type=int, default=2)
    parser.add_argument("--shock-seed", type=int, default=12345)
    parser.add_argument(
        "--shock-bank-max-child-shocks",
        type=int,
        default=None,
        help=(
            "Generate this many shocks before taking the first J; matrix mode sets this "
            "to max(J) so smaller-J cases are nested prefixes."
        ),
    )
    parser.add_argument(
        "--robustness-child-shocks",
        type=int,
        nargs="+",
        default=None,
        help="Optional J matrix; the primary --n-child-shocks case gets full plots.",
    )
    parser.add_argument(
        "--bp-teacher-margin-tol",
        type=float,
        default=1e-8,
        help="Minimum BPGridTeacher top-two objective margin for an identified bp target.",
    )
    return parser.parse_args()


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_file = ROOT / ".git"
        if git_file.is_file():
            git_dir = Path(git_file.read_text(encoding="utf-8").split(":", 1)[1].strip())
            head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
            if head.startswith("ref: "):
                common = (git_dir / (git_dir / "commondir").read_text().strip()).resolve()
                ref_path = common / head.removeprefix("ref: ")
                if ref_path.exists():
                    return ref_path.read_text(encoding="utf-8").strip()
        return "unavailable"


def _git_dirty() -> bool | None:
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_selected_surfaces(
    surfaces: Dict[str, np.ndarray],
    *,
    b_values: np.ndarray,
    z_values: np.ndarray,
    output_root: Path,
) -> None:
    groups = {
        "value": ["P0", "V0", "PI_low", "PI_mid", "PI_high", "VI_low", "VI_mid", "VI_high", "P", "Phat"],
        "default": ["Phat", "P", "default_region", "survival_prob", "bar_z"],
        "investment": [
            "bar_i_cond", "bar_i_cond_low", "bar_i_cond_mid", "bar_i_cond_high",
            "bar_i_eff", "bar_i_eff_low", "bar_i_eff_mid", "bar_i_eff_high",
        ],
        "q": ["Q", "q_unit"],
        "bp": [
            "bp0_raw", "bp0_survival",
            "bpI_raw", "bpI_survival",
            "bpI_low_raw", "bpI_low_survival",
            "bpI_mid_raw", "bpI_mid_survival",
            "bpI_high_raw", "bpI_high_survival",
            "bp_cond_raw", "bp_cond_survival",
            "bp_raw", "bp_survival",
        ],
    }
    for group, names in groups.items():
        directory = output_root / group
        directory.mkdir(parents=True, exist_ok=True)
        selected = {name: surfaces[name] for name in names}
        save_surface_csvs(directory, selected, b_values, z_values)
        for name, values in selected.items():
            plot_heatmap(
                values,
                b_values,
                z_values,
                directory / f"{name}.png",
                title=name,
                colorbar_label=name,
                cmap="gray_r" if name == "default_region" else "viridis",
            )


def _bp_boundary_summary(surfaces: Dict[str, np.ndarray]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for name in ("bp0", "bpI_low", "bpI_mid", "bpI_high", "bp"):
        raw = surfaces[f"{name}_raw"]
        survival = surfaces[f"{name}_survival"]
        valid = np.isfinite(survival)
        summary[f"{name}_raw_share_lt_0p05"] = float((raw < 0.05).mean())
        summary[f"{name}_raw_share_gt_0p95"] = float((raw > 0.95).mean())
        summary[f"{name}_share_lt_0p05"] = (
            float((survival[valid] < 0.05).mean()) if valid.any() else float("nan")
        )
        summary[f"{name}_share_gt_0p95"] = (
            float((survival[valid] > 0.95).mean()) if valid.any() else float("nan")
        )
    return summary


def evaluate(args: argparse.Namespace) -> tuple[pd.DataFrame, Dict[str, object]]:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_only = bool(getattr(args, "summary_only", False))
    loaded = load_analysis_checkpoint(
        args.checkpoint or args.pv_ckpt,
        sdf_checkpoint=args.sdf_ckpt,
        hyperparams_json=args.hyperparams_json,
        config_json=args.config_json,
        model_spec_json=args.model_spec_json,
        device=device,
        allow_default_hyperparams=bool(args.allow_default_hyperparams),
        allow_current_config=bool(args.allow_current_config),
        m_source="sdf_fc1",
    )
    model = loaded.models["policy_value"]
    sdf_fc1_model = loaded.models["sdf_fc1"]
    model.eval()
    sdf_fc1_model.eval()
    before_hashes = {
        "policy_value": _state_hash(model),
        "sdf_fc1": _state_hash(sdf_fc1_model),
    }

    _, reference = load_reference_state(args.firm_data, macro_path=args.macro_data)
    reference = dataclasses.replace(reference, eta=float(args.eta))
    b_min = float(Config.SIM_B_INIT_MIN if args.b_min is None else args.b_min)
    b_max = float(Config.SIM_B_INIT_MAX if args.b_max is None else args.b_max)
    grid = build_frozen_grid(
        reference,
        b_min=b_min,
        b_max=b_max,
        b_points=args.b_points,
        z_min=args.z_min,
        z_max=args.z_max,
        z_points=args.z_points,
        device=device,
    )
    surfaces = evaluate_firm_surfaces(
        model,
        grid,
        reference,
        chunk_size=args.forward_chunk_size,
    )
    if not summary_only:
        _write_selected_surfaces(
            surfaces,
            b_values=grid.b_values,
            z_values=grid.z_values,
            output_root=output,
        )

    boundary, boundary_summary = extract_phat_default_boundary(
        grid.b_values,
        grid.z_values,
        surfaces["Phat"],
    )
    default_comparison, default_comparison_summary = compare_hard_soft_default_boundaries(
        grid.b_values, grid.z_values, surfaces["Phat"], surfaces["bar_z"]
    )
    if not summary_only:
        (output / "default").mkdir(parents=True, exist_ok=True)
        boundary.to_csv(output / "default" / "default_boundary.csv", index=False)
        plot_default_boundary(boundary, output / "default" / "default_boundary.png")
        default_comparison.to_csv(output / "default" / "boundary_comparison.csv", index=False)
        plot_default_boundary_comparison(
            default_comparison, output / "default" / "boundary_comparison.png"
        )

    investment = evaluate_investment_cutoff(
        model,
        grid,
        reference,
        i_points=args.i_points,
        i_min=0.0,
        i_max=float(loaded.economic_config.I_THRESHOLD),
        chunk_size=args.forward_chunk_size,
        survival_mask=surfaces["survival_mask"],
    )
    investment_surfaces = {
        "i_star": investment["i_star"],
        "investment_region_mid": investment["investment_region_mid"],
        "investment_crossing_count": investment["crossing_count"].astype(np.float64),
    }
    margin_surfaces, investment_margin_summary = investment_margin_diagnostics(
        surfaces, investment
    )
    investment_surfaces.update(margin_surfaces)
    investment_boundary = pd.DataFrame(
        {
            "b": grid.mesh_b.reshape(-1),
            "z": grid.mesh_z.reshape(-1),
            "i_star": investment["i_star"].reshape(-1),
            "investment_status": investment["investment_status"].reshape(-1),
            "crossing_count": investment["crossing_count"].reshape(-1),
            "survival_region": investment["survival_mask"].reshape(-1),
        }
    )
    if not summary_only:
        save_surface_csvs(
            output / "investment", investment_surfaces, grid.b_values, grid.z_values,
        )
        investment_boundary.to_csv(output / "investment" / "investment_boundary.csv", index=False)
        for name, values in investment_surfaces.items():
            plot_heatmap(
                values, grid.b_values, grid.z_values,
                output / "investment" / f"{name}.png",
                title=name, colorbar_label=name,
                cmap="gray_r" if "investment_region" in name else "viridis",
            )
        plot_i_star_slices(
            investment["i_star"], grid.b_values, grid.z_values,
            output / "investment" / "i_star_z_slices.png",
        )

    if not summary_only:
        plot_b_slices(
            surfaces["Q"], grid.b_values, grid.z_values,
            output / "q" / "Q_b_slices.png", title="Q(b) at fixed z", ylabel="Q",
        )
        plot_b_slices(
            surfaces["q_unit"], grid.b_values, grid.z_values,
            output / "q" / "q_unit_b_slices.png", title="Unit bond price at fixed z", ylabel="q_unit",
        )
    peak_idx = np.nanargmax(surfaces["Q"], axis=0)
    q_peak = pd.DataFrame(
        {
            "z": grid.z_values,
            "b_peak": grid.b_values[peak_idx],
            "q_peak": surfaces["Q"][peak_idx, np.arange(len(grid.z_values))],
        }
    )
    if not summary_only:
        q_peak.to_csv(output / "q" / "Q_peak_by_z.csv", index=False)
        plot_q_peak(q_peak, output / "q" / "Q_peak_by_z.png")

    transition_data = build_frozen_transition_data(
        sdf_fc1_model, grid.base_states, reference, loaded.hyperparams,
        loaded.economic_config, n_child_shocks=args.n_child_shocks,
        shock_seed=args.shock_seed,
        shock_bank_max_child_shocks=args.shock_bank_max_child_shocks,
    )
    bellman_surfaces, bellman_summary = evaluate_bellman_residuals(
        model, grid, transition_data, loaded.economic_config,
        chunk_size=args.forward_chunk_size,
    )
    if not summary_only:
        save_surface_csvs(output / "bellman", bellman_surfaces, grid.b_values, grid.z_values)
        for name, values in bellman_surfaces.items():
            plot_heatmap(
                values, grid.b_values, grid.z_values, output / "bellman" / f"{name}.png",
                title=name, colorbar_label=name,
            )

    bp_surfaces, bp_summary, transition_meta = evaluate_bp_consistency(
        model,
        sdf_fc1_model,
        grid,
        reference,
        loaded.hyperparams,
        loaded.economic_config,
        output_dir=output / "objective_slices",
        n_child_shocks=args.n_child_shocks,
        shock_seed=args.shock_seed,
        teacher_margin_tol=args.bp_teacher_margin_tol,
        transition_data=transition_data,
        write_objective_slices=not summary_only,
    )
    if not summary_only:
        save_surface_csvs(output / "bp", bp_surfaces, grid.b_values, grid.z_values)
        for name, values in bp_surfaces.items():
            plot_heatmap(
                values, grid.b_values, grid.z_values, output / "bp" / f"{name}.png",
                title=name, colorbar_label=name,
            )
        audit = build_child_continuation_audit(
            model, grid, transition_data, loaded.economic_config,
            chunk_size=args.forward_chunk_size,
        )
        (output / "audits").mkdir(parents=True, exist_ok=True)
        audit.to_csv(output / "audits" / "child_continuation_audit.csv", index=False)
    else:
        audit = None

    summary_values: Dict[str, float] = {}
    summary_values.update(finite_difference_summary(surfaces))
    summary_values.update(boundary_summary)
    summary_values.update(default_comparison_summary)
    summary_values.update(_bp_boundary_summary(surfaces))
    summary_values.update(bp_summary)
    summary_values.update(bellman_summary)
    summary_values.update(investment_margin_summary)
    summary_values.update(q_unit_summary(surfaces["q_unit"], surfaces["survival_mask"]))
    summary_values.update(
        {
            "investment_i_star_observed_share": float(np.isfinite(investment["i_star"]).mean()),
            "investment_single_crossing_share": float(
                (investment["investment_status"] == "single_crossing").mean()
            ),
            "investment_multiple_crossing_share": float(
                (investment["investment_status"] == "multiple_crossings").mean()
            ),
            "investment_all_invest_share": float(
                (investment["investment_status"] == "all_invest").mean()
            ),
            "investment_all_no_invest_share": float(
                (investment["investment_status"] == "all_no_invest").mean()
            ),
            "investment_nonfinite_share": float(
                (investment["investment_status"] == "nonfinite").mean()
            ),
            "eta_parent": float(args.eta),
            "eta_next_active_share": float(transition_meta["eta_next_active_share"]),
            "n_child_shocks": int(args.n_child_shocks),
            "child_transition_max_error": (
                float(audit["child_b_identity_error"].max()) if audit is not None else float("nan")
            ),
        }
    )
    def _mean_summary(names: list[str]) -> float:
        values = np.asarray([summary_values.get(name, np.nan) for name in names], dtype=np.float64)
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else float("nan")

    def _max_summary(names: list[str]) -> float:
        values = np.asarray([summary_values.get(name, np.nan) for name in names], dtype=np.float64)
        finite = values[np.isfinite(values)]
        return float(finite.max()) if finite.size else float("nan")

    summary_values.update({
        "bp_mae_raw": _mean_summary(["p0_raw_mae", "pi_mid_raw_mae"]),
        "bp_mae_survival": _mean_summary(["p0_survival_mae", "pi_mid_survival_mae"]),
        "bp_mae_survival_identified": _mean_summary(["p0_mae", "pi_mid_mae"]),
        "bp_grid_star_mean": _mean_summary([
            "p0_bp_grid_star_mean", "pi_mid_bp_grid_star_mean"
        ]),
        "bp_continuation_at_coarse_star_mean": _mean_summary([
            "p0_coarse_continuation_at_star_mean",
            "pi_mid_coarse_continuation_at_star_mean",
        ]),
        "bp_regret_mean": _mean_summary(["p0_regret_mean", "pi_mid_regret_mean"]),
        "bp_regret_median": _mean_summary(["p0_regret_median", "pi_mid_regret_median"]),
        "bp_regret_p90": _mean_summary(["p0_regret_p90", "pi_mid_regret_p90"]),
        "bp_regret_p99": _mean_summary(["p0_regret_p99", "pi_mid_regret_p99"]),
        "bp_regret_max": _max_summary(["p0_regret_max", "pi_mid_regret_max"]),
        "teacher_identified_share": _mean_summary([
            "p0_teacher_identified_share", "pi_mid_teacher_identified_share"
        ]),
        "teacher_identified_share_survival": _mean_summary([
            "p0_teacher_identified_share_survival",
            "pi_mid_teacher_identified_share_survival",
        ]),
    })
    summary = pd.DataFrame([summary_values])
    summary.to_csv(output / "summary.csv", index=False)

    after_hashes = {
        "policy_value": _state_hash(model),
        "sdf_fc1": _state_hash(sdf_fc1_model),
    }
    if before_hashes != after_hashes:
        raise RuntimeError("Checkpoint model state changed during read-only evaluation")
    checkpoint_path = Path(args.checkpoint or args.pv_ckpt).expanduser().resolve()
    metadata: Dict[str, object] = {
        "evaluator": "firm_side_checkpoint_evaluator_v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "checkpoint": loaded.metadata,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_filename": checkpoint_path.name,
        "device": str(device),
        "firm_data": str(args.firm_data.resolve()),
        "macro_data": reference.macro_source,
        "bp_teacher_model": "policy_value",
        "parent_eta": float(args.eta),
        "n_child_shocks": int(args.n_child_shocks),
        "shock_seed": int(args.shock_seed),
        "common_random_numbers": True,
        "common_random_numbers_scope": (
            "within_eta_grid_and_prefix_of_Jmax"
            if transition_meta.get("nested_prefix_from_max_J")
            else "within_eta_J_frozen_grid"
        ),
        "shock_bank_max_child_shocks": int(
            transition_meta.get("shock_bank_max_child_shocks", args.n_child_shocks)
        ),
        "bp_margin_identification_threshold": float(args.bp_teacher_margin_tol),
        "reference_state": reference.to_dict(),
        "grid": {
            "b_min": b_min,
            "b_max": b_max,
            "b_points": int(args.b_points),
            "z_min": float(args.z_min),
            "z_max": float(args.z_max),
            "z_points": int(args.z_points),
            "eta": float(args.eta),
            "i_points": int(args.i_points),
            "i_min": 0.0,
            "i_max": float(loaded.economic_config.I_THRESHOLD),
        },
        "reference_transition_bank": transition_meta,
        "m_mode": transition_meta.get("m_mode"),
        "m_clamp_bounds": (
            {
                "min": float(getattr(loaded.hyperparams, "pv_m_clamp_min", 0.7)),
                "max": float(getattr(loaded.hyperparams, "pv_m_clamp_max", 1.3)),
            }
            if bool(getattr(loaded.hyperparams, "pv_use_clipped_m", True)) else None
        ),
        "model_state_hash_before": before_hashes,
        "model_state_hash_after": after_hashes,
        "model_state_unchanged": True,
        "semantics": {
            "default_boundary": (
                "linearly interpolated Phat=0 when exactly one crossing exists; "
                "NaN for zero or multiple crossings"
            ),
            "bar_i_cond": "conditional investment probability",
            "bar_i_eff": "survival-adjusted executed investment probability",
            "i_star": (
                "VI(i)-V0(i)=0 when exactly one crossing exists; NaN for zero/multiple "
                "crossings, nonfinite scans, or states outside Phat>0"
            ),
            "Q": "total debt value",
            "q_unit": "Q/b for b>1e-12; NaN at b=0",
            "bp_consistency_masks": (
                "Raw statistics use the full finite grid; survival statistics require finite Phat(i)>0; "
                "primary statistics additionally require BPGridTeacher top2_margin above the configured tolerance"
            ),
            "bp_consistency": (
                "PolicyValue output versus BPGridTeacher.compute using checkpoint SDF/FC1, "
                "ConvergenceShockBank, and build_child_exogenous_bundle"
            ),
            "bp_teacher_model": (
                "policy_value; this measures current policy consistency with the online value surface, "
                "not the historical firm_target training teacher"
            ),
            "child_leverage_timing": (
                "b_next = eta_next * bp_current + (1-eta_next) * b_current"
            ),
            "current_financing_eta": "eta_current",
            "bellman_residual": (
                "physical conditional mean: P0-CF0-E[M_used*P_child] and "
                "PI-CFI-G*E[M_used*P_child]"
            ),
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return summary, metadata


def evaluate_matrix(args: argparse.Namespace) -> tuple[pd.DataFrame, Dict[str, object]]:
    """Run eta/J cases with full plots for the primary J and compact robustness rows otherwise."""
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    eta_values = list(args.eta_values or [args.eta])
    j_values = list(args.robustness_child_shocks or [args.n_child_shocks])
    if int(args.n_child_shocks) not in j_values:
        j_values.insert(0, int(args.n_child_shocks))
    eta_values = list(dict.fromkeys(float(value) for value in eta_values))
    j_values = list(dict.fromkeys(int(value) for value in j_values))
    if any(value < 2 for value in j_values):
        raise ValueError("All robustness child-shock counts must be at least 2")

    rows = []
    case_metadata = []
    first_case_metadata: Dict[str, object] | None = None
    for eta in eta_values:
        eta_label = f"eta{eta:g}".replace("-", "m").replace(".", "p")
        for child_count in j_values:
            primary = child_count == int(args.n_child_shocks)
            case_output = (
                root / eta_label
                if primary
                else root / "robustness" / f"{eta_label}_J{child_count}"
            )
            case_args = argparse.Namespace(**vars(args))
            case_args.eta = eta
            case_args.n_child_shocks = child_count
            case_args.output_dir = case_output
            case_args.summary_only = not primary
            case_args.eta_values = None
            case_args.robustness_child_shocks = None
            case_args.shock_bank_max_child_shocks = max(j_values)
            summary, metadata = evaluate(case_args)
            if first_case_metadata is None:
                first_case_metadata = metadata
            row = summary.iloc[0].to_dict()
            row.update({
                "eta_parent": eta,
                "n_child_shocks": child_count,
                "full_output": primary,
                "case_output": str(case_output),
            })
            rows.append(row)
            case_metadata.append({
                "eta_parent": eta,
                "n_child_shocks": child_count,
                "full_output": primary,
                "output": str(case_output),
                "metadata": str(case_output / "metadata.json"),
                "model_state_unchanged": bool(metadata.get("model_state_unchanged")),
            })

    combined = pd.DataFrame(rows)
    combined.to_csv(root / "summary.csv", index=False)
    robustness = root / "robustness"
    robustness.mkdir(parents=True, exist_ok=True)
    combined.to_csv(robustness / "j_comparison.csv", index=False)
    first_case_metadata = first_case_metadata or {}
    metadata = {
        "evaluator": "firm_side_checkpoint_evaluator_v2_matrix",
        "git_commit_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "checkpoint_path": str(Path(args.checkpoint or args.pv_ckpt).expanduser().resolve()),
        "checkpoint_filename": Path(args.checkpoint or args.pv_ckpt).name,
        "eta_values": eta_values,
        "primary_n_child_shocks": int(args.n_child_shocks),
        "robustness_n_child_shocks": j_values,
        "shock_seed": int(args.shock_seed),
        "common_random_numbers": True,
        "common_random_numbers_scope": "within_each_eta_grid_and_nested_prefix_across_J",
        "shock_bank_max_child_shocks": max(j_values),
        "nested_shock_prefix_across_J": True,
        "bp_margin_identification_threshold": float(args.bp_teacher_margin_tol),
        "grid": first_case_metadata.get("grid"),
        "reference_state": first_case_metadata.get("reference_state"),
        "m_mode": first_case_metadata.get("m_mode"),
        "m_clamp_bounds": first_case_metadata.get("m_clamp_bounds"),
        "model_state_unchanged": all(
            bool(item.get("model_state_unchanged")) for item in case_metadata
        ),
        "cases": case_metadata,
        "semantics": {
            "child_leverage_timing": "b_next = eta_next * bp_current + (1-eta_next) * b_current",
            "current_financing_eta": "eta_current",
        },
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return combined, metadata


def main() -> None:
    args = parse_args()
    matrix = args.eta_values is not None or args.robustness_child_shocks is not None
    summary, metadata = evaluate_matrix(args) if matrix else evaluate(args)
    print(f"Saved firm-side checkpoint evaluation to: {args.output_dir.resolve()}")
    if "model_state_unchanged" in metadata:
        print(f"Model state unchanged: {metadata['model_state_unchanged']}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
