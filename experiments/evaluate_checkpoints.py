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
from evaluation.boundaries import extract_phat_default_boundary  # noqa: E402
from evaluation.bp_diagnostics import evaluate_bp_consistency  # noqa: E402
from evaluation.firm_surfaces import (  # noqa: E402
    evaluate_firm_surfaces,
    evaluate_investment_cutoff,
    finite_difference_summary,
)
from evaluation.grids import build_frozen_grid, load_reference_state  # noqa: E402
from evaluation.plotting import (  # noqa: E402
    plot_b_slices,
    plot_default_boundary,
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
    boundary.to_csv(output / "default" / "default_boundary.csv", index=False)
    plot_default_boundary(boundary, output / "default" / "default_boundary.png")

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
    save_surface_csvs(
        output / "investment",
        investment_surfaces,
        grid.b_values,
        grid.z_values,
    )
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
    investment_boundary.to_csv(output / "investment" / "investment_boundary.csv", index=False)
    for name, values in investment_surfaces.items():
        plot_heatmap(
            values,
            grid.b_values,
            grid.z_values,
            output / "investment" / f"{name}.png",
            title=name,
            colorbar_label=name,
            cmap="gray_r" if name == "investment_region_mid" else "viridis",
        )
    plot_i_star_slices(
        investment["i_star"],
        grid.b_values,
        grid.z_values,
        output / "investment" / "i_star_z_slices.png",
    )

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
    q_peak.to_csv(output / "q" / "Q_peak_by_z.csv", index=False)
    plot_q_peak(q_peak, output / "q" / "Q_peak_by_z.png")

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
    )
    save_surface_csvs(output / "bp", bp_surfaces, grid.b_values, grid.z_values)
    for name, values in bp_surfaces.items():
        plot_heatmap(
            values,
            grid.b_values,
            grid.z_values,
            output / "bp" / f"{name}.png",
            title=name,
            colorbar_label=name,
        )

    summary_values: Dict[str, float] = {}
    summary_values.update(finite_difference_summary(surfaces))
    summary_values.update(boundary_summary)
    summary_values.update(_bp_boundary_summary(surfaces))
    summary_values.update(bp_summary)
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
        }
    )
    summary = pd.DataFrame([summary_values])
    summary.to_csv(output / "summary.csv", index=False)

    after_hashes = {
        "policy_value": _state_hash(model),
        "sdf_fc1": _state_hash(sdf_fc1_model),
    }
    if before_hashes != after_hashes:
        raise RuntimeError("Checkpoint model state changed during read-only evaluation")
    metadata: Dict[str, object] = {
        "evaluator": "firm_side_checkpoint_evaluator_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": _git_sha(),
        "checkpoint": loaded.metadata,
        "device": str(device),
        "firm_data": str(args.firm_data.resolve()),
        "macro_data": reference.macro_source,
        "bp_teacher_model": "policy_value",
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
            "bp_survival": (
                "BP surfaces and primary statistics restricted to finite Phat(i)>0 states "
                "for each corresponding investment slice"
            ),
            "bp_consistency": (
                "PolicyValue output versus BPGridTeacher.compute using checkpoint SDF/FC1, "
                "ConvergenceShockBank, and build_child_exogenous_bundle"
            ),
            "bp_teacher_model": (
                "policy_value; this measures current policy consistency with the online value surface, "
                "not the historical firm_target training teacher"
            ),
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return summary, metadata


def main() -> None:
    args = parse_args()
    summary, metadata = evaluate(args)
    print(f"Saved firm-side checkpoint evaluation to: {args.output_dir.resolve()}")
    print(f"Model state unchanged: {metadata['model_state_unchanged']}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
