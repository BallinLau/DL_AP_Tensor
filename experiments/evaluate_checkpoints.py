from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
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
from losses.q_loss import resolve_q_regime_settings  # noqa: E402
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
    evaluate_bp_consistency_multi_j,
    resolve_bp_eval_max_expanded_states,
    slice_frozen_transition_data,
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


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    return int(raw)

_BP_FORWARD_STAT_COUNTERS = (
    "bp_model_forward_calls",
    "bp_child_equity_forward_calls",
    "bp_q_forward_calls",
    "bp_parent_chunks",
    "bp_candidate_chunks",
)


def _zero_bp_forward_stats(budget: int) -> Dict[str, object]:
    stats: Dict[str, object] = {name: 0 for name in _BP_FORWARD_STAT_COUNTERS}
    stats.update(
        {
            "bp_max_expanded_states": int(budget),
            "bp_max_actual_expanded_states": 0,
            "bp_multi_j_reuse_enabled": False,
            "bp_branch_reuse_enabled": False,
            "bp_grid_chunk_plans": [],
        }
    )
    return stats


def sum_bp_forward_stats(items) -> Dict[str, object]:
    """Aggregate per-case BP hot-path counters into one matrix-level summary."""
    totals: Dict[str, object] = {name: 0 for name in _BP_FORWARD_STAT_COUNTERS}
    max_expanded = 0
    max_actual = 0
    multi_j_reuse = False
    branch_reuse = False
    plans: list = []
    for stats in items:
        if not stats:
            continue
        for name in _BP_FORWARD_STAT_COUNTERS:
            totals[name] = int(totals[name]) + int(stats.get(name, 0))
        max_expanded = max(max_expanded, int(stats.get("bp_max_expanded_states", 0)))
        max_actual = max(max_actual, int(stats.get("bp_max_actual_expanded_states", 0)))
        multi_j_reuse = multi_j_reuse or bool(stats.get("bp_multi_j_reuse_enabled", False))
        branch_reuse = branch_reuse or bool(stats.get("bp_branch_reuse_enabled", False))
        plans.extend(stats.get("bp_grid_chunk_plans") or [])
    totals.update(
        {
            "bp_max_expanded_states": max_expanded,
            "bp_max_actual_expanded_states": max_actual,
            "bp_multi_j_reuse_enabled": multi_j_reuse,
            "bp_branch_reuse_enabled": branch_reuse,
            "bp_grid_chunk_plans": plans,
        }
    )
    return totals

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
    parser.add_argument(
        "--bp-eval-max-expanded-states",
        type=int,
        default=_env_int("BP_EVAL_MAX_EXPANDED_STATES"),
        help=(
            "Evaluator-only BP chunk budget (one of 65536/131072/262144/524288). "
            "Overrides the checkpoint's bp_grid_max_expanded_states for this read-only "
            "evaluation only; it never changes training semantics or model state. "
            "Defaults to a conservative GPU-memory tier on CUDA and 65536 otherwise. "
            "Falls back to the BP_EVAL_MAX_EXPANDED_STATES environment variable."
        ),
    )
    parser.add_argument(
        "--recovery-normalization-mode",
        choices=["asset_only", "legacy_b_times_unit"],
        default=None,
        help=(
            "Q 违约回收归一化口径。默认沿用 checkpoint hyperparams，再回落到 "
            "Config.RECOVERY_NORMALIZATION_MODE（asset_only，与 main_4.tex 一致）。"
        ),
    )
    parser.add_argument(
        "--q-parent-default-regime-mode",
        choices=["legacy_soft_penalty", "hard", "transition_band"],
        default=None,
        help=(
            "Q parent default regime gating 口径。默认沿用 checkpoint hyperparams，"
            "再回落到 Config.Q_PARENT_DEFAULT_REGIME_MODE（legacy_soft_penalty）。"
        ),
    )
    parser.add_argument("--q-parent-default-eps", type=float, default=None)
    parser.add_argument("--q-parent-default-tau", type=float, default=None)
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
    write_plots: bool = True,
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
        if not write_plots:
            continue
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


def _q_peak_diagnostics_frame(
    surfaces: Dict[str, np.ndarray],
    bellman_surfaces: Dict[str, np.ndarray],
    grid,
    peak_idx: np.ndarray,
) -> pd.DataFrame:
    """每个 z 的 Q peak 点上的 Q / recovery / regime 分解（自包含，免手工 join）。"""
    z_values = np.asarray(grid.z_values)
    columns = np.arange(len(z_values))

    def pick(name: str) -> np.ndarray:
        values = bellman_surfaces.get(name)
        if values is None:
            return np.full(len(z_values), np.nan, dtype=np.float64)
        return values[peak_idx, columns]

    return pd.DataFrame({
        "z": z_values,
        "b_peak": grid.b_values[peak_idx],
        "Q": surfaces["Q"][peak_idx, columns],
        "Phat": surfaces["Phat"][peak_idx, columns],
        "bar_z": surfaces["bar_z"][peak_idx, columns],
        "parent_hard_default": pick("parent_hard_default"),
        "parent_survival_weight": pick("parent_survival_weight"),
        "parent_default_weight": pick("parent_default_weight"),
        "bellman_weight": pick("parent_survival_weight"),
        "recovery_weight": pick("parent_default_weight"),
        "recovery_current": pick("recovery_current"),
        "Q_target": pick("Q_target"),
        "Q_target_survival": pick("Q_target_survival"),
        "Q_target_recovery": pick("Q_target_recovery"),
        # 显式 Bellman 口径分解。
        "Q_target_bellman": pick("Q_target_bellman"),
        "Q_target_survival_bellman": pick("Q_target_survival_bellman"),
        "Q_target_recovery_bellman": pick("Q_target_recovery_bellman"),
        # transition_band 下 AiO 非线性，w*Q_B + (1-w)*R 不是严格 optimizer target，
        # 只能作为 blended reference；Q_target_training 保留为兼容 alias。
        "Q_target_blended_reference": pick("Q_target_training"),
        "Q_target_training": pick("Q_target_training"),
        "Q_minus_recovery": pick("Q_minus_recovery"),
        "Q_minus_Q_target": pick("Q_minus_Q_target"),
        "Q_target_minus_recovery": pick("Q_target_minus_recovery"),
        "RQ_bellman_signed": pick("RQ_bellman_signed"),
        "RQ_blended_reference_signed": pick("RQ_training_signed"),
    })


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
    _sync_cuda(device)
    total_started = time.perf_counter()
    phase_timing: Dict[str, float] = {}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_only = bool(getattr(args, "summary_only", False))
    detailed_output = bool(getattr(args, "detailed_output", not summary_only))
    shared_cache = getattr(args, "_evaluation_cache", None)
    loaded = getattr(args, "loaded_checkpoint", None)
    if loaded is None and shared_cache is not None:
        loaded = shared_cache.get("loaded")
    if loaded is None:
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
        if shared_cache is not None:
            shared_cache["loaded"] = loaded
    model = loaded.models["policy_value"]
    sdf_fc1_model = loaded.models["sdf_fc1"]
    model.eval()
    sdf_fc1_model.eval()

    # 统一解析 Q regime / recovery 口径（CLI > checkpoint > legacy fallback > Config）。
    # 旧 checkpoint 的 hyperparams payload 不含 q_recovery_normalization_mode，
    # 必须靠 loader 记录的“真实字段集合”判定，否则会被静默解释为 Config.asset_only。
    recorded_fields = None
    if isinstance(getattr(loaded, "metadata", None), dict):
        recorded_fields = loaded.metadata.get("q_semantics_recorded_fields", None)
    q_settings = resolve_q_regime_settings(
        cli_recovery_normalization_mode=getattr(args, "recovery_normalization_mode", None),
        cli_parent_default_regime_mode=getattr(args, "q_parent_default_regime_mode", None),
        cli_parent_default_eps=getattr(args, "q_parent_default_eps", None),
        cli_parent_default_tau=getattr(args, "q_parent_default_tau", None),
        checkpoint_recovery_normalization_mode=getattr(
            loaded.hyperparams, "q_recovery_normalization_mode", None
        ),
        checkpoint_parent_default_regime_mode=getattr(
            loaded.hyperparams, "q_parent_default_regime_mode", None
        ),
        checkpoint_parent_default_eps=getattr(
            loaded.hyperparams, "q_parent_default_eps", None
        ),
        checkpoint_parent_default_tau=getattr(
            loaded.hyperparams, "q_parent_default_tau", None
        ),
        checkpoint_recorded_fields=recorded_fields,
    )
    recovery_normalization_mode = q_settings["recovery_normalization_mode"]
    q_parent_default_regime_mode = q_settings["q_parent_default_regime_mode"]
    q_parent_default_eps = q_settings["q_parent_default_eps"]
    q_parent_default_tau = q_settings["q_parent_default_tau"]
    hash_locally = not bool(getattr(args, "defer_model_state_hash", False))
    before_hashes = (
        {
            "policy_value": _state_hash(model),
            "sdf_fc1": _state_hash(sdf_fc1_model),
        }
        if hash_locally else None
    )

    static_cache = (
        shared_cache.setdefault("static_by_eta", {})
        if shared_cache is not None else {}
    )
    static_key = float(args.eta)
    cached_static = static_cache.get(static_key)
    if cached_static is None:
        _sync_cuda(device)
        static_started = time.perf_counter()
        _, reference = load_reference_state(args.firm_data, macro_path=args.macro_data)
        reference = dataclasses.replace(reference, eta=static_key)
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
        boundary, boundary_summary = extract_phat_default_boundary(
            grid.b_values,
            grid.z_values,
            surfaces["Phat"],
        )
        default_comparison, default_comparison_summary = compare_hard_soft_default_boundaries(
            grid.b_values, grid.z_values, surfaces["Phat"], surfaces["bar_z"]
        )
        _sync_cuda(device)
        phase_timing["firm_static_seconds"] = time.perf_counter() - static_started

        investment_started = time.perf_counter()
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
        _sync_cuda(device)
        phase_timing["investment_seconds"] = time.perf_counter() - investment_started
        cached_static = {
            "reference": reference, "b_min": b_min, "b_max": b_max,
            "grid": grid, "surfaces": surfaces,
            "boundary": boundary, "boundary_summary": boundary_summary,
            "default_comparison": default_comparison,
            "default_comparison_summary": default_comparison_summary,
            "investment": investment,
            "investment_surfaces": investment_surfaces,
            "investment_boundary": investment_boundary,
            "investment_margin_summary": investment_margin_summary,
        }
        if shared_cache is not None:
            static_cache[static_key] = cached_static
    else:
        phase_timing["firm_static_seconds"] = 0.0
        phase_timing["investment_seconds"] = 0.0
    reference = cached_static["reference"]
    b_min = cached_static["b_min"]
    b_max = cached_static["b_max"]
    grid = cached_static["grid"]
    surfaces = cached_static["surfaces"]
    boundary = cached_static["boundary"]
    boundary_summary = cached_static["boundary_summary"]
    default_comparison = cached_static["default_comparison"]
    default_comparison_summary = cached_static["default_comparison_summary"]
    investment = cached_static["investment"]
    investment_surfaces = cached_static["investment_surfaces"]
    investment_boundary = cached_static["investment_boundary"]
    investment_margin_summary = cached_static["investment_margin_summary"]
    if not summary_only:
        _write_selected_surfaces(
            surfaces,
            b_values=grid.b_values,
            z_values=grid.z_values,
            output_root=output,
            write_plots=detailed_output,
        )

    if not summary_only:
        (output / "default").mkdir(parents=True, exist_ok=True)
        boundary.to_csv(output / "default" / "default_boundary.csv", index=False)
        default_comparison.to_csv(output / "default" / "boundary_comparison.csv", index=False)
        if detailed_output:
            plot_default_boundary(boundary, output / "default" / "default_boundary.png")
            plot_default_boundary_comparison(
                default_comparison, output / "default" / "boundary_comparison.png"
            )

    if not summary_only:
        save_surface_csvs(
            output / "investment", investment_surfaces, grid.b_values, grid.z_values,
        )
        investment_boundary.to_csv(output / "investment" / "investment_boundary.csv", index=False)
        if detailed_output:
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

    if detailed_output:
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
        if detailed_output:
            plot_q_peak(q_peak, output / "q" / "Q_peak_by_z.png")

    transition_cache = (
        shared_cache.setdefault("transition_by_eta", {})
        if shared_cache is not None else {}
    )
    transition_max = transition_cache.get(static_key)
    # ``max_evaluated_child_shocks`` sizes the transition that is actually built;
    # ``canonical_shock_bank_max_child_shocks`` only sizes the deterministic bank
    # the prefix is drawn from. Keeping them separate means an episode that only
    # evaluates J=64 builds a J=64 transition (128 expanded children) instead of a
    # J=128 transition (256 expanded children) whose upper half is then discarded,
    # while still drawing from the same canonical bank for cross-episode CRN.
    max_evaluated_child_shocks = int(
        (
            shared_cache.get("max_evaluated_child_shocks")
            if shared_cache is not None else None
        )
        or args.n_child_shocks
    )
    canonical_shock_bank_max_child_shocks = int(
        (
            shared_cache.get("canonical_shock_bank_max_child_shocks")
            if shared_cache is not None else None
        )
        or getattr(args, "shock_bank_max_child_shocks", None)
        or max_evaluated_child_shocks
    )
    if canonical_shock_bank_max_child_shocks < max_evaluated_child_shocks:
        raise ValueError(
            "shock_bank_max_child_shocks "
            f"({canonical_shock_bank_max_child_shocks}) must cover the largest "
            f"evaluated J ({max_evaluated_child_shocks})"
        )
    if transition_max is None:
        transition_max = build_frozen_transition_data(
            sdf_fc1_model, grid.base_states, reference, loaded.hyperparams,
            loaded.economic_config, n_child_shocks=max_evaluated_child_shocks,
            shock_seed=args.shock_seed,
            shock_bank_max_child_shocks=canonical_shock_bank_max_child_shocks,
        )
        if shared_cache is not None:
            transition_cache[static_key] = transition_max
    transition_data = slice_frozen_transition_data(
        transition_max, n_continuous_children=int(args.n_child_shocks)
    )
    _sync_cuda(device)
    bellman_started = time.perf_counter()
    bellman_surfaces, bellman_summary = evaluate_bellman_residuals(
        model, grid, transition_data, loaded.economic_config,
        chunk_size=args.forward_chunk_size,
        recovery_normalization_mode=recovery_normalization_mode,
        parent_default_regime_mode=q_parent_default_regime_mode,
        parent_default_eps=q_parent_default_eps,
        parent_default_tau=q_parent_default_tau,
    )
    _sync_cuda(device)
    phase_timing["bellman_seconds"] = time.perf_counter() - bellman_started
    if not summary_only:
        save_surface_csvs(output / "bellman", bellman_surfaces, grid.b_values, grid.z_values)
        _q_peak_diagnostics_frame(
            surfaces, bellman_surfaces, grid, peak_idx
        ).to_csv(output / "q" / "Q_peak_diagnostics.csv", index=False)
        if detailed_output:
            for name, values in bellman_surfaces.items():
                plot_heatmap(
                    values, grid.b_values, grid.z_values, output / "bellman" / f"{name}.png",
                    title=name, colorbar_label=name,
                )

    bp_budget, bp_budget_resolution = resolve_bp_eval_max_expanded_states(
        getattr(args, "bp_eval_max_expanded_states", None), device
    )
    checkpoint_bp_budget = int(
        getattr(loaded.hyperparams, "bp_grid_max_expanded_states", 65536)
    )
    shared_bp_j = shared_cache.get("bp_j_values") if shared_cache is not None else None
    bp_multi_cache = (
        shared_cache.setdefault("bp_multi_j_by_eta", {})
        if shared_cache is not None else {}
    )
    bp_primary_output_dirs = (
        shared_cache.get("bp_primary_output_dirs") if shared_cache is not None else None
    )
    bp_objective_output_dir = (
        Path(bp_primary_output_dirs[static_key]) / "objective_slices"
        if bp_primary_output_dirs and static_key in bp_primary_output_dirs
        else output / "objective_slices"
    )
    if shared_bp_j:
        bp_bundle = bp_multi_cache.get(static_key)
        if bp_bundle is None:
            _sync_cuda(device)
            multi_started = time.perf_counter()
            results_by_j, bp_forward_stats = evaluate_bp_consistency_multi_j(
                model,
                sdf_fc1_model,
                grid,
                reference,
                loaded.hyperparams,
                loaded.economic_config,
                output_dir=bp_objective_output_dir,
                j_values=shared_bp_j,
                primary_j=int(shared_cache["bp_primary_j"]),
                transition_max=transition_max,
                shock_seed=args.shock_seed,
                teacher_margin_tol=args.bp_teacher_margin_tol,
                write_objective_slices=bool(
                    shared_cache.get("bp_write_objective_slices", True)
                ),
                bp_eval_max_expanded_states=bp_budget,
            )
            _sync_cuda(device)
            bp_bundle = {
                "results_by_j": results_by_j,
                "forward_stats": bp_forward_stats,
                "seconds": time.perf_counter() - multi_started,
            }
            bp_multi_cache[static_key] = bp_bundle
        else:
            # Every J of this eta shares one multi-J pass; only the leading case
            # carries its wall time and forward counters.
            bp_forward_stats = _zero_bp_forward_stats(bp_budget)
            bp_bundle = dict(bp_bundle, seconds=0.0, forward_stats=bp_forward_stats)
        entry = bp_bundle["results_by_j"][int(args.n_child_shocks)]
        bp_surfaces = entry["surfaces"]
        bp_summary = entry["summary"]
        transition_meta = entry["metadata"]
        phase_timing["bp_seconds"] = float(bp_bundle["seconds"])
    else:
        _sync_cuda(device)
        bp_started = time.perf_counter()
        results_by_j, bp_forward_stats = evaluate_bp_consistency_multi_j(
            model,
            sdf_fc1_model,
            grid,
            reference,
            loaded.hyperparams,
            loaded.economic_config,
            output_dir=output / "objective_slices",
            j_values=[int(args.n_child_shocks)],
            primary_j=int(args.n_child_shocks),
            transition_max=transition_max,
            shock_seed=args.shock_seed,
            teacher_margin_tol=args.bp_teacher_margin_tol,
            write_objective_slices=detailed_output,
            bp_eval_max_expanded_states=bp_budget,
        )
        _sync_cuda(device)
        phase_timing["bp_seconds"] = time.perf_counter() - bp_started
        entry = results_by_j[int(args.n_child_shocks)]
        bp_surfaces = entry["surfaces"]
        bp_summary = entry["summary"]
        transition_meta = entry["metadata"]
    if not summary_only:
        save_surface_csvs(output / "bp", bp_surfaces, grid.b_values, grid.z_values)
        if detailed_output:
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

    after_hashes = (
        {
            "policy_value": _state_hash(model),
            "sdf_fc1": _state_hash(sdf_fc1_model),
        }
        if hash_locally else None
    )
    if hash_locally and before_hashes != after_hashes:
        raise RuntimeError("Checkpoint model state changed during read-only evaluation")
    _sync_cuda(device)
    phase_timing["total_seconds"] = time.perf_counter() - total_started
    phase_timing["firm_structural_seconds"] = sum(
        float(phase_timing.get(name, 0.0))
        for name in (
            "firm_static_seconds", "investment_seconds", "bellman_seconds", "bp_seconds"
        )
    )
    phase_timing["cuda_peak_memory_mb"] = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
        if device.type == "cuda" and bool(getattr(args, "manage_cuda_peak_stats", True))
        else float("nan")
    )
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
        "q_parameterization": loaded.metadata.get("q_parameterization"),
        "q_parent_default_regime_mode": q_parent_default_regime_mode,
        "eps_default": q_parent_default_eps,
        "tau_parent_default": q_parent_default_tau,
        "recovery_normalization_mode": recovery_normalization_mode,
        # resolved 参数及其来源（cli / checkpoint / legacy_checkpoint_fallback /
        # config_default），避免 metadata 写成 None 而实际运行使用了别的值。
        "recovery_normalization_mode_resolved": q_settings["recovery_normalization_mode"],
        "recovery_normalization_mode_source": q_settings[
            "recovery_normalization_mode_source"
        ],
        "q_parent_default_regime_mode_source": q_settings[
            "q_parent_default_regime_mode_source"
        ],
        "q_parent_default_eps_source": q_settings["q_parent_default_eps_source"],
        "q_parent_default_tau_source": q_settings["q_parent_default_tau_source"],
        "q_boundary_low_margin": float(
            getattr(Config, "Q_BOUNDARY_LOW_MARGIN", 1e-3)
        ),
        "parent_eta": float(args.eta),
        "n_child_shocks": int(args.n_child_shocks),
        "eta_integration_mode": transition_meta.get("eta_integration_mode"),
        "formal_evaluator_eta_integration_mode": "exact",
        "formal_eta_integration_independent_of_training_ablation": True,
        "training_eta_integration_mode": transition_meta.get(
            "training_eta_integration_mode"
        ),
        "eta_probability": transition_meta.get("eta_probability"),
        "continuous_child_count": transition_meta.get("continuous_child_count"),
        "expanded_child_count": transition_meta.get("expanded_child_count"),
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
        "canonical_shock_bank_max_child_shocks": int(
            canonical_shock_bank_max_child_shocks
        ),
        "max_evaluated_child_shocks": int(max_evaluated_child_shocks),
        "evaluated_child_shocks": sorted(
            int(value) for value in (shared_bp_j or [args.n_child_shocks])
        ),
        "transition_continuous_child_count": int(
            transition_max.metadata.get(
                "continuous_child_count", max_evaluated_child_shocks
            )
        ),
        "transition_expanded_child_count": int(
            transition_max.metadata.get(
                "expanded_child_count", 2 * max_evaluated_child_shocks
            )
        ),
        "bp_margin_identification_threshold": float(args.bp_teacher_margin_tol),
        "checkpoint_bp_grid_max_expanded_states": checkpoint_bp_budget,
        "evaluator_bp_max_expanded_states": int(bp_budget),
        "bp_eval_max_expanded_states_resolution": bp_budget_resolution,
        "bp_forward_stats": bp_forward_stats,
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
        "model_state_unchanged": True if hash_locally else None,
        "model_state_hash_scope": "single_case" if hash_locally else "deferred_to_orchestrator",
        "timing": phase_timing,
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
            "q_parameterization": loaded.metadata.get("q_parameterization"),
            "q_unit": "derived reporting ratio Q/b for b>1e-12; NaN at b=0; not a direct-Q head output",
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
                "b_next = eta_current * bp_current + (1-eta_current) * b_current"
            ),
            "current_financing_eta": "eta_current",
            "bellman_residual": (
                "physical conditional mean: P0-CF0-E[M_used*P_child] and "
                "PI-CFI-G*E[M_used*P_child]"
            ),
            "bellman_residual_semantics": {
                "trainM": (
                    "current-policy fixed-point residual using M_used/clipped-M semantics; "
                    "not historical target-network training residual"
                ),
                "rawM": "current-policy fixed-point residual using raw SDF M",
            },
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
    requested_robustness = list(args.robustness_child_shocks or [])
    j_values = list(requested_robustness)
    if int(args.n_child_shocks) not in j_values:
        j_values.insert(0, int(args.n_child_shocks))
    eta_values = list(dict.fromkeys(float(value) for value in eta_values))
    j_values = sorted(set(int(value) for value in j_values))
    if any(value < 2 for value in j_values):
        raise ValueError("All robustness child-shock counts must be at least 2")
    # Two distinct quantities that must never be conflated:
    #   * ``max_evaluated_child_shocks`` is the largest J this episode actually
    #     evaluates; it sizes the transition that is built (and sliced).
    #   * ``canonical_shock_bank_max_child_shocks`` is the single shock bank every
    #     episode draws from, so cross-episode common random numbers line up. It
    #     can exceed the largest evaluated J (canonical 128, evaluated 64).
    max_evaluated_child_shocks = max(j_values)
    canonical_max_child_shocks = (
        int(args.shock_bank_max_child_shocks)
        if getattr(args, "shock_bank_max_child_shocks", None) is not None
        else max_evaluated_child_shocks
    )
    if canonical_max_child_shocks < max_evaluated_child_shocks:
        raise ValueError(
            "shock_bank_max_child_shocks "
            f"({canonical_max_child_shocks}) must cover the largest evaluated J "
            f"({max_evaluated_child_shocks})"
        )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    manage_cuda_peak_stats = bool(getattr(args, "manage_cuda_peak_stats", True))
    if device.type == "cuda" and manage_cuda_peak_stats:
        torch.cuda.reset_peak_memory_stats(device)
    matrix_started = time.perf_counter()
    loaded = getattr(args, "loaded_checkpoint", None)
    checkpoint_load_count_local = 0
    if loaded is None:
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
        checkpoint_load_count_local = 1
    for loaded_model in loaded.models.values():
        loaded_model.eval()
    evaluation_cache: Dict[str, object] = {
        "canonical_shock_bank_max_child_shocks": canonical_max_child_shocks,
        "max_evaluated_child_shocks": max_evaluated_child_shocks,
        "loaded": loaded,
    }
    defer_hash_to_outer = bool(getattr(args, "defer_model_state_hash_to_outer", False))
    active_model_names = list(loaded.metadata.get("loaded_model_keys", loaded.models.keys()))
    hash_before = (
        {name: _state_hash(loaded.models[name]) for name in active_model_names}
        if not defer_hash_to_outer else None
    )

    rows = []
    case_metadata = []
    requested_primary_eta = float(getattr(args, "eta", 1.0))
    primary_eta = requested_primary_eta if requested_primary_eta in eta_values else eta_values[0]
    # One shared BP pass per eta covers every requested J, so the J=Jmax child
    # equity forward and the coarse objective grid are never recomputed per J.
    evaluation_cache["bp_j_values"] = list(j_values)
    evaluation_cache["bp_primary_j"] = int(args.n_child_shocks)
    # Objective slices are per-eta artefacts: every eta writes its own primary-J
    # slices under ``root/<eta_label>/objective_slices``. The multi-J pass is
    # cached per eta, so the eta that happens to run first must not decide the
    # output root for the others.
    evaluation_cache["bp_primary_output_dirs"] = {
        float(eta): str(
            root / f"eta{eta:g}".replace("-", "m").replace(".", "p")
        )
        for eta in eta_values
    }
    evaluation_cache["bp_write_objective_slices"] = not bool(
        getattr(args, "summary_only_all", False)
    )
    primary_case_metadata: Dict[str, object] | None = None
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
            case_args.detailed_output = primary and not bool(
                getattr(args, "summary_only_all", False)
            )
            case_args.eta_values = None
            case_args.robustness_child_shocks = None
            case_args.shock_bank_max_child_shocks = canonical_max_child_shocks
            case_args._evaluation_cache = evaluation_cache
            case_args.loaded_checkpoint = loaded
            case_args.defer_model_state_hash = True
            case_args.manage_cuda_peak_stats = False
            summary, metadata = evaluate(case_args)
            if eta == primary_eta and child_count == int(args.n_child_shocks):
                primary_case_metadata = metadata
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
                "model_state_unchanged": None,
                "reference_transition_bank": metadata.get("reference_transition_bank"),
                "canonical_shock_bank_max_child_shocks": metadata.get(
                    "canonical_shock_bank_max_child_shocks"
                ),
                "max_evaluated_child_shocks": metadata.get("max_evaluated_child_shocks"),
                "evaluated_child_shocks": metadata.get("evaluated_child_shocks"),
                "transition_continuous_child_count": metadata.get(
                    "transition_continuous_child_count"
                ),
                "transition_expanded_child_count": metadata.get(
                    "transition_expanded_child_count"
                ),
                "timing": metadata.get("timing", {}),
                "bp_forward_stats": metadata.get("bp_forward_stats"),
                "bp_eval_max_expanded_states": metadata.get("evaluator_bp_max_expanded_states"),
            })

    combined = pd.DataFrame(rows)
    combined.to_csv(root / "summary.csv", index=False)
    robustness = root / "robustness"
    robustness.mkdir(parents=True, exist_ok=True)
    combined.to_csv(robustness / "j_comparison.csv", index=False)
    primary_case_metadata = primary_case_metadata or {}
    hash_after = (
        {name: _state_hash(loaded.models[name]) for name in active_model_names}
        if not defer_hash_to_outer else None
    )
    if not defer_hash_to_outer and hash_before != hash_after:
        raise RuntimeError("Checkpoint model state changed during matrix evaluation")
    matrix_state_unchanged = None if defer_hash_to_outer else True
    for item in case_metadata:
        item["model_state_unchanged"] = matrix_state_unchanged
        item["model_state_hash_scope"] = (
            "deferred_to_outer_episode" if defer_hash_to_outer else "matrix"
        )
    checkpoint_load_count = int(
        getattr(args, "checkpoint_load_count", checkpoint_load_count_local)
    )
    matrix_timing = {
        "total_seconds": time.perf_counter() - matrix_started,
        "firm_static_seconds": sum(
            float((item.get("timing") or {}).get("firm_static_seconds", 0.0))
            for item in case_metadata
        ),
        "investment_seconds": sum(
            float((item.get("timing") or {}).get("investment_seconds", 0.0))
            for item in case_metadata
        ),
        "bellman_seconds": sum(
            float((item.get("timing") or {}).get("bellman_seconds", 0.0))
            for item in case_metadata
        ),
        "bp_seconds": sum(
            float((item.get("timing") or {}).get("bp_seconds", 0.0))
            for item in case_metadata
        ),
        "transition_build_count": len(evaluation_cache.get("transition_by_eta", {})),
        "checkpoint_load_count": checkpoint_load_count,
        "checkpoint_load_count_local": checkpoint_load_count_local,
        "cuda_peak_memory_mb": (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
            if device.type == "cuda" and manage_cuda_peak_stats else float("nan")
        ),
    }
    matrix_timing["firm_structural_seconds"] = sum(
        float(matrix_timing[name])
        for name in ("firm_static_seconds", "investment_seconds", "bellman_seconds", "bp_seconds")
    )
    matrix_bp_forward_stats = sum_bp_forward_stats(
        item.get("bp_forward_stats") for item in case_metadata
    )
    matrix_timing.update(matrix_bp_forward_stats)
    metadata = {
        "evaluator": "firm_side_checkpoint_evaluator_v2_matrix",
        "git_commit_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "checkpoint_path": str(Path(args.checkpoint or args.pv_ckpt).expanduser().resolve()),
        "checkpoint_filename": Path(args.checkpoint or args.pv_ckpt).name,
        "eta_values": eta_values,
        "primary_eta": primary_eta,
        "primary_n_child_shocks": int(args.n_child_shocks),
        "robustness_n_child_shocks": j_values,
        "robustness_child_shocks": requested_robustness,
        "robustness_scope": getattr(args, "robustness_scope", "matrix_explicit"),
        "shock_seed": int(args.shock_seed),
        "common_random_numbers": True,
        "common_random_numbers_scope": "within_each_eta_grid_and_nested_prefix_across_J",
        "shock_bank_max_child_shocks": canonical_max_child_shocks,
        "canonical_shock_bank_max_child_shocks": canonical_max_child_shocks,
        "max_evaluated_child_shocks": max(j_values),
        "evaluated_child_shocks": j_values,
        "nested_shock_prefix_across_J": True,
        "formal_evaluator_eta_integration_mode": "exact",
        "formal_eta_integration_independent_of_training_ablation": True,
        "training_eta_integration_mode": primary_case_metadata.get(
            "training_eta_integration_mode"
        ),
        "bp_margin_identification_threshold": float(args.bp_teacher_margin_tol),
        "checkpoint_bp_grid_max_expanded_states": primary_case_metadata.get(
            "checkpoint_bp_grid_max_expanded_states"
        ),
        "evaluator_bp_max_expanded_states": primary_case_metadata.get(
            "evaluator_bp_max_expanded_states"
        ),
        "bp_eval_max_expanded_states_resolution": primary_case_metadata.get(
            "bp_eval_max_expanded_states_resolution"
        ),
        "bp_multi_j_shared_pass_per_eta": True,
        "bp_forward_stats": matrix_bp_forward_stats,
        "grid": primary_case_metadata.get("grid"),
        "reference_state": primary_case_metadata.get("reference_state"),
        "reference_transition_bank": primary_case_metadata.get("reference_transition_bank"),
        "m_mode": primary_case_metadata.get("m_mode"),
        "m_clamp_bounds": primary_case_metadata.get("m_clamp_bounds"),
        "q_parent_default_regime_mode": primary_case_metadata.get(
            "q_parent_default_regime_mode"
        ),
        "eps_default": primary_case_metadata.get("eps_default"),
        "tau_parent_default": primary_case_metadata.get("tau_parent_default"),
        "recovery_normalization_mode": primary_case_metadata.get(
            "recovery_normalization_mode"
        ),
        "recovery_normalization_mode_resolved": primary_case_metadata.get(
            "recovery_normalization_mode_resolved"
        ),
        "recovery_normalization_mode_source": primary_case_metadata.get(
            "recovery_normalization_mode_source"
        ),
        "q_parent_default_regime_mode_source": primary_case_metadata.get(
            "q_parent_default_regime_mode_source"
        ),
        "q_parent_default_eps_source": primary_case_metadata.get(
            "q_parent_default_eps_source"
        ),
        "q_parent_default_tau_source": primary_case_metadata.get(
            "q_parent_default_tau_source"
        ),
        "q_boundary_low_margin": primary_case_metadata.get("q_boundary_low_margin"),
        "model_state_hash_before": hash_before,
        "model_state_hash_after": hash_after,
        "model_state_hash_scope": "outer_episode" if defer_hash_to_outer else "matrix",
        "model_state_unchanged": matrix_state_unchanged,
        "timing": matrix_timing,
        "matrix_reuse": {
            "checkpoint_loaded_once": True,
            "static_surfaces_once_per_eta": True,
            "transition_built_at_Jmax_once_per_eta": True,
        },
        "cases": case_metadata,
        "case_metadata_by_eta_j": {
            f"eta{item['eta_parent']:g}_J{item['n_child_shocks']}": item
            for item in case_metadata
        },
        "semantics": {
            "child_leverage_timing": "b_next = eta_current * bp_current + (1-eta_current) * b_current",
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
