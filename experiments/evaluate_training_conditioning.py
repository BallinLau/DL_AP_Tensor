"""Read-only training-conditioning evaluator for Hybrid-Q conditioning runs.

Primary objective (per spec): TRAINING HEALTH / FUNCTION SHAPE / TEACHER-FIT
DIAGNOSTICS for the three c766537 conditioning fixes

1. Hybrid Q_claim low-b synthetic coverage
2. P-only current-eta balanced collocation
3. staged cached-P normalized Bellman loss

This evaluator deliberately does NOT issue an equilibrium convergence verdict.
Cross-episode function drift is reported as a SECONDARY diagnostic only.

Read-only guarantees:
- every model forward runs under ``torch.no_grad()``
- no optimizer / scheduler / firm_target update is ever constructed
- per-episode model state hashes are recorded before and after evaluation
- shock banks use dedicated seeded generators; global RNG state is untouched
- nothing inside ``--run-root`` is ever written

All heavy machinery is reused from the existing evaluator stack:
- analysis.checkpoint_loader.load_analysis_checkpoint
- evaluation.grids (reference state + frozen grids)
- evaluation.firm_surfaces.evaluate_firm_surfaces (production surfaces)
- evaluation.bp_diagnostics.build_frozen_transition_data /
  evaluate_bp_consistency_multi_j (shock bank + BPGridTeacher)
- evaluation.full_run_diagnostics.model_state_hash
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.checkpoint_loader import load_analysis_checkpoint  # noqa: E402
from evaluation.bp_diagnostics import (  # noqa: E402
    _checkpoint_economic_config,
    build_frozen_transition_data,
    evaluate_bp_consistency_multi_j,
)
from evaluation.boundaries import zero_crossings  # noqa: E402
from evaluation.convergence_artifacts import (  # noqa: E402
    discover_episode_firm_data,
    parse_episode_selection,
)
from evaluation.convergence_metrics import choose_representative_episodes  # noqa: E402
from evaluation.full_run_diagnostics import (  # noqa: E402
    evaluate_fc1_checkpoint,
    model_state_hash,
    write_json,
)
from evaluation.firm_surfaces import evaluate_firm_surfaces  # noqa: E402
from evaluation.grids import (  # noqa: E402
    FrozenFirmGrid,
    ReferenceFirmState,
    build_frozen_grid,
    load_reference_state,
)
from experiments.evaluate_full_run import _choose_reference  # noqa: E402
from losses import PILoss, P0Loss  # noqa: E402
from losses.utils import compute_cashflow  # noqa: E402
from training.bp_grid_teacher import BPGridTeacher  # noqa: E402


# ---------------------------------------------------------------------------
# Spec constants
# ---------------------------------------------------------------------------

# §5 expected run configuration. Any mismatch is reported as ``config_warning``
# and the evaluation continues (never silently overwritten).
EXPECTED_CONFIG: Tuple[Tuple[str, Any], ...] = (
    ("q_parameterization", "hybrid_regime"),
    ("pv_value_scale_mode", "exp_xz"),
    ("pv_bellman_normalize_by_value_scale", True),
    ("q_claim_coverage_enabled", True),
    ("q_claim_coverage_b_bins", 10),
    ("q_claim_coverage_low_b_enabled", True),
    ("q_claim_coverage_low_b_anchors", (0.005, 0.01, 0.025)),
    ("pv_current_eta_balance_enabled", True),
    ("pv_current_eta1_train_share", 0.50),
    ("pv_current_eta_balance_validation", True),
    ("pv_current_eta_balance_seed", 97531),
    ("pv_exact_eta_integration_enabled", True),
    ("pv_mixture_enabled", True),
    ("pv_mixture_ratio", 0.20),
    ("q_survival_ondist_share", 0.80),
    ("q_shape_weight_z", 0.0),
    ("q_shape_weight_b_low", 0.0),
    ("q_shape_weight_b_high", 0.0),
    ("q_boundary_match_weight", 0.0),
)

# §6 training-log diagnostics. Each entry maps the reported column name to the
# candidate raw keys inside the "Episode N done: {...}" payload.
TRAINING_LOG_FIELDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("claim_coverage_anchor_count", ("claim_coverage_anchor_count",)),
    ("claim_coverage_anchors_occupied", ("claim_coverage_anchors_occupied",)),
    ("claim_coverage_fraction_anchors_occupied", ("claim_coverage_fraction_anchors_occupied",)),
    ("claim_coverage_low_b_enabled", ("claim_coverage_low_b_enabled",)),
    ("claim_coverage_low_b_anchor_count", ("claim_coverage_low_b_anchor_count",)),
    ("claim_coverage_low_b_sample_count", ("claim_coverage_low_b_sample_count",)),
    ("claim_coverage_low_b_sample_share", ("claim_coverage_low_b_sample_share",)),
    ("claim_coverage_b_min", ("claim_coverage_b_min",)),
    ("claim_coverage_b_max", ("claim_coverage_b_max",)),
    ("p_train_eta1_share_before", ("p_train_eta1_share_before",)),
    ("p_train_eta1_share_after", ("p_train_eta1_share_after",)),
    ("p_validation_eta1_share_before", ("p_validation_eta1_share_before",)),
    ("p_validation_eta1_share_after", ("p_validation_eta1_share_after",)),
    ("p_eta_target_share", ("p_eta_target_share",)),
    ("q_batches_eta1_share", ("q_train_current_eta1_share", "q_batches_eta_rebalanced")),
    ("bp_batches_eta1_share", ("bp_train_current_eta1_share", "bp_batches_eta_rebalanced")),
)

Z_BINS: Tuple[Tuple[float, float], ...] = ((-4.0, -2.0), (-2.0, 0.0), (0.0, 1.0), (1.0, 2.0), (2.0, 4.0))
ANCHOR_Z_SLICES: Tuple[float, ...] = (-4.0, -2.0, 0.0, 2.0, 4.0)
DEFAULT_ANCHOR_B: Tuple[float, ...] = (
    0.0, 0.005, 0.01, 0.025, 0.05,
    0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95,
)
NEAR_DEFAULT_EPS: Tuple[float, ...] = (0.25, 0.5, 1.0)
EPISODE_DONE_RE = re.compile(
    r"Episode\s+(?P<episode>\d+)[^\n]*?done:\s*(?P<payload>\{.*\})", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def build_custom_grid(
    reference: ReferenceFirmState,
    b_values: np.ndarray,
    z_values: np.ndarray,
    *,
    eta: float,
    device: torch.device,
    i_value: Optional[float] = None,
    dtype: torch.dtype = torch.float32,
) -> FrozenFirmGrid:
    """Frozen grid over explicit b/z values (mirrors ``build_frozen_grid``)."""
    b_values = np.asarray(b_values, dtype=np.float64)
    z_values = np.asarray(z_values, dtype=np.float64)
    mesh_b, mesh_z = np.meshgrid(b_values, z_values, indexing="ij")
    states = torch.tensor(
        np.column_stack(
            [
                mesh_b.reshape(-1),
                mesh_z.reshape(-1),
                np.full(mesh_b.size, float(eta)),
                np.full(mesh_b.size, float(reference.i_mid if i_value is None else i_value)),
                np.full(mesh_b.size, reference.x),
                np.full(mesh_b.size, reference.hatcf),
                np.full(mesh_b.size, reference.lnkf),
            ]
        ),
        dtype=dtype,
        device=device,
    )
    return FrozenFirmGrid(
        b_values=b_values, z_values=z_values, mesh_b=mesh_b, mesh_z=mesh_z, base_states=states
    )


def build_low_b_values(
    low_b_max: float,
    low_b_points: int,
    anchors: Sequence[float],
) -> np.ndarray:
    """§7.2 low-b grid: union(linspace(0, low_b_max, points), anchors, low_b_max)."""
    grid = np.linspace(0.0, float(low_b_max), int(low_b_points), dtype=np.float64)
    values = np.concatenate([grid, np.asarray(list(anchors), dtype=np.float64), [float(low_b_max)]])
    return np.unique(np.sort(values))


def forward_model_fields(
    model: torch.nn.Module,
    states: torch.Tensor,
    fields: Sequence[str],
    *,
    chunk_size: int,
) -> Dict[str, torch.Tensor]:
    step = max(int(chunk_size), 1)
    collected: Dict[str, List[torch.Tensor]] = {name: [] for name in fields}
    with torch.no_grad():
        for start in range(0, states.shape[0], step):
            output = model(states[start : start + step])
            for name in fields:
                value = output[name] if isinstance(output, dict) else getattr(output, name)
                collected[name].append(value.detach())
    return {name: torch.cat(parts, dim=0) for name, parts in collected.items()}


def grid_forward_surfaces(
    model: torch.nn.Module,
    grid: FrozenFirmGrid,
    reference: ReferenceFirmState,
    *,
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    return evaluate_firm_surfaces(model, grid, reference, chunk_size=chunk_size)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _finite_stats(values: np.ndarray) -> Dict[str, float]:
    finite = np.isfinite(values)
    if not finite.any():
        return {key: float("nan") for key in ("mean", "median", "p90", "p95", "p99", "max")}
    selected = np.asarray(values[finite], dtype=np.float64)
    return {
        "mean": float(selected.mean()),
        "median": float(np.median(selected)),
        "p90": float(np.quantile(selected, 0.90)),
        "p95": float(np.quantile(selected, 0.95)),
        "p99": float(np.quantile(selected, 0.99)),
        "max": float(selected.max()),
    }


def finite_difference_stats(
    surface: np.ndarray,
    axis: int,
    coordinates: np.ndarray,
) -> Dict[str, float]:
    """§7.4 / §12 first- and second-order finite-difference quantiles."""
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.size < 2:
        return {f"d_abs_{key}": float("nan") for key in ("p95", "max")}
    moved = np.moveaxis(np.asarray(surface, dtype=np.float64), axis, 0)
    spacing = np.diff(coords)
    spacing2 = coords[2:] - coords[:-2]
    d1 = np.diff(moved, axis=0) / spacing.reshape((-1,) + (1,) * (moved.ndim - 1))
    stats: Dict[str, float] = {}
    d1_abs = np.abs(d1)
    for name, value in _finite_stats(d1_abs).items():
        if name in ("p95", "max", "p99"):
            stats[f"d_abs_{name}"] = value
    if coords.size >= 3:
        d2 = np.diff(moved, n=2, axis=0) / spacing2.reshape((-1,) + (1,) * (moved.ndim - 1))
        for name, value in _finite_stats(np.abs(d2)).items():
            if name in ("p95", "max", "p99"):
                stats[f"d2_abs_{name}"] = value
    else:
        stats[f"d2_abs_p95"] = float("nan")
        stats[f"d2_abs_max"] = float("nan")
    return stats


def value_scale_grid(model: torch.nn.Module, grid: FrozenFirmGrid) -> np.ndarray:
    """Production equity_value_scale helper evaluated on the frozen grid."""
    with torch.no_grad():
        scale = model.equity_value_scale(grid.base_states).detach()
    return scale.reshape(grid.shape).cpu().numpy().astype(np.float64)


def confusion_metrics(
    d_pred: np.ndarray, d_teacher: np.ndarray
) -> Dict[str, float]:
    """§10 predicted-vs-teacher default confusion (D = 1{Phat <= 0})."""
    d_pred = np.isfinite(d_pred) & (np.asarray(d_pred) > 0.5)
    d_teacher = np.isfinite(d_teacher) & (np.asarray(d_teacher) > 0.5)
    total = d_pred.size
    if total == 0:
        return {key: float("nan") for key in (
            "pred_default_share", "teacher_default_share", "agreement_share",
            "false_survival_share", "false_default_share",
        )}
    false_survival = (~d_pred) & d_teacher
    false_default = d_pred & (~d_teacher)
    return {
        "pred_default_share": float(d_pred.mean()),
        "teacher_default_share": float(d_teacher.mean()),
        "agreement_share": float((d_pred == d_teacher).mean()),
        "false_survival_share": float(false_survival.mean()),
        "false_default_share": float(false_default.mean()),
    }


def boundary_summary_for_surface(
    phat: np.ndarray,
    b_values: np.ndarray,
    z_values: np.ndarray,
) -> Dict[str, float]:
    """§11 default-boundary diagnostics along z for every fixed b row."""
    phat = np.asarray(phat, dtype=np.float64)
    n_b = phat.shape[0]
    crossing_counts = np.zeros(n_b, dtype=np.int64)
    all_survival = np.zeros(n_b, dtype=bool)
    all_default = np.zeros(n_b, dtype=bool)
    crossing_z: List[Optional[float]] = [None] * n_b
    crossing_sets: List[np.ndarray] = []
    for row in range(n_b):
        values = phat[row]
        finite = np.isfinite(values)
        if not finite.any():
            continue
        if bool((values[finite] > 0.0).all()):
            all_survival[row] = True
            continue
        if bool((values[finite] <= 0.0).all()):
            all_default[row] = True
            continue
        crossings = zero_crossings(z_values, values)
        crossing_counts[row] = len(crossings)
        crossing_sets.append(np.asarray(crossings, dtype=np.float64))
        if len(crossings) == 1:
            crossing_z[row] = float(crossings[0])
    valid = np.isfinite(phat).all(axis=1)
    crossing_share_rows = crossing_counts > 0
    single = crossing_counts == 1
    z_default_values = np.asarray(
        [value if value is not None else np.nan for value in crossing_z], dtype=np.float64
    )
    summary = {
        "all_survival_share": float((all_survival & valid).mean()) if valid.any() else float("nan"),
        "all_default_share": float((all_default & valid).mean()) if valid.any() else float("nan"),
        "crossing_share": float(crossing_share_rows.mean()),
        "crossing_count_max": int(crossing_counts.max()),
        "crossing_count_gt1_share": float((crossing_counts > 1).mean()),
        "z_default_median": float(np.nanmedian(z_default_values)) if single.any() else float("nan"),
        "z_default_p10": float(np.nanquantile(z_default_values, 0.10)) if single.any() else float("nan"),
        "z_default_p90": float(np.nanquantile(z_default_values, 0.90)) if single.any() else float("nan"),
    }
    return summary


def boundary_gap_stats(
    phat_pred: np.ndarray,
    phat_teacher: np.ndarray,
    b_values: np.ndarray,
    z_values: np.ndarray,
) -> Dict[str, float]:
    """§11 boundary_z_abs_gap on b rows where both surfaces cross."""
    gaps: List[float] = []
    for row in range(phat_pred.shape[0]):
        pred_cross = zero_crossings(z_values, phat_pred[row])
        teacher_cross = zero_crossings(z_values, phat_teacher[row])
        if len(pred_cross) == 0 or len(teacher_cross) == 0:
            continue
        pred_arr = np.asarray(pred_cross, dtype=np.float64)
        teacher_arr = np.asarray(teacher_cross, dtype=np.float64)
        gaps.append(float(np.min(np.abs(pred_arr[:, None] - teacher_arr[None, :]))))
    if not gaps:
        return {
            "boundary_z_abs_gap_mean": float("nan"),
            "boundary_z_abs_gap_p90": float("nan"),
            "boundary_z_abs_gap_max": float("nan"),
            "boundary_rows_compared": 0,
        }
    arr = np.asarray(gaps, dtype=np.float64)
    return {
        "boundary_z_abs_gap_mean": float(arr.mean()),
        "boundary_z_abs_gap_p90": float(np.quantile(arr, 0.90)),
        "boundary_z_abs_gap_max": float(arr.max()),
        "boundary_rows_compared": int(arr.size),
    }


def zbin_residual_frame(
    z_values: np.ndarray,
    phys: np.ndarray,
    norm: np.ndarray,
    scale: np.ndarray,
    *,
    eta: float,
    module: str,
    b_axis: int = 0,
    i_axis: Optional[int] = None,
) -> pd.DataFrame:
    """§9 z-bin grouped residuals (pooled over b and, when present, i)."""
    z_values = np.asarray(z_values, dtype=np.float64)
    phys = np.asarray(phys, dtype=np.float64)
    norm = np.asarray(norm, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    phys_pooled = phys
    norm_pooled = norm
    scale_pooled = np.broadcast_to(scale, phys.shape)
    if i_axis is not None:
        phys_pooled = np.moveaxis(phys, i_axis, 0).reshape(-1, *phys.shape[1:])
        norm_pooled = np.moveaxis(norm, i_axis, 0).reshape(-1, *norm.shape[1:])
        scale_pooled = np.broadcast_to(scale, phys.shape)
        scale_pooled = np.moveaxis(scale_pooled, i_axis, 0).reshape(-1, *phys.shape[1:])
        z_axis_in_pooled = 1 + (b_axis if b_axis > i_axis else b_axis)
    else:
        z_axis_in_pooled = b_axis
    rows = []
    for z_lo, z_hi in Z_BINS:
        lo = max(z_lo, float(z_values.min()))
        hi = min(z_hi, float(z_values.max()))
        if not lo < hi:
            continue
        mask = (z_values >= lo - 1e-12) & (z_values <= hi + 1e-12)
        if not mask.any():
            continue
        take = [slice(None)] * phys_pooled.ndim
        take[z_axis_in_pooled] = mask
        phys_bin = phys_pooled[tuple(take)]
        norm_bin = norm_pooled[tuple(take)]
        scale_bin = scale_pooled[tuple(take)]
        phys_stats = _finite_stats(np.abs(phys_bin))
        norm_stats = _finite_stats(np.abs(norm_bin))
        scale_stats = _finite_stats(scale_bin)
        rows.append({
            "eta": eta,
            "module": module,
            "z_bin": f"[{z_lo:g},{z_hi:g}]",
            "n_states": int(np.isfinite(phys_bin).sum()),
            "physical_abs_residual_mean": phys_stats["mean"],
            "physical_abs_residual_p90": phys_stats["p90"],
            "normalized_abs_residual_mean": norm_stats["mean"],
            "normalized_abs_residual_p90": norm_stats["p90"],
            "value_scale_mean": scale_stats["mean"],
            "value_scale_p90": scale_stats["p90"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §6 training-log conditioning extractor
# ---------------------------------------------------------------------------


def parse_conditioning_training_log(path: Optional[Path]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Parse "Episode N done: {...}" payload dicts from a training log."""
    if path is None:
        return pd.DataFrame(), {"status": "missing", "reason": "no --training-log given"}
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame(), {"status": "missing", "reason": f"file not found: {path}"}
    rows: Dict[int, Dict[str, Any]] = {}
    parse_errors = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EPISODE_DONE_RE.search(line)
        if not match:
            continue
        episode = int(match.group("episode"))
        try:
            payload = ast.literal_eval(match.group("payload"))
        except (SyntaxError, ValueError):
            parse_errors += 1
            continue
        if not isinstance(payload, dict):
            continue
        row: Dict[str, Any] = {"episode": episode}
        for column, candidates in TRAINING_LOG_FIELDS:
            value: Any = None
            for key in candidates:
                if key in payload:
                    value = payload[key]
                    break
            if isinstance(value, (list, tuple)):
                value = ",".join(repr(float(item)) for item in value)
            elif isinstance(value, (bool, int, float)):
                value = float(value) if not isinstance(value, bool) else bool(value)
            row[column] = float("nan") if value is None else value
        rows[episode] = row
    if not rows:
        return pd.DataFrame(), {
            "status": "missing",
            "reason": "no 'Episode N done: {...}' payloads found",
            "parse_errors": parse_errors,
        }
    frame = pd.DataFrame(list(rows.values())).sort_values("episode").reset_index(drop=True)
    present = [column for _, candidates in TRAINING_LOG_FIELDS for column in [candidates[0]] if column in frame.columns and frame[column].notna().any()]
    return frame, {
        "status": "ok",
        "path": str(path.resolve()),
        "episodes_parsed": int(len(frame)),
        "fields_with_values": present,
        "fields_missing": [name for name, _ in TRAINING_LOG_FIELDS if name not in frame.columns or frame[name].isna().all()],
        "parse_errors": parse_errors,
    }


# ---------------------------------------------------------------------------
# §8 teacher block: T0/TI over the shared i-grid
# ---------------------------------------------------------------------------


def teacher_surfaces_over_i_grid(
    teacher: BPGridTeacher,
    grids_by_eta: Dict[float, FrozenFirmGrid],
    transition,
    *,
    eta_values: Sequence[float],
    i_values: np.ndarray,
    prefix_child_counts: Sequence[int],
    progress_every: int = 20,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[float, Dict[str, np.ndarray]]:
    """One-step Bellman teacher per i slice.

    Children/shocks depend on parent (x, z) only — never on parent i/b/eta —
    so one transition per episode is shared by every i slice and every eta.
    ``compute_multi_j_branches`` requires identical parent leverage (column 0)
    across the branch-state list; the eta0/eta1 grids share the same b column.
    """
    children_max = transition.stacked_children()
    m_max = transition.stacked_m_used()
    child_weights = transition.branch_weights
    count = int(prefix_child_counts[0])
    output: Dict[float, Dict[str, np.ndarray]] = {
        eta: {"T0": [], "TI": []} for eta in eta_values
    }
    shape = next(iter(grids_by_eta.values())).shape
    for i_index, i_value in enumerate(np.asarray(i_values, dtype=np.float64)):
        branch_states: List[torch.Tensor] = []
        for eta in eta_values:
            states = grids_by_eta[eta].base_states.clone()
            states[:, 3] = float(i_value)
            branch_states.extend([states, states])
        with torch.no_grad():
            bundles = teacher.compute_multi_j_branches(
                branch_states,
                children_max,
                m_max,
                branches=["p0", "pi"] * len(eta_values),
                prefix_child_counts=[count],
                child_weights=child_weights,
            )
        for eta_index, eta in enumerate(eta_values):
            t0 = bundles[2 * eta_index][count]["value_star"].detach()
            ti = bundles[2 * eta_index + 1][count]["value_star"].detach()
            output[eta]["T0"].append(t0.reshape(shape).cpu().numpy().astype(np.float64))
            output[eta]["TI"].append(ti.reshape(shape).cpu().numpy().astype(np.float64))
        if log is not None and (i_index + 1) % progress_every == 0:
            log(f"    teacher i-slice {i_index + 1}/{len(i_values)}")
    for eta in eta_values:
        output[eta]["T0"] = np.stack(output[eta]["T0"], axis=0)
        output[eta]["TI"] = np.stack(output[eta]["TI"], axis=0)
    return output


def teacher_result_at_star_components(
    result: Dict[str, torch.Tensor],
) -> Dict[str, np.ndarray]:
    """Gather teacher objective components at the fine-grid star candidate."""
    argmax = result["argmax_index"]
    rows = torch.arange(argmax.shape[0], device=argmax.device)
    flat = argmax.reshape(-1)

    def gather(key: str) -> np.ndarray:
        return result[key][rows, flat].detach().cpu().numpy().astype(np.float64)

    components = {
        "value_at_star": gather("value_grid"),
        "cashflow_at_star": gather("cashflow_grid_mean"),
        "continuation_at_star": gather("continuation_grid_mean"),
        "bp_star": result["bp_star"].detach().reshape(-1).cpu().numpy().astype(np.float64),
        "q_issue_claim_at_star": result["q_issue_at_star"].detach().reshape(-1).cpu().numpy().astype(np.float64),
        "p_child_at_star": result["p_child_at_star"].detach().reshape(-1).cpu().numpy().astype(np.float64),
        "default_at_star": result["default_at_star"].detach().reshape(-1).cpu().numpy().astype(np.float64),
    }
    if "q_current_claim" in result:
        components["q_current_claim"] = result["q_current_claim"].detach().reshape(-1).cpu().numpy().astype(np.float64)
    return components


# ---------------------------------------------------------------------------
# Per-episode conditioning evaluation
# ---------------------------------------------------------------------------


def evaluate_episode_conditioning(
    *,
    episode: int,
    checkpoint: Path,
    reference: ReferenceFirmState,
    args: argparse.Namespace,
    device: torch.device,
    output_root: Path,
    log: Callable[[str], None],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate one episode. Returns (metrics_row, artifacts, errors)."""
    errors: List[Dict[str, Any]] = []
    timing = time.perf_counter()

    def record_error(block: str, exc: BaseException) -> None:
        errors.append({
            "episode": episode,
            "block": block,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=6),
        })
        log(f"  [ep{episode}] {block} FAILED: {type(exc).__name__}: {exc}")

    loaded = load_analysis_checkpoint(checkpoint, device=device, m_source="sdf_fc1")
    model = loaded.models["policy_value"]
    sdf_fc1_model = loaded.models["sdf_fc1"]
    model.eval()
    sdf_fc1_model.eval()
    hash_before = {
        "policy_value": model_state_hash(model),
        "sdf_fc1": model_state_hash(sdf_fc1_model),
    }

    # ---- §5 configuration audit ----
    config_audit: Dict[str, Any] = {}
    config_warnings: List[str] = []
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        hp_payload = payload.get("hyperparams") if isinstance(payload, dict) else None
        hp_dict = hp_payload if isinstance(hp_payload, dict) else (
            hp_payload.__dict__ if hp_payload is not None else {}
        )
        for key, expected in EXPECTED_CONFIG:
            actual = hp_dict.get(key, "<absent>")
            matches = (
                tuple(np.asarray(actual).ravel().tolist()) == tuple(np.asarray(expected).ravel().tolist())
                if isinstance(actual, (list, tuple, np.ndarray)) or isinstance(expected, (tuple, list))
                else actual == expected
            )
            config_audit[key] = {
                "expected": expected if not isinstance(expected, tuple) else list(expected),
                "actual": actual if not isinstance(actual, tuple) else list(actual),
                "match": bool(matches),
            }
            if not matches:
                config_warnings.append(f"{key}: expected {expected!r}, checkpoint has {actual!r}")
    except Exception as exc:  # noqa: BLE001
        record_error("config_audit", exc)

    eta_values = [float(value) for value in args.eta_values]
    chunk = int(args.forward_chunk_size)

    # ---- frozen grids: full b/z grid + dedicated low-b grid (§7.2) ----
    grids_by_eta: Dict[float, FrozenFirmGrid] = {}
    low_grids_by_eta: Dict[float, FrozenFirmGrid] = {}
    anchor_grids_by_eta: Dict[float, FrozenFirmGrid] = {}
    low_b_values = build_low_b_values(args.low_b_max, args.low_b_points, args.low_b_anchors)
    for eta in eta_values:
        reference_eta = dataclasses.replace(reference, eta=eta)
        grids_by_eta[eta] = build_frozen_grid(
            reference_eta,
            b_min=args.b_min, b_max=args.b_max, b_points=args.b_points,
            z_min=args.z_min, z_max=args.z_max, z_points=args.z_points,
            device=device,
        )
        low_grids_by_eta[eta] = build_custom_grid(
            reference_eta, low_b_values, grids_by_eta[eta].z_values,
            eta=eta, device=device,
        )
        anchor_grids_by_eta[eta] = build_custom_grid(
            reference_eta, np.asarray(args.anchor_b_values, dtype=np.float64),
            grids_by_eta[eta].z_values, eta=eta, device=device,
        )

    # ---- §7.1 production surfaces per eta ----
    surfaces_by_eta: Dict[float, Dict[str, np.ndarray]] = {}
    low_surfaces_by_eta: Dict[float, Dict[str, np.ndarray]] = {}
    anchor_surfaces_by_eta: Dict[float, Dict[str, np.ndarray]] = {}
    for eta in eta_values:
        reference_eta = dataclasses.replace(reference, eta=eta)
        try:
            surfaces_by_eta[eta] = grid_forward_surfaces(
                model, grids_by_eta[eta], reference_eta, chunk_size=chunk
            )
            low_surfaces_by_eta[eta] = grid_forward_surfaces(
                model, low_grids_by_eta[eta], reference_eta, chunk_size=chunk
            )
            anchor_surfaces_by_eta[eta] = grid_forward_surfaces(
                model, anchor_grids_by_eta[eta], reference_eta, chunk_size=chunk
            )
        except Exception as exc:  # noqa: BLE001
            record_error("q_surfaces", exc)

    metrics: Dict[str, Any] = {"episode": episode}
    per_eta_blocks: Dict[str, Dict[float, Any]] = {name: {} for name in (
        "q_low_b", "q_anchor", "p_residual", "zbin", "confusion", "boundary",
    )}

    # ---- §7 low-b Q metrics / structural zero ----
    for eta in eta_values:
        try:
            low = low_surfaces_by_eta[eta]
            full = surfaces_by_eta[eta]
            block: Dict[str, Any] = {}
            for name in ("q_unit", "Q_claim"):
                for stat, value in _finite_stats(low[name]).items():
                    block[f"{name}_low_b_{stat}"] = value
            block.update({
                f"qclaim_low_b_d1_{key}": value
                for key, value in finite_difference_stats(
                    low["Q_claim"], axis=0, coordinates=low_b_values
                ).items()
            })
            block.update({
                f"qclaim_full_d1_{key}": value
                for key, value in finite_difference_stats(
                    full["Q_claim"], axis=0, coordinates=grids_by_eta[eta].b_values
                ).items()
            })
            b0_mask = np.isclose(grids_by_eta[eta].b_values, 0.0)
            if b0_mask.any():
                block["Q_claim_b0_abs_max"] = float(np.nanmax(np.abs(full["Q_claim"][b0_mask, :])))
                block["q_unit_b0_mean"] = float(np.nanmean(full["q_unit"][b0_mask, :]))
            else:
                block["Q_claim_b0_abs_max"] = float("nan")
                block["q_unit_b0_mean"] = float("nan")
            per_eta_blocks["q_low_b"][eta] = block
        except Exception as exc:  # noqa: BLE001
            record_error("q_low_b_metrics", exc)

    # ---- §7.5 exact-anchor table ----
    anchor_rows: List[Dict[str, Any]] = []
    try:
        anchor_z_mask = np.isin(anchor_grids_by_eta[eta_values[0]].z_values, list(ANCHOR_Z_SLICES))
        for eta in eta_values:
            surf = anchor_surfaces_by_eta[eta]
            b_grid = anchor_grids_by_eta[eta].b_values
            z_grid = anchor_grids_by_eta[eta].z_values
            for b_index, b_value in enumerate(b_grid):
                for z_index in np.where(anchor_z_mask)[0]:
                    anchor_rows.append({
                        "episode": episode,
                        "eta": eta,
                        "b": float(b_value),
                        "z": float(z_grid[z_index]),
                        "q_unit": float(surf["q_unit"][b_index, z_index]),
                        "Q_claim": float(surf["Q_claim"][b_index, z_index]),
                        "Q_effective": float(surf["Q_effective"][b_index, z_index]),
                        "recovery": float(surf["recovery"][b_index, z_index]),
                        "Phat": float(surf["Phat"][b_index, z_index]),
                        "default_mask": float(surf["realized_default_mask"][b_index, z_index]),
                    })
    except Exception as exc:  # noqa: BLE001
        record_error("anchor_table", exc)

    # ---- shock transition: children depend on parent (x, z) only, so ONE
    #      transition (sized at the canonical bank) is shared by the teacher
    #      i-grid, the near-default decomposition, and the BP block. ----
    canonical_bank_max = max(
        [int(args.n_child_shocks), *(int(value) for value in args.robustness_child_shocks)]
    )
    transition = build_frozen_transition_data(
        sdf_fc1_model,
        grids_by_eta[eta_values[0]].base_states,
        dataclasses.replace(reference, eta=eta_values[0]),
        loaded.hyperparams,
        loaded.economic_config,
        n_child_shocks=canonical_bank_max,
        shock_seed=args.shock_seed,
        shock_bank_max_child_shocks=canonical_bank_max,
    )

    p0_loss = P0Loss(
        delta=loaded.economic_config.DELTA, tau=loaded.economic_config.TAU,
        kappa_b=loaded.economic_config.KAPPA_B, kappa_e=loaded.economic_config.KAPPA_E,
        aio_weight=loaded.economic_config.AIO_WEIGHT,
        alpha_z=loaded.economic_config.ALPHA_Z, beta_z=loaded.economic_config.BETA_Z,
        z0=loaded.economic_config.Z0,
    )
    pi_loss = PILoss(
        delta=loaded.economic_config.DELTA, tau=loaded.economic_config.TAU,
        g=loaded.economic_config.G,
        kappa_b=loaded.economic_config.KAPPA_B, kappa_e=loaded.economic_config.KAPPA_E,
        aio_weight=loaded.economic_config.AIO_WEIGHT,
        alpha_z=loaded.economic_config.ALPHA_Z, beta_z=loaded.economic_config.BETA_Z,
        z0=loaded.economic_config.Z0, b_penalty_weight=0.0,
    )
    teacher = BPGridTeacher.from_hyperparams(
        model, p0_loss, pi_loss, loaded.hyperparams,
        max_expanded_states_override=int(args.bp_eval_max_expanded_states),
    )
    teacher.reset_forward_stats()

    i_values = np.linspace(
        0.0, float(model.i_threshold), int(args.i_points), dtype=np.float64
    )

    # ---- §8 teacher + prediction surfaces over the shared i-grid ----
    p0_pred_by_eta: Dict[float, np.ndarray] = {}
    pi_pred_by_eta: Dict[float, np.ndarray] = {}
    scale_by_eta: Dict[float, np.ndarray] = {}
    teacher_by_eta: Dict[float, Dict[str, np.ndarray]] = {}
    try:
        for eta in eta_values:
            states_full = grids_by_eta[eta].base_states
            p0_slices, pi_slices = [], []
            for i_value in i_values:
                states_i = states_full.clone()
                states_i[:, 3] = float(i_value)
                fields = forward_model_fields(
                    model, states_i, ("P0", "PI"), chunk_size=chunk
                )
                p0_slices.append(fields["P0"].reshape(grids_by_eta[eta].shape).cpu().numpy().astype(np.float64))
                pi_slices.append(fields["PI"].reshape(grids_by_eta[eta].shape).cpu().numpy().astype(np.float64))
            p0_pred_by_eta[eta] = np.stack(p0_slices, axis=0)
            pi_pred_by_eta[eta] = np.stack(pi_slices, axis=0)
            scale_by_eta[eta] = value_scale_grid(model, grids_by_eta[eta])
        log(f"  [ep{episode}] prediction i-grid done ({len(i_values)} slices); running teacher")
        with _checkpoint_economic_config(loaded.economic_config):
            teacher_by_eta = teacher_surfaces_over_i_grid(
                teacher, grids_by_eta, transition,
                eta_values=eta_values, i_values=i_values,
                prefix_child_counts=[2 * int(args.n_child_shocks)],
                log=log,
            )
    except Exception as exc:  # noqa: BLE001
        record_error("teacher_i_grid", exc)

    # ---- §8 residuals (physical + production normalized) and §9/§10/§11 ----
    phat_pred_by_eta: Dict[float, np.ndarray] = {}
    phat_teacher_by_eta: Dict[float, np.ndarray] = {}
    for eta in eta_values:
        try:
            if eta not in teacher_by_eta or eta not in p0_pred_by_eta:
                continue
            t0 = teacher_by_eta[eta]["T0"]
            ti = teacher_by_eta[eta]["TI"]
            phat_teacher = np.mean(np.maximum(t0, ti), axis=0)
            phat_pred = np.mean(np.maximum(p0_pred_by_eta[eta], pi_pred_by_eta[eta]), axis=0)
            phat_pred_by_eta[eta] = phat_pred
            phat_teacher_by_eta[eta] = phat_teacher
            scale = scale_by_eta[eta]
            r0_phys = p0_pred_by_eta[eta] - t0
            ri_phys = pi_pred_by_eta[eta] - ti
            r0_norm = r0_phys / scale[None, :, :]
            ri_norm = ri_phys / scale[None, :, :]
            p_block: Dict[str, Any] = {}
            for name, arr in (
                ("P0_phys", r0_phys), ("PI_phys", ri_phys),
                ("P0_norm", r0_norm), ("PI_norm", ri_norm),
            ):
                for stat, value in _finite_stats(np.abs(arr)).items():
                    p_block[f"{name}_{stat}"] = value
            per_eta_blocks["p_residual"][eta] = p_block
            zbin_frames = []
            for module, phys, norm in (("P0", r0_phys, r0_norm), ("PI", ri_phys, ri_norm)):
                zbin_frames.append(zbin_residual_frame(
                    grids_by_eta[eta].z_values, phys, norm, scale,
                    eta=eta, module=module, b_axis=1, i_axis=0,
                ))
            per_eta_blocks["zbin"][eta] = pd.concat(zbin_frames, ignore_index=True)
            d_pred = (phat_pred <= 0.0).astype(np.float64)
            d_teacher = (phat_teacher <= 0.0).astype(np.float64)
            per_eta_blocks["confusion"][eta] = confusion_metrics(d_pred, d_teacher)
            boundary = boundary_summary_for_surface(
                phat_pred, grids_by_eta[eta].b_values, grids_by_eta[eta].z_values
            )
            boundary.update({
                f"teacher_{key}": value
                for key, value in boundary_summary_for_surface(
                    phat_teacher, grids_by_eta[eta].b_values, grids_by_eta[eta].z_values
                ).items()
            })
            boundary.update(boundary_gap_stats(
                phat_pred, phat_teacher,
                grids_by_eta[eta].b_values, grids_by_eta[eta].z_values,
            ))
            per_eta_blocks["boundary"][eta] = boundary
        except Exception as exc:  # noqa: BLE001
            record_error("p_residual_blocks", exc)

    # ---- dedicated i_mid teacher call: §12/§13/§14 ----
    mid_results: Dict[Tuple[float, str], Dict[str, Any]] = {}
    try:
        with _checkpoint_economic_config(loaded.economic_config):
            branch_states: List[torch.Tensor] = []
            for eta in eta_values:
                states = grids_by_eta[eta].base_states.clone()
                states[:, 3] = float(reference.i_mid)
                branch_states.extend([states, states])
            bundles = teacher.compute_multi_j_branches(
                branch_states,
                transition.stacked_children(),
                transition.stacked_m_used(),
                branches=["p0", "pi"] * len(eta_values),
                prefix_child_counts=[2 * int(args.n_child_shocks)],
                child_weights=transition.branch_weights,
            )
            for eta_index, eta in enumerate(eta_values):
                mid_results[(eta, "p0")] = bundles[2 * eta_index][2 * int(args.n_child_shocks)]
                mid_results[(eta, "pi")] = bundles[2 * eta_index + 1][2 * int(args.n_child_shocks)]
    except Exception as exc:  # noqa: BLE001
        record_error("teacher_i_mid", exc)

    # ---- §12 P0/PI shape diagnostics (production mid surfaces) ----
    shape_rows: List[Dict[str, Any]] = []
    for eta in eta_values:
        try:
            surf = surfaces_by_eta[eta]
            grid = grids_by_eta[eta]
            for name in ("P0_mid", "PI_mid"):
                for direction, axis, coords in (
                    ("b", 0, grid.b_values), ("z", 1, grid.z_values)
                ):
                    stats = finite_difference_stats(surf[name], axis=axis, coordinates=coords)
                    shape_rows.append({
                        "episode": episode, "eta": eta, "surface": name,
                        "direction": direction, **stats,
                    })
        except Exception as exc:  # noqa: BLE001
            record_error("p_shape", exc)

    # ---- §13 near-default decomposition + §14 Q->P linkage ----
    decomposition_frames: Dict[float, pd.DataFrame] = {}
    linkage_rows: List[Dict[str, Any]] = {}
    try:
        with _checkpoint_economic_config(loaded.economic_config):
            kappa_b = float(loaded.economic_config.KAPPA_B)
            kappa_e = float(loaded.economic_config.KAPPA_E)
            g_value = float(loaded.economic_config.G)
            for eta in eta_values:
                if (eta, "p0") not in mid_results or eta not in phat_teacher_by_eta:
                    continue
                grid = grids_by_eta[eta]
                mesh_b, mesh_z = grid.mesh_b, grid.mesh_z
                phat_teacher = phat_teacher_by_eta[eta]
                p0_comp = teacher_result_at_star_components(mid_results[(eta, "p0")])
                pi_comp = teacher_result_at_star_components(mid_results[(eta, "pi")])
                eta_col = np.full(mesh_b.size, eta)
                b_col = mesh_b.reshape(-1)
                z_col = mesh_z.reshape(-1)
                i_col = np.full(mesh_b.size, float(reference.i_mid))
                x_col = np.full(mesh_b.size, float(reference.x))
                states_cpu = grids_by_eta[eta].base_states.detach().cpu().numpy().astype(np.float64)
                with torch.no_grad():
                    prod = compute_cashflow(
                        torch.as_tensor(states_cpu[:, 4:5]),
                        torch.as_tensor(states_cpu[:, 1:2]),
                        torch.as_tensor(states_cpu[:, 0:1]),
                        float(loaded.economic_config.DELTA),
                        float(loaded.economic_config.TAU),
                    ).reshape(-1).cpu().numpy().astype(np.float64)
                q_current = p0_comp.get("q_current_claim", np.full(mesh_b.size, np.nan))
                q_issue = p0_comp["q_issue_claim_at_star"]
                net_financing_p0 = ((1.0 - kappa_b) * q_issue - q_current) * eta
                net_financing_pi = ((1.0 - kappa_b) * g_value * pi_comp["q_issue_claim_at_star"] - q_current) * eta
                frame = pd.DataFrame({
                    "episode": episode,
                    "eta": eta,
                    "b": b_col,
                    "z": z_col,
                    "bp_star_p0": p0_comp["bp_star"],
                    "bp_star_pi": pi_comp["bp_star"],
                    "production": prod,
                    "investment_cost": i_col,
                    "q_current_claim": q_current,
                    "q_issue_claim_at_star_p0": q_issue,
                    "q_issue_claim_at_star_pi": pi_comp["q_issue_claim_at_star"],
                    "net_debt_financing_p0": net_financing_p0,
                    "net_debt_financing_pi": net_financing_pi,
                    "cashflow_total_p0": p0_comp["cashflow_at_star"],
                    "cashflow_total_pi": pi_comp["cashflow_at_star"],
                    "continuation_at_star_p0": p0_comp["continuation_at_star"],
                    "g_continuation_at_star_pi": pi_comp["continuation_at_star"],
                    "total_target_p0": p0_comp["value_at_star"],
                    "total_target_pi": pi_comp["value_at_star"],
                    "equity_financing_cost_p0_residual": (
                        p0_comp["cashflow_at_star"] - prod - net_financing_p0
                    ),
                })
                for eps in NEAR_DEFAULT_EPS:
                    mask = (np.abs(phat_teacher.reshape(-1)) <= eps).astype(int)
                    frame[f"near_default_eps_{eps:g}"] = mask
                decomposition_frames[eta] = frame
                # §14 linkage summary on eta1 (and eta0 for reference)
                if eta == 1.0:
                    bp_star = p0_comp["bp_star"]
                    low_b_share_full = float(np.nanmean((bp_star <= 0.05).astype(np.float64)))
                    metrics["bp_star_low_b_share_full_grid_eta1"] = low_b_share_full
                    metrics["bp_star_p50_eta1"] = float(np.nanquantile(bp_star, 0.50))
                linkage_rows.append({
                    "episode": episode,
                    "eta": eta,
                    "low_b_bp_star_share_full_grid": float(
                        np.nanmean((p0_comp["bp_star"] <= 0.05).astype(np.float64))
                    ),
                    "q_issue_claim_at_star_mean": float(np.nanmean(q_issue)),
                    "q_current_claim_mean": float(np.nanmean(q_current)),
                    "net_financing_at_star_mean": float(np.nanmean(net_financing_p0)),
                    "continuation_at_star_mean": float(np.nanmean(p0_comp["continuation_at_star"])),
                    "T0_mean": float(np.nanmean(p0_comp["value_at_star"])),
                    "TI_mean": float(np.nanmean(pi_comp["value_at_star"])),
                })
    except Exception as exc:  # noqa: BLE001
        record_error("near_default_decomposition", exc)

    # ---- §14 low-b grid bp_star share (eta1) ----
    try:
        if 1.0 in eta_values:
            with _checkpoint_economic_config(loaded.economic_config):
                states_low = low_grids_by_eta[1.0].base_states.clone()
                states_low[:, 3] = float(reference.i_mid)
                bundles_low = teacher.compute_multi_j_branches(
                    [states_low],
                    transition.stacked_children(),
                    transition.stacked_m_used(),
                    branches=["p0"],
                    prefix_child_counts=[2 * int(args.n_child_shocks)],
                    child_weights=transition.branch_weights,
                )
                result_low = bundles_low[0][2 * int(args.n_child_shocks)]
                bp_star_low = result_low["bp_star"].detach().reshape(-1).cpu().numpy().astype(np.float64)
                metrics["bp_star_low_b_share_low_b_grid_eta1"] = float(
                    np.nanmean((bp_star_low <= 0.05).astype(np.float64))
                )
                metrics["bp_star_p50_low_b_grid_eta1"] = float(np.nanquantile(bp_star_low, 0.50))
    except Exception as exc:  # noqa: BLE001
        record_error("low_b_linkage", exc)

    # ---- §15 BP secondary block via the existing formal evaluator ----
    bp_rows: List[Dict[str, Any]] = []
    bp_output_root = output_root / "bp"
    try:
        selected_counts = select_episode_child_counts(
            episode=episode,
            representative_episodes=args._representative_episodes,
            primary_child_shocks=int(args.n_child_shocks),
            robustness_child_shocks=list(args.robustness_child_shocks),
            robustness_scope=args.robustness_scope,
        )
        with _checkpoint_economic_config(loaded.economic_config):
            for eta in eta_values:
                reference_eta = dataclasses.replace(reference, eta=eta)
                results_by_j, forward_stats = evaluate_bp_consistency_multi_j(
                    model,
                    sdf_fc1_model,
                    grids_by_eta[eta],
                    reference_eta,
                    loaded.hyperparams,
                    loaded.economic_config,
                    output_dir=bp_output_root / f"eta{eta:g}",
                    j_values=selected_counts,
                    primary_j=int(args.n_child_shocks),
                    transition_max=transition,
                    shock_seed=int(args.shock_seed),
                    teacher_margin_tol=float(args.bp_teacher_margin_tol),
                    write_objective_slices=True,
                    bp_eval_max_expanded_states=int(args.bp_eval_max_expanded_states),
                )
                for j_value, bundle in results_by_j.items():
                    summary = bundle["summary"]
                    row = {"episode": episode, "eta": eta, "n_child_shocks": j_value}
                    row.update(summary)
                    row.update({
                        f"forward_{key}": value
                        for key, value in forward_stats.items()
                        if isinstance(value, (int, float, bool))
                    })
                    bp_rows.append(row)
    except Exception as exc:  # noqa: BLE001
        record_error("bp_block", exc)

    # ---- §16 SDF context (M distribution on the frozen transition) ----
    sdf_row: Dict[str, Any] = {"episode": episode}
    try:
        metadata = dict(transition.metadata)
        for key in (
            "m_raw_mean", "m_raw_std", "m_mode", "eta_probability",
            "eta_next_active_share", "shock_seed", "n_child_shocks",
            "shock_bank_sha256", "m_source",
        ):
            if key in metadata:
                sdf_row[f"transition_{key}"] = metadata[key]
        m_raw = transition.stacked_m_raw().detach()
        m_used = transition.stacked_m_used().detach()
        sdf_row["m_raw_p50"] = float(torch.nanquantile(m_raw, 0.50).item())
        sdf_row["m_raw_p90"] = float(torch.nanquantile(m_raw, 0.90).item())
        sdf_row["m_raw_p99"] = float(torch.nanquantile(m_raw, 0.99).item())
        sdf_row["m_used_p50"] = float(torch.nanquantile(m_used, 0.50).item())
        sdf_row["m_used_p99"] = float(torch.nanquantile(m_used, 0.99).item())
        sdf_row["m_nonfinite_share"] = float((~torch.isfinite(m_raw)).to(torch.float64).mean().item())
    except Exception as exc:  # noqa: BLE001
        record_error("sdf_context", exc)

    # ---- §16 FC1 context ----
    fc1_row: Dict[str, Any] = {"episode": episode}
    try:
        fc1_summary, _, _ = evaluate_fc1_checkpoint(
            sdf_fc1_model, args._reference_frame, device=device
        )
        fc1_row.update(fc1_summary)
    except Exception as exc:  # noqa: BLE001
        fc1_row["status"] = f"missing ({type(exc).__name__})"
        record_error("fc1_context", exc)

    # ---- read-only verification ----
    hash_after = {
        "policy_value": model_state_hash(model),
        "sdf_fc1": model_state_hash(sdf_fc1_model),
    }
    metrics["model_state_unchanged"] = bool(hash_before == hash_after)
    metrics["hash_policy_value"] = hash_after["policy_value"]
    metrics["hash_sdf_fc1"] = hash_after["sdf_fc1"]
    metrics["config_warning"] = "; ".join(config_warnings) if config_warnings else ""
    metrics["transition_shock_bank_sha256"] = transition.metadata.get("shock_bank_sha256")
    metrics["transition_prefix_sha256"] = transition.metadata.get("prefix_sha256")
    metrics["transition_n_child_shocks"] = int(transition.metadata.get("n_child_shocks", 0))
    metrics["seconds"] = time.perf_counter() - timing

    artifacts = {
        "config_audit": config_audit,
        "config_warnings": config_warnings,
        "per_eta_blocks": per_eta_blocks,
        "anchor_rows": anchor_rows,
        "shape_rows": shape_rows,
        "decomposition_frames": decomposition_frames,
        "linkage_rows": linkage_rows,
        "bp_rows": bp_rows,
        "sdf_row": sdf_row,
        "fc1_row": fc1_row,
        "surfaces_by_eta": surfaces_by_eta,
        "low_surfaces_by_eta": low_surfaces_by_eta,
        "grids_by_eta": grids_by_eta,
        "low_grids_by_eta": low_grids_by_eta,
        "phat_pred_by_eta": phat_pred_by_eta,
        "phat_teacher_by_eta": phat_teacher_by_eta,
        "transition_metadata": dict(transition.metadata),
        "forward_stats": teacher.forward_stats(),
    }
    return metrics, artifacts, errors


def select_episode_child_counts(
    *,
    episode: int,
    representative_episodes: set[int],
    primary_child_shocks: int,
    robustness_child_shocks: List[int],
    robustness_scope: str,
) -> List[int]:
    if robustness_scope not in {"representative", "all", "none"}:
        raise ValueError(f"Unsupported robustness_scope={robustness_scope!r}")
    include_robustness = (
        robustness_scope == "all"
        or (robustness_scope == "representative" and episode in representative_episodes)
    )
    selected = robustness_child_shocks if include_robustness else []
    return sorted(set([int(primary_child_shocks), *(int(value) for value in selected)]))


# ---------------------------------------------------------------------------
# §17 cross-episode function drift (SECONDARY diagnostic)
# ---------------------------------------------------------------------------


DRIFT_SURFACES = ("Q_claim", "P0", "PI", "P", "bp", "bar_z")


def compute_conditioning_drift(
    surfaces_by_episode: Dict[int, Dict[float, Dict[str, np.ndarray]]],
    eta_values: Sequence[float],
) -> pd.DataFrame:
    rows = []
    episodes = sorted(surfaces_by_episode)
    for eta in eta_values:
        for index in range(1, len(episodes)):
            previous, current = episodes[index - 1], episodes[index]
            prev_surf = surfaces_by_episode[previous].get(eta, {})
            curr_surf = surfaces_by_episode[current].get(eta, {})
            for name in DRIFT_SURFACES:
                if name not in prev_surf or name not in curr_surf:
                    continue
                diff = np.abs(curr_surf[name] - prev_surf[name])
                rows.append({
                    "eta": eta,
                    "surface": name,
                    "episode": current,
                    "vs_episode": previous,
                    "mean_abs_diff": float(np.nanmean(diff)),
                    "p90_abs_diff": float(np.nanquantile(diff, 0.90)),
                    "max_abs_diff": float(np.nanmax(diff)),
                })
        if len(episodes) > 2:
            first = episodes[0]
            for current in episodes[1:]:
                prev_surf = surfaces_by_episode[first].get(eta, {})
                curr_surf = surfaces_by_episode[current].get(eta, {})
                for name in DRIFT_SURFACES:
                    if name not in prev_surf or name not in curr_surf:
                        continue
                    diff = np.abs(curr_surf[name] - prev_surf[name])
                    rows.append({
                        "eta": eta,
                        "surface": f"{name}_vs_first",
                        "episode": current,
                        "vs_episode": first,
                        "mean_abs_diff": float(np.nanmean(diff)),
                        "p90_abs_diff": float(np.nanquantile(diff, 0.90)),
                        "max_abs_diff": float(np.nanmax(diff)),
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §20 representative surface artifacts
# ---------------------------------------------------------------------------


def write_episode_surfaces(
    output_root: Path,
    episode: int,
    surfaces: Dict[str, np.ndarray],
    grid: FrozenFirmGrid,
    phat_pred: Optional[np.ndarray],
    phat_teacher: Optional[np.ndarray],
    eta: float,
) -> None:
    eta_dir = output_root / "representative_surfaces" / f"ep{episode}" / f"eta{eta:g}"
    eta_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "P0_mid", "PI_mid", "Q_claim", "q_unit", "Q_effective", "recovery", "P",
    ]
    for name in names:
        if name not in surfaces:
            continue
        frame = pd.DataFrame({
            "b": grid.mesh_b.reshape(-1),
            "z": grid.mesh_z.reshape(-1),
            f"{name}": surfaces[name].reshape(-1),
        })
        frame.to_csv(eta_dir / f"{name}.csv", index=False)
        plot_heatmap(
            surfaces[name], grid.b_values, grid.z_values,
            eta_dir / f"{name}.png", title=f"ep{episode} eta{eta:g} {name}",
        )
    if phat_pred is not None:
        for name, values in (("Phat_pred", phat_pred), ("default_pred", (phat_pred <= 0.0).astype(float))):
            frame = pd.DataFrame({
                "b": grid.mesh_b.reshape(-1), "z": grid.mesh_z.reshape(-1), name: values.reshape(-1),
            })
            frame.to_csv(eta_dir / f"{name}.csv", index=False)
            plot_heatmap(values, grid.b_values, grid.z_values, eta_dir / f"{name}.png", title=name)
    if phat_teacher is not None:
        for name, values in (("Phat_teacher", phat_teacher), ("default_teacher", (phat_teacher <= 0.0).astype(float))):
            frame = pd.DataFrame({
                "b": grid.mesh_b.reshape(-1), "z": grid.mesh_z.reshape(-1), name: values.reshape(-1),
            })
            frame.to_csv(eta_dir / f"{name}.csv", index=False)
            plot_heatmap(values, grid.b_values, grid.z_values, eta_dir / f"{name}.png", title=name)


def plot_heatmap(
    values: np.ndarray,
    b_values: np.ndarray,
    z_values: np.ndarray,
    path: Path,
    *,
    title: str,
    cmap: str = "viridis",
) -> None:
    finite = np.isfinite(values)
    if not finite.any():
        return
    fig, axis = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    image = axis.imshow(
        values, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        extent=[z_values[0], z_values[-1], b_values[0], b_values[-1]],
    )
    axis.set_xlabel("z")
    axis.set_ylabel("b")
    axis.set_title(title)
    fig.colorbar(image, ax=axis)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only training-conditioning evaluator (Q low-b health, P eta "
            "balance, normalized Bellman residuals, default diagnostics)."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--episodes", default=None, help="Comma/range selection, inclusive, e.g. 0:4")
    parser.add_argument("--training-log", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eta-values", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--b-min", type=float, default=0.0)
    parser.add_argument("--b-max", type=float, default=1.0)
    parser.add_argument("--b-points", type=int, default=101)
    parser.add_argument("--z-min", type=float, default=-4.0)
    parser.add_argument("--z-max", type=float, default=4.0)
    parser.add_argument("--z-points", type=int, default=101)
    parser.add_argument("--i-points", type=int, default=101)
    parser.add_argument("--n-child-shocks", type=int, default=64)
    parser.add_argument("--robustness-child-shocks", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--robustness-scope", choices=("representative", "all", "none"), default="representative")
    parser.add_argument("--shock-seed", type=int, default=12345)
    parser.add_argument("--forward-chunk-size", type=int, default=8192)
    parser.add_argument("--bp-teacher-margin-tol", type=float, default=1e-8)
    parser.add_argument("--bp-eval-max-expanded-states", type=int, default=None)
    parser.add_argument("--low-b-max", type=float, default=0.05)
    parser.add_argument("--low-b-points", type=int, default=101)
    parser.add_argument("--low-b-anchors", type=float, nargs="+", default=[0.005, 0.01, 0.025, 0.05])
    parser.add_argument("--anchor-b-values", type=float, nargs="+", default=list(DEFAULT_ANCHOR_B))
    parser.add_argument("--baseline-run-root", type=Path, default=None)
    parser.add_argument("--max-fc1-rows", type=int, default=200000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    run_root = args.run_root.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_root / "data" / "outputs" / f"training_conditioning_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output dir exists (use --overwrite): {output}")
    for name in ("tables", "figures", "episodes"):
        (output / name).mkdir(parents=True, exist_ok=True)

    # ---- §4 episode discovery (checkpoints_analysis first, then checkpoints) ----
    discovered = _discover_checkpoints(run_root)
    selection = parse_episode_selection(args.episodes)
    episodes = sorted(discovered) if selection is None else sorted(set(selection) & set(discovered))
    missing = sorted(set(selection or []) - set(discovered)) if selection else []
    if not episodes:
        raise SystemExit(f"no combined checkpoints found under {run_root}")
    representative = set(choose_representative_episodes(episodes))
    args._representative_episodes = representative

    # ---- shared reference state (SAME for every episode, §7.1) ----
    firms, _firm_warnings = discover_episode_firm_data(run_root)
    reference_firm, reference_macro = _choose_reference(run_root, firms, args)
    reference_frame, reference = load_reference_state(reference_firm, macro_path=reference_macro)
    args._reference_frame = reference_frame

    # ---- §6 training-log conditioning diagnostics ----
    log_frame, log_status = parse_conditioning_training_log(args.training_log)

    started = time.perf_counter()
    all_rows: List[Dict[str, Any]] = []
    all_errors: List[Dict[str, Any]] = []
    anchor_frames: List[pd.DataFrame] = []
    shape_frames: List[pd.DataFrame] = []
    decomposition_frames: List[pd.DataFrame] = []
    linkage_frames: List[pd.DataFrame] = []
    bp_frames: List[pd.DataFrame] = []
    sdf_rows: List[Dict[str, Any]] = []
    fc1_rows: List[Dict[str, Any]] = []
    surfaces_by_episode: Dict[int, Dict[float, Dict[str, np.ndarray]]] = {}

    for episode in episodes:
        print(f"[episode {episode}] evaluating {discovered[episode].name}", flush=True)
        try:
            row, artifacts, errors = evaluate_episode_conditioning(
                episode=episode,
                checkpoint=discovered[episode],
                reference=reference,
                args=args,
                device=device,
                output_root=output / "episodes" / f"ep{episode}",
                log=lambda message: print(message, flush=True),
            )
        except Exception as exc:  # noqa: BLE001
            all_errors.append({
                "episode": episode, "block": "episode", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            })
            print(f"  episode FAILED: {exc}", flush=True)
            continue
        all_errors.extend(errors)
        row["_blocks"] = artifacts["per_eta_blocks"]
        all_rows.append(row)
        eta_dir = output / "episodes" / f"ep{episode}"
        eta_dir.mkdir(parents=True, exist_ok=True)
        if artifacts["anchor_rows"]:
            anchor_frames.append(pd.DataFrame(artifacts["anchor_rows"]))
        if artifacts["shape_rows"]:
            shape_frames.append(pd.DataFrame(artifacts["shape_rows"]))
        for eta, frame in artifacts["decomposition_frames"].items():
            decomposition_frames.append(frame)
        if artifacts["linkage_rows"]:
            linkage_frames.append(pd.DataFrame(artifacts["linkage_rows"]))
        if artifacts["bp_rows"]:
            bp_frames.append(pd.DataFrame(artifacts["bp_rows"]))
        sdf_rows.append(artifacts["sdf_row"])
        fc1_rows.append(artifacts["fc1_row"])
        surfaces_by_episode[episode] = artifacts["surfaces_by_eta"]
        if episode in representative:
            for eta in args.eta_values:
                eta = float(eta)
                write_episode_surfaces(
                    output, episode,
                    artifacts["surfaces_by_eta"].get(eta, {}),
                    artifacts["grids_by_eta"][eta],
                    artifacts["phat_pred_by_eta"].get(eta),
                    artifacts["phat_teacher_by_eta"].get(eta),
                    eta,
                )
                low_grid = artifacts["low_grids_by_eta"][eta]
                low_surface = artifacts["low_surfaces_by_eta"].get(eta, {})
                if low_surface:
                    _write_low_b_line_plots(
                        output, episode, eta, low_grid, low_surface,
                    )
                    _write_boundary_comparison_plot(
                        output, episode, eta,
                        artifacts["phat_pred_by_eta"].get(eta),
                        artifacts["phat_teacher_by_eta"].get(eta),
                        artifacts["grids_by_eta"][eta],
                    )
        (eta_dir / "config_audit.json").write_text(
            json.dumps(artifacts["config_audit"], indent=1, default=str)
        )
    if not all_rows:
        raise SystemExit("every episode failed; see errors.csv")

    # ---- §6 merge ----
    metrics_frame = pd.DataFrame(
        [flatten_conditioning_row(row) for row in all_rows]
    )
    metrics_frame = add_eta_ratio_columns(metrics_frame)
    if not log_frame.empty:
        metrics_frame = metrics_frame.merge(log_frame, on="episode", how="left")

    # ---- §17 secondary drift ----
    drift_frame = compute_conditioning_drift(surfaces_by_episode, [float(v) for v in args.eta_values])

    # ---- tables (§21/§22) ----
    tables = output / "tables"
    metrics_frame.to_csv(tables / "conditioning_metrics_by_episode.csv", index=False)
    _write_per_eta_block_table(tables / "q_low_b_metrics_by_episode_eta.csv", all_rows, "q_low_b")
    if anchor_frames:
        pd.concat(anchor_frames, ignore_index=True).to_csv(tables / "q_anchor_values.csv", index=False)
    _write_p_residual_table(tables / "p_residual_metrics_by_episode_eta.csv", all_rows)
    _write_zbin_table(tables / "p_zbin_residuals.csv", all_rows)
    _write_per_eta_block_table(tables / "default_pred_teacher_metrics.csv", all_rows, "confusion")
    _write_per_eta_block_table(tables / "default_boundary_metrics.csv", all_rows, "boundary")
    if bp_frames:
        pd.concat(bp_frames, ignore_index=True).to_csv(tables / "bp_metrics.csv", index=False)
    pd.DataFrame(sdf_rows).to_csv(tables / "sdf_metrics.csv", index=False)
    pd.DataFrame(fc1_rows).to_csv(tables / "fc1_metrics.csv", index=False)
    log_frame.to_csv(tables / "training_conditioning_log_metrics.csv", index=False)
    drift_frame.to_csv(tables / "function_drift.csv", index=False)
    if shape_frames:
        pd.concat(shape_frames, ignore_index=True).to_csv(tables / "p_shape_diagnostics.csv", index=False)
    if decomposition_frames:
        pd.concat(decomposition_frames, ignore_index=True).to_csv(
            tables / "near_default_target_decomposition_all.csv", index=False
        )
    if linkage_frames:
        pd.concat(linkage_frames, ignore_index=True).to_csv(tables / "q_p_linkage.csv", index=False)

    # ---- errors ----
    pd.DataFrame(all_errors, columns=["episode", "block", "error", "traceback"]).to_csv(
        output / "errors.csv", index=False
    )

    # ---- optional §18 baseline comparison ----
    if args.baseline_run_root is not None:
        try:
            _run_baseline_comparison(args, output, reference, reference_frame, device, log_frame)
        except Exception as exc:  # noqa: BLE001
            all_errors.append({
                "episode": -1, "block": "baseline_comparison",
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=6),
            })

    # ---- figures ----
    _write_dashboards(output, metrics_frame, [float(v) for v in args.eta_values])

    # ---- metadata + summary ----
    git_head = ""
    try:
        import subprocess
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        git_head = "unknown"
    metadata = {
        "evaluator": "evaluate_training_conditioning.py",
        "purpose": "training health / function shape / teacher-fit diagnostics (not a convergence verdict)",
        "git_head": git_head,
        "run_root": str(run_root),
        "episodes": episodes,
        "missing_checkpoints": missing,
        "representative_episodes": sorted(representative),
        "reference_firm_data": str(reference_firm),
        "reference_macro_data": str(reference_macro) if reference_macro else None,
        "reference_state": reference.to_dict(),
        "eta_values": [float(v) for v in args.eta_values],
        "grid": {
            "b": [args.b_min, args.b_max, args.b_points],
            "z": [args.z_min, args.z_max, args.z_points],
            "i_points": args.i_points,
            "i_threshold": float(reference.i_mid),
            "low_b": {"max": args.low_b_max, "points": args.low_b_points, "anchors": list(args.low_b_anchors)},
        },
        "shock": {
            "seed": args.shock_seed,
            "n_child_shocks": args.n_child_shocks,
            "robustness_child_shocks": list(args.robustness_child_shocks),
            "robustness_scope": args.robustness_scope,
        },
        "training_log": log_status,
        "model_state_unchanged_all": bool(metrics_frame["model_state_unchanged"].all()) if "model_state_unchanged" in metrics_frame else False,
        "config_warnings": [
            {"episode": int(row["episode"]), "warning": row.get("config_warning", "")}
            for _, row in metrics_frame.iterrows() if row.get("config_warning")
        ],
        "errors": all_errors,
        "total_seconds": time.perf_counter() - started,
    }
    write_json(output / "metadata.json", metadata)

    summary_md, summary_json = build_training_health_summary(metrics_frame, log_frame, drift_frame)
    (output / "training_health_summary.md").write_text(summary_md, encoding="utf-8")
    write_json(output / "training_health_summary.json", summary_json)
    (output / "README.md").write_text(_readme_text(output), encoding="utf-8")

    print(f"\nconditioning evaluation complete -> {output}")
    print(f"episodes: {episodes}; errors: {len(all_errors)}; "
          f"model_state_unchanged_all={metadata['model_state_unchanged_all']}")


# ---- §22 metrics flattening ----------------------------------------------

# Block keys renamed to their §22 canonical stems when flattened into the
# per-episode metrics frame. Raw keys are kept as well (the summary rules and
# dashboards cite them).
CONDITIONING_KEY_RENAMES: Dict[str, str] = {
    "z_default_median": "pred_zdefault_median",
    "teacher_z_default_median": "teacher_zdefault_median",
    "boundary_z_abs_gap_p90": "boundary_gap_p90",
}

# Scalar columns kept under BOTH the raw stem and the d-alias stem so that
# §22/§23/§18 references (qclaim_low_b_d1_p95 / qclaim_low_b_d2_p95 /
# qclaim_full_d2_p95) resolve without duplicating the numbers.
_D_ALIAS_RENAMES: Tuple[Tuple[str, str], ...] = (
    ("_d1_d_abs_", "_d1_"),
    ("_d1_d2_abs_", "_d2_"),
)


def _eta_tag(eta: float) -> str:
    value = float(eta)
    return f"eta{int(value)}" if value.is_integer() else f"eta{value:g}"


def _is_scalar(value: Any) -> bool:
    return not isinstance(value, (dict, list, tuple, np.ndarray, pd.DataFrame, pd.Series))


def _emit_column(flat: Dict[str, Any], stem: str, eta: float, value: Any) -> None:
    flat[f"{stem}_{str(float(eta))}"] = value
    if float(eta).is_integer():
        flat[f"{stem}_{_eta_tag(eta)}"] = value


def flatten_conditioning_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten per-eta blocks into eta-suffixed scalar columns (§22)."""
    flat = {key: value for key, value in row.items() if key != "_blocks"}
    for block, per_eta in row.get("_blocks", {}).items():
        for eta, values in per_eta.items():
            if not isinstance(values, dict):
                continue  # zbin frames live in their dedicated table
            for key, value in values.items():
                if not _is_scalar(value):
                    continue
                _emit_column(flat, key, eta, value)
                for stem in _d_alias_stems(key):
                    _emit_column(flat, stem, eta, value)
                renamed = CONDITIONING_KEY_RENAMES.get(key)
                if renamed is not None:
                    _emit_column(flat, renamed, eta, value)
    return flat


def _d_alias_stems(key: str) -> Tuple[str, ...]:
    stems = []
    for old, new in _D_ALIAS_RENAMES:
        if key.startswith("qclaim_") and old in key:
            stems.append(key.replace(old, new))
    return tuple(stems)


def add_eta_ratio_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """§22 eta1/eta0 normalized-residual ratios (NaN when undefined)."""
    for module in ("P0", "PI"):
        eta1, eta0 = f"{module}_norm_mean_eta1", f"{module}_norm_mean_eta0"
        ratio = f"{module}_norm_eta1_eta0_ratio"
        if eta0 not in frame.columns:
            frame[ratio] = float("nan")
            continue
        denominator = pd.to_numeric(frame[eta0], errors="coerce").to_numpy(dtype=np.float64)
        if eta1 in frame.columns:
            numerator = pd.to_numeric(frame[eta1], errors="coerce").to_numpy(dtype=np.float64)
        else:
            numerator = np.full(frame.shape[0], np.nan)
        valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0.0)
        frame[ratio] = np.where(valid, numerator / np.where(denominator == 0.0, np.nan, denominator), np.nan)
    return frame


# ---- table writers -------------------------------------------------------


def _write_per_eta_block_table(path: Path, rows: List[Dict[str, Any]], block: str) -> None:
    records = []
    for row in rows:
        for eta, values in row.get("_blocks", {}).get(block, {}).items():
            if not isinstance(values, dict):
                continue
            records.append({"episode": row["episode"], "eta": eta, **values})
    pd.DataFrame(records).to_csv(path, index=False) if records else pd.DataFrame().to_csv(path, index=False)


def _write_p_residual_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    records = []
    for row in rows:
        for eta, values in row.get("_blocks", {}).get("p_residual", {}).items():
            if isinstance(values, dict):
                records.append({"episode": row["episode"], "eta": eta, **values})
    pd.DataFrame(records).to_csv(path, index=False) if records else pd.DataFrame().to_csv(path, index=False)


def _write_zbin_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    frames = []
    for row in rows:
        for eta, frame in row.get("_blocks", {}).get("zbin", {}).items():
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frames.append(frame)
    (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()).to_csv(path, index=False)


# ---- low-b line plots ----------------------------------------------------


def _write_low_b_line_plots(
    output: Path,
    episode: int,
    eta: float,
    grid: FrozenFirmGrid,
    surfaces: Dict[str, np.ndarray],
) -> None:
    eta_dir = output / "representative_surfaces" / f"ep{episode}"
    eta_dir.mkdir(parents=True, exist_ok=True)
    z_indices = [0, len(grid.z_values) // 4, len(grid.z_values) // 2, 3 * len(grid.z_values) // 4, len(grid.z_values) - 1]
    for name in ("Q_claim", "q_unit"):
        if name not in surfaces:
            continue
        fig, axis = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
        for z_index in z_indices:
            axis.plot(grid.b_values, surfaces[name][:, z_index], label=f"z={grid.z_values[z_index]:g}")
        axis.set_xlabel("b (low-b grid)")
        axis.set_ylabel(name)
        axis.set_title(f"ep{episode} eta{eta:g} low-b {name}")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.25)
        fig.savefig(eta_dir / f"low_b_{name}_eta{eta:g}.png", dpi=150)
        plt.close(fig)


def _write_boundary_comparison_plot(
    output: Path,
    episode: int,
    eta: float,
    phat_pred: Optional[np.ndarray],
    phat_teacher: Optional[np.ndarray],
    grid: FrozenFirmGrid,
) -> None:
    if phat_pred is None or phat_teacher is None:
        return
    eta_dir = output / "representative_surfaces" / f"ep{episode}"
    eta_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    for name, surface, style in (("pred", phat_pred, "-"), ("teacher", phat_teacher, "--")):
        z_star = []
        for row in range(surface.shape[0]):
            crossings = zero_crossings(grid.z_values, surface[row])
            z_star.append(crossings[0] if len(crossings) == 1 else np.nan)
        axis.plot(grid.b_values, z_star, style, label=f"{name} z_default")
    axis.set_xlabel("b")
    axis.set_ylabel("z at Phat=0 crossing")
    axis.set_title(f"ep{episode} eta{eta:g} default boundary pred vs teacher")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    fig.savefig(eta_dir / f"default_boundary_pred_vs_teacher_eta{eta:g}.png", dpi=150)
    plt.close(fig)


# ---- dashboards ----------------------------------------------------------


def _write_dashboards(output: Path, metrics_frame: pd.DataFrame, eta_values: Sequence[float]) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def plot_panels(frame: pd.DataFrame, metrics: Sequence[str], path: Path, title: str) -> None:
        available = [name for name in metrics if name in frame.columns]
        if not available:
            return
        ncols = min(3, len(available))
        nrows = int(np.ceil(len(available) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
        for axis, metric in zip(axes.reshape(-1), available):
            axis.plot(frame["episode"], pd.to_numeric(frame[metric], errors="coerce"), marker="o")
            axis.set_title(metric, fontsize=8)
            axis.set_xlabel("episode")
            axis.grid(alpha=0.25)
        for axis in axes.reshape(-1)[len(available):]:
            axis.axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    plot_panels(
        metrics_frame,
        ("q_unit_low_b_p95_1.0", "q_unit_low_b_max_1.0", "Q_claim_low_b_p95_1.0",
         "qclaim_low_b_d1_p95_1.0", "qclaim_low_b_d2_p95_1.0", "Q_claim_b0_abs_max_1.0"),
        figures / "q_low_b_dashboard.png",
        "Q low-b conditioning health (eta=1)",
    )
    plot_panels(
        metrics_frame,
        ("P0_norm_mean_0.0", "P0_norm_mean_1.0", "PI_norm_mean_0.0", "PI_norm_mean_1.0",
         "P0_norm_eta1_eta0_ratio", "PI_norm_eta1_eta0_ratio"),
        figures / "p_eta_residual_dashboard.png",
        "P normalized Bellman residuals by eta",
    )
    plot_panels(
        metrics_frame,
        ("pred_default_share_0.0", "pred_default_share_1.0",
         "teacher_default_share_0.0", "teacher_default_share_1.0",
         "false_survival_share_0.0", "false_survival_share_1.0"),
        figures / "default_pred_teacher_dashboard.png",
        "Predicted vs teacher default region",
    )
    plot_panels(
        metrics_frame,
        ("p_train_eta1_share_after", "p_validation_eta1_share_after",
         "claim_coverage_fraction_anchors_occupied", "claim_coverage_low_b_sample_share",
         "boundary_z_abs_gap_p90_1.0", "agreement_share_1.0"),
        figures / "conditioning_dashboard.png",
        "Training conditioning overview",
    )


# ---- §18 baseline comparison ---------------------------------------------


BASELINE_COMPARISON_METRICS = (
    "q_unit_low_b_p95_{eta}", "q_unit_low_b_max_{eta}",
    "qclaim_low_b_d1_p95_{eta}", "qclaim_low_b_d2_p95_{eta}",
    "P0_norm_mean_{eta}", "PI_norm_mean_{eta}",
    "P0_norm_eta1_eta0_ratio", "PI_norm_eta1_eta0_ratio",
    "false_survival_share_{eta}", "false_default_share_{eta}",
    "pred_default_share_{eta}", "teacher_default_share_{eta}",
    "pred_zdefault_median_{eta}", "teacher_zdefault_median_{eta}",
    "boundary_z_abs_gap_p90_{eta}",
)


def _run_baseline_comparison(
    args: argparse.Namespace,
    output: Path,
    reference: ReferenceFirmState,
    reference_frame: pd.DataFrame,
    device: torch.device,
    log_frame: pd.DataFrame,
) -> None:
    baseline_root = args.baseline_run_root.resolve()
    baseline_checkpoints = _discover_checkpoints(baseline_root)
    target_checkpoints = _discover_checkpoints(args.run_root.resolve())
    matched = sorted(set(baseline_checkpoints) & set(target_checkpoints))
    if not matched:
        raise ValueError("no matched episodes between target and baseline run roots")
    baseline_output = output / "baseline"
    baseline_output.mkdir(parents=True, exist_ok=True)
    baseline_rows: List[Dict[str, Any]] = []
    for episode in matched:
        row, artifacts, errors = evaluate_episode_conditioning(
            episode=episode,
            checkpoint=baseline_checkpoints[episode],
            reference=reference,
            args=args,
            device=device,
            output_root=baseline_output / f"ep{episode}",
            log=lambda message: print(message, flush=True),
        )
        row["_blocks"] = artifacts["per_eta_blocks"]
        baseline_rows.append(row)
    baseline_frame = pd.DataFrame(
        [flatten_conditioning_row(row) for row in baseline_rows]
    )
    baseline_frame = add_eta_ratio_columns(baseline_frame)
    if not log_frame.empty:
        baseline_frame = baseline_frame.merge(log_frame, on="episode", how="left")
    target_frame = pd.read_csv(output / "tables" / "conditioning_metrics_by_episode.csv")
    records = []
    for episode in matched:
        target_row = target_frame.loc[target_frame["episode"] == episode]
        baseline_row = baseline_frame.loc[baseline_frame["episode"] == episode]
        if target_row.empty or baseline_row.empty:
            continue
        for template in BASELINE_COMPARISON_METRICS:
            for eta in args.eta_values:
                name = template.format(eta=float(eta))
                if name in ("P0_norm_eta1_eta0_ratio", "PI_norm_eta1_eta0_ratio"):
                    pass
                if name not in target_row.columns or name not in baseline_row.columns:
                    continue
                records.append({
                    "episode": episode,
                    "metric": name,
                    "target": float(target_row.iloc[0][name]),
                    "baseline": float(baseline_row.iloc[0][name]),
                    "delta": float(target_row.iloc[0][name]) - float(baseline_row.iloc[0][name]),
                })
    pd.DataFrame(records).to_csv(output / "tables" / "baseline_comparison.csv", index=False)


# ---- §23 training-health summary -----------------------------------------


def _column_or_nan(frame: pd.DataFrame, name: str, episode: Optional[int] = None) -> float:
    if frame.empty or name not in frame.columns:
        return float("nan")
    series = pd.to_numeric(frame[name], errors="coerce")
    if episode is not None and "episode" in frame.columns:
        series = series.loc[pd.to_numeric(frame["episode"], errors="coerce") == episode]
    values = series.dropna()
    return float(values.iloc[-1]) if len(values) else float("nan")


def build_training_health_summary(
    metrics_frame: pd.DataFrame,
    log_frame: pd.DataFrame,
    drift_frame: pd.DataFrame,
) -> Tuple[str, Dict[str, Any]]:
    """§23 rule-based summary; every judgment cites concrete metrics."""
    last_episode = int(metrics_frame["episode"].max()) if not metrics_frame.empty else None
    columns = list(metrics_frame.columns)

    def m(name: str) -> float:
        return _column_or_nan(metrics_frame, name, last_episode)

    def fmt(value: float, digits: int = 4) -> str:
        return f"{value:.{digits}f}" if np.isfinite(value) else "n/a"

    eta1_columns = [name for name in columns if name.endswith("_1.0")]

    def eta1(name: str) -> str:
        return f"{name}_1.0" if f"{name}_1.0" in columns else (name if name in columns else f"{name}__missing")

    lines: List[str] = []
    json_summary: Dict[str, Any] = {"questions": {}, "warnings": []}

    # ---------------- A. Q low-b repair ----------------
    anchors_occupied = m("claim_coverage_fraction_anchors_occupied")
    low_b_share = m("claim_coverage_low_b_sample_share")
    d2_low = m("qclaim_low_b_d2_p95_1.0")
    d1_low = m("qclaim_low_b_d1_p95_1.0")
    d2_full = m("qclaim_full_d2_p95_1.0")
    q_unit_p95 = m("q_unit_low_b_p95_1.0")
    q_unit_max = m("q_unit_low_b_max_1.0")
    q_claim_b0 = m("Q_claim_b0_abs_max_1.0")
    ridge_ratio = d2_low / d2_full if np.isfinite(d2_low) and np.isfinite(d2_full) and d2_full > 0 else float("nan")
    lines.append("## A. Q low-b repair\n")
    lines.append(f"- low-b anchor coverage (training log, last episode): `claim_coverage_fraction_anchors_occupied` = {fmt(anchors_occupied)}, `claim_coverage_low_b_sample_share` = {fmt(low_b_share)}.")
    lines.append(f"- Q_claim low-b curvature: `qclaim_low_b_d2_p95` = {fmt(d2_low)} vs full-grid `qclaim_full_d2_p95` = {fmt(d2_full)} (ratio {fmt(ridge_ratio, 2)}).")
    lines.append(f"- q_unit tail on low-b grid: p95 = {fmt(q_unit_p95)}, max = {fmt(q_unit_max)} (diagnostic only; q_unit > 1 is not an error).")
    lines.append(f"- structural zero: `Q_claim_b0_abs_max` = {fmt(q_claim_b0)} (spec target: ~0).")
    if np.isfinite(ridge_ratio):
        ridge_flag = ridge_ratio > 10.0
        lines.append(f"- judgment: low-b artificial ridge {'PRESENT' if ridge_flag else 'NOT PRESENT'} by the d2-ratio rule (>10x full-grid).")
        json_summary["questions"]["A_q_low_b_repair"] = {
            "anchors_occupied": anchors_occupied,
            "low_b_sample_share": low_b_share,
            "qclaim_low_b_d2_p95": d2_low,
            "qclaim_full_d2_p95": d2_full,
            "ridge_ratio": ridge_ratio,
            "ridge_present": bool(ridge_flag),
            "q_unit_low_b_p95": q_unit_p95,
            "q_unit_low_b_max": q_unit_max,
            "Q_claim_b0_abs_max": q_claim_b0,
        }
        if ridge_flag:
            json_summary["warnings"].append("Q_claim low-b artificial ridge / extreme local curvature (qclaim_low_b_d2_p95 ratio)")

    # ---------------- B. P eta balance ----------------
    p_train_after = m("p_train_eta1_share_after")
    p_val_after = m("p_validation_eta1_share_after")
    q_eta_share = m("q_batches_eta1_share")
    bp_eta_share = m("bp_batches_eta1_share")
    p0_norm_ratio = m("P0_norm_eta1_eta0_ratio")
    pi_norm_ratio = m("PI_norm_eta1_eta0_ratio")
    balance_ok = (
        np.isfinite(p_train_after) and np.isfinite(p_val_after)
        and 0.45 <= p_train_after <= 0.55 and 0.45 <= p_val_after <= 0.55
    )
    lines.append("\n## B. P eta balance\n")
    lines.append(f"- P train eta1 share after = {fmt(p_train_after)}, P validation eta1 share after = {fmt(p_val_after)} (target 0.50; balanced = {balance_ok}).")
    lines.append(f"- Q natural share `q_batches_eta1_share` = {fmt(q_eta_share)}, BP natural share `bp_batches_eta1_share` = {fmt(bp_eta_share)} (missing means the field was absent from the training log).")
    lines.append(f"- normalized residual eta1/eta0 ratio: P0 = {fmt(p0_norm_ratio, 3)}, PI = {fmt(pi_norm_ratio, 3)}.")
    json_summary["questions"]["B_p_eta_balance"] = {
        "p_train_eta1_share_after": p_train_after,
        "p_validation_eta1_share_after": p_val_after,
        "balanced_50pct": bool(balance_ok),
        "q_batches_eta1_share": q_eta_share,
        "bp_batches_eta1_share": bp_eta_share,
        "P0_norm_eta1_eta0_ratio": p0_norm_ratio,
        "PI_norm_eta1_eta0_ratio": pi_norm_ratio,
    }

    # ---------------- C. normalized loss z-weighting ----------------
    zbin_path_note = "tables/p_zbin_residuals.csv"
    lines.append("\n## C. P normalized loss vs z-weighting\n")
    lines.append(f"- per-z-bin physical vs normalized residuals are in `{zbin_path_note}` (columns `physical_abs_residual_mean`, `normalized_abs_residual_mean`, `value_scale_mean`).")
    json_summary["questions"]["C_normalized_loss"] = {
        "table": zbin_path_note,
        "interpretation": "normalized residuals should stay comparable across z-bins while physical residuals grow with value_scale",
    }

    # ---------------- D. default region ----------------
    agree = m("agreement_share_1.0")
    false_survival_1 = m("false_survival_share_1.0")
    false_default_1 = m("false_default_share_1.0")
    pred_default_0 = m("pred_default_share_0.0")
    pred_default_1 = m("pred_default_share_1.0")
    teacher_default_0 = m("teacher_default_share_0.0")
    teacher_default_1 = m("teacher_default_share_1.0")
    lines.append("\n## D. default region\n")
    lines.append(f"- eta1 agreement = {fmt(agree)}, false_survival = {fmt(false_survival_1)}, false_default = {fmt(false_default_1)}.")
    lines.append(f"- pred default share eta0 = {fmt(pred_default_0)}, eta1 = {fmt(pred_default_1)}; teacher default share eta0 = {fmt(teacher_default_0)}, eta1 = {fmt(teacher_default_1)}.")
    if np.isfinite(false_survival_1) and false_survival_1 > 0.15:
        lines.append("- judgment: high false-survival share — the network misses teacher default states (network fitting issue, not a small teacher default region).")
        json_summary["warnings"].append("predicted-vs-teacher false survival share high")
    json_summary["questions"]["D_default_region"] = {
        "agreement_share_eta1": agree,
        "false_survival_share_eta1": false_survival_1,
        "false_default_share_eta1": false_default_1,
        "pred_default_share": {"eta0": pred_default_0, "eta1": pred_default_1},
        "teacher_default_share": {"eta0": teacher_default_0, "eta1": teacher_default_1},
    }

    # ---------------- E. P surface shape ----------------
    boundary_gt1 = m("crossing_count_gt1_share_1.0")
    boundary_gap_p90 = m("boundary_z_abs_gap_p90_1.0")
    lines.append("\n## E. P surface shape\n")
    lines.append(f"- eta1 multi-crossing boundary share = {fmt(boundary_gt1)}; pred-vs-teacher boundary gap p90 = {fmt(boundary_gap_p90)}.")
    lines.append("- P0/PI ridge/oscillation quantiles: see `tables/p_shape_diagnostics.csv` (`d2_abs_p95`, `d2_abs_max` per surface/direction).")
    if np.isfinite(boundary_gt1) and boundary_gt1 > 0.02:
        json_summary["warnings"].append("default boundary multiple crossings / checkerboard")
    json_summary["questions"]["E_p_surface"] = {
        "crossing_count_gt1_share_eta1": boundary_gt1,
        "boundary_z_abs_gap_p90_eta1": boundary_gap_p90,
        "shape_table": "tables/p_shape_diagnostics.csv",
    }

    # ---------------- F. issue ranking ----------------
    lines.append("\n## F. training issue ranking\n")
    ranking: Dict[str, List[str]] = {"primary": [], "secondary": [], "no_evidence": []}
    if np.isfinite(false_survival_1) and false_survival_1 > 0.15:
        ranking["primary"].append("P teacher-fit: false_survival_share = " + fmt(false_survival_1))
    if np.isfinite(p0_norm_ratio) and p0_norm_ratio > 1.5:
        ranking["primary"].append("P eta1 normalized residual gap: P0_norm_eta1_eta0_ratio = " + fmt(p0_norm_ratio, 2))
    if np.isfinite(ridge_ratio) and ridge_ratio > 10.0:
        ranking["primary"].append("Q low-b artificial ridge: d2 ratio = " + fmt(ridge_ratio, 1))
    if np.isfinite(false_survival_1) and 0.05 < false_survival_1 <= 0.15:
        ranking["secondary"].append("P false-survival share moderate: " + fmt(false_survival_1))
    if np.isfinite(boundary_gt1) and 0.0 < boundary_gt1 <= 0.02:
        ranking["secondary"].append("boundary multi-crossing share small but nonzero: " + fmt(boundary_gt1))
    if not np.isfinite(p_train_after) or not np.isfinite(p_val_after):
        ranking["secondary"].append("P eta-balance fields missing from the training log")
    if not ranking["primary"] and not ranking["secondary"]:
        ranking["no_evidence"].append("no metric exceeded a warning rule in this run")
    lines.append(f"- Primary: {'; '.join(ranking['primary']) if ranking['primary'] else 'none above rule threshold'}")
    lines.append(f"- Secondary: {'; '.join(ranking['secondary']) if ranking['secondary'] else 'none'}")
    lines.append(f"- No-evidence: {'; '.join(ranking['no_evidence']) if ranking['no_evidence'] else 'none'}")
    json_summary["questions"]["F_issue_ranking"] = ranking

    # ---------------- §24 non-failure notes ----------------
    lines.append("\n## Notes (not failure conditions)\n")
    lines.append("- q_unit > 1 on the low-b grid, Q_claim != recovery, eta1 default share < eta0, and residual cross-episode drift are diagnostics, not failure conditions.")
    if not drift_frame.empty:
        worst = drift_frame.loc[drift_frame["mean_abs_diff"].idxmax()]
        lines.append(
            f"- secondary outer-loop drift diagnostic: largest mean|Δsurface| = {fmt(float(worst['mean_abs_diff']))} "
            f"({worst['surface']}, eta={worst['eta']:g}, ep{int(worst['episode'])} vs ep{int(worst['vs_episode'])}); see `tables/function_drift.csv`."
        )
    lines.append("- primary objective of this report is training health; drift must not be read as a convergence verdict.")

    header = [
        "# Training-conditioning health summary",
        "",
        f"- episodes evaluated: {', '.join(str(int(v)) for v in metrics_frame['episode'])}" if not metrics_frame.empty else "",
        "- purpose: c766537 conditioning fixes (Q low-b coverage / P eta balance / normalized Bellman loss) health check — NOT a convergence verdict",
        "",
    ]
    return "\n".join([line for line in header if line] + lines), json_summary


# ---- README ---------------------------------------------------------------


def _readme_text(output: Path) -> str:
    return (
        "# Training-conditioning evaluator output\n\n"
        "Read-only training-health diagnostics for the Hybrid-Q conditioning run.\n\n"
        "- `tables/` metric CSVs (see metadata.json for the full column dictionary)\n"
        "- `figures/` dashboards\n"
        "- `episodes/epN/` per-episode artifacts (bp objective slices, config audit)\n"
        "- `representative_surfaces/` heatmaps and low-b line plots per representative episode\n"
        "- `training_health_summary.md` / `.json` rule-based A-F summary\n"
        "- `errors.csv` per-block failures (the run continues past block errors)\n\n"
        "This evaluator never modifies the run root; model state hashes are recorded\n"
        "before/after every episode in `tables/conditioning_metrics_by_episode.csv`.\n"
    )


# ---- discovery re-use ------------------------------------------------------


def _discover_checkpoints(run_root: Path) -> Dict[int, Path]:
    import re as _re
    pattern = _re.compile(r"^ep(?P<episode>\d+)_combined\.pt$")
    found: Dict[int, Path] = {}
    for directory in (run_root / "checkpoints_analysis", run_root / "checkpoints"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("ep*_combined.pt")):
            match = pattern.match(path.name)
            if match:
                found.setdefault(int(match.group("episode")), path.resolve())
    return dict(sorted(found.items()))


if __name__ == "__main__":
    main()
