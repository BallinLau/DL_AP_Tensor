"""Read-only BP policy fit diagnostics against the production grid teacher.

This evaluator intentionally does not construct an optimizer, call backward,
or modify training/checkpoint state.  It evaluates a small fixed state bank and
uses ``BPGridTeacher.compute_multi_j_branches`` for every economic objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
)
from evaluation.full_run_diagnostics import model_state_hash  # noqa: E402
from evaluation.grids import (  # noqa: E402
    ReferenceFirmState,
    select_parent_rows,
    load_reference_state,
)
from experiments.evaluate_p_teacher_drift import (  # noqa: E402
    _losses,
    atomic_write_csv,
    atomic_write_json,
    build_focus_states,
    choose_reference_artifacts,
)
from training.bp_grid_teacher import BPGridTeacher  # noqa: E402


DEFAULT_B_VALUES = (0.01, 0.05, 0.20, 0.50, 0.80)
DEFAULT_Z_VALUES = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)
TINY_B_VALUES = (0.05, 0.20, 0.50)
TINY_Z_VALUES = (0.0, 2.0, 4.0)
PRESENTATION_B_VALUES = (0.01, 0.05, 0.20, 0.50, 0.80)
PRESENTATION_Z_VALUES = (-2.0, 0.0, 1.0, 2.0, 3.0, 4.0)
MASK_TYPES = ("raw_refi", "survival_refi", "primary")
BRANCH_ORDER = ("p0", "pi_low", "pi_mid", "pi_high")
EPS = 1e-8
Z_BIN_EDGES = (-4.0, -2.0, 0.0, 1.0, 2.0, 3.0, 4.0000001)
B_BIN_EDGES = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0000001)
GAP_THRESHOLDS = (0.05, 0.10, 0.20, 0.40)
TINY_EXPECTED_MAE = 0.0062404


@dataclass(frozen=True)
class StateBank:
    name: str
    reference: ReferenceFirmState
    base_states: torch.Tensor
    hatc_cal: torch.Tensor
    lnk_cal: torch.Tensor
    context: pd.DataFrame
    metadata: dict[str, Any]
    grid_shape: tuple[int, int] | None = None
    b_values: np.ndarray | None = None
    z_values: np.ndarray | None = None
    transition_eta_override: float | None = None


@contextmanager
def timed_block(name: str, timings: dict[str, float], *, heartbeat_seconds: float = 30.0):
    start = time.monotonic()
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(heartbeat_seconds):
            elapsed = time.monotonic() - start
            print(f"[timing] {name}: still running after {elapsed:.1f}s", flush=True)

    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    print(f"[timing] {name}: started", flush=True)
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=1.0)
        elapsed = time.monotonic() - start
        timings[name] = elapsed
        print(f"[timing] {name}: finished in {elapsed:.2f}s", flush=True)


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _quantile(values: Iterable[float], q: float) -> float:
    array = _finite(values)
    return float(np.quantile(array, q)) if array.size else float("nan")


def pearson_correlation(x: Iterable[float], y: Iterable[float]) -> float:
    left = np.asarray(x, dtype=np.float64).reshape(-1)
    right = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(left) & np.isfinite(right)
    left, right = left[keep], right[keep]
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def spearman_correlation(x: Iterable[float], y: Iterable[float]) -> float:
    left = np.asarray(x, dtype=np.float64).reshape(-1)
    right = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(left) & np.isfinite(right)
    left, right = left[keep], right[keep]
    if left.size < 2:
        return float("nan")
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=np.float64)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=np.float64)
    return pearson_correlation(left_rank, right_rank)


def build_masks(
    refi_active: Iterable[bool],
    phat: Iterable[float],
    top2_margin: Iterable[float],
    *,
    teacher_margin_tol: float,
) -> dict[str, np.ndarray]:
    refi = np.asarray(refi_active, dtype=bool).reshape(-1)
    phat_array = np.asarray(phat, dtype=np.float64).reshape(-1)
    margin = np.asarray(top2_margin, dtype=np.float64).reshape(-1)
    survival = np.isfinite(phat_array) & (phat_array > 0.0)
    identified = np.isfinite(margin) & (margin > float(teacher_margin_tol))
    return {
        "raw_refi": refi,
        "survival_refi": refi & survival,
        "primary": refi & survival & identified,
        "survival": survival,
        "numerical_identified": identified,
    }


def normalized_regrets(
    regret: Iterable[float], value_star: Iterable[float], value_scale: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    regret_array = np.asarray(regret, dtype=np.float64)
    value_array = np.asarray(value_star, dtype=np.float64)
    scale_array = np.asarray(value_scale, dtype=np.float64)
    relative = regret_array / (np.abs(value_array) + EPS)
    scaled = np.full_like(regret_array, np.nan)
    valid_scale = np.isfinite(scale_array) & (scale_array > 0.0)
    scaled[valid_scale] = regret_array[valid_scale] / scale_array[valid_scale]
    return relative, scaled


def relative_margins(
    top2_margin: Iterable[float], value_star: Iterable[float]
) -> np.ndarray:
    margin = np.asarray(top2_margin, dtype=np.float64)
    value = np.asarray(value_star, dtype=np.float64)
    return margin / (np.abs(value) + EPS)


def separated_margin(
    coarse_bp_grid: np.ndarray,
    coarse_value_grid: np.ndarray,
    bp_star: np.ndarray,
    value_star: np.ndarray,
    *,
    delta: float,
) -> np.ndarray:
    bp_grid = np.asarray(coarse_bp_grid, dtype=np.float64)
    value_grid = np.asarray(coarse_value_grid, dtype=np.float64)
    star_bp = np.asarray(bp_star, dtype=np.float64).reshape(-1)
    star_value = np.asarray(value_star, dtype=np.float64).reshape(-1)
    if bp_grid.shape != value_grid.shape or bp_grid.shape[0] != star_bp.size:
        raise ValueError("coarse grid and star arrays have incompatible shapes")
    output = np.full(star_bp.shape, np.nan, dtype=np.float64)
    for row in range(star_bp.size):
        eligible = (
            np.isfinite(bp_grid[row])
            & np.isfinite(value_grid[row])
            & (np.abs(bp_grid[row] - star_bp[row]) >= float(delta))
        )
        if bool(eligible.any()) and np.isfinite(star_value[row]):
            output[row] = star_value[row] - float(np.max(value_grid[row, eligible]))
    return output


def _share(condition: np.ndarray) -> float:
    return float(np.mean(condition)) if condition.size else float("nan")


def summarize_sample(
    frame: pd.DataFrame,
    *,
    large_gap_threshold: float,
    high_regret_relative_threshold: float,
    weak_margin_relative_threshold: float,
) -> dict[str, Any]:
    n = int(len(frame))
    pred = frame["bp_pred"].to_numpy(dtype=np.float64)
    teacher = frame["bp_star"].to_numpy(dtype=np.float64)
    gap = frame["bp_gap"].to_numpy(dtype=np.float64)
    regret = frame["regret"].to_numpy(dtype=np.float64)
    regret_relative = frame["regret_relative"].to_numpy(dtype=np.float64)
    regret_scaled = frame["regret_scaled"].to_numpy(dtype=np.float64)
    margin = frame["top2_margin"].to_numpy(dtype=np.float64)
    relative_margin = frame["relative_margin"].to_numpy(dtype=np.float64)
    pred_std = float(np.std(pred)) if n else float("nan")
    teacher_std = float(np.std(teacher)) if n else float("nan")
    std_ratio = (
        pred_std / teacher_std
        if n and np.isfinite(teacher_std) and teacher_std > 0.0
        else float("nan")
    )
    large = gap >= float(large_gap_threshold)
    high_regret = regret_relative >= float(high_regret_relative_threshold)
    weak_margin = relative_margin < float(weak_margin_relative_threshold)
    return {
        "n": n,
        "pearson_r": pearson_correlation(pred, teacher),
        "spearman_r": spearman_correlation(pred, teacher),
        "bp_mae": float(np.mean(gap)) if n else float("nan"),
        "bp_gap_median": _quantile(gap, 0.50),
        "bp_gap_p90": _quantile(gap, 0.90),
        "bp_gap_p95": _quantile(gap, 0.95),
        "bp_gap_p99": _quantile(gap, 0.99),
        "bp_gap_max": float(np.max(gap)) if n else float("nan"),
        "pred_std": pred_std,
        "teacher_std": teacher_std,
        "std_ratio": std_ratio,
        "pred_mean": float(np.mean(pred)) if n else float("nan"),
        "teacher_mean": float(np.mean(teacher)) if n else float("nan"),
        "pred_p10": _quantile(pred, 0.10),
        "pred_p50": _quantile(pred, 0.50),
        "pred_p90": _quantile(pred, 0.90),
        "teacher_p10": _quantile(teacher, 0.10),
        "teacher_p50": _quantile(teacher, 0.50),
        "teacher_p90": _quantile(teacher, 0.90),
        "pred_low_boundary_share": _share(pred < 0.05),
        "pred_high_boundary_share": _share(pred > 0.95),
        "teacher_low_boundary_share": _share(teacher < 0.05),
        "teacher_high_boundary_share": _share(teacher > 0.95),
        "regret_mean": float(np.mean(regret)) if n else float("nan"),
        "regret_p50": _quantile(regret, 0.50),
        "regret_p90": _quantile(regret, 0.90),
        "regret_p95": _quantile(regret, 0.95),
        "regret_p99": _quantile(regret, 0.99),
        "regret_max": float(np.max(regret)) if n else float("nan"),
        "regret_relative_mean": float(np.mean(regret_relative)) if n else float("nan"),
        "regret_relative_p50": _quantile(regret_relative, 0.50),
        "regret_relative_p90": _quantile(regret_relative, 0.90),
        "regret_relative_p99": _quantile(regret_relative, 0.99),
        "regret_scaled_mean": float(np.mean(regret_scaled)) if n else float("nan"),
        "regret_scaled_p50": _quantile(regret_scaled, 0.50),
        "regret_scaled_p90": _quantile(regret_scaled, 0.90),
        "regret_scaled_p99": _quantile(regret_scaled, 0.99),
        "top2_margin_mean": float(np.mean(margin)) if n else float("nan"),
        "top2_margin_p10": _quantile(margin, 0.10),
        "top2_margin_p25": _quantile(margin, 0.25),
        "top2_margin_p50": _quantile(margin, 0.50),
        "top2_margin_p90": _quantile(margin, 0.90),
        "relative_margin_p10": _quantile(relative_margin, 0.10),
        "relative_margin_p50": _quantile(relative_margin, 0.50),
        "relative_margin_nonpositive_count": int(np.sum(relative_margin <= 0.0)),
        "weak_rel_margin_1e4": _share(relative_margin < 1e-4),
        "weak_rel_margin_1e3": _share(relative_margin < 1e-3),
        "weak_rel_margin_1e2": _share(relative_margin < 1e-2),
        "numerical_identified_share": _share(frame["numerical_identified"].to_numpy(dtype=bool)),
        "separated_margin_rel_0p05_p50": _quantile(frame["separated_margin_rel_0p05"], 0.50),
        "separated_margin_rel_0p10_p50": _quantile(frame["separated_margin_rel_0p10"], 0.50),
        "large_gap_share": _share(large),
        "large_gap_high_regret_share": _share(large & high_regret),
        "large_gap_low_regret_share": _share(large & ~high_regret),
        "large_gap_weak_margin_share": _share(large & weak_margin),
        "large_gap_strong_margin_share": _share(large & ~weak_margin),
        "gap_regret_pearson_r": pearson_correlation(gap, regret_relative),
        "gap_regret_spearman_r": spearman_correlation(gap, regret_relative),
        "margin_gap_pearson_r": pearson_correlation(relative_margin, gap),
        "margin_gap_spearman_r": spearman_correlation(relative_margin, gap),
        **{
            f"gap_gt_{threshold:.2f}_share".replace(".", "p"): _share(gap > threshold)
            for threshold in GAP_THRESHOLDS
        },
    }


def _checkpoint_payload(
    checkpoint: Path, combined_checkpoint: Path, device: torch.device
) -> Any:
    loaded = load_analysis_checkpoint(combined_checkpoint, device=device)
    if checkpoint.resolve() != combined_checkpoint.resolve():
        stage = torch.load(checkpoint, map_location=device)
        if not isinstance(stage, Mapping):
            raise ValueError("stage checkpoint must be a mapping")
        state = stage.get("models", {}).get("policy_value")
        if state is None:
            raise ValueError("stage checkpoint is missing models.policy_value")
        loaded.models["policy_value"].load_state_dict(state, strict=True)
        loaded.metadata["stage_checkpoint_path"] = str(checkpoint)
        loaded.metadata["checkpoint_stage"] = stage.get("stage")
        loaded.metadata["episode"] = stage.get("episode")
    return loaded


def _discover_combined(run_root: Path, episode: int) -> Path:
    candidates = [
        run_root / "checkpoints_analysis" / f"ep{episode}_combined.pt",
        run_root / "checkpoints" / f"ep{episode}_combined.pt",
    ]
    found = next((path for path in candidates if path.is_file()), None)
    if found is None:
        raise FileNotFoundError(f"No EP{episode} combined checkpoint found: {candidates}")
    return found.resolve()


def _branch_specs(model: torch.nn.Module, reference: ReferenceFirmState) -> dict[str, tuple[str, float]]:
    return {
        "p0": ("p0", float(reference.i_mid)),
        "pi_low": ("pi", 0.0),
        "pi_mid": ("pi", float(reference.i_mid)),
        "pi_high": ("pi", float(model.i_threshold)),
    }


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float64)


def build_grid_bank(
    name: str,
    reference: ReferenceFirmState,
    b_values: Sequence[float],
    z_values: Sequence[float],
    *,
    eta: float,
    device: torch.device,
    transition_eta_override: float | None = None,
) -> StateBank:
    base = build_focus_states(
        reference, b_values, z_values, eta=eta, i_value=reference.i_mid, device=device,
    )
    n = int(base.shape[0])
    context = pd.DataFrame({"bank_row": np.arange(n, dtype=np.int64)})
    b_array = np.asarray(b_values, dtype=np.float64)
    z_array = np.asarray(z_values, dtype=np.float64)
    return StateBank(
        name=name,
        reference=reference,
        base_states=base,
        hatc_cal=torch.full((n,), reference.hatc_cal, device=device),
        lnk_cal=torch.full((n,), reference.lnk_cal, device=device),
        context=context,
        metadata={
            "kind": "canonical_grid",
            "n_states": n,
            "b_values": b_array.tolist(),
            "z_values": z_array.tolist(),
            "eta": float(eta),
            "transition_eta_override": transition_eta_override,
            "ordering": "b-major_z-minor",
        },
        grid_shape=(len(b_array), len(z_array)),
        b_values=b_array,
        z_values=z_array,
        transition_eta_override=transition_eta_override,
    )


def _distribution_stats(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if numeric.size == 0:
        return {name: float("nan") for name in ("mean", "std", "p10", "p50", "p90")}
    return {
        "mean": float(np.mean(numeric)),
        "std": float(np.std(numeric)),
        "p10": float(np.quantile(numeric, 0.10)),
        "p50": float(np.quantile(numeric, 0.50)),
        "p90": float(np.quantile(numeric, 0.90)),
    }


def build_on_distribution_bank(
    frame: pd.DataFrame,
    *,
    reference: ReferenceFirmState,
    source: Path,
    macro_source: Path | None,
    max_states: int,
    seed: int,
    device: torch.device,
) -> StateBank:
    parents = select_parent_rows(frame)
    required = ["b", "z", "ETA", "i", "x", "Hatcf", "LnKF", "hatc_cal", "lnk_cal"]
    missing = [name for name in required if name not in parents.columns]
    if missing:
        raise ValueError(f"on-distribution parents are missing columns: {missing}")
    numeric = parents[required].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)
    usable = parents.loc[finite].copy()
    original_count = int(len(parents))
    sampled = deterministic_sample(usable, int(max_states), int(seed))
    states = torch.as_tensor(
        sampled[["b", "z", "ETA", "i", "x", "Hatcf", "LnKF"]].to_numpy(dtype=np.float32),
        device=device,
    )
    context_columns = [
        name for name in ("source_index", "path", "t", "branch", "ID") if name in sampled.columns
    ]
    context = sampled[context_columns].copy()
    context.insert(0, "bank_row", np.arange(len(sampled), dtype=np.int64))
    t_numeric = pd.to_numeric(sampled["t"], errors="coerce") if "t" in sampled else pd.Series(dtype=float)
    metadata = {
        "kind": "on_distribution",
        "source_artifact": str(source),
        "macro_artifact": None if macro_source is None else str(macro_source),
        "original_parent_states": original_count,
        "finite_parent_states": int(len(usable)),
        "nonfinite_rows_dropped": int(original_count - len(usable)),
        "sampled_states": int(len(sampled)),
        "sample_seed": int(seed),
        "burn_in_rule": "none",
        "b": _distribution_stats(sampled["b"]),
        "z": _distribution_stats(sampled["z"]),
        "eta_share": float(pd.to_numeric(sampled["ETA"], errors="coerce").mean()),
        "i": _distribution_stats(sampled["i"]),
        "simulation_t_min": float(t_numeric.min()) if len(t_numeric) else None,
        "simulation_t_max": float(t_numeric.max()) if len(t_numeric) else None,
    }
    return StateBank(
        name="on_distribution",
        reference=reference,
        base_states=states,
        hatc_cal=torch.as_tensor(sampled["hatc_cal"].to_numpy(dtype=np.float32), device=device),
        lnk_cal=torch.as_tensor(sampled["lnk_cal"].to_numpy(dtype=np.float32), device=device),
        context=context,
        metadata=metadata,
    )


def _empty_full_grid(n_rows: int, n_grid: int) -> dict[str, np.ndarray]:
    return {
        "coarse_bp_grid": np.full((n_rows, n_grid), np.nan, dtype=np.float64),
        "coarse_value_grid": np.full((n_rows, n_grid), np.nan, dtype=np.float64),
    }


def evaluate_state_bank(
    bank: StateBank,
    *,
    labels: Sequence[str],
    specs: Mapping[str, tuple[str, float]],
    model: torch.nn.Module,
    sdf_fc1: torch.nn.Module,
    loaded: Any,
    teacher: BPGridTeacher,
    episode: int,
    checkpoint_stage: str,
    n_child_shocks: int,
    shock_seed: int,
    teacher_margin_tol: float,
    timings: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, Any], list[str]]:
    n_rows = int(bank.base_states.shape[0])
    eta_values = bank.base_states[:, 2].detach().cpu().numpy()
    groups = [
        np.flatnonzero(np.isclose(eta_values, eta_value))
        for eta_value in sorted(np.unique(eta_values).tolist())
    ]
    frame_parts: list[pd.DataFrame] = []
    grids: dict[str, dict[str, np.ndarray]] = {}
    transition_records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for group_positions in groups:
        if group_positions.size == 0:
            continue
        index = torch.as_tensor(group_positions, dtype=torch.long, device=bank.base_states.device)
        transition_states = bank.base_states.index_select(0, index).clone()
        transition_states[:, 3] = float(bank.reference.i_mid)
        if bank.transition_eta_override is not None:
            transition_states[:, 2] = float(bank.transition_eta_override)
        # Keep the same exogenous shock realization for every branch and eta group.
        with timed_block(f"{bank.name}:transition_eta{eta_values[group_positions[0]]:g}", timings):
            transition = build_frozen_transition_data(
                sdf_fc1,
                transition_states,
                bank.reference,
                loaded.hyperparams,
                loaded.economic_config,
                n_child_shocks=int(n_child_shocks),
                shock_seed=int(shock_seed),
                shock_bank_max_child_shocks=int(n_child_shocks),
                hatc_cal_values=bank.hatc_cal.index_select(0, index),
                lnk_cal_values=bank.lnk_cal.index_select(0, index),
            )
        branch_states: list[torch.Tensor] = []
        predictions: list[torch.Tensor] = []
        outputs: list[Any] = []
        with torch.no_grad():
            for label in labels:
                branch, i_value = specs[label]
                state = bank.base_states.index_select(0, index).clone()
                state[:, 3] = float(i_value)
                output = model(state)
                branch_states.append(state)
                outputs.append(output)
                predictions.append(output.bp0 if branch == "p0" else output.bpI)
        child_count = int(transition.stacked_children().shape[1])
        with timed_block(f"{bank.name}:teacher_eta{eta_values[group_positions[0]]:g}", timings):
            with _checkpoint_economic_config(loaded.economic_config), torch.no_grad():
                bundles = teacher.compute_multi_j_branches(
                    branch_states,
                    transition.stacked_children(),
                    transition.stacked_m_used(),
                    branches=[specs[label][0] for label in labels],
                    prefix_child_counts=[child_count],
                    child_weights=transition.branch_weights,
                    bp_preds=predictions,
                )
        transition_records.append({
            "eta": float(eta_values[group_positions[0]]),
            "n_rows": int(group_positions.size),
            **transition.metadata,
        })
        for pos, label in enumerate(labels):
            result = bundles[pos][child_count]
            with torch.no_grad():
                scale = model.equity_value_scale(branch_states[pos])
            part, branch_grid = _state_rows(
                episode=episode,
                checkpoint_stage=checkpoint_stage,
                label=label,
                state=branch_states[pos],
                prediction=predictions[pos],
                phat=outputs[pos].Phat,
                value_scale=scale,
                result=result,
                teacher_margin_tol=teacher_margin_tol,
            )
            part.insert(0, "bank", bank.name)
            part.insert(1, "bank_row", group_positions)
            context = bank.context.iloc[group_positions].reset_index(drop=True)
            for column in context:
                if column not in part.columns:
                    part[column] = context[column].to_numpy()
            frame_parts.append(part)
            if bool(np.isclose(eta_values[group_positions[0]], 1.0)):
                if label not in grids:
                    grids[label] = _empty_full_grid(n_rows, branch_grid["coarse_bp_grid"].shape[1])
                for grid_name, values in branch_grid.items():
                    grids[label][grid_name][group_positions, :] = values
            for delta_name in ("separated_margin_0p05", "separated_margin_0p10"):
                negative = part[delta_name] < -1e-6
                if bool(negative.any()):
                    warnings.append(
                        f"{bank.name}/{label}: {int(negative.sum())} materially negative {delta_name} values"
                    )
    state_level = pd.concat(frame_parts, ignore_index=True)
    state_level = state_level.sort_values(["branch", "bank_row"], kind="stable").reset_index(drop=True)
    shock_hashes = sorted({record["shock_bank_sha256"] for record in transition_records})
    metadata = {
        **bank.metadata,
        "branches": list(labels),
        "n_child_shocks": int(n_child_shocks),
        "expanded_exact_eta_child_count": 2 * int(n_child_shocks),
        "shock_seed": int(shock_seed),
        "shock_bank_sha256": shock_hashes[0] if len(shock_hashes) == 1 else shock_hashes,
        "transition_groups": transition_records,
    }
    return state_level, grids, metadata, warnings


def _state_rows(
    *,
    episode: int,
    checkpoint_stage: str,
    label: str,
    state: torch.Tensor,
    prediction: torch.Tensor,
    phat: torch.Tensor,
    value_scale: torch.Tensor,
    result: Mapping[str, torch.Tensor],
    teacher_margin_tol: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    bp_pred = _as_numpy(prediction).reshape(-1)
    bp_star = _as_numpy(result["bp_star"]).reshape(-1)
    value_star = _as_numpy(result["value_star"]).reshape(-1)
    value_pred = _as_numpy(result["value_pred"]).reshape(-1)
    regret = _as_numpy(result["regret"]).reshape(-1)
    top2 = _as_numpy(result["top2_margin"]).reshape(-1)
    confidence = _as_numpy(result["confidence"]).reshape(-1)
    refi = _as_numpy(result["refi_active"]).reshape(-1) > 0.5
    phat_array = _as_numpy(phat).reshape(-1)
    scale = _as_numpy(value_scale).reshape(-1)
    masks = build_masks(refi, phat_array, top2, teacher_margin_tol=teacher_margin_tol)
    relative_regret, scaled_regret = normalized_regrets(regret, value_star, scale)
    relative_margin = relative_margins(top2, value_star)
    coarse_bp = _as_numpy(result["coarse_bp_grid"])
    coarse_value = _as_numpy(result["coarse_value_grid"])
    sep05 = separated_margin(coarse_bp, coarse_value, bp_star, value_star, delta=0.05)
    sep10 = separated_margin(coarse_bp, coarse_value, bp_star, value_star, delta=0.10)
    frame = pd.DataFrame({
        "episode": int(episode),
        "checkpoint_stage": checkpoint_stage,
        "branch": label,
        "b": _as_numpy(state[:, 0]).reshape(-1),
        "z": _as_numpy(state[:, 1]).reshape(-1),
        "eta": _as_numpy(state[:, 2]).reshape(-1),
        "i": _as_numpy(state[:, 3]).reshape(-1),
        "Phat": phat_array,
        "bp_pred": bp_pred,
        "bp_star": bp_star,
        "bp_gap": np.abs(bp_pred - bp_star),
        "value_star": value_star,
        "value_pred": value_pred,
        "regret": regret,
        "regret_relative": relative_regret,
        "value_scale": scale,
        "regret_scaled": scaled_regret,
        "top2_margin": top2,
        "relative_margin": relative_margin,
        "separated_margin_0p05": sep05,
        "separated_margin_rel_0p05": sep05 / (np.abs(value_star) + EPS),
        "separated_margin_0p10": sep10,
        "separated_margin_rel_0p10": sep10 / (np.abs(value_star) + EPS),
        "confidence": confidence,
        "refi_active": refi,
        "survival": masks["survival"],
        "numerical_identified": masks["numerical_identified"],
        "primary_mask": masks["primary"],
    })
    grids = {
        "coarse_bp_grid": coarse_bp,
        "coarse_value_grid": coarse_value,
    }
    return frame, grids


def _masked(frame: pd.DataFrame, mask_type: str) -> pd.DataFrame:
    column = {
        "raw_refi": "refi_active",
        "survival_refi": "survival",
        "primary": "primary_mask",
    }[mask_type]
    mask = frame[column].astype(bool)
    if mask_type == "survival_refi":
        mask &= frame["refi_active"].astype(bool)
    return frame.loc[mask].copy()


def build_summary(
    state_level: pd.DataFrame,
    *,
    large_gap_threshold: float,
    high_regret_relative_threshold: float,
    weak_margin_relative_threshold: float,
) -> pd.DataFrame:
    rows = []
    banks = state_level["bank"].drop_duplicates().tolist() if "bank" in state_level else [None]
    for bank in banks:
        bank_frame = state_level if bank is None else state_level[state_level["bank"] == bank]
        for branch in BRANCH_ORDER:
            branch_frame = bank_frame[bank_frame["branch"] == branch]
            if branch_frame.empty:
                continue
            for mask_type in MASK_TYPES:
                sample = _masked(branch_frame, mask_type)
                row = {
                    "branch": branch,
                    "mask_type": mask_type,
                    **summarize_sample(
                        sample,
                        large_gap_threshold=large_gap_threshold,
                        high_regret_relative_threshold=high_regret_relative_threshold,
                        weak_margin_relative_threshold=weak_margin_relative_threshold,
                    ),
                }
                if bank is not None:
                    row = {"bank": bank, **row}
                rows.append(row)
    return pd.DataFrame(rows)


def reshape_dense_surface(values: Iterable[float], shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    expected = int(shape[0]) * int(shape[1])
    if array.size != expected:
        raise ValueError(f"dense surface has {array.size} values, expected {expected}")
    # State construction is b-major, z-minor. Rows are b and columns are z.
    return array.reshape(shape)


def deterministic_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or len(frame) <= n:
        return frame.copy().reset_index(drop=False).rename(columns={"index": "source_index"})
    rng = np.random.default_rng(int(seed))
    positions = np.sort(rng.choice(len(frame), size=int(n), replace=False))
    sampled = frame.iloc[positions].copy()
    return sampled.reset_index(drop=False).rename(columns={"index": "source_index"})


def build_binned_summary(
    state_level: pd.DataFrame,
    *,
    column: str,
    edges: Sequence[float],
    bin_column: str,
) -> pd.DataFrame:
    primary = state_level[state_level["primary_mask"].astype(bool)].copy()
    labels = [f"[{edges[pos]:g},{edges[pos + 1]:g})" for pos in range(len(edges) - 1)]
    primary[bin_column] = pd.cut(
        primary[column], bins=list(edges), labels=labels, right=False, include_lowest=True,
    )
    rows: list[dict[str, Any]] = []
    group_columns = [name for name in ("bank", "branch", bin_column) if name in primary.columns]
    for key, sample in primary.groupby(group_columns, observed=True, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        identifiers = dict(zip(group_columns, key_values))
        metrics = summarize_sample(
            sample,
            large_gap_threshold=0.20,
            high_regret_relative_threshold=0.01,
            weak_margin_relative_threshold=0.001,
        )
        rows.append({
            **identifiers,
            "n": metrics["n"],
            "mae": metrics["bp_mae"],
            "p90_gap": metrics["bp_gap_p90"],
            "pearson_r": metrics["pearson_r"],
            "spearman_r": metrics["spearman_r"],
            "std_ratio": metrics["std_ratio"],
            "regret_scaled_mean": metrics["regret_scaled_mean"],
            "regret_scaled_p90": metrics["regret_scaled_p90"],
            "weak_margin_share": metrics["weak_rel_margin_1e3"],
        })
    return pd.DataFrame(rows)


def rank_worst_states(state_level: pd.DataFrame, *, top_n: int = 20) -> pd.DataFrame:
    primary = state_level[state_level["primary_mask"].astype(bool)].copy()
    primary["gap_x_regret_scaled"] = primary["bp_gap"] * primary["regret_scaled"]
    parts = []
    for category, column in (
        ("largest_gap", "bp_gap"),
        ("largest_regret_scaled", "regret_scaled"),
        ("largest_gap_x_regret", "gap_x_regret_scaled"),
    ):
        ranked = primary.sort_values(column, ascending=False).head(int(top_n)).copy()
        ranked.insert(0, "ranking_category", category)
        ranked.insert(1, "ranking_value", ranked[column])
        parts.append(ranked)
    return pd.concat(parts, ignore_index=True) if parts else primary.iloc[0:0]


def validate_tiny_smoke_mae(actual: float, expected: float, atol: float) -> None:
    if not math.isfinite(actual) or abs(float(actual) - float(expected)) > float(atol):
        raise RuntimeError(
            "tiny smoke regression failed: "
            f"MAE={actual:.9g}, expected={expected:.9g} within atol={atol:.3g}"
        )


def build_by_z(state_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = state_level[state_level["primary_mask"].astype(bool)]
    for (branch, z_value), sample in primary.groupby(["branch", "z"], sort=True):
        pred = sample["bp_pred"].to_numpy(dtype=np.float64)
        star = sample["bp_star"].to_numpy(dtype=np.float64)
        pred_std = float(np.std(pred)) if len(sample) else float("nan")
        teacher_std = float(np.std(star)) if len(sample) else float("nan")
        rows.append({
            "branch": branch,
            "z": float(z_value),
            "n": int(len(sample)),
            "mae": float(sample["bp_gap"].mean()),
            "pearson_r": pearson_correlation(pred, star),
            "mean_regret_relative": float(sample["regret_relative"].mean()),
            "p90_regret_relative": _quantile(sample["regret_relative"], 0.90),
            "mean_relative_margin": float(sample["relative_margin"].mean()),
            "weak_margin_share": float((sample["relative_margin"] < 1e-3).mean()),
            "std_ratio": pred_std / teacher_std if teacher_std > 0.0 else float("nan"),
        })
    return pd.DataFrame(rows)


def _scatter_identity(axis: plt.Axes, sample: pd.DataFrame, *, color: str, title: str) -> None:
    scatter = axis.scatter(sample["bp_star"], sample["bp_pred"], c=sample[color], cmap="viridis")
    axis.plot([0, 1], [0, 1], "k--", linewidth=1)
    axis.set(xlabel="teacher bp*", ylabel="network bp", title=title, xlim=(0, 1), ylim=(0, 1))
    plt.colorbar(scatter, ax=axis, label=color)


def _distribution_panel(axis: plt.Axes, sample: pd.DataFrame) -> None:
    bins = np.linspace(0.0, 1.0, 21)
    axis.hist(sample["bp_star"], bins=bins, alpha=0.55, label="teacher", density=True)
    axis.hist(sample["bp_pred"], bins=bins, alpha=0.55, label="prediction", density=True)
    axis.set(xlabel="bp", ylabel="density", title="Teacher and prediction distribution")
    axis.legend()


def _savefig_atomic(fig: plt.Figure, path: Path, *, dpi: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    fig.savefig(temporary, dpi=dpi)
    os.replace(temporary, path)
    plt.close(fig)


def _heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    bank: StateBank,
    *,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> Any:
    if bank.grid_shape is None or bank.b_values is None or bank.z_values is None:
        raise ValueError(f"bank {bank.name} is not a dense grid")
    matrix = reshape_dense_surface(values, bank.grid_shape)
    palette = plt.get_cmap(cmap).copy()
    palette.set_bad("#d9d9d9")
    image = axis.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        extent=[bank.z_values[0], bank.z_values[-1], bank.b_values[0], bank.b_values[-1]],
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap=palette,
    )
    axis.set(xlabel="z", ylabel="b", title=title)
    return image


def write_dense_figures(
    bank: StateBank,
    state_level: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    target = output_dir / "figures" / "dense"
    target.mkdir(parents=True, exist_ok=True)
    branches = state_level["branch"].drop_duplicates().tolist()
    primary_gap = state_level["bp_gap"].where(state_level["primary_mask"].astype(bool))
    finite_gap = _finite(primary_gap)
    shared_gap_max = float(np.max(finite_gap)) if finite_gap.size else 1.0
    written: list[str] = []
    specs = (
        ("bp_star", "bp_teacher_heatmap", "Teacher bp*", 0.0, 1.0, "viridis", False),
        ("bp_pred", "bp_prediction_heatmap", "Network bp", 0.0, 1.0, "viridis", False),
        ("bp_gap", "bp_gap_heatmap", "Absolute BP gap", 0.0, shared_gap_max, "magma", True),
        ("regret_scaled", "regret_scaled_heatmap", "Scaled regret", None, None, "magma", True),
        ("regret_relative", "regret_relative_heatmap", "Relative regret", None, None, "magma", True),
        ("relative_margin", "relative_margin_heatmap", "Relative top-2 margin", None, None, "viridis", True),
        ("separated_margin_rel_0p10", "separated_margin_heatmap", "Separated relative margin (delta=.10)", None, None, "viridis", True),
    )
    for branch in branches:
        sample = state_level[state_level["branch"] == branch].sort_values("bank_row")
        for column, suffix, title, vmin, vmax, cmap, mask_primary in specs:
            values = sample[column].to_numpy(dtype=np.float64)
            if mask_primary:
                values = np.where(sample["primary_mask"].to_numpy(dtype=bool), values, np.nan)
            fig, axis = plt.subplots(figsize=(7.2, 5.5))
            image = _heatmap(
                axis, values, bank, title=f"{branch}: {title}", vmin=vmin, vmax=vmax, cmap=cmap,
            )
            fig.colorbar(image, ax=axis)
            path = target / f"{branch}_{suffix}.png"
            fig.tight_layout(); _savefig_atomic(fig, path); written.append(str(path))

        fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
        panel_specs = (
            ("bp_star", "Teacher bp*", 0.0, 1.0, "viridis", False),
            ("bp_pred", "Network bp", 0.0, 1.0, "viridis", False),
            ("bp_gap", "Absolute gap", 0.0, shared_gap_max, "magma", True),
            ("regret_scaled", "Scaled regret", None, None, "magma", True),
            ("separated_margin_rel_0p10", "Separated relative margin", None, None, "viridis", True),
        )
        for axis, (column, title, vmin, vmax, cmap, mask_primary) in zip(axes.flat, panel_specs):
            values = sample[column].to_numpy(dtype=np.float64)
            if mask_primary:
                values = np.where(sample["primary_mask"].to_numpy(dtype=bool), values, np.nan)
            image = _heatmap(axis, values, bank, title=title, vmin=vmin, vmax=vmax, cmap=cmap)
            fig.colorbar(image, ax=axis, shrink=0.85)
        mask_image = _heatmap(
            axes.flat[-1], sample["primary_mask"].to_numpy(dtype=float), bank,
            title="Primary mask", vmin=0.0, vmax=1.0, cmap="gray_r",
        )
        fig.colorbar(mask_image, ax=axes.flat[-1], shrink=0.85)
        fig.suptitle(f"Dense BP diagnostics: {branch}")
        path = target / f"{branch}_diagnostic_panel.png"
        fig.tight_layout(); _savefig_atomic(fig, path); written.append(str(path))
    return written


def write_ondist_figures(
    state_level: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    target = output_dir / "figures" / "ondist"
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for branch in state_level["branch"].drop_duplicates():
        sample = state_level[
            (state_level["branch"] == branch) & state_level["primary_mask"].astype(bool)
        ]
        if sample.empty:
            continue
        metric = summary[
            (summary["branch"] == branch) & (summary["mask_type"] == "primary")
        ].iloc[0]
        fig, axis = plt.subplots(figsize=(6.5, 5.5))
        _scatter_identity(
            axis,
            sample,
            color="z",
            title=(
                f"{branch}: N={len(sample)}, MAE={metric.bp_mae:.3g}, "
                f"Pearson={metric.pearson_r:.3g}, Spearman={metric.spearman_r:.3g}, "
                f"std ratio={metric.std_ratio:.3g}"
            ),
        )
        path = target / f"{branch}_pred_vs_teacher.png"
        fig.tight_layout(); _savefig_atomic(fig, path); written.append(str(path))

        fig, axis = plt.subplots(figsize=(6.5, 5.5))
        scatter = axis.scatter(
            sample["bp_gap"], sample["regret_scaled"], c=sample["relative_margin"], cmap="plasma"
        )
        axis.set(xlabel="BP gap", ylabel="scaled regret", title=f"{branch}: gap vs regret")
        fig.colorbar(scatter, ax=axis, label="relative margin")
        path = target / f"{branch}_gap_vs_regret.png"
        fig.tight_layout(); _savefig_atomic(fig, path); written.append(str(path))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        _distribution_panel(axes[0], sample)
        for column, label in (("bp_star", "teacher"), ("bp_pred", "prediction")):
            values = np.sort(sample[column].to_numpy(dtype=np.float64))
            axes[1].step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=label)
        axes[1].set(xlabel="bp", ylabel="ECDF", title="Empirical CDF"); axes[1].legend()
        path = target / f"{branch}_teacher_pred_distribution.png"
        fig.tight_layout(); _savefig_atomic(fig, path); written.append(str(path))
    return written


def write_compression_figure(summary: pd.DataFrame, output_dir: Path) -> Path:
    sample = summary[summary["mask_type"] == "primary"].copy()
    sample["label"] = sample["bank"].astype(str) + " / " + sample["branch"].astype(str)
    fig, axis = plt.subplots(figsize=(max(8.0, 0.55 * len(sample)), 5.2))
    axis.bar(np.arange(len(sample)), sample["std_ratio"].to_numpy(dtype=float))
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(np.arange(len(sample)), sample["label"], rotation=60, ha="right")
    axis.set(ylabel="std(prediction) / std(teacher)", title="BP compression by state bank and branch")
    path = output_dir / "figures" / "compression_ratio_by_bank_branch.png"
    fig.tight_layout(); _savefig_atomic(fig, path)
    return path


def write_state_bank_coverage(banks: Mapping[str, StateBank], output_dir: Path) -> Path:
    fig, axis = plt.subplots(figsize=(8, 6))
    styles = {
        "dense_grid": {"s": 8, "alpha": 0.22, "label": "dense grid"},
        "on_distribution": {"s": 18, "alpha": 0.55, "label": "on-distribution"},
        "tiny_smoke": {"s": 65, "alpha": 1.0, "marker": "x", "label": "tiny smoke"},
        "presentation_exact": {"s": 45, "alpha": 0.8, "marker": "+", "label": "presentation"},
    }
    for name, bank in banks.items():
        style = styles.get(name, {"s": 20, "alpha": 0.5, "label": name})
        values = bank.base_states.detach().cpu().numpy()
        axis.scatter(values[:, 0], values[:, 1], **style)
    axis.set(xlabel="b", ylabel="z", title="State-bank coverage in (b,z)")
    axis.legend()
    path = output_dir / "figures" / "state_bank_coverage.png"
    fig.tight_layout(); _savefig_atomic(fig, path)
    return path


def write_presentation_reproduction(
    original: pd.DataFrame,
    reproduced: pd.DataFrame,
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    original_pairs = (
        ("bp0_teacher", "bp0_pred", "Original p0"),
        ("bpI_teacher", "bpI_pred", "Original pi_mid"),
    )
    for teacher_column, pred_column, label in original_pairs:
        if {teacher_column, pred_column}.issubset(original.columns):
            axes[0].scatter(original[teacher_column], original[pred_column], alpha=0.7, label=label)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set(xlabel="teacher", ylabel="prediction", title="Original presentation CSV", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend()
    for branch, sample in reproduced[reproduced["refi_active"].astype(bool)].groupby("branch"):
        axes[1].scatter(sample["bp_star"], sample["bp_pred"], alpha=0.7, label=branch)
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[1].set(xlabel="production teacher", ylabel="current checkpoint prediction", title="Current evaluator", xlim=(0, 1), ylim=(0, 1))
    axes[1].legend()
    path = output_dir / "figures" / "presentation_reproduction.png"
    fig.tight_layout(); _savefig_atomic(fig, path)
    return path


def write_figures(state_level: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for branch in BRANCH_ORDER:
        sample = state_level[(state_level["branch"] == branch) & state_level["primary_mask"].astype(bool)]
        metrics = summary[(summary["branch"] == branch) & (summary["mask_type"] == "primary")]
        metric = metrics.iloc[0] if not metrics.empty else {}
        if sample.empty:
            continue
        title = (
            f"EP2 eta=1 post_bp {branch} PRIMARY\n"
            f"MAE={metric.get('bp_mae', np.nan):.3g}, Pearson={metric.get('pearson_r', np.nan):.3g}, "
            f"Spearman={metric.get('spearman_r', np.nan):.3g}"
        )
        fig, axis = plt.subplots(figsize=(6.5, 5.5))
        _scatter_identity(axis, sample, color="z", title=title)
        fig.tight_layout(); fig.savefig(figures / f"bp_pred_vs_teacher_{branch}.png", dpi=160); plt.close(fig)

        fig, axis = plt.subplots(figsize=(6.5, 5.5))
        scatter = axis.scatter(sample["bp_gap"], sample["regret_scaled"], c=sample["relative_margin"], cmap="plasma")
        axis.set(xlabel="|bp_pred - bp_star|", ylabel="regret / value scale", title=f"Gap vs regret: {branch}")
        plt.colorbar(scatter, ax=axis, label="relative margin")
        fig.tight_layout(); fig.savefig(figures / f"bp_gap_vs_regret_{branch}.png", dpi=160); plt.close(fig)

        fig, axis = plt.subplots(figsize=(6.5, 5.5))
        scatter = axis.scatter(sample["relative_margin"], sample["bp_gap"], c=sample["z"], cmap="viridis")
        axis.set_xscale("symlog", linthresh=1e-8)
        axis.set(xlabel="relative top-2 margin", ylabel="bp gap", title=f"Margin vs BP gap: {branch}")
        plt.colorbar(scatter, ax=axis, label="z")
        fig.tight_layout(); fig.savefig(figures / f"bp_gap_vs_relative_margin_{branch}.png", dpi=160); plt.close(fig)

        fig, axis = plt.subplots(figsize=(6.5, 5.5))
        scatter = axis.scatter(sample["relative_margin"], sample["regret_scaled"], c=sample["bp_gap"], cmap="magma")
        axis.set_xscale("symlog", linthresh=1e-8)
        axis.set(xlabel="relative top-2 margin", ylabel="regret / value scale", title=f"Margin vs regret: {branch}")
        plt.colorbar(scatter, ax=axis, label="bp gap")
        fig.tight_layout(); fig.savefig(figures / f"bp_regret_vs_relative_margin_{branch}.png", dpi=160); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        _distribution_panel(axes[0], sample)
        for column, label in (("bp_star", "teacher"), ("bp_pred", "prediction")):
            values = np.sort(sample[column].to_numpy(dtype=np.float64))
            axes[1].step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=label)
        axes[1].set(xlabel="bp", ylabel="ECDF", title="Empirical CDF"); axes[1].legend()
        fig.tight_layout(); fig.savefig(figures / f"bp_teacher_pred_distribution_{branch}.png", dpi=160); plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        _scatter_identity(axes[0, 0], sample, color="z", title="Prediction vs teacher")
        axes[0, 1].scatter(sample["bp_gap"], sample["regret_scaled"], c=sample["relative_margin"], cmap="plasma")
        axes[0, 1].set(xlabel="bp gap", ylabel="regret / scale", title="Gap vs regret")
        axes[1, 0].scatter(sample["relative_margin"], sample["bp_gap"], c=sample["z"], cmap="viridis")
        axes[1, 0].set_xscale("symlog", linthresh=1e-8); axes[1, 0].set(xlabel="relative margin", ylabel="bp gap", title="Margin vs gap")
        _distribution_panel(axes[1, 1], sample)
        fig.suptitle(title); fig.tight_layout(); fig.savefig(figures / f"bp_fit_headline_{branch}.png", dpi=160); plt.close(fig)


def _example_indices(frame: pd.DataFrame, *, high_regret_threshold: float) -> list[tuple[str, int]]:
    primary = frame[frame["primary_mask"].astype(bool)]
    if primary.empty:
        return []
    primary = primary.copy()
    primary["gap_x_regret_scaled"] = primary["bp_gap"] * primary["regret_scaled"]
    groups = {
        "largest_gap": primary.sort_values("bp_gap", ascending=False),
        "largest_regret_scaled": primary.sort_values("regret_scaled", ascending=False),
        "largest_gap_x_regret": primary.sort_values("gap_x_regret_scaled", ascending=False),
    }
    chosen: list[tuple[str, int]] = []
    for category, candidates in groups.items():
        chosen.extend((category, int(index)) for index in candidates.head(5).index)
    return chosen


def write_objective_examples(
    state_level: pd.DataFrame,
    grids: Mapping[str, Mapping[str, np.ndarray]],
    output_dir: Path,
    *,
    high_regret_threshold: float,
    bank_name: str | None = None,
) -> None:
    target = output_dir / "objective_examples" / (bank_name or "bank")
    target.mkdir(parents=True, exist_ok=True)
    for branch in (label for label in BRANCH_ORDER if label in grids):
        frame = state_level[state_level["branch"] == branch]
        branch_grids = grids[branch]
        for order, (category, index) in enumerate(_example_indices(frame, high_regret_threshold=high_regret_threshold), start=1):
            row = state_level.loc[index]
            pos = int(row.bank_row)
            curve = pd.DataFrame({
                "bp_candidate": branch_grids["coarse_bp_grid"][pos],
                "value": branch_grids["coarse_value_grid"][pos],
            })
            stem = f"{branch}_{category}_{order}_b{row.b:.3g}_z{row.z:.3g}"
            atomic_write_csv(target / f"{stem}.csv", curve)
            fig, axis = plt.subplots(figsize=(7, 4.8))
            axis.plot(curve["bp_candidate"], curve["value"], marker="o", markersize=3)
            axis.axvline(row.bp_star, color="tab:green", linestyle="--", label="teacher bp*")
            axis.axvline(row.bp_pred, color="tab:red", linestyle=":", label="network bp")
            axis.set(
                xlabel="candidate bp",
                ylabel="V(bp)",
                title=(
                    f"{branch}: {category}, b={row.b:g}, z={row.z:g}\n"
                    f"regret={row.regret:.3g}, rel.margin={row.relative_margin:.3g}, "
                    f"sep.margin={row.separated_margin_rel_0p10:.3g}"
                ),
            )
            axis.legend(); fig.tight_layout(); _savefig_atomic(fig, target / f"{stem}.png")


def write_bank_artifacts(
    bank: StateBank,
    state_level: pd.DataFrame,
    grids: Mapping[str, Mapping[str, np.ndarray]],
    bank_metadata: Mapping[str, Any],
    output_dir: Path,
    *,
    large_gap_threshold: float,
    high_regret_relative_threshold: float,
    weak_margin_relative_threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    summary = build_summary(
        state_level,
        large_gap_threshold=large_gap_threshold,
        high_regret_relative_threshold=high_regret_relative_threshold,
        weak_margin_relative_threshold=weak_margin_relative_threshold,
    )
    atomic_write_csv(output_dir / "tables" / f"bp_fit_state_level_{bank.name}.csv", state_level)
    atomic_write_csv(output_dir / "tables" / f"bp_fit_summary_{bank.name}.csv", summary)
    atomic_write_json(output_dir / "metadata" / f"{bank.name}.json", bank_metadata)
    generated: list[str] = []
    if bank.name == "dense_grid":
        generated.extend(write_dense_figures(bank, state_level, output_dir))
        by_z = build_binned_summary(
            state_level, column="z", edges=Z_BIN_EDGES, bin_column="z_bin",
        )
        by_b = build_binned_summary(
            state_level, column="b", edges=B_BIN_EDGES, bin_column="b_bin",
        )
        atomic_write_csv(output_dir / "tables" / "bp_fit_by_z_bin.csv", by_z)
        atomic_write_csv(output_dir / "tables" / "bp_fit_by_b_bin.csv", by_b)
    elif bank.name == "on_distribution":
        generated.extend(write_ondist_figures(state_level, summary, output_dir))
    else:
        bank_output = output_dir / "banks" / bank.name
        write_figures(state_level, summary, bank_output)
        generated.extend(str(path) for path in (bank_output / "figures").glob("*.png"))
    if bank.name in {"dense_grid", "on_distribution"}:
        worst = rank_worst_states(state_level, top_n=20)
        atomic_write_csv(output_dir / "tables" / f"worst_states_{'dense' if bank.name == 'dense_grid' else 'ondist'}.csv", worst)
        write_objective_examples(
            state_level,
            grids,
            output_dir,
            high_regret_threshold=high_regret_relative_threshold,
            bank_name=bank.name,
        )
        generated.extend(
            str(path) for path in (output_dir / "objective_examples" / bank.name).glob("*.png")
        )
    return summary, generated


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.6g}" if math.isfinite(number) else "NaN"


def write_summary_markdown(
    summary: pd.DataFrame,
    by_z: pd.DataFrame,
    output_dir: Path,
    *,
    presentation: Mapping[str, Any] | None = None,
) -> None:
    primary = summary[summary["mask_type"] == "primary"].copy()

    def row(bank: str, branch: str = "pi_mid") -> Mapping[str, Any]:
        match = primary[(primary["bank"] == bank) & (primary["branch"] == branch)]
        return match.iloc[0].to_dict() if not match.empty else {}

    tiny = row("tiny_smoke")
    dense = row("dense_grid")
    ondist = row("on_distribution")
    dense_large = float(dense.get("bp_mae", float("nan"))) >= 0.05
    ondist_large = float(ondist.get("bp_mae", float("nan"))) >= 0.05
    case = {
        (False, False): "CASE A: dense and on-distribution errors are both small; inspect the old presentation pipeline.",
        (True, False): "CASE B: error is concentrated off distribution or in canonical tails.",
        (False, True): "CASE C: error is concentrated in actually visited states and is economically serious.",
        (True, True): "CASE D: BP fitting failure is broad across canonical and visited states.",
    }[(dense_large, ondist_large)]
    worst_branch = (
        primary[primary["bank"].isin(["dense_grid", "on_distribution"])]
        .sort_values("bp_mae", ascending=False)
        .head(1)
    )
    worst_branch_text = "unavailable" if worst_branch.empty else (
        f"{worst_branch.iloc[0].bank}/{worst_branch.iloc[0].branch} "
        f"(MAE={_fmt(worst_branch.iloc[0].bp_mae)})"
    )
    worst_z = by_z.sort_values("mae", ascending=False).head(1) if not by_z.empty else by_z
    worst_z_text = "unavailable" if worst_z.empty else (
        f"{worst_z.iloc[0].z_bin}, branch={worst_z.iloc[0].branch}, MAE={_fmt(worst_z.iloc[0].mae)}"
    )
    presentation = dict(presentation or {})
    lines = [
        "# BP Teacher Fit State-Space Summary",
        "",
        "This is a read-only diagnostic. No training, optimizer, backward pass, or checkpoint mutation was performed.",
        "",
        "1. **Why tiny smoke looks good.** Its nine points cover only b={0.05,0.20,0.50}, z={0,2,4}, eta=1 and pi_mid. "
        f"Its MAE is {_fmt(tiny.get('bp_mae'))}, so it is a regression probe rather than a state-space diagnosis.",
        f"2. **Dense-grid MAE.** pi_mid PRIMARY MAE={_fmt(dense.get('bp_mae'))}, p90 gap={_fmt(dense.get('bp_gap_p90'))}.",
        f"3. **On-distribution MAE.** pi_mid PRIMARY MAE={_fmt(ondist.get('bp_mae'))}, p90 gap={_fmt(ondist.get('bp_gap_p90'))}.",
        f"4. **Presentation reproduction.** source_found={presentation.get('source_found', False)}; "
        f"original pi MAE={_fmt(presentation.get('original_pi_mae'))}; reproduced pi_mid MAE={_fmt(presentation.get('reproduced_pi_mid_mae'))}.",
        f"5. **Where mismatch is largest.** Worst dense z bin: {worst_z_text}.",
        f"6. **Worst branch.** {worst_branch_text}.",
        "7. **State-region concentration.** See `bp_fit_by_z_bin.csv`, `bp_fit_by_b_bin.csv`, and dense heatmaps; no region is silently averaged away.",
        f"8. **Do large gaps carry loss?** Dense gap-regret Pearson={_fmt(dense.get('gap_regret_pearson_r'))}; "
        f"on-distribution={_fmt(ondist.get('gap_regret_pearson_r'))}.",
        f"9. **Are large gaps weakly identified?** Dense weak-margin share={_fmt(dense.get('weak_rel_margin_1e3'))}; "
        f"on-distribution={_fmt(ondist.get('weak_rel_margin_1e3'))}.",
        f"10. **State-space classification.** {case}",
        "11. **Next action.** Do not alter BP training from the tiny smoke alone. First compare current-production presentation reproduction, "
        "dense-grid, and visited-state regret/margin evidence; a presentation-version mismatch should be fixed before training changes.",
        "",
        "Raw top-2 margin can be small because the runner-up is adjacent. Interpret it jointly with regret and separated margins.",
    ]
    path = output_dir / "bp_teacher_fit_state_space_summary.md"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--combined-checkpoint", type=Path, default=None)
    parser.add_argument("--firm-data", type=Path, default=None)
    parser.add_argument("--macro-data", type=Path, default=None)
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--checkpoint-stage", default="post_bp")
    parser.add_argument("--b-values", type=float, nargs="+", default=list(DEFAULT_B_VALUES))
    parser.add_argument("--z-values", type=float, nargs="+", default=list(DEFAULT_Z_VALUES))
    parser.add_argument("--dense-points", type=int, default=41)
    parser.add_argument("--ondist-firm-data", type=Path, default=None)
    parser.add_argument("--ondist-macro-data", type=Path, default=None)
    parser.add_argument("--ondist-max-states", type=int, default=2000)
    parser.add_argument("--sample-seed", type=int, default=12345)
    parser.add_argument("--presentation-focus-csv", type=Path, default=None)
    parser.add_argument("--presentation-metadata-json", type=Path, default=None)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--branches", nargs="+", choices=BRANCH_ORDER, default=list(BRANCH_ORDER))
    parser.add_argument("--n-child-shocks", type=int, default=64)
    parser.add_argument("--shock-seed", type=int, default=12345)
    parser.add_argument("--teacher-margin-tol", type=float, default=1e-8)
    parser.add_argument("--large-gap-threshold", type=float, default=0.20)
    parser.add_argument("--high-regret-relative-threshold", type=float, default=0.01)
    parser.add_argument("--weak-margin-relative-threshold", type=float, default=0.001)
    parser.add_argument("--tiny-regression-mae", type=float, default=TINY_EXPECTED_MAE)
    parser.add_argument("--tiny-regression-atol", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timings: dict[str, float] = {}
    run_root = args.run_root.expanduser().resolve()
    checkpoint = (args.checkpoint or run_root / "episode_diagnostics" / f"ep_{args.episode:03d}" / f"{args.checkpoint_stage}.pt").expanduser().resolve()
    combined = (args.combined_checkpoint.expanduser().resolve() if args.combined_checkpoint else _discover_combined(run_root, args.episode))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    firm_data, auto_macro = choose_reference_artifacts(run_root) if args.firm_data is None else (args.firm_data.expanduser().resolve(), None)
    macro_data = args.macro_data.expanduser().resolve() if args.macro_data else auto_macro
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    with timed_block("checkpoint_load", timings):
        loaded = _checkpoint_payload(checkpoint, combined, device)
    model = loaded.models["policy_value"]
    sdf_fc1 = loaded.models["sdf_fc1"]
    model.eval(); sdf_fc1.eval()
    policy_hash_before = model_state_hash(model)
    sdf_hash_before = model_state_hash(sdf_fc1)
    with timed_block("reference_state_load", timings):
        _reference_frame, reference = load_reference_state(firm_data, macro_path=macro_data)
    specs = _branch_specs(model, reference)
    labels = [label for label in BRANCH_ORDER if label in args.branches]
    p0_loss, pi_loss = _losses(loaded)
    teacher = BPGridTeacher.from_hyperparams(model, p0_loss, pi_loss, loaded.hyperparams)
    banks: dict[str, StateBank] = {}
    all_frames: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    all_warnings: list[str] = []
    bank_metadata: dict[str, Any] = {}
    generated_figures: list[str] = []

    def run_bank(bank: StateBank, bank_labels: Sequence[str], n_shocks: int) -> pd.DataFrame:
        banks[bank.name] = bank
        with timed_block(f"{bank.name}:total", timings):
            state_level, grids, metadata, warnings = evaluate_state_bank(
                bank,
                labels=bank_labels,
                specs=specs,
                model=model,
                sdf_fc1=sdf_fc1,
                loaded=loaded,
                teacher=teacher,
                episode=args.episode,
                checkpoint_stage=args.checkpoint_stage,
                n_child_shocks=n_shocks,
                shock_seed=args.shock_seed,
                teacher_margin_tol=args.teacher_margin_tol,
                timings=timings,
            )
            summary, figures = write_bank_artifacts(
                bank,
                state_level,
                grids,
                metadata,
                output,
                large_gap_threshold=args.large_gap_threshold,
                high_regret_relative_threshold=args.high_regret_relative_threshold,
                weak_margin_relative_threshold=args.weak_margin_relative_threshold,
            )
        all_frames.append(state_level)
        all_summaries.append(summary)
        all_warnings.extend(warnings)
        bank_metadata[bank.name] = metadata
        generated_figures.extend(figures)
        return summary

    # The known nine-state probe is always evaluated first with J=8. A drift here
    # invalidates all later comparisons because it means the baseline changed.
    tiny = build_grid_bank(
        "tiny_smoke", reference, TINY_B_VALUES, TINY_Z_VALUES,
        eta=1.0, device=device,
    )
    tiny_summary = run_bank(tiny, ["pi_mid"], 8)
    tiny_primary = tiny_summary[
        (tiny_summary["branch"] == "pi_mid") & (tiny_summary["mask_type"] == "primary")
    ].iloc[0]
    tiny_mae = float(tiny_primary.bp_mae)
    validate_tiny_smoke_mae(
        tiny_mae, args.tiny_regression_mae, args.tiny_regression_atol,
    )

    dense_values = np.linspace(0.0, 1.0, int(args.dense_points))
    dense_z = np.linspace(-4.0, 4.0, int(args.dense_points))
    dense = build_grid_bank(
        "dense_grid", reference, dense_values, dense_z, eta=1.0, device=device,
    )
    run_bank(dense, labels, int(args.n_child_shocks))

    ondist_firm = (
        args.ondist_firm_data.expanduser().resolve()
        if args.ondist_firm_data else
        run_root / "data" / "outputs" / f"ep{args.episode}_stage_modeb.pkl"
    )
    ondist_macro = (
        args.ondist_macro_data.expanduser().resolve()
        if args.ondist_macro_data else
        ondist_firm.with_name(f"{ondist_firm.stem}_macro{ondist_firm.suffix}")
    )
    if not ondist_firm.is_file() or not ondist_macro.is_file():
        raise FileNotFoundError(
            f"EP{args.episode} on-distribution artifacts are required: {ondist_firm}, {ondist_macro}"
        )
    with timed_block("on_distribution:data_load", timings):
        ondist_frame, _ondist_reference = load_reference_state(ondist_firm, macro_path=ondist_macro)
    ondist = build_on_distribution_bank(
        ondist_frame,
        reference=reference,
        source=ondist_firm,
        macro_source=ondist_macro,
        max_states=args.ondist_max_states,
        seed=args.sample_seed,
        device=device,
    )
    run_bank(ondist, labels, int(args.n_child_shocks))

    presentation_info: dict[str, Any] = {
        "source_found": False,
        "exact_presentation_bank_reproduced": False,
    }
    if args.presentation_focus_csv is not None:
        presentation_csv = args.presentation_focus_csv.expanduser().resolve()
        if presentation_csv.is_file():
            original = pd.read_csv(presentation_csv)
            required = {"b", "z", "bp0_pred", "bp0_teacher", "bpI_pred", "bpI_teacher"}
            if not required.issubset(original.columns):
                raise ValueError(
                    f"presentation focus CSV is missing columns: {sorted(required - set(original.columns))}"
                )
            presentation_metadata_path = (
                args.presentation_metadata_json.expanduser().resolve()
                if args.presentation_metadata_json else
                presentation_csv.parent.parent / "00_manifest" / "metadata.json"
            )
            presentation_reference = reference
            presentation_source_metadata: dict[str, Any] = {}
            exact_reference_found = False
            if presentation_metadata_path.is_file():
                presentation_source_metadata = json.loads(
                    presentation_metadata_path.read_text(encoding="utf-8")
                )
                reference_firm = Path(presentation_source_metadata["reference_firm_pkl"])
                reference_macro = Path(presentation_source_metadata["reference_macro_pkl"])
                with timed_block("presentation_exact:reference_load", timings):
                    _presentation_frame, presentation_reference = load_reference_state(
                        reference_firm, macro_path=reference_macro,
                    )
                source_reference = presentation_source_metadata.get("reference", {})
                presentation_reference = replace(
                    presentation_reference,
                    i_mid=float(source_reference.get("i", presentation_reference.i_mid)),
                    x=float(source_reference.get("x", presentation_reference.x)),
                    hatcf=float(source_reference.get("hatcf", presentation_reference.hatcf)),
                    lnkf=float(source_reference.get("lnkf", presentation_reference.lnkf)),
                )
                exact_reference_found = True
            presentation = build_grid_bank(
                "presentation_exact",
                presentation_reference,
                sorted(original["b"].dropna().unique().tolist()),
                sorted(original["z"].dropna().unique().tolist()),
                eta=1.0,
                device=device,
                # This is an exact historical presentation semantic: transitions
                # were built from eta=0 states while teacher parents used eta=1.
                transition_eta_override=0.0,
            )
            presentation_labels = ["p0", "pi_mid"]
            presentation_summary = run_bank(
                presentation, presentation_labels, int(args.n_child_shocks),
            )
            presentation_state = all_frames[-1]
            reproduced_raw = presentation_summary[
                presentation_summary["mask_type"] == "raw_refi"
            ]
            p0_reproduced = presentation_state[
                presentation_state["branch"] == "p0"
            ].sort_values("bank_row")
            pi_reproduced = presentation_state[
                presentation_state["branch"] == "pi_mid"
            ].sort_values("bank_row")
            presentation_info = {
                "source_found": True,
                "exact_presentation_bank_reproduced": bool(
                    exact_reference_found and int(args.n_child_shocks) == 64
                ),
                "source_csv": str(presentation_csv),
                "source_metadata_json": str(presentation_metadata_path),
                "source_rows": int(len(original)),
                "source_semantics": (
                    "EP0 reference; transition parent eta=0; teacher/prediction parent eta=1; "
                    "original CSV used nearest saved candidate for regret; current reproduction "
                    "uses production bp_preds exact evaluation"
                ),
                "exact_reference_found": exact_reference_found,
                "source_reference": presentation_source_metadata.get("reference"),
                "original_p0_mae": float(np.mean(np.abs(original.bp0_pred - original.bp0_teacher))),
                "original_pi_mae": float(np.mean(np.abs(original.bpI_pred - original.bpI_teacher))),
                "reproduced_p0_mae": float(
                    reproduced_raw.loc[reproduced_raw.branch == "p0", "bp_mae"].iloc[0]
                ) if bool((reproduced_raw.branch == "p0").any()) else float("nan"),
                "reproduced_pi_mid_mae": float(
                    reproduced_raw.loc[reproduced_raw.branch == "pi_mid", "bp_mae"].iloc[0]
                ) if bool((reproduced_raw.branch == "pi_mid").any()) else float("nan"),
                "reproduction_mask": "raw_refi (all 30 eta=1 focus states, matching original scatter)",
                "p0_prediction_max_abs_diff_to_original": float(
                    np.max(np.abs(p0_reproduced.bp_pred.to_numpy() - original.bp0_pred.to_numpy()))
                ),
                "pi_prediction_max_abs_diff_to_original": float(
                    np.max(np.abs(pi_reproduced.bp_pred.to_numpy() - original.bpI_pred.to_numpy()))
                ),
                "p0_teacher_max_abs_diff_to_original": float(
                    np.max(np.abs(p0_reproduced.bp_star.to_numpy() - original.bp0_teacher.to_numpy()))
                ),
                "pi_teacher_max_abs_diff_to_original": float(
                    np.max(np.abs(pi_reproduced.bp_star.to_numpy() - original.bpI_teacher.to_numpy()))
                ),
                "original_n_child_shocks": 64,
                "reproduction_n_child_shocks": int(args.n_child_shocks),
                "known_source_commit": presentation_source_metadata.get(
                    "commit", "d6fe0ee73b1efa1298752ec2c3c3f09413cf25aa"
                ),
                "current_code_commit_expected": "f8e19ea2046566504884e134dd149b90630fd1cc",
            }
            generated_figures.append(
                str(write_presentation_reproduction(original, presentation_state, output))
            )

    state_level = pd.concat(all_frames, ignore_index=True)
    summary = pd.concat(all_summaries, ignore_index=True)
    atomic_write_csv(output / "tables" / "bp_fit_summary_by_bank.csv", summary)
    comparison_columns = [
        "bank", "branch", "mask_type", "n", "bp_mae", "pearson_r", "spearman_r",
        "std_ratio", "regret_scaled_mean", "regret_scaled_p90", "weak_rel_margin_1e3",
    ]
    atomic_write_csv(output / "tables" / "bank_comparison.csv", summary[comparison_columns])
    generated_figures.append(str(write_compression_figure(summary, output)))
    generated_figures.append(str(write_state_bank_coverage(banks, output)))
    dense_by_z = build_binned_summary(
        state_level[state_level["bank"] == "dense_grid"],
        column="z", edges=Z_BIN_EDGES, bin_column="z_bin",
    )
    write_summary_markdown(
        summary, dense_by_z, output, presentation=presentation_info,
    )
    policy_hash_after = model_state_hash(model)
    sdf_hash_after = model_state_hash(sdf_fc1)
    if policy_hash_before != policy_hash_after or sdf_hash_before != sdf_hash_after:
        raise RuntimeError("model state changed during read-only evaluation")
    metadata = {
        "episode": int(args.episode),
        "checkpoint_stage": args.checkpoint_stage,
        "checkpoint": str(checkpoint),
        "combined_checkpoint": str(combined),
        "firm_data": str(firm_data),
        "macro_data": None if macro_data is None else str(macro_data),
        "exact_presentation_bank_reproduced": bool(
            presentation_info.get("exact_presentation_bank_reproduced", False)
        ),
        "presentation_artifact_search": {
            "patterns": ["bpI pred vs teacher", "bp_pred_vs_teacher", "presentation_figures", "figure_manifest.csv"],
            "result": presentation_info,
        },
        "banks": bank_metadata,
        "ondist_firm_data": str(ondist_firm),
        "ondist_macro_data": str(ondist_macro),
        "sample_seed": int(args.sample_seed),
        "branch_semantics": {label: {"teacher_branch": specs[label][0], "i": specs[label][1]} for label in labels},
        "n_child_shocks": int(args.n_child_shocks),
        "shock_seed": int(args.shock_seed),
        "teacher_margin_tol": float(args.teacher_margin_tol),
        "mask_semantics": {
            "raw_refi": "refi_active",
            "survival_refi": "refi_active and finite(Phat) and Phat>0",
            "primary": "refi_active and finite(Phat) and Phat>0 and finite(top2_margin) and top2_margin>teacher_margin_tol",
        },
        "teacher_semantics": "BPGridTeacher.compute_multi_j_branches with bp_preds; production value_pred/regret/top2_margin",
        "regret_semantics": "clamp_min(value_star - value_pred, 0) from BPGridTeacher",
        "top2_margin_semantics": "coarse-grid best value minus second-best value",
        "separated_margin_semantics": "value_star - max coarse value at candidates with |bp-bp_star|>=delta",
        "thresholds": {
            "large_gap": float(args.large_gap_threshold),
            "high_regret_relative": float(args.high_regret_relative_threshold),
            "weak_margin_relative": float(args.weak_margin_relative_threshold),
        },
        "reference_state": reference.to_dict(),
        "warnings": all_warnings,
        "generated_figures": generated_figures,
        "timings_seconds": timings,
        "hashes": {
            "policy_value_before": policy_hash_before,
            "policy_value_after": policy_hash_after,
            "sdf_fc1_before": sdf_hash_before,
            "sdf_fc1_after": sdf_hash_after,
            "invariant": True,
        },
        "read_only_contract": {
            "training": False,
            "optimizer_constructed": False,
            "backward": False,
            "checkpoint_mutation": False,
            "economic_model_change": False,
        },
    }
    atomic_write_json(output / "metadata.json", metadata)
    print(f"Wrote BP teacher-fit diagnostics to {output}")
    print(summary[summary["mask_type"] == "primary"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
