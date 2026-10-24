"""Macro / SDF evaluator for the equilibrium convergence evaluator.

This module implements the macro-side and SDF-side evaluation block
("FC1/SDF Evaluator"): one-step macro forecast accuracy, recursive macro
stability under two explicitly separated calibres, and the SDF wealth-Euler /
conditional-moment / moment-constraint evaluation with standard errors and
numerical-safety accounting.

Economic semantics that this module must never blur (Phase A red lines):

* ``hatcf``/``lnkf`` (firm-state columns 5/6, macro belief columns) are the
  *forecast* of a node's macro state made one period earlier; ``Hatc``/``LnK``
  (macro realized aggregate) are the *realized* aggregate of the same node.
  Same-row ``hatcf_t`` vs ``Hatc_t`` is therefore exactly the one-step
  prediction-vs-realization comparison (``shift = 0``).  The two families are
  never mixed: pairing uses explicit ``(path, t, branch, economic_node_id)``
  keys taken from Bank D, never row adjacency.
* Recursive FC1 input follows the formal simulation rule: the FC1 call for
  node ``t`` is fed the *parent node's calculated macro state*
  (``hatc_cal_{t-1}``, ``lnk_cal_{t-1}``).  The two reported calibres are
  ``production_one_step`` (every call receives the realized/calculated macro
  state of its parent node -- the formal simulation call rule, errors do not
  compound) and ``autonomous_recursive`` (the model's own previous forecast is
  fed back, errors compound).  They live under disjoint metric prefixes and are
  never pooled.
* A negative **Signed AiO** sample mean is *not* evidence that the population
  conditional moment is negative.  The population-target estimate, its standard
  error and the numerical-safety constraints (non-finite ratios, clip shares)
  are reported separately and an explicit interpretation guard is recorded in
  the diagnostics.

Everything runs under ``torch.inference_mode()`` with every model in ``eval()``
mode and zero optimizer steps; the block is additionally wrapped in the
``param_hash_guard`` of ``evaluation.convergence.checkpoint_io`` so an accidental
parameter update during evaluation is detected.

Reused evaluation machinery (no equation is re-derived here):

* ``evaluation.full_run_diagnostics.evaluate_fc1_checkpoint`` -- one-step /
  timing-shift / rollout FC1 evaluation on a macro panel,
* ``evaluation.full_run_diagnostics.evaluate_sdf_heldout_multi_k`` and
  ``conditional_residual_summary`` -- nested-prefix SDF held-out evaluation,
* ``evaluation.convergence_metrics.regression_metrics`` /
  ``compute_shift_metrics`` -- regression and timing-shift metrics,
* ``losses.sdf_loss.SDFLoss.compute_wealth_residuals`` /
  ``compute_pooled_moment_constraints`` -- wealth Euler residuals and the
  economic moment constraints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from analysis.convergence_transition import ConvergenceShockBank
from evaluation.convergence_metrics import compute_shift_metrics, regression_metrics
from evaluation.full_run_diagnostics import (
    conditional_residual_summary,
    evaluate_fc1_checkpoint,
    evaluate_sdf_heldout_multi_k,
)
from losses.sdf_loss import SDFLoss, compute_pooled_moment_constraints

from .checkpoint_io import EvalCheckpoint, count_optimizer_steps, param_hash_guard


MACRO_STATE_LABELS = ("recession", "normal", "expansion")
CAPITAL_STATE_LABELS = ("small", "medium", "large")

RECURSION_CALIBER_PRODUCTION = "production_one_step"
RECURSION_CALIBER_AUTONOMOUS = "autonomous_recursive"
RECURSION_CALIBERS = (RECURSION_CALIBER_PRODUCTION, RECURSION_CALIBER_AUTONOMOUS)

_BANK_D_COLUMNS = (
    "x_t",
    "x_t1",
    "hatcf_t",
    "lnkf_t",
    "hatc_t",
    "lnk_t",
    "hatc_t1",
    "lnk_t1",
)

REASON_BANK_D_COLUMNS = "BANK_D_MISSING_COLUMNS"
REASON_BANK_D_DUPLICATE_KEY = "BANK_D_DUPLICATE_KEY"
REASON_BANK_D_EMPTY = "BANK_D_EMPTY"


class MacroSdfEvalError(ValueError):
    """Macro/SDF evaluation error carrying a stable machine reason code."""

    def __init__(self, reason_code: str, message: str):
        super().__init__(f"[{reason_code}] {message}")
        self.reason_code = reason_code
        self.message = message


@dataclass
class MacroSdfEvalConfig:
    """Tunables of the macro/SDF evaluation block (CPU-friendly defaults)."""

    device: str = "cpu"
    sdf_child_counts: Tuple[int, ...] = (2, 8, 32)
    sdf_shock_bank_max_children: Optional[int] = None
    sdf_seed: int = 20261025
    normalized_logr_clip: Optional[float] = None
    recursion_horizons: Tuple[int, ...] = (1, 2, 5, 10, 20, 50)
    recursion_path_horizon: int = 50
    anchor_x_mode: str = "constant"
    divergence_abs_bound: float = 1.0e3
    degenerate_std_tol: float = 1.0e-9
    macro_state_quantiles: Tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
    capital_state_quantiles: Tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
    rollout_horizons: Tuple[int, ...] = (1, 5, 10, 20)
    shifts: Tuple[int, ...] = (-2, -1, 0, 1, 2)

    def __post_init__(self) -> None:
        if self.anchor_x_mode not in {"constant", "bank_d"}:
            raise MacroSdfEvalError(
                "INVALID_ANCHOR_X_MODE",
                f"anchor_x_mode must be 'constant' or 'bank_d', got {self.anchor_x_mode!r}",
            )


@dataclass
class MacroSdfEvalResult:
    """Result of the macro/SDF evaluation block."""

    metrics: Dict[str, float] = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    surfaces: Dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------


def _to_float_array(values: Any) -> np.ndarray:
    series = pd.Series(np.asarray(values).reshape(-1))
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def _nan() -> float:
    return float("nan")


def _safe_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else _nan()


def _safe_std(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return _nan()
    return float(finite.std(ddof=0))


def _safe_se(values: np.ndarray) -> float:
    """Standard error of a sample mean (population std / sqrt(n))."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return _nan()
    return float(finite.std(ddof=0) / math.sqrt(float(finite.size)))


def error_metrics(calculated: Any, forecast: Any, *, prefix: str) -> Dict[str, float]:
    """MAE/RMSE/Bias/P90/R^2/regression slope of ``forecast`` vs ``calculated``.

    ``regression_metrics`` is reused for the regression block so R^2 / slope /
    intercept / rmse / mae keep the code-base semantics; bias and absolute-error
    quantiles are added on top.
    """
    calc = _to_float_array(calculated)
    fore = _to_float_array(forecast)
    if calc.shape != fore.shape:
        raise MacroSdfEvalError(
            "SHAPE_MISMATCH",
            f"error_metrics requires identically shaped inputs for {prefix!r}: "
            f"{calc.shape} vs {fore.shape}",
        )
    mask = np.isfinite(calc) & np.isfinite(fore)
    calc_f, fore_f = calc[mask], fore[mask]
    out: Dict[str, float] = {}
    base = regression_metrics(
        calc_f, fore_f, calculated_name="calculated", forecast_name="forecast"
    )
    for key, value in base.items():
        out[f"{prefix}_{key}"] = float(value)
    err = fore_f - calc_f
    absolute = np.abs(err)
    out[f"{prefix}_bias"] = float(err.mean()) if err.size else _nan()
    out[f"{prefix}_p90_abs"] = float(np.quantile(absolute, 0.90)) if err.size else _nan()
    out[f"{prefix}_p95_abs"] = float(np.quantile(absolute, 0.95)) if err.size else _nan()
    out[f"{prefix}_max_abs"] = float(absolute.max()) if err.size else _nan()
    out[f"{prefix}_nonfinite_ratio"] = float(1.0 - mask.mean()) if calc.size else _nan()
    return out


def state_cutpoints(values: Any, quantiles: Sequence[float]) -> List[float]:
    array = _to_float_array(values)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return [_nan() for _ in quantiles]
    return [float(np.quantile(finite, float(q))) for q in quantiles]


def state_labels(values: Any, cutpoints: Sequence[float], labels: Sequence[str]) -> np.ndarray:
    array = _to_float_array(values)
    out = np.empty(array.shape[0], dtype=object)
    out[:] = ""
    edges = [float(value) for value in cutpoints]
    if any(not math.isfinite(edge) for edge in edges):
        out[:] = str(labels[len(labels) // 2])
        return out.astype(str)
    bins = np.digitize(array, edges, right=True)
    for index, label in enumerate(labels):
        out[bins == index] = str(label)
    out[~np.isfinite(array)] = ""
    return out.astype(str)


# ---------------------------------------------------------------------------
# Bank D frame
# ---------------------------------------------------------------------------


def bank_d_frame(bank_d: Mapping[str, Any]) -> pd.DataFrame:
    """Materialize Bank D macro parent-child rows as a paired DataFrame.

    One row is one keyed macro transition ``(path, t, branch, economic_node_id)
    -> (path, t + 1, ...)``; the pairing keys come from the bank payload and are
    the only allowed pairing channel (row adjacency is forbidden).
    """
    missing = [name for name in _BANK_D_COLUMNS if name not in bank_d]
    if missing:
        raise MacroSdfEvalError(
            REASON_BANK_D_COLUMNS, f"Bank D payload is missing columns: {missing}"
        )
    n_rows = len(bank_d["x_t"])
    if n_rows == 0:
        raise MacroSdfEvalError(REASON_BANK_D_EMPTY, "Bank D payload has no rows")
    for name in _BANK_D_COLUMNS:
        if len(bank_d[name]) != n_rows:
            raise MacroSdfEvalError(
                REASON_BANK_D_COLUMNS,
                f"Bank D column {name!r} has {len(bank_d[name])} rows, expected {n_rows}",
            )

    frame = pd.DataFrame({name: _to_float_array(bank_d[name]) for name in _BANK_D_COLUMNS})
    keys = dict(bank_d.get("pairing_keys") or {})
    for name in ("path", "t", "branch", "economic_node_id", "t1", "economic_node_id_child"):
        if name in keys and len(keys[name]) == n_rows:
            frame[name] = pd.to_numeric(pd.Series(list(keys[name])), errors="coerce")
    for name, values in (
        ("path", np.arange(n_rows, dtype=np.float64)),
        ("t", np.zeros(n_rows, dtype=np.float64)),
        ("branch", np.zeros(n_rows, dtype=np.float64)),
        ("economic_node_id", np.arange(n_rows, dtype=np.float64)),
    ):
        if name not in frame.columns:
            frame[name] = values
    if "t1" not in frame.columns:
        frame["t1"] = frame["t"] + 1.0

    key_columns = ["path", "t", "branch", "economic_node_id"]
    duplicated = frame.duplicated(subset=key_columns, keep=False)
    if bool(duplicated.any()):
        raise MacroSdfEvalError(
            REASON_BANK_D_DUPLICATE_KEY,
            f"Bank D has duplicated pairing keys {key_columns}: "
            f"{int(duplicated.sum())} rows; pairing must be unambiguous",
        )
    return frame


def macro_node_series(pair_frame: pd.DataFrame) -> Tuple[Dict[int, pd.DataFrame], int]:
    """Per-path node series ``{path: DataFrame[node_t, x, hatc, lnk]}``.

    Built from the paired rows only (parent node ``t`` and child node ``t1``);
    duplicate ``(path, node_t)`` nodes keep the first occurrence and their count
    is returned for the caller's diagnostics.
    """
    parent = pair_frame[["path", "t", "x_t", "hatc_t", "lnk_t"]].rename(
        columns={"t": "node_t", "x_t": "x", "hatc_t": "hatc", "lnk_t": "lnk"}
    )
    child = pair_frame[["path", "t1", "x_t1", "hatc_t1", "lnk_t1"]].rename(
        columns={"t1": "node_t", "x_t1": "x", "hatc_t1": "hatc", "lnk_t1": "lnk"}
    )
    nodes = pd.concat([parent, child], ignore_index=True)
    nodes = nodes.sort_values(["path", "node_t"], kind="mergesort").reset_index(drop=True)
    conflicts = nodes.duplicated(subset=["path", "node_t"], keep=False)
    n_conflicts = int(conflicts.sum())
    nodes = nodes.drop_duplicates(subset=["path", "node_t"], keep="first").reset_index(drop=True)
    series: Dict[int, pd.DataFrame] = {}
    for path, group in nodes.groupby("path", sort=True):
        series[int(path)] = group.sort_values("node_t", kind="mergesort").reset_index(drop=True)
    return series, n_conflicts


# ---------------------------------------------------------------------------
# FC1 one-step (recorded belief vs realized; shift = 0)
# ---------------------------------------------------------------------------


def _stratified_error_metrics(
    frame: pd.DataFrame,
    *,
    calculated: str,
    forecast: str,
    prefix: str,
    group_column: str,
    labels: Sequence[str],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label in labels:
        subset = frame.loc[frame[group_column] == str(label)]
        if subset.empty:
            continue
        out.update(
            error_metrics(subset[calculated], subset[forecast], prefix=f"{prefix}_{label}")
        )
    return out


def fc1_one_step_belief_metrics(
    pair_frame: pd.DataFrame,
    *,
    config: MacroSdfEvalConfig,
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, Any]]:
    """``Hatcf_t`` vs ``Hatc_t`` and ``LnKF_t`` vs ``LnK_t`` on Bank D (shift=0).

    The belief columns are the forecast formed at ``t - 1`` for node ``t``; the
    realized columns are the same node's aggregate.  Alignment is same-node
    same-``t`` and comes straight from the Bank D key pairing.
    """
    frame = pair_frame.copy()
    macro_cut = state_cutpoints(frame["x_t"], config.macro_state_quantiles)
    capital_cut = state_cutpoints(frame["lnk_t"], config.capital_state_quantiles)
    frame["macro_state"] = state_labels(frame["x_t"], macro_cut, MACRO_STATE_LABELS)
    frame["capital_state"] = state_labels(frame["lnk_t"], capital_cut, CAPITAL_STATE_LABELS)

    metrics: Dict[str, float] = {}
    for name, belief, realized in (
        ("hatc", "hatcf_t", "hatc_t"),
        ("lnk", "lnkf_t", "lnk_t"),
    ):
        prefix = f"fc1_one_step_{name}"
        metrics.update(error_metrics(frame[realized], frame[belief], prefix=prefix))
        metrics.update(
            _stratified_error_metrics(
                frame,
                calculated=realized,
                forecast=belief,
                prefix=prefix,
                group_column="macro_state",
                labels=MACRO_STATE_LABELS,
            )
        )
        for label in MACRO_STATE_LABELS:
            for suffix in ("n", "mae", "rmse", "bias", "p90_abs", "r2", "slope", "intercept"):
                source = f"{prefix}_{label}_{suffix}"
                if source in metrics:
                    metrics[f"{prefix}_{suffix}_{label}"] = metrics.pop(source)
        for label in CAPITAL_STATE_LABELS:
            subset = frame.loc[frame["capital_state"] == str(label)]
            if subset.empty:
                continue
            metrics.update(
                error_metrics(
                    subset[realized],
                    subset[belief],
                    prefix=f"{prefix}_cap_{label}",
                )
            )
        for label in CAPITAL_STATE_LABELS:
            for suffix in (
                "n",
                "mae",
                "rmse",
                "bias",
                "p90_abs",
                "r2",
                "slope",
                "intercept",
            ):
                source = f"{prefix}_cap_{label}_{suffix}"
                if source in metrics:
                    metrics[f"{prefix}_{suffix}_cap_{label}"] = metrics.pop(source)

    diagnostics = {
        "fc1_pairing": {
            "pair_keys": ["path", "t", "branch", "economic_node_id"],
            "n_pairs": int(len(frame)),
            "shift": 0,
            "semantics": (
                "hatcf_t/lnkf_t is the t-1 forecast of node t; hatc_t/lnk_t is "
                "node t realized aggregate"
            ),
        },
        "macro_state_cutpoints_x_t": macro_cut,
        "capital_state_cutpoints_lnk_t": capital_cut,
    }
    return metrics, frame, diagnostics


def fc1_shift_alignment(
    pair_frame: pd.DataFrame, *, config: MacroSdfEvalConfig
) -> Tuple[Dict[str, float], Dict[str, pd.DataFrame]]:
    """Timing-shift alignment of the recorded belief against realized values.

    ``shift = 0`` is same-node alignment; ``shift = -1`` means the belief tracks
    the previous node's realized value (persistence).  Uses
    ``evaluation.convergence_metrics.compute_shift_metrics``.
    """
    metrics: Dict[str, float] = {}
    tables: Dict[str, pd.DataFrame] = {}
    unique = (
        pair_frame.sort_values(["path", "t"], kind="mergesort")
        .drop_duplicates(subset=["path", "t"], keep="first")
        .reset_index(drop=True)
    )
    for name, belief, realized in (
        ("hatc", "hatcf_t", "hatc_t"),
        ("lnk", "lnkf_t", "lnk_t"),
    ):
        frame = unique[["path", "t", belief, realized]].copy()
        table = compute_shift_metrics(
            frame,
            forecast_col=belief,
            calculated_col=realized,
            shifts=config.shifts,
        )
        tables[f"fc1_shift_{name}"] = table
        finite = table[np.isfinite(table["rmse"])].copy()
        if not finite.empty:
            best = finite.loc[finite["rmse"].idxmin()]
            metrics[f"fc1_shift_{name}_best_shift"] = float(best["shift"])
            metrics[f"fc1_shift_{name}_best_rmse"] = float(best["rmse"])
        shift0 = table.loc[table["shift"] == 0]
        if not shift0.empty:
            metrics[f"fc1_shift_{name}_shift0_rmse"] = float(shift0.iloc[0]["rmse"])
            metrics[f"fc1_shift_{name}_shift0_mae"] = float(shift0.iloc[0]["mae"])
    return metrics, tables


def fc1_model_one_step(
    cp: EvalCheckpoint,
    pair_frame: pd.DataFrame,
    *,
    config: MacroSdfEvalConfig,
) -> Dict[str, float]:
    """Fresh one-step forecasts of the checkpoint's own FC1 on Bank D pairs.

    The FC1 call for the child node is fed the *parent node's calculated macro
    state* exactly as the formal simulation does:
    ``forward_step(x_t, x_t1, hatc_t, lnk_t) -> (.., hatc_pred, lnk_pred)`` and
    the target is the child node's realized ``(hatc_t1, lnk_t1)``.
    """
    model = cp.models["sdf_fc1"]
    device = torch.device(config.device)
    x_prev = torch.as_tensor(
        pair_frame["x_t"].to_numpy(dtype=np.float32), device=device
    ).reshape(-1, 1)
    x_curr = torch.as_tensor(
        pair_frame["x_t1"].to_numpy(dtype=np.float32), device=device
    ).reshape(-1, 1)
    hatc_prev = torch.as_tensor(
        pair_frame["hatc_t"].to_numpy(dtype=np.float32), device=device
    ).reshape(-1, 1)
    lnk_prev = torch.as_tensor(
        pair_frame["lnk_t"].to_numpy(dtype=np.float32), device=device
    ).reshape(-1, 1)
    with torch.inference_mode():
        _, _, _, hatc_pred, lnk_pred = model.forward_step(
            x_prev, x_curr, hatc_prev, lnk_prev
        )
    metrics: Dict[str, float] = {}
    metrics.update(
        error_metrics(
            pair_frame["hatc_t1"],
            hatc_pred.detach().cpu().reshape(-1).numpy(),
            prefix="fc1_model_one_step_hatc",
        )
    )
    metrics.update(
        error_metrics(
            pair_frame["lnk_t1"],
            lnk_pred.detach().cpu().reshape(-1).numpy(),
            prefix="fc1_model_one_step_lnk",
        )
    )
    return metrics


def fc1_evaluator_pass(
    cp: EvalCheckpoint,
    pair_frame: pd.DataFrame,
    *,
    config: MacroSdfEvalConfig,
) -> Tuple[Dict[str, float], Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Reuse ``evaluation.full_run_diagnostics.evaluate_fc1_checkpoint``.

    A macro frame is rebuilt from the Bank D pairs (one row per ``(path, t)``
    node, first occurrence kept) so the evaluator's own one-step / timing-shift /
    rollout analysis runs on the holdout paths.
    """
    parent = pair_frame[["path", "t", "x_t", "hatc_t", "lnk_t"]].rename(
        columns={"x_t": "x", "hatc_t": "hatc_cal", "lnk_t": "lnk_cal"}
    )
    child = pair_frame[["path", "t1", "x_t1", "hatc_t1", "lnk_t1"]].rename(
        columns={"t1": "t", "x_t1": "x", "hatc_t1": "hatc_cal", "lnk_t1": "lnk_cal"}
    )
    macro_frame = pd.concat([parent, child], ignore_index=True)
    macro_frame = macro_frame.sort_values(["path", "t"], kind="mergesort")
    duplicate_rows = int(
        macro_frame.duplicated(subset=["path", "t"], keep=False).sum()
    )
    macro_frame = macro_frame.drop_duplicates(
        subset=["path", "t"], keep="first"
    ).reset_index(drop=True)

    summary, timing, rollout = evaluate_fc1_checkpoint(
        cp.models["sdf_fc1"],
        macro_frame,
        device=torch.device(config.device),
        rollout_horizons=tuple(config.rollout_horizons),
        shifts=tuple(config.shifts),
    )
    metrics = {f"fc1_eval_{key}": float(value) for key, value in summary.items()}
    tables = {
        "fc1_eval_timing": timing,
        "fc1_eval_rollout": rollout,
    }
    diagnostics = {
        "fc1_evaluator_macro_frame_duplicate_rows_dropped": duplicate_rows,
        "fc1_evaluator_source": (
            "evaluation.full_run_diagnostics.evaluate_fc1_checkpoint"
        ),
    }
    return metrics, tables, diagnostics


# ---------------------------------------------------------------------------
# FC1 recursive stability -- two calibres, reported separately
# ---------------------------------------------------------------------------


def _recursion_predictions(
    model: torch.nn.Module,
    *,
    x_path: np.ndarray,
    hatc_start: float,
    lnk_start: float,
    n_steps: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Autonomous recursion: feed the model's own previous forecast back."""
    hatc = np.full(int(n_steps) + 1, np.nan, dtype=np.float64)
    lnk = np.full(int(n_steps) + 1, np.nan, dtype=np.float64)
    hatc[0] = float(hatc_start)
    lnk[0] = float(lnk_start)
    for step in range(1, int(n_steps) + 1):
        x_prev = torch.tensor([[float(x_path[step - 1])]], dtype=torch.float32, device=device)
        x_curr = torch.tensor([[float(x_path[step])]], dtype=torch.float32, device=device)
        hatc_prev = torch.tensor([[float(hatc[step - 1])]], dtype=torch.float32, device=device)
        lnk_prev = torch.tensor([[float(lnk[step - 1])]], dtype=torch.float32, device=device)
        with torch.inference_mode():
            _, _, _, hatc_next, lnk_next = model.forward_step(
                x_prev, x_curr, hatc_prev, lnk_prev
            )
        hatc[step] = float(hatc_next.reshape(-1)[0].item())
        lnk[step] = float(lnk_next.reshape(-1)[0].item())
    return hatc, lnk


def _one_step_prediction(
    model: torch.nn.Module,
    *,
    x_prev: float,
    x_curr: float,
    hatc_prev: float,
    lnk_prev: float,
    device: torch.device,
) -> Tuple[float, float]:
    """One production FC1 call: parent node's *realized* macro state as input."""
    with torch.inference_mode():
        _, _, _, hatc_next, lnk_next = model.forward_step(
            torch.tensor([[float(x_prev)]], dtype=torch.float32, device=device),
            torch.tensor([[float(x_curr)]], dtype=torch.float32, device=device),
            torch.tensor([[float(hatc_prev)]], dtype=torch.float32, device=device),
            torch.tensor([[float(lnk_prev)]], dtype=torch.float32, device=device),
        )
    return float(hatc_next.reshape(-1)[0].item()), float(lnk_next.reshape(-1)[0].item())


def fc1_recursive_stability(
    cp: EvalCheckpoint,
    pair_frame: pd.DataFrame,
    bank_c_anchors: Optional[Mapping[str, Mapping[str, float]]],
    *,
    config: MacroSdfEvalConfig,
) -> Tuple[Dict[str, float], Dict[str, pd.DataFrame], Dict[str, np.ndarray], Dict[str, Any]]:
    """Recursive FC1 stability in the two calibres.

    ``production_one_step``: every FC1 call is fed the *realized/calculated*
    macro state of its parent node -- the formal simulation call rule -- so the
    horizon-``h`` error is the one-step error of the call forecasting node
    ``t0 + h`` and does not compound.

    ``autonomous_recursive``: the model's own previous forecast is fed back from
    the origin node's realized macro state, so errors compound with the horizon.

    Both report 1/2/5/10/20/50-period errors, long-run drift, out-of-training
    range share and divergence / degenerate-constant detection; anchor recursion
    paths start from the Bank C macro anchors (Recession/Normal/Expansion).
    """
    model = cp.models["sdf_fc1"]
    device = torch.device(config.device)
    series, n_conflicts = macro_node_series(pair_frame)
    horizons = sorted({int(h) for h in config.recursion_horizons if int(h) > 0})
    if not horizons:
        raise MacroSdfEvalError(
            "INVALID_HORIZONS", "recursion_horizons must contain positive integers"
        )
    max_horizon = max(horizons)

    all_hatc = _to_float_array(
        pair_frame["hatc_t"].tolist() + pair_frame["hatc_t1"].tolist()
    )
    all_lnk = _to_float_array(pair_frame["lnk_t"].tolist() + pair_frame["lnk_t1"].tolist())
    all_x = _to_float_array(pair_frame["x_t"].tolist() + pair_frame["x_t1"].tolist())

    def _bounds(values: np.ndarray) -> Tuple[float, float]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return (_nan(), _nan())
        return (float(finite.min()), float(finite.max()))

    training_range = {
        "x": _bounds(all_x),
        "hatc": _bounds(all_hatc),
        "lnk": _bounds(all_lnk),
    }

    def _out_of_range(values: np.ndarray, bounds: Tuple[float, float]) -> float:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return _nan()
        low, high = bounds
        if not (math.isfinite(low) and math.isfinite(high)):
            return _nan()
        return float(np.mean((finite < low) | (finite > high)))

    errors: Dict[Tuple[str, str, int], List[float]] = {}
    range_rows: List[Dict[str, float]] = []
    for path, nodes in series.items():
        x_values = nodes["x"].to_numpy(dtype=np.float64)
        hatc_values = nodes["hatc"].to_numpy(dtype=np.float64)
        lnk_values = nodes["lnk"].to_numpy(dtype=np.float64)
        n_nodes = int(len(nodes))
        for origin in range(0, n_nodes - 1):
            reachable = min(max_horizon, n_nodes - 1 - origin)
            if reachable <= 0:
                continue
            for horizon in horizons:
                if horizon > reachable:
                    continue
                hatc_pred, lnk_pred = _one_step_prediction(
                    model,
                    x_prev=float(x_values[origin + horizon - 1]),
                    x_curr=float(x_values[origin + horizon]),
                    hatc_prev=float(hatc_values[origin + horizon - 1]),
                    lnk_prev=float(lnk_values[origin + horizon - 1]),
                    device=device,
                )
                errors.setdefault(
                    (RECURSION_CALIBER_PRODUCTION, "hatc", horizon), []
                ).append(hatc_pred - float(hatc_values[origin + horizon]))
                errors.setdefault(
                    (RECURSION_CALIBER_PRODUCTION, "lnk", horizon), []
                ).append(lnk_pred - float(lnk_values[origin + horizon]))

            hatc_path, lnk_path = _recursion_predictions(
                model,
                x_path=x_values[origin : origin + reachable + 1],
                hatc_start=float(hatc_values[origin]),
                lnk_start=float(lnk_values[origin]),
                n_steps=reachable,
                device=device,
            )
            for horizon in horizons:
                if horizon > reachable:
                    continue
                errors.setdefault(
                    (RECURSION_CALIBER_AUTONOMOUS, "hatc", horizon), []
                ).append(float(hatc_path[horizon]) - float(hatc_values[origin + horizon]))
                errors.setdefault(
                    (RECURSION_CALIBER_AUTONOMOUS, "lnk", horizon), []
                ).append(float(lnk_path[horizon]) - float(lnk_values[origin + horizon]))

            combined_finite = np.isfinite(hatc_path) & np.isfinite(lnk_path)
            divergent = bool(
                (~combined_finite).any()
                or float(np.nanmax(np.abs(hatc_path))) > config.divergence_abs_bound
                or float(np.nanmax(np.abs(lnk_path))) > config.divergence_abs_bound
            )
            range_rows.append(
                {
                    "path": float(path),
                    "origin_t": float(nodes["node_t"].iloc[origin]),
                    "caliber": RECURSION_CALIBER_AUTONOMOUS,
                    "hatc_out_of_range_share": _out_of_range(
                        hatc_path, training_range["hatc"]
                    ),
                    "lnk_out_of_range_share": _out_of_range(
                        lnk_path, training_range["lnk"]
                    ),
                    "x_out_of_range_share": _out_of_range(
                        x_values[origin : origin + reachable + 1], training_range["x"]
                    ),
                    "nonfinite_share": float(np.mean(~combined_finite)),
                    "diverged": float(int(divergent)),
                    "degenerate_constant": float(
                        int(
                            _safe_std(hatc_path) <= config.degenerate_std_tol
                            and _safe_std(lnk_path) <= config.degenerate_std_tol
                        )
                    ),
                }
            )

    metrics: Dict[str, float] = {}
    rows: List[Dict[str, float]] = []
    for caliber in RECURSION_CALIBERS:
        for variable in ("hatc", "lnk"):
            for horizon in horizons:
                values = np.asarray(
                    errors.get((caliber, variable, horizon), []), dtype=np.float64
                )
                finite = values[np.isfinite(values)]
                absolute = np.abs(finite)
                key = f"fc1_rec_{caliber}_{variable}_h{horizon}"
                metrics[f"{key}_n"] = float(values.size)
                metrics[f"{key}_mae"] = float(absolute.mean()) if absolute.size else _nan()
                metrics[f"{key}_rmse"] = (
                    float(np.sqrt(np.mean(np.square(finite)))) if finite.size else _nan()
                )
                metrics[f"{key}_bias"] = _safe_mean(values)
                metrics[f"{key}_p90_abs"] = (
                    float(np.quantile(absolute, 0.90)) if absolute.size else _nan()
                )
                rows.append(
                    {
                        "caliber": caliber,
                        "variable": variable,
                        "horizon": float(horizon),
                        "n": float(values.size),
                        "mae": metrics[f"{key}_mae"],
                        "rmse": metrics[f"{key}_rmse"],
                        "bias": metrics[f"{key}_bias"],
                    }
                )
            bias_by_h = [
                metrics[f"fc1_rec_{caliber}_{variable}_h{horizon}_bias"]
                for horizon in horizons
            ]
            finite_pairs = [
                (float(h), float(b))
                for h, b in zip(horizons, bias_by_h)
                if math.isfinite(float(b))
            ]
            if len(finite_pairs) >= 2:
                hs = np.asarray([pair[0] for pair in finite_pairs], dtype=np.float64)
                bs = np.asarray([pair[1] for pair in finite_pairs], dtype=np.float64)
                design = np.column_stack([np.ones_like(hs), hs])
                intercept, slope = np.linalg.lstsq(design, bs, rcond=None)[0]
                metrics[f"fc1_rec_{caliber}_{variable}_drift_slope"] = float(slope)
                metrics[f"fc1_rec_{caliber}_{variable}_drift_intercept"] = float(intercept)
            else:
                metrics[f"fc1_rec_{caliber}_{variable}_drift_slope"] = _nan()
                metrics[f"fc1_rec_{caliber}_{variable}_drift_intercept"] = _nan()
            metrics[f"fc1_rec_{caliber}_{variable}_drift_at_max_h"] = float(bias_by_h[-1])

    range_frame = pd.DataFrame(range_rows)
    for caliber in RECURSION_CALIBERS:
        if range_frame.empty:
            continue
        subset = range_frame.loc[range_frame["caliber"] == caliber]
        if subset.empty:
            continue
        for column, key in (
            ("hatc_out_of_range_share", "hatc_out_of_training_range_share"),
            ("lnk_out_of_range_share", "lnk_out_of_training_range_share"),
            ("x_out_of_range_share", "x_out_of_training_range_share"),
            ("nonfinite_share", "nonfinite_share"),
            ("diverged", "diverged_share"),
            ("degenerate_constant", "degenerate_constant_share"),
        ):
            metrics[f"fc1_rec_{caliber}_{key}"] = _safe_mean(
                subset[column].to_numpy(dtype=np.float64)
            )

    # Anchor recursion paths from the Bank C macro anchors.
    anchor_tables: Dict[str, pd.DataFrame] = {}
    surfaces: Dict[str, np.ndarray] = {}
    x_pool = _to_float_array(pair_frame["x_t"].tolist())
    for label, anchor in dict(bank_c_anchors or {}).items():
        start_hatc = float(anchor.get("hatc_cal", anchor.get("hatc", _nan())))
        start_lnk = float(anchor.get("lnk_cal", anchor.get("lnk", _nan())))
        anchor_x = float(anchor.get("x", _nan()))
        n_steps = int(config.recursion_path_horizon)
        if config.anchor_x_mode == "bank_d" and x_pool.size:
            repeats = int(math.ceil((n_steps + 1) / max(x_pool.size, 1)))
            x_path = np.tile(x_pool, repeats)[: n_steps + 1]
        else:
            x_path = np.full(n_steps + 1, anchor_x, dtype=np.float64)
        hatc_path, lnk_path = _recursion_predictions(
            model,
            x_path=x_path,
            hatc_start=start_hatc,
            lnk_start=start_lnk,
            n_steps=n_steps,
            device=device,
        )
        slug = str(label).strip().lower().replace(" ", "_")
        table = pd.DataFrame(
            {
                "step": np.arange(n_steps + 1, dtype=np.float64),
                "x": x_path,
                "hatc_pred": hatc_path,
                "lnk_pred": lnk_path,
            }
        )
        table["hatc_in_training_range"] = (
            (table["hatc_pred"] >= training_range["hatc"][0])
            & (table["hatc_pred"] <= training_range["hatc"][1])
        ).astype(float)
        table["lnk_in_training_range"] = (
            (table["lnk_pred"] >= training_range["lnk"][0])
            & (table["lnk_pred"] <= training_range["lnk"][1])
        ).astype(float)
        anchor_tables[f"fc1_anchor_recursion_{slug}"] = table
        surfaces[f"fc1_anchor_recursion_{slug}"] = np.stack([hatc_path, lnk_path], axis=1)

        combined_finite = np.isfinite(hatc_path) & np.isfinite(lnk_path)
        divergent = bool(
            (~combined_finite).any()
            or float(np.nanmax(np.abs(hatc_path))) > config.divergence_abs_bound
            or float(np.nanmax(np.abs(lnk_path))) > config.divergence_abs_bound
        )
        metrics[f"fc1_anchor_{slug}_hatc_final"] = float(hatc_path[-1])
        metrics[f"fc1_anchor_{slug}_lnk_final"] = float(lnk_path[-1])
        metrics[f"fc1_anchor_{slug}_hatc_drift_per_step"] = (
            float((hatc_path[-1] - hatc_path[0]) / max(n_steps, 1))
            if bool(np.isfinite(hatc_path).all())
            else _nan()
        )
        metrics[f"fc1_anchor_{slug}_lnk_drift_per_step"] = (
            float((lnk_path[-1] - lnk_path[0]) / max(n_steps, 1))
            if bool(np.isfinite(lnk_path).all())
            else _nan()
        )
        metrics[f"fc1_anchor_{slug}_diverged"] = float(int(divergent))
        metrics[f"fc1_anchor_{slug}_degenerate_constant"] = float(
            int(
                _safe_std(hatc_path) <= config.degenerate_std_tol
                and _safe_std(lnk_path) <= config.degenerate_std_tol
            )
        )
        metrics[f"fc1_anchor_{slug}_out_of_training_range_share"] = _safe_mean(
            np.concatenate(
                [
                    1.0 - table["hatc_in_training_range"].to_numpy(dtype=np.float64),
                    1.0 - table["lnk_in_training_range"].to_numpy(dtype=np.float64),
                ]
            )
        )
        metrics[f"fc1_anchor_{slug}_nonfinite_share"] = float(np.mean(~combined_finite))

    tables = {
        "fc1_recursion_errors": pd.DataFrame(rows),
        **anchor_tables,
    }
    diagnostics = {
        "fc1_recursion_calibres": {
            RECURSION_CALIBER_PRODUCTION: (
                "every FC1 call is fed the realized/calculated macro state of its "
                "parent node (formal simulation call rule); the horizon-h error is "
                "the one-step error of the call forecasting node t0+h and does not "
                "compound"
            ),
            RECURSION_CALIBER_AUTONOMOUS: (
                "the model's own previous forecast is fed back from the origin "
                "node's realized macro state; errors compound with the horizon"
            ),
            "pooled": False,
        },
        "fc1_training_range": training_range,
        "fc1_recursion_horizons": horizons,
        "bank_d_node_conflicts": int(n_conflicts),
        "anchor_x_mode": config.anchor_x_mode,
    }
    return metrics, tables, surfaces, diagnostics


# ---------------------------------------------------------------------------
# SDF block
# ---------------------------------------------------------------------------


def _sdf_detail_pass(
    cp: EvalCheckpoint,
    pair_frame: pd.DataFrame,
    *,
    config: MacroSdfEvalConfig,
) -> Tuple[Dict[str, float], Dict[str, pd.DataFrame], Dict[str, Any]]:
    """SDF detail pass: standard errors, conditional distributions, safety.

    The shock bank is rebuilt from the same ``(n_parent, bank_size, seed)`` as
    :func:`evaluate_sdf_heldout_multi_k`, so this pass observes exactly the same
    fresh shock pairs as the canonical summary.
    """
    model = cp.models["sdf_fc1"]
    device = torch.device(config.device)
    dtype = torch.float32
    counts = sorted({int(value) for value in config.sdf_child_counts})
    if not counts or counts[0] < 2:
        raise MacroSdfEvalError(
            "INVALID_SDF_CHILD_COUNTS",
            "sdf_child_counts must contain integers >= 2",
        )
    max_children = max(counts)
    bank_size = int(config.sdf_shock_bank_max_children or max_children)
    if bank_size < max_children:
        raise MacroSdfEvalError(
            "INVALID_SDF_CHILD_COUNTS",
            "sdf_shock_bank_max_children must be >= max(sdf_child_counts)",
        )

    n_parent = int(len(pair_frame))
    parent_states = torch.zeros((n_parent, 7), dtype=dtype, device=device)
    parent_states[:, 4] = torch.as_tensor(
        pair_frame["x_t"].to_numpy(dtype=np.float32), device=device
    )
    hatc_cal = torch.as_tensor(
        pair_frame["hatc_t"].to_numpy(dtype=np.float32), device=device
    ).reshape(-1, 1)
    lnk_cal = torch.as_tensor(
        pair_frame["lnk_t"].to_numpy(dtype=np.float32), device=device
    ).reshape(-1, 1)

    full_bank = ConvergenceShockBank.create(
        n_parent, bank_size, seed=int(config.sdf_seed), device=device, dtype=dtype
    )
    bank = ConvergenceShockBank(
        eps_x=full_bank.eps_x[:, :max_children],
        eps_z=full_bank.eps_z[:, :max_children],
        u_eta=full_bank.u_eta[:, :max_children],
        u_i=full_bank.u_i[:, :max_children],
        seed=full_bank.seed,
    )
    economic_config = cp.economic_config
    x_prev = parent_states[:, 4:5]
    x_next = (
        (1.0 - economic_config.RHO_X) * economic_config.XBAR
        + economic_config.RHO_X * x_prev.unsqueeze(1)
        + economic_config.SIGMA_X * bank.eps_x
    )

    clip = config.normalized_logr_clip
    if clip is None:
        clip = float(getattr(cp.hyperparams, "sdf_normalized_logr_clip", 20.0))
    with torch.inference_mode():
        w_parent, w_children, m_raw, hatc_next, lnk_next = model.forward_step(
            x_prev=x_prev,
            x_curr=x_next,
            hatcf_prev=hatc_cal,
            lnkf_prev=lnk_cal,
            return_physical=True,
        )
        loss = SDFLoss(
            gamma=float(model.sdf_model.gamma),
            kappa=float(model.sdf_model.kappa),
            sigma=float(model.sdf_model.sigma),
            beta=float(model.sdf_model.beta),
            wealth_loss_mode="signed_aio",
        )
        residuals = loss.compute_wealth_residuals(
            w_parent=w_parent,
            w_children=w_children,
            k_parent=lnk_cal,
            k_children=lnk_next,
            c_parent=hatc_cal,
            c_children=hatc_next,
            normalized_logr_clip=float(clip),
        )

    raw_residual = residuals["raw"]
    normalized_residual = residuals["normalized"]
    log_r = residuals["log_R"]

    metrics: Dict[str, float] = {}
    metrics.update(conditional_residual_summary(raw_residual, prefix="sdf_raw"))
    metrics.update(
        conditional_residual_summary(normalized_residual, prefix="sdf_normalized")
    )

    # --- Signed AiO: estimate + standard error + explicit interpretation guard.
    with torch.inference_mode():
        _, aio_details = loss.compute_wealth_main_loss(normalized_residual[:, :2])
    signed_aio_values = (
        normalized_residual[:, 0].to(torch.float64)
        * normalized_residual[:, 1].to(torch.float64)
    )
    aio_array = signed_aio_values.detach().cpu().numpy()
    metrics["sdf_signed_aio_estimate"] = _safe_mean(aio_array)
    metrics["sdf_signed_aio_se"] = _safe_se(aio_array)
    metrics["sdf_signed_aio_std"] = _safe_std(aio_array)
    metrics["sdf_signed_aio_negative_share"] = (
        float(np.mean(aio_array[np.isfinite(aio_array)] < 0.0))
        if np.isfinite(aio_array).any()
        else _nan()
    )
    metrics["sdf_signed_aio_n"] = float(np.isfinite(aio_array).sum())
    metrics["sdf_signed_aio_loss_helper_estimate"] = float(
        aio_details["signed_aio"].detach().cpu().item()
    )
    metrics["sdf_signed_aio_loss_helper_se"] = float(
        aio_details["signed_aio_se"].detach().cpu().item()
    )

    # --- Population target estimate + standard error (per-parent conditional
    #     means of the normalized wealth Euler residual).
    per_parent_mean = normalized_residual.to(torch.float64).mean(dim=1)
    parent_means = per_parent_mean.detach().cpu().numpy()
    metrics["sdf_population_target_estimate"] = _safe_mean(parent_means)
    metrics["sdf_population_target_se"] = _safe_se(parent_means)
    metrics["sdf_population_target_std"] = _safe_std(parent_means)
    estimate = metrics["sdf_population_target_estimate"]
    se = metrics["sdf_population_target_se"]
    metrics["sdf_population_target_t_stat"] = (
        float(estimate / se)
        if math.isfinite(estimate) and math.isfinite(se) and se > 0
        else _nan()
    )
    metrics["sdf_population_target_abs"] = (
        float(abs(estimate)) if math.isfinite(estimate) else _nan()
    )

    # --- SDF first / second moments with standard errors.
    m_flat = m_raw.detach().reshape(-1).to(torch.float64).cpu().numpy()
    m_finite = m_flat[np.isfinite(m_flat)]
    metrics["sdf_M_mean"] = float(m_finite.mean()) if m_finite.size else _nan()
    metrics["sdf_M_mean_se"] = _safe_se(m_finite)
    metrics["sdf_M_second_moment"] = (
        float(np.mean(m_finite ** 2)) if m_finite.size else _nan()
    )
    metrics["sdf_M_second_moment_se"] = _safe_se(m_finite ** 2)
    metrics["sdf_M_var"] = float(m_finite.var()) if m_finite.size else _nan()
    metrics["sdf_M_std"] = _safe_std(m_finite)
    metrics["sdf_M_finite_ratio"] = (
        float(np.isfinite(m_flat).mean()) if m_flat.size else _nan()
    )

    # --- Economic moment constraints (never pooled with the AiO estimate).
    if m_finite.size:
        constraints = compute_pooled_moment_constraints(
            torch.as_tensor(m_finite, dtype=torch.float64),
            loss.mu_lo,
            loss.mu_hi,
            loss.var_hi,
        )
        metrics["sdf_g_mean_low"] = float(constraints["g_mean_low"].item())
        metrics["sdf_g_mean_high"] = float(constraints["g_mean_high"].item())
        metrics["sdf_g_var_high"] = float(constraints["g_var_high"].item())
        metrics["sdf_g_max"] = float(constraints["max_violation"].item())
    else:
        for name in ("sdf_g_mean_low", "sdf_g_mean_high", "sdf_g_var_high", "sdf_g_max"):
            metrics[name] = _nan()

    # --- Numerical safety constraints.
    metrics["sdf_nonfinite_ratio"] = float(
        (~np.isfinite(normalized_residual.detach().cpu().numpy())).mean()
    )
    metrics["sdf_nonfinite_parent_ratio"] = (
        float((~np.isfinite(parent_means)).mean()) if parent_means.size else _nan()
    )
    metrics["sdf_log_R_clip_share"] = float(
        (log_r.detach().abs() > float(clip)).double().mean().item()
    )
    metrics["sdf_valid_parent_ratio"] = float(
        torch.isfinite(normalized_residual).all(dim=1).double().mean().item()
    )
    metrics["sdf_n_parents"] = float(n_parent)
    metrics["sdf_n_children"] = float(int(normalized_residual.shape[1]))

    # --- Conditional distribution by macro state (Recession/Normal/Expansion).
    macro_cut = state_cutpoints(pair_frame["x_t"], config.macro_state_quantiles)
    labels = state_labels(pair_frame["x_t"], macro_cut, MACRO_STATE_LABELS)
    tables: Dict[str, pd.DataFrame] = {}
    state_rows: List[Dict[str, float]] = []
    for label in MACRO_STATE_LABELS:
        mask = torch.as_tensor(labels == str(label), device=device)
        n_rows = int(mask.sum().item())
        if n_rows == 0:
            continue
        subset = normalized_residual[mask]
        metrics.update(conditional_residual_summary(subset, prefix=f"sdf_{label}"))
        subset_means = subset.to(torch.float64).mean(dim=1).detach().cpu().numpy()
        m_subset = m_raw[mask].detach().reshape(-1).to(torch.float64).cpu().numpy()
        m_subset = m_subset[np.isfinite(m_subset)]
        metrics[f"sdf_{label}_M_mean"] = float(m_subset.mean()) if m_subset.size else _nan()
        metrics[f"sdf_{label}_M_mean_se"] = _safe_se(m_subset)
        metrics[f"sdf_{label}_n_parents"] = float(n_rows)
        state_rows.append(
            {
                "macro_state": str(label),
                "n_parents": float(n_rows),
                "mean_residual": _safe_mean(subset_means),
                "se_residual": _safe_se(subset_means),
                "M_mean": metrics[f"sdf_{label}_M_mean"],
            }
        )
    tables["sdf_macro_state_summary"] = pd.DataFrame(state_rows)

    diagnostics = {
        "sdf_signed_aio_interpretation_guard": {
            "negative_sample_mean_is_not_negative_population_moment": True,
            "reported_estimate": (
                "sdf_signed_aio_estimate (sample mean of r_1 * r_2 across parents)"
            ),
            "reported_standard_error": "sdf_signed_aio_se",
            "population_target_reported_separately": (
                "sdf_population_target_estimate / sdf_population_target_se"
            ),
            "numerical_safety_reported_separately": [
                "sdf_nonfinite_ratio",
                "sdf_nonfinite_parent_ratio",
                "sdf_log_R_clip_share",
                "sdf_valid_parent_ratio",
            ],
            "note": (
                "A negative Signed AiO sample mean must not be read as a negative "
                "population conditional moment; the estimate is reported with its "
                "standard error and the conditional-moment metrics are reported "
                "separately."
            ),
        },
        "sdf_residual_definitions": {
            "raw": (
                "A_j * w_{t+1,j}^kappa - (w_t - exp(c_t))^kappa "
                "(SDFLoss.compute_wealth_residuals)"
            ),
            "normalized": (
                "A_j * (w_{t+1,j} / (w_t - exp(c_t)))^kappa - 1"
            ),
            "signed_aio": "mean over parents of r_1 * r_2 (first two children)",
            "moment_constraints": (
                "losses.sdf_loss.compute_pooled_moment_constraints (mu_lo/mu_hi/var_hi)"
            ),
        },
        "sdf_macro_state_cutpoints_x_t": macro_cut,
        "sdf_child_counts": counts,
        "sdf_shock_bank_max_children": bank_size,
        "sdf_seed": int(config.sdf_seed),
        "sdf_normalized_logr_clip": float(clip),
    }
    return metrics, tables, diagnostics


def evaluate_sdf_block(
    cp: EvalCheckpoint,
    pair_frame: pd.DataFrame,
    *,
    config: MacroSdfEvalConfig,
) -> Tuple[Dict[str, float], Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Full SDF block: canonical multi-K summary + detail pass with std errors."""
    model = cp.models["sdf_fc1"]
    device = torch.device(config.device)
    n_parent = int(len(pair_frame))
    parent_states = torch.zeros((n_parent, 7), dtype=torch.float32, device=device)
    parent_states[:, 4] = torch.as_tensor(
        pair_frame["x_t"].to_numpy(dtype=np.float32), device=device
    )
    hatc_cal = torch.as_tensor(
        pair_frame["hatc_t"].to_numpy(dtype=np.float32), device=device
    ).reshape(-1, 1)
    lnk_cal = torch.as_tensor(
        pair_frame["lnk_t"].to_numpy(dtype=np.float32), device=device
    ).reshape(-1, 1)
    clip = config.normalized_logr_clip
    if clip is None:
        clip = float(getattr(cp.hyperparams, "sdf_normalized_logr_clip", 20.0))

    counts = sorted({int(value) for value in config.sdf_child_counts})
    summaries, metadata = evaluate_sdf_heldout_multi_k(
        model,
        parent_states,
        hatc_cal=hatc_cal,
        lnk_cal=lnk_cal,
        economic_config=cp.economic_config,
        child_counts=counts,
        seed=int(config.sdf_seed),
        shock_bank_max_children=config.sdf_shock_bank_max_children,
        normalized_logr_clip=float(clip),
    )
    metrics: Dict[str, float] = {}
    for count, summary in sorted(summaries.items()):
        for key, value in summary.items():
            metrics[f"sdf_k{int(count)}_{key}"] = float(value)

    detail_metrics, detail_tables, detail_diagnostics = _sdf_detail_pass(
        cp, pair_frame, config=config
    )
    metrics.update(detail_metrics)
    diagnostics = {
        "sdf_multi_k_metadata": metadata,
        "sdf_multi_k_source": (
            "evaluation.full_run_diagnostics.evaluate_sdf_heldout_multi_k"
        ),
        **detail_diagnostics,
    }
    return metrics, detail_tables, diagnostics


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def evaluate_macro_sdf(
    cp: EvalCheckpoint,
    bank_d: Mapping[str, Any],
    bank_c_anchors: Optional[Mapping[str, Mapping[str, float]]] = None,
    *,
    config: Optional[MacroSdfEvalConfig] = None,
) -> MacroSdfEvalResult:
    """Macro / SDF evaluation of one checkpoint on Bank D (+ Bank C anchors).

    Parameters
    ----------
    cp:
        Evaluation checkpoint (``evaluation.convergence.checkpoint_io``).
    bank_d:
        Bank D macro holdout payload (``evaluation.convergence.eval_bank``
        ``bank_d`` entry): keyed parent-child macro rows.
    bank_c_anchors:
        Bank C macro anchors (``derive_macro_anchors`` output) used as macro
        initial states of the anchor recursion paths.
    config:
        :class:`MacroSdfEvalConfig`; defaults are CPU-friendly.

    Returns
    -------
    MacroSdfEvalResult
        ``metrics`` (flat, prefixed ``fc1_*`` / ``sdf_*``), ``tables``,
        ``surfaces`` (anchor recursion paths) and ``diagnostics`` (pairing
        audit, calibre separation, Signed-AiO interpretation guard).
    """
    config = config or MacroSdfEvalConfig()
    for module in cp.models.values():
        if isinstance(module, torch.nn.Module):
            module.eval()

    pair_frame = bank_d_frame(bank_d)
    metrics: Dict[str, float] = {}
    tables: Dict[str, pd.DataFrame] = {}
    surfaces: Dict[str, np.ndarray] = {}
    diagnostics: Dict[str, Any] = {
        "optimizer_steps": count_optimizer_steps(),
        "evaluation_only": True,
        "belief_vs_realized": {
            "belief_columns": ["hatcf_t", "lnkf_t"],
            "realized_columns": ["hatc_t", "lnk_t"],
            "never_mixed": True,
        },
    }

    with param_hash_guard(cp):
        with torch.inference_mode():
            belief_metrics, stratified_frame, belief_diagnostics = (
                fc1_one_step_belief_metrics(pair_frame, config=config)
            )
            metrics.update(belief_metrics)
            diagnostics.update(belief_diagnostics)

            shift_metrics, shift_tables = fc1_shift_alignment(
                pair_frame, config=config
            )
            metrics.update(shift_metrics)
            tables.update(shift_tables)

            metrics.update(fc1_model_one_step(cp, pair_frame, config=config))

            eval_metrics, eval_tables, eval_diagnostics = fc1_evaluator_pass(
                cp, pair_frame, config=config
            )
            metrics.update(eval_metrics)
            tables.update(eval_tables)
            diagnostics.update(eval_diagnostics)

            rec_metrics, rec_tables, rec_surfaces, rec_diagnostics = (
                fc1_recursive_stability(
                    cp, pair_frame, bank_c_anchors, config=config
                )
            )
            metrics.update(rec_metrics)
            tables.update(rec_tables)
            surfaces.update(rec_surfaces)
            diagnostics.update(rec_diagnostics)

            sdf_metrics, sdf_tables, sdf_diagnostics = evaluate_sdf_block(
                cp, pair_frame, config=config
            )
            metrics.update(sdf_metrics)
            tables.update(sdf_tables)
            diagnostics.update(sdf_diagnostics)

    tables["fc1_one_step_pairs"] = stratified_frame
    diagnostics["optimizer_steps"] = count_optimizer_steps()
    return MacroSdfEvalResult(
        metrics=metrics, tables=tables, surfaces=surfaces, diagnostics=diagnostics
    )
