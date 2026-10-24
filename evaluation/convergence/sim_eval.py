"""Simulation / distribution evaluator with the capital Waterfall.

This module implements the macro-panel side of the equilibrium convergence
evaluator:

* **Capital Waterfall** (Phase 1 priority) -- rebuilds the exact endpoint
  identity of ``data.entry.capital_growth_decomposition`` from a simulate-style
  macro panel::

      K_{t+1}^{end} - K_t^{start}
          = deltaK_incumbent + K_{t+1}^{entry} - K_t^{exit} + accounting_residual

  per transition, with the same-node entrant exit reported as a *separate memo
  line* (those firms are in neither endpoint set, so adding them into the
  identity would double count), with dual-view nodes de-duplicated explicitly
  and with the start/end, ``economic_node_id`` and ``accounting_stage`` labels
  cross-checked.
* **Distribution evaluation** (Gomes Fig.2 style data layer) -- x-quantile
  Recession/Normal/Expansion splits, firm-count and capital-weighted dual
  calibres, leverage / productivity marginals, b-z joint histograms on a fixed
  grid, time slices, per-period path/firm accounting and distribution distances
  (1D Wasserstein-1 plus fixed-grid 2D histogram distance -- no full 2D optimal
  transport).
* **Fixed rollout driver** -- runs the existing simulation chain
  (``data.simulate_ts`` / ``data.simulate_ts_parallel``, call only, never
  modified) from a Bank E initial cross-section under the ``stable_firm_identity``
  keyed-RNG shock specification, with the initial h-star calibrated per
  checkpoint by the production ``_calibrate_initial_hstar_batched`` chain.

Economic semantics that must never blur:

* ``hatcf``/``lnkf`` are beliefs/forecasts, ``Hatc``/``LnK`` are realized
  aggregates; the FC1 capital forecast error is reported as ``LnKF vs LnK`` and
  never as a realized-vs-realized quantity.
* The simulate panel only writes surviving rows (exit = row disappearance,
  ``bar_z >= 0.5``); extinct paths stay in the statistics and are reported
  explicitly (``extinct_paths``, ``path_stop_times``), never silently dropped.
* Main-branch nodes are exported twice -- child view ``(t, branch=0)`` pre-exit
  and parent view ``(t, branch=-1)`` post-exit.  Every statistic here picks one
  view explicitly and reports how many dual-view rows it excluded.

Distribution statistics are computable on existing panel pickles through
:func:`evaluate_waterfall` / :func:`evaluate_distributions`; the rollout driver
:func:`evaluate_simulation` produces such panels from a checkpoint plus Bank E.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from analysis.economic_config import AnalysisEconomicConfig
from config import Config
from data.simulate_ts import SimulateTS

from .checkpoint_io import EvalCheckpoint, count_optimizer_steps, param_hash_guard


MACRO_STATE_LABELS = ("recession", "normal", "expansion")

VIEW_CHILD = "child_pre_exit"
VIEW_PARENT = "parent_post_exit"

REASON_MISSING_COLUMNS = "PANEL_MISSING_COLUMNS"
REASON_EMPTY_PANEL = "EMPTY_PANEL"


class SimEvalError(ValueError):
    """Simulation/distribution evaluation error with a stable reason code."""

    def __init__(self, reason_code: str, message: str):
        super().__init__(f"[{reason_code}] {message}")
        self.reason_code = reason_code
        self.message = message


@dataclass
class SimEvalConfig:
    """Tunables of the simulation / distribution evaluation block."""

    device: str = "cpu"
    horizon: int = 100
    variants: Tuple[str, ...] = ("Low", "Baseline", "High")
    branch_num: int = 1
    main_branch: int = 0
    enable_entry: bool = True
    enable_exit: bool = True
    capacity_extra_per_step: int = 20
    calibrate_initial_hstar: bool = True
    node_accounting_mode: Optional[str] = None
    # Waterfall.
    waterfall_transitions: Optional[Tuple[int, ...]] = None
    identity_tolerance: float = 1.0e-5
    # Distributions.
    time_slices: Tuple[int, ...] = (0, 1, 2, 5, 10, 25, 50)
    extended_time_slices: Tuple[int, ...] = (75, 100, 150, 200)
    macro_state_quantiles: Tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
    b_grid_min: float = 0.0
    b_grid_max: float = 2.0
    b_grid_points: int = 21
    z_grid_min: float = -4.0
    z_grid_max: float = 4.0
    z_grid_points: int = 21


@dataclass
class SimEvalResult:
    """Result of a simulation / distribution / waterfall evaluation block."""

    metrics: Dict[str, float] = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    surfaces: Dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# numeric helpers
# ---------------------------------------------------------------------------


def _nan() -> float:
    return float("nan")


def _numeric(frame: pd.DataFrame, name: str) -> Optional[np.ndarray]:
    if name not in frame.columns:
        return None
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
    return values


def _safe_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else _nan()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0:
        return _nan()
    return float(numerator / denominator)


def _weighted_moments(values: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    """Weighted mean / variance of ``values`` (weights normalized internally)."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not bool(mask.any()):
        return {"mean": _nan(), "var": _nan(), "total_weight": 0.0}
    v = values[mask]
    w = weights[mask]
    total = float(w.sum())
    w = w / total
    mean = float(np.sum(w * v))
    var = float(np.sum(w * (v - mean) ** 2))
    return {"mean": mean, "var": var, "total_weight": total}


def wasserstein_1(
    values_a: Any,
    values_b: Any,
    *,
    weights_a: Optional[Any] = None,
    weights_b: Optional[Any] = None,
) -> float:
    """Exact 1D Wasserstein-1 distance between two weighted samples.

    Computed as ``integral |F_a - F_b| dx`` over the merged support of the two
    empirical step CDFs -- no binning, no approximation.  With equal weights on
    identical shapes shifted by ``d`` the result is exactly ``|d|``.
    """
    a = np.asarray(values_a, dtype=np.float64).reshape(-1)
    b = np.asarray(values_b, dtype=np.float64).reshape(-1)
    wa = (
        np.ones_like(a, dtype=np.float64)
        if weights_a is None
        else np.asarray(weights_a, dtype=np.float64).reshape(-1)
    )
    wb = (
        np.ones_like(b, dtype=np.float64)
        if weights_b is None
        else np.asarray(weights_b, dtype=np.float64).reshape(-1)
    )
    if a.shape != wa.shape or b.shape != wb.shape:
        raise SimEvalError("SHAPE_MISMATCH", "wasserstein_1 values/weights shape mismatch")
    mask_a = np.isfinite(a) & np.isfinite(wa) & (wa > 0)
    mask_b = np.isfinite(b) & np.isfinite(wb) & (wb > 0)
    a, wa = a[mask_a], wa[mask_a]
    b, wb = b[mask_b], wb[mask_b]
    if a.size == 0 or b.size == 0:
        return _nan()
    wa = wa / wa.sum()
    wb = wb / wb.sum()

    support = np.unique(np.concatenate([a, b]))
    if support.size < 2:
        return 0.0

    order_a = np.argsort(a, kind="mergesort")
    order_b = np.argsort(b, kind="mergesort")
    a_sorted, wa_sorted = a[order_a], wa[order_a]
    b_sorted, wb_sorted = b[order_b], wb[order_b]
    cdf_a = np.cumsum(wa_sorted)
    cdf_b = np.cumsum(wb_sorted)

    idx_a = np.searchsorted(a_sorted, support, side="right") - 1
    idx_b = np.searchsorted(b_sorted, support, side="right") - 1
    fa = np.where(idx_a >= 0, cdf_a[np.maximum(idx_a, 0)], 0.0)
    fb = np.where(idx_b >= 0, cdf_b[np.maximum(idx_b, 0)], 0.0)

    widths = np.diff(support)
    return float(np.sum(np.abs(fa[:-1] - fb[:-1]) * widths))


def joint_histogram(
    values_b: Any,
    values_z: Any,
    *,
    b_edges: Any,
    z_edges: Any,
    weights: Optional[Any] = None,
) -> np.ndarray:
    """Normalized fixed-grid b-z joint histogram (sums to 1 when non-empty)."""
    b = np.asarray(values_b, dtype=np.float64).reshape(-1)
    z = np.asarray(values_z, dtype=np.float64).reshape(-1)
    w = (
        np.ones_like(b, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64).reshape(-1)
    )
    if b.shape != z.shape or b.shape != w.shape:
        raise SimEvalError("SHAPE_MISMATCH", "joint_histogram values/weights shape mismatch")
    mask = np.isfinite(b) & np.isfinite(z) & np.isfinite(w) & (w >= 0)
    b, z, w = b[mask], z[mask], w[mask]
    b_edges = np.asarray(b_edges, dtype=np.float64)
    z_edges = np.asarray(z_edges, dtype=np.float64)
    hist, _, _ = np.histogram2d(b, z, bins=[b_edges, z_edges], weights=w)
    total = float(hist.sum())
    if total > 0:
        hist = hist / total
    return hist


def joint_histogram_distance(
    hist_a: np.ndarray,
    hist_b: np.ndarray,
) -> Dict[str, float]:
    """Fixed-grid 2D histogram distance (L1 / total variation / overlap).

    Full 2D optimal transport is deliberately **not** used: the b-z joint
    distance is the L1 distance between the two normalized fixed-grid
    histograms, with total variation and overlap reported alongside.
    """
    a = np.asarray(hist_a, dtype=np.float64)
    b = np.asarray(hist_b, dtype=np.float64)
    if a.shape != b.shape:
        raise SimEvalError("SHAPE_MISMATCH", "joint_histogram_distance requires equal shapes")
    total_a, total_b = float(a.sum()), float(b.sum())
    if total_a > 0:
        a = a / total_a
    if total_b > 0:
        b = b / total_b
    l1 = float(np.abs(a - b).sum())
    return {
        "hist2d_l1": l1,
        "hist2d_tv": 0.5 * l1,
        "hist2d_overlap": float(np.minimum(a, b).sum()),
    }


# ---------------------------------------------------------------------------
# panel view selection (dual-view de-duplication)
# ---------------------------------------------------------------------------

_STAGE_NAME_TO_CODE = {
    "none": -1,
    "parent_pre_transition_alive": 0,
    "child_post_exit_endpoint": 1,
}


def _stage_code(values: pd.Series) -> np.ndarray:
    """Normalize ``decomposition_*_stage`` labels to codes (-1/0/1)."""
    out = np.full(len(values), np.nan, dtype=np.float64)
    for position, raw in enumerate(values.tolist()):
        if raw is None:
            continue
        if isinstance(raw, str):
            out[position] = float(_STAGE_NAME_TO_CODE.get(raw.strip(), np.nan))
        else:
            number = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
            out[position] = float(number)
    return out


def _branch_values(frame: pd.DataFrame) -> np.ndarray:
    if "branch" not in frame.columns:
        raise SimEvalError(
            REASON_MISSING_COLUMNS,
            "panel frame is missing the 'branch' column; explicit branch filtering "
            "is mandatory (dual-view nodes would otherwise be double counted)",
        )
    return pd.to_numeric(frame["branch"], errors="coerce").to_numpy(dtype=np.float64)


def _pick_view(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    preferred_branch: float,
    fallback_branch: float,
    view_name: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Pick one row per key, preferring one exported view of the dual-view node.

    Main-branch nodes are exported twice (child view ``branch=0`` pre-exit and
    parent view ``branch=-1`` post-exit); picking one view explicitly and
    counting the excluded counterpart rows is what keeps "two views never double
    counted" executable instead of a comment.
    """
    if frame.empty:
        return frame.copy(), {
            "view": view_name,
            "n_rows": 0,
            "dual_view_rows_excluded": 0,
            "duplicate_rows_excluded": 0,
        }
    branch = _branch_values(frame)
    candidates = frame.loc[(branch == preferred_branch) | (branch == fallback_branch)].copy()
    candidates["__branch"] = branch[(branch == preferred_branch) | (branch == fallback_branch)]
    candidates["__preferred"] = (candidates["__branch"] == preferred_branch).astype(int)
    candidates = candidates.sort_values(
        list(key_columns) + ["__preferred"], kind="mergesort", ascending=[True] * len(key_columns) + [False]
    )
    n_candidates = int(len(candidates))
    selected = candidates.drop_duplicates(subset=list(key_columns), keep="first")
    n_selected = int(len(selected))
    dual_view_rows = 0
    counts = candidates.groupby(list(key_columns), sort=False).size()
    if not counts.empty:
        dual_view_rows = int((counts > 1).sum())
    report = {
        "view": view_name,
        "preferred_branch": float(preferred_branch),
        "fallback_branch": float(fallback_branch),
        "n_rows": n_selected,
        "n_candidate_rows": n_candidates,
        "dual_view_keys": dual_view_rows,
        "dual_view_rows_excluded": n_candidates - n_selected,
        "key_columns": list(key_columns),
    }
    return selected.drop(columns=["__branch", "__preferred"]).reset_index(drop=True), report


def select_transition_rows(macro_frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Canonical capital-decomposition transition rows (one per (path, t)).

    A transition row carries the decomposition of ``node(t-1) -> node(t)`` and
    is exported on the child view ``(t, branch=0)`` (pre-exit, where the
    same-node entrant exits are still visible) and re-exported on the parent
    view ``(t, branch=-1)``.  The child view is canonical, the parent view is
    the fallback; everything else (counterfactual branches, invalid
    transitions) is excluded and reported.
    """
    required = {
        "path",
        "t",
        "branch",
        "decomposition_transition_valid",
        "deltaK_incumbent",
        "K_entry_endpoint",
        "K_exit_old",
        "K_decomposition_start",
        "K_decomposition_end",
    }
    missing = sorted(required - set(macro_frame.columns))
    if missing:
        raise SimEvalError(
            REASON_MISSING_COLUMNS,
            f"macro panel is missing capital decomposition columns: {missing}",
        )
    rows, report = _pick_view(
        macro_frame,
        key_columns=("path", "t"),
        preferred_branch=0.0,
        fallback_branch=-1.0,
        view_name=VIEW_CHILD,
    )
    valid = pd.to_numeric(rows["decomposition_transition_valid"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    keep = np.isfinite(valid) & (valid > 0.5)
    if "decomposition_start_stage" in rows.columns and "decomposition_end_stage" in rows.columns:
        start_stage = _stage_code(rows["decomposition_start_stage"])
        end_stage = _stage_code(rows["decomposition_end_stage"])
        stage_ok = (start_stage == 0.0) & (end_stage == 1.0)
        report["stage_label_violations"] = int((keep & ~stage_ok).sum())
        keep = keep & stage_ok
    else:
        report["stage_label_violations"] = 0
    report["invalid_transitions_excluded"] = int((~keep).sum())
    report["n_transitions"] = int(keep.sum())
    selected = rows.loc[keep].copy().reset_index(drop=True)
    selected["t"] = pd.to_numeric(selected["t"], errors="coerce")
    selected["t_from"] = selected["t"] - 1
    return selected, report


def select_cross_section_rows(
    firm_frame: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Canonical surviving cross-section rows, resolved **per node view**.

    Main-branch nodes are exported twice (child view ``branch=0`` pre-exit and
    parent view ``branch=-1`` post-exit).  The view is chosen per *node*, never
    per firm: when a node has parent-view rows the whole post-exit view is
    taken, otherwise the pre-exit child view is taken.  Rows are then unique per
    ``(path, t, ID)``.  This is what keeps the dual views from double counting
    while the surviving cross-section of a node stays on one single view.
    """
    required = {"path", "t", "branch", "ID"}
    missing = sorted(required - set(firm_frame.columns))
    if missing:
        raise SimEvalError(
            REASON_MISSING_COLUMNS, f"firm panel is missing columns: {missing}"
        )
    branch = _branch_values(firm_frame)
    path_values = pd.to_numeric(firm_frame["path"], errors="coerce").to_numpy(dtype=np.float64)
    t_values = pd.to_numeric(firm_frame["t"], errors="coerce").to_numpy(dtype=np.float64)
    has_parent = set()
    for position in range(len(firm_frame)):
        if branch[position] == -1.0:
            has_parent.add((path_values[position], t_values[position]))
    keep = np.zeros(len(firm_frame), dtype=bool)
    n_parent_rows = 0
    n_child_rows = 0
    for position in range(len(firm_frame)):
        key = (path_values[position], t_values[position])
        if key in has_parent:
            selected = branch[position] == -1.0
            n_parent_rows += int(selected)
        else:
            selected = branch[position] == 0.0
            n_child_rows += int(selected)
        keep[position] = selected
    selected_rows = firm_frame.loc[keep].copy()
    duplicates = int(
        selected_rows.duplicated(subset=["path", "t", "ID"], keep="first").sum()
    )
    selected_rows = selected_rows.drop_duplicates(
        subset=["path", "t", "ID"], keep="first"
    ).reset_index(drop=True)
    report = {
        "view": VIEW_PARENT,
        "view_rule": (
            "per node: branch==-1 (post-exit) when present, else branch==0 (pre-exit)"
        ),
        "n_rows": int(len(selected_rows)),
        "n_parent_view_rows": int(n_parent_rows),
        "n_child_view_rows": int(n_child_rows),
        "duplicate_rows_excluded": duplicates,
        "rows_excluded_by_view_filter": int(len(firm_frame)) - int(len(selected_rows)),
        "dual_view_rows_excluded": int(len(firm_frame)) - int(len(selected_rows)),
    }
    return selected_rows, report


def select_node_rows(macro_frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Canonical macro node rows (one per (path, t)); parent view preferred."""
    required = {"path", "t", "branch"}
    missing = sorted(required - set(macro_frame.columns))
    if missing:
        raise SimEvalError(
            REASON_MISSING_COLUMNS, f"macro panel is missing columns: {missing}"
        )
    return _pick_view(
        macro_frame,
        key_columns=("path", "t"),
        preferred_branch=-1.0,
        fallback_branch=0.0,
        view_name=VIEW_PARENT,
    )


def path_status_table(
    cross_section: pd.DataFrame, *, all_paths: Sequence[int]
) -> pd.DataFrame:
    """Per-path accounting: first/last observed t, stop time, extinction flag.

    A path is extinct once it stops writing rows (exit = row disappearance);
    extinct paths stay in this table and are counted in every period's
    ``extinct_paths`` column instead of being silently dropped.
    """
    rows: List[Dict[str, Any]] = []
    observed = (
        cross_section.groupby("path")["t"].agg(["min", "max", "count"])
        if not cross_section.empty
        else pd.DataFrame(columns=["min", "max", "count"])
    )
    max_t = int(pd.to_numeric(cross_section["t"], errors="coerce").max()) if not cross_section.empty else -1
    for path in sorted({int(value) for value in all_paths}):
        if path in observed.index:
            first_t = int(observed.loc[path, "min"])
            last_t = int(observed.loc[path, "max"])
            n_rows = int(observed.loc[path, "count"])
            extinct = bool(last_t < max_t)
            stop_time = last_t + 1 if extinct else None
        else:
            first_t = -1
            last_t = -1
            n_rows = 0
            extinct = True
            stop_time = 0
        rows.append(
            {
                "path": path,
                "first_t": first_t,
                "last_t": last_t,
                "n_rows": n_rows,
                "extinct": extinct,
                "stop_time": stop_time,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# capital Waterfall
# ---------------------------------------------------------------------------


def evaluate_waterfall(
    firm_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    *,
    config: Optional[SimEvalConfig] = None,
) -> SimEvalResult:
    """Rebuild the capital Waterfall of the ``capital_growth_decomposition`` identity.

    Per transition ``node(t) -> node(t+1)`` (rows ``t_from -> t_to``)::

        initial_capital = K_decomposition_start
        ending_capital  = initial + incumbent_dk + entrant_capital
                          - exiting_incumbent_capital + accounting_residual

    with

    * ``incumbent_dk`` = ``deltaK_incumbent`` (surviving firms' capital change),
    * ``entrant_capital`` = ``K_entry_endpoint`` (endpoint decomposition entry,
      **not** the gross ``K_entry_gross``),
    * ``exiting_incumbent_capital`` = ``K_exit_old``,
    * ``same_node_entrant_exit`` = ``K_entry_same_node_exit`` reported as a
      separate memo line: those firms are in neither endpoint set, so adding
      them into the identity would double count them.

    The identity residual is cross-checked against the recorded
    ``capital_accounting_residual`` and against the firm-panel capital sums of
    both endpoints; ``economic_node_id`` / ``accounting_stage`` continuity and
    dual-view de-duplication are reported in the diagnostics.
    """
    config = config or SimEvalConfig()
    transitions, view_report = select_transition_rows(macro_df)
    cross_section, cross_report = select_cross_section_rows(firm_df)
    node_rows, node_report = select_node_rows(macro_df)
    all_paths = sorted(
        int(value)
        for value in pd.to_numeric(
            pd.concat([firm_df["path"], macro_df["path"]], ignore_index=True),
            errors="coerce",
        ).dropna()
    )
    status_table = path_status_table(cross_section, all_paths=all_paths)

    if transitions.empty:
        raise SimEvalError(
            REASON_EMPTY_PANEL,
            "macro panel has no valid capital decomposition transition rows",
        )

    firm_k = _numeric(cross_section, "K")
    firm_ids = (
        cross_section["ID"].astype(str).to_numpy() if "ID" in cross_section.columns else None
    )
    cross_keys = list(
        zip(
            pd.to_numeric(cross_section["path"], errors="coerce").to_numpy(),
            pd.to_numeric(cross_section["t"], errors="coerce").to_numpy(),
        )
    )
    k_by_key: Dict[Tuple[int, int], float] = {}
    ids_by_key: Dict[Tuple[int, int], List[str]] = {}
    for position, key in enumerate(cross_keys):
        if firm_k is not None and np.isfinite(firm_k[position]):
            k_by_key[key] = k_by_key.get(key, 0.0) + float(firm_k[position])
        if firm_ids is not None:
            ids_by_key.setdefault(key, []).append(str(firm_ids[position]))

    node_index = {
        (int(path), int(t)): position
        for position, (path, t) in enumerate(
            zip(
                pd.to_numeric(node_rows["path"], errors="coerce").to_numpy(),
                pd.to_numeric(node_rows["t"], errors="coerce").to_numpy(),
            )
        )
    }

    rows: List[Dict[str, Any]] = []
    for _, transition in transitions.iterrows():
        path = int(pd.to_numeric(pd.Series([transition["path"]]), errors="coerce").iloc[0])
        t_from = int(transition["t_from"])
        t_to = int(transition["t"])
        if config.waterfall_transitions is not None and t_from not in tuple(
            int(value) for value in config.waterfall_transitions
        ):
            continue

        initial_capital = float(transition["K_decomposition_start"])
        ending_capital = float(transition["K_decomposition_end"])
        incumbent_dk = float(transition["deltaK_incumbent"])
        entrant_capital = float(transition["K_entry_endpoint"])
        exiting_capital = float(transition["K_exit_old"])
        recorded_residual = (
            float(transition["capital_accounting_residual"])
            if "capital_accounting_residual" in transition.index
            else _nan()
        )
        same_node_exit_capital = (
            float(transition["K_entry_same_node_exit"])
            if "K_entry_same_node_exit" in transition.index
            else _nan()
        )
        same_node_exit_count = (
            float(transition["same_node_entrant_exit_count"])
            if "same_node_entrant_exit_count" in transition.index
            else _nan()
        )
        entrants_gross = (
            float(transition["K_entry_gross"])
            if "K_entry_gross" in transition.index
            else _nan()
        )
        entrants_surviving = (
            float(transition["K_entry_surviving"])
            if "K_entry_surviving" in transition.index
            else _nan()
        )
        entry_event_residual = (
            float(transition["entry_capital_event_residual"])
            if "entry_capital_event_residual" in transition.index
            else _nan()
        )

        accounting_residual = (
            (ending_capital - initial_capital)
            - incumbent_dk
            - entrant_capital
            + exiting_capital
        )
        start_node_id = (
            float(transition["decomposition_start_node_id"])
            if "decomposition_start_node_id" in transition.index
            else _nan()
        )
        end_node_id = (
            float(transition["decomposition_end_node_id"])
            if "decomposition_end_node_id" in transition.index
            else _nan()
        )
        row_node_id = (
            float(transition["economic_node_id"])
            if "economic_node_id" in transition.index
            else _nan()
        )

        # Side-by-side same-period indicators.
        key_from = (path, t_from)
        key_to = (path, t_to)
        n_firms_from = len(ids_by_key.get(key_from, []))
        n_firms_to = len(ids_by_key.get(key_to, []))
        ids_from = set(ids_by_key.get(key_from, []))
        ids_to = set(ids_by_key.get(key_to, []))
        exited_ids = ids_from - ids_to
        default_rate = _safe_ratio(float(len(exited_ids)), float(n_firms_from))
        entry_count = 0.0
        investor_count = 0.0
        if "entry" in cross_section.columns:
            mask_to = (cross_keys == key_to) if isinstance(cross_keys, np.ndarray) else [
                key == key_to for key in cross_keys
            ]
            subset = cross_section.loc[np.asarray(mask_to)]
            if not subset.empty:
                entry_values = pd.to_numeric(subset["entry"], errors="coerce").to_numpy()
                entry_count = float(np.nansum(entry_values > 0.5))
                investment_column = None
                for candidate in ("Bar_i_executed", "Bar_i"):
                    if candidate in subset.columns:
                        investment_column = candidate
                        break
                if investment_column is not None:
                    investment_values = pd.to_numeric(
                        subset[investment_column], errors="coerce"
                    ).to_numpy()
                    investor_count = float(np.nansum(investment_values > 0.5))
        entry_rate = _safe_ratio(entry_count, float(n_firms_to))
        investment_rate = _safe_ratio(investor_count, float(n_firms_to))

        fc1_lnk_error = _nan()
        fc1_hatc_error = _nan()
        node_position = node_index.get(key_to)
        if node_position is not None:
            lnk_values = _numeric(node_rows, "LnK")
            lnkf_values = _numeric(node_rows, "lnkf")
            hatc_values = _numeric(node_rows, "Hatc")
            hatcf_values = _numeric(node_rows, "hatcf")
            if lnk_values is not None and lnkf_values is not None:
                fc1_lnk_error = float(lnkf_values[node_position] - lnk_values[node_position])
            if hatc_values is not None and hatcf_values is not None:
                fc1_hatc_error = float(hatcf_values[node_position] - hatc_values[node_position])

        rows.append(
            {
                "path": path,
                "t_from": t_from,
                "t_to": t_to,
                "initial_capital": initial_capital,
                "incumbent_dk": incumbent_dk,
                "exiting_incumbent_capital": exiting_capital,
                "entrant_capital": entrant_capital,
                "same_node_entrant_exit": same_node_exit_capital,
                "ending_capital": ending_capital,
                "accounting_residual": accounting_residual,
                "recorded_accounting_residual": recorded_residual,
                "entrant_capital_gross": entrants_gross,
                "entrant_capital_surviving": entrants_surviving,
                "same_node_entrant_exit_count": same_node_exit_count,
                "entry_capital_event_residual": entry_event_residual,
                "initial_capital_firm_sum": k_by_key.get(key_from, _nan()),
                "ending_capital_firm_sum": k_by_key.get(key_to, _nan()),
                "start_node_id": start_node_id,
                "end_node_id": end_node_id,
                "row_node_id": row_node_id,
                "n_firms_from": float(n_firms_from),
                "n_firms_to": float(n_firms_to),
                "default_rate": default_rate,
                "survival_rate": (
                    1.0 - default_rate if math.isfinite(default_rate) else _nan()
                ),
                "active_firms": float(n_firms_to),
                "active_paths": 1.0 if n_firms_to > 0 else 0.0,
                "investment_rate": investment_rate,
                "entry_rate": entry_rate,
                "fc1_lnk_error": fc1_lnk_error,
                "fc1_hatc_error": fc1_hatc_error,
            }
        )

    waterfall = pd.DataFrame(rows)
    if waterfall.empty:
        raise SimEvalError(
            REASON_EMPTY_PANEL,
            "no waterfall transitions selected (check waterfall_transitions)",
        )

    metrics: Dict[str, float] = {}
    residual = waterfall["accounting_residual"].to_numpy(dtype=np.float64)
    recorded = waterfall["recorded_accounting_residual"].to_numpy(dtype=np.float64)
    metrics["waterfall_identity_residual_abs_max"] = float(np.nanmax(np.abs(residual)))
    metrics["waterfall_identity_residual_abs_mean"] = _safe_mean(np.abs(residual))
    metrics["waterfall_recorded_residual_abs_max"] = (
        float(np.nanmax(np.abs(recorded))) if np.isfinite(recorded).any() else _nan()
    )
    finite_pairs = np.isfinite(residual) & np.isfinite(recorded)
    metrics["waterfall_identity_matches_recorded_abs_max"] = (
        float(np.nanmax(np.abs(residual[finite_pairs] - recorded[finite_pairs])))
        if bool(finite_pairs.any())
        else _nan()
    )
    metrics["waterfall_identity_within_tolerance_share"] = float(
        np.mean(np.abs(residual) <= float(config.identity_tolerance))
    )
    metrics["waterfall_firm_start_crosscheck_abs_max"] = float(
        np.nanmax(
            np.abs(
                waterfall["initial_capital"].to_numpy(dtype=np.float64)
                - waterfall["initial_capital_firm_sum"].to_numpy(dtype=np.float64)
            )
        )
    )
    metrics["waterfall_firm_end_crosscheck_abs_max"] = float(
        np.nanmax(
            np.abs(
                waterfall["ending_capital"].to_numpy(dtype=np.float64)
                - waterfall["ending_capital_firm_sum"].to_numpy(dtype=np.float64)
            )
        )
    )
    start_ids = waterfall["start_node_id"].to_numpy(dtype=np.float64)
    end_ids = waterfall["end_node_id"].to_numpy(dtype=np.float64)
    row_ids = waterfall["row_node_id"].to_numpy(dtype=np.float64)
    comparable = np.isfinite(end_ids) & np.isfinite(row_ids)
    metrics["waterfall_end_node_id_mismatch_count"] = float(
        int(np.nansum(end_ids[comparable] != row_ids[comparable]))
    )
    continuity_violations = 0
    for _, group in waterfall.groupby("path"):
        ordered = group.sort_values("t_to", kind="mergesort")
        ends = ordered["end_node_id"].to_numpy(dtype=np.float64)
        starts = ordered["start_node_id"].to_numpy(dtype=np.float64)
        for position in range(1, len(ordered)):
            if np.isfinite(ends[position - 1]) and np.isfinite(starts[position]):
                if ends[position - 1] != starts[position]:
                    continuity_violations += 1
    metrics["waterfall_node_id_continuity_violations"] = float(continuity_violations)
    metrics["waterfall_same_node_entrant_exit_total"] = float(
        np.nansum(waterfall["same_node_entrant_exit"].to_numpy(dtype=np.float64))
    )
    event_residual = waterfall["entry_capital_event_residual"].to_numpy(dtype=np.float64)
    metrics["waterfall_entrant_event_residual_abs_max"] = (
        float(np.nanmax(np.abs(event_residual)))
        if np.isfinite(event_residual).any()
        else _nan()
    )
    metrics["waterfall_dual_view_rows_excluded"] = float(
        view_report.get("dual_view_rows_excluded", 0)
    )
    metrics["waterfall_transitions_count"] = float(len(waterfall))
    metrics["waterfall_side_default_rate_mean"] = _safe_mean(
        waterfall["default_rate"].to_numpy(dtype=np.float64)
    )
    metrics["waterfall_side_survival_rate_mean"] = _safe_mean(
        waterfall["survival_rate"].to_numpy(dtype=np.float64)
    )
    metrics["waterfall_side_investment_rate_mean"] = _safe_mean(
        waterfall["investment_rate"].to_numpy(dtype=np.float64)
    )
    metrics["waterfall_side_entry_rate_mean"] = _safe_mean(
        waterfall["entry_rate"].to_numpy(dtype=np.float64)
    )
    metrics["waterfall_side_fc1_lnk_bias"] = _safe_mean(
        waterfall["fc1_lnk_error"].to_numpy(dtype=np.float64)
    )
    metrics["waterfall_side_fc1_lnk_mae"] = _safe_mean(
        np.abs(waterfall["fc1_lnk_error"].to_numpy(dtype=np.float64))
    )
    metrics["waterfall_side_fc1_hatc_bias"] = _safe_mean(
        waterfall["fc1_hatc_error"].to_numpy(dtype=np.float64)
    )
    metrics["waterfall_side_active_firms_mean"] = _safe_mean(
        waterfall["active_firms"].to_numpy(dtype=np.float64)
    )

    totals_rows: List[Dict[str, Any]] = []
    for (t_from, t_to), group in waterfall.groupby(["t_from", "t_to"], sort=True):
        totals_rows.append(
            {
                "t_from": float(t_from),
                "t_to": float(t_to),
                "initial_capital": float(group["initial_capital"].sum()),
                "incumbent_dk": float(group["incumbent_dk"].sum()),
                "exiting_incumbent_capital": float(group["exiting_incumbent_capital"].sum()),
                "entrant_capital": float(group["entrant_capital"].sum()),
                "same_node_entrant_exit": float(
                    np.nansum(group["same_node_entrant_exit"].to_numpy(dtype=np.float64))
                ),
                "ending_capital": float(group["ending_capital"].sum()),
                "accounting_residual": float(group["accounting_residual"].sum()),
                "default_rate": _safe_mean(group["default_rate"].to_numpy(dtype=np.float64)),
                "survival_rate": _safe_mean(group["survival_rate"].to_numpy(dtype=np.float64)),
                "investment_rate": _safe_mean(
                    group["investment_rate"].to_numpy(dtype=np.float64)
                ),
                "entry_rate": _safe_mean(group["entry_rate"].to_numpy(dtype=np.float64)),
                "active_firms": float(group["active_firms"].sum()),
                "active_paths": float(group["active_paths"].sum()),
                "fc1_lnk_error": _safe_mean(group["fc1_lnk_error"].to_numpy(dtype=np.float64)),
            }
        )

    diagnostics = {
        "waterfall_identity": (
            "K_end - K_start = deltaK_incumbent + K_entry_endpoint - K_exit_old "
            "+ accounting_residual (data/entry.py::capital_growth_decomposition)"
        ),
        "waterfall_entry_semantics": (
            "entrant_capital is the endpoint decomposition entry "
            "(K_entry_endpoint); entrant_capital_gross (K_entry_gross) is a memo "
            "event total and is never added into the identity"
        ),
        "waterfall_same_node_entrant_exit": (
            "same-node entrant exits are in neither endpoint set and are reported "
            "as a separate memo line; they are excluded from the identity sum"
        ),
        "transition_view_report": view_report,
        "cross_section_view_report": cross_report,
        "node_view_report": node_report,
        "identity_tolerance": float(config.identity_tolerance),
        "optimizer_steps": count_optimizer_steps(),
        "evaluation_only": True,
    }
    return SimEvalResult(
        metrics=metrics,
        tables={
            "capital_waterfall": waterfall,
            "capital_waterfall_totals": pd.DataFrame(totals_rows),
            "path_status": status_table,
        },
        surfaces={},
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# distribution evaluation (Gomes Fig.2 style data layer)
# ---------------------------------------------------------------------------


def _macro_state_column(values: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    out = np.empty(array.shape[0], dtype=object)
    out[:] = ""
    if finite.size == 0:
        return out.astype(str)
    edges = [float(np.quantile(finite, float(q))) for q in quantiles]
    bins = np.digitize(array, edges, right=True)
    for index, label in enumerate(MACRO_STATE_LABELS):
        out[bins == index] = str(label)
    out[~np.isfinite(array)] = ""
    return out.astype(str)


def evaluate_distributions(
    firm_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    *,
    config: Optional[SimEvalConfig] = None,
) -> SimEvalResult:
    """Cross-sectional distribution layer on an existing simulate panel.

    * x-quantile Recession/Normal/Expansion split with both firm-count and
      capital-weighted calibres,
    * leverage / productivity marginals and the b-z joint histogram on a fixed
      grid,
    * time slices ``0,1,2,5,10,25,50`` (plus ``75,100,150,200`` when the panel
      is long enough),
    * per-period ``active_paths / extinct_paths / n_alive_firms / n_entrants /
      n_defaults / n_investors`` accounting with extinct paths retained and
      explicitly annotated,
    * moments (b/z mean and variance, Cov(b,z), capital-weighted mean b,
      default/investment/entry rates) and distribution distances (1D
      Wasserstein-1 for b and z, fixed-grid 2D histogram distance for b-z --
      no full 2D optimal transport).
    """
    config = config or SimEvalConfig()
    cross_section, cross_report = select_cross_section_rows(firm_df)
    node_rows, node_report = select_node_rows(macro_df)
    if cross_section.empty:
        raise SimEvalError(REASON_EMPTY_PANEL, "firm panel has no cross-section rows")

    all_paths = sorted(
        {
            int(value)
            for value in pd.to_numeric(
                pd.concat([firm_df["path"], macro_df["path"]], ignore_index=True),
                errors="coerce",
            ).dropna()
        }
    )
    status = path_status_table(cross_section, all_paths=all_paths)

    t_values = pd.to_numeric(cross_section["t"], errors="coerce").to_numpy(dtype=np.float64)
    max_t = int(np.nanmax(t_values)) if np.isfinite(t_values).any() else 0
    slices = {int(value) for value in config.time_slices if int(value) <= max_t}
    if max_t >= min(int(value) for value in config.extended_time_slices):
        slices.update(int(value) for value in config.extended_time_slices if int(value) <= max_t)
    slices.add(max_t)
    ordered_slices = sorted(slices)

    b_values = _numeric(cross_section, "b")
    z_values = _numeric(cross_section, "z")
    k_values = _numeric(cross_section, "K")
    x_values = _numeric(cross_section, "x")
    if b_values is None or z_values is None:
        raise SimEvalError(
            REASON_MISSING_COLUMNS, "firm panel must provide 'b' and 'z' columns"
        )
    if k_values is None:
        k_values = np.ones_like(b_values)
    if x_values is None:
        x_values = np.zeros_like(b_values)
    macro_state = _macro_state_column(x_values, config.macro_state_quantiles)

    entry_values = _numeric(cross_section, "entry")
    if entry_values is None:
        entry_values = np.zeros_like(b_values)
    investment_column = None
    for candidate in ("Bar_i_executed", "Bar_i"):
        if candidate in cross_section.columns:
            investment_column = candidate
            break
    investment_values = (
        _numeric(cross_section, investment_column)
        if investment_column is not None
        else np.zeros_like(b_values)
    )

    path_values = pd.to_numeric(cross_section["path"], errors="coerce").to_numpy(dtype=np.float64)
    id_values = (
        cross_section["ID"].astype(str).to_numpy()
        if "ID" in cross_section.columns
        else np.array([""] * len(cross_section))
    )

    event_rows, _ = _pick_view(
        macro_df,
        key_columns=("path", "t"),
        preferred_branch=0.0,
        fallback_branch=-1.0,
        view_name=VIEW_CHILD,
    )
    same_node_exit_by_t: Dict[int, float] = {}
    accepted_count_by_t: Dict[int, float] = {}
    if "same_node_entrant_exit_count" in event_rows.columns:
        node_t = pd.to_numeric(event_rows["t"], errors="coerce").to_numpy(dtype=np.float64)
        node_same = _numeric(event_rows, "same_node_entrant_exit_count")
        if node_same is not None:
            for position, t_node in enumerate(node_t):
                if np.isfinite(t_node):
                    key = int(t_node)
                    value = node_same[position]
                    if np.isfinite(value):
                        same_node_exit_by_t[key] = same_node_exit_by_t.get(key, 0.0) + float(value)
        accepted_count_by_t: Dict[int, float] = {}
        accepted = _numeric(event_rows, "accepted_count")
        if accepted is not None:
            for position, t_node in enumerate(node_t):
                if np.isfinite(t_node) and np.isfinite(accepted[position]):
                    key = int(t_node)
                    accepted_count_by_t[key] = accepted_count_by_t.get(key, 0.0) + float(accepted[position])

    b_edges = np.linspace(config.b_grid_min, config.b_grid_max, int(config.b_grid_points) + 1)
    z_edges = np.linspace(config.z_grid_min, config.z_grid_max, int(config.z_grid_points) + 1)

    slice_rows: List[Dict[str, Any]] = []
    period_rows: List[Dict[str, Any]] = []
    marginal_rows: List[Dict[str, Any]] = []
    surfaces: Dict[str, np.ndarray] = {}
    ids_by_t: Dict[int, set] = {}
    for t_slice in ordered_slices:
        in_slice = np.isfinite(t_values) & (t_values == float(t_slice))
        ids_by_t[int(t_slice)] = set(id_values[in_slice].tolist())

    for t_slice in ordered_slices:
        in_slice = np.isfinite(t_values) & (t_values == float(t_slice))
        paths_at_t = sorted({int(value) for value in path_values[in_slice]})
        extinct_paths = [
            int(row["path"])
            for _, row in status.iterrows()
            if bool(row["extinct"]) and (row["stop_time"] is not None and int(row["stop_time"]) <= t_slice)
        ]
        n_entrants = (
            float(np.nansum(entry_values[in_slice] > 0.5)) if bool(in_slice.any()) else 0.0
        )
        n_investors = (
            float(np.nansum(investment_values[in_slice] > 0.5))
            if bool(in_slice.any())
            else 0.0
        )
        if t_slice > min(ordered_slices):
            previous_ids = ids_by_t.get(t_slice - 1, set())
            current_ids = ids_by_t.get(t_slice, set())
            n_defaults = float(len(previous_ids - current_ids))
        else:
            n_defaults = 0.0
        n_defaults_total = n_defaults + same_node_exit_by_t.get(int(t_slice), 0.0)
        n_entrants_event = accepted_count_by_t.get(int(t_slice), _nan())

        period_rows.append(
            {
                "t": float(t_slice),
                "active_paths": float(len(paths_at_t)),
                "extinct_paths": float(len(extinct_paths)),
                "extinct_path_ids": ",".join(str(value) for value in extinct_paths),
                "n_alive_firms": float(int(in_slice.sum())),
                "n_entrants": n_entrants,
                "n_entrants_event": n_entrants_event,
                "n_defaults": n_defaults_total,
                "n_defaults_incumbent": n_defaults,
                "n_same_node_entrant_exits": same_node_exit_by_t.get(int(t_slice), 0.0),
                "n_investors": n_investors,
            }
        )

        for caliber, weights in (
            ("count", np.ones_like(k_values)),
            ("capital", k_values),
        ):
            for state_label in ("all",) + MACRO_STATE_LABELS:
                mask = in_slice.copy()
                if state_label != "all":
                    mask = mask & (macro_state == str(state_label))
                if not bool(mask.any()):
                    continue
                w = weights[mask]
                b_slice = b_values[mask]
                z_slice = z_values[mask]
                k_slice = k_values[mask]
                b_moments = _weighted_moments(b_slice, w)
                z_moments = _weighted_moments(z_slice, w)
                total_w = float(np.nansum(w))
                if total_w > 0:
                    mean_b = float(np.nansum(w * b_slice) / total_w)
                    mean_z = float(np.nansum(w * z_slice) / total_w)
                    cov_bz = float(
                        np.nansum(w * (b_slice - mean_b) * (z_slice - mean_z)) / total_w
                    )
                else:
                    mean_b = mean_z = cov_bz = _nan()
                cap_total = float(np.nansum(k_slice))
                cap_mean_b = (
                    float(np.nansum(k_slice * b_slice) / cap_total)
                    if cap_total > 0
                    else _nan()
                )
                slice_rows.append(
                    {
                        "t": float(t_slice),
                        "caliber": caliber,
                        "macro_state": state_label,
                        "n_firms": float(int(mask.sum())),
                        "active_paths": float(len({int(v) for v in path_values[mask]})),
                        "extinct_paths": float(len(extinct_paths)),
                        "extinct_path_ids": ",".join(str(v) for v in extinct_paths),
                        "n_entrants": n_entrants,
                        "n_defaults": n_defaults_total,
                        "n_investors": n_investors,
                        "b_mean": b_moments["mean"],
                        "b_var": b_moments["var"],
                        "z_mean": z_moments["mean"],
                        "z_var": z_moments["var"],
                        "cov_bz": cov_bz,
                        "capital_weighted_mean_b": cap_mean_b,
                        "default_rate": _safe_ratio(
                            n_defaults_total, float(int(mask.sum()))
                        ),
                        "investment_rate": _safe_ratio(
                            n_investors, float(int(mask.sum()))
                        ),
                        "entry_rate": _safe_ratio(n_entrants, float(int(mask.sum()))),
                    }
                )

        for caliber, weights in (
            ("count", np.ones_like(k_values)),
            ("capital", k_values),
        ):
            w = weights[in_slice]
            b_slice = b_values[in_slice]
            z_slice = z_values[in_slice]
            hist_b, _ = np.histogram(b_slice, bins=b_edges, weights=w)
            hist_z, _ = np.histogram(z_slice, bins=z_edges, weights=w)
            hist_b = hist_b / hist_b.sum() if hist_b.sum() > 0 else hist_b
            hist_z = hist_z / hist_z.sum() if hist_z.sum() > 0 else hist_z
            for bin_index in range(len(hist_b)):
                marginal_rows.append(
                    {
                        "t": float(t_slice),
                        "caliber": caliber,
                        "axis": "b",
                        "bin_center": float(0.5 * (b_edges[bin_index] + b_edges[bin_index + 1])),
                        "weight": float(hist_b[bin_index]),
                    }
                )
            for bin_index in range(len(hist_z)):
                marginal_rows.append(
                    {
                        "t": float(t_slice),
                        "caliber": caliber,
                        "axis": "z",
                        "bin_center": float(0.5 * (z_edges[bin_index] + z_edges[bin_index + 1])),
                        "weight": float(hist_z[bin_index]),
                    }
                )
            hist2d = joint_histogram(
                b_slice, z_slice, b_edges=b_edges, z_edges=z_edges, weights=w
            )
            surfaces[f"bz_hist_t{t_slice}_{caliber}"] = hist2d

    # Distribution distances between consecutive slices and against t0.
    distance_rows: List[Dict[str, float]] = []
    first_slice = ordered_slices[0]
    for caliber, weights in (
        ("count", np.ones_like(k_values)),
        ("capital", k_values),
    ):
        pairs: List[Tuple[int, int]] = [
            (ordered_slices[index - 1], ordered_slices[index])
            for index in range(1, len(ordered_slices))
        ]
        pairs.extend((first_slice, t_slice) for t_slice in ordered_slices[1:])
        for t_from, t_to in pairs:
            mask_from = np.isfinite(t_values) & (t_values == float(t_from))
            mask_to = np.isfinite(t_values) & (t_values == float(t_to))
            if not (bool(mask_from.any()) and bool(mask_to.any())):
                continue
            w_from, w_to = weights[mask_from], weights[mask_to]
            distance_rows.append(
                {
                    "t_from": float(t_from),
                    "t_to": float(t_to),
                    "caliber": caliber,
                    "w1_b": wasserstein_1(
                        b_values[mask_from], b_values[mask_to],
                        weights_a=w_from, weights_b=w_to,
                    ),
                    "w1_z": wasserstein_1(
                        z_values[mask_from], z_values[mask_to],
                        weights_a=w_from, weights_b=w_to,
                    ),
                    **joint_histogram_distance(
                        surfaces[f"bz_hist_t{t_from}_{caliber}"],
                        surfaces[f"bz_hist_t{t_to}_{caliber}"],
                    ),
                }
            )
    distances = pd.DataFrame(distance_rows)

    metrics: Dict[str, float] = {
        "dist_n_slices": float(len(ordered_slices)),
        "dist_min_active_paths": float(
            min(row["active_paths"] for row in period_rows)
        ),
        "dist_max_active_paths": float(
            max(row["active_paths"] for row in period_rows)
        ),
        "dist_extinct_paths_total": float(
            int((status["extinct"].astype(bool)).sum()) if not status.empty else 0
        ),
        "dist_dual_view_rows_excluded": float(
            cross_report.get("dual_view_rows_excluded", 0)
        ),
    }
    if not distances.empty:
        for caliber in ("count", "capital"):
            for t_from, t_to in ((first_slice, ordered_slices[-1]),):
                subset = distances.loc[
                    (distances["caliber"] == caliber)
                    & (distances["t_from"] == float(t_from))
                    & (distances["t_to"] == float(t_to))
                ]
                if subset.empty:
                    continue
                row = subset.iloc[0]
                metrics[f"dist_w1_b_{t_from}_to_{t_to}_{caliber}"] = float(row["w1_b"])
                metrics[f"dist_w1_z_{t_from}_to_{t_to}_{caliber}"] = float(row["w1_z"])
                metrics[f"dist_hist2d_l1_{t_from}_to_{t_to}_{caliber}"] = float(
                    row["hist2d_l1"]
                )

    diagnostics = {
        "distribution_calibres": {
            "count": "firm-count weighted",
            "capital": "firm capital weighted (K)",
        },
        "macro_state_rule": {
            "column": "x (firm-row macro state)",
            "quantiles": list(config.macro_state_quantiles),
            "labels": list(MACRO_STATE_LABELS),
        },
        "joint_histogram_distance": (
            "fixed-grid L1 / total-variation distance of normalized b-z "
            "histograms; full 2D optimal transport is deliberately not used"
        ),
        "extinct_path_policy": (
            "extinct paths are retained in path_status and counted in every "
            "period's extinct_paths column; slices annotate their ids instead of "
            "silently dropping them"
        ),
        "time_slices": ordered_slices,
        "cross_section_view_report": cross_report,
        "node_view_report": node_report,
        "optimizer_steps": count_optimizer_steps(),
        "evaluation_only": True,
    }
    return SimEvalResult(
        metrics=metrics,
        tables={
            "distribution_slices": pd.DataFrame(slice_rows),
            "distribution_marginals": pd.DataFrame(marginal_rows),
            "distribution_distances": distances,
            "period_counts": pd.DataFrame(period_rows),
            "path_status": status,
        },
        surfaces=surfaces,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# fixed rollout driver (Bank E -> existing simulation chain)
# ---------------------------------------------------------------------------


@contextmanager
def _economic_config_context(config: AnalysisEconomicConfig):
    """Apply a checkpoint's economic config snapshot to the global ``Config``."""
    saved = {name: getattr(Config, name) for name in config.field_names()}
    try:
        for name, value in config.to_dict().items():
            setattr(Config, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(Config, name, value)


def _sim_config_class(economic_config: AnalysisEconomicConfig, *, device: torch.device):
    """Config *class* for ``SimulateTS`` carrying the checkpoint's economics."""
    namespace: Dict[str, Any] = {}
    for name in dir(Config):
        if name.isupper():
            namespace[name] = getattr(Config, name)
    namespace.update(economic_config.to_dict())
    namespace["DEVICE"] = device
    return type("EvalSimConfig", (), namespace)


def _hyperparam_or(hyperparams: Any, name: str, default: Any) -> Any:
    value = getattr(hyperparams, name, None)
    return default if value is None else value


def _build_initial_state(
    bank_e: Mapping[str, Any],
    variant_name: str,
    *,
    horizon: int,
    capacity_extra_per_step: int,
    initial_entry_flag: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Build the batched rollout state from a Bank E initial cross-section.

    The layout mirrors ``data.simulate_ts_parallel._initialize_batched_state``
    so the existing simulation chain consumes it unchanged; firm identities are
    the per-path positions (as in the production initializer), so the
    ``stable_firm_identity`` keyed RNG draws identical shocks for every
    checkpoint evaluated on the same bank.
    """
    variants = dict(bank_e.get("variants") or {})
    if variant_name not in variants:
        raise SimEvalError(
            "MISSING_LEVERAGE_VARIANT",
            f"Bank E has no leverage variant {variant_name!r}; available: {sorted(variants)}",
        )
    variant = variants[variant_name]
    b0 = torch.as_tensor(np.asarray(variant["b0"]).reshape(-1), dtype=torch.float32)
    z0 = torch.as_tensor(np.asarray(variant["z0"]).reshape(-1), dtype=torch.float32)
    eta0 = torch.as_tensor(np.asarray(variant["eta0"]).reshape(-1), dtype=torch.float32)
    i0 = torch.as_tensor(np.asarray(variant["i0"]).reshape(-1), dtype=torch.float32)
    k0 = torch.as_tensor(np.asarray(variant["K0"]).reshape(-1), dtype=torch.float32)
    x0 = torch.as_tensor(np.asarray(variant["x0"]).reshape(-1), dtype=torch.float32)
    n_firms = int(b0.shape[0])
    for name, tensor in (("z0", z0), ("eta0", eta0), ("i0", i0), ("K0", k0), ("x0", x0)):
        if int(tensor.shape[0]) != n_firms:
            raise SimEvalError(
                "INVALID_BANK_E_VARIANT",
                f"Bank E variant {variant_name!r} field {name} has {int(tensor.shape[0])} "
                f"rows, expected {n_firms}",
            )

    keys = dict(bank_e.get("pairing_keys") or {})
    path_ids = np.asarray(
        pd.to_numeric(pd.Series(list(keys.get("path", []))), errors="coerce").tolist()
        if "path" in keys and len(keys["path"]) == n_firms
        else [],
        dtype=np.float64,
    )
    if path_ids.size != n_firms:
        raise SimEvalError(
            "INVALID_BANK_E_VARIANT",
            "Bank E pairing_keys['path'] must provide one path id per firm",
        )
    paths = sorted(int(value) for value in np.unique(path_ids))
    n_paths = int(len(paths))
    counts = {path: int((path_ids == path).sum()) for path in paths}
    max_count = max(counts.values()) if counts else 1
    n_potential = max(int(capacity_extra_per_step), int(max_count * 0.1))
    capacity = max_count + int(horizon) * n_potential

    def _blank_firm(shape: Tuple[int, int], fill: float = 0.0) -> torch.Tensor:
        return torch.full(shape, float(fill), dtype=torch.float32, device=device)

    b = _blank_firm((n_paths, capacity))
    z = _blank_firm((n_paths, capacity))
    eta = _blank_firm((n_paths, capacity))
    i = _blank_firm((n_paths, capacity))
    k = _blank_firm((n_paths, capacity))
    alive = torch.zeros((n_paths, capacity), dtype=torch.bool, device=device)
    entry = _blank_firm((n_paths, capacity))
    firm_id = torch.full((n_paths, capacity), -1, dtype=torch.long, device=device)
    x_path = torch.zeros(n_paths, dtype=torch.float32, device=device)
    ids_by_path: Dict[int, List[str]] = {}
    id_strings = (
        [str(value) for value in keys["ID"]]
        if "ID" in keys and len(keys["ID"]) == n_firms
        else [str(index) for index in range(n_firms)]
    )
    for path_index, path in enumerate(paths):
        positions = np.nonzero(path_ids == path)[0]
        count = int(positions.size)
        b[path_index, :count] = b0[positions]
        z[path_index, :count] = z0[positions]
        eta[path_index, :count] = eta0[positions]
        i[path_index, :count] = i0[positions]
        k[path_index, :count] = k0[positions]
        alive[path_index, :count] = True
        firm_id[path_index, :count] = torch.arange(count, dtype=torch.long, device=device)
        x_path[path_index] = float(x0[positions[0]])
        ids_by_path[path] = [id_strings[int(position)] for position in positions]

    entry = alive.to(torch.float32) * float(initial_entry_flag)
    total_k = (k * alive.to(k.dtype)).sum(dim=1)
    state: Dict[str, torch.Tensor] = {
        "x": x_path,
        "b": b,
        "z": z,
        "eta": eta,
        "i": i,
        "K": k,
        "hatcf": torch.zeros(n_paths, dtype=torch.float32, device=device),
        "lnkf": torch.log(total_k.clamp_min(1e-12)).to(torch.float32),
        "M": torch.ones(n_paths, dtype=torch.float32, device=device),
        "alive": alive,
        "entry": entry,
        "initial_cohort": alive.to(torch.float32),
        "birth_time": torch.full_like(k, -1.0),
        "entry_cost": torch.zeros_like(k),
        "K_birth": torch.where(alive, k, torch.zeros_like(k)),
        "entry_P0": torch.full_like(k, float("nan")),
        "entry_PI": torch.full_like(k, float("nan")),
        "entry_value_gap": torch.full_like(k, float("nan")),
        "firm_id": firm_id,
        "next_firm_id": torch.full(
            (n_paths,), max_count, dtype=torch.long, device=device
        ),
        "bar_i": torch.zeros_like(b),
        "bar_z": torch.zeros_like(b),
        "bp": b.clone(),
    }
    report = {
        "variant": str(variant_name),
        "leverage_factor": variant.get("leverage_factor"),
        "n_paths": n_paths,
        "n_firms": n_firms,
        "capacity": capacity,
        "n_potential_per_step": n_potential,
        "K0_total": float(total_k.sum().item()),
        "ids_by_path": ids_by_path,
    }
    return state, report


def _nonfinite_share(frames: Sequence[pd.DataFrame], columns: Sequence[str]) -> Tuple[float, List[str]]:
    total = 0
    nonfinite = 0
    used: List[str] = []
    for frame in frames:
        for name in columns:
            if name not in frame.columns:
                continue
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
            total += int(values.size)
            nonfinite += int((~np.isfinite(values)).sum())
            if name not in used:
                used.append(name)
    if total == 0:
        return _nan(), used
    return float(nonfinite / total), used


def _evaluate_variant(
    cp: EvalCheckpoint,
    bank_e: Mapping[str, Any],
    variant_name: str,
    *,
    config: SimEvalConfig,
) -> Dict[str, Any]:
    """Run one fixed rollout (one leverage variant) through the simulate chain."""
    from data.simulate_ts_parallel import (  # local import: keeps module import light
        _calibrate_initial_hstar_batched,
        _ensure_entry_state_fields,
    )

    hyperparams = cp.hyperparams
    device = torch.device(config.device)
    horizon = int(config.horizon)
    state, state_report = _build_initial_state(
        bank_e,
        variant_name,
        horizon=horizon,
        capacity_extra_per_step=int(config.capacity_extra_per_step),
        initial_entry_flag=(1.0 if str(_hyperparam_or(hyperparams, "entry_mode", "legacy")) == "legacy" else 0.0),
        device=device,
    )
    _ensure_entry_state_fields(state)

    config_class = _sim_config_class(cp.economic_config, device=device)
    shock_spec = dict(bank_e.get("shock_spec") or {})
    common_transition_seed = shock_spec.get("seed", bank_e.get("seed"))
    sim_kwargs: Dict[str, Any] = {
        "models": cp.models,
        "config": config_class,
        "n_paths": int(state_report["n_paths"]),
        "group_size": int(state_report["n_firms"]) // max(int(state_report["n_paths"]), 1),
        "horizon": horizon,
        "branch_num": int(config.branch_num),
        "main_branch": int(config.main_branch),
        "enable_entry": bool(config.enable_entry),
        "enable_exit": bool(config.enable_exit),
        "device": device,
        "entry_mode": str(_hyperparam_or(hyperparams, "entry_mode", "legacy")),
        "entry_spec_version": str(
            _hyperparam_or(hyperparams, "entry_spec_version", "value_cost_v1")
        ),
        "entry_capital_ratio": float(
            _hyperparam_or(hyperparams, "entry_capital_ratio", 0.10)
        ),
        "entry_size_ratio": float(_hyperparam_or(hyperparams, "entry_size_ratio", 1.0)),
        "entry_cost_max": float(_hyperparam_or(hyperparams, "entry_cost_max", 1.0)),
        "entry_dummy_i": float(_hyperparam_or(hyperparams, "entry_dummy_i", 0.0)),
        "entry_inference_chunk_size": int(
            _hyperparam_or(hyperparams, "entry_inference_chunk_size", 65536)
        ),
        "entry_rng_seed": int(_hyperparam_or(hyperparams, "entry_rng_seed", 86420)),
        "entry_rng_mode": str(
            _hyperparam_or(hyperparams, "entry_rng_mode", "legacy_sequential")
        ),
        "consumption_aggregation_mode": str(
            _hyperparam_or(hyperparams, "consumption_aggregation_mode", "legacy_per_firm_clamp")
        ),
        "consumption_floor": float(_hyperparam_or(hyperparams, "consumption_floor", 0.0)),
        "node_accounting_mode": str(
            config.node_accounting_mode
            or _hyperparam_or(hyperparams, "node_accounting_mode", "auto")
        ),
        "entry_independent_resource_validation_enabled": bool(
            _hyperparam_or(hyperparams, "entry_independent_resource_validation_enabled", False)
        ),
        "preserve_global_rng_around_entry": bool(
            _hyperparam_or(hyperparams, "preserve_global_rng_around_entry", False)
        ),
        # Fixed Bank E shock specification: keyed RNG shared by every checkpoint.
        "transition_rng_mode": str(shock_spec.get("rng_mode", "stable_firm_identity")),
        "common_transition_seed": (
            None if common_transition_seed is None else int(common_transition_seed)
        ),
        # The initial h-star is calibrated explicitly below on the fixed Bank E
        # cross-section, so the in-chain calibration stays off.
        "initial_hstar_enabled": False,
        "initial_hstar_residual_tolerance": float(
            _hyperparam_or(hyperparams, "initial_hstar_residual_tolerance", 1e-5)
        ),
        "initial_hstar_parity_tolerance": float(
            _hyperparam_or(hyperparams, "initial_hstar_parity_tolerance", 1e-5)
        ),
        "initial_hstar_max_iterations": int(
            _hyperparam_or(hyperparams, "initial_hstar_max_iterations", 40)
        ),
        "initial_hstar_coarse_points": int(
            _hyperparam_or(hyperparams, "initial_hstar_coarse_points", 10)
        ),
        "initial_hstar_refinement_points": int(
            _hyperparam_or(hyperparams, "initial_hstar_refinement_points", 20)
        ),
        "initial_hstar_max_refinement_rounds": int(
            _hyperparam_or(hyperparams, "initial_hstar_max_refinement_rounds", 4)
        ),
        "initial_hstar_support_lower": float(
            _hyperparam_or(hyperparams, "initial_hstar_support_lower", -8.0)
        ),
        "initial_hstar_support_upper": float(
            _hyperparam_or(hyperparams, "initial_hstar_support_upper", 2.0)
        ),
        "initial_hstar_width_tolerance": float(
            _hyperparam_or(hyperparams, "initial_hstar_width_tolerance", 1e-5)
        ),
    }
    sim = SimulateTS(**sim_kwargs)

    hstar_diagnostics: Dict[str, Any] = {"enabled": bool(config.calibrate_initial_hstar)}
    if bool(config.calibrate_initial_hstar):
        # Same rule for every checkpoint, calibrated on this checkpoint's own
        # accounting operator; the result is recorded per checkpoint so the
        # differences across checkpoints are explicit.
        hstar_diagnostics = dict(_calibrate_initial_hstar_batched(sim, state))
        selected_h = getattr(sim, "_initial_hstar_selected_h", None)
        if selected_h is not None:
            hstar_diagnostics["selected_hstar_per_path"] = [
                float(value) for value in selected_h.detach().cpu().reshape(-1).tolist()
            ]
    sim.initial_state = state

    torch.manual_seed(int(bank_e.get("seed") or 0))
    with _economic_config_context(cp.economic_config):
        firm_df, macro_df = sim.simulate()
    meta = dict(getattr(sim, "last_simulation_meta", {}) or {})

    waterfall = evaluate_waterfall(firm_df, macro_df, config=config)
    distributions = evaluate_distributions(firm_df, macro_df, config=config)

    node_rows, _ = select_node_rows(macro_df)
    cross_section, _ = select_cross_section_rows(firm_df)
    t_values = pd.to_numeric(cross_section["t"], errors="coerce").to_numpy(dtype=np.float64)
    k_values = _numeric(cross_section, "K")
    id_values = (
        cross_section["ID"].astype(str).to_numpy()
        if "ID" in cross_section.columns
        else np.array([""] * len(cross_section))
    )
    k0_total = float(np.nansum(k_values[np.isfinite(t_values) & (t_values == 0.0)])) if k_values is not None else _nan()
    k1_total = float(np.nansum(k_values[np.isfinite(t_values) & (t_values == 1.0)])) if k_values is not None else _nan()
    ids0 = set(id_values[np.isfinite(t_values) & (t_values == 0.0)].tolist())
    ids1 = set(id_values[np.isfinite(t_values) & (t_values == 1.0)].tolist())
    initial_cohort_defaults = len(ids0 - ids1)

    lnk_values = _numeric(node_rows, "LnK")
    lnkf_values = _numeric(node_rows, "lnkf")
    hatc_values = _numeric(node_rows, "Hatc")
    hatcf_values = _numeric(node_rows, "hatcf")
    fc1_lnk_error = (
        lnkf_values - lnk_values
        if lnk_values is not None and lnkf_values is not None
        else np.asarray([_nan()])
    )
    fc1_hatc_error = (
        hatcf_values - hatc_values
        if hatc_values is not None and hatcf_values is not None
        else np.asarray([_nan()])
    )

    waterfall_table = waterfall.tables["capital_waterfall"]
    first_transition = waterfall_table.loc[
        (waterfall_table["t_from"] == 0) & (waterfall_table["t_to"] == 1)
    ]
    entrant_capital_t1 = (
        float(first_transition["entrant_capital"].sum())
        if not first_transition.empty
        else _nan()
    )
    incumbent_dk_t1 = (
        float(first_transition["incumbent_dk"].sum())
        if not first_transition.empty
        else _nan()
    )
    entrant_count_t1 = (
        float(np.nansum(first_transition["entrant_capital_gross"].to_numpy(dtype=np.float64)))
        if not first_transition.empty
        else _nan()
    )

    resource_feasible_share = _nan()
    if "resource_feasible" in macro_df.columns:
        values = pd.to_numeric(macro_df["resource_feasible"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        resource_feasible_share = float((finite > 0.5).mean()) if finite.size else _nan()
    nonfinite_share, nonfinite_columns = _nonfinite_share(
        [firm_df, macro_df],
        ("b", "z", "K", "x", "Hatcf", "LnKF", "Bar_i", "Bar_z", "P", "C", "LnK", "Hatc"),
    )

    fixed_shock_row = {
        "variant": str(variant_name),
        "leverage_factor": state_report.get("leverage_factor"),
        "horizon": float(horizon),
        "k0_total": k0_total,
        "k_t1_total": k1_total,
        "capital_change_t0_t1": _safe_ratio(
            (k1_total - k0_total) if math.isfinite(k1_total) and math.isfinite(k0_total) else _nan(),
            k0_total,
        ),
        "capital_drop_t0_t1": _safe_ratio(
            (k0_total - k1_total) if math.isfinite(k1_total) and math.isfinite(k0_total) else _nan(),
            k0_total,
        ),
        "incumbent_capital_change_t0_t1": _safe_ratio(incumbent_dk_t1, k0_total),
        "initial_cohort_default_count_t1": float(initial_cohort_defaults),
        "initial_cohort_default_share_t1": _safe_ratio(
            float(initial_cohort_defaults), float(len(ids0)) if ids0 else _nan()
        ),
        "entrant_capital_t1": entrant_capital_t1,
        "entrant_capital_gross_t1": entrant_count_t1,
        "entry_refill_ratio_t1": _safe_ratio(entrant_capital_t1, k0_total),
        "fc1_lnk_bias": _safe_mean(fc1_lnk_error),
        "fc1_lnk_mae": _safe_mean(np.abs(fc1_lnk_error)),
        "fc1_hatc_bias": _safe_mean(fc1_hatc_error),
        "completed_horizon": float(meta.get("completed_parent_steps", _nan())),
        "stopped_early": float(int(bool(meta.get("stopped_early", False)))),
        "n_extinct_paths": float(meta.get("n_extinct_paths", 0)),
        "resource_feasible_share": resource_feasible_share,
        "nonfinite_share": nonfinite_share,
    }

    rollout_diagnostics = {
        "variant": str(variant_name),
        "completed_horizon": int(meta.get("completed_parent_steps", horizon)),
        "stopped_early": bool(meta.get("stopped_early", False)),
        "stop_reason": meta.get("stop_reason"),
        "stop_time": meta.get("stop_time"),
        "extinct_paths": list(meta.get("extinct_paths", []) or []),
        "n_extinct_paths": int(meta.get("n_extinct_paths", 0)),
        "path_stop_times": list(meta.get("path_stop_times", []) or []),
        "resource_feasible_share": resource_feasible_share,
        "nonfinite_share": nonfinite_share,
        "nonfinite_columns": nonfinite_columns,
        "initial_state": state_report,
        "initial_hstar": hstar_diagnostics,
        "shock_spec": shock_spec,
        "simulate_kwargs": {
            key: value
            for key, value in sim_kwargs.items()
            if key not in {"models", "config", "initial_state"}
        },
    }
    return {
        "firm_df": firm_df,
        "macro_df": macro_df,
        "waterfall": waterfall,
        "distributions": distributions,
        "fixed_shock_row": fixed_shock_row,
        "rollout_diagnostics": rollout_diagnostics,
    }


def _variant_slug(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def evaluate_simulation(
    cp: EvalCheckpoint,
    bank_e: Mapping[str, Any],
    *,
    config: Optional[SimEvalConfig] = None,
) -> SimEvalResult:
    """Fixed-rollout simulation evaluation from a checkpoint plus Bank E.

    For every leverage variant of Bank E (Low / Baseline / High by default) the
    existing simulation chain (``data.simulate_ts`` -> ``simulate_ts_parallel``,
    called, never modified) is driven from the fixed initial cross-section under
    the bank's ``stable_firm_identity`` keyed-RNG shock specification.  The
    initial h-star is calibrated **per checkpoint** with the production
    ``_calibrate_initial_hstar_batched`` chain and its difference is recorded in
    the diagnostics; the FC1 beliefs are never overwritten after that.

    Scaled-down CPU smoke runs are supported through
    :class:`SimEvalConfig` (``horizon`` / ``capacity_extra_per_step`` /
    ``branch_num``); large-scale runs use the same entry points with production
    sizes.
    """
    config = config or SimEvalConfig()
    for module in cp.models.values():
        if isinstance(module, torch.nn.Module):
            module.eval()

    variants = dict(bank_e.get("variants") or {})
    selected = [str(name) for name in config.variants if str(name) in variants]
    if not selected:
        selected = [str(name) for name in variants.keys()]
    if not selected:
        raise SimEvalError(
            "MISSING_LEVERAGE_VARIANT", "Bank E payload has no leverage variants"
        )

    metrics: Dict[str, float] = {}
    tables: Dict[str, pd.DataFrame] = {}
    surfaces: Dict[str, np.ndarray] = {}
    diagnostics: Dict[str, Any] = {
        "bank_seed": bank_e.get("seed"),
        "shock_spec": dict(bank_e.get("shock_spec") or {}),
        "horizon": int(config.horizon),
        "variants": selected,
        "rollouts": {},
        "optimizer_steps": count_optimizer_steps(),
        "evaluation_only": True,
    }
    fixed_rows: List[Dict[str, Any]] = []

    with param_hash_guard(cp):
        with _economic_config_context(cp.economic_config):
            for variant_name in selected:
                outcome = _evaluate_variant(cp, bank_e, variant_name, config=config)
                slug = _variant_slug(variant_name)
                for source in (outcome["waterfall"], outcome["distributions"]):
                    for key, value in source.metrics.items():
                        metrics[f"sim_{slug}_{key}"] = value
                    for key, frame in source.tables.items():
                        tables[f"{key}_{slug}"] = frame
                    for key, array in source.surfaces.items():
                        surfaces[f"{key}_{slug}"] = array
                tables[f"firm_panel_{slug}"] = outcome["firm_df"]
                tables[f"macro_panel_{slug}"] = outcome["macro_df"]
                diagnostics["rollouts"][str(variant_name)] = outcome["rollout_diagnostics"]
                fixed_rows.append(outcome["fixed_shock_row"])

    fixed_frame = pd.DataFrame(fixed_rows)
    tables["fixed_shock_metrics"] = fixed_frame
    for _, row in fixed_frame.iterrows():
        slug = _variant_slug(row["variant"])
        for name in (
            "capital_change_t0_t1",
            "capital_drop_t0_t1",
            "initial_cohort_default_share_t1",
            "entry_refill_ratio_t1",
            "fc1_lnk_bias",
            "fc1_lnk_mae",
            "resource_feasible_share",
            "nonfinite_share",
        ):
            metrics[f"sim_{slug}_{name}"] = float(row[name])
    metrics["sim_completed_horizon_min"] = float(fixed_frame["completed_horizon"].min())
    metrics["sim_stopped_early_any"] = float(
        int(bool(fixed_frame["stopped_early"].fillna(0).astype(float).max()))
    )
    metrics["sim_n_extinct_paths_max"] = float(fixed_frame["n_extinct_paths"].max())
    metrics["sim_resource_feasible_share_min"] = float(
        fixed_frame["resource_feasible_share"].min()
    )
    metrics["sim_nonfinite_share_max"] = float(fixed_frame["nonfinite_share"].max())

    return SimEvalResult(
        metrics=metrics, tables=tables, surfaces=surfaces, diagnostics=diagnostics
    )


def compare_fixed_shock_rollouts(
    results: Mapping[str, SimEvalResult] | Sequence[Tuple[str, SimEvalResult]],
    *,
    config: Optional[SimEvalConfig] = None,
) -> pd.DataFrame:
    """Cross-checkpoint comparison under one fixed Bank E shock specification.

    For the same bank and the same initial cohort the table reports the t0->t1
    capital drop, the initial-cohort default change, the entry refill and the
    FC1 ``LnKF vs LnK`` bias per checkpoint (and per leverage variant), plus
    per-variant deltas against the first supplied checkpoint.
    """
    items = list(results.items()) if isinstance(results, Mapping) else list(results)
    if not items:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for label, result in items:
        frame = result.tables.get("fixed_shock_metrics")
        if frame is None or frame.empty:
            continue
        for _, row in frame.iterrows():
            record = row.to_dict()
            record["label"] = str(label)
            rows.append(record)
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    reference_label = str(items[0][0])
    reference = table.loc[table["label"] == reference_label]
    numeric_columns = [
        name
        for name in table.columns
        if name not in {"label", "variant"} and pd.api.types.is_numeric_dtype(table[name])
    ]
    for variant in sorted(set(table["variant"].tolist())):
        mask = table["variant"] == variant
        ref_rows = reference.loc[reference["variant"] == variant]
        if ref_rows.empty:
            continue
        ref_row = ref_rows.iloc[0]
        for name in numeric_columns:
            value = table.loc[mask, name]
            table.loc[mask, f"{name}_delta_vs_reference"] = value - float(ref_row[name])
    table.attrs["reference_label"] = reference_label
    return table
