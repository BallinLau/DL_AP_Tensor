"""Figure renderer for the equilibrium convergence evaluator.

Renders the Prompt figure set (F01-F12 main figures plus the V / P / M sub
series) from the result objects produced by the sibling evaluators:

* ``firm_bellman.BellmanEvalResult``  -- ``metrics`` / ``tables`` / ``surfaces``
* ``firm_policy.PolicyEvalResult``    -- ``metrics`` / ``tables`` / ``surfaces``
* ``macro_sdf.MacroSdfEvalResult``    -- ``metrics`` / ``tables`` / ``surfaces``
* ``sim_eval.SimEvalResult``          -- ``metrics`` / ``tables`` / ``surfaces``

Design rules enforced here:

* **No fabricated figures.**  A figure whose data is missing is *skipped*: it
  gets ``status="skipped"``, a stable ``missing_reason`` code, an explanatory
  ``caption`` and **no file on disk**.  A blank "successful" PNG is never
  written.
* **Unified axes and colour scale.**  Every firm surface is drawn on the same
  ``b``/``z`` support and with the same per-family ``(vmin, vmax)`` so that
  surfaces are comparable across episodes and banks.  The applied ranges are
  recorded in ``FigureRecord.meta`` (``b_range`` / ``z_range`` /
  ``value_limits``).
* **CSV / heatmap conventions** of ``evaluation/plotting.py`` are preserved:
  surfaces are indexed ``[b_index, z_index]`` and transposed for
  ``pcolormesh`` / ``plot_surface`` (``x = b``, ``y = z``).
* **Every figure carries a caption** (returned in ``FigureRecord.caption``)
  explaining its economic meaning, and a footer with the Episode label, macro
  anchor, ``eta`` view, ``i`` slice, data distribution source, sample count,
  capital weighting and whether any path went extinct.
* 3D panels use ``mpl_toolkits.mplot3d`` on the Agg backend and always include
  the canonical surface set ``P0 / PI / P / Q_claim / Grid bp0* / Grid bpI*``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)


# ---------------------------------------------------------------------------
# stable reason codes (missing data -> skipped figure)
# ---------------------------------------------------------------------------

REASON_NO_POLICY_SURFACES = "MISSING_POLICY_SURFACES"
REASON_NO_GRID_AXES = "MISSING_BZ_GRID_AXES"
REASON_NO_DEFAULT_BOUNDARY = "MISSING_DEFAULT_BOUNDARY"
REASON_NO_Q_SURFACES = "MISSING_Q_SURFACES"
REASON_NO_BP_SURFACES = "MISSING_BP_SURFACES"
REASON_NO_INVESTMENT_BOUNDARY = "MISSING_INVESTMENT_BOUNDARY"
REASON_NO_BELLMAN_RESULTS = "MISSING_BELLMAN_RESULTS"
REASON_NO_BELLMAN_METRICS = "MISSING_BELLMAN_METRICS"
REASON_NO_POLICY_ROWS = "MISSING_POLICY_ROWS"
REASON_NO_MACRO_PAIRS = "MISSING_MACRO_PAIRS"
REASON_NO_MACRO_ANCHOR_PATHS = "MISSING_MACRO_ANCHOR_PATHS"
REASON_NO_RECURSION_TABLE = "MISSING_FC1_RECURSION_TABLE"
REASON_NO_SDF_TABLE = "MISSING_SDF_TABLE"
REASON_NO_SIM_PANEL = "MISSING_SIM_PANEL"
REASON_NO_JOINT_HIST = "MISSING_JOINT_HISTOGRAM"
REASON_NO_MARGINALS = "MISSING_DISTRIBUTION_MARGINALS"
REASON_NO_CAPITAL_WATERFALL = "MISSING_CAPITAL_WATERFALL"
REASON_NO_SURVIVAL_ACCOUNT = "MISSING_PERIOD_COUNTS"
REASON_NO_OBJECTIVE_SLICES = "MISSING_OBJECTIVE_SLICES"
REASON_INSUFFICIENT_EPISODES = "INSUFFICIENT_EPISODES_FOR_DRIFT"
REASON_POLICY_ROWS_NOT_ALIGNED = "POLICY_ROWS_NOT_ALIGNED_ACROSS_EPISODES"
REASON_INSUFFICIENT_BLOCKS = "INSUFFICIENT_RESULT_BLOCKS"
REASON_NO_FINITE_SAMPLES = "NO_FINITE_SAMPLES"
REASON_NO_MACRO_PANEL = "MISSING_MACRO_PANEL"


# ---------------------------------------------------------------------------
# figure registry
# ---------------------------------------------------------------------------

FIGURE_FILENAMES = {
    "F01": "F01_p0_pi_p_surface.png",
    "F02": "F02_default_boundary_probability.png",
    "F03": "F03_q_caliber_surface_region.png",
    "F04": "F04_grid_bp_investment_boundary.png",
    "F05": "F05_bellman_residual_by_episode.png",
    "F06": "F06_p_q_policy_drift.png",
    "F07": "F07_fc1_pred_vs_actual.png",
    "F08": "F08_macro_dynamics_sdf.png",
    "F09": "F09_leverage_by_macro_state.png",
    "F10": "F10_bz_joint_distribution.png",
    "F11": "F11_capital_waterfall.png",
    "F12": "F12_convergence_dashboard.png",
    "V01": "V01_p0_surface.png",
    "V02": "V02_pi_surface.png",
    "V03": "V03_equity_p_surface.png",
    "V04": "V04_q_four_calibers.png",
    "V05": "V05_value_contour_default_boundary.png",
    "V06": "V06_value_drift_across_episodes.png",
    "P01": "P01_b0_star_surface.png",
    "P02": "P02_bi_star_surface.png",
    "P03": "P03_effective_debt_policy.png",
    "P04": "P04_bellman_objective_vs_bp.png",
    "M01": "M01_fc1_one_step_scatter.png",
    "M02": "M02_hatcf_vs_hatc_path.png",
    "M03": "M03_lnkf_vs_lnk_path.png",
    "M04": "M04_macro_recursive_error.png",
    "M05": "M05_sdf_distribution_constraints.png",
    "M06": "M06_macro_aggregate_time_series.png",
}

FIGURE_IDS = tuple(FIGURE_FILENAMES)

CAPTIONS = {
    "F01": (
        "F01 firm value and policy surfaces (P0/PI/P, 3D + contours). "
        "P0 is the equity value of the no-investment branch, PI that of the investment branch, "
        "and P the blended equity value after the investment choice. Q_claim is the debt claim "
        "value used for issuance pricing, and Grid bp0*/bpI* are the optimal next-period leverage "
        "of the two branches from the formal Grid-BP policy. Surfaces rise in productivity z and "
        "fall in leverage b, capturing debt overhang: at equal net asset value, higher leverage "
        "makes equity cheaper. Contours overlay the Phat=0 default boundary; below it lies the "
        "hard-default region."
    ),
    "F02": (
        "F02 default boundary and default probability (Gomes-style 2D). "
        "The left column shows the soft default probability bar_z=1-sigmoid(Phat/tau_z) together "
        "with the Phat=0 hard default boundary; bar_z is a soft probability, not a productivity "
        "threshold. The right column contrasts the hard boundary z_default(b), the soft boundary "
        "bar_z=0.5 and the conditional default probability. Economic meaning: the default boundary "
        "rises with leverage b (default becomes easier), and the hard-soft gap measures the caliber "
        "wedge between the soft penalty and the hard gate."
    ),
    "F03": (
        "F03 Q surfaces in four calibers and the default region (Q Claim / Q Unit / Recovery / Q Effective). "
        "Q_claim=b₊·q_unit is the only caliber allowed for issuance pricing and the Bellman residual. "
        "q_unit is the unit bond price, recovery=φ(1-δ+exp(x+z)) is the default recovery value, and "
        "Q_effective is the settlement caliber (only hybrid includes the hard default gate). Dark "
        "regions mark the hard-default area Phat≤0. Economic meaning: inside the default region the "
        "claim value collapses to recovery, while in the surviving region Q_claim first rises and then "
        "falls in b, reflecting the optimal capital structure."
    ),
    "F04": (
        "F04 Grid BP policy and investment boundary (3D + 2D). "
        "Grid bp0*/bpI* are the formal policies solved by the Grid-BP teacher on the candidate "
        "leverage grid (the PolicyValueModel bp head is compatibility output only), and the "
        "investment boundary i*(b,z) is the zero of PI-V0 along the investment cost axis. Economic "
        "meaning: higher productivity supports higher optimal leverage. Investment happens only "
        "below the investment boundary, and multi-root/no-root states are explicitly classified "
        "(always/never/single/multiple)."
    ),
    "F05": (
        "F05 Bellman residual convergence across episodes (line + P90). "
        "The frozen residual is the Bellman identity residual under the training-frozen policy, and "
        "the fresh residual is the optimality residual after re-solving greedily with the current "
        "model (optimality evidence uses fresh). P90/P99 are drawn together with MAE so the mean "
        "cannot mask the tail. Economic meaning: only residuals falling to the 1e-3 scale constitute "
        "a necessary condition for equilibrium convergence; training loss is not convergence evidence."
    ),
    "F06": (
        "F06 P / Q / policy drift across episodes (multi-panel lines). "
        "On the same fixed EvalBank states, consecutive episodes are compared row by row for P "
        "(equity value), Q (issuance pricing) and the Grid BP policy, plus policy/default/investment "
        "flip rates. Economic meaning: function drift vanishing is direct evidence of equilibrium "
        "convergence; flip rates measure the discrete switching share of the decision surface."
    ),
    "F07": (
        "F07 FC1 predicted vs realized (scatter + path). "
        "hatcf_t/lnkf_t are the t-1 belief forecasts of node t, hatc_t/lnk_t are the realized "
        "aggregates at node t, and same-row alignment gives the shift=0 one-step forecast. Economic "
        "meaning: forecast points hugging the 45-degree line mean the belief system is accurate; "
        "systematic slope/offset away from 45 degrees reflects forecast bias rather than noise."
    ),
    "F08": (
        "F08 macro dynamics and SDF (time series). "
        "On the Bank C macro anchors (Recession/Normal/Expansion): FC1 recursion paths, SDF "
        "conditional residuals decomposed by macro state, and the mean/standard error of M. Economic "
        "meaning: whether recursion paths drift out of the training range bounds the credibility of "
        "long simulations; SDF conditional moments should be stable across the three states, "
        "otherwise the asset-pricing kernel is inconsistent."
    ),
    "F09": (
        "F09 leverage distribution by macro state (Gomes-style histogram). "
        "The cross-section is split into Recession/Normal/Expansion by x quantiles, and leverage "
        "distributions are shown under firm-count weighting and capital weighting. Economic meaning: "
        "the recession distribution should shift left (deleveraging) and the capital-weighted one "
        "shift right (larger firms carry more leverage); the gap between the two calibers is the "
        "size-leverage correlation."
    ),
    "F10": (
        "F10 b-z joint distribution over time (heatmap + marginals). "
        "Normalized joint histograms on the fixed grid with their b/z marginals, under both count "
        "and capital weighting. Economic meaning: the evolving shape of the joint distribution "
        "reflects selection effects and endogenous exit; extinct paths are flagged explicitly and "
        "never dropped from the statistics."
    ),
    "F11": (
        "F11 t0→t1 capital waterfall and survival accounting. "
        "Per-period waterfall of the capital growth decomposition identity: ending capital = initial "
        "capital + ΔK of survivors + entrant capital (endpoint caliber) - exiting capital + "
        "accounting residual. Same-node entrants/exits are memo rows and stay outside the identity. "
        "Side-by-side period accounts of surviving/default/active paths are shown. Economic meaning: "
        "the residual should stay near zero, otherwise the capital ledger caliber has a gap."
    ),
    "F12": (
        "F12 general-equilibrium convergence dashboard. "
        "Trajectories across episodes of the Bellman residual, P/Q/policy drift, FC1 one-step error, "
        "SDF moments and distribution distances. Economic meaning: convergence in the "
        "general-equilibrium sense requires firm values/policies, macro beliefs and the pricing "
        "kernel to stabilize at the same time; any non-converging panel means the whole system has "
        "not converged."
    ),
    "V01": (
        "V01 P0 surface (no-investment branch equity value). "
        "Overlays the Phat=0 default boundary. The η=0 and η=1 counterfactual views are placed side "
        "by side on unified b/z ranges and colour scale for cross-episode comparison. Economic "
        "meaning: P0 is the liquidation-continuation value after forgoing the investment opportunity."
    ),
    "V02": (
        "V02 PI surface (investment branch equity value, labelled by investment cost i). "
        "Evaluated at the fixed reference cost i_mid (the median tier of the 10/50/90 quantiles). "
        "Economic meaning: the sign of PI-P0 decides whether to invest, and a higher i makes the "
        "investment branch less attractive."
    ),
    "V03": (
        "V03 equity value P and the Phat≤0 default region. "
        "P is the blended equity value after the investment choice, and Phat≤0 is the hard-default "
        "indicator (model caliber), strictly distinct from the soft probability bar_z. Economic "
        "meaning: inside the default region equity value collapses to zero, and the value gradient "
        "is steepest near the boundary."
    ),
    "V04": (
        "V04 four Q calibers compared (Q Claim / Q Unit / Q Effective / Recovery). "
        "Issuance and residuals always use the claim caliber; effective and recovery carry "
        "settlement semantics only. Economic meaning: the separation of the four calibers in the "
        "default region measures the gap between the hard default gate and the soft penalty."
    ),
    "V05": (
        "V05 value contours + default boundary. "
        "Contours of P0/PI/P overlaid with the hard Phat=0 boundary and the soft bar_z=0.5 "
        "boundary. Economic meaning: the kink of the contours at the default boundary shows debt "
        "overhang depressing marginal value."
    ),
    "V06": (
        "V06 value drift across episodes (value drift). "
        "Pointwise differences and summary statistics of the P0/PI/P surfaces between consecutive "
        "episodes. Economic meaning: drift vanishing shows the value function has stabilized on the "
        "fixed bank states, a necessary condition for convergence."
    ),
    "P01": (
        "P01 Grid bp0* surface (optimal next-period leverage of the no-investment branch). "
        "The greedy solution of the Grid-BP teacher, not the bp0 head output. Economic meaning: "
        "optimal leverage rises with productivity and falls with current leverage (mean reversion)."
    ),
    "P02": (
        "P02 Grid bpI* surface (optimal next-period leverage of the investment branch). "
        "The investment branch needs extra financing, so its optimal leverage is systematically "
        "above bp0*."
    ),
    "P03": (
        "P03 effective debt policy (realized b_next). "
        "b_next = η·bp* + (1-η)·b: η=0 forces b_next=b (refinancing off), η=1 gives b_next=bp*. The "
        "figure also shows the gap between the head output and Grid BP. Economic meaning: the "
        "refinancing gate determines what the policy actually executes, and intended vs executed "
        "policy must be reported separately."
    ),
    "P04": (
        "P04 Bellman objective vs candidate bp (Q Claim overlaid). "
        "At representative states: the Bellman right-hand side across candidate leverage versus "
        "issuance Q_claim. Economic meaning: the peak of the objective curve is Grid bp*, and the "
        "η=0 row must be flat (financing gate enforced)."
    ),
    "M01": (
        "M01 FC1 one-step forecast scatter. "
        "hatcf_t vs hatc_t and lnkf_t vs lnk_t (shift=0 same-row alignment), with the 45-degree line "
        "and R². Economic meaning: unbiased, on-line forecasts are a necessary condition for a "
        "consistent macro belief system."
    ),
    "M02": (
        "M02 Hatcf vs Hatc paths. "
        "Time paths per path of the belief forecast of the aggregate and its realization. Economic "
        "meaning: beliefs should fluctuate around realizations rather than deviate systematically."
    ),
    "M03": (
        "M03 LnKF vs LnK paths. "
        "Time paths per path of capital belief and realization. Economic meaning: persistent "
        "capital-belief deviation accumulates into wrong financing/investment decisions."
    ),
    "M04": (
        "M04 macro recursive forecast error. "
        "production_one_step (macro state computed from the parent node at every call, the formal "
        "simulation rule) and autonomous_recursive (fully autoregressive) are reported separately, "
        "and the slope of the error over horizon is the drift speed. Economic meaning: the drift "
        "slope under the autonomous caliber sets the credibility bound of long simulations."
    ),
    "M05": (
        "M05 SDF distribution and constraints. "
        "Mean/variance of M, conditional residuals by macro state, the Signed AiO estimate (with "
        "standard errors and interpretation guardrails) and the moment-constraint violations "
        "g_mean_low/g_mean_high/g_var_high. Economic meaning: a negative sample mean of Signed AiO "
        "is not a negative population conditional moment; constraint violations are the numerically "
        "safe criterion."
    ),
    "M06": (
        "M06 C/K, K, C, Y, I over time. "
        "Period paths of macro aggregates (C/K recovered from Hatc=log(C/K)). Economic meaning: a "
        "macro steady state means these aggregates stop drifting; persistent trends mean general "
        "equilibrium has not converged."
    ),
}

# Canonical 3D surface set required on every 3D figure.
CANONICAL_3D_NAMES = ("P0", "PI", "P", "Q_claim", "Grid bp0*", "Grid bpI*")


# ---------------------------------------------------------------------------
# dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FigureStyle:
    """Unified paper-grade style plus the shared axis / colour scale."""

    dpi: int = 160
    font_size: float = 9.0
    title_size: float = 10.5
    label_size: float = 8.5
    legend_size: float = 7.5
    footer_size: float = 6.8
    cmap_surface: str = "viridis"
    cmap_residual: str = "coolwarm"
    cmap_density: str = "viridis"
    cmap_region: str = "Reds"
    contour_levels: int = 20
    elev: float = 28.0
    azim: float = -132.0
    b_range: Optional[Tuple[float, float]] = None
    z_range: Optional[Tuple[float, float]] = None
    value_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def limit_for(self, family, values):
        limits = self.value_limits.get(family)
        if limits is not None:
            return float(limits[0]), float(limits[1])
        array = _finite(values)
        if array.size == 0:
            return None, None
        return float(np.min(array)), float(np.max(array))


@dataclass
class FigureRecord:
    """One rendered (or deliberately skipped) figure."""

    figure_id: str
    path: Optional[str] = None
    caption: str = ""
    status: str = "skipped"
    missing_reason: Optional[str] = None
    n_samples: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "figure_id": self.figure_id,
            "path": self.path,
            "caption": self.caption,
            "status": self.status,
            "missing_reason": self.missing_reason,
            "n_samples": self.n_samples,
            "meta": dict(self.meta),
        }


@dataclass
class FiguresManifest:
    """Result of :func:`render_all_figures`."""

    figures: Dict[str, FigureRecord] = field(default_factory=dict)
    output_dir: Optional[str] = None
    style: Optional[FigureStyle] = None

    def __iter__(self):
        return iter(self.figures.values())

    def __len__(self) -> int:
        return len(self.figures)

    def __getitem__(self, figure_id: str) -> FigureRecord:
        return self.figures[figure_id]

    @property
    def ok(self):
        return {key: value for key, value in self.figures.items() if value.status == "ok"}

    @property
    def skipped(self):
        return {key: value for key, value in self.figures.items() if value.status != "ok"}

    def to_dict(self):
        return {
            "output_dir": self.output_dir,
            "figures": {key: value.to_dict() for key, value in self.figures.items()},
        }


@dataclass
class _Panel:
    """Declarative panel description consumed by :func:`_render_grid`."""

    kind: str
    title: str = ""
    family: str = "value"
    xlabel: str = ""
    ylabel: str = ""
    zlabel: str = ""
    values: Any = None
    options: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------


def _finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _n_finite(values) -> int:
    return int(_finite(values).size)


def _episode_key(value) -> Tuple[int, Any]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _as_episode_map(value) -> Dict[Any, Any]:
    """Normalize ``{episode: result}`` / ``[result]`` / single result inputs."""
    if value is None:
        return {}
    items: List[Tuple[Any, Any]] = []
    if isinstance(value, Mapping):
        items = list(value.items())
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            episode = None
            diagnostics = getattr(item, "diagnostics", None)
            if isinstance(diagnostics, Mapping):
                episode = diagnostics.get("episode")
            items.append((episode if episode is not None else index, item))
    else:
        diagnostics = getattr(value, "diagnostics", None)
        episode = diagnostics.get("episode") if isinstance(diagnostics, Mapping) else None
        items = [(episode if episode is not None else 0, value)]
    ordered = sorted(items, key=lambda kv: _episode_key(kv[0]))
    return {key: item for key, item in ordered}


def _result_tables(result):
    tables = getattr(result, "tables", None)
    return dict(tables) if isinstance(tables, Mapping) else {}


def _result_surfaces(result):
    surfaces = getattr(result, "surfaces", None)
    return dict(surfaces) if isinstance(surfaces, Mapping) else {}


def _result_metrics(result):
    metrics = getattr(result, "metrics", None)
    return dict(metrics) if isinstance(metrics, Mapping) else {}


def _pick_table(result, base: str) -> Optional[pd.DataFrame]:
    """``base`` first, then any ``base_<slug>`` (sim_eval variant suffixes)."""
    tables = _result_tables(result)
    frame = tables.get(base)
    if isinstance(frame, pd.DataFrame):
        return frame
    suffixes = sorted(key for key in tables if str(key).startswith(base + "_"))
    for key in suffixes:
        candidate = tables[key]
        if isinstance(candidate, pd.DataFrame) and not candidate.empty:
            return candidate
    return None


def _table_column(frame, name: str):
    if frame is None or name not in frame.columns:
        return None
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)


_BZ_HIST_KEY = re.compile(r"^bz_hist_t(\d+)_(count|capital)(?:_(.+))?$")


def _joint_hists(result):
    out = {}
    by_key = {}
    for key, array in _result_surfaces(result).items():
        match = _BZ_HIST_KEY.match(str(key))
        if match is None:
            continue
        slot = (int(match.group(1)), str(match.group(2)))
        by_key.setdefault(slot, {})[str(match.group(3) or "")] = np.asarray(array)
    for slot, variants in by_key.items():
        if "" in variants:
            out[slot] = variants[""]
        else:
            out[slot] = variants[sorted(variants)[0]]
    return out


def _cmap(name: str):
    try:
        return matplotlib.colormaps[name]
    except Exception:  # pragma: no cover - older matplotlib
        return plt.get_cmap(name)


# ---------------------------------------------------------------------------
# style resolution (unified b/z support and colour limits)
# ---------------------------------------------------------------------------

_FAMILY_LIMIT_KEYS = (
    ("value", ("P0", "PI", "P", "V0", "VI", "value_star", "bellman_rhs", "continuation")),
    ("q", ("Q_claim", "Q_effective", "Q", "q_unit", "recovery", "q_claim_issue")),
    ("bp", ("bp0", "bpI", "bp", "b_next", "bp_candidate", "i_star")),
    ("prob", ("bar_z", "bar_i", "survival_prob", "default", "share", "rate", "prob")),
    ("residual", ("R0", "RI", "RQ", "residual", "error", "drift", "diff", "gap")),
)


def _family_of(name: str) -> str:
    text = str(name)
    for family, tokens in _FAMILY_LIMIT_KEYS:
        for token in tokens:
            if token in text:
                return family
    return "value"


def _collect_ranges(results):
    b_values = []
    z_values = []
    for result in results:
        for name in ("investment_boundary", "default_boundary", "hard_soft_boundary"):
            frame = _pick_table(result, name)
            if frame is None:
                continue
            for column, sink in (("b", b_values), ("z", z_values)):
                values = _table_column(frame, column)
                if values is not None:
                    good = _finite(values)
                    if good.size:
                        sink.append(float(np.min(good)))
                        sink.append(float(np.max(good)))
    b_range = (min(b_values), max(b_values)) if b_values else None
    z_range = (min(z_values), max(z_values)) if z_values else None
    return b_range, z_range


def _collect_value_limits(results):
    collected = {}
    for result in results:
        for name, values in _result_surfaces(result).items():
            array = np.asarray(values, dtype=np.float64)
            if array.ndim != 2:
                continue
            good = _finite(array)
            if good.size == 0:
                continue
            bucket = collected.setdefault(_family_of(name), [])
            bucket.append(float(np.min(good)))
            bucket.append(float(np.max(good)))
    return {family: (min(values), max(values)) for family, values in collected.items()}


def resolve_style(style: Optional[FigureStyle], results) -> FigureStyle:
    """Fill the shared ``b``/``z`` support and colour limits on ``style``.

    Ranges are the global extent over the supplied fixed-bank results, so every
    surface of every episode is drawn on one identical coordinate box and one
    identical colour scale.
    """
    base = replace(style) if style is not None else FigureStyle()
    if base.b_range is None or base.z_range is None:
        b_range, z_range = _collect_ranges(results)
        if base.b_range is None:
            base.b_range = b_range
        if base.z_range is None:
            base.z_range = z_range
    if not base.value_limits:
        base.value_limits = _collect_value_limits(results)
    return base


def _style_meta(style: FigureStyle):
    return {
        "b_range": list(style.b_range) if style.b_range is not None else None,
        "z_range": list(style.z_range) if style.z_range is not None else None,
        "value_limits": {
            family: [float(low), float(high)]
            for family, (low, high) in sorted(style.value_limits.items())
        },
        "cmap_surface": style.cmap_surface,
        "contour_levels": int(style.contour_levels),
        "unified_scale": True,
    }


# ---------------------------------------------------------------------------
# annotations / footer
# ---------------------------------------------------------------------------


def _annotation_lines(annotations, **overrides):
    values = {
        "Episode": "Not provided",
        "Macro Anchor": "Not provided",
        "η": "Not provided",
        "i": "Not provided",
        "Data source": "Not provided",
        "n": "Not provided",
        "Capital weighted": "Not provided",
        "Extinct paths": "Not provided",
    }
    if isinstance(annotations, Mapping):
        values.update({str(key): value for key, value in annotations.items()})
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return [f"{key}={value}" for key, value in values.items()]


def _describe_macro_anchor(bank) -> str:
    if not isinstance(bank, Mapping):
        return "Not provided"
    candidates = [bank]
    for key in ("C", "c", "bank_c", "bank_C"):
        if isinstance(bank.get(key), Mapping):
            candidates.append(bank[key])
    for candidate in candidates:
        anchor = candidate.get("macro_anchor")
        if isinstance(anchor, Mapping) and anchor:
            labels = sorted(str(key) for key in anchor)
            first = next(iter(anchor.values()))
            quantile = first.get("quantile") if isinstance(first, Mapping) else None
            suffix = f" @x q={quantile}" if quantile is not None else ""
            return f"{'/'.join(labels)}{suffix}"
        metadata = candidate.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("macro_anchor_source"):
            return str(metadata["macro_anchor_source"])
    return "Not provided"


def _describe_bank_source(bank) -> str:
    if not isinstance(bank, Mapping):
        return "Not provided"
    bank_id = bank.get("bank_id")
    bank_kind = bank.get("bank_kind")
    if bank_id or bank_kind:
        return f"{bank_id or 'bank'} ({bank_kind or 'unknown'})"
    for key in ("A", "B", "C", "D", "E"):
        entry = bank.get(key)
        if isinstance(entry, Mapping) and entry.get("bank_id"):
            return f"{entry['bank_id']} ({entry.get('bank_kind', 'unknown')})"
    return "Not provided"


def _describe_extinct(sim_result) -> str:
    if isinstance(sim_result, Mapping):
        episodes = _as_episode_map(sim_result)
        if not episodes:
            return "Not provided"
        sim_result = list(episodes.values())[-1]
    status = _pick_table(sim_result, "path_status")
    if status is None or "extinct" not in status.columns:
        return "Not provided"
    flags = status["extinct"].astype(bool).to_numpy()
    return f"{int(flags.sum())}/{int(flags.size)}"


# ---------------------------------------------------------------------------
# drawing core
# ---------------------------------------------------------------------------


def _apply_style(style: FigureStyle) -> None:
    plt.rcParams.update(
        {
            "font.size": style.font_size,
            "axes.titlesize": style.title_size,
            "axes.labelsize": style.label_size,
            "legend.fontsize": style.legend_size,
            "xtick.labelsize": style.label_size,
            "ytick.labelsize": style.label_size,
            "figure.dpi": style.dpi,
            "savefig.dpi": style.dpi,
            "axes.grid": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans", "sans-serif"],
            "axes.unicode_minus": False,
        }
    )


def _save(fig, path: Path, style: FigureStyle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=style.dpi, metadata={"Software": "convergence-eval-figures"})
    plt.close(fig)


def _set_xy_limits(ax, style: FigureStyle) -> None:
    if style.b_range is not None:
        ax.set_xlim(float(style.b_range[0]), float(style.b_range[1]))
    if style.z_range is not None:
        ax.set_ylim(float(style.z_range[0]), float(style.z_range[1]))


def _draw_surface_3d(ax, panel: _Panel, style: FigureStyle) -> bool:
    b_values = np.asarray(panel.options["b_values"], dtype=np.float64)
    z_values = np.asarray(panel.options["z_values"], dtype=np.float64)
    values = np.asarray(panel.values, dtype=np.float64)
    if values.shape != (b_values.size, z_values.size):
        return False
    mesh_b, mesh_z = np.meshgrid(b_values, z_values, indexing="ij")
    low, high = style.limit_for(panel.family, values)
    norm = None
    if low is not None and high is not None and math.isfinite(low) and math.isfinite(high):
        if high <= low:
            high = low + 1e-9
        norm = matplotlib.colors.Normalize(vmin=low, vmax=high)
    ax.plot_surface(
        mesh_b,
        mesh_z,
        values,
        cmap=_cmap(style.cmap_residual if panel.family == "residual" else style.cmap_surface),
        norm=norm,
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
    )
    ax.set_xlabel(panel.xlabel or "b = leverage ratio (debt/capital)")
    ax.set_ylabel(panel.ylabel or "z = firm productivity shock")
    ax.set_zlabel(panel.zlabel or panel.title)
    ax.view_init(elev=style.elev, azim=style.azim)
    ax.set_title(panel.title)
    if style.b_range is not None:
        ax.set_xlim3d(float(style.b_range[0]), float(style.b_range[1]))
    if style.z_range is not None:
        ax.set_ylim3d(float(style.z_range[0]), float(style.z_range[1]))
    return True


def _overlay_lines(ax, panel: _Panel) -> None:
    overlays = panel.options.get("overlays") or []
    for item in overlays:
        x = np.asarray(item.get("x"), dtype=np.float64)
        y = np.asarray(item.get("y"), dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            continue
        ax.plot(
            x[mask],
            y[mask],
            color=item.get("color", "black"),
            linestyle=item.get("linestyle", "-"),
            linewidth=item.get("linewidth", 1.4),
            label=item.get("label"),
        )
    if overlays:
        ax.legend()


def _shade_regions(ax, panel: _Panel, style: FigureStyle) -> None:
    region = panel.options.get("region")
    if region is None:
        return
    mask = np.asarray(region, dtype=bool)
    if mask.shape != np.asarray(panel.values).shape:
        return
    if not bool(mask.any()):
        return
    ax.contourf(
        np.asarray(panel.options["b_values"], dtype=np.float64),
        np.asarray(panel.options["z_values"], dtype=np.float64),
        mask.astype(np.float64).T,
        levels=[0.5, 1.5],
        colors=["#b2182b"],
        alpha=0.32,
    )


def _draw_contour(ax, panel: _Panel, style: FigureStyle) -> bool:
    b_values = np.asarray(panel.options["b_values"], dtype=np.float64)
    z_values = np.asarray(panel.options["z_values"], dtype=np.float64)
    values = np.asarray(panel.values, dtype=np.float64)
    if values.shape != (b_values.size, z_values.size):
        return False
    low, high = style.limit_for(panel.family, values)
    if low is None or not math.isfinite(low) or not math.isfinite(high) or high <= low:
        good = _finite(values)
        if good.size == 0:
            return False
        low, high = float(np.min(good)), float(np.max(good))
        if high <= low:
            high = low + 1e-9
    levels = np.linspace(low, high, int(style.contour_levels) + 1)
    mesh = ax.contourf(
        b_values,
        z_values,
        values.T,
        levels=levels,
        cmap=_cmap(style.cmap_residual if panel.family == "residual" else style.cmap_surface),
    )
    _shade_regions(ax, panel, style)
    _overlay_lines(ax, panel)
    fig = ax.get_figure()
    if fig is not None:
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel(panel.xlabel or "b = leverage ratio (debt/capital)")
    ax.set_ylabel(panel.ylabel or "z = firm productivity shock")
    ax.set_title(panel.title)
    _set_xy_limits(ax, style)
    return True


def _draw_heatmap(ax, panel: _Panel, style: FigureStyle) -> bool:
    b_values = np.asarray(panel.options["b_values"], dtype=np.float64)
    z_values = np.asarray(panel.options["z_values"], dtype=np.float64)
    values = np.asarray(panel.values, dtype=np.float64)
    if values.shape != (b_values.size, z_values.size):
        return False
    low, high = style.limit_for(panel.family, values)
    mesh = ax.pcolormesh(
        b_values,
        z_values,
        values.T,
        shading="auto",
        cmap=_cmap(style.cmap_residual if panel.family == "residual" else style.cmap_surface),
        vmin=low,
        vmax=high,
    )
    _shade_regions(ax, panel, style)
    _overlay_lines(ax, panel)
    fig = ax.get_figure()
    if fig is not None:
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel(panel.xlabel or "b = leverage ratio (debt/capital)")
    ax.set_ylabel(panel.ylabel or "z = firm productivity shock")
    ax.set_title(panel.title)
    _set_xy_limits(ax, style)
    return True


def _draw_line(ax, panel: _Panel, style: FigureStyle) -> bool:
    drawn = False
    for series in panel.options.get("series", []) or []:
        x = np.asarray(series.get("x"), dtype=np.float64)
        y = np.asarray(series.get("y"), dtype=np.float64)
        if x.size == 0:
            continue
        ax.plot(
            x,
            y,
            marker=series.get("marker", "o"),
            markersize=series.get("markersize", 3.5),
            linewidth=series.get("linewidth", 1.3),
            linestyle=series.get("linestyle", "-"),
            color=series.get("color"),
            label=series.get("label"),
        )
        drawn = True
    if panel.options.get("logy") == "symlog":
        ax.set_yscale("symlog")
    elif panel.options.get("logy"):
        ax.set_yscale("log")
    ax.set_xlabel(panel.xlabel)
    ax.set_ylabel(panel.ylabel)
    ax.set_title(panel.title)
    ax.grid(alpha=0.25)
    if panel.options.get("series"):
        ax.legend()
    return drawn


def _draw_scatter(ax, panel: _Panel, style: FigureStyle) -> bool:
    drawn = False
    for series in panel.options.get("series", []) or []:
        x = np.asarray(series.get("x"), dtype=np.float64)
        y = np.asarray(series.get("y"), dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            continue
        ax.scatter(
            x[mask],
            y[mask],
            s=series.get("size", 8),
            alpha=series.get("alpha", 0.55),
            label=series.get("label"),
            color=series.get("color"),
            edgecolors="none",
        )
        drawn = True
    if panel.options.get("diag", True):
        all_values = []
        for series in panel.options.get("series", []) or []:
            all_values.append(np.asarray(series.get("x"), dtype=np.float64))
            all_values.append(np.asarray(series.get("y"), dtype=np.float64))
        joined = _finite(np.concatenate(all_values)) if all_values else np.empty(0)
        if joined.size:
            low, high = float(np.min(joined)), float(np.max(joined))
            ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel(panel.xlabel)
    ax.set_ylabel(panel.ylabel)
    ax.set_title(panel.title)
    ax.grid(alpha=0.25)
    if panel.options.get("series"):
        ax.legend()
    return drawn


def _draw_bar(ax, panel: _Panel, style: FigureStyle) -> bool:
    labels = list(panel.options.get("labels", []) or [])
    values = np.asarray(panel.options.get("values", []), dtype=np.float64)
    if labels and values.size:
        positions = np.arange(len(labels))
        ax.bar(
            positions,
            values,
            yerr=panel.options.get("yerr"),
            color=panel.options.get("color", "tab:blue"),
            alpha=0.85,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=panel.options.get("rotation", 0))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel(panel.xlabel)
        ax.set_ylabel(panel.ylabel)
        ax.set_title(panel.title)
        ax.grid(alpha=0.2, axis="y")
        return True
    return False


def _draw_text(ax, panel: _Panel, style: FigureStyle) -> bool:
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        str(panel.options.get("text", "")),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=style.legend_size,
    )
    ax.set_title(panel.title)
    return True


def _draw_hist(ax, panel: _Panel, style: FigureStyle) -> bool:
    drawn = False
    for series in panel.options.get("series", []) or []:
        centers = np.asarray(series.get("x"), dtype=np.float64)
        values = np.asarray(series.get("y"), dtype=np.float64)
        if centers.size == 0:
            continue
        ax.plot(
            centers,
            values,
            drawstyle=series.get("drawstyle", "steps-mid"),
            linewidth=series.get("linewidth", 1.3),
            label=series.get("label"),
            color=series.get("color"),
        )
        if panel.options.get("fill", False):
            ax.fill_between(centers, values, step="mid", alpha=0.18)
        drawn = True
    ax.set_xlabel(panel.xlabel)
    ax.set_ylabel(panel.ylabel)
    ax.set_title(panel.title)
    ax.grid(alpha=0.25)
    if panel.options.get("series"):
        ax.legend()
    return drawn


_DRAW = {
    "3d": _draw_surface_3d,
    "contour": _draw_contour,
    "heatmap": _draw_heatmap,
    "line": _draw_line,
    "scatter": _draw_scatter,
    "bar": _draw_bar,
    "text": _draw_text,
    "hist": _draw_hist,
}


def _render_grid(
    figure_id: str,
    panels: Sequence[_Panel],
    *,
    rows: int,
    cols: int,
    output_dir: Path,
    style: FigureStyle,
    caption: str,
    meta: Dict[str, Any],
    n_samples,
    annotations=None,
    figsize=None,
) -> FigureRecord:
    """Draw ``rows x cols`` panels; skip the figure when nothing is drawable."""
    filename = FIGURE_FILENAMES[figure_id]
    output_dir = Path(output_dir)
    full_meta = dict(meta)
    full_meta.update(_style_meta(style))
    full_meta["panels"] = [panel.title for panel in panels]
    full_meta["dropped_panels"] = [panel.title for panel in panels[rows * cols :]]
    full_meta["annotations"] = _annotation_lines(annotations)

    drawable = [
        panel
        for panel in panels
        if panel.values is not None or panel.kind in ("text", "line", "scatter", "bar", "hist")
    ]
    if not drawable:
        return FigureRecord(
            figure_id=figure_id,
            caption=caption,
            status="skipped",
            missing_reason=str(meta.get("missing_reason") or REASON_NO_FINITE_SAMPLES),
            n_samples=n_samples,
            meta=full_meta,
        )

    _apply_style(style)
    width = float(figsize[0]) if figsize else 5.0 * cols
    height = float(figsize[1]) if figsize else 4.0 * rows
    fig = plt.figure(figsize=(width, height))
    drawn_any = False
    used = []
    for index, panel in enumerate(panels[: rows * cols]):
        projection = "3d" if panel.kind == "3d" else None
        ax = fig.add_subplot(rows, cols, index + 1, projection=projection)
        try:
            drawn = bool(_DRAW[panel.kind](ax, panel, style))
        except Exception as exc:  # a malformed panel must not fake a success figure
            drawn = False
            panel.title = f"{panel.title} [render failed: {type(exc).__name__}: {str(exc)[:160]}]"
        drawn_any = drawn_any or drawn
        used.append({"title": panel.title, "drawn": bool(drawn)})
    full_meta["panel_status"] = used
    full_meta["panel_errors"] = [
        item["title"] for item in used if not item["drawn"]
    ]

    if not drawn_any:
        plt.close(fig)
        return FigureRecord(
            figure_id=figure_id,
            caption=caption,
            status="skipped",
            missing_reason=str(meta.get("missing_reason") or REASON_NO_FINITE_SAMPLES),
            n_samples=n_samples,
            meta=full_meta,
        )

    first_sentence = re.split(r"(?:。|\.\s)", caption, maxsplit=1)[0]
    fig.suptitle(f"{figure_id}: {first_sentence}", fontsize=style.title_size + 1.5)
    fig.subplots_adjust(
        left=0.07, right=0.955, top=0.90, bottom=0.115, hspace=0.55, wspace=0.32
    )
    fig.text(
        0.005,
        0.005,
        " | ".join(_annotation_lines(annotations)),
        ha="left",
        va="bottom",
        fontsize=style.footer_size,
    )
    path = output_dir / filename
    _save(fig, path, style)
    return FigureRecord(
        figure_id=figure_id,
        path=str(path),
        caption=caption,
        status="ok",
        missing_reason=None,
        n_samples=n_samples,
        meta=full_meta,
    )


def _skip(figure_id: str, reason: str, meta=None) -> FigureRecord:
    return FigureRecord(
        figure_id=figure_id,
        path=None,
        caption=CAPTIONS.get(figure_id, ""),
        status="skipped",
        missing_reason=reason,
        n_samples=None,
        meta=dict(meta or {}),
    )


# ---------------------------------------------------------------------------
# firm-surface panel builders
# ---------------------------------------------------------------------------

_GRID_SOURCE_TABLES = ("investment_boundary", "default_boundary", "hard_soft_boundary")


def grid_axes(result):
    """Recover the ``(b_values, z_values)`` surface axes of a policy result.

    The surfaces themselves are ``(len(b), len(z))`` arrays; the axes come from
    the boundary tables, which carry the grid coordinates explicitly.
    """
    for name in _GRID_SOURCE_TABLES:
        frame = _pick_table(result, name)
        if frame is None:
            continue
        if {"b", "z"}.issubset(frame.columns):
            b_values = np.unique(_finite(_table_column(frame, "b")))
            z_values = np.unique(_finite(_table_column(frame, "z")))
            if b_values.size >= 2 and z_values.size >= 2:
                return b_values, z_values
        if {"b", "z_default"}.issubset(frame.columns):
            b_values = np.unique(_finite(_table_column(frame, "b")))
            z_values = np.unique(_finite(_table_column(frame, "z_default")))
            if b_values.size >= 2 and z_values.size >= 2:
                return b_values, z_values
    return None


# Key aliases for evaluator surfaces that are only exposed under a slice
# qualifier: ``evaluation.firm_surfaces.evaluate_firm_surfaces`` rolls most
# fields up under their plain name (the ``i_mid`` slice) but keeps ``PI`` only
# per i-slice, so ``{prefix}PI`` lookups fall back to the ``i_mid`` slice.
_SURFACE_KEY_ALIASES = {
    "PI": ("PI_mid",),
    "eta0_PI": ("eta0_PI_mid",),
    "eta1_PI": ("eta1_PI_mid",),
}


def _surface(result, name: str):
    surfaces = _result_surfaces(result)
    for key in (name, *_SURFACE_KEY_ALIASES.get(name, ())):
        values = surfaces.get(key)
        if values is None:
            continue
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 2:
            return array
    return None


def _canonical_surfaces(result):
    """The canonical 3D set; ``None`` + reason when any member is missing."""
    mapping = {
        "P0": "eta1_P0",
        "PI": "eta1_PI",
        "P": "eta1_P",
        "Q_claim": "eta1_Q_claim",
        "Grid bp0*": "eta1_bp0",
        "Grid bpI*": "eta1_bpI",
    }
    out = {}
    for label, key in mapping.items():
        values = _surface(result, key)
        if values is None:
            return None, REASON_NO_POLICY_SURFACES
        out[label] = values
    return out, None


def _boundary_overlay(result, style: FigureStyle):
    overlays = []
    frame = _pick_table(result, "default_boundary")
    if frame is not None and {"b", "z_default"}.issubset(frame.columns):
        b = _table_column(frame, "b")
        z = _table_column(frame, "z_default")
        if b is not None and z is not None:
            overlays.append({"x": b, "y": z, "label": "default boundary Phat=0", "color": "black"})
    hard = _pick_table(result, "hard_soft_boundary")
    if hard is not None and {"b", "z_soft_barz0p5"}.issubset(hard.columns):
        b = _table_column(hard, "b")
        z = _table_column(hard, "z_soft_barz0p5")
        if b is not None and z is not None:
            overlays.append(
                {
                    "x": b,
                    "y": z,
                    "label": "soft boundary bar_z=0.5",
                    "color": "tab:red",
                    "linestyle": "--",
                }
            )
    return overlays


def _or_nan(values, shape):
    return values if values is not None else np.full(shape, np.nan)


def _default_region(result, grid_shape):
    region = _surface(result, "eta1_default_region")
    if region is not None and region.shape == grid_shape:
        return region.astype(bool)
    phat = _surface(result, "eta1_Phat")
    if phat is not None and phat.shape == grid_shape:
        return phat <= 0.0
    return None


def _eta_label(view: str) -> str:
    return "η=1 (refi active, b_next=bp*)" if view == "eta1" else "η=0 (refi off, b_next=b)"


def _base_meta(
    *,
    episode=None,
    eta=None,
    i_slice=None,
    data_source=None,
    capital_weighted=None,
    extinct_paths=None,
    n_samples=None,
    extra=None,
):
    meta = {
        "episode": episode,
        "eta_view": eta,
        "i_slice": i_slice,
        "data_source": data_source,
        "capital_weighted": capital_weighted,
        "extinct_paths": extinct_paths,
        "n_samples": n_samples,
    }
    if extra:
        meta.update(extra)
    return meta


def _annotation_dict(
    annotations,
    *,
    episode=None,
    eta=None,
    i_slice=None,
    data_source=None,
    capital_weighted=None,
    extinct_paths=None,
    n_samples=None,
):
    merged = dict(annotations or {})
    if episode is not None:
        merged["Episode"] = episode
    if eta is not None:
        merged["η"] = eta
    if i_slice is not None:
        merged["i"] = i_slice
    if data_source is not None:
        merged["Data source"] = data_source
    if capital_weighted is not None:
        merged["Capital weighted"] = capital_weighted
    if extinct_paths is not None:
        merged["Extinct paths"] = extinct_paths
    if n_samples is not None:
        merged["n"] = n_samples
    return merged


def _select_episode(policy_results, episode):
    episodes = _as_episode_map(policy_results)
    if not episodes:
        return None, None, REASON_NO_POLICY_SURFACES
    if episode is None:
        key = list(episodes)[-1]
    elif episode in episodes:
        key = episode
    else:
        target = _episode_key(episode)
        match = [key for key in episodes if _episode_key(key) == target]
        if not match:
            return None, None, REASON_NO_POLICY_SURFACES
        key = match[0]
    return key, episodes[key], None


def _i_slice_text(policy_result) -> str:
    diagnostics = getattr(policy_result, "diagnostics", None) or {}
    reference = diagnostics.get("reference_i") if isinstance(diagnostics, Mapping) else None
    if reference is not None:
        return f"i_mid={reference}"
    return "i=i_mid (bank i median)"


# ---------------------------------------------------------------------------
# F-series renderers
# ---------------------------------------------------------------------------


def render_f01(
    *,
    policy_results,
    output_dir,
    style=None,
    episode=None,
    annotations=None,
) -> FigureRecord:
    """F01: P0 / PI / P surfaces (3D) + contours, plus the canonical 3D set."""
    episode_key, result, reason = _select_episode(policy_results, episode)
    if result is None:
        return _skip("F01", reason or REASON_NO_POLICY_SURFACES)
    axes_pair = grid_axes(result)
    if axes_pair is None:
        return _skip("F01", REASON_NO_GRID_AXES)
    b_values, z_values = axes_pair
    canonical, reason = _canonical_surfaces(result)
    if canonical is None:
        return _skip("F01", reason or REASON_NO_POLICY_SURFACES)

    style = resolve_style(style, [result])
    panels = []
    order = ("P0", "PI", "P", "Q_claim", "Grid bp0*", "Grid bpI*")
    for name in order:
        panels.append(
            _Panel(
                kind="3d",
                title=f"3D {name}",
                family=_family_of(name),
                values=canonical[name],
                options={"b_values": b_values, "z_values": z_values},
            )
        )
    region = _default_region(result, canonical["P0"].shape)
    overlays = _boundary_overlay(result, style)
    for name in ("P0", "PI", "P"):
        panels.append(
            _Panel(
                kind="contour",
                title=f"contour {name} + default boundary",
                family=_family_of(name),
                values=canonical[name],
                options={
                    "b_values": b_values,
                    "z_values": z_values,
                    "region": region,
                    "overlays": overlays,
                },
            )
        )

    n_samples = int(sum(_n_finite(canonical[name]) for name in order))
    meta = _base_meta(
        episode=episode_key,
        eta="η=1",
        i_slice=_i_slice_text(result),
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="no (surfaces evaluated on the fixed bank grid)",
        n_samples=n_samples,
        extra={
            "canonical_3d": list(CANONICAL_3D_NAMES),
            "grid_shape": list(canonical["P0"].shape),
        },
    )
    return _render_grid(
        "F01",
        panels,
        rows=3,
        cols=3,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS["F01"],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations,
            episode=episode_key,
            eta="η=1",
            i_slice=_i_slice_text(result),
            data_source=str(meta["data_source"]),
            capital_weighted="no",
            n_samples=n_samples,
        ),
    )


def render_f02(
    *,
    policy_results,
    output_dir,
    style=None,
    episode=None,
    annotations=None,
) -> FigureRecord:
    """F02: default boundary / default probability (Gomes-style 2D)."""
    episode_key, result, reason = _select_episode(policy_results, episode)
    if result is None:
        return _skip("F02", reason or REASON_NO_POLICY_SURFACES)
    axes_pair = grid_axes(result)
    if axes_pair is None:
        return _skip("F02", REASON_NO_GRID_AXES)
    b_values, z_values = axes_pair
    style = resolve_style(style, [result])

    bar_z = _surface(result, "eta1_bar_z")
    phat = _surface(result, "eta1_Phat")
    default_table = _pick_table(result, "default_boundary")
    hard_soft = _pick_table(result, "hard_soft_boundary")
    if bar_z is None and phat is None and default_table is None:
        return _skip("F02", REASON_NO_DEFAULT_BOUNDARY)

    shape = (b_values.size, z_values.size)
    region = _default_region(result, shape)
    overlays = _boundary_overlay(result, style)
    panels = []
    if bar_z is not None:
        panels.append(
            _Panel(
                kind="heatmap",
                title="bar_z = 1-sigmoid(Phat/tau_z) (soft default probability)",
                family="prob",
                values=bar_z,
                options={
                    "b_values": b_values,
                    "z_values": z_values,
                    "region": region,
                    "overlays": overlays,
                },
            )
        )
    if phat is not None:
        panels.append(
            _Panel(
                kind="heatmap",
                title="Phat(b,z) (equity value discriminant; ≤0 = hard default)",
                family="value",
                values=phat,
                options={
                    "b_values": b_values,
                    "z_values": z_values,
                    "region": region,
                    "overlays": overlays,
                },
            )
        )

    boundary_series = []
    if default_table is not None and {"b", "z_default"}.issubset(default_table.columns):
        boundary_series.append(
            {
                "x": _table_column(default_table, "b"),
                "y": _table_column(default_table, "z_default"),
                "label": "Phat=0 (hard)",
                "color": "black",
            }
        )
    if hard_soft is not None and {"b", "z_soft_barz0p5"}.issubset(hard_soft.columns):
        boundary_series.append(
            {
                "x": _table_column(hard_soft, "b"),
                "y": _table_column(hard_soft, "z_soft_barz0p5"),
                "label": "bar_z=0.5 (soft)",
                "color": "tab:red",
                "linestyle": "--",
            }
        )
    panels.append(
        _Panel(
            kind="line",
            title="default boundary z_default(b): hard vs soft",
            xlabel="b = leverage ratio (debt/capital)",
            ylabel="z = firm productivity shock (default threshold)",
            options={"series": boundary_series},
        )
    )

    rows = _pick_table(result, "policy_rows")
    if rows is not None and {"b", "conditional_default_prob"}.issubset(rows.columns):
        panels.append(
            _Panel(
                kind="scatter",
                title="conditional default probability (z transitions, not bar_z)",
                xlabel="b = leverage ratio",
                ylabel="Pr(Phat_child ≤ 0 | b_next)",
                options={
                    "series": [
                        {
                            "x": _table_column(rows, "b"),
                            "y": _table_column(rows, "conditional_default_prob"),
                            "label": "conditional default probability",
                        }
                    ],
                    "diag": False,
                },
            )
        )
    else:
        panels.append(
            _Panel(
                kind="text",
                title="conditional default probability",
                options={
                    "text": "policy_rows has no conditional_default_prob column\n(panel omitted; no blank figure is fabricated)"
                },
            )
        )

    status_counts = ""
    if default_table is not None and "boundary_status" in default_table.columns:
        counts = default_table["boundary_status"].value_counts().to_dict()
        status_counts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    n_samples = int(shape[0] * shape[1])
    meta = _base_meta(
        episode=episode_key,
        eta="η=1",
        i_slice=_i_slice_text(result),
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="no",
        n_samples=n_samples,
        extra={"boundary_status_counts": status_counts},
    )
    return _render_grid(
        "F02",
        panels,
        rows=2,
        cols=2,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS["F02"],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations,
            episode=episode_key,
            eta="η=1",
            i_slice=_i_slice_text(result),
            data_source=str(meta["data_source"]),
            capital_weighted="no",
            n_samples=n_samples,
        ),
    )


def render_f03(
    *,
    policy_results,
    output_dir,
    style=None,
    episode=None,
    annotations=None,
) -> FigureRecord:
    """F03: Q Claim / Q Unit / Recovery / Q Effective (3D + default regions)."""
    episode_key, result, reason = _select_episode(policy_results, episode)
    if result is None:
        return _skip("F03", reason or REASON_NO_POLICY_SURFACES)
    axes_pair = grid_axes(result)
    if axes_pair is None:
        return _skip("F03", REASON_NO_GRID_AXES)
    b_values, z_values = axes_pair
    canonical, reason = _canonical_surfaces(result)
    if canonical is None:
        return _skip("F03", reason or REASON_NO_POLICY_SURFACES)
    q_claim = _surface(result, "eta1_Q_claim")
    q_unit = _surface(result, "eta1_q_unit")
    recovery = _surface(result, "eta1_recovery")
    if q_claim is None or q_unit is None or recovery is None:
        return _skip("F03", REASON_NO_Q_SURFACES)
    q_effective = _surface(result, "eta1_Q_effective")

    style = resolve_style(style, [result])
    shape = canonical["P0"].shape
    region = _default_region(result, shape)

    panels = [
        _Panel(kind="3d", title="3D Q_claim (issuance pricing caliber)", family="q", values=q_claim,
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D q_unit (unit bond price)", family="q", values=q_unit,
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D Recovery (default recovery value)", family="q", values=recovery,
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D Q Effective (settlement caliber)", family="q",
               values=q_effective if q_effective is not None else np.full(shape, np.nan),
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D P0 (canonical)", family="value", values=canonical["P0"],
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D PI (canonical)", family="value", values=canonical["PI"],
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D P (canonical)", family="value", values=canonical["P"],
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D Grid bp0* (canonical)", family="bp", values=canonical["Grid bp0*"],
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D Grid bpI* (canonical)", family="bp", values=canonical["Grid bpI*"],
               options={"b_values": b_values, "z_values": z_values}),
    ]
    for name, values in (("Q Claim", q_claim), ("q_unit", q_unit), ("Recovery", recovery)):
        panels.append(
            _Panel(
                kind="contour",
                title=f"{name} + Phat≤0 default region",
                family="q",
                values=values,
                options={"b_values": b_values, "z_values": z_values, "region": region,
                         "overlays": _boundary_overlay(result, style)},
            )
        )

    n_samples = int(sum(_n_finite(item.values) for item in panels if item.values is not None))
    meta = _base_meta(
        episode=episode_key,
        eta="η=1",
        i_slice=_i_slice_text(result),
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="no",
        n_samples=n_samples,
        extra={"q_calibers": ["Q_claim", "q_unit", "Q_effective", "recovery"],
               "canonical_3d": list(CANONICAL_3D_NAMES)},
    )
    return _render_grid(
        "F03",
        panels,
        rows=4,
        cols=3,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS["F03"],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=episode_key, eta="η=1", i_slice=_i_slice_text(result),
            data_source=str(meta["data_source"]), capital_weighted="no", n_samples=n_samples,
        ),
    )


def render_f04(
    *,
    policy_results,
    output_dir,
    style=None,
    episode=None,
    annotations=None,
) -> FigureRecord:
    """F04: Grid BP policy and investment boundary (3D + 2D)."""
    episode_key, result, reason = _select_episode(policy_results, episode)
    if result is None:
        return _skip("F04", reason or REASON_NO_POLICY_SURFACES)
    axes_pair = grid_axes(result)
    if axes_pair is None:
        return _skip("F04", REASON_NO_GRID_AXES)
    b_values, z_values = axes_pair
    canonical, reason = _canonical_surfaces(result)
    if canonical is None:
        return _skip("F04", reason or REASON_NO_POLICY_SURFACES)
    i_star = _surface(result, "eta1_investment_i_star")
    investment_table = _pick_table(result, "investment_boundary")
    if i_star is None and investment_table is None:
        return _skip("F04", REASON_NO_INVESTMENT_BOUNDARY)

    style = resolve_style(style, [result])
    shape = canonical["P0"].shape
    region = _default_region(result, shape)

    panels = [
        _Panel(kind="3d", title="3D Grid bp0* (no-investment branch)", family="bp", values=canonical["Grid bp0*"],
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D Grid bpI* (investment branch)", family="bp", values=canonical["Grid bpI*"],
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D investment boundary i*(b,z)", family="bp",
               values=i_star if i_star is not None else np.full(shape, np.nan),
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D P0 (canonical)", family="value", values=canonical["P0"],
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D PI (canonical)", family="value", values=canonical["PI"],
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D P (canonical)", family="value", values=canonical["P"],
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="3d", title="3D Q_claim (canonical)", family="q", values=canonical["Q_claim"],
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="contour", title="Grid bp0* + default boundary", family="bp", values=canonical["Grid bp0*"],
               options={"b_values": b_values, "z_values": z_values, "region": region,
                        "overlays": _boundary_overlay(result, style)}),
        _Panel(kind="contour", title="Grid bpI* + default boundary", family="bp", values=canonical["Grid bpI*"],
               options={"b_values": b_values, "z_values": z_values, "region": region,
                        "overlays": _boundary_overlay(result, style)}),
        _Panel(kind="contour", title="investment boundary i*(b,z) (PI-V0 zero)", family="bp",
               values=i_star if i_star is not None else np.full(shape, np.nan),
               options={"b_values": b_values, "z_values": z_values, "region": region,
                        "overlays": _boundary_overlay(result, style)}),
        _Panel(kind="heatmap", title="Bar_i_model (investment intent)", family="prob",
               values=_or_nan(_surface(result, "eta1_bar_i_cond"), shape),
               options={"b_values": b_values, "z_values": z_values, "region": region}),
        _Panel(kind="heatmap", title="Bar_i_executed (executed investment)", family="prob",
               values=_or_nan(_surface(result, "eta1_bar_i_eff"), shape),
               options={"b_values": b_values, "z_values": z_values, "region": region}),
        _Panel(kind="contour", title="Grid bpI* - bp0* (investment-branch leverage premium)", family="residual",
               values=_or_nan(canonical["Grid bpI*"] - canonical["Grid bp0*"], shape),
               options={"b_values": b_values, "z_values": z_values, "region": region,
                        "overlays": _boundary_overlay(result, style)}),
        _Panel(kind="heatmap", title="survival indicator (Phat>0)", family="prob",
               values=_or_nan((1.0 - region.astype(np.float64)) if region is not None else None, shape),
               options={"b_values": b_values, "z_values": z_values,
                        "overlays": _boundary_overlay(result, style)}),
    ]

    if investment_table is not None and {"b", "z", "investment_status_label"}.issubset(
        investment_table.columns
    ):
        labels = sorted(set(investment_table["investment_status_label"].astype(str)))
        encoded = (
            investment_table["investment_status_label"]
            .astype(str)
            .map({label: index for index, label in enumerate(labels)})
            .to_numpy(dtype=np.float64)
        )
        pivoted = np.full(shape, np.nan)
        b_column = _table_column(investment_table, "b")
        z_column = _table_column(investment_table, "z")
        if b_column is not None and z_column is not None:
            b_lookup = {float(value): index for index, value in enumerate(b_values)}
            z_lookup = {float(value): index for index, value in enumerate(z_values)}
            for row in range(len(encoded)):
                i_pos = b_lookup.get(float(b_column[row]))
                j_pos = z_lookup.get(float(z_column[row]))
                if i_pos is not None and j_pos is not None:
                    pivoted[i_pos, j_pos] = encoded[row]
        panels.append(
            _Panel(
                kind="heatmap",
                title="investment boundary classification (0=" + ",1=".join(labels) + ")",
                family="bp",
                values=pivoted,
                options={"b_values": b_values, "z_values": z_values, "region": region},
            )
        )

    n_samples = int(sum(_n_finite(item.values) for item in panels if item.values is not None))
    status_counts = {}
    if investment_table is not None and "investment_status_label" in investment_table.columns:
        status_counts = (
            investment_table["investment_status_label"].astype(str).value_counts().to_dict()
        )
    meta = _base_meta(
        episode=episode_key,
        eta="η=1",
        i_slice=_i_slice_text(result),
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="no",
        n_samples=n_samples,
        extra={"canonical_3d": list(CANONICAL_3D_NAMES),
               "investment_status_counts": status_counts},
    )
    return _render_grid(
        "F04",
        panels,
        rows=5,
        cols=3,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS["F04"],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=episode_key, eta="η=1", i_slice=_i_slice_text(result),
            data_source=str(meta["data_source"]), capital_weighted="no", n_samples=n_samples,
        ),
    )


def render_f05(
    *,
    bellman_results,
    output_dir,
    style=None,
    annotations=None,
) -> FigureRecord:
    """F05: Bellman residual vs episode (line + P90)."""
    episodes = _as_episode_map(bellman_results)
    if not episodes:
        return _skip("F05", REASON_NO_BELLMAN_METRICS)
    x = np.arange(len(episodes), dtype=np.float64)
    keys = list(episodes)
    styles = (
        ("fresh_p0_normalized", "fresh P0 residual (normalized)"),
        ("fresh_pi_normalized", "fresh PI residual (normalized)"),
        ("frozen_p0_normalized", "frozen P0 residual (normalized)"),
        ("frozen_pi_normalized", "frozen PI residual (normalized)"),
        ("fresh_p0_physical", "fresh P0 residual (physical)"),
        ("fresh_pi_physical", "fresh PI residual (physical)"),
    )
    panels = []
    total = 0
    for prefix, label in styles:
        series = []
        for suffix, style_name, line in (
            ("mae", "MAE", "-"),
            ("p90", "P90", "--"),
            ("p99", "P99", ":"),
        ):
            values = []
            for key in keys:
                metrics = _result_metrics(episodes[key])
                values.append(float(metrics.get(f"{prefix}_{suffix}", float("nan"))))
            array = np.asarray(values, dtype=np.float64)
            if np.isfinite(array).any():
                total += int(np.isfinite(array).sum())
                series.append({"x": x, "y": array, "label": style_name, "linestyle": line})
        if series:
            panels.append(
                _Panel(
                    kind="line",
                    title=label,
                    xlabel="Episode (training rounds)",
                    ylabel="residual |R| (normalized=R/(1+exp(clamp(x+z))))",
                    options={"series": series, "logy": "symlog"},
                )
            )
    if not panels:
        return _skip("F05", REASON_NO_BELLMAN_METRICS)
    while len(panels) < 6:
        panels.append(_Panel(kind="text", title="(no data)", options={"text": "no residual statistics reported for this caliber"}))

    counts = []
    for key in keys:
        metrics = _result_metrics(episodes[key])
        counts.append(int(metrics.get("fresh_p0_normalized_n_finite", 0.0)))
    n_samples = int(sum(counts)) if sum(counts) else total
    meta = _base_meta(
        episode=f"{keys[0]}..{keys[-1]}",
        eta="η∈{0,1} mixed (full residual_stat_block)",
        i_slice="all bank states",
        data_source=_describe_bank_source(getattr(episodes[keys[-1]], "diagnostics", None)),
        capital_weighted="no",
        n_samples=n_samples,
        extra={
            "episodes": [key for key in keys],
            "residual_definition": (
                "frozen=Bellman identity residual under the training-frozen policy; "
                "fresh=optimality residual after re-solving greedily with the current model"
            ),
        },
    )
    return _render_grid(
        "F05",
        panels,
        rows=2,
        cols=3,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS["F05"],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations,
            episode=meta["episode"],
            eta="η∈{0,1}",
            i_slice="all bank states",
            data_source=str(meta["data_source"]),
            capital_weighted="no",
            n_samples=n_samples,
        ),
    )


def _aligned_policy_rows(results):
    frames = {}
    length = None
    index = None
    for key, result in results.items():
        frame = _pick_table(result, "policy_rows")
        if frame is None or frame.empty:
            return None
        frames[key] = frame
        if length is None:
            length = len(frame)
            index = frame["row_index"].to_numpy() if "row_index" in frame.columns else None
        else:
            if len(frame) != length:
                return None
            if index is not None and "row_index" in frame.columns:
                if not np.array_equal(index, frame["row_index"].to_numpy()):
                    return None
    return frames if frames else None


def render_f06(
    *,
    policy_results,
    output_dir,
    style=None,
    comparison=None,
    annotations=None,
) -> FigureRecord:
    """F06: P / Q / policy drift across episodes (multi-panel line)."""
    episodes = _as_episode_map(policy_results)
    if len(episodes) < 2:
        return _skip("F06", REASON_INSUFFICIENT_EPISODES)
    frames = _aligned_policy_rows(episodes)
    if frames is None:
        return _skip("F06", REASON_POLICY_ROWS_NOT_ALIGNED)
    keys = list(frames)
    x = np.arange(1, len(keys), dtype=np.float64)

    def _drift(column):
        values = []
        for previous, current in zip(keys[:-1], keys[1:]):
            left = _table_column(frames[previous], column)
            right = _table_column(frames[current], column)
            if left is None or right is None:
                values.append(np.nan)
            else:
                values.append(float(np.nanmean(np.abs(right - left))))
        array = np.asarray(values, dtype=np.float64)
        return array if np.isfinite(array).any() else None

    panels = []
    drift_series = {}
    for column in ("P", "P0", "PI", "q_issue_at_star_p0", "bp_star_mix", "b_next_mix"):
        array = _drift(column)
        if array is not None:
            drift_series[column] = array
    if not drift_series:
        return _skip("F06", REASON_NO_POLICY_ROWS)
    for column, array in drift_series.items():
        panels.append(
            _Panel(
                kind="line",
                title=f"{column} drift across episodes",
                xlabel="Episode e (difference vs e-1)",
                ylabel="mean |f_e - f_{e-1}| (row-by-row on fixed bank states)",
                family=_family_of(column),
                options={"series": [{"x": x, "y": array, "label": column}]},
            )
        )

    flip_series = []
    for name, label in (
        ("policy_flip_rate", "policy_flip_rate"),
        ("default_flip_rate", "default_flip_rate"),
        ("investment_flip_rate", "investment_flip_rate"),
    ):
        values = []
        for previous, current in zip(keys[:-1], keys[1:]):
            left = frames[previous]
            right = frames[current]
            if name == "policy_flip_rate":
                if not {"bp_star_mix", "eta"}.issubset(left.columns) or "bp_star_mix" not in right.columns:
                    values.append(np.nan)
                    continue
                eta = _table_column(left, "eta")
                diagnostics = getattr(episodes[current], "diagnostics", None) or {}
                tol = float(diagnostics.get("policy_flip_tol", 1e-2))
                mask = (
                    np.asarray(eta, dtype=np.float64) > 0.5
                    if eta is not None
                    else np.ones(len(left), dtype=bool)
                )
                delta = np.abs(
                    _table_column(right, "bp_star_mix") - _table_column(left, "bp_star_mix")
                )
                values.append(float(np.mean(delta[mask] > tol)) if mask.any() else np.nan)
            elif name == "default_flip_rate":
                if "Phat" not in left.columns or "Phat" not in right.columns:
                    values.append(np.nan)
                    continue
                values.append(
                    float(
                        np.mean(
                            (_table_column(left, "Phat") <= 0.0)
                            != (_table_column(right, "Phat") <= 0.0)
                        )
                    )
                )
            else:
                if not {"PI", "P0"}.issubset(left.columns) or not {"PI", "P0"}.issubset(right.columns):
                    values.append(np.nan)
                    continue
                left_dec = _table_column(left, "PI") > _table_column(left, "P0")
                right_dec = _table_column(right, "PI") > _table_column(right, "P0")
                values.append(float(np.mean(left_dec != right_dec)))
        array = np.asarray(values, dtype=np.float64)
        if np.isfinite(array).any():
            flip_series.append({"x": x, "y": array, "label": label})
    if flip_series:
        panels.append(
            _Panel(
                kind="line",
                title="decision flip rate",
                xlabel="Episode e (difference vs e-1)",
                ylabel="flip share (threshold policy_flip_tol)",
                options={"series": flip_series},
            )
        )

    comparison_series = []
    if isinstance(comparison, pd.DataFrame) and {"episode", "mean_abs_diff"}.issubset(
        comparison.columns
    ):
        group_column = "surface" if "surface" in comparison.columns else comparison.columns[0]
        for surface_name, group in comparison.groupby(group_column):
            comparison_series.append(
                {
                    "x": pd.to_numeric(group["episode"], errors="coerce").to_numpy(dtype=np.float64),
                    "y": pd.to_numeric(group["mean_abs_diff"], errors="coerce").to_numpy(dtype=np.float64),
                    "label": str(surface_name),
                }
            )
    if comparison_series:
        panels.append(
            _Panel(
                kind="line",
                title="comparator artifact drift (comparison)",
                xlabel="Episode",
                ylabel="mean_abs_diff",
                options={"series": comparison_series},
            )
        )
    else:
        panels.append(
            _Panel(
                kind="text",
                title="comparator input",
                options={
                    "text": "comparison=None or unsupported format\n(panel omitted; drift is recomputed row-by-row from policy_rows)"
                },
            )
        )

    while len(panels) < 9:
        panels.append(_Panel(kind="text", title="(no data)", options={"text": "drift caliber unavailable"}))

    n_samples = int(len(next(iter(frames.values())))) * max(len(keys) - 1, 1)
    meta = _base_meta(
        episode=f"{keys[0]}..{keys[-1]}",
        eta="η∈{0,1} (flip statistics on the η=1 row)",
        i_slice="all bank states",
        data_source=_describe_bank_source(getattr(episodes[keys[-1]], "diagnostics", None)),
        capital_weighted="no",
        n_samples=n_samples,
        extra={
            "episodes": [key for key in keys],
            "drift_definition": "mean of row-wise |f_e - f_{e-1}| on the same fixed bank states",
        },
    )
    return _render_grid(
        "F06",
        panels,
        rows=3,
        cols=3,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS["F06"],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=meta["episode"], eta="η∈{0,1}", i_slice="all bank states",
            data_source=str(meta["data_source"]), capital_weighted="no", n_samples=n_samples,
        ),
    )


def _macro_pairs(result):
    frame = _pick_table(result, "fc1_one_step_pairs")
    if frame is None or frame.empty:
        return None
    if not {"hatcf_t", "hatc_t", "lnkf_t", "lnk_t"}.issubset(frame.columns):
        return None
    return frame


def render_f07(
    *,
    macro_results,
    output_dir,
    style=None,
    annotations=None,
) -> FigureRecord:
    """F07: FC1 forecast vs realized (scatter + path)."""
    episodes = _as_episode_map(macro_results)
    if not episodes:
        return _skip("F07", REASON_NO_MACRO_PAIRS)
    key = list(episodes)[-1]
    result = episodes[key]
    frame = _macro_pairs(result)
    if frame is None:
        return _skip("F07", REASON_NO_MACRO_PAIRS)

    style = resolve_style(style, [result])
    hatc_x = _table_column(frame, "hatc_t")
    hatc_y = _table_column(frame, "hatcf_t")
    lnk_x = _table_column(frame, "lnk_t")
    lnk_y = _table_column(frame, "lnkf_t")

    panels = [
        _Panel(
            kind="scatter",
            title="Hatcf_t vs Hatc_t (one-step forecast vs realized, shift=0)",
            xlabel="Hatc_t = log(C/K) realized aggregate",
            ylabel="Hatcf_t = belief forecast of t",
            options={"series": [{"x": hatc_x, "y": hatc_y, "label": "Bank D macro holding period"}]},
        ),
        _Panel(
            kind="scatter",
            title="LnKF_t vs LnK_t (capital belief vs realized)",
            xlabel="LnK_t = log(ΣK) realized aggregate",
            ylabel="LnKF_t = belief forecast of t",
            options={"series": [{"x": lnk_x, "y": lnk_y, "label": "Bank D macro holding period"}]},
        ),
    ]

    path_series_hatc = []
    path_series_lnk = []
    if "path" in frame.columns and "t" in frame.columns:
        path_column = pd.to_numeric(frame["path"], errors="coerce")
        paths = sorted(path_column.dropna().unique())[:4]
        for path_id in paths:
            subset = frame.loc[path_column == path_id].sort_values("t", kind="mergesort")
            t_values = _table_column(subset, "t")
            path_series_hatc.append(
                {"x": t_values, "y": _table_column(subset, "hatc_t"), "label": f"path {int(path_id)} realized"}
            )
            path_series_hatc.append(
                {"x": t_values, "y": _table_column(subset, "hatcf_t"),
                 "label": f"path {int(path_id)} forecast", "linestyle": "--"}
            )
            path_series_lnk.append(
                {"x": t_values, "y": _table_column(subset, "lnk_t"), "label": f"path {int(path_id)} realized"}
            )
            path_series_lnk.append(
                {"x": t_values, "y": _table_column(subset, "lnkf_t"),
                 "label": f"path {int(path_id)} forecast", "linestyle": "--"}
            )
    panels.append(
        _Panel(
            kind="line",
            title="Hatcf vs Hatc paths (first 4 paths)",
            xlabel="t (macro node)",
            ylabel="Hatc / Hatcf",
            options={"series": path_series_hatc},
        )
    )
    panels.append(
        _Panel(
            kind="line",
            title="LnKF vs LnK paths (first 4 paths)",
            xlabel="t (macro node)",
            ylabel="LnK / LnKF",
            options={"series": path_series_lnk},
        )
    )

    stats_text = []
    for label, calculated, forecast in (("hatc", hatc_x, hatc_y), ("lnk", lnk_x, lnk_y)):
        mask = np.isfinite(calculated) & np.isfinite(forecast)
        if mask.any():
            err = forecast[mask] - calculated[mask]
            stats_text.append(
                f"{label}: n={int(mask.sum())}  MAE={np.mean(np.abs(err)):.4g}  "
                f"Bias={np.mean(err):.4g}  RMSE={np.sqrt(np.mean(err ** 2)):.4g}"
            )
    panels.append(
        _Panel(kind="text", title="one-step forecast error summary", options={"text": "\n".join(stats_text) or "no data"})
    )
    panels.append(
        _Panel(
            kind="text",
            title="caliber notes",
            options={
                "text": (
                    "hatcf/lnkf are beliefs (firm state columns 5/6)\n"
                    "Hatc/LnK are realized aggregates\n"
                    "same-row hatcf_t vs Hatc_t is the shift=0 one-step forecast\n"
                    "pairing uses only (path,t,branch,economic_node_id) keys"
                )
            },
        )
    )

    n_samples = int(len(frame))
    meta = _base_meta(
        episode=key,
        eta="n/a (macro rows)",
        i_slice="n/a",
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="no",
        n_samples=n_samples,
        extra={"pair_keys": ["path", "t", "branch", "economic_node_id"], "shift": 0},
    )
    return _render_grid(
        "F07",
        panels,
        rows=2,
        cols=3,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS["F07"],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=key, data_source=str(meta["data_source"]),
            capital_weighted="no", n_samples=n_samples,
        ),
    )


def _anchor_path_tables(result):
    tables = _result_tables(result)
    return {
        str(key): frame
        for key, frame in tables.items()
        if str(key).startswith("fc1_anchor_recursion_")
        and isinstance(frame, pd.DataFrame)
        and not frame.empty
    }


def render_f08(
    *,
    macro_results,
    output_dir,
    style=None,
    annotations=None,
) -> FigureRecord:
    """F08: macro dynamics / SDF time series."""
    episodes = _as_episode_map(macro_results)
    if not episodes:
        return _skip("F08", REASON_NO_MACRO_ANCHOR_PATHS)
    key = list(episodes)[-1]
    result = episodes[key]
    style = resolve_style(style, [result])

    anchors = _anchor_path_tables(result)
    sdf_table = _pick_table(result, "sdf_macro_state_summary")
    if not anchors and sdf_table is None:
        return _skip("F08", REASON_NO_MACRO_ANCHOR_PATHS)

    panels = []
    n_samples = 0
    for name, frame in sorted(anchors.items()):
        panels.append(
            _Panel(
                kind="line",
                title=f"macro-anchor recursion path {name.replace('fc1_anchor_recursion_', '')}",
                xlabel="recursion step h",
                ylabel="forecast (hatc_pred / lnk_pred)",
                options={
                    "series": [
                        {"x": _table_column(frame, "step"), "y": _table_column(frame, "hatc_pred"),
                         "label": "hatc_pred"},
                        {"x": _table_column(frame, "step"), "y": _table_column(frame, "lnk_pred"),
                         "label": "lnk_pred", "linestyle": "--"},
                    ]
                },
            )
        )
        n_samples += int(len(frame))

    if sdf_table is not None and {"macro_state", "M_mean"}.issubset(sdf_table.columns):
        labels = sdf_table["macro_state"].astype(str).tolist()
        panels.append(
            _Panel(
                kind="bar",
                title="SDF M conditional mean (by macro state)",
                xlabel="macro state (x quantile: Recession/Normal/Expansion)",
                ylabel="E[M | macro state] (± standard error)",
                options={"labels": labels, "values": _table_column(sdf_table, "M_mean"),
                         "yerr": _table_column(sdf_table, "M_mean_se"), "rotation": 15},
            )
        )
        panels.append(
            _Panel(
                kind="bar",
                title="normalized wealth Euler residual (by macro state)",
                xlabel="macro state",
                ylabel="mean residual (± standard error)",
                options={"labels": labels, "values": _table_column(sdf_table, "mean_residual"),
                         "yerr": _table_column(sdf_table, "se_residual"), "rotation": 15},
            )
        )
        n_samples += int(len(sdf_table))

    metrics = _result_metrics(result)
    lines = []
    for name in (
        "sdf_M_mean", "sdf_M_mean_se", "sdf_M_var", "sdf_signed_aio_estimate",
        "sdf_signed_aio_se", "sdf_population_target_estimate", "sdf_population_target_se",
        "sdf_g_mean_low", "sdf_g_mean_high", "sdf_g_var_high", "sdf_g_max",
        "sdf_nonfinite_ratio", "sdf_log_R_clip_share", "sdf_valid_parent_ratio",
    ):
        if name in metrics:
            lines.append(f"{name} = {float(metrics[name]):.6g}")
    panels.append(
        _Panel(kind="text", title="SDF numerical safety and moment constraints", options={"text": "\n".join(lines) or "no SDF metrics"})
    )
    panels.append(
        _Panel(
            kind="text",
            title="interpretation guardrails",
            options={
                "text": (
                    "sample mean of Signed AiO ≠ population conditional moment\n"
                    "population targets are reported as sdf_population_target_*\n"
                    "moment constraints g_* and AiO are reported separately, never pooled"
                )
            },
        )
    )
    while len(panels) < 9:
        panels.append(_Panel(kind="text", title="(no data)", options={"text": "panel data missing"}))

    meta = _base_meta(
        episode=key,
        eta="n/a (macro rows)",
        i_slice="n/a",
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="no",
        n_samples=n_samples,
        extra={"macro_anchors": sorted(anchors), "sdf_table_present": sdf_table is not None},
    )
    return _render_grid(
        "F08",
        panels,
        rows=3,
        cols=3,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS["F08"],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=key, data_source=str(meta["data_source"]),
            capital_weighted="no", n_samples=n_samples,
        ),
    )


def _macro_state_labels(x_values, quantiles=(1.0 / 3.0, 2.0 / 3.0)):
    array = np.asarray(x_values, dtype=np.float64)
    out = np.empty(array.shape[0], dtype=object)
    out[:] = ""
    finite = np.isfinite(array)
    if not finite.any():
        return out.astype(str)
    edges = [float(np.quantile(array[finite], float(q))) for q in quantiles]
    bins = np.digitize(array, edges, right=True)
    labels = ("recession", "normal", "expansion")
    for index, label in enumerate(labels):
        out[bins == index] = label
    out[~finite] = ""
    return out.astype(str)


def _sim_firm_panel(result):
    frame = _pick_table(result, "firm_panel")
    if frame is None or frame.empty:
        return None
    if not {"b", "x"}.issubset(frame.columns):
        return None
    return frame


def render_f09(
    *,
    sim_results,
    output_dir,
    style=None,
    annotations=None,
) -> FigureRecord:
    """F09: leverage distribution by macro state (Gomes-style histogram)."""
    episodes = _as_episode_map(sim_results)
    if not episodes:
        return _skip("F09", REASON_NO_SIM_PANEL)
    key = list(episodes)[-1]
    result = episodes[key]
    panel_frame = _sim_firm_panel(result)
    if panel_frame is None:
        return _skip("F09", REASON_NO_SIM_PANEL)

    style = resolve_style(style, [result])
    b_values = _table_column(panel_frame, "b")
    x_values = _table_column(panel_frame, "x")
    k_values = _table_column(panel_frame, "K")
    if b_values is None or x_values is None:
        return _skip("F09", REASON_NO_SIM_PANEL)
    if k_values is None:
        k_values = np.ones_like(b_values)

    t_column = _table_column(panel_frame, "t")
    states = _macro_state_labels(x_values)
    slices = []
    if t_column is not None and np.isfinite(t_column).any():
        unique = sorted(set(_finite(t_column).tolist()))
        if unique:
            picks = {unique[0], unique[-1]}
            if len(unique) > 2:
                picks.add(unique[len(unique) // 2])
            slices = sorted(picks)
    if not slices:
        slices = [float("nan")]

    b_low = float(style.b_range[0]) if style.b_range is not None else float(np.nanmin(b_values))
    b_high = float(style.b_range[1]) if style.b_range is not None else float(np.nanmax(b_values))
    if not math.isfinite(b_low) or not math.isfinite(b_high) or b_high <= b_low:
        good = _finite(b_values)
        if good.size == 0:
            return _skip("F09", REASON_NO_FINITE_SAMPLES)
        b_low, b_high = float(np.min(good)), float(np.max(good))
        if b_high <= b_low:
            b_high = b_low + 1e-9
    edges = np.linspace(b_low, b_high, 21)

    panels = []
    n_samples = 0
    for caliber, weights, label in (
        ("count", np.ones_like(b_values), "count weighted"),
        ("capital", k_values, "capital weighted"),
    ):
        for state in ("recession", "normal", "expansion"):
            series = []
            for slice_value in slices:
                mask = states == state
                if t_column is not None and math.isfinite(slice_value):
                    mask = mask & np.isclose(t_column, slice_value)
                subset = b_values[mask]
                w = weights[mask]
                if subset.size == 0:
                    continue
                hist, bin_edges = np.histogram(subset, bins=edges, weights=w)
                total = float(np.nansum(hist))
                if total <= 0:
                    continue
                hist = hist / total
                centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                series.append(
                    {
                        "x": centers,
                        "y": hist,
                        "label": f"t={slice_value:g}" if math.isfinite(slice_value) else "all t",
                    }
                )
                n_samples += int(np.isfinite(subset).sum())
            panels.append(
                _Panel(
                    kind="hist",
                    title=f"{state} | {label}",
                    xlabel="b = leverage ratio (debt/capital)",
                    ylabel="distribution mass (normalized)",
                    options={"series": series, "fill": caliber == "count"},
                )
            )
    if not any(item.options.get("series") for item in panels):
        return _skip("F09", REASON_NO_FINITE_SAMPLES)

    meta = _base_meta(
        episode=key,
        eta="n/a (simulation cross-section)",
        i_slice="n/a",
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="count and capital weighted (top row count, bottom row capital)",
        extinct_paths=_describe_extinct(result),
        n_samples=n_samples,
        extra={"macro_state_rule": "x quantiles (1/3, 2/3) → recession/normal/expansion",
               "time_slices": [float(v) for v in slices]},
    )
    return _render_grid(
        "F09",
        panels,
        rows=2,
        cols=3,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS["F09"],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=key, data_source=str(meta["data_source"]),
            capital_weighted="both calibers", extinct_paths=str(meta["extinct_paths"]),
            n_samples=n_samples,
        ),
    )


def render_f10(
    *,
    sim_results,
    output_dir,
    style=None,
    annotations=None,
) -> FigureRecord:
    """F10: b-z joint distribution over time (heatmap + marginals)."""
    episodes = _as_episode_map(sim_results)
    if not episodes:
        return _skip("F10", REASON_NO_JOINT_HIST)
    key = list(episodes)[-1]
    result = episodes[key]
    hists = _joint_hists(result)
    if not hists:
        return _skip("F10", REASON_NO_JOINT_HIST)
    marginals = _pick_table(result, "distribution_marginals")
    style = resolve_style(style, [result])

    slots = sorted(hists)
    selected = slots[:6]
    rows = 2 if len(selected) > 3 else 1
    cols = 3 if len(selected) > 3 else max(len(selected), 1)

    _apply_style(style)
    fig = plt.figure(figsize=(5.0 * cols, 4.6 * rows))
    outer = fig.add_gridspec(rows, cols)
    drawn = 0
    used = []
    total_cells = 0
    for index, (t_slice, caliber) in enumerate(selected):
        row, col = divmod(index, cols)
        hist = np.asarray(hists[(t_slice, caliber)], dtype=np.float64)
        if hist.ndim != 2 or hist.size == 0:
            used.append({"title": f"t={t_slice}/{caliber}", "drawn": False})
            continue
        b_range = style.b_range or (0.0, 1.0)
        z_range = style.z_range or (-1.0, 1.0)
        nb, nz = hist.shape
        b_edges = np.linspace(float(b_range[0]), float(b_range[1]), nb + 1)
        z_edges = np.linspace(float(z_range[0]), float(z_range[1]), nz + 1)
        b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
        z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

        sub = outer[row, col].subgridspec(
            2, 2, width_ratios=[1.0, 0.24], height_ratios=[0.24, 1.0],
            wspace=0.08, hspace=0.08,
        )
        ax_top = fig.add_subplot(sub[0, 0])
        ax_main = fig.add_subplot(sub[1, 0], sharex=ax_top)
        ax_side = fig.add_subplot(sub[1, 1], sharey=ax_main)

        total = float(np.nansum(hist))
        normalised = hist / total if total > 0 else hist
        total_cells += int(round(total))
        mesh = ax_main.pcolormesh(
            b_edges, z_edges, normalised.T, shading="flat", cmap=_cmap(style.cmap_density)
        )
        fig.colorbar(mesh, ax=ax_main, fraction=0.046, pad=0.02)
        ax_main.set_xlabel("b = leverage ratio")
        ax_main.set_ylabel("z = firm productivity shock")
        ax_main.set_xlim(float(b_range[0]), float(b_range[1]))
        ax_main.set_ylim(float(z_range[0]), float(z_range[1]))

        ax_top.plot(b_centers, normalised.sum(axis=1), color="tab:blue")
        ax_top.fill_between(b_centers, normalised.sum(axis=1), alpha=0.25, color="tab:blue")
        ax_top.set_ylabel("b marginal")
        ax_top.tick_params(labelbottom=False)
        ax_side.plot(normalised.sum(axis=0), z_centers, color="tab:orange")
        ax_side.fill_betweenx(z_centers, normalised.sum(axis=0), alpha=0.25, color="tab:orange")
        ax_side.set_xlabel("z marginal")
        ax_side.tick_params(labelleft=False)

        if marginals is not None and {"t", "caliber", "axis", "bin_center", "weight"}.issubset(
            marginals.columns
        ):
            t_numeric = pd.to_numeric(marginals["t"], errors="coerce").to_numpy(dtype=np.float64)
            for axis_name, axis in (("b", ax_top), ("z", ax_side)):
                subset = marginals.loc[
                    (marginals["axis"] == axis_name)
                    & np.isclose(t_numeric, float(t_slice))
                    & (marginals["caliber"].astype(str) == str(caliber))
                ]
                if subset.empty:
                    continue
                centers = pd.to_numeric(subset["bin_center"], errors="coerce").to_numpy(dtype=np.float64)
                weights = pd.to_numeric(subset["weight"], errors="coerce").to_numpy(dtype=np.float64)
                if axis_name == "b":
                    axis.plot(centers, weights, color="black", linestyle="--", linewidth=1.0)
                else:
                    axis.plot(weights, centers, color="black", linestyle="--", linewidth=1.0)

        ax_top.set_title(
            f"t={t_slice} | caliber={caliber} ({'capital weighted' if caliber == 'capital' else 'count weighted'})"
        )
        drawn += 1
        used.append({"title": f"t={t_slice}/{caliber}", "drawn": True})

    if drawn == 0:
        plt.close(fig)
        return _skip("F10", REASON_NO_JOINT_HIST)

    first_sentence = re.split(r"(?:。|\.\s)", CAPTIONS["F10"], maxsplit=1)[0]
    fig.suptitle(f"F10: {first_sentence}", fontsize=style.title_size + 1.5)
    fig.subplots_adjust(left=0.06, right=0.955, top=0.90, bottom=0.115)
    footer = " | ".join(
        _annotation_lines(
            _annotation_dict(
                annotations,
                episode=key,
                data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
                capital_weighted="count/capital dual caliber",
                extinct_paths=_describe_extinct(result),
                n_samples=total_cells,
            )
        )
    )
    fig.text(0.005, 0.005, footer, ha="left", va="bottom", fontsize=style.footer_size)
    path = Path(output_dir) / FIGURE_FILENAMES["F10"]
    _save(fig, path, style)

    meta = _base_meta(
        episode=key,
        eta="n/a (simulation cross-section)",
        i_slice="n/a",
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="count/capital dual caliber",
        extinct_paths=_describe_extinct(result),
        n_samples=total_cells,
        extra={"time_slices": [{"t": t, "caliber": c} for t, c in selected],
               "marginals_present": marginals is not None,
               "hist_shape": list(np.asarray(hists[selected[0]]).shape)},
    )
    meta.update(_style_meta(style))
    meta["panel_status"] = used
    return FigureRecord(
        figure_id="F10",
        path=str(path),
        caption=CAPTIONS["F10"],
        status="ok",
        missing_reason=None,
        n_samples=total_cells,
        meta=meta,
    )


def render_f11_waterfall(
    *,
    sim_results,
    output_dir,
    style=None,
    annotations=None,
) -> FigureRecord:
    """F11: t0->t1 capital waterfall + survival / default / active paths."""
    episodes = _as_episode_map(sim_results)
    if not episodes:
        return _skip("F11", REASON_NO_CAPITAL_WATERFALL)
    key = list(episodes)[-1]
    result = episodes[key]
    waterfall = _pick_table(result, "capital_waterfall")
    if waterfall is None or waterfall.empty:
        return _skip("F11", REASON_NO_CAPITAL_WATERFALL)
    period_counts = _pick_table(result, "period_counts")
    path_status = _pick_table(result, "path_status")
    if period_counts is None and path_status is None:
        return _skip("F11", REASON_NO_SURVIVAL_ACCOUNT)

    required = (
        "initial_capital", "ending_capital", "incumbent_dk",
        "entrant_capital", "exiting_incumbent_capital",
    )
    if not set(required).issubset(waterfall.columns):
        return _skip("F11", REASON_NO_CAPITAL_WATERFALL)

    style = resolve_style(style, [result])
    _apply_style(style)
    fig = plt.figure(figsize=(17.0, 9.0))
    grid = fig.add_gridspec(2, 3)

    totals = _pick_table(result, "capital_waterfall_totals")
    source = totals if totals is not None and not totals.empty else waterfall
    steps = []
    if {"t_from", "t_to"}.issubset(source.columns):
        t_from = pd.to_numeric(source["t_from"], errors="coerce").to_numpy(dtype=np.float64)
        t_to = pd.to_numeric(source["t_to"], errors="coerce").to_numpy(dtype=np.float64)
        steps = list(zip(t_from, t_to))
    if not steps:
        plt.close(fig)
        return _skip("F11", REASON_NO_CAPITAL_WATERFALL)

    n_samples = int(len(waterfall))
    for index, (t_from, t_to) in enumerate(steps[:3]):
        ax = fig.add_subplot(grid[0, index])
        if totals is not None and not totals.empty and index < len(source):
            row = source.iloc[index]
        else:
            t_from_col = pd.to_numeric(waterfall["t_from"], errors="coerce")
            t_to_col = pd.to_numeric(waterfall["t_to"], errors="coerce")
            subset = waterfall.loc[np.isclose(t_from_col, t_from) & np.isclose(t_to_col, t_to)]
            row = subset.iloc[0] if not subset.empty else None
        if row is None:
            ax.axis("off")
            continue
        labels = ["initial capital", "survivor ΔK", "entrant capital", "exiting capital", "accounting residual", "ending capital"]
        values = [
            float(row["initial_capital"]),
            float(row["incumbent_dk"]),
            float(row["entrant_capital"]),
            -float(row["exiting_incumbent_capital"]),
            float(row.get("accounting_residual", 0.0)),
            float(row["ending_capital"]),
        ]
        running = 0.0
        bottoms = []
        heights = []
        colors = []
        for position, value in enumerate(values):
            if position in (0, len(values) - 1):
                bottoms.append(0.0)
                heights.append(value)
                colors.append("tab:blue")
                running = value
            else:
                if value >= 0:
                    bottoms.append(running)
                    heights.append(value)
                    colors.append("tab:green")
                else:
                    bottoms.append(running + value)
                    heights.append(-value)
                    colors.append("tab:red")
                running += value
        ax.bar(np.arange(len(labels)), heights, bottom=bottoms, color=colors, alpha=0.85)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("capital (K)")
        ax.set_title(f"capital waterfall t={int(t_from)}→{int(t_to)}")
        ax.grid(alpha=0.2, axis="y")
        residual = float(row.get("accounting_residual", float("nan")))
        ax.text(
            0.02, 0.96, f"identity residual = {residual:.3e}",
            transform=ax.transAxes, va="top", fontsize=style.legend_size,
        )

    ax = fig.add_subplot(grid[1, 0])
    if period_counts is not None and {"t", "active_paths"}.issubset(period_counts.columns):
        t_values = _table_column(period_counts, "t")
        for column, label in (
            ("active_paths", "active paths"),
            ("extinct_paths", "extinct paths"),
            ("n_alive_firms", "surviving firms"),
            ("n_defaults", "defaults/exits"),
            ("n_entrants", "entrants"),
        ):
            values = _table_column(period_counts, column)
            if values is not None:
                ax.plot(t_values, values, marker="o", markersize=3, label=label)
        ax.set_xlabel("t (simulation period)")
        ax.set_ylabel("count")
        ax.set_title("surviving / default / active path account by period")
        ax.legend()
        ax.grid(alpha=0.25)
    elif path_status is not None and {"path", "extinct"}.issubset(path_status.columns):
        flags = path_status["extinct"].astype(bool).to_numpy()
        ax.bar(np.arange(2), [int((~flags).sum()), int(flags.sum())],
               color=["tab:green", "tab:red"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["surviving paths", "extinct paths"])
        ax.set_title("path status (extinct paths stay in the statistics)")
        ax.grid(alpha=0.2, axis="y")

    ax = fig.add_subplot(grid[1, 1])
    if period_counts is not None:
        t_values = _table_column(period_counts, "t")
        for column, label in (
            ("n_defaults", "n_defaults"),
            ("n_entrants", "n_entrants"),
            ("n_investors", "n_investors"),
        ):
            values = _table_column(period_counts, column)
            if values is not None:
                ax.plot(t_values, values, marker="o", markersize=3, label=label)
        ax.set_xlabel("t")
        ax.set_ylabel("count")
        ax.set_title("period event counts")
        ax.legend()
        ax.grid(alpha=0.25)
    elif {"default_rate", "survival_rate"}.issubset(waterfall.columns):
        t_to = _table_column(waterfall, "t_to")
        ax.plot(t_to, _table_column(waterfall, "survival_rate"), marker="o", label="survival_rate")
        ax.plot(t_to, _table_column(waterfall, "default_rate"), marker="o", label="default_rate")
        ax.set_xlabel("t_to")
        ax.set_ylabel("rate")
        ax.set_title("survival rate / default rate")
        ax.legend()
        ax.grid(alpha=0.25)

    ax = fig.add_subplot(grid[1, 2])
    if "accounting_residual" in waterfall.columns:
        t_to = _table_column(waterfall, "t_to")
        ax.plot(t_to, _table_column(waterfall, "accounting_residual"),
                marker="o", linestyle="none", markersize=3)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xlabel("t_to")
        ax.set_ylabel("accounting_residual")
        ax.set_title("capital identity residual (should stay near 0)")
        ax.grid(alpha=0.25)

    first_sentence = re.split(r"(?:。|\.\s)", CAPTIONS["F11"], maxsplit=1)[0]
    fig.suptitle(f"F11: {first_sentence}", fontsize=style.title_size + 1.5)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.135, hspace=0.55, wspace=0.30)
    footer = " | ".join(
        _annotation_lines(
            _annotation_dict(
                annotations,
                episode=key,
                data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
                capital_weighted="capital ledger (aggregate caliber)",
                extinct_paths=_describe_extinct(result),
                n_samples=n_samples,
            )
        )
    )
    fig.text(0.005, 0.005, footer, ha="left", va="bottom", fontsize=style.footer_size)
    path = Path(output_dir) / FIGURE_FILENAMES["F11"]
    _save(fig, path, style)

    meta = _base_meta(
        episode=key,
        eta="n/a (simulation cross-section)",
        i_slice="n/a",
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="capital ledger (aggregate caliber)",
        extinct_paths=_describe_extinct(result),
        n_samples=n_samples,
        extra={
            "transitions": [[float(a), float(b)] for a, b in steps[:3]],
            "identity": "K_end-K_start = deltaK_incumbent + K_entry_endpoint - K_exit_old + residual",
            "survival_panel_source": "period_counts" if period_counts is not None else "path_status",
        },
    )
    meta.update(_style_meta(style))
    return FigureRecord(
        figure_id="F11",
        path=str(path),
        caption=CAPTIONS["F11"],
        status="ok",
        missing_reason=None,
        n_samples=n_samples,
        meta=meta,
    )


def render_f12(
    *,
    output_dir,
    bellman_results=None,
    policy_results=None,
    macro_results=None,
    sim_results=None,
    comparison=None,
    style=None,
    annotations=None,
) -> FigureRecord:
    """F12: general-equilibrium convergence dashboard."""
    blocks = {
        "bellman": _as_episode_map(bellman_results),
        "policy": _as_episode_map(policy_results),
        "macro": _as_episode_map(macro_results),
        "sim": _as_episode_map(sim_results),
    }
    present = [name for name, values in blocks.items() if values]
    if len(present) < 2:
        return _skip("F12", REASON_INSUFFICIENT_BLOCKS)
    style = resolve_style(style, [item for values in blocks.values() for item in values.values()])

    panels = []
    n_samples = 0

    bellman = blocks["bellman"]
    if bellman:
        keys = list(bellman)
        x = np.arange(len(keys), dtype=np.float64)
        series = []
        for metric, label in (
            ("fresh_p0_normalized_mae", "fresh P0 MAE"),
            ("fresh_pi_normalized_mae", "fresh PI MAE"),
            ("frozen_p0_normalized_mae", "frozen P0 MAE"),
            ("frozen_pi_normalized_mae", "frozen PI MAE"),
        ):
            array = np.asarray(
                [float(_result_metrics(bellman[key]).get(metric, float("nan"))) for key in keys],
                dtype=np.float64,
            )
            if np.isfinite(array).any():
                series.append({"x": x, "y": array, "label": label})
                n_samples += int(np.isfinite(array).sum())
        panels.append(
            _Panel(
                kind="line",
                title="Bellman residual convergence",
                xlabel="Episode",
                ylabel="MAE (normalized)",
                options={"series": series, "logy": "symlog"},
            )
        )
    else:
        panels.append(_Panel(kind="text", title="Bellman", options={"text": "no bellman results"}))

    policy = blocks["policy"]
    if len(policy) >= 2:
        keys = list(policy)
        drift_x = np.arange(1, len(keys), dtype=np.float64)
        series = []
        for column, label in (("P", "mean|ΔP|"), ("bp_star_mix", "mean|Δbp*|")):
            values = []
            for previous, current in zip(keys[:-1], keys[1:]):
                left = _table_column(_pick_table(policy[previous], "policy_rows"), column)
                right = _table_column(_pick_table(policy[current], "policy_rows"), column)
                values.append(
                    float(np.nanmean(np.abs(right - left)))
                    if left is not None and right is not None
                    else np.nan
                )
            array = np.asarray(values, dtype=np.float64)
            if np.isfinite(array).any():
                series.append({"x": drift_x, "y": array, "label": label})
                n_samples += int(np.isfinite(array).sum())
        panels.append(
            _Panel(
                kind="line",
                title="P / policy drift",
                xlabel="Episode e (vs e-1)",
                ylabel="mean |Δ|",
                options={"series": series},
            )
        )
    else:
        panels.append(_Panel(kind="text", title="value/policy drift", options={"text": "insufficient episodes (<2)"}))

    macro = blocks["macro"]
    if macro:
        keys = list(macro)
        x = np.arange(len(keys), dtype=np.float64)
        series = []
        for metric, label in (
            ("fc1_one_step_hatc_mae", "hatc one-step MAE"),
            ("fc1_one_step_lnk_mae", "lnk one-step MAE"),
            ("sdf_g_max", "max SDF moment-constraint violation"),
        ):
            array = np.asarray(
                [float(_result_metrics(macro[key]).get(metric, float("nan"))) for key in keys],
                dtype=np.float64,
            )
            if np.isfinite(array).any():
                series.append({"x": x, "y": array, "label": label})
                n_samples += int(np.isfinite(array).sum())
        panels.append(
            _Panel(
                kind="line",
                title="FC1 one-step error and SDF constraints",
                xlabel="Episode",
                ylabel="metric value",
                options={"series": series},
            )
        )
    else:
        panels.append(_Panel(kind="text", title="macro / SDF", options={"text": "no macro results"}))

    sim = blocks["sim"]
    if sim:
        keys = list(sim)
        x = np.arange(len(keys), dtype=np.float64)
        series = []
        for metric, label in (
            ("sim_baseline_capital_change_t0_t1", "capital change t0→t1"),
            ("sim_baseline_initial_cohort_default_share_t1", "initial cohort default share"),
        ):
            array = np.asarray(
                [float(_result_metrics(sim[key]).get(metric, float("nan"))) for key in keys],
                dtype=np.float64,
            )
            if np.isfinite(array).any():
                series.append({"x": x, "y": array, "label": label})
                n_samples += int(np.isfinite(array).sum())
        panels.append(
            _Panel(
                kind="line",
                title="fixed-shock rollout summary",
                xlabel="Episode",
                ylabel="metric value",
                options={"series": series},
            )
        )
    else:
        panels.append(_Panel(kind="text", title="simulation / distribution", options={"text": "no sim results"}))

    coverage = "result block coverage: " + ", ".join(
        f"{name}={'yes' if blocks[name] else 'no'}" for name in blocks
    )
    panels.append(_Panel(kind="text", title="data coverage", options={"text": coverage}))
    panels.append(
        _Panel(
            kind="text",
            title="criterion notes",
            options={
                "text": (
                    "convergence = firm values/policies + macro beliefs + pricing kernel all stable\n"
                    "training loss is not convergence evidence\n"
                    "no PASS/FAIL verdict when thresholds are unconfigured\n"
                    "(THRESHOLD_NOT_CONFIGURED)"
                )
            },
        )
    )

    meta = _base_meta(
        episode="multi-episode summary",
        eta="η∈{0,1}",
        i_slice="all bank states",
        data_source="multi-bank summary",
        capital_weighted="per-panel caliber",
        n_samples=n_samples,
        extra={"result_blocks": present},
    )
    return _render_grid(
        "F12",
        panels,
        rows=2,
        cols=3,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS["F12"],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode="multi-episode", eta="η∈{0,1}",
            data_source="multi-bank summary", capital_weighted="mixed calibers", n_samples=n_samples,
        ),
    )


# ---------------------------------------------------------------------------
# V-series renderers
# ---------------------------------------------------------------------------


def _value_series_figure(
    figure_id: str,
    surface_key: str,
    family: str,
    *,
    policy_results,
    output_dir,
    style=None,
    episode=None,
    annotations=None,
) -> FigureRecord:
    episode_key, result, reason = _select_episode(policy_results, episode)
    if result is None:
        return _skip(figure_id, reason or REASON_NO_POLICY_SURFACES)
    axes_pair = grid_axes(result)
    if axes_pair is None:
        return _skip(figure_id, REASON_NO_GRID_AXES)
    b_values, z_values = axes_pair
    if (
        _surface(result, f"eta1_{surface_key}") is None
        and _surface(result, f"eta0_{surface_key}") is None
    ):
        return _skip(figure_id, REASON_NO_POLICY_SURFACES)
    style = resolve_style(style, [result])
    region = _default_region(result, (b_values.size, z_values.size))
    overlays = _boundary_overlay(result, style)

    panels = []
    for view in ("eta1", "eta0"):
        view_values = _surface(result, f"{view}_{surface_key}")
        if view_values is None:
            continue
        panels.append(
            _Panel(
                kind="3d",
                title=f"3D {surface_key} ({_eta_label(view)})",
                family=family,
                values=view_values,
                options={"b_values": b_values, "z_values": z_values},
            )
        )
        panels.append(
            _Panel(
                kind="contour",
                title=f"contour {surface_key} ({_eta_label(view)}) + default boundary",
                family=family,
                values=view_values,
                options={"b_values": b_values, "z_values": z_values,
                         "region": region, "overlays": overlays},
            )
        )
    while len(panels) < 4:
        panels.append(_Panel(kind="text", title="(no data)", options={"text": "surface missing for this η view"}))

    n_samples = int(sum(_n_finite(item.values) for item in panels if item.values is not None))
    meta = _base_meta(
        episode=episode_key,
        eta="η=1 / η=0 dual view",
        i_slice=_i_slice_text(result),
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="no",
        n_samples=n_samples,
        extra={"surface": surface_key, "canonical_3d_note": "canonical set: see F01"},
    )
    return _render_grid(
        figure_id,
        panels,
        rows=2,
        cols=2,
        output_dir=Path(output_dir),
        style=style,
        caption=CAPTIONS[figure_id],
        meta=meta,
        n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=episode_key, eta="η=1/η=0", i_slice=_i_slice_text(result),
            data_source=str(meta["data_source"]), capital_weighted="no", n_samples=n_samples,
        ),
    )


def render_v01(*, policy_results, output_dir, style=None, episode=None, annotations=None) -> FigureRecord:
    """V01: P0 surface."""
    return _value_series_figure(
        "V01", "P0", "value", policy_results=policy_results,
        output_dir=output_dir, style=style, episode=episode, annotations=annotations,
    )


def render_v02(*, policy_results, output_dir, style=None, episode=None, annotations=None) -> FigureRecord:
    """V02: PI surface with the investment cost ``i`` annotated."""
    return _value_series_figure(
        "V02", "PI", "value", policy_results=policy_results,
        output_dir=output_dir, style=style, episode=episode, annotations=annotations,
    )


def render_v03(*, policy_results, output_dir, style=None, episode=None, annotations=None) -> FigureRecord:
    """V03: equity P with Phat and the Phat<=0 default region."""
    episode_key, result, reason = _select_episode(policy_results, episode)
    if result is None:
        return _skip("V03", reason or REASON_NO_POLICY_SURFACES)
    axes_pair = grid_axes(result)
    if axes_pair is None:
        return _skip("V03", REASON_NO_GRID_AXES)
    b_values, z_values = axes_pair
    p_values = _surface(result, "eta1_P")
    phat = _surface(result, "eta1_Phat")
    if p_values is None or phat is None:
        return _skip("V03", REASON_NO_POLICY_SURFACES)
    style = resolve_style(style, [result])
    region = phat <= 0.0
    overlays = _boundary_overlay(result, style)
    panels = [
        _Panel(kind="3d", title="3D P (η=1, equity value after investment choice)", family="value", values=p_values,
               options={"b_values": b_values, "z_values": z_values}),
        _Panel(kind="contour", title="P + Phat≤0 default region", family="value", values=p_values,
               options={"b_values": b_values, "z_values": z_values,
                        "region": region, "overlays": overlays}),
        _Panel(kind="heatmap", title="Phat(b,z) (hard-default discriminant, ≤0 = default)", family="value", values=phat,
               options={"b_values": b_values, "z_values": z_values,
                        "region": region, "overlays": overlays}),
        _Panel(kind="heatmap", title="default region indicator (Phat≤0)", family="prob",
               values=region.astype(np.float64),
               options={"b_values": b_values, "z_values": z_values, "overlays": overlays}),
    ]
    n_samples = int(_n_finite(p_values))
    meta = _base_meta(
        episode=episode_key,
        eta="η=1",
        i_slice=_i_slice_text(result),
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="no",
        n_samples=n_samples,
        extra={"default_region_share": float(region.mean()),
               "phat_semantics": "Phat≤0 = hard default; bar_z is a soft probability, never mixed"},
    )
    return _render_grid(
        "V03", panels, rows=2, cols=2, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS["V03"], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=episode_key, eta="η=1", i_slice=_i_slice_text(result),
            data_source=str(meta["data_source"]), capital_weighted="no", n_samples=n_samples,
        ),
    )


def render_v04(*, policy_results, output_dir, style=None, episode=None, annotations=None) -> FigureRecord:
    """V04: the four Q calibers (claim / unit / effective / recovery)."""
    episode_key, result, reason = _select_episode(policy_results, episode)
    if result is None:
        return _skip("V04", reason or REASON_NO_POLICY_SURFACES)
    axes_pair = grid_axes(result)
    if axes_pair is None:
        return _skip("V04", REASON_NO_GRID_AXES)
    b_values, z_values = axes_pair
    spec = (
        ("eta1_Q_claim", "Q Claim = b₊·q_unit (issuance pricing caliber)"),
        ("eta1_q_unit", "q_unit (unit bond price)"),
        ("eta1_Q_effective", "Q Effective (settlement caliber, hybrid includes the hard default gate)"),
        ("eta1_recovery", "recovery = φ(1-δ+exp(x+z)) (default recovery)"),
    )
    panels = []
    for key, label in spec:
        values = _surface(result, key)
        if values is None:
            return _skip("V04", REASON_NO_Q_SURFACES)
        panels.append(
            _Panel(
                kind="heatmap",
                title=label,
                family="q",
                values=values,
                options={"b_values": b_values, "z_values": z_values,
                         "region": _default_region(result, values.shape),
                         "overlays": _boundary_overlay(result, style)},
            )
        )
    style = resolve_style(style, [result])
    n_samples = int(sum(_n_finite(item.values) for item in panels))
    meta = _base_meta(
        episode=episode_key,
        eta="η=1",
        i_slice=_i_slice_text(result),
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="no",
        n_samples=n_samples,
        extra={"q_calibers": ["Q_claim", "q_unit", "Q_effective", "recovery"]},
    )
    return _render_grid(
        "V04", panels, rows=2, cols=2, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS["V04"], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=episode_key, eta="η=1", i_slice=_i_slice_text(result),
            data_source=str(meta["data_source"]), capital_weighted="no", n_samples=n_samples,
        ),
    )


def render_v05(*, policy_results, output_dir, style=None, episode=None, annotations=None) -> FigureRecord:
    """V05: value contours + default boundaries."""
    episode_key, result, reason = _select_episode(policy_results, episode)
    if result is None:
        return _skip("V05", reason or REASON_NO_POLICY_SURFACES)
    axes_pair = grid_axes(result)
    if axes_pair is None:
        return _skip("V05", REASON_NO_GRID_AXES)
    b_values, z_values = axes_pair
    style = resolve_style(style, [result])
    overlays = _boundary_overlay(result, style)
    region = _default_region(result, (b_values.size, z_values.size))
    panels = []
    for key, label in (("eta1_P0", "P0"), ("eta1_PI", "PI"), ("eta1_P", "P")):
        values = _surface(result, key)
        if values is None:
            return _skip("V05", REASON_NO_POLICY_SURFACES)
        panels.append(
            _Panel(
                kind="contour",
                title=f"{label} contours + hard/soft default boundaries",
                family=_family_of(key),
                values=values,
                options={"b_values": b_values, "z_values": z_values,
                         "region": region, "overlays": overlays},
            )
        )
    hard_soft = _pick_table(result, "hard_soft_boundary")
    if hard_soft is not None and {"b", "z_hard_phat0"}.issubset(hard_soft.columns):
        panels.append(
            _Panel(
                kind="line",
                title="hard boundary Phat=0 vs soft boundary bar_z=0.5",
                xlabel="b = leverage ratio",
                ylabel="z boundary",
                options={
                    "series": [
                        {"x": _table_column(hard_soft, "b"),
                         "y": _table_column(hard_soft, "z_hard_phat0"), "label": "Phat=0"},
                        {"x": _table_column(hard_soft, "b"),
                         "y": _table_column(hard_soft, "z_soft_barz0p5"),
                         "label": "bar_z=0.5", "linestyle": "--"},
                    ]
                },
            )
        )
    else:
        panels.append(_Panel(kind="text", title="boundary comparison", options={"text": "hard_soft_boundary table missing"}))
    n_samples = int(sum(_n_finite(item.values) for item in panels if item.values is not None))
    meta = _base_meta(
        episode=episode_key,
        eta="η=1",
        i_slice=_i_slice_text(result),
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="no",
        n_samples=n_samples,
    )
    return _render_grid(
        "V05", panels, rows=2, cols=2, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS["V05"], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=episode_key, eta="η=1", i_slice=_i_slice_text(result),
            data_source=str(meta["data_source"]), capital_weighted="no", n_samples=n_samples,
        ),
    )


def render_v06(*, policy_results, output_dir, style=None, annotations=None) -> FigureRecord:
    """V06: value drift across episodes."""
    episodes = _as_episode_map(policy_results)
    if len(episodes) < 2:
        return _skip("V06", REASON_INSUFFICIENT_EPISODES)
    keys = list(episodes)
    axes_pair = grid_axes(episodes[keys[-1]])
    if axes_pair is None:
        return _skip("V06", REASON_NO_GRID_AXES)
    b_values, z_values = axes_pair
    style = resolve_style(style, list(episodes.values()))

    panels = []
    n_samples = 0
    summary_series = []
    for key, family in (("eta1_P0", "P0"), ("eta1_PI", "PI"), ("eta1_P", "P")):
        differences = []
        means = []
        for previous, current in zip(keys[:-1], keys[1:]):
            left = _surface(episodes[previous], key)
            right = _surface(episodes[current], key)
            if left is None or right is None or left.shape != right.shape:
                differences.append(None)
                means.append(np.nan)
                continue
            delta = right - left
            differences.append(delta)
            good = _finite(delta)
            means.append(float(np.mean(np.abs(good))) if good.size else np.nan)
        array = np.asarray(means, dtype=np.float64)
        if np.isfinite(array).any():
            summary_series.append(
                {"x": np.arange(1, len(keys), dtype=np.float64), "y": array,
                 "label": f"mean|Δ{family}|"}
            )
        last_delta = next((item for item in reversed(differences) if item is not None), None)
        if last_delta is not None:
            panels.append(
                _Panel(
                    kind="heatmap",
                    title=f"Δ{family} ({keys[-2]} → {keys[-1]})",
                    family="residual",
                    values=last_delta,
                    options={"b_values": b_values, "z_values": z_values,
                             "region": _default_region(episodes[keys[-1]], last_delta.shape)},
                )
            )
            n_samples += _n_finite(last_delta)
    if not panels:
        return _skip("V06", REASON_NO_POLICY_SURFACES)
    while len(panels) < 3:
        panels.append(_Panel(kind="text", title="(no data)", options={"text": "surface missing"}))
    panels.append(
        _Panel(
            kind="line",
            title="value drift summary",
            xlabel="Episode e (vs e-1)",
            ylabel="mean |Δvalue|",
            options={"series": summary_series},
        )
    )
    meta = _base_meta(
        episode=f"{keys[0]}..{keys[-1]}",
        eta="η=1",
        i_slice=_i_slice_text(episodes[keys[-1]]),
        data_source=_describe_bank_source(getattr(episodes[keys[-1]], "diagnostics", None)),
        capital_weighted="no",
        n_samples=n_samples,
        extra={"episodes": [key for key in keys]},
    )
    return _render_grid(
        "V06", panels, rows=2, cols=2, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS["V06"], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=meta["episode"], eta="η=1", i_slice=meta["i_slice"],
            data_source=str(meta["data_source"]), capital_weighted="no", n_samples=n_samples,
        ),
    )


# ---------------------------------------------------------------------------
# P-series renderers
# ---------------------------------------------------------------------------


def render_p01(*, policy_results, output_dir, style=None, episode=None, annotations=None) -> FigureRecord:
    """P01: Grid bp0* surface."""
    return _value_series_figure(
        "P01", "bp0", "bp", policy_results=policy_results,
        output_dir=output_dir, style=style, episode=episode, annotations=annotations,
    )


def render_p02(*, policy_results, output_dir, style=None, episode=None, annotations=None) -> FigureRecord:
    """P02: Grid bpI* surface."""
    return _value_series_figure(
        "P02", "bpI", "bp", policy_results=policy_results,
        output_dir=output_dir, style=style, episode=episode, annotations=annotations,
    )


def render_p03(*, policy_results, output_dir, style=None, episode=None, annotations=None) -> FigureRecord:
    """P03: effective debt policy (executed ``b_next``)."""
    episode_key, result, reason = _select_episode(policy_results, episode)
    if result is None:
        return _skip("P03", reason or REASON_NO_POLICY_ROWS)
    rows = _pick_table(result, "policy_rows")
    required = {"b", "z", "eta", "b_next_mix", "bp_star_mix"}
    if rows is None or not required.issubset(rows.columns):
        return _skip("P03", REASON_NO_POLICY_ROWS)
    style = resolve_style(style, [result])

    b = _table_column(rows, "b")
    z = _table_column(rows, "z")
    eta = _table_column(rows, "eta")
    b_next = _table_column(rows, "b_next_mix")
    bp_star = _table_column(rows, "bp_star_mix")
    eta1 = eta > 0.5
    eta0 = ~eta1

    head_values = None
    if "bp0_head" in rows.columns:
        head_values = _table_column(rows, "bp0_head")
    panels = [
        _Panel(
            kind="scatter",
            title="b_next vs b (η=0 forces b_next=b)",
            xlabel="b current leverage",
            ylabel="b_next = η·bp* + (1-η)·b",
            options={
                "series": [
                    {"x": b[eta0], "y": b_next[eta0], "label": "η=0 (refi off)", "color": "tab:blue"},
                    {"x": b[eta1], "y": b_next[eta1], "label": "η=1 (refi active)", "color": "tab:orange"},
                ]
            },
        ),
        _Panel(
            kind="scatter",
            title="b_next vs z (productivity effect on executed leverage)",
            xlabel="z = firm productivity shock",
            ylabel="b_next",
            options={
                "series": [
                    {"x": z[eta0], "y": b_next[eta0], "label": "η=0", "color": "tab:blue"},
                    {"x": z[eta1], "y": b_next[eta1], "label": "η=1", "color": "tab:orange"},
                ],
                "diag": False,
            },
        ),
        _Panel(
            kind="scatter",
            title="Grid bp* vs head output bp_head (head is compatibility only)",
            xlabel="Grid bp* (formal policy)",
            ylabel="bp_head (compatibility output)",
            options={
                "series": [
                    {"x": bp_star[eta1],
                     "y": (head_values[eta1] if head_values is not None else bp_star[eta1]),
                     "label": "η=1"},
                ]
            },
        ),
    ]
    delta = b_next - b
    panels.append(
        _Panel(
            kind="line",
            title="Δb = b_next - b (by η)",
            xlabel="b current leverage",
            ylabel="Δb",
            options={
                "series": [
                    {"x": b[eta0], "y": delta[eta0], "label": "η=0 (must be 0)",
                     "marker": "o", "markersize": 2},
                    {"x": b[eta1], "y": delta[eta1], "label": "η=1",
                     "marker": "o", "markersize": 2},
                ]
            },
        )
    )

    n_samples = int(len(rows))
    eta0_error = float(np.nanmax(np.abs(delta[eta0]))) if eta0.any() else float("nan")
    meta = _base_meta(
        episode=episode_key,
        eta="η=0 / η=1",
        i_slice=_i_slice_text(result),
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="no",
        n_samples=n_samples,
        extra={
            "financing_gate": "b_next = eta_t * bp + (1 - eta_t) * b_t",
            "eta0_financing_gate_max_error": eta0_error,
            "formal_policy": "grid_bp (BPGridTeacher)",
        },
    )
    return _render_grid(
        "P03", panels, rows=2, cols=2, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS["P03"], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=episode_key, eta="η=0/η=1", i_slice=meta["i_slice"],
            data_source=str(meta["data_source"]), capital_weighted="no", n_samples=n_samples,
        ),
    )


def render_p04(*, policy_results, output_dir, style=None, episode=None, annotations=None) -> FigureRecord:
    """P04: Bellman objective vs candidate ``bp`` with the Q claim overlay."""
    episode_key, result, reason = _select_episode(policy_results, episode)
    if result is None:
        return _skip("P04", reason or REASON_NO_OBJECTIVE_SLICES)
    slices = _pick_table(result, "objective_slices")
    required = {"row_index", "branch", "bp_candidate", "bellman_rhs", "q_claim_issue"}
    if slices is None or slices.empty or not required.issubset(slices.columns):
        return _skip("P04", REASON_NO_OBJECTIVE_SLICES)
    style = resolve_style(style, [result])

    groups = list(slices.groupby(["row_index", "branch"], sort=True))
    panels = []
    n_samples = 0
    for (row_index, branch), group in groups[:4]:
        bp = _table_column(group, "bp_candidate")
        rhs = _table_column(group, "bellman_rhs")
        q_claim = _table_column(group, "q_claim_issue")
        cashflow = _table_column(group, "cashflow") if "cashflow" in group.columns else None
        continuation = _table_column(group, "continuation") if "continuation" in group.columns else None
        series = [{"x": bp, "y": rhs, "label": "Bellman RHS"}]
        if cashflow is not None:
            series.append({"x": bp, "y": cashflow, "label": "cashflow", "linestyle": "--"})
        if continuation is not None:
            series.append({"x": bp, "y": continuation, "label": "continuation", "linestyle": ":"})
        if q_claim is not None:
            series.append({"x": bp, "y": q_claim, "label": "Q_claim (issuance pricing)", "linestyle": "-."})
        eta_value = float(group["eta"].iloc[0]) if "eta" in group.columns else float("nan")
        panels.append(
            _Panel(
                kind="line",
                title=f"row {int(row_index)} | branch={branch} | η={eta_value:g}",
                xlabel="candidate next-period leverage bp",
                ylabel="objective value / issuance pricing",
                options={"series": series},
            )
        )
        n_samples += int(len(group))
    while len(panels) < 4:
        panels.append(_Panel(kind="text", title="(no data)", options={"text": "insufficient representative states"}))

    metrics = _result_metrics(result)
    flatness = float(metrics.get("objective_slice_eta0_flatness_max_abs", float("nan")))
    meta = _base_meta(
        episode=episode_key,
        eta="η=0 / η=1 (η=0 objective must be flat)",
        i_slice=_i_slice_text(result),
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="no",
        n_samples=n_samples,
        extra={"eta0_objective_flatness_max_abs": flatness,
               "representative_states": sorted({int(v) for v in slices["row_index"].tolist()})},
    )
    return _render_grid(
        "P04", panels, rows=2, cols=2, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS["P04"], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=episode_key, eta="η=0/η=1", i_slice=meta["i_slice"],
            data_source=str(meta["data_source"]), capital_weighted="no", n_samples=n_samples,
        ),
    )


# ---------------------------------------------------------------------------
# M-series renderers
# ---------------------------------------------------------------------------


def _regression_stats(calculated, forecast) -> str:
    mask = np.isfinite(calculated) & np.isfinite(forecast)
    if not mask.any():
        return "n=0"
    err = forecast[mask] - calculated[mask]
    denominator = float(np.sum((calculated[mask] - np.mean(calculated[mask])) ** 2))
    r2 = 1.0 - float(np.sum(err ** 2)) / denominator if denominator > 0 else float("nan")
    return (
        f"n={int(mask.sum())}  MAE={np.mean(np.abs(err)):.4g}  "
        f"Bias={np.mean(err):.4g}  R2={r2:.4g}"
    )


def render_m01(*, macro_results, output_dir, style=None, annotations=None) -> FigureRecord:
    """M01: FC1 one-step scatter."""
    episodes = _as_episode_map(macro_results)
    if not episodes:
        return _skip("M01", REASON_NO_MACRO_PAIRS)
    key = list(episodes)[-1]
    result = episodes[key]
    frame = _macro_pairs(result)
    if frame is None:
        return _skip("M01", REASON_NO_MACRO_PAIRS)
    style = resolve_style(style, [result])

    hatc_x = _table_column(frame, "hatc_t")
    hatc_y = _table_column(frame, "hatcf_t")
    lnk_x = _table_column(frame, "lnk_t")
    lnk_y = _table_column(frame, "lnkf_t")
    panels = [
        _Panel(
            kind="scatter",
            title="Hatcf_t vs Hatc_t",
            xlabel="Hatc_t realized (log(C/K))",
            ylabel="Hatcf_t belief forecast",
            options={"series": [{"x": hatc_x, "y": hatc_y, "label": "hatc"}]},
        ),
        _Panel(
            kind="scatter",
            title="LnKF_t vs LnK_t",
            xlabel="LnK_t realized (log ΣK)",
            ylabel="LnKF_t belief forecast",
            options={"series": [{"x": lnk_x, "y": lnk_y, "label": "lnk"}]},
        ),
        _Panel(
            kind="text",
            title="one-step forecast statistics",
            options={
                "text": (
                    "hatc: " + _regression_stats(hatc_x, hatc_y) + "\n"
                    "lnk:  " + _regression_stats(lnk_x, lnk_y)
                )
            },
        ),
        _Panel(
            kind="text",
            title="alignment semantics",
            options={
                "text": (
                    "hatcf_t/lnkf_t = t-1 forecasts of node t\n"
                    "hatc_t/lnk_t = node t realized aggregates\n"
                    "same-row alignment = shift=0 (one-step forecast)\n"
                    "pairing key (path,t,branch,economic_node_id)"
                )
            },
        ),
    ]
    n_samples = int(len(frame))
    meta = _base_meta(
        episode=key,
        eta="n/a",
        i_slice="n/a",
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="no",
        n_samples=n_samples,
    )
    return _render_grid(
        "M01", panels, rows=2, cols=2, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS["M01"], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=key, data_source=str(meta["data_source"]),
            capital_weighted="no", n_samples=n_samples,
        ),
    )


def _path_figure(
    figure_id: str,
    *,
    macro_results,
    output_dir,
    style=None,
    annotations=None,
    realized: str,
    forecast: str,
    ylabel: str,
    title: str,
) -> FigureRecord:
    episodes = _as_episode_map(macro_results)
    if not episodes:
        return _skip(figure_id, REASON_NO_MACRO_PAIRS)
    key = list(episodes)[-1]
    result = episodes[key]
    frame = _macro_pairs(result)
    if frame is None or not {"path", "t", realized, forecast}.issubset(frame.columns):
        return _skip(figure_id, REASON_NO_MACRO_PAIRS)
    style = resolve_style(style, [result])

    path_column = pd.to_numeric(frame["path"], errors="coerce")
    paths = sorted(path_column.dropna().unique())
    selected = paths[:6]
    panels = []
    n_samples = 0
    for path_id in selected:
        subset = frame.loc[path_column == path_id].sort_values("t", kind="mergesort")
        panels.append(
            _Panel(
                kind="line",
                title=f"{title} | path {int(path_id)}",
                xlabel="t (macro node)",
                ylabel=ylabel,
                options={
                    "series": [
                        {"x": _table_column(subset, "t"), "y": _table_column(subset, forecast),
                         "label": forecast, "linestyle": "--"},
                        {"x": _table_column(subset, "t"), "y": _table_column(subset, realized),
                         "label": realized},
                    ]
                },
            )
        )
        n_samples += int(len(subset))
    if not panels:
        return _skip(figure_id, REASON_NO_MACRO_PAIRS)
    while len(panels) < 6:
        panels.append(_Panel(kind="text", title="(no data)", options={"text": "insufficient paths"}))

    meta = _base_meta(
        episode=key,
        eta="n/a",
        i_slice="n/a",
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="no",
        n_samples=n_samples,
        extra={"paths": [int(value) for value in selected], "n_paths_total": int(len(paths))},
    )
    return _render_grid(
        figure_id, panels, rows=2, cols=3, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS[figure_id], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=key, data_source=str(meta["data_source"]),
            capital_weighted="no", n_samples=n_samples,
        ),
    )


def render_m02(*, macro_results, output_dir, style=None, annotations=None) -> FigureRecord:
    """M02: Hatcf vs Hatc paths."""
    return _path_figure(
        "M02", macro_results=macro_results, output_dir=output_dir, style=style,
        annotations=annotations, realized="hatc_t", forecast="hatcf_t",
        ylabel="Hatc / Hatcf", title="Hatcf vs Hatc",
    )


def render_m03(*, macro_results, output_dir, style=None, annotations=None) -> FigureRecord:
    """M03: LnKF vs LnK paths."""
    return _path_figure(
        "M03", macro_results=macro_results, output_dir=output_dir, style=style,
        annotations=annotations, realized="lnk_t", forecast="lnkf_t",
        ylabel="LnK / LnKF", title="LnKF vs LnK",
    )


def render_m04(*, macro_results, output_dir, style=None, annotations=None) -> FigureRecord:
    """M04: macro recursive forecast error."""
    episodes = _as_episode_map(macro_results)
    if not episodes:
        return _skip("M04", REASON_NO_RECURSION_TABLE)
    key = list(episodes)[-1]
    result = episodes[key]
    table = _pick_table(result, "fc1_recursion_errors")
    required = {"caliber", "variable", "horizon", "mae", "rmse", "bias"}
    if table is None or table.empty or not required.issubset(table.columns):
        return _skip("M04", REASON_NO_RECURSION_TABLE)
    style = resolve_style(style, [result])

    panels = []
    for variable, label in (("hatc", "hatc"), ("lnk", "lnk")):
        subset = table.loc[table["variable"].astype(str) == variable]
        series_mae = []
        series_bias = []
        for caliber, group in subset.groupby("caliber"):
            horizon = _table_column(group, "horizon")
            series_mae.append({"x": horizon, "y": _table_column(group, "mae"),
                               "label": f"{caliber} MAE"})
            series_mae.append({"x": horizon, "y": _table_column(group, "rmse"),
                               "label": f"{caliber} RMSE", "linestyle": "--"})
            series_bias.append({"x": horizon, "y": _table_column(group, "bias"),
                                "label": f"{caliber} bias"})
        panels.append(
            _Panel(
                kind="line",
                title=f"{label} recursion error vs horizon",
                xlabel="horizon h (recursion steps)",
                ylabel="error (MAE / RMSE)",
                options={"series": series_mae},
            )
        )
        panels.append(
            _Panel(
                kind="line",
                title=f"{label} recursion bias vs horizon",
                xlabel="horizon h",
                ylabel="bias (forecast - realized)",
                options={"series": series_bias},
            )
        )
    metrics = _result_metrics(result)
    lines = [
        f"{name} = {float(metrics[name]):.6g}"
        for name in sorted(metrics)
        if "drift_slope" in name or "out_of_training_range" in name or name.endswith("_diverged")
    ]
    panels.append(_Panel(kind="text", title="drift and out-of-training-range",
                         options={"text": "\n".join(lines[:18]) or "none"}))
    panels.append(
        _Panel(
            kind="text",
            title="two-caliber definitions",
            options={
                "text": (
                    "production_one_step: macro state computed from the parent node at every call\n"
                    "autonomous_recursive: fully autoregressive, error accumulates with h\n"
                    "the two calibers are reported separately, never merged"
                )
            },
        )
    )
    while len(panels) < 6:
        panels.append(_Panel(kind="text", title="(no data)", options={"text": "variable missing"}))

    n_samples = int(table["n"].sum()) if "n" in table.columns else int(len(table))
    meta = _base_meta(
        episode=key,
        eta="n/a",
        i_slice="n/a",
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="no",
        n_samples=n_samples,
        extra={"calibers": sorted(set(table["caliber"].astype(str)))},
    )
    return _render_grid(
        "M04", panels, rows=2, cols=3, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS["M04"], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=key, data_source=str(meta["data_source"]),
            capital_weighted="no", n_samples=n_samples,
        ),
    )


def render_m05(*, macro_results, output_dir, style=None, annotations=None) -> FigureRecord:
    """M05: SDF distribution and constraints."""
    episodes = _as_episode_map(macro_results)
    if not episodes:
        return _skip("M05", REASON_NO_SDF_TABLE)
    key = list(episodes)[-1]
    result = episodes[key]
    table = _pick_table(result, "sdf_macro_state_summary")
    metrics = _result_metrics(result)
    sdf_metrics = {name: value for name, value in metrics.items() if str(name).startswith("sdf_")}
    if table is None and not sdf_metrics:
        return _skip("M05", REASON_NO_SDF_TABLE)
    style = resolve_style(style, [result])

    panels = []
    if table is not None and not table.empty:
        labels = table["macro_state"].astype(str).tolist()
        panels.append(
            _Panel(kind="bar", title="E[M | macro state] (± standard error)",
                   xlabel="macro state", ylabel="M mean",
                   options={"labels": labels, "values": _table_column(table, "M_mean"),
                            "yerr": _table_column(table, "M_mean_se"), "rotation": 15})
        )
        panels.append(
            _Panel(kind="bar", title="normalized wealth Euler residual (± standard error)",
                   xlabel="macro state", ylabel="mean residual",
                   options={"labels": labels, "values": _table_column(table, "mean_residual"),
                            "yerr": _table_column(table, "se_residual"), "rotation": 15})
        )
        panels.append(
            _Panel(kind="bar", title="sample count by macro state",
                   xlabel="macro state", ylabel="n_parents",
                   options={"labels": labels, "values": _table_column(table, "n_parents"),
                            "rotation": 15})
        )
    else:
        for title in ("macro-state decomposition", "residual decomposition", "sample size"):
            panels.append(_Panel(kind="text", title=title,
                                 options={"text": "sdf_macro_state_summary missing"}))

    labels = []
    values = []
    for name in ("sdf_g_mean_low", "sdf_g_mean_high", "sdf_g_var_high", "sdf_g_max"):
        if name in sdf_metrics:
            labels.append(name.replace("sdf_", ""))
            values.append(float(sdf_metrics[name]))
    panels.append(
        _Panel(kind="bar", title="moment-constraint violations (g_*, should be ≤0)",
               xlabel="constraint", ylabel="violation",
               options={"labels": labels, "values": np.asarray(values, dtype=np.float64),
                        "rotation": 15})
    )
    lines = []
    for name in (
        "sdf_M_mean", "sdf_M_mean_se", "sdf_M_var", "sdf_M_std",
        "sdf_signed_aio_estimate", "sdf_signed_aio_se", "sdf_signed_aio_negative_share",
        "sdf_population_target_estimate", "sdf_population_target_se",
        "sdf_population_target_t_stat", "sdf_nonfinite_ratio",
        "sdf_log_R_clip_share", "sdf_valid_parent_ratio",
    ):
        if name in sdf_metrics:
            lines.append(f"{name} = {float(sdf_metrics[name]):.6g}")
    panels.append(_Panel(kind="text", title="SDF distribution and numerical safety",
                         options={"text": "\n".join(lines) or "none"}))
    panels.append(
        _Panel(
            kind="text",
            title="interpretation guardrails",
            options={
                "text": (
                    "negative sample mean of Signed AiO ≠ negative population conditional moment\n"
                    "population targets are reported as sdf_population_target_*\n"
                    "constraint violations and AiO are reported separately\n"
                    "M caliber: rawM vs trainM(clipped) must be declared"
                )
            },
        )
    )
    while len(panels) < 6:
        panels.append(_Panel(kind="text", title="(no data)", options={"text": "panel data missing"}))

    n_samples = (
        int(table["n_parents"].sum())
        if table is not None and "n_parents" in table.columns
        else len(sdf_metrics)
    )
    meta = _base_meta(
        episode=key,
        eta="n/a",
        i_slice="n/a",
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="no",
        n_samples=n_samples,
        extra={"sdf_metric_count": len(sdf_metrics)},
    )
    return _render_grid(
        "M05", panels, rows=2, cols=3, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS["M05"], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=key, data_source=str(meta["data_source"]),
            capital_weighted="no", n_samples=n_samples,
        ),
    )


def render_m06(*, sim_results, output_dir, style=None, annotations=None) -> FigureRecord:
    """M06: C/K, K, C, Y, I over time."""
    episodes = _as_episode_map(sim_results)
    if not episodes:
        return _skip("M06", REASON_NO_MACRO_PANEL)
    key = list(episodes)[-1]
    result = episodes[key]
    macro_panel = _pick_table(result, "macro_panel")
    if macro_panel is None or macro_panel.empty or not {"t", "K", "C"}.issubset(macro_panel.columns):
        return _skip("M06", REASON_NO_MACRO_PANEL)
    style = resolve_style(style, [result])

    frame = macro_panel
    if {"path", "branch"}.issubset(frame.columns):
        branch = pd.to_numeric(frame["branch"], errors="coerce")
        parent = frame.loc[branch < 0]
        if not parent.empty:
            frame = parent
        else:
            child = frame.loc[branch == 0]
            if not child.empty:
                frame = child
    frame = frame.sort_values("t", kind="mergesort")

    def _aggregate(values):
        if values is None:
            return None, None
        grouped = pd.DataFrame({"t": _table_column(frame, "t"), "v": values}).groupby(
            "t", sort=True
        )["v"].mean()
        return grouped.index.to_numpy(dtype=np.float64), grouped.to_numpy(dtype=np.float64)

    t_k, k_series = _aggregate(_table_column(frame, "K"))
    t_c, c_series = _aggregate(_table_column(frame, "C"))
    t_h, hatc_series = _aggregate(_table_column(frame, "Hatc"))
    ck_series = np.exp(hatc_series) if hatc_series is not None else None

    panels = []
    if ck_series is not None and t_h is not None:
        panels.append(
            _Panel(kind="line", title="C/K (= exp(Hatc), Hatc=log(C/K))",
                   xlabel="t (simulation period)", ylabel="C/K",
                   options={"series": [{"x": t_h, "y": ck_series, "label": "C/K"}]})
        )
    if k_series is not None:
        panels.append(
            _Panel(kind="line", title="total capital K", xlabel="t", ylabel="K = Σ K_i",
                   options={"series": [{"x": t_k, "y": k_series, "label": "K"}]})
        )
    if c_series is not None:
        panels.append(
            _Panel(kind="line", title="total consumption C", xlabel="t", ylabel="C",
                   options={"series": [{"x": t_c, "y": c_series, "label": "C"}]})
        )

    firm_panel = _sim_firm_panel(result)
    if firm_panel is not None and {"t", "Y"}.issubset(firm_panel.columns):
        grouped = pd.DataFrame(
            {"t": _table_column(firm_panel, "t"), "v": _table_column(firm_panel, "Y")}
        ).groupby("t", sort=True)["v"].sum()
        panels.append(
            _Panel(kind="line", title="total output Y", xlabel="t", ylabel="Y = Σ Y_i",
                   options={"series": [{"x": grouped.index.to_numpy(dtype=np.float64),
                                        "y": grouped.to_numpy(dtype=np.float64), "label": "Y"}]})
        )
    else:
        panels.append(_Panel(kind="text", title="total output Y",
                             options={"text": "firm_panel has no Y column (panel omitted)"}))

    drawn_investment = False
    for column, label in (("I_total", "total investment I"), ("I_oper", "operating investment"), ("I_entry", "entry investment")):
        if column in frame.columns:
            t_i, values = _aggregate(_table_column(frame, column))
            if values is not None and t_i is not None:
                panels.append(
                    _Panel(kind="line", title=label, xlabel="t", ylabel=column,
                           options={"series": [{"x": t_i, "y": values, "label": column}]})
                )
                drawn_investment = True
                break
    if not drawn_investment:
        panels.append(_Panel(kind="text", title="investment I",
                             options={"text": "macro_panel has no I_* columns (panel omitted)"}))

    while len(panels) < 6:
        panels.append(_Panel(kind="text", title="(no data)", options={"text": "aggregate missing"}))

    n_samples = int(len(frame))
    meta = _base_meta(
        episode=key,
        eta="n/a (simulation aggregates)",
        i_slice="n/a",
        data_source=_describe_bank_source(getattr(result, "diagnostics", None)),
        capital_weighted="aggregate caliber (Σ)",
        extinct_paths=_describe_extinct(result),
        n_samples=n_samples,
        extra={"aggregation": "sum/mean across firms within (path,t), then averaged across paths",
               "view_filter": "branch<0 preferred (parent view), else branch==0"},
    )
    return _render_grid(
        "M06", panels, rows=2, cols=3, output_dir=Path(output_dir), style=style,
        caption=CAPTIONS["M06"], meta=meta, n_samples=n_samples,
        annotations=_annotation_dict(
            annotations, episode=key, data_source=str(meta["data_source"]),
            capital_weighted="aggregate caliber", extinct_paths=str(meta["extinct_paths"]),
            n_samples=n_samples,
        ),
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

_SINGLE_RENDERERS = (
    ("F01", render_f01, ("policy_results",)),
    ("F02", render_f02, ("policy_results",)),
    ("F03", render_f03, ("policy_results",)),
    ("F04", render_f04, ("policy_results",)),
    ("F05", render_f05, ("bellman_results",)),
    ("F06", render_f06, ("policy_results", "comparison")),
    ("F07", render_f07, ("macro_results",)),
    ("F08", render_f08, ("macro_results",)),
    ("F09", render_f09, ("sim_results",)),
    ("F10", render_f10, ("sim_results",)),
    ("F11", render_f11_waterfall, ("sim_results",)),
    ("F12", render_f12,
     ("bellman_results", "policy_results", "macro_results", "sim_results", "comparison")),
    ("V01", render_v01, ("policy_results",)),
    ("V02", render_v02, ("policy_results",)),
    ("V03", render_v03, ("policy_results",)),
    ("V04", render_v04, ("policy_results",)),
    ("V05", render_v05, ("policy_results",)),
    ("V06", render_v06, ("policy_results",)),
    ("P01", render_p01, ("policy_results",)),
    ("P02", render_p02, ("policy_results",)),
    ("P03", render_p03, ("policy_results",)),
    ("P04", render_p04, ("policy_results",)),
    ("M01", render_m01, ("macro_results",)),
    ("M02", render_m02, ("macro_results",)),
    ("M03", render_m03, ("macro_results",)),
    ("M04", render_m04, ("macro_results",)),
    ("M05", render_m05, ("macro_results",)),
    ("M06", render_m06, ("sim_results",)),
)


def render_all_figures(
    *,
    bellman_results=None,
    policy_results=None,
    macro_results=None,
    sim_results=None,
    comparison=None,
    bank=None,
    output_dir,
    style=None,
) -> FiguresManifest:
    """Render the full figure set into ``output_dir``.

    Every figure id in :data:`FIGURE_IDS` gets a :class:`FigureRecord`: ``ok``
    with a file on disk, or ``skipped`` with a stable ``missing_reason`` and no
    file.  Multi-episode inputs share one horizontal episode axis and one
    ``b``/``z``/colour scale.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for source in (bellman_results, policy_results, macro_results, sim_results):
        all_results.extend(_as_episode_map(source).values())
    shared_style = resolve_style(style, all_results)

    annotations = {
        "Macro Anchor": _describe_macro_anchor(bank),
        "Data source": _describe_bank_source(bank),
        "Capital weighted": "see each figure's caliber",
        "Extinct paths": _describe_extinct(sim_results),
    }

    inputs = {
        "bellman_results": bellman_results,
        "policy_results": policy_results,
        "macro_results": macro_results,
        "sim_results": sim_results,
        "comparison": comparison,
    }
    manifest = FiguresManifest(output_dir=str(output_dir), style=shared_style)
    for figure_id, renderer, wanted in _SINGLE_RENDERERS:
        kwargs = {"output_dir": output_dir, "style": shared_style}
        for name in wanted:
            kwargs[name] = inputs.get(name)
        kwargs["annotations"] = annotations
        try:
            record = renderer(**kwargs)
        except Exception as exc:  # never fake a successful figure on error
            record = _skip(
                figure_id,
                f"RENDER_ERROR:{type(exc).__name__}",
                {"error": str(exc)[:400]},
            )
            record.caption = CAPTIONS.get(figure_id, "")
        record.meta.setdefault("n_samples", record.n_samples)
        for _key, _value in _style_meta(shared_style).items():
            record.meta.setdefault(_key, _value)
        manifest.figures[figure_id] = record
    return manifest


__all__ = [
    "CANONICAL_3D_NAMES",
    "CAPTIONS",
    "FIGURE_FILENAMES",
    "FIGURE_IDS",
    "FigureRecord",
    "FigureStyle",
    "FiguresManifest",
    "grid_axes",
    "render_all_figures",
    "render_f01",
    "render_f02",
    "render_f03",
    "render_f04",
    "render_f05",
    "render_f06",
    "render_f07",
    "render_f08",
    "render_f09",
    "render_f10",
    "render_f11_waterfall",
    "render_f12",
    "render_m01",
    "render_m02",
    "render_m03",
    "render_m04",
    "render_m05",
    "render_m06",
    "render_p01",
    "render_p02",
    "render_p03",
    "render_p04",
    "render_v01",
    "render_v02",
    "render_v03",
    "render_v04",
    "render_v05",
    "render_v06",
    "resolve_style",
]
