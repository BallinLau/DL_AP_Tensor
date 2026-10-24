"""Cross-Episode comparison and convergence gates (Outer Fixed Point Dashboard).

This module is the cross-episode layer of the equilibrium convergence
evaluator: it consumes one :class:`EpisodeEvalRecord` per outer episode (the
checkpoint semantic record, the four per-episode evaluator results and the
EvalBank reference they were computed on) and produces

* the **six metric groups** of the dashboard on a unified episode axis
  (``firm_bellman`` / ``value_drift`` / ``policy_drift`` / ``macro`` /
  ``simulation`` / ``distribution``), always keeping absolute errors and sample
  sizes next to every relative figure;
* a **comparability pre-check** per episode pair
  (:func:`check_semantic_compatibility` + EvalBank file hashes + grid
  consistency via :func:`evaluation.convergence_artifacts.validate_grid_comparability`);
  an incomparable pair is never silently compared -- the affected metric groups
  are flagged ``NOT_COMPARABLE`` with an explicit reason code;
* the **convergence decision** (Prompt section 10): four gate groups
  (hard economic / firm / macro / outer), each gate in
  ``PASS / FAIL / INSUFFICIENT_EVIDENCE / THRESHOLD_NOT_CONFIGURED`` with a
  machine-readable ``reason_code``, and the ``Converged Candidate`` decision
  which requires *three consecutive comparable outer episodes* satisfying every
  necessary condition;
* a **threshold calibration report framework** (:func:`build_threshold_calibration`)
  that derives suggested intervals from the numerical-error economic scale, the
  historical residual distribution on the fixed EvalBank, multi-seed Monte Carlo
  spread and a :func:`path_clustered_bootstrap` confidence interval -- it never
  picks a threshold value by itself (``recommended_threshold`` is always ``null``).

Threshold discipline (the hard rule of Prompt section 10): **no threshold is
invented here**.  :class:`ThresholdConfig` is JSON-serializable and every gate
defaults to ``null`` (= unconfigured); an unconfigured gate is judged
``THRESHOLD_NOT_CONFIGURED`` and can never turn into ``PASS`` because a loss
"looks small".

Everything in this module is bookkeeping over already-computed evaluator
results: no model is loaded, no simulation is run, no parameter is touched.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from evaluation.convergence_artifacts import validate_grid_comparability

from .checkpoint_io import EvalCheckpoint, check_semantic_compatibility
from .firm_bellman import BellmanEvalResult, residual_stat_block
from .firm_policy import PolicyEvalResult, compare_policy_across_episodes
from .macro_sdf import MacroSdfEvalResult
from .sim_eval import SimEvalResult, compare_fixed_shock_rollouts, wasserstein_1


# ---------------------------------------------------------------------------
# status / reason codes
# ---------------------------------------------------------------------------

STATUS_OK = "OK"
STATUS_NOT_EVALUATED = "NOT_EVALUATED"
STATUS_NOT_COMPARABLE = "NOT_COMPARABLE"
STATUS_MISSING = "MISSING"

REASON_NO_PREVIOUS_EPISODE = "NO_PREVIOUS_EPISODE"
REASON_NO_RECORD_FOR_EPISODE = "NO_RECORD_FOR_EPISODE"
REASON_MISSING_BELLMAN_RESULT = "MISSING_BELLMAN_RESULT"
REASON_MISSING_POLICY_RESULT = "MISSING_POLICY_RESULT"
REASON_MISSING_MACRO_RESULT = "MISSING_MACRO_RESULT"
REASON_MISSING_SIMULATION_RESULT = "MISSING_SIMULATION_RESULT"
REASON_NON_CONSECUTIVE_EPISODE_GAP = "NON_CONSECUTIVE_EPISODE_GAP"
REASON_SEMANTIC_MISMATCH = "SEMANTIC_MISMATCH"
REASON_BANK_FILE_HASH_MISMATCH = "BANK_FILE_HASH_MISMATCH"
REASON_BANK_CONTENT_HASH_MISMATCH = "BANK_CONTENT_HASH_MISMATCH"
REASON_FIXED_STATE_MISMATCH = "FIXED_STATE_MISMATCH"
REASON_GRID_MISMATCH = "GRID_MISMATCH"
REASON_EVALUATED = "EVALUATED"

GATE_PASS = "PASS"
GATE_FAIL = "FAIL"
GATE_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
GATE_NOT_CONFIGURED = "THRESHOLD_NOT_CONFIGURED"

GATE_REASON_WITHIN_THRESHOLD = "PASS_WITHIN_THRESHOLD"
GATE_REASON_ABOVE_MINIMUM = "PASS_ABOVE_MINIMUM"
GATE_REASON_FAIL_ABOVE = "FAIL_ABOVE_THRESHOLD"
GATE_REASON_FAIL_BELOW = "FAIL_BELOW_MINIMUM"
GATE_REASON_THRESHOLD_NOT_CONFIGURED = "THRESHOLD_NOT_CONFIGURED"
GATE_REASON_METRIC_UNAVAILABLE = "METRIC_UNAVAILABLE"
GATE_REASON_METRIC_NONFINITE = "METRIC_NONFINITE"
GATE_REASON_GROUP_NOT_COMPARABLE = "GROUP_NOT_COMPARABLE"
REASON_NO_PREVIOUS_FOR_DRIFT_GATE = "NO_PREVIOUS_EPISODE_FOR_DRIFT_GATE"

CONVERGED_CANDIDATE = "CONVERGED_CANDIDATE"
NOT_CONVERGED = "NOT_CONVERGED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

# Metric-table groups on the unified episode axis.  ``comparability`` is a
# seventh, gate-only table with the pre-check evidence.
METRIC_GROUPS: Tuple[str, ...] = (
    "firm_bellman",
    "value_drift",
    "policy_drift",
    "macro",
    "simulation",
    "distribution",
)
COMPARABILITY_GROUP = "comparability"
ALL_TABLES: Tuple[str, ...] = (COMPARABILITY_GROUP,) + METRIC_GROUPS


# Value-drift quantities on the fixed evaluation state set A of the EvalBank
# parent rows (columns of ``BellmanEvalResult.tables["rows"]``).
VALUE_DRIFT_QUANTITIES: Tuple[str, ...] = (
    "p0_physical",
    "pi_physical",
    "p0_normalized",
    "pi_normalized",
    "P",
    "Q_claim",
)
VALUE_DRIFT_SURFACE_KEYS: Dict[str, str] = {
    "p0_physical": "P0_physical",
    "pi_physical": "PI_physical",
    "p0_normalized": "P0_normalized",
    "pi_normalized": "PI_normalized",
    "P": "P",
    "Q_claim": "Q_claim",
}


# ---------------------------------------------------------------------------
# input / output containers
# ---------------------------------------------------------------------------


@dataclass
class EpisodeEvalRecord:
    """Everything one outer episode contributes to the cross-episode dashboard.

    * ``episode`` -- outer episode id (the episode axis value).
    * ``semantic`` -- the checkpoint semantic record
      (:func:`evaluation.convergence.checkpoint_io.semantic_record`).
    * ``bellman`` / ``policy`` / ``macro`` / ``simulation`` -- the per-episode
      evaluator results; any of them may be ``None`` (the group row then carries
      an explicit ``MISSING_*_RESULT`` reason code).
    * ``bank_ref`` -- the fixed EvalBank reference the results were computed on:
      ``bank_id`` / ``bank_version`` / ``bank_file_sha256`` / ``bank_content_sha256``
      / ``parent_states_sha256`` / ``grid_metadata`` (the nested mapping consumed
      by :func:`evaluation.convergence_artifacts.validate_grid_comparability`).
    * ``checkpoint`` -- optional live :class:`EvalCheckpoint`; when present the
      semantic pre-check goes through
      :func:`check_semantic_compatibility`, otherwise the recorded semantic
      dicts are compared field-by-field.
    * ``seed`` -- optional Monte-Carlo seed of the evaluation (used by the
      threshold calibration's multi-seed basis).
    """

    episode: int
    semantic: Dict[str, Any]
    bellman: Optional[BellmanEvalResult] = None
    policy: Optional[PolicyEvalResult] = None
    macro: Optional[MacroSdfEvalResult] = None
    simulation: Optional[SimEvalResult] = None
    bank_ref: Dict[str, Any] = field(default_factory=dict)
    checkpoint: Optional[EvalCheckpoint] = None
    seed: Optional[int] = None


@dataclass
class ComparisonResult:
    """Output of :func:`compare_episodes`."""

    metrics: Dict[str, pd.DataFrame]
    convergence_status: Dict[str, Any]
    diagnostics: Dict[str, Any]


# ---------------------------------------------------------------------------
# thresholds (JSON-serializable, everything defaults to null)
# ---------------------------------------------------------------------------


@dataclass
class GateSpec:
    """Static specification of one convergence gate.

    ``direction="max"``: the gate passes when ``value <= threshold``.
    ``direction="min"``: the gate passes when ``value >= threshold``.
    ``metric_key`` is read from the episode row of ``table``; alternatively
    ``metric_pattern`` (regex over the table's column names) is reduced with
    ``reducer``.  ``table="*"`` searches every metric table.
    ``requires_predecessor`` marks drift gates: without a previous comparable
    episode they are not applicable and are excluded from that episode's
    necessary conditions (never silently passed).
    """

    gate_id: str
    group: str
    table: str
    direction: str
    metric_key: Optional[str] = None
    metric_pattern: Optional[str] = None
    reducer: str = "max"
    required: bool = True
    requires_predecessor: bool = False
    description: str = ""


GATE_SPECS: Dict[str, GateSpec] = {
    spec.gate_id: spec
    for spec in (
        # ---- Hard Economic Gates ----------------------------------------
        GateSpec(
            gate_id="hard_resource_ledger_residual",
            group="hard_economic",
            table="simulation",
            direction="max",
            metric_pattern=r"(^|_)waterfall_identity_residual_abs_max$",
            reducer="max",
            description="capital/resource ledger accounting residual (absolute)",
        ),
        GateSpec(
            gate_id="hard_capital_transition_validity",
            group="hard_economic",
            table="simulation",
            direction="max",
            metric_pattern=r"waterfall_(node_id_continuity_violations|end_node_id_mismatch_count)$",
            reducer="max",
            description="capital timing / transition validity violations",
        ),
        GateSpec(
            gate_id="hard_consumption_feasibility",
            group="hard_economic",
            table="simulation",
            direction="min",
            metric_key="sim_resource_feasible_share_min",
            description="share of nodes with a feasible consumption/resource account",
        ),
        GateSpec(
            gate_id="hard_nonfinite",
            group="hard_economic",
            table="*",
            direction="max",
            metric_pattern="nonfinite",
            reducer="max",
            description="nonfinite share/ratio anywhere in the dashboard",
        ),
        GateSpec(
            gate_id="hard_q_semantic_consistency",
            group="hard_economic",
            table=COMPARABILITY_GROUP,
            direction="max",
            metric_key="q_semantic_diff_count",
            description="cross-episode Q-caliber semantic inconsistencies",
        ),
        GateSpec(
            gate_id="hard_sim_path_completeness",
            group="hard_economic",
            table="simulation",
            direction="min",
            metric_key="sim_completed_horizon_min",
            description="shortest completed simulation horizon across rollout variants",
        ),
        # ---- Firm Gates -------------------------------------------------
        GateSpec(
            gate_id="firm_p_bellman_residual",
            group="firm",
            table="firm_bellman",
            direction="max",
            metric_pattern=r"^fresh_p(0|i)_normalized_mae$",
            reducer="max",
            description="P0/PI fresh-greedy Bellman residual (normalized MAE)",
        ),
        GateSpec(
            gate_id="firm_q_bellman_residual",
            group="firm",
            table="firm_bellman",
            direction="max",
            metric_pattern=r"^q_.*_claim_bellman_trainM_mae$",
            reducer="max",
            description="Q claim Bellman residual across Q regions (trainM)",
        ),
        GateSpec(
            gate_id="firm_policy_stability",
            group="firm",
            table="policy_drift",
            direction="max",
            metric_key="policy_flip_rate",
            requires_predecessor=True,
            description="policy flip rate on the fixed bank states",
        ),
        GateSpec(
            gate_id="firm_value_drift",
            group="firm",
            table="value_drift",
            direction="max",
            metric_pattern=r"^(p0_normalized|pi_normalized|P|Q_claim)_relative_mae$",
            reducer="max",
            requires_predecessor=True,
            description="relative-scale value drift D_e on the fixed state set",
        ),
        GateSpec(
            gate_id="firm_bp_star_drift",
            group="firm",
            table="policy_drift",
            direction="max",
            metric_key="bp_star_drift_max",
            requires_predecessor=True,
            description="Grid-BP drift |bp*_e - bp*_{e-1}| max over eta=1 rows",
        ),
        # ---- Macro Gates ------------------------------------------------
        GateSpec(
            gate_id="macro_fc1_one_step",
            group="macro",
            table="macro",
            direction="max",
            metric_pattern=r"^fc1_one_step_.*_(rmse|mae)$",
            reducer="max",
            description="FC1 one-step forecast error (belief vs realized)",
        ),
        GateSpec(
            gate_id="macro_fc1_recursive",
            group="macro",
            table="macro",
            direction="max",
            metric_pattern=r"^fc1_rec_.*_rmse$",
            reducer="max",
            description="FC1 recursive (autonomous / production) forecast error",
        ),
        GateSpec(
            gate_id="macro_sdf_euler_moments",
            group="macro",
            table="macro",
            direction="max",
            metric_pattern=r"^(sdf_g_max|sdf_population_target_abs|sdf_nonfinite_ratio)$",
            reducer="max",
            description="SDF Euler/moment-constraint violations",
        ),
        # ---- Outer Gates -----------------------------------------------
        GateSpec(
            gate_id="outer_fixed_shock_rollout",
            group="outer",
            table="simulation",
            direction="max",
            metric_key="fixed_shock_rollout_drift_abs_max",
            requires_predecessor=True,
            description="fixed-shock rollout drift across episodes (Bank E)",
        ),
        GateSpec(
            gate_id="outer_macro_trajectory_drift",
            group="outer",
            table="macro",
            direction="max",
            metric_key="macro_trajectory_drift_abs_max",
            requires_predecessor=True,
            description="macro trajectory metric drift across episodes",
        ),
        GateSpec(
            gate_id="outer_distribution_drift",
            group="outer",
            table="distribution",
            direction="max",
            metric_key="distribution_drift_abs_max",
            requires_predecessor=True,
            description="Wasserstein drift of the b/z distributions across episodes",
        ),
        GateSpec(
            gate_id="outer_extinct_paths",
            group="outer",
            table="simulation",
            direction="max",
            metric_key="sim_n_extinct_paths_max",
            description="extinct paths in the fixed-shock rollouts",
        ),
    )
}

GATE_GROUPS: Tuple[str, ...] = ("hard_economic", "firm", "macro", "outer")


@dataclass
class ThresholdConfig:
    """Per-gate thresholds; every gate defaults to ``null`` (unconfigured).

    This object is the *only* place a convergence threshold can come from and it
    never contains a default value: a gate whose threshold is ``None`` is judged
    ``THRESHOLD_NOT_CONFIGURED``.  ``to_dict`` / ``from_dict`` round-trip
    through JSON.
    """

    thresholds: Dict[str, Optional[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for gate_id, value in dict(self.thresholds).items():
            if gate_id not in GATE_SPECS:
                raise ValueError(f"unknown gate id in ThresholdConfig: {gate_id!r}")
            if value is None:
                continue
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"threshold for {gate_id} must be finite or null, got {value!r}")
            self.thresholds[gate_id] = number

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "ThresholdConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("ThresholdConfig payload must be a mapping of gate_id -> float|null")
        return cls(thresholds={str(key): value for key, value in payload.items()})

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {gate_id: self.thresholds.get(gate_id) for gate_id in sorted(GATE_SPECS)}

    def get(self, gate_id: str) -> Optional[float]:
        return self.thresholds.get(gate_id)

    def missing_required_gate_ids(self) -> List[str]:
        return sorted(
            gate_id
            for gate_id, spec in GATE_SPECS.items()
            if spec.required and self.thresholds.get(gate_id) is None
        )

    def is_complete(self, *, required_only: bool = True) -> bool:
        specs = [
            spec for spec in GATE_SPECS.values() if (spec.required or not required_only)
        ]
        return all(self.thresholds.get(spec.gate_id) is not None for spec in specs)


# ---------------------------------------------------------------------------
# path-clustered bootstrap
# ---------------------------------------------------------------------------


def _resolve_stat(stat: Union[str, Callable[[np.ndarray], float]]) -> Callable[[np.ndarray], float]:
    if callable(stat):
        return stat
    name = str(stat)
    if name == "mean":
        return lambda values: float(np.mean(values))
    if name == "median":
        return lambda values: float(np.median(values))
    raise ValueError(f"stat must be 'mean', 'median' or a callable, got {stat!r}")


def path_clustered_bootstrap(
    values: Any,
    path_ids: Any,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    stat: Union[str, Callable[[np.ndarray], float]] = "mean",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Path-clustered bootstrap confidence interval of a statistic.

    Rows sharing one ``path_id`` form a cluster (an autocorrelated simulation
    path / episode); clusters are resampled with replacement and every row of a
    drawn cluster enters the statistic.  This is the CI engine used by
    :func:`build_threshold_calibration` and available for metric CIs.

    Returns a JSON-safe dict with the point estimate, the percentile interval
    ``[ci_low, ci_high]`` at level ``1 - alpha``, the bootstrap mean/std and the
    cluster/observation counts.  Degenerate inputs return NaN estimates with a
    ``reason_code`` instead of raising.
    """
    if int(n_boot) < 1:
        raise ValueError("n_boot must be positive")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    stat_fn = _resolve_stat(stat)
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    ids = np.asarray(path_ids).reshape(-1)
    if array.shape[0] != ids.shape[0]:
        raise ValueError(
            f"values and path_ids must have equal length, got {array.shape[0]} vs {ids.shape[0]}"
        )
    finite = np.isfinite(array)
    array = array[finite]
    ids = ids[finite]
    n_obs = int(array.size)
    unique_paths = np.unique(ids)
    n_paths = int(unique_paths.size)
    base = {
        "stat": stat if isinstance(stat, str) else getattr(stat, "__name__", "callable"),
        "alpha": float(alpha),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "n_obs": n_obs,
        "n_paths": n_paths,
    }
    if n_obs == 0 or n_paths == 0:
        return {
            **base,
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "bootstrap_mean": float("nan"),
            "bootstrap_std": float("nan"),
            "reason_code": "NO_FINITE_OBSERVATIONS",
        }

    rows_by_path: Dict[Any, np.ndarray] = {
        path: array[ids == path] for path in unique_paths
    }
    estimate = float(stat_fn(array))
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_boot), dtype=np.float64)
    path_list = list(unique_paths)
    for draw in range(int(n_boot)):
        picked = rng.choice(len(path_list), size=len(path_list), replace=True)
        sample = np.concatenate([rows_by_path[path_list[index]] for index in picked])
        draws[draw] = float(stat_fn(sample))
    finite_draws = draws[np.isfinite(draws)]
    if finite_draws.size == 0:
        ci_low = ci_high = bootstrap_mean = bootstrap_std = float("nan")
    else:
        ci_low = float(np.quantile(finite_draws, float(alpha) / 2.0))
        ci_high = float(np.quantile(finite_draws, 1.0 - float(alpha) / 2.0))
        bootstrap_mean = float(finite_draws.mean())
        bootstrap_std = float(finite_draws.std(ddof=0)) if finite_draws.size > 1 else 0.0
    return {
        **base,
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_mean": bootstrap_mean,
        "bootstrap_std": bootstrap_std,
        "reason_code": "OK",
    }


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _nan() -> float:
    return float("nan")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _to_float(value: Any) -> float:
    if _is_number(value):
        return float(value)
    return _nan()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    return value


def _finite_concat(values: Sequence[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(values)


DIST_KEY_RE = re.compile(r"(^|_)dist_")


def _metric_in_group(key: str, group: str) -> bool:
    """Classify one evaluator metric key into one dashboard metric group."""
    if group == "distribution":
        return bool(DIST_KEY_RE.search(key))
    if group == "simulation":
        return (key.startswith("sim_") or key.startswith("waterfall_")) and not bool(DIST_KEY_RE.search(key))
    if group == "firm_bellman":
        return key.startswith(("fresh_", "frozen_", "q_", "parity_"))
    return key.startswith(("fc1_", "sdf_"))


def _reduce(values: np.ndarray, reducer: str) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return _nan()
    if reducer == "max":
        return float(finite.max())
    if reducer == "min":
        return float(finite.min())
    if reducer == "mean":
        return float(finite.mean())
    raise ValueError(f"unknown reducer: {reducer!r}")


# ---------------------------------------------------------------------------
# comparability pre-check
# ---------------------------------------------------------------------------


def _semantic_flat_view(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Flat per-field semantic view (mirrors ``checkpoint_io._compatibility_view``)."""
    view: Dict[str, Any] = {
        "q_parameterization": record.get("q_parameterization"),
        "q_parameterization_source": record.get("q_parameterization_source"),
        "q_recovery_normalization_mode": record.get("q_recovery_normalization_mode"),
        "policy_value_model_spec": record.get("policy_value_model_spec"),
    }
    value_parameterization = record.get("value_parameterization") or {}
    for key in ("mode", "scale_formula", "bellman_normalization", "log_max"):
        view[f"value_parameterization.{key}"] = value_parameterization.get(key)
    economic_config = record.get("economic_config") or {}
    for name in sorted(economic_config.keys()):
        view[f"economic_config.{name}"] = economic_config[name]
    return view


def _semantic_diffs(records: Sequence[EpisodeEvalRecord]) -> Dict[str, Dict[str, Any]]:
    """Per-field semantic diffs across the records (semantic incompatibility)."""
    if len(records) < 2:
        return {}
    checkpoints = [record.checkpoint for record in records]
    if all(cp is not None for cp in checkpoints):
        result = check_semantic_compatibility([cp for cp in checkpoints if cp is not None])
        return dict(result.get("diffs") or {})
    views = [_semantic_flat_view(record.semantic) for record in records]
    labels = [f"episode_{record.episode}" for record in records]
    diffs: Dict[str, Dict[str, Any]] = {}
    for field_name in sorted({key for view in views for key in view}):
        values = {labels[i]: views[i].get(field_name) for i in range(len(views))}
        serialized = {
            label: repr(value)
            if not isinstance(value, (bool, int, float, str, type(None)))
            else value
            for label, value in values.items()
        }
        if len({repr(value) for value in serialized.values()}) > 1:
            diffs[field_name] = values
    return diffs


def _pair_comparability(
    previous: EpisodeEvalRecord,
    current: EpisodeEvalRecord,
) -> Dict[str, Any]:
    """Comparability verdict for one episode pair; conflicts veto comparison."""
    reasons: List[str] = []
    unverified: List[str] = []

    semantic_diffs = _semantic_diffs([previous, current])
    q_semantic_diffs = sorted(
        name for name in semantic_diffs if "q_parameterization" in name or "q_recovery" in name
    )
    if semantic_diffs:
        reasons.append(REASON_SEMANTIC_MISMATCH)

    def _hash_check(field_name: str, reason_code: str) -> Optional[bool]:
        old = (previous.bank_ref or {}).get(field_name)
        new = (current.bank_ref or {}).get(field_name)
        if old is None or new is None:
            unverified.append(field_name)
            return None
        match = str(old) == str(new)
        if not match:
            reasons.append(reason_code)
        return match

    bank_file_match = _hash_check("bank_file_sha256", REASON_BANK_FILE_HASH_MISMATCH)
    bank_content_match = _hash_check("bank_content_sha256", REASON_BANK_CONTENT_HASH_MISMATCH)
    old_states = (previous.bank_ref or {}).get("parent_states_sha256") or _fixed_state_fingerprint(previous)
    new_states = (current.bank_ref or {}).get("parent_states_sha256") or _fixed_state_fingerprint(current)
    if old_states is None or new_states is None:
        states_match: Optional[bool] = None
        unverified.append("parent_states_sha256")
    else:
        states_match = str(old_states) == str(new_states)
        if not states_match:
            reasons.append(REASON_FIXED_STATE_MISMATCH)

    previous_grid = (previous.bank_ref or {}).get("grid_metadata")
    current_grid = (current.bank_ref or {}).get("grid_metadata")
    if previous_grid is None or current_grid is None:
        grid_match: Optional[bool] = None
        grid_differences: List[str] = []
        unverified.append("grid_metadata")
    else:
        grid_match, grid_differences = validate_grid_comparability(previous_grid, current_grid)
        if not grid_match:
            reasons.append(REASON_GRID_MISMATCH)

    return {
        "comparable": not reasons,
        "reason_codes": sorted(set(reasons)),
        "unverified_fields": sorted(set(unverified)),
        "semantic_diff_fields": sorted(semantic_diffs.keys()),
        "q_semantic_diff_fields": q_semantic_diffs,
        "bank_file_hash_match": bank_file_match,
        "bank_content_hash_match": bank_content_match,
        "parent_states_match": states_match,
        "grid_match": grid_match,
        "grid_differences": list(grid_differences),
    }


def _fixed_state_fingerprint(record: EpisodeEvalRecord) -> Optional[str]:
    """Fixed-state identity of a record (bank parent rows), if recorded."""
    for source in (
        (record.bank_ref or {}).get("parent_states_sha256"),
        (record.bellman.diagnostics or {}).get("parent_states_sha256")
        if record.bellman is not None
        else None,
        (record.policy.diagnostics or {}).get("parent_states_sha256")
        if record.policy is not None
        else None,
    ):
        if source is not None:
            return str(source)
    return None


# ---------------------------------------------------------------------------
# per-group metric tables (unified episode axis)
# ---------------------------------------------------------------------------


def _result_metrics(record: EpisodeEvalRecord, group: str) -> Optional[Dict[str, float]]:
    if group == "firm_bellman":
        result = record.bellman
    elif group == "macro":
        result = record.macro
    elif group in ("simulation", "distribution"):
        result = record.simulation
    else:
        return None
    if result is None:
        return None
    return dict(result.metrics or {})


def _missing_reason(group: str) -> str:
    return {
        "firm_bellman": REASON_MISSING_BELLMAN_RESULT,
        "macro": REASON_MISSING_MACRO_RESULT,
        "simulation": REASON_MISSING_SIMULATION_RESULT,
        "distribution": REASON_MISSING_SIMULATION_RESULT,
        "policy_drift": REASON_MISSING_POLICY_RESULT,
        "value_drift": REASON_MISSING_BELLMAN_RESULT,
    }.get(group, REASON_NO_RECORD_FOR_EPISODE)


def _group_metric_columns(records: Sequence[EpisodeEvalRecord], group: str) -> List[str]:

    columns: set = set()
    for record in records:
        metrics = _result_metrics(record, group)
        if metrics is None:
            continue
        for key in metrics:
            if _metric_in_group(key, group):
                columns.add(key)
    return sorted(columns)


def _build_metric_group_table(
    records: Sequence[EpisodeEvalRecord],
    episodes: Sequence[int],
    group: str,
) -> pd.DataFrame:
    """Episode-axis table of one metric group (raw evaluator metric keys)."""
    columns = _group_metric_columns(records, group)
    by_episode = {record.episode: record for record in records}
    rows: List[Dict[str, Any]] = []
    for episode in episodes:
        record = by_episode.get(episode)
        row: Dict[str, Any] = {"episode": episode}
        for name in columns:
            row[name] = _nan()
        if record is None:
            row["status"] = STATUS_MISSING
            row["reason_code"] = REASON_NO_RECORD_FOR_EPISODE
            rows.append(row)
            continue
        metrics = _result_metrics(record, group)
        if metrics is None:
            row["status"] = STATUS_MISSING
            row["reason_code"] = _missing_reason(group)
            rows.append(row)
            continue
        for name in columns:
            if name in metrics:
                row[name] = _to_float(metrics[name])
        row["status"] = STATUS_OK
        row["reason_code"] = REASON_EVALUATED
        rows.append(row)
    return pd.DataFrame(rows, columns=["episode"] + columns + ["status", "reason_code"])


def _distribution_slice_columns(record: EpisodeEvalRecord) -> Dict[str, float]:
    """b/z distribution moments + capital-weighted moments from the slice table."""
    out: Dict[str, float] = {}
    if record.simulation is None:
        return out
    frame = (record.simulation.tables or {}).get("distribution_slices")
    if frame is None or frame.empty:
        return out
    required = {"t", "caliber", "macro_state"}
    if not required.issubset(frame.columns):
        return out
    for _, row in frame.iterrows():
        if str(row.get("macro_state")) != "all":
            continue
        suffix = f"t{int(float(row['t']))}_{row['caliber']}"
        for source, target in (
            ("b_mean", f"dist_slice_b_mean_{suffix}"),
            ("b_var", f"dist_slice_b_var_{suffix}"),
            ("z_mean", f"dist_slice_z_mean_{suffix}"),
            ("z_var", f"dist_slice_z_var_{suffix}"),
            ("capital_weighted_mean_b", f"dist_slice_capital_weighted_mean_b_{suffix}"),
            ("default_rate", f"dist_slice_default_rate_{suffix}"),
            ("investment_rate", f"dist_slice_investment_rate_{suffix}"),
            ("entry_rate", f"dist_slice_entry_rate_{suffix}"),
            ("extinct_paths", f"dist_slice_extinct_paths_{suffix}"),
        ):
            if source in frame.columns:
                out[target] = _to_float(row[source])
    return out


# ---------------------------------------------------------------------------
# cross-episode drift tables
# ---------------------------------------------------------------------------


def _align_rows_frame(result: BellmanEvalResult) -> Optional[pd.DataFrame]:
    frame = (result.tables or {}).get("rows")
    if frame is None or frame.empty:
        return None
    return frame


def _surface_drift_columns(
    previous: BellmanEvalResult,
    current: BellmanEvalResult,
) -> Dict[str, float]:
    """b/z surface drift of the value quantities via ``function_drift_metrics``.

    Reuses ``evaluation.convergence_metrics.function_drift_metrics`` whose
    signature is ``(current, previous)``; surfaces of the same fixed bank are
    exactly aligned by construction (same shape and same b/z axes).
    """
    from evaluation.convergence_artifacts import SurfaceData
    from evaluation.convergence_metrics import function_drift_metrics

    columns: Dict[str, float] = {}
    previous_surfaces = previous.surfaces or {}
    current_surfaces = current.surfaces or {}
    for quantity, surface_key in VALUE_DRIFT_SURFACE_KEYS.items():
        old = previous_surfaces.get(surface_key)
        new = current_surfaces.get(surface_key)
        if old is None or new is None:
            continue
        old_array = np.asarray(old, dtype=np.float64)
        new_array = np.asarray(new, dtype=np.float64)
        if old_array.shape != new_array.shape:
            continue
        if old_array.ndim == 1:
            old_array = old_array.reshape(1, -1)
            new_array = new_array.reshape(1, -1)
        b_axis = np.arange(old_array.shape[0], dtype=np.float64)
        z_axis = np.arange(old_array.shape[1], dtype=np.float64)
        try:
            drift = function_drift_metrics(
                SurfaceData(values=new_array, b=b_axis, z=z_axis, path=Path("<current>")),
                SurfaceData(values=old_array, b=b_axis, z=z_axis, path=Path("<previous>")),
            )
        except ValueError:
            continue
        for key in (
            "n_aligned",
            "max_abs_diff",
            "mean_abs_diff",
            "rmse_diff",
            "p90_abs_diff",
            "p99_abs_diff",
            "normalized_mean_abs_diff",
            "normalized_max_abs_diff",
        ):
            columns[f"fdrift_{quantity}_{key}"] = _to_float(drift.get(key))
    return columns


def _value_drift_row(
    previous_record: EpisodeEvalRecord,
    current_record: EpisodeEvalRecord,
) -> Dict[str, float]:
    """Value drift of one comparable episode pair on the fixed state set A.

    ``D_e^Q = E_A[|Q_e(s) - Q_{e-1}(s)|] / E_A[1 + |Q_e(s)|]`` is reported as
    ``{quantity}_relative_mae``; the absolute errors (MAE/RMSE/P90/P99/Max),
    signed mean, sample sizes and the eta / source strata come from
    ``firm_bellman.residual_stat_block`` and are always kept alongside.
    """
    previous = previous_record.bellman
    current = current_record.bellman
    previous_frame = _align_rows_frame(previous)
    current_frame = _align_rows_frame(current)
    row: Dict[str, float] = {}
    if previous_frame is None or current_frame is None:
        return row
    if len(previous_frame) != len(current_frame):
        return row
    if "row_index" in previous_frame.columns and "row_index" in current_frame.columns:
        if not np.array_equal(
            previous_frame["row_index"].to_numpy(), current_frame["row_index"].to_numpy()
        ):
            return row
    eta = (
        current_frame["eta"].to_numpy(dtype=np.float64)
        if "eta" in current_frame.columns
        else None
    )
    source = (
        [str(item) for item in current_frame["source_label"].tolist()]
        if "source_label" in current_frame.columns
        else None
    )
    for quantity in VALUE_DRIFT_QUANTITIES:
        column = quantity
        if column not in previous_frame.columns or column not in current_frame.columns:
            continue
        old = previous_frame[column].to_numpy(dtype=np.float64)
        new = current_frame[column].to_numpy(dtype=np.float64)
        if old.shape != new.shape:
            continue
        row.update(
            residual_stat_block(
                new - old,
                prefix=quantity,
                level=new,
                eta=eta,
                source=source,
            )
        )
    row.update(_surface_drift_columns(previous, current))
    return row


def _policy_drift_row(
    previous_record: EpisodeEvalRecord,
    current_record: EpisodeEvalRecord,
) -> Tuple[Dict[str, float], Optional[str]]:
    """Policy drift of one comparable episode pair (consumes the policy tables).

    The Grid-BP / flip-rate block comes straight from
    ``firm_policy.compare_policy_across_episodes``; the investment-decision and
    default-boundary drifts are added on top of the same fixed bank rows.
    """
    previous = previous_record.policy
    current = current_record.policy
    if previous is None or current is None:
        return {}, REASON_MISSING_POLICY_RESULT
    try:
        comparison = compare_policy_across_episodes([previous, current])
    except ValueError as exc:
        return {}, f"POLICY_COMPARISON_REFUSED:{exc}"
    if comparison.empty:
        return {}, REASON_MISSING_POLICY_RESULT
    row: Dict[str, float] = {}
    record = comparison.iloc[0].to_dict()
    for key, value in record.items():
        if key in ("episode_previous", "episode_current"):
            row[key] = _to_float(value)
        elif _is_number(value):
            row[key] = _to_float(value)
    row["episode_previous"] = _to_float(previous_record.episode)
    row["episode_current"] = _to_float(current_record.episode)

    previous_rows = (previous.tables or {}).get("policy_rows")
    current_rows = (current.tables or {}).get("policy_rows")
    if previous_rows is not None and current_rows is not None and len(previous_rows) == len(current_rows):
        for column, target in (
            ("bar_i_executed", "bar_i_executed_drift_mean_abs"),
            ("bar_i_model", "bar_i_model_drift_mean_abs"),
            ("Phat", "phat_drift_mean_abs"),
            ("P0", "p0_drift_mean_abs"),
            ("PI", "pi_drift_mean_abs"),
            ("b_next_mix", "b_next_mix_drift_mean_abs"),
        ):
            if column in previous_rows.columns and column in current_rows.columns:
                old = previous_rows[column].to_numpy(dtype=np.float64)
                new = current_rows[column].to_numpy(dtype=np.float64)
                finite = np.isfinite(old) & np.isfinite(new)
                row[target] = (
                    float(np.mean(np.abs(new[finite] - old[finite]))) if finite.any() else _nan()
                )
        if "Bar_i" in previous_rows.columns and "Bar_i" in current_rows.columns:
            old = previous_rows["Bar_i"].to_numpy(dtype=np.float64)
            new = current_rows["Bar_i"].to_numpy(dtype=np.float64)
            finite = np.isfinite(old) & np.isfinite(new)
            if finite.any():
                row["bar_i_decision_flip_rate"] = float(
                    np.mean((old[finite] > 0.5) != (new[finite] > 0.5))
                )

    previous_boundary = (previous.tables or {}).get("default_boundary")
    current_boundary = (current.tables or {}).get("default_boundary")
    if (
        previous_boundary is not None
        and current_boundary is not None
        and "z_hard_phat0" in previous_boundary.columns
        and "z_hard_phat0" in current_boundary.columns
    ):
        for view in sorted(
            set(previous_boundary.get("eta_view", pd.Series(dtype=float)).dropna().tolist())
            & set(current_boundary.get("eta_view", pd.Series(dtype=float)).dropna().tolist())
        ):
            suffix = f"eta{int(round(float(view)))}"
            old = previous_boundary.loc[
                previous_boundary["eta_view"] == view, "z_hard_phat0"
            ].to_numpy(dtype=np.float64)
            new = current_boundary.loc[
                current_boundary["eta_view"] == view, "z_hard_phat0"
            ].to_numpy(dtype=np.float64)
            if old.shape != new.shape:
                continue
            finite = np.isfinite(old) & np.isfinite(new)
            if finite.any():
                drift = np.abs(new[finite] - old[finite])
                row[f"default_boundary_z_drift_mean_abs_{suffix}"] = float(drift.mean())
                row[f"default_boundary_z_drift_max_{suffix}"] = float(drift.max())
    return row, None


FIXED_SHOCK_ROLLOUT_KEYS = re.compile(
    r"^sim_.*_(capital_drop_t0_t1|capital_change_t0_t1|entry_refill_ratio_t1"
    r"|initial_cohort_default_share_t1|k0_total|k_t1_total|fc1_lnk_bias|fc1_hatc_bias)$"
)
MACRO_TRAJECTORY_KEYS = re.compile(r"^fc1_(rec|anchor)_")


def _numeric_metric_diff(
    previous: Mapping[str, float],
    current: Mapping[str, float],
    pattern: re.Pattern,
) -> Tuple[float, Dict[str, float]]:
    deltas: Dict[str, float] = {}
    for key in sorted(set(previous) & set(current)):
        if not pattern.search(key):
            continue
        old = _to_float(previous.get(key))
        new = _to_float(current.get(key))
        if math.isfinite(old) and math.isfinite(new):
            deltas[key] = new - old
    if not deltas:
        return _nan(), deltas
    return float(max(abs(value) for value in deltas.values())), deltas


def _distribution_cross_wasserstein(
    previous_record: EpisodeEvalRecord,
    current_record: EpisodeEvalRecord,
) -> Dict[str, float]:
    """Wasserstein-1 drift of the b/z marginals between two episodes.

    Both results must have been produced on the same fixed evaluation
    specification, so their ``distribution_marginals`` bins line up on
    ``(t, caliber, axis, bin_center)``.
    """
    out: Dict[str, float] = {}
    previous = previous_record.simulation
    current = current_record.simulation
    if previous is None or current is None:
        return out
    previous_frame = (previous.tables or {}).get("distribution_marginals")
    current_frame = (current.tables or {}).get("distribution_marginals")
    if previous_frame is None or current_frame is None:
        return out
    required = {"t", "caliber", "axis", "bin_center", "weight"}
    if not required.issubset(previous_frame.columns) or not required.issubset(current_frame.columns):
        return out
    group_keys = ["t", "caliber", "axis"]
    previous_groups = dict(tuple(previous_frame.groupby(group_keys)))
    current_groups = dict(tuple(current_frame.groupby(group_keys)))
    for key in sorted(set(previous_groups) & set(current_groups)):
        old = previous_groups[key]
        new = current_groups[key]
        old = old.sort_values("bin_center")
        new = new.sort_values("bin_center")
        if not np.allclose(
            old["bin_center"].to_numpy(dtype=np.float64),
            new["bin_center"].to_numpy(dtype=np.float64),
        ):
            continue
        distance = wasserstein_1(
            old["bin_center"].to_numpy(dtype=np.float64),
            new["bin_center"].to_numpy(dtype=np.float64),
            weights_a=old["weight"].to_numpy(dtype=np.float64),
            weights_b=new["weight"].to_numpy(dtype=np.float64),
        )
        t_value, caliber, axis = key
        out[f"cross_w1_{axis}_t{int(float(t_value))}_{caliber}"] = _to_float(distance)
    return out


# ---------------------------------------------------------------------------
# gate evaluation
# ---------------------------------------------------------------------------


def _gate_metric_lookup(
    spec: GateSpec,
    tables: Mapping[str, pd.DataFrame],
    episode: int,
) -> Tuple[float, str, bool]:
    """Resolve the gate metric for one episode row.

    Returns ``(value, metric_description, available)``; ``available=False``
    means the metric is absent or non-finite (never silently treated as small).
    """
    table_names = [spec.table] if spec.table != "*" else list(ALL_TABLES)
    collected: List[np.ndarray] = []
    matched: List[str] = []
    for table_name in table_names:
        frame = tables.get(table_name)
        if frame is None or frame.empty or "episode" not in frame.columns:
            continue
        rows = frame.loc[frame["episode"] == episode]
        if rows.empty:
            continue
        row = rows.iloc[0]
        if spec.metric_key is not None:
            if spec.metric_key in frame.columns:
                collected.append(np.asarray([_to_float(row[spec.metric_key])], dtype=np.float64))
                matched.append(f"{table_name}.{spec.metric_key}")
        else:
            pattern = re.compile(spec.metric_pattern or "")
            for name in frame.columns:
                if name in ("episode", "status", "reason_code") or not pattern.search(str(name)):
                    continue
                collected.append(np.asarray([_to_float(row[name])], dtype=np.float64))
                matched.append(f"{table_name}.{name}")
    if not collected:
        return _nan(), "", False
    values = _finite_concat(collected)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return _nan(), ";".join(matched), False
    return _reduce(values, spec.reducer), ";".join(matched), True


def _evaluate_gate(
    spec: GateSpec,
    tables: Mapping[str, pd.DataFrame],
    episode: int,
    threshold_config: ThresholdConfig,
    *,
    applicable: bool,
    group_blocked: bool,
) -> Dict[str, Any]:
    threshold = threshold_config.get(spec.gate_id)
    out: Dict[str, Any] = {
        "group": spec.group,
        "table": spec.table,
        "direction": spec.direction,
        "reducer": spec.reducer if spec.metric_pattern else None,
        "metric": spec.metric_key or spec.metric_pattern,
        "required": bool(spec.required),
        "requires_predecessor": bool(spec.requires_predecessor),
        "applicable": bool(applicable),
        "value": None,
        "threshold": threshold,
    }
    if not applicable:
        out["status"] = GATE_INSUFFICIENT
        out["reason_code"] = REASON_NO_PREVIOUS_FOR_DRIFT_GATE
        return out
    if threshold is None:
        out["status"] = GATE_NOT_CONFIGURED
        out["reason_code"] = GATE_REASON_THRESHOLD_NOT_CONFIGURED
        value, metric_description, available = _gate_metric_lookup(spec, tables, episode)
        out["value"] = value if available else None
        out["metric_sources"] = metric_description
        return out
    value, metric_description, available = _gate_metric_lookup(spec, tables, episode)
    out["metric_sources"] = metric_description
    if group_blocked:
        out["status"] = GATE_INSUFFICIENT
        out["reason_code"] = GATE_REASON_GROUP_NOT_COMPARABLE
        out["value"] = value if available else None
        return out
    if not available:
        out["status"] = GATE_INSUFFICIENT
        out["reason_code"] = GATE_REASON_METRIC_UNAVAILABLE
        return out
    out["value"] = value
    if spec.direction == "max":
        if value <= threshold:
            out["status"] = GATE_PASS
            out["reason_code"] = GATE_REASON_WITHIN_THRESHOLD
        else:
            out["status"] = GATE_FAIL
            out["reason_code"] = GATE_REASON_FAIL_ABOVE
    else:
        if value >= threshold:
            out["status"] = GATE_PASS
            out["reason_code"] = GATE_REASON_ABOVE_MINIMUM
        else:
            out["status"] = GATE_FAIL
            out["reason_code"] = GATE_REASON_FAIL_BELOW
    return out


# ---------------------------------------------------------------------------
# dashboard assembly
# ---------------------------------------------------------------------------


def _require_unique_episodes(records: Sequence[EpisodeEvalRecord]) -> None:
    seen: Dict[int, int] = {}
    for record in records:
        episode = int(record.episode)
        seen[episode] = seen.get(episode, 0) + 1
    duplicated = sorted(episode for episode, count in seen.items() if count > 1)
    if duplicated:
        raise ValueError(
            "compare_episodes requires at most one EpisodeEvalRecord per episode; "
            f"duplicated episodes: {duplicated}"
        )


def compare_episodes(
    records: Sequence[EpisodeEvalRecord],
    *,
    threshold_config: Optional[ThresholdConfig] = None,
    required_consecutive_episodes: int = 3,
) -> ComparisonResult:
    """Build the Outer Fixed Point Dashboard for a run of outer episodes.

    Returns a :class:`ComparisonResult` with

    * ``metrics`` -- one episode-axis table per dashboard group (plus the
      ``comparability`` pre-check table); every table has ``status`` and
      ``reason_code`` columns so missing episodes / missing results are explicit
      holes, never silently dropped;
    * ``convergence_status`` -- the ``convergence_status.json`` payload (per-gate
      states and reason codes, participating episodes, consecutive-satisfied
      counts);
    * ``diagnostics`` -- the comparability verdicts per pair, per-metric
      cross-episode drift maps and the fixed-shock rollout comparison.

    Drift groups only compare *consecutive, comparable* episodes on the same
    fixed evaluation bank; otherwise the group row is ``NOT_COMPARABLE`` with
    the reason codes of the failed pre-check.
    """
    records = list(records)
    _require_unique_episodes(records)
    threshold_config = threshold_config if threshold_config is not None else ThresholdConfig()
    if int(required_consecutive_episodes) < 1:
        raise ValueError("required_consecutive_episodes must be positive")
    episodes = sorted(int(record.episode) for record in records)
    by_episode = {int(record.episode): record for record in records}

    # ---- static per-episode metric tables --------------------------------
    tables: Dict[str, pd.DataFrame] = {}
    for group in ("firm_bellman", "macro"):
        tables[group] = _build_metric_group_table(records, episodes, group)
    simulation_table = _build_metric_group_table(records, episodes, "simulation")
    distribution_table = _build_metric_group_table(records, episodes, "distribution")
    # distribution moments from the distribution_slices table (b/z + capital
    # weighted + extinction shares) widen the distribution group
    extra_columns: set = set()
    for record in records:
        extra_columns.update(_distribution_slice_columns(record).keys())
    for name in sorted(extra_columns):
        distribution_table[name] = _nan()
    for record in records:
        extras = _distribution_slice_columns(record)
        mask = distribution_table["episode"] == int(record.episode)
        for name, value in extras.items():
            distribution_table.loc[mask, name] = value
    tables["simulation"] = simulation_table
    tables["distribution"] = distribution_table

    # ---- comparability pre-check ----------------------------------------
    pair_info: Dict[int, Dict[str, Any]] = {}
    comparability_rows: List[Dict[str, Any]] = []
    semantic_diffs_all = _semantic_diffs(records)
    for index, episode in enumerate(episodes):
        record = by_episode[episode]
        row: Dict[str, Any] = {
            "episode": episode,
            "semantic_diff_count": float(len(semantic_diffs_all)),
            "q_semantic_diff_count": float(
                sum(
                    1
                    for name in semantic_diffs_all
                    if "q_parameterization" in name or "q_recovery" in name
                )
            ),
        }
        if index == 0:
            row.update(
                {
                    "comparable": True,
                    "status": STATUS_OK,
                    "reason_code": REASON_EVALUATED,
                    "episode_previous": _nan(),
                    "episode_gap": _nan(),
                    "bank_file_hash_match": _nan(),
                    "bank_content_hash_match": _nan(),
                    "parent_states_match": _nan(),
                    "grid_match": _nan(),
                }
            )
            pair_info[episode] = {
                "comparable": True,
                "reason_codes": [],
                "has_predecessor": False,
                "episode_gap": None,
            }
            comparability_rows.append(row)
            continue
        previous_episode = episodes[index - 1]
        previous_record = by_episode[previous_episode]
        verdict = _pair_comparability(previous_record, record)
        gap = episode - previous_episode
        if gap != 1:
            verdict = dict(verdict)
            verdict["reason_codes"] = sorted(
                set(verdict["reason_codes"]) | {REASON_NON_CONSECUTIVE_EPISODE_GAP}
            )
            verdict["comparable"] = False
        row.update(
            {
                "comparable": bool(verdict["comparable"]),
                "status": STATUS_OK if verdict["comparable"] else STATUS_NOT_COMPARABLE,
                "reason_code": (
                    REASON_EVALUATED
                    if verdict["comparable"]
                    else ";".join(verdict["reason_codes"])
                ),
                "episode_previous": float(previous_episode),
                "episode_gap": float(gap),
                "bank_file_hash_match": _to_float(
                    np.nan if verdict["bank_file_hash_match"] is None else verdict["bank_file_hash_match"]
                ),
                "bank_content_hash_match": _to_float(
                    np.nan
                    if verdict["bank_content_hash_match"] is None
                    else verdict["bank_content_hash_match"]
                ),
                "parent_states_match": _to_float(
                    np.nan if verdict["parent_states_match"] is None else verdict["parent_states_match"]
                ),
                "grid_match": _to_float(
                    np.nan if verdict["grid_match"] is None else verdict["grid_match"]
                ),
            }
        )
        pair_info[episode] = {
            **verdict,
            "has_predecessor": True,
            "episode_previous": previous_episode,
            "episode_gap": gap,
        }
        comparability_rows.append(row)
    tables[COMPARABILITY_GROUP] = pd.DataFrame(comparability_rows)

    # ---- drift groups ----------------------------------------------------
    value_rows: List[Dict[str, Any]] = []
    policy_rows: List[Dict[str, Any]] = []
    metric_drift: Dict[str, Dict[str, Any]] = {"macro": {}, "simulation": {}, "distribution": {}}
    rollout_records: Dict[str, SimEvalResult] = {}
    for episode in episodes:
        record = by_episode[episode]
        info = pair_info[episode]
        for group, sink, missing_group in (
            ("value_drift", value_rows, "bellman"),
            ("policy_drift", policy_rows, "policy"),
        ):
            row = {"episode": episode}
            if not info["has_predecessor"]:
                row["status"] = STATUS_NOT_EVALUATED
                row["reason_code"] = REASON_NO_PREVIOUS_EPISODE
                sink.append(row)
                continue
            previous_record = by_episode[info["episode_previous"]]
            previous_result = getattr(previous_record, missing_group)
            current_result = getattr(record, missing_group)
            if previous_result is None or current_result is None:
                row["status"] = STATUS_MISSING
                row["reason_code"] = _missing_reason(group)
                sink.append(row)
                continue
            if not info["comparable"]:
                row["status"] = STATUS_NOT_COMPARABLE
                row["reason_code"] = ";".join(info["reason_codes"])
                sink.append(row)
                continue
            if group == "value_drift":
                drift = _value_drift_row(previous_record, record)
                error_reason = None if drift else "DRIFT_ALIGNMENT_FAILED"
            else:
                drift, error_reason = _policy_drift_row(previous_record, record)
            if error_reason is not None:
                row["status"] = STATUS_NOT_COMPARABLE
                row["reason_code"] = error_reason
                sink.append(row)
                continue
            row.update(drift)
            row["episode_previous"] = float(info["episode_previous"])
            row["episode_gap"] = float(info["episode_gap"])
            row["status"] = STATUS_OK
            row["reason_code"] = REASON_EVALUATED
            sink.append(row)

        # cross-episode drift of macro / simulation / distribution metrics
        if info["has_predecessor"] and info["comparable"]:
            previous_record = by_episode[info["episode_previous"]]
            macro_delta: Dict[str, float] = {}
            simulation_delta: Dict[str, float] = {}
            if previous_record.macro is not None and record.macro is not None:
                _, macro_delta = _numeric_metric_diff(
                    previous_record.macro.metrics or {},
                    record.macro.metrics or {},
                    MACRO_TRAJECTORY_KEYS,
                )
            if previous_record.simulation is not None and record.simulation is not None:
                fixed_drift, simulation_delta = _numeric_metric_diff(
                    previous_record.simulation.metrics or {},
                    record.simulation.metrics or {},
                    FIXED_SHOCK_ROLLOUT_KEYS,
                )
                tables["simulation"].loc[
                    tables["simulation"]["episode"] == episode, "fixed_shock_rollout_drift_abs_max"
                ] = fixed_drift
            macro_drift = (
                float(max(abs(value) for value in macro_delta.values()))
                if macro_delta
                else _nan()
            )
            tables["macro"].loc[
                tables["macro"]["episode"] == episode, "macro_trajectory_drift_abs_max"
            ] = macro_drift
            cross_w1 = _distribution_cross_wasserstein(previous_record, record)
            distribution_values = [abs(value) for value in cross_w1.values()]
            distribution_slice_delta: Dict[str, float] = {}
            if previous_record.simulation is not None and record.simulation is not None:
                previous_extras = _distribution_slice_columns(previous_record)
                current_extras = _distribution_slice_columns(record)
                for name in sorted(set(previous_extras) & set(current_extras)):
                    old = previous_extras[name]
                    new = current_extras[name]
                    if math.isfinite(old) and math.isfinite(new):
                        distribution_slice_delta[name] = new - old
                        distribution_values.append(abs(new - old))
            for name, value in cross_w1.items():
                tables["distribution"].loc[
                    tables["distribution"]["episode"] == episode, name
                ] = value
            tables["distribution"].loc[
                tables["distribution"]["episode"] == episode, "distribution_drift_abs_max"
            ] = (
                float(max(distribution_values)) if distribution_values else _nan()
            )
            metric_drift["macro"][str(episode)] = macro_delta
            metric_drift["simulation"][str(episode)] = simulation_delta
            metric_drift["distribution"][str(episode)] = {
                **cross_w1,
                **{f"{name}": value for name, value in distribution_slice_delta.items()},
            }

        if record.simulation is not None:
            rollout_records[f"episode_{episode}"] = record.simulation

    tables["value_drift"] = _ordered_frame(value_rows, episodes)
    tables["policy_drift"] = _ordered_frame(policy_rows, episodes)

    # ---- gates + convergence decision -----------------------------------
    episode_status: List[Dict[str, Any]] = []
    consecutive = 0
    best_consecutive = 0
    previous_satisfied = False
    any_fail = False
    for index, episode in enumerate(episodes):
        info = pair_info[episode]
        gates: Dict[str, Dict[str, Any]] = {group: {} for group in GATE_GROUPS}
        blocked_tables = {
            "value_drift": False,
            "policy_drift": False,
            "macro": False,
            "simulation": False,
            "distribution": False,
            "firm_bellman": False,
            COMPARABILITY_GROUP: False,
        }
        if info["has_predecessor"] and not info["comparable"]:
            for name in ("value_drift", "policy_drift", "macro", "simulation", "distribution"):
                blocked_tables[name] = True
        has_predecessor = bool(info.get("has_predecessor"))
        for gate_id, spec in GATE_SPECS.items():
            applicable = not (spec.requires_predecessor and not has_predecessor)
            gate = _evaluate_gate(
                spec,
                tables,
                episode,
                threshold_config,
                applicable=applicable,
                group_blocked=blocked_tables.get(spec.table, False),
            )
            gates[spec.group][gate_id] = gate
        necessary = [
            gate
            for group in GATE_GROUPS
            for gate in gates[group].values()
            if gate["required"] and gate["applicable"]
        ]
        statuses = [gate["status"] for gate in necessary]
        if any(status == GATE_FAIL for status in statuses):
            satisfied: Optional[bool] = False
            any_fail = True
        elif statuses and all(status == GATE_PASS for status in statuses):
            satisfied = True
        else:
            satisfied = None

        gap = info.get("episode_gap")
        gap_ok = (not has_predecessor) or gap == 1
        if satisfied is True:
            if previous_satisfied and gap_ok:
                consecutive += 1
            else:
                consecutive = 1
            streak_reset_reason = None
        else:
            consecutive = 0
            if satisfied is False:
                streak_reset_reason = "GATE_FAIL"
            elif not info["comparable"]:
                streak_reset_reason = "NOT_COMPARABLE"
            elif not gap_ok:
                streak_reset_reason = REASON_NON_CONSECUTIVE_EPISODE_GAP
            else:
                streak_reset_reason = "INSUFFICIENT_EVIDENCE"
        best_consecutive = max(best_consecutive, consecutive)
        previous_satisfied = satisfied is True
        episode_status.append(
            {
                "episode": episode,
                "comparable": bool(info["comparable"]),
                "has_predecessor": has_predecessor,
                "episode_previous": info.get("episode_previous"),
                "episode_gap": gap,
                "not_comparable_reason_code": (
                    None if info["comparable"] else ";".join(info["reason_codes"])
                ),
                "unverified_fields": list(info.get("unverified_fields") or []),
                "satisfied": satisfied,
                "consecutive_satisfied_count": consecutive,
                "streak_reset_reason": streak_reset_reason,
                "gates": gates,
            }
        )

    if best_consecutive >= int(required_consecutive_episodes):
        convergence_status = CONVERGED_CANDIDATE
        converged_candidate = True
        reason_code = "THREE_CONSECUTIVE_COMPARABLE_EPISODES_SATISFIED"
    else:
        converged_candidate = False
        if any_fail:
            convergence_status = NOT_CONVERGED
            reason_code = "REQUIRED_GATE_FAILED"
        else:
            convergence_status = INSUFFICIENT_EVIDENCE
            reason_code = "INSUFFICIENT_EVIDENCE_FOR_CONVERGENCE"

    status_payload = {
        "schema_version": "convergence_status_v1",
        "convergence_status": convergence_status,
        "converged_candidate": bool(converged_candidate),
        "reason_code": reason_code,
        "required_consecutive_comparable_episodes": int(required_consecutive_episodes),
        "max_consecutive_satisfied_count": int(best_consecutive),
        "final_consecutive_satisfied_count": int(
            episode_status[-1]["consecutive_satisfied_count"] if episode_status else 0
        ),
        "episodes": episodes,
        "participating_episodes": [
            entry["episode"]
            for entry in episode_status
            if entry["comparable"] or entry["has_predecessor"]
        ],
        "comparable_episode_pairs": [
            [entry["episode_previous"], entry["episode"]]
            for entry in episode_status
            if entry["has_predecessor"] and entry["comparable"]
        ],
        "episode_status": episode_status,
        "threshold_config": threshold_config.to_dict(),
        "threshold_config_complete": bool(threshold_config.is_complete()),
        "missing_required_thresholds": threshold_config.missing_required_gate_ids(),
        "gate_groups": list(GATE_GROUPS),
    }

    diagnostics: Dict[str, Any] = {
        "episode_axis": episodes,
        "semantic_diffs": _json_safe(semantic_diffs_all),
        "pair_comparability": {
            str(episode): _json_safe(pair_info[episode])
            for episode in episodes
            if pair_info[episode].get("has_predecessor")
        },
        "metric_drift": _json_safe(metric_drift),
        "unverified_comparability_fields_note": (
            "unverified fields could not be checked from the recorded metadata; "
            "they never veto a comparison but are always surfaced"
        ),
        "fixed_shock_rollout_comparison": _json_safe(_rollout_comparison(rollout_records)),
        "value_drift_definition": (
            "D_e^Q = E_A[|Q_e(s) - Q_{e-1}(s)|] / E_A[1 + |Q_e(s)|] over the "
            "fixed EvalBank state set A; absolute errors and sample sizes are "
            "reported next to it ({quantity}_mae/_rmse/_p90/_p99/_max/_n)"
        ),
        "evaluation_only": True,
        "optimizer_steps": 0,
    }
    return ComparisonResult(
        metrics=tables,
        convergence_status=status_payload,
        diagnostics=diagnostics,
    )


def _ordered_frame(rows: List[Dict[str, Any]], episodes: Sequence[int]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame({"episode": list(episodes)})
    frame = pd.DataFrame(rows)
    ordered = ["episode"]
    for name in sorted(frame.columns):
        if name not in ordered and name not in ("status", "reason_code"):
            ordered.append(name)
    ordered.extend(["status", "reason_code"])
    return frame.reindex(columns=ordered)


def _rollout_comparison(
    rollout_records: Mapping[str, SimEvalResult],
) -> List[Dict[str, Any]]:
    """Fixed-shock rollout cross-checkpoint comparison (Bank E)."""
    if len(rollout_records) < 2:
        return []
    try:
        table = compare_fixed_shock_rollouts(dict(rollout_records))
    except Exception as exc:  # pragma: no cover - defensive, recorded verbatim
        return [{"error": f"FIXED_SHOCK_ROLLOUT_COMPARISON_FAILED: {exc}"}]
    if table is None or table.empty:
        return []
    return _json_safe(table.to_dict(orient="records"))


# ---------------------------------------------------------------------------
# threshold calibration report framework
# ---------------------------------------------------------------------------

# Economic feasibility criteria are *reported as basis*, never written into the
# thresholds: the resource-ledger tolerances are the defaults of the production
# validator ``data/entry.py::validate_node_resource_account`` (atol/rtol) and
# the nonfinite / Q-semantic criteria are exact zeros.
ECONOMIC_FEASIBILITY_CRITERIA: Dict[str, Dict[str, Any]] = {
    "hard_resource_ledger_residual": {
        "criterion": "resource_ledger_residual <= atol + rtol * |reference|",
        "atol": 1e-6,
        "rtol": 1e-5,
        "source": "data/entry.py::validate_node_resource_account defaults",
    },
    "hard_nonfinite": {"criterion": "nonfinite share/ratio == 0", "value": 0.0},
    "hard_q_semantic_consistency": {"criterion": "Q semantic diffs == 0", "value": 0.0},
    "hard_sim_path_completeness": {
        "criterion": "every rollout completes its requested horizon",
        "value": None,
    },
    "outer_extinct_paths": {"criterion": "no extinct paths in the fixed rollouts", "value": 0.0},
}

_CALIBRATION_NOTES = [
    "This report never picks a threshold: 'recommended_threshold' is always null and "
    "'suggested_interval' is a data-derived interval, not a decision.",
    "The interval basis is (1) the numerical-error economic scale (absolute errors and "
    "sample sizes are retained next to every relative figure), (2) the historical "
    "residual distribution on the fixed EvalBank, (3) multi-seed Monte-Carlo spread and "
    "(4) a path-clustered bootstrap confidence interval; economic feasibility criteria "
    "are recorded separately and bound the decision from below/above.",
    "A gate is only judged when its threshold is configured; unconfigured gates report "
    "THRESHOLD_NOT_CONFIGURED and can never be passed by a small-looking loss.",
]


def _gate_observations(
    records: Sequence[EpisodeEvalRecord],
    gate_ids: Sequence[str],
    threshold_config: ThresholdConfig,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    """Per-gate observed values across the records (one compare per seed group)."""
    observations: Dict[str, List[Dict[str, Any]]] = {gate_id: [] for gate_id in gate_ids}
    seeds = sorted({record.seed for record in records}, key=lambda value: (value is None, value))
    warnings: List[str] = []
    for seed in seeds:
        group = [record for record in records if record.seed == seed]
        try:
            result = compare_episodes(group, threshold_config=threshold_config)
        except ValueError as exc:
            warnings.append(f"seed={seed}: episodes could not be compared ({exc})")
            continue
        for entry in result.convergence_status["episode_status"]:
            episode = int(entry["episode"])
            for gate_id in gate_ids:
                spec = GATE_SPECS[gate_id]
                gate = entry["gates"][spec.group][gate_id]
                if not gate["applicable"]:
                    continue
                value = gate["value"]
                if value is None or not math.isfinite(float(value)):
                    continue
                observations[gate_id].append(
                    {
                        "episode": episode,
                        "seed": seed,
                        "value": float(value),
                        "metric": gate["metric"],
                        "companion": _companion_stats(result.metrics, spec, episode),
                    }
                )
    return observations, warnings


def _companion_stats(
    tables: Mapping[str, pd.DataFrame],
    spec: GateSpec,
    episode: int,
) -> Dict[str, Any]:
    """Absolute errors and sample sizes that belong to the gate's metric(s)."""
    table_names = [spec.table] if spec.table != "*" else list(ALL_TABLES)
    stems: List[str] = []
    for table_name in table_names:
        frame = tables.get(table_name)
        if frame is None or frame.empty or "episode" not in frame.columns:
            continue
        rows = frame.loc[frame["episode"] == episode]
        if rows.empty:
            continue
        if spec.metric_key is not None:
            candidates = [spec.metric_key] if spec.metric_key in frame.columns else []
        else:
            pattern = re.compile(spec.metric_pattern or "")
            candidates = [
                name
                for name in frame.columns
                if name not in ("episode", "status", "reason_code") and pattern.search(str(name))
            ]
        for name in candidates:
            stem = re.sub(r"_(relative_mae|mae|rmse|p90|p99|max)$", "", str(name))
            stems.append(f"{table_name}:{stem}")
    absolute: List[float] = []
    counts: List[float] = []
    finite_counts: List[float] = []
    for entry in sorted(set(stems)):
        table_name, stem = entry.split(":", 1)
        frame = tables.get(table_name)
        if frame is None or frame.empty:
            continue
        rows = frame.loc[frame["episode"] == episode]
        if rows.empty:
            continue
        row = rows.iloc[0]
        for suffix, sink in (("_mae", absolute), ("_n", counts), ("_n_finite", finite_counts)):
            column = f"{stem}{suffix}"
            if column in frame.columns and math.isfinite(_to_float(row[column])):
                sink.append(float(row[column]))
    return {
        "absolute_mae_mean": float(np.mean(absolute)) if absolute else None,
        "n_mean": float(np.mean(counts)) if counts else None,
        "n_finite_mean": float(np.mean(finite_counts)) if finite_counts else None,
        "n_sources": len(set(stems)),
    }


def build_threshold_calibration(
    records: Sequence[EpisodeEvalRecord],
    *,
    gate_ids: Optional[Sequence[str]] = None,
    threshold_config: Optional[ThresholdConfig] = None,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    stat: Union[str, Callable[[np.ndarray], float]] = "mean",
) -> Dict[str, Any]:
    """Threshold calibration report framework (suggestions, never decisions).

    For every candidate gate the report collects the observed metric values on
    the fixed EvalBank across episodes (and seeds), the retained absolute errors
    and sample sizes, the historical residual distribution, the multi-seed
    Monte-Carlo spread, a :func:`path_clustered_bootstrap` confidence interval
    (clusters = episodes/paths) and the economic feasibility criteria, and
    derives ``suggested_interval`` from the observed distribution's bootstrap
    interval.  ``recommended_threshold`` stays ``null``: choosing the threshold
    is the caller's economic decision.
    """
    records = list(records)
    threshold_config = threshold_config if threshold_config is not None else ThresholdConfig()
    if gate_ids is None:
        selected = sorted(GATE_SPECS)
    else:
        selected = []
        for gate_id in gate_ids:
            if gate_id not in GATE_SPECS:
                raise ValueError(f"unknown gate id: {gate_id!r}")
            selected.append(gate_id)
    observations, warnings = _gate_observations(records, selected, threshold_config)

    candidates: Dict[str, Any] = {}
    for gate_id in selected:
        spec = GATE_SPECS[gate_id]
        entries = sorted(observations[gate_id], key=lambda item: (item["episode"], str(item["seed"])))
        values = np.asarray([entry["value"] for entry in entries], dtype=np.float64)
        episodes = [entry["episode"] for entry in entries]
        per_seed: Dict[str, List[float]] = {}
        for entry in entries:
            per_seed.setdefault(str(entry["seed"]), []).append(float(entry["value"]))
        seed_means = {
            name: float(np.mean(seed_values)) if seed_values else None
            for name, seed_values in per_seed.items()
        }
        finite_seed_means = [
            value for value in seed_means.values() if value is not None and math.isfinite(value)
        ]
        if len(per_seed) >= 2 and len(finite_seed_means) >= 2:
            multi_seed: Dict[str, Any] = {
                "available": True,
                "n_seeds": len(per_seed),
                "mean_by_seed": seed_means,
                "std_across_seeds": float(np.std(finite_seed_means, ddof=0)),
            }
        else:
            multi_seed = {
                "available": False,
                "n_seeds": len(per_seed),
                "mean_by_seed": seed_means,
                "std_across_seeds": None,
                "reason_code": "NEEDS_AT_LEAST_TWO_SEEDS",
            }
        bootstrap = path_clustered_bootstrap(
            values if values.size else np.empty(0),
            np.asarray(episodes),
            n_boot=n_boot,
            seed=seed,
            stat=stat,
            alpha=alpha,
        )
        finite = values[np.isfinite(values)]
        distribution = {
            "n": int(finite.size),
            "mean": float(finite.mean()) if finite.size else None,
            "std": float(finite.std(ddof=0)) if finite.size > 1 else (0.0 if finite.size else None),
            "p50": float(np.quantile(finite, 0.50)) if finite.size else None,
            "p90": float(np.quantile(finite, 0.90)) if finite.size else None,
            "p99": float(np.quantile(finite, 0.99)) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
        }
        absolute_maes = [
            entry["companion"]["absolute_mae_mean"]
            for entry in entries
            if entry["companion"].get("absolute_mae_mean") is not None
        ]
        sample_sizes = [
            entry["companion"]["n_mean"]
            for entry in entries
            if entry["companion"].get("n_mean") is not None
        ]
        ci_low = bootstrap.get("ci_low")
        ci_high = bootstrap.get("ci_high")
        if (
            finite.size >= 2
            and ci_low is not None
            and ci_high is not None
            and math.isfinite(float(ci_low))
            and math.isfinite(float(ci_high))
        ):
            suggested_interval: Optional[List[float]] = [float(ci_low), float(ci_high)]
            interval_reason = "BOOTSTRAP_CI_OF_OBSERVED_DISTRIBUTION"
        else:
            suggested_interval = None
            interval_reason = "INSUFFICIENT_OBSERVATIONS"
        candidates[gate_id] = {
            "gate_id": gate_id,
            "group": spec.group,
            "table": spec.table,
            "metric": spec.metric_key or spec.metric_pattern,
            "direction": spec.direction,
            "required": bool(spec.required),
            "configured_threshold": threshold_config.get(gate_id),
            "values_by_episode": {
                str(entry["episode"]): entry["value"] for entry in entries
            },
            "historical_residual_distribution": distribution,
            "absolute_error_companion": {
                "absolute_mae_by_episode": {
                    str(entry["episode"]): entry["companion"]["absolute_mae_mean"]
                    for entry in entries
                },
                "sample_size_by_episode": {
                    str(entry["episode"]): entry["companion"]["n_mean"] for entry in entries
                },
                "mean_absolute_mae": float(np.mean(absolute_maes)) if absolute_maes else None,
                "mean_sample_size": float(np.mean(sample_sizes)) if sample_sizes else None,
            },
            "multi_seed_monte_carlo": multi_seed,
            "path_clustered_bootstrap": bootstrap,
            "economic_scale": {
                "normalization": (
                    "relative-scale error E[|delta|]/E[1+|level|]"
                    if str(spec.metric_key or spec.metric_pattern or "").endswith("relative_mae")
                    else "absolute units of the underlying evaluator metric"
                ),
                "observed_scale": distribution,
            },
            "economic_feasibility": ECONOMIC_FEASIBILITY_CRITERIA.get(
                gate_id, {"criterion": None, "source": None}
            ),
            "suggested_interval": suggested_interval,
            "suggested_interval_reason_code": interval_reason,
            "recommended_threshold": None,
            "basis": [
                "numerical_error_economic_scale",
                "historical_residual_distribution_on_fixed_evalbank",
                (
                    "multi_seed_monte_carlo"
                    if multi_seed["available"]
                    else "multi_seed_monte_carlo_unavailable"
                ),
                "path_clustered_bootstrap",
                "economic_feasibility_criteria",
            ],
        }
    return {
        "schema_version": "threshold_calibration_v1",
        "n_records": len(records),
        "episodes": sorted({int(record.episode) for record in records}),
        "seeds": sorted({record.seed for record in records}, key=lambda value: (value is None, value)),
        "bootstrap_config": {
            "n_boot": int(n_boot),
            "seed": int(seed),
            "alpha": float(alpha),
            "stat": stat if isinstance(stat, str) else getattr(stat, "__name__", "callable"),
            "cluster_unit": "episode / simulation path",
        },
        "candidates": candidates,
        "warnings": warnings,
        "notes": list(_CALIBRATION_NOTES),
    }


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def save_comparison(
    result: ComparisonResult,
    output_dir: str | Path,
) -> Dict[str, str]:
    """Write the dashboard to ``output_dir``.

    Emits ``metrics_<group>.csv`` per group (unified episode axis), one combined
    ``metrics.csv`` in long format, ``convergence_status.json`` and
    ``comparison_diagnostics.json``.  Returns the written paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    long_rows: List[Dict[str, Any]] = []
    for group, frame in result.metrics.items():
        target = output_path / f"metrics_{group}.csv"
        frame.to_csv(target, index=False)
        written[f"metrics_{group}"] = str(target)
        for _, row in frame.iterrows():
            episode = row.get("episode")
            for name in frame.columns:
                if name in ("episode", "status", "reason_code"):
                    continue
                long_rows.append(
                    {
                        "episode": episode,
                        "group": group,
                        "metric": name,
                        "value": row[name],
                        "status": row.get("status"),
                        "reason_code": row.get("reason_code"),
                    }
                )
    long_frame = pd.DataFrame(
        long_rows, columns=["episode", "group", "metric", "value", "status", "reason_code"]
    )
    long_target = output_path / "metrics.csv"
    long_frame.to_csv(long_target, index=False)
    written["metrics"] = str(long_target)

    status_target = output_path / "convergence_status.json"
    status_target.write_text(
        json.dumps(_json_safe(result.convergence_status), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    written["convergence_status"] = str(status_target)

    diagnostics_target = output_path / "comparison_diagnostics.json"
    diagnostics_target.write_text(
        json.dumps(_json_safe(result.diagnostics), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    written["comparison_diagnostics"] = str(diagnostics_target)
    return written
