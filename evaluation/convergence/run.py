"""Evaluation-Only orchestration + CLI for the equilibrium convergence evaluator.

``run_convergence_evaluation`` drives the full evaluation pipeline over one or
more combined checkpoints:

1. load checkpoints (``checkpoint_io.load_eval_checkpoint``) one episode at a
   time and release them again (chunked load/release so the whole network set
   never has to stay resident on the GPU), with the entire evaluation wrapped
   in ``param_hash_guard`` and ``optimizer_steps == 0`` recorded everywhere;
2. per episode run the firm Bellman / firm policy / macro+SDF / simulation
   evaluators on the fixed ``evaluation_bank_v1`` banks (mode subsets:
   ``all`` / ``firm`` / ``macro`` / ``sim``); a failing evaluator is recorded
   with a reason code and the run continues;
3. cross-episode comparison (``comparator.compare_episodes`` +
   ``save_comparison``);
4. figure rendering (``figures.render_all_figures``);
5. report building (``report.build_convergence_report``);
6. write the Prompt §13 output tree under
   ``<output_root>/convergence_<run_id>/{manifest.json, bank/, metrics/,
   figures/, tables/, report/, diagnostics/}``.

CLI::

    python -m evaluation.convergence.run \
        --checkpoint ep3_combined.pt --checkpoint ep4_combined.pt \
        --bank-root evaluation_results/evaluation_bank_v1 \
        --output-root evaluation_results --run-id smoke20261025 \
        [--episodes 3 4] [--mode all] [--device cuda:0] [--seed 20261025] \
        [--chunk-size 256] [--rollout-horizon 100] \
        [--threshold-config thresholds.json] [--build-bank firm.pkl,macro.pkl]
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from analysis.economic_config import AnalysisEconomicConfig
from training.bp_grid_teacher import BPGridTeacher

from .checkpoint_io import (
    EvalCheckpoint,
    EvalCheckpointError,
    compute_param_hashes,
    count_optimizer_steps,
    load_eval_checkpoint,
    param_hash_guard,
    semantic_record,
)
from .comparator import (
    ComparisonResult,
    EpisodeEvalRecord,
    ThresholdConfig,
    compare_episodes,
    save_comparison,
)
from .eval_bank import (
    DEFAULT_BANK_VERSION,
    EvalBankError,
    TAG_REFERENCE_POLICY_DISTRIBUTION,
    build_bank_a,
    build_bank_b,
    build_bank_c,
    build_bank_d,
    build_bank_e,
    compute_config_snapshot_hash,
    content_sha256,
    derive_macro_anchors,
    evaluation_code_commit,
    load_bank,
    save_bank,
)
from .figures import FiguresManifest, render_all_figures
from .firm_bellman import BellmanEvalConfig, BellmanEvalResult, evaluate_firm_bellman
from .firm_policy import PolicyEvalConfig, PolicyEvalResult, evaluate_firm_policy
from .macro_sdf import MacroSdfEvalConfig, MacroSdfEvalResult, evaluate_macro_sdf
from .report import build_convergence_report
from .sim_eval import SimEvalConfig, SimEvalResult, evaluate_simulation

__all__ = ["MODES", "run_convergence_evaluation", "main"]

MODES: Tuple[str, ...] = ("all", "firm", "macro", "sim")

_BLOCKS_BY_MODE: Dict[str, Tuple[str, ...]] = {
    "all": ("bellman", "policy", "macro", "simulation"),
    "firm": ("bellman", "policy"),
    "macro": ("macro",),
    "sim": ("simulation",),
}

# Which fixed bank feeds which evaluator (see eval_bank bank semantics).
_BANK_FOR_BLOCK: Dict[str, str] = {
    "bellman": "B",
    "policy": "C",
    "macro": "D",
    "simulation": "E",
}

REASON_EVALUATOR_FAILED = "EVALUATOR_FAILED"
REASON_BLOCK_NOT_IN_MODE = "NOT_IN_SELECTED_MODE"
REASON_CHECKPOINT_LOAD_FAILED = "CHECKPOINT_LOAD_FAILED"
REASON_PARAM_HASH_MISMATCH = "PARAM_HASH_MISMATCH"
REASON_NOT_EVALUATED = "NOT_EVALUATED"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))
    return str(value)


class _BankCache:
    """Tiny content cache over the fixed bank payloads (hit-rate accounting)."""

    def __init__(self, banks: Mapping[str, Any]):
        self._items = dict(banks)
        self._cache: Dict[str, Any] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any:
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        self._cache[key] = self._items[key]
        return self._cache[key]

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "kind": "fixed_evalbank_payload_cache",
            "cache_items": len(self._cache),
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate": (float(self.hits) / float(total)) if total else None,
            "note": (
                "bank payloads are content-fixed per run; repeated per-episode "
                "requests hit the cache and never reload/re-hash the file"
            ),
        }


@contextmanager
def _track_grid_teachers() -> Iterator[List[Any]]:
    """Record every ``BPGridTeacher`` built during the block (instrumentation).

    The evaluators build their Grid-BP teacher internally; wrapping the single
    class-level factory ``BPGridTeacher.from_hyperparams`` lets the runner sum
    the real ``teacher.forward_stats()`` counters afterwards without touching
    evaluator code.
    """
    created: List[Any] = []
    original_descriptor = BPGridTeacher.__dict__["from_hyperparams"]
    original_fn = original_descriptor.__func__

    def _wrapper(cls, *args, **kwargs):
        teacher = original_fn(cls, *args, **kwargs)
        created.append(teacher)
        return teacher

    BPGridTeacher.from_hyperparams = classmethod(_wrapper)
    try:
        yield created
    finally:
        BPGridTeacher.from_hyperparams = original_descriptor


def _aggregate_teacher_stats(teachers: Sequence[Any]) -> Dict[str, Any]:
    sums: Dict[str, float] = {}
    maxima: Dict[str, float] = {}
    flags: Dict[str, bool] = {}
    plans: List[Dict[str, Any]] = []
    for teacher in teachers:
        stats = teacher.forward_stats()
        for name, value in stats.items():
            if isinstance(value, bool):
                flags[name] = bool(flags.get(name, False) or value)
            elif isinstance(value, (int, float)):
                if name.startswith("bp_max_"):
                    maxima[name] = max(float(maxima.get(name, 0.0)), float(value))
                else:
                    sums[name] = float(sums.get(name, 0.0)) + float(value)
        for plan in teacher.grid_chunk_plans():
            if plan not in plans:
                plans.append(plan)
    return {
        "n_grid_teachers": len(teachers),
        "forward_stats_total": {**sums, **maxima, **flags},
        "forward_stats_semantics": {
            "bp_model_forward_calls": "sum over all Grid-BP teachers (teacher.forward_stats())",
            "bp_child_equity_forward_calls": "child equity forwards in the BP hot path",
            "bp_q_forward_calls": "issuance Q forwards in the BP hot path",
            "bp_max_actual_expanded_states": "max across teachers",
        },
        "grid_chunk_plans": plans,
    }


def _peak_vram_bytes(device: str) -> Optional[int]:
    if not str(device).startswith("cuda"):
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def _is_numeric_array(values: Any) -> bool:
    try:
        array = np.asarray(values)
    except Exception:  # noqa: BLE001 - non-array surface payloads are simply skipped
        return False
    return np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_


def _numeric_result_copy(result: Any) -> Any:
    """Numeric-only metric/surface view for the figure renderer.

    Evaluator results also carry classification entries (strings such as
    ``never_invest`` in ``investment_status`` surfaces/metrics);
    ``figures.render_all_figures`` plots numeric arrays only, so the renderer
    receives a shallow copy keeping numeric metrics/surfaces.  Tables and
    diagnostics (where the classification labels live) pass through unchanged.
    """
    if result is None:
        return None
    numeric_metrics = {
        name: value
        for name, value in (getattr(result, "metrics", {}) or {}).items()
        if isinstance(value, (int, float, bool, np.integer, np.floating))
    }
    numeric_surfaces = {
        name: values
        for name, values in (getattr(result, "surfaces", {}) or {}).items()
        if _is_numeric_array(values)
    }
    return replace(result, metrics=numeric_metrics, surfaces=numeric_surfaces)


def _build_and_save_bank(bank_root: Path, seed: int, spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Build ``evaluation_bank_v1`` banks A-E from source panels and save them.

    ``spec`` keys: ``firm_panel`` (required pickle), ``macro_panel`` (required
    for banks C/D), optional ``checkpoint_ref``, ``exclude_lineages``,
    ``n_rows``, ``n_firms``, ``n_paths``, ``horizon``.  An existing non-empty
    bank root is never overwritten (``save_bank`` raises ``BANK_ROOT_EXISTS``).
    """
    firm_panel = spec.get("firm_panel")
    if not firm_panel:
        raise EvalBankError("MISSING_FIRM_PANEL", "build_bank requires 'firm_panel' (pickle)")
    firm_frame = pd.read_pickle(firm_panel)
    macro_frame = None
    if spec.get("macro_panel"):
        macro_frame = pd.read_pickle(spec["macro_panel"])
    if macro_frame is None:
        raise EvalBankError(
            "MISSING_MACRO_PANEL",
            "build_bank requires 'macro_panel' (banks C/D are macro anchored)",
        )
    config_hash = compute_config_snapshot_hash(AnalysisEconomicConfig.from_current_config())
    economic_config = AnalysisEconomicConfig.from_current_config()
    bank_a = build_bank_a(
        firm_frame,
        macro_frame,
        seed=int(seed),
        config_hash=config_hash,
        n_rows=spec.get("n_rows"),
        checkpoint_ref=spec.get("checkpoint_ref"),
        exclude_lineages=spec.get("exclude_lineages"),
        source_kind=TAG_REFERENCE_POLICY_DISTRIBUTION,
    )
    bank_b = build_bank_b(bank_a, seed=int(seed) + 1, config_hash=config_hash)
    bank_d = build_bank_d(macro_frame, seed=int(seed) + 3, config_hash=config_hash)
    anchors = derive_macro_anchors(bank_d)
    bank_c = build_bank_c(seed=int(seed) + 2, macro_anchors=anchors, config_hash=config_hash)
    bank_e = build_bank_e(
        seed=int(seed) + 4,
        n_firms=int(spec.get("n_firms", 64)),
        n_paths=int(spec.get("n_paths", 4)),
        horizon=int(spec.get("horizon", 20)),
        economic_config=economic_config,
        config_hash=config_hash,
    )
    banks = {"A": bank_a, "B": bank_b, "C": bank_c, "D": bank_d, "E": bank_e}
    manifest = {
        "bank_version": DEFAULT_BANK_VERSION,
        "config_snapshot_hash": config_hash,
        "seeds": {
            "bank_a": int(seed),
            "bank_b": int(seed) + 1,
            "bank_c": int(seed) + 2,
            "bank_d": int(seed) + 3,
            "bank_e": int(seed) + 4,
        },
        "source": {
            "kind": TAG_REFERENCE_POLICY_DISTRIBUTION,
            "checkpoint_ref": spec.get("checkpoint_ref"),
            "train_lineage_exclusion": list(spec.get("exclude_lineages") or []),
        },
    }
    return save_bank(bank_root, banks, manifest)


def _write_csv(frame: Optional[pd.DataFrame], path: Path, *, empty_note: str) -> None:
    if frame is None:
        frame = pd.DataFrame(
            [{"episode": float("nan"), "status": "NOT_EVALUATED", "reason_code": empty_note}]
        )
    frame.to_csv(path, index=False)


def _write_metrics_contract(metrics_dir: Path, comparison: ComparisonResult) -> List[str]:
    """Write the Prompt §13 metrics set (7 CSVs + convergence_status.json)."""
    frames: Dict[str, Optional[pd.DataFrame]] = dict(comparison.metrics or {})
    written: List[str] = []
    mapping = {
        "bellman_metrics.csv": (frames.get("firm_bellman"), "MISSING_BELLMAN_RESULT"),
        "value_drift.csv": (frames.get("value_drift"), "MISSING_BELLMAN_RESULT"),
        "policy_drift.csv": (frames.get("policy_drift"), "MISSING_POLICY_RESULT"),
        "simulation_metrics.csv": (frames.get("simulation"), "MISSING_SIMULATION_RESULT"),
        "distribution_metrics.csv": (frames.get("distribution"), "MISSING_SIMULATION_RESULT"),
    }
    for name, (frame, note) in mapping.items():
        target = metrics_dir / name
        _write_csv(frame, target, empty_note=note)
        written.append(str(target))
    macro = frames.get("macro")
    base_columns = ["episode", "status", "reason_code"]
    for name, prefix in (("macro_metrics.csv", "fc1_"), ("sdf_metrics.csv", "sdf_")):
        target = metrics_dir / name
        if macro is None or macro.empty:
            _write_csv(None, target, empty_note="MISSING_MACRO_RESULT")
        else:
            columns = [c for c in macro.columns if c in base_columns or str(c).startswith(prefix)]
            ordered = ["episode"] + [c for c in columns if c not in base_columns] + [
                c for c in ("status", "reason_code") if c in columns
            ]
            macro.reindex(columns=ordered).to_csv(target, index=False)
        written.append(str(target))
    status_target = metrics_dir / "convergence_status.json"
    written.append(str(status_target))
    return written


def _build_summary_table(comparison: ComparisonResult, episodes: Sequence[int]) -> pd.DataFrame:
    frames: Dict[str, Optional[pd.DataFrame]] = dict(comparison.metrics or {})

    def _value(group: str, episode: int, column: str) -> Any:
        frame = frames.get(group)
        if frame is None or frame.empty or column not in frame.columns:
            return float("nan")
        rows = frame.loc[frame["episode"] == episode]
        if rows.empty:
            return float("nan")
        return rows.iloc[-1][column]

    rows = []
    for episode in episodes:
        rows.append(
            {
                "episode": episode,
                "bellman_fresh_p0_normalized_mae": _value("firm_bellman", episode, "fresh_p0_normalized_mae"),
                "bellman_fresh_pi_normalized_mae": _value("firm_bellman", episode, "fresh_pi_normalized_mae"),
                "bellman_frozen_p0_normalized_mae": _value("firm_bellman", episode, "frozen_p0_normalized_mae"),
                "value_p0_normalized_relative_mae": _value("value_drift", episode, "p0_normalized_relative_mae"),
                "value_Q_claim_relative_mae": _value("value_drift", episode, "Q_claim_relative_mae"),
                "policy_flip_rate": _value("policy_drift", episode, "policy_flip_rate"),
                "bp_star_drift_max": _value("policy_drift", episode, "bp_star_drift_max"),
                "fc1_one_step_hatc_mae": _value("macro", episode, "fc1_one_step_hatc_mae"),
                "sdf_g_max": _value("macro", episode, "sdf_g_max"),
                "sim_resource_feasible_share_min": _value("simulation", episode, "sim_resource_feasible_share_min"),
                "sim_completed_horizon_min": _value("simulation", episode, "sim_completed_horizon_min"),
                "fixed_shock_rollout_drift_abs_max": _value("simulation", episode, "fixed_shock_rollout_drift_abs_max"),
                "distribution_drift_abs_max": _value("distribution", episode, "distribution_drift_abs_max"),
                "macro_trajectory_drift_abs_max": _value("macro", episode, "macro_trajectory_drift_abs_max"),
            }
        )
    return pd.DataFrame(rows)


def _build_macro_state_table(
    results_by_episode: Mapping[int, Mapping[str, Any]], episodes: Sequence[int]
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for episode in episodes:
        macro = (results_by_episode.get(episode) or {}).get("macro")
        metrics = dict(getattr(macro, "metrics", {}) or {}) if macro is not None else {}
        for state in ("recession", "normal", "expansion"):
            key_n = f"fc1_one_step_hatc_n_{state}"
            record: Dict[str, Any] = {
                "episode": episode,
                "macro_state": state,
                "n": metrics.get(key_n, float("nan")),
                "fc1_one_step_hatc_mae": metrics.get(f"fc1_one_step_hatc_mae_{state}", float("nan")),
                "fc1_one_step_hatc_bias": metrics.get(f"fc1_one_step_hatc_bias_{state}", float("nan")),
                "fc1_one_step_lnk_mae": metrics.get(f"fc1_one_step_lnk_mae_{state}", float("nan")),
                "fc1_one_step_lnk_bias": metrics.get(f"fc1_one_step_lnk_bias_{state}", float("nan")),
                "sdf_conditional_mean_abs": metrics.get(f"sdf_{state}_conditional_mean_abs", float("nan")),
                "sdf_M_mean": metrics.get(f"sdf_{state}_M_mean", float("nan")),
                "sdf_M_mean_se": metrics.get(f"sdf_{state}_M_mean_se", float("nan")),
                "sdf_n_parents": metrics.get(f"sdf_{state}_n_parents", float("nan")),
            }
            if macro is None:
                record["status"] = "NOT_EVALUATED"
                record["reason_code"] = "MISSING_MACRO_RESULT"
            elif f"fc1_one_step_hatc_mae_{state}" not in metrics:
                record["status"] = "NOT_EVALUATED"
                record["reason_code"] = "NO_ROWS_IN_MACRO_STATE"
            else:
                record["status"] = "OK"
                record["reason_code"] = "EVALUATED"
            rows.append(record)
    return pd.DataFrame(rows)


def run_convergence_evaluation(
    *,
    checkpoint_paths: Sequence[Union[str, Path]],
    bank_root: Union[str, Path],
    output_root: Union[str, Path],
    run_id: str,
    episodes: Optional[Sequence[int]] = None,
    mode: str = "all",
    device: str = "cpu",
    seed: int = 20261025,
    chunk_size: int = 256,
    rollout_horizon: int = 100,
    threshold_config: Optional[Union[ThresholdConfig, Mapping[str, Any]]] = None,
    build_bank: Optional[Mapping[str, Any]] = None,
    eval_config_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the full Evaluation-Only convergence pipeline; return a summary dict.

    ``eval_config_overrides`` optionally scales individual evaluator
    configurations (keys ``bellman`` / ``policy`` / ``macro`` / ``simulation``,
    values = dataclass field overrides) for smoke runs; production runs use the
    chunk/horizon parameters above and the fixed defaults.
    """
    mode = str(mode)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    paths = [Path(item) for item in checkpoint_paths]
    if not paths:
        raise ValueError("checkpoint_paths must not be empty")
    if episodes is None:
        episode_ids: List[Optional[int]] = [None] * len(paths)
    else:
        episode_ids = [int(item) for item in episodes]
        if len(episode_ids) != len(paths):
            raise ValueError("episodes must align 1:1 with checkpoint_paths")
    blocks = _BLOCKS_BY_MODE[mode]

    bank_root_path = Path(bank_root)
    if build_bank is not None:
        _build_and_save_bank(bank_root_path, int(seed), build_bank)
    loaded = load_bank(bank_root_path, verify=True)
    bank_manifest: Dict[str, Any] = dict(loaded["manifest"])
    banks: Dict[str, Any] = dict(loaded["banks"])
    cache = _BankCache(banks)
    banks_info = dict(bank_manifest.get("banks") or {})
    macro_anchors = (banks.get("C") or {}).get("macro_anchor")
    if not macro_anchors and "D" in banks:
        macro_anchors = derive_macro_anchors(banks["D"])

    # ---- unique output directory (repeated runs never overwrite) ----------
    output_root_path = Path(output_root)
    run_dir = output_root_path / f"convergence_{run_id}"
    counter = 2
    while run_dir.exists() and any(run_dir.iterdir()):
        run_dir = output_root_path / f"convergence_{run_id}__v{counter}"
        counter += 1
    metrics_dir = run_dir / "metrics"
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    report_dir = run_dir / "report"
    diagnostics_dir = run_dir / "diagnostics"
    for directory in (run_dir, run_dir / "bank", metrics_dir, figures_dir, tables_dir, report_dir, diagnostics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if isinstance(threshold_config, ThresholdConfig):
        thresholds = threshold_config
    else:
        thresholds = ThresholdConfig.from_dict(threshold_config)

    # ---- evaluator configurations ---------------------------------------
    overrides = dict(eval_config_overrides or {})
    bellman_config = replace(
        BellmanEvalConfig(parent_chunk_size=int(chunk_size)),
        **dict(overrides.get("bellman") or {}),
    )
    policy_config = replace(
        PolicyEvalConfig(parent_chunk_size=int(chunk_size)),
        **dict(overrides.get("policy") or {}),
    )
    macro_config = replace(
        MacroSdfEvalConfig(device=str(device), sdf_seed=int(seed)),
        **dict(overrides.get("macro") or {}),
    )
    sim_config = replace(
        SimEvalConfig(
            device=str(device),
            horizon=int(rollout_horizon),
            # The t=0 h-star calibration and the capital-waterfall identity need
            # the node resource ledger fields (data/simulate_ts_parallel);
            # eval_config_overrides["simulation"] can still change the mode.
            node_accounting_mode="economic_node_ledger",
        ),
        **dict(overrides.get("simulation") or {}),
    )

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    failures: List[Dict[str, Any]] = []
    skipped_blocks: List[Dict[str, Any]] = []
    results_by_episode: Dict[int, Dict[str, Any]] = {}
    records: List[EpisodeEvalRecord] = []
    checkpoint_entries: List[Dict[str, Any]] = []
    param_hash_records: Dict[str, Any] = {}
    block_times: Dict[str, float] = {name: 0.0 for name in ("bellman", "policy", "macro", "simulation")}
    episode_times: Dict[str, float] = {}
    grid_metadata: Optional[Dict[str, Any]] = None
    all_hashes_unchanged = True

    def _bank_payload_for(block: str) -> Any:
        key = _BANK_FOR_BLOCK[block]
        if key in banks:
            return cache.get(key)
        if block == "bellman" and "A" in banks:
            return cache.get("A")
        raise EvalBankError("MISSING_BANK", f"bank {key} required by block {block} is absent")

    def _grid_metadata_from(cp: EvalCheckpoint) -> Dict[str, Any]:
        hp = cp.hyperparams
        anchors = dict((banks.get("C") or {}).get("macro_anchor") or {})
        reference = dict(next(iter(anchors.values())) or {}) if anchors else {}
        i_max = policy_config.i_max
        return {
            "grid": {
                "b_min": policy_config.b_min,
                "b_max": policy_config.b_max,
                "b_points": int(policy_config.b_points),
                "z_min": policy_config.z_min,
                "z_max": policy_config.z_max,
                "z_points": int(policy_config.z_points),
                "i_min": float(policy_config.i_min),
                "i_max": i_max,
                "i_points": int(policy_config.i_points),
                "eta": [float(value) for value in policy_config.eta_views],
            },
            "reference_state": {
                "x": reference.get("x"),
                "hatcf": reference.get("hatcf"),
                "lnkf": reference.get("lnkf"),
                "hatc_cal": reference.get("hatc_cal"),
                "lnk_cal": reference.get("lnk_cal"),
                "i_low": float(policy_config.i_min),
                "i_mid": (
                    float(policy_config.i_min)
                    if i_max is None
                    else (float(policy_config.i_min) + float(i_max)) / 2.0
                ),
                "i_high": i_max,
                "eta": 1.0,
            },
            "m_mode": "clipped_train_m" if getattr(hp, "pv_use_clipped_m", False) else "raw_sdf_m",
            "m_clamp_bounds": [
                getattr(hp, "pv_m_clamp_min", None),
                getattr(hp, "pv_m_clamp_max", None),
            ],
        }

    def _run_block(block: str, cp: EvalCheckpoint) -> Any:
        if block == "bellman":
            return evaluate_firm_bellman(cp, _bank_payload_for(block), config=bellman_config)
        if block == "policy":
            return evaluate_firm_policy(cp, _bank_payload_for(block), config=policy_config)
        if block == "macro":
            return evaluate_macro_sdf(cp, _bank_payload_for(block), macro_anchors, config=macro_config)
        if block == "simulation":
            return evaluate_simulation(cp, _bank_payload_for(block), config=sim_config)
        raise ValueError(f"unknown block: {block!r}")

    run_started = time.perf_counter()
    with _track_grid_teachers() as tracked_teachers:
        for position, path in enumerate(paths):
            episode_fallback = position + 1
            episode_id = episode_ids[position]
            episode_started = time.perf_counter()
            block_results: Dict[str, Any] = {
                "bellman": None,
                "policy": None,
                "macro": None,
                "simulation": None,
            }
            entry: Dict[str, Any] = {
                "episode": episode_id,
                "checkpoint_path": str(path),
                "status": "OK",
                "reason_code": "EVALUATED",
            }
            try:
                cp = load_eval_checkpoint(path, device=device, label=f"episode_{episode_id}")
            except Exception as exc:  # noqa: BLE001 - recorded, run continues
                entry["episode"] = episode_id if episode_id is not None else episode_fallback
                entry["status"] = "FAILED"
                entry["reason_code"] = REASON_CHECKPOINT_LOAD_FAILED
                entry["detail"] = f"{type(exc).__name__}: {exc}"
                checkpoint_entries.append(entry)
                failures.append(
                    {
                        "episode": entry["episode"],
                        "evaluator": "checkpoint_io.load_eval_checkpoint",
                        "reason_code": REASON_CHECKPOINT_LOAD_FAILED,
                        "detail": entry["detail"],
                    }
                )
                continue

            episode = int(cp.episode if cp.episode is not None else (episode_id or episode_fallback))
            entry["episode"] = episode
            semantic = semantic_record(cp)
            entry.setdefault("checkpoint_sha256", semantic.get("checkpoint_sha256"))
            entry.setdefault("git_commit", semantic.get("git_commit"))
            for name in (
                "label",
                "checkpoint_format",
                "q_parameterization",
                "q_parameterization_source",
                "q_parameterization_assumed_legacy",
                "q_recovery_normalization_mode",
                "q_recovery_normalization_mode_source",
                "q_recovery_normalization_mode_assumed_legacy",
                "value_parameterization",
            ):
                entry[name] = semantic.get(name)

            for module in cp.models.values():
                if isinstance(module, torch.nn.Module):
                    module.eval()
            before = compute_param_hashes(cp.models)
            if grid_metadata is None:
                grid_metadata = _grid_metadata_from(cp)
            try:
                with param_hash_guard(cp):
                    with torch.inference_mode():
                        for block in blocks:
                            started = time.perf_counter()
                            try:
                                block_results[block] = _run_block(block, cp)
                            except Exception as exc:  # noqa: BLE001 - one bad evaluator never kills the run
                                failures.append(
                                    {
                                        "episode": episode,
                                        "evaluator": block,
                                        "reason_code": REASON_EVALUATOR_FAILED,
                                        "detail": f"{type(exc).__name__}: {exc}",
                                    }
                                )
                            finally:
                                block_times[block] = block_times.get(block, 0.0) + (
                                    time.perf_counter() - started
                                )
            except EvalCheckpointError as exc:
                all_hashes_unchanged = False
                entry["status"] = "FAILED"
                entry["reason_code"] = getattr(exc, "reason_code", REASON_PARAM_HASH_MISMATCH)
                entry["detail"] = str(exc)
                failures.append(
                    {
                        "episode": episode,
                        "evaluator": "param_hash_guard",
                        "reason_code": entry["reason_code"],
                        "detail": str(exc),
                    }
                )
            after = compute_param_hashes(cp.models)
            unchanged = before == after
            all_hashes_unchanged = bool(all_hashes_unchanged and unchanged)
            param_hash_records[str(episode)] = {
                "before": dict(before),
                "after": dict(after),
                "unchanged": bool(unchanged),
                "optimizer_steps": count_optimizer_steps(),
            }
            entry["param_hashes_unchanged"] = bool(unchanged)
            entry["optimizer_steps"] = count_optimizer_steps()
            checkpoint_entries.append(entry)

            for block in blocks:
                if block_results.get(block) is None and not any(
                    item["episode"] == episode and item["evaluator"] == block for item in failures
                ):
                    skipped_blocks.append(
                        {
                            "episode": episode,
                            "evaluator": block,
                            "reason_code": REASON_NOT_EVALUATED,
                            "detail": "in-mode block produced no result (evaluation aborted)",
                        }
                    )
            for block in _BLOCKS_BY_MODE["all"]:
                if block not in blocks:
                    skipped_blocks.append(
                        {
                            "episode": episode,
                            "evaluator": block,
                            "reason_code": REASON_BLOCK_NOT_IN_MODE,
                            "detail": f"mode={mode}",
                        }
                    )

            results_by_episode[episode] = block_results
            primary_key = _BANK_FOR_BLOCK[blocks[0]] if blocks else "B"
            if primary_key not in banks:
                primary_key = "B" if "B" in banks else "A"
            primary_info = dict(banks_info.get(primary_key) or {})
            primary_payload = banks.get(primary_key) or {}
            bank_ref = {
                "bank_id": primary_key,
                "bank_version": bank_manifest.get("bank_version"),
                "bank_file_sha256": primary_info.get("sha256"),
                "bank_content_sha256": primary_info.get("content_sha256"),
                "parent_states_sha256": (
                    content_sha256(torch.as_tensor(primary_payload["parent_states"]))
                    if primary_payload.get("parent_states") is not None
                    else None
                ),
                "grid_metadata": grid_metadata,
            }
            records.append(
                EpisodeEvalRecord(
                    episode=episode,
                    semantic=semantic,
                    bellman=block_results.get("bellman"),
                    policy=block_results.get("policy"),
                    macro=block_results.get("macro"),
                    simulation=block_results.get("simulation"),
                    bank_ref=bank_ref,
                    checkpoint=None,
                    seed=int(seed),
                )
            )
            episode_times[str(episode)] = time.perf_counter() - episode_started
            del cp
            gc.collect()
            if str(device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

        teacher_stats = _aggregate_teacher_stats(tracked_teachers)

    # ---- cross-episode comparison ---------------------------------------
    comparison = compare_episodes(records, threshold_config=thresholds)
    comparison_paths = save_comparison(comparison, metrics_dir)
    contract_metrics = _write_metrics_contract(metrics_dir, comparison)
    episodes_sorted = sorted(results_by_episode.keys())

    # ---- figures ---------------------------------------------------------
    figures_started = time.perf_counter()
    figures_manifest: FiguresManifest = render_all_figures(
        bellman_results={
            ep: _numeric_result_copy(res["bellman"]) for ep, res in results_by_episode.items() if res.get("bellman") is not None
        },
        policy_results={
            ep: _numeric_result_copy(res["policy"]) for ep, res in results_by_episode.items() if res.get("policy") is not None
        },
        macro_results={
            ep: _numeric_result_copy(res["macro"]) for ep, res in results_by_episode.items() if res.get("macro") is not None
        },
        sim_results={
            ep: _numeric_result_copy(res["simulation"]) for ep, res in results_by_episode.items() if res.get("simulation") is not None
        },
        comparison=comparison,
        bank=banks,
        output_dir=figures_dir,
    )
    figure_seconds = time.perf_counter() - figures_started

    # ---- tables ----------------------------------------------------------
    summary_table = _build_summary_table(comparison, episodes_sorted)
    summary_table.to_csv(tables_dir / "summary_table.csv", index=False)
    comparability = (comparison.metrics or {}).get("comparability")
    _write_csv(comparability, tables_dir / "episode_comparison.csv", empty_note="NO_RECORD_FOR_EPISODE")
    macro_state_table = _build_macro_state_table(results_by_episode, episodes_sorted)
    macro_state_table.to_csv(tables_dir / "macro_state_table.csv", index=False)

    # ---- diagnostics -----------------------------------------------------
    total_wall = time.perf_counter() - run_started
    runtime = {
        "evaluation_wall_time_seconds": total_wall,
        "wall_time_seconds_by_episode": episode_times,
        "wall_time_seconds_by_block": {**block_times, "figures": figure_seconds},
        "grid_teacher": teacher_stats,
        "peak_vram_bytes": _peak_vram_bytes(device),
        "peak_vram_device": device if str(device).startswith("cuda") else None,
        "cache": cache.stats(),
        "device": device,
        "optimizer_steps": count_optimizer_steps(),
        "evaluation_only": True,
    }
    (diagnostics_dir / "runtime.json").write_text(
        json.dumps(_json_safe(runtime), indent=2, sort_keys=True), encoding="utf-8"
    )

    bank_coverage: Dict[str, Any] = {}
    for bank_key in sorted(banks.keys()):
        payload = banks[bank_key] or {}
        info = dict(banks_info.get(bank_key) or {})
        bank_coverage[bank_key] = {
            "n_rows": payload.get("n_rows", info.get("n_rows")),
            "eta_share": payload.get("eta_share", info.get("eta_share")),
            "source_share": payload.get("source_share", info.get("source_share")),
            "seed": payload.get("seed", info.get("seed")),
            "bank_kind": payload.get("bank_kind"),
            "tags": list(payload.get("tags") or []),
            "sha256": info.get("sha256"),
            "content_sha256": info.get("content_sha256"),
            "participating_episodes": sorted(results_by_episode.keys()),
        }
    data_coverage = {
        "bank_root": str(bank_root_path),
        "bank_version": bank_manifest.get("bank_version"),
        "source": bank_manifest.get("source"),
        "config_snapshot_hash": bank_manifest.get("config_snapshot_hash"),
        "banks": bank_coverage,
        "evaluator_bank_assignment": {
            block: key for block, key in _BANK_FOR_BLOCK.items()
        },
        "participating_episodes": sorted(results_by_episode.keys()),
        "mode": mode,
    }
    (diagnostics_dir / "data_coverage.json").write_text(
        json.dumps(_json_safe(data_coverage), indent=2, sort_keys=True), encoding="utf-8"
    )

    skipped_figures = [
        {
            "item": f"figures/{figure_id}",
            "reason_code": record.missing_reason or REASON_NOT_EVALUATED,
            "detail": record.caption,
        }
        for figure_id, record in figures_manifest.figures.items()
        if record.status != "ok"
    ]
    failure_report = {
        "failures": failures,
        "skipped_blocks": skipped_blocks,
        "skipped_figures": skipped_figures,
        "n_failures": len(failures),
        "evaluation_only": True,
    }
    (diagnostics_dir / "failure_report.json").write_text(
        json.dumps(_json_safe(failure_report), indent=2, sort_keys=True), encoding="utf-8"
    )

    # ---- manifest + report ----------------------------------------------
    code_commit = evaluation_code_commit()
    convergence_status = dict(comparison.convergence_status or {})
    missing_items = [
        {"item": f"evaluator:{item['evaluator']}", "episode": item["episode"],
         "reason_code": item["reason_code"], "detail": item.get("detail")}
        for item in failures
    ] + skipped_figures
    manifest: Dict[str, Any] = {
        "schema_version": "convergence_run_manifest_v1",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "created_utc": _utc_now(),
        "mode": mode,
        "device": device,
        "seed": int(seed),
        "chunk_size": int(chunk_size),
        "rollout_horizon": int(rollout_horizon),
        "episodes": episodes_sorted,
        "code_commit": code_commit,
        "threshold_config": thresholds.to_dict(),
        "required_consecutive_episodes": 3,
        "checkpoints": checkpoint_entries,
        "param_hash": {
            "all_unchanged": bool(all_hashes_unchanged),
            "optimizer_steps": count_optimizer_steps(),
            "per_episode": param_hash_records,
            "evaluation_only": True,
        },
        "bank": {
            "root": str(bank_root_path),
            "bank_version": bank_manifest.get("bank_version"),
            "banks": banks_info,
            "source": bank_manifest.get("source"),
            "config_snapshot_hash": bank_manifest.get("config_snapshot_hash"),
            "evaluator_bank_assignment": dict(_BANK_FOR_BLOCK),
        },
        "evaluation": {
            "blocks_requested": list(blocks),
            "failures": failures,
            "skipped_blocks": skipped_blocks,
            "results_present": {
                str(ep): sorted(name for name, value in res.items() if value is not None)
                for ep, res in results_by_episode.items()
            },
        },
        "figures": figures_manifest.to_dict(),
        "artifacts": {
            "bank_manifest": str(run_dir / "bank" / "bank_manifest.json"),
            "metrics": [Path(item).name for item in contract_metrics]
            + [Path(item).name for item in comparison_paths.values()],
            "tables": ["summary_table.csv", "episode_comparison.csv", "macro_state_table.csv"],
            "report": "report/convergence_report.md",
            "diagnostics": ["runtime.json", "data_coverage.json", "failure_report.json"],
        },
        "missing_items": missing_items,
        "convergence": {
            "convergence_status": convergence_status.get("convergence_status"),
            "converged_candidate": convergence_status.get("converged_candidate"),
            "reason_code": convergence_status.get("reason_code"),
            "max_consecutive_satisfied_count": convergence_status.get(
                "max_consecutive_satisfied_count"
            ),
        },
        "param_hash_guard": "evaluation.convergence.checkpoint_io.param_hash_guard",
    }



    shutil.copy2(bank_root_path / "bank_manifest.json", run_dir / "bank" / "bank_manifest.json")

    report_path = report_dir / "convergence_report.md"
    build_convergence_report(
        manifest=manifest,
        comparison=comparison,
        figures_manifest=figures_manifest,
        results_by_episode=results_by_episode,
        bank_manifest=bank_manifest,
        output_path=report_path,
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "episodes": episodes_sorted,
        "mode": mode,
        "manifest_path": str(run_dir / "manifest.json"),
        "report_path": str(report_path),
        "metrics_dir": str(metrics_dir),
        "figures_dir": str(figures_dir),
        "tables_dir": str(tables_dir),
        "diagnostics_dir": str(diagnostics_dir),
        "bank_root": str(bank_root_path),
        "convergence_status": convergence_status.get("convergence_status"),
        "converged_candidate": convergence_status.get("converged_candidate"),
        "n_failures": len(failures),
        "figures_ok": len(figures_manifest.ok),
        "figures_skipped": len(figures_manifest.skipped),
        "param_hashes_unchanged": bool(all_hashes_unchanged),
        "optimizer_steps": count_optimizer_steps(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _split_paths(values: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for value in values:
        paths.extend(item.strip() for item in str(value).split(",") if item.strip())
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.convergence.run",
        description="Evaluation-Only equilibrium convergence evaluation (never trains).",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="combined checkpoint path (epN_combined.pt); repeatable / comma-separated",
    )
    parser.add_argument("--bank-root", required=True, help="evaluation_bank_v1 root directory")
    parser.add_argument("--output-root", required=True, help="parent of convergence_<run_id>/")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--mode", default="all", choices=MODES)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--rollout-horizon", type=int, default=100)
    parser.add_argument(
        "--threshold-config",
        default=None,
        help="JSON file with {gate_id: threshold|null}; omitted => all gates THRESHOLD_NOT_CONFIGURED",
    )
    parser.add_argument(
        "--build-bank",
        default=None,
        help="FIRM_PKL[,MACRO_PKL] source panels: build evaluation_bank_v1 under --bank-root first",
    )
    args = parser.parse_args(argv)

    checkpoint_paths = _split_paths(args.checkpoint)
    thresholds = None
    if args.threshold_config:
        payload = json.loads(Path(args.threshold_config).read_text(encoding="utf-8"))
        thresholds = ThresholdConfig.from_dict(payload)
    build_spec = None
    if args.build_bank:
        parts = [item.strip() for item in str(args.build_bank).split(",") if item.strip()]
        if not parts:
            parser.error("--build-bank requires at least the firm panel pickle")
        build_spec = {
            "firm_panel": parts[0],
            "macro_panel": parts[1] if len(parts) > 1 else None,
        }
    summary = run_convergence_evaluation(
        checkpoint_paths=checkpoint_paths,
        bank_root=args.bank_root,
        output_root=args.output_root,
        run_id=args.run_id,
        episodes=args.episodes,
        mode=args.mode,
        device=args.device,
        seed=args.seed,
        chunk_size=args.chunk_size,
        rollout_horizon=args.rollout_horizon,
        threshold_config=thresholds,
        build_bank=build_spec,
    )
    print(
        json.dumps(
            {
                "run_dir": summary["run_dir"],
                "manifest_path": summary["manifest_path"],
                "report_path": summary["report_path"],
                "episodes": summary["episodes"],
                "convergence_status": summary["convergence_status"],
                "converged_candidate": summary["converged_candidate"],
                "n_failures": summary["n_failures"],
                "figures_ok": summary["figures_ok"],
                "figures_skipped": summary["figures_skipped"],
                "param_hashes_unchanged": summary["param_hashes_unchanged"],
                "optimizer_steps": summary["optimizer_steps"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
