"""Utilities for deterministic frozen-environment P/Q fixed-point experiments.

This module deliberately contains no Bellman equations.  The experiment runner
uses :class:`training.episode.Episode` for every fitted P/Q update and the
existing evaluation package for contemporaneous residuals.
"""

from __future__ import annotations

import copy
import hashlib
import ast
import inspect
import json
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch

from training.bp_grid_teacher import BPGridTeacher


BATCH_BANK_FORMAT = "frozen_pq_batch_bank_v1"
REQUIRED_PANEL_STAGE = "post_sdf_refresh_pre_pv"
GRID_PQ_MAPPING_COMMIT = "4b236eaacf47b2ac7cb506508e54b21a199e89b0"
RELEVANT_PQ_HYPERPARAMETER_FIELDS = (
    "pv_training_flow",
    "pv_batch_size",
    "pv_eval_epochs",
    "pv_value_scale_mode",
    "pv_value_scale_log_max",
    "pv_bellman_normalize_by_value_scale",
    "pv_mixture_enabled",
    "pv_mixture_ratio",
    "pv_mixture_start_episode",
    "pv_mixture_budget_mode",
    "pv_mixture_sampling_mode",
    "pv_mixture_coverage_group_size",
    "pv_mixture_seed",
    "pv_mixture_stratified_validation",
    "pv_exact_eta_integration_enabled",
    "pv_current_eta_balance_enabled",
    "pv_current_eta1_train_share",
    "pv_current_eta_balance_validation",
    "pv_current_eta_balance_seed",
    "pv_bp_training_mode",
    "pv_bp_head_training_enabled",
    "simulation_bp_action_source",
    "q_parameterization",
    "q_target_refresh_mode",
    "q_bellman_normalize_by_target_scale",
    "q_zero_boundary_epochs",
    "q_default_pretrain_epochs",
    "q_survival_aio_epochs",
    "q_mixed_polish_epochs",
    "q_zero_sample_share",
    "q_default_sample_share",
    "q_survival_sample_share",
    "q_survival_ondist_share",
    "q_claim_coverage_enabled",
    "q_claim_coverage_b_bins",
    "q_claim_coverage_start_episode",
    "bp_grid_coarse_size",
    "bp_grid_fine_size",
    "bp_grid_refine_enabled",
    "entry_mode",
    "entry_spec_version",
)
PRODUCTION_PQ_METHOD_FINGERPRINTS = {
    "_build_pq_value_target_cache": "d14786049af942ca976bbc8eb45690c603fb00162cf99cfd3fc11a0bdf1f0485",
    "_compute_cached_pq_loss": "50f96721a22d3cf89c6cabba9f5717cf8336b4e7830fa26385fe09678fb02bdd",
    "_run_policy_value_evaluation_stage": "6283334391705dd004cad53576ef18f51b1e59ed4ad35a9e93439f2d86269040",
    "_run_q_regime_phase": "a68c436b375a3051f2c97254245b358848d0b6d4b93fcb490574b9fe8beae420",
    "_run_q_regime_training": "bd1d2fd4f56de0e72bc71043ea78bea1727daeee4dc3a3f319ea6be1442b7c0c",
}


def _update_digest(digest: "hashlib._Hash", value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=str):
            digest.update(str(key).encode("utf-8"))
            _update_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence")
        for item in value:
            _update_digest(digest, item)
    else:
        digest.update(repr(value).encode("utf-8"))


def object_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def model_state_sha256(module: torch.nn.Module) -> str:
    return object_sha256(module.state_dict())


def cpu_detached_tree(value: Any) -> Any:
    """Clone a nested training object onto CPU without mutating the source."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_detached_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_detached_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_detached_tree(item) for item in value)
    return value


def relevant_pq_hyperparameter_payload(hyperparams: Any) -> Dict[str, Any]:
    source = hyperparams if isinstance(hyperparams, Mapping) else vars(hyperparams)
    return {
        name: source.get(name)
        for name in RELEVANT_PQ_HYPERPARAMETER_FIELDS
    }


def relevant_pq_hyperparameter_fingerprint(hyperparams: Any) -> str:
    return object_sha256(relevant_pq_hyperparameter_payload(hyperparams))


def make_frozen_batch_bank_payload(
    *,
    train_batches: Sequence[Mapping[str, Any]],
    validation_batches: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    train_cpu = cpu_detached_tree(list(train_batches))
    validation_cpu = cpu_detached_tree(list(validation_batches))
    payload = {
        "format": BATCH_BANK_FORMAT,
        "provenance": dict(provenance),
        "train_batches": train_cpu,
        "validation_batches": validation_cpu,
    }
    payload["hashes"] = {
        "train_batches_sha256": object_sha256(train_cpu),
        "validation_batches_sha256": object_sha256(validation_cpu),
        "dataset_sha256": object_sha256(
            {"train": train_cpu, "validation": validation_cpu}
        ),
    }
    validate_frozen_batch_bank(payload)
    return payload


def save_frozen_batch_bank(
    path: str | Path,
    *,
    train_batches: Sequence[Mapping[str, Any]],
    validation_batches: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = make_frozen_batch_bank_payload(
        train_batches=train_batches,
        validation_batches=validation_batches,
        provenance=provenance,
    )
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return {**payload["hashes"], "path": str(target)}


def checkpoint_provenance(path: str | Path) -> Dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    models = payload.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("checkpoint is missing models")
    hyperparams = payload.get("hyperparams")
    config = payload.get("config_snapshot")
    if not isinstance(hyperparams, Mapping):
        raise ValueError("checkpoint is missing hyperparams")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint is missing config_snapshot")
    if "policy_value" not in models or "sdf_fc1" not in models:
        raise ValueError("checkpoint must contain policy_value and sdf_fc1")
    run_root = checkpoint_path.parent.parent
    return {
        "episode": int(payload.get("episode", -1)),
        "source_run_commit": payload.get("git_commit"),
        "run_root": str(run_root),
        "source_run_identity": str(run_root),
        "policy_value_checkpoint_hash": object_sha256(models["policy_value"]),
        "sdf_fc1_hash": object_sha256(models["sdf_fc1"]),
        "economic_config_hash": object_sha256(dict(config)),
        "hyperparameter_fingerprint": relevant_pq_hyperparameter_fingerprint(
            hyperparams
        ),
        "pv_training_flow": hyperparams.get("pv_training_flow"),
        "q_target_refresh_mode": hyperparams.get("q_target_refresh_mode"),
        "simulation_bp_action_source": hyperparams.get(
            "simulation_bp_action_source"
        ),
        "pv_bp_head_training_enabled": hyperparams.get(
            "pv_bp_head_training_enabled"
        ),
    }


def validate_checkpoint_bank_provenance(
    checkpoint: Mapping[str, Any],
    bank: Mapping[str, Any],
) -> Dict[str, Any]:
    hard_equal = (
        "episode",
        "source_run_commit",
        "source_run_identity",
        "sdf_fc1_hash",
        "economic_config_hash",
        "hyperparameter_fingerprint",
        "pv_training_flow",
        "q_target_refresh_mode",
        "simulation_bp_action_source",
        "pv_bp_head_training_enabled",
    )
    unavailable = [
        key
        for key in hard_equal
        if checkpoint.get(key) is None or bank.get(key) is None
    ]
    if unavailable:
        raise ValueError(
            "checkpoint/bank provenance hard requirements unavailable: "
            + ", ".join(unavailable)
        )
    mismatches = {
        key: {"checkpoint": checkpoint.get(key), "bank": bank.get(key)}
        for key in hard_equal
        if checkpoint.get(key) != bank.get(key)
    }
    if mismatches:
        raise ValueError(f"checkpoint/bank provenance mismatch: {mismatches}")
    if checkpoint.get("episode") != 2:
        raise ValueError("fixed-point checkpoint and bank must both be episode 2")
    if checkpoint.get("simulation_bp_action_source") != "grid":
        raise ValueError("fixed-point checkpoint must use GRID simulation BP actions")
    if checkpoint.get("pv_bp_head_training_enabled") is not False:
        raise ValueError("fixed-point checkpoint must disable BP-head training")
    return {"passed": True, "checked_fields": list(hard_equal)}


def production_pq_method_fingerprints(episode_cls: type) -> Dict[str, str]:
    fingerprints: Dict[str, str] = {}
    for name in PRODUCTION_PQ_METHOD_FINGERPRINTS:
        source = textwrap.dedent(inspect.getsource(getattr(episode_cls, name)))
        node = ast.parse(source).body[0]
        fingerprints[name] = hashlib.sha256(
            ast.dump(node, include_attributes=False).encode("utf-8")
        ).hexdigest()
    return fingerprints


def verify_production_pq_method_fingerprints(episode_cls: type) -> Dict[str, str]:
    actual = production_pq_method_fingerprints(episode_cls)
    mismatches = {
        name: {"expected": expected, "actual": actual.get(name)}
        for name, expected in PRODUCTION_PQ_METHOD_FINGERPRINTS.items()
        if actual.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(
            "production P/Q staged methods differ from the audited 4b236ea mapping: "
            f"{mismatches}"
        )
    return actual


def validate_cycle_teacher_hash(
    *,
    cycle_start_hash: str,
    teacher_hash: str,
    previous_teacher_hash: str | None = None,
    previous_cycle_end_hash: str | None = None,
) -> None:
    if previous_cycle_end_hash is not None and cycle_start_hash != previous_cycle_end_hash:
        raise RuntimeError("cycle X^k does not equal the previous cycle end state")
    if (
        previous_teacher_hash is not None
        and previous_cycle_end_hash is not None
        and previous_cycle_end_hash != previous_teacher_hash
        and teacher_hash == previous_teacher_hash
    ):
        raise RuntimeError("cycle teacher remained pinned to an older X snapshot")
    if teacher_hash != cycle_start_hash:
        raise RuntimeError("cycle teacher hash does not equal the X^k start hash")


def parameter_subset_sha256(
    module: torch.nn.Module,
    parameter_ids: Iterable[int],
) -> str:
    selected = set(int(value) for value in parameter_ids)
    payload = {
        name: parameter.detach()
        for name, parameter in module.named_parameters()
        if id(parameter) in selected
    }
    return object_sha256(payload)


def move_tree(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tree(item, device) for item in value)
    return value


def validate_frozen_batch_bank(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if payload.get("format") != BATCH_BANK_FORMAT:
        raise ValueError(
            f"frozen batch bank format must be {BATCH_BANK_FORMAT!r}, "
            f"got {payload.get('format')!r}"
        )
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("frozen batch bank is missing provenance")
    required = {
        "episode": 2,
        "panel_stage": REQUIRED_PANEL_STAGE,
        "simulation_reused_without_rerun": True,
        "batch_composition_frozen": True,
        "batch_order_frozen": True,
        "validation_split_frozen": True,
    }
    for key, expected in required.items():
        actual = provenance.get(key)
        if actual != expected:
            raise ValueError(
                f"frozen batch bank provenance {key!r} must be {expected!r}, "
                f"got {actual!r}"
            )
    required_provenance_fields = (
        "source_commit",
        "source_run_commit",
        "source_run_identity",
        "run_root",
        "seed",
        "policy_value_checkpoint_hash_pre_pv",
        "sdf_fc1_hash",
        "economic_config_hash",
        "hyperparameter_fingerprint",
        "pv_training_flow",
        "q_target_refresh_mode",
        "simulation_bp_action_source",
        "pv_bp_head_training_enabled",
    )
    unavailable = [
        key for key in required_provenance_fields if provenance.get(key) is None
    ]
    if unavailable:
        raise ValueError(
            "frozen batch bank hard provenance unavailable: "
            + ", ".join(unavailable)
        )
    if provenance.get("source_is_post_pv_output", False):
        raise ValueError(
            "post-PV simulation output cannot reproduce the EP2 pre-PV P/Q environment"
        )
    train = payload.get("train_batches")
    validation = payload.get("validation_batches")
    if not isinstance(train, list) or not train:
        raise ValueError("frozen batch bank must contain non-empty train_batches")
    if not isinstance(validation, list) or not validation:
        raise ValueError("frozen batch bank must contain non-empty validation_batches")
    for split_name, batches in (("train", train), ("validation", validation)):
        for index, batch in enumerate(batches):
            if not isinstance(batch, Mapping) or "parent" not in batch:
                raise ValueError(f"{split_name} batch {index} is missing parent")
            children = batch.get("children")
            if children is None:
                children = [batch.get("child0"), batch.get("child1")]
            if not children or any(not torch.is_tensor(item) for item in children):
                raise ValueError(f"{split_name} batch {index} has invalid children")
            n_parent = int(batch["parent"].shape[0])
            if any(int(item.shape[0]) != n_parent for item in children):
                raise ValueError(f"{split_name} batch {index} child rows do not match parent")
    hashes = {
        "train_batches_sha256": object_sha256(train),
        "validation_batches_sha256": object_sha256(validation),
        "dataset_sha256": object_sha256({"train": train, "validation": validation}),
    }
    recorded = payload.get("hashes", {})
    for key, actual in hashes.items():
        expected = recorded.get(key) if isinstance(recorded, Mapping) else None
        if expected is not None and expected != actual:
            raise ValueError(f"frozen batch bank hash mismatch for {key}")
    return {**hashes, "provenance": dict(provenance)}


def load_frozen_batch_bank(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"frozen batch bank not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("frozen batch bank must contain a mapping")
    metadata = validate_frozen_batch_bank(payload)
    metadata.update({"path": str(path), "file_sha256": file_sha256(path)})
    return (
        move_tree(payload["train_batches"], device),
        move_tree(payload["validation_batches"], device),
        metadata,
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def seed_fixed_mapping(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def distribution_summary(values: np.ndarray) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {key: float("nan") for key in ("mean", "p50", "p90", "p99", "max")}
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def absolute_gap_summary(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    finite = np.isfinite(prediction) & np.isfinite(target)
    return distribution_summary(np.abs(prediction[finite] - target[finite]))


def evaluate_grid_p_fixed_point_residuals(
    episode: Any,
    canonical_batch: Mapping[str, Any],
    *,
    current_model: torch.nn.Module | None = None,
) -> Dict[str, Any]:
    """Evaluate P self-consistency against the production GRID argmax target.

    The current model supplies the left-hand-side P0/PI predictions.  A frozen
    deepcopy of that same model supplies both continuation equity and Q pricing
    inside ``BPGridTeacher``.  This is therefore a contemporaneous GRID
    Bellman residual, not a residual evaluated at the model's BP-head action.
    """
    model = current_model or episode.models["policy_value"]
    model_hash_before = model_state_sha256(model)
    self_teacher = copy.deepcopy(model).to(episode.device)
    self_teacher.eval()
    self_teacher.requires_grad_(False)

    parent_state, children, m_list, *_hashes = (
        episode._policy_batch_hash_components(canonical_batch)
    )
    children, _m_raw, m_list, child_weights = (
        episode._expand_policy_expectation_children(children, m_list, m_list)
    )
    teacher = BPGridTeacher.from_hyperparams(
        self_teacher,
        episode.loss_fns["p0"],
        episode.loss_fns["pi"],
        episode.hyperparams,
        q_target_model=self_teacher,
    )

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            value_fn = getattr(model, "_value_outputs", None)
            if callable(value_fn):
                p0_pred, pi_pred = value_fn(parent_state)
            else:
                output = model(parent_state)
                p0_pred = output["P0"] if isinstance(output, dict) else output.P0
                pi_pred = output["PI"] if isinstance(output, dict) else output.PI

            branch_results: Dict[str, Dict[str, torch.Tensor]] = {}
            for branch in ("p0", "pi"):
                branch_results[branch] = teacher.compute_value_target(
                    parent_state=parent_state,
                    children=children,
                    m_list=m_list,
                    branch=branch,
                    child_weights=child_weights,
                )
    finally:
        model.train(was_training)

    if model_state_sha256(model) != model_hash_before:
        raise RuntimeError("GRID P fixed-point evaluation mutated the current model")

    predictions = {"p0": p0_pred, "pi": pi_pred}
    result: Dict[str, Any] = {}
    summary: Dict[str, float] = {}
    branch_summaries: Dict[str, Dict[str, float]] = {}
    for branch in ("p0", "pi"):
        prediction = predictions[branch].detach().cpu().numpy().reshape(-1)
        value_star = (
            branch_results[branch]["value_star"].detach().cpu().numpy().reshape(-1)
        )
        bp_star = (
            branch_results[branch]["bp_star"].detach().cpu().numpy().reshape(-1)
        )
        residual = prediction - value_star
        stats = distribution_summary(np.abs(residual))
        branch_summaries[branch] = stats
        result[f"{branch}_prediction"] = prediction
        result[f"{branch}_value_star"] = value_star
        result[f"{branch}_bp_star"] = bp_star
        result[f"{branch}_grid_residual"] = residual
        for statistic, value in stats.items():
            summary[f"{branch}_grid_residual_abs_{statistic}"] = value

    summary["P_grid_fixed_point_residual_mean"] = float(
        np.mean([branch_summaries["p0"]["mean"], branch_summaries["pi"]["mean"]])
    )
    summary["P_grid_fixed_point_residual_p90"] = float(
        max(branch_summaries["p0"]["p90"], branch_summaries["pi"]["p90"])
    )
    result["summary"] = summary
    return result


def fixed_rms_scale(*values: np.ndarray, eps: float = 1e-6) -> float:
    flattened = [np.asarray(value, dtype=np.float64).reshape(-1) for value in values]
    combined = np.concatenate(flattened)
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        raise ValueError("cycle-0 scale has no finite observations")
    return max(float(np.sqrt(np.mean(np.square(finite)))), float(eps))


def normalized_rms_error(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    fixed_scale: float,
) -> float:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    finite = np.isfinite(prediction) & np.isfinite(target)
    if not finite.any():
        return float("nan")
    rms = float(np.sqrt(np.mean(np.square(prediction[finite] - target[finite]))))
    return rms / max(float(fixed_scale), 1e-12)


@dataclass(frozen=True)
class DriftResult:
    p_stats: Dict[str, float]
    q_stats: Dict[str, float]
    d_p: float
    d_q: float
    d_joint: float
    delta_vector: np.ndarray


def function_drift(
    previous: Mapping[str, np.ndarray],
    current: Mapping[str, np.ndarray],
    *,
    p_scale: float,
    q_scale: float,
) -> DriftResult:
    p_delta = np.asarray(current["P"], dtype=np.float64) - np.asarray(
        previous["P"], dtype=np.float64
    )
    q_delta = np.asarray(current["Q"], dtype=np.float64) - np.asarray(
        previous["Q"], dtype=np.float64
    )
    p_abs = np.abs(p_delta.reshape(-1))
    q_abs = np.abs(q_delta.reshape(-1))
    d_p = float(np.sqrt(np.mean(np.square(p_delta))) / max(float(p_scale), 1e-12))
    d_q = float(np.sqrt(np.mean(np.square(q_delta))) / max(float(q_scale), 1e-12))
    vector = np.concatenate(
        [p_delta.reshape(-1) / max(float(p_scale), 1e-12),
         q_delta.reshape(-1) / max(float(q_scale), 1e-12)]
    )
    return DriftResult(
        p_stats=distribution_summary(p_abs),
        q_stats=distribution_summary(q_abs),
        d_p=d_p,
        d_q=d_q,
        d_joint=float(np.hypot(d_p, d_q)),
        delta_vector=vector,
    )


def update_cosine(current: np.ndarray, previous: np.ndarray) -> float:
    current = np.asarray(current, dtype=np.float64).reshape(-1)
    previous = np.asarray(previous, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(current) * np.linalg.norm(previous))
    if denominator <= 1e-15:
        return float("nan")
    return float(np.dot(current, previous) / denominator)


def normalized_state_distance(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
    *,
    p_scale: float,
    q_scale: float,
) -> float:
    drift = function_drift(first, second, p_scale=p_scale, q_scale=q_scale)
    return drift.d_joint


def choose_verdict(
    rows: Sequence[Mapping[str, Any]],
    *,
    fit_normalized_tolerance: float,
    distance_tolerance: float,
    residual_improvement_ratio: float,
    boundary_drift_tolerance: float,
    two_cycle_ratio_threshold: float,
) -> Dict[str, Any]:
    """Classify fitted mapping behavior without equating fit and residual gaps."""
    labels = {
        "A": "Convergent",
        "B": "Fitted-update failure",
        "C": "Coverage/projection failure",
        "D": "Oscillatory",
        "E": "Expansive/divergent",
        "F": "Plateau/inconclusive",
    }
    thresholds = {
        "fit_normalized_tolerance": float(fit_normalized_tolerance),
        "distance_tolerance": float(distance_tolerance),
        "residual_improvement_ratio": float(residual_improvement_ratio),
        "boundary_drift_tolerance": float(boundary_drift_tolerance),
        "two_cycle_ratio_threshold": float(two_cycle_ratio_threshold),
    }
    if not rows:
        return {
            "primary": "F",
            "label": labels["F"],
            "reason": "no completed cycles",
            "conditions": {},
            "thresholds": thresholds,
            "statistics": {},
        }

    n_rows = len(rows)
    window = 3 if n_rows >= 6 else max(1, n_rows // 2)
    early = list(rows[:window])
    tail = list(rows[-window:])

    def _values(items: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
        values = np.asarray([float(item.get(key, np.nan)) for item in items])
        return values[np.isfinite(values)]

    def _mean(items: Sequence[Mapping[str, Any]], key: str) -> float:
        values = _values(items, key)
        return float(values.mean()) if values.size else float("nan")

    def _median(items: Sequence[Mapping[str, Any]], key: str) -> float:
        values = _values(items, key)
        return float(np.median(values)) if values.size else float("nan")

    def _ratio(tail_value: float, early_value: float) -> float:
        if not np.isfinite(tail_value) or not np.isfinite(early_value):
            return float("nan")
        return float(tail_value / max(abs(early_value), 1e-12))

    statistics: Dict[str, Any] = {
        "n_completed_cycles": n_rows,
        "early_window": [int(row.get("cycle", index + 1)) for index, row in enumerate(early)],
        "tail_window": [int(row.get("cycle", n_rows - window + index + 1)) for index, row in enumerate(tail)],
    }
    for prefix, key in (
        ("d_joint", "d_joint"),
        ("P_residual", "P_grid_fixed_point_residual_mean"),
        ("Q_residual", "Q_fixed_point_residual_mean"),
        ("boundary_switch", "boundary_switch_share"),
        ("bar_z_drift", "bar_z_mean_abs_drift"),
        ("boundary_drift", "boundary_drift"),
    ):
        early_mean = _mean(early, key)
        tail_mean = _mean(tail, key)
        statistics[f"{prefix}_early_mean"] = early_mean
        statistics[f"{prefix}_tail_mean"] = tail_mean
        statistics[f"{prefix}_tail_over_early"] = _ratio(tail_mean, early_mean)
    statistics.update(
        {
            "rho_tail_median": _median(tail, "rho"),
            "cos_tail_median": _median(tail, "cos_theta"),
            "two_step_tail_median": _median(tail, "two_step_distance"),
            "one_step_tail_median": _median(tail, "d_joint"),
            "P_stage_fit_on_norm_tail_max": float(max(
                _values(tail, "P_stage_fit_on_norm"), default=float("nan")
            )),
            "Q_stage_fit_on_norm_tail_max": float(max(
                _values(tail, "Q_stage_fit_on_norm"), default=float("nan")
            )),
            "P_stage_fit_canonical_norm_tail_max": float(max(
                _values(tail, "P_stage_fit_canonical_norm"), default=float("nan")
            )),
            "Q_stage_fit_canonical_norm_tail_max": float(max(
                _values(tail, "Q_stage_fit_canonical_norm"), default=float("nan")
            )),
        }
    )
    on_fit = np.asarray(
        [
            statistics["P_stage_fit_on_norm_tail_max"],
            statistics["Q_stage_fit_on_norm_tail_max"],
        ],
        dtype=np.float64,
    )
    canonical_fit = np.asarray(
        [
            statistics["P_stage_fit_canonical_norm_tail_max"],
            statistics["Q_stage_fit_canonical_norm_tail_max"],
        ],
        dtype=np.float64,
    )
    stage_fit_controlled = bool(
        np.isfinite(on_fit).all()
        and float(on_fit.max()) <= float(fit_normalized_tolerance)
    )
    joint_distance_shrinking = bool(
        np.isfinite(statistics["d_joint_tail_mean"])
        and (
            statistics["d_joint_tail_mean"] <= distance_tolerance
            or statistics["d_joint_tail_over_early"] < residual_improvement_ratio
        )
    )
    tail_rho_below_one = bool(
        np.isfinite(statistics["rho_tail_median"])
        and statistics["rho_tail_median"] < 1.0
    )
    p_residual_improving = bool(
        np.isfinite(statistics["P_residual_tail_over_early"])
        and statistics["P_residual_tail_over_early"] < residual_improvement_ratio
    )
    q_residual_improving = bool(
        np.isfinite(statistics["Q_residual_tail_over_early"])
        and statistics["Q_residual_tail_over_early"] < residual_improvement_ratio
    )
    two_step_ratio = _ratio(
        statistics["two_step_tail_median"],
        statistics["one_step_tail_median"],
    )
    statistics["two_step_over_one_step_tail"] = two_step_ratio
    oscillatory = bool(
        np.isfinite(statistics["cos_tail_median"])
        and statistics["cos_tail_median"] < -0.5
        and np.isfinite(two_step_ratio)
        and two_step_ratio < two_cycle_ratio_threshold
    )
    boundary_switch_ok = bool(
        np.isfinite(statistics["boundary_switch_tail_mean"])
        and (
            statistics["boundary_switch_tail_mean"] <= boundary_drift_tolerance
            or statistics["boundary_switch_tail_over_early"] < residual_improvement_ratio
        )
    )
    bar_z_ok = bool(
        np.isfinite(statistics["bar_z_drift_tail_mean"])
        and (
            statistics["bar_z_drift_tail_mean"] <= boundary_drift_tolerance
            or statistics["bar_z_drift_tail_over_early"] < residual_improvement_ratio
        )
    )
    boundary_stabilizing = boundary_switch_ok and bar_z_ok
    canonical_bad = bool(
        np.isfinite(canonical_fit).all()
        and float(canonical_fit.max()) > float(fit_normalized_tolerance) * 5.0
    )
    expansive = bool(
        np.isfinite(statistics["rho_tail_median"])
        and statistics["rho_tail_median"] > 1.0
        and statistics["d_joint_tail_over_early"] > 1.0
        and statistics["P_residual_tail_over_early"] > 1.0
        and statistics["Q_residual_tail_over_early"] > 1.0
    )
    conditions = {
        "stage_fit_controlled": stage_fit_controlled,
        "joint_distance_shrinking": joint_distance_shrinking,
        "tail_rho_below_one": tail_rho_below_one,
        "p_residual_improving": p_residual_improving,
        "q_residual_improving": q_residual_improving,
        "no_two_cycle": not oscillatory,
        "boundary_stabilizing": boundary_stabilizing,
        "canonical_fit_materially_worse": canonical_bad,
        "expansive": expansive,
    }

    if not stage_fit_controlled:
        primary = "B"
        reason = "on-distribution normalized fitted-stage error is not controlled"
    elif canonical_bad and not boundary_stabilizing:
        primary = "C"
        reason = "on-distribution fit is controlled but canonical projection and boundary drift are not"
    elif oscillatory:
        primary = "D"
        reason = "tail updates reverse direction and two-step distance is comparatively small"
    elif expansive:
        primary = "E"
        reason = "controlled fitted updates expand joint distance and fixed-point residuals"
    elif all(
        conditions[key]
        for key in (
            "stage_fit_controlled",
            "joint_distance_shrinking",
            "tail_rho_below_one",
            "p_residual_improving",
            "q_residual_improving",
            "no_two_cycle",
            "boundary_stabilizing",
        )
    ):
        primary = "A"
        reason = "all fitted-update, contraction, residual, oscillation, and boundary conditions pass"
    else:
        primary = "F"
        reason = "fixed-point evidence plateaus or remains incomplete"
    return {
        "primary": primary,
        "label": labels[primary],
        "reason": reason,
        "conditions": conditions,
        "thresholds": thresholds,
        "statistics": statistics,
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
