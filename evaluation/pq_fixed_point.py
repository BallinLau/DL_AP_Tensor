"""Utilities for deterministic frozen-environment P/Q fixed-point experiments.

This module deliberately contains no Bellman equations.  The experiment runner
uses :class:`training.episode.Episode` for every fitted P/Q update and the
existing evaluation package for contemporaneous residuals.
"""

from __future__ import annotations

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


BATCH_BANK_FORMAT = "frozen_pq_batch_bank_v1"
REQUIRED_PANEL_STAGE = "post_sdf_refresh_pre_pv"
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
    fit_tolerance: float,
    distance_tolerance: float,
) -> tuple[str, str]:
    if not rows:
        return "F", "no completed cycles"
    fit_keys = (
        "P_stage_fit_on_mean",
        "Q_stage_fit_on_mean",
    )
    tail = list(rows[max(0, len(rows) // 2):])
    fit_values = [float(row.get(key, np.nan)) for row in tail for key in fit_keys]
    if not fit_values or not np.isfinite(fit_values).all() or max(fit_values) > fit_tolerance:
        return "B", "on-distribution fitted-stage error is not controlled"
    canonical_fit = [
        float(row.get(key, np.nan))
        for row in tail
        for key in ("P_stage_fit_canonical_mean", "Q_stage_fit_canonical_mean")
    ]
    if np.isfinite(canonical_fit).all() and max(canonical_fit) > fit_tolerance * 5.0:
        return "C", "on-distribution fit is controlled but canonical fit is materially worse"
    distances = np.asarray([float(row.get("d_joint", np.nan)) for row in rows])
    rhos = np.asarray([float(row.get("rho", np.nan)) for row in tail])
    cosines = np.asarray([float(row.get("cos_theta", np.nan)) for row in tail])
    two_step = np.asarray([float(row.get("two_step_distance", np.nan)) for row in tail])
    if np.isfinite(distances[-1]) and distances[-1] <= distance_tolerance:
        return "A", "joint function distance reached the configured fixed-point tolerance"
    finite_cos = cosines[np.isfinite(cosines)]
    finite_two = two_step[np.isfinite(two_step)]
    if finite_cos.size and np.median(finite_cos) < -0.5:
        if finite_two.size and np.median(finite_two) < np.nanmedian(distances[-len(finite_two):]):
            return "D", "updates reverse direction and two-step distance is smaller"
    finite_rho = rhos[np.isfinite(rhos)]
    if finite_rho.size and np.median(finite_rho) > 1.0 and distances[-1] > distances[0]:
        return "E", "tail contraction ratios exceed one and joint distance grows"
    if finite_rho.size and np.median(finite_rho) < 1.0 and distances[-1] < distances[0]:
        return "A", "tail contraction ratios are below one and joint distance declines"
    return "F", "joint distance and residual evidence plateau or remain inconclusive"


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
