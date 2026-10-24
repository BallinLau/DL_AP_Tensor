"""Evaluation-only checkpoint loading for the equilibrium convergence evaluator.

This module is the single entry point through which the convergence evaluator is
allowed to load a checkpoint.  It builds on
``analysis.checkpoint_loader.load_analysis_checkpoint`` (which performs the
semantic validation of ``value_parameterization`` and rebuilds
``policy_value`` from ``policy_value_model_spec``) and adds the guarantees the
evaluation protocol requires:

* **combined-only**: only ``epN_combined.pt`` style payloads (a dict with a
  ``models`` mapping) are accepted.  A bare ``epN_policy_value.pt`` state_dict
  cannot certify its own Q semantics (``direct`` vs legacy ``b_times_unit`` have
  an identically shaped q-head), so it is rejected with an explicit reason code.
* **parameter hashes**: a stable sha256 is computed per model state_dict
  (per-parameter tensor bytes) right after loading; ``param_hash_guard`` guards
  an evaluation block against accidental parameter updates (evaluation is
  Evaluation-Only and must leave every parameter bit-identical).
* **semantic record**: every checkpoint carries the Q parameterization, the Q
  semantics recorded fields, the recovery normalization mode and the value
  parameterization, with legacy checkpoints explicitly annotated instead of
  silently defaulted.
* **cross-checkpoint semantics**: ``check_semantic_compatibility`` reports
  per-field diffs (architecture / Q parameterization / value scaling /
  economic parameters) before any cross-episode comparison is performed.

No optimizer is ever constructed here: ``count_optimizer_steps`` is the recorded
constant 0 of the evaluation side.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch

from analysis.checkpoint_loader import load_analysis_checkpoint
from analysis.economic_config import AnalysisEconomicConfig
from config import HyperParams


class EvalCheckpointError(ValueError):
    """Evaluation-side checkpoint error carrying a stable machine reason code."""

    def __init__(self, reason_code: str, message: str):
        super().__init__(f"[{reason_code}] {message}")
        self.reason_code = reason_code
        self.message = message


# Reason codes used by this module.
REASON_NOT_COMBINED = "NOT_COMBINED_CHECKPOINT"
REASON_MISSING_MODEL = "MISSING_REQUIRED_MODEL"
REASON_PARAM_HASH_MISMATCH = "PARAM_HASH_MISMATCH"
REASON_SEMANTIC_MISMATCH = "SEMANTIC_MISMATCH"
REASON_EMPTY_CHECKPOINT_LIST = "EMPTY_CHECKPOINT_LIST"


def state_dict_sha256(state_dict: Dict[str, Any]) -> str:
    """Stable sha256 over every parameter of a state_dict.

    Mirrors ``evaluation.full_run_diagnostics.model_state_hash`` but covers any
    state_dict (including ``sdf_fc1``/``fc2``/``firm_target``) and is therefore
    used for all models of a checkpoint.
    """
    digest = hashlib.sha256()
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        digest.update(name.encode("utf-8"))
        if torch.is_tensor(tensor):
            value = tensor.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("utf-8"))
            digest.update(value.numpy().tobytes())
        else:
            digest.update(repr(tensor).encode("utf-8"))
    return digest.hexdigest()


def model_state_sha256(model: torch.nn.Module) -> str:
    return state_dict_sha256(model.state_dict())


def compute_param_hashes(models: Dict[str, torch.nn.Module]) -> Dict[str, str]:
    return {name: model_state_sha256(module) for name, module in sorted(models.items())}


def count_optimizer_steps() -> int:
    """Optimizer steps performed by the evaluation side: always 0.

    The evaluator never builds an optimizer; this constant is recorded in every
    semantic record so that evaluation outputs can prove Evaluation-Only usage.
    """
    return 0


@dataclass
class EvalCheckpoint:
    models: Dict[str, torch.nn.Module]
    hyperparams: HyperParams
    economic_config: AnalysisEconomicConfig
    metadata: Dict[str, Any]
    episode: Optional[int]
    checkpoint_path: str
    param_hashes: Dict[str, str]


def _load_raw_payload(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_eval_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    label: Optional[str] = None,
) -> EvalCheckpoint:
    """Load a combined checkpoint for evaluation.

    Only combined checkpoints (``epN_combined.pt``, payload dict containing
    ``models``) are accepted.  Bare ``epN_policy_value.pt`` state_dicts are
    rejected with reason code ``NOT_COMBINED_CHECKPOINT`` because they cannot
    self-certify their Q semantics (see module docstring).
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise EvalCheckpointError(
            "CHECKPOINT_NOT_FOUND", f"checkpoint file does not exist: {checkpoint_path}"
        )
    payload = _load_raw_payload(checkpoint_path)
    if not isinstance(payload, dict) or "models" not in payload:
        raise EvalCheckpointError(
            REASON_NOT_COMBINED,
            "only combined checkpoints (payload dict with a 'models' mapping) are "
            f"accepted: {checkpoint_path}. A bare state_dict cannot self-certify its "
            "Q semantics (direct vs legacy b_times_unit share the q-head shape), so "
            "epN_policy_value.pt style files must not be used for evaluation.",
        )
    model_states = payload.get("models")
    if not isinstance(model_states, dict):
        raise EvalCheckpointError(
            REASON_NOT_COMBINED,
            f"checkpoint 'models' entry must be a mapping: {checkpoint_path}",
        )
    for required in ("policy_value", "sdf_fc1"):
        if required not in model_states:
            raise EvalCheckpointError(
                REASON_MISSING_MODEL,
                f"combined checkpoint is missing models[{required!r}]: {checkpoint_path}",
            )

    loaded = load_analysis_checkpoint(checkpoint_path, device=device, label=label)
    metadata = dict(loaded.metadata)
    metadata.setdefault("git_commit", payload.get("git_commit"))
    metadata.setdefault("episode", payload.get("episode"))
    param_hashes = compute_param_hashes(loaded.models)
    episode = payload.get("episode")
    return EvalCheckpoint(
        models=loaded.models,
        hyperparams=loaded.hyperparams,
        economic_config=loaded.economic_config,
        metadata=metadata,
        episode=None if episode is None else int(episode),
        checkpoint_path=str(checkpoint_path),
        param_hashes=param_hashes,
    )


def semantic_record(cp: EvalCheckpoint) -> Dict[str, Any]:
    """Full semantic record of a checkpoint for convergence-surface metadata.

    Legacy checkpoints (missing ``q_parameterization`` in the model spec and/or
    missing ``q_recovery_normalization_mode`` in hyperparams) are annotated with
    the assumed legacy semantics instead of being silently defaulted.
    """
    metadata = cp.metadata
    spec = metadata.get("policy_value_model_spec")
    spec_records_q = isinstance(spec, dict) and "q_parameterization" in spec

    resolved_q = metadata.get("q_parameterization")
    resolved_q = str(resolved_q) if resolved_q else "b_times_unit"
    if spec_records_q:
        q_source = "checkpoint_spec"
        q_assumed_legacy = False
    else:
        q_source = "q_parameterization_assumed_legacy"
        q_assumed_legacy = True

    recorded_q_fields = list(metadata.get("q_semantics_recorded_fields") or [])
    recorded_recovery = "q_recovery_normalization_mode" in recorded_q_fields
    recovery_value = getattr(cp.hyperparams, "q_recovery_normalization_mode", None)
    if recorded_recovery and recovery_value is not None:
        q_recovery_mode = str(recovery_value)
        q_recovery_source = "checkpoint_hyperparams"
        q_recovery_assumed_legacy = False
    else:
        q_recovery_mode = "legacy_b_times_unit"
        q_recovery_source = "q_recovery_normalization_mode_assumed_legacy"
        q_recovery_assumed_legacy = True

    value_parameterization = dict((metadata.get("value_parameterization") or {}).get("current") or {})
    if not value_parameterization:
        value_parameterization = {
            "mode": "none",
            "scale_formula": "1",
            "bellman_normalization": False,
            "log_max": 20.0,
        }

    return {
        "label": metadata.get("label"),
        "checkpoint_path": cp.checkpoint_path,
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "episode": cp.episode,
        "git_commit": metadata.get("git_commit"),
        "checkpoint_format": metadata.get("checkpoint_format"),
        "loaded_model_keys": list(metadata.get("loaded_model_keys") or sorted(cp.models.keys())),
        "q_parameterization": resolved_q,
        "q_parameterization_source": q_source,
        "q_parameterization_assumed_legacy": q_assumed_legacy,
        "q_semantics_recorded_fields": recorded_q_fields,
        "q_recovery_normalization_mode": q_recovery_mode,
        "q_recovery_normalization_mode_source": q_recovery_source,
        "q_recovery_normalization_mode_assumed_legacy": q_recovery_assumed_legacy,
        "value_parameterization": value_parameterization,
        "policy_value_model_spec": spec,
        "economic_config": cp.economic_config.to_dict(),
        "param_hashes": dict(cp.param_hashes),
        "optimizer_steps": count_optimizer_steps(),
        "evaluation_only": True,
        "missing_optional_fields": list(metadata.get("missing_optional_fields") or []),
    }


def _compatibility_view(cp: EvalCheckpoint) -> Dict[str, Any]:
    """Flat per-field view used for cross-checkpoint semantic diffs."""
    record = semantic_record(cp)
    view: Dict[str, Any] = {
        "q_parameterization": record["q_parameterization"],
        "q_parameterization_source": record["q_parameterization_source"],
        "q_recovery_normalization_mode": record["q_recovery_normalization_mode"],
        "policy_value_model_spec": record["policy_value_model_spec"],
    }
    for key in ("mode", "scale_formula", "bellman_normalization", "log_max"):
        view[f"value_parameterization.{key}"] = record["value_parameterization"].get(key)
    for name in sorted(record["economic_config"].keys()):
        view[f"economic_config.{name}"] = record["economic_config"][name]
    return view


def check_semantic_compatibility(
    cps: Sequence[EvalCheckpoint],
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    """Compare checkpoints on architecture / Q parameterization / value scaling /
    economic parameters and report per-field diffs.

    Inconsistencies are only reported (``strict=False`` default) so that the
    caller can decide how to surface them; with ``strict=True`` the first
    inconsistency raises ``EvalCheckpointError(SEMANTIC_MISMATCH)``.
    """
    if not cps:
        raise EvalCheckpointError(
            REASON_EMPTY_CHECKPOINT_LIST,
            "check_semantic_compatibility requires at least one checkpoint",
        )
    labels: List[str] = []
    views: List[Dict[str, Any]] = []
    for index, cp in enumerate(cps):
        label = cp.metadata.get("label") or f"checkpoint_{index}"
        labels.append(str(label))
        views.append(_compatibility_view(cp))

    fields: Dict[str, Any] = {}
    diffs: Dict[str, Any] = {}
    checked_fields = sorted({key for view in views for key in view})
    for field_name in checked_fields:
        values = {labels[i]: views[i].get(field_name) for i in range(len(views))}
        serialized = {
            label: repr(value) if not isinstance(value, (bool, int, float, str, type(None))) else value
            for label, value in values.items()
        }
        consistent = len({repr(v) for v in serialized.values()}) == 1
        fields[field_name] = {"consistent": consistent, "values": values}
        if not consistent:
            diffs[field_name] = values

    result = {
        "compatible": not diffs,
        "n_checkpoints": len(cps),
        "labels": labels,
        "checked_fields": checked_fields,
        "fields": fields,
        "diffs": diffs,
    }
    if strict and diffs:
        raise EvalCheckpointError(
            REASON_SEMANTIC_MISMATCH,
            "checkpoints are not semantically compatible on fields: "
            + ", ".join(sorted(diffs.keys())),
        )
    return result


@contextmanager
def param_hash_guard(cp: EvalCheckpoint) -> Iterator[EvalCheckpoint]:
    """Guard an evaluation block: parameters must be bit-identical on exit.

    Evaluation is Evaluation-Only; if any model parameter changed while the
    guard was active (an accidental optimizer step, an in-place update, ...),
    ``EvalCheckpointError(PARAM_HASH_MISMATCH)`` is raised listing the models
    whose state_dict hash changed.
    """
    before = compute_param_hashes(cp.models)
    try:
        yield cp
    finally:
        after = compute_param_hashes(cp.models)
    changed = sorted(name for name, digest in before.items() if after.get(name) != digest)
    if changed:
        raise EvalCheckpointError(
            REASON_PARAM_HASH_MISMATCH,
            "model parameters changed during evaluation (evaluation must not "
            f"update parameters): {changed}",
        )
