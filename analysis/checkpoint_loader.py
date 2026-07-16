from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from config import HyperParams
from experiments.run_utils import build_models
from .economic_config import AnalysisEconomicConfig


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _state_hash(state: Optional[Dict[str, torch.Tensor]]) -> Optional[str]:
    if state is None:
        return None
    h = hashlib.sha256()
    for key in sorted(state.keys()):
        tensor = state[key]
        h.update(key.encode("utf-8"))
        if torch.is_tensor(tensor):
            arr = tensor.detach().cpu().contiguous()
            h.update(str(arr.dtype).encode("utf-8"))
            h.update(str(tuple(arr.shape)).encode("utf-8"))
            h.update(arr.numpy().tobytes())
        else:
            h.update(repr(tensor).encode("utf-8"))
    return h.hexdigest()


def _hyperparams_from_dict(values: Dict[str, Any]) -> HyperParams:
    hp = HyperParams()
    for key, value in values.items():
        if hasattr(hp, key):
            setattr(hp, key, value)
    return hp


def _load_hyperparams_json(path: Path) -> HyperParams:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "hyperparams" in payload and isinstance(payload["hyperparams"], dict):
        payload = payload["hyperparams"]
    if not isinstance(payload, dict):
        raise ValueError(f"hyperparams_json must contain a JSON object: {path}")
    return _hyperparams_from_dict(payload)


@dataclass
class AnalysisCheckpoint:
    models: Dict[str, torch.nn.Module]
    hyperparams: HyperParams
    economic_config: AnalysisEconomicConfig
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class CheckpointSpec:
    checkpoint_path: Optional[str | Path] = None
    policy_checkpoint: Optional[str | Path] = None
    sdf_checkpoint: Optional[str | Path] = None
    hyperparams_json: Optional[str | Path] = None
    config_json: Optional[str | Path] = None
    label: Optional[str] = None


def _load_config_payload(
    *,
    payload: Optional[Dict[str, Any]],
    config_json: Optional[str | Path],
    allow_current_config: bool,
) -> tuple[AnalysisEconomicConfig, str]:
    if config_json is not None:
        return AnalysisEconomicConfig.from_json(config_json), "json"
    if isinstance(payload, dict) and "config_snapshot" in payload:
        snapshot = payload["config_snapshot"]
        if not isinstance(snapshot, dict):
            raise ValueError("checkpoint config_snapshot must be a dictionary")
        return AnalysisEconomicConfig.from_dict(snapshot), "checkpoint"
    if allow_current_config:
        return AnalysisEconomicConfig.from_current_config(), "current_explicit"
    raise ValueError(
        "checkpoint analysis requires config_snapshot or config_json; "
        "set allow_current_config=True only for explicit current-config analysis"
    )


def load_analysis_checkpoint(
    checkpoint_path: Optional[str | Path] = None,
    *,
    policy_checkpoint: Optional[str | Path] = None,
    sdf_checkpoint: Optional[str | Path] = None,
    hyperparams_json: Optional[str | Path] = None,
    config_json: Optional[str | Path] = None,
    device: torch.device | str = "cpu",
    allow_default_hyperparams: bool = False,
    allow_current_config: bool = False,
    m_source: str = "sdf_fc1",
    label: Optional[str] = None,
) -> AnalysisCheckpoint:
    if m_source != "sdf_fc1":
        raise ValueError("Only m_source='sdf_fc1' is supported by convergence surface analysis")
    device = torch.device(device)
    models = build_models(device)
    missing_optional_fields = []
    checkpoint_format = None
    checkpoint_hash = None
    policy_state = None
    sdf_state = None
    firm_target_state = None
    payload_for_config: Optional[Dict[str, Any]] = None
    policy_path_for_meta = policy_checkpoint
    sdf_path_for_meta = sdf_checkpoint

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        policy_path_for_meta = checkpoint_path
        checkpoint_hash = _sha256_file(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location=device)
        payload_for_config = payload if isinstance(payload, dict) else None
        if isinstance(payload, dict) and "models" in payload:
            checkpoint_format = "combined"
            model_states = payload["models"]
            if "policy_value" not in model_states:
                raise ValueError("combined checkpoint is missing models['policy_value']")
            if "sdf_fc1" not in model_states and m_source == "sdf_fc1":
                raise ValueError("combined checkpoint is missing models['sdf_fc1']")
            policy_state = model_states["policy_value"]
            sdf_state = model_states.get("sdf_fc1")
            firm_target_state = model_states.get("firm_target")
            if firm_target_state is None:
                missing_optional_fields.append("models.firm_target")
            hp_payload = payload.get("hyperparams")
            if hp_payload is None:
                if not allow_default_hyperparams:
                    raise ValueError("combined checkpoint is missing hyperparams")
                hyperparams = HyperParams()
                hp_source = "default"
            elif isinstance(hp_payload, HyperParams):
                hyperparams = hp_payload
                hp_source = "checkpoint"
            elif isinstance(hp_payload, dict):
                hyperparams = _hyperparams_from_dict(hp_payload)
                hp_source = "checkpoint"
            else:
                raise ValueError("unsupported hyperparams payload in combined checkpoint")
        else:
            checkpoint_format = "raw_policy_state_dict"
            policy_state = payload
            if m_source == "sdf_fc1" and sdf_checkpoint is None:
                raise ValueError("raw policy state_dict requires sdf_checkpoint when m_source='sdf_fc1'")
            if hyperparams_json is None and not allow_default_hyperparams:
                raise ValueError("raw policy state_dict requires hyperparams_json unless allow_default_hyperparams=True")
            if sdf_checkpoint is not None:
                sdf_checkpoint = Path(sdf_checkpoint)
                sdf_state = torch.load(sdf_checkpoint, map_location=device)
            if hyperparams_json is not None:
                hyperparams = _load_hyperparams_json(Path(hyperparams_json))
                hp_source = "json"
            else:
                hyperparams = HyperParams()
                hp_source = "default"
    else:
        if policy_checkpoint is None:
            raise ValueError("Either checkpoint_path or policy_checkpoint must be provided")
        checkpoint_format = "raw_policy_state_dict"
        policy_checkpoint = Path(policy_checkpoint)
        checkpoint_hash = _sha256_file(policy_checkpoint)
        policy_state = torch.load(policy_checkpoint, map_location=device)
        if m_source == "sdf_fc1" and sdf_checkpoint is None:
            raise ValueError("raw policy state_dict requires sdf_checkpoint when m_source='sdf_fc1'")
        if hyperparams_json is None and not allow_default_hyperparams:
            raise ValueError("raw policy state_dict requires hyperparams_json unless allow_default_hyperparams=True")
        if sdf_checkpoint is not None:
            sdf_checkpoint = Path(sdf_checkpoint)
            sdf_path_for_meta = sdf_checkpoint
            sdf_state = torch.load(sdf_checkpoint, map_location=device)
        if hyperparams_json is not None:
            hyperparams = _load_hyperparams_json(Path(hyperparams_json))
            hp_source = "json"
        else:
            hyperparams = HyperParams()
            hp_source = "default"

    try:
        models["policy_value"].load_state_dict(policy_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError("policy_value state_dict is incompatible with current model architecture") from exc
    if sdf_state is not None:
        try:
            models["sdf_fc1"].load_state_dict(sdf_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError("sdf_fc1 state_dict is incompatible with current model architecture") from exc
    elif m_source == "sdf_fc1":
        raise ValueError("checkpoint is missing sdf_fc1 but m_source='sdf_fc1'")

    if firm_target_state is not None:
        firm_target = build_models(device)["policy_value"]
        try:
            firm_target.load_state_dict(firm_target_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError("firm_target state_dict is incompatible with current model architecture") from exc
        models["firm_target"] = firm_target

    economic_config, config_source = _load_config_payload(
        payload=payload_for_config,
        config_json=config_json,
        allow_current_config=allow_current_config,
    )

    resolved_label = label
    if resolved_label is None:
        stem_source = checkpoint_path or policy_checkpoint
        stem = Path(stem_source).stem if stem_source is not None else "checkpoint"
        suffix = (checkpoint_hash or _state_hash(policy_state) or "unknown")[:8]
        resolved_label = f"{stem}_{suffix}"

    metadata = {
        "label": resolved_label,
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "policy_checkpoint_path": None if policy_path_for_meta is None else str(policy_path_for_meta),
        "sdf_checkpoint_path": None if sdf_path_for_meta is None else str(sdf_path_for_meta),
        "checkpoint_sha256": checkpoint_hash,
        "policy_state_hash": _state_hash(policy_state),
        "sdf_state_hash": _state_hash(sdf_state),
        "checkpoint_format": checkpoint_format,
        "loaded_model_keys": ["policy_value"] + (["sdf_fc1"] if sdf_state is not None else []),
        "hyperparameter_source": hp_source,
        "config_source": config_source,
        "config_snapshot": economic_config.to_dict(),
        "missing_optional_fields": missing_optional_fields,
    }
    return AnalysisCheckpoint(
        models=models,
        hyperparams=hyperparams,
        economic_config=economic_config,
        metadata=metadata,
    )


def load_analysis_checkpoint_spec(
    spec: CheckpointSpec,
    *,
    device: torch.device | str = "cpu",
    allow_default_hyperparams: bool = False,
    allow_current_config: bool = False,
) -> AnalysisCheckpoint:
    return load_analysis_checkpoint(
        spec.checkpoint_path,
        policy_checkpoint=spec.policy_checkpoint,
        sdf_checkpoint=spec.sdf_checkpoint,
        hyperparams_json=spec.hyperparams_json,
        config_json=spec.config_json,
        device=device,
        allow_default_hyperparams=allow_default_hyperparams,
        allow_current_config=allow_current_config,
        label=spec.label,
    )
