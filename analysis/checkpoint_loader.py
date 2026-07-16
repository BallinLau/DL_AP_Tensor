from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from config import HyperParams
from experiments.run_utils import build_models


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
    metadata: Dict[str, Any]


def load_analysis_checkpoint(
    checkpoint_path: Optional[str | Path] = None,
    *,
    policy_checkpoint: Optional[str | Path] = None,
    sdf_checkpoint: Optional[str | Path] = None,
    hyperparams_json: Optional[str | Path] = None,
    device: torch.device | str = "cpu",
    allow_default_hyperparams: bool = False,
    m_source: str = "sdf_fc1",
) -> AnalysisCheckpoint:
    device = torch.device(device)
    models = build_models(device)
    missing_optional_fields = []
    checkpoint_format = None
    checkpoint_hash = None
    policy_state = None
    sdf_state = None
    firm_target_state = None

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_hash = _sha256_file(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location=device)
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

    metadata = {
        "checkpoint_sha256": checkpoint_hash,
        "policy_state_hash": _state_hash(policy_state),
        "sdf_state_hash": _state_hash(sdf_state),
        "checkpoint_format": checkpoint_format,
        "hyperparameter_source": hp_source,
        "missing_optional_fields": missing_optional_fields,
    }
    return AnalysisCheckpoint(models=models, hyperparams=hyperparams, metadata=metadata)

