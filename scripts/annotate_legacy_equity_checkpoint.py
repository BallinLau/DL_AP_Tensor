"""Annotate legacy unscaled equity checkpoints with explicit metadata."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.economic_config import AnalysisEconomicConfig
from models import build_policy_value_from_checkpoint_spec


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _state_hash(state: Optional[Dict[str, Any]]) -> Optional[str]:
    if state is None:
        return None
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        h.update(key.encode("utf-8"))
        if torch.is_tensor(value):
            t = value.detach().cpu().contiguous()
            h.update(str(t.dtype).encode("utf-8"))
            h.update(str(tuple(t.shape)).encode("utf-8"))
            h.update(t.numpy().tobytes())
        else:
            h.update(repr(value).encode("utf-8"))
    return h.hexdigest()


def _load_json_object(path: Path, *, wrapper_key: Optional[str] = None) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if wrapper_key and isinstance(payload, dict) and wrapper_key in payload:
        payload = payload[wrapper_key]
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_hyperparams_json(path: Path) -> Dict[str, Any]:
    payload = _load_json_object(path)
    if "hyperparams" in payload and isinstance(payload["hyperparams"], dict):
        return dict(payload["hyperparams"])
    return dict(payload)


def _validate_legacy_payload(payload: Dict[str, Any], *, model_spec: Dict[str, Any], config_snapshot: Dict[str, Any]) -> None:
    existing = payload.get("value_parameterization")
    if isinstance(existing, dict) and str(existing.get("mode", "none")).lower() != "none":
        raise ValueError("Refusing to annotate an existing scaled checkpoint as legacy mode='none'")
    AnalysisEconomicConfig.from_dict(config_snapshot)
    model = build_policy_value_from_checkpoint_spec({"policy_value_model_spec": model_spec}, value_scale_mode="none")
    policy_state = payload.get("models", {}).get("policy_value")
    if policy_state is None:
        raise ValueError("checkpoint is missing models.policy_value")
    model.load_state_dict(policy_state, strict=True)
    firm_state = payload.get("models", {}).get("firm_target")
    if firm_state is not None:
        firm_target = build_policy_value_from_checkpoint_spec({"policy_value_model_spec": model_spec}, value_scale_mode="none")
        firm_target.load_state_dict(firm_state, strict=True)
    for key in ("delta", "phi", "g"):
        if key not in model_spec:
            raise ValueError(f"policy_value_model_spec is missing {key}")
    tol = 1e-6
    if abs(float(model_spec["delta"]) - float(model.delta.detach().cpu().item())) > tol:
        raise ValueError("model spec delta does not match policy_value state buffer")
    if abs(float(model_spec["phi"]) - float(model.phi.detach().cpu().item())) > tol:
        raise ValueError("model spec phi does not match policy_value state buffer")
    if abs(float(model_spec["g"]) - float(model.g.detach().cpu().item())) > tol:
        raise ValueError("model spec g does not match policy_value state buffer")
    if abs(float(config_snapshot["DELTA"]) - float(model_spec["delta"])) > tol:
        raise ValueError("config_snapshot DELTA does not match model spec delta")
    if abs(float(config_snapshot["G"]) - float(model_spec["g"])) > tol:
        raise ValueError("config_snapshot G does not match model spec g")


def _combined_payload(path: Path) -> tuple[Dict[str, Any], Dict[str, Optional[str]], str]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "models" not in payload:
        raise ValueError("legacy checkpoint must be a combined checkpoint with a models dictionary")
    source_hash = _sha256_file(path)
    return payload, {
        "source_format": "combined",
        "policy_source_sha256": source_hash,
        "sdf_source_sha256": source_hash,
        "firm_target_source_sha256": source_hash if payload.get("models", {}).get("firm_target") is not None else None,
        "hyperparams_source_sha256": source_hash,
        "model_spec_source_sha256": None,
        "economic_config_source_sha256": None,
    }, source_hash


def _raw_payload(args: argparse.Namespace) -> tuple[Dict[str, Any], Dict[str, Optional[str]], str]:
    required = {
        "policy_checkpoint": args.policy_checkpoint,
        "sdf_checkpoint": args.sdf_checkpoint,
        "hyperparams_json": args.hyperparams_json,
        "model_spec_json": args.model_spec_json,
        "economic_config_json": args.economic_config_json,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"raw annotation mode is missing required inputs: {missing}")
    policy_path = Path(args.policy_checkpoint)
    sdf_path = Path(args.sdf_checkpoint)
    firm_target_path = Path(args.firm_target_checkpoint) if args.firm_target_checkpoint else None
    policy_state = torch.load(policy_path, map_location="cpu")
    sdf_state = torch.load(sdf_path, map_location="cpu")
    firm_target_state = torch.load(firm_target_path, map_location="cpu") if firm_target_path else policy_state
    payload = {
        "models": {
            "policy_value": policy_state,
            "sdf_fc1": sdf_state,
            "firm_target": firm_target_state,
        },
        "hyperparams": _load_hyperparams_json(Path(args.hyperparams_json)),
    }
    return payload, {
        "source_format": "raw_components",
        "policy_source_sha256": _sha256_file(policy_path),
        "sdf_source_sha256": _sha256_file(sdf_path),
        "firm_target_source_sha256": _sha256_file(firm_target_path) if firm_target_path else _sha256_file(policy_path),
        "hyperparams_source_sha256": _sha256_file(Path(args.hyperparams_json)),
        "model_spec_source_sha256": _sha256_file(Path(args.model_spec_json)),
        "economic_config_source_sha256": _sha256_file(Path(args.economic_config_json)),
    }, _sha256_file(policy_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--legacy-checkpoint")
    group.add_argument("--policy-checkpoint")
    parser.add_argument("--sdf-checkpoint")
    parser.add_argument("--hyperparams-json")
    parser.add_argument("--firm-target-checkpoint")
    parser.add_argument("--model-spec-json", required=True)
    parser.add_argument("--economic-config-json", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    args = parser.parse_args()

    model_spec = _load_json_object(Path(args.model_spec_json), wrapper_key="policy_value_model_spec")
    config_snapshot = _load_json_object(Path(args.economic_config_json), wrapper_key="config_snapshot")

    if args.legacy_checkpoint:
        payload, source_meta, source_sha = _combined_payload(Path(args.legacy_checkpoint))
    else:
        payload, source_meta, source_sha = _raw_payload(args)
    source_meta["model_spec_source_sha256"] = source_meta.get("model_spec_source_sha256") or _sha256_file(Path(args.model_spec_json))
    source_meta["economic_config_source_sha256"] = source_meta.get("economic_config_source_sha256") or _sha256_file(Path(args.economic_config_json))
    _validate_legacy_payload(payload, model_spec=model_spec, config_snapshot=config_snapshot)

    before_hashes = {
        "models.policy_value": _state_hash(payload.get("models", {}).get("policy_value")),
        "models.sdf_fc1": _state_hash(payload.get("models", {}).get("sdf_fc1")),
        "models.firm_target": _state_hash(payload.get("models", {}).get("firm_target")),
        "optimizers": _state_hash(payload.get("optimizers")),
    }

    out = copy.deepcopy(payload)
    out["value_parameterization"] = {
        "mode": "none",
        "scale_formula": "1",
        "bellman_normalization": False,
        "log_max": 20.0,
    }
    out["policy_value_model_spec"] = model_spec
    out["config_snapshot"] = config_snapshot
    out["legacy_annotation"] = {
        **source_meta,
        "source": str(args.legacy_checkpoint or args.policy_checkpoint),
        "source_sha256": source_sha,
        "state_hashes_before": before_hashes,
    }

    after_hashes = {
        "models.policy_value": _state_hash(out.get("models", {}).get("policy_value")),
        "models.sdf_fc1": _state_hash(out.get("models", {}).get("sdf_fc1")),
        "models.firm_target": _state_hash(out.get("models", {}).get("firm_target")),
        "optimizers": _state_hash(out.get("optimizers")),
    }
    if before_hashes != after_hashes:
        raise RuntimeError("annotation unexpectedly changed model or optimizer state hashes")
    out["legacy_annotation"]["state_hashes_after"] = after_hashes

    output_path = Path(args.output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_path)
    print(json.dumps({"output": str(output_path), "source_sha256": out["legacy_annotation"]["source_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
