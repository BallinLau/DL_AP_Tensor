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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-checkpoint", required=True)
    parser.add_argument("--model-spec-json", required=True)
    parser.add_argument("--economic-config-json", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    args = parser.parse_args()

    legacy_path = Path(args.legacy_checkpoint)
    model_spec = _load_json_object(Path(args.model_spec_json), wrapper_key="policy_value_model_spec")
    config_snapshot = _load_json_object(Path(args.economic_config_json), wrapper_key="config_snapshot")

    payload = torch.load(legacy_path, map_location="cpu")
    if not isinstance(payload, dict) or "models" not in payload:
        raise ValueError("legacy checkpoint must be a combined checkpoint with a models dictionary")

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
        "source": str(legacy_path),
        "source_sha256": _sha256_file(legacy_path),
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
