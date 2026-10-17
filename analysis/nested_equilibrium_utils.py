"""Crash-safe utilities for the formal nested-equilibrium runner.

This module contains orchestration and persistence helpers only.  Economic
operators remain in the validated production/analysis modules imported by
``run_full_nested_equilibrium.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence, TypeVar

import numpy as np
import pandas as pd
import torch


T = TypeVar("T")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def stable_json_hash(payload: Any) -> str:
    body = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return sha256(body).hexdigest()


def tensor_hash(items: Sequence[tuple[str, torch.Tensor]]) -> str:
    digest = sha256()
    for name, tensor in items:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def module_hash(module: Optional[torch.nn.Module]) -> str:
    if module is None:
        return "ABSENT"
    return tensor_hash(sorted(module.state_dict().items()))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def atomic_write_json(path: Path, payload: Any) -> None:
    body = json.dumps(
        _jsonable(payload), indent=2, sort_keys=True, default=str, allow_nan=False
    ).encode("utf-8")
    atomic_write_bytes(path, body + b"\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_write_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
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
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_module(module: torch.nn.Module) -> bool:
    return all(torch.isfinite(value).all().item() for value in module.state_dict().values())


def require_finite_module(name: str, module: torch.nn.Module) -> None:
    bad = [
        key for key, value in module.state_dict().items()
        if not bool(torch.isfinite(value).all())
    ]
    if bad:
        raise RuntimeError(f"nonfinite parameters in {name}: {bad[:20]}")


def require_finite_metrics(label: str, payload: Mapping[str, Any], keys: Sequence[str]) -> None:
    invalid = {}
    for key in keys:
        value = payload.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            invalid[key] = value
            continue
        if not math.isfinite(number):
            invalid[key] = value
    if invalid:
        raise RuntimeError(f"nonfinite {label} metrics: {invalid}")


def cuda_memory(device: torch.device) -> dict[str, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "allocated_memory": 0,
            "reserved_memory": 0,
            "max_memory_allocated": 0,
        }
    return {
        "allocated_memory": int(torch.cuda.memory_allocated(device)),
        "reserved_memory": int(torch.cuda.memory_reserved(device)),
        "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
    }


def is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        "out of memory" in str(error).lower() and "cuda" in str(error).lower()
    )


@dataclass(frozen=True)
class OOMEvent:
    block: str
    equilibrium_outer: int
    firm_inner: int
    macro_inner: int
    old_batch_size: int
    new_batch_size: int
    memory: Mapping[str, int]
    timestamp: float


def run_with_oom_fallback(
    *,
    block: str,
    initial_batch_size: int,
    fallback_batch_size: int,
    execute: Callable[[int], T],
    reload_block_start: Callable[[], None],
    events: list[dict[str, Any]],
    events_path: Path,
    device: torch.device,
    indices: Mapping[str, int],
) -> tuple[T, int]:
    """Run one block and retry exactly once from its checkpoint after CUDA OOM."""
    batch_size = int(initial_batch_size)
    try:
        return execute(batch_size), batch_size
    except BaseException as error:
        if not is_cuda_oom(error):
            raise
        if fallback_batch_size >= initial_batch_size or fallback_batch_size <= 0:
            raise RuntimeError(
                f"invalid OOM fallback for {block}: {initial_batch_size}->{fallback_batch_size}"
            ) from error
        event = OOMEvent(
            block=block,
            equilibrium_outer=int(indices.get("equilibrium_outer", -1)),
            firm_inner=int(indices.get("firm_inner", -1)),
            macro_inner=int(indices.get("macro_inner", -1)),
            old_batch_size=int(initial_batch_size),
            new_batch_size=int(fallback_batch_size),
            memory=cuda_memory(device),
            timestamp=time.time(),
        )
        events.append(_jsonable(event.__dict__))
        atomic_write_json(events_path, events)
        reload_block_start()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            return execute(int(fallback_batch_size)), int(fallback_batch_size)
        except BaseException as retry_error:
            if is_cuda_oom(retry_error):
                raise RuntimeError(
                    f"CUDA OOM persisted after one fallback in {block}: "
                    f"{initial_batch_size}->{fallback_batch_size}"
                ) from retry_error
            raise


def make_checkpoint_payload(
    *,
    models: Mapping[str, torch.nn.Module],
    state: Mapping[str, Any],
    git_head: str,
    hyperparameter_fingerprint: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    model_states = {
        name: {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        for name, model in models.items()
    }
    return {
        "format": "full_nested_equilibrium_v1",
        "git_head": git_head,
        "hyperparameter_fingerprint": hyperparameter_fingerprint,
        "state": dict(state),
        "models": model_states,
        "model_hashes": {name: tensor_hash(sorted(values.items())) for name, values in model_states.items()},
        "rng_state": capture_rng_state(),
        "optimizer_state_restored": False,
        "extra": dict(extra or {}),
    }


def verify_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("format") != "full_nested_equilibrium_v1":
        raise ValueError("unexpected nested-equilibrium checkpoint format")
    if not isinstance(payload.get("models"), Mapping):
        raise ValueError("checkpoint is missing models")
    for name, state in payload["models"].items():
        if not isinstance(state, Mapping) or not state:
            raise ValueError(f"checkpoint model state is empty: {name}")
        if not all(torch.is_tensor(value) for value in state.values()):
            raise ValueError(f"checkpoint model state contains non-tensors: {name}")
        if not all(bool(torch.isfinite(value).all()) for value in state.values()):
            raise ValueError(f"checkpoint model state contains nonfinite values: {name}")
        observed = tensor_hash(sorted(state.items()))
        if observed != payload.get("model_hashes", {}).get(name):
            raise ValueError(f"checkpoint hash mismatch: {name}")


def publish_checkpoint(
    *,
    path: Path,
    payload: Mapping[str, Any],
    run_root: Path,
) -> None:
    atomic_torch_save(path, dict(payload))
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    verify_checkpoint_payload(loaded)
    latest_path = run_root / "latest_checkpoint.json"
    previous: Optional[str] = None
    if latest_path.is_file():
        try:
            previous = json.loads(latest_path.read_text(encoding="utf-8")).get("latest")
        except (OSError, json.JSONDecodeError):
            previous = None
    atomic_write_json(
        latest_path,
        {
            "latest": str(path.resolve()),
            "previous": previous,
            "state": payload.get("state", {}),
            "git_head": payload.get("git_head"),
            "hyperparameter_fingerprint": payload.get("hyperparameter_fingerprint"),
        },
    )


def load_checkpoint(
    path: Path,
    *,
    models: Mapping[str, torch.nn.Module],
    git_head: str,
    hyperparameter_fingerprint: str,
    allow_code_mismatch: bool,
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    verify_checkpoint_payload(payload)
    mismatches = {}
    if payload.get("git_head") != git_head:
        mismatches["git_head"] = (payload.get("git_head"), git_head)
    if payload.get("hyperparameter_fingerprint") != hyperparameter_fingerprint:
        mismatches["hyperparameter_fingerprint"] = (
            payload.get("hyperparameter_fingerprint"), hyperparameter_fingerprint
        )
    if mismatches and not allow_code_mismatch:
        raise RuntimeError(f"resume provenance mismatch: {mismatches}")
    missing = sorted(set(models) - set(payload["models"]))
    if missing:
        raise RuntimeError(f"resume checkpoint is missing model states: {missing}")
    for name, model in models.items():
        model.load_state_dict(payload["models"][name], strict=True)
        require_finite_module(name, model)
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return dict(payload)


def append_summary_row(path: Path, row: Mapping[str, Any]) -> pd.DataFrame:
    current = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    addition = pd.DataFrame([_jsonable(dict(row))])
    if current.empty:
        columns = list(dict.fromkeys([*current.columns, *addition.columns]))
        frame = addition.reindex(columns=columns)
    else:
        frame = pd.concat([current, addition], ignore_index=True)
    atomic_write_csv(path, frame)
    return frame

