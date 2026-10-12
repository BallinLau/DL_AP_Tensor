"""Versioned CPU state bank for Mode B simulation warm starts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch


STATE_BANK_FORMAT_VERSION = "state_bank_v1"

_NAN_PAD_FIELDS = {
    "entry_P0",
    "entry_PI",
    "entry_value_gap",
    "transition_eps_x",
    "transition_eps_z",
    "transition_u_eta",
    "transition_u_i",
}
_NEGATIVE_ONE_PAD_FIELDS = {"firm_id", "birth_time"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def compatibility_fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(payload)).encode("utf-8")).hexdigest()


def build_state_bank_compatibility(
    *,
    config: type,
    hyperparams: Any,
    branch_num: int,
    main_branch: int = 0,
) -> Dict[str, Any]:
    """Build the economic/simulation compatibility payload for one bank."""
    config_names = (
        "RHO_X",
        "SIGMA_X",
        "XBAR",
        "RHO_Z",
        "SIGMA_Z",
        "ZBAR",
        "ZETA",
        "I_THRESHOLD",
        "DELTA",
        "PHI",
        "G",
    )
    hp_names = (
        "entry_mode",
        "entry_spec_version",
        "resource_accounting_version",
        "entry_capital_ratio",
        "entry_size_ratio",
        "entry_cost_max",
        "entry_dummy_i",
        "entry_rng_seed",
        "consumption_aggregation_mode",
        "consumption_floor",
        "node_accounting_mode",
        "simulation_bp_action_source",
    )
    return {
        "format_version": STATE_BANK_FORMAT_VERSION,
        "branch_num": int(branch_num),
        "main_branch": int(main_branch),
        "enable_entry": True,
        "enable_exit": True,
        "config": {name: getattr(config, name) for name in config_names},
        "hyperparams": {name: getattr(hyperparams, name) for name in hp_names},
    }


def _clone_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu").clone().contiguous()


def clone_state(
    state: Mapping[str, torch.Tensor],
    *,
    device: Optional[torch.device | str] = None,
) -> Dict[str, torch.Tensor]:
    """Clone a batched state so simulations cannot mutate the stored source."""
    target = None if device is None else torch.device(device)
    return {
        name: tensor.detach().clone().to(target or tensor.device)
        for name, tensor in state.items()
    }


class StateBank:
    """A collection of full parent-ready path states stored on CPU."""

    def __init__(
        self,
        *,
        compatibility: Mapping[str, Any],
        source_episode: int,
        entries: Optional[List[Dict[str, Any]]] = None,
        field_schema: Optional[Dict[str, Dict[str, Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.compatibility = dict(compatibility)
        self.compatibility_fingerprint = compatibility_fingerprint(self.compatibility)
        self.source_episode = int(source_episode)
        self.entries: List[Dict[str, Any]] = list(entries or [])
        self.field_schema: Dict[str, Dict[str, Any]] = dict(field_schema or {})
        self.metadata = dict(metadata or {})
        self._entry_index: Dict[str, Dict[str, Any]] = {}
        self._reindex()

    def _reindex(self) -> None:
        self._entry_index = {}
        for entry in self.entries:
            entry_id = str(entry["entry_id"])
            if entry_id in self._entry_index:
                raise ValueError(f"Duplicate State Bank entry id: {entry_id}")
            self._entry_index[entry_id] = entry

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def entry_ids(self) -> List[str]:
        return [str(entry["entry_id"]) for entry in self.entries]

    def validate_compatibility(self, expected: Mapping[str, Any]) -> None:
        expected_fingerprint = compatibility_fingerprint(expected)
        if expected_fingerprint != self.compatibility_fingerprint:
            raise ValueError(
                "State Bank compatibility fingerprint mismatch: "
                f"bank={self.compatibility_fingerprint}, expected={expected_fingerprint}"
            )

    def _infer_schema(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> Dict[str, Dict[str, Any]]:
        if "b" not in state or state["b"].ndim != 2:
            raise ValueError("State Bank capture requires two-dimensional state['b']")
        n_paths, capacity = state["b"].shape
        schema: Dict[str, Dict[str, Any]] = {}
        for name, tensor in sorted(state.items()):
            if not torch.is_tensor(tensor):
                raise TypeError(f"State field {name!r} is not a tensor")
            if tensor.ndim == 0 or int(tensor.shape[0]) != int(n_paths):
                raise ValueError(
                    f"State field {name!r} does not have path dimension {n_paths}: "
                    f"shape={tuple(tensor.shape)}"
                )
            role = (
                "firm"
                if tensor.ndim >= 2 and int(tensor.shape[1]) == int(capacity)
                else "path"
            )
            schema[name] = {
                "role": role,
                "dtype": str(tensor.dtype),
                "rank_after_path": int(tensor.ndim - 1),
                "fixed_shape_after_path": (
                    list(tensor.shape[1:]) if role == "path" else list(tensor.shape[2:])
                ),
            }
        return schema

    def _validate_state_schema(self, state: Mapping[str, torch.Tensor]) -> None:
        schema = self._infer_schema(state)
        if not self.field_schema:
            self.field_schema = schema
            return
        if schema != self.field_schema:
            missing = sorted(set(self.field_schema).difference(schema))
            extra = sorted(set(schema).difference(self.field_schema))
            changed = sorted(
                name
                for name in set(schema).intersection(self.field_schema)
                if schema[name] != self.field_schema[name]
            )
            raise ValueError(
                "State Bank state schema changed across captures: "
                f"missing={missing}, extra={extra}, changed={changed}"
            )

    def capture(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        capture_time: int,
        max_entries: int = 0,
    ) -> int:
        """Capture every non-extinct path from one parent-ready state."""
        self._validate_state_schema(state)
        n_paths = int(state["b"].shape[0])
        alive = state.get("alive")
        captured = 0
        for path_index in range(n_paths):
            if max_entries > 0 and len(self.entries) >= int(max_entries):
                break
            if alive is not None and not bool(alive[path_index].any()):
                continue
            fields = {
                name: _clone_cpu(tensor[path_index])
                for name, tensor in state.items()
            }
            entry_id = (
                f"ep{self.source_episode}:t{int(capture_time)}:"
                f"path{int(path_index)}"
            )
            entry = {
                "entry_id": entry_id,
                "source_episode": self.source_episode,
                "capture_time": int(capture_time),
                "source_path": int(path_index),
                "firm_capacity": int(state["b"].shape[1]),
                "fields": fields,
            }
            if entry_id in self._entry_index:
                raise ValueError(f"State Bank entry already captured: {entry_id}")
            self.entries.append(entry)
            self._entry_index[entry_id] = entry
            captured += 1
        return captured

    def sample_entry_ids(
        self,
        n_paths: int,
        *,
        seed: int,
        episode_id: int,
    ) -> List[str]:
        if not self.entries:
            raise ValueError("Cannot sample from an empty State Bank")
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + int(episode_id) * 100_003)
        if n_paths <= len(self.entries):
            positions = torch.randperm(len(self.entries), generator=generator)[:n_paths]
        else:
            positions = torch.randint(
                len(self.entries),
                (n_paths,),
                generator=generator,
            )
        return [self.entries[int(index)]["entry_id"] for index in positions.tolist()]

    @staticmethod
    def _padding_value(name: str, tensor: torch.Tensor) -> float | int | bool:
        if name in _NAN_PAD_FIELDS and tensor.is_floating_point():
            return float("nan")
        if name in _NEGATIVE_ONE_PAD_FIELDS:
            return -1
        if tensor.dtype == torch.bool:
            return False
        return 0

    def stack_entries(
        self,
        entry_ids: Sequence[str],
        *,
        device: Optional[torch.device | str] = None,
    ) -> Dict[str, torch.Tensor]:
        if not entry_ids:
            raise ValueError("entry_ids must be non-empty")
        try:
            entries = [self._entry_index[str(entry_id)] for entry_id in entry_ids]
        except KeyError as exc:
            raise KeyError(f"Unknown State Bank entry id: {exc.args[0]}") from exc
        max_capacity = max(int(entry["firm_capacity"]) for entry in entries)
        stacked: Dict[str, torch.Tensor] = {}
        for name, schema in self.field_schema.items():
            values = [entry["fields"][name] for entry in entries]
            if schema["role"] == "path":
                result = torch.stack(values, dim=0)
            else:
                padded = []
                for value in values:
                    pad_length = max_capacity - int(value.shape[0])
                    if pad_length > 0:
                        pad_shape = (pad_length, *value.shape[1:])
                        pad = torch.full(
                            pad_shape,
                            self._padding_value(name, value),
                            dtype=value.dtype,
                        )
                        value = torch.cat([value, pad], dim=0)
                    padded.append(value)
                result = torch.stack(padded, dim=0)
            if device is not None:
                result = result.to(device)
            stacked[name] = result
        return stacked

    def to_payload(self) -> Dict[str, Any]:
        return {
            "format_version": STATE_BANK_FORMAT_VERSION,
            "source_episode": self.source_episode,
            "compatibility": self.compatibility,
            "compatibility_fingerprint": self.compatibility_fingerprint,
            "field_schema": self.field_schema,
            "metadata": self.metadata,
            "entries": self.entries,
        }

    def save(self, path: Path | str, *, atomic: bool = True) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if atomic:
            temporary = target.with_suffix(target.suffix + ".tmp")
            try:
                torch.save(self.to_payload(), temporary)
                os.replace(temporary, target)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        else:
            torch.save(self.to_payload(), target)
        return target

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        expected_compatibility: Optional[Mapping[str, Any]] = None,
    ) -> "StateBank":
        source = Path(path)
        try:
            payload = torch.load(source, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(source, map_location="cpu")
        if payload.get("format_version") != STATE_BANK_FORMAT_VERSION:
            raise ValueError(
                "Unsupported State Bank format: "
                f"{payload.get('format_version')!r}"
            )
        bank = cls(
            compatibility=payload["compatibility"],
            source_episode=int(payload["source_episode"]),
            entries=payload["entries"],
            field_schema=payload["field_schema"],
            metadata=payload.get("metadata", {}),
        )
        recorded = payload.get("compatibility_fingerprint")
        if recorded != bank.compatibility_fingerprint:
            raise ValueError(
                "State Bank compatibility payload fingerprint is corrupt: "
                f"recorded={recorded}, recomputed={bank.compatibility_fingerprint}"
            )
        if expected_compatibility is not None:
            bank.validate_compatibility(expected_compatibility)
        return bank


class StateBankCapture:
    """Capture policy called at the top of the tensor simulation time loop."""

    def __init__(
        self,
        bank: StateBank,
        *,
        burn_in: int,
        capture_stride: int = 1,
        max_entries: int = 0,
    ) -> None:
        if burn_in < 1:
            raise ValueError("State Bank burn_in must be at least 1")
        if capture_stride <= 0:
            raise ValueError("State Bank capture_stride must be positive")
        if max_entries < 0:
            raise ValueError("State Bank max_entries must be non-negative")
        self.bank = bank
        self.burn_in = int(burn_in)
        self.capture_stride = int(capture_stride)
        self.max_entries = int(max_entries)
        self.capture_calls = 0
        self.entries_captured = 0

    def capture_parent_ready(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        time_index: int,
    ) -> None:
        if time_index < self.burn_in:
            return
        if (time_index - self.burn_in) % self.capture_stride != 0:
            return
        if self.max_entries > 0 and len(self.bank) >= self.max_entries:
            return
        self.capture_calls += 1
        self.entries_captured += self.bank.capture(
            state,
            capture_time=int(time_index),
            max_entries=self.max_entries,
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "burn_in": self.burn_in,
            "capture_stride": self.capture_stride,
            "max_entries": self.max_entries,
            "capture_calls": self.capture_calls,
            "entries_captured": self.entries_captured,
            "bank_size": len(self.bank),
        }
