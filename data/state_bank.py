"""Versioned CPU state bank for Mode B simulation warm starts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


STATE_BANK_FORMAT_VERSION = "state_bank_v2"

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
        "TAU",
        "KAPPA_B",
        "KAPPA_E",
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
        "transition_rng_mode",
        "common_transition_seed",
        "preserve_global_rng_around_entry",
        "entry_rng_mode",
    )
    return {
        "format_version": STATE_BANK_FORMAT_VERSION,
        "branch_num": int(branch_num),
        "main_branch": int(main_branch),
        "enable_entry": True,
        "enable_exit": True,
        "state_bank_schema_version": STATE_BANK_FORMAT_VERSION,
        "config": {name: getattr(config, name) for name in config_names},
        "hyperparams": {name: getattr(hyperparams, name) for name in hp_names},
    }


def _flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key in sorted(value):
        name = f"{prefix}.{key}" if prefix else str(key)
        item = value[key]
        if isinstance(item, Mapping):
            flattened.update(_flatten_mapping(item, name))
        else:
            flattened[name] = item
    return flattened


def compatibility_field_diff(
    recorded: Mapping[str, Any], expected: Mapping[str, Any]
) -> Dict[str, Any]:
    left = _flatten_mapping(recorded)
    right = _flatten_mapping(expected)
    return {
        "missing": {key: left[key] for key in sorted(set(left) - set(right))},
        "extra": {key: right[key] for key in sorted(set(right) - set(left))},
        "changed": {
            key: {"bank": left[key], "expected": right[key]}
            for key in sorted(set(left).intersection(right))
            if left[key] != right[key]
        },
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
            diff = compatibility_field_diff(self.compatibility, expected)
            raise ValueError(
                "State Bank compatibility fingerprint mismatch: "
                f"bank={self.compatibility_fingerprint}, expected={expected_fingerprint}; "
                f"field_diff={json.dumps(diff, sort_keys=True, default=str)}"
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
        lineage_ids: Optional[Sequence[str]] = None,
        source_path_ids: Optional[Sequence[int]] = None,
        entry_namespace: str = "",
        reservoir_generator: Optional[torch.Generator] = None,
    ) -> int:
        """Capture every non-extinct path from one parent-ready state."""
        self._validate_state_schema(state)
        n_paths = int(state["b"].shape[0])
        alive = state.get("alive")
        transition_path_id = state.get(
            "transition_path_id", torch.arange(n_paths, device=state["b"].device)
        )
        if lineage_ids is None:
            generation = str(
                self.metadata.get("generation_id", f"ep{self.source_episode}")
            )
            lineage_ids = [
                f"root:{generation}:path:{int(transition_path_id[path].item())}"
                for path in range(n_paths)
            ]
        if len(lineage_ids) != n_paths:
            raise ValueError("lineage_ids must match the captured path dimension")
        if source_path_ids is None:
            source_path_ids = [int(transition_path_id[path].item()) for path in range(n_paths)]
        if len(source_path_ids) != n_paths:
            raise ValueError("source_path_ids must match the captured path dimension")
        if max_entries > 0 and reservoir_generator is None:
            raise ValueError("capped State Bank capture requires a reservoir generator")
        captured = 0
        for path_index in range(n_paths):
            if alive is not None and not bool(alive[path_index].any()):
                continue
            fields = {
                name: _clone_cpu(tensor[path_index])
                for name, tensor in state.items()
            }
            economic_time = int(
                state.get(
                    "economic_time",
                    torch.full((n_paths,), int(capture_time), device=state["b"].device),
                )[path_index].item()
            )
            namespace = f":{entry_namespace}" if entry_namespace else ""
            entry_id = (
                f"ep{self.source_episode}{namespace}:t{int(capture_time)}:"
                f"path{int(path_index)}:econ{economic_time}"
            )
            entry = {
                "entry_id": entry_id,
                "lineage_id": str(lineage_ids[path_index]),
                "source_episode": self.source_episode,
                "capture_time": int(capture_time),
                "economic_time": economic_time,
                "source_path": int(source_path_ids[path_index]),
                "firm_capacity": int(state["b"].shape[1]),
                "fields": fields,
            }
            if entry_id in self._entry_index:
                raise ValueError(f"State Bank entry already captured: {entry_id}")
            seen = int(self.metadata.get("reservoir_candidates_seen", 0)) + 1
            self.metadata["reservoir_candidates_seen"] = seen
            if max_entries <= 0 or len(self.entries) < int(max_entries):
                self.entries.append(entry)
                self._entry_index[entry_id] = entry
                captured += 1
            else:
                replacement = int(
                    torch.randint(seen, (1,), generator=reservoir_generator).item()
                )
                if replacement < int(max_entries):
                    old_id = str(self.entries[replacement]["entry_id"])
                    self._entry_index.pop(old_id, None)
                    self.entries[replacement] = entry
                    self._entry_index[entry_id] = entry
                    captured += 1
        return captured

    def sample_entry_ids(
        self,
        n_paths: int,
        *,
        seed: int,
        episode_id: int,
        mode: str = "uniform_entry",
    ) -> List[str]:
        if not self.entries:
            raise ValueError("Cannot sample from an empty State Bank")
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + int(episode_id) * 100_003)
        mode = str(mode).strip().lower()
        if mode == "uniform_entry":
            if n_paths <= len(self.entries):
                positions = torch.randperm(len(self.entries), generator=generator)[:n_paths]
            else:
                positions = torch.randint(
                    len(self.entries),
                    (n_paths,),
                    generator=generator,
                )
            return [self.entries[int(index)]["entry_id"] for index in positions.tolist()]
        if mode != "uniform_lineage":
            raise ValueError(
                "state_bank_sampling_mode must be uniform_entry or uniform_lineage"
            )
        by_lineage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for entry in self.entries:
            by_lineage[str(entry["lineage_id"])].append(entry)
        lineage_names = sorted(by_lineage)
        lineage_positions_parts = []
        remaining = int(n_paths)
        while remaining > 0:
            order = torch.randperm(len(lineage_names), generator=generator)
            take = min(remaining, len(lineage_names))
            lineage_positions_parts.append(order[:take])
            remaining -= take
        lineage_positions = torch.cat(lineage_positions_parts, dim=0)
        sampled: List[str] = []
        for position in lineage_positions.tolist():
            candidates = by_lineage[lineage_names[int(position)]]
            state_position = int(
                torch.randint(len(candidates), (1,), generator=generator).item()
            )
            sampled.append(str(candidates[state_position]["entry_id"]))
        return sampled

    def lineage_ids_for_entries(self, entry_ids: Sequence[str]) -> List[str]:
        return [str(self._entry_index[str(entry_id)]["lineage_id"]) for entry_id in entry_ids]

    def sampling_summary(self, entry_ids: Sequence[str]) -> Dict[str, Any]:
        lineages = self.lineage_ids_for_entries(entry_ids)
        counts = Counter(lineages)
        n = len(entry_ids)
        return {
            "n_sampled_states": int(n),
            "n_unique_entries": int(len(set(map(str, entry_ids)))),
            "n_unique_lineages": int(len(counts)),
            "unique_lineage_ratio": float(len(counts) / max(n, 1)),
            "lineage_concentration": float(
                sum((count / max(n, 1)) ** 2 for count in counts.values())
            ),
            "max_states_from_same_lineage": int(max(counts.values(), default=0)),
        }

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

    def distribution_summary(self) -> Dict[str, Any]:
        if not self.entries:
            return {
                "bank_entry_count": 0,
                "unique_lineages": 0,
                "bank_unique_lineage_ratio": 0.0,
                "largest_lineage_share": 0.0,
            }

        def _quantiles(values: Sequence[float]) -> Dict[str, float]:
            tensor = torch.as_tensor(values, dtype=torch.float64)
            finite = tensor[torch.isfinite(tensor)]
            if not finite.numel():
                return {name: float("nan") for name in ("p10", "p50", "p90")}
            q = torch.quantile(finite, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))
            return {"p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2])}

        def _stats(values: Sequence[float], *, p99: bool = False) -> Dict[str, float]:
            tensor = torch.as_tensor(values, dtype=torch.float64)
            finite = tensor[torch.isfinite(tensor)]
            if not finite.numel():
                names = ["mean", "std", "p10", "p50", "p90"] + (["p99"] if p99 else [])
                return {name: float("nan") for name in names}
            levels = [0.1, 0.5, 0.9] + ([0.99] if p99 else [])
            q = torch.quantile(finite, torch.tensor(levels, dtype=torch.float64))
            result = {
                "mean": float(finite.mean()),
                "std": float(finite.std(unbiased=False)),
                "p10": float(q[0]),
                "p50": float(q[1]),
                "p90": float(q[2]),
            }
            if p99:
                result["p99"] = float(q[3])
            return result

        lineage_counts = Counter(str(entry["lineage_id"]) for entry in self.entries)
        b_values: List[float] = []
        capital_weights: List[float] = []
        capital_weighted_b: List[float] = []
        total_k: List[float] = []
        lnk: List[float] = []
        hatc: List[float] = []
        hatcf: List[float] = []
        firm_count: List[float] = []
        entry_share: List[float] = []
        initial_share: List[float] = []
        default_rate: List[float] = []
        investment_rate: List[float] = []
        lnk_by_lineage: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for entry in self.entries:
            fields = entry["fields"]
            alive = fields["alive"].to(torch.bool)
            b = fields["b"][alive].to(torch.float64)
            K = fields["K"][alive].to(torch.float64)
            b_values.extend(b.tolist())
            capital_weights.extend(K.tolist())
            capital_weighted_b.extend((b * K).tolist())
            total_k.append(float(K.sum()))
            firm_count.append(float(alive.sum()))
            lnk_value = float(fields["lnkf"].reshape(-1)[0])
            hatc_value = float(
                fields.get("hatc_cal", fields["hatcf"]).reshape(-1)[0]
            )
            lnk.append(lnk_value)
            hatc.append(hatc_value)
            hatcf.append(float(fields["hatcf"].reshape(-1)[0]))
            lnk_by_lineage[str(entry["lineage_id"])].append(
                (int(entry["economic_time"]), lnk_value)
            )
            if alive.any():
                entry_share.append(float(fields["entry"][alive].float().mean()))
                initial_share.append(float(fields["initial_cohort"][alive].float().mean()))
                default_rate.append(float((fields["bar_z"][alive] >= 0.5).float().mean()))
                investment_rate.append(float(fields["bar_i"][alive].float().mean()))
        weighted_b_mean = (
            float(sum(capital_weighted_b) / max(sum(capital_weights), 1e-12))
            if capital_weights else float("nan")
        )
        lineage_weighted_b: List[float] = []
        for entry in self.entries:
            fields = entry["fields"]
            alive = fields["alive"].to(torch.bool)
            weights = fields["K"][alive].to(torch.float64)
            if weights.sum() > 0:
                lineage_weighted_b.append(
                    float((fields["b"][alive].to(torch.float64) * weights).sum() / weights.sum())
                )
        dlnk: List[float] = []
        for sequence in lnk_by_lineage.values():
            ordered = sorted(sequence)
            dlnk.extend(
                ordered[index][1] - ordered[index - 1][1]
                for index in range(1, len(ordered))
                if ordered[index][0] != ordered[index - 1][0]
            )
        return {
            "bank_entry_count": int(len(self.entries)),
            "unique_lineages": int(len(lineage_counts)),
            "bank_unique_lineage_ratio": float(len(lineage_counts) / len(self.entries)),
            "largest_lineage_share": float(max(lineage_counts.values()) / len(self.entries)),
            "capture_time": _quantiles([float(entry["capture_time"]) for entry in self.entries]),
            "economic_time": _quantiles([float(entry["economic_time"]) for entry in self.entries]),
            "b": _stats(b_values, p99=True),
            "capital_weighted_b": {
                "mean": weighted_b_mean,
                **_quantiles(lineage_weighted_b),
            },
            "K": _stats(total_k),
            "LnK": _stats(lnk),
            "Hatc": _stats(hatc),
            "hatcf": _stats(hatcf),
            "firm_count": _quantiles(firm_count),
            "entry_share": float(sum(entry_share) / max(len(entry_share), 1)),
            "initial_cohort_share": float(sum(initial_share) / max(len(initial_share), 1)),
            "default_rate": float(sum(default_rate) / max(len(default_rate), 1)),
            "investment_rate": float(sum(investment_rate) / max(len(investment_rate), 1)),
            "mean_LnK": float(sum(lnk) / max(len(lnk), 1)),
            "p10_LnK": _quantiles(lnk)["p10"],
            "p50_LnK": _quantiles(lnk)["p50"],
            "p90_LnK": _quantiles(lnk)["p90"],
            "mean_dLnK": float(sum(dlnk) / len(dlnk)) if dlnk else float("nan"),
        }


@dataclass(frozen=True)
class StateBankRolloutChunk:
    chunk_id: int
    path_offset: int
    entry_ids: Tuple[str, ...]
    lineage_ids: Tuple[str, ...]
    initial_state_cpu: Dict[str, torch.Tensor]
    transition_seed: int

    @property
    def n_paths(self) -> int:
        return len(self.entry_ids)


@dataclass(frozen=True)
class StateBankRolloutPlan:
    episode_id: int
    short_horizon: int
    requested_parent_transitions: int
    sampling_mode: str
    chunks: Tuple[StateBankRolloutChunk, ...]

    @classmethod
    def build(
        cls,
        bank: StateBank,
        *,
        episode_id: int,
        n_paths_per_chunk: int,
        short_horizon: int,
        parent_transition_budget: int,
        sample_seed: int,
        sampling_mode: str,
        transition_seed: int,
    ) -> "StateBankRolloutPlan":
        if n_paths_per_chunk <= 0 or short_horizon <= 0:
            raise ValueError("rollout chunk paths and horizon must be positive")
        if parent_transition_budget <= 0:
            raise ValueError("parent_transition_budget must be positive")
        total_paths = int(math.ceil(parent_transition_budget / short_horizon))
        entry_ids = bank.sample_entry_ids(
            total_paths,
            seed=sample_seed,
            episode_id=episode_id,
            mode=sampling_mode,
        )
        chunks: List[StateBankRolloutChunk] = []
        for chunk_id, start in enumerate(range(0, total_paths, n_paths_per_chunk)):
            ids = tuple(entry_ids[start : start + n_paths_per_chunk])
            chunks.append(
                StateBankRolloutChunk(
                    chunk_id=int(chunk_id),
                    path_offset=int(start),
                    entry_ids=ids,
                    lineage_ids=tuple(bank.lineage_ids_for_entries(ids)),
                    initial_state_cpu=bank.stack_entries(ids, device="cpu"),
                    transition_seed=int(transition_seed),
                )
            )
        return cls(
            episode_id=int(episode_id),
            short_horizon=int(short_horizon),
            requested_parent_transitions=int(parent_transition_budget),
            sampling_mode=str(sampling_mode),
            chunks=tuple(chunks),
        )

    @property
    def entry_ids(self) -> List[str]:
        return [entry_id for chunk in self.chunks for entry_id in chunk.entry_ids]

    @property
    def lineage_ids(self) -> List[str]:
        return [lineage for chunk in self.chunks for lineage in chunk.lineage_ids]

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            _canonical_json(
                {
                    "episode_id": self.episode_id,
                    "short_horizon": self.short_horizon,
                    "requested_parent_transitions": self.requested_parent_transitions,
                    "sampling_mode": self.sampling_mode,
                }
            ).encode("utf-8")
        )
        for chunk in self.chunks:
            digest.update(
                _canonical_json(
                    {
                        "chunk_id": chunk.chunk_id,
                        "path_offset": chunk.path_offset,
                        "entry_ids": chunk.entry_ids,
                        "lineage_ids": chunk.lineage_ids,
                        "transition_seed": chunk.transition_seed,
                    }
                ).encode("utf-8")
            )
            for name in sorted(chunk.initial_state_cpu):
                tensor = chunk.initial_state_cpu[name].detach().cpu().contiguous()
                digest.update(name.encode("utf-8"))
                digest.update(str(tensor.dtype).encode("utf-8"))
                digest.update(str(tuple(tensor.shape)).encode("utf-8"))
                digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def summary(self) -> Dict[str, Any]:
        counts = Counter(self.lineage_ids)
        sampled = len(self.entry_ids)
        return {
            "requested_parent_budget": int(self.requested_parent_transitions),
            "rollout_chunk_count": int(len(self.chunks)),
            "sampled_bank_entry_count": int(sampled),
            "unique_sampled_bank_entries": int(len(set(self.entry_ids))),
            "unique_sampled_lineages": int(len(counts)),
            "unique_lineage_ratio": float(len(counts) / max(sampled, 1)),
            "lineage_concentration": float(
                sum((count / max(sampled, 1)) ** 2 for count in counts.values())
            ),
            "max_states_from_same_lineage": int(max(counts.values(), default=0)),
            "sampling_mode": self.sampling_mode,
            "short_horizon": int(self.short_horizon),
            "plan_fingerprint": self.fingerprint(),
        }


class StateBankCapture:
    """Capture policy called at the top of the tensor simulation time loop."""

    def __init__(
        self,
        bank: StateBank,
        *,
        burn_in: int,
        capture_stride: int = 1,
        max_entries: int = 0,
        lineage_ids: Optional[Sequence[str]] = None,
        entry_namespace: str = "",
        reservoir_seed: int = 0,
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
        self.lineage_ids = None if lineage_ids is None else tuple(map(str, lineage_ids))
        self.entry_namespace = str(entry_namespace)
        self.reservoir_seed = int(reservoir_seed)
        self.reservoir_generator = torch.Generator(device="cpu")
        self.reservoir_generator.manual_seed(self.reservoir_seed)
        self.capture_calls = 0
        self.entries_captured = 0

    def set_context(
        self,
        *,
        lineage_ids: Optional[Sequence[str]],
        entry_namespace: str,
    ) -> None:
        """Reuse one reservoir stream across independently simulated chunks."""
        self.lineage_ids = (
            None if lineage_ids is None else tuple(map(str, lineage_ids))
        )
        self.entry_namespace = str(entry_namespace)

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
        self.capture_calls += 1
        self.entries_captured += self.bank.capture(
            state,
            capture_time=int(time_index),
            max_entries=self.max_entries,
            lineage_ids=self.lineage_ids,
            entry_namespace=self.entry_namespace,
            reservoir_generator=self.reservoir_generator,
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "burn_in": self.burn_in,
            "capture_stride": self.capture_stride,
            "max_entries": self.max_entries,
            "reservoir_seed": self.reservoir_seed,
            "capture_calls": self.capture_calls,
            "entries_captured": self.entries_captured,
            "bank_size": len(self.bank),
        }
