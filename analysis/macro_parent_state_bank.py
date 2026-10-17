"""Macro parent-bank construction using production short simulations.

The module deliberately owns only *which production states are simulated*.
It does not reproduce firm transitions, entry/exit, accounting, FC1 targets, or
SDF pairing.  Complete parent-ready states are captured through ``StateBank``
at the top of the production tensor simulation loop and are resumed through
``SimulateTS(initial_state=..., horizon=1, branch_num=2)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from config import Config
from data.simulate_ts import SimulateTS
from data.state_bank import (
    StateBank,
    build_state_bank_compatibility,
    clone_state,
)
from data.tensor_data import TensorSimulationOutput, TensorTable


SUMMARY_FEATURES = (
    "b_mean",
    "b_std",
    "z_mean",
    "z_std",
    "corr_b_z",
    "log_K_total",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _tensor_digest(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(items):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parse_capture_times(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        requested = tuple(int(part) for part in parts)
    else:
        requested = tuple(int(part) for part in value)
    if not requested or any(time_index < 0 for time_index in requested):
        raise ValueError("macro parent capture times must be non-empty and non-negative")
    return tuple(sorted(set(requested)))


def effective_capture_times(
    requested: str | Sequence[int], simulation_horizon: int
) -> tuple[int, ...]:
    """Map requested endpoints to stable parent-ready StateBank times.

    Production StateBank entries begin at local ``t=1``.  The fresh ``t=0``
    state has not yet acquired the transition-provenance fields established by
    the first node processing, so mixing it with later states would violate the
    authoritative StateBank schema.  The upper endpoint is similarly mapped
    from economic-horizon notation to ``range(horizon)`` indexing.
    """
    if int(simulation_horizon) < 2:
        raise ValueError("short rollout horizon must be at least 2 for parent-bank capture")
    last_parent_index = int(simulation_horizon) - 1
    return tuple(
        sorted({max(1, min(time_index, last_parent_index)) for time_index in parse_capture_times(requested)})
    )


@dataclass(frozen=True)
class ParentSourceKey:
    split: str
    initial_state_seed: int
    initial_path_id: int
    capture_time: int
    source_type: str = "ENDOGENOUS_SHORT_ROLLOUT"

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "initial_state_seed": int(self.initial_state_seed),
            "initial_path_id": int(self.initial_path_id),
            "capture_time": int(self.capture_time),
            "source_type": self.source_type,
        }

    @property
    def canonical(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParentSourceKey":
        return cls(
            split=str(value["split"]),
            initial_state_seed=int(value["initial_state_seed"]),
            initial_path_id=int(value["initial_path_id"]),
            capture_time=int(value["capture_time"]),
            source_type=str(value.get("source_type", "ENDOGENOUS_SHORT_ROLLOUT")),
        )


@dataclass(frozen=True)
class MacroParentSourceSchedule:
    split: str
    initial_state_seed: int
    transition_seed: int
    entry_seed: int
    candidate_paths: int
    short_rollout_horizon: int
    requested_capture_times: tuple[int, ...]
    effective_capture_times: tuple[int, ...]
    source_keys: tuple[ParentSourceKey, ...]
    endogenous_frequency_share: float
    coverage_aware_share: float

    @property
    def fingerprint(self) -> str:
        return _json_hash(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value = {
            "split": self.split,
            "initial_state_seed": int(self.initial_state_seed),
            "transition_seed": int(self.transition_seed),
            "entry_seed": int(self.entry_seed),
            "candidate_paths": int(self.candidate_paths),
            "short_rollout_horizon": int(self.short_rollout_horizon),
            "requested_capture_times": list(self.requested_capture_times),
            "effective_capture_times": list(self.effective_capture_times),
            "source_keys": [key.to_dict() for key in self.source_keys],
            "endogenous_frequency_share": float(self.endogenous_frequency_share),
            "coverage_aware_share": float(self.coverage_aware_share),
        }
        if include_fingerprint:
            value["fingerprint"] = self.fingerprint
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MacroParentSourceSchedule":
        result = cls(
            split=str(value["split"]),
            initial_state_seed=int(value["initial_state_seed"]),
            transition_seed=int(value["transition_seed"]),
            entry_seed=int(value["entry_seed"]),
            candidate_paths=int(value["candidate_paths"]),
            short_rollout_horizon=int(value["short_rollout_horizon"]),
            requested_capture_times=tuple(map(int, value["requested_capture_times"])),
            effective_capture_times=tuple(map(int, value["effective_capture_times"])),
            source_keys=tuple(
                ParentSourceKey.from_dict(item) for item in value["source_keys"]
            ),
            endogenous_frequency_share=float(value["endogenous_frequency_share"]),
            coverage_aware_share=float(value["coverage_aware_share"]),
        )
        recorded = value.get("fingerprint")
        if recorded is not None and str(recorded) != result.fingerprint:
            raise RuntimeError("macro parent source schedule fingerprint mismatch")
        return result


@dataclass
class HarvestedParentBank:
    bank: StateBank
    selected_entry_ids: list[str]
    summary: pd.DataFrame
    occupancy: np.ndarray
    simulation_output: TensorSimulationOutput
    initial_hash_audit: dict[str, Any]

    @property
    def n_parents(self) -> int:
        return len(self.selected_entry_ids)

    def stacked_state(self, *, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        return self.bank.stack_entries(self.selected_entry_ids, device=device)


@dataclass
class OneStepMacroDataset:
    macro: TensorTable
    pairs: TensorTable
    batches: dict[str, list[dict[str, torch.Tensor]]]
    pairing_audit: dict[str, Any]
    used_parent_batch_size: int


class ExactTimeStateBankCapture:
    """StateBank capture policy for an explicit set of parent-ready times.

    ``StateBankCapture`` historically requires ``burn_in >= 1``.  Macro banks
    need the production parent at local time zero as well, so this policy calls
    the same authoritative ``StateBank.capture`` serializer at exact times.
    """

    def __init__(
        self,
        bank: StateBank,
        *,
        split: str,
        capture_times: Sequence[int],
        initial_state_seed: int,
    ) -> None:
        self.bank = bank
        self.split = str(split)
        self.capture_times = frozenset(map(int, capture_times))
        self.initial_state_seed = int(initial_state_seed)
        self.capture_calls = 0

    def capture_parent_ready(
        self, state: Mapping[str, torch.Tensor], *, time_index: int
    ) -> None:
        if int(time_index) not in self.capture_times:
            return
        n_paths = int(state["b"].shape[0])
        lineage_ids = [
            f"{self.split}:seed{self.initial_state_seed}:path{path}"
            for path in range(n_paths)
        ]
        self.bank.capture(
            state,
            capture_time=int(time_index),
            lineage_ids=lineage_ids,
            source_path_ids=list(range(n_paths)),
            entry_namespace=self.split,
        )
        self.capture_calls += 1

    def summary(self) -> dict[str, Any]:
        return {
            "capture_mode": "exact_parent_ready_times",
            "capture_times": sorted(self.capture_times),
            "capture_calls": int(self.capture_calls),
            "entries_captured": len(self.bank),
            "bank_size": len(self.bank),
            "split": self.split,
            "initial_state_seed": self.initial_state_seed,
        }


def initial_cross_section_hash_audit(
    state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    n_paths = int(state["b"].shape[0])
    economic_cross_section_fields = (
        "b", "z", "eta", "i", "K", "alive", "entry", "initial_cohort"
    )
    hashes = []
    for path in range(n_paths):
        fields = []
        for name in economic_cross_section_fields:
            value = state.get(name)
            if value is None:
                continue
            if value.ndim and int(value.shape[0]) == n_paths:
                fields.append((name, value[path]))
        hashes.append(_tensor_digest(fields))
    return {
        "n_paths": n_paths,
        "hashed_economic_fields": list(economic_cross_section_fields),
        "unique_initial_cross_section_count": len(set(hashes)),
        "all_initial_cross_sections_unique": len(set(hashes)) == n_paths,
        "path_hashes": hashes,
    }


def _simulation_kwargs(episode: Any, transition_seed: int, entry_seed: int) -> dict[str, Any]:
    # This is the exact production constructor surface used by
    # analysis.run_macro_outer_inner_pq._simulation_kwargs.  Keeping it local
    # avoids an import cycle with the full nested runner; tests compare both.
    names = (
        "entry_mode",
        "entry_spec_version",
        "entry_capital_ratio",
        "entry_size_ratio",
        "entry_cost_max",
        "entry_dummy_i",
        "entry_inference_chunk_size",
        "consumption_aggregation_mode",
        "consumption_floor",
        "node_accounting_mode",
        "entry_independent_resource_validation_enabled",
        "preserve_global_rng_around_entry",
    )
    kwargs = {name: getattr(episode.hyperparams, name) for name in names}
    kwargs.update(
        {
            "entry_rng_seed": int(entry_seed),
            "entry_rng_mode": "stable_economic_identity",
            "common_transition_seed": int(transition_seed),
            "transition_rng_mode": "stable_firm_identity",
            "bp_action_source": "head",
        }
    )
    return kwargs


def _allowed_simulation_models(
    models: Mapping[str, torch.nn.Module],
) -> dict[str, torch.nn.Module]:
    return {
        name: model
        for name, model in models.items()
        if name in {"policy_value", "sdf_fc1", "dist_b"}
    }


def run_short_rollout_harvest(
    *,
    models: Mapping[str, torch.nn.Module],
    episode: Any,
    initial_state: Mapping[str, torch.Tensor],
    split: str,
    source_episode: int,
    initial_state_seed: int,
    transition_seed: int,
    entry_seed: int,
    short_rollout_horizon: int,
    capture_times: Sequence[int],
    branch_num: int,
    device: torch.device,
) -> tuple[StateBank, TensorSimulationOutput, dict[str, Any]]:
    effective = effective_capture_times(capture_times, short_rollout_horizon)
    compatibility = build_state_bank_compatibility(
        config=episode.config,
        hyperparams=episode.hyperparams,
        branch_num=int(branch_num),
    )
    bank = StateBank(
        compatibility=compatibility,
        source_episode=int(source_episode),
        metadata={
            "purpose": "macro_parent_state_bank",
            "split": split,
            "initial_state_seed": int(initial_state_seed),
            "requested_capture_times": list(parse_capture_times(capture_times)),
            "effective_capture_times": list(effective),
        },
    )
    capture = ExactTimeStateBankCapture(
        bank,
        split=split,
        capture_times=effective,
        initial_state_seed=initial_state_seed,
    )
    n_paths = int(initial_state["b"].shape[0])
    simulator = SimulateTS(
        models=_allowed_simulation_models(models),
        config=episode.config,
        n_paths=n_paths,
        group_size=int(initial_state["b"].shape[1]),
        horizon=int(short_rollout_horizon),
        branch_num=int(branch_num),
        device=device,
        initial_state=clone_state(initial_state, device="cpu"),
        state_bank_capture=capture,
        **_simulation_kwargs(episode, transition_seed, entry_seed),
    )
    output = simulator.simulate_tensor()
    audit = initial_cross_section_hash_audit(initial_state)
    audit.update(
        {
            "capture_calls": int(capture.capture_calls),
            "candidate_parent_count": len(bank),
            "requested_capture_times": list(parse_capture_times(capture_times)),
            "effective_capture_times": list(effective),
            "short_rollout_horizon": int(short_rollout_horizon),
        }
    )
    return bank, output, audit


def _finite_quantiles(values: torch.Tensor) -> dict[str, float]:
    finite = values.detach().to(torch.float64)
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        return {name: float("nan") for name in ("mean", "std", "p10", "p50", "p90")}
    q = torch.quantile(finite, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(unbiased=False)),
        "p10": float(q[0]),
        "p50": float(q[1]),
        "p90": float(q[2]),
    }


def _diagnostic_bin_edges(config: type = Config) -> tuple[np.ndarray, np.ndarray]:
    b_internal = np.linspace(
        float(config.SIM_B_INIT_MIN), float(config.SIM_B_INIT_MAX), 11
    )[1:-1]
    stationary_z_std = float(config.SIGMA_Z) / math.sqrt(
        max(1.0 - float(config.RHO_Z) ** 2, 1e-12)
    )
    z_internal = np.linspace(
        float(config.ZBAR) - 4.0 * stationary_z_std,
        float(config.ZBAR) + 4.0 * stationary_z_std,
        11,
    )[1:-1]
    # Infinite edge bins retain, rather than clip or discard, off-range states.
    return (
        np.concatenate(([-np.inf], b_internal, [np.inf])),
        np.concatenate(([-np.inf], z_internal, [np.inf])),
    )


def summarize_parent_bank(
    bank: StateBank,
    macro_table: TensorTable,
    *,
    split: str,
    initial_state_seed: int,
    config: type = Config,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    columns = {name: index for index, name in enumerate(macro_table.columns)}
    required = {"path", "t", "branch", "Hatc", "LnK", "x"}
    missing = sorted(required.difference(columns))
    if missing:
        raise RuntimeError(f"macro parent summary missing production columns: {missing}")
    macro = macro_table.data.detach().cpu()
    parent_mask = macro[:, columns["branch"]].to(torch.long) == -1
    lookup: dict[tuple[int, int], torch.Tensor] = {}
    for row in macro[parent_mask]:
        key = (int(row[columns["path"]].item()), int(row[columns["t"]].item()))
        if key in lookup:
            raise RuntimeError(f"duplicate production parent macro row: {key}")
        lookup[key] = row

    b_edges, z_edges = _diagnostic_bin_edges(config)
    occupancies = []
    rows = []
    for entry in bank.entries:
        path = int(entry["source_path"])
        capture_time = int(entry["capture_time"])
        macro_row = lookup.get((path, capture_time))
        if macro_row is None:
            raise RuntimeError(
                f"captured parent has no production macro row: path={path}, t={capture_time}"
            )
        fields = entry["fields"]
        alive = fields["alive"].to(torch.bool)
        b = fields["b"][alive].to(torch.float64)
        z = fields["z"][alive].to(torch.float64)
        eta = fields["eta"][alive].to(torch.float64)
        investment_cost = fields["i"][alive].to(torch.float64)
        capital = fields["K"][alive].to(torch.float64)
        if b.numel() > 1 and float(b.std(unbiased=False)) > 0 and float(z.std(unbiased=False)) > 0:
            corr = float(torch.corrcoef(torch.stack([b, z]))[0, 1])
        else:
            corr = float("nan")
        b_stats = _finite_quantiles(b)
        z_stats = _finite_quantiles(z)
        i_stats = _finite_quantiles(investment_cost)
        k_total = float(capital.sum())
        source_key = ParentSourceKey(
            split=split,
            initial_state_seed=initial_state_seed,
            initial_path_id=path,
            capture_time=capture_time,
        )
        occupancy, _, _ = np.histogram2d(
            b.numpy(), z.numpy(), bins=(b_edges, z_edges)
        )
        occupancies.append(occupancy.astype(np.int64))
        rows.append(
            {
                "source_key": source_key.canonical,
                **source_key.to_dict(),
                "entry_id": str(entry["entry_id"]),
                "economic_time": int(entry["economic_time"]),
                "n_alive": int(alive.sum()),
                **{f"b_{key}": value for key, value in b_stats.items()},
                **{f"z_{key}": value for key, value in z_stats.items()},
                "corr_b_z": corr,
                "eta_share": float(eta.mean()) if eta.numel() else float("nan"),
                "i_mean": i_stats["mean"],
                "i_std": i_stats["std"],
                "K_total": k_total,
                "K_mean": float(capital.mean()) if capital.numel() else float("nan"),
                "log_K_total": math.log(max(k_total, 1e-30)),
                "Hatc": float(macro_row[columns["Hatc"]]),
                "LnK": float(macro_row[columns["LnK"]]),
                "x": float(macro_row[columns["x"]]),
            }
        )
    occupancy_array = np.stack(occupancies, axis=0) if occupancies else np.zeros((0, 10, 10), dtype=np.int64)
    metadata = {
        "b_range": [float(config.SIM_B_INIT_MIN), float(config.SIM_B_INIT_MAX)],
        "z_reasonable_range": [float(z_edges[1]), float(z_edges[-2])],
        "outliers_retained_in_infinite_edge_bins": True,
        "occupancy_shape": list(occupancy_array.shape),
    }
    return pd.DataFrame(rows), occupancy_array, metadata


def _coverage_matrix(frame: pd.DataFrame) -> np.ndarray:
    raw = frame.loc[:, SUMMARY_FEATURES].to_numpy(dtype=np.float64, copy=True)
    for column in range(raw.shape[1]):
        finite = np.isfinite(raw[:, column])
        replacement = float(np.median(raw[finite, column])) if finite.any() else 0.0
        raw[~finite, column] = replacement
    center = np.median(raw, axis=0)
    scale = np.quantile(raw, 0.75, axis=0) - np.quantile(raw, 0.25, axis=0)
    fallback = raw.std(axis=0)
    scale = np.where(scale > 1e-12, scale, np.where(fallback > 1e-12, fallback, 1.0))
    return (raw - center) / scale


def deterministic_parent_selection(
    candidate_summary: pd.DataFrame,
    *,
    target_count: int,
    coverage_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if candidate_summary.empty:
        raise RuntimeError("cannot select from an empty macro parent candidate pool")
    if not 0.0 <= float(coverage_fraction) <= 1.0:
        raise ValueError("coverage fraction must be in [0, 1]")
    count = min(int(target_count), len(candidate_summary))
    n_coverage = min(count, int(round(count * float(coverage_fraction))))
    n_endogenous = count - n_coverage
    generator = np.random.default_rng(int(seed))
    permutation = generator.permutation(len(candidate_summary))
    endogenous = list(map(int, permutation[:n_endogenous]))
    remaining = list(map(int, permutation[n_endogenous:]))
    standardized = _coverage_matrix(candidate_summary)
    if endogenous:
        distance = np.min(
            np.sum(
                (standardized[remaining, None, :] - standardized[endogenous][None, :, :]) ** 2,
                axis=2,
            ),
            axis=1,
        )
    else:
        center = np.median(standardized, axis=0)
        distance = np.sum((standardized[remaining] - center) ** 2, axis=1)
    coverage: list[int] = []
    for _ in range(n_coverage):
        if not remaining:
            break
        position = int(np.argmax(distance))
        selected_index = remaining.pop(position)
        coverage.append(selected_index)
        distance = np.delete(distance, position)
        if remaining:
            new_distance = np.sum(
                (standardized[remaining] - standardized[selected_index]) ** 2, axis=1
            )
            distance = np.minimum(distance, new_distance)
    selected_indices = endogenous + coverage
    selected = candidate_summary.iloc[selected_indices].copy().reset_index(drop=True)
    selected["selection_component"] = (
        ["ENDOGENOUS_FREQUENCY"] * len(endogenous)
        + ["COVERAGE_AWARE"] * len(coverage)
    )
    report = {
        "candidate_count": int(len(candidate_summary)),
        "requested_count": int(target_count),
        "selected_count": int(len(selected)),
        "endogenous_frequency_count": int(len(endogenous)),
        "coverage_aware_count": int(len(coverage)),
        "endogenous_frequency_share": float(len(endogenous) / max(len(selected), 1)),
        "coverage_aware_share": float(len(coverage) / max(len(selected), 1)),
        "selection_seed": int(seed),
        "candidate_shortfall": max(0, int(target_count) - len(candidate_summary)),
    }
    return selected, report


def build_source_schedule(
    selected_summary: pd.DataFrame,
    *,
    split: str,
    initial_state_seed: int,
    transition_seed: int,
    entry_seed: int,
    candidate_paths: int,
    short_rollout_horizon: int,
    requested_capture_times: Sequence[int],
) -> MacroParentSourceSchedule:
    keys = tuple(
        ParentSourceKey.from_dict(row)
        for row in selected_summary.loc[
            :, ("split", "initial_state_seed", "initial_path_id", "capture_time", "source_type")
        ].to_dict(orient="records")
    )
    component = selected_summary.get("selection_component", pd.Series(dtype=str))
    n = max(len(selected_summary), 1)
    return MacroParentSourceSchedule(
        split=split,
        initial_state_seed=int(initial_state_seed),
        transition_seed=int(transition_seed),
        entry_seed=int(entry_seed),
        candidate_paths=int(candidate_paths),
        short_rollout_horizon=int(short_rollout_horizon),
        requested_capture_times=parse_capture_times(requested_capture_times),
        effective_capture_times=effective_capture_times(
            requested_capture_times, short_rollout_horizon
        ),
        source_keys=keys,
        endogenous_frequency_share=float((component == "ENDOGENOUS_FREQUENCY").sum() / n),
        coverage_aware_share=float((component == "COVERAGE_AWARE").sum() / n),
    )


def select_scheduled_entries(
    bank: StateBank,
    summary: pd.DataFrame,
    schedule: MacroParentSourceSchedule,
) -> tuple[list[str], pd.DataFrame, np.ndarray]:
    by_key = {str(row.source_key): str(row.entry_id) for row in summary.itertuples()}
    requested = [key.canonical for key in schedule.source_keys]
    missing = [key for key in requested if key not in by_key]
    if missing:
        raise RuntimeError(
            "fixed macro parent source schedule could not be reproduced; "
            f"missing={len(missing)}, first={missing[:3]}"
        )
    entry_ids = [by_key[key] for key in requested]
    frame = summary.set_index("source_key").loc[requested].reset_index()
    b_edges, z_edges = _diagnostic_bin_edges(Config)
    occupancy = []
    for entry_id in entry_ids:
        fields = bank._entry_index[entry_id]["fields"]
        alive = fields["alive"].to(torch.bool)
        hist, _, _ = np.histogram2d(
            fields["b"][alive].to(torch.float64).numpy(),
            fields["z"][alive].to(torch.float64).numpy(),
            bins=(b_edges, z_edges),
        )
        occupancy.append(hist.astype(np.int64))
    return entry_ids, frame, np.stack(occupancy, axis=0)


def _offset_table_paths(table: TensorTable, offset: int) -> TensorTable:
    data = table.data.detach().cpu().clone()
    column = table.columns.index("path")
    data[:, column] += int(offset)
    return TensorTable(data=data, columns=table.columns)


def _cat_tables(tables: Sequence[TensorTable]) -> TensorTable:
    if not tables:
        raise RuntimeError("cannot concatenate an empty production simulation output")
    columns = tables[0].columns
    if any(table.columns != columns for table in tables):
        raise RuntimeError("production simulation table schemas differ across chunks")
    return TensorTable(data=torch.cat([table.data for table in tables], dim=0), columns=columns)


def _simulate_one_step_with_batch_size(
    *,
    models: Mapping[str, torch.nn.Module],
    episode: Any,
    bank: StateBank,
    entry_ids: Sequence[str],
    parent_batch_size: int,
    branch_num: int,
    transition_seed: int,
    entry_seed: int,
    device: torch.device,
    path_offset: int,
) -> TensorTable:
    macro_chunks = []
    for start in range(0, len(entry_ids), int(parent_batch_size)):
        ids = list(entry_ids[start : start + int(parent_batch_size)])
        initial = bank.stack_entries(ids, device="cpu")
        simulator = SimulateTS(
            models=_allowed_simulation_models(models),
            config=episode.config,
            n_paths=len(ids),
            group_size=int(initial["b"].shape[1]),
            horizon=1,
            branch_num=int(branch_num),
            device=device,
            initial_state=initial,
            **_simulation_kwargs(episode, transition_seed, entry_seed),
        )
        output = simulator.simulate_tensor()
        macro_chunks.append(_offset_table_paths(output.macro, path_offset + start))
        del output, simulator, initial
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return _cat_tables(macro_chunks)


def simulate_one_step_chunked(
    *,
    models: Mapping[str, torch.nn.Module],
    episode: Any,
    bank: StateBank,
    entry_ids: Sequence[str],
    parent_batch_size: int,
    branch_num: int,
    transition_seed: int,
    entry_seed: int,
    device: torch.device,
    path_offset: int = 0,
) -> tuple[TensorTable, int]:
    if int(branch_num) != 2:
        raise ValueError("macro one-step SDF dataset requires branch_num=2")
    candidates = []
    for size in (int(parent_batch_size), 128, 64):
        size = min(size, len(entry_ids))
        if size > 0 and size not in candidates:
            candidates.append(size)
    last_error: Optional[BaseException] = None
    for size in candidates:
        try:
            return (
                _simulate_one_step_with_batch_size(
                    models=models,
                    episode=episode,
                    bank=bank,
                    entry_ids=entry_ids,
                    parent_batch_size=size,
                    branch_num=branch_num,
                    transition_seed=transition_seed,
                    entry_seed=entry_seed,
                    device=device,
                    path_offset=path_offset,
                ),
                size,
            )
        except torch.cuda.OutOfMemoryError as exc:
            last_error = exc
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError(
        f"macro parent one-step simulation OOM persisted through {candidates}"
    ) from last_error


def audit_one_step_pairing(
    pairs: TensorTable,
    *,
    expected_parent_count: int,
    branch_num: int,
    path_offset: int = 0,
) -> dict[str, Any]:
    columns = {name: index for index, name in enumerate(pairs.columns)}
    required = {"path", "t", "branch"}
    if not required.issubset(columns):
        raise RuntimeError("production pair table lacks path/t/branch identity")
    data = pairs.data.detach().cpu()
    paths = data[:, columns["path"]].to(torch.long)
    times = data[:, columns["t"]].to(torch.long)
    branches = data[:, columns["branch"]].to(torch.long)
    expected_paths = set(range(int(path_offset), int(path_offset) + int(expected_parent_count)))
    seen_paths = set(map(int, torch.unique(paths).tolist()))
    failures = []
    for path in sorted(expected_paths):
        mask = paths == path
        observed = sorted(map(int, branches[mask].tolist()))
        observed_times = sorted(set(map(int, times[mask].tolist())))
        if observed != list(range(int(branch_num))) or observed_times != [1]:
            failures.append(
                {"path": path, "branches": observed, "child_times": observed_times}
            )
    unexpected = sorted(seen_paths.difference(expected_paths))
    audit = {
        "expected_parent_count": int(expected_parent_count),
        "observed_parent_count": int(len(seen_paths)),
        "pair_row_count": int(len(data)),
        "branch_num": int(branch_num),
        "expected_pair_row_count": int(expected_parent_count * branch_num),
        "unexpected_paths": unexpected,
        "failed_parent_groups": failures,
        "each_parent_has_exact_children": not failures and not unexpected and seen_paths == expected_paths,
        "parent_local_t": 0,
        "child_local_t": 1,
    }
    if not audit["each_parent_has_exact_children"]:
        raise RuntimeError(f"one-step parent/child pairing invariant failed: {audit}")
    return audit


def build_one_step_macro_dataset(
    *,
    models: Mapping[str, torch.nn.Module],
    episode: Any,
    parent_bank: HarvestedParentBank,
    parent_batch_size: int,
    macro_batch_size: int,
    branch_num: int,
    transition_seed: int,
    entry_seed: int,
    device: torch.device,
    path_offset: int = 0,
) -> OneStepMacroDataset:
    macro, used_batch_size = simulate_one_step_chunked(
        models=models,
        episode=episode,
        bank=parent_bank.bank,
        entry_ids=parent_bank.selected_entry_ids,
        parent_batch_size=parent_batch_size,
        branch_num=branch_num,
        transition_seed=transition_seed,
        entry_seed=entry_seed,
        device=device,
        path_offset=path_offset,
    )
    pairs = episode._build_sdf_pairs_from_macro_tensor(macro.to(device))
    audit = audit_one_step_pairing(
        pairs,
        expected_parent_count=parent_bank.n_parents,
        branch_num=branch_num,
        path_offset=path_offset,
    )
    episode.add_FC1loss = True
    target_batches = int(getattr(episode.hyperparams, "sdf_true_target_batches", 20))
    min_parents = int(getattr(episode.hyperparams, "sdf_min_parent_groups_per_batch", 256))
    batches = {
        "fc1": episode._create_sdf_batches_from_macro_tensor(
            pairs, batch_size=macro_batch_size, n_branches=branch_num
        ),
        "sdf": episode._create_sdf_batches_from_macro_tensor(
            pairs,
            batch_size=macro_batch_size,
            n_branches=branch_num,
            target_num_batches=target_batches,
            min_parent_groups_per_batch=min_parents,
        ),
    }
    if not batches["fc1"] or not batches["sdf"]:
        raise RuntimeError("one-step production pairing produced an empty macro batch stream")
    return OneStepMacroDataset(
        macro=macro,
        pairs=pairs,
        batches=batches,
        pairing_audit=audit,
        used_parent_batch_size=used_batch_size,
    )


def distribution_coverage(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"count": int(len(frame))}
    for name in (*SUMMARY_FEATURES, "K_total", "Hatc", "LnK", "n_alive"):
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if not finite.size:
            result[name] = {key: None for key in ("min", "p10", "p50", "p90", "max")}
            continue
        result[name] = {
            "min": float(np.min(finite)),
            "p10": float(np.quantile(finite, 0.10)),
            "p50": float(np.quantile(finite, 0.50)),
            "p90": float(np.quantile(finite, 0.90)),
            "max": float(np.max(finite)),
        }
    return result


def parent_summary_movement(
    old: pd.DataFrame, new: pd.DataFrame
) -> dict[str, float]:
    left = old.set_index("source_key")
    right = new.set_index("source_key")
    if set(left.index) != set(right.index):
        raise RuntimeError("macro parent-bank movement source keys are not aligned")
    right = right.loc[left.index]
    result = {}
    for name in (*SUMMARY_FEATURES, "K_total", "Hatc", "LnK", "n_alive"):
        old_value = torch.as_tensor(left[name].to_numpy(dtype=np.float64))
        new_value = torch.as_tensor(right[name].to_numpy(dtype=np.float64))
        finite = torch.isfinite(old_value) & torch.isfinite(new_value)
        relative = (new_value[finite] - old_value[finite]).abs() / (
            1.0 + old_value[finite].abs()
        )
        result[f"{name}_nmae"] = float(relative.mean()) if relative.numel() else float("nan")
        result[f"{name}_p90"] = (
            float(torch.quantile(relative, 0.90)) if relative.numel() else float("nan")
        )
    return result


def save_schedule(path: Path, schedule: MacroParentSourceSchedule) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(schedule.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )


def load_schedule(path: Path) -> MacroParentSourceSchedule:
    return MacroParentSourceSchedule.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


def save_bank_shards(
    output: Path,
    harvested: HarvestedParentBank,
    *,
    shard_size: int = 128,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    selected_entries = [
        harvested.bank._entry_index[entry_id] for entry_id in harvested.selected_entry_ids
    ]
    shard_paths = []
    for shard_id, start in enumerate(range(0, len(selected_entries), int(shard_size))):
        shard = StateBank(
            compatibility=harvested.bank.compatibility,
            source_episode=harvested.bank.source_episode,
            entries=selected_entries[start : start + int(shard_size)],
            field_schema=harvested.bank.field_schema,
            metadata={**harvested.bank.metadata, "shard_id": shard_id},
        )
        path = output / f"bank_shard_{shard_id:03d}.pt"
        shard.save(path)
        shard_paths.append(path.name)
    metadata = {
        "n_parents": harvested.n_parents,
        "shard_size": int(shard_size),
        "shards": shard_paths,
        "state_schema": harvested.bank.field_schema,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def source_overlap(
    train: MacroParentSourceSchedule, validation: MacroParentSourceSchedule
) -> dict[str, Any]:
    train_keys = {key.canonical for key in train.source_keys}
    validation_keys = {key.canonical for key in validation.source_keys}
    overlap = sorted(train_keys.intersection(validation_keys))
    return {
        "train_source_count": len(train_keys),
        "validation_source_count": len(validation_keys),
        "overlap_count": len(overlap),
        "overlap": overlap,
    }


def prepare_macro_parent_schedule(
    *,
    models: Mapping[str, torch.nn.Module],
    episode: Any,
    initial_state: Mapping[str, torch.Tensor],
    split: str,
    source_episode: int,
    initial_state_seed: int,
    transition_seed: int,
    entry_seed: int,
    short_rollout_horizon: int,
    capture_times: Sequence[int],
    branch_num: int,
    target_count: int,
    coverage_fraction: float,
    selection_seed: int,
    device: torch.device,
) -> tuple[
    MacroParentSourceSchedule,
    HarvestedParentBank,
    pd.DataFrame,
    np.ndarray,
    dict[str, Any],
]:
    bank, output, initial_audit = run_short_rollout_harvest(
        models=models,
        episode=episode,
        initial_state=initial_state,
        split=split,
        source_episode=source_episode,
        initial_state_seed=initial_state_seed,
        transition_seed=transition_seed,
        entry_seed=entry_seed,
        short_rollout_horizon=short_rollout_horizon,
        capture_times=capture_times,
        branch_num=branch_num,
        device=device,
    )
    candidate, candidate_occupancy, histogram_metadata = summarize_parent_bank(
        bank,
        output.macro,
        split=split,
        initial_state_seed=initial_state_seed,
        config=episode.config,
    )
    selected, selection_report = deterministic_parent_selection(
        candidate,
        target_count=target_count,
        coverage_fraction=coverage_fraction,
        seed=selection_seed,
    )
    schedule = build_source_schedule(
        selected,
        split=split,
        initial_state_seed=initial_state_seed,
        transition_seed=transition_seed,
        entry_seed=entry_seed,
        candidate_paths=int(initial_state["b"].shape[0]),
        short_rollout_horizon=short_rollout_horizon,
        requested_capture_times=capture_times,
    )
    entry_ids, selected_aligned, selected_occupancy = select_scheduled_entries(
        bank, candidate, schedule
    )
    selected_aligned = selected_aligned.merge(
        selected.loc[:, ["source_key", "selection_component"]],
        on="source_key",
        how="left",
        validate="one_to_one",
    )
    harvested = HarvestedParentBank(
        bank=bank,
        selected_entry_ids=entry_ids,
        summary=selected_aligned,
        occupancy=selected_occupancy,
        simulation_output=output,
        initial_hash_audit=initial_audit,
    )
    report = {
        **selection_report,
        "schedule_fingerprint": schedule.fingerprint,
        "candidate_coverage": distribution_coverage(candidate),
        "selected_coverage": distribution_coverage(selected_aligned),
        "histogram": histogram_metadata,
        "unique_source_paths": int(selected_aligned["initial_path_id"].nunique()),
        "capture_time_distribution": {
            str(int(key)): int(value)
            for key, value in selected_aligned["capture_time"].value_counts().sort_index().items()
        },
        "initial_cross_sections": initial_audit,
    }
    return (
        schedule,
        harvested,
        candidate,
        candidate_occupancy,
        report,
    )


def harvest_macro_parent_schedule(
    *,
    models: Mapping[str, torch.nn.Module],
    episode: Any,
    initial_state: Mapping[str, torch.Tensor],
    schedule: MacroParentSourceSchedule,
    source_episode: int,
    branch_num: int,
    device: torch.device,
) -> HarvestedParentBank:
    bank, output, initial_audit = run_short_rollout_harvest(
        models=models,
        episode=episode,
        initial_state=initial_state,
        split=schedule.split,
        source_episode=source_episode,
        initial_state_seed=schedule.initial_state_seed,
        transition_seed=schedule.transition_seed,
        entry_seed=schedule.entry_seed,
        short_rollout_horizon=schedule.short_rollout_horizon,
        capture_times=schedule.requested_capture_times,
        branch_num=branch_num,
        device=device,
    )
    summary, _, _ = summarize_parent_bank(
        bank,
        output.macro,
        split=schedule.split,
        initial_state_seed=schedule.initial_state_seed,
        config=episode.config,
    )
    entry_ids, selected, occupancy = select_scheduled_entries(bank, summary, schedule)
    return HarvestedParentBank(
        bank=bank,
        selected_entry_ids=entry_ids,
        summary=selected,
        occupancy=occupancy,
        simulation_output=output,
        initial_hash_audit=initial_audit,
    )


def harvest_macro_parent_schedule_with_candidate(
    *,
    models: Mapping[str, torch.nn.Module],
    episode: Any,
    initial_state: Mapping[str, torch.Tensor],
    schedule: MacroParentSourceSchedule,
    source_episode: int,
    branch_num: int,
    device: torch.device,
) -> tuple[HarvestedParentBank, pd.DataFrame, np.ndarray]:
    bank, output, initial_audit = run_short_rollout_harvest(
        models=models,
        episode=episode,
        initial_state=initial_state,
        split=schedule.split,
        source_episode=source_episode,
        initial_state_seed=schedule.initial_state_seed,
        transition_seed=schedule.transition_seed,
        entry_seed=schedule.entry_seed,
        short_rollout_horizon=schedule.short_rollout_horizon,
        capture_times=schedule.requested_capture_times,
        branch_num=branch_num,
        device=device,
    )
    candidate, candidate_occupancy, _ = summarize_parent_bank(
        bank,
        output.macro,
        split=schedule.split,
        initial_state_seed=schedule.initial_state_seed,
        config=episode.config,
    )
    entry_ids, selected, occupancy = select_scheduled_entries(bank, candidate, schedule)
    return (
        HarvestedParentBank(
            bank=bank,
            selected_entry_ids=entry_ids,
            summary=selected,
            occupancy=occupancy,
            simulation_output=output,
            initial_hash_audit=initial_audit,
        ),
        candidate,
        candidate_occupancy,
    )


def write_coverage_artifacts(
    output: Path,
    *,
    train_candidate: pd.DataFrame,
    train_selected: pd.DataFrame,
    validation_candidate: pd.DataFrame,
    validation_selected: pd.DataFrame,
    train_candidate_occupancy: np.ndarray,
    train_selected_occupancy: np.ndarray,
    validation_candidate_occupancy: np.ndarray,
    validation_selected_occupancy: np.ndarray,
    train_schedule: MacroParentSourceSchedule,
    validation_schedule: MacroParentSourceSchedule,
    train_report: Mapping[str, Any],
    validation_report: Mapping[str, Any],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    save_schedule(output / "train_source_schedule.json", train_schedule)
    save_schedule(output / "validation_source_schedule.json", validation_schedule)
    combined_candidate = pd.concat(
        [train_candidate, validation_candidate], ignore_index=True
    )
    combined_candidate.to_csv(output / "candidate_summary.csv", index=False)
    train_selected.to_csv(output / "selected_train_summary.csv", index=False)
    validation_selected.to_csv(output / "selected_validation_summary.csv", index=False)
    np.save(output / "candidate_bz_occupancy.npy", np.concatenate(
        [train_candidate_occupancy, validation_candidate_occupancy], axis=0
    ))
    np.save(output / "train_bz_occupancy.npy", train_selected_occupancy)
    np.save(output / "validation_bz_occupancy.npy", validation_selected_occupancy)
    split = source_overlap(train_schedule, validation_schedule)
    if split["overlap_count"]:
        raise RuntimeError(f"macro parent train/validation source leakage: {split}")
    coverage_warning = bool(
        train_report.get("candidate_shortfall", 0)
        or validation_report.get("candidate_shortfall", 0)
        or train_report["selected_coverage"].get("b_std", {}).get("max")
        == train_report["selected_coverage"].get("b_std", {}).get("min")
    )
    report = {
        "train": dict(train_report),
        "validation": dict(validation_report),
        "train_validation_source_split": split,
        "coverage_warning": coverage_warning,
        "coverage_warning_label": "COVERAGE_WARNING" if coverage_warning else None,
        "synthetic_state_perturbation_used": False,
    }
    (output / "coverage_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_coverage_figures(
        output / "figures",
        candidate=train_candidate,
        selected=train_selected,
        selected_occupancy=train_selected_occupancy,
    )
    return report


def _write_coverage_figures(
    output: Path,
    *,
    candidate: pd.DataFrame,
    selected: pd.DataFrame,
    selected_occupancy: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(output / name, dpi=160)
        plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(
        candidate["b_mean"], candidate["z_mean"], s=8, alpha=0.25, label="candidate"
    )
    plt.scatter(
        selected["b_mean"], selected["z_mean"], s=12, alpha=0.65, label="selected"
    )
    plt.xlabel("parent mean b")
    plt.ylabel("parent mean z")
    plt.legend()
    save("selected_parent_mean_b_vs_mean_z.png")

    for variable in ("b_mean", "z_mean"):
        plt.figure(figsize=(6, 4))
        plt.hist(candidate[variable], bins=30, alpha=0.45, density=True, label="candidate")
        plt.hist(selected[variable], bins=30, alpha=0.55, density=True, label="selected")
        plt.xlabel(variable)
        plt.ylabel("density")
        plt.legend()
        save(f"candidate_vs_selected_{variable}.png")

    plt.figure(figsize=(6, 4))
    plt.hist(candidate["corr_b_z"].dropna(), bins=30, alpha=0.45, density=True, label="candidate")
    plt.hist(selected["corr_b_z"].dropna(), bins=30, alpha=0.55, density=True, label="selected")
    plt.xlabel("corr(b,z) within parent economy")
    plt.ylabel("density")
    plt.legend()
    save("parent_corr_b_z_distribution.png")

    plt.figure(figsize=(6, 4))
    plt.hist(candidate["K_total"], bins=30, alpha=0.45, density=True, label="candidate")
    plt.hist(selected["K_total"], bins=30, alpha=0.55, density=True, label="selected")
    plt.xlabel("parent K_total")
    plt.ylabel("density")
    plt.legend()
    save("parent_K_total_distribution.png")

    plt.figure(figsize=(6, 4))
    selected["capture_time"].value_counts().sort_index().plot(kind="bar")
    plt.xlabel("capture time")
    plt.ylabel("selected parent count")
    save("capture_time_distribution.png")

    plt.figure(figsize=(6, 5))
    plt.scatter(selected["b_mean"], selected["corr_b_z"], s=12, alpha=0.65)
    plt.xlabel("parent mean b")
    plt.ylabel("parent corr(b,z)")
    save("parent_mean_b_vs_corr_bz.png")

    plt.figure(figsize=(6, 5))
    aggregate = selected_occupancy.sum(axis=0)
    plt.imshow(aggregate.T, origin="lower", aspect="auto")
    plt.colorbar(label="firm observations")
    plt.xlabel("b diagnostic bin")
    plt.ylabel("z diagnostic bin")
    save("selected_parent_bz_occupancy.png")


def write_macro_inner_movement_figure(history: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    if history.empty:
        return
    output.mkdir(parents=True, exist_ok=True)
    x = history["macro_inner"]
    for filename, columns, ylabel in (
        (
            "macro_inner_parent_bank_movement.png",
            ("parent_bank_summary_movement_max",),
            "parent summary relative movement",
        ),
        (
            "macro_inner_one_step_target_movement.png",
            ("Hatc_target_move", "LnK_target_move"),
            "one-step target NMAE",
        ),
    ):
        plt.figure(figsize=(6, 4))
        for column in columns:
            if column in history:
                plt.plot(x, history[column], marker="o", label=column)
        if len(columns) > 1:
            plt.legend()
        plt.xlabel("macro_inner")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(output / filename, dpi=160)
        plt.close()

