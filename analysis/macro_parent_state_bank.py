"""Macro parent-bank construction using production short simulations.

The module deliberately owns only *which production states are simulated*.
It does not reproduce firm transitions, entry/exit, accounting, FC1 targets, or
SDF pairing.  Complete parent-ready states are captured through ``StateBank``
at the top of the production tensor simulation loop and are resumed through
``SimulateTS(initial_state=..., horizon=1, branch_num=2)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.special import betaincinv

from config import Config
from data.boundary_aware_distribution import (
    BOUNDARY_AWARE_CONTINUUM_V1,
    PreDecisionDistribution,
    gaussian_mixture_moments,
    propagate_ar1_with_eta_mixture,
)
from data.boundary_aware_market_clearing import (
    BoundaryAwareMarketClearingController,
    BoundaryIntegrationConfig,
)
from data.entry import EntryCandidates
from data.market_clearing import (
    FAST_STATUS_LABELS,
    MarketClearingConfig,
    MarketClearingError,
    empty_entry_candidates,
)
from data.simulate_ts import SimulateTS
from data.simulate_ts_parallel import (
    _keyed_normal,
    _keyed_uniform,
    _predict_macro_fc1_batched,
)
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


MACRO_CANDIDATE_STATUS_VALID = "VALID_COMPLETE_PAIR"
MACRO_CANDIDATE_FAILURE_STATUSES = (
    "INVALID_INITIAL_DISTRIBUTION",
    "PARENT_CLEARING_FAILED",
    "CHILD_0_CLEARING_FAILED",
    "CHILD_1_CLEARING_FAILED",
    "CHILD_EXTINCT",
    "NUMERICAL_MASS_LOSS",
    "NUMERICAL_FAILURE",
    "ENGINEERING_FAILURE",
)

# These strata are a training-support design, not economic calibration.  The
# intervals deliberately overlap and Sobol coordinates vary alpha/beta inside
# every stratum, so the schedule is not a short list of repeated fixed laws.
B_DISTRIBUTION_STRATA: tuple[tuple[str, tuple[float, float], tuple[float, float]], ...] = (
    ("low_leverage_concentration", (0.65, 2.00), (4.00, 10.00)),
    ("medium_low_leverage_concentration", (1.50, 3.50), (3.50, 7.00)),
    ("middle_concentration", (2.00, 8.00), (2.00, 8.00)),
    ("medium_high_leverage_concentration", (3.50, 7.00), (1.50, 3.50)),
    ("high_leverage_concentration", (4.00, 10.00), (0.65, 2.00)),
    ("wide_dispersion", (0.80, 2.25), (0.80, 2.25)),
    ("two_tail_concentration", (0.25, 0.90), (0.25, 0.90)),
)

# New candidate schedules default to the leverage neighborhood of validation
# candidate 42000000 (observed mean 0.254999).  Existing serialized schedules
# remain replayable because the cap is applied only while constructing a new
# schedule; it never rewrites or clips an already committed distribution.
DEFAULT_MAX_INITIAL_WEIGHTED_B_MEAN = 0.30
DEFAULT_MACRO_CANDIDATE_DISTRIBUTION_DESIGN = "stratified_beta_v1"
RESTRICTED_BETA_FULL_OUTER_V1 = "restricted_beta_full_outer_v1"
MACRO_CANDIDATE_DISTRIBUTION_DESIGNS = (
    DEFAULT_MACRO_CANDIDATE_DISTRIBUTION_DESIGN,
    RESTRICTED_BETA_FULL_OUTER_V1,
)
RESTRICTED_BETA_FULL_OUTER_ALPHA_RANGE = (1.8, 3.2)
RESTRICTED_BETA_FULL_OUTER_BETA_RANGE = (4.8, 7.8)


def midpoint_beta_mean(alpha: float, beta: float, points: int) -> float:
    """Mean of the exact midpoint-quantile representation used at simulation."""

    if float(alpha) <= 0.0 or float(beta) <= 0.0 or int(points) < 2:
        raise ValueError("Beta parameters must be positive and points >= 2")
    quantiles = (np.arange(int(points), dtype=np.float64) + 0.5) / float(points)
    values = betaincinv(float(alpha), float(beta), quantiles)
    if not np.isfinite(values).all():
        raise RuntimeError("Beta inverse quantiles are nonfinite")
    return float(np.mean(values))


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


def _model_state_digest(model: torch.nn.Module) -> str:
    return _tensor_digest(
        (name, value) for name, value in model.state_dict().items()
    )


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
    candidate_audit: Optional[pd.DataFrame] = None
    distribution_manifest: Optional[pd.DataFrame] = None
    dataset_status: str = "TARGET_VALID_DATASET"


def _identity_metadata_payload(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    for key in ("canonical_candidate_ids", "local_path_ids"):
        value = result.get(key)
        if torch.is_tensor(value):
            result[key] = value.detach().cpu().to(torch.int64)
    return result


def one_step_macro_dataset_fingerprint(dataset: OneStepMacroDataset) -> str:
    """Semantic fingerprint for an immutable committed Macro dataset."""

    identity = _identity_metadata_payload(dataset.pairs.metadata)
    return _json_hash(
        {
            "macro_tensor": _tensor_digest((("macro", dataset.macro.data),)),
            "pair_tensor": _tensor_digest((("pairs", dataset.pairs.data),)),
            "macro_columns": list(dataset.macro.columns),
            "pair_columns": list(dataset.pairs.columns),
            "identity_schema": identity.get("identity_schema"),
            "canonical_candidate_ids": (
                identity.get("canonical_candidate_ids", torch.empty(0, dtype=torch.int64))
                .tolist()
            ),
            "canonical_to_local": identity.get("canonical_to_local", {}),
            "accepted_candidate_ids": dataset.pairing_audit.get(
                "candidate_coverage", {}
            ).get("accepted_candidate_ids", []),
            "dataset_status": dataset.dataset_status,
        }
    )


def save_committed_one_step_macro_dataset(
    path: Path, dataset: OneStepMacroDataset
) -> str:
    """Atomically persist exactly the tensors consumed by the next Macro inner."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = one_step_macro_dataset_fingerprint(dataset)
    payload = {
        "format": "committed_one_step_macro_dataset_v1",
        "fingerprint": fingerprint,
        "macro": {
            "data": dataset.macro.data.detach().cpu(),
            "columns": list(dataset.macro.columns),
            "metadata": _identity_metadata_payload(dataset.macro.metadata),
        },
        "pairs": {
            "data": dataset.pairs.data.detach().cpu(),
            "columns": list(dataset.pairs.columns),
            "metadata": _identity_metadata_payload(dataset.pairs.metadata),
        },
        "pairing_audit": dataset.pairing_audit,
        "used_parent_batch_size": int(dataset.used_parent_batch_size),
        "dataset_status": str(dataset.dataset_status),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return fingerprint


def load_committed_one_step_macro_dataset(
    path: Path,
    *,
    episode: Any,
    macro_batch_size: int,
    n_branches: int,
    device: torch.device,
    build_batches: bool = False,
) -> OneStepMacroDataset:
    """Load a committed dataset without evaluating or resampling candidates."""

    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"committed Macro dataset is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "committed_one_step_macro_dataset_v1":
        raise RuntimeError(f"unexpected committed Macro dataset format: {path}")
    macro_payload = payload["macro"]
    pair_payload = payload["pairs"]
    macro = TensorTable(
        data=macro_payload["data"].to(device),
        columns=list(macro_payload["columns"]),
        metadata=dict(macro_payload.get("metadata", {})),
    )
    pairs = TensorTable(
        data=pair_payload["data"].to(device),
        columns=list(pair_payload["columns"]),
        metadata=dict(pair_payload.get("metadata", {})),
    )
    batches: dict[str, list[dict[str, torch.Tensor]]] = {"fc1": [], "sdf": []}
    if build_batches:
        original_mode = episode.train_mode
        original_add_fc1 = episode.add_FC1loss
        try:
            episode.train_mode = "2time"
            episode.add_FC1loss = True
            target_batches = int(
                getattr(episode.hyperparams, "sdf_true_target_batches", 20)
            )
            min_parents = int(
                getattr(episode.hyperparams, "sdf_min_parent_groups_per_batch", 256)
            )
            batches = {
                "fc1": episode._create_sdf_batches_from_macro_tensor(
                    pairs,
                    batch_size=int(macro_batch_size),
                    n_branches=int(n_branches),
                ),
                "sdf": episode._create_sdf_batches_from_macro_tensor(
                    pairs,
                    batch_size=int(macro_batch_size),
                    n_branches=int(n_branches),
                    target_num_batches=target_batches,
                    min_parent_groups_per_batch=min_parents,
                ),
            }
        finally:
            episode.train_mode = original_mode
            episode.add_FC1loss = original_add_fc1
    candidate_path = path.parent / "candidate_outcomes.csv"
    manifest_path = path.parent / "initial_b_distribution_coverage.csv"
    result = OneStepMacroDataset(
        macro=macro,
        pairs=pairs,
        batches=batches,
        pairing_audit=dict(payload["pairing_audit"]),
        used_parent_batch_size=int(payload["used_parent_batch_size"]),
        candidate_audit=(
            pd.read_csv(candidate_path) if candidate_path.is_file() else None
        ),
        distribution_manifest=(
            pd.read_csv(manifest_path) if manifest_path.is_file() else None
        ),
        dataset_status=str(payload["dataset_status"]),
    )
    observed = one_step_macro_dataset_fingerprint(result)
    if observed != payload.get("fingerprint"):
        raise RuntimeError(
            f"committed Macro dataset fingerprint mismatch: {observed} != "
            f"{payload.get('fingerprint')}"
        )
    return result


@dataclass(frozen=True)
class MacroCandidateSpec:
    candidate_id: int
    lineage_id: int
    split: str
    distribution_parameter_id: int
    distribution_family: str
    alpha: float
    beta: float
    initial_state_seed: int
    transition_seed: int
    entry_seed: int
    shock_key_child_0: int
    shock_key_child_1: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MacroCandidateSchedule:
    split: str
    namespace: str
    target_valid: int
    initial_candidate_budget: int
    max_candidate_budget: int
    minimum_partial_valid: int
    firms_per_parent: int
    seed: int
    candidate_specs: tuple[MacroCandidateSpec, ...]
    locked_candidate_ids: tuple[int, ...] = ()

    def validate(self) -> None:
        if self.target_valid < 1 or self.firms_per_parent < 2:
            raise ValueError("candidate schedule requires positive target and firms")
        if not (
            self.target_valid <= self.initial_candidate_budget <= self.max_candidate_budget
        ):
            raise ValueError("candidate budgets must satisfy target <= initial <= max")
        if len(self.candidate_specs) != self.max_candidate_budget:
            raise ValueError("candidate schedule does not cover max_candidate_budget")
        ids = [spec.candidate_id for spec in self.candidate_specs]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate ids must be unique")
        if any(spec.split != self.split for spec in self.candidate_specs):
            raise ValueError("candidate split mismatch")
        if not set(self.locked_candidate_ids).issubset(ids):
            raise ValueError("locked candidate ids are outside the schedule")

    @property
    def fingerprint(self) -> str:
        return _json_hash(self.to_dict(include_fingerprint=False))

    @property
    def parameter_fingerprint(self) -> str:
        """Hash candidate parameters/shocks while excluding the lock decision."""

        value = self.to_dict(include_fingerprint=False)
        value["locked_candidate_ids"] = []
        return _json_hash(value)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value = {
            "split": self.split,
            "namespace": self.namespace,
            "target_valid": self.target_valid,
            "initial_candidate_budget": self.initial_candidate_budget,
            "max_candidate_budget": self.max_candidate_budget,
            "minimum_partial_valid": self.minimum_partial_valid,
            "firms_per_parent": self.firms_per_parent,
            "seed": self.seed,
            "candidate_specs": [spec.to_dict() for spec in self.candidate_specs],
            "locked_candidate_ids": list(self.locked_candidate_ids),
        }
        if include_fingerprint:
            value["fingerprint"] = self.fingerprint
            value["parameter_fingerprint"] = self.parameter_fingerprint
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MacroCandidateSchedule":
        schedule = cls(
            split=str(value["split"]),
            namespace=str(value["namespace"]),
            target_valid=int(value["target_valid"]),
            initial_candidate_budget=int(value["initial_candidate_budget"]),
            max_candidate_budget=int(value["max_candidate_budget"]),
            minimum_partial_valid=int(value["minimum_partial_valid"]),
            firms_per_parent=int(value["firms_per_parent"]),
            seed=int(value["seed"]),
            candidate_specs=tuple(
                MacroCandidateSpec(**item) for item in value["candidate_specs"]
            ),
            locked_candidate_ids=tuple(map(int, value.get("locked_candidate_ids", ()))),
        )
        schedule.validate()
        recorded = value.get("fingerprint")
        if recorded is not None and str(recorded) != schedule.fingerprint:
            raise RuntimeError("macro candidate schedule fingerprint mismatch")
        recorded_parameters = value.get("parameter_fingerprint")
        if (
            recorded_parameters is not None
            and str(recorded_parameters) != schedule.parameter_fingerprint
        ):
            raise RuntimeError("macro candidate parameter fingerprint mismatch")
        return schedule


def build_macro_candidate_schedule(
    *,
    split: str,
    namespace: str,
    target_valid: int,
    initial_candidate_budget: int,
    max_candidate_budget: int,
    minimum_partial_valid: int,
    firms_per_parent: int,
    seed: int,
    candidate_id_offset: int,
    max_initial_weighted_b_mean: Optional[float] = (
        DEFAULT_MAX_INITIAL_WEIGHTED_B_MEAN
    ),
    distribution_design: str = DEFAULT_MACRO_CANDIDATE_DISTRIBUTION_DESIGN,
) -> MacroCandidateSchedule:
    """Create a fixed path-specific Beta schedule before any model evaluation.

    The Sobol points vary the shape continuously within seven declared support
    strata.  Candidate identity, distribution parameters and shock keys are
    therefore independent of batch size, failure outcomes and resume timing.
    """

    if max_candidate_budget < 1:
        raise ValueError("max_candidate_budget must be positive")
    if distribution_design not in MACRO_CANDIDATE_DISTRIBUTION_DESIGNS:
        raise ValueError(
            f"unknown macro candidate distribution design: {distribution_design}"
        )
    if max_initial_weighted_b_mean is not None and not (
        0.0 < float(max_initial_weighted_b_mean) <= 1.0
    ):
        raise ValueError("max_initial_weighted_b_mean must be in (0, 1]")
    # Draw a deterministic oversupply and accept only laws whose *actual*
    # midpoint-quantile mean passes the cap.  This avoids relying on the
    # analytical Beta mean when the simulator uses a finite quadrature.
    draw_budget = (
        int(max_candidate_budget)
        if max_initial_weighted_b_mean is None
        else max(1024, int(max_candidate_budget) * 16)
    )
    sobol = torch.quasirandom.SobolEngine(
        dimension=2, scramble=True, seed=int(seed)
    ).draw(draw_budget).to(torch.float64)
    # Rotate the balanced family allocation by a seed-dependent offset.  This
    # is stratification; alpha/beta themselves are not repeated templates.
    strata = (
        (
            (
                RESTRICTED_BETA_FULL_OUTER_V1,
                RESTRICTED_BETA_FULL_OUTER_ALPHA_RANGE,
                RESTRICTED_BETA_FULL_OUTER_BETA_RANGE,
            ),
        )
        if distribution_design == RESTRICTED_BETA_FULL_OUTER_V1
        else B_DISTRIBUTION_STRATA
    )
    rotation = int(seed) % len(strata)
    specs: list[MacroCandidateSpec] = []
    for draw_index in range(draw_budget):
        family_index = (draw_index + rotation) % len(strata)
        family, alpha_range, beta_range = strata[family_index]
        alpha = alpha_range[0] + float(sobol[draw_index, 0]) * (
            alpha_range[1] - alpha_range[0]
        )
        beta = beta_range[0] + float(sobol[draw_index, 1]) * (
            beta_range[1] - beta_range[0]
        )
        # The analytical mean is only a cheap rejection prefilter. Every
        # accepted law is checked against the exact finite midpoint-quantile
        # representation used by the simulator below.
        if (
            max_initial_weighted_b_mean is not None
            and alpha / (alpha + beta)
            > float(max_initial_weighted_b_mean) + 0.01
        ):
            continue
        observed_mean = midpoint_beta_mean(alpha, beta, firms_per_parent)
        if (
            max_initial_weighted_b_mean is not None
            and observed_mean > float(max_initial_weighted_b_mean) + 1.0e-12
        ):
            continue
        position = len(specs)
        candidate_id = int(candidate_id_offset + position)
        specs.append(
            MacroCandidateSpec(
                candidate_id=candidate_id,
                lineage_id=candidate_id,
                split=str(split),
                distribution_parameter_id=position,
                distribution_family=family,
                alpha=float(alpha),
                beta=float(beta),
                initial_state_seed=int(seed),
                transition_seed=int(seed + 10_000_019),
                entry_seed=int(seed + 20_000_033),
                shock_key_child_0=int(candidate_id * 2),
                shock_key_child_1=int(candidate_id * 2 + 1),
            )
        )
        if len(specs) >= int(max_candidate_budget):
            break
    if len(specs) != int(max_candidate_budget):
        raise RuntimeError(
            "unable to construct enough candidate distributions under "
            f"max_initial_weighted_b_mean={max_initial_weighted_b_mean}: "
            f"accepted={len(specs)}, required={max_candidate_budget}, "
            f"draw_budget={draw_budget}"
        )
    schedule = MacroCandidateSchedule(
        split=str(split),
        namespace=str(namespace),
        target_valid=int(target_valid),
        initial_candidate_budget=int(initial_candidate_budget),
        max_candidate_budget=int(max_candidate_budget),
        minimum_partial_valid=int(minimum_partial_valid),
        firms_per_parent=int(firms_per_parent),
        seed=int(seed),
        candidate_specs=tuple(specs),
    )
    schedule.validate()
    return schedule


def save_candidate_schedule(path: Path, schedule: MacroCandidateSchedule) -> None:
    schedule.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(schedule.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_candidate_schedule(path: Path) -> MacroCandidateSchedule:
    return MacroCandidateSchedule.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


def lock_validation_candidate_ids(
    schedule: MacroCandidateSchedule, candidate_ids: Sequence[int]
) -> MacroCandidateSchedule:
    """Return the immutable validation identity set used by later outers."""

    ids = tuple(map(int, candidate_ids))
    if len(ids) != len(set(ids)):
        raise ValueError("validation candidate ids must be unique")
    locked = replace(schedule, locked_candidate_ids=ids)
    locked.validate()
    return locked


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
            "aggregation_mode": getattr(
                episode.hyperparams, "aggregation_mode", "legacy_mixed_replay"
            ),
            "continuum_i_points": getattr(
                episode.hyperparams, "continuum_i_points", 9
            ),
            "continuum_z_points": getattr(
                episode.hyperparams, "continuum_z_points", 7
            ),
            "continuum_projection_tolerance": getattr(
                episode.hyperparams, "continuum_projection_tolerance", 1.0e-6
            ),
            "bp_actual_use_trace_enabled": getattr(
                episode.hyperparams, "bp_actual_use_trace_enabled", False
            ),
        }
    )
    market_defaults = {
        "market_clearing_mode": "disabled",
        "market_clearing_coarse_points": 10,
        "market_clearing_refinement_points": 20,
        "market_clearing_max_refinement_rounds": 4,
        "market_clearing_support_lower": -8.0,
        "market_clearing_support_upper": 2.0,
        "market_clearing_support_padding": 2.0,
        "market_clearing_residual_tolerance": 1.0e-5,
        "market_clearing_width_tolerance": 1.0e-5,
        "market_clearing_candidate_chunk_size": 262144,
    }
    kwargs.update({
        name: getattr(episode.hyperparams, name, default)
        for name, default in market_defaults.items()
    })
    source = str(
        getattr(episode.hyperparams, "simulation_bp_action_source", "head")
    ).strip().lower()
    if source not in {"head", "grid"}:
        raise ValueError(f"invalid simulation_bp_action_source={source!r}")
    kwargs.update(
        {
            "entry_rng_seed": int(entry_seed),
            "entry_rng_mode": "stable_economic_identity",
            "common_transition_seed": int(transition_seed),
            "transition_rng_mode": "stable_firm_identity",
            "bp_action_source": source,
        }
    )
    if source == "grid":
        kwargs["bp_grid_policy"] = episode._build_simulation_bp_grid_policy()
    return kwargs


def _allowed_simulation_models(
    models: Mapping[str, torch.nn.Module],
) -> dict[str, torch.nn.Module]:
    return {
        name: model
        for name, model in models.items()
        if name in {"policy_value", "sdf_fc1", "dist_b"}
    }


def _candidate_state_hash(state: Mapping[str, torch.Tensor], path: int) -> str:
    return _tensor_digest(
        (name, value[path])
        for name, value in state.items()
        if torch.is_tensor(value)
        and value.ndim > 0
        and int(value.shape[0]) == int(state["b"].shape[0])
    )


def make_boundary_candidate_initial_state(
    sim: SimulateTS,
    specs: Sequence[MacroCandidateSpec],
    *,
    firms: int,
    initial_total_k_min: Optional[float] = None,
    initial_total_k_max: Optional[float] = None,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame, dict[str, Any]]:
    """Build path-specific B-only initial distributions with keyed CRN.

    All non-b fields are generated once from the same keyed baseline state and
    then left byte-identical while b is replaced by the scheduled Beta law.
    Midpoint quantiles avoid endpoint singularities for alpha/beta below one.
    """

    if len(specs) != int(sim.n_paths):
        raise ValueError("candidate spec count must match simulator n_paths")
    if firms < 2:
        raise ValueError("at least two weighted components are required")
    device = sim.device
    dtype = torch.float32
    B, N = len(specs), int(firms)
    candidate_ids = torch.tensor(
        [spec.candidate_id for spec in specs], device=device, dtype=torch.long
    )
    path = candidate_ids[:, None].expand(B, N)
    identity = torch.arange(N, device=device, dtype=torch.long)[None, :].expand(B, N)
    seed = int(specs[0].initial_state_seed)
    if any(int(spec.initial_state_seed) != seed for spec in specs):
        raise ValueError("a candidate batch must share its initial-state seed")

    stationary_x_std = float(sim.config.SIGMA_X) / math.sqrt(
        max(1.0 - float(sim.config.RHO_X) ** 2, 1.0e-30)
    )
    x = float(sim.config.XBAR) + stationary_x_std * _keyed_normal(
        seed=seed,
        path_id=candidate_ids,
        economic_time=0,
        branch_id=-1,
        identity=torch.zeros_like(candidate_ids),
        shock_type=101,
        dtype=dtype,
    )
    if (initial_total_k_min is None) != (initial_total_k_max is None):
        raise ValueError("initial total K bounds must be provided together")
    if initial_total_k_min is not None:
        if not 0.0 < float(initial_total_k_min) <= float(initial_total_k_max):
            raise ValueError("initial total K bounds must be positive and ordered")
        log_k_min = math.log(float(initial_total_k_min))
        log_k_max = math.log(float(initial_total_k_max))
        lnkf = log_k_min + (log_k_max - log_k_min) * _keyed_uniform(
            seed=seed,
            path_id=candidate_ids,
            economic_time=0,
            branch_id=-1,
            identity=torch.zeros_like(candidate_ids),
            shock_type=102,
            dtype=dtype,
        )
    else:
        lnkf = 4.0 + _keyed_normal(
            seed=seed,
            path_id=candidate_ids,
            economic_time=0,
            branch_id=-1,
            identity=torch.zeros_like(candidate_ids),
            shock_type=102,
            dtype=dtype,
        )
    hatcf = -2.0 + _keyed_normal(
        seed=seed,
        path_id=candidate_ids,
        economic_time=0,
        branch_id=-1,
        identity=torch.zeros_like(candidate_ids),
        shock_type=103,
        dtype=dtype,
    )
    weights = _keyed_uniform(
        seed=seed,
        path_id=path,
        economic_time=0,
        branch_id=-1,
        identity=identity,
        shock_type=104,
        dtype=dtype,
    ).clamp_min(torch.finfo(dtype).eps)
    weights = weights / weights.sum(dim=1, keepdim=True)
    capital = torch.exp(lnkf)[:, None] * weights
    baseline_b = float(sim.config.SIM_B_INIT_MIN) + (
        float(sim.config.SIM_B_INIT_MAX) - float(sim.config.SIM_B_INIT_MIN)
    ) * _keyed_uniform(
        seed=seed,
        path_id=path,
        economic_time=0,
        branch_id=-1,
        identity=identity,
        shock_type=105,
        dtype=dtype,
    )

    quantiles = (torch.arange(N, dtype=torch.float64) + 0.5) / float(N)
    order_key = _keyed_uniform(
        seed=seed,
        path_id=path,
        economic_time=0,
        branch_id=-1,
        identity=identity,
        shock_type=106,
        dtype=torch.float64,
    )
    permutation = torch.argsort(order_key, dim=1)
    q = quantiles.to(device=device)[None, :].expand(B, N)
    q = torch.gather(q, 1, permutation).cpu().numpy()
    alpha = np.asarray([spec.alpha for spec in specs], dtype=np.float64)[:, None]
    beta = np.asarray([spec.beta for spec in specs], dtype=np.float64)[:, None]
    unit_b = betaincinv(alpha, beta, q)
    if not np.isfinite(unit_b).all():
        raise RuntimeError("Beta inverse quantile mapping produced nonfinite b")
    b = torch.as_tensor(unit_b, device=device, dtype=dtype)
    b = float(sim.config.SIM_B_INIT_MIN) + (
        float(sim.config.SIM_B_INIT_MAX) - float(sim.config.SIM_B_INIT_MIN)
    ) * b

    duplicate = lambda value: torch.repeat_interleave(value, 2, dim=1)
    eta_probability = torch.tensor(
        [1.0 - float(sim.config.ZETA), float(sim.config.ZETA)],
        device=device,
        dtype=dtype,
    )[None, None, :]
    mass = (
        torch.ones((B, N, 1), device=device, dtype=dtype) * eta_probability
    ).reshape(B, 2 * N)
    mask = mass > 0.0
    component_id = torch.arange(2 * N, device=device, dtype=torch.long)[None, :].expand(B, -1)
    eta = torch.tensor([0.0, 1.0], device=device, dtype=dtype)[None, None, :]
    eta = eta.expand(B, N, 2).reshape(B, 2 * N)
    stationary_z_std = float(sim.config.SIGMA_Z) / math.sqrt(
        max(1.0 - float(sim.config.RHO_Z) ** 2, 1.0e-30)
    )
    zeros_long = torch.zeros((B, 2 * N), device=device, dtype=torch.long)
    distribution = PreDecisionDistribution(
        component_mass=mass,
        b=duplicate(b),
        capital_per_unit=duplicate(capital),
        z_conditional_mean=torch.full_like(mass, float(sim.config.ZBAR)),
        z_conditional_std=torch.full_like(mass, stationary_z_std),
        component_mask=mask,
        economic_node_id=zeros_long,
        component_id=component_id,
        origin_kind=zeros_long,
        origin_source_id=component_id,
        origin_time=zeros_long,
        evaluation_time=zeros_long,
        cohort=zeros_long,
        birth_time=torch.full_like(zeros_long, -1),
        maintenance_eligibility=torch.ones_like(mask),
        eta_state=eta,
    ).validate()
    state = distribution.as_state_tensors()
    state.update(
        {
            "x": x,
            "hatcf": hatcf,
            "lnkf": lnkf,
            "M": torch.ones(B, device=device, dtype=dtype),
            "economic_node_id": torch.zeros(B, device=device, dtype=torch.long),
            "economic_time": torch.zeros(B, device=device, dtype=torch.long),
            "transition_path_id": candidate_ids,
            "market_previous_root": hatcf.clone(),
            "i": torch.zeros_like(mass),
            "firm_id": component_id,
        }
    )

    rows: list[dict[str, Any]] = []
    for local_path, spec in enumerate(specs):
        values = b[local_path].to(torch.float64)
        mean = values.mean()
        sorted_values = torch.sort(values).values
        digest = hashlib.sha256(values.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        rows.append(
            {
                **spec.to_dict(),
                "weighted_b_mean": float(mean),
                "weighted_b_std": float(values.std(unbiased=False)),
                "weighted_b_p10": float(torch.quantile(sorted_values, 0.10)),
                "weighted_b_p50": float(torch.quantile(sorted_values, 0.50)),
                "weighted_b_p90": float(torch.quantile(sorted_values, 0.90)),
                "high_b_mass": float((values >= 0.75).to(torch.float64).mean()),
                "K_total": float(capital[local_path].sum()),
                "z_conditional_mean": float(sim.config.ZBAR),
                "z_conditional_std": stationary_z_std,
                "distribution_hash": digest,
            }
        )
    preservation = {
        "z_conditional_law_unchanged": True,
        "eta_law_unchanged": True,
        "investment_cost_law_unchanged": True,
        "initial_K_unchanged": initial_total_k_min is None,
        "initial_macro_state_distribution_restricted": (
            initial_total_k_min is not None
        ),
        "initial_total_K_bounds": (
            None
            if initial_total_k_min is None
            else [float(initial_total_k_min), float(initial_total_k_max)]
        ),
        "initial_x_unchanged": True,
        "firm_mass_unchanged": True,
        "baseline_b_hash": hashlib.sha256(
            baseline_b.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest(),
        "only_b_replaced": True,
    }
    return state, pd.DataFrame(rows), preservation


def boundary_distribution_from_state(
    state: Mapping[str, torch.Tensor],
) -> PreDecisionDistribution:
    return PreDecisionDistribution(
        component_mass=state["component_mass"],
        b=state["b"],
        capital_per_unit=state["K"],
        z_conditional_mean=state["z_conditional_mean"],
        z_conditional_std=state["z_conditional_std"],
        component_mask=state["component_mask"],
        economic_node_id=state["economic_node_id_component"],
        component_id=state["component_id"],
        origin_kind=state["origin_kind"],
        origin_source_id=state["origin_source_id"],
        origin_time=state["origin_time"],
        evaluation_time=state["evaluation_time"],
        cohort=state["cohort"],
        birth_time=state["birth_time"],
        maintenance_eligibility=state["maintenance_eligibility"],
        eta_state=state["eta"],
    ).validate()


def boundary_entry_candidates(
    sim: SimulateTS, distribution: PreDecisionDistribution
) -> EntryCandidates:
    total_mass = distribution.total_mass()
    total_capital = distribution.total_capital()
    mean_capital = total_capital / total_mass.clamp_min(1.0e-30)
    capital_per_entrant = float(sim.entry_size_ratio) * mean_capital
    potential_capital = float(sim.entry_capital_ratio) * total_capital
    active = (capital_per_entrant > 0.0) & (potential_capital > 0.0)
    B = distribution.batch_size
    dtype = distribution.component_mass.dtype
    device = distribution.component_mass.device
    return EntryCandidates(
        counts=active.to(torch.long),
        mask=active[:, None],
        z=torch.zeros((B, 1), device=device, dtype=dtype),
        entry_cost=torch.zeros((B, 1), device=device, dtype=dtype),
        K_birth=capital_per_entrant[:, None],
        potential_capital_nominal=potential_capital,
        candidate_capital_realized=capital_per_entrant,
    )


def boundary_child_state(
    sim: SimulateTS,
    parent: Mapping[str, torch.Tensor],
    *,
    branch: int,
) -> dict[str, torch.Tensor]:
    record = parent["_boundary_decision_record"]
    distribution = propagate_ar1_with_eta_mixture(
        record.transport_atoms,
        rho_z=float(sim.config.RHO_Z),
        sigma_z=float(sim.config.SIGMA_Z),
        z_bar=float(sim.config.ZBAR),
        zeta=float(sim.config.ZETA),
    )
    path_ids = parent["transition_path_id"].to(torch.long)
    eps = _keyed_normal(
        seed=int(sim.common_transition_seed),
        path_id=path_ids,
        economic_time=torch.ones_like(path_ids),
        branch_id=int(branch),
        identity=torch.zeros_like(path_ids),
        shock_type=1,
        dtype=parent["x"].dtype,
    )
    x_next = (
        (1.0 - float(sim.config.RHO_X)) * float(sim.config.XBAR)
        + float(sim.config.RHO_X) * parent["x"]
        + float(sim.config.SIGMA_X) * eps
    )
    hatcf, lnkf, M = _predict_macro_fc1_batched(
        sim,
        parent["x"],
        x_next,
        parent["hatc_cal"],
        parent["lnk_cal"],
    )
    child = distribution.as_state_tensors()
    B, L = distribution.component_mass.shape
    node_id = torch.full(
        (B,), branch + 1, device=sim.device, dtype=torch.long
    )
    child["economic_node_id_component"] = node_id[:, None].expand(B, L)
    child.update(
        {
            "x": x_next,
            "hatcf": hatcf,
            "lnkf": lnkf,
            "M": M,
            "economic_node_id": node_id,
            "economic_time": torch.ones(B, device=sim.device, dtype=torch.long),
            "transition_path_id": path_ids.clone(),
            "market_previous_root": parent["hatc_cleared"].clone(),
            "i": torch.zeros((B, L), device=sim.device, dtype=distribution.component_mass.dtype),
            "firm_id": distribution.component_id,
        }
    )
    return child


def boundary_macro_rows(
    state: Mapping[str, torch.Tensor], *, t: int, branch: int, split: str
) -> list[dict[str, Any]]:
    distribution = boundary_distribution_from_state(state)
    moments = gaussian_mixture_moments(distribution)
    path_ids = state["transition_path_id"].to(torch.long)
    rows: list[dict[str, Any]] = []
    for local in range(distribution.batch_size):
        C = state["accounting_C_raw"][local]
        K = state["accounting_K_total"][local]
        h_implied = (
            float(torch.log(C / K))
            if bool((C > 0.0) & (K > 0.0))
            else float("nan")
        )
        h_belief = float(state.get("hatc_belief", state["hatcf"])[local])
        h_policy = state.get("h_policy_input", state["hatcf"])
        h_calculated = state.get(
            "h_cal_from_ledger", state.get("hatc_cal", state["hatc_cleared"])
        )
        h_cleared = float(h_calculated[local])
        h_firm_input = float(h_policy[local])
        lnk_belief = float(state["lnkf"][local])
        lnk_current = float(state["lnk_cal"][local])
        candidate_id = int(path_ids[local])
        rows.append(
            {
                "split": str(split),
                "path": candidate_id,
                "lineage_id": candidate_id,
                "candidate_id": candidate_id,
                "t": int(t),
                "branch": int(branch),
                "x": float(state["x"][local]),
                "hatcf": h_firm_input,
                # The FC1 pair builder consumes ``hatcf``/``lnkf``.  On a
                # cleared node these columns must be the true current cleared
                # state, while beliefs remain separate audit-only columns.
                "lnkf": lnk_current,
                "Hatc": h_cleared,
                "LnK": lnk_current,
                "M": float(state.get("M", torch.ones_like(state["x"]))[local]),
                "C": float(C),
                "K": float(K),
                "I_oper": float(state["accounting_I_oper"][local]),
                "I_entry": float(state["accounting_I_entry"][local]),
                "I_maintenance": float(
                    state["accounting_I_maintenance"][local]
                ),
                "n_firms": float(distribution.total_mass()[local]),
                "default_mass": float(
                    state.get(
                        "decision_default_mass", torch.zeros_like(state["x"])
                    )[local]
                ),
                "investment_mass": float(
                    state.get(
                        "decision_investment_mass", torch.zeros_like(state["x"])
                    )[local]
                ),
                "mean_b": float(
                    (
                        distribution.component_mass[local]
                        * distribution.b[local]
                    ).sum()
                    / distribution.component_mass[local].sum().clamp_min(1.0e-30)
                ),
                "h_belief": h_belief,
                "lnk_belief": lnk_belief,
                "h_selected_root": h_cleared,
                "h_implied_from_accounting": h_implied,
                "h_used_in_firm_policy": h_firm_input,
                "root_status": (
                    FAST_STATUS_LABELS[int(state["fast_solver_status"][local])]
                    if "fast_solver_status" in state
                    else "ROOT_FOUND"
                ),
                "h_fc1_original": float(
                    state.get("h_fc1_original", state.get("hatc_belief", state["hatcf"]))[local]
                ),
                "h_policy_input": h_firm_input,
                "h_cal_from_ledger": h_cleared,
                "h_policy_minus_h_cal": h_firm_input - h_cleared,
                "baseline_G": float(
                    state.get("baseline_G", torch.full_like(state["x"], float("nan")))[local]
                ),
                "corrected_G": float(
                    state.get("corrected_G", torch.full_like(state["x"], float("nan")))[local]
                ),
                "G_evaluation_count": int(
                    state.get("G_evaluation_count", torch.zeros_like(path_ids))[local]
                ),
                "root_residual": float(state["market_root_residual"][local]),
                "root_residual_normalized": float(
                    state["market_root_residual_normalized"][local]
                ),
                "firm_mass": float(distribution.total_mass()[local]),
                "total_capital": float(distribution.total_capital()[local]),
                "aggregate_debt": float(distribution.aggregate_debt()[local]),
                "z_mixture_mean": float(moments["mean"][local]),
                "z_mixture_variance": float(moments["variance"][local]),
                "component_count": int(distribution.component_mask[local].sum()),
                "representation": BOUNDARY_AWARE_CONTINUUM_V1,
            }
        )
    return rows


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


def build_legacy_harvested_one_step_macro_dataset(
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
    """Archived short-rollout interface; canonical training never calls it."""
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


class _CandidateEconomicFailure(RuntimeError):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = str(status)
        self.reason = str(reason)


def _candidate_failure_status(stage: str, reason: str) -> str:
    upper = reason.upper()
    if "NO ACTIVE ECONOMIC NODE" in upper or "EXTINCT" in upper:
        return "CHILD_EXTINCT" if stage.startswith("child") else "PARENT_CLEARING_FAILED"
    if "NONFINITE" in upper or "NAN" in upper or "INF" in upper:
        return "NUMERICAL_FAILURE"
    if stage == "parent":
        return "PARENT_CLEARING_FAILED"
    if stage == "child_0":
        return "CHILD_0_CLEARING_FAILED"
    if stage == "child_1":
        return "CHILD_1_CLEARING_FAILED"
    return "NUMERICAL_FAILURE"


def _evaluate_boundary_candidate_batch(
    *,
    models: Mapping[str, torch.nn.Module],
    episode: Any,
    specs: Sequence[MacroCandidateSpec],
    firms_per_parent: int,
    device: torch.device,
    integration: BoundaryIntegrationConfig,
    root_config: MarketClearingConfig,
    initial_total_k_min: Optional[float] = None,
    initial_total_k_max: Optional[float] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate one vectorized candidate batch or raise a typed economic failure."""

    if not specs:
        return [], [], []
    sim = SimulateTS(
        models=_allowed_simulation_models(models),
        config=episode.config,
        n_paths=len(specs),
        group_size=int(firms_per_parent),
        horizon=1,
        branch_num=2,
        device=device,
        **_simulation_kwargs(
            episode, int(specs[0].transition_seed), int(specs[0].entry_seed)
        ),
    )
    firms = int(firms_per_parent)
    if firms < 2:
        raise RuntimeError("candidate evaluator missing firms_per_parent contract")
    sim.aggregation_mode = BOUNDARY_AWARE_CONTINUUM_V1
    controller = BoundaryAwareMarketClearingController(root_config, integration)
    sim.market_clearing_controller = controller
    try:
        state, manifest, preservation = make_boundary_candidate_initial_state(
            sim,
            specs,
            firms=firms,
            initial_total_k_min=initial_total_k_min,
            initial_total_k_max=initial_total_k_max,
        )
    except (ValueError, FloatingPointError) as error:
        raise _CandidateEconomicFailure(
            "INVALID_INITIAL_DISTRIBUTION", repr(error)
        ) from error

    initial_state_hashes = [
        _candidate_state_hash(state, local) for local in range(len(specs))
    ]
    stage = "parent"
    try:
        state = controller.clear_state(
            sim, state, empty_entry_candidates(state), t=0, branch=-1
        )
        controller.validate_committed(state, t=0, branch=-1)
        parent_record = state["_boundary_decision_record"]
        parent_transport = parent_record.transport_atoms
        parent_transport_fingerprint = parent_transport.fingerprint()
        torch.testing.assert_close(
            parent_transport.total_mass(),
            state["decision_transport_mass"].to(torch.float64),
            atol=5.0e-6,
            rtol=5.0e-6,
        )
        if bool(
            (
                parent_transport.source_time[parent_transport.atom_mask]
                != 0
            ).any()
        ) or bool(
            (
                parent_transport.destination_time[parent_transport.atom_mask]
                != 1
            ).any()
        ):
            raise RuntimeError("parent transport timing parity failed")
        parent_rows = boundary_macro_rows(state, t=0, branch=-1, split=specs[0].split)
        entry = boundary_entry_candidates(sim, boundary_distribution_from_state(state))
        children: list[dict[str, torch.Tensor]] = []
        child_rows: list[dict[str, Any]] = []
        for branch in range(2):
            stage = f"child_{branch}"
            child = boundary_child_state(sim, state, branch=branch)
            if parent_transport.fingerprint() != parent_transport_fingerprint:
                raise RuntimeError("child transition mutated parent transport storage")
            child_mask = child["component_mask"].bool()
            child_mass = torch.where(
                child_mask,
                child["component_mass"],
                torch.zeros_like(child["component_mass"]),
            )
            torch.testing.assert_close(
                child_mass.sum(dim=1).to(torch.float64),
                parent_transport.total_mass(),
                atol=5.0e-6,
                rtol=5.0e-6,
            )
            torch.testing.assert_close(
                (child_mass * child["K"]).sum(dim=1).to(torch.float64),
                parent_transport.total_capital(),
                atol=5.0e-6,
                rtol=5.0e-6,
            )
            torch.testing.assert_close(
                (child_mass * child["K"] * child["b"]).sum(dim=1).to(torch.float64),
                parent_transport.aggregate_debt(),
                atol=5.0e-6,
                rtol=5.0e-6,
            )
            controller.record_gpu_memory("child_transition", device)
            child = controller.clear_state(
                sim, child, entry, t=1, branch=branch
            )
            controller.validate_committed(child, t=1, branch=branch)
            children.append(child)
            child_rows.extend(
                boundary_macro_rows(
                    child, t=1, branch=branch, split=specs[0].split
                )
            )
    except MarketClearingError as error:
        raise _CandidateEconomicFailure(
            _candidate_failure_status(stage, str(error)), str(error)
        ) from error

    rows_by_id: dict[int, list[dict[str, Any]]] = {
        spec.candidate_id: [] for spec in specs
    }
    for row in (*parent_rows, *child_rows):
        rows_by_id[int(row["candidate_id"])].append(row)
    def clearing_validity(
        cleared: Mapping[str, torch.Tensor], local: int
    ) -> tuple[bool, str]:
        defined = bool(
            cleared.get(
                "economic_state_defined",
                torch.ones(len(specs), device=device, dtype=torch.bool),
            )[local]
        )
        if not defined:
            return False, "UNDEFINED_ECONOMIC_STATE"
        if bool(cleared["market_root_found"][local]):
            return True, "STRICT_ROOT"
        if root_config.solver_mode != "fc1_bounded_improvement_pilot":
            return False, "STRICT_ROOT_NOT_FOUND"
        normalized = abs(float(cleared["corrected_normalized_residual"][local]))
        log_mismatch = abs(float(cleared["h_policy_minus_h_cal"][local]))
        improvement = float(cleared["improvement_fraction"][local])
        accepted = bool(
            normalized <= 1.0e-3
            and log_mismatch <= 0.005
            and improvement >= 0.90
        )
        return accepted, (
            "APPROXIMATE_PILOT_ACCEPTED"
            if accepted
            else "APPROXIMATE_PILOT_REJECTED"
        )

    if parent_transport.fingerprint() != parent_transport_fingerprint:
        raise RuntimeError("selected transport fingerprint changed after child clearing")
    parent_bp_trace = [
        item
        for item in sim.bp_actual_use_trace
        if item.get("decision_stage") == "boundary_transport_parent_policy"
    ]
    bp_trace_required = bool(sim.bp_actual_use_trace_enabled)
    bp_action_parity = (not bp_trace_required) or (
        bool(parent_bp_trace)
        and all(
            item.get("executed_bp_source") == "grid"
            for item in parent_bp_trace
        )
    )
    if not bp_action_parity:
        raise RuntimeError("parent BP grid action parity trace failed")
    controller_summary = controller.summary()
    memory_stages = controller_summary.get("gpu_memory_stages", {})
    records: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    manifest_rows = manifest.to_dict("records")
    for local, spec in enumerate(specs):
        parent_valid, parent_clearing_status = clearing_validity(state, local)
        child_validity = []
        child_clearing_status = []
        for child in children:
            valid, clearing_status = clearing_validity(child, local)
            child_validity.append(
                valid
                and bool(child["accounting_C_raw"][local] > 0.0)
                and bool(child["accounting_K_total"][local] > 0.0)
            )
            child_clearing_status.append(clearing_status)
        if not parent_valid:
            status = "PARENT_CLEARING_FAILED"
        elif not child_validity[0] or not child_validity[1]:
            status = "CHILD_EXTINCT"
        else:
            status = MACRO_CANDIDATE_STATUS_VALID
            valid_rows.extend(rows_by_id[spec.candidate_id])
        record = {
            **spec.to_dict(),
            "initial_distribution_hash": initial_state_hashes[local],
            "parent_status": "VALID" if parent_valid else "INVALID",
            "child_0_status": "VALID" if child_validity[0] else "INVALID",
            "child_1_status": "VALID" if child_validity[1] else "INVALID",
            "parent_clearing_status": parent_clearing_status,
            "child_0_clearing_status": child_clearing_status[0],
            "child_1_clearing_status": child_clearing_status[1],
            "status": status,
            "failure_reason": "" if status == MACRO_CANDIDATE_STATUS_VALID else status,
            "accepted_for_training": False,
            "boundary_aware_representation": BOUNDARY_AWARE_CONTINUUM_V1,
            "root_residual_parent": float(state["market_root_residual_normalized"][local]),
            "root_residual_child_0": float(children[0]["market_root_residual_normalized"][local]),
            "root_residual_child_1": float(children[1]["market_root_residual_normalized"][local]),
            "only_b_replaced": bool(preservation["only_b_replaced"]),
            "gpu_memory_stages": memory_stages,
            "candidate_committed_ledger_parity": True,
            "transport_fingerprint_parity": True,
            "capital_debt_mass_accounting": True,
            "parent_child_timing": True,
            "bp_action_parity": bp_action_parity,
        }
        records.append(record)
        manifest_rows[local]["candidate_status"] = status
    return valid_rows, records, manifest_rows


def _evaluate_boundary_candidates_with_isolation(
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    specs = tuple(kwargs.pop("specs"))
    try:
        return _evaluate_boundary_candidate_batch(specs=specs, **kwargs)
    except _CandidateEconomicFailure as error:
        if len(specs) > 1:
            middle = len(specs) // 2
            left = _evaluate_boundary_candidates_with_isolation(
                specs=specs[:middle], **kwargs
            )
            right = _evaluate_boundary_candidates_with_isolation(
                specs=specs[middle:], **kwargs
            )
            return tuple(left[i] + right[i] for i in range(3))  # type: ignore[return-value]
        spec = specs[0]
        return [], [
            {
                **spec.to_dict(),
                "initial_distribution_hash": None,
                "parent_status": "INVALID" if error.status == "PARENT_CLEARING_FAILED" else "UNKNOWN",
                "child_0_status": "INVALID" if error.status in {"CHILD_0_CLEARING_FAILED", "CHILD_EXTINCT"} else "UNKNOWN",
                "child_1_status": "INVALID" if error.status in {"CHILD_1_CLEARING_FAILED", "CHILD_EXTINCT"} else "UNKNOWN",
                "status": error.status,
                "failure_reason": error.reason,
                "accepted_for_training": False,
                "boundary_aware_representation": BOUNDARY_AWARE_CONTINUUM_V1,
            }
        ], [{**spec.to_dict(), "candidate_status": error.status}]


def _macro_rows_to_table(rows: Sequence[Mapping[str, Any]], device: torch.device) -> TensorTable:
    columns = [
        "path", "t", "branch", "x", "hatcf", "lnkf", "Hatc", "LnK",
        "M", "C", "K", "I_oper", "I_entry", "I_maintenance", "n_firms",
        "default_mass", "investment_mass", "mean_b",
        "lineage_id",
    ]
    canonical_ids = list(dict.fromkeys(int(row["candidate_id"]) for row in rows))
    canonical_to_local = {
        candidate_id: local_id
        for local_id, candidate_id in enumerate(canonical_ids)
    }
    if len(canonical_to_local) > 16_777_216:
        raise RuntimeError("dataset-local candidate ids exceed exact float32 range")
    local_rows = []
    for row in rows:
        canonical_id = int(row["candidate_id"])
        local_id = canonical_to_local[canonical_id]
        local = dict(row)
        local["path"] = local_id
        local["lineage_id"] = local_id
        local_rows.append(local)
    data = torch.tensor(
        [[float(row[name]) for name in columns] for row in local_rows],
        device=device,
        dtype=torch.float32,
    )
    return TensorTable(
        data=data,
        columns=columns,
        metadata={
            "identity_schema": "canonical_candidate_to_local_path_v1",
            "canonical_candidate_ids": torch.tensor(
                canonical_ids, dtype=torch.int64
            ),
            "local_path_ids": torch.arange(len(canonical_ids), dtype=torch.int64),
            "canonical_to_local": {
                str(candidate_id): int(local_id)
                for candidate_id, local_id in canonical_to_local.items()
            },
            "local_to_canonical": {
                str(local_id): int(candidate_id)
                for candidate_id, local_id in canonical_to_local.items()
            },
        },
    )


def audit_candidate_pairing(
    pairs: TensorTable, accepted_candidate_ids: Sequence[int]
) -> dict[str, Any]:
    col = {name: index for index, name in enumerate(pairs.columns)}
    required = {"path", "t", "branch", "lineage_id"}
    if not required.issubset(col):
        raise RuntimeError("candidate pair table lacks identity fields")
    data = pairs.data.detach().cpu()
    expected = set(map(int, accepted_candidate_ids))
    identity = dict(pairs.metadata)
    if identity.get("identity_schema") != "canonical_candidate_to_local_path_v1":
        raise RuntimeError("candidate pair table lacks lossless canonical/local identity metadata")
    canonical_to_local = {
        int(candidate_id): int(local_id)
        for candidate_id, local_id in identity.get("canonical_to_local", {}).items()
    }
    local_to_canonical = {
        int(local_id): int(candidate_id)
        for local_id, candidate_id in identity.get("local_to_canonical", {}).items()
    }
    if set(canonical_to_local) != expected:
        raise RuntimeError(
            "candidate identity mapping does not match accepted canonical ids: "
            f"expected={sorted(expected)[:10]}, mapped={sorted(canonical_to_local)[:10]}"
        )
    if any(local_to_canonical.get(local) != canonical for canonical, local in canonical_to_local.items()):
        raise RuntimeError("candidate canonical/local identity mapping is not bijective")
    failures: list[dict[str, Any]] = []
    for candidate_id in sorted(expected):
        local_id = canonical_to_local[candidate_id]
        mask = data[:, col["path"]].to(torch.long) == local_id
        branches = sorted(map(int, data[mask, col["branch"]].tolist()))
        times = sorted(set(map(int, data[mask, col["t"]].tolist())))
        lineages = set(map(int, data[mask, col["lineage_id"]].tolist()))
        if branches != [0, 1] or times != [1] or lineages != {local_id}:
            failures.append(
                {
                    "candidate_id": candidate_id,
                    "local_path_id": local_id,
                    "branches": branches,
                    "times": times,
                    "lineages": sorted(lineages),
                }
            )
    observed_local = set(map(int, data[:, col["path"]].to(torch.long).tolist()))
    observed = {local_to_canonical[value] for value in observed_local}
    audit = {
        "expected_parent_count": len(expected),
        "observed_parent_count": len(observed),
        "pair_row_count": len(data),
        "expected_pair_row_count": 2 * len(expected),
        "failed_parent_groups": failures,
        "unexpected_candidate_ids": sorted(observed - expected),
        "identity_schema": identity["identity_schema"],
        "canonical_to_local": {
            str(key): value for key, value in sorted(canonical_to_local.items())
        },
        "local_to_canonical": {
            str(key): value for key, value in sorted(local_to_canonical.items())
        },
        "each_parent_has_exact_children": (
            not failures and observed == expected and len(data) == 2 * len(expected)
        ),
        "parent_local_t": 0,
        "child_local_t": 1,
    }
    if not audit["each_parent_has_exact_children"]:
        raise RuntimeError(f"candidate parent/child pairing invariant failed: {audit}")
    return audit


def build_one_step_macro_dataset(
    *,
    models: Mapping[str, torch.nn.Module],
    episode: Any,
    candidate_schedule: MacroCandidateSchedule,
    parent_batch_size: int,
    macro_batch_size: int,
    device: torch.device,
    output_dir: Optional[Path] = None,
    candidate_ids: Optional[Sequence[int]] = None,
    allow_replenishment: bool = True,
    require_cached_shards: bool = False,
    integration: Optional[BoundaryIntegrationConfig] = None,
    root_config: Optional[MarketClearingConfig] = None,
    source_checkpoint_provenance: Optional[Mapping[str, Any]] = None,
    initial_total_k_min: Optional[float] = None,
    initial_total_k_max: Optional[float] = None,
) -> OneStepMacroDataset:
    """Canonical boundary-aware one-step Macro dataset builder.

    Candidate failures are audited and isolated.  Only complete parent/two-
    child pairs enter optimization batches; engineering/schema errors are not
    caught here and therefore remain fail-fast.
    """

    candidate_schedule.validate()
    if int(parent_batch_size) < 1 or int(macro_batch_size) < 1:
        raise ValueError("batch sizes must be positive")
    integration = integration or BoundaryIntegrationConfig()
    root_config = root_config or MarketClearingConfig()
    spec_by_id = {spec.candidate_id: spec for spec in candidate_schedule.candidate_specs}
    if candidate_ids is None:
        ordered_specs = list(candidate_schedule.candidate_specs)
    else:
        requested = tuple(map(int, candidate_ids))
        missing = sorted(set(requested) - set(spec_by_id))
        if missing:
            raise ValueError(f"candidate ids outside schedule: {missing[:10]}")
        ordered_specs = [spec_by_id[value] for value in requested]
        if allow_replenishment:
            used = set(requested)
            ordered_specs.extend(
                spec for spec in candidate_schedule.candidate_specs
                if spec.candidate_id not in used
            )

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    accepted_ids: list[int] = []
    attempted_ids: set[int] = set()
    output = None if output_dir is None else Path(output_dir)
    provenance = {
        "candidate_parameter_fingerprint": candidate_schedule.parameter_fingerprint,
        "split": candidate_schedule.split,
        "firms_per_parent": candidate_schedule.firms_per_parent,
        "integration_config": asdict(integration),
        "root_config": asdict(root_config),
        "initial_total_K_bounds": (
            None
            if initial_total_k_min is None
            else [float(initial_total_k_min), float(initial_total_k_max)]
        ),
        "model_hashes_at_generation": {
            name: _model_state_digest(model)
            for name, model in sorted(models.items())
        },
        "source_checkpoint_provenance": dict(
            source_checkpoint_provenance or {
                "policy": "CURRENT_FIRM_MODEL_STATE",
                "fc1_sdf": "CURRENT_MACRO_MODEL_STATE",
            }
        ),
    }
    provenance["fingerprint"] = _json_hash(provenance)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        save_candidate_schedule(output / "candidate_schedule.json", candidate_schedule)
        provenance_path = output / "dataset_provenance.json"
        if provenance_path.is_file():
            recorded = json.loads(provenance_path.read_text(encoding="utf-8"))
            for key in (
                "candidate_parameter_fingerprint",
                "split",
                "firms_per_parent",
                "integration_config",
                "root_config",
                "model_hashes_at_generation",
                "source_checkpoint_provenance",
                "initial_total_K_bounds",
            ):
                if _json_hash(recorded.get(key)) != _json_hash(provenance.get(key)):
                    raise RuntimeError(
                        f"candidate dataset resume provenance mismatch: {key}"
                    )
            provenance = recorded
        else:
            provenance_path.write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    for start in range(0, len(ordered_specs), int(parent_batch_size)):
        if len(accepted_ids) >= candidate_schedule.target_valid:
            break
        batch_specs = ordered_specs[start : start + int(parent_batch_size)]
        shard_path = (
            None
            if output is None
            else output / "candidate_shards" / f"batch_{start:06d}_{start + len(batch_specs):06d}.pt"
        )
        if shard_path is not None and shard_path.is_file():
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            if (
                payload["candidate_parameter_fingerprint"]
                != candidate_schedule.parameter_fingerprint
            ):
                raise RuntimeError("candidate shard schedule mismatch")
            if payload["dataset_provenance_fingerprint"] != provenance["fingerprint"]:
                raise RuntimeError("candidate shard provenance mismatch")
            batch_rows = list(payload["rows"])
            batch_records = list(payload["records"])
            batch_manifest = list(payload["manifest"])
        else:
            if require_cached_shards and shard_path is not None:
                raise RuntimeError(
                    f"resume requires committed candidate shard: {shard_path}"
                )
            batch_rows, batch_records, batch_manifest = (
                _evaluate_boundary_candidates_with_isolation(
                    models=models,
                    episode=episode,
                    specs=batch_specs,
                    firms_per_parent=candidate_schedule.firms_per_parent,
                    device=device,
                    integration=integration,
                    root_config=root_config,
                    initial_total_k_min=initial_total_k_min,
                    initial_total_k_max=initial_total_k_max,
                )
            )
            if shard_path is not None:
                shard_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = shard_path.with_suffix(".tmp")
                torch.save(
                    {
                        "candidate_parameter_fingerprint": (
                            candidate_schedule.parameter_fingerprint
                        ),
                        "dataset_provenance_fingerprint": provenance["fingerprint"],
                        "rows": batch_rows,
                        "records": batch_records,
                        "manifest": batch_manifest,
                    },
                    temporary,
                )
                temporary.replace(shard_path)
        rows_by_id: dict[int, list[dict[str, Any]]] = {}
        for row in batch_rows:
            rows_by_id.setdefault(int(row["candidate_id"]), []).append(row)
        for record in batch_records:
            candidate_id = int(record["candidate_id"])
            record["candidate_phase"] = (
                "INITIAL_BUDGET"
                if int(record["distribution_parameter_id"])
                < candidate_schedule.initial_candidate_budget
                else "BOUNDED_REPLENISHMENT"
            )
            attempted_ids.add(candidate_id)
            if (
                record["status"] == MACRO_CANDIDATE_STATUS_VALID
                and len(accepted_ids) < candidate_schedule.target_valid
            ):
                record["accepted_for_training"] = True
                accepted_ids.append(candidate_id)
                rows.extend(rows_by_id[candidate_id])
            records.append(record)
        manifest_rows.extend(batch_manifest)
        if output is not None:
            cursor = {
                "schedule_fingerprint": candidate_schedule.fingerprint,
                "candidate_parameter_fingerprint": (
                    candidate_schedule.parameter_fingerprint
                ),
                "dataset_provenance_fingerprint": provenance["fingerprint"],
                "candidate_cursor": start + len(batch_specs),
                "attempted_candidate_ids": sorted(attempted_ids),
                "accepted_candidate_ids": accepted_ids,
            }
            temporary = output / "candidate_cursor.tmp"
            temporary.write_text(json.dumps(cursor, indent=2, sort_keys=True) + "\n")
            temporary.replace(output / "candidate_cursor.json")

    accepted_count = len(accepted_ids)
    if accepted_count >= candidate_schedule.target_valid:
        dataset_status = "TARGET_VALID_DATASET"
    elif accepted_count >= candidate_schedule.minimum_partial_valid:
        dataset_status = "PARTIAL_BOOTSTRAP_TRAINING"
    else:
        dataset_status = "INSUFFICIENT_VALID_PAIRS"
    if not rows:
        empty = torch.empty((0, 19), device=device, dtype=torch.float32)
        macro = TensorTable(
            data=empty,
            columns=[
                "path", "t", "branch", "x", "hatcf", "lnkf", "Hatc", "LnK",
                "M", "C", "K", "I_oper", "I_entry", "I_maintenance", "n_firms",
                "default_mass", "investment_mass", "mean_b",
                "lineage_id",
            ],
            metadata={
                "identity_schema": "canonical_candidate_to_local_path_v1",
                "canonical_candidate_ids": torch.empty(0, dtype=torch.int64),
                "local_path_ids": torch.empty(0, dtype=torch.int64),
                "canonical_to_local": {},
                "local_to_canonical": {},
            },
        )
        pairs = TensorTable(
            data=torch.empty((0, 12), device=device, dtype=torch.float32),
            columns=[
                "path", "t", "branch", "x_t", "x_t1", "Hatcf_t", "LnKF_t",
                "Hatc_t", "LnK_t", "Hatc_t1", "LnK_t1", "lineage_id",
            ],
            metadata=dict(macro.metadata),
        )
        batches = {"fc1": [], "sdf": []}
        pairing_audit = {
            "each_parent_has_exact_children": False,
            "reason": "NO_VALID_COMPLETE_PAIR",
        }
    else:
        macro = _macro_rows_to_table(rows, device)
        pairs = episode._build_sdf_pairs_from_macro_tensor(macro)
        pairing_audit = audit_candidate_pairing(pairs, accepted_ids)
        original_mode = episode.train_mode
        original_add_fc1 = episode.add_FC1loss
        try:
            episode.train_mode = "2time"
            episode.add_FC1loss = True
            target_batches = int(
                getattr(episode.hyperparams, "sdf_true_target_batches", 20)
            )
            min_parents = int(
                getattr(episode.hyperparams, "sdf_min_parent_groups_per_batch", 256)
            )
            batches = {
                "fc1": episode._create_sdf_batches_from_macro_tensor(
                    pairs, batch_size=macro_batch_size, n_branches=2
                ),
                "sdf": episode._create_sdf_batches_from_macro_tensor(
                    pairs,
                    batch_size=macro_batch_size,
                    n_branches=2,
                    target_num_batches=target_batches,
                    min_parent_groups_per_batch=min_parents,
                ),
            }
        finally:
            episode.train_mode = original_mode
            episode.add_FC1loss = original_add_fc1
        if dataset_status != "INSUFFICIENT_VALID_PAIRS" and not all(batches.values()):
            raise RuntimeError("valid candidate pairs produced an empty training stream")

    candidate_frame = pd.DataFrame(records)
    manifest_frame = pd.DataFrame(manifest_rows)
    acceptance = candidate_frame.get(
        "accepted_for_training", pd.Series(dtype=bool)
    ).astype(bool)
    coverage = {
        "original_candidate_count": int(
            len(candidate_ids)
            if candidate_ids is not None
            else candidate_schedule.initial_candidate_budget
        ),
        "attempted_candidate_count": int(len(candidate_frame)),
        "valid_complete_pair_count": int(
            (candidate_frame.get("status") == MACRO_CANDIDATE_STATUS_VALID).sum()
        ) if not candidate_frame.empty else 0,
        "accepted_training_pair_count": accepted_count,
        "target_valid": candidate_schedule.target_valid,
        "candidate_budget_exhausted": len(attempted_ids) >= len(ordered_specs),
        "initial_candidate_budget": candidate_schedule.initial_candidate_budget,
        "max_candidate_budget": candidate_schedule.max_candidate_budget,
        "initial_budget_attempted": int(
            (
                candidate_frame.get("candidate_phase") == "INITIAL_BUDGET"
            ).sum()
        ) if not candidate_frame.empty else 0,
        "replenishment_attempted": int(
            (
                candidate_frame.get("candidate_phase")
                == "BOUNDED_REPLENISHMENT"
            ).sum()
        ) if not candidate_frame.empty else 0,
        "dataset_status": dataset_status,
        "accepted_candidate_ids": accepted_ids,
        "failure_status_counts": (
            candidate_frame.loc[~acceptance, "status"].value_counts().to_dict()
            if not candidate_frame.empty else {}
        ),
        "failed_parent_count": int(
            (candidate_frame.get("status") == "PARENT_CLEARING_FAILED").sum()
        ) if not candidate_frame.empty else 0,
        "failed_child0_count": int(
            (candidate_frame.get("status") == "CHILD_0_CLEARING_FAILED").sum()
        ) if not candidate_frame.empty else 0,
        "failed_child1_count": int(
            (candidate_frame.get("status") == "CHILD_1_CLEARING_FAILED").sum()
        ) if not candidate_frame.empty else 0,
        "extinct_count": int(
            (candidate_frame.get("status") == "CHILD_EXTINCT").sum()
        ) if not candidate_frame.empty else 0,
        "strict_root_failure_count": int(
            candidate_frame.get("status", pd.Series(dtype=str)).isin(
                [
                    "PARENT_CLEARING_FAILED",
                    "CHILD_0_CLEARING_FAILED",
                    "CHILD_1_CLEARING_FAILED",
                ]
            ).sum()
        ) if not candidate_frame.empty else 0,
        "valid_subset_fraction": float(
            accepted_count
            / max(
                len(candidate_ids)
                if candidate_ids is not None
                else candidate_schedule.initial_candidate_budget,
                1,
            )
        ),
        "selection_interpretation": "conditional_on_complete_parent_two_child_pair",
        "outcome_by_distribution_family": (
            candidate_frame.groupby(["distribution_family", "status"])
            .size()
            .rename("count")
            .reset_index()
            .to_dict(orient="records")
            if not candidate_frame.empty else []
        ),
    }
    pairing_audit["candidate_coverage"] = coverage
    if output is not None:
        candidate_frame.to_csv(output / "candidate_outcomes.csv", index=False)
        manifest_frame.to_csv(output / "initial_b_distribution_coverage.csv", index=False)
        (output / "initial_b_distribution_manifest.json").write_text(
            json.dumps(
                {
                    "schedule_fingerprint": candidate_schedule.fingerprint,
                    "sampling_design": (
                        candidate_schedule.candidate_specs[0].distribution_family
                        if candidate_schedule.candidate_specs
                        else "UNKNOWN"
                    ),
                    "b_support": [
                        float(episode.config.SIM_B_INIT_MIN),
                        float(episode.config.SIM_B_INIT_MAX),
                    ],
                    "families": [item[0] for item in B_DISTRIBUTION_STRATA],
                    "coverage": coverage,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
    return OneStepMacroDataset(
        macro=macro,
        pairs=pairs,
        batches=batches,
        pairing_audit=pairing_audit,
        used_parent_batch_size=int(parent_batch_size),
        candidate_audit=candidate_frame,
        distribution_manifest=manifest_frame,
        dataset_status=dataset_status,
    )


def validation_coverage_record(
    *,
    original_candidate_ids: Sequence[int],
    before: OneStepMacroDataset,
    after: OneStepMacroDataset,
) -> dict[str, Any]:
    """Report fixed-schedule validation coverage without imputing failures."""

    original = tuple(map(int, original_candidate_ids))
    before_coverage = before.pairing_audit.get("candidate_coverage", {})
    after_coverage = after.pairing_audit.get("candidate_coverage", {})
    before_ids = set(map(int, before_coverage.get("accepted_candidate_ids", [])))
    after_ids = set(map(int, after_coverage.get("accepted_candidate_ids", [])))
    unexpected = sorted((before_ids | after_ids).difference(original))
    if unexpected:
        raise RuntimeError(
            "validation coverage contains replacement identities: "
            f"{unexpected[:20]}"
        )
    failure_counts = after_coverage.get("failure_status_counts", {})
    matched = before_ids & after_ids
    return {
        "original_validation_candidate_count": len(original),
        "valid_before_fc1": len(before_ids),
        "valid_after_fc1": len(after_ids),
        "matched_valid_subset_count": len(matched),
        "failed_parent_count": int(
            failure_counts.get("PARENT_CLEARING_FAILED", 0)
        ),
        "failed_child0_count": int(
            failure_counts.get("CHILD_0_CLEARING_FAILED", 0)
        ),
        "failed_child1_count": int(
            failure_counts.get("CHILD_1_CLEARING_FAILED", 0)
        ),
        "extinct_count": int(failure_counts.get("CHILD_EXTINCT", 0)),
        "strict_root_failure_count": int(
            failure_counts.get("PARENT_CLEARING_FAILED", 0)
            + failure_counts.get("CHILD_0_CLEARING_FAILED", 0)
            + failure_counts.get("CHILD_1_CLEARING_FAILED", 0)
        ),
        "valid_subset_fraction": float(len(after_ids) / max(len(original), 1)),
        "metrics_interpretation": "conditional_on_matched_valid_subset",
        "coverage_interpretation": "fixed_original_schedule_no_replacement",
    }


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
) -> dict[str, Any]:
    left = old.set_index("source_key")
    right = new.set_index("source_key")
    left_keys = list(map(str, left.index))
    right_keys = set(map(str, right.index))
    common = [key for key in left_keys if key in right_keys]
    if not common:
        raise RuntimeError("macro parent-bank movement has no common source keys")
    left = left.loc[common]
    right = right.loc[common]
    result: dict[str, Any] = {
        "old_source_count": int(len(old)),
        "new_source_count": int(len(new)),
        "common_source_count": int(len(common)),
        "dropped_source_count": int(len(set(left_keys).difference(right_keys))),
        "added_source_count": int(
            len(set(map(str, new["source_key"])).difference(left_keys))
        ),
        "common_source_share_of_old": float(len(common) / max(len(old), 1)),
        "movement_alignment": "COMMON_FIXED_SOURCE_KEYS_ONLY",
    }
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


def restrict_schedule_to_available_sources(
    schedule: MacroParentSourceSchedule,
    candidate_summary: pd.DataFrame,
) -> tuple[MacroParentSourceSchedule, dict[str, Any]]:
    """Keep the fixed source order while censoring economically extinct rows.

    A global validation schedule is fixed across equilibrium outers, while the
    Firm solution is not.  Production ``StateBank`` correctly omits a path once
    its cross-section is extinct, so such a source cannot supply the required
    two one-step child branches.  We never replace it with a new random source:
    the active schedule is the ordered available subset, and the censoring is
    recorded explicitly for cross-outer diagnostics.
    """
    available = set(map(str, candidate_summary["source_key"].tolist()))
    retained = tuple(key for key in schedule.source_keys if key.canonical in available)
    missing = tuple(key for key in schedule.source_keys if key.canonical not in available)
    if not retained:
        raise RuntimeError(
            "fixed macro parent source schedule has no non-extinct sources under "
            "the current Firm solution"
        )
    active = replace(schedule, source_keys=retained)
    audit = {
        "global_schedule_fingerprint": schedule.fingerprint,
        "active_schedule_fingerprint": active.fingerprint,
        "scheduled_source_count": len(schedule.source_keys),
        "available_source_count": len(retained),
        "missing_extinct_source_count": len(missing),
        "availability_rate": float(len(retained) / max(len(schedule.source_keys), 1)),
        "missing_extinct_source_keys": [key.to_dict() for key in missing],
        "replacement_sources_used": False,
        "censoring_reason": "PRODUCTION_ROLLOUT_PATH_EXTINCT",
    }
    return active, audit


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


def harvest_available_macro_parent_schedule_with_candidate(
    *,
    models: Mapping[str, torch.nn.Module],
    episode: Any,
    initial_state: Mapping[str, torch.Tensor],
    schedule: MacroParentSourceSchedule,
    source_episode: int,
    branch_num: int,
    device: torch.device,
) -> tuple[
    MacroParentSourceSchedule,
    HarvestedParentBank,
    pd.DataFrame,
    np.ndarray,
    dict[str, Any],
]:
    """Harvest the ordered non-extinct subset of a cross-outer schedule."""
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
    active_schedule, availability_audit = restrict_schedule_to_available_sources(
        schedule, candidate
    )
    entry_ids, selected, occupancy = select_scheduled_entries(
        bank, candidate, active_schedule
    )
    harvested = HarvestedParentBank(
        bank=bank,
        selected_entry_ids=entry_ids,
        summary=selected,
        occupancy=occupancy,
        simulation_output=output,
        initial_hash_audit=initial_audit,
    )
    return (
        active_schedule,
        harvested,
        candidate,
        candidate_occupancy,
        availability_audit,
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

