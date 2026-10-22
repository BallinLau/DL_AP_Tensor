"""Shared contracts and helpers for the EP3-H10 hybrid experiment.

The runner is deliberately plan-only unless an explicit, content-bound
approval is supplied.  This module contains no optimizer code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

import torch

from analysis.run_full_nested_finite_direct_outer4 import Budget, stages
from analysis.test_frozen_policy_root_solve import BP_PREFIXES, P_PREFIXES, Q_PREFIXES


EXPERIMENT_ID = "hybrid_ep3_h10_unified_20261023"
SCHEMA = "HYBRID_EP3_H10_UNIFIED_APPROVAL_V1"
HYBRID_OUTER_COUNT = 2


@dataclass(frozen=True)
class HybridFresh100Contract:
    """Frozen EP3 Firm Inner data semantics."""

    mode: str = "modeb"
    paths: int = 200
    firms: int = 200
    horizon: int = 100
    branches: int = 2
    entry_mode: str = "investment_compare"
    state_bank_enabled: bool = False
    pv_mixture_enabled: bool = True
    pv_mixture_ratio: float = 0.20
    bp_head_training_enabled: bool = False
    simulation_bp_action_source: str = "grid"
    episode_id: int = 1

    def validate(self) -> None:
        expected = asdict(self)
        checks = {
            "mode": expected["mode"] == "modeb",
            "paths": expected["paths"] == 200,
            "firms": expected["firms"] == 200,
            "horizon": expected["horizon"] == 100,
            "branches": expected["branches"] == 2,
            "entry_mode": expected["entry_mode"] == "investment_compare",
            "state_bank_enabled": expected["state_bank_enabled"] is False,
            "pv_mixture_enabled": expected["pv_mixture_enabled"] is True,
            "pv_mixture_ratio": abs(expected["pv_mixture_ratio"] - 0.20) < 1e-12,
            "bp_head_training_enabled": expected["bp_head_training_enabled"] is False,
            "simulation_bp_action_source": expected["simulation_bp_action_source"] == "grid",
            "episode_id": expected["episode_id"] > 0,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"invalid Fresh100 contract fields: {failed}")


@dataclass(frozen=True)
class HybridMacroMixtureContract:
    """Train-only 80% on-policy / 20% coverage; fixed separate Validation."""

    on_policy_fraction: float = 0.80
    target: int = 4096
    validation_target: int = 1024
    validation_mixed: bool = False

    def validate(self) -> None:
        if not 0.0 <= self.on_policy_fraction <= 1.0:
            raise ValueError("on_policy_fraction must be in [0,1]")
        if min(self.target, self.validation_target) <= 0:
            raise ValueError("macro mixture targets must be positive")
        if self.validation_mixed:
            raise ValueError("Validation must use the fixed H10 Validation bank")


def hybrid_budget() -> Budget:
    """H10 budgets, with the two-Outer continuation budget only."""

    return Budget(equilibrium_outer_max=HYBRID_OUTER_COUNT)


def hybrid_stage_plan() -> list[dict[str, Any]]:
    """Execution stages for two hybrid Outers.

    ``firm_bank`` is interpreted as one unified Fresh100 data generation per
    Outer.  P consumes the same parent/children tensors as Q.  BP remains a
    stage but is a no-op guard when the BP head is disabled.
    """

    output: list[dict[str, Any]] = []
    for outer, kind, indices in stages(hybrid_budget()):
        if outer == 0 and kind == "reuse_firm":
            continue
        if outer == 0 and kind == "data" and indices.get("version") == 0:
            kind = "unified_firm_data"
        if kind == "firm_bank":
            kind = "unified_firm_data"
        output.append({"outer": outer, "kind": kind, "indices": indices})
    return output


def stage_id(outer: int, kind: str, indices: dict[str, int]) -> str:
    suffix = "_".join(f"{key}{value}" for key, value in indices.items())
    return f"outer_{outer:03d}/{kind}" + (f"/{suffix}" if suffix else "")


def parameter_state_hash(model: torch.nn.Module, prefixes: tuple[str, ...]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in model.named_parameters():
        if any(name.startswith(f"{prefix}.") for prefix in prefixes):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def stage_parameter_hashes(model: torch.nn.Module) -> dict[str, str]:
    return {
        "P": parameter_state_hash(model, P_PREFIXES),
        "Q": parameter_state_hash(model, Q_PREFIXES),
        "BP": parameter_state_hash(model, BP_PREFIXES),
    }


def tensor_pool_fingerprint(pool: Any) -> str:
    """Hash the immutable unified P/Q parent-children pool."""

    digest = hashlib.sha256()
    for tensor in (pool.parent, *pool.children):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def pool_from_batches(
    batches: list[dict[str, Any]],
) -> Any | None:
    """Reconstruct the immutable tensor pool consumed by a batch list."""

    if not batches:
        return None
    parents = [batch["parent"].detach().cpu() for batch in batches]
    branches = [[child.detach().cpu() for child in batch["children"]] for batch in batches]
    return SimpleNamespace(
        parent=torch.cat(parents, dim=0),
        children=[
            torch.cat([branch[j] for branch in branches], dim=0)
            for j in range(len(branches[0]))
        ],
    )


def build_unified_fresh100_batches(
    episode: Any,
    table: Any,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Build one pool consumed unchanged by both H10-P and EP3-Q."""

    contract = HybridFresh100Contract()
    contract.validate()
    if episode._simulation_bp_action_source() != "grid":
        raise RuntimeError("Fresh100 simulation must use BP grid")
    train_batches, validation_batches, summary = episode._prepare_mixed_policy_value_batches(
        table,
        batch_size=batch_size,
        n_branches=contract.branches,
    )
    if not train_batches:
        raise RuntimeError("Fresh100 produced no P/Q training batches")
    unified_pool = pool_from_batches(train_batches)
    if unified_pool is None:
        raise RuntimeError("Fresh100 tensor produced no unified P/Q pool")
    fingerprint = tensor_pool_fingerprint(unified_pool)
    validation_pool = pool_from_batches(validation_batches)
    return {
        "fingerprint": fingerprint,
        "pool": unified_pool,
        "validation_pool": validation_pool,
        "train_batches": train_batches,
        "validation_batches": validation_batches,
        "summary": summary,
    }


def run_h10_p_on_unified_pool(
    *,
    model: torch.nn.Module,
    episode: Any,
    pool: dict[str, Any],
    batch_size: int,
    args: Any,
    policy_outer: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """H10 P solver with explicit same-pool and P-only guards."""

    from analysis.run_full_nested_equilibrium import _p_policy_outer

    assert_same_fresh100_pool(pool["fingerprint"], tensor_pool_fingerprint(pool["pool"]))
    before = stage_parameter_hashes(model)
    # H10 P accepts validation tensors. Fresh100's own internal validation is
    # fixed and must not be re-mixed with synthetic coverage.
    validation = pool.get("validation_pool")
    if validation is None:
        validation = pool["pool"]
    train_children = torch.stack(pool["pool"].children, dim=1)
    validation_children = torch.stack(validation.children, dim=1)
    if tuple(train_children.shape[1:]) != (2, train_children.shape[2]):
        raise RuntimeError("unified Fresh100 pool must contain exactly two Child branches")
    solved, row, history = _p_policy_outer(
        model=model,
        episode=episode,
        train_parents=pool["pool"].parent,
        train_children=train_children,
        validation_parents=validation.parent,
        validation_children=validation_children,
        batch_size=batch_size,
        args=args,
        policy_outer=policy_outer,
    )
    after = stage_parameter_hashes(solved)
    assert_p_stage_isolation(before, after)
    return solved, {"row": row, "history": history, "optimizer_executed": len(history) > 0}


def run_ep3_q_regimes_on_unified_pool(
    *,
    model: torch.nn.Module,
    episode: Any,
    pool: dict[str, Any],
) -> dict[str, Any]:
    """EP3 direct-Q zero/default/survival/polish on the same Fresh100 pool."""

    if str(getattr(model, "q_parameterization", "b_times_unit")).lower() != "direct":
        raise RuntimeError("EP3 Q regime training requires direct Q")
    assert_same_fresh100_pool(pool["fingerprint"], tensor_pool_fingerprint(pool["pool"]))
    before = stage_parameter_hashes(model)
    frozen_p = model
    summary = episode._run_q_regime_training(
        pool["train_batches"],
        frozen_p,
        validation_batches=pool["validation_batches"] or None,
    )
    after = stage_parameter_hashes(model)
    assert_q_stage_isolation(before, after)
    summary = dict(summary)
    summary.update(
        {
            "unified_fresh100_fingerprint": pool["fingerprint"],
            "q_regime_phases": ["zero", "default", "survival", "polish"],
            "optimizer_executed": int(summary.get("optimizer_steps", 0)) > 0,
        }
    )
    return summary


def bp_disabled_stage_result(episode: Any, model: torch.nn.Module) -> dict[str, Any]:
    before = stage_parameter_hashes(model)
    if bool(getattr(episode.hyperparams, "pv_bp_head_training_enabled", True)):
        raise RuntimeError("BP head must be disabled")
    if episode._simulation_bp_action_source() != "grid":
        raise RuntimeError("BP action source must be grid")
    return {
        "BP_HEAD_TRAINING_SKIPPED": True,
        "BP_HEAD_OPTIMIZER_STEPS": 0,
        "SIMULATION_BP_ACTION_SOURCE": "GRID",
        "OPTIMIZER_STEPS": 0,
        "OPTIMIZER_EXECUTED": False,
        "BP_HASH_BEFORE": before["BP"],
    }


def fresh100_tensor_to_macro_replay(
    tensor_macro: Any,
    *,
    split: str,
    burn_in: int = 20,
) -> pd.DataFrame:
    """Convert EP3 Fresh100 macro rows to the H10 replay ledger schema.

    This is an adapter only. It never invents a missing Child, fills an
    undefined account, or changes an economic label.
    """

    if split not in {"train", "validation"}:
        raise ValueError("Fresh100 replay split must be train or validation")
    frame = tensor_macro.to_dataframe().copy()
    integer_columns = ("path", "t", "branch")
    for name in integer_columns:
        if name in frame:
            frame[name] = np.rint(frame[name]).astype(np.int64)
    required = {
        "path", "t", "branch", "Hatc", "LnK", "hatcf", "lnkf",
        "C", "K", "M", "n_firms",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Fresh100 macro tensor missing columns: {missing}")
    label_columns = [
        "Hatc_cal_t", "LnK_cal_t", "Hatc_cal_t1", "LnK_cal_t1", "C_raw_t1", "K_cal_t1"
    ]
    frame = frame.rename(columns={
        "path": "parent_candidate_id",
        "Hatc": "Hatc_cal_t",
        "LnK": "LnK_cal_t",
        "hatcf": "Hatc_cal_t1",
        "lnkf": "LnK_cal_t1",
        "C": "C_raw_t1",
        "K": "K_cal_t1",
    })
    frame["split"] = split
    frame["child_branch"] = frame["branch"].astype(int)
    frame["x_t"] = np.nan
    frame["x_t1"] = np.nan
    frame["economic_state_valid"] = (
        np.isfinite(frame[label_columns].to_numpy()).all(axis=1)
        & (frame["C_raw_t1"] > 0)
        & (frame["K_cal_t1"] > 0)
    )
    # H10 replay requires t-1 >= burn_in and the primary single-spine Child.
    frame = frame.loc[(frame["t"] - 1 >= burn_in) & (frame["child_branch"] == 0)].copy()
    if frame.empty:
        return frame
    frame["parent_time"] = frame["t"] - 1
    frame["source_outer"] = "HYBRID_FRESH100"
    frame["replay_semantics"] = "ON_POLICY_100_PERIOD_PRIMARY_BRANCH"
    frame["original_lineage"] = frame["parent_candidate_id"]
    frame["valid_label"] = frame["economic_state_valid"]
    frame["initial_mode"] = "EP3_MODE_B"
    if frame.duplicated(["parent_candidate_id", "t"]).any():
        raise ValueError("Fresh100 replay contains duplicated parent-time rows")
    return frame.reset_index(drop=True)


def select_macro_mixture(
    *,
    coverage: Any,
    on_policy_replay: Any,
    edges: Any,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """80/20 Train mixture; fixed H10 Validation is supplied separately."""

    from data.finite_direct_longrun_replay import select_mixture

    contract = HybridMacroMixtureContract()
    contract.validate()
    mixed, audit = select_mixture(
        coverage,
        on_policy_replay,
        edges=edges,
        seed=seed,
        target=contract.target,
        on_policy_fraction=contract.on_policy_fraction,
    )
    return mixed, audit


def configure_episode_for_hybrid(episode: Any) -> dict[str, Any]:
    """Apply only the reviewed hybrid runtime switches, never economics."""

    contract = HybridFresh100Contract()
    contract.validate()
    hp = episode.hyperparams
    hp.entry_mode = contract.entry_mode
    hp.entry_spec_version = "investment_compare_v1"
    hp.entry_capital_ratio = 0.05
    hp.entry_size_ratio = 1.0
    hp.entry_cost_max = 0.5
    hp.entry_dummy_i = 0.0
    hp.consumption_aggregation_mode = "raw"
    hp.consumption_floor = 0.0
    hp.node_accounting_mode = "economic_node_ledger"
    hp.aggregation_mode = "legacy_mixed_replay"
    hp.market_clearing_mode = "disabled"
    hp.simulate_horizon = contract.horizon
    hp.use_tensor_pipeline = True
    hp.pv_training_flow = "staged"
    hp.firm_target_update = "stage_hard"
    hp.q_checkpoint_loaded = True
    hp.pv_bp_head_training_enabled = contract.bp_head_training_enabled
    hp.simulation_bp_action_source = contract.simulation_bp_action_source
    hp.state_bank_enabled = contract.state_bank_enabled
    hp.pv_mixture_enabled = contract.pv_mixture_enabled
    hp.pv_mixture_ratio = contract.pv_mixture_ratio
    episode.episode_id = contract.episode_id
    effective_grid = episode._simulation_bp_action_source() == "grid"
    if not effective_grid:
        raise RuntimeError("hybrid simulation did not resolve BP action source to grid")
    return asdict(contract)


def assert_same_fresh100_pool(fingerprint_a: Any, fingerprint_b: Any) -> None:
    if fingerprint_a != fingerprint_b:
        raise ValueError("P and Q must consume the same unified Fresh100 pool")


def assert_p_stage_isolation(before: dict[str, str], after: dict[str, str]) -> None:
    if before["Q"] != after["Q"] or before["BP"] != after["BP"]:
        raise RuntimeError("P stage changed Q or BP parameters")


def assert_q_stage_isolation(before: dict[str, str], after: dict[str, str]) -> None:
    if before["P"] != after["P"] or before["BP"] != after["BP"]:
        raise RuntimeError("Q stage changed P or BP parameters")


def assert_bp_disabled(before: dict[str, str], after: dict[str, str]) -> None:
    if before["BP"] != after["BP"]:
        raise RuntimeError("BP parameters changed while BP head is disabled")


def training_plan() -> dict[str, Any]:
    fresh = HybridFresh100Contract()
    fresh.validate()
    macro = HybridMacroMixtureContract()
    macro.validate()
    return {
        "version": "HYBRID_EP3_H10_UNIFIED_V1",
        "experiment_id": EXPERIMENT_ID,
        "outer_count": HYBRID_OUTER_COUNT,
        "budget": asdict(hybrid_budget()),
        "fresh100": asdict(fresh),
        "macro_mixture": asdict(macro),
        "initialization": "EP3_WEIGHTS_ONLY_NEW_OPTIMIZERS",
        "firm_training": {
            "P": "H10_EXPLICIT_P_SOLVER",
            "Q": "EP3_DIRECT_Q_ZERO_DEFAULT_SURVIVAL_POLISH",
            "unified_data_pool": True,
            "bp_head_training": False,
            "bp_action_source": "grid",
        },
        "outer_training": "H10_ARCHITECTURE_FC1_SDF_CROSS_BLOCK",
        "training_distribution_is_not_population": True,
        "stages": hybrid_stage_plan(),
    }
