#!/usr/bin/env python3
"""Formal resumable nested equilibrium training on one CUDA device.

The runner is orchestration only.  It reuses the validated production Bellman,
Q, simulation, FC1, SDF, and BP-grid implementations.  The hierarchy is named
unambiguously throughout:

``equilibrium_outer -> firm_inner -> (p_policy_outer, q_picard_outer)`` and
``equilibrium_outer -> macro_inner -> sdf_chunk``.

FC2 is structurally excluded.  Recursive SDF training is disabled.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.economic_config import AnalysisEconomicConfig  # noqa: E402
from analysis.nested_equilibrium_utils import (  # noqa: E402
    append_summary_row,
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_json,
    capture_episode_runtime_state,
    capture_training_transaction,
    cuda_memory,
    load_checkpoint,
    make_checkpoint_payload,
    module_hash,
    publish_checkpoint,
    require_finite_metrics,
    require_finite_module,
    restore_training_transaction,
    run_with_oom_fallback,
    safe_torch_load,
    seed_all,
    stable_json_hash,
    tensor_hash,
)
from analysis.macro_parent_state_bank import (  # noqa: E402
    MacroCandidateSchedule,
    build_macro_candidate_schedule,
    build_one_step_macro_dataset,
    load_committed_one_step_macro_dataset,
    load_candidate_schedule,
    lock_validation_candidate_ids,
    one_step_macro_dataset_fingerprint,
    save_candidate_schedule,
    save_committed_one_step_macro_dataset,
    validation_coverage_record,
)
from analysis.market_cleared_banks import (  # noqa: E402
    build_market_cleared_transition_bank,
)
from data.boundary_aware_market_clearing import BoundaryIntegrationConfig  # noqa: E402
from data.market_clearing import MarketClearingConfig  # noqa: E402
from analysis.run_coupled_fixed_macro import (  # noqa: E402
    build_coupled_view,
    coupled_transition_metrics,
)
from analysis.run_macro_block_isolation import (  # noqa: E402
    aligned_macro_vectors,
    build_in_memory_batches,
    firm_hashes,
    flatten_batches,
    evaluate_fc1,
    movement_metrics,
    movement_relation,
    run_fc1_fit,
    run_sdf_chunks,
    set_macro_scope,
    snapshot_functions,
    table_hash,
    two_cycle_detect,
    TwoCycleDiagnosticsConfig,
)
from analysis.run_macro_outer_inner_pq import (  # noqa: E402
    _to_frame,
    audit_m_parity,
    install_fc2_guard,
    macro_diagnostics,
    make_initial_state,
    refresh_transition_bank,
    run_simulation,
    simulation_policy_metrics,
    state_hash,
)
from analysis.run_neural_policy_iteration import (  # noqa: E402
    _solve_frozen_policy,
    build_policy_view,
    function_transition_metrics,
    policy_transition_metrics,
    resolve_policy_evaluation_target,
)
from analysis.run_q_root_solve_fixed_macro import (  # noqa: E402
    _non_q_hash,
    build_q_survival_problem,
    evaluate_q_problem,
    refresh_q_targets,
    run_q_adam,
)
from analysis.test_frozen_policy_root_solve import (  # noqa: E402
    BP_PREFIXES,
    P_PREFIXES,
    Q_PREFIXES,
    _state_hash,
    build_problem,
    evaluate_problem,
)
from config import Config  # noqa: E402
from experiments.evaluate_bp_teacher_fit import normalized_regrets  # noqa: E402
from experiments.run_utils import build_hyperparams, build_models, build_optimizers  # noqa: E402
from training.bp_grid_teacher import BPGridTeacher  # noqa: E402
from training.episode import Episode  # noqa: E402
from training.inner_fixed_point import (  # noqa: E402
    InnerFPChildBank,
    InnerFPCollocationBank,
    build_child_bank,
    build_collocation_bank,
    make_batches,
)


P_EVAL_SCHEDULE = (0.04, 0.03, 0.02, 0.015, 0.01)
SUMMARY_COLUMNS = (
    "stage", "equilibrium_outer", "firm_inner", "p_policy_outer",
    "q_picard_outer", "macro_inner", "p_policy_outer_count",
    "q_picard_outer_count", "macro_inner_count", "P0_residual",
    "PI_residual", "Q_residual", "P0_move_p90", "PI_move_p90",
    "Q_move_p90", "investment_flip", "default_flip", "bp0_move_p90",
    "bpI_move_p90", "M_function_move", "FC1_C_function_move",
    "FC1_K_function_move", "Hatc_target_move", "LnK_target_move",
    "Hatc_path_move", "LnK_path_move", "C_path_move", "K_path_move",
    "FC1_C_val_R2", "FC1_K_val_R2", "SDF_objective", "SDF_moment1",
    "SDF_moment2", "within_D_M_movement", "investment_rate",
    "default_rate", "mean_b", "mean_M", "mean_firms", "mean_C", "mean_K",
    "GPU_max_memory",
    "stage_seconds", "total_elapsed",
)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _git_branch() -> str:
    return subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()


def _quantile(value: torch.Tensor, q: float) -> float:
    flat = value.detach().reshape(-1).to(torch.float64)
    flat = flat[torch.isfinite(flat)]
    return float(torch.quantile(flat, q)) if flat.numel() else float("nan")


def _model_states(models: Mapping[str, torch.nn.Module]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        for name, model in models.items()
    }


def _restore_model_states(
    models: Mapping[str, torch.nn.Module],
    states: Mapping[str, Mapping[str, torch.Tensor]],
) -> None:
    for name, model in models.items():
        model.load_state_dict(states[name], strict=True)


def _load_parent_bank(path: Path) -> InnerFPCollocationBank:
    payload = safe_torch_load(path, map_location="cpu")
    bank = InnerFPCollocationBank(
        parent_states=payload["parent_states"],
        source_kind=payload["source_kind"],
        seed=int(payload["seed"]), split=str(payload["split"]),
        metadata=dict(payload.get("metadata", {})),
    )
    if bank.fingerprint != payload.get("fingerprint"):
        raise RuntimeError(f"collocation bank hash mismatch: {path}")
    return bank


def _load_child_bank(payload: Mapping[str, Any]) -> InnerFPChildBank:
    bank = InnerFPChildBank(
        children=payload["children"], eps_x=payload["eps_x"],
        eps_z=payload["eps_z"], u_eta=payload["u_eta"],
        u_i=payload["u_i"], seed=int(payload["seed"]),
        metadata=dict(payload.get("metadata", {})),
    )
    if bank.fingerprint != payload.get("fingerprint"):
        raise RuntimeError("child shock bank hash mismatch")
    return bank


def _args_fingerprint(args: argparse.Namespace) -> str:
    ignored = {
        "cold_start", "resume", "allow_code_mismatch", "output_dir", "smoke",
        "device",
    }
    return stable_json_hash({key: value for key, value in vars(args).items() if key not in ignored})


_RESUME_RUNTIME_ONLY_ARGS = {
    "cold_start",
    "resume",
    "allow_code_mismatch",
    "output_dir",
    "device",
}
_PATH_ARGS = {"resume", "output_dir", "initial_checkpoint"}


def _resolve_resume(args: argparse.Namespace) -> tuple[Optional[Path], Optional[Path]]:
    if args.resume is None:
        return None, None
    requested = Path(args.resume).expanduser().resolve()
    if requested.is_dir():
        run_root = requested
        pointer_path = run_root / "latest_checkpoint.json"
        if not pointer_path.is_file():
            raise RuntimeError(f"resume run is missing latest_checkpoint.json: {run_root}")
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        checkpoint = Path(pointer["latest"]).expanduser().resolve()
    elif requested.is_file():
        checkpoint = requested
        run_root = next(
            (
                parent
                for parent in (requested.parent, *requested.parents)
                if (parent / "metadata" / "run.json").is_file()
            ),
            None,
        )
        if run_root is None:
            raise RuntimeError(
                f"cannot locate run root for resume checkpoint: {requested}"
            )
    else:
        raise RuntimeError(f"resume target does not exist: {requested}")
    if not checkpoint.is_file():
        raise RuntimeError(f"resume checkpoint does not exist: {checkpoint}")
    return Path(run_root), checkpoint


def _explicit_cli_destinations(argv: Sequence[str]) -> set[str]:
    destinations: set[str] = set()
    for token in argv:
        if not str(token).startswith("--"):
            continue
        name = str(token).split("=", 1)[0][2:]
        if name.startswith("no-"):
            name = name[3:]
        destinations.add(name.replace("-", "_"))
    return destinations


def _coerce_recorded_arg(name: str, value: Any) -> Any:
    if name in _PATH_ARGS:
        return None if value is None else Path(value)
    return value


def _resume_with_recorded_config(
    args: argparse.Namespace,
    argv: Sequence[str],
) -> tuple[argparse.Namespace, Optional[Path], Optional[Path]]:
    """Hydrate immutable resume arguments from the original run metadata."""

    run_root, checkpoint = _resolve_resume(args)
    if run_root is None:
        return args, None, None
    metadata_path = run_root / "metadata" / "run.json"
    if not metadata_path.is_file():
        raise RuntimeError("resume run is missing metadata/run.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    recorded = metadata.get("args")
    if not isinstance(recorded, Mapping):
        raise RuntimeError("resume metadata does not contain original args")
    explicit = _explicit_cli_destinations(argv)
    conflicts: dict[str, dict[str, Any]] = {}
    for name, stored_value in recorded.items():
        if name in _RESUME_RUNTIME_ONLY_ARGS or not hasattr(args, name):
            continue
        restored = _coerce_recorded_arg(name, stored_value)
        current = getattr(args, name)
        if name in explicit and current != restored:
            conflicts[name] = {"recorded": stored_value, "requested": current}
        setattr(args, name, restored)
    if conflicts:
        raise RuntimeError(f"resume immutable argument conflict: {conflicts}")
    args.cold_start = False
    observed = _args_fingerprint(args)
    expected = metadata.get("hyperparameter_fingerprint")
    if observed != expected:
        raise RuntimeError(
            "resume metadata fingerprint mismatch after immutable config restore: "
            f"{observed} != {expected}"
        )
    return args, run_root, checkpoint


def _configure_hyperparams(args: argparse.Namespace):
    hp = build_hyperparams()
    hp.n_paths = int(args.macro_train_target_valid)
    hp.batch_size = int(args.macro_batch_size)
    hp.simulate_horizon = 1
    hp.fc1_only_epochs = int(args.fc1_production_epochs)
    hp.sdf_true_only_epochs = int(args.sdf_chunks * args.sdf_epochs_per_chunk)
    hp.sdf_recursive_only_epochs = 0
    hp.fc1_max_rounds = 1
    # Recursive FC1 diagnostics belong to the separate long-horizon rollout.
    # Enabling this on a one-step primary table makes the production batch
    # builder demand unavailable future rows and discard every valid parent.
    hp.fc1_rollout_diagnostic_enabled = False
    hp.bp_distill_epochs = int(args.policy_distill_max_epochs)
    hp.bp_distill_patience = int(args.policy_distill_patience)
    hp.bp_distill_max_optimizer_steps = int(args.policy_distill_max_steps)
    hp.bp_distill_grad_clip_norm = 10.0
    hp.pv_bp_head_training_enabled = True
    hp.market_clearing_mode = str(args.market_clearing_mode)
    hp.market_clearing_coarse_points = int(args.market_clearing_coarse_points)
    hp.market_clearing_refinement_points = int(
        args.market_clearing_refinement_points
    )
    hp.market_clearing_max_refinement_rounds = int(
        args.market_clearing_max_refinement_rounds
    )
    hp.market_clearing_support_lower = float(
        args.market_clearing_support_lower
    )
    hp.market_clearing_support_upper = float(
        args.market_clearing_support_upper
    )
    hp.market_clearing_support_padding = float(
        args.market_clearing_support_padding
    )
    hp.market_clearing_residual_tolerance = float(
        args.market_clearing_residual_tolerance
    )
    hp.market_clearing_width_tolerance = float(
        args.market_clearing_width_tolerance
    )
    hp.market_clearing_candidate_chunk_size = int(
        args.market_clearing_candidate_chunk_size
    )
    hp.aggregation_mode = str(args.aggregation_mode)
    hp.continuum_i_points = int(args.continuum_i_points)
    hp.continuum_z_points = int(args.continuum_z_points)
    hp.continuum_projection_tolerance = float(
        args.continuum_projection_tolerance
    )
    hp.bp_actual_use_trace_enabled = bool(args.bp_actual_use_trace)
    if hp.market_clearing_mode != "disabled":
        # Lock to the already-audited Gomes/root-diagnostic economic setup.
        hp.entry_mode = "investment_compare"
        hp.entry_spec_version = "investment_compare_v1"
        hp.entry_capital_ratio = 0.05
        hp.entry_size_ratio = 1.0
        hp.entry_cost_max = 0.5
        hp.entry_dummy_i = 0.0
        hp.entry_rng_mode = "stable_economic_identity"
        hp.transition_rng_mode = "stable_firm_identity"
        hp.consumption_aggregation_mode = "raw"
        hp.consumption_floor = 0.0
        hp.node_accounting_mode = "economic_node_ledger"
        hp.simulation_bp_action_source = "grid"
        hp.simulation_bp_grid_n_child_shocks = 2
    return hp


def _candidate_operator_configs(
    args: argparse.Namespace,
) -> tuple[BoundaryIntegrationConfig, MarketClearingConfig]:
    integration = BoundaryIntegrationConfig(
        z_scan_points=int(args.boundary_z_scan_points),
        z_quadrature_points=int(args.boundary_z_quadrature_points),
        i_scan_points=int(args.boundary_i_scan_points),
        i_quadrature_points=int(args.boundary_i_quadrature_points),
        boundary_refinement_rounds=int(args.boundary_refinement_rounds),
        scan_tail_probability=float(args.boundary_scan_tail_probability),
        candidate_h_microbatch_size=int(args.boundary_candidate_h_microbatch_size),
    )
    root = MarketClearingConfig(
        solver_mode=str(args.macro_clearing_solver_mode),
        coarse_points=int(args.market_clearing_coarse_points),
        refinement_points=int(args.market_clearing_refinement_points),
        max_refinement_rounds=int(args.market_clearing_max_refinement_rounds),
        support_lower=float(args.market_clearing_support_lower),
        support_upper=float(args.market_clearing_support_upper),
        support_padding=float(args.market_clearing_support_padding),
        residual_tolerance=float(args.market_clearing_residual_tolerance),
        width_tolerance=float(args.market_clearing_width_tolerance),
        candidate_chunk_size=int(args.market_clearing_candidate_chunk_size),
    )
    integration.validate()
    root.validate()
    return integration, root


def _setup_fresh(args: argparse.Namespace, device: torch.device):
    Config.DEVICE = device
    hp = _configure_hyperparams(args)
    models = build_models(device)
    q_mode = str(getattr(models["policy_value"], "q_parameterization", "")).lower()
    if q_mode != "direct":
        raise RuntimeError(f"formal nested run requires direct Q, got {q_mode!r}")
    # A resume restores the complete current model/optimizer state below.  The
    # original warm-start path remains immutable provenance in run.json and in
    # the fingerprint, but it must not be loaded a second time or be required
    # to exist on the resume host.
    if args.initial_checkpoint is not None and args.resume is None:
        payload = safe_torch_load(
            args.initial_checkpoint.expanduser().resolve(), map_location="cpu"
        )
        source = payload.get("models", payload) if isinstance(payload, Mapping) else None
        if not isinstance(source, Mapping):
            raise RuntimeError("initial checkpoint does not expose model states")
        missing = [
            name for name in ("policy_value", "sdf_fc1") if name not in source
        ]
        if missing:
            raise RuntimeError(
                f"initial checkpoint is missing required model states: {missing}"
            )
        for name in ("policy_value", "sdf_fc1"):
            state = source[name]
            if isinstance(state, torch.nn.Module):
                state = state.state_dict()
            if not isinstance(state, Mapping):
                raise RuntimeError(
                    f"initial checkpoint state for {name} is malformed"
                )
            models[name].load_state_dict(state, strict=True)
            require_finite_module(name, models[name])
    optimizers = build_optimizers(models, hp)
    episode = Episode(
        models=models, optimizers=optimizers, config=Config, hyperparams=hp,
        device=device, episode_id=1, q_checkpoint_loaded=False,
    )
    return episode, hp, models


def _fixed_bank_hashes(
    train_bank: InnerFPCollocationBank,
    validation_bank: InnerFPCollocationBank,
    train_children: InnerFPChildBank,
    validation_children: InnerFPChildBank,
    macro_probe: Mapping[str, torch.Tensor],
) -> dict[str, str]:
    return {
        "collocation_train": train_bank.fingerprint,
        "collocation_validation": validation_bank.fingerprint,
        "child_train": train_children.fingerprint,
        "child_validation": validation_children.fingerprint,
        "macro_probe": tensor_hash(sorted(macro_probe.items())),
    }


def _save_stage_checkpoint(
    *, run_root: Path, path: Path, models: Mapping[str, torch.nn.Module],
    optimizers: Mapping[str, torch.optim.Optimizer],
    episode: Episode,
    state: Mapping[str, Any], git_head: str, fingerprint: str,
    bank_hashes: Mapping[str, str], extra: Optional[Mapping[str, Any]] = None,
) -> None:
    payload = make_checkpoint_payload(
        models=models,
        optimizers=optimizers,
        schedulers=getattr(episode, "lr_schedulers", {}),
        episode_runtime_state=capture_episode_runtime_state(episode),
        state=state, git_head=git_head,
        hyperparameter_fingerprint=fingerprint,
        extra={"bank_hashes": dict(bank_hashes), **dict(extra or {})},
    )
    publish_checkpoint(path=path, payload=payload, run_root=run_root)
    atomic_write_json(run_root / "state.json", {**dict(state), "checkpoint": str(path.resolve())})


def _checkpoint_path(run_root: Path, *parts: str) -> Path:
    return run_root.joinpath(*parts)


def _build_or_load_fixed_banks(
    *, args: argparse.Namespace, run_root: Path, episode: Episode,
    models: Mapping[str, torch.nn.Module], device: torch.device,
) -> tuple[
    InnerFPCollocationBank, InnerFPCollocationBank,
    InnerFPChildBank, InnerFPChildBank,
]:
    fixed = run_root / "fixed_point"
    train_path = fixed / "collocation_train.pt"
    validation_path = fixed / "collocation_val.pt"
    child_path = fixed / "child_shock_bank.pt"
    if all(path.is_file() for path in (train_path, validation_path, child_path)):
        train_bank = _load_parent_bank(train_path)
        validation_bank = _load_parent_bank(validation_path)
        child_payload = safe_torch_load(child_path, map_location="cpu")
        train_children = _load_child_bank(child_payload["train"])
        validation_children = _load_child_bank(child_payload["validation"])
        return train_bank, validation_bank, train_children, validation_children

    fixed.mkdir(parents=True, exist_ok=True)
    bootstrap = run_root / "bootstrap"
    bootstrap.mkdir(parents=True, exist_ok=True)
    if args.market_clearing_mode != "disabled":
        # A Firm-first cold start needs matched conditional transitions, not a
        # pre-solved long economic path.  One production step supplies parent
        # firms and both independently cleared children without moving any
        # trainable parameter before the Firm solve.
        bootstrap_horizon = 1
        train_initial = make_initial_state(
            models=models,
            episode=episode,
            paths=args.bootstrap_paths,
            firms=args.firms_per_path,
            horizon=bootstrap_horizon,
            branch_num=args.branch_num,
            seed=args.seed + 10,
            device=device,
        )
        validation_initial = make_initial_state(
            models=models,
            episode=episode,
            paths=args.bootstrap_validation_paths,
            firms=args.firms_per_path,
            horizon=bootstrap_horizon,
            branch_num=args.branch_num,
            seed=args.seed + 20,
            device=device,
        )
        train_output = run_simulation(
            models=models,
            episode=episode,
            initial_state=train_initial,
            paths=args.bootstrap_paths,
            firms=args.firms_per_path,
            horizon=bootstrap_horizon,
            branch_num=args.branch_num,
            transition_seed=args.seed + 11,
            entry_seed=args.seed + 12,
            device=device,
        )
        validation_output = run_simulation(
            models=models,
            episode=episode,
            initial_state=validation_initial,
            paths=args.bootstrap_validation_paths,
            firms=args.firms_per_path,
            horizon=bootstrap_horizon,
            branch_num=args.branch_num,
            transition_seed=args.seed + 21,
            entry_seed=args.seed + 22,
            device=device,
        )
        train_bank, train_children, train_audit = (
            build_market_cleared_transition_bank(
                train_output,
                n_parents=args.train_limit,
                seed=args.seed + 101,
                split="train",
                root_tolerance=args.market_clearing_residual_tolerance,
            )
        )
        validation_bank, validation_children, validation_audit = (
            build_market_cleared_transition_bank(
                validation_output,
                n_parents=args.validation_limit,
                seed=args.seed + 202,
                split="validation",
                root_tolerance=args.market_clearing_residual_tolerance,
            )
        )
        atomic_torch_save(train_path, train_bank.payload())
        atomic_torch_save(validation_path, validation_bank.payload())
        atomic_torch_save(
            child_path,
            {
                "train": train_children.payload(),
                "validation": validation_children.payload(),
            },
        )
        atomic_write_json(
            fixed / "metadata.json",
            {
                "source": "root_cleared_full_economy_matched_transitions",
                "market_clearing_mode": args.market_clearing_mode,
                "train": train_audit,
                "validation": validation_audit,
                "train_simulation": train_output.meta,
                "validation_simulation": validation_output.meta,
            },
        )
        atomic_write_json(
            bootstrap / "summary.json",
            {
                "support_source": "root_cleared_full_economy_matched_transitions",
                "train_paths": args.bootstrap_paths,
                "validation_paths": args.bootstrap_validation_paths,
                "firms_per_path": args.firms_per_path,
                "horizon": bootstrap_horizon,
                "requested_long_bootstrap_horizon": args.bootstrap_horizon,
                "long_rollout_used": False,
                "train_initial_state_hash": state_hash(train_initial),
                "validation_initial_state_hash": state_hash(validation_initial),
                "train": train_audit,
                "validation": validation_audit,
            },
        )
        return train_bank, validation_bank, train_children, validation_children
    bootstrap_horizon = 1
    initial = make_initial_state(
        models=models, episode=episode, paths=args.bootstrap_paths,
        firms=args.firms_per_path, horizon=bootstrap_horizon,
        branch_num=args.branch_num, seed=args.seed + 10, device=device,
    )
    output = run_simulation(
        models=models, episode=episode, initial_state=initial,
        paths=args.bootstrap_paths, firms=args.firms_per_path,
        horizon=bootstrap_horizon, branch_num=args.branch_num,
        transition_seed=args.seed + 11, entry_seed=args.seed + 12, device=device,
    )
    firm_frame, macro_frame = _to_frame(output.firm), _to_frame(output.macro)
    if firm_frame.empty or macro_frame.empty:
        raise RuntimeError("cold-start bootstrap simulation produced an empty support table")
    economic = AnalysisEconomicConfig.from_current_config()
    train_bank = build_collocation_bank(
        firm_frame, macro_frame, n_parents=args.train_limit, seed=args.seed + 101,
        split="train", economic_config=economic,
    )
    validation_bank = build_collocation_bank(
        firm_frame, macro_frame, n_parents=args.validation_limit,
        seed=args.seed + 202, split="validation", economic_config=economic,
    )
    models["sdf_fc1"].eval().requires_grad_(False)
    train_children = build_child_bank(
        models["sdf_fc1"], train_bank, n_child_shocks=args.child_shocks,
        seed=args.seed + 303, economic_config=economic, device=device,
    )
    validation_children = build_child_bank(
        models["sdf_fc1"], validation_bank, n_child_shocks=args.child_shocks,
        seed=args.seed + 404, economic_config=economic, device=device,
    )
    atomic_torch_save(train_path, train_bank.payload())
    atomic_torch_save(validation_path, validation_bank.payload())
    atomic_torch_save(child_path, {
        "train": train_children.payload(), "validation": validation_children.payload()
    })
    atomic_write_json(fixed / "metadata.json", {
        "train_parent_fingerprint": train_bank.fingerprint,
        "validation_parent_fingerprint": validation_bank.fingerprint,
        "train_child_fingerprint": train_children.fingerprint,
        "validation_child_fingerprint": validation_children.fingerprint,
        "train_seed": train_bank.seed,
        "validation_seed": validation_bank.seed,
        "child_train_seed": train_children.seed,
        "child_validation_seed": validation_children.seed,
        "source": "cold_start_production_simulation",
    })
    atomic_write_json(bootstrap / "summary.json", {
        "support_source": "cold_start_production_simulation",
        "paths": args.bootstrap_paths, "firms_per_path": args.firms_per_path,
        "horizon": bootstrap_horizon,
        "requested_long_bootstrap_horizon": args.bootstrap_horizon,
        "long_rollout_used": False,
        "firm_rows": len(firm_frame), "macro_rows": len(macro_frame),
        "initial_state_hash": state_hash(initial),
    })
    return train_bank, validation_bank, train_children, validation_children


def _build_or_load_macro_probe(
    *, args: argparse.Namespace, run_root: Path, episode: Episode,
    models: Mapping[str, torch.nn.Module],
    train_initial: Mapping[str, torch.Tensor],
    validation_initial: Mapping[str, torch.Tensor], device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build the fixed macro probe from production macro-pair tensors once."""
    path = run_root / "fixed_point" / "macro_probe_bank.pt"
    if path.is_file():
        probe = torch.load(path, map_location="cpu", weights_only=True)
    else:
        train_out = run_simulation(
            models=models,
            episode=episode,
            initial_state=train_initial,
            paths=args.long_diagnostic_paths,
            firms=args.firms_per_path,
            horizon=1,
            branch_num=args.branch_num,
            transition_seed=args.seed + 1011,
            entry_seed=args.seed + 1012,
            device=device,
        )
        validation_out = run_simulation(
            models=models,
            episode=episode,
            initial_state=validation_initial,
            paths=args.long_diagnostic_paths,
            firms=args.firms_per_path,
            horizon=1,
            branch_num=args.branch_num,
            transition_seed=args.seed + 2011,
            entry_seed=args.seed + 2012,
            device=device,
        )
        batches = build_in_memory_batches(
            episode, train_out.macro.to(device), validation_out.macro.to(device), args
        )
        parents, children = flatten_batches(batches["sdf_validation"])
        count = min(int(args.macro_probe_size), int(parents.shape[0]))
        if count < 1:
            raise RuntimeError("production macro-pair generator produced an empty probe bank")
        probe = {
            "parents": parents[:count].detach().cpu(),
            "children": children[:count].detach().cpu(),
        }
        atomic_torch_save(path, probe)
    parents, children = probe.get("parents"), probe.get("children")
    if not torch.is_tensor(parents) or not torch.is_tensor(children):
        raise RuntimeError("macro probe bank is malformed")
    if parents.ndim != 2 or parents.shape[-1] < 9:
        raise RuntimeError(f"macro probe parent shape is invalid: {tuple(parents.shape)}")
    if children.ndim != 3 or children.shape[0] != parents.shape[0]:
        raise RuntimeError(f"macro probe child shape is invalid: {tuple(children.shape)}")
    if not torch.isfinite(parents).all() or not torch.isfinite(children).all():
        raise RuntimeError("macro probe bank contains NaN/Inf")
    return {"parents": parents, "children": children}


def _refresh_children(
    *, models: Mapping[str, torch.nn.Module], episode: Episode,
    hp: Any, train_bank: InnerFPCollocationBank,
    validation_bank: InnerFPCollocationBank,
    train_children: InnerFPChildBank, validation_children: InnerFPChildBank,
    device: torch.device, preserve_child_macro: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    m_min = float(getattr(hp, "pv_m_clamp_min", 0.7))
    m_max = float(getattr(hp, "pv_m_clamp_max", 1.3))
    train, train_diag = refresh_transition_bank(
        models["sdf_fc1"], train_bank.parent_states.to(device),
        train_children.children.to(device), m_min=m_min, m_max=m_max,
        preserve_child_macro=preserve_child_macro,
    )
    validation, validation_diag = refresh_transition_bank(
        models["sdf_fc1"], validation_bank.parent_states.to(device),
        validation_children.children.to(device), m_min=m_min, m_max=m_max,
        preserve_child_macro=preserve_child_macro,
    )
    parity = audit_m_parity(episode, validation)
    return train, validation, {
        "train": train_diag, "validation": validation_diag, "parity": parity
    }


def _p_policy_outer(
    *, model: torch.nn.Module, episode: Episode,
    train_parents: torch.Tensor, train_children: torch.Tensor,
    validation_parents: torch.Tensor, validation_children: torch.Tensor,
    batch_size: int, args: argparse.Namespace, policy_outer: int,
) -> tuple[torch.nn.Module, dict[str, Any], pd.DataFrame]:
    train_problem = build_problem(
        episode, model, train_parents, train_children, batch_size,
        snapshot_batch_size=min(512, batch_size), require_legacy_m1=False,
    )
    validation_problem = build_problem(
        episode, model, validation_parents, validation_children, batch_size,
        snapshot_batch_size=min(512, batch_size), require_legacy_m1=False,
    )
    old_view = build_policy_view(model, validation_problem)
    target = resolve_policy_evaluation_target(policy_outer, P_EVAL_SCHEDULE)
    q_before = _state_hash(model, Q_PREFIXES)
    bp_before = _state_hash(model, BP_PREFIXES)
    solved, history, info = _solve_frozen_policy(
        model, episode, train_problem, validation_problem,
        adam_steps=args.p_adam_steps, adam_lr=args.p_adam_lr,
        eval_every=args.p_eval_every, lbfgs_max_iter=args.p_lbfgs_max_iter,
        lbfgs_history_size=50,
        lbfgs_chunk_iterations=args.p_lbfgs_chunk_iterations,
        target_residual=target,
    )
    if _state_hash(solved, Q_PREFIXES) != q_before:
        raise RuntimeError("P solver changed Q parameters")
    if _state_hash(solved, BP_PREFIXES) != bp_before:
        raise RuntimeError("P solver changed policy/BP parameters")
    fresh_problem = build_problem(
        episode, solved, validation_parents, validation_children, batch_size,
        snapshot_batch_size=min(512, batch_size), require_legacy_m1=False,
    )
    greedy = evaluate_problem(solved, fresh_problem)
    new_view = build_policy_view(solved, fresh_problem)
    movement = {
        **function_transition_metrics(old_view, new_view),
        **policy_transition_metrics(old_view, new_view),
    }
    row = {
        "p_policy_outer": policy_outer,
        "evaluation_target": target,
        "initial_frozen_residual": info["initial"]["normalized_mae"],
        "final_frozen_residual": info["final"]["normalized_mae"],
        "full_greedy_residual": greedy["normalized_mae"],
        "full_greedy_P0_residual": greedy["p0_norm_mae"],
        "full_greedy_PI_residual": greedy["pi_norm_mae"],
        "optimizer_stop_reason": info["stop_reason"],
        **movement,
    }
    require_finite_metrics(
        "P", row,
        ("full_greedy_P0_residual", "full_greedy_PI_residual"),
    )
    return solved, row, history


def _q_view(model: torch.nn.Module, parents: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model._q_claim_output(parents[:, :7]).detach()


def _q_picard_outer(
    *, model: torch.nn.Module, episode: Episode,
    train_parents: torch.Tensor, train_children: torch.Tensor,
    validation_parents: torch.Tensor, validation_children: torch.Tensor,
    batch_size: int, args: argparse.Namespace, picard_outer: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frozen_p = deepcopy(model).to(train_parents.device).eval().requires_grad_(False)
    train, train_regimes = build_q_survival_problem(
        episode=episode, initial_model=model, frozen_p_model=frozen_p,
        parents=train_parents, children=train_children, batch_size=batch_size,
        m_mode="current_sdf_child", include_all_regimes=True,
    )
    validation, validation_regimes = build_q_survival_problem(
        episode=episode, initial_model=model, frozen_p_model=frozen_p,
        parents=validation_parents, children=validation_children,
        batch_size=batch_size, m_mode="current_sdf_child", include_all_regimes=True,
    )
    fixed_train = refresh_q_targets(model, episode, train)
    fixed_validation = refresh_q_targets(model, episode, validation)
    non_q_before = _non_q_hash(model)
    old_q = _q_view(model, validation_parents)
    history = run_q_adam(
        model=model, episode=episode, train=fixed_train,
        validation=fixed_validation, steps=args.q_fit_steps, lr=args.q_lr,
        frozen_target=True, eval_every=max(1, min(args.q_fit_steps, 10)),
        phase="frozen_target_fit",
    )
    if _non_q_hash(model) != non_q_before:
        raise RuntimeError("Q Picard solver changed P or policy/BP parameters")
    fresh = evaluate_q_problem(model, episode, validation)
    frozen = evaluate_q_problem(model, episode, fixed_validation, frozen_target=True)
    new_q = _q_view(model, validation_parents)
    movement = (new_q - old_q).abs() / (1.0 + old_q.abs())
    row = {
        "q_picard_outer": picard_outer,
        "fresh_Q_residual": fresh["normalized_mae"],
        "frozen_fit_residual": frozen["normalized_mae"],
        "Q_mean": fresh["q_mean"], "Q_p90": fresh["q_p90"],
        "Q_max": fresh["q_max"], "Q_function_move_mean": float(movement.mean()),
        "Q_function_move_p90": _quantile(movement, 0.90),
        "train_regimes": train_regimes, "validation_regimes": validation_regimes,
        **{key: value for key, value in fresh.items() if "residual" in key or "recursion_gain" in key},
    }
    require_finite_metrics("Q", row, ("fresh_Q_residual", "Q_function_move_p90"))
    return row, history


def _firm_gate(row: Mapping[str, Any]) -> bool:
    return bool(
        float(row["P0_residual"]) <= 0.02
        and float(row["PI_residual"]) <= 0.02
        and float(row["Q_residual"]) <= 0.02
        and float(row["P0_move_p90"]) <= 0.01
        and float(row["PI_move_p90"]) <= 0.01
        and float(row["Q_move_p90"]) <= 0.01
        and float(row["investment_flip"]) <= 0.01
        and float(row["default_flip"]) <= 0.01
        and float(row["bp0_move_p90"]) <= 0.025
        and float(row["bpI_move_p90"]) <= 0.025
    )


def _bp_teacher_diagnostics(
    episode: Episode, model: torch.nn.Module,
    batches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    teacher = BPGridTeacher.from_hyperparams(
        model, episode.loss_fns["p0"], episode.loss_fns["pi"],
        episode.hyperparams, q_target_model=model,
    )
    rows: list[dict[str, np.ndarray]] = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            parent = batch["parent"][:, :7]
            children = batch["children"]
            raw_m, used_m = episode._build_policy_m_lists(
                batch["parent"], children,
                float(getattr(episode.hyperparams, "pv_m_clamp_min", 0.7)),
                float(getattr(episode.hyperparams, "pv_m_clamp_max", 1.3)),
            )
            children, _, used_m, weights = episode._expand_policy_expectation_children(
                children, raw_m, used_m
            )
            output = model(parent)
            scale = episode._pv_value_scale(model, parent).detach()
            for label, prediction in (("bp0", output.bp0), ("bpI", output.bpI)):
                result = teacher.compute(
                    parent_state=parent, children=children, m_list=used_m,
                    branch="p0" if label == "bp0" else "pi",
                    bp_pred=prediction, child_weights=weights,
                )
                active = parent[:, 2:3] > 0.5
                rows.append({
                    "label": np.repeat(label, int(active.sum().item())),
                    "prediction": prediction[active].cpu().numpy().reshape(-1),
                    "target": result["bp_star"][active].cpu().numpy().reshape(-1),
                    "regret": result["regret"][active].cpu().numpy().reshape(-1),
                    "value_star": result["value_star"][active].cpu().numpy().reshape(-1),
                    "value_scale": scale[active].cpu().numpy().reshape(-1),
                    "margin": result["top2_margin"][active].cpu().numpy().reshape(-1),
                })
    output: dict[str, Any] = {}
    for label in ("bp0", "bpI"):
        selected = [row for row in rows if row["label"].size and row["label"][0] == label]
        if not selected:
            continue
        pred = np.concatenate([row["prediction"] for row in selected])
        target = np.concatenate([row["target"] for row in selected])
        regret = np.concatenate([row["regret"] for row in selected])
        value_star = np.concatenate([row["value_star"] for row in selected])
        value_scale = np.concatenate([row["value_scale"] for row in selected])
        margin = np.concatenate([row["margin"] for row in selected])
        relative, scaled = normalized_regrets(regret, value_star, value_scale)
        gap = np.abs(pred - target)
        weak = margin / (np.abs(value_star) + 1e-12) < 1e-3
        large = gap > 0.05
        output[label] = {
            "n": int(gap.size), "mae": float(gap.mean()),
            "signed_bias": float((pred - target).mean()),
            "gap_p90": float(np.quantile(gap, 0.90)),
            "regret_p50": float(np.quantile(regret, 0.50)),
            "regret_p90": float(np.quantile(regret, 0.90)),
            "regret_p99": float(np.quantile(regret, 0.99)),
            "regret_max": float(np.max(regret)),
            "relative_regret_p90": float(np.nanquantile(relative, 0.90)),
            "scale_normalized_regret_p90": float(np.nanquantile(scaled, 0.90)),
            "weak_margin_share": float(weak.mean()),
            "large_action_error_share": float(large.mean()),
            "large_action_error_regret_p90": (
                float(np.quantile(regret[large], 0.90)) if large.any() else 0.0
            ),
        }
    return output


def _distill_policy_heads(
    *, episode: Episode, model: torch.nn.Module,
    train_bank: InnerFPCollocationBank, validation_bank: InnerFPCollocationBank,
    train_children: torch.Tensor, validation_children: torch.Tensor,
    batch_size: int, args: argparse.Namespace, output: Path, device: torch.device,
) -> dict[str, Any]:
    train_child_bank = InnerFPChildBank(
        children=train_children.detach().cpu(),
        eps_x=torch.zeros_like(train_children[..., :1].cpu()),
        eps_z=torch.zeros_like(train_children[..., :1].cpu()),
        u_eta=torch.zeros_like(train_children[..., :1].cpu()),
        u_i=torch.zeros_like(train_children[..., :1].cpu()), seed=-1,
    )
    validation_child_bank = InnerFPChildBank(
        children=validation_children.detach().cpu(),
        eps_x=torch.zeros_like(validation_children[..., :1].cpu()),
        eps_z=torch.zeros_like(validation_children[..., :1].cpu()),
        u_eta=torch.zeros_like(validation_children[..., :1].cpu()),
        u_i=torch.zeros_like(validation_children[..., :1].cpu()), seed=-1,
    )
    train_batches = make_batches(train_bank, train_child_bank, batch_size=batch_size, device=device)
    validation_batches = make_batches(
        validation_bank, validation_child_bank, batch_size=batch_size, device=device
    )
    p_before = _state_hash(model, P_PREFIXES)
    q_before = _state_hash(model, Q_PREFIXES)
    teacher_snapshot = deepcopy(model).to(device).eval().requires_grad_(False)
    train_cache = episode._build_bp_target_cache(train_batches, teacher_snapshot)
    validation_cache = episode._build_bp_target_cache(validation_batches, teacher_snapshot)
    summary = episode._run_bp_distillation_stage(
        train_cache, validation_cache, teacher_snapshot,
        n_epochs=args.policy_distill_max_epochs,
    )
    if _state_hash(model, P_PREFIXES) != p_before or _state_hash(model, Q_PREFIXES) != q_before:
        raise RuntimeError("policy-head distillation changed P or Q parameters")
    summary["diagnostics"] = _bp_teacher_diagnostics(episode, model, validation_batches)
    atomic_write_json(output / "policy_head_summary.json", summary)
    return summary


def _path_and_target_movements(old: pd.DataFrame, new: pd.DataFrame) -> dict[str, float]:
    aligned = aligned_macro_vectors(old, new, ("Hatc", "LnK", "C", "K"))
    result: dict[str, float] = {}
    for name, (left, right) in aligned.items():
        metrics = movement_metrics(left, right, name.lower())
        result[f"{name}_path_move"] = metrics[f"{name.lower()}_nmae"]
        if name in {"Hatc", "LnK"}:
            result[f"{name}_target_move"] = metrics[f"{name.lower()}_nmae"]
    return result


def _pairs_with_source_keys(pairs: Any, summary: pd.DataFrame) -> pd.DataFrame:
    frame = _to_frame(pairs).copy()
    paths = sorted(map(int, frame["path"].unique()))
    keys = list(map(str, summary["source_key"].tolist()))
    if len(paths) != len(keys):
        raise RuntimeError(
            "one-step pair/source-key cardinality mismatch: "
            f"paths={len(paths)}, source_keys={len(keys)}"
        )
    frame["source_key"] = frame["path"].map(dict(zip(paths, keys)))
    if frame["source_key"].isna().any():
        raise RuntimeError("one-step pair rows could not be mapped to source keys")
    return frame


def _one_step_target_movements(
    old: Any,
    new: Any,
    old_summary: pd.DataFrame,
    new_summary: pd.DataFrame,
) -> dict[str, Any]:
    left = _pairs_with_source_keys(old, old_summary)
    right = _pairs_with_source_keys(new, new_summary)
    identity = ["source_key", "t", "branch"]
    aligned = left.merge(
        right,
        on=identity,
        how="inner",
        suffixes=("_old", "_new"),
        validate="one_to_one",
    ).sort_values(identity).reset_index(drop=True)
    if aligned.empty:
        raise RuntimeError("one-step target movement has no common source keys")
    common_sources = int(aligned["source_key"].nunique())
    result: dict[str, Any] = {
        "old_source_count": int(len(old_summary)),
        "new_source_count": int(len(new_summary)),
        "common_source_count": common_sources,
        "common_pair_row_count": int(len(aligned)),
        "movement_alignment": "COMMON_FIXED_SOURCE_KEYS_ONLY",
    }
    for column, label in (("Hatc_t1", "Hatc"), ("LnK_t1", "LnK")):
        metrics = movement_metrics(
            torch.as_tensor(aligned[f"{column}_old"].to_numpy()),
            torch.as_tensor(aligned[f"{column}_new"].to_numpy()),
            label.lower(),
        )
        result[f"{label}_target_move"] = metrics[f"{label.lower()}_nmae"]
    return result


def _candidate_target_movements(old: Any, new: Any) -> dict[str, Any]:
    """Align one-step targets on immutable candidate/branch identity."""

    def canonical_frame(table: Any) -> pd.DataFrame:
        frame = _to_frame(table)
        metadata = dict(getattr(table, "metadata", {}))
        if metadata.get("identity_schema") != "canonical_candidate_to_local_path_v1":
            raise RuntimeError(
                "candidate target movement requires canonical/local identity metadata"
            )
        local_to_canonical = {
            int(local): int(canonical)
            for local, canonical in metadata.get("local_to_canonical", {}).items()
        }
        frame["candidate_id"] = (
            frame["path"].round().astype(np.int64).map(local_to_canonical)
        )
        if frame["candidate_id"].isna().any():
            raise RuntimeError("candidate target movement found an unmapped local id")
        frame["candidate_id"] = frame["candidate_id"].astype(np.int64)
        return frame

    left = canonical_frame(old)
    right = canonical_frame(new)
    identity = ["candidate_id", "t", "branch"]
    aligned = left.merge(
        right,
        on=identity,
        how="inner",
        suffixes=("_old", "_new"),
        validate="one_to_one",
    ).sort_values(identity).reset_index(drop=True)
    if aligned.empty:
        raise RuntimeError("candidate target movement has no common identities")
    result: dict[str, Any] = {
        "old_pair_count": int(len(left)),
        "new_pair_count": int(len(right)),
        "common_pair_count": int(len(aligned)),
        "common_candidate_count": int(aligned["candidate_id"].nunique()),
        "movement_alignment": "IMMUTABLE_CANDIDATE_BRANCH_IDENTITY",
    }
    for column, label in (("Hatc_t1", "Hatc"), ("LnK_t1", "LnK")):
        metrics = movement_metrics(
            torch.as_tensor(aligned[f"{column}_old"].to_numpy()),
            torch.as_tensor(aligned[f"{column}_new"].to_numpy()),
            label.lower(),
        )
        result[f"{label}_target_move"] = metrics[f"{label.lower()}_nmae"]
    return result


def _macro_gate(row: Mapping[str, Any]) -> bool:
    return bool(
        float(row["M_function_move"]) < 0.005
        and float(row["FC1_C_function_move"]) < 0.005
        and float(row["FC1_K_function_move"]) < 0.005
        and float(row["Hatc_target_move"]) < 0.01
        and float(row["LnK_target_move"]) < 0.01
    )


def _overall_gate(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("firm_conditional_fixed_point_reached", False)
        and row.get("macro_conditional_fixed_point_reached", False)
        and float(row.get("P0_move_p90", math.inf)) < 0.01
        and float(row.get("PI_move_p90", math.inf)) < 0.01
        and float(row.get("Q_move_p90", math.inf)) < 0.01
        and float(row.get("investment_flip", math.inf)) < 0.01
        and float(row.get("default_flip", math.inf)) < 0.01
        and float(row.get("M_function_move", math.inf)) < 0.01
        and float(row.get("FC1_C_function_move", math.inf)) < 0.01
        and float(row.get("FC1_K_function_move", math.inf)) < 0.01
    )


def _simulation_pair(
    *, args: argparse.Namespace, episode: Episode,
    models: Mapping[str, torch.nn.Module], train_initial: Mapping[str, torch.Tensor],
    validation_initial: Mapping[str, torch.Tensor], device: torch.device,
) -> tuple[Any, Any, float]:
    started = time.perf_counter()
    train_out = run_simulation(
        models=models, episode=episode, initial_state=train_initial,
        paths=args.long_diagnostic_paths, firms=args.firms_per_path,
        horizon=args.long_diagnostic_horizon, branch_num=args.branch_num,
        transition_seed=args.seed + 1011, entry_seed=args.seed + 1012,
        device=device,
    )
    validation_out = run_simulation(
        models=models, episode=episode, initial_state=validation_initial,
        paths=args.long_diagnostic_paths, firms=args.firms_per_path,
        horizon=args.long_diagnostic_horizon, branch_num=args.branch_num,
        transition_seed=args.seed + 2011, entry_seed=args.seed + 2012,
        device=device,
    )
    return train_out, validation_out, time.perf_counter() - started


def _write_simulation_metadata(
    path: Path, *, args: argparse.Namespace, train_out: Any, validation_out: Any,
    train_initial: Mapping[str, torch.Tensor],
    validation_initial: Mapping[str, torch.Tensor], label: str,
) -> None:
    atomic_write_json(path, {
        "label": label,
        "train_macro_hash": table_hash(train_out.macro),
        "train_firm_hash": table_hash(train_out.firm),
        "validation_macro_hash": table_hash(validation_out.macro),
        "validation_firm_hash": table_hash(validation_out.firm),
        "train_macro_rows": len(train_out.macro),
        "train_firm_rows": len(train_out.firm),
        "validation_macro_rows": len(validation_out.macro),
        "validation_firm_rows": len(validation_out.firm),
        "train_initial_hash": state_hash(train_initial),
        "validation_initial_hash": state_hash(validation_initial),
        "train_transition_seed": args.seed + 1011,
        "train_entry_seed": args.seed + 1012,
        "validation_transition_seed": args.seed + 2011,
        "validation_entry_seed": args.seed + 2012,
        "paths": args.long_diagnostic_paths,
        "validation_paths": args.long_diagnostic_paths,
        "firms_per_path": args.firms_per_path,
        "horizon": args.long_diagnostic_horizon,
        "branch_num": args.branch_num,
        "purpose": "LONG_HORIZON_DIAGNOSTIC_ONLY",
        "used_for_primary_macro_training": False,
    })


def _write_long_diagnostic_artifacts(
    output: Path,
    *,
    train_out: Any,
    validation_out: Any,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    keep_macro = [
        "path", "t", "branch", "Hatc", "LnK", "C", "K", "n_firms",
        "M", "x", "hatcf", "lnkf",
        "Hatc_belief", "Hatc_cleared", "Hatc_firm_input",
        "LnK_belief", "LnK_calculated", "C_raw",
        "market_root_found", "market_root_count",
        "market_root_residual", "market_root_residual_normalized",
        "market_root_bracket_count", "market_root_refinement_rounds",
        "market_root_branch_id", "market_root_branch_switch",
        "market_root_policy_jump",
        "market_investment_flip_vs_belief",
        "market_default_flip_vs_belief",
        "market_entry_flip_vs_belief",
        "market_root_operator_evaluations",
        "market_committed_residual",
        "market_committed_residual_normalized",
    ]
    for split, simulation in (("train", train_out), ("validation", validation_out)):
        macro = _to_frame(simulation.macro)
        firm = _to_frame(simulation.firm)
        atomic_write_csv(
            output / f"{split}_macro_paths.csv",
            macro.loc[:, [name for name in keep_macro if name in macro]],
        )
        diagnostics = macro_diagnostics(macro)
        diagnostics.update(simulation_policy_metrics(firm))
        parent_firm = firm[firm["branch"] == -1].copy()
        if parent_firm.empty:
            parent_firm = firm.copy()
        firm_path_summary = (
            parent_firm.groupby(["path", "t", "branch"], as_index=False)
            .agg(
                investment_rate=("Bar_i_executed", "mean"),
                default_rate=("P", lambda value: float((value <= 0.0).mean())),
                mean_b=("b", "mean"),
                mean_z=("z", "mean"),
                firm_rows=("ID", "size"),
            )
        )
        atomic_write_csv(
            output / f"{split}_firm_path_summary.csv", firm_path_summary
        )
        if (
            "market_root_found" in macro
            and float(macro["market_root_found"].sum()) > 0.0
        ):
            residual = pd.to_numeric(
                macro["market_committed_residual_normalized"], errors="coerce"
            ).abs()
            diagnostics.update({
                "market_total_nodes": int(len(macro)),
                "market_root_success_rate": float(macro["market_root_found"].mean()),
                "market_multiple_root_rate": float((macro["market_root_count"] > 1).mean()),
                "market_root_residual_p50": float(residual.quantile(0.50)),
                "market_root_residual_p90": float(residual.quantile(0.90)),
                "market_root_residual_p99": float(residual.quantile(0.99)),
                "market_mean_abs_belief_minus_root": float(
                    (macro["Hatc_belief"] - macro["Hatc_cleared"]).abs().mean()
                ),
                "market_root_branch_switch_rate": float(
                    macro["market_root_branch_switch"].mean()
                ),
                "market_policy_jump_rate": float(
                    macro["market_root_policy_jump"].mean()
                ),
                "market_root_operator_evaluations": float(
                    macro["market_root_operator_evaluations"].sum()
                ),
            })
        parent = macro[macro["branch"] == -1].copy()
        if parent.empty:
            raise RuntimeError(f"long diagnostic {split} has no parent rows")
        for forecast, realized, label in (
            ("hatcf", "Hatc", "recursive_fc1_hatc"),
            ("lnkf", "LnK", "recursive_fc1_lnk"),
        ):
            diagnostics[f"{label}_nmae"] = movement_metrics(
                torch.as_tensor(parent[forecast].to_numpy()),
                torch.as_tensor(parent[realized].to_numpy()),
                label,
            )[f"{label}_nmae"]
        tail_start = int(math.floor(0.8 * float(parent["t"].max())))
        tail = parent[parent["t"] >= tail_start]
        diagnostics.update({
            "tail_start_t": tail_start,
            "tail_M_mean": float(tail["M"].mean()),
            "tail_M_std": float(tail["M"].std(ddof=0)),
            "tail_Hatc_mean": float(tail["Hatc"].mean()),
            "tail_LnK_mean": float(tail["LnK"].mean()),
            "tail_C_mean": float(tail["C"].mean()),
            "tail_K_mean": float(tail["K"].mean()),
            "tail_mean_firms": float(tail["n_firms"].mean()),
            "stationary_mean_b": float(
                firm[firm["t"] >= tail_start]["b"].mean()
            ),
            "stationary_mean_z": float(
                firm[firm["t"] >= tail_start]["z"].mean()
            ),
        })
        summaries[split] = diagnostics
    atomic_write_json(output / "summary.json", {
        "purpose": "LONG_HORIZON_DIAGNOSTIC_ONLY",
        "used_for_primary_macro_training": False,
        "splits": summaries,
    })
    return summaries


def _matched_legacy_market_comparison(
    *,
    output: Path,
    label: str,
    args: argparse.Namespace,
    episode: Episode,
    models: Mapping[str, torch.nn.Module],
    initial_state: Mapping[str, torch.Tensor],
    market_output: Any,
    paths: int,
    transition_seed: int,
    entry_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Replay one locked CRN sample with legacy direct aggregation.

    A legacy resource-feasibility failure is evidence, not a reason to alter
    consumption or shocks.  It is therefore captured verbatim and does not get
    converted into a filtered comparison sample.
    """
    output.mkdir(parents=True, exist_ok=True)
    market_macro = _to_frame(market_output.macro)
    market_firm = _to_frame(market_output.firm)
    atomic_write_csv(output / "market_macro.csv", market_macro)
    atomic_write_csv(output / "market_firm.csv", market_firm)
    result: dict[str, Any] = {
        "label": label,
        "initial_state_hash": state_hash(initial_state),
        "transition_seed": int(transition_seed),
        "entry_seed": int(entry_seed),
        "market_macro_hash": table_hash(market_output.macro),
        "market_firm_hash": table_hash(market_output.firm),
        "market_mode": args.market_clearing_mode,
        "legacy_mode": "disabled",
        "same_initial_state_and_crn": True,
        "economic_equations_changed": False,
    }
    original_mode = str(episode.hyperparams.market_clearing_mode)
    try:
        episode.hyperparams.market_clearing_mode = "disabled"
        legacy_output = run_simulation(
            models=models,
            episode=episode,
            initial_state=initial_state,
            paths=paths,
            firms=args.firms_per_path,
            horizon=args.long_diagnostic_horizon,
            branch_num=args.branch_num,
            transition_seed=transition_seed,
            entry_seed=entry_seed,
            device=device,
        )
    except Exception as error:  # strict evidence capture; never censor/reseed
        result.update({
            "status": "LEGACY_DIRECT_AGGREGATION_FAILED",
            "legacy_error_type": type(error).__name__,
            "legacy_error": str(error),
            "legacy_traceback": traceback.format_exc(),
        })
        atomic_write_json(output / "summary.json", result)
        return result
    finally:
        episode.hyperparams.market_clearing_mode = original_mode

    legacy_macro = _to_frame(legacy_output.macro)
    legacy_firm = _to_frame(legacy_output.firm)
    atomic_write_csv(output / "legacy_macro.csv", legacy_macro)
    atomic_write_csv(output / "legacy_firm.csv", legacy_firm)
    macro_keys = ["path", "t", "branch"]
    macro_fields = [
        "C_raw", "C", "K", "Hatc", "LnK", "M", "n_firms",
        "accepted_count", "same_node_entrant_exit_count",
    ]
    macro_fields = [
        name for name in macro_fields
        if name in market_macro and name in legacy_macro
    ]
    matched_macro = legacy_macro[macro_keys + macro_fields].merge(
        market_macro[macro_keys + macro_fields],
        on=macro_keys,
        how="inner",
        suffixes=("_legacy", "_market"),
        validate="one_to_one",
    )
    atomic_write_csv(output / "matched_macro.csv", matched_macro)
    metrics: dict[str, Any] = {"matched_macro_rows": int(len(matched_macro))}
    for name in macro_fields:
        old = matched_macro[f"{name}_legacy"].to_numpy(dtype=float)
        new = matched_macro[f"{name}_market"].to_numpy(dtype=float)
        finite = np.isfinite(old) & np.isfinite(new)
        if finite.any():
            metrics[f"{name}_legacy_mean"] = float(old[finite].mean())
            metrics[f"{name}_market_mean"] = float(new[finite].mean())
            metrics[f"{name}_market_minus_legacy_mae"] = float(
                np.abs(new[finite] - old[finite]).mean()
            )
    firm_keys = ["path", "t", "branch", "ID"]
    firm_fields = [
        name for name in ("Bar_i_executed", "Bar_z", "P", "BP")
        if name in market_firm and name in legacy_firm
    ]
    matched_firm = legacy_firm[firm_keys + firm_fields].merge(
        market_firm[firm_keys + firm_fields],
        on=firm_keys,
        how="inner",
        suffixes=("_legacy", "_market"),
        validate="one_to_one",
    )
    atomic_write_csv(output / "matched_firm.csv", matched_firm)
    metrics["matched_firm_rows"] = int(len(matched_firm))
    for name in firm_fields:
        old = matched_firm[f"{name}_legacy"].to_numpy(dtype=float)
        new = matched_firm[f"{name}_market"].to_numpy(dtype=float)
        finite = np.isfinite(old) & np.isfinite(new)
        if finite.any():
            metrics[f"{name}_market_minus_legacy_mae"] = float(
                np.abs(new[finite] - old[finite]).mean()
            )
    result.update({
        "status": "PASS",
        "legacy_macro_hash": table_hash(legacy_output.macro),
        "legacy_firm_hash": table_hash(legacy_output.firm),
        "metrics": metrics,
    })
    atomic_write_json(output / "summary.json", result)
    return result


def _write_metadata(
    *, run_root: Path, args: argparse.Namespace, hp: Any,
    models: Mapping[str, torch.nn.Module], git_head: str,
    fingerprint: str, device: torch.device, bank_hashes: Mapping[str, str],
) -> None:
    initial_checkpoint = None
    if args.initial_checkpoint is not None:
        checkpoint_path = args.initial_checkpoint.expanduser().resolve()
        digest = sha256()
        with checkpoint_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        initial_checkpoint = {
            "path": str(checkpoint_path),
            "sha256": digest.hexdigest(),
        }
    cuda = {
        "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "gpu_total_memory": (
            torch.cuda.get_device_properties(device).total_memory
            if device.type == "cuda" else 0
        ),
    }
    atomic_write_json(run_root / "metadata" / "run.json", {
        "initialization_mode": (
            "WARM_START_CHECKPOINT" if initial_checkpoint is not None else "COLD_START"
        ),
        "initial_checkpoint": initial_checkpoint,
        "branch": _git_branch(), "git_head": git_head,
        "hyperparameter_fingerprint": fingerprint,
        "args": vars(args), "hyperparams": vars(hp),
        "economic_config": AnalysisEconomicConfig.from_current_config().to_dict(),
        "bank_hashes": bank_hashes, "cuda": cuda,
        "fc2": "COMPLETELY_DISABLED",
        "sdf_mode": "SDF_TRUE_ONLY", "sdf_recursive_epochs": 0,
        "macro_primary_training_source": "MANY_PARENT_CROSS_SECTIONS_ONE_STEP",
        "macro_parent_generation": "PATH_SPECIFIC_BETA_ONE_STEP_BOUNDARY_AWARE",
        "macro_candidate_budgets": {
            "train": {
                "target": args.macro_train_target_valid,
                "initial": args.macro_train_initial_candidate_budget,
                "maximum": args.macro_train_max_candidate_budget,
            },
            "validation": {
                "target": args.macro_validation_target_valid,
                "initial": args.macro_validation_initial_candidate_budget,
                "maximum": args.macro_validation_max_candidate_budget,
            },
        },
        "macro_one_step_horizon": 1,
        "long_rollout_training_used": False,
        "long_rollout_diagnostics_enabled": bool(args.run_long_diagnostics),
        "long_rollout_horizon": int(args.long_diagnostic_horizon),
        "m_semantics": "CURRENT_SDF_CHILD shared by P and Q; one common clamp",
        "model_hashes_at_initialization": {
            name: module_hash(model) for name, model in models.items()
        },
    })


def _candidate_schedule_overlap(
    train: MacroCandidateSchedule,
    validation: MacroCandidateSchedule,
) -> dict[str, Any]:
    train_ids = {spec.candidate_id for spec in train.candidate_specs}
    validation_ids = {spec.candidate_id for spec in validation.candidate_specs}
    overlap = sorted(train_ids.intersection(validation_ids))
    audit = {
        "train_namespace": train.namespace,
        "validation_namespace": validation.namespace,
        "train_candidate_count": len(train_ids),
        "validation_candidate_count": len(validation_ids),
        "overlap_count": len(overlap),
        "overlap_candidate_ids": overlap[:20],
        "disjoint": not overlap,
    }
    if overlap:
        raise RuntimeError(f"macro train/validation candidate leakage: {audit}")
    return audit


def _load_committed_macro_inner_inputs(
    *,
    macro_root: Path,
    macro_inner: int,
    episode: Episode,
    macro_batch_size: int,
    branch_num: int,
    device: torch.device,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load exactly the previous committed Macro-inner output datasets."""

    if int(macro_inner) <= 0:
        raise ValueError("previous committed Macro inputs require macro_inner > 0")
    previous_inner_root = macro_root / f"macro_inner_{int(macro_inner) - 1:03d}"
    manifest_path = previous_inner_root / "committed_dataset_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            "resume requires the previous Macro inner committed dataset "
            f"manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = load_committed_one_step_macro_dataset(
        previous_inner_root / "post_fc1_train" / "committed_dataset.pt",
        episode=episode,
        macro_batch_size=macro_batch_size,
        n_branches=branch_num,
        device=device,
    )
    validation = load_committed_one_step_macro_dataset(
        previous_inner_root / "post_fc1_validation" / "committed_dataset.pt",
        episode=episode,
        macro_batch_size=macro_batch_size,
        n_branches=branch_num,
        device=device,
    )
    observed = {
        "train": one_step_macro_dataset_fingerprint(train),
        "validation": one_step_macro_dataset_fingerprint(validation),
    }
    if observed != manifest.get("output_dataset_fingerprints"):
        raise RuntimeError(
            "previous Macro inner committed dataset fingerprint mismatch: "
            f"{observed} != {manifest.get('output_dataset_fingerprints')}"
        )
    if manifest.get("committed_stage") != "macro_inner_complete":
        raise RuntimeError("previous Macro dataset was not committed")
    return train, validation, manifest


def _build_or_refresh_macro_candidate_context(
    *,
    args: argparse.Namespace,
    run_root: Path,
    outer_root: Path,
    equilibrium_outer: int,
) -> tuple[MacroCandidateSchedule, MacroCandidateSchedule, dict[str, Any]]:
    """Create immutable train/validation candidate schedules before simulation.

    The train schedule is outer-specific.  Validation parameters, shocks and
    candidate identities live under a global run directory and are reused by
    every outer.  Neither schedule depends on Firm/FC1/SDF outputs or candidate
    success, which makes replenishment and resume deterministic.
    """

    candidate_root = outer_root / "macro_candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)
    train_path = candidate_root / "train_candidate_schedule.json"
    if train_path.is_file():
        train = load_candidate_schedule(train_path)
    else:
        train = build_macro_candidate_schedule(
            split=f"train_outer_{equilibrium_outer:03d}",
            namespace=f"train_outer_{equilibrium_outer:03d}",
            target_valid=args.macro_train_target_valid,
            initial_candidate_budget=args.macro_train_initial_candidate_budget,
            max_candidate_budget=args.macro_train_max_candidate_budget,
            minimum_partial_valid=args.macro_minimum_partial_train,
            firms_per_parent=args.firms_per_path,
            seed=int(args.seed + 30_000 + equilibrium_outer * 100_000),
            candidate_id_offset=int(equilibrium_outer * 10_000_000),
        )
        save_candidate_schedule(train_path, train)

    validation_root = run_root / "fixed_point" / "macro_global_validation"
    validation_root.mkdir(parents=True, exist_ok=True)
    validation_path = validation_root / "validation_candidate_schedule.json"
    if validation_path.is_file():
        validation = load_candidate_schedule(validation_path)
    else:
        validation = build_macro_candidate_schedule(
            split="global_validation",
            namespace="global_validation",
            target_valid=args.macro_validation_target_valid,
            initial_candidate_budget=args.macro_validation_initial_candidate_budget,
            max_candidate_budget=args.macro_validation_max_candidate_budget,
            minimum_partial_valid=args.macro_minimum_partial_validation,
            firms_per_parent=args.firms_per_path,
            seed=int(args.seed + 40_000),
            candidate_id_offset=1_000_000_000,
        )
        save_candidate_schedule(validation_path, validation)

    outer_validation_path = candidate_root / "validation_candidate_schedule.json"
    if outer_validation_path.is_file():
        outer_validation = load_candidate_schedule(outer_validation_path)
        if outer_validation.fingerprint != validation.fingerprint:
            raise RuntimeError("outer/global validation candidate schedule mismatch")
    else:
        save_candidate_schedule(outer_validation_path, validation)
    split_audit = _candidate_schedule_overlap(train, validation)
    atomic_write_json(candidate_root / "train_validation_split_audit.json", split_audit)
    return train, validation, split_audit


FIRM_FIRST_STAGE_ORDER = (
    "firm_solve",
    "macro_candidate_generation",
    "fc1_fit",
    "post_fc1_refresh",
    "sdf_fit",
    "outer_evaluation",
)


def validate_firm_first_stage_trace(trace: Sequence[str], outer_count: int) -> None:
    expected = list(FIRM_FIRST_STAGE_ORDER) * int(outer_count)
    if list(trace) != expected:
        raise RuntimeError(
            f"Firm-first orchestration order mismatch: expected={expected}, actual={list(trace)}"
        )


def run_firm_first_orchestration_mock(
    *, outer_count: int, resume_after: int = 0
) -> dict[str, Any]:
    """Exercise the production stage contract without models or simulation."""

    full = list(FIRM_FIRST_STAGE_ORDER) * int(outer_count)
    if not 0 <= int(resume_after) <= len(full):
        raise ValueError("resume_after is outside the mock stage trace")
    persisted = full[: int(resume_after)]
    executed = full[int(resume_after) :]
    trace = persisted + executed
    validate_firm_first_stage_trace(trace, outer_count)
    return {
        "outer_count": int(outer_count),
        "resume_after": int(resume_after),
        "persisted_trace": persisted,
        "executed_trace": executed,
        "full_trace": trace,
        "status": "PASS",
    }


def _record_firm_first_stage(
    run_root: Path, *, equilibrium_outer: int, stage: str
) -> None:
    if stage not in FIRM_FIRST_STAGE_ORDER:
        raise ValueError(f"unknown Firm-first stage: {stage}")
    path = run_root / "metadata" / "firm_first_stage_trace.json"
    payload = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {"entries": []}
    )
    entries = list(payload.get("entries", []))
    identity = {"equilibrium_outer": int(equilibrium_outer), "stage": stage}
    if identity in entries:
        return
    expected_index = len(entries)
    expected_outer = expected_index // len(FIRM_FIRST_STAGE_ORDER)
    expected_stage = FIRM_FIRST_STAGE_ORDER[
        expected_index % len(FIRM_FIRST_STAGE_ORDER)
    ]
    if int(equilibrium_outer) != expected_outer or stage != expected_stage:
        raise RuntimeError(
            "Firm-first stage transition violation: "
            f"expected outer={expected_outer} stage={expected_stage}, got {identity}"
        )
    entries.append(identity)
    atomic_write_json(
        path,
        {
            "stage_order": list(FIRM_FIRST_STAGE_ORDER),
            "entries": entries,
            "completed_outer_count": len(entries) // len(FIRM_FIRST_STAGE_ORDER),
        },
    )


def _apply_smoke(args: argparse.Namespace) -> None:
    args.equilibrium_outer_max = 1
    args.firm_inner_max = 1
    args.p_policy_outer_max = 1
    args.q_picard_outer_max = 1
    args.macro_inner_max = 1
    args.simulation_paths = 4
    args.validation_paths = 4
    args.bootstrap_paths = 4
    args.bootstrap_validation_paths = 4
    args.firms_per_path = 32
    args.simulation_horizon = 8
    args.bootstrap_horizon = 8
    args.macro_train_target_valid = 8
    args.macro_validation_target_valid = 4
    args.macro_train_initial_candidate_budget = 16
    args.macro_train_max_candidate_budget = 32
    args.macro_validation_initial_candidate_budget = 8
    args.macro_validation_max_candidate_budget = 16
    args.macro_minimum_partial_train = 4
    args.macro_minimum_partial_validation = 2
    args.macro_parent_batch_size = 4
    args.long_diagnostic_paths = 4
    args.long_diagnostic_horizon = 8
    args.train_limit = 64
    args.validation_limit = 32
    args.child_shocks = 2
    args.macro_probe_size = 32
    args.p_batch_size = 32
    args.q_batch_size = 32
    args.macro_batch_size = 32
    args.p_adam_steps = 2
    args.p_lbfgs_max_iter = 2
    args.p_lbfgs_chunk_iterations = 1
    args.q_fit_steps = 2
    args.fc1_production_epochs = 1
    args.fc1_max_epochs = 1
    args.fc1_patience = 1
    args.sdf_chunks = 1
    args.sdf_epochs_per_chunk = 1
    args.policy_distill_max_epochs = 1
    args.policy_distill_max_steps = 2


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cold-start", action="store_true")
    mode.add_argument("--resume", type=Path)
    parser.add_argument("--allow-code-mismatch", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--equilibrium-outer-max", type=int, default=10)
    parser.add_argument("--firm-inner-max", type=int, default=8)
    parser.add_argument("--p-batch-size", type=int, default=1024)
    parser.add_argument("--p-policy-outer-max", type=int, default=8)
    parser.add_argument("--p-adam-steps", type=int, default=100)
    parser.add_argument("--p-adam-lr", type=float, default=1e-4)
    parser.add_argument("--p-eval-every", type=int, default=10)
    parser.add_argument("--p-lbfgs-max-iter", type=int, default=100)
    parser.add_argument("--p-lbfgs-chunk-iterations", type=int, default=20)
    parser.add_argument("--q-solver", choices=("picard",), default="picard")
    parser.add_argument("--q-batch-size", type=int, default=2048)
    parser.add_argument("--q-picard-outer-max", type=int, default=8)
    parser.add_argument("--q-fit-steps", type=int, default=100)
    parser.add_argument("--q-lr", type=float, default=1e-4)
    parser.add_argument("--macro-inner-max", type=int, default=20)
    parser.add_argument("--simulation-paths", type=int, default=400, help="deprecated; not used for primary Macro training")
    parser.add_argument("--validation-paths", type=int, default=400, help="deprecated; not used for primary Macro training")
    parser.add_argument("--bootstrap-paths", type=int, default=64)
    parser.add_argument("--bootstrap-validation-paths", type=int, default=32)
    parser.add_argument("--firms-per-path", type=int, default=200)
    parser.add_argument("--simulation-horizon", type=int, default=100, help="deprecated; not used for primary Macro training")
    parser.add_argument("--bootstrap-horizon", type=int, default=100)
    parser.add_argument("--branch-num", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=8192)
    parser.add_argument("--validation-limit", type=int, default=2048)
    parser.add_argument("--child-shocks", type=int, default=16)
    parser.add_argument("--macro-probe-size", type=int, default=2048)
    parser.add_argument("--macro-batch-size", type=int, default=4096)
    parser.add_argument("--macro-train-target-valid", type=int, default=2048)
    parser.add_argument("--macro-validation-target-valid", type=int, default=512)
    parser.add_argument(
        "--macro-train-initial-candidate-budget", type=int, default=4096
    )
    parser.add_argument(
        "--macro-train-max-candidate-budget", type=int, default=8192
    )
    parser.add_argument(
        "--macro-validation-initial-candidate-budget", type=int, default=1024
    )
    parser.add_argument(
        "--macro-validation-max-candidate-budget", type=int, default=2048
    )
    parser.add_argument("--macro-minimum-partial-train", type=int, default=1024)
    parser.add_argument("--macro-minimum-partial-validation", type=int, default=256)
    parser.add_argument("--macro-parent-batch-size", type=int, default=256)
    parser.add_argument("--long-diagnostic-paths", type=int, default=50)
    parser.add_argument("--long-diagnostic-horizon", type=int, default=100)
    parser.add_argument(
        "--run-long-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--use-long-rollout-for-macro-training",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--fc1-production-epochs", type=int, default=10)
    parser.add_argument("--fc1-max-epochs", type=int, default=100)
    parser.add_argument("--fc1-patience", type=int, default=20)
    parser.add_argument("--sdf-chunks", type=int, default=4)
    parser.add_argument("--sdf-epochs-per-chunk", type=int, default=100)
    parser.add_argument("--sdf-recursive-epochs", type=int, default=0)
    parser.add_argument("--policy-distill-max-epochs", type=int, default=1000)
    parser.add_argument("--policy-distill-max-steps", type=int, default=1000)
    parser.add_argument("--policy-distill-patience", type=int, default=20)
    parser.add_argument("--disable-fc2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--market-clearing-mode",
        choices=("disabled", "market_cleared_experimental_v1"),
        default="market_cleared_experimental_v1",
    )
    parser.add_argument("--market-clearing-coarse-points", type=int, default=10)
    parser.add_argument(
        "--macro-clearing-solver-mode",
        choices=("strict_global", "fc1_bounded_improvement_pilot"),
        default="strict_global",
    )
    parser.add_argument("--market-clearing-refinement-points", type=int, default=20)
    parser.add_argument(
        "--market-clearing-max-refinement-rounds", type=int, default=4
    )
    parser.add_argument("--market-clearing-support-lower", type=float, default=-8.0)
    parser.add_argument("--market-clearing-support-upper", type=float, default=2.0)
    parser.add_argument("--market-clearing-support-padding", type=float, default=2.0)
    parser.add_argument(
        "--market-clearing-residual-tolerance", type=float, default=1.0e-5
    )
    parser.add_argument(
        "--market-clearing-width-tolerance", type=float, default=1.0e-5
    )
    parser.add_argument(
        "--market-clearing-candidate-chunk-size", type=int, default=262144
    )
    parser.add_argument(
        "--aggregation-mode",
        choices=(
            "legacy_mixed_replay",
            "hard_realized_reference",
            "continuum_mass_consistent_v1",
        ),
        default="continuum_mass_consistent_v1",
    )
    parser.add_argument("--continuum-i-points", type=int, default=9)
    parser.add_argument("--continuum-z-points", type=int, default=7)
    parser.add_argument(
        "--continuum-projection-tolerance", type=float, default=1.0e-6
    )
    parser.add_argument("--boundary-z-scan-points", type=int, default=65)
    parser.add_argument("--boundary-z-quadrature-points", type=int, default=8)
    parser.add_argument("--boundary-i-scan-points", type=int, default=33)
    parser.add_argument("--boundary-i-quadrature-points", type=int, default=8)
    parser.add_argument("--boundary-refinement-rounds", type=int, default=16)
    parser.add_argument(
        "--boundary-candidate-h-microbatch-size", type=int, default=1
    )
    parser.add_argument(
        "--boundary-scan-tail-probability", type=float, default=1.0e-8
    )
    parser.add_argument(
        "--bp-actual-use-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        _apply_smoke(args)
    if not args.disable_fc2:
        parser.error("formal nested training requires --disable-fc2")
    if args.sdf_recursive_epochs != 0:
        parser.error("formal nested training requires --sdf-recursive-epochs 0")
    if args.use_long_rollout_for_macro_training:
        parser.error("long rollout is diagnostic-only and cannot enter primary Macro training")
    if args.branch_num != 2:
        parser.error("one-step conditional SDF training requires --branch-num 2")
    if (
        args.aggregation_mode != "legacy_mixed_replay"
        and args.market_clearing_mode == "disabled"
    ):
        parser.error("hard/continuum aggregation requires market clearing")
    if args.continuum_i_points < 2 or args.continuum_z_points < 2:
        parser.error("continuum quadrature point counts must be at least two")
    try:
        _candidate_operator_configs(args)
    except ValueError as error:
        parser.error(str(error))
    if args.bootstrap_validation_paths < 1:
        parser.error("--bootstrap-validation-paths must be positive")
    budget_triplets = (
        (
            args.macro_train_target_valid,
            args.macro_train_initial_candidate_budget,
            args.macro_train_max_candidate_budget,
            args.macro_minimum_partial_train,
            "train",
        ),
        (
            args.macro_validation_target_valid,
            args.macro_validation_initial_candidate_budget,
            args.macro_validation_max_candidate_budget,
            args.macro_minimum_partial_validation,
            "validation",
        ),
    )
    for target, initial, maximum, partial, label in budget_triplets:
        if not (1 <= partial <= target <= initial <= maximum):
            parser.error(
                f"invalid {label} candidate budgets: require "
                "1 <= partial <= target <= initial <= maximum"
            )
    for name in ("equilibrium_outer_max", "firm_inner_max", "p_policy_outer_max",
                 "q_picard_outer_max", "macro_inner_max"):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    args, resume_root, resume_checkpoint = _resume_with_recorded_config(
        args, raw_argv
    )
    device = torch.device(args.device)
    if device.type == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, "0"}:
        raise RuntimeError("formal CUDA run requires CUDA_VISIBLE_DEVICES=0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    if device.type == "cuda" and device.index not in {None, 0}:
        raise RuntimeError("formal run is single-device cuda:0 only")
    seed_all(args.seed)
    started_at = time.time()
    git_head = _git_head()
    fingerprint = _args_fingerprint(args)
    if args.output_dir is not None:
        run_root = args.output_dir.expanduser().resolve()
    elif resume_root is not None:
        run_root = resume_root
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = (ROOT / "cachedir" / f"full_nested_equilibrium_{stamp}").resolve()
    for name in ("metadata", "fixed_point", "bootstrap", "checkpoints", "figures", "logs", "reports"):
        (run_root / name).mkdir(parents=True, exist_ok=True)

    episode, hp, models = _setup_fresh(args, device)
    fc2 = models.get("fc2")
    fc2_counter = {"calls": 0}
    fc2_initial_hash = module_hash(fc2)
    if fc2 is not None:
        fc2.requires_grad_(False).eval()
    install_fc2_guard(fc2, fc2_counter)
    train_bank, validation_bank, raw_train_children, raw_validation_children = (
        _build_or_load_fixed_banks(
            args=args, run_root=run_root, episode=episode, models=models, device=device
        )
    )
    state: dict[str, Any] = {
        "equilibrium_outer": 0, "phase": "firm_p", "firm_inner": 0,
        "p_policy_outer": 0, "q_picard_outer": 0, "macro_inner": 0,
        "sdf_chunk": 0, "firm_gate_streak": 0, "macro_gate_streak": 0,
        "overall_gate_streak": 0,
    }
    if resume_checkpoint is not None:
        payload = load_checkpoint(
            resume_checkpoint, models=models, optimizers=episode.optimizers,
            schedulers=getattr(episode, "lr_schedulers", {}), episode=episode,
            git_head=git_head,
            hyperparameter_fingerprint=fingerprint,
            allow_code_mismatch=args.allow_code_mismatch,
        )
        state.update(payload["state"])
        expected_banks = payload.get("extra", {}).get("bank_hashes")

    fixed = run_root / "fixed_point"
    train_initial_path = fixed / "simulation_train_initial.pt"
    validation_initial_path = fixed / "simulation_validation_initial.pt"
    if train_initial_path.is_file() and validation_initial_path.is_file():
        train_initial = torch.load(train_initial_path, map_location="cpu", weights_only=True)
        validation_initial = torch.load(validation_initial_path, map_location="cpu", weights_only=True)
    else:
        train_initial = make_initial_state(
            models=models, episode=episode, paths=args.long_diagnostic_paths,
            firms=args.firms_per_path, horizon=args.long_diagnostic_horizon,
            branch_num=args.branch_num, seed=args.seed + 1000, device=device,
        )
        validation_initial = make_initial_state(
            models=models, episode=episode, paths=args.long_diagnostic_paths,
            firms=args.firms_per_path, horizon=args.long_diagnostic_horizon,
            branch_num=args.branch_num, seed=args.seed + 2000, device=device,
        )
        if state_hash(train_initial) == state_hash(validation_initial):
            raise RuntimeError("train and validation initial cross sections are identical")
        atomic_torch_save(train_initial_path, train_initial)
        atomic_torch_save(validation_initial_path, validation_initial)

    macro_probe = _build_or_load_macro_probe(
        args=args, run_root=run_root, episode=episode, models=models,
        train_initial=train_initial, validation_initial=validation_initial,
        device=device,
    )
    bank_hashes = _fixed_bank_hashes(
        train_bank, validation_bank, raw_train_children,
        raw_validation_children, macro_probe,
    )
    bank_hashes.update({
        "crn_train_initial": state_hash(train_initial),
        "crn_validation_initial": state_hash(validation_initial),
        "crn_train_shocks": stable_json_hash([
            args.seed + 1011, args.seed + 1012, args.branch_num,
            args.long_diagnostic_horizon,
        ]),
        "crn_validation_shocks": stable_json_hash([
            args.seed + 2011, args.seed + 2012, args.branch_num,
            args.long_diagnostic_horizon,
        ]),
    })
    if (
        resume_checkpoint is not None
        and args.market_clearing_mode == "disabled"
        and expected_banks != bank_hashes
    ):
        raise RuntimeError(
            f"resume fixed-bank fingerprint mismatch: {expected_banks} != {bank_hashes}"
        )
    metadata_path = run_root / "metadata" / "run.json"
    if resume_checkpoint is None:
        _write_metadata(
            run_root=run_root, args=args, hp=hp, models=models, git_head=git_head,
            fingerprint=fingerprint, device=device, bank_hashes=bank_hashes,
        )
    elif not metadata_path.is_file():
        raise RuntimeError("resume run is missing metadata/run.json")
    else:
        resume_events_path = run_root / "metadata" / "resume_events.json"
        resume_events = (
            json.loads(resume_events_path.read_text(encoding="utf-8"))
            if resume_events_path.is_file() else []
        )
        resume_events.append({
            "checkpoint": str(resume_checkpoint), "timestamp": time.time(),
            "git_head": git_head, "hyperparameter_fingerprint": fingerprint,
            "allow_code_mismatch": bool(args.allow_code_mismatch),
        })
        atomic_write_json(resume_events_path, resume_events)

    oom_events_path = run_root / "metadata" / "oom_events.json"
    oom_events = (
        json.loads(oom_events_path.read_text(encoding="utf-8"))
        if oom_events_path.is_file() else []
    )
    summary_path = run_root / "summary.csv"
    if not summary_path.is_file():
        atomic_write_csv(summary_path, pd.DataFrame(columns=SUMMARY_COLUMNS))
    previous_equilibrium_snapshot: Optional[dict[str, torch.Tensor]] = None
    resumed_outer = int(state.get("equilibrium_outer", 0))
    if resumed_outer > 0:
        previous_probe_path = (
            run_root / f"equilibrium_outer_{resumed_outer - 1:03d}" / "equilibrium_probe.pt"
        )
        if previous_probe_path.is_file():
            previous_equilibrium_snapshot = torch.load(
                previous_probe_path, map_location="cpu", weights_only=True
            )
    overall_gate_streak = int(state.get("overall_gate_streak", 0))
    final_status = "MAX_EQUILIBRIUM_OUTER"

    initial_comparison_path = (
        run_root / "evaluation" / "historical_canonical" / "initial" / "summary.json"
    )
    if (
        args.market_clearing_mode != "disabled"
        and args.run_long_diagnostics
        and resume_checkpoint is None
        and not initial_comparison_path.is_file()
    ):
        initial_market = run_simulation(
            models=models,
            episode=episode,
            initial_state=validation_initial,
            paths=args.long_diagnostic_paths,
            firms=args.firms_per_path,
            horizon=args.long_diagnostic_horizon,
            branch_num=args.branch_num,
            transition_seed=args.seed + 2011,
            entry_seed=args.seed + 2012,
            device=device,
        )
        _matched_legacy_market_comparison(
            output=initial_comparison_path.parent,
            label="initial_checkpoint",
            args=args,
            episode=episode,
            models=models,
            initial_state=validation_initial,
            market_output=initial_market,
            paths=args.long_diagnostic_paths,
            transition_seed=args.seed + 2011,
            entry_seed=args.seed + 2012,
            device=device,
        )

    for equilibrium_outer in range(
        int(state.get("equilibrium_outer", 0)), args.equilibrium_outer_max
    ):
        outer_started = time.perf_counter()
        outer_root = run_root / f"equilibrium_outer_{equilibrium_outer:03d}"
        firm_root = outer_root / "firm"
        macro_root = outer_root / "macro"
        checkpoints = outer_root / "checkpoints"
        for path in (firm_root, macro_root, checkpoints):
            path.mkdir(parents=True, exist_ok=True)
        if (
            args.market_clearing_mode != "disabled"
            and equilibrium_outer > 0
        ):
            # Macro changed at the previous outer boundary. Rebuild a matched
            # full-economy Firm transition bank under the frozen current Macro
            # law before beginning this Firm conditional solve.
            (
                train_bank,
                validation_bank,
                raw_train_children,
                raw_validation_children,
            ) = _build_or_load_fixed_banks(
                args=args,
                run_root=outer_root,
                episode=episode,
                models=models,
                device=device,
            )
            refreshed = _fixed_bank_hashes(
                train_bank,
                validation_bank,
                raw_train_children,
                raw_validation_children,
                macro_probe,
            )
            bank_hashes.update(refreshed)
            atomic_write_json(
                outer_root / "firm_bank_refresh.json",
                {
                    "equilibrium_outer": equilibrium_outer,
                    "reason": "macro_changed_rebuild_root_cleared_transitions",
                    "bank_hashes": refreshed,
                },
            )
        train_parents = train_bank.parent_states.to(device)
        validation_parents = validation_bank.parent_states.to(device)
        train_children, validation_children, m_diag = _refresh_children(
            models=models, episode=episode, hp=hp, train_bank=train_bank,
            validation_bank=validation_bank, train_children=raw_train_children,
            validation_children=raw_validation_children, device=device,
            preserve_child_macro=(args.market_clearing_mode != "disabled"),
        )
        atomic_write_json(outer_root / "m_parity.json", m_diag)
        current_anchor_hashes = _fixed_bank_hashes(
            train_bank, validation_bank, raw_train_children,
            raw_validation_children, macro_probe,
        )
        if any(bank_hashes.get(key) != value for key, value in current_anchor_hashes.items()):
            raise RuntimeError("fixed bank or macro probe hash changed")

        phase = str(state.get("phase", "firm_p")) if equilibrium_outer == int(state.get("equilibrium_outer", 0)) else "firm_p"
        firm_gate_streak = int(state.get("firm_gate_streak", 0)) if equilibrium_outer == int(state.get("equilibrium_outer", 0)) else 0
        firm_reached = firm_gate_streak >= 2
        last_firm_row: dict[str, Any] = {}
        p_count = q_count = 0
        firm_summary_path = firm_root / "firm_summary.csv"
        if firm_summary_path.is_file():
            existing_firm_summary = pd.read_csv(firm_summary_path)
            if not existing_firm_summary.empty:
                last_firm_row = existing_firm_summary.iloc[-1].to_dict()
            p_count = int(existing_firm_summary.get("p_policy_outer_count", pd.Series(dtype=float)).fillna(0).sum())
            q_count = int(existing_firm_summary.get("q_picard_outer_count", pd.Series(dtype=float)).fillna(0).sum())
        start_firm_inner = int(state.get("firm_inner", 0)) if phase in {"firm_p", "firm_q", "firm_inner"} else args.firm_inner_max
        macro_hash_for_firm = module_hash(models["sdf_fc1"])

        for firm_inner in range(start_firm_inner, args.firm_inner_max):
            firm_inner_root = firm_root / f"firm_inner_{firm_inner:03d}"
            p_root, q_root = firm_inner_root / "p", firm_inner_root / "q"
            p_root.mkdir(parents=True, exist_ok=True)
            q_root.mkdir(parents=True, exist_ok=True)
            initial_problem = build_problem(
                episode, models["policy_value"], validation_parents,
                validation_children, args.p_batch_size,
                snapshot_batch_size=min(512, args.p_batch_size), require_legacy_m1=False,
            )
            initial_view = build_coupled_view(models["policy_value"], initial_problem)
            initial_p_metrics = evaluate_problem(models["policy_value"], initial_problem)
            initial_q_problem, _ = build_q_survival_problem(
                episode=episode, initial_model=models["policy_value"],
                frozen_p_model=deepcopy(models["policy_value"]).eval().requires_grad_(False),
                parents=validation_parents, children=validation_children,
                batch_size=args.q_batch_size, m_mode="current_sdf_child",
                include_all_regimes=True,
            )
            initial_q_metrics = evaluate_q_problem(
                models["policy_value"], episode, initial_q_problem
            )
            p_rows: list[dict[str, Any]] = []
            p_streak = 0
            p_start = int(state.get("p_policy_outer", 0)) if phase == "firm_p" and firm_inner == start_firm_inner else 0
            if phase != "firm_q" or firm_inner != start_firm_inner:
                for p_outer in range(p_start, args.p_policy_outer_max):
                    block_start = capture_training_transaction(
                        models=models,
                        optimizers=episode.optimizers,
                        episode=episode,
                        stage_progress={
                            **state,
                            "equilibrium_outer": equilibrium_outer,
                            "phase": "firm_p",
                            "firm_inner": firm_inner,
                            "p_policy_outer": p_outer,
                        },
                    )

                    def run_p(batch_size: int):
                        return _p_policy_outer(
                            model=models["policy_value"], episode=episode,
                            train_parents=train_parents, train_children=train_children,
                            validation_parents=validation_parents,
                            validation_children=validation_children,
                            batch_size=batch_size, args=args, policy_outer=p_outer,
                        )

                    result, used_batch = run_with_oom_fallback(
                        block="P", initial_batch_size=args.p_batch_size,
                        fallback_batch_size=max(1, args.p_batch_size // 2),
                        execute=run_p,
                        reload_block_start=lambda: restore_training_transaction(
                            block_start,
                            models=models,
                            optimizers=episode.optimizers,
                            episode=episode,
                        ),
                        events=oom_events, events_path=oom_events_path, device=device,
                        indices={"equilibrium_outer": equilibrium_outer, "firm_inner": firm_inner, "macro_inner": -1},
                    )
                    solved, row, history = result
                    models["policy_value"].load_state_dict(solved.state_dict(), strict=True)
                    episode.models["policy_value"] = models["policy_value"]
                    row["batch_size"] = used_batch
                    p_rows.append(row)
                    p_count += 1
                    atomic_write_csv(p_root / f"optimizer_{p_outer:03d}.csv", history)
                    atomic_write_csv(p_root / "p_policy_history.csv", pd.DataFrame(p_rows))
                    p_pass = bool(
                        row["full_greedy_P0_residual"] <= 0.02
                        and row["full_greedy_PI_residual"] <= 0.02
                        and row["investment_selector_flip_share"] <= 0.01
                        and row["default_survival_flip_share"] <= 0.01
                        and row["bp0_p90"] <= 0.025 and row["bpI_p90"] <= 0.025
                    )
                    p_streak = p_streak + 1 if p_pass else 0
                    next_state = {
                        **state, "equilibrium_outer": equilibrium_outer,
                        "phase": "firm_p", "firm_inner": firm_inner,
                        "p_policy_outer": p_outer + 1, "q_picard_outer": 0,
                    }
                    _save_stage_checkpoint(
                        run_root=run_root,
                        path=p_root / f"policy_outer_{p_outer:03d}.pt",
                        models=models, optimizers=episode.optimizers, episode=episode,
                        state=next_state, git_head=git_head,
                        fingerprint=fingerprint, bank_hashes=bank_hashes,
                        extra={"row": row, "m_parity": m_diag["parity"]},
                    )
                    append_summary_row(summary_path, {
                        "stage": "p_policy_outer",
                        "equilibrium_outer": equilibrium_outer,
                        "firm_inner": firm_inner, "p_policy_outer": p_outer,
                        "P0_residual": row["full_greedy_P0_residual"],
                        "PI_residual": row["full_greedy_PI_residual"],
                        "P0_move_p90": row["p0_function_change_p90"],
                        "PI_move_p90": row["pi_function_change_p90"],
                        "investment_flip": row["investment_selector_flip_share"],
                        "default_flip": row["default_survival_flip_share"],
                        "bp0_move_p90": row["bp0_p90"],
                        "bpI_move_p90": row["bpI_p90"],
                        "GPU_max_memory": cuda_memory(device)["max_memory_allocated"],
                        "total_elapsed": time.time() - started_at,
                    })
                    if p_streak >= 2 or (
                        p_pass and row["p0_function_change_p90"] <= 0.005
                        and row["pi_function_change_p90"] <= 0.005
                    ):
                        break
            if module_hash(models["sdf_fc1"]) != macro_hash_for_firm:
                raise RuntimeError("Macro hash changed during frozen FirmSolve P stage")

            q_rows: list[dict[str, Any]] = []
            q_streak = 0
            q_start = int(state.get("q_picard_outer", 0)) if phase == "firm_q" and firm_inner == start_firm_inner else 0
            for q_outer in range(q_start, args.q_picard_outer_max):
                block_start = capture_training_transaction(
                    models=models,
                    optimizers=episode.optimizers,
                    episode=episode,
                    stage_progress={
                        **state,
                        "equilibrium_outer": equilibrium_outer,
                        "phase": "firm_q",
                        "firm_inner": firm_inner,
                        "q_picard_outer": q_outer,
                    },
                )

                def run_q(batch_size: int):
                    return _q_picard_outer(
                        model=models["policy_value"], episode=episode,
                        train_parents=train_parents, train_children=train_children,
                        validation_parents=validation_parents,
                        validation_children=validation_children,
                        batch_size=batch_size, args=args, picard_outer=q_outer,
                    )

                result, used_batch = run_with_oom_fallback(
                    block="Q", initial_batch_size=args.q_batch_size,
                    fallback_batch_size=max(1, args.q_batch_size // 2),
                    execute=run_q,
                    reload_block_start=lambda: restore_training_transaction(
                        block_start,
                        models=models,
                        optimizers=episode.optimizers,
                        episode=episode,
                    ),
                    events=oom_events, events_path=oom_events_path, device=device,
                    indices={"equilibrium_outer": equilibrium_outer, "firm_inner": firm_inner, "macro_inner": -1},
                )
                row, history = result
                row["batch_size"] = used_batch
                q_rows.append(row)
                q_count += 1
                atomic_write_csv(q_root / f"optimizer_{q_outer:03d}.csv", pd.DataFrame(history))
                atomic_write_csv(q_root / "q_picard_history.csv", pd.DataFrame(q_rows))
                q_pass = bool(
                    row["fresh_Q_residual"] <= 0.02
                    and row["Q_function_move_p90"] <= 0.01
                )
                q_streak = q_streak + 1 if q_pass else 0
                next_state = {
                    **state, "equilibrium_outer": equilibrium_outer,
                    "phase": "firm_q", "firm_inner": firm_inner,
                    "p_policy_outer": 0, "q_picard_outer": q_outer + 1,
                }
                _save_stage_checkpoint(
                    run_root=run_root,
                    path=q_root / f"picard_outer_{q_outer:03d}.pt",
                    models=models, optimizers=episode.optimizers, episode=episode,
                    state=next_state, git_head=git_head,
                    fingerprint=fingerprint, bank_hashes=bank_hashes,
                    extra={"row": row, "m_parity": m_diag["parity"]},
                )
                append_summary_row(summary_path, {
                    "stage": "q_picard_outer",
                    "equilibrium_outer": equilibrium_outer,
                    "firm_inner": firm_inner, "q_picard_outer": q_outer,
                    "Q_residual": row["fresh_Q_residual"],
                    "Q_move_p90": row["Q_function_move_p90"],
                    "GPU_max_memory": cuda_memory(device)["max_memory_allocated"],
                    "total_elapsed": time.time() - started_at,
                })
                if q_streak >= 2:
                    break
            if module_hash(models["sdf_fc1"]) != macro_hash_for_firm:
                raise RuntimeError("Macro hash changed during frozen FirmSolve Q stage")

            fresh_p_problem = build_problem(
                episode, models["policy_value"], validation_parents,
                validation_children, args.p_batch_size,
                snapshot_batch_size=min(512, args.p_batch_size), require_legacy_m1=False,
            )
            p_metrics = evaluate_problem(models["policy_value"], fresh_p_problem)
            q_problem, _ = build_q_survival_problem(
                episode=episode, initial_model=models["policy_value"],
                frozen_p_model=deepcopy(models["policy_value"]).eval().requires_grad_(False),
                parents=validation_parents, children=validation_children,
                batch_size=args.q_batch_size, m_mode="current_sdf_child",
                include_all_regimes=True,
            )
            q_metrics = evaluate_q_problem(models["policy_value"], episode, q_problem)
            final_view = build_coupled_view(models["policy_value"], fresh_p_problem)
            movement = coupled_transition_metrics(initial_view, final_view)
            last_firm_row = {
                "equilibrium_outer": equilibrium_outer, "firm_inner": firm_inner,
                "P0_residual_pre": initial_p_metrics["p0_norm_mae"],
                "PI_residual_pre": initial_p_metrics["pi_norm_mae"],
                "Q_residual_pre": initial_q_metrics["normalized_mae"],
                "P0_residual": p_metrics["p0_norm_mae"],
                "PI_residual": p_metrics["pi_norm_mae"],
                "Q_residual": q_metrics["normalized_mae"],
                "P0_move_p90": movement["p0_function_change_p90"],
                "PI_move_p90": movement["pi_function_change_p90"],
                "Q_move_p90": movement["q_movement_p90"],
                "investment_flip": movement["investment_selector_flip_share"],
                "default_flip": movement["default_survival_flip_share"],
                "bp0_move_p90": movement["bp0_p90"],
                "bpI_move_p90": movement["bpI_p90"],
                "PI_active_share": movement["pi_active_share_new"],
                "default_share": movement["default_share_new"],
                "p_policy_outer_count": len(p_rows),
                "q_picard_outer_count": len(q_rows),
            }
            firm_gate_streak = firm_gate_streak + 1 if _firm_gate(last_firm_row) else 0
            last_firm_row["firm_gate_pass"] = _firm_gate(last_firm_row)
            last_firm_row["firm_gate_streak"] = firm_gate_streak
            atomic_write_csv(
                firm_root / "firm_summary.csv",
                pd.concat([
                    pd.read_csv(firm_root / "firm_summary.csv")
                    if (firm_root / "firm_summary.csv").is_file() else pd.DataFrame(),
                    pd.DataFrame([last_firm_row]),
                ], ignore_index=True),
            )
            state = {
                **state, "equilibrium_outer": equilibrium_outer,
                "phase": "firm_inner", "firm_inner": firm_inner + 1,
                "p_policy_outer": 0, "q_picard_outer": 0,
                "firm_gate_streak": firm_gate_streak,
            }
            _save_stage_checkpoint(
                run_root=run_root,
                path=firm_inner_root / "firm_inner_complete.pt",
                models=models, optimizers=episode.optimizers, episode=episode,
                state=state, git_head=git_head,
                fingerprint=fingerprint, bank_hashes=bank_hashes,
                extra={"row": last_firm_row, "m_parity": m_diag["parity"]},
            )
            append_summary_row(summary_path, {
                "stage": "firm_inner", **last_firm_row,
                "p_policy_outer_count": len(p_rows),
                "q_picard_outer_count": len(q_rows),
                "GPU_max_memory": cuda_memory(device)["max_memory_allocated"],
                "total_elapsed": time.time() - started_at,
            })
            phase = "firm_inner"
            if firm_gate_streak >= 2:
                firm_reached = True
                break

        policy_root = firm_root / "policy_distill"
        policy_root.mkdir(parents=True, exist_ok=True)
        if phase != "macro_inner":
            block_start = capture_training_transaction(
                models=models,
                optimizers=episode.optimizers,
                episode=episode,
                stage_progress=state,
            )

            def run_policy(batch_size: int):
                return _distill_policy_heads(
                    episode=episode, model=models["policy_value"],
                    train_bank=train_bank, validation_bank=validation_bank,
                    train_children=train_children, validation_children=validation_children,
                    batch_size=batch_size, args=args, output=policy_root, device=device,
                )

            policy_summary, _ = run_with_oom_fallback(
                block="POLICY_BP", initial_batch_size=args.p_batch_size,
                fallback_batch_size=max(1, args.p_batch_size // 2), execute=run_policy,
                reload_block_start=lambda: restore_training_transaction(
                    block_start,
                    models=models,
                    optimizers=episode.optimizers,
                    episode=episode,
                ),
                events=oom_events, events_path=oom_events_path, device=device,
                indices={"equilibrium_outer": equilibrium_outer, "firm_inner": -1, "macro_inner": -1},
            )
            atomic_torch_save(firm_root / "policy_value.pt", models["policy_value"].state_dict())
            firm_probe = build_coupled_view(models["policy_value"], build_problem(
                episode, models["policy_value"], validation_parents,
                validation_children, args.p_batch_size,
                snapshot_batch_size=min(512, args.p_batch_size), require_legacy_m1=False,
            ))
            atomic_torch_save(firm_root / "validation_probe.pt", {
                "P0": firm_probe.policy.p0.cpu(), "PI": firm_probe.policy.pi.cpu(),
                "Q": firm_probe.q.cpu(), "bp0": firm_probe.policy.bp0.cpu(),
                "bpI": firm_probe.policy.bpI.cpu(),
                "investment": firm_probe.policy.investment_selector.cpu(),
                "survival": firm_probe.policy.survival.cpu(),
            })
            state = {
                **state, "equilibrium_outer": equilibrium_outer,
                "phase": "macro_inner", "macro_inner": 0, "sdf_chunk": 0,
            }
            _save_stage_checkpoint(
                run_root=run_root, path=policy_root / "policy_distill_complete.pt",
                models=models, optimizers=episode.optimizers, episode=episode,
                state=state, git_head=git_head,
                fingerprint=fingerprint, bank_hashes=bank_hashes,
                extra={"policy_summary": policy_summary},
            )
            append_summary_row(summary_path, {
                "stage": "policy_distill", "equilibrium_outer": equilibrium_outer,
                "GPU_max_memory": cuda_memory(device)["max_memory_allocated"],
                "total_elapsed": time.time() - started_at,
            })

        _record_firm_first_stage(
            run_root, equilibrium_outer=equilibrium_outer, stage="firm_solve"
        )
        firm_hash_for_macro = firm_hashes(models["policy_value"])
        macro_gate_streak = int(state.get("macro_gate_streak", 0))
        macro_reached = macro_gate_streak >= 2
        start_macro_inner = int(state.get("macro_inner", 0))
        probe_parents = macro_probe["parents"].to(device)
        probe_children = macro_probe["children"].to(device)
        current_functions = snapshot_functions(models["sdf_fc1"], probe_parents, probe_children)
        macro_outer_start_path = macro_root / "macro_outer_start_functions.pt"
        if macro_outer_start_path.is_file():
            macro_outer_start_functions = torch.load(
                macro_outer_start_path, map_location="cpu", weights_only=True
            )
        else:
            macro_outer_start_functions = {
                key: value.detach().cpu().clone()
                for key, value in current_functions.items()
            }
            atomic_torch_save(macro_outer_start_path, macro_outer_start_functions)
        function_history_path = macro_root / "macro_probe_history.pt"
        if function_history_path.is_file():
            function_history = torch.load(
                function_history_path, map_location="cpu", weights_only=True
            )
            if not function_history:
                raise RuntimeError("resume macro probe history is empty")
            if str(state.get("macro_stage", "fc1_fit")) == "fc1_fit":
                for key in current_functions:
                    if not torch.equal(
                        current_functions[key].detach().cpu(),
                        function_history[-1][key],
                    ):
                        raise RuntimeError(f"resume macro probe mismatch: {key}")
        else:
            function_history = [
                {key: value.detach().cpu() for key, value in current_functions.items()}
            ]
            atomic_torch_save(function_history_path, function_history)
        (
            train_candidate_schedule,
            validation_candidate_schedule,
            train_validation_split_audit,
        ) = _build_or_refresh_macro_candidate_context(
            args=args,
            run_root=run_root,
            outer_root=outer_root,
            equilibrium_outer=equilibrium_outer,
        )
        boundary_integration, candidate_root_config = _candidate_operator_configs(args)
        long_start_train = long_start_validation = None
        if args.run_long_diagnostics:
            long_start_train, long_start_validation, _ = _simulation_pair(
                args=args, episode=episode, models=models,
                train_initial=train_initial, validation_initial=validation_initial,
                device=device,
            )
            _write_simulation_metadata(
                macro_root / "long_diagnostic_start.json",
                args=args,
                train_out=long_start_train,
                validation_out=long_start_validation,
                train_initial=train_initial,
                validation_initial=validation_initial,
                label=f"equilibrium_outer_{equilibrium_outer}_macro_start",
            )
            _write_long_diagnostic_artifacts(
                macro_root / "long_diagnostic_start",
                train_out=long_start_train,
                validation_out=long_start_validation,
            )
        if start_macro_inner > 0:
            (
                current_train_dataset,
                current_validation_dataset,
                _committed_manifest,
            ) = _load_committed_macro_inner_inputs(
                macro_root=macro_root,
                macro_inner=start_macro_inner,
                episode=episode,
                macro_batch_size=args.macro_batch_size,
                branch_num=args.branch_num,
                device=device,
            )
        else:
            current_train_dataset = build_one_step_macro_dataset(
                models=models,
                episode=episode,
                candidate_schedule=train_candidate_schedule,
                parent_batch_size=args.macro_parent_batch_size,
                macro_batch_size=args.macro_batch_size,
                device=device,
                output_dir=macro_root / "candidate_data" / "pre_fc1_train",
                integration=boundary_integration,
                root_config=candidate_root_config,
                require_cached_shards=str(state.get("macro_stage", "")) in {
                    "post_fc1_refresh", "sdf_fit", "outer_evaluation"
                },
            )
            current_validation_dataset = build_one_step_macro_dataset(
                models=models,
                episode=episode,
                candidate_schedule=validation_candidate_schedule,
                parent_batch_size=args.macro_parent_batch_size,
                macro_batch_size=args.macro_batch_size,
                device=device,
                output_dir=macro_root / "candidate_data" / "pre_fc1_validation",
                candidate_ids=(
                    validation_candidate_schedule.locked_candidate_ids or None
                ),
                allow_replenishment=not bool(
                    validation_candidate_schedule.locked_candidate_ids
                ),
                integration=boundary_integration,
                root_config=candidate_root_config,
                require_cached_shards=str(state.get("macro_stage", "")) in {
                    "post_fc1_refresh", "sdf_fit", "outer_evaluation"
                },
            )
        if current_train_dataset.dataset_status == "INSUFFICIENT_VALID_PAIRS":
            raise RuntimeError("insufficient valid train parent-two-child pairs")
        if current_validation_dataset.dataset_status == "INSUFFICIENT_VALID_PAIRS":
            raise RuntimeError("insufficient valid validation parent-two-child pairs")
        accepted_validation_ids = current_validation_dataset.pairing_audit[
            "candidate_coverage"
        ]["accepted_candidate_ids"]
        locked_validation = lock_validation_candidate_ids(
            validation_candidate_schedule, accepted_validation_ids
        )
        global_validation_schedule_path = (
            run_root / "fixed_point" / "macro_global_validation"
            / "validation_candidate_schedule.json"
        )
        if validation_candidate_schedule.locked_candidate_ids:
            unexpected_validation = sorted(
                set(map(int, accepted_validation_ids)).difference(
                    map(int, validation_candidate_schedule.locked_candidate_ids)
                )
            )
            if unexpected_validation:
                raise RuntimeError(
                    "fixed validation candidate identity replacement detected: "
                    f"{unexpected_validation[:20]}"
                )
        else:
            save_candidate_schedule(global_validation_schedule_path, locked_validation)
            save_candidate_schedule(
                outer_root / "macro_candidates" / "validation_candidate_schedule.json",
                locked_validation,
            )
            validation_candidate_schedule = locked_validation
        if start_macro_inner == 0 and str(
            state.get("macro_stage", "candidate_generation")
        ) in {
            "candidate_generation", "fc1_fit"
        }:
            _save_stage_checkpoint(
                run_root=run_root,
                path=macro_root / "candidate_generation_complete.pt",
                models=models,
                optimizers=episode.optimizers,
                episode=episode,
                state={
                    **state,
                    "equilibrium_outer": equilibrium_outer,
                    "phase": "macro_inner",
                    "macro_inner": start_macro_inner,
                    "macro_stage": "fc1_fit",
                },
                git_head=git_head,
                fingerprint=fingerprint,
                bank_hashes=bank_hashes,
                extra={
                    "train_schedule_fingerprint": train_candidate_schedule.fingerprint,
                    "validation_schedule_fingerprint": validation_candidate_schedule.fingerprint,
                    "train_pairing": current_train_dataset.pairing_audit,
                    "validation_pairing": current_validation_dataset.pairing_audit,
                },
            )
        _record_firm_first_stage(
            run_root,
            equilibrium_outer=equilibrium_outer,
            stage="macro_candidate_generation",
        )
        current_validation_frame = _to_frame(current_validation_dataset.macro)
        last_macro_row: dict[str, Any] = {}
        macro_history_path = macro_root / "macro_history.csv"
        if start_macro_inner > 0 and macro_history_path.is_file():
            existing_macro_history = pd.read_csv(macro_history_path)
            if not existing_macro_history.empty:
                last_macro_row = existing_macro_history.iloc[-1].to_dict()
        previous_core_move: Optional[float] = None
        if last_macro_row:
            previous_core_move = max(
                float(last_macro_row.get(key, 0.0)) for key in (
                    "M_function_move", "FC1_C_function_move", "FC1_K_function_move",
                    "Hatc_target_move", "LnK_target_move",
                )
            )
        explosion_streak = 0

        for macro_inner in range(start_macro_inner, args.macro_inner_max):
            macro_inner_root = macro_root / f"macro_inner_{macro_inner:03d}"
            macro_inner_root.mkdir(parents=True, exist_ok=True)
            resume_macro_stage = (
                str(state.get("macro_stage", "fc1_fit"))
                if macro_inner == start_macro_inner
                else "fc1_fit"
            )
            atomic_write_json(
                macro_inner_root / "input_parent_bank_metadata.json",
                {
                    "macro_primary_training_source": "PATH_SPECIFIC_BETA_BOUNDARY_AWARE_ONE_STEP",
                    "train_parent_count": current_train_dataset.pairing_audit[
                        "candidate_coverage"
                    ]["accepted_training_pair_count"],
                    "validation_parent_count": current_validation_dataset.pairing_audit[
                        "candidate_coverage"
                    ]["accepted_training_pair_count"],
                    "one_step_horizon": 1,
                    "branch_num": args.branch_num,
                    "long_rollout_training_used": False,
                    "train_schedule_fingerprint": train_candidate_schedule.fingerprint,
                    "validation_schedule_fingerprint": validation_candidate_schedule.fingerprint,
                    "firm_hashes": firm_hash_for_macro,
                },
            )
            atomic_write_json(
                macro_inner_root / "one_step_pairing_audit.json",
                {
                    "train": current_train_dataset.pairing_audit,
                    "validation": current_validation_dataset.pairing_audit,
                    "train_validation_source_split": train_validation_split_audit,
                },
            )
            block_start = capture_training_transaction(
                models=models,
                optimizers=episode.optimizers,
                episode=episode,
                stage_progress={
                    **state,
                    "equilibrium_outer": equilibrium_outer,
                    "macro_inner": macro_inner,
                    "macro_stage": resume_macro_stage,
                },
                dataset_provenance=input_dataset_fingerprints,
            )
            input_function_path = macro_inner_root / "input_function_snapshot.pt"
            if input_function_path.is_file():
                old_functions_cpu = torch.load(
                    input_function_path, map_location="cpu", weights_only=True
                )
                old_functions = {
                    key: value.to(device)
                    for key, value in old_functions_cpu.items()
                }
            else:
                old_functions = {
                    key: value.detach().clone()
                    for key, value in current_functions.items()
                }
                atomic_torch_save(
                    input_function_path,
                    {
                        key: value.detach().cpu()
                        for key, value in old_functions.items()
                    },
                )
            old_validation_frame = current_validation_frame
            old_validation_pairs = current_validation_dataset.pairs
            input_dataset_fingerprints = {
                "train": one_step_macro_dataset_fingerprint(
                    current_train_dataset
                ),
                "validation": one_step_macro_dataset_fingerprint(
                    current_validation_dataset
                ),
            }

            def run_macro(batch_size: int):
                target_batches = int(getattr(
                    episode.hyperparams, "sdf_true_target_batches", 20
                ))
                min_parents = int(getattr(
                    episode.hyperparams, "sdf_min_parent_groups_per_batch", 256
                ))
                train_fc1_batches = episode._create_sdf_batches_from_macro_tensor(
                    current_train_dataset.pairs,
                    batch_size=batch_size,
                    n_branches=args.branch_num,
                )
                validation_fc1_batches = episode._create_sdf_batches_from_macro_tensor(
                    current_validation_dataset.pairs,
                    batch_size=batch_size,
                    n_branches=args.branch_num,
                )
                fc1_summary_path = macro_inner_root / "fc1" / "summary.json"
                if resume_macro_stage in {
                    "post_fc1_refresh", "sdf_fit", "outer_evaluation"
                }:
                    if not fc1_summary_path.is_file():
                        raise RuntimeError(
                            "FC1 resume stage is missing fc1/summary.json"
                        )
                    fc1_summary = json.loads(
                        fc1_summary_path.read_text(encoding="utf-8")
                    )
                else:
                    fc1_summary = run_fc1_fit(
                        episode,
                        train_fc1_batches,
                        validation_fc1_batches,
                        macro_inner_root / "fc1", block="joint",
                        production_epochs=args.fc1_production_epochs,
                        max_epochs=args.fc1_max_epochs, patience=args.fc1_patience,
                        log_interval=1_000_000,
                    )
                    _save_stage_checkpoint(
                        run_root=run_root,
                        path=macro_inner_root / "fc1_complete.pt",
                        models=models,
                        optimizers=episode.optimizers,
                        episode=episode,
                        state={
                            **state,
                            "equilibrium_outer": equilibrium_outer,
                            "phase": "macro_inner",
                            "macro_inner": macro_inner,
                            "macro_stage": "post_fc1_refresh",
                        },
                        git_head=git_head,
                        fingerprint=fingerprint,
                        bank_hashes=bank_hashes,
                        extra={"fc1_summary": fc1_summary},
                    )
                if macro_inner == args.macro_inner_max - 1:
                    _record_firm_first_stage(
                        run_root,
                        equilibrium_outer=equilibrium_outer,
                        stage="fc1_fit",
                    )
                simulation_started = time.perf_counter()
                accepted_train_ids = current_train_dataset.pairing_audit[
                    "candidate_coverage"
                ]["accepted_candidate_ids"]
                accepted_validation_ids = current_validation_dataset.pairing_audit[
                    "candidate_coverage"
                ]["accepted_candidate_ids"]
                committed_train_path = (
                    macro_inner_root / "post_fc1_train" / "committed_dataset.pt"
                )
                committed_validation_path = (
                    macro_inner_root / "post_fc1_validation" / "committed_dataset.pt"
                )
                if resume_macro_stage in {"sdf_fit", "outer_evaluation"}:
                    post_train_dataset = load_committed_one_step_macro_dataset(
                        committed_train_path,
                        episode=episode,
                        macro_batch_size=batch_size,
                        n_branches=args.branch_num,
                        device=device,
                    )
                    post_validation_dataset = load_committed_one_step_macro_dataset(
                        committed_validation_path,
                        episode=episode,
                        macro_batch_size=batch_size,
                        n_branches=args.branch_num,
                        device=device,
                    )
                else:
                    post_train_dataset = build_one_step_macro_dataset(
                        models=models,
                        episode=episode,
                        candidate_schedule=train_candidate_schedule,
                        parent_batch_size=args.macro_parent_batch_size,
                        macro_batch_size=batch_size,
                        device=device,
                        output_dir=macro_inner_root / "post_fc1_train",
                        candidate_ids=accepted_train_ids,
                        allow_replenishment=True,
                        integration=boundary_integration,
                        root_config=candidate_root_config,
                    )
                    post_validation_dataset = build_one_step_macro_dataset(
                        models=models,
                        episode=episode,
                        candidate_schedule=validation_candidate_schedule,
                        parent_batch_size=args.macro_parent_batch_size,
                        macro_batch_size=batch_size,
                        device=device,
                        output_dir=macro_inner_root / "post_fc1_validation",
                        candidate_ids=accepted_validation_ids,
                        allow_replenishment=False,
                        integration=boundary_integration,
                        root_config=candidate_root_config,
                    )
                    train_output_fingerprint = save_committed_one_step_macro_dataset(
                        committed_train_path, post_train_dataset
                    )
                    validation_output_fingerprint = (
                        save_committed_one_step_macro_dataset(
                            committed_validation_path, post_validation_dataset
                        )
                    )
                    post_train_ids = post_train_dataset.pairing_audit[
                        "candidate_coverage"
                    ]["accepted_candidate_ids"]
                    post_validation_ids = post_validation_dataset.pairing_audit[
                        "candidate_coverage"
                    ]["accepted_candidate_ids"]
                    replenishment_audit = {
                        "pre_fc1_train_candidate_ids": list(accepted_train_ids),
                        "post_fc1_train_candidate_ids": list(post_train_ids),
                        "failed_previously_valid_train_ids": sorted(
                            set(map(int, accepted_train_ids)).difference(post_train_ids)
                        ),
                        "replacement_train_ids": sorted(
                            set(map(int, post_train_ids)).difference(accepted_train_ids)
                        ),
                        "train_duplicate_ids": len(post_train_ids)
                        != len(set(map(int, post_train_ids))),
                        "pre_fc1_validation_candidate_ids": list(
                            accepted_validation_ids
                        ),
                        "post_fc1_validation_candidate_ids": list(
                            post_validation_ids
                        ),
                        "validation_replacement_ids": sorted(
                            set(map(int, post_validation_ids)).difference(
                                accepted_validation_ids
                            )
                        ),
                        "train_replenishment_bounded_by_fixed_schedule": True,
                        "validation_replenishment_enabled": False,
                    }
                    if replenishment_audit["train_duplicate_ids"]:
                        raise RuntimeError("post-FC1 train replenishment reused an id")
                    if replenishment_audit["validation_replacement_ids"]:
                        raise RuntimeError(
                            "post-FC1 validation candidate replacement detected"
                        )
                    atomic_write_json(
                        macro_inner_root / "post_fc1_replenishment_audit.json",
                        replenishment_audit,
                    )
                    atomic_write_json(
                        macro_inner_root / "committed_dataset_manifest.json",
                        {
                            "macro_inner": macro_inner,
                            "committed_stage": "post_fc1_refresh_complete",
                            "input_dataset_fingerprints": input_dataset_fingerprints,
                            "output_dataset_fingerprints": {
                                "train": train_output_fingerprint,
                                "validation": validation_output_fingerprint,
                            },
                            "accepted_train_candidate_ids": list(post_train_ids),
                            "accepted_validation_candidate_ids": list(
                                post_validation_ids
                            ),
                            "train_schedule_fingerprint": (
                                train_candidate_schedule.fingerprint
                            ),
                            "validation_schedule_fingerprint": (
                                validation_candidate_schedule.fingerprint
                            ),
                            "fc1_model_hash_at_production": module_hash(
                                models["sdf_fc1"]
                            ),
                            "firm_model_hash_at_production": module_hash(
                                models["policy_value"]
                            ),
                        },
                    )
                if post_train_dataset.dataset_status == "INSUFFICIENT_VALID_PAIRS":
                    raise RuntimeError("post-FC1 train refresh lost the minimum valid pairs")
                if post_validation_dataset.dataset_status == "INSUFFICIENT_VALID_PAIRS":
                    raise RuntimeError(
                        "post-FC1 fixed validation refresh lost the minimum valid pairs"
                    )
                validation_coverage = validation_coverage_record(
                    original_candidate_ids=(
                        validation_candidate_schedule.locked_candidate_ids
                        or accepted_validation_ids
                    ),
                    before=current_validation_dataset,
                    after=post_validation_dataset,
                )
                if macro_inner == args.macro_inner_max - 1:
                    _record_firm_first_stage(
                        run_root,
                        equilibrium_outer=equilibrium_outer,
                        stage="post_fc1_refresh",
                    )
                if resume_macro_stage != "outer_evaluation":
                    _save_stage_checkpoint(
                        run_root=run_root,
                        path=macro_inner_root / "post_fc1_refresh_complete.pt",
                        models=models,
                        optimizers=episode.optimizers,
                        episode=episode,
                        state={
                            **state,
                            "equilibrium_outer": equilibrium_outer,
                            "phase": "macro_inner",
                            "macro_inner": macro_inner,
                            "macro_stage": "sdf_fit",
                        },
                        git_head=git_head,
                        fingerprint=fingerprint,
                        bank_hashes=bank_hashes,
                        extra={
                            "train_pairing": post_train_dataset.pairing_audit,
                            "validation_pairing": post_validation_dataset.pairing_audit,
                            "input_dataset_fingerprints": input_dataset_fingerprints,
                            "output_dataset_fingerprints": {
                                "train": one_step_macro_dataset_fingerprint(
                                    post_train_dataset
                                ),
                                "validation": one_step_macro_dataset_fingerprint(
                                    post_validation_dataset
                                ),
                            },
                        },
                    )
                # Frozen-target post-fit metrics are produced inside
                # run_fc1_fit.  These are the re-simulation metrics on the new
                # endogenous root-cleared paths, before any SDF update.
                post_validation_fc1_batches = (
                    episode._create_sdf_batches_from_macro_tensor(
                        post_validation_dataset.pairs,
                        batch_size=batch_size,
                        n_branches=args.branch_num,
                    )
                )
                fc1_resimulation = evaluate_fc1(
                    episode.models["sdf_fc1"], post_validation_fc1_batches
                )
                fc1_summary["resimulation_validation"] = fc1_resimulation
                atomic_write_json(
                    macro_inner_root / "fc1" / "conditional_metrics.json",
                    {
                        "pre_fit_same_pairs": fc1_summary["initial_validation"],
                        "post_fit_same_pairs": fc1_summary["best_validation"],
                        "post_fit_resimulation": fc1_resimulation,
                        "target_semantics": {
                            "hatc": "root_cleared_child_Hatc",
                            "lnk": "production_child_LnK_calculated",
                            "delta_lnk": "child_LnK_minus_parent_LnK",
                        },
                    },
                )

                # SDF is trained only after the final FC1 for this conditional
                # round has generated a fresh fixed two-child root-cleared pair
                # bank.  It never trains on the pre-FC1 paths.
                sdf_summary = None
                if macro_inner == args.macro_inner_max - 1:
                    train_sdf_batches = episode._create_sdf_batches_from_macro_tensor(
                        post_train_dataset.pairs,
                        batch_size=batch_size,
                        n_branches=args.branch_num,
                        target_num_batches=target_batches,
                        min_parent_groups_per_batch=min_parents,
                    )
                    validation_sdf_batches = episode._create_sdf_batches_from_macro_tensor(
                        post_validation_dataset.pairs,
                        batch_size=batch_size,
                        n_branches=args.branch_num,
                    )
                    sdf_summary_path = macro_inner_root / "sdf" / "summary.json"
                    if resume_macro_stage == "outer_evaluation":
                        if not sdf_summary_path.is_file():
                            raise RuntimeError(
                                "SDF resume stage is missing sdf/summary.json"
                            )
                        sdf_summary = json.loads(
                            sdf_summary_path.read_text(encoding="utf-8")
                        )
                    else:
                        sdf_summary = run_sdf_chunks(
                            episode,
                            train_sdf_batches,
                            validation_sdf_batches,
                            probe_parents, probe_children, macro_inner_root / "sdf",
                            chunks=args.sdf_chunks,
                            epochs_per_chunk=args.sdf_epochs_per_chunk,
                            log_interval=1_000_000,
                        )
                        chunk_detail_path = (
                            macro_inner_root / "sdf" /
                            f"history_chunk_{args.sdf_chunks:03d}.json"
                        )
                        if not chunk_detail_path.is_file():
                            raise RuntimeError(f"missing SDF chunk diagnostics: {chunk_detail_path}")
                        chunk_detail = json.loads(chunk_detail_path.read_text(encoding="utf-8"))
                        chunk_history = chunk_detail.get("history", [])
                        final_losses = (
                            chunk_history[-1].get("train_summary", {}).get("final_losses", {})
                            if chunk_history else {}
                        )
                        sdf_summary["production_metrics"] = {
                            "objective": final_losses.get("sdf"),
                            "moment1": final_losses.get("sdf_normalized_r1_abs_mean"),
                            "moment2": final_losses.get("sdf_normalized_r2_abs_mean"),
                            "moment_loss": final_losses.get("sdf_moment_loss"),
                            "constraint_max_violation": final_losses.get(
                                "sdf_constraint_max_violation"
                            ),
                        }
                        require_finite_metrics(
                            "SDF production", sdf_summary["production_metrics"],
                            ("objective", "moment1", "moment2"),
                        )
                        atomic_write_json(sdf_summary_path, sdf_summary)
                        _save_stage_checkpoint(
                            run_root=run_root,
                            path=macro_inner_root / "sdf_complete.pt",
                            models=models,
                            optimizers=episode.optimizers,
                            episode=episode,
                            state={
                                **state,
                                "equilibrium_outer": equilibrium_outer,
                                "phase": "macro_inner",
                                "macro_inner": macro_inner,
                                "macro_stage": "outer_evaluation",
                            },
                            git_head=git_head,
                            fingerprint=fingerprint,
                            bank_hashes=bank_hashes,
                            extra={"sdf_summary": sdf_summary},
                        )
                    _record_firm_first_stage(
                        run_root,
                        equilibrium_outer=equilibrium_outer,
                        stage="sdf_fit",
                    )
                return (
                    fc1_summary,
                    sdf_summary,
                    post_train_dataset,
                    post_validation_dataset,
                    {
                        "train": post_train_dataset.pairing_audit["candidate_coverage"],
                        "validation": post_validation_dataset.pairing_audit["candidate_coverage"],
                        "validation_coverage": validation_coverage,
                    },
                    time.perf_counter() - simulation_started,
                )

            result, used_batch = run_with_oom_fallback(
                block="MACRO", initial_batch_size=args.macro_batch_size,
                fallback_batch_size=max(1, args.macro_batch_size // 2),
                execute=run_macro,
                reload_block_start=lambda: restore_training_transaction(
                    block_start,
                    models=models,
                    optimizers=episode.optimizers,
                    episode=episode,
                ),
                events=oom_events, events_path=oom_events_path, device=device,
                indices={"equilibrium_outer": equilibrium_outer, "firm_inner": -1, "macro_inner": macro_inner},
            )
            (
                fc1_summary,
                sdf_summary,
                current_train_dataset,
                current_validation_dataset,
                source_availability,
                simulation_seconds,
            ) = result
            append_summary_row(
                run_root / "validation_coverage.csv",
                {
                    "equilibrium_outer": equilibrium_outer,
                    "macro_inner": macro_inner,
                    **source_availability["validation_coverage"],
                },
            )
            atomic_write_json(
                macro_inner_root / "output_parent_bank_metadata.json",
                {
                    "train_bank_hash": table_hash(current_train_dataset.macro),
                    "validation_bank_hash": table_hash(current_validation_dataset.macro),
                    "train_pair_rows": len(current_train_dataset.pairs),
                    "validation_pair_rows": len(current_validation_dataset.pairs),
                    "train_parent_batch_size": current_train_dataset.used_parent_batch_size,
                    "validation_parent_batch_size": current_validation_dataset.used_parent_batch_size,
                    "source_availability": source_availability,
                    "long_rollout_training_used": False,
                },
            )
            if firm_hashes(models["policy_value"]) != firm_hash_for_macro:
                raise RuntimeError("Firm hash changed during MacroSolve")
            if fc2_counter["calls"] != 0 or module_hash(fc2) != fc2_initial_hash:
                raise RuntimeError("FC2 exclusion invariant failed")
            current_functions = snapshot_functions(
                models["sdf_fc1"], probe_parents, probe_children
            )
            function_history.append({
                key: value.detach().cpu() for key, value in current_functions.items()
            })
            current_validation_frame = _to_frame(current_validation_dataset.macro)
            function_moves = {}
            for name in ("m", "fc1_c", "fc1_k"):
                function_moves[name] = movement_metrics(
                    old_functions[name], current_functions[name], name
                )
            target_moves = _candidate_target_movements(
                old_validation_pairs, current_validation_dataset.pairs
            )
            atomic_write_json(
                macro_inner_root / "source_availability_audit.json",
                {
                    **source_availability,
                    "target_movement_alignment": {
                        key: value for key, value in target_moves.items()
                        if key not in {"Hatc_target_move", "LnK_target_move"}
                    },
                    "candidate_identity_alignment": "IMMUTABLE_CANDIDATE_BRANCH_IDENTITY",
                },
            )
            macro_diag = macro_diagnostics(current_validation_frame)
            positive_mass = current_validation_frame["n_firms"].clip(lower=1.0e-30)
            macro_diag.update({
                "investment_rate_firm": float(
                    (current_validation_frame["investment_mass"] / positive_mass).mean()
                ),
                "default_rate": float(
                    (current_validation_frame["default_mass"] / positive_mass).mean()
                ),
                "b_mean": float(current_validation_frame["mean_b"].mean()),
            })
            if sdf_summary is None:
                fc1_round_row = {
                    "equilibrium_outer": equilibrium_outer,
                    "fc1_inner_round": macro_inner,
                    "FC1_C_pre_R2": fc1_summary["initial_validation"].get(
                        "hatc_r2", float("nan")
                    ),
                    "FC1_C_post_R2": fc1_summary["best_validation"].get(
                        "hatc_r2", float("nan")
                    ),
                    "FC1_C_resim_R2": fc1_summary[
                        "resimulation_validation"
                    ].get("hatc_r2", float("nan")),
                    "FC1_K_pre_R2": fc1_summary["initial_validation"].get(
                        "lnk_r2", float("nan")
                    ),
                    "FC1_K_post_R2": fc1_summary["best_validation"].get(
                        "lnk_r2", float("nan")
                    ),
                    "FC1_K_resim_R2": fc1_summary[
                        "resimulation_validation"
                    ].get("lnk_r2", float("nan")),
                    "FC1_delta_LnK_pre_R2": fc1_summary[
                        "initial_validation"
                    ].get("delta_lnk_r2", float("nan")),
                    "FC1_delta_LnK_post_R2": fc1_summary[
                        "best_validation"
                    ].get("delta_lnk_r2", float("nan")),
                    "FC1_delta_LnK_resim_R2": fc1_summary[
                        "resimulation_validation"
                    ].get("delta_lnk_r2", float("nan")),
                    "FC1_C_function_move": function_moves["fc1_c"][
                        "fc1_c_nmae"
                    ],
                    "FC1_K_function_move": function_moves["fc1_k"][
                        "fc1_k_nmae"
                    ],
                    "Hatc_target_move": target_moves["Hatc_target_move"],
                    "LnK_target_move": target_moves["LnK_target_move"],
                    "simulation_seconds": simulation_seconds,
                    "SDF_training_executed": False,
                }
                require_finite_metrics(
                    "FC1 conditional",
                    fc1_round_row,
                    (
                        "FC1_C_post_R2",
                        "FC1_C_resim_R2",
                        "FC1_K_post_R2",
                        "FC1_K_resim_R2",
                        "FC1_C_function_move",
                        "FC1_K_function_move",
                    ),
                )
                fc1_history_path = macro_root / "fc1_conditional_history.csv"
                existing_fc1 = (
                    pd.read_csv(fc1_history_path)
                    if fc1_history_path.is_file()
                    else pd.DataFrame()
                )
                atomic_write_csv(
                    fc1_history_path,
                    pd.concat(
                        [existing_fc1, pd.DataFrame([fc1_round_row])],
                        ignore_index=True,
                    ),
                )
                state = {
                    **state,
                    "equilibrium_outer": equilibrium_outer,
                    "phase": "macro_inner",
                    "macro_inner": macro_inner + 1,
                    "sdf_chunk": 0,
                    "macro_gate_streak": 0,
                }
                committed_manifest_path = (
                    macro_inner_root / "committed_dataset_manifest.json"
                )
                if not committed_manifest_path.is_file():
                    raise RuntimeError(
                        "Macro inner completion is missing committed dataset manifest"
                    )
                committed_manifest = json.loads(
                    committed_manifest_path.read_text(encoding="utf-8")
                )
                committed_manifest.update(
                    {
                        "committed_stage": "macro_inner_complete",
                        "macro_model_hash_at_commit": module_hash(
                            models["sdf_fc1"]
                        ),
                        "sdf_training_executed": False,
                        "next_macro_inner": macro_inner + 1,
                    }
                )
                atomic_write_json(committed_manifest_path, committed_manifest)
                _save_stage_checkpoint(
                    run_root=run_root,
                    path=macro_inner_root / "macro_inner_complete.pt",
                    models=models,
                    optimizers=episode.optimizers,
                    episode=episode,
                    state=state,
                    git_head=git_head,
                    fingerprint=fingerprint,
                    bank_hashes=bank_hashes,
                    extra={
                        "row": fc1_round_row,
                        "firm_hashes": firm_hash_for_macro,
                        "input_dataset_fingerprints": input_dataset_fingerprints,
                        "output_dataset_fingerprints": committed_manifest[
                            "output_dataset_fingerprints"
                        ],
                    },
                )
                append_summary_row(
                    summary_path,
                    {
                        "stage": "fc1_inner",
                        **fc1_round_row,
                        "GPU_max_memory": cuda_memory(device)[
                            "max_memory_allocated"
                        ],
                        "total_elapsed": time.time() - started_at,
                    },
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            final_sdf = sdf_summary.get("final_row", {})
            production_sdf = sdf_summary["production_metrics"]
            last_macro_row = {
                "equilibrium_outer": equilibrium_outer,
                "macro_inner": macro_inner,
                "batch_size": used_batch,
                "M_function_move": function_moves["m"]["m_nmae"],
                "FC1_C_function_move": function_moves["fc1_c"]["fc1_c_nmae"],
                "FC1_K_function_move": function_moves["fc1_k"]["fc1_k_nmae"],
                "Hatc_target_move": target_moves["Hatc_target_move"],
                "LnK_target_move": target_moves["LnK_target_move"],
                "Hatc_path_move": float("nan"),
                "LnK_path_move": float("nan"),
                "C_path_move": float("nan"),
                "K_path_move": float("nan"),
                "path_gate_pass": False,
                "long_path_metrics_deferred_to_diagnostic": True,
                "parent_bank_summary_movement_max": max(
                    float(target_moves["Hatc_target_move"]),
                    float(target_moves["LnK_target_move"]),
                ),
                "FC1_C_val_R2": fc1_summary["best_validation"].get("hatc_r2", float("nan")),
                "FC1_K_val_R2": fc1_summary["best_validation"].get("lnk_r2", float("nan")),
                "FC1_C_pre_R2": fc1_summary["initial_validation"].get("hatc_r2", float("nan")),
                "FC1_C_post_R2": fc1_summary["best_validation"].get("hatc_r2", float("nan")),
                "FC1_C_resim_R2": fc1_summary["resimulation_validation"].get("hatc_r2", float("nan")),
                "FC1_K_pre_R2": fc1_summary["initial_validation"].get("lnk_r2", float("nan")),
                "FC1_K_post_R2": fc1_summary["best_validation"].get("lnk_r2", float("nan")),
                "FC1_K_resim_R2": fc1_summary["resimulation_validation"].get("lnk_r2", float("nan")),
                "FC1_delta_LnK_pre_R2": fc1_summary["initial_validation"].get("delta_lnk_r2", float("nan")),
                "FC1_delta_LnK_post_R2": fc1_summary["best_validation"].get("delta_lnk_r2", float("nan")),
                "FC1_delta_LnK_resim_R2": fc1_summary["resimulation_validation"].get("delta_lnk_r2", float("nan")),
                "SDF_objective": production_sdf["objective"],
                "SDF_moment1": production_sdf["moment1"],
                "SDF_moment2": production_sdf["moment2"],
                "within_D_M_movement": float(final_sdf.get("M_function_nmae", float("nan"))),
                "simulation_seconds": simulation_seconds,
                "investment_rate": macro_diag.get(
                    "investment_rate_firm", macro_diag.get("investment_rate", float("nan"))
                ),
                "investment_intensity": macro_diag.get("investment_rate", float("nan")),
                "default_rate": macro_diag.get("default_rate", float("nan")),
                "mean_b": macro_diag.get("b_mean", float("nan")),
                "mean_M": macro_diag.get("sdf_M_mean", float("nan")),
                "mean_firms": macro_diag.get("mean_firms", float("nan")),
                "mean_C": float(current_validation_frame["C"].mean()),
                "mean_K": float(current_validation_frame["K"].mean()),
            }
            if len(function_history) >= 3:
                for name in ("m", "fc1_c", "fc1_k"):
                    relation = movement_relation(
                        function_history[-2][name] - function_history[-3][name],
                        function_history[-1][name] - function_history[-2][name],
                    )
                    last_macro_row.update({f"{name}_{key}": value for key, value in relation.items()})
                    last_macro_row[f"{name}_strict_two_cycle"] = two_cycle_detect(
                        function_history[-3][name], function_history[-2][name],
                        function_history[-1][name], config=TwoCycleDiagnosticsConfig(),
                    )
            require_finite_metrics(
                "Macro", last_macro_row,
                ("M_function_move", "FC1_C_function_move", "FC1_K_function_move",
                 "Hatc_target_move", "LnK_target_move", "SDF_objective",
                 "SDF_moment1", "SDF_moment2", "mean_M", "mean_firms",
                 "mean_C", "mean_K", "mean_b", "investment_rate",
                 "default_rate"),
            )
            core_move = max(
                float(last_macro_row["M_function_move"]),
                float(last_macro_row["FC1_C_function_move"]),
                float(last_macro_row["FC1_K_function_move"]),
                float(last_macro_row["Hatc_target_move"]),
                float(last_macro_row["LnK_target_move"]),
            )
            if previous_core_move is not None and core_move > 2.0 * max(previous_core_move, 1e-12):
                explosion_streak += 1
            else:
                explosion_streak = 0
            if explosion_streak >= 2:
                raise RuntimeError("severe macro numerical explosion: core movement doubled twice")
            previous_core_move = core_move
            macro_gate_streak = macro_gate_streak + 1 if _macro_gate(last_macro_row) else 0
            last_macro_row["macro_gate_pass"] = _macro_gate(last_macro_row)
            last_macro_row["macro_gate_streak"] = macro_gate_streak
            existing = pd.read_csv(macro_history_path) if macro_history_path.is_file() else pd.DataFrame()
            atomic_write_csv(
                macro_history_path,
                pd.concat([existing, pd.DataFrame([last_macro_row])], ignore_index=True),
            )
            atomic_torch_save(macro_root / "macro_probe_history.pt", function_history)
            state = {
                **state, "equilibrium_outer": equilibrium_outer,
                "phase": "macro_inner", "macro_inner": macro_inner + 1,
                "sdf_chunk": args.sdf_chunks, "macro_gate_streak": macro_gate_streak,
            }
            committed_manifest_path = (
                macro_inner_root / "committed_dataset_manifest.json"
            )
            if not committed_manifest_path.is_file():
                raise RuntimeError(
                    "Macro inner completion is missing committed dataset manifest"
                )
            committed_manifest = json.loads(
                committed_manifest_path.read_text(encoding="utf-8")
            )
            committed_manifest.update(
                {
                    "committed_stage": "macro_inner_complete",
                    "macro_model_hash_at_commit": module_hash(models["sdf_fc1"]),
                    "sdf_training_executed": True,
                    "next_macro_inner": macro_inner + 1,
                }
            )
            atomic_write_json(committed_manifest_path, committed_manifest)
            _save_stage_checkpoint(
                run_root=run_root,
                path=macro_inner_root / "macro_inner_complete.pt",
                models=models, optimizers=episode.optimizers, episode=episode,
                state=state, git_head=git_head,
                fingerprint=fingerprint, bank_hashes=bank_hashes,
                extra={
                    "row": last_macro_row,
                    "firm_hashes": firm_hash_for_macro,
                    "input_dataset_fingerprints": input_dataset_fingerprints,
                    "output_dataset_fingerprints": committed_manifest[
                        "output_dataset_fingerprints"
                    ],
                },
            )
            append_summary_row(summary_path, {
                "stage": "macro_inner", **last_macro_row,
                "GPU_max_memory": cuda_memory(device)["max_memory_allocated"],
                "total_elapsed": time.time() - started_at,
            })
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if macro_gate_streak >= 2:
                macro_reached = True
                break

        macro_stop_reason = (
            "CONDITIONAL_FIXED_POINT_REACHED"
            if macro_reached else "MAX_INNER_BUT_STABLE"
        )
        long_path_moves: dict[str, Any] = {
            "long_diagnostics_executed": False,
            "long_rollout_training_used": False,
        }
        if args.run_long_diagnostics:
            long_final_train, long_final_validation, long_final_seconds = _simulation_pair(
                args=args,
                episode=episode,
                models=models,
                train_initial=train_initial,
                validation_initial=validation_initial,
                device=device,
            )
            _write_simulation_metadata(
                macro_root / "long_diagnostic_final.json",
                args=args,
                train_out=long_final_train,
                validation_out=long_final_validation,
                train_initial=train_initial,
                validation_initial=validation_initial,
                label=f"equilibrium_outer_{equilibrium_outer}_macro_final_and_boundary",
            )
            _write_long_diagnostic_artifacts(
                macro_root / "long_diagnostic_final",
                train_out=long_final_train,
                validation_out=long_final_validation,
            )
            if args.market_clearing_mode != "disabled" and equilibrium_outer == 0:
                _matched_legacy_market_comparison(
                    output=run_root / "evaluation" / "historical_canonical" / "outer_000",
                    label="after_outer_000",
                    args=args,
                    episode=episode,
                    models=models,
                    initial_state=validation_initial,
                    market_output=long_final_validation,
                    paths=args.long_diagnostic_paths,
                    transition_seed=args.seed + 2011,
                    entry_seed=args.seed + 2012,
                    device=device,
                )
            assert long_start_validation is not None
            long_path_moves = _path_and_target_movements(
                _to_frame(long_start_validation.macro),
                _to_frame(long_final_validation.macro),
            )
            long_path_moves.update({
                "long_diagnostics_executed": True,
                "long_rollout_training_used": False,
            })
            atomic_write_json(
                macro_root / "long_diagnostic_movement.json",
                {**long_path_moves, "simulation_seconds": long_final_seconds},
            )
        (
            _post_macro_train_children,
            post_macro_validation_children,
            post_macro_m_diag,
        ) = _refresh_children(
            models=models,
            episode=episode,
            hp=hp,
            train_bank=train_bank,
            validation_bank=validation_bank,
            train_children=raw_train_children,
            validation_children=raw_validation_children,
            device=device,
            preserve_child_macro=(args.market_clearing_mode != "disabled"),
        )
        post_macro_p_problem = build_problem(
            episode,
            models["policy_value"],
            validation_parents,
            post_macro_validation_children,
            args.p_batch_size,
            snapshot_batch_size=min(512, args.p_batch_size),
            require_legacy_m1=False,
        )
        post_macro_p_metrics = evaluate_problem(
            models["policy_value"], post_macro_p_problem
        )
        post_macro_q_problem, _ = build_q_survival_problem(
            episode=episode,
            initial_model=models["policy_value"],
            frozen_p_model=deepcopy(models["policy_value"]).eval().requires_grad_(False),
            parents=validation_parents,
            children=post_macro_validation_children,
            batch_size=args.q_batch_size,
            m_mode="current_sdf_child",
            include_all_regimes=True,
        )
        post_macro_q_metrics = evaluate_q_problem(
            models["policy_value"], episode, post_macro_q_problem
        )
        cross_block_firm = {
            "firm_bank_semantics": (
                "SAME_REALIZED_ROOT_CLEARED_TRANSITIONS_WITH_CURRENT_M"
                if args.market_clearing_mode != "disabled"
                else "SAME_FIXED_SHOCKS_WITH_CURRENT_FC1_AND_M"
            ),
            "P0_residual_under_old_macro": last_firm_row.get(
                "P0_residual", float("nan")
            ),
            "PI_residual_under_old_macro": last_firm_row.get(
                "PI_residual", float("nan")
            ),
            "Q_residual_under_old_macro": last_firm_row.get(
                "Q_residual", float("nan")
            ),
            "P0_residual_under_new_macro": post_macro_p_metrics["p0_norm_mae"],
            "PI_residual_under_new_macro": post_macro_p_metrics["pi_norm_mae"],
            "Q_residual_under_new_macro": post_macro_q_metrics["normalized_mae"],
            "current_m_parity": post_macro_m_diag["parity"],
        }
        atomic_write_json(
            outer_root / "metrics" / "cross_block_firm_residual.json",
            cross_block_firm,
        )
        if last_macro_row:
            last_macro_row["macro_stop_reason"] = macro_stop_reason
            last_macro_row.update({
                key: value
                for key, value in long_path_moves.items()
                if key.endswith("_path_move")
            })
        atomic_torch_save(macro_root / "sdf_fc1.pt", models["sdf_fc1"].state_dict())
        current_firm_probe = build_coupled_view(models["policy_value"], build_problem(
            episode, models["policy_value"], validation_parents,
            validation_children, args.p_batch_size,
            snapshot_batch_size=min(512, args.p_batch_size), require_legacy_m1=False,
        ))
        current_macro_probe = snapshot_functions(
            models["sdf_fc1"], probe_parents, probe_children
        )
        equilibrium_snapshot = {
            "parent_states": validation_parents.detach().cpu(),
            "P0": current_firm_probe.policy.p0.cpu(),
            "PI": current_firm_probe.policy.pi.cpu(), "Q": current_firm_probe.q.cpu(),
            "bp0": current_firm_probe.policy.bp0.cpu(),
            "bpI": current_firm_probe.policy.bpI.cpu(),
            "investment": current_firm_probe.policy.investment_selector.cpu(),
            "survival": current_firm_probe.policy.survival.cpu(),
            "M": current_macro_probe["m"].cpu(),
            "FC1_C": current_macro_probe["fc1_c"].cpu(),
            "FC1_K": current_macro_probe["fc1_k"].cpu(),
        }
        outer_moves = {
            "P0_move_p90": last_firm_row.get("P0_move_p90", float("nan")),
            "PI_move_p90": last_firm_row.get("PI_move_p90", float("nan")),
            "Q_move_p90": last_firm_row.get("Q_move_p90", float("nan")),
            "bp0_move_p90": last_firm_row.get("bp0_move_p90", float("nan")),
            "bpI_move_p90": last_firm_row.get("bpI_move_p90", float("nan")),
            "investment_flip": last_firm_row.get("investment_flip", float("nan")),
            "default_flip": last_firm_row.get("default_flip", float("nan")),
            "M_function_move": movement_metrics(
                macro_outer_start_functions["m"], current_macro_probe["m"], "M"
            )["M_nmae"],
            "FC1_C_function_move": movement_metrics(
                macro_outer_start_functions["fc1_c"], current_macro_probe["fc1_c"], "FC1_C"
            )["FC1_C_nmae"],
            "FC1_K_function_move": movement_metrics(
                macro_outer_start_functions["fc1_k"], current_macro_probe["fc1_k"], "FC1_K"
            )["FC1_K_nmae"],
        }
        if previous_equilibrium_snapshot is not None:
            for name in ("P0", "PI", "Q", "bp0", "bpI"):
                scale = 1.0 + previous_equilibrium_snapshot[name].abs()
                move = (equilibrium_snapshot[name] - previous_equilibrium_snapshot[name]).abs() / scale
                outer_moves[f"{name}_move_p90"] = _quantile(move, 0.90)
            outer_moves["investment_flip"] = float(
                (equilibrium_snapshot["investment"] != previous_equilibrium_snapshot["investment"])
                .to(torch.float32).mean()
            )
            outer_moves["default_flip"] = float(
                (equilibrium_snapshot["survival"] != previous_equilibrium_snapshot["survival"])
                .to(torch.float32).mean()
            )
        previous_equilibrium_snapshot = equilibrium_snapshot
        atomic_torch_save(outer_root / "equilibrium_probe.pt", equilibrium_snapshot)
        total_elapsed = time.time() - started_at
        summary_row = {
            "stage": "equilibrium_outer",
            "equilibrium_outer": equilibrium_outer,
            "firm_inner": last_firm_row.get("firm_inner", -1),
            "p_policy_outer_count": p_count,
            "q_picard_outer_count": q_count,
            "macro_inner_count": int(state.get("macro_inner", 0)),
            "P0_residual": last_firm_row.get("P0_residual", float("nan")),
            "PI_residual": last_firm_row.get("PI_residual", float("nan")),
            "Q_residual": last_firm_row.get("Q_residual", float("nan")),
            "P0_residual_post_macro": cross_block_firm[
                "P0_residual_under_new_macro"
            ],
            "PI_residual_post_macro": cross_block_firm[
                "PI_residual_under_new_macro"
            ],
            "Q_residual_post_macro": cross_block_firm[
                "Q_residual_under_new_macro"
            ],
            **outer_moves,
            **{key: last_macro_row.get(key, float("nan")) for key in (
                "M_function_move", "FC1_C_function_move", "FC1_K_function_move",
                "Hatc_target_move", "LnK_target_move", "Hatc_path_move",
                "LnK_path_move", "C_path_move", "K_path_move", "FC1_C_val_R2",
                "FC1_K_val_R2", "FC1_C_pre_R2", "FC1_C_post_R2",
                "FC1_C_resim_R2", "FC1_K_pre_R2", "FC1_K_post_R2",
                "FC1_K_resim_R2", "FC1_delta_LnK_pre_R2",
                "FC1_delta_LnK_post_R2", "FC1_delta_LnK_resim_R2",
                "SDF_objective", "SDF_moment1", "SDF_moment2",
                "within_D_M_movement", "investment_rate", "default_rate",
                "mean_b", "mean_M", "mean_firms", "mean_C", "mean_K",
            )},
            "firm_conditional_fixed_point_reached": firm_reached,
            "macro_conditional_fixed_point_reached": macro_reached,
            "firm_stop_reason": (
                "CONDITIONAL_FIXED_POINT_REACHED" if firm_reached else "MAX_INNER"
            ),
            "macro_stop_reason": macro_stop_reason,
            "GPU_max_memory": cuda_memory(device)["max_memory_allocated"],
            "stage_seconds": time.perf_counter() - outer_started,
            "total_elapsed": total_elapsed,
        }
        summary_row.update(outer_moves)
        overall_gate_streak = overall_gate_streak + 1 if _overall_gate(summary_row) else 0
        summary_row["overall_gate_pass"] = _overall_gate(summary_row)
        summary_row["overall_gate_streak"] = overall_gate_streak
        _record_firm_first_stage(
            run_root,
            equilibrium_outer=equilibrium_outer,
            stage="outer_evaluation",
        )
        append_summary_row(summary_path, summary_row)
        next_state = {
            **state, "equilibrium_outer": equilibrium_outer + 1,
            "phase": "firm_p", "firm_inner": 0, "p_policy_outer": 0,
            "q_picard_outer": 0, "macro_inner": 0, "sdf_chunk": 0,
            "macro_stage": "candidate_generation",
            "firm_gate_streak": 0, "macro_gate_streak": 0,
            "overall_gate_streak": overall_gate_streak,
        }
        _save_stage_checkpoint(
            run_root=run_root,
            path=checkpoints / "equilibrium_outer_complete.pt",
            models=models, optimizers=episode.optimizers, episode=episode,
            state=next_state, git_head=git_head,
            fingerprint=fingerprint, bank_hashes=bank_hashes,
            extra={"row": summary_row, "m_parity": m_diag["parity"]},
        )
        state = next_state
        print(json.dumps({"equilibrium_outer_complete": equilibrium_outer, **summary_row}, default=str), flush=True)
        if overall_gate_streak >= 2:
            final_status = "OVERALL_EQUILIBRIUM_CONVERGED"
            break

    for name, model in models.items():
        require_finite_module(name, model)
    if fc2_counter["calls"] != 0 or module_hash(fc2) != fc2_initial_hash:
        raise RuntimeError("FC2 final invariant failed")
    final_state = {**state, "phase": "complete", "status": final_status}
    _save_stage_checkpoint(
        run_root=run_root, path=run_root / "checkpoints" / "final.pt",
        models=models, optimizers=episode.optimizers, episode=episode,
        state=final_state, git_head=git_head,
        fingerprint=fingerprint, bank_hashes=bank_hashes,
        extra={"fc2_forward_calls": fc2_counter["calls"]},
    )
    atomic_write_json(run_root / "reports" / "final_summary.json", {
        "status": final_status, "run_root": str(run_root),
        "fc2_forward_calls": fc2_counter["calls"],
        "fc2_hash_invariant": module_hash(fc2) == fc2_initial_hash,
        "total_elapsed": time.time() - started_at,
    })
    print(f"FULL_NESTED_EQUILIBRIUM_RUN_ROOT={run_root}", flush=True)
    print(f"FINAL_STATUS={final_status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

