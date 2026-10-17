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
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
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
    cuda_memory,
    load_checkpoint,
    make_checkpoint_payload,
    module_hash,
    publish_checkpoint,
    require_finite_metrics,
    require_finite_module,
    run_with_oom_fallback,
    seed_all,
    stable_json_hash,
    tensor_hash,
)
from analysis.macro_parent_state_bank import (  # noqa: E402
    MacroParentSourceSchedule,
    build_one_step_macro_dataset,
    distribution_coverage,
    effective_capture_times,
    harvest_macro_parent_schedule,
    harvest_macro_parent_schedule_with_candidate,
    load_schedule,
    parent_summary_movement,
    parse_capture_times,
    prepare_macro_parent_schedule,
    save_bank_shards,
    save_schedule,
    source_overlap,
    write_coverage_artifacts,
    write_macro_inner_movement_figure,
)
from analysis.run_coupled_fixed_macro import (  # noqa: E402
    build_coupled_view,
    coupled_transition_metrics,
)
from analysis.run_macro_block_isolation import (  # noqa: E402
    aligned_macro_vectors,
    build_in_memory_batches,
    firm_hashes,
    flatten_batches,
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
    payload = torch.load(path, map_location="cpu", weights_only=False)
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
    ignored = {"cold_start", "resume", "allow_code_mismatch", "output_dir", "smoke"}
    return stable_json_hash({key: value for key, value in vars(args).items() if key not in ignored})


def _configure_hyperparams(args: argparse.Namespace):
    hp = build_hyperparams()
    hp.n_paths = int(args.macro_train_parent_count)
    hp.batch_size = int(args.macro_batch_size)
    hp.simulate_horizon = int(args.macro_bank_rollout_horizon)
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
    return hp


def _setup_fresh(args: argparse.Namespace, device: torch.device):
    Config.DEVICE = device
    hp = _configure_hyperparams(args)
    models = build_models(device)
    q_mode = str(getattr(models["policy_value"], "q_parameterization", "")).lower()
    if q_mode != "direct":
        raise RuntimeError(f"formal nested run requires direct Q, got {q_mode!r}")
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
    state: Mapping[str, Any], git_head: str, fingerprint: str,
    bank_hashes: Mapping[str, str], extra: Optional[Mapping[str, Any]] = None,
) -> None:
    payload = make_checkpoint_payload(
        models=models, state=state, git_head=git_head,
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
        child_payload = torch.load(child_path, map_location="cpu", weights_only=False)
        train_children = _load_child_bank(child_payload["train"])
        validation_children = _load_child_bank(child_payload["validation"])
        return train_bank, validation_bank, train_children, validation_children

    fixed.mkdir(parents=True, exist_ok=True)
    bootstrap = run_root / "bootstrap"
    bootstrap.mkdir(parents=True, exist_ok=True)
    initial = make_initial_state(
        models=models, episode=episode, paths=args.bootstrap_paths,
        firms=args.firms_per_path, horizon=args.bootstrap_horizon,
        branch_num=args.branch_num, seed=args.seed + 10, device=device,
    )
    output = run_simulation(
        models=models, episode=episode, initial_state=initial,
        paths=args.bootstrap_paths, firms=args.firms_per_path,
        horizon=args.bootstrap_horizon, branch_num=args.branch_num,
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
        "horizon": args.bootstrap_horizon,
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
        train_out, validation_out, _ = _simulation_pair(
            args=args, episode=episode, models=models,
            train_initial=train_initial, validation_initial=validation_initial,
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
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    m_min = float(getattr(hp, "pv_m_clamp_min", 0.7))
    m_max = float(getattr(hp, "pv_m_clamp_max", 1.3))
    train, train_diag = refresh_transition_bank(
        models["sdf_fc1"], train_bank.parent_states.to(device),
        train_children.children.to(device), m_min=m_min, m_max=m_max,
    )
    validation, validation_diag = refresh_transition_bank(
        models["sdf_fc1"], validation_bank.parent_states.to(device),
        validation_children.children.to(device), m_min=m_min, m_max=m_max,
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


def _one_step_target_movements(old: Any, new: Any) -> dict[str, float]:
    left = _to_frame(old).sort_values(["path", "t", "branch"]).reset_index(drop=True)
    right = _to_frame(new).sort_values(["path", "t", "branch"]).reset_index(drop=True)
    identity = ["path", "t", "branch"]
    if not left[identity].equals(right[identity]):
        raise RuntimeError("one-step target rows are not source-key aligned")
    result = {}
    for column, label in (("Hatc_t1", "Hatc"), ("LnK_t1", "LnK")):
        metrics = movement_metrics(
            torch.as_tensor(left[column].to_numpy()),
            torch.as_tensor(right[column].to_numpy()),
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
    ]
    for split, simulation in (("train", train_out), ("validation", validation_out)):
        macro = _to_frame(simulation.macro)
        firm = _to_frame(simulation.firm)
        atomic_write_csv(output / f"{split}_macro_paths.csv", macro.loc[:, keep_macro])
        diagnostics = macro_diagnostics(macro)
        diagnostics.update(simulation_policy_metrics(firm))
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


def _write_metadata(
    *, run_root: Path, args: argparse.Namespace, hp: Any,
    models: Mapping[str, torch.nn.Module], git_head: str,
    fingerprint: str, device: torch.device, bank_hashes: Mapping[str, str],
) -> None:
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
        "initialization_mode": "COLD_START",
        "branch": _git_branch(), "git_head": git_head,
        "hyperparameter_fingerprint": fingerprint,
        "args": vars(args), "hyperparams": vars(hp),
        "economic_config": AnalysisEconomicConfig.from_current_config().to_dict(),
        "bank_hashes": bank_hashes, "cuda": cuda,
        "fc2": "COMPLETELY_DISABLED",
        "sdf_mode": "SDF_TRUE_ONLY", "sdf_recursive_epochs": 0,
        "macro_primary_training_source": "MANY_PARENT_CROSS_SECTIONS_ONE_STEP",
        "macro_parent_generation": "SHORT_ENDOGENOUS_ROLLOUT_HARVEST",
        "macro_short_rollout_horizon": int(args.macro_bank_rollout_horizon),
        "macro_one_step_horizon": 1,
        "long_rollout_training_used": False,
        "long_rollout_horizon": int(args.long_diagnostic_horizon),
        "m_semantics": "CURRENT_SDF_CHILD shared by P and Q; one common clamp",
        "model_hashes_at_initialization": {
            name: module_hash(model) for name, model in models.items()
        },
    })


def _schedule_report(
    *,
    candidate: pd.DataFrame,
    selected: pd.DataFrame,
    schedule: MacroParentSourceSchedule,
    initial_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_count": int(len(candidate)),
        "requested_count": int(len(schedule.source_keys)),
        "selected_count": int(len(selected)),
        "endogenous_frequency_count": int(round(
            len(selected) * schedule.endogenous_frequency_share
        )),
        "coverage_aware_count": int(round(
            len(selected) * schedule.coverage_aware_share
        )),
        "endogenous_frequency_share": schedule.endogenous_frequency_share,
        "coverage_aware_share": schedule.coverage_aware_share,
        "candidate_shortfall": 0,
        "schedule_fingerprint": schedule.fingerprint,
        "candidate_coverage": distribution_coverage(candidate),
        "selected_coverage": distribution_coverage(selected),
        "unique_source_paths": int(selected["initial_path_id"].nunique()),
        "capture_time_distribution": {
            str(int(key)): int(value)
            for key, value in selected["capture_time"].value_counts().sort_index().items()
        },
        "initial_cross_sections": dict(initial_audit),
    }


def _build_or_refresh_macro_parent_context(
    *,
    args: argparse.Namespace,
    run_root: Path,
    outer_root: Path,
    equilibrium_outer: int,
    models: Mapping[str, torch.nn.Module],
    episode: Episode,
    device: torch.device,
) -> tuple[
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
    MacroParentSourceSchedule,
    MacroParentSourceSchedule,
    Any,
    Any,
]:
    parent_root = outer_root / "macro_parent_bank"
    parent_root.mkdir(parents=True, exist_ok=True)
    requested_times = parse_capture_times(args.macro_bank_capture_times)

    train_seed = int(args.seed + 30_000 + equilibrium_outer * 100_000)
    train_transition_seed = train_seed + 11
    train_entry_seed = train_seed + 12
    train_initial_path = parent_root / "train_initial_state.pt"
    if train_initial_path.is_file():
        macro_train_initial = torch.load(
            train_initial_path, map_location="cpu", weights_only=True
        )
    else:
        macro_train_initial = make_initial_state(
            models=models,
            episode=episode,
            paths=args.macro_bank_candidate_paths,
            firms=args.firms_per_path,
            horizon=args.macro_bank_rollout_horizon,
            branch_num=args.branch_num,
            seed=train_seed,
            device=device,
        )
        atomic_torch_save(train_initial_path, macro_train_initial)

    train_schedule_path = parent_root / "train_source_schedule.json"
    if train_schedule_path.is_file():
        train_schedule = load_schedule(train_schedule_path)
        train_harvest, train_candidate, train_candidate_occupancy = (
            harvest_macro_parent_schedule_with_candidate(
                models=models,
                episode=episode,
                initial_state=macro_train_initial,
                schedule=train_schedule,
                source_episode=equilibrium_outer,
                branch_num=args.branch_num,
                device=device,
            )
        )
        train_report = _schedule_report(
            candidate=train_candidate,
            selected=train_harvest.summary,
            schedule=train_schedule,
            initial_audit=train_harvest.initial_hash_audit,
        )
    else:
        (
            train_schedule,
            train_harvest,
            train_candidate,
            train_candidate_occupancy,
            train_report,
        ) = prepare_macro_parent_schedule(
            models=models,
            episode=episode,
            initial_state=macro_train_initial,
            split=f"train_outer_{equilibrium_outer:03d}",
            source_episode=equilibrium_outer,
            initial_state_seed=train_seed,
            transition_seed=train_transition_seed,
            entry_seed=train_entry_seed,
            short_rollout_horizon=args.macro_bank_rollout_horizon,
            capture_times=requested_times,
            branch_num=args.branch_num,
            target_count=args.macro_train_parent_count,
            coverage_fraction=args.macro_bank_coverage_fraction,
            selection_seed=train_seed + 21,
            device=device,
        )
        save_schedule(train_schedule_path, train_schedule)

    global_validation_root = run_root / "fixed_point" / "macro_global_validation"
    global_validation_root.mkdir(parents=True, exist_ok=True)
    validation_seed = int(args.seed + 40_000)
    validation_initial_path = global_validation_root / "initial_state.pt"
    if validation_initial_path.is_file():
        macro_validation_initial = torch.load(
            validation_initial_path, map_location="cpu", weights_only=True
        )
    else:
        macro_validation_initial = make_initial_state(
            models=models,
            episode=episode,
            paths=args.macro_bank_validation_paths,
            firms=args.firms_per_path,
            horizon=args.macro_bank_rollout_horizon,
            branch_num=args.branch_num,
            seed=validation_seed,
            device=device,
        )
        atomic_torch_save(validation_initial_path, macro_validation_initial)

    validation_schedule_path = global_validation_root / "validation_source_schedule.json"
    if validation_schedule_path.is_file():
        validation_schedule = load_schedule(validation_schedule_path)
        validation_harvest, validation_candidate, validation_candidate_occupancy = (
            harvest_macro_parent_schedule_with_candidate(
                models=models,
                episode=episode,
                initial_state=macro_validation_initial,
                schedule=validation_schedule,
                source_episode=equilibrium_outer,
                branch_num=args.branch_num,
                device=device,
            )
        )
        validation_report = _schedule_report(
            candidate=validation_candidate,
            selected=validation_harvest.summary,
            schedule=validation_schedule,
            initial_audit=validation_harvest.initial_hash_audit,
        )
    else:
        (
            validation_schedule,
            validation_harvest,
            validation_candidate,
            validation_candidate_occupancy,
            validation_report,
        ) = prepare_macro_parent_schedule(
            models=models,
            episode=episode,
            initial_state=macro_validation_initial,
            split="global_validation",
            source_episode=equilibrium_outer,
            initial_state_seed=validation_seed,
            transition_seed=validation_seed + 11,
            entry_seed=validation_seed + 12,
            short_rollout_horizon=args.macro_bank_rollout_horizon,
            capture_times=requested_times,
            branch_num=args.branch_num,
            target_count=args.macro_validation_parent_count,
            coverage_fraction=args.macro_bank_coverage_fraction,
            selection_seed=validation_seed + 21,
            device=device,
        )
        save_schedule(validation_schedule_path, validation_schedule)
    save_schedule(parent_root / "validation_source_schedule.json", validation_schedule)

    split_audit = source_overlap(train_schedule, validation_schedule)
    if split_audit["overlap_count"]:
        raise RuntimeError(f"macro train/validation parent leakage: {split_audit}")
    write_coverage_artifacts(
        parent_root,
        train_candidate=train_candidate,
        train_selected=train_harvest.summary,
        validation_candidate=validation_candidate,
        validation_selected=validation_harvest.summary,
        train_candidate_occupancy=train_candidate_occupancy,
        train_selected_occupancy=train_harvest.occupancy,
        validation_candidate_occupancy=validation_candidate_occupancy,
        validation_selected_occupancy=validation_harvest.occupancy,
        train_schedule=train_schedule,
        validation_schedule=validation_schedule,
        train_report=train_report,
        validation_report=validation_report,
    )
    return (
        macro_train_initial,
        macro_validation_initial,
        train_schedule,
        validation_schedule,
        train_harvest,
        validation_harvest,
    )


def _resolve_resume(args: argparse.Namespace) -> tuple[Optional[Path], Optional[Path]]:
    if args.resume is None:
        return None, None
    supplied = args.resume.expanduser().resolve()
    if supplied.suffix.lower() == ".json":
        pointer = json.loads(supplied.read_text(encoding="utf-8"))
        target = pointer.get("checkpoint") or pointer.get("latest")
        if not target:
            raise RuntimeError(f"resume JSON has no checkpoint/latest pointer: {supplied}")
        checkpoint = Path(target).expanduser().resolve()
    else:
        checkpoint = supplied
    for candidate in (supplied.parent, *supplied.parents):
        if (candidate / "metadata" / "run.json").is_file() and (
            candidate / "fixed_point"
        ).is_dir():
            return candidate, checkpoint
    raise RuntimeError(f"cannot locate nested-equilibrium run root above {supplied}")


def _apply_smoke(args: argparse.Namespace) -> None:
    args.equilibrium_outer_max = 1
    args.firm_inner_max = 1
    args.p_policy_outer_max = 1
    args.q_picard_outer_max = 1
    args.macro_inner_max = 1
    args.simulation_paths = 4
    args.validation_paths = 4
    args.bootstrap_paths = 4
    args.firms_per_path = 32
    args.simulation_horizon = 8
    args.bootstrap_horizon = 8
    args.macro_bank_candidate_paths = 8
    args.macro_bank_validation_paths = 4
    args.macro_bank_rollout_horizon = 3
    args.macro_bank_capture_times = "0,1,2"
    args.macro_train_parent_count = 16
    args.macro_validation_parent_count = 8
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
    parser.add_argument("--firms-per-path", type=int, default=200)
    parser.add_argument("--simulation-horizon", type=int, default=100, help="deprecated; not used for primary Macro training")
    parser.add_argument("--bootstrap-horizon", type=int, default=100)
    parser.add_argument("--branch-num", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=8192)
    parser.add_argument("--validation-limit", type=int, default=2048)
    parser.add_argument("--child-shocks", type=int, default=16)
    parser.add_argument("--macro-probe-size", type=int, default=2048)
    parser.add_argument("--macro-batch-size", type=int, default=4096)
    parser.add_argument("--macro-bank-candidate-paths", type=int, default=1024)
    parser.add_argument("--macro-bank-validation-paths", type=int, default=256)
    parser.add_argument("--macro-bank-rollout-horizon", type=int, default=8)
    parser.add_argument("--macro-bank-capture-times", default="0,2,4,6,8")
    parser.add_argument("--macro-train-parent-count", type=int, default=4096)
    parser.add_argument("--macro-validation-parent-count", type=int, default=1024)
    parser.add_argument("--macro-bank-candidate-multiplier", type=float, default=1.5)
    parser.add_argument("--macro-bank-coverage-fraction", type=float, default=0.20)
    parser.add_argument("--macro-parent-batch-size", type=int, default=256)
    parser.add_argument("--macro-bank-shard-size", type=int, default=128)
    parser.add_argument("--long-diagnostic-paths", type=int, default=50)
    parser.add_argument("--long-diagnostic-horizon", type=int, default=100)
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
    parse_capture_times(args.macro_bank_capture_times)
    if not 0.0 <= args.macro_bank_coverage_fraction <= 1.0:
        parser.error("--macro-bank-coverage-fraction must be in [0, 1]")
    for name in ("equilibrium_outer_max", "firm_inner_max", "p_policy_outer_max",
                 "q_picard_outer_max", "macro_inner_max"):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
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
    resume_root, resume_checkpoint = _resolve_resume(args)
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
            resume_checkpoint, models=models, git_head=git_head,
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
    if resume_checkpoint is not None and expected_banks != bank_hashes:
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
        train_parents = train_bank.parent_states.to(device)
        validation_parents = validation_bank.parent_states.to(device)
        train_children, validation_children, m_diag = _refresh_children(
            models=models, episode=episode, hp=hp, train_bank=train_bank,
            validation_bank=validation_bank, train_children=raw_train_children,
            validation_children=raw_validation_children, device=device,
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
                    block_start = _model_states(models)

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
                        reload_block_start=lambda: _restore_model_states(models, block_start),
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
                        models=models, state=next_state, git_head=git_head,
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
                block_start = _model_states(models)

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
                    reload_block_start=lambda: _restore_model_states(models, block_start),
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
                    models=models, state=next_state, git_head=git_head,
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
                models=models, state=state, git_head=git_head,
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
            block_start = _model_states(models)

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
                reload_block_start=lambda: _restore_model_states(models, block_start),
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
                models=models, state=state, git_head=git_head,
                fingerprint=fingerprint, bank_hashes=bank_hashes,
                extra={"policy_summary": policy_summary},
            )
            append_summary_row(summary_path, {
                "stage": "policy_distill", "equilibrium_outer": equilibrium_outer,
                "GPU_max_memory": cuda_memory(device)["max_memory_allocated"],
                "total_elapsed": time.time() - started_at,
            })

        firm_hash_for_macro = firm_hashes(models["policy_value"])
        macro_gate_streak = int(state.get("macro_gate_streak", 0))
        macro_reached = macro_gate_streak >= 2
        start_macro_inner = int(state.get("macro_inner", 0))
        probe_parents = macro_probe["parents"].to(device)
        probe_children = macro_probe["children"].to(device)
        current_functions = snapshot_functions(models["sdf_fc1"], probe_parents, probe_children)
        macro_outer_start_functions = {
            key: value.detach().cpu().clone() for key, value in current_functions.items()
        }
        function_history_path = macro_root / "macro_probe_history.pt"
        if start_macro_inner > 0 and function_history_path.is_file():
            function_history = torch.load(
                function_history_path, map_location="cpu", weights_only=True
            )
            if not function_history:
                raise RuntimeError("resume macro probe history is empty")
            for key in current_functions:
                if not torch.equal(
                    current_functions[key].detach().cpu(), function_history[-1][key]
                ):
                    raise RuntimeError(f"resume macro probe mismatch: {key}")
        else:
            function_history = [
                {key: value.detach().cpu() for key, value in current_functions.items()}
            ]
        (
            macro_train_initial,
            macro_validation_initial,
            train_source_schedule,
            validation_source_schedule,
            current_train_harvest,
            current_validation_harvest,
        ) = _build_or_refresh_macro_parent_context(
            args=args,
            run_root=run_root,
            outer_root=outer_root,
            equilibrium_outer=equilibrium_outer,
            models=models,
            episode=episode,
            device=device,
        )
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
        current_train_dataset = build_one_step_macro_dataset(
            models=models,
            episode=episode,
            parent_bank=current_train_harvest,
            parent_batch_size=args.macro_parent_batch_size,
            macro_batch_size=args.macro_batch_size,
            branch_num=args.branch_num,
            transition_seed=train_source_schedule.transition_seed + 1000,
            entry_seed=train_source_schedule.entry_seed + 1000,
            device=device,
            path_offset=0,
        )
        current_validation_dataset = build_one_step_macro_dataset(
            models=models,
            episode=episode,
            parent_bank=current_validation_harvest,
            parent_batch_size=args.macro_parent_batch_size,
            macro_batch_size=args.macro_batch_size,
            branch_num=args.branch_num,
            transition_seed=validation_source_schedule.transition_seed + 1000,
            entry_seed=validation_source_schedule.entry_seed + 1000,
            device=device,
            path_offset=1_000_000,
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
            atomic_write_json(
                macro_inner_root / "input_parent_bank_metadata.json",
                {
                    "macro_primary_training_source": "MANY_PARENT_CROSS_SECTIONS_ONE_STEP",
                    "train_parent_count": current_train_harvest.n_parents,
                    "validation_parent_count": current_validation_harvest.n_parents,
                    "short_rollout_horizon": args.macro_bank_rollout_horizon,
                    "effective_capture_times": list(effective_capture_times(
                        args.macro_bank_capture_times,
                        args.macro_bank_rollout_horizon,
                    )),
                    "one_step_horizon": 1,
                    "branch_num": args.branch_num,
                    "long_rollout_training_used": False,
                    "train_schedule_fingerprint": train_source_schedule.fingerprint,
                    "validation_schedule_fingerprint": validation_source_schedule.fingerprint,
                    "firm_hashes": firm_hash_for_macro,
                },
            )
            atomic_write_json(
                macro_inner_root / "one_step_pairing_audit.json",
                {
                    "train": current_train_dataset.pairing_audit,
                    "validation": current_validation_dataset.pairing_audit,
                    "train_validation_source_split": source_overlap(
                        train_source_schedule, validation_source_schedule
                    ),
                },
            )
            if macro_inner == 0:
                save_bank_shards(
                    macro_root / "macro_parent_bank" / "macro_inner_000_train",
                    current_train_harvest,
                    shard_size=args.macro_bank_shard_size,
                )
                save_bank_shards(
                    macro_root / "macro_parent_bank" / "macro_inner_000_validation",
                    current_validation_harvest,
                    shard_size=args.macro_bank_shard_size,
                )
            block_start = _model_states(models)
            old_functions = current_functions
            old_validation_frame = current_validation_frame
            old_validation_pairs = current_validation_dataset.pairs
            old_validation_parent_summary = current_validation_harvest.summary

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
                train_sdf_batches = episode._create_sdf_batches_from_macro_tensor(
                    current_train_dataset.pairs,
                    batch_size=batch_size,
                    n_branches=args.branch_num,
                    target_num_batches=target_batches,
                    min_parent_groups_per_batch=min_parents,
                )
                validation_sdf_batches = episode._create_sdf_batches_from_macro_tensor(
                    current_validation_dataset.pairs,
                    batch_size=batch_size,
                    n_branches=args.branch_num,
                )
                fc1_summary = run_fc1_fit(
                    episode,
                    train_fc1_batches,
                    validation_fc1_batches,
                    macro_inner_root / "fc1", block="joint",
                    production_epochs=args.fc1_production_epochs,
                    max_epochs=args.fc1_max_epochs, patience=args.fc1_patience,
                    log_interval=1_000_000,
                )
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
                atomic_write_json(macro_inner_root / "sdf" / "summary.json", sdf_summary)
                simulation_started = time.perf_counter()
                post_train_harvest = harvest_macro_parent_schedule(
                    models=models,
                    episode=episode,
                    initial_state=macro_train_initial,
                    schedule=train_source_schedule,
                    source_episode=equilibrium_outer,
                    branch_num=args.branch_num,
                    device=device,
                )
                post_validation_harvest = harvest_macro_parent_schedule(
                    models=models,
                    episode=episode,
                    initial_state=macro_validation_initial,
                    schedule=validation_source_schedule,
                    source_episode=equilibrium_outer,
                    branch_num=args.branch_num,
                    device=device,
                )
                post_train_dataset = build_one_step_macro_dataset(
                    models=models,
                    episode=episode,
                    parent_bank=post_train_harvest,
                    parent_batch_size=args.macro_parent_batch_size,
                    macro_batch_size=batch_size,
                    branch_num=args.branch_num,
                    transition_seed=train_source_schedule.transition_seed + 1000,
                    entry_seed=train_source_schedule.entry_seed + 1000,
                    device=device,
                    path_offset=0,
                )
                post_validation_dataset = build_one_step_macro_dataset(
                    models=models,
                    episode=episode,
                    parent_bank=post_validation_harvest,
                    parent_batch_size=args.macro_parent_batch_size,
                    macro_batch_size=batch_size,
                    branch_num=args.branch_num,
                    transition_seed=validation_source_schedule.transition_seed + 1000,
                    entry_seed=validation_source_schedule.entry_seed + 1000,
                    device=device,
                    path_offset=1_000_000,
                )
                return (
                    fc1_summary,
                    sdf_summary,
                    post_train_harvest,
                    post_validation_harvest,
                    post_train_dataset,
                    post_validation_dataset,
                    time.perf_counter() - simulation_started,
                )

            result, used_batch = run_with_oom_fallback(
                block="MACRO", initial_batch_size=args.macro_batch_size,
                fallback_batch_size=max(1, args.macro_batch_size // 2),
                execute=run_macro,
                reload_block_start=lambda: _restore_model_states(models, block_start),
                events=oom_events, events_path=oom_events_path, device=device,
                indices={"equilibrium_outer": equilibrium_outer, "firm_inner": -1, "macro_inner": macro_inner},
            )
            (
                fc1_summary,
                sdf_summary,
                current_train_harvest,
                current_validation_harvest,
                current_train_dataset,
                current_validation_dataset,
                simulation_seconds,
            ) = result
            atomic_write_json(
                macro_inner_root / "output_parent_bank_metadata.json",
                {
                    "train_bank_hash": table_hash(current_train_dataset.macro),
                    "validation_bank_hash": table_hash(current_validation_dataset.macro),
                    "train_pair_rows": len(current_train_dataset.pairs),
                    "validation_pair_rows": len(current_validation_dataset.pairs),
                    "train_parent_batch_size": current_train_dataset.used_parent_batch_size,
                    "validation_parent_batch_size": current_validation_dataset.used_parent_batch_size,
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
            target_moves = _one_step_target_movements(
                old_validation_pairs, current_validation_dataset.pairs
            )
            parent_moves = parent_summary_movement(
                old_validation_parent_summary, current_validation_harvest.summary
            )
            atomic_write_json(
                macro_inner_root / "parent_bank_summary_movement.json", parent_moves
            )
            macro_diag = macro_diagnostics(current_validation_frame)
            macro_diag.update(simulation_policy_metrics(
                _to_frame(current_validation_harvest.simulation_output.firm)
            ))
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
                    value for value in parent_moves.values() if math.isfinite(value)
                ),
                "FC1_C_val_R2": fc1_summary["best_validation"].get("hatc_r2", float("nan")),
                "FC1_K_val_R2": fc1_summary["best_validation"].get("lnk_r2", float("nan")),
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
            write_macro_inner_movement_figure(
                pd.read_csv(macro_history_path),
                macro_root / "macro_parent_bank" / "figures",
            )
            atomic_torch_save(macro_root / "macro_probe_history.pt", function_history)
            state = {
                **state, "equilibrium_outer": equilibrium_outer,
                "phase": "macro_inner", "macro_inner": macro_inner + 1,
                "sdf_chunk": args.sdf_chunks, "macro_gate_streak": macro_gate_streak,
            }
            _save_stage_checkpoint(
                run_root=run_root,
                path=macro_inner_root / "macro_inner_complete.pt",
                models=models, state=state, git_head=git_head,
                fingerprint=fingerprint, bank_hashes=bank_hashes,
                extra={"row": last_macro_row, "firm_hashes": firm_hash_for_macro},
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
        save_bank_shards(
            macro_root / "macro_parent_bank" / "final_train",
            current_train_harvest,
            shard_size=args.macro_bank_shard_size,
        )
        save_bank_shards(
            macro_root / "macro_parent_bank" / "final_validation",
            current_validation_harvest,
            shard_size=args.macro_bank_shard_size,
        )
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
        long_path_moves = _path_and_target_movements(
            _to_frame(long_start_validation.macro),
            _to_frame(long_final_validation.macro),
        )
        atomic_write_json(
            macro_root / "long_diagnostic_movement.json",
            {**long_path_moves, "simulation_seconds": long_final_seconds},
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
            **outer_moves,
            **{key: last_macro_row.get(key, float("nan")) for key in (
                "M_function_move", "FC1_C_function_move", "FC1_K_function_move",
                "Hatc_target_move", "LnK_target_move", "Hatc_path_move",
                "LnK_path_move", "C_path_move", "K_path_move", "FC1_C_val_R2",
                "FC1_K_val_R2", "SDF_objective", "SDF_moment1", "SDF_moment2",
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
        append_summary_row(summary_path, summary_row)
        next_state = {
            **state, "equilibrium_outer": equilibrium_outer + 1,
            "phase": "firm_p", "firm_inner": 0, "p_policy_outer": 0,
            "q_picard_outer": 0, "macro_inner": 0, "sdf_chunk": 0,
            "firm_gate_streak": 0, "macro_gate_streak": 0,
            "overall_gate_streak": overall_gate_streak,
        }
        _save_stage_checkpoint(
            run_root=run_root,
            path=checkpoints / "equilibrium_outer_complete.pt",
            models=models, state=next_state, git_head=git_head,
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
        models=models, state=final_state, git_head=git_head,
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

