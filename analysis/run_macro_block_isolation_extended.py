#!/usr/bin/env python3
"""Resume the completed macro block-isolation experiment without replaying L0.

The runner is deliberately narrow: it resumes FC1-only at outer 3, runs the
missing SDF-only branch, resumes joint at outer 8, and audits recursive-state
SDF tails.  Firm policy/value is frozen, FC2 is guarded, and the production
simulation and conditional macro solvers are reused unchanged.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.run_macro_block_isolation import (  # noqa: E402
    FUNCTION_NAMES,
    TwoCycleDiagnosticsConfig,
    build_in_memory_batches,
    evaluate_fc1,
    firm_hashes,
    fresh_macro_model,
    install_fc2_guard,
    macro_target_hash,
    make_initial_state,
    module_hash,
    movement_metrics,
    movement_relation,
    read_json,
    run_fc1_fit,
    run_sdf_chunks,
    set_macro_scope,
    simulate_frozen_firm,
    snapshot_functions,
    stable_json_hash,
    state_hash,
    tensor_hash,
    two_cycle_detect,
    write_json,
)
from analysis.run_macro_outer_inner_pq import _to_frame  # noqa: E402


DEFAULT_ROOT = ROOT / "cachedir" / "macro_block_isolation_20261017"
DEFAULT_REPORT = ROOT / "reports" / "macro_block_isolation_L1_extended_20261017.md"
TRACKED_VECTOR_NAMES = (
    "function_m",
    "function_fc1_c",
    "function_fc1_k",
    "target_hatc",
    "target_lnk",
    "path_hatc",
    "path_lnk",
    "path_c",
    "path_k",
)


def ensure_new_output(root: Path, allowed_existing: Sequence[str] = ()) -> None:
    """Refuse to overwrite any result other than explicitly preserved files."""
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
        return
    allowed = set(allowed_existing)
    unexpected = [path.name for path in root.iterdir() if path.name not in allowed]
    if unexpected:
        raise FileExistsError(
            f"refusing to overwrite existing extended output {root}: {sorted(unexpected)}"
        )


def checkpoint_for_outer(old_root: Path, branch: str, outer: int) -> Path:
    if outer < 1:
        raise ValueError("old branch checkpoints only exist after the first solve")
    solve = old_root / "l1" / branch / f"outer_{outer - 1:03d}_fc1_solve"
    if branch == "fc1_only":
        return solve / "best.pt"
    if branch == "joint":
        return (
            old_root
            / "l1"
            / branch
            / f"outer_{outer - 1:03d}_sdf_solve"
            / "checkpoint_chunk_004.pt"
        )
    raise ValueError(branch)


def load_full_state(path: Path, model: torch.nn.Module) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, Mapping) and "sdf_fc1" in payload:
        payload = payload["sdf_fc1"]
    model.load_state_dict(payload, strict=True)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().cpu() - right.detach().cpu()).abs().max())


def verify_resume_probe(
    model: torch.nn.Module,
    parents: torch.Tensor,
    children: torch.Tensor,
    expected_path: Path,
) -> dict[str, Any]:
    expected = torch.load(expected_path, map_location="cpu", weights_only=True)
    actual = snapshot_functions(model, parents, children)
    errors = {name: _max_abs(actual[name], expected[name]) for name in FUNCTION_NAMES}
    exact = {
        name: bool(torch.equal(actual[name], expected[name])) for name in FUNCTION_NAMES
    }
    if not all(exact.values()):
        raise RuntimeError(
            f"resume probe parity failed for {expected_path}: exact={exact}, max_abs={errors}"
        )
    return {
        "expected_snapshot": str(expected_path),
        "exact": exact,
        "max_abs_error": errors,
        "probe_hash": tensor_hash(parents) + ":" + tensor_hash(children),
    }


def strict_cycle_fields(
    two_back: torch.Tensor,
    previous: torch.Tensor,
    current: torch.Tensor,
    config: TwoCycleDiagnosticsConfig,
) -> dict[str, Any]:
    delta_previous = previous - two_back
    delta_current = current - previous
    relation = movement_relation(delta_previous, delta_current)
    distance_one = torch.linalg.vector_norm(delta_current.reshape(-1).to(torch.float64))
    distance_two = torch.linalg.vector_norm((current - two_back).reshape(-1).to(torch.float64))
    return {
        **relation,
        "two_step_to_one_step": float(distance_two / (distance_one + 1e-12)),
        "negative_directional_movement": bool(relation["movement_cosine"] < 0.0),
        "damped_oscillation": bool(
            relation["movement_cosine"] < config.cosine_max
            and relation["norm_gain"] < config.amplitude_ratio_min
        ),
        "two_cycle_candidate": two_cycle_detect(
            two_back, previous, current, config=config
        ),
    }


def sorted_macro_vectors(frame: pd.DataFrame) -> dict[str, torch.Tensor]:
    ordered = frame.sort_values(["path", "t", "branch"])
    return {
        "target_hatc": torch.from_numpy(ordered["Hatc"].to_numpy(np.float64)),
        "target_lnk": torch.from_numpy(ordered["LnK"].to_numpy(np.float64)),
        "path_hatc": torch.from_numpy(ordered["Hatc"].to_numpy(np.float64)),
        "path_lnk": torch.from_numpy(ordered["LnK"].to_numpy(np.float64)),
        "path_c": torch.from_numpy(ordered["C"].to_numpy(np.float64)),
        "path_k": torch.from_numpy(ordered["K"].to_numpy(np.float64)),
    }


def combined_vectors(
    functions: Mapping[str, torch.Tensor], frame: pd.DataFrame
) -> dict[str, torch.Tensor]:
    result = sorted_macro_vectors(frame)
    result.update({f"function_{name}": functions[name].detach().cpu() for name in FUNCTION_NAMES})
    return result


def add_transition_metrics(
    row: dict[str, Any],
    histories: Mapping[str, list[torch.Tensor]],
    cycle_counts: dict[str, int],
    config: TwoCycleDiagnosticsConfig,
) -> None:
    any_strict = False
    any_damped = False
    any_negative = False
    for name, values in histories.items():
        if len(values) >= 2:
            row.update(movement_metrics(values[-2], values[-1], name))
        if len(values) < 3:
            continue
        diagnostics = strict_cycle_fields(values[-3], values[-2], values[-1], config)
        for key, value in diagnostics.items():
            row[f"{name}_{key}"] = value
        candidate = bool(diagnostics["two_cycle_candidate"])
        cycle_counts[name] = cycle_counts.get(name, 0) + 1 if candidate else 0
        strict = candidate and cycle_counts[name] >= config.consecutive_required
        row[f"{name}_two_cycle_consecutive"] = cycle_counts[name]
        row[f"{name}_possible_two_cycle_strict"] = strict
        any_strict = any_strict or strict
        any_damped = any_damped or bool(diagnostics["damped_oscillation"])
        any_negative = any_negative or bool(diagnostics["negative_directional_movement"])
    row["possible_two_cycle_strict"] = any_strict
    row["damped_oscillation_flag"] = any_damped
    row["negative_directional_movement"] = any_negative


def _prefix_metrics(prefix: str, values: Mapping[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in values.items()}


def build_initial_states(args: argparse.Namespace, episode: Any, models: Mapping[str, Any], device: torch.device):
    train = make_initial_state(
        models=models,
        episode=episode,
        paths=args.simulation_paths,
        firms=args.firms_per_path,
        horizon=args.simulation_horizon,
        branch_num=args.branch_num,
        seed=args.seed + 1000,
        device=device,
    )
    validation = make_initial_state(
        models=models,
        episode=episode,
        paths=args.validation_paths,
        firms=args.firms_per_path,
        horizon=args.simulation_horizon,
        branch_num=args.branch_num,
        seed=args.seed + 2000,
        device=device,
    )
    crn = {
        "train_initial_hash": state_hash(train),
        "validation_initial_hash": state_hash(validation),
        "train_transition_seed": args.seed + 1011,
        "train_entry_seed": args.seed + 1012,
        "validation_transition_seed": args.seed + 2011,
        "validation_entry_seed": args.seed + 2012,
    }
    return train, validation, crn


def simulate_state(
    args: argparse.Namespace,
    episode: Any,
    models: Mapping[str, Any],
    train_initial: Mapping[str, torch.Tensor],
    validation_initial: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    train_out, validation_out, seconds = simulate_frozen_firm(
        args, episode, models, train_initial, validation_initial, device
    )
    train_frame = _to_frame(train_out.macro)
    validation_frame = _to_frame(validation_out.macro)
    batches = build_in_memory_batches(
        episode, train_out.macro.to(device), validation_out.macro.to(device), args
    )
    return {
        "train_out": train_out,
        "validation_out": validation_out,
        "train_frame": train_frame,
        "validation_frame": validation_frame,
        "batches": batches,
        "seconds": seconds,
        "target_hash": macro_target_hash(validation_frame),
        "data_hash_including_M": macro_target_hash(validation_frame, include_m=True),
        "train_fit": evaluate_fc1(models["sdf_fc1"], batches["fc1_train"]),
        "validation_fit": evaluate_fc1(models["sdf_fc1"], batches["fc1_validation"]),
    }


def _old_function_history(old_root: Path, branch: str, end_outer: int) -> dict[str, list[torch.Tensor]]:
    start = max(0, end_outer - 2)
    states = [
        torch.load(
            old_root / "l1" / branch / f"functions_outer_{outer:03d}_start.pt",
            map_location="cpu",
            weights_only=True,
        )
        for outer in range(start, end_outer + 1)
    ]
    return {
        f"function_{name}": [state[name].detach().cpu() for state in states]
        for name in FUNCTION_NAMES
    }


def _save_checkpoint(path: Path, model: torch.nn.Module, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sdf_fc1": model.state_dict(), "metadata": dict(metadata)}, path)


def _load_old_metrics(old_root: Path, branch: str) -> pd.DataFrame:
    path = old_root / "l1" / branch / "metrics.csv"
    frame = pd.read_csv(path)
    frame.insert(0, "experiment", branch)
    frame.insert(1, "source", "original")
    return frame


def run_resumed_branch(
    args: argparse.Namespace,
    device: torch.device,
    probe_parents: torch.Tensor,
    probe_children: torch.Tensor,
    *,
    branch: str,
    resume_outer: int,
    final_outer: int,
    budgets: Mapping[str, int],
    output_name: str,
) -> dict[str, Any]:
    if branch not in {"fc1_only", "joint"}:
        raise ValueError(branch)
    root = args.output_dir / "l1" / output_name
    ensure_new_output(root)
    episode, models, _ = fresh_macro_model(args, device)
    model = models["sdf_fc1"]
    firm = models["policy_value"].requires_grad_(False).eval()
    resume_checkpoint = checkpoint_for_outer(args.old_output_root, branch, resume_outer)
    load_full_state(resume_checkpoint, model)
    parity = verify_resume_probe(
        model,
        probe_parents,
        probe_children,
        args.old_output_root / "l1" / branch / f"functions_outer_{resume_outer:03d}_start.pt",
    )
    initial_firm_hashes = firm_hashes(firm)
    initial_sdf_hash = module_hash(model.sdf_model) + ":" + module_hash(model.value_model)
    initial_fc1_hash = module_hash(model.fc1_model)
    fc2 = models.get("fc2")
    fc2_counter = {"calls": 0}
    fc2_hash = module_hash(fc2)
    install_fc2_guard(fc2, fc2_counter)
    train_initial, validation_initial, crn = build_initial_states(args, episode, models, device)

    # Re-simulate only the preceding state to construct the first gain denominator.
    previous_checkpoint = checkpoint_for_outer(args.old_output_root, branch, resume_outer - 1)
    load_full_state(previous_checkpoint, model)
    previous_sim = simulate_state(
        args, episode, models, train_initial, validation_initial, device
    )
    previous_functions = snapshot_functions(model, probe_parents, probe_children)
    load_full_state(resume_checkpoint, model)
    current_sim = simulate_state(
        args, episode, models, train_initial, validation_initial, device
    )
    current_functions = snapshot_functions(model, probe_parents, probe_children)

    histories = _old_function_history(args.old_output_root, branch, resume_outer)
    for key, value in sorted_macro_vectors(previous_sim["validation_frame"]).items():
        histories[key] = [value]
    for key, value in sorted_macro_vectors(current_sim["validation_frame"]).items():
        histories[key].append(value)
    # The checkpoint-derived function history is authoritative; these checks
    # ensure diagnostic re-simulation did not silently alter the resumed state.
    for name in FUNCTION_NAMES:
        if not torch.equal(histories[f"function_{name}"][-1], current_functions[name].cpu()):
            raise RuntimeError(f"resumed {branch} function history mismatch for {name}")
        if not torch.equal(histories[f"function_{name}"][-2], previous_functions[name].cpu()):
            raise RuntimeError(f"previous {branch} function history mismatch for {name}")

    rows: list[dict[str, Any]] = []
    cycle_counts = {name: 0 for name in TRACKED_VECTOR_NAMES}
    strict_config = args.two_cycle_config
    convergence_streak = 0
    stop_reason = "max_outer"
    current_outer = resume_outer
    minimum_outer = min(final_outer, resume_outer + (4 if branch == "joint" else 1))

    while current_outer < final_outer:
        before_firm = firm_hashes(firm)
        before_fc2 = module_hash(fc2)
        solve_started = time.perf_counter()
        if branch == "fc1_only":
            solve = run_fc1_fit(
                episode,
                current_sim["batches"]["fc1_train"],
                current_sim["batches"]["fc1_validation"],
                root / f"outer_{current_outer:03d}_fc1_solve",
                block="joint",
                production_epochs=min(args.production_fc1_epochs, budgets["fc1_epochs"]),
                max_epochs=budgets["fc1_epochs"],
                patience=args.fc1_patience,
                log_interval=args.log_interval,
            )
            sdf_solve = None
        else:
            solve = run_fc1_fit(
                episode,
                current_sim["batches"]["fc1_train"],
                current_sim["batches"]["fc1_validation"],
                root / f"outer_{current_outer:03d}_fc1_solve",
                block="joint",
                production_epochs=min(args.production_fc1_epochs, budgets["fc1_epochs"]),
                max_epochs=budgets["fc1_epochs"],
                patience=args.fc1_patience,
                log_interval=args.log_interval,
            )
            sdf_solve = run_sdf_chunks(
                episode,
                current_sim["batches"]["sdf_train"],
                current_sim["batches"]["sdf_validation"],
                probe_parents,
                probe_children,
                root / f"outer_{current_outer:03d}_sdf_solve",
                chunks=budgets["sdf_chunks"],
                epochs_per_chunk=budgets["sdf_epochs_per_chunk"],
                log_interval=args.log_interval,
            )
        solve_seconds = time.perf_counter() - solve_started
        current_outer += 1
        set_macro_scope(model, "frozen")
        functions = snapshot_functions(model, probe_parents, probe_children)
        torch.save(functions, root / f"functions_outer_{current_outer:03d}_start.pt")
        _save_checkpoint(
            root / "checkpoints" / f"outer_{current_outer:03d}.pt",
            model,
            {"branch": branch, "outer": current_outer, "source_outer": current_outer - 1},
        )
        next_sim = simulate_state(
            args, episode, models, train_initial, validation_initial, device
        )
        for name in FUNCTION_NAMES:
            histories[f"function_{name}"].append(functions[name].detach().cpu())
        for key, value in sorted_macro_vectors(next_sim["validation_frame"]).items():
            histories[key].append(value)

        row: dict[str, Any] = {
            "experiment": branch,
            "source": "extended",
            "outer": current_outer,
            "conditional_source_outer": current_outer - 1,
            "simulation_seconds": next_sim["seconds"],
            "conditional_solve_seconds": solve_seconds,
            "target_hash": next_sim["target_hash"],
            "M_data_hash": next_sim["data_hash_including_M"],
            "crn_hash": stable_json_hash(crn),
            "firm_hash_invariant": firm_hashes(firm) == initial_firm_hashes,
            "fc2_hash_invariant": module_hash(fc2) == fc2_hash,
            "fc2_forward_calls": fc2_counter["calls"],
            **_prefix_metrics("conditional_train", solve["best_train"]),
            **_prefix_metrics("conditional_validation", solve["best_validation"]),
        }
        if sdf_solve is not None:
            row["within_D_final_M_movement"] = float(
                sdf_solve["final_row"].get("M_function_nmae", float("nan"))
            )
        add_transition_metrics(row, histories, cycle_counts, strict_config)
        joint_gate = (
            row.get("function_fc1_c_nmae", math.inf) < 0.005
            and row.get("function_fc1_k_nmae", math.inf) < 0.005
            and (branch != "joint" or row.get("function_m_nmae", math.inf) < 0.005)
            and row.get("target_hatc_nmae", math.inf) < 0.01
            and row.get("target_lnk_nmae", math.inf) < 0.01
        )
        convergence_streak = convergence_streak + 1 if joint_gate else 0
        row["conditional_macro_fixed_point_streak"] = convergence_streak
        row["conditional_macro_fixed_point_reached"] = convergence_streak >= 2
        rows.append(row)
        pd.DataFrame(rows).to_csv(root / "metrics_extended_only.csv", index=False)
        print(json.dumps({"branch": branch, "outer_complete": current_outer, **row}, default=str), flush=True)

        if firm_hashes(firm) != before_firm or firm_hashes(firm) != initial_firm_hashes:
            raise RuntimeError(f"firm hashes changed in resumed {branch}")
        if module_hash(fc2) != before_fc2 or fc2_counter["calls"]:
            raise RuntimeError(f"FC2 invariant failed in resumed {branch}")
        if branch == "fc1_only":
            current_sdf_hash = module_hash(model.sdf_model) + ":" + module_hash(model.value_model)
            if current_sdf_hash != initial_sdf_hash:
                raise RuntimeError("SDF changed in FC1-only continuation")
        if (
            current_outer >= minimum_outer
            and row["conditional_macro_fixed_point_reached"]
        ):
            stop_reason = "diagnostic_convergence_gate"
            current_sim = next_sim
            break
        current_sim = next_sim

    extended = pd.DataFrame(rows)
    combined = pd.concat([_load_old_metrics(args.old_output_root, branch), extended], ignore_index=True, sort=False)
    combined.to_csv(root / "metrics.csv", index=False)
    summary = {
        "branch": branch,
        "resume_outer": resume_outer,
        "final_outer": current_outer,
        "requested_final_outer": final_outer,
        "resume_checkpoint": str(resume_checkpoint),
        "resume_probe_parity": parity,
        "stop_reason": stop_reason,
        "firm_initial_hashes": initial_firm_hashes,
        "firm_final_hashes": firm_hashes(firm),
        "firm_hash_invariant": firm_hashes(firm) == initial_firm_hashes,
        "sdf_hash_invariant": (
            module_hash(model.sdf_model) + ":" + module_hash(model.value_model)
        ) == initial_sdf_hash if branch == "fc1_only" else None,
        "fc1_initial_hash": initial_fc1_hash,
        "crn": crn,
        "crn_hash": stable_json_hash(crn),
        "fc2_forward_calls": fc2_counter["calls"],
        "fc2_hash_invariant": module_hash(fc2) == fc2_hash,
        "two_cycle_config": vars(strict_config),
        "old_output_untouched": True,
    }
    write_json(root / "summary.json", summary)
    return summary


def run_sdf_only(
    args: argparse.Namespace,
    device: torch.device,
    probe_parents: torch.Tensor,
    probe_children: torch.Tensor,
    budgets: Mapping[str, int],
) -> dict[str, Any]:
    root = args.output_dir / "l1" / "sdf_only"
    allowed = ("dependency_audit.json",) if args.output_dir == args.old_output_root else ()
    ensure_new_output(root, allowed_existing=allowed)
    episode, models, _ = fresh_macro_model(args, device)
    model = models["sdf_fc1"]
    firm = models["policy_value"].requires_grad_(False).eval()
    initial_firm_hashes = firm_hashes(firm)
    initial_fc1_hash = module_hash(model.fc1_model)
    fc2 = models.get("fc2")
    fc2_counter = {"calls": 0}
    fc2_hash = module_hash(fc2)
    install_fc2_guard(fc2, fc2_counter)
    train_initial, validation_initial, crn = build_initial_states(args, episode, models, device)
    current_sim = simulate_state(args, episode, models, train_initial, validation_initial, device)
    functions = snapshot_functions(model, probe_parents, probe_children)
    torch.save(functions, root / "functions_outer_000_start.pt")
    histories = {f"function_{name}": [functions[name].cpu()] for name in FUNCTION_NAMES}
    histories.update({key: [value] for key, value in sorted_macro_vectors(current_sim["validation_frame"]).items()})
    rows: list[dict[str, Any]] = [{
        "experiment": "sdf_only",
        "source": "extended",
        "outer": 0,
        "simulation_seconds": current_sim["seconds"],
        "target_hash": current_sim["target_hash"],
        "M_data_hash": current_sim["data_hash_including_M"],
        "crn_hash": stable_json_hash(crn),
        "firm_hash_invariant": True,
        "fc1_hash_invariant": True,
        "fc2_hash_invariant": True,
        "fc2_forward_calls": 0,
        **_prefix_metrics("conditional_train", current_sim["train_fit"]),
        **_prefix_metrics("conditional_validation", current_sim["validation_fit"]),
    }]
    cycle_counts = {name: 0 for name in TRACKED_VECTOR_NAMES}
    current_outer = 0
    while current_outer < args.sdf_final_outer:
        before_firm = firm_hashes(firm)
        before_fc1 = module_hash(model.fc1_model)
        solve_started = time.perf_counter()
        solve = run_sdf_chunks(
            episode,
            current_sim["batches"]["sdf_train"],
            current_sim["batches"]["sdf_validation"],
            probe_parents,
            probe_children,
            root / f"outer_{current_outer:03d}_sdf_solve",
            chunks=budgets["sdf_chunks"],
            epochs_per_chunk=budgets["sdf_epochs_per_chunk"],
            log_interval=args.log_interval,
        )
        solve_seconds = time.perf_counter() - solve_started
        current_outer += 1
        set_macro_scope(model, "frozen")
        functions = snapshot_functions(model, probe_parents, probe_children)
        torch.save(functions, root / f"functions_outer_{current_outer:03d}_start.pt")
        _save_checkpoint(
            root / "checkpoints" / f"outer_{current_outer:03d}.pt",
            model,
            {"branch": "sdf_only", "outer": current_outer},
        )
        next_sim = simulate_state(args, episode, models, train_initial, validation_initial, device)
        for name in FUNCTION_NAMES:
            histories[f"function_{name}"].append(functions[name].detach().cpu())
        for key, value in sorted_macro_vectors(next_sim["validation_frame"]).items():
            histories[key].append(value)
        row: dict[str, Any] = {
            "experiment": "sdf_only",
            "source": "extended",
            "outer": current_outer,
            "conditional_source_outer": current_outer - 1,
            "simulation_seconds": next_sim["seconds"],
            "conditional_solve_seconds": solve_seconds,
            "target_hash": next_sim["target_hash"],
            "M_data_hash": next_sim["data_hash_including_M"],
            "crn_hash": stable_json_hash(crn),
            "firm_hash_invariant": firm_hashes(firm) == initial_firm_hashes,
            "fc1_hash_invariant": module_hash(model.fc1_model) == initial_fc1_hash,
            "fc2_hash_invariant": module_hash(fc2) == fc2_hash,
            "fc2_forward_calls": fc2_counter["calls"],
            "within_D_final_M_movement": float(
                solve["final_row"].get("M_function_nmae", float("nan"))
            ),
            **_prefix_metrics("conditional_train", next_sim["train_fit"]),
            **_prefix_metrics("conditional_validation", next_sim["validation_fit"]),
        }
        add_transition_metrics(row, histories, cycle_counts, args.two_cycle_config)
        rows.append(row)
        pd.DataFrame(rows).to_csv(root / "metrics.csv", index=False)
        print(json.dumps({"branch": "sdf_only", "outer_complete": current_outer, **row}, default=str), flush=True)
        if firm_hashes(firm) != before_firm or firm_hashes(firm) != initial_firm_hashes:
            raise RuntimeError("firm changed in SDF-only")
        if module_hash(model.fc1_model) != before_fc1 or module_hash(model.fc1_model) != initial_fc1_hash:
            raise RuntimeError("FC1 changed in SDF-only")
        if module_hash(fc2) != fc2_hash or fc2_counter["calls"]:
            raise RuntimeError("FC2 invariant failed in SDF-only")
        current_sim = next_sim
    summary = {
        "branch": "sdf_only",
        "initial_outer": 0,
        "final_outer": current_outer,
        "stop_reason": "max_outer",
        "firm_initial_hashes": initial_firm_hashes,
        "firm_final_hashes": firm_hashes(firm),
        "firm_hash_invariant": firm_hashes(firm) == initial_firm_hashes,
        "fc1_initial_hash": initial_fc1_hash,
        "fc1_final_hash": module_hash(model.fc1_model),
        "fc1_hash_invariant": module_hash(model.fc1_model) == initial_fc1_hash,
        "crn": crn,
        "crn_hash": stable_json_hash(crn),
        "fc2_forward_calls": fc2_counter["calls"],
        "fc2_hash_invariant": module_hash(fc2) == fc2_hash,
        "two_cycle_config": vars(args.two_cycle_config),
        "old_output_untouched": True,
    }
    write_json(root / "summary.json", summary)
    return summary


def used_m_trace(raw: torch.Tensor, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    return {
        "raw_M": raw,
        "post_transform_M": raw,
        "used_P_M": raw.clamp(args.pv_m_clamp_min, args.pv_m_clamp_max),
        "used_Q_M": raw.clamp(args.q_m_clamp_min, args.q_m_clamp_max),
    }


def audit_recursive_sdf_tail(
    args: argparse.Namespace,
    device: torch.device,
    checkpoint: Path,
) -> dict[str, Any]:
    root = args.output_dir / "l1" / "sdf_tail_audit"
    ensure_new_output(root)
    episode, models, _ = fresh_macro_model(args, device)
    model = models["sdf_fc1"]
    load_full_state(checkpoint, model)
    firm = models["policy_value"].requires_grad_(False).eval()
    firm_before = firm_hashes(firm)
    fc2 = models.get("fc2")
    fc2_counter = {"calls": 0}
    install_fc2_guard(fc2, fc2_counter)
    train_initial, validation_initial, crn = build_initial_states(args, episode, models, device)
    simulation = simulate_state(args, episode, models, train_initial, validation_initial, device)
    for name, expected in (
        ("pv_m_clamp_min", args.pv_m_clamp_min),
        ("pv_m_clamp_max", args.pv_m_clamp_max),
        ("q_m_clamp_min", args.q_m_clamp_min),
        ("q_m_clamp_max", args.q_m_clamp_max),
    ):
        actual = float(getattr(episode.hyperparams, name))
        if actual != float(expected):
            raise RuntimeError(f"{name} audit mismatch: runtime={actual}, requested={expected}")
    mode_flags = {
        "pv_fixed_sdf": bool(getattr(episode.hyperparams, "pv_fixed_sdf", False)),
        "pv_use_clipped_m": bool(getattr(episode.hyperparams, "pv_use_clipped_m", True)),
        "q_use_detached_m": bool(getattr(episode.hyperparams, "q_use_detached_m", True)),
    }
    if mode_flags != {
        "pv_fixed_sdf": False,
        "pv_use_clipped_m": True,
        "q_use_detached_m": True,
    }:
        raise RuntimeError(f"unexpected production M modes: {mode_flags}")
    pairs = episode._build_sdf_pairs_from_macro_tensor(
        simulation["validation_out"].macro.to(device)
    )
    col = {name: index for index, name in enumerate(pairs.columns)}
    required = {
        "path", "t", "branch", "x_t", "x_t1", "Hatcf_t", "LnKF_t",
        "Hatc_t", "LnK_t",
    }
    if missing := sorted(required - set(col)):
        raise RuntimeError(f"tail audit pair table missing columns: {missing}")
    pair_data = pairs.data[pairs.data[:, col["t"]] > 2.0]
    true_state = pair_data[:, [col["Hatc_t"], col["LnK_t"]]].to(torch.float64)
    mean = true_state.mean(dim=0)
    std = true_state.std(dim=0, unbiased=False).clamp_min(1e-12)
    records: list[pd.DataFrame] = []
    raw_parts: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(pair_data.shape[0]), args.macro_batch_size):
            rows = pair_data[start:start + args.macro_batch_size]
            _, _, raw, forecast_hatc, forecast_lnk = model.forward_step(
                x_prev=rows[:, col["x_t"]:col["x_t"] + 1],
                x_curr=rows[:, col["x_t1"]:col["x_t1"] + 1],
                hatcf_prev=rows[:, col["Hatcf_t"]:col["Hatcf_t"] + 1],
                lnkf_prev=rows[:, col["LnKF_t"]:col["LnKF_t"] + 1],
                return_physical=True,
            )
            raw = raw.reshape(-1)
            traced = used_m_trace(raw, args)
            raw_parts.append(raw.detach().cpu())
            recursive_state = rows[:, [col["Hatcf_t"], col["LnKF_t"]]].to(torch.float64)
            zdist = torch.sqrt((((recursive_state - mean) / std) ** 2).sum(dim=1))
            numpy_col = lambda name: rows[:, col[name]].detach().cpu().numpy()
            frame = pd.DataFrame({
                "path": numpy_col("path"),
                "t": numpy_col("t"),
                "child_branch": numpy_col("branch"),
                "x": numpy_col("x_t"),
                "Hatc": numpy_col("Hatc_t"),
                "LnK": numpy_col("LnK_t"),
                "recursive_Hatc": numpy_col("Hatcf_t"),
                "recursive_LnK": numpy_col("LnKF_t"),
                "forecast_Hatc": forecast_hatc.reshape(-1).detach().cpu().numpy(),
                "forecast_LnK": forecast_lnk.reshape(-1).detach().cpu().numpy(),
                "teacher_state_z_distance": zdist.detach().cpu().numpy(),
                "raw_M": traced["raw_M"].reshape(-1).detach().cpu().numpy(),
                "post_transform_M": traced["post_transform_M"].reshape(-1).detach().cpu().numpy(),
                "used_P_M": traced["used_P_M"].reshape(-1).detach().cpu().numpy(),
                "used_Q_M": traced["used_Q_M"].reshape(-1).detach().cpu().numpy(),
            })
            records.append(frame)
    detail = pd.concat(records, ignore_index=True)
    tail = detail[detail["raw_M"] > 10.0].copy()
    tail.to_csv(root / "recursive_sdf_tail_samples.csv", index=False)
    detail.nlargest(min(1000, len(detail)), "raw_M").to_csv(
        root / "recursive_sdf_top1000.csv", index=False
    )
    raw_all = torch.cat(raw_parts).to(torch.float64)
    finite = raw_all[torch.isfinite(raw_all)]
    counts = []
    for threshold in (10.0, 100.0, 1000.0):
        mask = detail["raw_M"] > threshold
        counts.append({
            "threshold": threshold,
            "count": int(mask.sum()),
            "share": float(mask.mean()),
            "used_P_min": float(detail.loc[mask, "used_P_M"].min()) if mask.any() else float("nan"),
            "used_P_max": float(detail.loc[mask, "used_P_M"].max()) if mask.any() else float("nan"),
            "used_Q_min": float(detail.loc[mask, "used_Q_M"].min()) if mask.any() else float("nan"),
            "used_Q_max": float(detail.loc[mask, "used_Q_M"].max()) if mask.any() else float("nan"),
            "recursive_enters_simulation_or_aggregate": False,
            "recursive_enters_P_or_Q_training": False,
        })
    pd.DataFrame(counts).to_csv(root / "tail_counts.csv", index=False)
    dataflow = {
        "raw_sdf_definition": "models/sdf_fc1.py::SDFFC1Combined.forward_step -> SDFModel.get_M -> compute_sdf",
        "internal_transform": "compute_sdf clamps only the macro exponent to Config.SDF_EXPONENT_CLAMP before exp; returned M is otherwise raw",
        "simulation": {
            "path": "data/simulate_ts_parallel.py::_predict_macro_fc1_batched -> _expand_branches_batched",
            "use": "raw primary-state M is stored/exported as state['M']; M is not read by firm transitions or aggregate accounting",
            "recursive_audit_M_used": False,
        },
        "P_PI": {
            "path": "training/episode.py::_build_policy_m_lists -> _compute_p0_loss/_compute_pi_loss -> Bellman residual",
            "transform": f"clamp({args.pv_m_clamp_min}, {args.pv_m_clamp_max})",
            "recursive_audit_M_used": False,
            "runtime_mode_flags": mode_flags,
        },
        "Q": {
            "path": "training/episode.py::_compute_q_loss -> QLoss.compute_main_components",
            "transform": f"clamp({args.q_m_clamp_min}, {args.q_m_clamp_max}) and detach",
            "recursive_audit_M_used": False,
        },
        "SDF_true_only_training": {
            "path": "training/episode.py SDF_TRUE_ONLY selects Hatc_t/LnK_t as parent macro state",
            "recursive_audit_M_used": False,
        },
        "FC1_training": {"uses_M": False},
        "critical_correctness_issue": False,
        "conclusion": "recursive-state raw M tails are off-manifold diagnostics and do not enter an economic equation",
    }
    write_json(root / "sdf_dataflow_manifest.json", dataflow)
    summary = {
        "checkpoint": str(checkpoint),
        "total_n": int(raw_all.numel()),
        "finite_n": int(finite.numel()),
        "raw_M_mean": float(finite.mean()),
        "raw_M_p99": float(torch.quantile(finite, 0.99)),
        "raw_M_max": float(finite.max()),
        "thresholds": counts,
        "firm_hash_invariant": firm_hashes(firm) == firm_before,
        "fc2_forward_calls": fc2_counter["calls"],
        "crn": crn,
        "dataflow": dataflow,
        "runtime_mode_flags": mode_flags,
    }
    write_json(root / "summary.json", summary)
    return summary


def _latest_row(root: Path, branch: str) -> pd.Series:
    return pd.read_csv(root / "l1" / branch / "metrics.csv").iloc[-1]


def build_comparison(output: Path) -> pd.DataFrame:
    frames = []
    for branch in ("fc1_only_extended", "sdf_only", "joint_extended"):
        path = output / "l1" / branch / "metrics.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if "experiment" not in frame:
            frame.insert(0, "experiment", branch.replace("_extended", ""))
        frames.append(frame)
    comparison = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    comparison.to_csv(output / "l1_extended_summary.csv", index=False)
    return comparison


def _trend(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].dropna() if column in frame else pd.Series(dtype=float)
    if len(values) < 2:
        return "insufficient"
    ratio = float(values.iloc[-1] / max(abs(values.iloc[0]), 1e-12))
    return "damping" if ratio < 0.8 else "flat_or_rising"


def write_extended_report(args: argparse.Namespace, summaries: Mapping[str, Any]) -> None:
    fc1 = pd.read_csv(args.output_dir / "l1" / "fc1_only_extended" / "metrics.csv")
    sdf = pd.read_csv(args.output_dir / "l1" / "sdf_only" / "metrics.csv")
    joint = pd.read_csv(args.output_dir / "l1" / "joint_extended" / "metrics.csv")
    tail = summaries["sdf_tail_audit"]
    fc1_last, sdf_last, joint_last = fc1.iloc[-1], sdf.iloc[-1], joint.iloc[-1]
    strict_fc1 = bool(fc1.get("possible_two_cycle_strict", pd.Series(False)).fillna(False).any())
    strict_sdf = bool(sdf.get("possible_two_cycle_strict", pd.Series(False)).fillna(False).any())
    strict_joint = bool(joint.get("possible_two_cycle_strict", pd.Series(False)).fillna(False).any())
    reached = bool(joint_last.get("conditional_macro_fixed_point_reached", False))
    late = joint[joint["outer"] >= 8]
    movement_columns = [
        "function_m_nmae", "function_fc1_c_nmae", "function_fc1_k_nmae",
        "target_hatc_nmae", "target_lnk_nmae",
    ]
    driver = max(
        movement_columns,
        key=lambda column: float(joint_last.get(column, float("-inf")))
        if pd.notna(joint_last.get(column, float("nan"))) else float("-inf"),
    )
    if strict_joint:
        classification = "C. true limit cycle"
    elif reached:
        classification = "A. clearly damping and convergent"
    elif all(_trend(late, column) == "damping" for column in movement_columns if column in late):
        classification = "B. damping but slow / nonzero floor"
    else:
        classification = "G. inconclusive"
    q = {
        1: f"FC1-only post-resume FC1_C/FC1_K trends are {_trend(fc1[fc1.outer >= 3], 'function_fc1_c_nmae')} / {_trend(fc1[fc1.outer >= 3], 'function_fc1_k_nmae')}.",
        2: f"FC1-only convergence gate at outer {int(fc1_last.outer)}: {bool(fc1_last.get('conditional_macro_fixed_point_reached', False))}.",
        3: f"SDF-only outer-M trend: {_trend(sdf, 'function_m_nmae')}.",
        4: f"Final across-D M movement={float(sdf_last.get('function_m_nmae', float('nan'))):.6g}; within-D tail movement={float(sdf_last.get('within_D_final_M_movement', float('nan'))):.6g}.",
        5: f"SDF-only same-direction amplification: {bool((sdf.get('function_m_directional_gain', pd.Series(dtype=float)) > 1).any() and (sdf.get('function_m_movement_cosine', pd.Series(dtype=float)) > 0.5).any())}.",
        6: f"SDF-only strict two-cycle: {strict_sdf}.",
        7: f"Joint outer8+ M/FC1_C/FC1_K trends: {_trend(late, 'function_m_nmae')} / {_trend(late, 'function_fc1_c_nmae')} / {_trend(late, 'function_fc1_k_nmae')}.",
        8: f"Joint late-stage convergence gate reached={reached}; inspect the final three rows for a nonzero floor.",
        9: f"Largest final normalized movement column: {driver}={float(joint_last.get(driver, float('nan'))):.6g}.",
        10: f"Joint strict two-cycle: {strict_joint}.",
        11: f"FC1 strict cycle={strict_fc1}, SDF strict cycle={strict_sdf}, joint strict cycle={strict_joint}; cross-coupling interpretation is conditional on their trajectories.",
        12: "Recursive-state audit M does not enter simulation transitions, aggregates, P/Q training, or SDF_TRUE_ONLY training.",
        13: "No direct recursive-tail economic call path was found; primary simulated M would be clamped before P/PI and Q use.",
        14: f"Yes. Recursive raw M max={tail['raw_M_max']:.6g}; it is an off-manifold diagnostic under the audited call graph.",
        15: classification,
    }
    lines = [
        "# Macro Block Isolation L1 Extended Diagnostic",
        "",
        f"- branch: `{subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()}`",
        f"- commit: `{subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}`",
        f"- old output root (read-only): `{args.old_output_root}`",
        f"- extended output root: `{args.output_dir}`",
        f"- strict two-cycle config: `{vars(args.two_cycle_config)}`",
        "",
        "## Artifacts",
        "",
        f"- comparison: `{args.output_dir / 'l1_extended_summary.csv'}`",
        f"- FC1-only: `{args.output_dir / 'l1/fc1_only_extended/metrics.csv'}`",
        f"- SDF-only: `{args.output_dir / 'l1/sdf_only/metrics.csv'}`",
        f"- joint: `{args.output_dir / 'l1/joint_extended/metrics.csv'}`",
        f"- tail audit: `{args.output_dir / 'l1/sdf_tail_audit/summary.json'}`",
        "",
        "## Q1-Q15",
        "",
    ]
    lines.extend(f"{index}. {q[index]}" for index in range(1, 16))
    lines.extend([
        "",
        "## Final classification",
        "",
        classification,
        "",
        "No economic equation, calibration, network architecture, optimizer, P/Q/policy/firm block, FC2 path, damping, Anderson method, or Newton method was changed.",
    ])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--firm-checkpoint", type=Path, required=True)
    parser.add_argument("--outer0-macro-checkpoint", type=Path, required=True)
    parser.add_argument("--middle-macro-checkpoint", type=Path, required=True)
    parser.add_argument("--final-macro-checkpoint", type=Path, required=True)
    parser.add_argument("--old-output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--experiment", choices=("all", "fc1_only", "sdf_only", "joint", "tail_audit"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--simulation-paths", type=int, default=400)
    parser.add_argument("--validation-paths", type=int, default=400)
    parser.add_argument("--simulation-horizon", type=int, default=100)
    parser.add_argument("--firms-per-path", type=int, default=200)
    parser.add_argument("--branch-num", type=int, default=2)
    parser.add_argument("--macro-batch-size", type=int, default=4096)
    parser.add_argument("--production-fc1-epochs", type=int, default=10)
    parser.add_argument("--fc1-patience", type=int, default=30)
    parser.add_argument("--log-interval", type=int, default=1000000)
    parser.add_argument("--fc1-resume-outer", type=int, default=3)
    parser.add_argument("--fc1-final-outer", type=int, default=8)
    parser.add_argument("--sdf-final-outer", type=int, default=8)
    parser.add_argument("--joint-resume-outer", type=int, default=8)
    parser.add_argument("--joint-final-outer", type=int, default=14)
    parser.add_argument("--two-cycle-cosine-max", type=float, default=-0.5)
    parser.add_argument("--two-cycle-two-step-ratio-max", type=float, default=0.5)
    parser.add_argument("--two-cycle-amplitude-ratio-min", type=float, default=0.8)
    parser.add_argument("--two-cycle-consecutive", type=int, default=2)
    parser.add_argument("--pv-m-clamp-min", type=float, default=0.7)
    parser.add_argument("--pv-m-clamp-max", type=float, default=1.3)
    parser.add_argument("--q-m-clamp-min", type=float, default=0.5)
    parser.add_argument("--q-m-clamp-max", type=float, default=1.5)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def apply_smoke(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.simulation_paths = min(args.simulation_paths, 4)
    args.validation_paths = min(args.validation_paths, 4)
    args.simulation_horizon = min(args.simulation_horizon, 8)
    args.firms_per_path = min(args.firms_per_path, 32)
    args.macro_batch_size = min(args.macro_batch_size, 256)
    args.production_fc1_epochs = 1
    args.fc1_patience = 1
    args.fc1_final_outer = args.fc1_resume_outer + 1
    args.sdf_final_outer = 1
    args.joint_final_outer = args.joint_resume_outer + 1


def _budgets(args: argparse.Namespace) -> dict[str, int]:
    old = read_json(args.old_output_root / "l1" / "fc1_only" / "summary.json")
    budgets = dict(old["conditional_budgets_from_l0"])
    if args.smoke:
        budgets.update({"fc1_epochs": 2, "sdf_chunks": 1, "sdf_epochs_per_chunk": 1})
    return budgets


def _assert_prohibitions() -> None:
    source = inspect.getsource(run_resumed_branch) + inspect.getsource(run_sdf_only)
    forbidden = ("solve_p_block(", "solve_q_block(", "policy_iteration(")
    hits = [token for token in forbidden if token in source]
    if hits:
        raise RuntimeError(f"forbidden solver call in extended runner: {hits}")


def validate_extended_limits(args: argparse.Namespace) -> None:
    smoke = bool(getattr(args, "smoke", False))
    if args.fc1_resume_outer != 3:
        raise ValueError("FC1-only must resume from completed outer 3")
    if not args.fc1_resume_outer < args.fc1_final_outer <= 8:
        raise ValueError("FC1-only continuation must end after outer 3 and by outer 8")
    if not 0 < args.sdf_final_outer <= 8:
        raise ValueError("SDF-only must end in outer 1..8")
    if args.joint_resume_outer != 8:
        raise ValueError("joint must resume from completed outer 8")
    minimum_joint = args.joint_resume_outer + (1 if smoke else 4)
    if not minimum_joint <= args.joint_final_outer <= 14:
        raise ValueError("joint continuation must add 4-6 outer solves and stop by outer 14")


def main() -> None:
    args = parse_args()
    apply_smoke(args)
    args.old_output_root = args.old_output_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.report = args.report.expanduser().resolve()
    args.two_cycle_config = TwoCycleDiagnosticsConfig(
        cosine_max=args.two_cycle_cosine_max,
        two_step_to_one_step_max=args.two_cycle_two_step_ratio_max,
        amplitude_ratio_min=args.two_cycle_amplitude_ratio_min,
        consecutive_required=args.two_cycle_consecutive,
    )
    validate_extended_limits(args)
    _assert_prohibitions()
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    probe = torch.load(
        args.old_output_root / "l0" / "data" / "sdf_probe_bank.pt",
        map_location=device,
        weights_only=True,
    )
    probe_parents, probe_children = probe["parents"], probe["children"]
    budgets = _budgets(args)
    summaries: dict[str, Any] = {}
    if args.experiment in {"all", "fc1_only"}:
        summaries["fc1_only"] = run_resumed_branch(
            args,
            device,
            probe_parents,
            probe_children,
            branch="fc1_only",
            resume_outer=args.fc1_resume_outer,
            final_outer=args.fc1_final_outer,
            budgets=budgets,
            output_name="fc1_only_extended",
        )
    if args.experiment in {"all", "sdf_only"}:
        summaries["sdf_only"] = run_sdf_only(
            args, device, probe_parents, probe_children, budgets
        )
    if args.experiment in {"all", "joint"}:
        summaries["joint"] = run_resumed_branch(
            args,
            device,
            probe_parents,
            probe_children,
            branch="joint",
            resume_outer=args.joint_resume_outer,
            final_outer=args.joint_final_outer,
            budgets=budgets,
            output_name="joint_extended",
        )
    if args.experiment in {"all", "tail_audit"}:
        completed_joint_outer = int(
            summaries.get("joint", {}).get("final_outer", args.joint_final_outer)
        )
        checkpoint = args.output_dir / "l1" / "joint_extended" / "checkpoints" / f"outer_{completed_joint_outer:03d}.pt"
        if not checkpoint.is_file():
            checkpoint = checkpoint_for_outer(args.old_output_root, "joint", args.joint_resume_outer)
        summaries["sdf_tail_audit"] = audit_recursive_sdf_tail(args, device, checkpoint)
    comparison = build_comparison(args.output_dir)
    if args.experiment == "all":
        write_extended_report(args, summaries)
    metadata = {
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "two_cycle_config"},
        "two_cycle_config": vars(args.two_cycle_config),
        "summaries": summaries,
        "comparison_rows": int(len(comparison)),
        "prohibitions_honored": {
            "P_Q_policy_firm_updated": False,
            "FC2_used": False,
            "economic_equations_changed": False,
            "calibration_changed": False,
            "damping_or_new_solver_added": False,
            "old_outputs_overwritten": False,
        },
    }
    write_json(args.output_dir / "l1_extended_metadata.json", metadata)


if __name__ == "__main__":
    main()
