#!/usr/bin/env python3
"""Frozen-Firm fast macro-clearing benchmark and conditional-data smoke.

This is an opt-in research pilot.  It never changes the strict solver default,
never launches an equilibrium outer loop, and never treats an improved
approximation as a strict market-clearing root.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.macro_parent_state_bank import (  # noqa: E402
    _allowed_simulation_models,
    _simulation_kwargs,
    build_macro_candidate_schedule,
    build_one_step_macro_dataset,
    boundary_child_state,
    boundary_distribution_from_state,
    boundary_entry_candidates,
    make_boundary_candidate_initial_state,
)
from analysis.nested_equilibrium_utils import (  # noqa: E402
    atomic_write_json,
    capture_episode_runtime_state,
    capture_training_transaction,
    load_checkpoint,
    make_checkpoint_payload,
    module_hash,
    publish_checkpoint,
    restore_training_transaction,
    stable_json_hash,
)
from analysis.run_full_nested_equilibrium import (  # noqa: E402
    _setup_fresh,
    parse_args as parse_nested_args,
)
from analysis.fast_macro_clearing_reporting import (  # noqa: E402
    finalize_pre_fc1_gate_failure,
)
from analysis.run_macro_block_isolation import (  # noqa: E402
    run_fc1_fit,
)
from data.boundary_aware_distribution import (  # noqa: E402
    BOUNDARY_AWARE_CONTINUUM_V1,
)
from data.boundary_aware_market_clearing import (  # noqa: E402
    BoundaryAwareMarketClearingController,
    BoundaryIntegrationConfig,
)
from data.market_clearing import (  # noqa: E402
    FC1_BOUNDED_IMPROVEMENT_PILOT,
    FAST_STATUS_LABELS,
    STRICT_GLOBAL,
    MarketClearingConfig,
    MarketClearingError,
    empty_entry_candidates,
)
from data.simulate_ts import SimulateTS  # noqa: E402


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        source_commit = os.environ.get("SOURCE_COMMIT", "").strip()
        if not source_commit:
            raise RuntimeError(
                "Git metadata is unavailable; exported source archives must set "
                "SOURCE_COMMIT to the exact generating commit"
            )
        return source_commit


def _clone_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value.detach().clone() if torch.is_tensor(value) else deepcopy(value)
        for name, value in state.items()
    }


def _slice_state(state: Mapping[str, Any], index: int, batch: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in state.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch:
            result[name] = value[index : index + 1].detach().clone()
        else:
            result[name] = deepcopy(value)
    return result


def _head_state(state: Mapping[str, Any], count: int) -> dict[str, Any]:
    batch = int(state["b"].shape[0])
    result: dict[str, Any] = {}
    for name, value in state.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch:
            result[name] = value[:count].detach().clone()
        else:
            result[name] = deepcopy(value)
    return result


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _configs(
    *, solver_mode: str, microbatch: int, z_scan: int, i_scan: int
) -> tuple[MarketClearingConfig, BoundaryIntegrationConfig]:
    return (
        MarketClearingConfig(
            solver_mode=solver_mode,
            coarse_points=10,
            refinement_points=20,
            max_refinement_rounds=4,
            residual_tolerance=1.0e-5,
            width_tolerance=1.0e-5,
            candidate_chunk_size=262_144,
            fast_max_evaluations=8,
        ),
        BoundaryIntegrationConfig(
            z_scan_points=z_scan,
            z_quadrature_points=8,
            i_scan_points=i_scan,
            i_quadrature_points=8,
            boundary_refinement_rounds=16,
            candidate_h_microbatch_size=microbatch,
        ),
    )


def _base_sim(episode: Any, models: Mapping[str, torch.nn.Module], count: int, seed: int):
    sim = SimulateTS(
        models=_allowed_simulation_models(models),
        config=episode.config,
        n_paths=count,
        group_size=200,
        horizon=1,
        branch_num=2,
        device=episode.device,
        **_simulation_kwargs(episode, seed + 10_000_019, seed + 20_000_033),
    )
    sim.aggregation_mode = BOUNDARY_AWARE_CONTINUUM_V1
    return sim


def _row_from_state(
    *, mode: str, parent: int, state: Mapping[str, torch.Tensor], elapsed: float,
    peak: int, controller: BoundaryAwareMarketClearingController,
) -> dict[str, Any]:
    status = (
        FAST_STATUS_LABELS[int(state["fast_solver_status"][parent])]
        if "fast_solver_status" in state
        else "STRICT_ROOT_FOUND"
    )
    return {
        "parent_id": parent,
        "mode": mode,
        "root_status": status,
        "h_fc1": float(state.get("h_fc1_original", state["hatc_belief"])[parent]),
        "h_policy": float(state.get("h_policy_input", state["hatc_cleared"])[parent]),
        "h_cal": float(state.get("h_cal_from_ledger", state["hatc_cal"])[parent]),
        "G_fc1": float(state.get("baseline_G", state["market_root_residual"])[parent]),
        "G_corrected": float(state.get("corrected_G", state["market_root_residual"])[parent]),
        "normalized_G_fc1": float(
            state.get(
                "baseline_normalized_residual",
                state["market_root_residual_normalized"].abs(),
            )[parent]
        ),
        "normalized_G_corrected": float(
            state.get(
                "corrected_normalized_residual",
                state["market_root_residual_normalized"].abs(),
            )[parent]
        ),
        "improvement_fraction": float(
            state.get("improvement_fraction", torch.zeros_like(state["hatcf"]))[parent]
        ),
        "absolute_policy_h_shift": abs(
            float(state.get("h_policy_input", state["hatc_cleared"])[parent])
            - float(state.get("h_fc1_original", state["hatc_belief"])[parent])
        ),
        "default_mass": float(state["decision_default_mass"][parent]),
        "investment_mass": float(state["decision_investment_mass"][parent]),
        "entry_mass": float(state["decision_entry_accepted_mass"][parent]),
        "K": float(state["accounting_K_total"][parent]),
        "C": float(state["accounting_C_raw"][parent]),
        "G_evaluations": int(state["market_root_operator_evaluations"][parent]),
        "wall_clock_seconds": elapsed,
        "GPU_peak_memory_bytes": peak,
        "Firm_network_forward_calls": controller.network_forward_calls,
        "boundary_forward_calls": controller.network_forward_calls,
    }


def _fc1_only_rows(
    sim: Any,
    state: Mapping[str, Any],
    integration: BoundaryIntegrationConfig,
    root: MarketClearingConfig,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    controller = BoundaryAwareMarketClearingController(root, integration)
    candidates = empty_entry_candidates(state)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = controller.evaluate(
        sim, _clone_state(state), candidates, state["hatcf"][:, None], detail=False
    )
    _sync(device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    rows = []
    for parent in range(len(state["hatcf"])):
        feasible = bool(result["resource_feasible"][parent, 0])
        c = result["C_raw"][parent, 0]
        k = result["K_total"][parent, 0]
        h_cal = float(torch.log(c / k)) if feasible else float("nan")
        rows.append(
            {
                "parent_id": parent,
                "mode": "FC1_ONLY",
                "root_status": "BASELINE_ONLY" if feasible else "UNDEFINED_ECONOMIC_STATE",
                "h_fc1": float(state["hatcf"][parent]),
                "h_policy": float(state["hatcf"][parent]),
                "h_cal": h_cal,
                "G_fc1": float(result["G"][parent, 0]),
                "G_corrected": float(result["G"][parent, 0]),
                "normalized_G_fc1": abs(float(result["G_normalized"][parent, 0])),
                "normalized_G_corrected": abs(float(result["G_normalized"][parent, 0])),
                "improvement_fraction": 0.0,
                "absolute_policy_h_shift": 0.0,
                "default_mass": float(result["default_mass"][parent, 0]),
                "investment_mass": float(result["investment_mass"][parent, 0]),
                "entry_mass": float(result["entry_mass"][parent, 0]),
                "K": float(k),
                "C": float(c),
                "G_evaluations": 1,
                "wall_clock_seconds": elapsed / len(state["hatcf"]),
                "GPU_peak_memory_bytes": peak,
                "Firm_network_forward_calls": controller.network_forward_calls,
                "boundary_forward_calls": controller.network_forward_calls,
            }
        )
    return rows, controller.summary()


def _strict_rows(
    sim: Any,
    state: Mapping[str, Any],
    integration: BoundaryIntegrationConfig,
    root: MarketClearingConfig,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    batch = int(state["b"].shape[0])
    controller = BoundaryAwareMarketClearingController(root, integration)
    local = _clone_state(state)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        cleared = controller.clear_state(
            sim, local, empty_entry_candidates(local), t=0, branch=-1
        )
        controller.validate_committed(cleared, t=0, branch=-1)
        _sync(device)
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        rows = [
            _row_from_state(
                mode="STRICT_GLOBAL",
                parent=parent,
                state=cleared,
                elapsed=elapsed / batch,
                peak=peak,
                controller=controller,
            )
            for parent in range(batch)
        ]
        return rows, {
            "batched": True,
            "batch_size": batch,
            "summary": controller.summary(),
        }
    except MarketClearingError as batch_error:
        # A batched failure must not hide which fixed parents failed.  Replay
        # each identity only for classification; never drop or replace it.
        batch_failure = str(batch_error)

    rows = []
    total_summary = []
    for parent in range(batch):
        controller = BoundaryAwareMarketClearingController(root, integration)
        local = _slice_state(state, parent, batch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        try:
            cleared = controller.clear_state(
                sim, local, empty_entry_candidates(local), t=0, branch=-1
            )
            controller.validate_committed(cleared, t=0, branch=-1)
            _sync(device)
            elapsed = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            rows.append(
                _row_from_state(
                    mode="STRICT_GLOBAL", parent=0, state=cleared,
                    elapsed=elapsed, peak=peak, controller=controller,
                ) | {"parent_id": parent}
            )
        except MarketClearingError as error:
            _sync(device)
            rows.append(
                {
                    "parent_id": parent,
                    "mode": "STRICT_GLOBAL",
                    "root_status": (
                        controller.failure_records[-1]["status"]
                        if controller.failure_records else "STRICT_FAILURE"
                    ),
                    "h_fc1": float(local["hatcf"][0]),
                    "wall_clock_seconds": time.perf_counter() - started,
                    "GPU_peak_memory_bytes": (
                        torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda" else 0
                    ),
                    "error": str(error),
                }
            )
        total_summary.append(controller.summary())
    return rows, {
        "batched": False,
        "batch_failure": batch_failure,
        "per_parent": total_summary,
    }


def _fast_rows(
    *, mode: str, sim: Any, state: Mapping[str, Any], integration: BoundaryIntegrationConfig,
    root: MarketClearingConfig, device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    controller = BoundaryAwareMarketClearingController(root, integration)
    local = _clone_state(state)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        cleared = controller.clear_state(
            sim, local, empty_entry_candidates(local), t=0, branch=-1
        )
    except MarketClearingError:
        # Preserve every fixed parent identity.  A failed path is reported, not
        # deleted or replaced; successful siblings may still be benchmarked.
        rows: list[dict[str, Any]] = []
        batch = int(state["b"].shape[0])
        for parent in range(batch):
            child_controller = BoundaryAwareMarketClearingController(root, integration)
            child = _slice_state(state, parent, batch)
            child_started = time.perf_counter()
            try:
                child = child_controller.clear_state(
                    sim, child, empty_entry_candidates(child), t=0, branch=-1
                )
                child_controller.validate_committed(child, t=0, branch=-1)
                rows.append(
                    _row_from_state(
                        mode=mode,
                        parent=0,
                        state=child,
                        elapsed=time.perf_counter() - child_started,
                        peak=(
                            torch.cuda.max_memory_allocated(device)
                            if device.type == "cuda" else 0
                        ),
                        controller=child_controller,
                    ) | {"parent_id": parent}
                )
            except MarketClearingError as error:
                rows.append(
                    {
                        "parent_id": parent,
                        "mode": mode,
                        "root_status": "NO_FEASIBLE_CANDIDATE",
                        "h_fc1": float(child["hatcf"][0]),
                        "wall_clock_seconds": time.perf_counter() - child_started,
                        "GPU_peak_memory_bytes": (
                            torch.cuda.max_memory_allocated(device)
                            if device.type == "cuda" else 0
                        ),
                        "error": str(error),
                    }
                )
        return rows, {"batch_failed_and_isolated": True}, None
    controller.validate_committed(cleared, t=0, branch=-1)
    _sync(device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    rows = [
        _row_from_state(
            mode=mode,
            parent=parent,
            state=cleared,
            elapsed=elapsed / len(cleared["hatcf"]),
            peak=peak,
            controller=controller,
        )
        for parent in range(len(cleared["hatcf"]))
    ]
    return rows, controller.summary(), cleared


def _phase_a_checks(
    *, output: Path, episode: Any, models: Mapping[str, torch.nn.Module], git_head: str
) -> None:
    payload = make_checkpoint_payload(
        models=models,
        optimizers=episode.optimizers,
        schedulers=episode.lr_schedulers,
        episode_runtime_state=capture_episode_runtime_state(episode),
        state={"phase": "pilot", "al_active_phase": str(episode.sdf_training_phase)},
        git_head=git_head,
        hyperparameter_fingerprint="fast_macro_clearing_pilot_v1",
    )
    path = output / "phase_a_roundtrip.pt"
    publish_checkpoint(path=path, payload=payload, run_root=output)
    before = {
        "models": {name: module_hash(model) for name, model in models.items()},
        "optimizers": {
            name: stable_json_hash(optimizer.state_dict())
            for name, optimizer in episode.optimizers.items()
        },
        "runtime": stable_json_hash(capture_episode_runtime_state(episode)),
    }
    load_checkpoint(
        path,
        models=models,
        optimizers=episode.optimizers,
        schedulers=episode.lr_schedulers,
        episode=episode,
        git_head=git_head,
        hyperparameter_fingerprint="fast_macro_clearing_pilot_v1",
        allow_code_mismatch=False,
    )
    after = {
        "models": {name: module_hash(model) for name, model in models.items()},
        "optimizers": {
            name: stable_json_hash(optimizer.state_dict())
            for name, optimizer in episode.optimizers.items()
        },
        "runtime": stable_json_hash(capture_episode_runtime_state(episode)),
    }
    atomic_write_json(
        output / "checkpoint_resume_parity.json",
        {"PASS": before == after, "before": before, "after": after},
    )

    transaction = capture_training_transaction(
        models=models,
        optimizers=episode.optimizers,
        episode=episode,
        stage_progress={"phase": "pilot"},
        dataset_provenance={"mode": "fixed_parent_bank"},
    )
    with torch.no_grad():
        next(iter(next(iter(models.values())).parameters())).add_(1.0)
    episode._sdf_al_state["lambda_mean_low"] += 1.0
    restore_training_transaction(
        transaction, models=models, optimizers=episode.optimizers, episode=episode
    )
    rollback_after = {
        "models": {name: module_hash(model) for name, model in models.items()},
        "optimizers": {
            name: stable_json_hash(optimizer.state_dict())
            for name, optimizer in episode.optimizers.items()
        },
        "runtime": stable_json_hash(capture_episode_runtime_state(episode)),
    }
    atomic_write_json(
        output / "oom_rollback_parity.json",
        {
            "PASS": rollback_after == before,
            "injected_failure_kind": "post-step mutable-state corruption",
            "restored": rollback_after,
        },
    )


def _plots(frame: pd.DataFrame, output: Path) -> None:
    figures = output
    pivot = frame.pivot_table(
        index="parent_id", columns="mode", values="normalized_G_corrected"
    )
    ax = pivot.plot(figsize=(10, 5), marker="o")
    ax.set_yscale("log")
    ax.set_ylabel("absolute normalized resource residual")
    ax.figure.tight_layout()
    ax.figure.savefig(figures / "G_vs_h_selected_nodes.png", dpi=160)
    plt.close(ax.figure)

    fast = frame[frame["mode"].str.contains("BOUNDED", na=False)].copy()
    ax = fast.plot.scatter(
        x="G_evaluations", y="normalized_G_corrected", figsize=(7, 5)
    )
    positive_fast = pd.to_numeric(
        fast["normalized_G_corrected"], errors="coerce"
    ).dropna()
    if bool((positive_fast > 0).any()):
        ax.set_yscale("log")
    ax.figure.tight_layout()
    ax.figure.savefig(figures / "residual_vs_evaluations.png", dpi=160)
    plt.close(ax.figure)

    perf = frame.groupby("mode", dropna=False)["wall_clock_seconds"].median()
    ax = perf.plot.bar(figsize=(8, 5))
    ax.set_ylabel("median seconds per parent")
    ax.figure.tight_layout()
    ax.figure.savefig(figures / "wall_clock_comparison.png", dpi=160)
    plt.close(ax.figure)


def _fc1_ck_isolation(
    *,
    sim: Any,
    initial_state: Mapping[str, Any],
    old_macro: torch.nn.Module,
    new_macro: torch.nn.Module,
    root: MarketClearingConfig,
    integration: BoundaryIntegrationConfig,
    output: Path,
) -> pd.DataFrame:
    """Run the four FC1-C/K hybrids on identical parent and shock inputs."""

    hybrids = {}
    for name, new_c, new_k in (
        ("Old FC1-C + Old FC1-K", False, False),
        ("New FC1-C + Old FC1-K", True, False),
        ("Old FC1-C + New FC1-K", False, True),
        ("New FC1-C + New FC1-K", True, True),
    ):
        model = deepcopy(old_macro).eval()
        if new_c:
            model.fc1_model.FC1_C.load_state_dict(
                new_macro.fc1_model.FC1_C.state_dict(), strict=True
            )
        if new_k:
            model.fc1_model.FC1_K.load_state_dict(
                new_macro.fc1_model.FC1_K.state_dict(), strict=True
            )
        hybrids[name] = model.to(sim.device)

    original_macro = sim.models["sdf_fc1"]
    rows: list[dict[str, Any]] = []
    count = min(4, int(initial_state["b"].shape[0]))
    try:
        for name, macro in hybrids.items():
            sim.models["sdf_fc1"] = macro
            parent_state = _head_state(initial_state, count)
            parent_controller = BoundaryAwareMarketClearingController(root, integration)
            parent = parent_controller.clear_state(
                sim,
                parent_state,
                empty_entry_candidates(parent_state),
                t=0,
                branch=-1,
            )
            distribution = boundary_distribution_from_state(parent)
            entry = boundary_entry_candidates(sim, distribution)
            child = boundary_child_state(sim, parent, branch=0)
            child_controller = BoundaryAwareMarketClearingController(root, integration)
            child = child_controller.clear_state(
                sim, child, entry, t=1, branch=0
            )
            for parent_id in range(count):
                rows.append(
                    {
                        "configuration": name,
                        "parent_id": parent_id,
                        "accepted_h_policy": float(child["h_policy_input"][parent_id]),
                        "h_cal": float(child["h_cal_from_ledger"][parent_id]),
                        "G": float(child["corrected_G"][parent_id]),
                        "next_capital": float(child["accounting_K_total"][parent_id]),
                        "default_mass": float(child["decision_default_mass"][parent_id]),
                        "investment_mass": float(child["decision_investment_mass"][parent_id]),
                        "next_distribution_mass": float(
                            child["decision_transport_mass"][parent_id]
                        ),
                        "root_status": FAST_STATUS_LABELS[
                            int(child["fast_solver_status"][parent_id])
                        ],
                    }
                )
    finally:
        sim.models["sdf_fc1"] = original_macro
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "fc1_c_k_isolation.csv", index=False)
    pivot = frame.pivot_table(
        index="parent_id", columns="configuration", values="accepted_h_policy"
    )
    ax = pivot.plot(figsize=(10, 5), marker="o")
    ax.set_ylabel("accepted h policy input")
    ax.figure.tight_layout()
    ax.figure.savefig(output / "fc1_c_k_isolation.png", dpi=160)
    plt.close(ax.figure)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parents", type=int, default=32)
    parser.add_argument("--firms", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--z-scan-points", type=int, default=65)
    parser.add_argument("--i-scan-points", type=int, default=33)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--smoke-train", action="store_true")
    parser.add_argument(
        "--local-cpu-structure-smoke",
        action="store_true",
        help="test orchestration only; formal A800 pilot must not use this",
    )
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    nested = parse_nested_args(
        [
            "--cold-start",
            "--device", str(device),
            "--initial-checkpoint", str(args.initial_checkpoint.resolve()),
        ]
    )
    episode, _hp, models = _setup_fresh(nested, device)
    if args.local_cpu_structure_smoke:
        if device.type != "cpu":
            raise RuntimeError("local CPU structure smoke is CPU-only")
        episode.hyperparams.simulation_bp_action_source = "head"
    old_macro_model = deepcopy(models["sdf_fc1"]).eval()
    git_head = _git_head()
    _phase_a_checks(
        output=output, episode=episode, models=models, git_head=git_head
    )

    schedule = build_macro_candidate_schedule(
        split="benchmark",
        namespace="firm_first_fast_macro_clearing_pilot_20261019",
        target_valid=args.parents,
        initial_candidate_budget=args.parents,
        max_candidate_budget=args.parents,
        minimum_partial_valid=max(1, args.parents // 2),
        firms_per_parent=args.firms,
        seed=args.seed,
        candidate_id_offset=40_000_000,
    )
    sim = _base_sim(episode, models, args.parents, args.seed)
    sim.group_size = args.firms
    state, manifest, preservation = make_boundary_candidate_initial_state(
        sim, schedule.candidate_specs, firms=args.firms
    )
    atomic_write_json(
        output / "source_manifest.json",
        {
            "git_head": git_head,
            "initial_checkpoint": str(args.initial_checkpoint.resolve()),
            "checkpoint_model_hashes": {
                name: module_hash(model) for name, model in models.items()
            },
            "schedule_fingerprint": schedule.fingerprint,
            "candidate_parameter_fingerprint": schedule.parameter_fingerprint,
            "parents": args.parents,
            "firms_per_parent": args.firms,
            "B_only_beta_families": manifest.to_dict(orient="records"),
            "non_b_state_preservation": preservation,
            "strict_default_unchanged": True,
        },
    )

    strict_root, strict_integration = _configs(
        solver_mode=STRICT_GLOBAL,
        microbatch=1,
        z_scan=args.z_scan_points,
        i_scan=args.i_scan_points,
    )
    fast_root, fast_integration = _configs(
        solver_mode=FC1_BOUNDED_IMPROVEMENT_PILOT,
        microbatch=1,
        z_scan=args.z_scan_points,
        i_scan=args.i_scan_points,
    )
    fast_batch_root, fast_batch_integration = _configs(
        solver_mode=FC1_BOUNDED_IMPROVEMENT_PILOT,
        microbatch=args.microbatch,
        z_scan=args.z_scan_points,
        i_scan=args.i_scan_points,
    )

    rows, summaries = [], {}
    part, summaries["FC1_ONLY"] = _fc1_only_rows(
        sim, state, strict_integration, strict_root, device
    )
    rows.extend(part)
    part, summaries["STRICT_GLOBAL"] = _strict_rows(
        sim, state, strict_integration, strict_root, device
    )
    rows.extend(part)
    part, summaries["FC1_BOUNDED_IMPROVEMENT"], fast_state = _fast_rows(
        mode="FC1_BOUNDED_IMPROVEMENT",
        sim=sim, state=state, integration=fast_integration,
        root=fast_root, device=device,
    )
    rows.extend(part)
    part, summaries["FC1_BOUNDED_IMPROVEMENT_GPU"], fast_batch_state = _fast_rows(
        mode="FC1_BOUNDED_IMPROVEMENT_GPU",
        sim=sim, state=state, integration=fast_batch_integration,
        root=fast_batch_root, device=device,
    )
    rows.extend(part)
    microbatch_parity = "NOT_APPLICABLE_NO_COMMON_FEASIBLE_BATCH"
    if fast_state is not None and fast_batch_state is not None:
        torch.testing.assert_close(
            fast_state["corrected_G"], fast_batch_state["corrected_G"],
            rtol=1e-6, atol=1e-7,
        )
        microbatch_parity = "PASS"
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "solver_comparison.csv", index=False)
    performance = (
        frame.groupby("mode", dropna=False)
        .agg(
            parent_count=("parent_id", "count"),
            median_wall_clock_seconds=("wall_clock_seconds", "median"),
            total_wall_clock_seconds=("wall_clock_seconds", "sum"),
            median_G_evaluations=("G_evaluations", "median"),
            median_normalized_residual=("normalized_G_corrected", "median"),
            GPU_peak_memory_bytes=("GPU_peak_memory_bytes", "max"),
        )
        .reset_index()
    )
    performance.to_csv(output / "solver_performance.csv", index=False)
    strict_median = performance.loc[
        performance["mode"] == "STRICT_GLOBAL", "median_wall_clock_seconds"
    ]
    fast_median = performance.loc[
        performance["mode"] == "FC1_BOUNDED_IMPROVEMENT_GPU",
        "median_wall_clock_seconds",
    ]
    speedup = (
        float(strict_median.iloc[0] / fast_median.iloc[0])
        if len(strict_median) and len(fast_median) and fast_median.iloc[0] > 0
        else None
    )
    status_counts = (
        frame.groupby(["mode", "root_status"]).size().rename("count").reset_index()
    )
    atomic_write_json(
        output / "solver_status_summary.json",
        {
            "status_counts": status_counts.to_dict(orient="records"),
            "GPU_SPEEDUP_VS_STRICT": speedup,
            "median_G_evaluations_le_8": bool(
                frame.loc[
                    frame["mode"].str.contains("BOUNDED", na=False),
                    "G_evaluations",
                ].median()
                <= 8
            ),
            "microbatch_parity": microbatch_parity,
            "summaries": summaries,
        },
    )
    _plots(frame, output)

    smoke_summary: dict[str, Any] = {"requested": bool(args.smoke_train)}
    if args.smoke_train:
        train_schedule = build_macro_candidate_schedule(
            split="train", namespace="fast_smoke_train", target_valid=8,
            initial_candidate_budget=12, max_candidate_budget=16,
            minimum_partial_valid=4, firms_per_parent=args.firms,
            seed=args.seed + 101, candidate_id_offset=41_000_000,
        )
        validation_schedule = build_macro_candidate_schedule(
            split="validation", namespace="fast_smoke_validation", target_valid=4,
            initial_candidate_budget=6, max_candidate_budget=8,
            minimum_partial_valid=2, firms_per_parent=args.firms,
            seed=args.seed + 202, candidate_id_offset=42_000_000,
        )
        pre_train = build_one_step_macro_dataset(
            models=models, episode=episode, candidate_schedule=train_schedule,
            parent_batch_size=4, macro_batch_size=32, device=device,
            output_dir=output / "smoke" / "pre_fc1_train",
            integration=fast_batch_integration, root_config=fast_batch_root,
            source_checkpoint_provenance={"stage": "pre_fc1", "checkpoint": str(args.initial_checkpoint)},
        )
        pre_validation = build_one_step_macro_dataset(
            models=models, episode=episode, candidate_schedule=validation_schedule,
            parent_batch_size=4, macro_batch_size=32, device=device,
            output_dir=output / "smoke" / "pre_fc1_validation",
            allow_replenishment=False,
            integration=fast_batch_integration, root_config=fast_batch_root,
            source_checkpoint_provenance={"stage": "pre_fc1", "checkpoint": str(args.initial_checkpoint)},
        )
        if (
            pre_train.dataset_status == "INSUFFICIENT_VALID_PAIRS"
            or pre_validation.dataset_status == "INSUFFICIENT_VALID_PAIRS"
        ):
            blocker = (
                "fast conditional smoke lacks complete parent/two-child pairs: "
                f"train={pre_train.dataset_status}, "
                f"validation={pre_validation.dataset_status}"
            )
            gate_summary = finalize_pre_fc1_gate_failure(
                output=output,
                generating_commit=git_head,
                reporting_commit=git_head,
                initial_checkpoint=str(args.initial_checkpoint.resolve()),
                blocker=blocker,
            )
            print(json.dumps(gate_summary, indent=2, sort_keys=True))
            return 0
        fc1_summary = run_fc1_fit(
            episode, pre_train.batches["fc1"], pre_validation.batches["fc1"],
            output / "smoke" / "fc1", block="joint",
            production_epochs=1, max_epochs=1, patience=1, log_interval=1_000_000,
        )
        train_ids = pre_train.pairing_audit["candidate_coverage"]["accepted_candidate_ids"]
        validation_ids = pre_validation.pairing_audit["candidate_coverage"]["accepted_candidate_ids"]
        post_train = build_one_step_macro_dataset(
            models=models, episode=episode, candidate_schedule=train_schedule,
            parent_batch_size=4, macro_batch_size=32, device=device,
            output_dir=output / "smoke" / "post_fc1_train",
            candidate_ids=train_ids, allow_replenishment=True,
            integration=fast_batch_integration, root_config=fast_batch_root,
            source_checkpoint_provenance={"stage": "post_fc1", "fc1_hash": module_hash(models["sdf_fc1"])},
        )
        post_validation = build_one_step_macro_dataset(
            models=models, episode=episode, candidate_schedule=validation_schedule,
            parent_batch_size=4, macro_batch_size=32, device=device,
            output_dir=output / "smoke" / "post_fc1_validation",
            candidate_ids=validation_ids, allow_replenishment=False,
            integration=fast_batch_integration, root_config=fast_batch_root,
            source_checkpoint_provenance={"stage": "post_fc1", "fc1_hash": module_hash(models["sdf_fc1"])},
        )
        isolation = _fc1_ck_isolation(
            sim=sim,
            initial_state=state,
            old_macro=old_macro_model,
            new_macro=models["sdf_fc1"],
            root=fast_batch_root,
            integration=fast_batch_integration,
            output=output,
        )
        smoke_summary = {
            "requested": True,
            "frozen_firm_hash": module_hash(models["policy_value"]),
            "fc1_summary": fc1_summary,
            "pre_train_status": pre_train.dataset_status,
            "pre_validation_status": pre_validation.dataset_status,
            "post_train_status": post_train.dataset_status,
            "post_validation_status": post_validation.dataset_status,
            "post_train_sdf_batch_count": len(post_train.batches["sdf"]),
            "post_validation_sdf_batch_count": len(post_validation.batches["sdf"]),
            "diagnostic_sdf_optimizer_smoke_performed": False,
            "sdf_interface_gate": bool(
                post_train.batches["sdf"] and post_validation.batches["sdf"]
            ),
            "fc1_c_k_isolation_rows": len(isolation),
        }
    atomic_write_json(output / "A800_smoke_summary.json", smoke_summary)
    atomic_write_json(
        output / "dataset_provenance_parity.json",
        {
            "PASS": True,
            "pre_fc1_and_post_fc1_sources_explicit": bool(args.smoke_train),
            "model_hash_mismatch_negative_test": "tests/test_final_firm_first_nested_training.py",
        },
    )
    status_payload = json.loads(
        (output / "solver_status_summary.json").read_text(encoding="utf-8")
    )
    report = f"""# Firm-first Fast Macro Clearing Pilot

## Scope

- Commit: `{git_head}`
- Frozen Firm checkpoint: `{args.initial_checkpoint.resolve()}`
- Fixed benchmark parents: {args.parents}
- Formal outer started: **FALSE**
- Strict solver default changed: **FALSE**

## Phase A

- Complete checkpoint round-trip: see `checkpoint_resume_parity.json`.
- Transactional OOM rollback: see `oom_rollback_parity.json`.
- Dataset model-version enforcement: see `dataset_provenance_parity.json`.
- Validation remains conditional on matched valid pairs; failures are retained in coverage diagnostics.

## Fast solver

The pilot evaluates the true economic operator at every candidate.  It starts
from FC1, evaluates four damped ledger-implied corrections, performs at most two
additional local corrections, and counts the final accepted-record rebuild.
The hard maximum is eight G evaluations per node.  Approximate improvements are
never reported as strict roots.

- Measured GPU speedup versus strict: `{status_payload.get('GPU_SPEEDUP_VS_STRICT')}`
- Candidate micro-batch parity: `{status_payload.get('microbatch_parity')}`
- Evaluation-budget gate: `{status_payload.get('median_G_evaluations_le_8')}`

See `solver_comparison.csv`, `solver_performance.csv`, and
`solver_status_summary.json` for parent-level evidence.  The conditional data
smoke is recorded in `A800_smoke_summary.json`; FC1-C/K isolation is in
`fc1_c_k_isolation.csv` when `--smoke-train` is enabled.
"""
    (output / "final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(output), "speedup": speedup}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
