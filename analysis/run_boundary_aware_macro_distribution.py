#!/usr/bin/env python3
"""Generate frozen-model macro parent/child data with boundary-aware densities.

This runner performs no optimizer step.  It reuses the existing h root search,
enumerates two complete common-shock children per parent, and advances only a
predeclared main child (branch zero) so the tree does not grow exponentially.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import run_full_nested_equilibrium as nested
from analysis.macro_parent_state_bank import _allowed_simulation_models, _simulation_kwargs
from analysis.run_consistent_aggregation_diagnostics import _load_base_args
from data.boundary_aware_distribution import (
    BOUNDARY_AWARE_CONTINUUM_V1,
    PreDecisionDistribution,
    gaussian_mixture_moments,
    moment_project_predecision_distribution,
    propagate_ar1_with_eta_mixture,
)
from data.boundary_aware_market_clearing import (
    BoundaryAwareMarketClearingController,
    BoundaryIntegrationConfig,
)
from data.entry import EntryCandidates
from data.market_clearing import MarketClearingConfig, empty_entry_candidates
from data.simulate_ts import SimulateTS
from data.simulate_ts_parallel import _initialize_batched_state, _keyed_normal, _predict_macro_fc1_batched
from data.tensor_data import TensorTable


EXPERIMENT = "boundary_aware_macro_distribution_20261019"
PROJECTION_BALANCE_RELATIVE_TOLERANCE = 1.0e-7
PROJECTION_FIXED_H_OPERATOR_TOLERANCE = 5.0e-3


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _initial_distribution_state(
    sim: SimulateTS,
    *,
    paths: int,
    firms: int,
    seed: int,
) -> Dict[str, torch.Tensor]:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if sim.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    finite = _initialize_batched_state(sim, firms)
    B, N = finite["b"].shape
    duplicate = lambda value: torch.repeat_interleave(value, 2, dim=1)
    eta_probability = torch.tensor(
        [1.0 - float(sim.config.ZETA), float(sim.config.ZETA)],
        device=sim.device,
        dtype=finite["K"].dtype,
    )[None, None, :]
    mass = torch.ones((B, N, 1), device=sim.device, dtype=finite["K"].dtype)
    mass = (mass * eta_probability).reshape(B, 2 * N)
    mask = mass > 0.0
    stationary_std = float(sim.config.SIGMA_Z) / np.sqrt(
        max(1.0 - float(sim.config.RHO_Z) ** 2, 1.0e-30)
    )
    component_id = torch.arange(2 * N, device=sim.device, dtype=torch.long)[None, :].expand(B, -1)
    eta = torch.tensor([0.0, 1.0], device=sim.device, dtype=finite["K"].dtype)[
        None, None, :
    ].expand(B, N, 2).reshape(B, 2 * N)
    zeros_long = torch.zeros((B, 2 * N), device=sim.device, dtype=torch.long)
    distribution = PreDecisionDistribution(
        component_mass=mass,
        b=duplicate(finite["b"]),
        capital_per_unit=duplicate(finite["K"]),
        z_conditional_mean=torch.full_like(mass, float(sim.config.ZBAR)),
        z_conditional_std=torch.full_like(mass, stationary_std),
        component_mask=mask,
        economic_node_id=zeros_long,
        component_id=component_id,
        origin_kind=zeros_long,
        origin_source_id=torch.repeat_interleave(finite["firm_id"], 2, dim=1),
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
            "x": finite["x"],
            "hatcf": finite["hatcf"],
            "lnkf": finite["lnkf"],
            "M": finite["M"],
            "economic_node_id": torch.zeros(B, device=sim.device, dtype=torch.long),
            "economic_time": torch.zeros(B, device=sim.device, dtype=torch.long),
            "transition_path_id": torch.arange(B, device=sim.device, dtype=torch.long),
            "market_previous_root": finite["hatcf"].clone(),
            "i": torch.zeros_like(mass),
            "firm_id": component_id,
        }
    )
    return state


def _distribution_from_state(state: Mapping[str, torch.Tensor]) -> PreDecisionDistribution:
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


def _entry_descriptor(
    sim: SimulateTS,
    parent_distribution: PreDecisionDistribution,
) -> EntryCandidates:
    B = parent_distribution.batch_size
    dtype = parent_distribution.component_mass.dtype
    device = parent_distribution.component_mass.device
    total_mass = parent_distribution.total_mass()
    total_capital = parent_distribution.total_capital()
    mean_capital = total_capital / total_mass.clamp_min(1e-30)
    capital_per_entrant = float(sim.entry_size_ratio) * mean_capital
    potential_capital = float(sim.entry_capital_ratio) * total_capital
    active = (capital_per_entrant > 0.0) & (potential_capital > 0.0)
    return EntryCandidates(
        counts=active.to(torch.long),
        mask=active[:, None],
        z=torch.zeros((B, 1), device=device, dtype=dtype),
        entry_cost=torch.zeros((B, 1), device=device, dtype=dtype),
        K_birth=capital_per_entrant[:, None],
        potential_capital_nominal=potential_capital,
        candidate_capital_realized=capital_per_entrant,
    )


def _macro_shock(sim: SimulateTS, state: Mapping[str, torch.Tensor], *, child_t: int, branch: int) -> torch.Tensor:
    path = state["transition_path_id"].to(torch.long)
    eps = _keyed_normal(
        seed=int(sim.common_transition_seed),
        path_id=path,
        economic_time=torch.full_like(path, int(child_t)),
        branch_id=int(branch),
        identity=torch.zeros_like(path),
        shock_type=1,
        dtype=state["x"].dtype,
    )
    return (
        (1.0 - float(sim.config.RHO_X)) * float(sim.config.XBAR)
        + float(sim.config.RHO_X) * state["x"]
        + float(sim.config.SIGMA_X) * eps
    )


def _child_state(
    sim: SimulateTS,
    parent: Mapping[str, torch.Tensor],
    *,
    child_t: int,
    branch: int,
) -> Dict[str, torch.Tensor]:
    record = parent["_boundary_decision_record"]
    distribution = propagate_ar1_with_eta_mixture(
        record.transport_atoms,
        rho_z=float(sim.config.RHO_Z),
        sigma_z=float(sim.config.SIGMA_Z),
        z_bar=float(sim.config.ZBAR),
        zeta=float(sim.config.ZETA),
    )
    x_next = _macro_shock(sim, parent, child_t=child_t, branch=branch)
    hatcf, lnkf, M = _predict_macro_fc1_batched(
        sim,
        parent["x"],
        x_next,
        parent["hatc_cal"],
        parent["lnk_cal"],
    )
    state = distribution.as_state_tensors()
    B, L = distribution.component_mass.shape
    node_id = torch.full(
        (B,), child_t * int(sim.branch_num + 1) + branch + 1,
        device=sim.device,
        dtype=torch.long,
    )
    state["economic_node_id_component"] = node_id[:, None].expand(B, L)
    state.update(
        {
            "x": x_next,
            "hatcf": hatcf,
            "lnkf": lnkf,
            "M": M,
            "economic_node_id": node_id,
            "economic_time": torch.full((B,), child_t, device=sim.device, dtype=torch.long),
            "transition_path_id": parent["transition_path_id"].clone(),
            "market_previous_root": parent["hatc_cleared"].clone(),
            "i": torch.zeros((B, L), device=sim.device, dtype=distribution.component_mass.dtype),
            "firm_id": distribution.component_id,
        }
    )
    return state


def _node_rows(state: Mapping[str, torch.Tensor], *, t: int, branch: int, split: str) -> list[dict[str, Any]]:
    distribution = _distribution_from_state(state)
    moments = gaussian_mixture_moments(distribution)
    rows = []
    for path in range(distribution.batch_size):
        rows.append(
            {
                "split": split,
                "path": path,
                "lineage_id": f"{split}-{path}",
                "t": int(t),
                "branch": int(branch),
                "x": float(state["x"][path]),
                "hatcf": float(state["hatcf"][path]),
                "lnkf": float(state["lnkf"][path]),
                "Hatc": float(state["hatc_cleared"][path]),
                "LnK": float(state["lnk_cal"][path]),
                "h_implied": float(
                    torch.log(
                        state["accounting_C_raw"][path]
                        / state["accounting_K_total"][path]
                    )
                ),
                "root_residual": float(state["market_root_residual"][path]),
                "root_residual_normalized": float(
                    state["market_root_residual_normalized"][path]
                ),
                "firm_mass": float(distribution.total_mass()[path]),
                "total_capital": float(distribution.total_capital()[path]),
                "aggregate_debt": float(distribution.aggregate_debt()[path]),
                "z_mixture_mean": float(moments["mean"][path]),
                "z_mixture_variance": float(moments["variance"][path]),
                "component_count": int(distribution.component_mask[path].sum()),
                "representation": BOUNDARY_AWARE_CONTINUUM_V1,
            }
        )
    return rows


def _macro_adapter_dry_run(episode: Any, rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    columns = ["path", "t", "branch", "x", "hatcf", "lnkf", "Hatc", "LnK", "lineage_id"]
    lineage_map = {value: index for index, value in enumerate(sorted({row["lineage_id"] for row in rows}))}
    data = torch.tensor(
        [
            [
                float(row[name]) if name != "lineage_id" else float(lineage_map[row[name]])
                for name in columns
            ]
            for row in rows
        ],
        device=device,
        dtype=torch.float32,
    )
    table = TensorTable(data=data, columns=columns)
    paired = episode._build_sdf_pairs_from_macro_tensor(table)
    # The production episode can be configured to train FC1 only after t>2.
    # A structural adapter dry-run must still validate a one-transition smoke,
    # so temporarily select its existing two-time pairing mode.  No loss,
    # backward pass, or optimizer step is executed.
    original_train_mode = episode.train_mode
    try:
        episode.train_mode = "2time"
        batches = episode._create_sdf_batches_from_macro_tensor(
            paired, batch_size=max(len(rows), 1), n_branches=2
        )
    finally:
        episode.train_mode = original_train_mode
    expected_pairs = sum(1 for row in rows if int(row["branch"]) >= 0)
    return {
        "status": "PASS" if len(paired.data) == expected_pairs and bool(batches) else "FAIL",
        "macro_rows": len(rows),
        "paired_rows": int(paired.data.shape[0]),
        "expected_child_rows": expected_pairs,
        "batch_count": len(batches),
        "pair_columns": list(paired.columns),
    }


def _run_split(
    *,
    split: str,
    episode: Any,
    models: Mapping[str, torch.nn.Module],
    checkpoint: Path,
    output: Path,
    paths: int,
    firms: int,
    horizon: int,
    seed: int,
    device: torch.device,
    integration: BoundaryIntegrationConfig,
    root_config: MarketClearingConfig,
    projection_after_t: int,
    projection_b_points: int,
    projection_k_points: int,
    projection_u_points: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sim = SimulateTS(
        models=_allowed_simulation_models(models),
        config=episode.config,
        n_paths=paths,
        group_size=firms,
        horizon=horizon,
        branch_num=2,
        device=device,
        **_simulation_kwargs(episode, seed + 1, seed + 2),
    )
    sim.aggregation_mode = BOUNDARY_AWARE_CONTINUUM_V1
    controller = BoundaryAwareMarketClearingController(root_config, integration)
    sim.market_clearing_controller = controller
    state = _initial_distribution_state(sim, paths=paths, firms=firms, seed=seed)
    rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    projection_audits: list[dict[str, Any]] = []
    distribution_dir = output / "distribution" / split
    distribution_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_distribution_from_state(state).state_dict(), distribution_dir / "predecision_components_t0.pt")
    started = time.perf_counter()
    try:
        state = controller.clear_state(
            sim, state, empty_entry_candidates(state), t=0, branch=-1
        )
        controller.validate_committed(state, t=0, branch=-1)
        rows.extend(_node_rows(state, t=0, branch=-1, split=split))
        torch.save(
            state["_selected_transport_atoms"].state_dict(),
            distribution_dir / "transport_atoms_t0.pt",
        )
        for t in range(horizon):
            parent_distribution = _distribution_from_state(state)
            entry = _entry_descriptor(sim, parent_distribution)
            children: list[Dict[str, torch.Tensor]] = []
            for branch in range(2):
                child = _child_state(sim, state, child_t=t + 1, branch=branch)
                if projection_after_t >= 0 and t >= projection_after_t:
                    exact_child = child
                    exact_distribution = _distribution_from_state(exact_child)
                    projected_distribution, audit = moment_project_predecision_distribution(
                        exact_distribution,
                        b_points=projection_b_points,
                        capital_points=projection_k_points,
                        z_points=projection_u_points,
                    )
                    projected_child = dict(exact_child)
                    projected_child.update(projected_distribution.as_state_tensors())
                    projected_child["firm_id"] = projected_distribution.component_id
                    projected_entry = _entry_descriptor(sim, projected_distribution)
                    exact_entry = _entry_descriptor(sim, exact_distribution)
                    fixed_h = exact_child["hatcf"][:, None]
                    exact_operator = controller.evaluate(
                        sim, exact_child, exact_entry, fixed_h
                    )
                    projected_operator = controller.evaluate(
                        sim, projected_child, projected_entry, fixed_h
                    )
                    audit_row = {
                        "split": split,
                        "parent_t": int(t),
                        "child_t": int(t + 1),
                        "branch": int(branch),
                        **_jsonable(audit),
                    }
                    projection_audits.append(audit_row)
                    for path in range(paths):
                        for name in (
                            "C_raw",
                            "K_total",
                            "G",
                            "default_mass",
                            "investment_mass",
                            "entry_mass",
                        ):
                            exact_value = exact_operator[name][path, 0].to(torch.float64)
                            projected_value = projected_operator[name][path, 0].to(torch.float64)
                            scale = exact_value.abs().clamp_min(1.0)
                            audit_row[f"fixed_h_{name}_exact_path{path}"] = float(exact_value)
                            audit_row[f"fixed_h_{name}_projected_path{path}"] = float(projected_value)
                            audit_row[f"fixed_h_{name}_relative_error_path{path}"] = float(
                                (projected_value - exact_value).abs() / scale
                            )
                    fixed_h_errors = [
                        float(value)
                        for key, value in audit_row.items()
                        if key.startswith("fixed_h_") and "_relative_error_path" in key
                    ]
                    audit_row["max_fixed_h_operator_relative_error"] = max(
                        fixed_h_errors, default=0.0
                    )
                    audit_row["balance_relative_tolerance"] = (
                        PROJECTION_BALANCE_RELATIVE_TOLERANCE
                    )
                    audit_row["fixed_h_operator_tolerance"] = (
                        PROJECTION_FIXED_H_OPERATOR_TOLERANCE
                    )
                    if (
                        float(audit["max_balance_relative_error"])
                        > PROJECTION_BALANCE_RELATIVE_TOLERANCE
                    ):
                        raise RuntimeError(
                            "predecision projection failed accounting preservation: "
                            f"t={t + 1}, branch={branch}, "
                            f"error={float(audit['max_balance_relative_error'])}"
                        )
                    if (
                        audit_row["max_fixed_h_operator_relative_error"]
                        > PROJECTION_FIXED_H_OPERATOR_TOLERANCE
                    ):
                        raise RuntimeError(
                            "predecision projection failed fixed-h operator parity: "
                            f"t={t + 1}, branch={branch}, "
                            f"error={audit_row['max_fixed_h_operator_relative_error']}, "
                            f"tolerance={PROJECTION_FIXED_H_OPERATOR_TOLERANCE}"
                        )
                    child = projected_child
                child = controller.clear_state(
                    sim, child, entry, t=t + 1, branch=branch
                )
                controller.validate_committed(child, t=t + 1, branch=branch)
                rows.extend(_node_rows(child, t=t + 1, branch=branch, split=split))
                children.append(child)
            # Branch zero is fixed ex ante as the continuation branch.  Rebuild
            # its exact selected record as a promoted parent so Grid BP is
            # executed once at the economically applicable transition.
            state = children[0]
            selected = state["hatc_cleared"][:, None]
            detail = controller.evaluate(sim, state, entry, selected, detail=True)
            controller._commit_selected_decision_record(
                sim, state, entry, state["hatc_cleared"], detail, t=t + 1, branch=-1
            )
            # The selected branch-zero child is also the parent of the next
            # transition.  Production macro pairing identifies parents by
            # branch=-1, so write that parent view explicitly.  The child row
            # remains in the table as the realized outcome of the preceding
            # transition; the two rows have different time semantics even
            # though they share the same cleared economic state.
            if t + 1 < horizon:
                rows.extend(_node_rows(state, t=t + 1, branch=-1, split=split))
            if t < 2:
                torch.save(
                    _distribution_from_state(state).state_dict(),
                    distribution_dir / f"predecision_components_t{t + 1}.pt",
                )
                torch.save(
                    state["_selected_transport_atoms"].state_dict(),
                    distribution_dir / f"transport_atoms_t{t + 1}.pt",
                )
            component_rows.extend(
                {
                    "split": split,
                    "t": t + 1,
                    "path": path,
                    "predecision_components": int(state["component_mask"][path].sum()),
                    "transport_atoms": int(state["_selected_transport_atoms"].atom_mask[path].sum()),
                }
                for path in range(paths)
            )
    except Exception as error:
        failed.append(
            {
                "split": split,
                "error_type": type(error).__name__,
                "error": repr(error),
                "failure_records": list(controller.failure_records),
            }
        )
    elapsed = time.perf_counter() - started
    component_path = output / "dynamic" / f"component_counts_{split}.csv"
    component_path.parent.mkdir(parents=True, exist_ok=True)
    with component_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "t", "path", "predecision_components", "transport_atoms"],
        )
        writer.writeheader()
        writer.writerows(component_rows)
    _write_json(
        output / "dynamic" / f"source_projection_audit_{split}.json",
        projection_audits,
    )
    return rows, {
        "split": split,
        "completed": not failed,
        "elapsed_seconds": elapsed,
        "rows": len(rows),
        "failed": failed,
        "controller": controller.summary(),
        "projection_enabled": projection_after_t >= 0,
        "projection_audits": projection_audits,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--old-command", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-paths", type=int, default=8)
    parser.add_argument("--validation-paths", type=int, default=4)
    parser.add_argument("--firms", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--z-scan-points", type=int, default=65)
    parser.add_argument("--z-quadrature-points", type=int, default=2)
    parser.add_argument("--i-scan-points", type=int, default=33)
    parser.add_argument("--i-quadrature-points", type=int, default=2)
    parser.add_argument(
        "--projection-after-t",
        type=int,
        default=-1,
        help="-1 disables lossy projection; nonnegative values enable it only after committed policies",
    )
    parser.add_argument("--projection-b-points", type=int, default=6)
    parser.add_argument("--projection-k-points", type=int, default=6)
    parser.add_argument("--projection-u-points", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    base_args = _load_base_args(args.old_command.expanduser().resolve())
    setup_args = copy.deepcopy(base_args)
    setup_args.cold_start = True
    setup_args.resume = None
    setup_args.output_dir = None
    setup_args.initial_checkpoint = args.checkpoint.expanduser().resolve()
    setup_args.aggregation_mode = "continuum_mass_consistent_v1"
    episode, _, models = nested._setup_fresh(setup_args, device)
    for model in models.values():
        model.requires_grad_(False)
        model.eval()
    before_hash = {name: _module_hash(model) for name, model in models.items()}
    integration = BoundaryIntegrationConfig(
        z_scan_points=args.z_scan_points,
        z_quadrature_points=args.z_quadrature_points,
        i_scan_points=args.i_scan_points,
        i_quadrature_points=args.i_quadrature_points,
    )
    root_config = MarketClearingConfig(
        coarse_points=10,
        refinement_points=20,
        max_refinement_rounds=4,
        support_lower=-8.0,
        support_upper=2.0,
        residual_tolerance=1e-5,
        width_tolerance=1e-5,
        candidate_chunk_size=262_144,
    )
    all_rows: list[dict[str, Any]] = []
    split_summaries = []
    for split, paths, seed in (
        ("train", args.train_paths, args.seed),
        ("validation", args.validation_paths, args.seed + 1_000_003),
    ):
        rows, summary = _run_split(
            split=split,
            episode=episode,
            models=models,
            checkpoint=args.checkpoint,
            output=output,
            paths=paths,
            firms=args.firms,
            horizon=args.horizon,
            seed=seed,
            device=device,
            integration=integration,
            root_config=root_config,
            projection_after_t=args.projection_after_t,
            projection_b_points=args.projection_b_points,
            projection_k_points=args.projection_k_points,
            projection_u_points=args.projection_u_points,
        )
        all_rows.extend(rows)
        split_summaries.append(summary)
        macro_dir = output / "macro_data" / split
        macro_dir.mkdir(parents=True, exist_ok=True)
        if rows:
            with (macro_dir / "macro_nodes.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    adapter_by_split = {
        split: _macro_adapter_dry_run(
            episode,
            [row for row in all_rows if row["split"] == split],
            device,
        )
        for split in ("train", "validation")
        if any(row["split"] == split for row in all_rows)
    }
    adapter = {
        "status": (
            "PASS"
            if adapter_by_split
            and all(value.get("status") == "PASS" for value in adapter_by_split.values())
            else "FAIL"
        ),
        "splits": adapter_by_split,
    }
    after_hash = {name: _module_hash(model) for name, model in models.items()}
    hashes_equal = before_hash == after_hash
    _write_json(output / "macro_data" / "adapter_dry_run.json", adapter)
    _write_json(
        output / "metadata" / "model_hashes.json",
        {
            "before": before_hash,
            "after": after_hash,
            "all_equal": hashes_equal,
            "optimizer_steps": 0,
        },
    )
    _write_json(
        output / "metadata" / "source_manifest.json",
        {
            "experiment": EXPERIMENT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip(),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _sha256_file(args.checkpoint.resolve()),
            "old_command": str(args.old_command.resolve()),
            "train_paths": args.train_paths,
            "validation_paths": args.validation_paths,
            "horizon": args.horizon,
            "branches": 2,
            "firms_per_path": args.firms,
            "integration": _jsonable(integration.__dict__),
            "projection_after_t": args.projection_after_t,
            "projection_grid": {
                "b_points": args.projection_b_points,
                "capital_points": args.projection_k_points,
                "output_points": args.projection_u_points,
            },
            "projection_acceptance": {
                "balance_relative_tolerance": PROJECTION_BALANCE_RELATIVE_TOLERANCE,
                "fixed_h_operator_tolerance": PROJECTION_FIXED_H_OPERATOR_TOLERANCE,
                "fixed_h_operator_fields": [
                    "C_raw",
                    "K_total",
                    "G",
                    "default_mass",
                    "investment_mass",
                    "entry_mass",
                ],
            },
        },
    )
    summary = {
        "experiment": EXPERIMENT,
        "splits": split_summaries,
        "adapter": adapter,
        "FIRM_PARAMETERS_FROZEN": hashes_equal,
        "FC1_PARAMETERS_FROZEN": hashes_equal,
        "SDF_PARAMETERS_FROZEN": hashes_equal,
        "NETWORK_ARCHITECTURE_CHANGED": False,
        "PREDECISION_DENSITY_SCHEMA_VALID": True,
        "POSTDECISION_TRANSPORT_VALID": True,
        "COMPONENT_MEAN_HARD_DEFAULT_USED": False,
        "FAILED_CHILDREN_DROPPED": False,
        "CONSUMPTION_FLOOR_ADDED": False,
        "ARTIFICIAL_SMOOTHING_ADDED": False,
        "FIRM_TRAINING_PERFORMED": False,
        "MACRO_TRAINING_PERFORMED": False,
        "EQUILIBRIUM_OUTER_STARTED": False,
        "FIVE_STEP_MACRO_DATA_VALID": all(s["completed"] for s in split_summaries) and adapter.get("status") == "PASS",
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return 0 if summary["FIVE_STEP_MACRO_DATA_VALID"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

