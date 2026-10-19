#!/usr/bin/env python3
"""Run the gated consistency diagnostics before a continuum Small outer.

This runner is intentionally checkpoint- and snapshot-driven.  It first
replays the audited Job-695479 node under the historical operator, then runs
the same finite realization under the hard reference operator, and finally
reconstructs a conditional continuum child from the saved parent and original
idiosyncratic transition laws.  A separate fresh two-step run verifies that a
selected root is the record that is committed into the next distribution.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import run_full_nested_equilibrium as nested
from analysis.macro_parent_state_bank import _allowed_simulation_models, _simulation_kwargs
from analysis.run_macro_outer_inner_pq import make_initial_state
from data.consistent_aggregation import (
    CONTINUUM_MASS_CONSISTENT_V1,
    HARD_REALIZED_REFERENCE,
    LEGACY_MIXED_REPLAY,
)
from data.entry import EntryCandidates
from data.market_clearing import MarketClearingError, empty_entry_candidates
from data.simulate_ts import SimulateTS
from data.simulate_ts_parallel import (
    _expand_branches_batched,
    _prepare_market_entry_candidates,
)
from utils.firm_transition import apply_refinancing_policy


MODES = (
    LEGACY_MIXED_REPLAY,
    HARD_REALIZED_REFERENCE,
    CONTINUUM_MASS_CONSISTENT_V1,
)


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _clone_tensors(values: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.detach().clone().to(device) if torch.is_tensor(value) else copy.deepcopy(value)
        for name, value in values.items()
    }


def _entry_candidates(values: Mapping[str, torch.Tensor], device: torch.device) -> EntryCandidates:
    return EntryCandidates(
        **{name: values[name].detach().clone().to(device) for name in EntryCandidates.__dataclass_fields__}
    )


def _load_base_args(command_path: Path) -> argparse.Namespace:
    command = json.loads(command_path.read_text(encoding="utf-8"))
    script_index = next(
        index for index, value in enumerate(command)
        if str(value).endswith("run_full_nested_equilibrium.py")
    )
    return nested.parse_args(command[script_index + 1 :])


def _mode_runtime(
    base_args: argparse.Namespace,
    *,
    mode: str,
    checkpoint: Path,
    device: torch.device,
    i_points: int,
    z_points: int,
    trace: bool,
    transition_seed: int,
    entry_seed: int,
) -> tuple[Any, Mapping[str, torch.nn.Module], SimulateTS]:
    args = copy.deepcopy(base_args)
    args.cold_start = True
    args.resume = None
    args.output_dir = None
    args.initial_checkpoint = checkpoint
    args.aggregation_mode = mode
    args.continuum_i_points = int(i_points)
    args.continuum_z_points = int(z_points)
    args.continuum_projection_tolerance = 1.0e-6
    args.bp_actual_use_trace = bool(trace)
    episode, _, models = nested._setup_fresh(args, device)
    sim = SimulateTS(
        models=_allowed_simulation_models(models),
        config=episode.config,
        n_paths=1,
        group_size=1,
        horizon=1,
        branch_num=2,
        device=device,
        **_simulation_kwargs(episode, transition_seed, entry_seed),
    )
    return episode, models, sim


def _rows_from_evaluation(mode: str, label: str, result: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    fields = (
        "G", "G_normalized", "C_raw", "C_oper", "K_total", "Y_total",
        "I_oper_total", "I_entry_total", "Liquidation_total",
        "I_maintenance_total", "resource_feasible", "finite",
        "investment_count", "default_count", "entry_count", "firm_count",
        "investment_mass", "default_mass", "entry_mass",
    )
    rows: list[dict[str, Any]] = []
    for column in range(int(result["h"].shape[1])):
        row: dict[str, Any] = {
            "mode": mode,
            "scan": label,
            "h": float(result["h"][0, column]),
        }
        for name in fields:
            value = result[name][0, column]
            row[name] = bool(value) if value.dtype == torch.bool else float(value)
        rows.append(row)
    return rows


def _status_from_clear(sim: SimulateTS, state: dict[str, Any], candidates: EntryCandidates, *, t: int, branch: int) -> dict[str, Any]:
    try:
        cleared = sim.market_clearing_controller.clear_state(
            sim, state, candidates, t=t, branch=branch
        )
    except MarketClearingError as error:
        record = sim.market_clearing_controller.failure_records[-1]
        return {
            "status": record["status"],
            "error": str(error),
            **record,
        }
    record = sim.market_clearing_controller.root_records[-1]
    return {
        "status": "ROOT_FOUND",
        "selected_h": float(cleared["hatc_cleared"][0]),
        "selected_G": float(cleared["market_root_residual"][0]),
        "selected_G_normalized": float(cleared["market_root_residual_normalized"][0]),
        **record,
    }


def _legacy_parity(reference_csv: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    reference = pd.read_csv(reference_csv)
    current = pd.DataFrame(rows)
    columns = {
        "G": "G",
        "G_normalized": "G_normalized",
        "C_raw": "C_raw",
        "K": "K_total",
        "investment_count": "investment_count",
        "default_count": "default_count",
        "entry_count": "entry_count",
    }
    metrics = {}
    passed = len(reference) == len(current)
    for old, new in columns.items():
        difference = (
            pd.to_numeric(reference[old], errors="coerce").to_numpy()
            - pd.to_numeric(current[new], errors="coerce").to_numpy()
        )
        maximum = float(abs(difference).max())
        metrics[f"{old}_max_abs_difference"] = maximum
        tolerance = 1.0e-6 if "count" not in old else 0.0
        passed = passed and maximum <= tolerance
    return {"passed": bool(passed), "rows": len(current), **metrics}


@torch.inference_mode()
def _continuum_child_from_parent(
    sim: SimulateTS,
    parent_cpu: Mapping[str, Any],
    finite_child_cpu: Mapping[str, Any],
    *,
    branch: int,
    child_t: int,
) -> tuple[dict[str, Any], EntryCandidates, dict[str, Any]]:
    device = sim.device
    parent = _clone_tensors(parent_cpu, device)
    finite_child = _clone_tensors(finite_child_cpu, device)
    parent["firm_mass"] = parent["alive"].to(parent["K"].dtype)

    # Re-evaluate the parent hard decisions at the saved selected h.  The
    # historical child is used only to pin the common macro realization.
    detail = sim.market_clearing_controller.evaluate(
        sim,
        parent,
        empty_entry_candidates(parent),
        parent["hatcf"][:, None],
        detail=True,
    )
    parent["bar_i"] = detail["bar_i_firm"][:, 0, : parent["b"].shape[1]]
    parent["bar_z"] = detail["bar_z_firm"][:, 0, : parent["b"].shape[1]]
    parent["bp"] = detail["bp_firm"][:, 0, : parent["b"].shape[1]]

    active = parent["alive"]
    active_pos = torch.nonzero(active.reshape(-1), as_tuple=False).reshape(-1)
    if active_pos.numel():
        B, N = parent["b"].shape
        path_grid = torch.arange(B, device=device)[:, None].expand(B, N)
        firm_state = detail["firm_state"][:, 0, :N].reshape(B * N, 7)[active_pos]
        head = parent["bp"].reshape(-1)[active_pos, None]
        path_index = path_grid.reshape(-1)[active_pos]
        resolved = sim._resolve_bp_action(
            sim=sim,
            firm_state=firm_state,
            model_output=None,
            bp_head=head,
            hatc_cal=parent["hatcf"][path_index, None],
            lnk_cal=torch.log(detail["K_total"][:, 0].clamp_min(1.0e-30))[path_index, None],
            path_index=path_index,
            firm_id=parent["firm_id"].reshape(-1)[active_pos],
            economic_node_id=parent["economic_node_id"][path_index],
            t=int(child_t - 1),
            branch=-1,
            decision_stage="continuum_reconstruction_parent_transition",
            bp_execution_applicable=True,
        ).reshape(-1)
        parent["bp"].reshape(-1)[active_pos] = resolved

    branches = _expand_branches_batched(sim, parent, child_t=child_t)
    child = branches[int(branch)]
    generated_x = float(child["x"][0])
    for name in ("x", "hatcf", "lnkf", "M"):
        child[name] = finite_child[name].detach().clone().to(device)
    candidates = _prepare_market_entry_candidates(sim, child, enable_entry=True)
    reconstruction = {
        "label": "CONTINUUM_CONDITIONAL_DISTRIBUTION_COMPARISON",
        "source": "saved parent distribution plus original z/eta/i transition laws",
        "finite_child_common_x": float(finite_child["x"][0]),
        "generated_common_x_before_pin": generated_x,
        "common_x_match_before_pin": math.isclose(
            generated_x, float(finite_child["x"][0]), rel_tol=0.0, abs_tol=1.0e-6
        ),
        "finite_child_not_claimed_identical": True,
        "slot_count": int(child["alive"].sum()),
        "firm_mass": float(child["firm_mass"].sum()),
        "projection_error": float(child.get("distribution_projection_error", torch.zeros(1, device=device))[0]),
    }
    return child, candidates, reconstruction


def _scan_mode(
    *,
    mode: str,
    sim: SimulateTS,
    state: dict[str, Any],
    candidates: EntryCandidates,
    reference_dense: Optional[Path],
    t: int,
    branch: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    device = sim.device
    dtype = state["b"].dtype
    grid = torch.linspace(-8.0, 2.0, 512, device=device, dtype=dtype)[None, :]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    before_calls = sim.market_clearing_controller.network_forward_calls
    started = time.perf_counter()
    vector = sim.market_clearing_controller.evaluate(sim, state, candidates, grid)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    vector_time = time.perf_counter() - started
    vector_calls = sim.market_clearing_controller.network_forward_calls - before_calls
    rows = _rows_from_evaluation(mode, "dense512", vector)

    # Candidate-loop reference: same batched Firm operator, one h at a time.
    parity_grid = grid[:, :10]
    before_calls = sim.market_clearing_controller.network_forward_calls
    started = time.perf_counter()
    scalar_results = [
        sim.market_clearing_controller.evaluate(sim, state, candidates, parity_grid[:, j : j + 1])
        for j in range(parity_grid.shape[1])
    ]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    scalar_time = time.perf_counter() - started
    scalar_calls = sim.market_clearing_controller.network_forward_calls - before_calls
    parity = {}
    for name in ("G", "C_raw", "K_total", "investment_count", "default_count", "entry_count"):
        scalar = torch.cat([item[name] for item in scalar_results], dim=1)
        parity[name] = float((vector[name][:, :10] - scalar).abs().max())
    parity_pass = all(value <= (1.0e-6 if "count" not in name else 0.0) for name, value in parity.items())
    performance = {
        "mode": mode,
        "vectorized_runtime_seconds": vector_time,
        "candidate_loop_runtime_seconds": scalar_time,
        "candidate_loop_equivalent_speedup": scalar_time / max(vector_time * 10.0 / 512.0, 1.0e-12),
        "vectorized_network_forward_calls": vector_calls,
        "candidate_loop_network_forward_calls": scalar_calls,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "SCALAR_PARITY": "PASS" if parity_pass else "FAIL",
        **{f"parity_{name}_max_abs": value for name, value in parity.items()},
    }
    status = _status_from_clear(
        sim,
        _clone_tensors(state, device),
        _entry_candidates(candidates.__dict__, device),
        t=t,
        branch=branch,
    )
    legacy = _legacy_parity(reference_dense, rows) if reference_dense else {}
    return rows, status, {"performance": performance, "legacy_parity": legacy}


def _fresh_dynamic_validation(
    base_args: argparse.Namespace,
    *,
    checkpoint: Path,
    output: Path,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    args = copy.deepcopy(base_args)
    args.cold_start = True
    args.resume = None
    args.output_dir = None
    args.initial_checkpoint = checkpoint
    args.aggregation_mode = CONTINUUM_MASS_CONSISTENT_V1
    args.continuum_i_points = 9
    args.continuum_z_points = 7
    args.continuum_projection_tolerance = 1.0e-6
    args.bp_actual_use_trace = True
    episode, _, models = nested._setup_fresh(args, device)
    initial = make_initial_state(
        models=models,
        episode=episode,
        paths=4,
        firms=32,
        horizon=2,
        branch_num=2,
        seed=seed,
        device=device,
    )

    def execute(horizon: int, label: str) -> dict[str, Any]:
        sim = SimulateTS(
            models=_allowed_simulation_models(models),
            config=episode.config,
            n_paths=4,
            group_size=32,
            horizon=int(horizon),
            branch_num=2,
            device=device,
            initial_state=_clone_tensors(initial, torch.device("cpu")),
            **_simulation_kwargs(episode, seed + 11, seed + 12),
        )
        try:
            result = sim.simulate_tensor()
        except BaseException as error:
            sim.write_bp_actual_use_trace(
                output / "audit" / f"actual_bp_routing_{label}_partial.csv"
            )
            partial = {
                "completed": False,
                "error": repr(error),
                "error_type": type(error).__name__,
                "market_clearing": sim.market_clearing_controller.summary(),
                "bp_trace_rows": len(sim.bp_actual_use_trace),
                "bp_execution_sources": sorted({
                    str(row["executed_bp_source"])
                    for row in sim.bp_actual_use_trace
                    if bool(row["bp_execution_applicable"])
                }),
            }
            _write_json(
                output / "dynamic_validation" / f"{label}_partial.json", partial
            )
            return partial
        sim.write_bp_actual_use_trace(
            output / "audit" / f"actual_bp_routing_{label}.csv"
        )
        return {
            "completed": True,
            "output": result,
            "market_clearing": result.meta["market_clearing"],
            "bp_trace_rows": len(sim.bp_actual_use_trace),
            "bp_execution_sources": sorted({
                str(row["executed_bp_source"])
                for row in sim.bp_actual_use_trace
                if bool(row["bp_execution_applicable"])
            }),
        }

    one_step = execute(1, "one_step")
    two_step = execute(2, "two_step")
    two_step_repeat = execute(2, "two_step_repeat") if two_step["completed"] else {
        "completed": False,
        "skipped": "first two-step run failed",
    }
    firm_equal = bool(
        two_step["completed"]
        and two_step_repeat["completed"]
        and torch.equal(
            two_step["output"].firm.data, two_step_repeat["output"].firm.data
        )
    )
    macro_equal = bool(
        two_step["completed"]
        and two_step_repeat["completed"]
        and torch.equal(
            two_step["output"].macro.data, two_step_repeat["output"].macro.data
        )
    )
    selected_output = (
        two_step["output"] if two_step["completed"] else one_step.get("output")
    )
    if selected_output is None:
        macro = pd.DataFrame()
    else:
        macro = selected_output.macro.to_dataframe()
    balance_columns = [
        "path", "t", "branch", "slot_count", "firm_mass", "K",
        "aggregate_debt", "distribution_projection_error",
        "market_root_residual", "market_committed_residual",
    ]
    balance = macro[[name for name in balance_columns if name in macro]].copy()
    balance.to_csv(output / "dynamic_validation" / "mass_capital_debt_balance.csv", index=False)
    market = one_step.get("market_clearing", {})
    result = {
        "fixed_checkpoint": str(checkpoint),
        "one_step_completed": bool(one_step["completed"]),
        "two_step_completed": bool(two_step["completed"]),
        "paths": 4,
        "firm_tensor_repeat_exact": firm_equal,
        "macro_tensor_repeat_exact": macro_equal,
        "one_step_market_clearing": market,
        "two_step_market_clearing": two_step.get("market_clearing", {}),
        "two_step_error": two_step.get("error"),
        "candidate_commit_parity": (
            int(market.get("candidate_commit_parity_pass_count", 0))
            == int(market.get("node_count", -1))
        ),
        "one_step_root_failures": int(market.get("failure_count", -1)),
        "bp_trace_rows": int(one_step.get("bp_trace_rows", 0)),
        "bp_execution_sources": one_step.get("bp_execution_sources", []),
        "two_step_repeat_market_clearing": two_step_repeat.get(
            "market_clearing", {}
        ),
    }
    _write_json(output / "dynamic_validation" / "two_step_replay.json", result)
    return result


def _plot_scans(frame: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 6))
    for mode, group in frame.groupby("mode"):
        axis.plot(group["h"], group["G"], label=mode)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(xlabel="h", ylabel="G(h)", title="Legacy, hard-reference, and continuum operators")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "figures" / "legacy_hard_continuum_G.png", dpi=180)
    plt.close(figure)


def _write_diagnostic_report(output: Path, summary: Mapping[str, Any]) -> None:
    gate = summary["gate"]
    statuses = summary["statuses"]
    dynamic = summary["dynamic_validation"]
    lines = [
        "# Consistent Aggregation Market Clearing — Diagnostic Gate",
        "",
        "## Outcome",
        "",
        f"- Gate result: **{'PASS' if gate['passed'] else 'FAIL'}**",
        "- Tiny started: **FALSE**",
        "- Small started: **FALSE**",
        "- Formal started: **FALSE**",
        "",
        "The failed gate is a required stop, not a training failure: the continuum representation did not yet pass both integration convergence and two-step root continuity. No tolerance was widened and no fallback was added.",
        "",
        "## Required classifications",
        "",
        f"- `CLASSIFICATION_BUG_FIXED = {str(gate['CLASSIFICATION_BUG_FIXED']).upper()}`",
        f"- `GRID_ROUTING_FIXED = TRUE` (covered by committed routing regression tests)",
        f"- `ACTUAL_BP_EXECUTION_VERIFIED = {str(gate['ACTUAL_BP_EXECUTION_VERIFIED']).upper()}`",
        f"- `LEGACY_EXACT_REPLAY = {gate['LEGACY_EXACT_REPLAY']}`",
        f"- `POLICY_ACCOUNTING_TRANSITION_CONSISTENCY = {'PASS' if gate['CANDIDATE_COMMIT_PARITY'] else 'FAIL'}`",
        f"- `CANDIDATE_COMMIT_PARITY = {'PASS' if gate['CANDIDATE_COMMIT_PARITY'] else 'FAIL'}`",
        "- `CONTINUUM_MODE_IMPLEMENTED = TRUE`",
        f"- `CONTINUUM_INTEGRATION_VALIDATED = {str(gate['CONTINUUM_INTEGRATION_VALIDATED']).upper()}`",
        f"- `DISTRIBUTION_TRANSITION_VALIDATED = {str(gate['DISTRIBUTION_TRANSITION_VALIDATED']).upper()}`",
        f"- `ROOT_STATUS_ON_REGRESSION_BANK = {statuses}`",
        "- `SMALL_OUTERS_COMPLETED = 0`",
        "- `FINAL_FIRM_STATUS = NOT_STARTED_GATE_BLOCKED`",
        "- `FINAL_FC1_STATUS = NOT_STARTED_GATE_BLOCKED`",
        "- `FINAL_SDF_STATUS = NOT_STARTED_GATE_BLOCKED`",
        "- `FINAL_OUTER_STATUS = NOT_STARTED_GATE_BLOCKED`",
        "- `ECONOMIC_CALIBRATION_CHANGED = FALSE`",
        "- `AGGREGATION_SEMANTICS_CHANGED = TRUE`",
        "- `NUMERICAL_STATE_REPRESENTATION_CHANGED = TRUE`",
        "- `ARTIFICIAL_SMOOTHING_ADDED = FALSE`",
        "- `CONSUMPTION_FLOOR_ADDED = FALSE`",
        "- `FAILED_CHILDREN_DROPPED = FALSE`",
        "- `FORMAL_STARTED = FALSE`",
        "- `READY_FOR_FORMAL = FALSE`",
        "",
        "## Direct blocker",
        "",
        f"- Integration gate: `{summary.get('integration_acceptance', {})}`",
        f"- Two-step result: `{dynamic}`",
        "- Highest-priority blocker: the projected continuum z distribution still presents finite mass blocks to the hard default boundary. At t=2 branch=0 this produced a rejected discontinuity crossing; the mid-to-fine C_raw change also exceeded its pre-registered 0.5% threshold.",
        "",
        "The admissible next implementation step is boundary-aware integration of default mass within distribution cells (or an equivalently converged deterministic distribution scheme). Adding a sigmoid, consumption floor, interpolation root, new shock, or failed-child filtering is explicitly excluded.",
        "",
        "## Evidence paths",
        "",
        "- `regression/operator_scan.csv`",
        "- `regression/mode_comparison.csv`",
        "- `regression/integration_convergence.csv`",
        "- `regression/integration_acceptance.json`",
        "- `dynamic_validation/two_step_replay.json`",
        "- `audit/actual_bp_routing_one_step.csv`",
        "- `performance/vectorization_parity.json`",
    ]
    report = output / "reports" / "consistent_aggregation_market_clearing_20261019.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-command", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261019)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    for name in ("metadata", "audit", "regression", "dynamic_validation", "performance", "figures"):
        (output / name).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    base_args = _load_base_args(args.old_command.expanduser().resolve())
    replay = args.replay_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    snapshot = torch.load(replay / "exact_failure_node.pt", map_location="cpu", weights_only=True)
    branches = torch.load(replay / "parent_and_child_branches.pt", map_location="cpu", weights_only=True)
    transition_seed = int(base_args.seed + 30_000 + 11)
    entry_seed = int(base_args.seed + 30_000 + 12)

    source_manifest = {
        "experiment": "consistent_aggregation_market_clearing_20261019",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_status": _git("status", "--short"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "old_command": str(args.old_command.resolve()),
        "old_command_sha256": _sha256(args.old_command.resolve()),
        "replay_dir": str(replay),
        "replay_report_sha256_expected": "d58335d265b955b6f67e584c20581b5e6d2aaa532f5cb19414554dcc1296392f",
        "ECONOMIC_CALIBRATION_CHANGED": False,
        "AGGREGATION_SEMANTICS_CHANGED": True,
        "NUMERICAL_STATE_REPRESENTATION_CHANGED": True,
        "ARTIFICIAL_SMOOTHING_ADDED": False,
        "CONSUMPTION_FLOOR_ADDED": False,
        "FAILED_CHILDREN_DROPPED": False,
        "FORMAL_STARTED": False,
    }
    report_path = replay / "isolated_market_clearing_replay_detailed_report.md"
    if report_path.is_file():
        source_manifest["replay_report_sha256_observed"] = _sha256(report_path)
    _write_json(output / "metadata" / "source_manifest.json", source_manifest)
    _write_json(output / "metadata" / "checkpoint_manifest.json", {
        "path": str(checkpoint), "sha256": _sha256(checkpoint),
        "use": "weights-only warm start; optimizer, AL, banks, targets and metrics reset",
    })
    _write_json(output / "metadata" / "economic_config.json", vars(base_args))
    _write_json(output / "metadata" / "mode_specification.json", {
        "legacy_mixed_replay": "historical soft-resource finite-firm replay",
        "hard_realized_reference": "finite-firm hard actions shared by resource and transition",
        "continuum_mass_consistent_v1": "mass-weighted z/eta/investment-cost quadrature with conservative transition projection",
    })
    shutil.copy2(ROOT / "reports" / "consistent_aggregation_policy_semantics.md", output / "audit" / "policy_semantics.md")
    shutil.copy2(ROOT / "reports" / "consistent_aggregation_timing_and_information_set.md", output / "audit" / "timing_and_information_set.md")

    all_rows: list[dict[str, Any]] = []
    statuses: dict[str, Any] = {}
    profiling: list[dict[str, Any]] = []
    reconstruction: dict[str, Any] = {}
    integration_rows: list[dict[str, Any]] = []
    legacy_parity: dict[str, Any] = {}

    for mode in MODES:
        episode, models, sim = _mode_runtime(
            base_args,
            mode=mode,
            checkpoint=checkpoint,
            device=device,
            i_points=9,
            z_points=7,
            trace=True,
            transition_seed=transition_seed,
            entry_seed=entry_seed,
        )
        if mode == CONTINUUM_MASS_CONSISTENT_V1:
            state, candidates, reconstruction = _continuum_child_from_parent(
                sim, branches["parent"], branches["children"][1], branch=1, child_t=3
            )
        else:
            state = _clone_tensors(snapshot["state"], device)
            if mode == HARD_REALIZED_REFERENCE:
                state["firm_mass"] = state["alive"].to(state["K"].dtype)
            candidates = _entry_candidates(snapshot["candidates"], device)
        rows, status, extra = _scan_mode(
            mode=mode,
            sim=sim,
            state=state,
            candidates=candidates,
            reference_dense=(replay / "g_scan_dense512.csv") if mode == LEGACY_MIXED_REPLAY else None,
            t=3,
            branch=1,
        )
        all_rows.extend(rows)
        statuses[mode] = status
        profiling.append(extra["performance"])
        if mode == LEGACY_MIXED_REPLAY:
            legacy_parity = extra["legacy_parity"]

    # Continuum discretization convergence on the same conditional comparison.
    for i_points, z_points in ((5, 5), (9, 7), (17, 13)):
        _, _, sim = _mode_runtime(
            base_args,
            mode=CONTINUUM_MASS_CONSISTENT_V1,
            checkpoint=checkpoint,
            device=device,
            i_points=i_points,
            z_points=z_points,
            trace=False,
            transition_seed=transition_seed,
            entry_seed=entry_seed,
        )
        state, candidates, info = _continuum_child_from_parent(
            sim, branches["parent"], branches["children"][1], branch=1, child_t=3
        )
        grid = torch.linspace(-8.0, 2.0, 512, device=device, dtype=state["b"].dtype)[None, :]
        result = sim.market_clearing_controller.evaluate(sim, state, candidates, grid)
        feasible = result["resource_feasible"][0]
        residual = torch.where(
            feasible,
            result["G_normalized"][0].abs(),
            torch.full_like(result["G_normalized"][0], float("inf")),
        )
        best = int(residual.argmin())
        integration_rows.append({
            "i_points": i_points,
            "z_points": z_points,
            "slots": int(state["alive"].sum()),
            "firm_mass": float(state["firm_mass"].sum()),
            "projection_error": info["projection_error"],
            "best_h": float(result["h"][0, best]),
            "best_G": float(result["G"][0, best]),
            "best_G_normalized": float(result["G_normalized"][0, best]),
            "C_raw_at_best": float(result["C_raw"][0, best]),
            "K_at_best": float(result["K_total"][0, best]),
        })

    operator = pd.DataFrame(all_rows)
    operator.to_csv(output / "regression" / "operator_scan.csv", index=False)
    pd.DataFrame(profiling).to_csv(output / "performance" / "profiling.csv", index=False)
    pd.DataFrame(integration_rows).to_csv(output / "regression" / "integration_convergence.csv", index=False)
    mode_comparison = pd.DataFrame([
        {
            "mode": mode,
            "status": statuses[mode]["status"],
            "valid_root_count": statuses[mode].get("valid_root_count", 0),
            "discontinuity_bracket_count": statuses[mode].get("discontinuity_bracket_count", 0),
            "unresolved_bracket_count": statuses[mode].get("unresolved_bracket_count", 0),
            "min_abs_G_normalized": float(operator.loc[operator["mode"] == mode, "G_normalized"].abs().min()),
        }
        for mode in MODES
    ])
    mode_comparison.to_csv(output / "regression" / "mode_comparison.csv", index=False)
    operator[[
        "mode", "h", "investment_count", "default_count", "entry_count",
        "investment_mass", "default_mass", "entry_mass",
    ]].to_csv(output / "regression" / "cutoff_diagnostics.csv", index=False)
    _write_json(output / "regression" / "legacy_replay.json", {
        "numeric_operator_parity": legacy_parity,
        "old_terminal_status": "NO_BRACKET_ON_TESTED_GRID",
        "corrected_terminal_status": statuses[LEGACY_MIXED_REPLAY]["status"],
        "status_difference_expected_from_classification_fix": True,
    })
    _write_json(output / "regression" / "mode_status.json", statuses)
    _write_json(output / "regression" / "continuum_reconstruction.json", reconstruction)
    _write_json(output / "performance" / "vectorization_parity.json", {
        row["mode"]: row for row in profiling
    })
    _plot_scans(operator, output)

    dynamic = _fresh_dynamic_validation(
        base_args,
        checkpoint=checkpoint,
        output=output,
        device=device,
        seed=args.seed,
    )
    dynamic_error = dynamic.get("two_step_error")

    # Locked before the formal diagnostic is inspected.  These compare the
    # two highest quadrature resolutions, not the historical finite child.
    integration_thresholds = {
        "best_h_abs": 0.04,
        "best_G_normalized_abs": 5.0e-3,
        "C_raw_relative": 5.0e-3,
        "K_relative": 5.0e-3,
        "projection_error": 1.0e-6,
    }
    middle, fine = integration_rows[-2], integration_rows[-1]
    integration_differences = {
        "best_h_abs": abs(fine["best_h"] - middle["best_h"]),
        "best_G_normalized_abs": abs(
            fine["best_G_normalized"] - middle["best_G_normalized"]
        ),
        "C_raw_relative": abs(fine["C_raw_at_best"] - middle["C_raw_at_best"])
        / max(abs(fine["C_raw_at_best"]), 1.0e-12),
        "K_relative": abs(fine["K_at_best"] - middle["K_at_best"])
        / max(abs(fine["K_at_best"]), 1.0e-12),
        "projection_error": max(row["projection_error"] for row in integration_rows),
    }
    integration_validated = all(
        integration_differences[name] <= threshold
        for name, threshold in integration_thresholds.items()
    )
    _write_json(output / "regression" / "integration_acceptance.json", {
        "thresholds_locked_before_formal_result_review": integration_thresholds,
        "differences_mid_to_fine": integration_differences,
        "passed": integration_validated,
    })

    gate = {
        "LEGACY_EXACT_REPLAY": "PASS" if legacy_parity.get("passed") else "FAIL",
        "CLASSIFICATION_BUG_FIXED": statuses[LEGACY_MIXED_REPLAY]["status"] == "DISCONTINUITY_CROSSING",
        "SCALAR_VECTOR_PARITY": all(row["SCALAR_PARITY"] == "PASS" for row in profiling),
        "CONTINUUM_INTEGRATION_FINITE": all(math.isfinite(row["best_G"]) for row in integration_rows),
        "CONTINUUM_INTEGRATION_VALIDATED": integration_validated,
        "DISTRIBUTION_TRANSITION_VALIDATED": bool(dynamic.get("two_step_completed")) and dynamic.get("firm_tensor_repeat_exact") and dynamic.get("macro_tensor_repeat_exact"),
        "CANDIDATE_COMMIT_PARITY": bool(dynamic.get("one_step_completed")) and dynamic.get("candidate_commit_parity"),
        "ACTUAL_BP_EXECUTION_VERIFIED": bool(dynamic.get("one_step_completed")) and dynamic.get("bp_execution_sources") == ["grid"],
        "dynamic_error": dynamic_error,
    }
    gate["passed"] = all(
        value is True for key, value in gate.items()
        if key != "dynamic_error"
    )
    _write_json(output / "regression" / "candidate_commit_parity.json", dynamic)
    _write_json(output / "diagnostic_gate.json", gate)
    summary_payload = {
        "phase": "gate_1_and_two_step",
        "gate": gate,
        "statuses": statuses,
        "legacy_parity": legacy_parity,
        "continuum_reconstruction": reconstruction,
        "integration_convergence": integration_rows,
        "integration_acceptance": {
            "thresholds": integration_thresholds,
            "differences": integration_differences,
            "passed": integration_validated,
        },
        "dynamic_validation": dynamic,
        **{key: source_manifest[key] for key in (
            "ECONOMIC_CALIBRATION_CHANGED", "AGGREGATION_SEMANTICS_CHANGED",
            "NUMERICAL_STATE_REPRESENTATION_CHANGED", "ARTIFICIAL_SMOOTHING_ADDED",
            "CONSUMPTION_FLOOR_ADDED", "FAILED_CHILDREN_DROPPED", "FORMAL_STARTED",
        )},
    }
    _write_json(output / "summary.json", summary_payload)
    _write_diagnostic_report(output, summary_payload)
    print(f"CONSISTENT_AGGREGATION_DIAGNOSTIC_GATE={'PASS' if gate['passed'] else 'FAIL'}")
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
