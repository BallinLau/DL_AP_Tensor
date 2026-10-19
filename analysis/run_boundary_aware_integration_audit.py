#!/usr/bin/env python3
"""Frozen-node fixed-h quadrature and reference-root audit."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import run_full_nested_equilibrium as nested
from analysis.macro_parent_state_bank import _allowed_simulation_models, _simulation_kwargs
from analysis.run_boundary_aware_macro_distribution import (
    _entry_descriptor,
    _distribution_from_state,
    _initial_distribution_state,
    _load_base_args,
    _module_hash,
)
from data.boundary_aware_market_clearing import (
    BoundaryAwareMarketClearingController,
    BoundaryIntegrationConfig,
)
from data.market_clearing import MarketClearingConfig
from data.simulate_ts import SimulateTS


RESOLUTIONS: Mapping[str, tuple[int, int, int, int]] = {
    "low": (65, 8, 33, 8),
    "medium": (129, 16, 65, 16),
    "high": (513, 32, 257, 32),
    "reference": (1025, 48, 513, 48),
}


def _clone_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for name, value in state.items()
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _integration_config(values: tuple[int, int, int, int]) -> BoundaryIntegrationConfig:
    z_scan, z_quad, i_scan, i_quad = values
    return BoundaryIntegrationConfig(
        z_scan_points=z_scan,
        z_quadrature_points=z_quad,
        i_scan_points=i_scan,
        i_quadrature_points=i_quad,
    )


def _root_config() -> MarketClearingConfig:
    return MarketClearingConfig(
        coarse_points=10,
        refinement_points=20,
        max_refinement_rounds=4,
        support_lower=-8.0,
        support_upper=2.0,
        residual_tolerance=1e-5,
        width_tolerance=1e-5,
        candidate_chunk_size=262_144,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--old-command", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--firms", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--operator-scan-points", type=int, default=0)
    return parser.parse_args(argv)


@torch.inference_mode()
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
    hashes_before = {name: _module_hash(model) for name, model in models.items()}
    sim = SimulateTS(
        models=_allowed_simulation_models(models),
        config=episode.config,
        n_paths=1,
        group_size=args.firms,
        horizon=1,
        branch_num=2,
        device=device,
        **_simulation_kwargs(episode, args.seed + 1, args.seed + 2),
    )
    state = _initial_distribution_state(
        sim, paths=1, firms=args.firms, seed=args.seed
    )
    entry = _entry_descriptor(
        sim,
        _distribution_from_state(state),
    )
    controllers = {
        name: BoundaryAwareMarketClearingController(
            _root_config(), _integration_config(values)
        )
        for name, values in RESOLUTIONS.items()
    }
    root_rows: list[dict[str, Any]] = []
    roots: dict[str, float] = {}
    for name in ("low", "medium", "high"):
        candidate_state = _clone_state(state)
        try:
            cleared = controllers[name].clear_state(
                sim, candidate_state, entry, t=0, branch=-1
            )
            controllers[name].validate_committed(cleared, t=0, branch=-1)
            root = float(cleared["hatc_cleared"][0])
            roots[name] = root
            root_rows.append(
                {
                    "resolution": name,
                    "root_status": "ROOT_FOUND",
                    "h_root": root,
                    "own_operator_G": float(cleared["market_root_residual"][0]),
                    "own_operator_G_normalized": float(
                        cleared["market_root_residual_normalized"][0]
                    ),
                }
            )
        except Exception as error:
            root_rows.append(
                {
                    "resolution": name,
                    "root_status": type(error).__name__,
                    "h_root": float("nan"),
                    "own_operator_G": float("nan"),
                    "own_operator_G_normalized": float("nan"),
                }
            )
    if "high" not in roots:
        raise RuntimeError("high-resolution operator did not produce a reference fixed h")
    fixed_h = torch.tensor([[roots["high"]]], device=device, dtype=state["b"].dtype)
    fixed_rows: list[dict[str, Any]] = []
    aggregate_debt = float(
        (state["component_mass"] * state["K"] * state["b"])[
            state["component_mask"]
        ].sum()
    )
    for name, values in RESOLUTIONS.items():
        result = controllers[name].evaluate(sim, state, entry, fixed_h)
        fixed_rows.append(
            {
                "resolution": name,
                "z_scan_points": values[0],
                "z_quadrature_points": values[1],
                "i_scan_points": values[2],
                "i_quadrature_points": values[3],
                "fixed_h": roots["high"],
                "default_mass": float(result["default_mass"][0, 0]),
                "investment_mass": float(result["investment_mass"][0, 0]),
                "entry_mass": float(result["entry_mass"][0, 0]),
                "C_raw": float(result["C_raw"][0, 0]),
                "K_total": float(result["K_total"][0, 0]),
                "aggregate_debt": aggregate_debt,
                "G": float(result["G"][0, 0]),
                "G_normalized": float(result["G_normalized"][0, 0]),
            }
        )
    reference = controllers["reference"]
    for row in root_rows:
        if row["root_status"] != "ROOT_FOUND":
            row["reference_G"] = float("nan")
            row["reference_G_normalized"] = float("nan")
            row["reference_root_pass"] = False
            continue
        h = torch.tensor([[row["h_root"]]], device=device, dtype=state["b"].dtype)
        result = reference.evaluate(sim, state, entry, h)
        row["reference_G"] = float(result["G"][0, 0])
        row["reference_G_normalized"] = float(result["G_normalized"][0, 0])
        row["reference_root_pass"] = bool(
            abs(row["reference_G_normalized"]) <= _root_config().residual_tolerance
        )

    detail = controllers["high"].evaluate(sim, state, entry, fixed_h, detail=True)
    boundary_rows: list[dict[str, Any]] = []
    default_probability = detail["default_boundary_probability"]
    default_mask = detail["default_boundary_mask"]
    default_residual = detail["default_boundary_residual"]
    for task, index in torch.nonzero(default_mask, as_tuple=False).detach().cpu().tolist():
        boundary_rows.append(
            {
                "boundary_kind": "incumbent_default",
                "task_index": task,
                "source_z": float("nan"),
                "probability_coordinate": float(default_probability[task, index]),
                "residual": float(default_residual[task, index]),
                "resolution": "high",
            }
        )
    investment_probability = detail["investment_boundary_probability"]
    investment_mask = detail["investment_boundary_mask"]
    investment_residual = detail["investment_boundary_residual"]
    investment_z = detail["investment_boundary_source_z"]
    for task, index in torch.nonzero(investment_mask, as_tuple=False).detach().cpu().tolist():
        boundary_rows.append(
            {
                "boundary_kind": "incumbent_investment",
                "task_index": task,
                "source_z": float(investment_z[task]),
                "probability_coordinate": float(investment_probability[task, index]),
                "residual": float(investment_residual[task, index]),
                "resolution": "high",
            }
        )

    _write_csv(output / "regression" / "fixed_h_integration_convergence.csv", fixed_rows)
    _write_csv(output / "regression" / "roots_under_reference_integrator.csv", root_rows)
    _write_csv(output / "regression" / "boundary_locations.csv", boundary_rows)
    operator_scan_rows: list[dict[str, Any]] = []
    if args.operator_scan_points > 1:
        scan_h = torch.linspace(
            max(-1.6, roots["high"] - 0.5),
            min(-0.2, roots["high"] + 0.5),
            args.operator_scan_points,
            device=device,
            dtype=state["b"].dtype,
        )
        for name in ("high", "reference"):
            controller = controllers[name]
            for start in range(0, int(scan_h.numel()), 5):
                values = scan_h[start : start + 5][None, :]
                result = controller.evaluate(sim, state, entry, values)
                for local in range(values.shape[1]):
                    operator_scan_rows.append(
                        {
                            "resolution": name,
                            "h": float(values[0, local]),
                            "G": float(result["G"][0, local]),
                            "G_normalized": float(result["G_normalized"][0, local]),
                            "C_raw": float(result["C_raw"][0, local]),
                            "K_total": float(result["K_total"][0, local]),
                            "default_mass": float(result["default_mass"][0, local]),
                            "investment_mass": float(result["investment_mass"][0, local]),
                            "entry_mass": float(result["entry_mass"][0, local]),
                        }
                    )
    _write_csv(output / "regression" / "operator_scan.csv", operator_scan_rows)
    hashes_after = {name: _module_hash(model) for name, model in models.items()}
    reference_row = fixed_rows[-1]
    convergence_fields = (
        "default_mass",
        "investment_mass",
        "entry_mass",
        "C_raw",
        "K_total",
        "G",
    )
    high_row = next(row for row in fixed_rows if row["resolution"] == "high")
    max_fixed_h_relative_error = max(
        abs(high_row[name] - reference_row[name]) / max(abs(reference_row[name]), 1.0)
        for name in convergence_fields
    )
    summary = {
        "fixed_h": roots["high"],
        "max_high_vs_reference_fixed_h_relative_error": max_fixed_h_relative_error,
        "fixed_h_convergence_tolerance": 5e-3,
        "FIXED_H_INTEGRATION_CONVERGENCE": max_fixed_h_relative_error <= 5e-3,
        "ROOT_REFERENCE_RESIDUAL_CHECK": bool(
            next(
                (
                    row.get("reference_root_pass", False)
                    for row in root_rows
                    if row["resolution"] == "high"
                    and row["root_status"] == "ROOT_FOUND"
                ),
                False,
            )
        ),
        "FIRM_PARAMETERS_FROZEN": hashes_before == hashes_after,
        "optimizer_steps": 0,
        "resolutions": RESOLUTIONS,
        "controller_summaries": {
            name: controller.summary() for name, controller in controllers.items()
        },
    }
    (output / "regression" / "integration_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["FIXED_H_INTEGRATION_CONVERGENCE"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
