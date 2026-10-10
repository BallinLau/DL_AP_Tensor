"""Short frozen-checkpoint comparison for the entry-mechanism experiment.

This command runs simulation only. It never trains, updates, or saves model
parameters. Initial states are matched by resetting the global seed; transition
innovations use canonical identity keys, and entry uses a separate RNG stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.checkpoint_loader import load_analysis_checkpoint
from config import Config
from data.entry import EntryEconomicInfeasibilityError
from data.simulate_ts import SimulateTS
from losses import P0Loss, PILoss
from training.bp_simulation_policy import GridBPSimulationPolicy


BASE_COMMIT = "4b236eaacf47b2ac7cb506508e54b21a199e89b0"
HARDENING_BASE_COMMIT = "5195ad058077f5fd45865c0ce8ed050be0f4fb6a"


def _state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_config(economic_config) -> type:
    values = economic_config.to_dict()
    values.update(
        DEVICE=Config.DEVICE,
        SIM_B_INIT_MIN=Config.SIM_B_INIT_MIN,
        SIM_B_INIT_MAX=Config.SIM_B_INIT_MAX,
        ENTRY_B_MIN=Config.ENTRY_B_MIN,
        ENTRY_B_MAX=Config.ENTRY_B_MAX,
    )
    return type("EntrySmokeEconomicConfig", (), values)


def _reset_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _finite_summary(series: pd.Series) -> Dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _summarize_macro(df: pd.DataFrame) -> Dict[str, Any]:
    # Child and promoted-parent rows are two views of one economic node. Keep
    # the first authoritative account so time-series totals are not doubled.
    nodes = (
        df.sort_values(["path", "economic_node_id", "accounting_stage"])
        .drop_duplicates(["path", "economic_node_id"], keep="first")
        .reset_index(drop=True)
    )
    fields = (
        "potential_count", "accepted_count", "acceptance_rate",
        "reference_K", "reference_N", "mean_K_ref", "K_per_entrant",
        "potential_capital_nominal", "candidate_capital_realized",
        "K_entry_gross", "K_entry_surviving", "I_entry",
        "entry_cutoff_mean", "entry_cutoff_p10", "entry_cutoff_p50", "entry_cutoff_p90",
        "entry_P0_mean", "entry_P0_p10", "entry_P0_p50", "entry_P0_p90",
        "entry_PI_mean", "entry_PI_p10", "entry_PI_p50", "entry_PI_p90",
        "entry_value_gap_mean", "entry_value_gap_p10",
        "entry_value_gap_p50", "entry_value_gap_p90",
        "entry_cost_mean", "entry_cost_p10", "entry_cost_p50", "entry_cost_p90",
        "same_node_entrant_exit_count", "K_entry_same_node_exit",
        "deltaK_incumbent", "K_entry_endpoint", "K_exit_old",
        "C_oper", "C_raw", "resource_accounting_residual",
        "I_entry_rebuilt", "entry_spend_residual", "C_rebuilt",
        "legacy_clamp_adjustment",
        "capital_accounting_residual", "tensor_capacity",
        "K_node_pre_exit", "K_endpoint_post_exit",
        "K_decomposition_start", "K_decomposition_end",
        "entry_capital_event_residual",
    )
    summary = {field: _finite_summary(nodes[field]) for field in fields}
    summary.update(
        economic_nodes=int(len(nodes)),
        resource_infeasible_count=int((nodes["resource_feasible"] < 0.5).sum()),
        max_abs_resource_accounting_residual=float(
            nodes["resource_accounting_residual"].abs().max()
        ),
        max_abs_capital_accounting_residual=float(
            nodes["capital_accounting_residual"].abs().max()
        ),
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-paths", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--branch-num", type=int, default=2)
    parser.add_argument("--entry-capital-ratio", type=float, default=0.10)
    parser.add_argument("--entry-size-ratio", type=float, default=0.10)
    parser.add_argument("--entry-cost-max", type=float, default=1.0)
    parser.add_argument("--entry-dummy-i", type=float, default=0.0)
    parser.add_argument("--entry-rng-seed", type=int, default=86420)
    parser.add_argument("--entry-inference-chunk-size", type=int, default=65536)
    parser.add_argument(
        "--simulation-bp-action-source",
        choices=["checkpoint", "head", "grid"],
        default="checkpoint",
        help=(
            "checkpoint uses the explicitly recorded source; head/grid are "
            "diagnostic overrides and are recorded in the report"
        ),
    )
    return parser.parse_args(argv)


def _jsonable_namespace(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def resolve_simulation_bp_source(loaded, requested_source: str) -> Dict[str, Any]:
    requested = str(requested_source).strip().lower()
    if requested not in {"checkpoint", "head", "grid"}:
        raise ValueError(f"unknown simulation BP source request: {requested!r}")
    recorded = set(loaded.metadata.get("hyperparameter_recorded_fields", []))
    source_recorded = "simulation_bp_action_source" in recorded
    checkpoint_source = (
        str(getattr(loaded.hyperparams, "simulation_bp_action_source", "")).lower()
        if source_recorded
        else None
    )
    if source_recorded and checkpoint_source not in {"head", "grid"}:
        raise ValueError(
            "checkpoint records an invalid simulation_bp_action_source: "
            f"{checkpoint_source!r}"
        )
    if requested == "checkpoint":
        if checkpoint_source is None:
            raise ValueError(
                "checkpoint does not record simulation_bp_action_source; "
                "pass --simulation-bp-action-source head or grid explicitly"
            )
        resolved = checkpoint_source
        provenance = "checkpoint_hyperparams"
    else:
        resolved = requested
        provenance = "explicit_diagnostic_override"

    head_status_recorded = "pv_bp_head_training_enabled" in recorded
    head_training_enabled = (
        bool(getattr(loaded.hyperparams, "pv_bp_head_training_enabled"))
        if head_status_recorded
        else None
    )
    warning = None
    if resolved == "head" and head_training_enabled is False:
        if requested == "checkpoint":
            raise ValueError(
                "checkpoint selects head simulation but records BP head training disabled; "
                "an explicit --simulation-bp-action-source head override is required"
            )
        warning = (
            "explicitly using BP head although checkpoint records "
            "pv_bp_head_training_enabled=False"
        )
        warnings.warn(warning, RuntimeWarning)
    return {
        "requested_source": requested,
        "checkpoint_source": checkpoint_source,
        "resolved_source": resolved,
        "resolution_provenance": provenance,
        "head_training_status_recorded": head_status_recorded,
        "head_training_enabled": head_training_enabled,
        "warning": warning,
    }


def _build_grid_policy(loaded, config, *, device: torch.device):
    target_name = "firm_target" if loaded.models.get("firm_target") is not None else "policy_value"
    target = loaded.models[target_name]
    economic = loaded.economic_config
    p0_loss = P0Loss(
        delta=economic.DELTA,
        tau=economic.TAU,
        kappa_b=economic.KAPPA_B,
        kappa_e=economic.KAPPA_E,
        aio_weight=economic.AIO_WEIGHT,
        alpha_z=economic.ALPHA_Z,
        beta_z=economic.BETA_Z,
        z0=economic.Z0,
    )
    pi_loss = PILoss(
        delta=economic.DELTA,
        tau=economic.TAU,
        g=economic.G,
        kappa_b=economic.KAPPA_B,
        kappa_e=economic.KAPPA_E,
        aio_weight=economic.AIO_WEIGHT,
        alpha_z=economic.ALPHA_Z,
        beta_z=economic.BETA_Z,
        z0=economic.Z0,
        b_penalty_weight=0.0,
    )
    policy = GridBPSimulationPolicy(
        target_model=target,
        sdf_fc1_model=loaded.models.get("sdf_fc1"),
        p0_loss=p0_loss,
        pi_loss=pi_loss,
        hyperparams=loaded.hyperparams,
        economic_config=config,
        n_child_shocks=int(loaded.hyperparams.simulation_bp_grid_n_child_shocks),
        shock_seed=int(loaded.hyperparams.simulation_bp_grid_shock_seed),
        require_cuda=True,
    )
    return policy, target_name


def _crn_pair_summary(reference, comparison) -> Dict[str, Any]:
    reference_firm, reference_macro = reference
    comparison_firm, comparison_macro = comparison
    result: Dict[str, Any] = {}
    macro_keys = ["path", "t", "branch", "economic_node_id"]
    macro = reference_macro[macro_keys + ["eps_x_transition"]].merge(
        comparison_macro[macro_keys + ["eps_x_transition"]],
        on=macro_keys,
        suffixes=("_reference", "_comparison"),
    )
    macro_valid = macro["eps_x_transition_reference"].notna() & macro[
        "eps_x_transition_comparison"
    ].notna()
    macro_diff = (
        macro.loc[macro_valid, "eps_x_transition_reference"]
        - macro.loc[macro_valid, "eps_x_transition_comparison"]
    ).abs()
    result["macro_common_samples"] = int(macro_valid.sum())
    result["eps_x_max_abs_diff"] = None if macro_diff.empty else float(macro_diff.max())

    firm_keys = ["path", "t", "branch", "ID"]
    shock_fields = ["eps_z_transition", "u_eta_transition", "u_i_transition"]
    left = reference_firm[reference_firm["initial_cohort"] > 0.5][firm_keys + shock_fields]
    right = comparison_firm[comparison_firm["initial_cohort"] > 0.5][firm_keys + shock_fields]
    merged = left.merge(right, on=firm_keys, suffixes=("_reference", "_comparison"))
    result["initial_cohort_common_rows"] = int(len(merged))
    for field in shock_fields:
        valid = merged[f"{field}_reference"].notna() & merged[f"{field}_comparison"].notna()
        diff = (
            merged.loc[valid, f"{field}_reference"]
            - merged.loc[valid, f"{field}_comparison"]
        ).abs()
        result[f"{field}_common_samples"] = int(valid.sum())
        result[f"{field}_max_abs_diff"] = None if diff.empty else float(diff.max())
    result["entrant_identity_matching"] = "not_claimed"
    return result


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    loaded = load_analysis_checkpoint(args.checkpoint, device=device)
    models = {
        "policy_value": loaded.models["policy_value"],
        "sdf_fc1": loaded.models.get("sdf_fc1"),
        "dist_b": loaded.models.get("dist_b"),
        "firm_target": loaded.models.get("firm_target"),
    }
    models = {name: model for name, model in models.items() if model is not None}
    for model in models.values():
        if isinstance(model, torch.nn.Module):
            model.eval()
            model.requires_grad_(False)
    hashes_before = {
        name: _state_hash(model)
        for name, model in models.items()
        if isinstance(model, torch.nn.Module)
    }
    config = _checkpoint_config(loaded.economic_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arms = {
        "A_legacy_entry_legacy_aggregation": ("legacy", "legacy_per_firm_clamp"),
        "B_legacy_entry_raw_aggregation": ("legacy", "raw"),
        "C_value_cost_entry_raw_aggregation": ("value_cost", "raw"),
        "D_investment_compare_entry_raw_aggregation": (
            "investment_compare", "raw"
        ),
    }
    report: Dict[str, Any] = {
        "historical_behavior_base_commit": BASE_COMMIT,
        "hardening_base_commit": HARDENING_BASE_COMMIT,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_metadata": loaded.metadata,
        "smoke_parameters": _jsonable_namespace(args),
        "bp_action_resolution": None,
        "comparison_design": {
            "node_accounting_mode": "economic_node_ledger",
            "transition_rng_mode": "stable_firm_identity",
            "initial_cohort_shock_matching": True,
            "entrant_shock_matching": False,
            "historical_4b236ea_reproduction_arm": False,
        },
        "interpretation_warning": (
            "B/C/D comparisons jointly change entry screening, birth state, and entrant "
            "scale; they do not isolate any one submechanism or represent a solved new "
            "equilibrium. investment_compare is the explicit PI>=P0 approximation."
        ),
        "arms": {},
    }
    outputs = {}
    pending_error: Optional[BaseException] = None
    try:
        try:
            bp_resolution = resolve_simulation_bp_source(
                loaded, args.simulation_bp_action_source
            )
            report["bp_action_resolution"] = bp_resolution
        except Exception as exc:
            report["preflight_error"] = {
                "stage": "resolve_simulation_bp_action_source",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            pending_error = exc

        if pending_error is not None:
            bp_resolution = None
        for label, (entry_mode, aggregation_mode) in arms.items():
            if pending_error is not None:
                break
            _reset_seed(args.seed, device)
            arm_dir = args.output_dir / label
            arm_dir.mkdir(parents=True, exist_ok=True)
            try:
                grid_policy = None
                target_name = None
                if bp_resolution["resolved_source"] == "grid":
                    grid_policy, target_name = _build_grid_policy(
                        loaded, config, device=device
                    )
                simulator = SimulateTS(
                    models=models,
                    config=config,
                    n_paths=args.n_paths,
                    group_size=args.group_size,
                    horizon=args.horizon,
                    branch_num=args.branch_num,
                    enable_entry=True,
                    enable_exit=True,
                    device=device,
                    bp_action_source=bp_resolution["resolved_source"],
                    bp_grid_policy=grid_policy,
                    entry_mode=entry_mode,
                    entry_spec_version=(
                        "investment_compare_v1"
                        if entry_mode == "investment_compare"
                        else "value_cost_v1"
                    ),
                    entry_capital_ratio=args.entry_capital_ratio,
                    entry_size_ratio=args.entry_size_ratio,
                    entry_cost_max=(
                        float(config.I_THRESHOLD)
                        if entry_mode == "investment_compare"
                        else args.entry_cost_max
                    ),
                    entry_dummy_i=args.entry_dummy_i,
                    entry_inference_chunk_size=args.entry_inference_chunk_size,
                    entry_rng_seed=args.entry_rng_seed,
                    consumption_aggregation_mode=aggregation_mode,
                    node_accounting_mode="economic_node_ledger",
                    preserve_global_rng_around_entry=True,
                    common_transition_seed=args.seed + 700_001,
                    transition_rng_mode="stable_firm_identity",
                )
                firm, macro = simulator.simulate()
            except EntryEconomicInfeasibilityError as exc:
                report["arms"][label] = {
                    "status": "economic_infeasible",
                    "failure_class": "economic_infeasibility",
                    "entry_mode": entry_mode,
                    "consumption_aggregation_mode": aggregation_mode,
                    "node_accounting_mode": "economic_node_ledger",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                continue
            except Exception as exc:
                report["arms"][label] = {
                    "status": "program_or_configuration_error",
                    "failure_class": "program_or_configuration_error",
                    "entry_mode": entry_mode,
                    "consumption_aggregation_mode": aggregation_mode,
                    "node_accounting_mode": "economic_node_ledger",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                pending_error = exc
                break
            firm.to_pickle(arm_dir / "firm.pkl")
            macro.to_pickle(arm_dir / "macro.pkl")
            macro.to_csv(arm_dir / "macro.csv", index=False)
            report["arms"][label] = {
                "status": "completed",
                "entry_mode": entry_mode,
                "consumption_aggregation_mode": aggregation_mode,
                "node_accounting_mode": "economic_node_ledger",
                "bp_action_source": bp_resolution["resolved_source"],
                "bp_grid_target_model": target_name,
                "simulation_meta": simulator.last_simulation_meta,
                "summary": _summarize_macro(macro),
            }
            outputs[label] = (firm, macro)
        if outputs:
            reference_label = next(iter(outputs))
            report["common_random_numbers"] = {
                label: _crn_pair_summary(outputs[reference_label], output)
                for label, output in outputs.items()
                if label != reference_label
            }
    finally:
        hashes_after = {
            name: _state_hash(model)
            for name, model in models.items()
            if isinstance(model, torch.nn.Module)
        }
        report["model_hashes_before"] = hashes_before
        report["model_hashes_after"] = hashes_after
        report["model_hash_invariant"] = hashes_before == hashes_after
        (args.output_dir / "summary.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
    if not report["model_hash_invariant"]:
        raise RuntimeError("frozen entry smoke modified checkpoint model state")
    if pending_error is not None:
        raise pending_error
    completed = sum(
        arm.get("status") == "completed" for arm in report["arms"].values()
    )
    if completed == 0:
        raise RuntimeError("all entry smoke arms failed; see summary.json")
    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    main()
