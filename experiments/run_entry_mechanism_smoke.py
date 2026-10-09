"""Short frozen-checkpoint comparison for the entry-mechanism experiment.

This command runs simulation only. It never trains, updates, or saves model
parameters. The same global seed is restored before each arm so incumbent and
macro draws are common; value-cost entry itself uses its separate entry RNG.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.checkpoint_loader import load_analysis_checkpoint
from config import Config
from data.simulate_ts import SimulateTS


BASE_COMMIT = "4b236eaacf47b2ac7cb506508e54b21a199e89b0"


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
        "entry_cost_mean", "entry_cost_p10", "entry_cost_p50", "entry_cost_p90",
        "same_node_entrant_exit_count", "K_entry_same_node_exit",
        "deltaK_incumbent", "K_entry_endpoint", "K_exit_old",
        "C_oper", "C_raw", "resource_accounting_residual",
        "capital_accounting_residual", "tensor_capacity",
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


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    loaded = load_analysis_checkpoint(args.checkpoint, device=device)
    models = {
        "policy_value": loaded.models["policy_value"],
        "sdf_fc1": loaded.models.get("sdf_fc1"),
        "dist_b": loaded.models.get("dist_b"),
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
    }
    report: Dict[str, Any] = {
        "base_commit": BASE_COMMIT,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_metadata": loaded.metadata,
        "smoke_parameters": vars(args),
        "interpretation_warning": (
            "B versus C jointly changes entry screening, birth debt, and entrant scale; "
            "it does not isolate any one submechanism and is not a solved new equilibrium."
        ),
        "arms": {},
    }
    report["smoke_parameters"]["checkpoint"] = str(args.checkpoint)
    report["smoke_parameters"]["output_dir"] = str(args.output_dir)

    for label, (entry_mode, aggregation_mode) in arms.items():
        _reset_seed(args.seed, device)
        arm_dir = args.output_dir / label
        arm_dir.mkdir(parents=True, exist_ok=True)
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
            entry_mode=entry_mode,
            entry_capital_ratio=args.entry_capital_ratio,
            entry_size_ratio=args.entry_size_ratio,
            entry_cost_max=args.entry_cost_max,
            entry_dummy_i=args.entry_dummy_i,
            entry_inference_chunk_size=args.entry_inference_chunk_size,
            entry_rng_seed=args.entry_rng_seed,
            consumption_aggregation_mode=aggregation_mode,
            preserve_global_rng_around_entry=True,
            common_transition_seed=args.seed + 700_001,
        )
        try:
            firm, macro = simulator.simulate()
            firm.to_pickle(arm_dir / "firm.pkl")
            macro.to_pickle(arm_dir / "macro.pkl")
            macro.to_csv(arm_dir / "macro.csv", index=False)
            report["arms"][label] = {
                "status": "completed",
                "entry_mode": entry_mode,
                "consumption_aggregation_mode": aggregation_mode,
                "simulation_meta": simulator.last_simulation_meta,
                "summary": _summarize_macro(macro),
            }
        except RuntimeError as exc:
            report["arms"][label] = {
                "status": "infeasible_or_failed",
                "entry_mode": entry_mode,
                "consumption_aggregation_mode": aggregation_mode,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }

    hashes_after = {
        name: _state_hash(model)
        for name, model in models.items()
        if isinstance(model, torch.nn.Module)
    }
    if hashes_before != hashes_after:
        raise RuntimeError("frozen entry smoke modified checkpoint model state")
    report["model_hashes_before"] = hashes_before
    report["model_hashes_after"] = hashes_after
    report["model_hash_invariant"] = True
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
