#!/usr/bin/env python3
"""Solve the firm-side fixed point under the frozen Episode-1 macro law."""

from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.economic_config import AnalysisEconomicConfig  # noqa: E402
from analysis.inner_fixed_point_report import build_inner_fixed_point_report  # noqa: E402
from config import Config, HyperParams  # noqa: E402
from experiments.run_utils import build_optimizers, load_training_warmstart  # noqa: E402
from training.episode import Episode  # noqa: E402
from training.inner_fixed_point import (  # noqa: E402
    InnerFixedPointSolver,
    InnerFPSolverConfig,
    build_boundary_grid,
    build_child_bank,
    build_collocation_bank,
    write_json,
)


DEFAULT_SOURCE = Path(
    "/home/fit/zhuyingz/WORK/LiuHao/cachedir/"
    "normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_"
    "investment_compare_fresh100_20261014_144528"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Firm-side P0/PI/Q fixed point under a frozen Macro Law"
    )
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--fp-train-parents", type=int, default=8192)
    parser.add_argument("--fp-validation-parents", type=int, default=2048)
    parser.add_argument("--fp-child-shocks", type=int, default=16)
    parser.add_argument("--fp-batch-size", type=int, default=256)
    parser.add_argument("--fp-max-sweeps", type=int, default=30)
    parser.add_argument("--fp-min-sweeps", type=int, default=3)
    parser.add_argument("--fp-consecutive-passes", type=int, default=3)
    parser.add_argument("--fp-p-epochs", type=int, default=None)
    parser.add_argument("--fp-q-zero-epochs", type=int, default=None)
    parser.add_argument("--fp-q-default-epochs", type=int, default=None)
    parser.add_argument("--fp-q-survival-epochs", type=int, default=None)
    parser.add_argument("--fp-q-polish-epochs", type=int, default=None)
    parser.add_argument("--fp-damping", type=float, default=1.0)
    parser.add_argument("--fp-change-mean-tol", type=float, default=1e-3)
    parser.add_argument("--fp-change-p90-tol", type=float, default=5e-3)
    parser.add_argument("--fp-default-flip-tol", type=float, default=1e-3)
    parser.add_argument("--fp-bstar-p90-tol", type=float, default=1e-2)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--skip-report", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required source metadata is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _strict_source_payload(source: Path, checkpoint: Path) -> tuple[dict, dict, dict, dict]:
    meta = source / "metadata"
    hyperparams = _load_json(meta / "hyperparams.json")
    config = _load_json(meta / "config_snapshot.json")
    spec_file = _load_json(meta / "policy_value_model_spec.json")
    spec = spec_file.get("policy_value_model_spec", spec_file)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("episode", -1)) != 1:
        raise ValueError(f"Expected Episode 1 checkpoint, got {payload.get('episode')!r}")
    required = {"sdf_fc1", "policy_value", "fc2", "firm_target"}
    missing = sorted(required - set(payload.get("models", {})))
    if missing:
        raise ValueError(f"Combined checkpoint is missing model states: {missing}")
    checks = {
        "hyperparams": (payload.get("hyperparams"), hyperparams),
        "economic config": (payload.get("config_snapshot"), config),
        "policy_value model spec": (payload.get("policy_value_model_spec"), spec),
    }
    for label, (embedded, external) in checks.items():
        if _canonical(embedded) != _canonical(external):
            raise ValueError(f"Checkpoint/external {label} mismatch; refusing semantic drift")
    q_mode = str(spec.get("q_parameterization", "")).lower()
    if q_mode != "direct":
        raise ValueError(f"Expected inherited direct-Q semantics, got {q_mode!r}")
    value_meta = payload.get("value_parameterization")
    if not isinstance(value_meta, dict):
        raise ValueError("Checkpoint is missing value_parameterization metadata")
    for key, expected in (
        ("mode", hyperparams.get("pv_value_scale_mode")),
        ("log_max", hyperparams.get("pv_value_scale_log_max")),
        ("bellman_normalization", hyperparams.get("pv_bellman_normalize_by_value_scale")),
    ):
        if value_meta.get(key) != expected:
            raise ValueError(
                f"value_parameterization[{key}]={value_meta.get(key)!r} "
                f"does not match source hyperparams value {expected!r}"
            )
    if not payload.get("git_commit"):
        raise ValueError("Checkpoint is missing git_commit provenance")
    return payload, hyperparams, config, spec


def _source_data_path(source: Path, name: str) -> Path:
    expected = source / "data" / "outputs" / name
    if expected.is_file():
        return expected
    available = sorted(str(path.relative_to(source)) for path in source.rglob("ep1_stage_*.pkl"))
    raise FileNotFoundError(
        f"Required Episode-1 artifact is missing: {expected}. Found: {available}"
    )


def _restore_hyperparams(values: Dict[str, Any]) -> HyperParams:
    hp = HyperParams()
    for key, value in values.items():
        setattr(hp, key, value)
    return hp


def main() -> int:
    args = parse_args()
    source = args.source_run.expanduser().resolve()
    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else source / "checkpoints" / "ep1_combined.pt"
    )
    if not source.is_dir():
        raise FileNotFoundError(f"Source run does not exist: {source}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Combined checkpoint does not exist: {checkpoint}")
    payload, hp_values, config_values, spec = _strict_source_payload(source, checkpoint)
    economic_config = AnalysisEconomicConfig.from_dict(config_values)
    for name, value in economic_config.to_dict().items():
        setattr(Config, name, value)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    Config.DEVICE = device
    hp = _restore_hyperparams(hp_values)
    hp.bp_distill_epochs = 0
    hp.pv_bp_head_training_enabled = False
    overrides = {
        "q_zero_boundary_epochs": args.fp_q_zero_epochs,
        "q_default_pretrain_epochs": args.fp_q_default_epochs,
        "q_survival_aio_epochs": args.fp_q_survival_epochs,
        "q_mixed_polish_epochs": args.fp_q_polish_epochs,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(hp, name, int(value))
    p_epochs = args.fp_p_epochs
    if p_epochs is None:
        p_epochs = int(hp.pv_eval_epochs if hp.pv_eval_epochs is not None else hp.epochs)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.run_root.expanduser().resolve()
        if args.run_root is not None
        else (ROOT / "cachedir" / f"inner_fixed_point_from_fresh100_ep1_{stamp}").resolve()
    )
    if args.resume is None:
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        if not output_dir.is_dir():
            raise FileNotFoundError(
                f"--resume requires its existing --run-root directory: {output_dir}"
            )

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    models, firm_target, loader_provenance = load_training_warmstart(
        checkpoint,
        device,
        expected_episode=1,
        expected_q_parameterization="direct",
        expected_value_scale_mode=str(payload["value_parameterization"]["mode"]),
        expected_value_scale_log_max=float(payload["value_parameterization"]["log_max"]),
        expected_bellman_normalization=bool(payload["value_parameterization"]["bellman_normalization"]),
    )
    optimizers = build_optimizers(models, hp)
    episode = Episode(
        models=models,
        optimizers=optimizers,
        config=Config,
        hyperparams=hp,
        device=device,
        episode_id=1,
        firm_target=firm_target,
        q_checkpoint_loaded=True,
    )

    firm_path = _source_data_path(source, "ep1_stage_modeb.pkl")
    macro_path = _source_data_path(source, "ep1_stage_modeb_macro.pkl")
    firm_frame = pd.read_pickle(firm_path)
    macro_frame = pd.read_pickle(macro_path)
    train_bank = build_collocation_bank(
        firm_frame, macro_frame,
        n_parents=args.fp_train_parents,
        seed=args.seed + 101,
        split="train",
        economic_config=economic_config,
    )
    val_bank = build_collocation_bank(
        firm_frame, macro_frame,
        n_parents=args.fp_validation_parents,
        seed=args.seed + 202,
        split="validation",
        economic_config=economic_config,
    )
    models["sdf_fc1"].eval().requires_grad_(False)
    train_children = build_child_bank(
        models["sdf_fc1"], train_bank,
        n_child_shocks=args.fp_child_shocks,
        seed=args.seed + 303,
        economic_config=economic_config,
        device=device,
    )
    val_children = build_child_bank(
        models["sdf_fc1"], val_bank,
        n_child_shocks=args.fp_child_shocks,
        seed=args.seed + 404,
        economic_config=economic_config,
        device=device,
    )
    boundary_grid = build_boundary_grid(macro_frame, economic_config)

    provenance = {
        **loader_provenance,
        "experiment": "firm_side_inner_fixed_point_given_macro_law",
        "source_run": str(source),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_git_commit": payload["git_commit"],
        "source_checkpoint_sha256": _sha256(checkpoint),
        "source_firm_data": str(firm_path),
        "source_firm_data_sha256": _sha256(firm_path),
        "source_macro_data": str(macro_path),
        "source_macro_data_sha256": _sha256(macro_path),
        "q_parameterization": spec["q_parameterization"],
        "value_parameterization": payload["value_parameterization"],
        "economic_config": config_values,
        "outer_simulation": False,
        "entry": False,
        "bp_distillation": False,
        "macro_law_update": False,
    }
    write_json(output_dir / "metadata" / "provenance.json", provenance)
    write_json(output_dir / "metadata" / "source_hyperparams.json", hp_values)
    write_json(output_dir / "metadata" / "solver_cli.json", vars(args))

    solver_config = InnerFPSolverConfig(
        max_sweeps=args.fp_max_sweeps,
        min_sweeps=args.fp_min_sweeps,
        consecutive_passes=args.fp_consecutive_passes,
        batch_size=args.fp_batch_size,
        p_epochs=p_epochs,
        damping=args.fp_damping,
        change_mean_tol=args.fp_change_mean_tol,
        change_p90_tol=args.fp_change_p90_tol,
        default_flip_tol=args.fp_default_flip_tol,
        bstar_p90_tol=args.fp_bstar_p90_tol,
        seed=args.seed,
    )
    solver = InnerFixedPointSolver(
        episode=episode,
        train_bank=train_bank,
        validation_bank=val_bank,
        train_children=train_children,
        validation_children=val_children,
        boundary_grid=boundary_grid,
        output_dir=output_dir,
        config=solver_config,
        source_provenance=provenance,
    )
    start_sweep = solver.load_resume(args.resume) if args.resume is not None else 1
    summary = solver.run(start_sweep=start_sweep)
    if not args.skip_report:
        build_inner_fixed_point_report(output_dir)
    print(json.dumps(summary, indent=2, default=str))
    print(f"INNER_FP_RUN_ROOT={output_dir}")
    return 0 if summary.get("numerical_failure") is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
