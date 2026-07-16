from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analysis.convergence_surface import evaluate_checkpoint_convergence_surfaces


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot offline Bellman fixed-point convergence surfaces.")
    p.add_argument("--checkpoint", action="append", default=[])
    p.add_argument("--policy-checkpoint")
    p.add_argument("--sdf-checkpoint")
    p.add_argument("--hyperparams-json")
    p.add_argument("--config-json")
    p.add_argument("--allow-default-hyperparams", action="store_true")
    p.add_argument("--allow-current-config", action="store_true")
    p.add_argument("--checkpoint-label", action="append", default=[])
    p.add_argument("--state-mode", choices=["fixed_slice", "fixed_grid", "reference_distribution"], required=True)
    p.add_argument("--reference-data")
    p.add_argument("--eta", type=float)
    p.add_argument("--i", type=float)
    p.add_argument("--x", type=float)
    p.add_argument("--hatcf", type=float)
    p.add_argument("--lnkf", type=float)
    p.add_argument("--hatc-cal", type=float)
    p.add_argument("--lnk-cal", type=float)
    p.add_argument("--b-min", type=float, default=0.0)
    p.add_argument("--b-max", type=float, default=1.0)
    p.add_argument("--b-points", type=int, default=80)
    p.add_argument("--z-min", type=float, default=-4.0)
    p.add_argument("--z-max", type=float, default=4.0)
    p.add_argument("--z-points", type=int, default=80)
    p.add_argument("--n-reference-states", type=int, default=128)
    p.add_argument("--n-child-shocks", type=int, default=64)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--parent-chunk-size", type=int, default=512)
    p.add_argument("--child-chunk-size", type=int, default=8192)
    p.add_argument("--max-child-state-evals", type=int)
    p.add_argument("--allow-large-run", action="store_true")
    p.add_argument("--include-raw", action="store_true")
    p.add_argument("--include-signed", action="store_true")
    p.add_argument("--residual-threshold", type=float)
    p.add_argument("--log-residual-scale", action="store_true")
    p.add_argument("--no-support-mask", action="store_true")
    p.add_argument("--support-radius", type=float)
    p.add_argument("--device")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = list(args.checkpoint)
    if not checkpoints and args.policy_checkpoint:
        # Loader accepts raw policy path as checkpoint when paired with sdf/hparams.
        checkpoints = [args.policy_checkpoint]
    if not checkpoints:
        raise SystemExit("At least one --checkpoint or --policy-checkpoint is required.")
    fixed_state = None
    if args.state_mode in {"fixed_slice", "fixed_grid"}:
        fixed_args = {
            "eta": args.eta,
            "i": args.i,
            "x": args.x,
            "hatcf": args.hatcf,
            "lnkf": args.lnkf,
            "hatc_cal": args.hatc_cal,
            "lnk_cal": args.lnk_cal,
        }
        missing = [key for key, value in fixed_args.items() if value is None]
        if missing:
            raise SystemExit(f"fixed_slice is missing required arguments: {missing}")
        fixed_state = {
            key: float(value) for key, value in fixed_args.items()
        }
    b_grid = np.linspace(args.b_min, args.b_max, args.b_points)
    z_grid = np.linspace(args.z_min, args.z_max, args.z_points)
    evaluate_checkpoint_convergence_surfaces(
        checkpoints,
        b_grid=b_grid,
        z_grid=z_grid,
        state_mode=args.state_mode,
        sdf_checkpoint=args.sdf_checkpoint,
        hyperparams_json=args.hyperparams_json,
        config_json=args.config_json,
        allow_default_hyperparams=args.allow_default_hyperparams,
        allow_current_config=args.allow_current_config,
        checkpoint_labels=args.checkpoint_label or None,
        fixed_state=fixed_state,
        reference_data=args.reference_data,
        n_reference_states=args.n_reference_states,
        n_child_shocks=args.n_child_shocks,
        seed=args.seed,
        parent_chunk_size=args.parent_chunk_size,
        child_chunk_size=args.child_chunk_size,
        max_child_state_evals=args.max_child_state_evals,
        allow_large_run=args.allow_large_run,
        include_raw_plots=args.include_raw,
        apply_support_mask=not args.no_support_mask,
        include_signed=args.include_signed,
        residual_threshold=args.residual_threshold,
        log_residual_scale=args.log_residual_scale,
        support_radius=args.support_radius,
        device=args.device,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
