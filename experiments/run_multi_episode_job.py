"""
CLI-friendly multi-episode runner for Slurm/sbatch.

This mirrors the notebook `run_multi_episode.ipynb` setup and the existing
`run_multi_episode.py`, but adds CLI switches for quick tests and explicit run
root selection so it can be invoked from a Slurm batch script.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
import sys
import json
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
print(f"Added {ROOT} to sys.path for imports")
from config import Config  # noqa: E402
from training.episode import Episode, NumericalStageFailure  # noqa: E402
from data.state_bank import (  # noqa: E402
    StateBank,
    summarize_stored_economic_feasibility,
)
from data.simulate_ts import SimulateTS  # noqa: E402
from experiments.run_utils import (  # noqa: E402
    resolve_base_dir,
    build_models,
    build_optimizers,
    build_hyperparams,
    load_training_warmstart,
    ensure_dirs,
    save_models,
    save_stage_df,
    plot_surfaces,
    plot_bari_istar_heatmap,
    plot_i_star_b,
    plot_distributions,
    plot_macro_series,
    plot_firm_b_window_distribution,
)
from utils.gpu_monitor import get_monitor, reset_monitor
from evaluation.episode_diagnostics import save_episode_diagnostics


def validate_sdf_fresh_pair_config(hyperparams):
    mode = str(getattr(hyperparams, "sdf_wealth_loss_mode", "legacy_abs_log1p")).lower()
    fresh = bool(getattr(hyperparams, "sdf_fresh_pair_enabled", False))
    if mode == "signed_aio" and not fresh:
        raise ValueError("signed_aio requires --sdf-fresh-pair-enabled for valid double sampling.")
    if mode != "signed_aio" and fresh:
        warnings.warn(
            "--sdf-fresh-pair-enabled with legacy SDF wealth loss is allowed, "
            "but the fresh pair bank is primarily designed for signed_aio.",
            RuntimeWarning,
            stacklevel=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged multi-episode training")
    parser.add_argument("--run-root", type=Path, default=None, help="Override output root; default cachedir/timestamp")
    parser.add_argument("--seed", type=int, default=12345, help="Global random seed for Python, NumPy, and Torch")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help=(
            "Combined checkpoint used to initialize all models. Optimizers and RNG "
            "state are recreated; simulation/data state is never restored."
        ),
    )
    parser.add_argument(
        "--initial-checkpoint-mode",
        choices=["continuation", "model_init"],
        default="continuation",
        help=(
            "How to interpret --initial-checkpoint: continuation starts at source "
            "episode + 1; model_init starts a fresh run at Episode 0 using only the "
            "checkpoint model weights."
        ),
    )
    parser.add_argument(
        "--initial-checkpoint-episode",
        type=int,
        default=None,
        help="Expected source episode recorded in --initial-checkpoint",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=None,
        help=(
            "Logical first episode to run. continuation requires source episode + 1; "
            "model_init and runs without a checkpoint require Episode 0."
        ),
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per stage")
    parser.add_argument("--n-paths", type=int, default=None, help="Override n_paths for data")
    parser.add_argument("--post0-n-paths", type=int, default=None, help="Override n_paths for episodes > 0; default keeps full n_paths")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    parser.add_argument(
        "--pv-batch-size",
        type=int,
        default=None,
        help="Policy/Value firm-level batch size. Falls back to --batch-size when omitted.",
    )
    parser.add_argument(
        "--sdf-fc1-batch-size",
        type=int,
        default=None,
        help=(
            "FC1, Episode-0 SDF bootstrap, SDF validation, and post-refresh "
            "SDF gate batch size. Falls back to --batch-size when omitted."
        ),
    )
    parser.add_argument("--simulate-group-size", type=int, default=None, help="Override firm count per simulated path for episodes > 0")
    parser.add_argument("--simulate-horizon", type=int, default=None, help="Override simulate horizon")
    parser.add_argument(
        "--entry-mode",
        choices=["legacy", "value_cost", "investment_compare"],
        default=None,
    )
    parser.add_argument("--entry-capital-ratio", type=float, default=None)
    parser.add_argument("--entry-size-ratio", type=float, default=None)
    parser.add_argument("--entry-cost-max", type=float, default=None)
    parser.add_argument("--entry-dummy-i", type=float, default=None)
    parser.add_argument("--entry-inference-chunk-size", type=int, default=None)
    parser.add_argument("--entry-rng-seed", type=int, default=None)
    parser.add_argument(
        "--consumption-aggregation-mode",
        choices=["legacy_per_firm_clamp", "raw"],
        default=None,
    )
    parser.add_argument("--consumption-floor", type=float, default=None)
    parser.add_argument(
        "--node-accounting-mode",
        choices=["auto", "legacy_recompute", "economic_node_ledger"],
        default=None,
    )
    parser.add_argument(
        "--entry-independent-resource-validation-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Run the diagnostic second-pass resource-account reconstruction. "
            "Disabling it keeps the authoritative ledger and feasibility guard."
        ),
    )
    parser.add_argument(
        "--entry-hatc-failure-scan-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "On an entry resource/accounting failure, scan a fixed node over "
            "trial Hatc values without replaying or retraining"
        ),
    )
    parser.add_argument(
        "--entry-hatc-failure-scan-points",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--entry-hatc-failure-scan-half-width",
        type=float,
        default=None,
        help="Symmetric log-consumption width around the current FC1 Hatc input",
    )
    parser.add_argument(
        "--entry-macro-progress-plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Generate cumulative entry/macro diagnostics after fixed episode "
            "milestones and after the final requested episode"
        ),
    )
    parser.add_argument(
        "--entry-macro-progress-every",
        type=int,
        default=3,
        help="Generate cumulative entry/macro plots after every N completed episodes",
    )
    parser.add_argument(
        "--entry-macro-growth-start-time",
        type=int,
        default=2,
        help="First simulation time included in non-entry capital growth-rate plots",
    )
    parser.add_argument(
        "--state-bank-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable State Bank v1 warm starts for Mode B Episode >= 2",
    )
    parser.add_argument("--state-bank-short-horizon", type=int, default=None)
    parser.add_argument("--state-bank-bootstrap-burn-in", type=int, default=None)
    parser.add_argument("--state-bank-bootstrap-capture-stride", type=int, default=None)
    parser.add_argument("--state-bank-refresh-burn-in", type=int, default=None)
    parser.add_argument("--state-bank-refresh-capture-stride", type=int, default=None)
    parser.add_argument("--state-bank-parent-transition-budget", type=int, default=None)
    parser.add_argument("--state-bank-parent-budget-warn-ratio", type=float, default=None)
    parser.add_argument("--state-bank-burn-in", type=int, default=None)
    parser.add_argument("--state-bank-capture-stride", type=int, default=None)
    parser.add_argument("--state-bank-max-entries", type=int, default=None)
    parser.add_argument(
        "--state-bank-sampling-mode",
        choices=["uniform_entry", "uniform_lineage"],
        default=None,
    )
    parser.add_argument(
        "--state-bank-acceptance-mode",
        choices=["numerical", "strict"],
        default=None,
    )
    parser.add_argument(
        "--state-bank-sdf-publication-gate-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Allow SDF continuation/progress/convergence diagnostics to block "
            "State Bank publication in strict acceptance mode"
        ),
    )
    parser.add_argument(
        "--state-bank-pq-decoupling-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use a State-Bank-specific natural/coverage mixture for P/Q only.",
    )
    parser.add_argument(
        "--state-bank-pq-coverage-ratio",
        type=float,
        default=None,
        help="Coverage parent-group share for decoupled State Bank P/Q training.",
    )
    parser.add_argument(
        "--state-bank-economic-feasibility-filter-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Persist only accounting-valid, resource-feasible State Bank "
            "snapshots with finite positive raw consumption and no floor"
        ),
    )
    parser.add_argument("--state-bank-min-unique-lineages", type=int, default=None)
    parser.add_argument("--state-bank-sample-seed", type=int, default=None)
    parser.add_argument("--state-bank-refresh-seed", type=int, default=None)
    parser.add_argument("--state-bank-reservoir-seed", type=int, default=None)
    parser.add_argument(
        "--transition-rng-mode",
        choices=["legacy_position", "stable_firm_identity"],
        default=None,
    )
    parser.add_argument("--common-transition-seed", type=int, default=None)
    parser.add_argument(
        "--entry-rng-mode",
        choices=["legacy_sequential", "stable_economic_identity"],
        default=None,
    )
    parser.add_argument(
        "--preserve-global-rng-around-entry",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cuda:0 or cpu")
    parser.add_argument("--quick-test", action="store_true", help="Shrink workload for smoke tests (n_paths=10, epochs=20, horizon=20)")
    parser.add_argument(
        "--enable-fc2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to run FC2 stage in each episode (default: disabled; use --enable-fc2 to enable)",
    )
    parser.add_argument(
        "--post0-mode",
        type=str.lower,
        default="modeb",
        choices=["modea", "modeb", "alternate"],
        help="Episode>0 mode: modea, modeb, or alternate between them (default: modeb)",
    )
    parser.add_argument(
        "--alternate-start",
        type=str.lower,
        default="modea",
        choices=["modea", "modeb"],
        help="When --post0-mode=alternate, choose which mode starts at episode 1",
    )
    parser.add_argument(
        "--ablation-mode",
        type=str.lower,
        default="baseline",
        choices=["baseline", "bellman_only", "fixed_sdf", "fixed_policy"],
        help="Policy/value ablation mode for Episode 2 instability diagnosis",
    )
    parser.add_argument("--max-firm-train-units", type=int, default=None, help="Cap firm parent transitions per stage")
    parser.add_argument("--pv-fixed-sdf-value", type=float, default=None, help="Fixed SDF value for fixed_sdf ablation")
    parser.add_argument("--pv-fixed-policy-mode", type=str.lower, default=None, choices=["parent_b", "zero", "one"], help="Fixed policy rule")
    parser.add_argument("--policy-grad-threshold", type=float, default=None, help="Absolute policy/value grad gate")
    parser.add_argument("--policy-rolling-grad-threshold", type=float, default=None, help="Rolling policy/value grad gate floor")
    parser.add_argument("--policy-loss-threshold", type=float, default=None, help="Absolute policy/value loss gate")
    parser.add_argument("--pv-sdf-clip-ratio-gate", type=float, default=None, help="Reject stage if raw SDF clip ratio exceeds this value")
    parser.add_argument(
        "--sdf-wealth-loss-mode",
        type=str.lower,
        default="signed_aio",
        choices=["legacy_abs_log1p", "signed_aio"],
        help="SDF wealth-equation objective; default signed_aio for formal B, legacy_abs_log1p for explicit ablations",
    )
    parser.add_argument(
        "--sdf-fresh-pair-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use refreshable fresh aggregate shock pairs for SDF wealth loss",
    )
    parser.add_argument("--sdf-child-bank-size", type=int, default=16, help="Fresh SDF shock bank size per parent")
    parser.add_argument(
        "--sdf-child-bank-refresh-epochs",
        type=int,
        default=1,
        help="Refresh the SDF shock bank every N epochs",
    )
    parser.add_argument("--sdf-child-bank-seed", type=int, default=12345, help="Base seed for SDF shock bank")
    parser.add_argument(
        "--episode0-sdf-epochs-per-round",
        type=int,
        default=0,
        help="Episode 0 SDF bootstrap epochs per gate round; <=0 uses --epochs",
    )
    parser.add_argument(
        "--episode0-sdf-max-rounds",
        type=int,
        default=10,
        help="Maximum Episode 0 SDF bootstrap gate rounds before stopping",
    )
    parser.add_argument(
        "--episode0-sdf-log-mean-error-max",
        type=float,
        default=0.25,
        help="Episode 0 bootstrap safety gate max |log E[M] - target|",
    )
    parser.add_argument(
        "--episode0-sdf-clip-low-ratio-max",
        type=float,
        default=0.20,
        help="Episode 0 bootstrap safety gate max share of raw M below 0.7",
    )
    parser.add_argument(
        "--episode0-sdf-finite-ratio-min",
        type=float,
        default=1.0,
        help="Episode 0 bootstrap safety gate minimum finite M ratio",
    )
    parser.add_argument(
        "--sdf-true-only-epochs",
        type=int,
        default=None,
        help="Override SDF_TRUE_ONLY epochs; use 5 for normalized residual smoke tests",
    )
    parser.add_argument(
        "--sdf-true-start-episode",
        type=int,
        default=None,
        help="First post-bootstrap episode that runs formal FC1/SDF_TRUE training",
    )
    parser.add_argument(
        "--sdf-recursive-only-epochs",
        type=int,
        default=None,
        help="Override SDF_RECURSIVE_ONLY epochs; use 0 to disable the recursive stage",
    )
    parser.add_argument("--sdf-stage2-lr", type=float, default=None, help="Override Stage2 SDF/FC1 learning rate")
    parser.add_argument("--sdf-true-only-lr", type=float, default=None, help="Override SDF_TRUE_ONLY SDF/value learning rate")
    parser.add_argument("--sdf-true-target-batches", type=int, default=None, help="Target number of SDF_TRUE parent-group batches")
    parser.add_argument("--sdf-min-parent-groups-per-batch", type=int, default=None, help="Minimum parent groups per SDF_TRUE batch")
    parser.add_argument("--sdf-true-moment-weight", type=float, default=None, help="Override SDF_TRUE moment weight")
    parser.add_argument("--sdf-true-anchor-weight", type=float, default=None, help="Override SDF_TRUE log-mean anchor weight")
    parser.add_argument(
        "--sdf-moment-constraint-mode",
        type=str.lower,
        default=None,
        choices=["legacy_penalty", "augmented_lagrangian"],
        help="SDF_TRUE_ONLY moment handling: legacy fixed penalties or pooled PHR AL",
    )
    parser.add_argument("--sdf-al-rho", type=float, default=None, help="Fixed PHR penalty parameter for SDF_TRUE_ONLY")
    parser.add_argument(
        "--sdf-al-primal-epochs-per-dual-update",
        type=int,
        default=None,
        help="Accepted SDF_TRUE primal epochs per PHR dual update",
    )
    parser.add_argument(
        "--sdf-al-gate-tolerance",
        type=float,
        default=None,
        help="Allowed normalized moment-constraint violation in the SDF_TRUE strict gate",
    )
    parser.add_argument("--sdf-continue-constraint-tol", type=float, default=None)
    parser.add_argument("--sdf-aio-progress-ratio-max", type=float, default=None)
    parser.add_argument("--sdf-aio-good-abs-tol", type=float, default=None)
    parser.add_argument("--sdf-final-constraint-tol", type=float, default=None)
    parser.add_argument("--sdf-final-max-signed-t-abs", type=float, default=None)
    parser.add_argument("--sdf-final-aio-mean-tol", type=float, default=None)
    parser.add_argument(
        "--sdf-al-dual-max-batches",
        type=int,
        default=None,
        help="Accepted-epoch train batches used by the dual estimator; <=0 uses the full split",
    )
    parser.add_argument(
        "--sdf-al-reset-on-true-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reset SDF PHR multipliers when SDF_TRUE_ONLY begins",
    )
    parser.add_argument(
        "--sdf-al-strict-semantics-guard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require signed AiO with normalized residuals for formal SDF_TRUE AL runs",
    )
    parser.add_argument(
        "--sdf-al-noise-tax-warn-ratio",
        type=float,
        default=None,
        help="Warn when the estimated AL constraint-noise tax exceeds this AiO-loss ratio",
    )
    parser.add_argument(
        "--sdf-reset-optimizer-on-true-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reset SDF/value Adam moments before SDF_TRUE_ONLY starts",
    )
    parser.add_argument(
        "--sdf-restore-best-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Restore the best SDF phase checkpoint after epochwise validation",
    )
    parser.add_argument(
        "--sdf-collapse-log-mean-error",
        type=float,
        default=None,
        help="Collapse threshold for |log E[M] - target| during SDF phase validation",
    )
    parser.add_argument(
        "--sdf-collapse-mean-ratio",
        type=float,
        default=None,
        help="Collapse threshold for E[M] relative to target mean during SDF phase validation",
    )
    parser.add_argument(
        "--fc1-jacobian-penalty-interval",
        type=int,
        default=None,
        help="Compute FC1 Jacobian penalty every N optimizer steps; <=0 disables it",
    )
    parser.add_argument(
        "--pv-bp-training-mode",
        type=str.lower,
        default=None,
        choices=["target_grid", "legacy_foc_kkt"],
        help="bp training mode: target_grid distillation or legacy FOC/KKT",
    )
    parser.add_argument(
        "--pv-bp-head-training-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Train BP policy heads; disable for a diagnostic-only grid reference run",
    )
    parser.add_argument(
        "--simulation-bp-action-source",
        type=str.lower,
        choices=["head", "grid"],
        default=None,
        help="Actual leverage action used by Episode>0 simulation",
    )
    parser.add_argument(
        "--simulation-bp-grid-n-child-shocks",
        type=int,
        default=None,
        help="Continuous child shocks used by the vectorized simulation grid policy",
    )
    parser.add_argument(
        "--simulation-bp-grid-shock-seed",
        type=int,
        default=None,
        help="Independent fixed shock-bank seed for simulation grid policy",
    )
    parser.add_argument("--bp-grid-coarse-size", type=int, default=None, help="Coarse bp-grid size for target_grid mode")
    parser.add_argument("--bp-grid-fine-size", type=int, default=None, help="Local fine bp-grid size for target_grid mode")
    parser.add_argument(
        "--bp-grid-refine-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable coarse-to-fine local bp-grid refinement",
    )
    parser.add_argument("--bp-grid-policy-weight", type=float, default=None, help="Weight on bp policy distillation loss")
    parser.add_argument("--bp-grid-mix-policy-weight", type=float, default=None, help="Weight on simulation-action mixed bp distillation loss")
    parser.add_argument(
        "--bp-grid-policy-loss-space",
        type=str.lower,
        default=None,
        choices=["output", "logit"],
        help="BP branch target-grid distillation space: output probability or pre-sigmoid logit",
    )
    parser.add_argument("--bp-grid-logit-target-eps", type=float, default=None, help="Clipping eps for logit-space BP targets")
    parser.add_argument("--bp-grid-logit-huber-delta", type=float, default=None, help="Huber delta for logit-space BP targets")
    parser.add_argument("--bp-grid-margin-scale", type=float, default=None, help="Top-two value margin scale for policy target confidence")
    parser.add_argument("--bp-grid-parent-chunk-size", type=int, default=None, help="Parent chunk size for bp-grid teacher")
    parser.add_argument("--bp-grid-candidate-chunk-size", type=int, default=None, help="Candidate bp chunk size for bp-grid teacher")
    parser.add_argument("--bp-grid-max-expanded-states", type=int, default=None, help="Max B*J states per bp-grid teacher forward chunk")
    parser.add_argument("--pv-target-grid-val-fraction", type=float, default=None, help="Tail batch fraction reserved for target-grid policy validation")
    parser.add_argument(
        "--pv-training-flow",
        type=str.lower,
        default=None,
        choices=["joint", "staged"],
        help="Policy/value training flow: legacy joint or staged P/Q evaluation plus BP distillation",
    )
    parser.add_argument(
        "--pv-value-scale-mode",
        type=str.lower,
        default=None,
        choices=["none", "exp_xz"],
        help="Equity value parameterization: physical values or exp(x+z)-scaled latent values",
    )
    parser.add_argument("--pv-value-scale-log-max", type=float, default=None, help="Clamp max for exp_xz value scale log component")
    parser.add_argument(
        "--pv-bellman-normalize-by-value-scale",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Train P0/PI Bellman value residuals divided by parent equity value scale",
    )
    parser.add_argument("--pv-eval-epochs", type=int, default=None, help="P/Q evaluation epochs for staged policy/value flow")
    parser.add_argument(
        "--q-parameterization",
        type=str.lower,
        default=None,
        choices=["direct", "b_times_unit", "hybrid_regime"],
        help="Q network parameterization; formal regime training supports direct and hybrid_regime",
    )
    parser.add_argument("--q-zero-boundary-epochs", type=int, default=None, help="Exact Q(b=0)=0 phase epochs")
    parser.add_argument("--q-default-pretrain-epochs", type=int, default=None, help="Default-parent recovery phase epochs")
    parser.add_argument("--q-survival-aio-epochs", type=int, default=None, help="Survival-parent Bellman/AiO phase epochs")
    parser.add_argument("--q-mixed-polish-epochs", type=int, default=None, help="Regime-exclusive mixed Q polishing epochs")
    parser.add_argument(
        "--q-target-refresh-mode",
        type=str.lower,
        default=None,
        choices=["phase", "stage"],
        help="Refresh Q continuation target per phase or once before survival",
    )
    parser.add_argument(
        "--q-bellman-normalize-by-target-scale",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Normalize Q Bellman residuals by a detached parent-common target scale",
    )
    parser.add_argument("--q-stage-lr", type=float, default=None, help="Q-only stage LR; omitted falls back to policy_lr")
    parser.add_argument(
        "--q-epoch-validation-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Validate learned-Q phases on one fixed natural-distribution bank",
    )
    parser.add_argument("--q-validation-max-batches", type=int, default=None)
    parser.add_argument("--q-validation-min-delta", type=float, default=None)
    parser.add_argument(
        "--q-restore-best-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--q-no-improvement-patience-episodes", type=int, default=None)
    parser.add_argument(
        "--save-episode-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--save-intermediate-stage-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--q-zero-sample-share", type=float, default=None, help="Q0 sample share during mixed polishing")
    parser.add_argument("--q-default-sample-share", type=float, default=None, help="QD sample share during mixed polishing")
    parser.add_argument("--q-survival-sample-share", type=float, default=None, help="QS sample share during mixed polishing")
    parser.add_argument("--q-default-phat-eps", type=float, default=None, help="Negative frozen-Phat margin for synthetic QD coverage")
    parser.add_argument("--q-zero-b-eps", type=float, default=None, help="Debt tolerance classified as the exact-zero Q regime")
    parser.add_argument("--q-default-candidate-multiplier", type=int, default=None, help="QD candidate oversampling multiplier before frozen-P filtering")
    parser.add_argument("--q-default-b-bins", type=int, default=None, help="Positive-debt support bins for QD coverage")
    parser.add_argument("--q-survival-ondist-share", type=float, default=None, help="On-distribution share in QS matched-transition batches")
    parser.add_argument(
        "--q-claim-coverage-enabled",
        type=_str2bool,
        default=None,
        help="Hybrid-only deterministic synthetic-b replay for live claim-Q pricing",
    )
    parser.add_argument(
        "--q-claim-coverage-b-bins",
        type=int,
        default=None,
        help="Number of leverage bins used by hybrid claim-Q coverage replay",
    )
    parser.add_argument(
        "--q-claim-coverage-start-episode",
        type=int,
        default=None,
        help="First episode that enables hybrid claim-Q coverage replay",
    )
    parser.add_argument(
        "--q-claim-coverage-low-b-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Add explicit positive low-debt anchors to hybrid claim-Q replay",
    )
    parser.add_argument(
        "--q-claim-coverage-low-b-anchors",
        type=_parse_float_csv,
        default=None,
        help="Comma-separated positive low-debt anchors for hybrid claim-Q replay",
    )
    parser.add_argument("--q-zero-loss-weight", type=float, default=None, help="Q0 objective weight")
    parser.add_argument("--q-default-loss-weight", type=float, default=None, help="QD recovery objective weight")
    parser.add_argument("--q-survival-loss-weight", type=float, default=None, help="QS Bellman/AiO objective weight")
    parser.add_argument("--q-nonnegative-weight", type=float, default=None, help="Direct-Q negative-output penalty weight")
    parser.add_argument(
        "--q-shape-weight-z", type=float, default=None,
        help="Q monotone-in-z shape prior weight (set 0 for the direct-Q baseline ablation)",
    )
    parser.add_argument(
        "--q-shape-weight-b-low", type=float, default=None,
        help="Q increasing-in-b at low b shape prior weight (set 0 for the ablation)",
    )
    parser.add_argument(
        "--q-shape-weight-b-high", type=float, default=None,
        help="Q decreasing-in-b at high b shape prior weight (set 0 for the ablation)",
    )
    parser.add_argument("--q-bootstrap-epochs", type=int, default=None, help="Episode-0 direct-Q cold-start bootstrap epochs")
    parser.add_argument(
        "--q-bootstrap-mode", type=str, default=None, choices=["constant_unit"],
        help="Episode-0 bootstrap target mode (constant_unit: b=0 -> 0, b>0 -> b * unit value)",
    )
    parser.add_argument("--q-bootstrap-unit-value", type=float, default=None, help="Bootstrap target unit value for positive debt")
    parser.add_argument("--q-bootstrap-nonnegative-weight", type=float, default=None, help="Bootstrap ReLU(-Q)^2 penalty weight")
    parser.add_argument("--q-structural-zero-tol", type=float, default=None, help="Hybrid Q0 structural identity tolerance")
    parser.add_argument("--q-structural-recovery-tol", type=float, default=None, help="Hybrid QD recovery identity tolerance")
    parser.add_argument("--q-boundary-match-weight", type=float, default=None, help="Optional hybrid survival/recovery boundary matching weight")
    parser.add_argument("--q-boundary-match-phat-eps", type=float, default=None, help="Frozen-Phat band used by optional hybrid boundary matching")
    parser.add_argument("--q-boundary-match-min-samples", type=int, default=None, help="Minimum boundary rows before applying optional hybrid boundary matching")
    parser.add_argument(
        "--q-require-zero-phase", type=_str2bool, default=None,
        help="Treat a missing Q0 optimizer step as a Q-stage failure",
    )
    parser.add_argument(
        "--q-require-default-phase", type=_str2bool, default=None,
        help="Treat missing QD default coverage as an explicit Q-stage failure",
    )
    parser.add_argument(
        "--q-require-survival-phase", type=_str2bool, default=None,
        help="Treat a missing QS Bellman optimizer step as a Q-stage failure",
    )
    parser.add_argument("--bp-distill-epochs", type=int, default=None, help="BP distillation epochs for staged policy/value flow")
    parser.add_argument(
        "--firm-inner-cycles",
        type=int,
        default=None,
        help="Staged P/Q Gauss-Seidel cycles per Episode with FC1/SDF frozen; 1 is the ep3 baseline",
    )
    parser.add_argument(
        "--p-solver",
        type=str,
        default=None,
        choices=["fixed_target_cache", "frozen_policy_dynamic_residual"],
        help=(
            "P stage solver: legacy fixed-target-cache Huber regression or the "
            "Hybrid frozen-policy dynamic Bellman residual policy iteration"
        ),
    )
    parser.add_argument(
        "--p-policy-iterations-per-inner",
        type=int,
        default=None,
        help="P policy iterations executed before the EP3 Q staged block of each Firm Inner",
    )
    parser.add_argument("--p-root-adam-steps", type=int, default=None, help="Adam steps per P policy iteration")
    parser.add_argument("--p-root-adam-lr", type=float, default=None, help="Adam LR per P policy iteration")
    parser.add_argument("--p-root-adam-clip", type=float, default=None, help="Adam grad-clip norm per P policy iteration")
    parser.add_argument("--p-root-eval-every", type=int, default=None, help="Validation interval in Adam steps")
    parser.add_argument("--p-root-lbfgs-max-iter", type=int, default=None, help="L-BFGS optimizer budget per P policy iteration")
    parser.add_argument("--p-root-lbfgs-chunk-iterations", type=int, default=None, help="L-BFGS chunk size in optimizer iterations")
    parser.add_argument("--p-root-lbfgs-history-size", type=int, default=None, help="L-BFGS history size")
    parser.add_argument("--p-root-target-residual", type=float, default=None, help="Optional early-stop normalized residual target")
    parser.add_argument(
        "--pv-bp-drift-mode",
        type=str.lower,
        default=None,
        choices=["full", "subset", "off"],
        help="Greedy BP drift observation density for P policy iterations",
    )
    parser.add_argument(
        "--p-bp-drift-subset-rows",
        type=int,
        default=None,
        help="Max re-solved train rows for greedy BP drift in subset mode",
    )
    parser.add_argument(
        "--pv-fresh-initial-source",
        type=str.lower,
        default=None,
        choices=["resolve", "reuse_build"],
        help="Source of the fresh-greedy initial residual rows",
    )
    parser.add_argument(
        "--q-solver",
        type=str,
        default=None,
        choices=["ep3_regime_staged"],
        help="Q stage solver; only the EP3 regime-staged Q training is permitted",
    )
    parser.add_argument(
        "--initial-hstar-enabled",
        dest="initial_hstar_enabled",
        action="store_true",
        default=None,
        help="Calibrate the t=0 macro belief with the production clearing root; later periods keep FC1 forecasts",
    )
    parser.add_argument(
        "--no-initial-hstar",
        dest="initial_hstar_enabled",
        action="store_false",
        default=None,
        help="Disable t=0 h-star calibration (ep3 baseline behavior)",
    )
    parser.add_argument(
        "--initial-hstar-residual-tolerance",
        type=float,
        default=None,
        help="Required absolute clearing residual for the t=0 h-star root",
    )
    parser.add_argument(
        "--initial-hstar-parity-tolerance",
        type=float,
        default=None,
        help="Required ledger parity tolerance between the root candidate and the production t=0 node",
    )
    parser.add_argument(
        "--initial-hstar-max-iterations",
        type=int,
        default=None,
        help="Maximum bisection iterations for the t=0 h-star root",
    )
    parser.add_argument(
        "--initial-hstar-coarse-points",
        type=int,
        default=None,
        help="STRICT_GLOBAL coarse support scan points",
    )
    parser.add_argument(
        "--initial-hstar-refinement-points",
        type=int,
        default=None,
        help="STRICT_GLOBAL refinement grid points per round",
    )
    parser.add_argument(
        "--initial-hstar-max-refinement-rounds",
        type=int,
        default=None,
        help="STRICT_GLOBAL refinement round budget",
    )
    parser.add_argument(
        "--initial-hstar-support-lower",
        type=float,
        default=None,
        help="STRICT_GLOBAL declared support lower bound",
    )
    parser.add_argument(
        "--initial-hstar-support-upper",
        type=float,
        default=None,
        help="STRICT_GLOBAL declared support upper bound",
    )
    parser.add_argument(
        "--initial-hstar-width-tolerance",
        type=float,
        default=None,
        help="STRICT_GLOBAL terminal bracket width",
    )
    parser.add_argument(
        "--initial-hstar-audit-output-dir",
        type=str,
        default=None,
        help="Optional per-path h-star audit JSON output directory",
    )
    parser.add_argument(
        "--initial-hstar-audit-episode-id",
        type=int,
        default=None,
        help="Episode id stamped into the h-star audit filename",
    )
    parser.add_argument(
        "--hstar-recalib-count",
        type=int,
        default=None,
        help="Number of periodic h-star recalibrations per simulation rollout "
        "(t=0 initial calibration excluded); resolves to every=max(1, horizon//(count+1)); "
        "0 = initial calibration only",
    )
    parser.add_argument(
        "--hstar-recalib-every",
        type=int,
        default=None,
        help="Engine-level periodic h-star recalibration cadence in time steps "
        "(0 = initial calibration only); ignored when --hstar-recalib-count > 0",
    )
    parser.add_argument(
        "--pv-eta1-b-augment-mode",
        type=str,
        default=None,
        choices=["none", "uniform", "b_uniform"],
        help="P-stage eta=1 b-distribution augmentation (train stream only); "
        "'uniform' draws the configured share of the eta=1 half from the "
        "global train pool with inverse-density weights over parent b. "
        "Requires pv-current-eta-balance-enabled",
    )
    parser.add_argument(
        "--pv-eta1-b-augment-share",
        type=float,
        default=None,
        help="Fraction of the eta=1 half drawn from the global b-uniform pool",
    )
    parser.add_argument(
        "--pv-eta1-b-augment-bins",
        type=int,
        default=None,
        help="Parent-b histogram bins for inverse-density sampling weights",
    )
    parser.add_argument(
        "--pv-eta1-b-augment-seed",
        type=int,
        default=None,
        help="Seed for the eta=1 b-uniform augmentation sampler",
    )
    parser.add_argument("--bp-distill-patience", type=int, default=None, help="BP distillation early-stop patience")
    parser.add_argument("--bp-distill-min-delta", type=float, default=None, help="BP distillation validation min delta")
    parser.add_argument(
        "--bp-distill-max-optimizer-steps",
        type=int,
        default=None,
        help="Maximum successful BP optimizer steps per episode; 0 keeps legacy epoch control",
    )
    parser.add_argument(
        "--bp-current-eta-resample-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resample only the staged BP train cache by current parent eta_t",
    )
    parser.add_argument(
        "--bp-current-eta1-train-share",
        type=float,
        default=None,
        help="Target current parent eta_t=1 share in the staged BP train cache",
    )
    parser.add_argument(
        "--bp-current-eta-resample-seed",
        type=int,
        default=None,
        help="Deterministic CPU RNG seed for current parent eta_t BP resampling",
    )
    parser.add_argument(
        "--pv-current-eta-balance-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Balance current parent eta_t only in staged P collocation batches",
    )
    parser.add_argument(
        "--pv-current-eta1-train-share",
        type=float,
        default=None,
        help="Target eta_t=1 share in staged P-only train/validation collocation",
    )
    parser.add_argument(
        "--pv-current-eta-balance-validation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply current-eta balance to staged P-only validation batches",
    )
    parser.add_argument(
        "--pv-current-eta-balance-seed",
        type=int,
        default=None,
        help="Deterministic CPU RNG seed for staged P-only eta balancing",
    )
    parser.add_argument(
        "--p-eta-natural-diag-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the independent Natural Validation diagnostic when a "
        "second validation caliber exists (diagnostics only)",
    )
    parser.add_argument(
        "--p-eta-natural-diag-max-parents",
        type=int,
        default=None,
        help="Max parent-group budget for the Natural Validation diagnostic",
    )
    parser.add_argument(
        "--p-eta-natural-diag-chunk-size",
        type=int,
        default=None,
        help="Parent chunk size for chunked Natural Validation diagnostics",
    )
    parser.add_argument(
        "--p-eta-natural-diag-frequency",
        type=str.lower,
        default=None,
        choices=["firm_inner_end", "per_p_iteration", "off"],
        help="Natural Validation diagnostic frequency",
    )
    parser.add_argument(
        "--p-eta-natural-diag-seed",
        type=int,
        default=None,
        help="Deterministic CPU RNG seed for Natural Validation subset selection",
    )
    parser.add_argument(
        "--pv-rollback-on-soft-spikes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow consecutive finite soft gradient spikes to rollback a PV epoch",
    )
    parser.add_argument(
        "--bp-distill-trainable-scope",
        type=str.lower,
        default=None,
        choices=["heads_only", "policy_path"],
        help="Trainable BP distillation parameter scope",
    )
    parser.add_argument(
        "--pv-mixture-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable fixed-total SimulateTS/Sample mixture for episode>0 Policy/Value training.",
    )
    parser.add_argument("--pv-mixture-ratio", type=float, default=None, help="Coverage Sample parent-group share in fixed-total PV mixture")
    parser.add_argument("--pv-mixture-start-episode", type=int, default=None, help="First episode id eligible for PV mixture")
    parser.add_argument(
        "--pv-mixture-budget-mode",
        type=str.lower,
        default=None,
        choices=["fixed_total"],
        help="PV mixture budget mode; current formal implementation uses fixed_total.",
    )
    parser.add_argument(
        "--pv-mixture-sampling-mode",
        type=str.lower,
        default=None,
        choices=["uniform", "feasible", "realbz"],
        help="Coverage Sample sampling mode for PV mixture.",
    )
    parser.add_argument("--pv-mixture-coverage-group-size", type=int, default=None, help="Coverage Sample firm group size")
    parser.add_argument("--pv-mixture-seed", type=int, default=None, help="Base seed for deterministic PV mixture coverage data")
    parser.add_argument(
        "--pv-mixture-stratified-validation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Split SimulateTS and coverage parent groups separately into train/validation.",
    )
    parser.add_argument(
        "--pv-mixture-preserve-rng",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Restore Python/NumPy/Torch/CUDA RNG state after building coverage Sample data.",
    )
    parser.add_argument(
        "--pv-eta-resample-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable child-ETA oversampling only for the staged BP distillation train cache.",
    )
    parser.add_argument(
        "--pv-exact-eta-integration-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Exactly marginalize future Bernoulli ETA in Bellman and BP teacher expectations.",
    )
    parser.add_argument(
        "--bp-grid-confidence-relative",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Scale policy confidence by coarse top-two margin relative to |V*|",
    )
    parser.add_argument(
        "--firm-target-update",
        type=str.lower,
        default=None,
        choices=["soft", "hard", "epoch_hard", "epoch_soft", "stage_hard", "none"],
        help="Firm target update schedule",
    )
    parser.add_argument(
        "--modeb-resimulate-after-pv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Treatment B: resimulate Mode B data after Policy/Value training and before SDF/FC1 Stage 2",
    )
    return parser.parse_args()


def _str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _parse_float_csv(value):
    if isinstance(value, (tuple, list)):
        return tuple(float(item) for item in value)
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated floats, got {value!r}"
        ) from exc


def configure_hyperparams(args: argparse.Namespace):
    hyperparams = build_hyperparams()
    if args.quick_test:
        hyperparams.n_paths = 10
        hyperparams.epochs = 20
        hyperparams.simulate_horizon = 15
        hyperparams.batch_size = min(hyperparams.batch_size, 1024)
    if args.n_paths is not None:
        hyperparams.n_paths = args.n_paths
    if args.batch_size is not None:
        hyperparams.batch_size = int(args.batch_size)
    fallback_batch_size = int(hyperparams.batch_size)
    hyperparams.pv_batch_size = int(
        args.pv_batch_size
        if args.pv_batch_size is not None
        else fallback_batch_size
    )
    hyperparams.sdf_fc1_batch_size = int(
        args.sdf_fc1_batch_size
        if args.sdf_fc1_batch_size is not None
        else fallback_batch_size
    )
    if hyperparams.pv_batch_size <= 0:
        raise ValueError(
            "pv_batch_size must be positive, "
            f"got {hyperparams.pv_batch_size}."
        )
    if hyperparams.sdf_fc1_batch_size <= 0:
        raise ValueError(
            "sdf_fc1_batch_size must be positive, "
            f"got {hyperparams.sdf_fc1_batch_size}."
        )
    if args.epochs is not None:
        hyperparams.epochs = args.epochs
    if args.simulate_horizon is not None:
        hyperparams.simulate_horizon = args.simulate_horizon
    for arg_name, hp_name, cast in (
        ("entry_mode", "entry_mode", str),
        ("entry_capital_ratio", "entry_capital_ratio", float),
        ("entry_size_ratio", "entry_size_ratio", float),
        ("entry_cost_max", "entry_cost_max", float),
        ("entry_dummy_i", "entry_dummy_i", float),
        ("entry_inference_chunk_size", "entry_inference_chunk_size", int),
        ("entry_rng_seed", "entry_rng_seed", int),
        ("consumption_aggregation_mode", "consumption_aggregation_mode", str),
        ("consumption_floor", "consumption_floor", float),
        ("node_accounting_mode", "node_accounting_mode", str),
        (
            "entry_independent_resource_validation_enabled",
            "entry_independent_resource_validation_enabled",
            bool,
        ),
        (
            "entry_hatc_failure_scan_enabled",
            "entry_hatc_failure_scan_enabled",
            bool,
        ),
        (
            "entry_hatc_failure_scan_points",
            "entry_hatc_failure_scan_points",
            int,
        ),
        (
            "entry_hatc_failure_scan_half_width",
            "entry_hatc_failure_scan_half_width",
            float,
        ),
        ("state_bank_enabled", "state_bank_enabled", bool),
        ("state_bank_short_horizon", "state_bank_short_horizon", int),
        ("state_bank_bootstrap_burn_in", "state_bank_bootstrap_burn_in", int),
        (
            "state_bank_bootstrap_capture_stride",
            "state_bank_bootstrap_capture_stride",
            int,
        ),
        ("state_bank_refresh_burn_in", "state_bank_refresh_burn_in", int),
        (
            "state_bank_refresh_capture_stride",
            "state_bank_refresh_capture_stride",
            int,
        ),
        (
            "state_bank_parent_transition_budget",
            "state_bank_parent_transition_budget",
            int,
        ),
        (
            "state_bank_parent_budget_warn_ratio",
            "state_bank_parent_budget_warn_ratio",
            float,
        ),
        ("state_bank_burn_in", "state_bank_burn_in", int),
        ("state_bank_capture_stride", "state_bank_capture_stride", int),
        ("state_bank_max_entries", "state_bank_max_entries", int),
        ("state_bank_sampling_mode", "state_bank_sampling_mode", str),
        ("state_bank_acceptance_mode", "state_bank_acceptance_mode", str),
        (
            "state_bank_sdf_publication_gate_enabled",
            "state_bank_sdf_publication_gate_enabled",
            bool,
        ),
        (
            "state_bank_pq_decoupling_enabled",
            "state_bank_pq_decoupling_enabled",
            bool,
        ),
        (
            "state_bank_pq_coverage_ratio",
            "state_bank_pq_coverage_ratio",
            float,
        ),
        (
            "state_bank_economic_feasibility_filter_enabled",
            "state_bank_economic_feasibility_filter_enabled",
            bool,
        ),
        (
            "state_bank_min_unique_lineages",
            "state_bank_min_unique_lineages",
            int,
        ),
        ("state_bank_sample_seed", "state_bank_sample_seed", int),
        ("state_bank_refresh_seed", "state_bank_refresh_seed", int),
        ("state_bank_reservoir_seed", "state_bank_reservoir_seed", int),
        ("transition_rng_mode", "transition_rng_mode", str),
        ("common_transition_seed", "common_transition_seed", int),
        ("entry_rng_mode", "entry_rng_mode", str),
        (
            "preserve_global_rng_around_entry",
            "preserve_global_rng_around_entry",
            bool,
        ),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            setattr(hyperparams, hp_name, cast(value))
    if args.state_bank_burn_in is not None:
        if args.state_bank_bootstrap_burn_in is None:
            hyperparams.state_bank_bootstrap_burn_in = int(args.state_bank_burn_in)
        if args.state_bank_refresh_burn_in is None:
            hyperparams.state_bank_refresh_burn_in = int(args.state_bank_burn_in)
    if args.state_bank_capture_stride is not None:
        if args.state_bank_bootstrap_capture_stride is None:
            hyperparams.state_bank_bootstrap_capture_stride = int(
                args.state_bank_capture_stride
            )
        if args.state_bank_refresh_capture_stride is None:
            hyperparams.state_bank_refresh_capture_stride = int(
                args.state_bank_capture_stride
            )
    if hyperparams.entry_mode == "investment_compare":
        from data.entry import INVESTMENT_COMPARE_SPEC_VERSION

        resolved_i_max = float(Config.I_THRESHOLD)
        if (
            args.entry_cost_max is not None
            and not np.isclose(
                float(args.entry_cost_max), resolved_i_max, rtol=0.0, atol=1e-12
            )
        ):
            raise ValueError(
                "--entry-cost-max cannot differ from ordinary investment-cost "
                f"support in investment_compare mode: requested={args.entry_cost_max}, "
                f"I_THRESHOLD={resolved_i_max}"
            )
        hyperparams.entry_spec_version = INVESTMENT_COMPARE_SPEC_VERSION
        hyperparams.entry_cost_max = resolved_i_max
    from data.entry import resource_accounting_metadata

    hyperparams.resource_accounting_version = resource_accounting_metadata(
        hyperparams.entry_mode
    )["resource_accounting_version"]
    if (
        hyperparams.entry_mode in {"value_cost", "investment_compare"}
        and hyperparams.consumption_aggregation_mode != "raw"
    ):
        raise ValueError(
            f"entry_mode={hyperparams.entry_mode!r} requires "
            "consumption_aggregation_mode='raw'"
        )
    if hyperparams.entry_capital_ratio < 0.0:
        raise ValueError("entry_capital_ratio must be nonnegative")
    if not 0.0 < hyperparams.entry_size_ratio <= 1.0:
        raise ValueError("entry_size_ratio must be in (0, 1]")
    if hyperparams.entry_cost_max <= 0.0:
        raise ValueError("entry_cost_max must be positive")
    if hyperparams.entry_inference_chunk_size <= 0:
        raise ValueError("entry_inference_chunk_size must be positive")
    if not np.isfinite(hyperparams.consumption_floor) or hyperparams.consumption_floor < 0.0:
        raise ValueError("consumption_floor must be finite and nonnegative")
    if str(hyperparams.node_accounting_mode).lower() not in {
        "auto", "legacy_recompute", "economic_node_ledger"
    }:
        raise ValueError("invalid node_accounting_mode")
    if int(hyperparams.entry_hatc_failure_scan_points) < 3:
        raise ValueError("entry_hatc_failure_scan_points must be at least 3")
    if int(hyperparams.entry_hatc_failure_scan_points) % 2 == 0:
        raise ValueError("entry_hatc_failure_scan_points must be odd")
    if (
        not np.isfinite(hyperparams.entry_hatc_failure_scan_half_width)
        or float(hyperparams.entry_hatc_failure_scan_half_width) <= 0.0
    ):
        raise ValueError(
            "entry_hatc_failure_scan_half_width must be finite and positive"
        )
    if (
        hyperparams.entry_mode in {"value_cost", "investment_compare"}
        and str(hyperparams.node_accounting_mode).lower() == "legacy_recompute"
    ):
        raise ValueError(
            f"{hyperparams.entry_mode} entry cannot use legacy_recompute accounting"
        )
    if int(hyperparams.state_bank_short_horizon) <= 0:
        raise ValueError("state_bank_short_horizon must be positive")
    if int(hyperparams.state_bank_bootstrap_burn_in) < 1:
        raise ValueError("state_bank_bootstrap_burn_in must be at least 1")
    if int(hyperparams.state_bank_refresh_burn_in) < 1:
        raise ValueError("state_bank_refresh_burn_in must be at least 1")
    if int(hyperparams.state_bank_bootstrap_capture_stride) <= 0:
        raise ValueError("state_bank_bootstrap_capture_stride must be positive")
    if int(hyperparams.state_bank_refresh_capture_stride) <= 0:
        raise ValueError("state_bank_refresh_capture_stride must be positive")
    if int(hyperparams.state_bank_parent_transition_budget) < 0:
        raise ValueError("state_bank_parent_transition_budget must be non-negative")
    if not 0.0 < float(hyperparams.state_bank_parent_budget_warn_ratio) <= 1.0:
        raise ValueError(
            "state_bank_parent_budget_warn_ratio must be in (0, 1]"
        )
    if int(hyperparams.state_bank_max_entries) < 0:
        raise ValueError("state_bank_max_entries must be non-negative")
    if int(hyperparams.state_bank_refresh_burn_in) >= int(hyperparams.state_bank_short_horizon):
        raise ValueError(
            "state_bank_refresh_burn_in must be smaller than state_bank_short_horizon"
        )
    if (
        bool(hyperparams.state_bank_enabled)
        and int(hyperparams.state_bank_bootstrap_burn_in) >= int(hyperparams.simulate_horizon)
    ):
        raise ValueError(
            "state_bank_bootstrap_burn_in must be smaller than simulate_horizon so "
            "Episode 1 can produce Bank_1"
        )
    if str(hyperparams.state_bank_sampling_mode) not in {
        "uniform_entry",
        "uniform_lineage",
    }:
        raise ValueError("invalid state_bank_sampling_mode")
    if str(hyperparams.state_bank_acceptance_mode) not in {"numerical", "strict"}:
        raise ValueError("invalid state_bank_acceptance_mode")
    if int(hyperparams.state_bank_min_unique_lineages) < 0:
        raise ValueError("state_bank_min_unique_lineages must be non-negative")
    bellman_mean_threshold = getattr(
        hyperparams,
        "bellman_conditional_mean_thresh",
        None,
    )
    bellman_p90_threshold = getattr(
        hyperparams,
        "bellman_conditional_p90_thresh",
        None,
    )
    if (bellman_mean_threshold is None) != (bellman_p90_threshold is None):
        raise ValueError(
            "Bellman conditional thresholds must be both configured or both None"
        )
    if (
        str(hyperparams.transition_rng_mode) == "stable_firm_identity"
        and hyperparams.common_transition_seed is None
    ):
        raise ValueError(
            "stable_firm_identity transition RNG requires common_transition_seed"
        )
    hyperparams.ablation_mode = args.ablation_mode
    hyperparams.sdf_wealth_loss_mode = args.sdf_wealth_loss_mode
    hyperparams.sdf_fresh_pair_enabled = bool(args.sdf_fresh_pair_enabled)
    hyperparams.sdf_child_bank_size = args.sdf_child_bank_size
    hyperparams.sdf_child_bank_refresh_epochs = args.sdf_child_bank_refresh_epochs
    hyperparams.sdf_child_bank_seed = args.sdf_child_bank_seed
    hyperparams.episode0_sdf_epochs_per_round = int(args.episode0_sdf_epochs_per_round)
    hyperparams.episode0_sdf_max_rounds = int(args.episode0_sdf_max_rounds)
    hyperparams.episode0_sdf_log_mean_error_max = float(args.episode0_sdf_log_mean_error_max)
    hyperparams.episode0_sdf_clip_low_ratio_max = float(args.episode0_sdf_clip_low_ratio_max)
    hyperparams.episode0_sdf_finite_ratio_min = float(args.episode0_sdf_finite_ratio_min)
    if args.sdf_true_only_epochs is not None:
        hyperparams.sdf_true_only_epochs = int(args.sdf_true_only_epochs)
    if args.sdf_true_start_episode is not None:
        hyperparams.sdf_true_start_episode = int(args.sdf_true_start_episode)
    if args.sdf_recursive_only_epochs is not None:
        hyperparams.sdf_recursive_only_epochs = int(args.sdf_recursive_only_epochs)
    if args.sdf_stage2_lr is not None:
        hyperparams.sdf_stage2_lr = float(args.sdf_stage2_lr)
    if args.sdf_true_only_lr is not None:
        hyperparams.sdf_true_only_lr = float(args.sdf_true_only_lr)
    if args.sdf_true_target_batches is not None:
        hyperparams.sdf_true_target_batches = int(args.sdf_true_target_batches)
    if args.sdf_min_parent_groups_per_batch is not None:
        hyperparams.sdf_min_parent_groups_per_batch = int(args.sdf_min_parent_groups_per_batch)
    if args.sdf_true_moment_weight is not None:
        hyperparams.sdf_true_moment_weight = float(args.sdf_true_moment_weight)
    if args.sdf_true_anchor_weight is not None:
        hyperparams.sdf_true_anchor_weight = float(args.sdf_true_anchor_weight)
    if args.sdf_moment_constraint_mode is not None:
        hyperparams.sdf_moment_constraint_mode = str(args.sdf_moment_constraint_mode)
    if args.sdf_al_rho is not None:
        hyperparams.sdf_al_rho = float(args.sdf_al_rho)
    if args.sdf_al_primal_epochs_per_dual_update is not None:
        hyperparams.sdf_al_primal_epochs_per_dual_update = int(
            args.sdf_al_primal_epochs_per_dual_update
        )
    if args.sdf_al_gate_tolerance is not None:
        hyperparams.sdf_al_gate_tolerance = float(args.sdf_al_gate_tolerance)
    if args.sdf_continue_constraint_tol is not None:
        hyperparams.sdf_continue_constraint_tol = float(args.sdf_continue_constraint_tol)
    if args.sdf_aio_progress_ratio_max is not None:
        hyperparams.sdf_aio_progress_ratio_max = float(args.sdf_aio_progress_ratio_max)
    if args.sdf_aio_good_abs_tol is not None:
        hyperparams.sdf_aio_good_abs_tol = float(args.sdf_aio_good_abs_tol)
    if args.sdf_final_constraint_tol is not None:
        hyperparams.sdf_final_constraint_tol = float(args.sdf_final_constraint_tol)
    if args.sdf_final_max_signed_t_abs is not None:
        hyperparams.sdf_final_max_signed_t_abs = float(args.sdf_final_max_signed_t_abs)
    if args.sdf_final_aio_mean_tol is not None:
        hyperparams.sdf_final_aio_mean_tol = float(args.sdf_final_aio_mean_tol)
    if args.sdf_al_dual_max_batches is not None:
        hyperparams.sdf_al_dual_max_batches = int(args.sdf_al_dual_max_batches)
    if args.sdf_al_reset_on_true_start is not None:
        hyperparams.sdf_al_reset_on_true_start = bool(args.sdf_al_reset_on_true_start)
    if args.sdf_al_strict_semantics_guard is not None:
        hyperparams.sdf_al_strict_semantics_guard = bool(
            args.sdf_al_strict_semantics_guard
        )
    if args.sdf_al_noise_tax_warn_ratio is not None:
        hyperparams.sdf_al_noise_tax_warn_ratio = float(
            args.sdf_al_noise_tax_warn_ratio
        )
    if args.sdf_reset_optimizer_on_true_start is not None:
        hyperparams.sdf_reset_optimizer_on_true_start = bool(args.sdf_reset_optimizer_on_true_start)
    if args.sdf_restore_best_checkpoint is not None:
        hyperparams.sdf_restore_best_checkpoint = bool(args.sdf_restore_best_checkpoint)
    if args.sdf_collapse_log_mean_error is not None:
        hyperparams.sdf_collapse_log_mean_error = float(args.sdf_collapse_log_mean_error)
    if args.sdf_collapse_mean_ratio is not None:
        hyperparams.sdf_collapse_mean_ratio = float(args.sdf_collapse_mean_ratio)
    if args.fc1_jacobian_penalty_interval is not None:
        hyperparams.fc1_jacobian_penalty_interval = args.fc1_jacobian_penalty_interval
    if args.pv_bp_training_mode is not None:
        hyperparams.pv_bp_training_mode = args.pv_bp_training_mode
    if args.pv_bp_head_training_enabled is not None:
        hyperparams.pv_bp_head_training_enabled = bool(
            args.pv_bp_head_training_enabled
        )
    if args.simulation_bp_action_source is not None:
        hyperparams.simulation_bp_action_source = args.simulation_bp_action_source
    if args.simulation_bp_grid_n_child_shocks is not None:
        hyperparams.simulation_bp_grid_n_child_shocks = int(
            args.simulation_bp_grid_n_child_shocks
        )
    if args.simulation_bp_grid_shock_seed is not None:
        hyperparams.simulation_bp_grid_shock_seed = int(
            args.simulation_bp_grid_shock_seed
        )
    if args.bp_grid_coarse_size is not None:
        hyperparams.bp_grid_coarse_size = args.bp_grid_coarse_size
    if args.bp_grid_fine_size is not None:
        hyperparams.bp_grid_fine_size = args.bp_grid_fine_size
    if args.bp_grid_refine_enabled is not None:
        hyperparams.bp_grid_refine_enabled = bool(args.bp_grid_refine_enabled)
    if args.bp_grid_policy_weight is not None:
        hyperparams.bp_grid_policy_weight = args.bp_grid_policy_weight
    if args.bp_grid_mix_policy_weight is not None:
        hyperparams.bp_grid_mix_policy_weight = args.bp_grid_mix_policy_weight
    if args.bp_grid_policy_loss_space is not None:
        hyperparams.bp_grid_policy_loss_space = args.bp_grid_policy_loss_space
    if args.bp_grid_logit_target_eps is not None:
        hyperparams.bp_grid_logit_target_eps = float(args.bp_grid_logit_target_eps)
    if args.bp_grid_logit_huber_delta is not None:
        hyperparams.bp_grid_logit_huber_delta = float(args.bp_grid_logit_huber_delta)
    if args.bp_grid_margin_scale is not None:
        hyperparams.bp_grid_margin_scale = args.bp_grid_margin_scale
    if args.bp_grid_parent_chunk_size is not None:
        hyperparams.bp_grid_parent_chunk_size = args.bp_grid_parent_chunk_size
    if args.bp_grid_candidate_chunk_size is not None:
        hyperparams.bp_grid_candidate_chunk_size = args.bp_grid_candidate_chunk_size
    if args.bp_grid_max_expanded_states is not None:
        hyperparams.bp_grid_max_expanded_states = args.bp_grid_max_expanded_states
    if str(hyperparams.simulation_bp_action_source).lower() not in {"head", "grid"}:
        raise ValueError("simulation_bp_action_source must be 'head' or 'grid'")
    if int(hyperparams.simulation_bp_grid_n_child_shocks) < 1:
        raise ValueError("simulation_bp_grid_n_child_shocks must be positive")
    if args.bp_grid_confidence_relative is not None:
        hyperparams.bp_grid_confidence_relative = bool(args.bp_grid_confidence_relative)
    if args.pv_target_grid_val_fraction is not None:
        hyperparams.pv_target_grid_val_fraction = args.pv_target_grid_val_fraction
    if args.pv_training_flow is not None:
        hyperparams.pv_training_flow = args.pv_training_flow
    if args.pv_value_scale_mode is not None:
        hyperparams.pv_value_scale_mode = args.pv_value_scale_mode
    if args.pv_value_scale_log_max is not None:
        hyperparams.pv_value_scale_log_max = float(args.pv_value_scale_log_max)
    if args.pv_bellman_normalize_by_value_scale is not None:
        hyperparams.pv_bellman_normalize_by_value_scale = bool(args.pv_bellman_normalize_by_value_scale)
    if args.pv_eval_epochs is not None:
        hyperparams.pv_eval_epochs = args.pv_eval_epochs
    if args.q_parameterization is not None:
        hyperparams.q_parameterization = args.q_parameterization
    Config.Q_PARAMETERIZATION = str(hyperparams.q_parameterization).lower()
    q_scalar_overrides = {
        "q_zero_boundary_epochs": args.q_zero_boundary_epochs,
        "q_default_pretrain_epochs": args.q_default_pretrain_epochs,
        "q_survival_aio_epochs": args.q_survival_aio_epochs,
        "q_mixed_polish_epochs": args.q_mixed_polish_epochs,
        "q_target_refresh_mode": args.q_target_refresh_mode,
        "q_bellman_normalize_by_target_scale": args.q_bellman_normalize_by_target_scale,
        "q_stage_lr": args.q_stage_lr,
        "q_epoch_validation_enabled": args.q_epoch_validation_enabled,
        "q_validation_max_batches": args.q_validation_max_batches,
        "q_validation_min_delta": args.q_validation_min_delta,
        "q_restore_best_checkpoint": args.q_restore_best_checkpoint,
        "q_no_improvement_patience_episodes": args.q_no_improvement_patience_episodes,
        "save_episode_diagnostics": args.save_episode_diagnostics,
        "save_intermediate_stage_checkpoints": args.save_intermediate_stage_checkpoints,
        "q_zero_sample_share": args.q_zero_sample_share,
        "q_default_sample_share": args.q_default_sample_share,
        "q_survival_sample_share": args.q_survival_sample_share,
        "q_default_phat_eps": args.q_default_phat_eps,
        "q_zero_b_eps": args.q_zero_b_eps,
        "q_default_candidate_multiplier": args.q_default_candidate_multiplier,
        "q_default_b_bins": args.q_default_b_bins,
        "q_survival_ondist_share": args.q_survival_ondist_share,
        "q_claim_coverage_enabled": args.q_claim_coverage_enabled,
        "q_claim_coverage_b_bins": args.q_claim_coverage_b_bins,
        "q_claim_coverage_start_episode": args.q_claim_coverage_start_episode,
        "q_claim_coverage_low_b_enabled": args.q_claim_coverage_low_b_enabled,
        "q_claim_coverage_low_b_anchors": args.q_claim_coverage_low_b_anchors,
        "q_zero_loss_weight": args.q_zero_loss_weight,
        "q_default_loss_weight": args.q_default_loss_weight,
        "q_survival_loss_weight": args.q_survival_loss_weight,
        "q_nonnegative_weight": args.q_nonnegative_weight,
        "q_shape_weight_z": args.q_shape_weight_z,
        "q_shape_weight_b_low": args.q_shape_weight_b_low,
        "q_shape_weight_b_high": args.q_shape_weight_b_high,
        "q_bootstrap_epochs": args.q_bootstrap_epochs,
        "q_bootstrap_mode": args.q_bootstrap_mode,
        "q_bootstrap_unit_value": args.q_bootstrap_unit_value,
        "q_bootstrap_nonnegative_weight": args.q_bootstrap_nonnegative_weight,
        "q_structural_zero_tol": args.q_structural_zero_tol,
        "q_structural_recovery_tol": args.q_structural_recovery_tol,
        "q_boundary_match_weight": args.q_boundary_match_weight,
        "q_boundary_match_phat_eps": args.q_boundary_match_phat_eps,
        "q_boundary_match_min_samples": args.q_boundary_match_min_samples,
        "q_require_zero_phase": args.q_require_zero_phase,
        "q_require_default_phase": args.q_require_default_phase,
        "q_require_survival_phase": args.q_require_survival_phase,
    }
    for name, value in q_scalar_overrides.items():
        if value is not None:
            setattr(hyperparams, name, value)
    if str(hyperparams.q_parameterization).lower() not in {"direct", "hybrid_regime"}:
        raise ValueError(
            "Formal staged Q regime training requires q_parameterization='direct' "
            "or 'hybrid_regime'."
        )
    for name in (
        "q_zero_boundary_epochs",
        "q_default_pretrain_epochs",
        "q_survival_aio_epochs",
        "q_mixed_polish_epochs",
    ):
        if int(getattr(hyperparams, name)) < 0:
            raise ValueError(f"{name} must be non-negative")
    if str(hyperparams.q_target_refresh_mode).lower() not in {"phase", "stage"}:
        raise ValueError("q_target_refresh_mode must be 'phase' or 'stage'")
    if hyperparams.q_stage_lr is not None and float(hyperparams.q_stage_lr) <= 0.0:
        raise ValueError("q_stage_lr must be positive when provided")
    if int(hyperparams.q_validation_max_batches) <= 0:
        raise ValueError("q_validation_max_batches must be positive")
    if float(hyperparams.q_validation_min_delta) < 0.0:
        raise ValueError("q_validation_min_delta must be non-negative")
    if int(hyperparams.q_no_improvement_patience_episodes) < 0:
        raise ValueError("q_no_improvement_patience_episodes must be non-negative")
    q_shares = [
        float(getattr(hyperparams, "q_zero_sample_share")),
        float(getattr(hyperparams, "q_default_sample_share")),
        float(getattr(hyperparams, "q_survival_sample_share")),
    ]
    if any(value < 0.0 for value in q_shares) or not np.isclose(sum(q_shares), 1.0):
        raise ValueError("Q polishing sample shares must be non-negative and sum to 1.")
    if float(hyperparams.q_zero_b_eps) < 0.0:
        raise ValueError("q_zero_b_eps must be non-negative")
    if int(hyperparams.q_default_candidate_multiplier) <= 0 or int(hyperparams.q_default_b_bins) <= 0:
        raise ValueError("Q default coverage multiplier and b bins must be positive")
    for name in (
        "q_structural_zero_tol",
        "q_structural_recovery_tol",
        "q_boundary_match_weight",
        "q_boundary_match_phat_eps",
    ):
        if float(getattr(hyperparams, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if int(hyperparams.q_boundary_match_min_samples) < 0:
        raise ValueError("q_boundary_match_min_samples must be non-negative")
    if not 0.0 < float(hyperparams.q_survival_ondist_share) <= 1.0:
        raise ValueError("q_survival_ondist_share must be in (0, 1]")
    if int(hyperparams.q_claim_coverage_b_bins) < 2:
        raise ValueError("q_claim_coverage_b_bins must be at least 2")
    if int(hyperparams.q_claim_coverage_start_episode) < 0:
        raise ValueError("q_claim_coverage_start_episode must be non-negative")
    low_b_anchors = tuple(
        float(value)
        for value in getattr(hyperparams, "q_claim_coverage_low_b_anchors", ())
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in low_b_anchors):
        raise ValueError("q_claim_coverage_low_b_anchors must be finite and positive")
    for name in (
        "q_zero_loss_weight",
        "q_default_loss_weight",
        "q_survival_loss_weight",
        "q_nonnegative_weight",
    ):
        if float(getattr(hyperparams, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    for name in (
        "q_shape_weight_z",
        "q_shape_weight_b_low",
        "q_shape_weight_b_high",
    ):
        if float(getattr(hyperparams, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if int(getattr(hyperparams, "q_bootstrap_epochs", 0)) < 0:
        raise ValueError("q_bootstrap_epochs must be non-negative")
    if float(getattr(hyperparams, "q_bootstrap_nonnegative_weight", 0.0)) < 0.0:
        raise ValueError("q_bootstrap_nonnegative_weight must be non-negative")
    if str(getattr(hyperparams, "q_bootstrap_mode", "constant_unit")).lower() != "constant_unit":
        raise ValueError(
            "Only q_bootstrap_mode='constant_unit' is implemented; a legacy direct-Q "
            "distillation mode would require a migration tool that does not exist yet."
        )
    print(
        "[q-baseline] q_shape_weight_z={:g} q_shape_weight_b_low={:g} "
        "q_shape_weight_b_high={:g} (shape priors are OFF only when all three are 0)".format(
            float(hyperparams.q_shape_weight_z),
            float(hyperparams.q_shape_weight_b_low),
            float(hyperparams.q_shape_weight_b_high),
        )
    )
    print(
        "[q-baseline] q_bootstrap_epochs={} mode={} unit_value={:g} | "
        "required-phase gate zero/default/survival={}/{}/{}".format(
            int(hyperparams.q_bootstrap_epochs),
            hyperparams.q_bootstrap_mode,
            float(hyperparams.q_bootstrap_unit_value),
            bool(hyperparams.q_require_zero_phase),
            bool(hyperparams.q_require_default_phase),
            bool(hyperparams.q_require_survival_phase),
        )
    )
    if args.bp_distill_epochs is not None:
        hyperparams.bp_distill_epochs = args.bp_distill_epochs
    if args.firm_inner_cycles is not None:
        hyperparams.firm_inner_cycles = args.firm_inner_cycles
    if args.p_solver is not None:
        hyperparams.p_solver = args.p_solver
    if args.p_policy_iterations_per_inner is not None:
        hyperparams.p_policy_iterations_per_inner = args.p_policy_iterations_per_inner
    if args.p_root_adam_steps is not None:
        hyperparams.p_root_adam_steps = args.p_root_adam_steps
    if args.p_root_adam_lr is not None:
        hyperparams.p_root_adam_lr = args.p_root_adam_lr
    if args.p_root_adam_clip is not None:
        hyperparams.p_root_adam_clip = args.p_root_adam_clip
    if args.p_root_eval_every is not None:
        hyperparams.p_root_eval_every = args.p_root_eval_every
    if args.p_root_lbfgs_max_iter is not None:
        hyperparams.p_root_lbfgs_max_iter = args.p_root_lbfgs_max_iter
    if args.p_root_lbfgs_chunk_iterations is not None:
        hyperparams.p_root_lbfgs_chunk_iterations = args.p_root_lbfgs_chunk_iterations
    if args.p_root_lbfgs_history_size is not None:
        hyperparams.p_root_lbfgs_history_size = args.p_root_lbfgs_history_size
    if args.p_root_target_residual is not None:
        hyperparams.p_root_target_residual = args.p_root_target_residual
    if args.pv_bp_drift_mode is not None:
        hyperparams.pv_bp_drift_mode = str(args.pv_bp_drift_mode)
    if args.p_bp_drift_subset_rows is not None:
        hyperparams.p_bp_drift_subset_rows = int(args.p_bp_drift_subset_rows)
    if args.pv_fresh_initial_source is not None:
        hyperparams.pv_fresh_initial_source = str(args.pv_fresh_initial_source)
    if args.q_solver is not None:
        hyperparams.q_solver = args.q_solver
    if args.initial_hstar_enabled is not None:
        hyperparams.initial_hstar_enabled = bool(args.initial_hstar_enabled)
    if args.initial_hstar_residual_tolerance is not None:
        hyperparams.initial_hstar_residual_tolerance = (
            args.initial_hstar_residual_tolerance
        )
    if args.initial_hstar_parity_tolerance is not None:
        hyperparams.initial_hstar_parity_tolerance = (
            args.initial_hstar_parity_tolerance
        )
    if args.initial_hstar_max_iterations is not None:
        hyperparams.initial_hstar_max_iterations = (
            args.initial_hstar_max_iterations
        )
    if args.initial_hstar_coarse_points is not None:
        hyperparams.initial_hstar_coarse_points = args.initial_hstar_coarse_points
    if args.initial_hstar_refinement_points is not None:
        hyperparams.initial_hstar_refinement_points = (
            args.initial_hstar_refinement_points
        )
    if args.initial_hstar_max_refinement_rounds is not None:
        hyperparams.initial_hstar_max_refinement_rounds = (
            args.initial_hstar_max_refinement_rounds
        )
    if args.initial_hstar_support_lower is not None:
        hyperparams.initial_hstar_support_lower = args.initial_hstar_support_lower
    if args.initial_hstar_support_upper is not None:
        hyperparams.initial_hstar_support_upper = args.initial_hstar_support_upper
    if args.initial_hstar_width_tolerance is not None:
        hyperparams.initial_hstar_width_tolerance = args.initial_hstar_width_tolerance
    if args.initial_hstar_audit_output_dir is not None:
        hyperparams.initial_hstar_audit_output_dir = (
            args.initial_hstar_audit_output_dir
        )
    if args.initial_hstar_audit_episode_id is not None:
        hyperparams.initial_hstar_audit_episode_id = args.initial_hstar_audit_episode_id
    if args.hstar_recalib_count is not None:
        hyperparams.hstar_recalib_count = int(args.hstar_recalib_count)
    if getattr(args, "pv_eta1_b_augment_mode", None) is not None:
        hyperparams.pv_eta1_b_augment_mode = str(args.pv_eta1_b_augment_mode)
    if getattr(args, "pv_eta1_b_augment_share", None) is not None:
        hyperparams.pv_eta1_b_augment_share = float(args.pv_eta1_b_augment_share)
    if getattr(args, "pv_eta1_b_augment_bins", None) is not None:
        hyperparams.pv_eta1_b_augment_bins = int(args.pv_eta1_b_augment_bins)
    if getattr(args, "pv_eta1_b_augment_seed", None) is not None:
        hyperparams.pv_eta1_b_augment_seed = int(args.pv_eta1_b_augment_seed)
    if args.hstar_recalib_every is not None:
        hyperparams.hstar_recalib_every = int(args.hstar_recalib_every)
    if args.bp_distill_patience is not None:
        hyperparams.bp_distill_patience = args.bp_distill_patience
    if args.bp_distill_min_delta is not None:
        hyperparams.bp_distill_min_delta = args.bp_distill_min_delta
    if args.bp_distill_max_optimizer_steps is not None:
        hyperparams.bp_distill_max_optimizer_steps = int(args.bp_distill_max_optimizer_steps)
    if args.bp_current_eta_resample_enabled is not None:
        hyperparams.bp_current_eta_resample_enabled = bool(
            args.bp_current_eta_resample_enabled
        )
    if args.bp_current_eta1_train_share is not None:
        hyperparams.bp_current_eta1_train_share = float(
            args.bp_current_eta1_train_share
        )
    if args.bp_current_eta_resample_seed is not None:
        hyperparams.bp_current_eta_resample_seed = int(
            args.bp_current_eta_resample_seed
        )
    if args.pv_current_eta_balance_enabled is not None:
        hyperparams.pv_current_eta_balance_enabled = bool(
            args.pv_current_eta_balance_enabled
        )
    if args.pv_current_eta1_train_share is not None:
        hyperparams.pv_current_eta1_train_share = float(
            args.pv_current_eta1_train_share
        )
    if args.pv_current_eta_balance_validation is not None:
        hyperparams.pv_current_eta_balance_validation = bool(
            args.pv_current_eta_balance_validation
        )
    if args.pv_current_eta_balance_seed is not None:
        hyperparams.pv_current_eta_balance_seed = int(
            args.pv_current_eta_balance_seed
        )
    if args.p_eta_natural_diag_enabled is not None:
        hyperparams.p_eta_natural_diag_enabled = bool(
            args.p_eta_natural_diag_enabled
        )
    if args.p_eta_natural_diag_max_parents is not None:
        hyperparams.p_eta_natural_diag_max_parents = int(
            args.p_eta_natural_diag_max_parents
        )
    if args.p_eta_natural_diag_chunk_size is not None:
        hyperparams.p_eta_natural_diag_chunk_size = int(
            args.p_eta_natural_diag_chunk_size
        )
    if args.p_eta_natural_diag_frequency is not None:
        hyperparams.p_eta_natural_diag_frequency = str(
            args.p_eta_natural_diag_frequency
        )
    if args.p_eta_natural_diag_seed is not None:
        hyperparams.p_eta_natural_diag_seed = int(args.p_eta_natural_diag_seed)
    if int(getattr(hyperparams, "p_eta_natural_diag_max_parents", 4096)) < 1:
        raise ValueError("p_eta_natural_diag_max_parents must be at least 1")
    if int(getattr(hyperparams, "p_eta_natural_diag_chunk_size", 512)) < 1:
        raise ValueError("p_eta_natural_diag_chunk_size must be at least 1")
    if str(getattr(hyperparams, "p_eta_natural_diag_frequency", "firm_inner_end")) not in {
        "firm_inner_end",
        "per_p_iteration",
        "off",
    }:
        raise ValueError(
            "p_eta_natural_diag_frequency must be 'firm_inner_end', "
            "'per_p_iteration' or 'off'"
        )
    if str(getattr(hyperparams, "pv_bp_drift_mode", "full")).lower() not in {
        "full",
        "subset",
        "off",
    }:
        raise ValueError("pv_bp_drift_mode must be 'full', 'subset' or 'off'")
    if int(getattr(hyperparams, "p_bp_drift_subset_rows", 4096)) < 1:
        raise ValueError("p_bp_drift_subset_rows must be at least 1")
    if str(getattr(hyperparams, "pv_fresh_initial_source", "resolve")).lower() not in {
        "resolve",
        "reuse_build",
    }:
        raise ValueError("pv_fresh_initial_source must be 'resolve' or 'reuse_build'")
    if int(getattr(hyperparams, "bp_distill_max_optimizer_steps", 0)) < 0:
        raise ValueError("bp_distill_max_optimizer_steps must be non-negative")
    current_eta_share = float(
        getattr(hyperparams, "bp_current_eta1_train_share", 0.25)
    )
    if not 0.0 <= current_eta_share <= 1.0:
        raise ValueError("bp_current_eta1_train_share must be in [0, 1]")
    p_current_eta_share = float(
        getattr(hyperparams, "pv_current_eta1_train_share", 0.50)
    )
    if not 0.0 <= p_current_eta_share <= 1.0:
        raise ValueError("pv_current_eta1_train_share must be in [0, 1]")
    if args.pv_rollback_on_soft_spikes is not None:
        hyperparams.pv_rollback_on_soft_spikes = bool(args.pv_rollback_on_soft_spikes)
    if args.bp_distill_trainable_scope is not None:
        hyperparams.bp_distill_trainable_scope = args.bp_distill_trainable_scope
    if args.pv_mixture_enabled is not None:
        hyperparams.pv_mixture_enabled = bool(args.pv_mixture_enabled)
    if args.pv_mixture_ratio is not None:
        hyperparams.pv_mixture_ratio = float(args.pv_mixture_ratio)
    if args.pv_mixture_start_episode is not None:
        hyperparams.pv_mixture_start_episode = int(args.pv_mixture_start_episode)
    if args.pv_mixture_budget_mode is not None:
        hyperparams.pv_mixture_budget_mode = args.pv_mixture_budget_mode
    if args.pv_mixture_sampling_mode is not None:
        hyperparams.pv_mixture_sampling_mode = args.pv_mixture_sampling_mode
    if args.pv_mixture_coverage_group_size is not None:
        hyperparams.pv_mixture_coverage_group_size = int(args.pv_mixture_coverage_group_size)
    if args.pv_mixture_seed is not None:
        hyperparams.pv_mixture_seed = int(args.pv_mixture_seed)
    if args.pv_mixture_stratified_validation is not None:
        hyperparams.pv_mixture_stratified_validation = bool(args.pv_mixture_stratified_validation)
    if args.pv_mixture_preserve_rng is not None:
        hyperparams.pv_mixture_preserve_rng = bool(args.pv_mixture_preserve_rng)
    if not 0.0 <= float(getattr(hyperparams, "pv_mixture_ratio", 0.0)) <= 1.0:
        raise ValueError("pv_mixture_ratio must be in [0, 1].")
    if not 0.0 <= float(getattr(hyperparams, "state_bank_pq_coverage_ratio", 0.80)) <= 1.0:
        raise ValueError("state_bank_pq_coverage_ratio must be in [0, 1].")
    if str(getattr(hyperparams, "pv_mixture_budget_mode", "fixed_total")).lower() != "fixed_total":
        raise ValueError("pv_mixture_budget_mode must be fixed_total.")
    if int(getattr(hyperparams, "pv_mixture_coverage_group_size", 2)) <= 0:
        raise ValueError("pv_mixture_coverage_group_size must be positive.")
    if args.pv_eta_resample_enabled is not None:
        hyperparams.pv_eta_resample_enabled = bool(args.pv_eta_resample_enabled)
    if args.pv_exact_eta_integration_enabled is not None:
        hyperparams.pv_exact_eta_integration_enabled = bool(
            args.pv_exact_eta_integration_enabled
        )
    if bool(getattr(hyperparams, "pv_mixture_enabled", False)) or (
        bool(getattr(hyperparams, "state_bank_enabled", False))
        and bool(getattr(hyperparams, "state_bank_pq_decoupling_enabled", False))
    ):
        if bool(getattr(hyperparams, "pv_eta_resample_enabled", False)):
            raise ValueError("PV mixture currently requires --no-pv-eta-resample-enabled.")
        if str(getattr(hyperparams, "pv_training_flow", "joint")).lower() != "staged":
            raise ValueError("PV mixture currently requires --pv-training-flow staged.")
    if args.firm_target_update is not None:
        hyperparams.firm_target_update = args.firm_target_update
    if int(getattr(hyperparams, "firm_inner_cycles", 1)) < 1:
        raise ValueError("firm_inner_cycles must be at least 1.")
    if int(getattr(hyperparams, "firm_inner_cycles", 1)) > 1 and (
        str(getattr(hyperparams, "pv_training_flow", "joint")).lower() != "staged"
    ):
        raise ValueError("firm_inner_cycles>1 requires --pv-training-flow staged.")
    if str(getattr(hyperparams, "q_solver", "ep3_regime_staged")).lower() != (
        "ep3_regime_staged"
    ):
        raise ValueError(
            "q_solver must remain 'ep3_regime_staged'; the EP3 Q0/QD/QS/Polish "
            "regime-staged training is the only permitted Q solver."
        )
    p_solver_mode = str(getattr(hyperparams, "p_solver", "fixed_target_cache")).lower()
    if p_solver_mode not in {"fixed_target_cache", "frozen_policy_dynamic_residual"}:
        raise ValueError(
            "p_solver must be 'fixed_target_cache' or "
            f"'frozen_policy_dynamic_residual', got {p_solver_mode!r}"
        )
    if p_solver_mode == "frozen_policy_dynamic_residual":
        if str(getattr(hyperparams, "pv_training_flow", "joint")).lower() != "staged":
            raise ValueError(
                "p_solver=frozen_policy_dynamic_residual requires --pv-training-flow staged."
            )
        if int(getattr(hyperparams, "p_policy_iterations_per_inner", 3)) < 1:
            raise ValueError("p_policy_iterations_per_inner must be at least 1.")
        for name, minimum in (
            ("p_root_adam_steps", 1),
            ("p_root_adam_lr", 0.0),
            ("p_root_adam_clip", 0.0),
            ("p_root_eval_every", 1),
            ("p_root_lbfgs_max_iter", 0),
            ("p_root_lbfgs_chunk_iterations", 1),
            ("p_root_lbfgs_history_size", 1),
        ):
            value = float(getattr(hyperparams, name))
            lower_bound = float(minimum)
            if name == "p_root_lbfgs_max_iter":
                if value < lower_bound:
                    raise ValueError(f"{name} is out of range: {value!r}")
            elif value <= lower_bound:
                raise ValueError(f"{name} is out of range: {value!r}")
        target_residual = getattr(hyperparams, "p_root_target_residual", None)
        if target_residual is not None and float(target_residual) <= 0.0:
            raise ValueError("p_root_target_residual must be positive when set.")
    if bool(getattr(hyperparams, "initial_hstar_enabled", False)):
        if bool(getattr(hyperparams, "state_bank_enabled", False)):
            raise ValueError(
                "initial_hstar_enabled requires a fresh t=0 initialization and "
                "cannot run with state_bank_enabled."
            )
        if float(getattr(hyperparams, "initial_hstar_residual_tolerance", 1e-5)) <= 0.0:
            raise ValueError("initial_hstar_residual_tolerance must be positive.")
        if float(getattr(hyperparams, "initial_hstar_parity_tolerance", 1e-5)) <= 0.0:
            raise ValueError("initial_hstar_parity_tolerance must be positive.")
    if int(getattr(hyperparams, "hstar_recalib_count", 0)) < 0:
        raise ValueError("hstar_recalib_count must be nonnegative.")
    if int(getattr(hyperparams, "hstar_recalib_every", 0)) < 0:
        raise ValueError("hstar_recalib_every must be nonnegative.")
    hyperparams.policy_value_bellman_only = args.ablation_mode == "bellman_only"
    hyperparams.pv_fixed_sdf = args.ablation_mode == "fixed_sdf"
    hyperparams.pv_fixed_policy = args.ablation_mode == "fixed_policy"
    if str(getattr(hyperparams, "bp_grid_policy_loss_space", "output")).lower() not in {"output", "logit"}:
        raise ValueError(f"Unknown bp_grid_policy_loss_space={hyperparams.bp_grid_policy_loss_space!r}")
    if not 0.0 < float(getattr(hyperparams, "bp_grid_logit_target_eps", 1e-4)) < 0.5:
        raise ValueError("bp_grid_logit_target_eps must be in (0, 0.5)")
    if float(getattr(hyperparams, "bp_grid_logit_huber_delta", 1.0)) <= 0.0:
        raise ValueError("bp_grid_logit_huber_delta must be positive")
    if hyperparams.bp_grid_policy_loss_space == "logit" and hyperparams.pv_fixed_policy:
        raise ValueError("bp_grid_policy_loss_space='logit' is incompatible with pv_fixed_policy=True.")
    if str(getattr(hyperparams, "pv_training_flow", "joint")).lower() == "staged":
        firm_mode = str(getattr(hyperparams, "firm_target_update", "epoch_hard")).lower()
        if firm_mode not in {"stage_hard", "none"}:
            raise ValueError(
                "pv_training_flow='staged' requires --firm-target-update stage_hard or none; "
                f"got {firm_mode!r}."
            )
    if args.max_firm_train_units is not None:
        hyperparams.max_firm_train_units = args.max_firm_train_units
    if args.pv_fixed_sdf_value is not None:
        hyperparams.pv_fixed_sdf_value = args.pv_fixed_sdf_value
    if args.pv_fixed_policy_mode is not None:
        hyperparams.pv_fixed_policy_mode = args.pv_fixed_policy_mode
    if args.policy_grad_threshold is not None:
        hyperparams.policy_value_grad_fail_threshold = args.policy_grad_threshold
    if args.policy_rolling_grad_threshold is not None:
        hyperparams.policy_value_rolling_grad_fail_threshold = args.policy_rolling_grad_threshold
    if args.policy_loss_threshold is not None:
        hyperparams.policy_value_loss_fail_threshold = args.policy_loss_threshold
    if args.pv_sdf_clip_ratio_gate is not None:
        hyperparams.pv_sdf_clip_ratio_gate = args.pv_sdf_clip_ratio_gate
    validate_sdf_fresh_pair_config(hyperparams)
    return hyperparams


def _format_cuda_mem(n_bytes: int | float) -> str:
    return f"{float(n_bytes) / (1024 ** 2):.1f} MB"


def log_gpu_stats(prefix: str, device: torch.device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    idx = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    allocated = torch.cuda.memory_allocated(idx)
    reserved = torch.cuda.memory_reserved(idx)
    max_allocated = torch.cuda.max_memory_allocated(idx)
    print(f"{prefix} GPU: {props.name}")
    print(
        f"{prefix}   Allocated: {_format_cuda_mem(allocated)} / "
        f"{_format_cuda_mem(props.total_memory)} ({allocated / max(props.total_memory, 1) * 100:.1f}%)"
    )
    print(f"{prefix}   Reserved: {_format_cuda_mem(reserved)}")
    print(f"{prefix}   Max Allocated: {_format_cuda_mem(max_allocated)}")


def make_run_root(arg_path: Path | None) -> Path:
    if arg_path is not None:
        return arg_path.expanduser().resolve()
    return ROOT.parent / "cachedir" / datetime.now().strftime("%Y%m%d_%H%M")


def resolve_start_episode(
    *,
    requested_start_episode: int | None,
    warmstart_provenance: dict | None,
    initial_checkpoint_mode: str = "continuation",
) -> int:
    """Resolve numbering for continuation versus model-only initialization."""
    checkpoint_mode = str(initial_checkpoint_mode).lower()
    if checkpoint_mode not in {"continuation", "model_init"}:
        raise ValueError(
            "--initial-checkpoint-mode must be 'continuation' or 'model_init'"
        )
    if warmstart_provenance is None:
        if checkpoint_mode == "model_init":
            raise ValueError(
                "--initial-checkpoint-mode=model_init requires --initial-checkpoint"
            )
        resolved = 0 if requested_start_episode is None else int(requested_start_episode)
        if resolved != 0:
            raise ValueError("--start-episode must be 0 when --initial-checkpoint is omitted")
        return resolved

    if checkpoint_mode == "model_init":
        resolved = 0 if requested_start_episode is None else int(requested_start_episode)
        if resolved != 0:
            raise ValueError(
                "Model-only checkpoint initialization must begin at Episode 0: "
                f"got ep{resolved}"
            )
        return resolved

    source_episode = int(warmstart_provenance["source_episode"])
    required_start = source_episode + 1
    resolved = required_start if requested_start_episode is None else int(requested_start_episode)
    if resolved != required_start:
        raise ValueError(
            "Continuation warm-start must begin at the next logical episode: "
            f"source=ep{source_episode}, required start=ep{required_start}, got ep{resolved}"
        )
    return resolved


def build_and_commit_state_bank(
    episode: Episode,
    *,
    output_path: Path,
    n_paths: int,
    group_size: int,
    n_branches: int,
    simulate_kwargs: dict,
    episode_summary: dict | None = None,
    previous_bank: StateBank | None = None,
) -> StateBank:
    """Build, validate, then atomically publish one accepted candidate Bank."""
    candidate = episode.build_state_bank_after_success(
        n_paths=n_paths,
        group_size=group_size,
        n_branches=n_branches,
        simulate_kwargs=simulate_kwargs,
    )
    episode_summary = {} if episode_summary is None else episode_summary
    acceptance = evaluate_state_bank_acceptance(
        episode_summary=episode_summary,
        candidate_bank=candidate,
        hyperparams=episode.hyperparams,
    )
    candidate.metadata["acceptance"] = acceptance
    episode_summary.setdefault("module_summaries", {}).setdefault(
        "state_bank", {}
    )["acceptance"] = acceptance
    sdf_publication = acceptance["sdf_publication_gate"]
    print(
        "SDF publication status: "
        f"required={sdf_publication['required']} "
        f"safe_to_continue={sdf_publication['safe_to_continue']} "
        f"stage_progress={sdf_publication['stage_progress']} "
        f"converged={sdf_publication['converged']} "
        f"publication_passed={sdf_publication['publication_passed']} "
        f"reason={sdf_publication['reason']}"
    )
    print(
        "State Bank strict acceptance: "
        f"checks={acceptance['strict_checks']} "
        f"bellman={acceptance['bellman_gate']['status']} "
        f"bank_integrity={acceptance['refresh_finite_and_accounting_valid']} "
        f"publish={acceptance['episode_acceptance_passed']}"
    )
    if not acceptance["episode_acceptance_passed"]:
        raise RuntimeError(
            "State Bank candidate rejected before atomic publish: "
            + "; ".join(acceptance["reasons"])
        )
    previous_lineages = (
        0
        if previous_bank is None
        else len({str(entry["lineage_id"]) for entry in previous_bank.entries})
    )
    candidate_lineages = len(
        {str(entry["lineage_id"]) for entry in candidate.entries}
    )
    lineage_refresh = {
        "previous_unique_lineages": previous_lineages,
        "candidate_unique_lineages": candidate_lineages,
        "change": candidate_lineages - previous_lineages,
    }
    candidate.metadata["lineage_refresh"] = lineage_refresh
    if previous_lineages and candidate_lineages < previous_lineages:
        print(
            "WARNING: State Bank lineage count declined: "
            f"{previous_lineages} -> {candidate_lineages}"
        )
    candidate.save(output_path, atomic=True)
    distribution = candidate.distribution_summary()
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "acceptance": acceptance,
                "distribution": distribution,
                "economic_feasibility": distribution["economic_feasibility"],
                "rollout": candidate.metadata.get("rollout_diagnostics", {}),
                "lineage_refresh": lineage_refresh,
                "compatibility_fingerprint": candidate.compatibility_fingerprint,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    return candidate


def _state_bank_core_fields_are_finite(bank: StateBank) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for entry in bank.entries:
        fields = entry["fields"]
        alive = fields.get("alive")
        for name in ("x", "b", "z", "eta", "i", "K", "hatcf", "lnkf", "M"):
            value = fields.get(name)
            if not torch.is_tensor(value):
                errors.append(f"{entry['entry_id']}:missing:{name}")
                continue
            selected = value
            if alive is not None and value.ndim >= 1 and value.shape[0] == alive.shape[0]:
                selected = value[alive.to(torch.bool)]
            if selected.numel() and not bool(torch.isfinite(selected).all()):
                errors.append(f"{entry['entry_id']}:nonfinite:{name}")
        accounting = fields.get("accounting_valid")
        if accounting is not None and not bool(accounting.to(torch.bool).all()):
            errors.append(f"{entry['entry_id']}:accounting_invalid")
        if len(errors) >= 20:
            break
    return not errors, errors


def evaluate_state_bank_acceptance(
    *,
    episode_summary: dict,
    candidate_bank: StateBank,
    hyperparams,
) -> dict:
    mode = str(getattr(hyperparams, "state_bank_acceptance_mode", "numerical"))
    reasons: list[str] = []
    finite, finite_errors = _state_bank_core_fields_are_finite(candidate_bank)
    if not candidate_bank.entries:
        reasons.append("refresh_bank_empty")
    if not finite:
        reasons.append("refresh_bank_nonfinite_or_accounting_invalid")
    unique_lineages = len(
        {str(entry["lineage_id"]) for entry in candidate_bank.entries}
    )
    minimum = int(getattr(hyperparams, "state_bank_min_unique_lineages", 0))
    if unique_lineages < minimum:
        reasons.append(
            f"unique_lineages={unique_lineages}<required={minimum}"
        )

    economic_filter_enabled = bool(
        getattr(
            hyperparams,
            "state_bank_economic_feasibility_filter_enabled",
            False,
        )
    )
    stored_feasibility = summarize_stored_economic_feasibility(candidate_bank)
    economic_gate_passed = stored_feasibility["invalid_entry_count"] == 0
    economic_gate = {
        "required": economic_filter_enabled,
        "passed": bool(economic_gate_passed),
        "stored_entry_count": stored_feasibility["stored_entry_count"],
        "invalid_entry_count": stored_feasibility["invalid_entry_count"],
        "invalid_reasons": stored_feasibility["invalid_reasons"],
    }
    if economic_filter_enabled and not economic_gate_passed:
        reasons.append("state_bank_contains_economically_ineligible_entries")

    modules = episode_summary.get("module_summaries", {})
    strict_checks: dict[str, object] = {}
    bellman_mean_threshold = getattr(
        hyperparams,
        "bellman_conditional_mean_thresh",
        None,
    )
    bellman_p90_threshold = getattr(
        hyperparams,
        "bellman_conditional_p90_thresh",
        None,
    )
    if (bellman_mean_threshold is None) != (bellman_p90_threshold is None):
        raise ValueError(
            "Bellman conditional thresholds must be both configured or both None"
        )
    bellman_gate_configured = (
        bellman_mean_threshold is not None
        and bellman_p90_threshold is not None
    )
    convergence = episode_summary.get("convergence")
    bellman_passed = (
        convergence.get("bellman_passed")
        if isinstance(convergence, dict)
        else None
    )
    bellman_gate = {
        "configured": bellman_gate_configured,
        "required": bellman_gate_configured,
        "status": (
            "passed"
            if bellman_gate_configured and bellman_passed is True
            else "failed"
            if bellman_gate_configured
            else "diagnostic_only"
        ),
        "mean_threshold": bellman_mean_threshold,
        "p90_threshold": bellman_p90_threshold,
        "bellman_passed": bellman_passed,
    }
    schedule = modules.get("sdf_formal_stage_schedule", {})
    formal_sdf_active = bool(schedule.get("formal_sdf_active", False))
    sdf_publication_gate_enabled = bool(
        getattr(hyperparams, "state_bank_sdf_publication_gate_enabled", False)
    )
    sdf_publication_gate = {
        "formal_sdf_active": formal_sdf_active,
        "required": bool(formal_sdf_active and sdf_publication_gate_enabled),
        "safe_to_continue": None,
        "stage_progress": None,
        "converged": None,
        "publication_passed": True,
        "reason": "formal_sdf_inactive_not_required",
    }
    if mode == "strict":
        if formal_sdf_active:
            sdf_gate = modules.get("sdf_fc1_post_refresh_gate")
            safe_to_continue = bool(
                isinstance(sdf_gate, dict)
                and sdf_gate.get("sdf_safe_to_continue") is True
            )
            stage_progress = bool(
                isinstance(sdf_gate, dict)
                and sdf_gate.get("sdf_stage_progress") is True
            )
            converged = bool(
                isinstance(sdf_gate, dict)
                and sdf_gate.get("sdf_converged") is True
            )
            publication_passed = bool(
                safe_to_continue and (stage_progress or converged)
            )
            if not isinstance(sdf_gate, dict):
                publication_reason = "sdf_continuation_status_missing"
            elif not safe_to_continue:
                publication_reason = "sdf_unsafe_for_continuation"
            elif converged:
                publication_reason = "sdf_converged"
            elif stage_progress:
                publication_reason = "safe_and_progressing_not_yet_converged"
            else:
                publication_reason = "sdf_safe_but_no_progress"
            sdf_publication_gate.update(
                {
                    "safe_to_continue": safe_to_continue,
                    "stage_progress": stage_progress,
                    "converged": converged,
                    "publication_passed": publication_passed,
                    "reason": publication_reason,
                }
            )
            if sdf_publication_gate_enabled:
                strict_checks["sdf_publication_gate_passed"] = publication_passed
                if not publication_passed:
                    reasons.append(publication_reason)

        policy = modules.get("policy_value")
        metadata = policy.get("metadata", {}) if isinstance(policy, dict) else {}
        stage_passed = metadata.get("policy_value_stage_status") == "accepted"
        q_passed = metadata.get("q_stage_required_gate_passed") is True
        bp_status = metadata.get("bp_distillation_stage", {}).get("status")
        bp_passed = bp_status == "accepted"
        strict_checks.update(
            {
                "policy_value_stage_passed": stage_passed,
                "q_required_gate_passed": q_passed,
                "bp_required_gate_passed": bp_passed,
            }
        )
        for key, passed in strict_checks.items():
            if passed is not True:
                reasons.append(f"{key}=False_or_unavailable")
        strict_checks["bellman_fixed_point_gate"] = (
            bool(bellman_passed)
            if bellman_gate_configured
            else "not_required_unconfigured"
        )
        if bellman_gate_configured and bellman_passed is not True:
            reasons.append("bellman_required_gate_failed")
    elif mode != "numerical":
        raise ValueError(f"Unknown state_bank_acceptance_mode={mode!r}")

    return {
        "mode": mode,
        "episode_acceptance_passed": not reasons,
        "reasons": reasons,
        "refresh_nonempty": bool(candidate_bank.entries),
        "refresh_finite_and_accounting_valid": finite,
        "refresh_validation_errors": finite_errors,
        "unique_lineages": unique_lineages,
        "minimum_unique_lineages": minimum,
        "strict_checks": strict_checks,
        "bank_economic_feasibility_gate": economic_gate,
        "sdf_publication_gate": sdf_publication_gate,
        "bellman_gate": bellman_gate,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    run_root = make_run_root(args.run_root)
    ensure_dirs(run_root)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    Config.DEVICE = device

    hyperparams = configure_hyperparams(args)
    hyperparams.entry_hatc_failure_scan_output_dir = str(
        run_root / "diagnostics" / "entry_hatc_failure_scans"
    )
    print(f"Global random seed: {args.seed}")
    print(
        "Entry configuration: "
        + json.dumps(
            {
                "entry_mode": hyperparams.entry_mode,
                "entry_spec_version": hyperparams.entry_spec_version,
                "resource_accounting_version": hyperparams.resource_accounting_version,
                "entry_capital_ratio": hyperparams.entry_capital_ratio,
                "entry_size_ratio": hyperparams.entry_size_ratio,
                "entry_cost_max": hyperparams.entry_cost_max,
                "entry_dummy_i": hyperparams.entry_dummy_i,
                "entry_inference_chunk_size": hyperparams.entry_inference_chunk_size,
                "entry_rng_seed": hyperparams.entry_rng_seed,
                "consumption_aggregation_mode": hyperparams.consumption_aggregation_mode,
                "consumption_floor": hyperparams.consumption_floor,
                "node_accounting_mode": hyperparams.node_accounting_mode,
                "entry_independent_resource_validation_enabled": (
                    hyperparams.entry_independent_resource_validation_enabled
                ),
                "entry_hatc_failure_scan_enabled": (
                    hyperparams.entry_hatc_failure_scan_enabled
                ),
                "entry_hatc_failure_scan_points": (
                    hyperparams.entry_hatc_failure_scan_points
                ),
                "entry_hatc_failure_scan_half_width": (
                    hyperparams.entry_hatc_failure_scan_half_width
                ),
                "entry_hatc_failure_scan_output_dir": (
                    hyperparams.entry_hatc_failure_scan_output_dir
                ),
            },
            sort_keys=True,
        )
    )
    firm_target = None
    warmstart_provenance = None
    if args.initial_checkpoint is None:
        if args.initial_checkpoint_episode is not None:
            raise ValueError(
                "--initial-checkpoint-episode requires --initial-checkpoint"
            )
        models = build_models(device)
    else:
        models, firm_target, warmstart_provenance = load_training_warmstart(
            args.initial_checkpoint,
            device,
            expected_episode=args.initial_checkpoint_episode,
            expected_q_parameterization=hyperparams.q_parameterization,
            expected_value_scale_mode=hyperparams.pv_value_scale_mode,
            expected_value_scale_log_max=hyperparams.pv_value_scale_log_max,
            expected_bellman_normalization=(
                hyperparams.pv_bellman_normalize_by_value_scale
            ),
        )
    start_episode = resolve_start_episode(
        requested_start_episode=args.start_episode,
        warmstart_provenance=warmstart_provenance,
        initial_checkpoint_mode=args.initial_checkpoint_mode,
    )
    if warmstart_provenance is not None:
        if args.initial_checkpoint_mode == "model_init":
            warmstart_provenance["mode"] = "model_initialization_fresh_training"
        else:
            warmstart_provenance["mode"] = "continuation_warmstart_fresh_optimizers"
        warmstart_provenance["start_episode"] = start_episode
        warmstart_provenance["simulation_state_restored"] = False
        warmstart_provenance["training_data_restored"] = False
        provenance_path = (
            run_root / "metadata" / "checkpoint_initialization_provenance.json"
        )
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(
            json.dumps(warmstart_provenance, indent=2, default=str),
            encoding="utf-8",
        )
        print(
            "Checkpoint initialization: "
            + json.dumps(warmstart_provenance, sort_keys=True)
        )
    else:
        print("Model initialization: fresh Xavier weights with fresh optimizers")
    optimizers = build_optimizers(models, hyperparams)
    post0_n_paths = args.post0_n_paths if args.post0_n_paths is not None else hyperparams.n_paths
    simulate_group_size = args.simulate_group_size if args.simulate_group_size is not None else Config.SIMULATE_GROUP_SIZE

    # Initialize GPU monitor
    gpu_monitor = get_monitor(device, log_interval=10)
    print(f"GPU Monitor initialized: {device}")

    summaries = []
    failure_report = None
    canonical_diagnostic_reference = None
    active_state_bank = None
    state_bank_dir = resolve_base_dir(run_root, ROOT) / "state_banks"
    if bool(hyperparams.state_bank_enabled):
        state_bank_dir.mkdir(parents=True, exist_ok=True)
        print(
            "State Bank v2 enabled: "
            f"directory={state_bank_dir}, "
            f"short_horizon={hyperparams.state_bank_short_horizon}, "
            f"bootstrap=({hyperparams.state_bank_bootstrap_burn_in},"
            f"{hyperparams.state_bank_bootstrap_capture_stride}), "
            f"refresh=({hyperparams.state_bank_refresh_burn_in},"
            f"{hyperparams.state_bank_refresh_capture_stride}), "
            f"parent_budget={hyperparams.state_bank_parent_transition_budget}, "
            f"budget_warn_ratio={hyperparams.state_bank_parent_budget_warn_ratio}, "
            f"sampling={hyperparams.state_bank_sampling_mode}, "
            f"acceptance={hyperparams.state_bank_acceptance_mode}, "
            "sdf_publication_gate="
            f"{hyperparams.state_bank_sdf_publication_gate_enabled}, "
            "economic_feasibility_filter="
            f"{hyperparams.state_bank_economic_feasibility_filter_enabled}, "
            "pq_decoupling="
            f"{hyperparams.state_bank_pq_decoupling_enabled}, "
            "pq_coverage_ratio="
            f"{hyperparams.state_bank_pq_coverage_ratio}, "
            f"max_entries={hyperparams.state_bank_max_entries}"
        )
    episode = Episode(
            models=models,
            optimizers=optimizers,
            config=Config,
            hyperparams=hyperparams,
            device=device,
            episode_id=start_episode,
            gpu_monitor=gpu_monitor,
            firm_target=firm_target,
            q_checkpoint_loaded=warmstart_provenance is not None,
        )
    for episode_offset in range(args.n_episodes):
        ep = start_episode + episode_offset
        episode.episode_id = ep  # update episode ID for logging/saving
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        data_kwargs = {
            "n_samples": hyperparams.n_samples,
            "n_paths": hyperparams.n_paths if ep == 0 else post0_n_paths,
            "group_size": 2 if ep == 0 else simulate_group_size,
            "n_branches": Config.BRANCH_NUM,
        }
        if ep == 0:
            episode_mode = "mode0"
        else:
            if args.post0_mode == "alternate":
                start = args.alternate_start
                offset = ep - 1
                if offset % 2 == 0:
                    episode_mode = start
                else:
                    episode_mode = "modeb" if start == "modea" else "modea"
            else:
                episode_mode = args.post0_mode

        train_modules = ["sdf_fc1", "policy_value"]
        if args.enable_fc2:
            train_modules.append("fc2")
        simulate_kwargs = {
            "horizon_mode1": 1,
            "horizon": hyperparams.simulate_horizon,
            "modeb_resimulate_after_pv": args.modeb_resimulate_after_pv,
        }
        try:
            if bool(hyperparams.state_bank_enabled) and ep >= 2:
                required_bank_path = state_bank_dir / f"bank_ep{ep - 1}.pt"
                if (
                    active_state_bank is None
                    or int(active_state_bank.source_episode) != ep - 1
                ):
                    try:
                        active_state_bank = StateBank.load(
                            required_bank_path,
                            expected_compatibility=episode._state_bank_compatibility(
                                data_kwargs["n_branches"]
                            ),
                        )
                    except Exception as exc:
                        raise NumericalStageFailure(
                            f"Episode {ep} could not load required State Bank",
                            diagnostics={
                                "stage": "state_bank_load",
                                "episode": ep,
                                "required_path": str(required_bank_path),
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                            },
                        ) from exc
                print(
                    f"[Episode {ep}] loaded State Bank: {required_bank_path} "
                    f"entries={len(active_state_bank)}"
                )
            summary = episode.run_episode(
                n_epochs=hyperparams.epochs,
                batch_size=hyperparams.batch_size,
                pv_batch_size=hyperparams.pv_batch_size,
                sdf_fc1_batch_size=hyperparams.sdf_fc1_batch_size,
                log_interval=50,
                train_modules=train_modules,
                simulate_kwargs=simulate_kwargs,
                episode_mode=episode_mode,
                state_bank=active_state_bank,
                **data_kwargs,
            )
            if bool(hyperparams.state_bank_enabled) and ep >= 1:
                try:
                    bank_path = state_bank_dir / f"bank_ep{ep}.pt"
                    candidate_bank = build_and_commit_state_bank(
                        episode,
                        output_path=bank_path,
                        n_paths=data_kwargs["n_paths"],
                        group_size=data_kwargs["group_size"],
                        n_branches=data_kwargs["n_branches"],
                        simulate_kwargs=simulate_kwargs,
                        episode_summary=summary,
                        previous_bank=active_state_bank,
                    )
                except Exception as exc:
                    acceptance = summary.get("module_summaries", {}).get(
                        "state_bank", {}
                    ).get("acceptance")
                    acceptance_rejected = bool(
                        isinstance(acceptance, dict)
                        and acceptance.get("episode_acceptance_passed") is False
                    )
                    raise NumericalStageFailure(
                        (
                            f"Episode {ep} State Bank acceptance rejected"
                            if acceptance_rejected
                            else f"Episode {ep} State Bank refresh failed"
                        ),
                        diagnostics={
                            "stage": (
                                "state_bank_acceptance"
                                if acceptance_rejected
                                else "state_bank_refresh"
                            ),
                            "episode": ep,
                            "previous_bank_source_episode": (
                                None
                                if active_state_bank is None
                                else int(active_state_bank.source_episode)
                            ),
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                            "acceptance": acceptance,
                        },
                    ) from exc
                active_state_bank = candidate_bank
                state_bank_summary = summary.setdefault(
                    "module_summaries", {}
                ).setdefault("state_bank", {})
                state_bank_summary.update(
                    {
                        "output_path": str(bank_path),
                        "output_entry_count": len(candidate_bank),
                        "output_fingerprint": (
                            candidate_bank.compatibility_fingerprint
                        ),
                        "output_capture": candidate_bank.metadata.get(
                            "capture", {}
                        ),
                        "output_distribution_summary": (
                            candidate_bank.distribution_summary()
                        ),
                        "output_rollout_diagnostics": candidate_bank.metadata.get(
                            "rollout_diagnostics", {}
                        ),
                        "lineage_refresh": candidate_bank.metadata.get(
                            "lineage_refresh", {}
                        ),
                        "output_summary_path": str(
                            bank_path.with_suffix(".summary.json")
                        ),
                    }
                )
                print(
                    f"[Episode {ep}] committed State Bank: {bank_path} "
                    f"entries={len(candidate_bank)}"
                )
        except NumericalStageFailure as exc:
            failure_report = {
                "failed_episode": ep,
                "episode_mode": episode_mode,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "stage_diagnostics": getattr(exc, "diagnostics", {}),
                "partial_module_summaries": getattr(
                    episode,
                    "_last_partial_module_summaries",
                    {},
                ),
                "last_failed_stage_diagnostics": getattr(
                    episode,
                    "_last_failed_stage_diagnostics",
                    {},
                ),
                "ablation_mode": args.ablation_mode,
                "gate_context": getattr(episode, "_last_policy_value_gate_context", {}),
                "last_policy_value_stage_summary": getattr(episode, "_last_policy_value_stage_summary", {}),
                "completed_summaries": summaries,
                "run_parameters": {
                    "n_episodes": args.n_episodes,
                    "start_episode": start_episode,
                    "warmstart": warmstart_provenance,
                    "epochs": hyperparams.epochs,
                    "n_paths": data_kwargs["n_paths"],
                    "batch_size": hyperparams.batch_size,
                    "pv_batch_size": hyperparams.pv_batch_size,
                    "sdf_fc1_batch_size": hyperparams.sdf_fc1_batch_size,
                    "simulate_group_size": simulate_group_size,
                    "simulate_horizon": hyperparams.simulate_horizon,
                    "state_bank_enabled": hyperparams.state_bank_enabled,
                    "state_bank_short_horizon": hyperparams.state_bank_short_horizon,
                    "state_bank_bootstrap_burn_in": hyperparams.state_bank_bootstrap_burn_in,
                    "state_bank_bootstrap_capture_stride": hyperparams.state_bank_bootstrap_capture_stride,
                    "state_bank_refresh_burn_in": hyperparams.state_bank_refresh_burn_in,
                    "state_bank_refresh_capture_stride": hyperparams.state_bank_refresh_capture_stride,
                    "state_bank_parent_transition_budget": hyperparams.state_bank_parent_transition_budget,
                    "state_bank_parent_budget_warn_ratio": hyperparams.state_bank_parent_budget_warn_ratio,
                    "state_bank_max_entries": hyperparams.state_bank_max_entries,
                    "state_bank_sampling_mode": hyperparams.state_bank_sampling_mode,
                    "state_bank_acceptance_mode": hyperparams.state_bank_acceptance_mode,
                    "state_bank_sdf_publication_gate_enabled": hyperparams.state_bank_sdf_publication_gate_enabled,
                    "state_bank_pq_decoupling_enabled": hyperparams.state_bank_pq_decoupling_enabled,
                    "state_bank_pq_coverage_ratio": hyperparams.state_bank_pq_coverage_ratio,
                    "state_bank_economic_feasibility_filter_enabled": hyperparams.state_bank_economic_feasibility_filter_enabled,
                    "state_bank_min_unique_lineages": hyperparams.state_bank_min_unique_lineages,
                    "state_bank_sample_seed": hyperparams.state_bank_sample_seed,
                    "state_bank_refresh_seed": hyperparams.state_bank_refresh_seed,
                    "state_bank_reservoir_seed": hyperparams.state_bank_reservoir_seed,
                    "post0_mode": args.post0_mode,
                    "sdf_wealth_loss_mode": hyperparams.sdf_wealth_loss_mode,
                    "sdf_true_only_epochs": hyperparams.sdf_true_only_epochs,
                    "sdf_true_start_episode": hyperparams.sdf_true_start_episode,
                    "sdf_recursive_only_epochs": hyperparams.sdf_recursive_only_epochs,
                    "sdf_stage2_lr": hyperparams.sdf_stage2_lr,
                    "sdf_true_only_lr": hyperparams.sdf_true_only_lr,
                    "sdf_true_target_batches": hyperparams.sdf_true_target_batches,
                    "sdf_min_parent_groups_per_batch": hyperparams.sdf_min_parent_groups_per_batch,
                    "sdf_true_moment_weight": hyperparams.sdf_true_moment_weight,
                    "sdf_true_anchor_weight": hyperparams.sdf_true_anchor_weight,
                    "sdf_moment_constraint_mode": hyperparams.sdf_moment_constraint_mode,
                    "sdf_al_rho": hyperparams.sdf_al_rho,
                    "sdf_al_primal_epochs_per_dual_update": hyperparams.sdf_al_primal_epochs_per_dual_update,
                    "sdf_al_lambda_init": hyperparams.sdf_al_lambda_init,
                    "sdf_al_eps": hyperparams.sdf_al_eps,
                    "sdf_al_gate_tolerance": hyperparams.sdf_al_gate_tolerance,
                    "sdf_continue_constraint_tol": hyperparams.sdf_continue_constraint_tol,
                    "sdf_aio_progress_ratio_max": hyperparams.sdf_aio_progress_ratio_max,
                    "sdf_aio_good_abs_tol": hyperparams.sdf_aio_good_abs_tol,
                    "sdf_final_constraint_tol": hyperparams.sdf_final_constraint_tol,
                    "sdf_final_max_signed_t_abs": hyperparams.sdf_final_max_signed_t_abs,
                    "sdf_final_aio_mean_tol": hyperparams.sdf_final_aio_mean_tol,
                    "sdf_al_dual_max_batches": hyperparams.sdf_al_dual_max_batches,
                    "sdf_al_reset_on_true_start": hyperparams.sdf_al_reset_on_true_start,
                    "sdf_al_strict_semantics_guard": hyperparams.sdf_al_strict_semantics_guard,
                    "sdf_al_noise_tax_warn_ratio": hyperparams.sdf_al_noise_tax_warn_ratio,
                    "sdf_reset_optimizer_on_true_start": hyperparams.sdf_reset_optimizer_on_true_start,
                    "sdf_restore_best_checkpoint": hyperparams.sdf_restore_best_checkpoint,
                    "sdf_collapse_log_mean_error": hyperparams.sdf_collapse_log_mean_error,
                    "sdf_collapse_mean_ratio": hyperparams.sdf_collapse_mean_ratio,
                    "pv_bp_training_mode": hyperparams.pv_bp_training_mode,
                    "pv_bp_head_training_enabled": hyperparams.pv_bp_head_training_enabled,
                    "simulation_bp_action_source": hyperparams.simulation_bp_action_source,
                    "simulation_bp_grid_n_child_shocks": hyperparams.simulation_bp_grid_n_child_shocks,
                    "simulation_bp_grid_shock_seed": hyperparams.simulation_bp_grid_shock_seed,
                    "pv_training_flow": hyperparams.pv_training_flow,
                    "pv_eval_epochs": hyperparams.pv_eval_epochs,
                    "firm_inner_cycles": hyperparams.firm_inner_cycles,
                    "p_solver": hyperparams.p_solver,
                    "p_policy_iterations_per_inner": (
                        hyperparams.p_policy_iterations_per_inner
                    ),
                    "p_root_adam_steps": hyperparams.p_root_adam_steps,
                    "p_root_adam_lr": hyperparams.p_root_adam_lr,
                    "p_root_adam_clip": hyperparams.p_root_adam_clip,
                    "p_root_eval_every": hyperparams.p_root_eval_every,
                    "p_root_lbfgs_max_iter": hyperparams.p_root_lbfgs_max_iter,
                    "p_root_lbfgs_chunk_iterations": (
                        hyperparams.p_root_lbfgs_chunk_iterations
                    ),
                    "p_root_lbfgs_history_size": hyperparams.p_root_lbfgs_history_size,
                    "p_root_target_residual": hyperparams.p_root_target_residual,
                    "pv_bp_drift_mode": hyperparams.pv_bp_drift_mode,
                    "p_bp_drift_subset_rows": hyperparams.p_bp_drift_subset_rows,
                    "pv_fresh_initial_source": hyperparams.pv_fresh_initial_source,
                    "q_solver": hyperparams.q_solver,
                    "initial_hstar_enabled": hyperparams.initial_hstar_enabled,
                    "initial_hstar_residual_tolerance": (
                        hyperparams.initial_hstar_residual_tolerance
                    ),
                    "initial_hstar_parity_tolerance": (
                        hyperparams.initial_hstar_parity_tolerance
                    ),
                    "initial_hstar_max_iterations": (
                        hyperparams.initial_hstar_max_iterations
                    ),
                    "initial_hstar_coarse_points": (
                        hyperparams.initial_hstar_coarse_points
                    ),
                    "initial_hstar_refinement_points": (
                        hyperparams.initial_hstar_refinement_points
                    ),
                    "initial_hstar_max_refinement_rounds": (
                        hyperparams.initial_hstar_max_refinement_rounds
                    ),
                    "initial_hstar_support_lower": (
                        hyperparams.initial_hstar_support_lower
                    ),
                    "initial_hstar_support_upper": (
                        hyperparams.initial_hstar_support_upper
                    ),
                    "initial_hstar_width_tolerance": (
                        hyperparams.initial_hstar_width_tolerance
                    ),
                    "bp_distill_epochs": hyperparams.bp_distill_epochs,
                    "bp_distill_patience": hyperparams.bp_distill_patience,
                    "bp_distill_max_optimizer_steps": hyperparams.bp_distill_max_optimizer_steps,
                    "bp_current_eta_resample_enabled": hyperparams.bp_current_eta_resample_enabled,
                    "bp_current_eta1_train_share": hyperparams.bp_current_eta1_train_share,
                    "bp_current_eta_resample_seed": hyperparams.bp_current_eta_resample_seed,
                    "pv_mixture_enabled": hyperparams.pv_mixture_enabled,
                    "pv_mixture_ratio": hyperparams.pv_mixture_ratio,
                    "pv_mixture_start_episode": hyperparams.pv_mixture_start_episode,
                    "pv_mixture_budget_mode": hyperparams.pv_mixture_budget_mode,
                    "pv_mixture_sampling_mode": hyperparams.pv_mixture_sampling_mode,
                    "pv_mixture_coverage_group_size": hyperparams.pv_mixture_coverage_group_size,
                    "pv_mixture_seed": hyperparams.pv_mixture_seed,
                    "pv_eta_resample_enabled": hyperparams.pv_eta_resample_enabled,
                    "pv_exact_eta_integration_enabled": hyperparams.pv_exact_eta_integration_enabled,
                    "eta_active_reweight_enabled": hyperparams.eta_active_reweight_enabled,
                    "pv_rollback_on_soft_spikes": hyperparams.pv_rollback_on_soft_spikes,
                    "firm_target_update": hyperparams.firm_target_update,
                    "modeb_resimulate_after_pv": args.modeb_resimulate_after_pv,
                    "max_firm_train_units": hyperparams.max_firm_train_units,
                    "entry_mode": hyperparams.entry_mode,
                    "entry_spec_version": hyperparams.entry_spec_version,
                    "resource_accounting_version": hyperparams.resource_accounting_version,
                    "entry_capital_ratio": hyperparams.entry_capital_ratio,
                    "entry_size_ratio": hyperparams.entry_size_ratio,
                    "entry_cost_max": hyperparams.entry_cost_max,
                    "entry_dummy_i": hyperparams.entry_dummy_i,
                    "entry_inference_chunk_size": hyperparams.entry_inference_chunk_size,
                    "entry_rng_seed": hyperparams.entry_rng_seed,
                    "consumption_aggregation_mode": hyperparams.consumption_aggregation_mode,
                    "consumption_floor": hyperparams.consumption_floor,
                    "node_accounting_mode": hyperparams.node_accounting_mode,
                    "entry_hatc_failure_scan_enabled": hyperparams.entry_hatc_failure_scan_enabled,
                    "entry_hatc_failure_scan_points": hyperparams.entry_hatc_failure_scan_points,
                    "entry_hatc_failure_scan_half_width": hyperparams.entry_hatc_failure_scan_half_width,
                    "entry_hatc_failure_scan_output_dir": hyperparams.entry_hatc_failure_scan_output_dir,
                },
            }
            failure_path = resolve_base_dir(run_root, ROOT) / "failure_report.json"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(json.dumps(failure_report, indent=2, default=str))
            print(f"Numerical stage failure saved to: {failure_path}")
            print(json.dumps(failure_report, indent=2, default=str))
            break
        ep_summary = summary.get("module_summaries", summary)
        save_stage_df(ep, episode_mode, resolve_base_dir(run_root, ROOT), episode.df, episode.df_macro, episode.df_sdf)
        if args.entry_macro_progress_plots:
            from analysis.entry_macro_progress import (  # noqa: E402
                generate_entry_macro_progress_report,
                is_entry_macro_plot_milestone,
            )

            if is_entry_macro_plot_milestone(
                episode_offset,
                args.n_episodes,
                args.entry_macro_progress_every,
            ):
                report_dir = (
                    resolve_base_dir(run_root, ROOT)
                    / "analysis"
                    / f"macro_growth_cumulative_ep{ep}"
                )
                try:
                    report = generate_entry_macro_progress_report(
                        resolve_base_dir(run_root, ROOT),
                        report_dir,
                        latest_episode=ep,
                        growth_start_time=args.entry_macro_growth_start_time,
                    )
                    print(
                        f"[Episode {ep}] cumulative entry/macro plots written to "
                        f"{report_dir}; episodes={report['episodes']}"
                    )
                except Exception as exc:
                    failure = {
                        "episode": ep,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "report_dir": str(report_dir),
                    }
                    failure_path = (
                        resolve_base_dir(run_root, ROOT)
                        / "analysis"
                        / f"macro_growth_plot_failure_ep{ep}.json"
                    )
                    failure_path.parent.mkdir(parents=True, exist_ok=True)
                    failure_path.write_text(
                        json.dumps(failure, indent=2),
                        encoding="utf-8",
                    )
                    print(
                        f"WARNING: cumulative entry/macro plotting failed for "
                        f"Episode {ep}: {exc}. Details: {failure_path}"
                    )
        if bool(getattr(hyperparams, "save_episode_diagnostics", False)):
            canonical_diagnostic_reference = save_episode_diagnostics(
                run_root=resolve_base_dir(run_root, ROOT),
                episode=ep,
                model=models["policy_value"],
                firm_df=episode.df,
                module_summary=ep_summary,
                canonical_reference=canonical_diagnostic_reference,
            )
        if bool(
            getattr(hyperparams, "save_intermediate_stage_checkpoints", False)
        ):
            stage_dir = (
                resolve_base_dir(run_root, ROOT)
                / "episode_diagnostics"
                / f"ep_{ep:03d}"
            )
            stage_dir.mkdir(parents=True, exist_ok=True)
            for stage_name, state in episode._intermediate_stage_states.items():
                torch.save(
                    {
                        "episode": ep,
                        "stage": stage_name,
                        "models": {"policy_value": state},
                    },
                    stage_dir / f"{stage_name}.pt",
                )

        # 裸 state_dict 无法区分 direct-Q 与 legacy `b * q_unit`（q-head shape 相同），
        # 因此同时落盘 metadata/（hyperparams / config snapshot / model spec）与 combined ckpt。
        saved_paths = save_models(
            models,
            ep,
            resolve_base_dir(run_root, ROOT),
            hyperparams=hyperparams,
            extra_models={"firm_target": episode.firm_target},
            extra_metadata={
                "episode_mode": episode_mode,
                "warmstart": warmstart_provenance,
                # P 专用 Collocation Augmentation 的采样方法标记：这是对
                # Bellman 状态空间的反事实 Parent η 覆盖，不是对真实 η=1
                # 企业的 oversampling，也不是对真实 η 分布的重新估计。
                "p_eta_sampling_method": (
                    "counterfactual_parent_eta_balance"
                    if bool(
                        getattr(hyperparams, "pv_current_eta_balance_enabled", False)
                    )
                    else "natural_parent_eta"
                ),
            },
        )
        print(f"Checkpoint metadata written: {saved_paths}")

        parent_df = episode.df[episode.df["branch"] <= 0] if "branch" in episode.df.columns else episode.df
        ref_state = {
            "eta": 1.0,
            "i": parent_df["i"].median(),
            "x": parent_df["x"].median(),
            "hatcf": parent_df["Hatcf"].median(),
            "lnkf": parent_df["LnKF"].median(),
        }
        plot_surfaces(
            ep,
            models["policy_value"],
            models.get("sdf_fc1"),
            ref_state,
            device,
            resolve_base_dir(run_root, ROOT),
        )
        # Diagnostics-only additions: BARI threshold line + i*(b) curves.
        # Failures must never kill the training loop.
        try:
            plot_bari_istar_heatmap(
                ep,
                models["policy_value"],
                models.get("sdf_fc1"),
                ref_state,
                device,
                resolve_base_dir(run_root, ROOT),
            )
        except Exception as exc:
            print(f"WARNING: bari/istar heatmap plotting failed for Episode {ep}: {exc}")
        try:
            plot_i_star_b(
                ep,
                models["policy_value"],
                ref_state,
                device,
                resolve_base_dir(run_root, ROOT),
            )
        except Exception as exc:
            print(f"WARNING: i*(b) plotting failed for Episode {ep}: {exc}")
        plot_distributions(
            ep,
            episode.df,
            models["policy_value"],
            device,
            resolve_base_dir(run_root, ROOT),
            df_macro=episode.df_macro,
        )
        plot_macro_series(ep, episode.df_macro, resolve_base_dir(run_root, ROOT))

        summaries.append({
            "episode": ep,
            "episode_mode": episode_mode,
            "sdf_wealth_loss_mode": hyperparams.sdf_wealth_loss_mode,
            "pv_bp_training_mode": hyperparams.pv_bp_training_mode,
            "pv_bp_head_training_enabled": hyperparams.pv_bp_head_training_enabled,
            "simulation_bp_action_source": hyperparams.simulation_bp_action_source,
            "pv_training_flow": hyperparams.pv_training_flow,
            "firm_target_update": hyperparams.firm_target_update,
            "module_summaries": ep_summary,
            "gpu_memory": summary.get("gpu_memory", {})
        })
        print(
            f"[Episode {ep}] mode={episode_mode} "
            f"batch_size={hyperparams.batch_size} "
            f"pv_batch_size={hyperparams.pv_batch_size} "
            f"sdf_fc1_batch_size={hyperparams.sdf_fc1_batch_size} "
            f"n_paths={data_kwargs['n_paths']} "
            f"group_size={data_kwargs['group_size']} "
            f"horizon={hyperparams.simulate_horizon} "
            f"sdf_wealth_loss_mode={hyperparams.sdf_wealth_loss_mode} "
            f"pv_bp_training_mode={hyperparams.pv_bp_training_mode} "
            f"pv_bp_head_training_enabled={hyperparams.pv_bp_head_training_enabled} "
            f"simulation_bp_action_source={hyperparams.simulation_bp_action_source} "
            f"pv_training_flow={hyperparams.pv_training_flow} "
            f"firm_target_update={hyperparams.firm_target_update}"
        )
        log_gpu_stats(f"[Episode {ep}]", device)
        print(f"Episode {ep} ({episode_mode}) done: {ep_summary}")

    if failure_report is None:
        print("All episodes done.")
    else:
        print("Training stopped safely after numerical stage failure.")
    print(summaries)

    # Save GPU memory monitoring results to JSON
    gpu_monitor = get_monitor()
    if gpu_monitor:
        gpu_json_path = resolve_base_dir(run_root, ROOT) / "gpu_memory_monitoring.json"
        gpu_monitor.save_to_json(gpu_json_path)
        print(f"GPU memory monitoring saved to: {gpu_json_path}")
        reset_monitor()

    if failure_report is not None:
        return

    final_sim_kwargs = episode._prepare_simulation_kwargs({})
    final_sim = SimulateTS(
        models=models,
        config=Config,
        n_paths=hyperparams.n_paths,
        group_size=simulate_group_size,
        branch_num=Config.BRANCH_NUM,
        horizon=hyperparams.simulate_horizon,
        device=device,
        **final_sim_kwargs,
    )
    df_firm_sim, df_macro_sim = final_sim.simulate()
    out_dir = resolve_base_dir(run_root, ROOT) / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_firm_sim.to_pickle(out_dir / "final_simulate_firm.pkl")
    df_macro_sim.to_pickle(out_dir / "final_simulate_macro.pkl")
    plot_macro_series(-1, df_macro_sim, resolve_base_dir(run_root, ROOT))
    plot_firm_b_window_distribution(df_firm_sim, resolve_base_dir(run_root, ROOT))


if __name__ == "__main__":
    main()
