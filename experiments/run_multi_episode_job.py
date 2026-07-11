"""
CLI-friendly multi-episode runner for Slurm/sbatch.

This mirrors the notebook `run_multi_episode.ipynb` setup and the existing
`run_multi_episode.py`, but adds CLI switches for quick tests and explicit run
root selection so it can be invoked from a Slurm batch script.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import json
import warnings

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
print(f"Added {ROOT} to sys.path for imports")
from config import Config  # noqa: E402
from training.episode import Episode, NumericalStageFailure  # noqa: E402
from data.simulate_ts import SimulateTS  # noqa: E402
from experiments.run_utils import (  # noqa: E402
    resolve_base_dir,
    build_models,
    build_optimizers,
    build_hyperparams,
    ensure_dirs,
    save_models,
    save_stage_df,
    plot_surfaces,
    plot_distributions,
    plot_macro_series,
    plot_firm_b_window_distribution,
)
from utils.gpu_monitor import get_monitor, reset_monitor


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
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per stage")
    parser.add_argument("--n-paths", type=int, default=None, help="Override n_paths for data")
    parser.add_argument("--post0-n-paths", type=int, default=None, help="Override n_paths for episodes > 0; default keeps full n_paths")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--simulate-group-size", type=int, default=None, help="Override firm count per simulated path for episodes > 0")
    parser.add_argument("--simulate-horizon", type=int, default=None, help="Override simulate horizon")
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
        "--sdf-recursive-only-epochs",
        type=int,
        default=None,
        help="Override SDF_RECURSIVE_ONLY epochs; use 0 to disable the recursive stage",
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
    parser.add_argument("--bp-grid-margin-scale", type=float, default=None, help="Top-two value margin scale for policy target confidence")
    parser.add_argument("--bp-grid-parent-chunk-size", type=int, default=None, help="Parent chunk size for bp-grid teacher")
    parser.add_argument("--bp-grid-candidate-chunk-size", type=int, default=None, help="Candidate bp chunk size for bp-grid teacher")
    parser.add_argument("--bp-grid-max-expanded-states", type=int, default=None, help="Max B*J states per bp-grid teacher forward chunk")
    parser.add_argument("--pv-target-grid-val-fraction", type=float, default=None, help="Tail batch fraction reserved for target-grid policy validation")
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
        choices=["soft", "hard", "epoch_hard", "epoch_soft", "none"],
        help="Firm target update schedule",
    )
    parser.add_argument(
        "--modeb-resimulate-after-pv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Treatment B: resimulate Mode B data after Policy/Value training and before SDF/FC1 Stage 2",
    )
    return parser.parse_args()


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
        hyperparams.batch_size = args.batch_size
    if args.epochs is not None:
        hyperparams.epochs = args.epochs
    if args.simulate_horizon is not None:
        hyperparams.simulate_horizon = args.simulate_horizon
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
    if args.sdf_recursive_only_epochs is not None:
        hyperparams.sdf_recursive_only_epochs = int(args.sdf_recursive_only_epochs)
    if args.fc1_jacobian_penalty_interval is not None:
        hyperparams.fc1_jacobian_penalty_interval = args.fc1_jacobian_penalty_interval
    if args.pv_bp_training_mode is not None:
        hyperparams.pv_bp_training_mode = args.pv_bp_training_mode
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
    if args.bp_grid_margin_scale is not None:
        hyperparams.bp_grid_margin_scale = args.bp_grid_margin_scale
    if args.bp_grid_parent_chunk_size is not None:
        hyperparams.bp_grid_parent_chunk_size = args.bp_grid_parent_chunk_size
    if args.bp_grid_candidate_chunk_size is not None:
        hyperparams.bp_grid_candidate_chunk_size = args.bp_grid_candidate_chunk_size
    if args.bp_grid_max_expanded_states is not None:
        hyperparams.bp_grid_max_expanded_states = args.bp_grid_max_expanded_states
    if args.bp_grid_confidence_relative is not None:
        hyperparams.bp_grid_confidence_relative = bool(args.bp_grid_confidence_relative)
    if args.pv_target_grid_val_fraction is not None:
        hyperparams.pv_target_grid_val_fraction = args.pv_target_grid_val_fraction
    if args.firm_target_update is not None:
        hyperparams.firm_target_update = args.firm_target_update
    hyperparams.policy_value_bellman_only = args.ablation_mode == "bellman_only"
    hyperparams.pv_fixed_sdf = args.ablation_mode == "fixed_sdf"
    hyperparams.pv_fixed_policy = args.ablation_mode == "fixed_policy"
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


def main():
    args = parse_args()
    run_root = make_run_root(args.run_root)
    ensure_dirs(run_root)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    Config.DEVICE = device

    hyperparams = configure_hyperparams(args)
    models = build_models(device)
    optimizers = build_optimizers(models, hyperparams)
    post0_n_paths = args.post0_n_paths if args.post0_n_paths is not None else hyperparams.n_paths
    simulate_group_size = args.simulate_group_size if args.simulate_group_size is not None else Config.SIMULATE_GROUP_SIZE

    # Initialize GPU monitor
    gpu_monitor = get_monitor(device, log_interval=10)
    print(f"GPU Monitor initialized: {device}")

    summaries = []
    failure_report = None
    episode = Episode(
            models=models,
            optimizers=optimizers,
            config=Config,
            hyperparams=hyperparams,
            device=device,
            episode_id=0,
            gpu_monitor=gpu_monitor,
        )
    for ep in range(args.n_episodes):
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
            summary = episode.run_episode(
                n_epochs=hyperparams.epochs,
                batch_size=hyperparams.batch_size,
                log_interval=50,
                train_modules=train_modules,
                simulate_kwargs=simulate_kwargs,
                episode_mode=episode_mode,
                **data_kwargs,
            )
        except NumericalStageFailure as exc:
            failure_report = {
                "failed_episode": ep,
                "episode_mode": episode_mode,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "ablation_mode": args.ablation_mode,
                "gate_context": getattr(episode, "_last_policy_value_gate_context", {}),
                "last_policy_value_stage_summary": getattr(episode, "_last_policy_value_stage_summary", {}),
                "completed_summaries": summaries,
                "run_parameters": {
                    "n_episodes": args.n_episodes,
                    "epochs": hyperparams.epochs,
                    "n_paths": data_kwargs["n_paths"],
                    "batch_size": hyperparams.batch_size,
                    "simulate_group_size": simulate_group_size,
                    "simulate_horizon": hyperparams.simulate_horizon,
                    "post0_mode": args.post0_mode,
                    "sdf_wealth_loss_mode": hyperparams.sdf_wealth_loss_mode,
                    "pv_bp_training_mode": hyperparams.pv_bp_training_mode,
                    "firm_target_update": hyperparams.firm_target_update,
                    "modeb_resimulate_after_pv": args.modeb_resimulate_after_pv,
                    "max_firm_train_units": hyperparams.max_firm_train_units,
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

        save_models(models, ep, resolve_base_dir(run_root, ROOT))

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
            "episode_mode": episode_mode,
            "sdf_wealth_loss_mode": hyperparams.sdf_wealth_loss_mode,
            "pv_bp_training_mode": hyperparams.pv_bp_training_mode,
            "firm_target_update": hyperparams.firm_target_update,
            "module_summaries": ep_summary,
            "gpu_memory": summary.get("gpu_memory", {})
        })
        print(
            f"[Episode {ep}] mode={episode_mode} "
            f"batch_size={hyperparams.batch_size} "
            f"n_paths={data_kwargs['n_paths']} "
            f"group_size={data_kwargs['group_size']} "
            f"horizon={hyperparams.simulate_horizon} "
            f"sdf_wealth_loss_mode={hyperparams.sdf_wealth_loss_mode} "
            f"pv_bp_training_mode={hyperparams.pv_bp_training_mode} "
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

    final_sim = SimulateTS(
        models=models,
        config=Config,
        n_paths=hyperparams.n_paths,
        group_size=simulate_group_size,
        branch_num=Config.BRANCH_NUM,
        horizon=hyperparams.simulate_horizon,
        device=device,
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
