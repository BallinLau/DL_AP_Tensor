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

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
print(f"Added {ROOT} to sys.path for imports")
from config import Config  # noqa: E402
from training.episode import Episode  # noqa: E402
from data.simulate_ts import SimulateTS  # noqa: E402
from experiments.run_utils import (  # noqa: E402
    resolve_base_dir,
    build_models,
    build_optimizers,
    build_hyperparams,
    build_policy_ref_state,
    build_outer_drift_ref_state,
    compute_macro_moment_summary,
    compute_macro_outer_drift,
    compute_policy_surface_snapshot,
    compute_policy_surface_drift,
    ensure_dirs,
    save_models,
    save_stage_df,
    plot_surfaces,
    plot_bp_diagnostic_curves,
    plot_distributions,
    plot_macro_series,
    plot_outer_drift,
    plot_firm_b_window_distribution,
)
from utils.gpu_monitor import get_monitor, reset_monitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged multi-episode training")
    parser.add_argument("--run-root", type=Path, default=None, help="Override output root; default cachedir/timestamp")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per stage")
    parser.add_argument("--n-paths", type=int, default=None, help="Override n_paths for data")
    parser.add_argument("--post0-n-paths", type=int, default=None, help="Override n_paths for episodes > 0; default keeps full n_paths")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--sample-group-size", type=int, default=None, help="Override firm count per sampled path (Sample data source)")
    parser.add_argument("--simulate-group-size", type=int, default=None, help="Override firm count per simulated path (SimulateTS data source)")
    parser.add_argument("--simulate-horizon", type=int, default=None, help="Override simulate horizon")
    parser.add_argument(
        "--fc1-forecast-only-ablation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Train sdf_fc1 in forecast-only identification mode: disable Euler/moment/anchor and update only FC1 forecast heads.",
    )
    parser.add_argument(
        "--final-sim",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to run the extra final SimulateTS export after training. Defaults to off for q-only / q-joint ablations.",
    )
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cuda:0 or cpu")
    parser.add_argument("--quick-test", action="store_true", help="Shrink workload for smoke tests (n_paths=10, epochs=20, horizon=10)")
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
        choices=["mode0", "modea", "modeb", "alternate"],
        help="Episode>0 mode: mode0, modea, modeb, or alternate. Episode 0 is always mode0.",
    )
    parser.add_argument(
        "--alternate-start",
        type=str.lower,
        default="modea",
        choices=["modea", "modeb"],
        help="When --post0-mode=alternate, choose which mode starts at episode 1",
    )
    parser.add_argument(
        "--q-only-ablation",
        action="store_true",
        help="Run a Q-only ablation: only train policy_value, and keep the whole run inside q-only stage.",
    )
    parser.add_argument(
        "--q-joint-continuation-ablation",
        action="store_true",
        help="Run policy_value only with staged Q training followed by staged PV/BP training, and export stage-end artifacts.",
    )
    parser.add_argument(
        "--q-stage-epochs",
        "--q-only-epochs",
        dest="q_stage_epochs",
        type=int,
        default=None,
        help="Override Q-stage epochs for staged policy_value training.",
    )
    parser.add_argument(
        "--pvbp-stage-epochs",
        "--joint-epochs",
        dest="pvbp_stage_epochs",
        type=int,
        default=None,
        help="Override PV/BP-stage epochs for staged policy_value training.",
    )
    return parser.parse_args()


def configure_hyperparams(args: argparse.Namespace):
    if args.q_only_ablation and args.q_joint_continuation_ablation:
        raise ValueError("Cannot enable both --q-only-ablation and --q-joint-continuation-ablation")
    hyperparams = build_hyperparams()
    if args.quick_test:
        hyperparams.n_paths = 10
        hyperparams.epochs = 20
        hyperparams.simulate_horizon = 10
        hyperparams.batch_size = min(hyperparams.batch_size, 1024)
    if args.n_paths is not None:
        hyperparams.n_paths = args.n_paths
    if args.batch_size is not None:
        hyperparams.batch_size = args.batch_size
    if args.epochs is not None:
        hyperparams.epochs = args.epochs
    if args.simulate_horizon is not None:
        hyperparams.simulate_horizon = args.simulate_horizon
    if args.fc1_forecast_only_ablation is not None:
        hyperparams.fc1_forecast_only_ablation = bool(args.fc1_forecast_only_ablation)
    if args.q_only_ablation:
        q_only_epochs = max(100, int(hyperparams.epochs))
        hyperparams.q_stage_epochs = q_only_epochs
        hyperparams.pvbp_stage_epochs = 0
        hyperparams.q_pretrain_epochs = q_only_epochs
        hyperparams.q_warmstart_epochs = q_only_epochs
        hyperparams.bp_diag_enabled = True
        hyperparams.bp_diag_every_n_episodes = 1
        hyperparams.bp_diag_states = "safe"
    if args.q_joint_continuation_ablation:
        q_stage_epochs = int(args.q_stage_epochs if args.q_stage_epochs is not None else hyperparams.q_stage_epochs)
        pvbp_stage_epochs = int(args.pvbp_stage_epochs if args.pvbp_stage_epochs is not None else hyperparams.pvbp_stage_epochs)
        hyperparams.q_stage_epochs = max(100, q_stage_epochs)
        hyperparams.pvbp_stage_epochs = max(100, pvbp_stage_epochs)
        hyperparams.q_pretrain_epochs = hyperparams.q_stage_epochs
        hyperparams.q_warmstart_epochs = hyperparams.q_stage_epochs
        hyperparams.epochs = hyperparams.q_stage_epochs + hyperparams.pvbp_stage_epochs
        hyperparams.bp_diag_enabled = True
        hyperparams.bp_diag_every_n_episodes = 1
        hyperparams.bp_diag_states = "safe"
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
    return (ROOT / "cachedir" / datetime.now().strftime("%Y%m%d_%H%M")).resolve()


def materialize_episode_outputs(episode: Episode) -> None:
    if episode.df is None and episode.tensor_firm is not None:
        episode.df = episode._table_to_dataframe(episode.tensor_firm)
    if episode.df_macro is None and episode.tensor_macro is not None:
        episode.df_macro = episode._table_to_dataframe(episode.tensor_macro)
    if episode.df_sdf is None and episode.tensor_sdf is not None:
        episode.df_sdf = episode._table_to_dataframe(episode.tensor_sdf)


def make_json_safe(value):
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def export_policy_stage_artifacts(
    episode: Episode,
    ep: int,
    episode_mode: str,
    stage_tag: str,
    stage_summary: dict,
    run_root: Path,
    models: dict,
    hyperparams,
    device: torch.device,
    sdf_model,
):
    materialize_episode_outputs(episode)
    if episode.df is None or episode.df.empty:
        print(f"[Episode {ep}] skip stage export for {stage_tag}: empty firm DataFrame")
        return

    base_dir = resolve_base_dir(run_root, ROOT)
    ref_state = build_policy_ref_state(episode.df)
    save_models(models, ep, base_dir, tag=stage_tag)
    plot_surfaces(
        ep,
        models["policy_value"],
        sdf_model,
        ref_state,
        device,
        base_dir,
        tag=stage_tag,
    )
    if hyperparams.bp_diag_enabled:
        plot_bp_diagnostic_curves(
            ep,
            models["policy_value"],
            sdf_model,
            ref_state,
            device,
            base_dir,
            hyperparams=hyperparams,
            tag=stage_tag,
        )
    plot_distributions(
        ep,
        episode.df,
        models["policy_value"],
        device,
        base_dir,
        df_macro=episode.df_macro,
        tag=stage_tag,
    )

    out_dir = base_dir / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"ep{ep}_{stage_tag}_summary.json"
    payload = {
        "episode_id": ep,
        "episode_mode": episode_mode,
        "stage_tag": stage_tag,
        "stage_summary": make_json_safe(stage_summary),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    run_root = make_run_root(args.run_root)
    ensure_dirs(run_root)
    print(f"Run root: {run_root}")
    print(f"Checkpoints dir: {run_root / 'checkpoints'}")
    print(f"Outputs dir: {run_root / 'data' / 'outputs'}")
    print(f"Figures dir: {run_root / 'experiments' / 'figs'}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    Config.DEVICE = device

    hyperparams = configure_hyperparams(args)
    models = build_models(device)
    optimizers = build_optimizers(models, hyperparams)
    post0_n_paths = args.post0_n_paths if args.post0_n_paths is not None else hyperparams.n_paths
    sample_group_size = args.sample_group_size if args.sample_group_size is not None else Config.GROUP_SIZE
    simulate_group_size = args.simulate_group_size if args.simulate_group_size is not None else Config.SIMULATE_GROUP_SIZE
    run_final_sim = args.final_sim
    if run_final_sim is None:
        run_final_sim = not (args.q_only_ablation or args.q_joint_continuation_ablation)

    # Initialize GPU monitor
    gpu_monitor = get_monitor(device, log_interval=10)
    print(f"GPU Monitor initialized: {device}")
    print(f"Final simulation enabled: {int(bool(run_final_sim))}")

    summaries = []
    prev_macro_source_df = None
    prev_macro_df_for_drift = None
    prev_surface_snapshot = None
    drift_ref_state = build_outer_drift_ref_state()
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
        episode.macro_source_df = prev_macro_source_df
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

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

        data_kwargs = {
            "n_samples": hyperparams.n_samples,
            "n_paths": hyperparams.n_paths if ep == 0 else post0_n_paths,
            "sample_group_size": sample_group_size,
            "simulate_group_size": simulate_group_size,
            "n_branches": Config.BRANCH_NUM,
        }

        train_modules = ["policy_value"] if (args.q_only_ablation or args.q_joint_continuation_ablation) else ["sdf_fc1", "policy_value"]
        if args.enable_fc2 and not (args.q_only_ablation or args.q_joint_continuation_ablation):
            train_modules.append("fc2")
        simulate_kwargs = {
            "horizon_mode1": 1,
            "horizon": hyperparams.simulate_horizon,
        }
        stage_exports = []

        def on_policy_stage(stage_summary: dict):
            if not args.q_joint_continuation_ablation:
                return
            phase_to_tag = {
                "q_stage_end": "q_stage_end",
                "pvbp_stage_end": "pvbp_stage_end",
                "policy_end": "policy_end",
            }
            stage_tag = phase_to_tag.get(stage_summary.get("phase", "policy_stage"), str(stage_summary.get("phase", "policy_stage")))
            sdf_model_for_plots = None if args.q_joint_continuation_ablation else models.get("sdf_fc1")
            export_policy_stage_artifacts(
                episode=episode,
                ep=ep,
                episode_mode=episode_mode,
                stage_tag=stage_tag,
                stage_summary=stage_summary,
                run_root=run_root,
                models=models,
                hyperparams=hyperparams,
                device=device,
                sdf_model=sdf_model_for_plots,
            )
            stage_exports.append({
                "phase": stage_summary.get("phase"),
                "tag": stage_tag,
                "epoch": stage_summary.get("epoch"),
            })

        summary = episode.run_episode(
            n_epochs=hyperparams.epochs,
            batch_size=hyperparams.batch_size,
            log_interval=50,
            train_modules=train_modules,
            simulate_kwargs=simulate_kwargs,
            episode_mode=episode_mode,
            policy_stage_callback=on_policy_stage if args.q_joint_continuation_ablation else None,
            **data_kwargs,
        )
        ep_summary = summary.get("module_summaries", summary)
        if stage_exports:
            ep_summary["policy_stage_exports"] = stage_exports
        materialize_episode_outputs(episode)
        outer_drift = {}
        if episode.df_macro is not None and not episode.df_macro.empty:
            outer_drift.update(compute_macro_outer_drift(prev_macro_df_for_drift, episode.df_macro))
            outer_drift["current_macro_moments"] = compute_macro_moment_summary(episode.df_macro)
            prev_macro_source_df = episode.df_macro.copy()
            prev_macro_df_for_drift = episode.df_macro.copy()
        current_surface_snapshot = compute_policy_surface_snapshot(
            models["policy_value"],
            device=device,
            ref_state=drift_ref_state,
        )
        surface_drift = compute_policy_surface_drift(prev_surface_snapshot, current_surface_snapshot)
        if surface_drift:
            outer_drift["policy_surface_drift"] = surface_drift
        prev_surface_snapshot = current_surface_snapshot
        if outer_drift:
            ep_summary["outer_drift"] = outer_drift
        save_stage_df(ep, episode_mode, resolve_base_dir(run_root, ROOT), episode.df, episode.df_macro, episode.df_sdf)

        save_models(models, ep, resolve_base_dir(run_root, ROOT))

        ref_state = build_policy_ref_state(episode.df)
        sdf_model_for_plots = None if (args.q_only_ablation or args.q_joint_continuation_ablation) else models.get("sdf_fc1")
        plot_surfaces(
            ep,
            models["policy_value"],
            sdf_model_for_plots,
            ref_state,
            device,
            resolve_base_dir(run_root, ROOT),
        )
        if hyperparams.bp_diag_enabled and (
            ep % max(1, hyperparams.bp_diag_every_n_episodes) == 0 or ep == args.n_episodes - 1
        ):
            plot_bp_diagnostic_curves(
                ep,
                models["policy_value"],
                sdf_model_for_plots,
                ref_state,
                device,
                resolve_base_dir(run_root, ROOT),
                hyperparams=hyperparams,
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
            "module_summaries": ep_summary,
            "gpu_memory": summary.get("gpu_memory", {})
        })
        plot_outer_drift(summaries, resolve_base_dir(run_root, ROOT) / "experiments" / "figs")
        print(
            f"[Episode {ep}] mode={episode_mode} "
            f"batch_size={hyperparams.batch_size} "
            f"n_paths={data_kwargs['n_paths']} "
            f"sample_group_size={data_kwargs['sample_group_size']} "
            f"simulate_group_size={data_kwargs['simulate_group_size']} "
            f"horizon={hyperparams.simulate_horizon} "
            f"q_only_ablation={int(args.q_only_ablation)} "
            f"q_joint_continuation_ablation={int(args.q_joint_continuation_ablation)}"
        )
        log_gpu_stats(f"[Episode {ep}]", device)
        print(f"Episode {ep} ({episode_mode}) done: {ep_summary}")

    print("All episodes done.")
    print(summaries)
    plot_outer_drift(summaries, resolve_base_dir(run_root, ROOT) / "experiments" / "figs")

    # Save GPU memory monitoring results to JSON
    gpu_monitor = get_monitor()
    if gpu_monitor:
        gpu_json_path = resolve_base_dir(run_root, ROOT) / "gpu_memory_monitoring.json"
        gpu_monitor.save_to_json(gpu_json_path)
        print(f"GPU memory monitoring saved to: {gpu_json_path}")
        reset_monitor()

    if run_final_sim:
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
    else:
        print("Skipping final SimulateTS export for this run.")


if __name__ == "__main__":
    main()
