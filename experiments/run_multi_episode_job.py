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
    ensure_dirs,
    save_models,
    save_stage_df,
    plot_surfaces,
    plot_bp_diagnostic_curves,
    plot_distributions,
    plot_macro_series,
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
        "--q-only-ablation",
        action="store_true",
        help="Run a Q-only ablation: only train policy_value, and keep the whole run inside q-only stage.",
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
    if args.q_only_ablation:
        q_only_epochs = int(hyperparams.epochs)
        hyperparams.q_pretrain_epochs = q_only_epochs
        hyperparams.q_warmstart_epochs = q_only_epochs
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

        train_modules = ["policy_value"] if args.q_only_ablation else ["sdf_fc1", "policy_value"]
        if args.enable_fc2 and not args.q_only_ablation:
            train_modules.append("fc2")
        simulate_kwargs = {
            "horizon_mode1": 1,
            "horizon": hyperparams.simulate_horizon,
        }
        summary = episode.run_episode(
            n_epochs=hyperparams.epochs,
            batch_size=hyperparams.batch_size,
            log_interval=50,
            train_modules=train_modules,
            simulate_kwargs=simulate_kwargs,
            episode_mode=episode_mode,
            **data_kwargs,
        )
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
        if hyperparams.bp_diag_enabled and (
            ep % max(1, hyperparams.bp_diag_every_n_episodes) == 0 or ep == args.n_episodes - 1
        ):
            plot_bp_diagnostic_curves(
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
            "module_summaries": ep_summary,
            "gpu_memory": summary.get("gpu_memory", {})
        })
        print(
            f"[Episode {ep}] mode={episode_mode} "
            f"batch_size={hyperparams.batch_size} "
            f"n_paths={data_kwargs['n_paths']} "
            f"group_size={data_kwargs['group_size']} "
            f"horizon={hyperparams.simulate_horizon} "
            f"q_only_ablation={int(args.q_only_ablation)}"
        )
        log_gpu_stats(f"[Episode {ep}]", device)
        print(f"Episode {ep} ({episode_mode}) done: {ep_summary}")

    print("All episodes done.")
    print(summaries)

    # Save GPU memory monitoring results to JSON
    gpu_monitor = get_monitor()
    if gpu_monitor:
        gpu_json_path = resolve_base_dir(run_root, ROOT) / "gpu_memory_monitoring.json"
        gpu_monitor.save_to_json(gpu_json_path)
        print(f"GPU memory monitoring saved to: {gpu_json_path}")
        reset_monitor()

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
