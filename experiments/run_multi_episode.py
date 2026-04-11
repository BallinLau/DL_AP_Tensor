"""
Multi-episode staged training runner.

Episode 0 uses Sample data.
Episode >=1 uses SimulateTS (generated with weights from previous episode).
Within each episode we call Episode.run_episode once per stage sequence:
    sdf_fc1 -> policy_value -> sdf_fc1 -> fc2
Models/optimizers are continuous across episodes.

Each episode saves:
- checkpoints/ep{episode}_{model}.pt
- data/outputs/ep{episode}_stage_{name}.pkl (+_macro if available)
- experiments/figs/ep{episode}_{p0,pi,bp,q}_{heatmap,surface}.png
- experiments/figs/ep{episode}_{m,bp}_hist.png
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config, HyperParams  # noqa: E402
from training.episode import Episode  # noqa: E402
from data.simulate_ts import SimulateTS  # noqa: E402
from experiments.run_utils import (  # noqa: E402
    resolve_base_dir,
    build_models,
    build_optimizers,
    build_hyperparams,
    build_outer_drift_ref_state,
    compute_macro_moment_summary,
    compute_macro_outer_drift,
    compute_policy_surface_snapshot,
    compute_policy_surface_drift,
    ensure_dirs,
    save_models,
    save_stage_df,
    plot_surfaces,
    plot_distributions,
    plot_fc2_fit_diagnostics,
    plot_fc2_consumption_distribution,
    plot_fc2_episode_metrics,
    plot_outer_drift,
    plot_firm_b_window_distribution,
)

# Per-run cache directory (set in main). Lives outside code tree at ROOT.parent / cachedir.
RUN_ROOT: Path | None = None


def get_base_dir() -> Path:
    """Return the run-specific base dir if set, otherwise project root."""
    return resolve_base_dir(RUN_ROOT, ROOT)



def plot_stage_losses(all_summaries, figs_dir: Path):
    """
    Plot loss trajectories per stage across episodes, and per-episode bar charts.
    all_summaries: list of ep_summary dicts (one per episode)
    """
    if not all_summaries:
        return
    stage_loss_history = {}
    for ep_idx, ep_summary in enumerate(all_summaries):
        for stage, summary in ep_summary.items():
            if not isinstance(summary, dict):
                continue
            losses = summary.get("final_losses", summary)
            if not isinstance(losses, dict):
                continue
            for k, v in losses.items():
                if not isinstance(v, (int, float, np.floating)):
                    continue
                stage_loss_history.setdefault(stage, {}).setdefault(k, []).append(v)

    figs_dir.mkdir(parents=True, exist_ok=True)
    eps = list(range(len(all_summaries)))

    # 跨 episode 轨迹
    for stage, loss_dict in stage_loss_history.items():
        for loss_name, values in loss_dict.items():
            padded = values + [float("nan")] * (len(eps) - len(values))
            plt.figure(figsize=(5, 3))
            plt.plot(eps, padded, marker="o")
            plt.xlabel("episode")
            plt.ylabel(loss_name)
            plt.title(f"{stage} - {loss_name}")
            plt.tight_layout()
            plt.savefig(figs_dir / f"loss_{stage}_{loss_name}.png", dpi=150)
            plt.close()

    # 每个 episode 单独画一张（按 stage 汇总）
    for ep_idx, ep_summary in enumerate(all_summaries):
        for stage, summary in ep_summary.items():
            if not isinstance(summary, dict):
                continue
            losses = summary.get("final_losses", summary)
            if not isinstance(losses, dict) or not losses:
                continue
            names = list(losses.keys())
            vals = [losses[n] for n in names]
            plt.figure(figsize=(6, 4))
            plt.bar(range(len(names)), vals)
            plt.xticks(range(len(names)), names, rotation=45, ha="right")
            plt.ylabel("loss")
            plt.title(f"episode {ep_idx} - {stage}")
            plt.tight_layout()
            plt.savefig(figs_dir / f"loss_ep{ep_idx}_{stage}.png", dpi=150)
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Multi-episode staged training runner")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of episodes")
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
        "--fc1-forecast-only-ablation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Train sdf_fc1 in forecast-only identification mode: disable Euler/moment/anchor and update only FC1 forecast heads.",
    )
    parser.add_argument(
        "--fc2-as-main-macro-state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="When enabled, use FC2 as the node-level main macro state generator on the tensor recursive path.",
    )
    parser.add_argument(
        "--fc2-supervised-pretrain-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run only FC2 supervised pretrain and skip FC2 closure finetune.",
    )
    args = parser.parse_args()

    global RUN_ROOT
    RUN_ROOT = ROOT.parent / "cachedir" / datetime.now().strftime("%Y%m%d_%H%M")
    ensure_dirs(RUN_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Config.DEVICE = device

    n_episodes = args.n_episodes  # episode 0 + simulate episodes
    hyperparams = build_hyperparams()
    if args.fc1_forecast_only_ablation is not None:
        hyperparams.fc1_forecast_only_ablation = bool(args.fc1_forecast_only_ablation)
    if args.fc2_as_main_macro_state is not None:
        hyperparams.fc2_as_main_macro_state = bool(args.fc2_as_main_macro_state)
    if args.fc2_supervised_pretrain_only is not None:
        hyperparams.fc2_supervised_pretrain_only = bool(args.fc2_supervised_pretrain_only)
    models = build_models(device)
    optimizers = build_optimizers(models, hyperparams)

    summaries = []
    sample_group_size = Config.GROUP_SIZE
    simulate_group_size = Config.SIMULATE_GROUP_SIZE
    prev_macro_source_df = None
    prev_macro_df_for_drift = None
    prev_surface_snapshot = None
    drift_ref_state = build_outer_drift_ref_state()
    for ep in range(n_episodes):
        episode = Episode(
            models=models,
            optimizers=optimizers,
            config=Config,
            hyperparams=hyperparams,
            device=device,
            episode_id=ep,
        )
        episode.macro_source_df = prev_macro_source_df

        # data settings per episode
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
            "n_paths": hyperparams.n_paths if ep == 0 else min(100, hyperparams.n_paths),
            "sample_group_size": sample_group_size,
            "simulate_group_size": simulate_group_size,
            "n_branches": Config.BRANCH_NUM,
        }

        train_modules = ["sdf_fc1", "policy_value"]
        if args.enable_fc2:
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
        save_stage_df(ep, episode_mode, get_base_dir(), episode.df, episode.df_macro, episode.df_sdf)

        # save models after each episode
        save_models(models, ep, get_base_dir())

        # plots for this episode using latest df
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
            get_base_dir(),
        )
        plot_distributions(
            ep,
            episode.df,
            models["policy_value"],
            device,
            get_base_dir(),
            df_macro=episode.df_macro,
        )
        plot_fc2_fit_diagnostics(
            ep,
            getattr(episode, "_latest_fc2_fit_df", None),
            getattr(episode, "_latest_fc2_outer_df", None),
            get_base_dir(),
        )
        plot_fc2_consumption_distribution(
            ep,
            getattr(episode, "_latest_fc2_current_macro_panel_df", None),
            getattr(episode, "_latest_fc2_outer_macro_panel_df", None),
            get_base_dir(),
        )

        summaries.append(ep_summary)
        plot_fc2_episode_metrics(summaries, get_base_dir() / "experiments" / "figs")
        plot_outer_drift(summaries, get_base_dir() / "experiments" / "figs")
        print(f"Episode {ep} ({episode_mode}) done: {ep_summary}")

    print("All episodes done.")
    print(summaries)
    plot_stage_losses(summaries, get_base_dir() / "experiments" / "figs")
    plot_fc2_episode_metrics(summaries, get_base_dir() / "experiments" / "figs")
    plot_outer_drift(summaries, get_base_dir() / "experiments" / "figs")

    # 额外模拟一次使用最终模型的数据，并导出以便宏观画图
    final_sim = SimulateTS(
        models=models,
        config=Config,
        n_paths=hyperparams.n_paths,
        group_size=Config.SIMULATE_GROUP_SIZE,
        branch_num=Config.BRANCH_NUM,
        horizon=hyperparams.simulate_horizon,
        device=device,
    )
    df_firm_sim, df_macro_sim = final_sim.simulate()
    out_dir = get_base_dir() / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_firm_sim.to_pickle(out_dir / "final_simulate_firm.pkl")
    df_macro_sim.to_pickle(out_dir / "final_simulate_macro.pkl")
    plot_firm_b_window_distribution(df_firm_sim, get_base_dir())


if __name__ == "__main__":
    main()
