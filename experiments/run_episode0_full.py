"""
Run a minimal Episode 0 training on sample data, save artifacts, and
plot policy/value surfaces plus M and bp distributions.

Outputs:
- checkpoints/episode0_sdf_fc1.pt
- checkpoints/episode0_policy_value.pt
- checkpoints/episode0_fc2.pt (if trained)
- data/outputs/episode0_stage_sdf1.pkl
- data/outputs/episode0_stage_pv.pkl
- data/outputs/episode0_stage_sdf2.pkl
- data/outputs/episode0_stage_fc2.pkl
- data/outputs/episode0_stage_fc2_macro.pkl (if available)
- experiments/figs/{p0,pi,bp,q}_surface.png
- experiments/figs/{m,bp}_hist.png
"""

import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config, HyperParams  # noqa: E402
from models import SDFFC1Combined, PolicyValueModel, FC2Model  # noqa: E402
from training.episode import Episode  # noqa: E402
from data.simulate_ts import SimulateTS  # noqa: E402


def ensure_dirs():
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    (ROOT / "data" / "outputs").mkdir(parents=True, exist_ok=True)
    (ROOT / "experiments" / "figs").mkdir(parents=True, exist_ok=True)


def build_models(device: torch.device):
    models = {
        "sdf_fc1": SDFFC1Combined(
            sdf_input_dim=Config.FC1_INPUT_DIM,
            fc1_input_dim=Config.FC1_INPUT_DIM,
            sdf_hidden_dims=Config.SDF_HIDDEN_DIMS,
            fc1_hidden_dims=Config.FC1_HIDDEN_DIMS,
            w_hidden_dims=Config.SDF_HIDDEN_DIMS,
        ).to(device),
        "policy_value": PolicyValueModel().to(device),
        "fc2": FC2Model(
            input_dim=Config.FC2_INPUT_DIM,
            hidden_dims=Config.FC2_HIDDEN_DIMS,
            quantile_num=Config.QUANTILE_NUM,
        ).to(device),
    }
    return models


def build_optimizers(models):
    opts = {}
    for name, model in models.items():
        opts[name] = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    return opts


def build_hyperparams():
    hp = HyperParams(
        n_samples=5000,
        n_paths=500,
        batch_size=256,
        epochs=2,
        simulate_horizon=5,
    )
    # Episode expects these fields
    hp.lr = 1e-3
    hp.min_lr = 1e-6
    hp.warmup_steps = 0
    hp.max_steps = 20_000
    hp.w_sdf = 1.0
    hp.w_p0 = 1.0
    hp.w_pi = 1.0
    hp.w_q = 1.0
    hp.w_fc2 = 1.0
    return hp


def save_models(models, episode_idx: int = 0):
    torch.save(models["sdf_fc1"].state_dict(), ROOT / "checkpoints" / f"episode{episode_idx}_sdf_fc1.pt")
    torch.save(models["policy_value"].state_dict(), ROOT / "checkpoints" / f"episode{episode_idx}_policy_value.pt")
    if models.get("fc2") is not None:
        torch.save(models["fc2"].state_dict(), ROOT / "checkpoints" / f"episode{episode_idx}_fc2.pt")


def save_stage_df(episode_idx: int, name: str, df_firm: pd.DataFrame = None, df_macro: pd.DataFrame = None, df_sdf: pd.DataFrame = None):
    out_dir = ROOT / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    target_df = df_firm if df_firm is not None else df_sdf
    if target_df is not None:
        target_df.to_pickle(out_dir / f"episode{episode_idx}_stage_{name}.pkl")
    if df_macro is not None:
        df_macro.to_pickle(out_dir / f"episode{episode_idx}_stage_{name}_macro.pkl")


def plot_surfaces(pv_model: PolicyValueModel, ref_state: dict, device: torch.device):
    """Plot P0/PI/bp/Q heatmaps (b as x, z as y) and 3D surfaces."""
    b_grid = torch.linspace(0, 1, 50, device=device)
    z_grid = torch.linspace(-1, 1, 50, device=device)
    B, Z = torch.meshgrid(b_grid, z_grid, indexing="ij")

    base = torch.stack(
        [
            B.reshape(-1),
            Z.reshape(-1),
            torch.full_like(B.reshape(-1), ref_state["eta"]),
            torch.full_like(B.reshape(-1), ref_state["i"]),
            torch.full_like(B.reshape(-1), ref_state["x"]),
            torch.full_like(B.reshape(-1), ref_state["hatcf"]),
            torch.full_like(B.reshape(-1), ref_state["lnkf"]),
        ],
        dim=1,
    )
    with torch.no_grad():
        out = pv_model(base)
        P0 = out.P0.reshape(B.shape).cpu().numpy()
        PI = out.PI.reshape(B.shape).cpu().numpy()
        bp = out.bp.reshape(B.shape).cpu().numpy()
        Q = out.Q.reshape(B.shape).cpu().numpy()

    figs_dir = ROOT / "experiments" / "figs"
    for name, arr in [("p0", P0), ("pi", PI), ("bp", bp), ("q", Q)]:
        # Heatmap with b as x-axis, z as y-axis
        plt.figure(figsize=(6, 4))
        cs = plt.contourf(B.cpu().numpy(), Z.cpu().numpy(), arr, levels=30, cmap="viridis")
        plt.colorbar(cs)
        plt.xlabel("b")
        plt.ylabel("z")
        plt.title(f"{name.upper()} heatmap")
        plt.tight_layout()
        plt.savefig(figs_dir / f"{name}_heatmap.png", dpi=150)
        plt.close()

        # 3D surface
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(B.cpu().numpy(), Z.cpu().numpy(), arr, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel("b")
        ax.set_ylabel("z")
        ax.set_zlabel(name.upper())
        ax.set_title(f"{name.upper()} surface")
        plt.tight_layout()
        plt.savefig(figs_dir / f"{name}_surface.png", dpi=150)
        plt.close()


def plot_distributions(df: pd.DataFrame, pv_model: PolicyValueModel, device: torch.device):
    figs_dir = ROOT / "experiments" / "figs"

    # M distribution
    if "M" in df.columns:
        plt.figure(figsize=(5, 3))
        df["M"].dropna().hist(bins=40)
        plt.title("M distribution")
        plt.xlabel("M")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(figs_dir / "m_hist.png", dpi=150)
        plt.close()

    # bp distribution from model on parent rows (branch == 0 or -1)
    if "branch" in df.columns:
        parent_df = df[df["branch"] <= 0].copy()
    else:
        parent_df = df

    cols = ["b", "z", "ETA", "i", "x", "Hatcf", "LnKF"]
    X = torch.tensor(parent_df[cols].values, device=device, dtype=torch.float32)
    with torch.no_grad():
        bp = pv_model(X).bp.cpu().numpy()

    plt.figure(figsize=(5, 3))
    plt.hist(bp, bins=40)
    plt.title("bp distribution (parent states)")
    plt.xlabel("bp")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(figs_dir / "bp_hist.png", dpi=150)
    plt.close()


def main():
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Config.DEVICE = device

    models = build_models(device)
    optimizers = build_optimizers(models)
    hyperparams = build_hyperparams()

    n_episodes = 3  # 可以按需调整
    all_summaries = []

    for ep in range(n_episodes):
        episode = Episode(
            models=models,
            optimizers=optimizers,
            config=Config,
            hyperparams=hyperparams,
            device=device,
            episode_id=ep,
        )

        # Stage 1: train SDF/FC1
        summary_sdf1 = episode.run_episode(
            n_epochs=20,
            batch_size=hyperparams.batch_size,
            log_interval=50,
            n_samples=hyperparams.n_samples,
            n_paths=hyperparams.n_paths,
            group_size=2,
            n_branches=Config.BRANCH_NUM,
            train_modules=["sdf_fc1"],
            simulate_kwargs=None,
        )
        save_stage_df(ep, "sdf1", episode.df, episode.df_macro, episode.df_sdf)

        # Stage 2: train Policy/Value using updated SDF/FC1
        summary_pv = episode.run_episode(
            n_epochs=100,
            batch_size=hyperparams.batch_size,
            log_interval=50,
            n_samples=hyperparams.n_samples,
            n_paths=hyperparams.n_paths,
            group_size=2,
            n_branches=Config.BRANCH_NUM,
            train_modules=["policy_value"],
            simulate_kwargs=None,
        )
        save_stage_df(ep, "pv", episode.df, episode.df_macro)

        # Stage 3: re-train SDF/FC1
        summary_sdf2 = episode.run_episode(
            n_epochs=20,
            batch_size=hyperparams.batch_size,
            log_interval=50,
            n_samples=hyperparams.n_samples,
            n_paths=hyperparams.n_paths,
            group_size=2,
            n_branches=Config.BRANCH_NUM,
            train_modules=["sdf_fc1"],
            simulate_kwargs=None,
        )
        save_stage_df(ep, "sdf2", episode.df, episode.df_macro, episode.df_sdf)

        # Stage 4: train FC2 using fresh SimulateTS data (uses current sdf_fc1 + policy_value)
        sim = SimulateTS(
            models=models,
            config=Config,
            n_paths=min(50, hyperparams.n_paths),
            group_size=Config.SIMULATE_GROUP_SIZE,
            branch_num=Config.BRANCH_NUM,
            horizon=2,
            device=device,
        )
        df_firm_fc2, df_macro_fc2 = sim.simulate()
        episode.df = df_firm_fc2
        episode.df_macro = df_macro_fc2
        fc2_batches = episode._create_fc2_batches(df_firm_fc2, batch_size=hyperparams.batch_size, n_branches=Config.BRANCH_NUM)
        if fc2_batches:
            summary_fc2 = episode._run_batches(fc2_batches, n_epochs=20, log_interval=5, train_modules=["fc2"], desc_prefix="FC2 ")
        else:
            summary_fc2 = {}
        save_stage_df(ep, "fc2", df_firm_fc2, df_macro_fc2)

        # Save models and final data per episode
        save_models(models, ep)
        save_stage_df(ep, "final", episode.df, episode.df_macro)

        # Reference state for surfaces: use medians from parent rows of latest df
        parent_df = episode.df[episode.df["branch"] <= 0]
        ref_state = {
            "eta": 1.0,
            "i": parent_df["i"].median(),
            "x": parent_df["x"].median(),
            "hatcf": parent_df["Hatcf"].median(),
            "lnkf": parent_df["LnKF"].median(),
        }
        plot_surfaces(models["policy_value"], ref_state, device)
        plot_distributions(episode.df, models["policy_value"], device)

        all_summaries.append(
            {
                "sdf1": summary_sdf1.get("module_summaries", summary_sdf1),
                "pv": summary_pv.get("module_summaries", summary_pv),
                "sdf2": summary_sdf2.get("module_summaries", summary_sdf2),
                "fc2": summary_fc2.get("module_summaries", summary_fc2),
            }
        )

    print("Staged training finished. Summaries:")
    print(all_summaries)
    print("Artifacts saved to checkpoints/, data/outputs/, experiments/figs/.")


if __name__ == "__main__":
    main()
