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
from losses import P0Loss, PILoss  # noqa: E402
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


def build_optimizers(models, hyperparams: HyperParams | None = None):
    hp = hyperparams or HyperParams()
    opts = {}
    for name, model in models.items():
        if name == "sdf_fc1":
            opts[name] = torch.optim.AdamW(
                [
                    {
                        "params": model.sdf_model.parameters(),
                        "lr": hp.sdf_lr,
                        "base_lr": hp.sdf_lr,
                        "weight_decay": hp.sdf_weight_decay,
                        "group_name": "sdf_core",
                    },
                    {
                        "params": model.value_model.parameters(),
                        "lr": hp.sdf_lr,
                        "base_lr": hp.sdf_lr,
                        "weight_decay": hp.sdf_weight_decay,
                        "group_name": "value",
                    },
                    {
                        "params": model.fc1_model.parameters(),
                        "lr": hp.fc1_lr,
                        "base_lr": hp.fc1_lr,
                        "weight_decay": hp.fc1_weight_decay,
                        "group_name": "fc1",
                    },
                ]
            )
        elif name == "policy_value":
            opts[name] = torch.optim.AdamW(
                model.parameters(),
                lr=hp.policy_lr,
                weight_decay=hp.policy_weight_decay,
            )
        elif name == "fc2":
            opts[name] = torch.optim.AdamW(
                model.parameters(),
                lr=hp.fc2_lr,
                weight_decay=hp.fc2_weight_decay,
            )
        else:
            opts[name] = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=1e-4)
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
    hp.fc1_recon_weight = 1.0
    hp.fc1_recursive_aux_training_enabled = False
    hp.fc1_rollout_diagnostic_enabled = True
    hp.fc1_forecast_recon_weight = 0.0
    hp.fc1_hatc_recon_weight = 1.0
    hp.fc1_lnk_recon_weight = 0.25
    hp.fc1_delta_penalty_weight = 0.0
    hp.fc1_delta_hatc_abs_max = 0.50
    hp.fc1_delta_lnk_abs_max = 0.30
    hp.fc1_jacobian_penalty_weight = 0.0
    hp.fc1_use_true_macro_state_in_stage2 = True
    hp.sdf_wealth_residual_mode = "normalized_ratio"
    hp.sdf_gate_residual_mode = "normalized_ratio"
    hp.sdf_normalized_logr_clip = 20.0
    hp.sdf_true_only_epochs = 5
    hp.sdf_recursive_only_epochs = 5
    hp.sdf_true_moment_weight = 5e-4
    hp.sdf_true_anchor_weight = 0.05
    hp.sdf_epoch_validation_enabled = True
    hp.sdf_restore_best_checkpoint = True
    hp.sdf_stop_when_gate_passes = True
    hp.sdf_required_consecutive_passes = 1
    hp.sdf_collapse_log_mean_error = 0.5
    hp.sdf_collapse_mean_ratio = 0.10
    hp.sdf_collapse_patience = 1
    hp.sdf_reset_optimizer_on_true_start = False
    hp.sdf_clear_optimizer_after_restore = True
    hp.stage_parameter_invariance_check_enabled = True
    hp.stage_epochwise_validation = True
    hp.sdf_collapse_lower_ratio = 0.1
    hp.sdf_collapse_upper_ratio = 10.0
    hp.fc1_sdf_preserve_ratio = 0.5
    hp.stage_min_improvement = 1e-4
    hp.stage_lr_decay_on_reject = 0.1
    hp.stage_max_retries = 1
    hp.sdf_score_t_weight = 0.05
    hp.sdf_score_t_cap = 20.0
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


def _compute_policy_diagnostic_surfaces(
    pv_model: PolicyValueModel,
    sdf_model: SDFFC1Combined | None,
    base: torch.Tensor,
):
    p0_loss = P0Loss()
    pi_loss = PILoss()

    out = pv_model(base)
    bp0 = out.bp0
    bpI = out.bpI

    child_p0_state = base.clone()
    child_p0_state[:, 0:1] = bp0
    child_pI_state = base.clone()
    child_pI_state[:, 0:1] = bpI

    out_p0_child = pv_model(child_p0_state)
    out_pI_child = pv_model(child_pI_state)

    x = base[:, 4:5]
    z = base[:, 1:2]
    b = base[:, 0:1]
    i = base[:, 3:4]
    eta = base[:, 2:3]

    cf0 = p0_loss.compute_cashflow_p0(x, z, b, out.Q, out_p0_child.Q, eta)
    cfi = pi_loss.compute_cashflow_pi(x, z, b, i, out.Q, out_pI_child.Q, eta)

    if sdf_model is not None:
        _, _, m_next, _, _ = sdf_model.forward_step(
            x_prev=base[:, 4:5],
            x_curr=base[:, 4:5],
            hatcf_prev=base[:, 5:6],
            lnkf_prev=base[:, 6:7],
            return_physical=True,
        )
    else:
        m_next = torch.ones_like(out.P0)

    cont0 = m_next * out_p0_child.P * (1.0 - out_p0_child.bar_z)
    contI = Config.G * m_next * out_pI_child.P * (1.0 - out_pI_child.bar_z)

    diagnostics = {
        "pidiff": out.PI - out.P0,
        "cfdiff": cfi - cf0,
        "contdiff": contI - cont0,
    }
    return out, diagnostics


def plot_surfaces(pv_model: PolicyValueModel, sdf_model: SDFFC1Combined | None, ref_state: dict, device: torch.device):
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
        out, diagnostics = _compute_policy_diagnostic_surfaces(pv_model, sdf_model, base)
        P0 = out.P0.reshape(B.shape).cpu().numpy()
        PI = out.PI.reshape(B.shape).cpu().numpy()
        P = out.P.reshape(B.shape).cpu().numpy()
        bar_z = out.bar_z.reshape(B.shape).cpu().numpy()
        bp = out.bp.reshape(B.shape).cpu().numpy()
        Q = out.Q.reshape(B.shape).cpu().numpy()
        pi_diff = diagnostics["pidiff"].reshape(B.shape).cpu().numpy()
        cf_diff = diagnostics["cfdiff"].reshape(B.shape).cpu().numpy()
        cont_diff = diagnostics["contdiff"].reshape(B.shape).cpu().numpy()
        survive_mask = (P > 0.0) & (bar_z < 0.5)

    figs_dir = ROOT / "experiments" / "figs"
    for name, arr in [
        ("p0", P0),
        ("pi", PI),
        ("bp", bp),
        ("q", Q),
        ("pidiff", pi_diff),
        ("cfdiff", cf_diff),
        ("contdiff", cont_diff),
    ]:
        plot_arr = arr
        if name in {"bp", "pidiff", "cfdiff", "contdiff"}:
            plot_arr = np.where(survive_mask, arr, np.nan)
        # Heatmap with b as x-axis, z as y-axis
        plt.figure(figsize=(6, 4))
        cs = plt.contourf(B.cpu().numpy(), Z.cpu().numpy(), plot_arr, levels=30, cmap="viridis")
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
        ax.plot_surface(B.cpu().numpy(), Z.cpu().numpy(), plot_arr, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel("b")
        ax.set_ylabel("z")
        ax.set_zlabel(name.upper())
        ax.set_title(f"{name.upper()} surface")
        plt.tight_layout()
        plt.savefig(figs_dir / f"{name}_surface.png", dpi=150)
        plt.close()


def plot_distributions(
    df: pd.DataFrame,
    pv_model: PolicyValueModel,
    device: torch.device,
    df_macro: pd.DataFrame | None = None,
):
    figs_dir = ROOT / "experiments" / "figs"

    def _split_parent_child(df_in: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if "branch" not in df_in.columns:
            return df_in, df_in
        if (df_in["branch"] < 0).any():
            parent_mask = df_in["branch"] < 0
            child_mask = df_in["branch"] >= 0
        else:
            parent_mask = df_in["branch"] == 0
            child_mask = df_in["branch"] > 0
        return df_in[parent_mask].copy(), df_in[child_mask].copy()

    parent_df, child_df = _split_parent_child(df)

    if df_macro is not None and not df_macro.empty and "M" in df_macro.columns:
        macro_parent_df, macro_child_df = _split_parent_child(df_macro)
        m_child_source = macro_child_df["M"].dropna()
        m_parent_source = macro_parent_df["M"].dropna()
    elif "M" in df.columns:
        m_child_source = child_df["M"].dropna()
        m_parent_source = parent_df["M"].dropna()
    else:
        m_child_source = None
        m_parent_source = None

    if m_child_source is not None and len(m_child_source) > 0:
        plt.figure(figsize=(5, 3))
        m_child_source.hist(bins=40)
        plt.title("M distribution (child macro states)")
        plt.xlabel("M")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(figs_dir / "m_hist.png", dpi=150)
        plt.close()

    if m_parent_source is not None and len(m_parent_source) > 0:
        plt.figure(figsize=(5, 3))
        m_parent_source.hist(bins=40)
        plt.title("M distribution (parent macro states)")
        plt.xlabel("M")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(figs_dir / "m_parent_hist.png", dpi=150)
        plt.close()

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

    hyperparams = build_hyperparams()
    models = build_models(device)
    optimizers = build_optimizers(models, hyperparams)

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
        plot_surfaces(models["policy_value"], models.get("sdf_fc1"), ref_state, device)
        plot_distributions(episode.df, models["policy_value"], device, df_macro=episode.df_macro)

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
