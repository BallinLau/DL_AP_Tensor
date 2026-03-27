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
    hp.fc1_recon_weight = 0.0
    hp.fc1_forecast_recon_weight = 1.0
    hp.fc1_hatc_recon_weight = 1.0
    hp.fc1_lnk_recon_weight = 0.25
    hp.fc1_delta_penalty_weight = 10.0
    hp.fc1_delta_hatc_abs_max = 0.50
    hp.fc1_delta_lnk_abs_max = 0.30
    hp.fc1_jacobian_penalty_weight = 1.0
    hp.fc1_use_true_macro_state_in_stage2 = False
    hp.bp_foc_use_phat_children = False
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
        bar_i_cond = out.bar_i_cond.reshape(B.shape).cpu().numpy()
        bar_i = out.bar_i.reshape(B.shape).cpu().numpy()
        chi = out.chi.reshape(B.shape).cpu().numpy()
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
        ("bari_cond", bar_i_cond),
        ("bari", bar_i),
        ("chi", chi),
        ("bp", bp),
        ("q", Q),
        ("pidiff", pi_diff),
        ("cfdiff", cf_diff),
        ("contdiff", cont_diff),
    ]:
        plot_arr = arr
        if name in {"bari_cond", "bari", "chi", "bp", "pidiff", "cfdiff", "contdiff"}:
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


def plot_bp_diagnostic_curves(
    pv_model: PolicyValueModel,
    sdf_model: SDFFC1Combined | None,
    ref_state: dict,
    device: torch.device,
):
    figs_dir = ROOT / "experiments" / "figs"
    p0_loss = P0Loss()
    pi_loss = PILoss()
    hp = build_hyperparams()
    target_states = [
        ("safe", 0.10, 0.80),
        ("mid", 0.35, 0.20),
        ("risky", 0.60, -0.20),
        ("distress", 0.80, -0.60),
    ]
    bp_grid = torch.linspace(0.0, 1.0, 201, device=device).unsqueeze(-1)

    for label, b_val, z_val in target_states:
        parent = torch.tensor(
            [[b_val, z_val, ref_state["eta"], ref_state["i"], ref_state["x"], ref_state["hatcf"], ref_state["lnkf"]]],
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            parent_out = pv_model(parent)
        if float(parent_out.P.item()) <= 0.0 or float(parent_out.bar_z.item()) >= 0.5:
            continue

        bp_var = bp_grid.clone().detach().requires_grad_(True)
        child = parent.repeat(bp_var.shape[0], 1)
        child[:, 0:1] = bp_var
        child_out = pv_model(child)

        x = parent[:, 4:5].expand_as(bp_var)
        z = parent[:, 1:2].expand_as(bp_var)
        b = parent[:, 0:1].expand_as(bp_var)
        i = parent[:, 3:4].expand_as(bp_var)
        eta = parent[:, 2:3].expand_as(bp_var)
        q_parent = parent_out.Q.expand_as(bp_var)

        cf0 = p0_loss.compute_cashflow_p0(x, z, b, q_parent, child_out.Q, eta)
        cfi = pi_loss.compute_cashflow_pi(x, z, b, i, q_parent, child_out.Q, eta)

        prod = torch.exp(x + z) - Config.DELTA - b
        prod = prod - Config.TAU * torch.relu(prod)
        debt0 = ((1.0 - Config.KAPPA_B) * child_out.Q - q_parent) * eta
        raw0 = prod + debt0
        eqcost0 = Config.KAPPA_E * torch.relu(-raw0)
        debtI = ((1.0 - Config.KAPPA_B) * Config.G * child_out.Q - q_parent) * eta
        rawI = prod - i + debtI
        eqcostI = Config.KAPPA_E * torch.relu(-rawI)

        with torch.no_grad():
            if sdf_model is not None:
                _, _, m_next, _, _ = sdf_model.forward_step(
                    x_prev=parent[:, 4:5].expand_as(bp_var),
                    x_curr=parent[:, 4:5].expand_as(bp_var),
                    hatcf_prev=parent[:, 5:6].expand_as(bp_var),
                    lnkf_prev=parent[:, 6:7].expand_as(bp_var),
                    return_physical=True,
                )
            else:
                m_next = torch.ones_like(bp_var)

        cont0 = m_next * child_out.P * (1.0 - child_out.bar_z)
        contI = Config.G * m_next * child_out.P * (1.0 - child_out.bar_z)
        v0_diag = cf0 + cont0
        vi_diag = cfi + contI

        use_phat_for_bp_foc = bool(getattr(hp, "bp_foc_use_phat_children", True))
        p_child_for_foc = child_out.Phat if use_phat_for_bp_foc else child_out.P
        foc0 = p0_loss.compute_foc_residual_from_bp(
            CF0p=[cf0],
            M_list=[m_next],
            P_children=[p_child_for_foc],
            bar_z_children=[child_out.bar_z],
            bp=bp_var,
            eta=[eta],
        )[0]
        foci = pi_loss.compute_foc_residual_from_bp(
            CFip=[cfi],
            M_list=[m_next],
            P_children=[p_child_for_foc],
            bar_z_children=[child_out.bar_z],
            bp=bp_var,
            eta=[eta],
        )[0]

        eps_default = float(getattr(hp, "kkt_boundary_eps", 0.02))
        eps_low = getattr(hp, "kkt_boundary_eps_low", None)
        eps_high = getattr(hp, "kkt_boundary_eps_high", None)
        eps_low = eps_default if eps_low is None else float(eps_low)
        eps_high = eps_default if eps_high is None else float(eps_high)
        temp = float(getattr(hp, "kkt_boundary_temp", 40.0))
        w_high_cfg = float(getattr(hp, "kkt_high_weight", 3.0))
        w_low = torch.sigmoid(temp * (eps_low - bp_var))
        w_high = torch.sigmoid(temp * (bp_var - (1.0 - eps_high)))
        w_inner = (1.0 - w_low) * (1.0 - w_high)
        kkt0_point = w_inner * foc0.pow(2) + w_low * torch.relu(foc0) + w_high * w_high_cfg * torch.relu(-foc0)
        kkti_point = w_inner * foci.pow(2) + w_low * torch.relu(foci) + w_high * w_high_cfg * torch.relu(-foci)

        with torch.no_grad():
            bp_np = bp_grid.squeeze(-1).cpu().numpy()
            q_np = child_out.Q.squeeze(-1).cpu().numpy()
            q_unit_np = (
                child_out.Q.squeeze(-1) / torch.clamp_min(bp_grid.squeeze(-1), 1e-6)
            ).cpu().numpy()
            p_np = child_out.P.squeeze(-1).cpu().numpy()
            barz_np = child_out.bar_z.squeeze(-1).cpu().numpy()
            cf0_np = cf0.squeeze(-1).cpu().numpy()
            cfi_np = cfi.squeeze(-1).cpu().numpy()
            prod_np = prod.squeeze(-1).cpu().numpy()
            debt0_np = debt0.squeeze(-1).cpu().numpy()
            debtI_np = debtI.squeeze(-1).cpu().numpy()
            invest_np = i.squeeze(-1).cpu().numpy()
            eqcost0_np = eqcost0.squeeze(-1).cpu().numpy()
            eqcostI_np = eqcostI.squeeze(-1).cpu().numpy()
            cont0_np = cont0.squeeze(-1).cpu().numpy()
            contI_np = contI.squeeze(-1).cpu().numpy()
            v0_np = v0_diag.squeeze(-1).cpu().numpy()
            vi_np = vi_diag.squeeze(-1).cpu().numpy()
            foc0_np = foc0.squeeze(-1).cpu().numpy()
            foci_np = foci.squeeze(-1).cpu().numpy()
            kkt0_np = kkt0_point.squeeze(-1).cpu().numpy()
            kkti_np = kkti_point.squeeze(-1).cpu().numpy()
            bp0_star = float(parent_out.bp0.item())
            bpI_star = float(parent_out.bpI.item())
            bp_star = float(parent_out.bp.item())
            bp_v0_argmax = float(bp_np[int(np.argmax(v0_np))])
            bp_vi_argmax = float(bp_np[int(np.argmax(vi_np))])
            dcf0_np = np.gradient(cf0_np, bp_np)
            dcfi_np = np.gradient(cfi_np, bp_np)
            dcont0_np = np.gradient(cont0_np, bp_np)
            dcontI_np = np.gradient(contI_np, bp_np)
            dv0_np = np.gradient(v0_np, bp_np)
            dvi_np = np.gradient(vi_np, bp_np)
            p_zero_idx = np.where(p_np <= 1e-8)[0]
            bp_p_zero = float(bp_np[p_zero_idx[0]]) if len(p_zero_idx) > 0 else None
            z_half_idx = np.where(barz_np >= 0.5)[0]
            bp_z_half = float(bp_np[z_half_idx[0]]) if len(z_half_idx) > 0 else None
            foc_mid_np = 0.5 * (foc0_np + foci_np)
            foc_jump_idx = int(np.argmax(np.abs(np.diff(foc_mid_np))))
            bp_foc_jump = float(bp_np[foc_jump_idx + 1])

        fig, axes = plt.subplots(6, 2, figsize=(11, 19))
        (
            ax_q,
            ax_qunit,
            ax_p,
            ax_barz,
            ax_cf,
            ax_cont,
            ax_foc,
            ax_kkt,
            ax_cf0_decomp,
            ax_cfi_decomp,
            ax_dv0,
            ax_dvi,
        ) = axes.flatten()
        ax_q.plot(bp_np, q_np, color="tab:blue")
        ax_q.set_title("Q(bp)")
        ax_qunit.plot(bp_np, q_unit_np, color="tab:purple")
        ax_qunit.set_title("q_unit(bp)=Q(bp)/bp")
        ax_p.plot(bp_np, p_np, color="tab:green")
        ax_p.set_title("P_{t+1}(bp)")
        ax_barz.plot(bp_np, barz_np, color="tab:red")
        ax_barz.set_title("bar_z_{t+1}(bp)")
        ax_cf.plot(bp_np, cf0_np, label="CF0(bp)", color="tab:blue")
        ax_cf.plot(bp_np, cfi_np, label="CFI(bp)", color="tab:orange")
        ax_cf.plot(bp_np, v0_np, label="V0 diag(bp)", color="tab:blue", linestyle="--", alpha=0.7)
        ax_cf.plot(bp_np, vi_np, label="VI diag(bp)", color="tab:orange", linestyle="--", alpha=0.7)
        ax_cf.set_title("Current cash flow and one-step V")
        ax_cf.legend(frameon=False, fontsize=8)
        ax_cont.plot(bp_np, cont0_np, label="cont0(bp)", color="tab:blue")
        ax_cont.plot(bp_np, contI_np, label="contI(bp)", color="tab:orange")
        ax_cont.set_title("Continuation terms")
        ax_cont.legend(frameon=False, fontsize=8)
        ax_foc.plot(bp_np, foc0_np, label="FOC0(bp)", color="tab:blue")
        ax_foc.plot(bp_np, foci_np, label="FOCI(bp)", color="tab:orange")
        ax_foc.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        ax_foc.set_title("Training FOC residuals")
        ax_foc.legend(frameon=False, fontsize=8)
        ax_kkt.plot(bp_np, kkt0_np, label="KKT0 point penalty", color="tab:blue")
        ax_kkt.plot(bp_np, kkti_np, label="KKTI point penalty", color="tab:orange")
        ax_kkt.set_title("Training KKT point penalties")
        ax_kkt.legend(frameon=False, fontsize=8)
        ax_cf0_decomp.plot(bp_np, prod_np, label="prod(bp)", color="tab:green")
        ax_cf0_decomp.plot(bp_np, debt0_np, label="debt_adj0(bp)", color="tab:purple")
        ax_cf0_decomp.plot(bp_np, -eqcost0_np, label="-eq_cost0(bp)", color="tab:red")
        ax_cf0_decomp.plot(bp_np, cf0_np, label="CF0(bp)", color="tab:blue", linestyle="--")
        ax_cf0_decomp.set_title("CF0 decomposition")
        ax_cf0_decomp.legend(frameon=False, fontsize=8)
        ax_cfi_decomp.plot(bp_np, prod_np, label="prod(bp)", color="tab:green")
        ax_cfi_decomp.plot(bp_np, debtI_np, label="debt_adjI(bp)", color="tab:purple")
        ax_cfi_decomp.plot(bp_np, -invest_np, label="-i(bp)", color="tab:brown")
        ax_cfi_decomp.plot(bp_np, -eqcostI_np, label="-eq_costI(bp)", color="tab:red")
        ax_cfi_decomp.plot(bp_np, cfi_np, label="CFI(bp)", color="tab:orange", linestyle="--")
        ax_cfi_decomp.set_title("CFI decomposition")
        ax_cfi_decomp.legend(frameon=False, fontsize=8)
        ax_dv0.plot(bp_np, dcf0_np, label="dCF0/dbp", color="tab:blue")
        ax_dv0.plot(bp_np, dcont0_np, label="dcont0/dbp", color="tab:orange")
        ax_dv0.plot(bp_np, dv0_np, label="dV0/dbp", color="black", linestyle="--")
        ax_dv0.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        ax_dv0.set_title("Derivative decomposition: V0")
        ax_dv0.legend(frameon=False, fontsize=8)
        ax_dvi.plot(bp_np, dcfi_np, label="dCFI/dbp", color="tab:blue")
        ax_dvi.plot(bp_np, dcontI_np, label="dcontI/dbp", color="tab:orange")
        ax_dvi.plot(bp_np, dvi_np, label="dVI/dbp", color="black", linestyle="--")
        ax_dvi.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        ax_dvi.set_title("Derivative decomposition: VI")
        ax_dvi.legend(frameon=False, fontsize=8)
        for ax in axes.flatten():
            ax.axvline(bp0_star, color="tab:blue", linestyle="--", linewidth=1)
            ax.axvline(bpI_star, color="tab:orange", linestyle="--", linewidth=1)
            ax.axvline(bp_star, color="black", linestyle=":", linewidth=1)
            ax.set_xlabel("bp")
            ax.grid(True, alpha=0.25)
        fig.suptitle(
            f"bp diagnostics: {label} state "
            f"(b={b_val:.2f}, z={z_val:.2f}, P={float(parent_out.P.item()):.3f}, bar_z={float(parent_out.bar_z.item()):.3f})"
        )
        fig.text(
            0.5,
            0.955,
            f"bp0*={bp0_star:.3f}  |  bpI*={bpI_star:.3f}  |  bp*={bp_star:.3f}  "
            f"|  argmax V0={bp_v0_argmax:.3f}  |  argmax VI={bp_vi_argmax:.3f}",
            ha="center",
            va="top",
            fontsize=9,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(figs_dir / f"bp_diag_{label}.png", dpi=150)
        plt.close(fig)

        fig2, (ax_b1, ax_b2) = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True)
        ax_b1.plot(bp_np, p_np, label="P_{t+1}(bp)", color="tab:green")
        ax_b1_t = ax_b1.twinx()
        ax_b1_t.plot(bp_np, barz_np, label="bar_z_{t+1}(bp)", color="tab:red")
        ax_b1.set_ylabel("P")
        ax_b1_t.set_ylabel("bar_z")
        ax_b1.set_title("Survival/default boundary")
        ax_b2.plot(bp_np, foc0_np, label="FOC0(bp)", color="tab:blue")
        ax_b2.plot(bp_np, foci_np, label="FOCI(bp)", color="tab:orange")
        ax_b2.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        ax_b2.set_xlabel("bp")
        ax_b2.set_ylabel("FOC")
        ax_b2.set_title("FOC around boundary")
        boundary_lines = [
            ("P=0", bp_p_zero, "tab:green"),
            ("bar_z=0.5", bp_z_half, "tab:red"),
            ("FOC jump", bp_foc_jump, "tab:purple"),
            ("bp*", bp_star, "black"),
        ]
        for _, xval, color in boundary_lines:
            if xval is None:
                continue
            ax_b1.axvline(xval, color=color, linestyle="--", linewidth=1)
            ax_b1_t.axvline(xval, color=color, linestyle="--", linewidth=1)
            ax_b2.axvline(xval, color=color, linestyle="--", linewidth=1)
        lines1, labels1 = ax_b1.get_legend_handles_labels()
        lines2, labels2 = ax_b1_t.get_legend_handles_labels()
        ax_b1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8, loc="center right")
        ax_b2.legend(frameon=False, fontsize=8)
        boundary_text = " | ".join(
            [
                f"P=0 @ {bp_p_zero:.3f}" if bp_p_zero is not None else "P=0 @ NA",
                f"bar_z=0.5 @ {bp_z_half:.3f}" if bp_z_half is not None else "bar_z=0.5 @ NA",
                f"FOC jump @ {bp_foc_jump:.3f}",
                f"bp* @ {bp_star:.3f}",
            ]
        )
        fig2.suptitle(f"bp boundary summary: {label} state (b={b_val:.2f}, z={z_val:.2f})")
        fig2.text(0.5, 0.955, boundary_text, ha="center", va="top", fontsize=9)
        ax_b1.grid(True, alpha=0.25)
        ax_b2.grid(True, alpha=0.25)
        fig2.tight_layout(rect=[0, 0, 1, 0.92])
        fig2.savefig(figs_dir / f"bp_boundary_{label}.png", dpi=150)
        plt.close(fig2)


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
        plot_bp_diagnostic_curves(models["policy_value"], models.get("sdf_fc1"), ref_state, device)
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
