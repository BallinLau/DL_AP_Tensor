"""
Utilities for multi-episode runs: model/optimizer/hparam builders, I/O helpers, and plots.
"""

from pathlib import Path
from typing import Optional
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import Config, HyperParams
from models import SDFFC1Combined, PolicyValueModel, FC2HatcModel, FC2LnkModel
from losses import P0Loss, PILoss


def resolve_base_dir(run_root: Optional[Path], project_root: Path) -> Path:
    """Return run_root if provided; otherwise fall back to project_root."""
    return run_root if run_root is not None else project_root


def _load_state_dict(ckpt_path: Path, device: torch.device):
    """
    Load a plain tensor state dict without opt-in pickle execution.

    `weights_only=True` is the forward-compatible PyTorch setting for model
    checkpoints. Fall back for older torch versions that do not expose it.
    """
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


def _artifact_suffix(tag: Optional[str]) -> str:
    return f"_{tag}" if tag else ""


def _episode_prefix(ep: int, tag: Optional[str] = None) -> str:
    return f"ep{ep}{_artifact_suffix(tag)}"


def _episode_title_prefix(ep: int, tag: Optional[str] = None) -> str:
    return f"EP{ep}" if not tag else f"EP{ep} [{tag}]"


def build_models(device: torch.device, ckpt_dir: Optional[Path | str] = None, ckpt_prefix: str | None = None, strict: bool = True):
    models = {
        "sdf_fc1": SDFFC1Combined(
            sdf_input_dim=Config.FC1_INPUT_DIM,
            fc1_input_dim=Config.FC1_INPUT_DIM,
            sdf_hidden_dims=Config.SDF_HIDDEN_DIMS,
            fc1_hidden_dims=Config.FC1_HIDDEN_DIMS,
            w_hidden_dims=Config.SDF_HIDDEN_DIMS,
        ).to(device),
        "policy_value": PolicyValueModel().to(device),
        "fc2_hatc": FC2HatcModel(
            input_dim=Config.FC2_INPUT_DIM,
            hidden_dims=Config.FC2_HIDDEN_DIMS,
            quantile_num=Config.QUANTILE_NUM,
        ).to(device),
        "fc2_lnk": FC2LnkModel(
            input_dim=Config.FC2_INPUT_DIM + Config.QUANTILE_NUM,
            hidden_dims=Config.FC2_HIDDEN_DIMS,
            quantile_num=Config.QUANTILE_NUM,
        ).to(device),
    }

    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)
        prefix = f"{ckpt_prefix}_" if ckpt_prefix else ""
        mapping = {
            "sdf_fc1": "sdf_fc1",
            "policy_value": "policy_value",
            "fc2_hatc": "fc2_hatc",
            "fc2_lnk": "fc2_lnk",
        }
        for key, stem in mapping.items():
            if key == "policy_value":
                split_q = ckpt_dir / f"{prefix}{stem}_q.pt"
                split_pvbp = ckpt_dir / f"{prefix}{stem}_pvbp.pt"
                merged = ckpt_dir / f"{prefix}{stem}.pt"
                if split_q.exists() and split_pvbp.exists():
                    q_state = _load_state_dict(split_q, device)
                    pvbp_state = _load_state_dict(split_pvbp, device)
                    models[key].q_model.load_state_dict(q_state, strict=strict)
                    models[key].pvbp_model.load_state_dict(pvbp_state, strict=strict)
                    print(f"[build_models] loaded {split_q}")
                    print(f"[build_models] loaded {split_pvbp}")
                elif merged.exists():
                    state = _load_state_dict(merged, device)
                    is_old_joint_ckpt = any(
                        str(k).startswith("shared_model.") or str(k).startswith("combined_model.")
                        for k in state.keys()
                    )
                    if is_old_joint_ckpt:
                        print(f"[build_models] skip incompatible legacy policy_value ckpt: {merged}")
                    else:
                        models[key].load_state_dict(state, strict=strict)
                        print(f"[build_models] loaded {merged}")
                else:
                    print(f"[build_models] skip missing ckpt: {split_q} / {split_pvbp} / {merged}")
            elif key in {"fc2_hatc", "fc2_lnk"}:
                ckpt_path = ckpt_dir / f"{prefix}{stem}.pt"
                legacy_path = ckpt_dir / f"{prefix}fc2.pt"
                if ckpt_path.exists():
                    state = _load_state_dict(ckpt_path, device)
                    try:
                        models[key].load_state_dict(state, strict=strict)
                        print(f"[build_models] loaded {ckpt_path}")
                    except RuntimeError as exc:
                        print(f"[build_models] skip incompatible ckpt: {ckpt_path} ({exc})")
                elif legacy_path.exists():
                    state = _load_state_dict(legacy_path, device)
                    if key == "fc2_hatc":
                        trunk_prefix = "hatc_model.trunk."
                        head_prefix = "hatc_model.head."
                        legacy_trunk_prefix = "hatc_trunk."
                        legacy_head_prefix = "hatc_head."
                    else:
                        trunk_prefix = "lnk_model.trunk."
                        head_prefix = "lnk_model.head."
                        legacy_trunk_prefix = "lnk_trunk."
                        legacy_head_prefix = "lnk_head."
                    mapped = {}
                    for k, v in state.items():
                        if k.startswith(legacy_trunk_prefix):
                            mapped["trunk." + k[len(legacy_trunk_prefix):]] = v
                        elif k.startswith(legacy_head_prefix):
                            mapped["head." + k[len(legacy_head_prefix):]] = v
                        elif k.startswith(trunk_prefix):
                            mapped["trunk." + k[len(trunk_prefix):]] = v
                        elif k.startswith(head_prefix):
                            mapped["head." + k[len(head_prefix):]] = v
                    if mapped:
                        try:
                            models[key].load_state_dict(mapped, strict=strict)
                            print(f"[build_models] loaded legacy split from {legacy_path} -> {key}")
                        except RuntimeError as exc:
                            print(f"[build_models] skip incompatible legacy fc2 ckpt: {legacy_path} ({exc})")
                    else:
                        print(f"[build_models] skip legacy fc2 ckpt without split keys: {legacy_path}")
                else:
                    print(f"[build_models] skip missing ckpt: {ckpt_path} / {legacy_path}")
            else:
                ckpt_path = ckpt_dir / f"{prefix}{stem}.pt"
                if ckpt_path.exists():
                    state = _load_state_dict(ckpt_path, device)
                    try:
                        models[key].load_state_dict(state, strict=strict)
                        print(f"[build_models] loaded {ckpt_path}")
                    except RuntimeError as exc:
                        print(f"[build_models] skip incompatible ckpt: {ckpt_path} ({exc})")
                else:
                    print(f"[build_models] skip missing ckpt: {ckpt_path}")

    return models


def build_optimizers(models, hyperparams: Optional[HyperParams] = None):
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
            opts["policy_value_q"] = torch.optim.AdamW(
                model.q_model.parameters(),
                lr=hp.q_lr,
                weight_decay=hp.q_weight_decay,
            )
            opts["policy_value_pvbp"] = torch.optim.AdamW(
                model.pvbp_model.parameters(),
                lr=hp.pvbp_lr,
                weight_decay=hp.pvbp_weight_decay,
            )
        elif name == "fc2":
            opts[name] = torch.optim.AdamW(
                model.parameters(),
                lr=hp.fc2_lr,
                weight_decay=hp.fc2_weight_decay,
            )
        elif name in {"fc2_hatc", "fc2_lnk"}:
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
        n_samples=2000,
        n_paths=200,
        batch_size=4096,
        epochs=20,
        simulate_horizon=100,
    )
    hp.lr = 1e-3
    hp.min_lr = 1e-6
    hp.warmup_steps = 0
    hp.max_steps = 20000
    hp.w_sdf = 1.0
    hp.w_p0 = 1.0
    hp.w_pi = 1.0
    hp.w_q = 1.0
    hp.w_fc2 = 1.0
    # SDF 两阶段稳定配置（与 notebook 口径对齐）
    hp.fc1_recon_weight = 0.0
    hp.fc1_forecast_recon_weight = 1.0
    hp.fc1_hatc_recon_weight = 1.0
    hp.fc1_lnk_recon_weight = 0.25
    hp.fc1_delta_penalty_weight = 10.0
    hp.fc1_delta_hatc_abs_max = 0.50
    hp.fc1_delta_lnk_abs_max = 0.30
    hp.fc1_jacobian_penalty_weight = 1.0
    hp.fc1_use_true_macro_state_in_stage2 = True
    hp.sdf_stage1_lr = 1e-4
    hp.sdf_stage2_lr = 2e-4
    hp.sdf_stage1_moment_weight = 5.0
    hp.sdf_moment_weight = 5.0
    hp.sdf_log_mean_anchor_weight_stage1 = 1.0
    hp.sdf_log_mean_anchor_weight_stage2 = 5.0
    # 显式覆盖 Q 训练入口，避免 HyperParams 默认值和运行入口脱节。
    hp.q_pretrain_epochs = 40
    hp.q_stage_epochs = 40
    hp.pvbp_stage_epochs = 60
    hp.policy_separate_q_pvbp_training = True
    hp.q_refresh_stage_epochs = 10
    hp.q_warmstart_epochs = 40
    hp.q_lr = hp.policy_lr
    hp.q_weight_decay = hp.policy_weight_decay
    hp.pvbp_lr = hp.policy_lr
    hp.pvbp_weight_decay = hp.policy_weight_decay
    hp.q_pretrain_trainable_scope = "q_path"
    hp.q_shape_weight_z = 1.0
    hp.q_shape_weight_b_low = 1.0
    hp.q_shape_weight_b_high = 0.0
    # 先关闭 bp 的额外边界推进，避免 bp 长期贴到 1
    hp.bp_adaptive_enabled = False
    hp.bp_refine_steps_per_epoch = 0
    hp.bp_foc_use_phat_children = False
    hp.bp_diag_enabled = True
    hp.bp_diag_every_n_episodes = 5
    hp.bp_diag_states = "safe"
    hp.bp_diag_grid_points = 101
    hp.bp_diag_use_autograd_foc = False
    hp.bp_survival_reweight_enabled = True
    hp.bp_survival_tau_p = 20.0
    hp.bp_survival_tau_z = 20.0
    hp.bp_survival_barz_threshold = 0.5
    hp.bp_value_supervision_enabled = True
    hp.bp_value_weight = 1.0
    hp.bp_value_grid_points = 21
    hp.bp_value_sample_cap = 256
    hp.bp_value_survival_only = True
    hp.bp_value_barz_threshold = 0.5
    # P0/PI 默认与 raw-M Bellman operator 对齐；如需旧口径，可切回 clip/anneal。
    hp.pv_m_mode = "raw"
    hp.pv_m_clip_anneal_epochs = 0
    hp.pv_use_clipped_m = False
    hp.pv_m_clamp_min = 0.7
    hp.pv_m_clamp_max = 1.3
    return hp


def ensure_dirs(base_dir: Path):
    (base_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (base_dir / "data" / "outputs").mkdir(parents=True, exist_ok=True)
    (base_dir / "experiments" / "figs").mkdir(parents=True, exist_ok=True)


def save_models(models, episode: int, base_dir: Path, tag: Optional[str] = None):
    prefix = _episode_prefix(episode, tag)
    torch.save(models["sdf_fc1"].state_dict(), base_dir / "checkpoints" / f"{prefix}_sdf_fc1.pt")
    torch.save(models["policy_value"].state_dict(), base_dir / "checkpoints" / f"{prefix}_policy_value.pt")
    torch.save(models["policy_value"].q_model.state_dict(), base_dir / "checkpoints" / f"{prefix}_policy_value_q.pt")
    torch.save(models["policy_value"].pvbp_model.state_dict(), base_dir / "checkpoints" / f"{prefix}_policy_value_pvbp.pt")
    if "fc2_hatc" in models:
        torch.save(models["fc2_hatc"].state_dict(), base_dir / "checkpoints" / f"{prefix}_fc2_hatc.pt")
    if "fc2_lnk" in models:
        torch.save(models["fc2_lnk"].state_dict(), base_dir / "checkpoints" / f"{prefix}_fc2_lnk.pt")


def save_stage_df(ep: int, name: str, base_dir: Path, df_firm: pd.DataFrame = None, df_macro: pd.DataFrame = None, df_sdf: pd.DataFrame = None):
    out_dir = base_dir / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    target_df = df_firm if df_firm is not None else df_sdf
    if target_df is not None:
        target_df.to_pickle(out_dir / f"ep{ep}_stage_{name}.pkl")
    if df_macro is not None:
        df_macro.to_pickle(out_dir / f"ep{ep}_stage_{name}_macro.pkl")


def build_policy_ref_state(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        raise ValueError("Cannot build policy diagnostic state from an empty DataFrame")
    parent_df = df[df["branch"] <= 0] if "branch" in df.columns else df
    if parent_df.empty:
        parent_df = df
    return {
        "eta": 1.0,
        "i": parent_df["i"].median(),
        "x": parent_df["x"].median(),
        "hatcf": parent_df["Hatcf"].median(),
        "lnkf": parent_df["LnKF"].median(),
    }


def build_outer_drift_ref_state() -> dict:
    """
    Fixed reference state for cross-episode surface drift.

    Use a constant state so drift is not contaminated by episode-specific medians.
    """
    return {
        "eta": 1.0,
        "i": float(Config.I_THRESHOLD) * 0.5,
        "x": float(Config.XBAR),
        "hatcf": -2.0,
        "lnkf": 4.0,
    }


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


def _pick_col(df: pd.DataFrame, candidates) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _binned_x_curve(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    if mask.sum() < 3:
        return np.asarray([]), np.asarray([])
    x = x_vals[mask]
    y = y_vals[mask]
    mids = []
    means = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        mk = (x >= lo) & (x <= hi if hi == edges[-1] else x < hi)
        if mk.sum() < 3:
            continue
        mids.append(float(x[mk].mean()))
        means.append(float(y[mk].mean()))
    return np.asarray(mids), np.asarray(means)


def compute_macro_moment_summary(df_macro: pd.DataFrame | None) -> dict:
    if df_macro is None or df_macro.empty:
        return {}
    _, child_df = _split_parent_child(df_macro)
    use_df = child_df if not child_df.empty else df_macro
    out: dict[str, float] = {"n_rows": float(len(use_df))}
    for label, cols in [
        ("hatc", ["Hatc", "hatc"]),
        ("lnk", ["LnK", "lnk"]),
        ("hatcf", ["hatcf", "Hatcf"]),
        ("lnkf", ["lnkf", "LnKF"]),
        ("m", ["M"]),
        ("x", ["x"]),
    ]:
        col = _pick_col(use_df, cols)
        if col is None:
            continue
        vals = pd.to_numeric(use_df[col], errors="coerce").to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out[f"{label}_mean"] = float(np.mean(vals))
        out[f"{label}_std"] = float(np.std(vals))
    return out


def compute_macro_outer_drift(
    prev_df_macro: pd.DataFrame | None,
    curr_df_macro: pd.DataFrame | None,
    n_bins: int = 15,
) -> dict:
    if prev_df_macro is None or curr_df_macro is None or prev_df_macro.empty or curr_df_macro.empty:
        return {}

    prev_parent, prev_child = _split_parent_child(prev_df_macro)
    curr_parent, curr_child = _split_parent_child(curr_df_macro)
    prev_use = prev_child if not prev_child.empty else prev_df_macro
    curr_use = curr_child if not curr_child.empty else curr_df_macro

    out: dict[str, float] = {}

    prev_mom = compute_macro_moment_summary(prev_df_macro)
    curr_mom = compute_macro_moment_summary(curr_df_macro)
    for key in set(prev_mom.keys()) & set(curr_mom.keys()):
        if key.endswith("_mean") or key.endswith("_std"):
            out[f"delta_{key}"] = float(curr_mom[key] - prev_mom[key])
            out[f"abs_delta_{key}"] = float(abs(curr_mom[key] - prev_mom[key]))

    x_col_prev = _pick_col(prev_use, ["x"])
    x_col_curr = _pick_col(curr_use, ["x"])
    if x_col_prev is None or x_col_curr is None:
        return out

    x_all = np.concatenate(
        [
            pd.to_numeric(prev_use[x_col_prev], errors="coerce").to_numpy(),
            pd.to_numeric(curr_use[x_col_curr], errors="coerce").to_numpy(),
        ]
    )
    x_all = x_all[np.isfinite(x_all)]
    if x_all.size < max(10, n_bins):
        return out
    edges = np.quantile(x_all, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 4:
        return out

    for label, cols in [
        ("hatc_true", ["Hatc", "hatc"]),
        ("lnk_true", ["LnK", "lnk"]),
        ("hatc_pred", ["hatcf", "Hatcf"]),
        ("lnk_pred", ["lnkf", "LnKF"]),
    ]:
        prev_col = _pick_col(prev_use, cols)
        curr_col = _pick_col(curr_use, cols)
        if prev_col is None or curr_col is None:
            continue
        _, prev_curve = _binned_x_curve(
            pd.to_numeric(prev_use[x_col_prev], errors="coerce").to_numpy(),
            pd.to_numeric(prev_use[prev_col], errors="coerce").to_numpy(),
            edges,
        )
        _, curr_curve = _binned_x_curve(
            pd.to_numeric(curr_use[x_col_curr], errors="coerce").to_numpy(),
            pd.to_numeric(curr_use[curr_col], errors="coerce").to_numpy(),
            edges,
        )
        if prev_curve.size == 0 or curr_curve.size == 0:
            continue
        n = min(prev_curve.size, curr_curve.size)
        diff = curr_curve[:n] - prev_curve[:n]
        out[f"{label}_x_curve_rmse"] = float(np.sqrt(np.mean(diff ** 2)))
        out[f"{label}_x_curve_maxabs"] = float(np.max(np.abs(diff)))

    return out


def compute_policy_surface_snapshot(
    pv_model: PolicyValueModel,
    device: torch.device,
    ref_state: Optional[dict] = None,
    grid_points: int = 31,
) -> dict[str, np.ndarray]:
    ref = ref_state or build_outer_drift_ref_state()
    b_grid = torch.linspace(0, 1, grid_points, device=device)
    z_grid = torch.linspace(-4, 4, grid_points, device=device)
    B, Z = torch.meshgrid(b_grid, z_grid, indexing="ij")
    base = torch.stack(
        [
            B.reshape(-1),
            Z.reshape(-1),
            torch.full_like(B.reshape(-1), ref["eta"]),
            torch.full_like(B.reshape(-1), ref["i"]),
            torch.full_like(B.reshape(-1), ref["x"]),
            torch.full_like(B.reshape(-1), ref["hatcf"]),
            torch.full_like(B.reshape(-1), ref["lnkf"]),
        ],
        dim=1,
    )
    with torch.no_grad():
        out = pv_model(base)
    shape = B.shape
    return {
        "Q": out.Q.reshape(shape).detach().cpu().numpy(),
        "P": out.P.reshape(shape).detach().cpu().numpy(),
        "bar_z": out.bar_z.reshape(shape).detach().cpu().numpy(),
        "bp": out.bp.reshape(shape).detach().cpu().numpy(),
    }


def compute_policy_surface_drift(
    prev_snapshot: Optional[dict[str, np.ndarray]],
    curr_snapshot: Optional[dict[str, np.ndarray]],
) -> dict:
    if not prev_snapshot or not curr_snapshot:
        return {}
    out: dict[str, float] = {}
    for key in ["Q", "P", "bar_z", "bp"]:
        if key not in prev_snapshot or key not in curr_snapshot:
            continue
        prev_arr = np.asarray(prev_snapshot[key], dtype=float)
        curr_arr = np.asarray(curr_snapshot[key], dtype=float)
        if prev_arr.shape != curr_arr.shape or prev_arr.size == 0:
            continue
        diff = curr_arr - prev_arr
        out[f"{key}_mae"] = float(np.mean(np.abs(diff)))
        out[f"{key}_rmse"] = float(np.sqrt(np.mean(diff ** 2)))
        out[f"{key}_maxabs"] = float(np.max(np.abs(diff)))
    return out


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


def plot_surfaces(
    ep: int,
    pv_model: PolicyValueModel,
    sdf_model: SDFFC1Combined | None,
    ref_state: dict,
    device: torch.device,
    base_dir: Path,
    tag: Optional[str] = None,
):
    figs_dir = base_dir / "experiments" / "figs"
    prefix = _episode_prefix(ep, tag)
    title_prefix = _episode_title_prefix(ep, tag)
    b_grid = torch.linspace(0, 1, 50, device=device)
    z_grid = torch.linspace(-4, 4, 50, device=device)
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

    for name, arr in [
        ("p0", P0),
        ("pi", PI),
        ("p", P),
        ("bari_cond", bar_i_cond),
        ("bari", bar_i),
        ("chi", chi),
        ("barz", bar_z),
        ("bp", bp),
        ("q", Q),
        ("pidiff", pi_diff),
        ("cfdiff", cf_diff),
        ("contdiff", cont_diff),
    ]:
        plot_arr = arr
        if name in {"bari_cond", "bari", "chi", "bp", "pidiff", "cfdiff", "contdiff"}:
            plot_arr = np.where(survive_mask, arr, np.nan)
        plt.figure(figsize=(6, 4))
        cs = plt.contourf(B.cpu().numpy(), Z.cpu().numpy(), plot_arr, levels=30, cmap="viridis")
        plt.colorbar(cs)
        plt.xlabel("b")
        plt.ylabel("z")
        plt.title(f"{title_prefix} {name.upper()} heatmap")
        plt.tight_layout()
        plt.savefig(figs_dir / f"{prefix}_{name}_heatmap.png", dpi=150)
        plt.close()

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(B.cpu().numpy(), Z.cpu().numpy(), plot_arr, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel("b")
        ax.set_ylabel("z")
        ax.set_zlabel(name.upper())
        ax.set_title(f"{title_prefix} {name.upper()} surface")
        plt.tight_layout()
        plt.savefig(figs_dir / f"{prefix}_{name}_surface.png", dpi=150)
        plt.close()


def plot_bp_diagnostic_curves(
    ep: int,
    pv_model: PolicyValueModel,
    sdf_model: SDFFC1Combined | None,
    ref_state: dict,
    device: torch.device,
    base_dir: Path,
    hyperparams: Optional[HyperParams] = None,
    tag: Optional[str] = None,
):
    figs_dir = base_dir / "experiments" / "figs"
    p0_loss = P0Loss()
    pi_loss = PILoss()
    hp = hyperparams or build_hyperparams()
    prefix = _episode_prefix(ep, tag)
    title_prefix = _episode_title_prefix(ep, tag)

    target_state_map = [
        ("safe", 0.10, 1.50),
        ("mid", 0.35, 0.50),
        ("risky", 0.60, -0.50),
        ("distress", 0.80, -1.00),
    ]
    enabled_labels = {
        s.strip() for s in str(getattr(hp, "bp_diag_states", "safe")).split(",") if s.strip()
    }
    target_states = [item for item in target_state_map if item[0] in enabled_labels] or [target_state_map[0]]
    grid_points = int(getattr(hp, "bp_diag_grid_points", 101))
    bp_grid = torch.linspace(0.0, 1.0, grid_points, device=device).unsqueeze(-1)
    use_autograd_foc = bool(getattr(hp, "bp_diag_use_autograd_foc", False))

    for label, b_val, z_val in target_states:
        parent = torch.tensor(
            [[
                b_val,
                z_val,
                ref_state["eta"],
                ref_state["i"],
                ref_state["x"],
                ref_state["hatcf"],
                ref_state["lnkf"],
            ]],
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            parent_out = pv_model(parent)
        parent_default_flag = (
            float(parent_out.P.item()) <= 0.0 or float(parent_out.bar_z.item()) >= 0.5
        )

        with torch.no_grad():
            bp_var = bp_grid
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

            # Decompose current cash flow into economically interpretable pieces.
            prod = torch.exp(x + z) - Config.DELTA - b
            prod = prod - Config.TAU * torch.relu(prod)
            debt0 = ((1.0 - Config.KAPPA_B) * child_out.Q - q_parent) * eta
            raw0 = prod + debt0
            eqcost0 = Config.KAPPA_E * torch.relu(-raw0)
            debtI = ((1.0 - Config.KAPPA_B) * Config.G * child_out.Q - q_parent) * eta
            rawI = prod - i + debtI
            eqcostI = Config.KAPPA_E * torch.relu(-rawI)

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

            # Compare invest branch under multiple i values on the same bp grid.
            i_compare_vals = torch.linspace(0.0, Config.I_THRESHOLD, steps=5, device=device)
            cfi_compare = []
            vi_compare = []
            for i_val in i_compare_vals:
                i_cmp = torch.full_like(bp_var, float(i_val.item()))
                cfi_cmp = pi_loss.compute_cashflow_pi(x, z, b, i_cmp, q_parent, child_out.Q, eta)
                vi_cmp = cfi_cmp + contI
                cfi_compare.append(cfi_cmp.squeeze(-1).cpu().numpy())
                vi_compare.append(vi_cmp.squeeze(-1).cpu().numpy())
            i_compare_np = i_compare_vals.cpu().numpy()

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
            if use_autograd_foc:
                pass

            p_zero_idx = np.where(p_np <= 1e-8)[0]
            bp_p_zero = float(bp_np[p_zero_idx[0]]) if len(p_zero_idx) > 0 else None
            z_half_idx = np.where(barz_np >= 0.5)[0]
            bp_z_half = float(bp_np[z_half_idx[0]]) if len(z_half_idx) > 0 else None
            foc0_np = dv0_np
            foci_np = dvi_np
            eps_default = float(getattr(hp, "kkt_boundary_eps", 0.02))
            eps_low = getattr(hp, "kkt_boundary_eps_low", None)
            eps_high = getattr(hp, "kkt_boundary_eps_high", None)
            eps_low = eps_default if eps_low is None else float(eps_low)
            eps_high = eps_default if eps_high is None else float(eps_high)
            temp = float(getattr(hp, "kkt_boundary_temp", 40.0))
            w_high_cfg = float(getattr(hp, "kkt_high_weight", 3.0))
            w_low_np = 1.0 / (1.0 + np.exp(-temp * (eps_low - bp_np)))
            w_high_np = 1.0 / (1.0 + np.exp(-temp * (bp_np - (1.0 - eps_high))))
            w_inner_np = (1.0 - w_low_np) * (1.0 - w_high_np)
            kkt0_np = w_inner_np * (foc0_np ** 2) + w_low_np * np.maximum(foc0_np, 0.0) + w_high_np * w_high_cfg * np.maximum(-foc0_np, 0.0)
            kkti_np = w_inner_np * (foci_np ** 2) + w_low_np * np.maximum(foci_np, 0.0) + w_high_np * w_high_cfg * np.maximum(-foci_np, 0.0)
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
        ax_q.set_xlabel("bp")
        ax_q.set_ylabel("Q")

        ax_qunit.plot(bp_np, q_unit_np, color="tab:purple")
        ax_qunit.set_title("q_unit(bp)=Q(bp)/bp")
        ax_qunit.set_xlabel("bp")
        ax_qunit.set_ylabel("q_unit")

        ax_p.plot(bp_np, p_np, color="tab:green")
        ax_p.set_title("P_{t+1}(bp)")
        ax_p.set_xlabel("bp")
        ax_p.set_ylabel("P")

        ax_barz.plot(bp_np, barz_np, color="tab:red")
        ax_barz.set_title("bar_z_{t+1}(bp)")
        ax_barz.set_xlabel("bp")
        ax_barz.set_ylabel("bar_z")

        ax_cf.plot(bp_np, cf0_np, label="CF0(bp)", color="tab:blue")
        ax_cf.plot(bp_np, v0_np, label="V0 diag(bp)", color="tab:blue", linestyle="--", alpha=0.7)
        compare_colors = plt.cm.plasma(np.linspace(0.15, 0.90, len(i_compare_np)))
        ref_i_val = float(parent[:, 3:4].item())
        ref_idx = int(np.argmin(np.abs(i_compare_np - ref_i_val)))
        for idx, (i_val, cfi_curve, vi_curve, color) in enumerate(zip(i_compare_np, cfi_compare, vi_compare, compare_colors)):
            is_ref = idx == ref_idx
            line_alpha = 0.95 if is_ref else 0.55
            line_width = 2.0 if is_ref else 1.1
            label_suffix = " [ref]" if is_ref else ""
            ax_cf.plot(
                bp_np,
                cfi_curve,
                color=color,
                linewidth=line_width,
                alpha=line_alpha,
                label=f"CFI(i={i_val:.3f}){label_suffix}",
            )
            ax_cf.plot(
                bp_np,
                vi_curve,
                color=color,
                linewidth=line_width,
                alpha=line_alpha,
                linestyle="--",
                label=f"VI(i={i_val:.3f}){label_suffix}",
            )
        ax_cf.set_title("Current cash flow and one-step V")
        ax_cf.set_xlabel("bp")
        ax_cf.set_ylabel("value")
        ax_cf.legend(frameon=False, fontsize=8)

        ax_cont.plot(bp_np, cont0_np, label="cont0(bp)", color="tab:blue")
        ax_cont.plot(bp_np, contI_np, label="contI(bp)", color="tab:orange")
        ax_cont.set_title("Continuation terms")
        ax_cont.set_xlabel("bp")
        ax_cont.set_ylabel("continuation")
        ax_cont.legend(frameon=False, fontsize=8)

        ax_foc.plot(bp_np, foc0_np, label="FOC0(bp)", color="tab:blue")
        ax_foc.plot(bp_np, foci_np, label="FOCI(bp)", color="tab:orange")
        ax_foc.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        ax_foc.set_title("Approx. FOC residuals")
        ax_foc.set_xlabel("bp")
        ax_foc.set_ylabel("FOC")
        ax_foc.legend(frameon=False, fontsize=8)

        ax_kkt.plot(bp_np, kkt0_np, label="KKT0 point penalty", color="tab:blue")
        ax_kkt.plot(bp_np, kkti_np, label="KKTI point penalty", color="tab:orange")
        ax_kkt.set_title("Approx. KKT point penalties")
        ax_kkt.set_xlabel("bp")
        ax_kkt.set_ylabel("penalty")
        ax_kkt.legend(frameon=False, fontsize=8)

        ax_cf0_decomp.plot(bp_np, prod_np, label="prod(bp)", color="tab:green")
        ax_cf0_decomp.plot(bp_np, debt0_np, label="debt_adj0(bp)", color="tab:purple")
        ax_cf0_decomp.plot(bp_np, -eqcost0_np, label="-eq_cost0(bp)", color="tab:red")
        ax_cf0_decomp.plot(bp_np, cf0_np, label="CF0(bp)", color="tab:blue", linestyle="--")
        ax_cf0_decomp.set_title("CF0 decomposition")
        ax_cf0_decomp.set_xlabel("bp")
        ax_cf0_decomp.set_ylabel("value")
        ax_cf0_decomp.legend(frameon=False, fontsize=8)

        ax_cfi_decomp.plot(bp_np, prod_np, label="prod(bp)", color="tab:green")
        ax_cfi_decomp.plot(bp_np, debtI_np, label="debt_adjI(bp)", color="tab:purple")
        ax_cfi_decomp.plot(bp_np, -invest_np, label="-i(bp)", color="tab:brown")
        ax_cfi_decomp.plot(bp_np, -eqcostI_np, label="-eq_costI(bp)", color="tab:red")
        ax_cfi_decomp.plot(bp_np, cfi_np, label="CFI(bp)", color="tab:orange", linestyle="--")
        ax_cfi_decomp.set_title("CFI decomposition")
        ax_cfi_decomp.set_xlabel("bp")
        ax_cfi_decomp.set_ylabel("value")
        ax_cfi_decomp.legend(frameon=False, fontsize=8)

        ax_dv0.plot(bp_np, dcf0_np, label="dCF0/dbp", color="tab:blue")
        ax_dv0.plot(bp_np, dcont0_np, label="dcont0/dbp", color="tab:orange")
        ax_dv0.plot(bp_np, dv0_np, label="dV0/dbp", color="black", linestyle="--")
        ax_dv0.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        ax_dv0.set_title("Derivative decomposition: V0")
        ax_dv0.set_xlabel("bp")
        ax_dv0.set_ylabel("derivative")
        ax_dv0.legend(frameon=False, fontsize=8)

        ax_dvi.plot(bp_np, dcfi_np, label="dCFI/dbp", color="tab:blue")
        ax_dvi.plot(bp_np, dcontI_np, label="dcontI/dbp", color="tab:orange")
        ax_dvi.plot(bp_np, dvi_np, label="dVI/dbp", color="black", linestyle="--")
        ax_dvi.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        ax_dvi.set_title("Derivative decomposition: VI")
        ax_dvi.set_xlabel("bp")
        ax_dvi.set_ylabel("derivative")
        ax_dvi.legend(frameon=False, fontsize=8)

        for ax in axes.flatten():
            ax.axvline(bp0_star, color="tab:blue", linestyle="--", linewidth=1)
            ax.axvline(bpI_star, color="tab:orange", linestyle="--", linewidth=1)
            ax.axvline(bp_star, color="black", linestyle=":", linewidth=1)
            ax.grid(True, alpha=0.25)

        fig.suptitle(
            f"{title_prefix} bp diagnostics: {label} state "
            f"(b={b_val:.2f}, z={z_val:.2f}, P={float(parent_out.P.item()):.3f}, "
            f"bar_z={float(parent_out.bar_z.item()):.3f}, "
            f"parent={'default' if parent_default_flag else 'survive'})"
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
        fig.savefig(figs_dir / f"{prefix}_bp_diag_{label}.png", dpi=150)
        plt.close(fig)

        # Separate boundary figure: emphasize the survival/default regime switch.
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
        ax_b2.set_title("Approx. FOC around boundary")

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
        fig2.suptitle(
            f"{title_prefix} bp boundary summary: {label} state "
            f"(b={b_val:.2f}, z={z_val:.2f}, parent={'default' if parent_default_flag else 'survive'})"
        )
        fig2.text(0.5, 0.955, boundary_text, ha="center", va="top", fontsize=9)
        ax_b1.grid(True, alpha=0.25)
        ax_b2.grid(True, alpha=0.25)
        fig2.tight_layout(rect=[0, 0, 1, 0.92])
        fig2.savefig(figs_dir / f"{prefix}_bp_boundary_{label}.png", dpi=150)
        plt.close(fig2)


def plot_distributions(
    ep: int,
    df: pd.DataFrame,
    pv_model: PolicyValueModel,
    device: torch.device,
    base_dir: Path,
    df_macro: pd.DataFrame | None = None,
    tag: Optional[str] = None,
):
    figs_dir = base_dir / "experiments" / "figs"
    prefix = _episode_prefix(ep, tag)
    title_prefix = _episode_title_prefix(ep, tag)

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

    m_child_source = None
    m_parent_source = None
    if df_macro is not None and not df_macro.empty and "M" in df_macro.columns:
        macro_parent_df, macro_child_df = _split_parent_child(df_macro)
        m_child_source = macro_child_df["M"].dropna()
        m_parent_source = macro_parent_df["M"].dropna()
    elif "M" in df.columns and not child_df.empty:
        m_child_source = child_df["M"].dropna()
        m_parent_source = parent_df["M"].dropna()

    if m_child_source is not None and len(m_child_source) > 0:
        plt.figure(figsize=(5, 3))
        m_child_source.hist(bins=40)
        plt.title(f"{title_prefix} M distribution (child macro states)")
        plt.xlabel("M")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(figs_dir / f"{prefix}_m_hist.png", dpi=150)
        plt.close()

    if m_parent_source is not None and len(m_parent_source) > 0:
        plt.figure(figsize=(5, 3))
        m_parent_source.hist(bins=40)
        plt.title(f"{title_prefix} M distribution (parent macro states)")
        plt.xlabel("M")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(figs_dir / f"{prefix}_m_parent_hist.png", dpi=150)
        plt.close()

    if parent_df.empty:
        return

    cols = ["b", "z", "ETA", "i", "x", "Hatcf", "LnKF"]
    X = torch.tensor(parent_df[cols].values, device=device, dtype=torch.float32)
    with torch.no_grad():
        bp = pv_model(X).bp.cpu().numpy()

    plt.figure(figsize=(5, 3))
    plt.hist(bp, bins=40)
    plt.title(f"{title_prefix} bp distribution (parent states)")
    plt.xlabel("bp")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(figs_dir / f"{prefix}_bp_hist.png", dpi=150)
    plt.close()


def plot_firm_b_window_distribution(
    df_firm: pd.DataFrame,
    base_dir: Path,
    filename: str = "final_b_hist_t50_t150_parent.png",
    t_min: int = 50,
    t_max: int = 150,
) -> None:
    if df_firm is None or df_firm.empty or "b" not in df_firm.columns or "t" not in df_firm.columns:
        return

    use_df = df_firm.copy()
    if "branch" in use_df.columns:
        if (use_df["branch"] < 0).any():
            use_df = use_df[use_df["branch"] < 0].copy()
        else:
            use_df = use_df[use_df["branch"] == 0].copy()

    use_df = use_df[(use_df["t"] >= t_min) & (use_df["t"] <= t_max)].copy()
    if use_df.empty:
        return

    b = pd.to_numeric(use_df["b"], errors="coerce").dropna()
    if b.empty:
        return

    figs_dir = base_dir / "experiments" / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    b.hist(bins=50)
    plt.title(f"b distribution (parent firm states, t={t_min}-{t_max})")
    plt.xlabel("b")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(figs_dir / filename, dpi=150)
    plt.close()


def plot_outer_drift(all_summaries, figs_dir: Path) -> None:
    """
    Plot episode-to-episode outer-drift diagnostics stored in ep_summary["outer_drift"].
    """
    if not all_summaries:
        return

    figs_dir.mkdir(parents=True, exist_ok=True)
    eps = list(range(len(all_summaries)))

    scalar_series: dict[str, list[float]] = {}
    surface_series: dict[str, list[float]] = {}

    for raw_summary in all_summaries:
        if not isinstance(raw_summary, dict):
            continue
        ep_summary = raw_summary.get("module_summaries", raw_summary)
        if not isinstance(ep_summary, dict):
            continue
        drift = ep_summary.get("outer_drift", {})
        for key, value in drift.items():
            if isinstance(value, (int, float, np.floating)):
                scalar_series.setdefault(key, []).append(float(value))
            elif isinstance(value, dict):
                target = surface_series if key == "policy_surface_drift" else scalar_series
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, (int, float, np.floating)):
                        target.setdefault(f"{key}.{sub_key}", []).append(float(sub_val))

    def _plot_group(series: dict[str, list[float]], filename: str, title: str, keys: list[str]) -> None:
        available = [k for k in keys if k in series]
        if not available:
            return
        plt.figure(figsize=(7, 4.5))
        for key in available:
            vals = series[key]
            padded = vals + [float("nan")] * (len(eps) - len(vals))
            plt.plot(eps, padded, marker="o", label=key)
        plt.xlabel("episode")
        plt.ylabel("drift")
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(figs_dir / filename, dpi=150)
        plt.close()

    _plot_group(
        scalar_series,
        "outer_drift_macro_moments.png",
        "Outer Drift: Macro Moment Changes",
        [
            "abs_delta_hatc_mean",
            "abs_delta_hatc_std",
            "abs_delta_lnk_mean",
            "abs_delta_lnk_std",
            "abs_delta_hatcf_mean",
            "abs_delta_hatcf_std",
            "abs_delta_lnkf_mean",
            "abs_delta_lnkf_std",
            "abs_delta_m_mean",
            "abs_delta_m_std",
        ],
    )
    _plot_group(
        scalar_series,
        "outer_drift_macro_xcurves.png",
        "Outer Drift: Macro X-Response Curves",
        [
            "hatc_true_x_curve_rmse",
            "hatc_pred_x_curve_rmse",
            "lnk_true_x_curve_rmse",
            "lnk_pred_x_curve_rmse",
        ],
    )
    _plot_group(
        surface_series,
        "outer_drift_policy_surfaces.png",
        "Outer Drift: Policy Surface Changes",
        [
            "policy_surface_drift.Q_rmse",
            "policy_surface_drift.P_rmse",
            "policy_surface_drift.bar_z_rmse",
            "policy_surface_drift.bp_rmse",
            "policy_surface_drift.Q_maxabs",
            "policy_surface_drift.P_maxabs",
            "policy_surface_drift.bar_z_maxabs",
            "policy_surface_drift.bp_maxabs",
        ],
    )


def plot_fc2_fit_diagnostics(
    ep: int,
    fit_df: Optional[pd.DataFrame],
    outer_df: Optional[pd.DataFrame],
    base_dir: Path,
) -> None:
    figs_dir = base_dir / "experiments" / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    def _fit_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() < 2:
            return {}
        yt = y_true[mask]
        yp = y_pred[mask]
        yt_mean = float(yt.mean())
        yp_mean = float(yp.mean())
        yt_std = float(np.std(yt))
        yp_std = float(np.std(yp))
        resid = yt - yp
        sst = float(np.sum((yt - yt_mean) ** 2))
        sse = float(np.sum(resid ** 2))
        corr = float(np.corrcoef(yt, yp)[0, 1]) if yt_std > 1e-12 and yp_std > 1e-12 else float("nan")
        slope = float("nan")
        intercept = float("nan")
        denom = float(np.sum((yt - yt_mean) ** 2))
        if denom > 1e-12:
            slope = float(np.sum((yt - yt_mean) * (yp - yp_mean)) / denom)
            intercept = float(yp_mean - slope * yt_mean)
        return {
            "r2": float("nan") if sst <= 1e-12 else float(1.0 - sse / sst),
            "corr": corr,
            "slope": slope,
            "std_ratio": float(yp_std / yt_std) if yt_std > 1e-12 else float("nan"),
        }

    def _plot_one(df: Optional[pd.DataFrame], true_col: str, pred_col: str, title: str, save_path: Path) -> None:
        if df is None or df.empty or true_col not in df.columns or pred_col not in df.columns:
            return
        x = pd.to_numeric(df[true_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            return
        x = x[mask]
        y = y[mask]
        stats = _fit_stats(x, y)
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        if hi - lo < 1e-8:
            hi = lo + 1e-4
        pad = 0.05 * (hi - lo)
        lo -= pad
        hi += pad
        plt.figure(figsize=(5.8, 5.2))
        plt.scatter(x, y, s=8, alpha=0.25, edgecolors="none")
        plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.xlabel("true")
        plt.ylabel("pred")
        if stats:
            title = (
                f"{title}\n"
                f"R2={stats.get('r2', float('nan')):.4f} | "
                f"corr={stats.get('corr', float('nan')):.4f} | "
                f"slope={stats.get('slope', float('nan')):.4f} | "
                f"stdr={stats.get('std_ratio', float('nan')):.4f}"
            )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    _plot_one(fit_df, "hatc_true", "hatc_pred", f"EP{ep} FC2 fit-after-train Hatc", figs_dir / f"ep{ep}_fc2_fit_after_train_hatc.png")
    _plot_one(fit_df, "lnk_true", "lnk_pred", f"EP{ep} FC2 fit-after-train LnK", figs_dir / f"ep{ep}_fc2_fit_after_train_lnk.png")
    _plot_one(outer_df, "hatc_true", "hatc_pred", f"EP{ep} FC2 outer-after-resim Hatc", figs_dir / f"ep{ep}_fc2_outer_after_resim_hatc.png")
    _plot_one(outer_df, "lnk_true", "lnk_pred", f"EP{ep} FC2 outer-after-resim LnK", figs_dir / f"ep{ep}_fc2_outer_after_resim_lnk.png")


def plot_fc2_consumption_distribution(
    ep: int,
    current_macro_df: Optional[pd.DataFrame],
    outer_macro_df: Optional[pd.DataFrame],
    base_dir: Path,
) -> None:
    figs_dir = base_dir / "experiments" / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    def _select_rows(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None or df.empty or 'C_raw' not in df.columns:
            return None
        return df.copy()

    def _plot_one(df: Optional[pd.DataFrame], title: str, save_path: Path) -> None:
        use_df = _select_rows(df)
        if use_df is None or use_df.empty:
            return
        groups: list[tuple[str, pd.DataFrame]]
        if 'branch' in use_df.columns:
            parent_df = use_df[use_df['branch'] == -1]
            child_df = use_df[use_df['branch'] >= 0]
            groups = []
            if not parent_df.empty:
                groups.append(("parent", parent_df))
            if not child_df.empty:
                groups.append(("children", child_df))
            if not groups:
                groups = [("all", use_df)]
        else:
            groups = [("all", use_df)]

        fig, axes = plt.subplots(1, len(groups), figsize=(7.2 * len(groups), 4.8), squeeze=False)
        fig.suptitle(title)
        for ax, (label, grp) in zip(axes[0], groups):
            c_raw = pd.to_numeric(grp['C_raw'], errors='coerce').to_numpy(dtype=float)
            c_raw = c_raw[np.isfinite(c_raw)]
            if c_raw.size == 0:
                ax.set_visible(False)
                continue
            feasible = pd.to_numeric(grp.get('feasible_raw'), errors='coerce')
            feasible_share = float(feasible.mean()) if feasible is not None else float('nan')
            neg_share = pd.to_numeric(grp.get('neg_c_mass_share'), errors='coerce')
            neg_share_mean = float(neg_share.mean()) if neg_share is not None else float('nan')
            ax.hist(c_raw, bins=40, alpha=0.8, color="#4c78a8", edgecolor="white")
            ax.axvline(0.0, color="#d62728", linestyle="--", linewidth=1.5)
            ax.set_xlabel("sum C by path-node")
            ax.set_ylabel("count")
            ax.set_title(label)
            stats_text = "\n".join([
                f"n={c_raw.size}",
                f"mean={np.mean(c_raw):.4f}",
                f"p05={np.quantile(c_raw, 0.05):.4f}",
                f"p50={np.quantile(c_raw, 0.50):.4f}",
                f"p95={np.quantile(c_raw, 0.95):.4f}",
                f"feasible={feasible_share:.3f}",
                f"neg_share={neg_share_mean:.3f}",
            ])
            ax.text(
                0.98,
                0.98,
                stats_text,
                transform=ax.transAxes,
                va="top",
                ha="right",
                fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
            )
            ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    _plot_one(
        current_macro_df,
        f"EP{ep} FC2 pretrain/input raw sum C distribution",
        figs_dir / f"ep{ep}_fc2_before_train_c_raw_distribution.png",
    )
    _plot_one(
        outer_macro_df,
        f"EP{ep} FC2 outer-after-resim raw sum C distribution",
        figs_dir / f"ep{ep}_fc2_outer_after_resim_c_raw_distribution.png",
    )


def plot_fc2_episode_metrics(all_summaries, figs_dir: Path) -> None:
    """Plot FC2 fit-after-train and outer-after-resim metrics across episodes."""
    if not all_summaries:
        return

    figs_dir.mkdir(parents=True, exist_ok=True)
    episodes = list(range(len(all_summaries)))
    tracked_metrics = ("corr", "slope", "std_ratio", "rmse")
    tracked_targets = ("hatc", "lnk")
    tracked_groups = ("fit_after_train", "outer_after_resim")
    series: dict[tuple[str, str, str], list[float]] = {}

    def _append_value(group: str, target: str, metric: str, value: float | int | np.floating | None) -> None:
        key = (group, target, metric)
        series.setdefault(key, []).append(float(value) if value is not None else float("nan"))

    for raw_summary in all_summaries:
        ep_summary = raw_summary.get("module_summaries", raw_summary) if isinstance(raw_summary, dict) else {}
        fc2_summary = ep_summary.get("fc2", {}) if isinstance(ep_summary, dict) else {}
        if not isinstance(fc2_summary, dict):
            fc2_summary = {}

        group_maps = {
            "fit_after_train": fc2_summary.get("fit_after_train", {}),
            "outer_after_resim": fc2_summary.get("outer_after_resim", {}),
        }
        for group in tracked_groups:
            metrics_map = group_maps.get(group, {})
            for target in tracked_targets:
                for metric in tracked_metrics:
                    key = f"fc2_{group}_{target}_{metric}"
                    value = metrics_map.get(key) if isinstance(metrics_map, dict) else None
                    if isinstance(value, (int, float, np.floating)):
                        _append_value(group, target, metric, value)
                    else:
                        _append_value(group, target, metric, None)

    def _plot_group(group: str, target: str) -> None:
        available = [
            metric for metric in tracked_metrics
            if (group, target, metric) in series
        ]
        if not available:
            return
        plt.figure(figsize=(7.2, 4.8))
        for metric in available:
            values = series[(group, target, metric)]
            if not any(np.isfinite(values)):
                continue
            plt.plot(episodes, values, marker="o", linewidth=1.8, label=metric)
        plt.xlabel("episode")
        plt.ylabel("metric value")
        plt.title(f"FC2 {group.replace('_', '-')} {target} metrics")
        plt.grid(True, alpha=0.25)
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(figs_dir / f"fc2_{group}_{target}_by_episode.png", dpi=150)
        plt.close()

    for group in tracked_groups:
        for target in tracked_targets:
            _plot_group(group, target)
