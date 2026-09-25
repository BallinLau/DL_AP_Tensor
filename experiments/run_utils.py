"""
Utilities for multi-episode runs: model/optimizer/hparam builders, I/O helpers, and plots.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import Config, HyperParams
from models import SDFFC1Combined, PolicyValueModel, FC2Model
from losses import P0Loss, PILoss
from utils.firm_transition import apply_refinancing_policy


POLICY_VALUE_SPEC_FILENAME = "policy_value_model_spec.json"
METADATA_DIRNAME = "metadata"
Q_PARAMETERIZATION_MODES = ("direct", "b_times_unit", "hybrid_regime")


def resolve_base_dir(run_root: Optional[Path], project_root: Path) -> Path:
    """Return run_root if provided; otherwise fall back to project_root."""
    return run_root if run_root is not None else project_root


def _policy_value_spec_candidates(ckpt_dir: Path, ckpt_prefix: Optional[str]) -> list:
    prefix = f"{ckpt_prefix}_" if ckpt_prefix else ""
    return [
        ckpt_dir / f"{prefix}{POLICY_VALUE_SPEC_FILENAME}",
        ckpt_dir.parent / METADATA_DIRNAME / f"{prefix}{POLICY_VALUE_SPEC_FILENAME}",
        ckpt_dir.parent / METADATA_DIRNAME / POLICY_VALUE_SPEC_FILENAME,
    ]


def load_policy_value_model_spec(
    ckpt_dir: Optional[Path | str],
    ckpt_prefix: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load the saved ``policy_value`` model spec for a checkpoint directory.

    Returns ``None`` when no spec file exists (raw / legacy checkpoint layout).
    """
    if ckpt_dir is None:
        return None
    ckpt_dir = Path(ckpt_dir)
    for candidate in _policy_value_spec_candidates(ckpt_dir, ckpt_prefix):
        if not candidate.exists():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("policy_value_model_spec"), dict):
            return payload["policy_value_model_spec"]
        if isinstance(payload, dict):
            return payload
    return None


def add_raw_q_parameterization_argument(parser) -> None:
    """Register the shared ``--raw-q-parameterization`` CLI flag on a parser."""
    parser.add_argument(
        "--raw-q-parameterization",
        type=str,
        default=None,
        choices=list(Q_PARAMETERIZATION_MODES),
        help=(
            "Q semantics of a metadata-less raw policy_value checkpoint. Required when "
            "metadata/policy_value_model_spec.json is absent; ignored when it is present."
        ),
    )


def resolve_raw_q_parameterization(mode: Optional[str]) -> Optional[str]:
    """Validate an explicit ``q_parameterization`` for a metadata-less checkpoint."""
    if mode is None:
        return None
    resolved = str(mode).strip().lower()
    if resolved not in Q_PARAMETERIZATION_MODES:
        raise ValueError(
            f"raw_q_parameterization must be one of {Q_PARAMETERIZATION_MODES}, got {mode!r}"
        )
    return resolved


def resolve_expected_q_parameterization(mode: Optional[str]) -> Optional[str]:
    """Validate an explicit ``expected_q_parameterization`` (resume-training guard)."""
    if mode is None:
        return None
    resolved = str(mode).strip().lower()
    if resolved not in Q_PARAMETERIZATION_MODES:
        raise ValueError(
            f"expected_q_parameterization must be one of {Q_PARAMETERIZATION_MODES}, got {mode!r}"
        )
    return resolved


def build_models(
    device: torch.device,
    ckpt_dir: Optional[Path | str] = None,
    ckpt_prefix: str | None = None,
    strict: bool = True,
    allow_unsafe_raw_checkpoint: bool = False,
    raw_q_parameterization: Optional[str] = None,
    expected_q_parameterization: Optional[str] = None,
):
    """Build the run models, optionally loading a checkpoint with verified Q semantics.

    ``policy_value`` 的 q-head 在 ``direct`` 与 ``b_times_unit`` 下 shape 相同，因此
    裸 state_dict 无法自证语义。这里把两个职责分开：

    - **evaluation / diagnostics**：``expected_q_parameterization=None``（默认）。
      有 ``policy_value_model_spec`` 时完全相信 checkpoint，按 spec 的
      ``q_parameterization`` 构造模型，**不要求**与 ``Config.Q_PARAMETERIZATION`` 一致。
    - **resume training**：显式传入 ``expected_q_parameterization``。checkpoint 的
      resolved 语义（spec 或 ``raw_q_parameterization``）必须与它相等，否则拒绝。

    无 spec 的裸 checkpoint 仍必须显式给出 ``raw_q_parameterization``，不猜测。
    ``Config.Q_PARAMETERIZATION`` 只决定**未加载 checkpoint 时** fresh model 的默认
    architecture。``allow_unsafe_raw_checkpoint`` 仅为向后兼容保留，不能单独决定语义。
    """
    raw_mode = resolve_raw_q_parameterization(raw_q_parameterization)
    expected_mode = resolve_expected_q_parameterization(expected_q_parameterization)
    spec = load_policy_value_model_spec(ckpt_dir, ckpt_prefix) if ckpt_dir is not None else None
    spec_q = None
    if spec is not None:
        # spec 决定 architecture；缺失 q_parameterization 视为 legacy b_times_unit。
        spec_q = str(spec.get("q_parameterization", "b_times_unit")).lower()
    # checkpoint 自己的语义优先；Config 只在没有 checkpoint 语义时决定 fresh model。
    resolved_q = spec_q if spec_q is not None else raw_mode

    models = {
        "sdf_fc1": SDFFC1Combined(
            sdf_input_dim=Config.FC1_INPUT_DIM,
            fc1_input_dim=Config.FC1_INPUT_DIM,
            sdf_hidden_dims=Config.SDF_HIDDEN_DIMS,
            fc1_hidden_dims=Config.FC1_HIDDEN_DIMS,
            w_hidden_dims=Config.SDF_HIDDEN_DIMS,
        ).to(device),
        "policy_value": (
            PolicyValueModel(q_parameterization=resolved_q).to(device)
            if resolved_q is not None
            else PolicyValueModel().to(device)
        ),
        "fc2": FC2Model(
            input_dim=Config.FC2_INPUT_DIM,
            hidden_dims=Config.FC2_HIDDEN_DIMS,
            quantile_num=Config.QUANTILE_NUM,
        ).to(device),
    }

    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)
        prefix = f"{ckpt_prefix}_" if ckpt_prefix else ""
        pv_ckpt_path = ckpt_dir / f"{prefix}policy_value.pt"
        if spec_q is None and pv_ckpt_path.exists() and raw_mode is None:
            # 不许猜：legacy b_times_unit 与 direct-Q 的 q-head shape 相同，
            # 按当前 Config 强行 load 会静默改变 Q 的经济含义。
            raise ValueError(
                "Raw policy_value checkpoint without policy_value_model_spec: "
                f"{pv_ckpt_path}. The q-head of a legacy b_times_unit state_dict is "
                "shape-compatible with direct-Q, so the semantic mode must be stated "
                "explicitly. Pass raw_q_parameterization='direct' or "
                "raw_q_parameterization='b_times_unit'"
                + (
                    " (allow_unsafe_raw_checkpoint=True alone is not sufficient)."
                    if allow_unsafe_raw_checkpoint
                    else ", or save metadata/policy_value_model_spec.json."
                )
            )
        # resume-training guard：只有显式给出 expected 时才检查，evaluation 不受 Config 影响。
        if expected_mode is not None and resolved_q is not None and resolved_q != expected_mode:
            source = "checkpoint spec" if spec_q is not None else "raw_q_parameterization"
            raise ValueError(
                f"policy_value q_parameterization mismatch: {source} declares "
                f"{resolved_q!r} but expected_q_parameterization={expected_mode!r} "
                f"({pv_ckpt_path}). Resuming training across a Q semantic change is not "
                "allowed; explicit migration is required."
            )
        if spec_q is not None:
            print(
                "[build_models] policy_value_model_spec declares "
                f"q_parameterization={spec_q!r}; using the checkpoint semantics "
                f"({pv_ckpt_path})."
            )
        elif pv_ckpt_path.exists():
            print(
                "[build_models] raw policy_value checkpoint without "
                f"policy_value_model_spec ({pv_ckpt_path}); using explicit "
                f"raw_q_parameterization={raw_mode!r}."
            )
        mapping = {
            "sdf_fc1": "sdf_fc1",
            "policy_value": "policy_value",
            "fc2": "fc2",
        }
        for key, stem in mapping.items():
            ckpt_path = ckpt_dir / f"{prefix}{stem}.pt"
            if ckpt_path.exists():
                state = torch.load(ckpt_path, map_location=device)
                models[key].load_state_dict(state, strict=strict)
                print(f"[build_models] loaded {ckpt_path}")
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
    # SDF 两阶段稳定配置：calculated-state one-step 是主监督；
    # forecast-state reconstruction 默认关闭，仅作为显式对照开关保留。
    hp.sdf_training_schedule_enabled = True
    hp.fc1_only_epochs = 10
    hp.sdf_true_only_epochs = 5
    hp.sdf_true_start_episode = 2
    hp.sdf_recursive_only_epochs = 0
    hp.sdf_wealth_residual_mode = "normalized_ratio"
    hp.sdf_gate_residual_mode = "normalized_ratio"
    hp.sdf_normalized_logr_clip = 20.0
    hp.fc1_recon_weight = 1.0
    hp.fc1_recursive_aux_training_enabled = False
    hp.fc1_rollout_diagnostic_enabled = True
    hp.fc1_forecast_recon_weight = 0.0
    hp.fc1_rollout_weight = 0.0
    hp.fc1_rollout_horizon = 5
    hp.fc1_hatc_recon_weight = 1.0
    hp.fc1_lnk_recon_weight = 0.25
    hp.fc1_delta_penalty_weight = 0.0
    hp.fc1_delta_hatc_abs_max = 0.50
    hp.fc1_delta_lnk_abs_max = 0.30
    hp.fc1_jacobian_penalty_weight = 0.0
    hp.fc1_jacobian_penalty_interval = 10
    hp.fc1_use_true_macro_state_in_stage2 = True
    hp.sdf_euler_weight = 1.0
    hp.sdf_true_moment_weight = 5e-4
    hp.sdf_true_anchor_weight = 0.05
    hp.sdf_moment_constraint_mode = "augmented_lagrangian"
    hp.sdf_al_rho = 2.0
    hp.sdf_al_primal_epochs_per_dual_update = 5
    hp.sdf_al_lambda_init = 0.0
    hp.sdf_al_eps = 1e-8
    hp.sdf_al_gate_tolerance = 0.0
    hp.sdf_continue_constraint_tol = 0.02
    hp.sdf_aio_progress_ratio_max = 0.80
    hp.sdf_aio_good_abs_tol = 1e-3
    hp.sdf_final_constraint_tol = 1e-3
    hp.sdf_final_max_signed_t_abs = 2.0
    hp.sdf_final_aio_mean_tol = 1e-3
    hp.sdf_al_dual_max_batches = 0
    hp.sdf_al_reset_on_true_start = True
    hp.sdf_al_strict_semantics_guard = True
    hp.sdf_al_noise_tax_warn_ratio = 1.0
    hp.sdf_recursive_loss_weight = 0.25
    hp.sdf_recursive_moment_weight = 5e-4
    hp.sdf_recursive_anchor_weight = 0.05
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
    hp.fc1_epochs_per_round = 0
    hp.fc1_max_rounds = 8
    hp.fc1_plateau_patience = 2
    hp.fc1_min_relative_improvement = 0.01
    hp.fc1_gate_min_pairs = 128
    hp.fc1_target_std_floor = 1e-4
    hp.fc1_one_step_r2_min = 0.0
    hp.fc1_one_step_skill_min = 0.0
    hp.fc1_one_step_hatc_rmse_abs_max = 0.05
    hp.fc1_one_step_lnk_rmse_abs_max = 0.05
    hp.fc1_persistence_rmse_floor = 1e-6
    hp.fc1_rollout_finite_ratio_min = 1.0
    hp.sdf_fc1_val_fraction = 0.2
    hp.sdf_fc1_val_seed = 12345
    hp.sdf_gate_m_finite_ratio_min = 1.0
    hp.sdf_gate_m_p99_max = float("inf")
    hp.sdf_gate_m_max_max = float("inf")
    hp.episode0_sdf_epochs_per_round = 0
    hp.episode0_sdf_max_rounds = 10
    hp.episode0_sdf_log_mean_error_max = 0.25
    hp.episode0_sdf_clip_low_ratio_max = 0.20
    hp.episode0_sdf_finite_ratio_min = 1.0
    hp.allow_in_sample_sdf_gate_for_debug = False
    hp.sdf_post_refresh_gate_enabled = True
    hp.sdf_post_refresh_gate_mode = "safety"
    hp.stage_gate_required_consecutive_passes = 1
    hp.sdf_stage1_lr = 1e-4
    hp.sdf_stage2_lr = 2e-4
    hp.sdf_true_only_lr = float(os.environ.get("SDF_TRUE_ONLY_LR", 4e-5))
    hp.sdf_true_target_batches = int(os.environ.get("SDF_TRUE_TARGET_BATCHES", 20))
    hp.sdf_min_parent_groups_per_batch = int(
        os.environ.get("SDF_MIN_PARENT_GROUPS_PER_BATCH", 256)
    )
    hp.sdf_stage1_moment_weight = 5.0
    hp.sdf_moment_weight = 5.0
    hp.sdf_log_mean_anchor_weight_stage1 = 1.0
    hp.sdf_log_mean_anchor_weight_stage2 = 5.0
    # 先关闭 bp 的额外边界推进，避免 bp 长期贴到 1
    hp.bp_adaptive_enabled = False
    hp.bp_refine_steps_per_epoch = 0
    # P0/PI 对上游 M 的鲁棒化（避免 M 偏高直接抬高 P）
    hp.pv_use_clipped_m = True
    hp.pv_m_clamp_min = 0.7
    hp.pv_m_clamp_max = 1.3
    hp.bp_grid_parent_chunk_size = 2048
    hp.bp_grid_candidate_chunk_size = 0
    hp.bp_grid_max_expanded_states = 65536
    return hp


def ensure_dirs(base_dir: Path):
    (base_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (base_dir / "data" / "outputs").mkdir(parents=True, exist_ok=True)
    (base_dir / "experiments" / "figs").mkdir(parents=True, exist_ok=True)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _git_commit_hash() -> Optional[str]:
    try:
        import subprocess

        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(Path(__file__).resolve().parent),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def save_checkpoint_metadata(
    models,
    episode: int,
    base_dir: Path,
    *,
    hyperparams: Optional[HyperParams] = None,
    config_snapshot: Optional[Dict[str, Any]] = None,
    value_parameterization: Optional[Dict[str, Any]] = None,
    extra_models: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Write ``metadata/`` (hyperparams / config snapshot / model spec) + combined checkpoint.

    裸 state_dict 无法定义 direct-Q 与 legacy ``b q_unit`` 的区别（q-head shape 相同），
    因此 strict resume / evaluation 需要这里保存的 ``policy_value_model_spec``。
    """
    from analysis.economic_config import AnalysisEconomicConfig

    base_dir = Path(base_dir)
    meta_dir = base_dir / METADATA_DIRNAME
    ckpt_dir = base_dir / "checkpoints"
    meta_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if hyperparams is None:
        hyperparams = HyperParams()
    if config_snapshot is None:
        config_snapshot = AnalysisEconomicConfig.from_current_config().to_dict()
    if value_parameterization is None:
        # 实际训练的 value parameterization 由 HyperParams 配置给 PolicyValueModel /
        # Episode；Config 上的 PV_VALUE_SCALE_* 从不被 configure_hyperparams 回写，
        # 用 Config 推断会写出自相矛盾的 metadata（hyperparams=exp_xz / mode=none）。
        mode = str(getattr(hyperparams, "pv_value_scale_mode", "none")).lower()
        log_max = float(getattr(hyperparams, "pv_value_scale_log_max", 20.0))
        value_parameterization = {
            "mode": mode,
            "scale_formula": f"1+exp(clamp(x+z,max={log_max:g}))" if mode == "exp_xz" else "1",
            "bellman_normalization": bool(
                getattr(hyperparams, "pv_bellman_normalize_by_value_scale", False)
            ),
            "log_max": log_max,
        }

    model_spec = models["policy_value"].model_spec()
    hyperparams_path = meta_dir / "hyperparams.json"
    config_path = meta_dir / "config_snapshot.json"
    spec_path = meta_dir / POLICY_VALUE_SPEC_FILENAME
    combined_path = ckpt_dir / f"ep{episode}_combined.pt"

    hyperparams_path.write_text(
        json.dumps(_jsonable(vars(hyperparams)), indent=2), encoding="utf-8"
    )
    config_path.write_text(
        json.dumps(_jsonable(config_snapshot), indent=2), encoding="utf-8"
    )
    spec_path.write_text(
        json.dumps({"policy_value_model_spec": model_spec}, indent=2), encoding="utf-8"
    )

    payload_models = {
        name: model.state_dict()
        for name, model in {**models, **(extra_models or {})}.items()
        if model is not None
    }
    payload = {
        "models": payload_models,
        "hyperparams": vars(hyperparams),
        "config_snapshot": config_snapshot,
        "policy_value_model_spec": model_spec,
        "value_parameterization": value_parameterization,
        "episode": int(episode),
        "git_commit": _git_commit_hash(),
    }
    if extra_metadata:
        payload.update(extra_metadata)
    torch.save(payload, combined_path)
    return {
        "hyperparams": str(hyperparams_path),
        "config_snapshot": str(config_path),
        "policy_value_model_spec": str(spec_path),
        "combined_checkpoint": str(combined_path),
    }


def save_models(
    models,
    episode: int,
    base_dir: Path,
    *,
    hyperparams: Optional[HyperParams] = None,
    config_snapshot: Optional[Dict[str, Any]] = None,
    value_parameterization: Optional[Dict[str, Any]] = None,
    extra_models: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Save raw module checkpoints **and** the semantic metadata that defines them."""
    base_dir = Path(base_dir)
    ckpt_dir = base_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(models["sdf_fc1"].state_dict(), ckpt_dir / f"ep{episode}_sdf_fc1.pt")
    torch.save(models["policy_value"].state_dict(), ckpt_dir / f"ep{episode}_policy_value.pt")
    torch.save(models["fc2"].state_dict(), ckpt_dir / f"ep{episode}_fc2.pt")
    return save_checkpoint_metadata(
        models,
        episode,
        base_dir,
        hyperparams=hyperparams,
        config_snapshot=config_snapshot,
        value_parameterization=value_parameterization,
        extra_models=extra_models,
        extra_metadata=extra_metadata,
    )


def save_stage_df(ep: int, name: str, base_dir: Path, df_firm: pd.DataFrame = None, df_macro: pd.DataFrame = None, df_sdf: pd.DataFrame = None):
    out_dir = base_dir / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    target_df = df_firm if df_firm is not None else df_sdf
    if target_df is not None:
        target_df.to_pickle(out_dir / f"ep{ep}_stage_{name}.pkl")
    if df_macro is not None:
        df_macro.to_pickle(out_dir / f"ep{ep}_stage_{name}_macro.pkl")


def _compute_policy_diagnostic_surfaces(
    pv_model: PolicyValueModel,
    sdf_model: SDFFC1Combined | None,
    base: torch.Tensor,
    *,
    eta_next_assumed: float,
):
    """Compute a one-child diagnostic conditional on an explicit child eta."""
    p0_loss = P0Loss()
    pi_loss = PILoss()

    out = pv_model(base)
    bp0 = out.bp0
    bpI = out.bpI
    eta_next = torch.full_like(base[:, 2:3], float(eta_next_assumed))

    child_p0_state = base.clone()
    child_p0_state[:, 0:1] = apply_refinancing_policy(
        b_current=base[:, 0:1],
        bp_candidate=bp0,
        eta_current=base[:, 2:3],
    )
    child_p0_state[:, 2:3] = eta_next
    child_pI_state = base.clone()
    child_pI_state[:, 0:1] = apply_refinancing_policy(
        b_current=base[:, 0:1],
        bp_candidate=bpI,
        eta_current=base[:, 2:3],
    )
    child_pI_state[:, 2:3] = eta_next

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
):
    figs_dir = base_dir / "experiments" / "figs"
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
        # Legacy surface: continuation is conditional on a refinancing-active child.
        out, diagnostics = _compute_policy_diagnostic_surfaces(
            pv_model,
            sdf_model,
            base,
            eta_next_assumed=1.0,
        )
        P0 = out.P0.reshape(B.shape).cpu().numpy()
        PI = out.PI.reshape(B.shape).cpu().numpy()
        P = out.P.reshape(B.shape).cpu().numpy()
        bar_i = out.bar_i.reshape(B.shape).cpu().numpy()
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
        ("bari", bar_i),
        ("barz", bar_z),
        ("bp", bp),
        ("q", Q),
        ("pidiff", pi_diff),
        ("cfdiff", cf_diff),
        ("contdiff", cont_diff),
    ]:
        plot_arr = arr
        if name in {"bari", "bp", "pidiff", "cfdiff", "contdiff"}:
            plot_arr = np.where(survive_mask, arr, np.nan)
        plt.figure(figsize=(6, 4))
        cs = plt.contourf(B.cpu().numpy(), Z.cpu().numpy(), plot_arr, levels=30, cmap="viridis")
        plt.colorbar(cs)
        plt.xlabel("b")
        plt.ylabel("z")
        plt.title(f"EP{ep} {name.upper()} heatmap")
        plt.tight_layout()
        plt.savefig(figs_dir / f"ep{ep}_{name}_heatmap.png", dpi=150)
        plt.close()

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(B.cpu().numpy(), Z.cpu().numpy(), plot_arr, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel("b")
        ax.set_ylabel("z")
        ax.set_zlabel(name.upper())
        ax.set_title(f"EP{ep} {name.upper()} surface")
        plt.tight_layout()
        plt.savefig(figs_dir / f"ep{ep}_{name}_surface.png", dpi=150)
        plt.close()


def plot_distributions(
    ep: int,
    df: pd.DataFrame,
    pv_model: PolicyValueModel,
    device: torch.device,
    base_dir: Path,
    df_macro: pd.DataFrame | None = None,
):
    figs_dir = base_dir / "experiments" / "figs"

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
        plt.title(f"EP{ep} M distribution (child macro states)")
        plt.xlabel("M")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(figs_dir / f"ep{ep}_m_hist.png", dpi=150)
        plt.close()

    if m_parent_source is not None and len(m_parent_source) > 0:
        plt.figure(figsize=(5, 3))
        m_parent_source.hist(bins=40)
        plt.title(f"EP{ep} M distribution (parent macro states)")
        plt.xlabel("M")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(figs_dir / f"ep{ep}_m_parent_hist.png", dpi=150)
        plt.close()

    if parent_df.empty:
        return

    cols = ["b", "z", "ETA", "i", "x", "Hatcf", "LnKF"]
    X = torch.tensor(parent_df[cols].values, device=device, dtype=torch.float32)
    with torch.no_grad():
        bp = pv_model(X).bp.cpu().numpy()

    plt.figure(figsize=(5, 3))
    plt.hist(bp, bins=40)
    plt.title(f"EP{ep} bp distribution (parent states)")
    plt.xlabel("bp")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(figs_dir / f"ep{ep}_bp_hist.png", dpi=150)
    plt.close()


def plot_macro_series(ep: int, df_macro: pd.DataFrame, base_dir: Path):
    if df_macro is None or df_macro.empty:
        return

    def _pick_col(df: pd.DataFrame, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() < 2:
            return float("nan")
        yt = y_true[mask]
        yp = y_pred[mask]
        sst = float(np.sum((yt - yt.mean()) ** 2))
        if sst <= 1e-12:
            return float("nan")
        sse = float(np.sum((yt - yp) ** 2))
        return 1.0 - sse / sst

    def _plot_scatter_with_identity(
        x_true: np.ndarray,
        y_pred: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        save_path: Path,
        branch_vals: np.ndarray | None = None
    ) -> None:
        mask = np.isfinite(x_true) & np.isfinite(y_pred)
        if mask.sum() < 2:
            return
        x = x_true[mask]
        y = y_pred[mask]
        b = branch_vals[mask] if branch_vals is not None else None
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return
        if hi - lo < 1e-8:
            hi = lo + 1e-4
        pad = 0.05 * (hi - lo)
        lo -= pad
        hi += pad
        plt.figure(figsize=(5, 5))
        if b is not None and len(b) == len(x):
            unique_branch = sorted(pd.unique(pd.Series(b).dropna()))
            if unique_branch:
                for br in unique_branch:
                    br_mask = (b == br)
                    if np.any(br_mask):
                        br_r2 = _safe_r2(x[br_mask], y[br_mask])
                        br_label = f"branch={int(br)}" if float(br).is_integer() else f"branch={br}"
                        if np.isfinite(br_r2):
                            br_label += f" (R2={br_r2:.4f})"
                        plt.scatter(
                            x[br_mask],
                            y[br_mask],
                            s=9,
                            alpha=0.28,
                            edgecolors="none",
                            label=br_label
                        )
            else:
                plt.scatter(x, y, s=8, alpha=0.25, edgecolors="none", label="samples")
        else:
            plt.scatter(x, y, s=8, alpha=0.25, edgecolors="none", label="samples")
        plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y=x")
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    figs_dir = base_dir / "experiments" / "figs"
    use_df = df_macro
    branch_arr = use_df["branch"].to_numpy() if "branch" in use_df.columns else None

    hatc_true_col = _pick_col(use_df, ["Hatc", "hatc"])
    lnk_true_col = _pick_col(use_df, ["LnK", "lnk"])
    hatc_pred_col = _pick_col(use_df, ["hatcf", "Hatcf"])
    lnk_pred_col = _pick_col(use_df, ["lnkf", "LnKF"])

    if hatc_true_col is None or lnk_true_col is None:
        return

    r2_hatc = float("nan")
    r2_lnk = float("nan")
    if hatc_pred_col is not None:
        r2_hatc = _safe_r2(use_df[hatc_true_col].to_numpy(), use_df[hatc_pred_col].to_numpy())
    if lnk_pred_col is not None:
        r2_lnk = _safe_r2(use_df[lnk_true_col].to_numpy(), use_df[lnk_pred_col].to_numpy())

    if hatc_pred_col is not None:
        title_hatc = f"EP{ep} Hatc pred vs true"
        if np.isfinite(r2_hatc):
            title_hatc += f" (R2={r2_hatc:.4f})"
        _plot_scatter_with_identity(
            use_df[hatc_true_col].to_numpy(),
            use_df[hatc_pred_col].to_numpy(),
            title_hatc,
            "Hatc true",
            "Hatc pred",
            figs_dir / f"ep{ep}_macro_hatc.png",
            branch_vals=branch_arr
        )

    if lnk_pred_col is not None:
        title_lnk = f"EP{ep} LnK pred vs true"
        if np.isfinite(r2_lnk):
            title_lnk += f" (R2={r2_lnk:.4f})"
        _plot_scatter_with_identity(
            use_df[lnk_true_col].to_numpy(),
            use_df[lnk_pred_col].to_numpy(),
            title_lnk,
            "LnK true",
            "LnK pred",
            figs_dir / f"ep{ep}_macro_lnk.png",
            branch_vals=branch_arr
        )

    # Delta 图仍按 t 聚合均值构造，但使用全样本口径（不筛 branch）
    if "t" not in use_df.columns:
        return
    series_hatc = use_df.groupby("t")[hatc_true_col].mean().reset_index().rename(columns={hatc_true_col: "hatc_true"})
    series_lnk = use_df.groupby("t")[lnk_true_col].mean().reset_index().rename(columns={lnk_true_col: "lnk_true"})
    series = pd.merge(series_hatc, series_lnk, on="t", how="inner").sort_values("t").reset_index(drop=True)
    if hatc_pred_col is not None:
        s_hatc_pred = use_df.groupby("t")[hatc_pred_col].mean().reset_index().rename(columns={hatc_pred_col: "hatc_pred"})
        series = pd.merge(series, s_hatc_pred, on="t", how="left")
    if lnk_pred_col is not None:
        s_lnk_pred = use_df.groupby("t")[lnk_pred_col].mean().reset_index().rename(columns={lnk_pred_col: "lnk_pred"})
        series = pd.merge(series, s_lnk_pred, on="t", how="left")

    # Delta lnK_t = lnK_t - lnK_{t-1}
    series["d_lnk_true"] = series["lnk_true"].diff()
    if "lnk_pred" in series.columns:
        series["d_lnk_pred"] = series["lnk_pred"].diff()
    d_lnk = series.dropna(subset=["d_lnk_true"])

    plt.figure(figsize=(6, 3))
    plt.plot(d_lnk["t"], d_lnk["d_lnk_true"], marker="o", label="Delta LnK true")
    if "d_lnk_pred" in d_lnk.columns:
        plt.plot(d_lnk["t"], d_lnk["d_lnk_pred"], marker="x", linestyle="--", label="Delta LnK pred")
    plt.xlabel("t")
    plt.ylabel("Delta LnK")
    plt.title(f"EP{ep} Delta LnK vs t")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / f"ep{ep}_macro_delta_lnk.png", dpi=150)
    plt.close()

    # Delta lnC_t with lnC_t = Hatc_t + LnK_t
    series["lnc_true"] = series["hatc_true"] + series["lnk_true"]
    series["d_lnc_true"] = series["lnc_true"].diff()
    if "hatc_pred" in series.columns and "lnk_pred" in series.columns:
        series["lnc_pred"] = series["hatc_pred"] + series["lnk_pred"]
        series["d_lnc_pred"] = series["lnc_pred"].diff()
    d_lnc = series.dropna(subset=["d_lnc_true"])

    plt.figure(figsize=(6, 3))
    plt.plot(d_lnc["t"], d_lnc["d_lnc_true"], marker="o", label="Delta lnC true")
    if "d_lnc_pred" in d_lnc.columns:
        plt.plot(d_lnc["t"], d_lnc["d_lnc_pred"], marker="x", linestyle="--", label="Delta lnC pred")
    plt.xlabel("t")
    plt.ylabel("Delta lnC (Hatc + LnK)")
    plt.title(f"EP{ep} Delta lnC vs t")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / f"ep{ep}_macro_delta_lnc.png", dpi=150)
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
