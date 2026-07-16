"""
超参数配置类
用于管理训练过程中的超参数
"""

from dataclasses import dataclass, field
from typing import List, Optional
import math
import torch


@dataclass
class HyperParams:
    """
    训练超参数配置
    使用 dataclass 便于序列化和修改
    """
    
    # ========== 训练基础参数 ==========
    epochs: int = 100
    batch_size: int = 8192
    # Stage-specific batch sizes.
    #
    # A value <= 0 means falling back to the legacy/global batch_size.
    # These values are normally resolved by the CLI runner.
    pv_batch_size: int = 0
    sdf_fc1_batch_size: int = 0
    
    # ========== 优化器参数 ==========
    # SDF & FC1
    sdf_lr: float = 5e-4
    sdf_weight_decay: float = 1e-3
    fc1_lr: float = 1e-3
    fc1_weight_decay: float = 1e-4
    # 通用学习率（调度器基准值，缺省时沿用 sdf_lr）
    lr: float = 5e-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    
    # Policy & Value
    policy_lr: float = 1e-3
    policy_weight_decay: float = 1e-6
    # Policy/Value 消融实验：baseline, bellman_only, fixed_sdf, fixed_policy
    ablation_mode: str = 'baseline'
    policy_value_bellman_only: bool = False
    pv_fixed_sdf: bool = False
    pv_fixed_sdf_value: float = 0.98
    pv_fixed_policy: bool = False
    pv_fixed_policy_mode: str = 'parent_b'
    
    # FC2
    fc2_lr: float = 1e-4
    fc2_weight_decay: float = 1e-4
    
    # ========== 学习率调度 ==========
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 2
    lr_min: float = 1e-6
    # 通用学习率下限（LearningRateScheduler 使用）
    min_lr: float = 1e-6
    
    # 动态学习率冷却
    auto_cooldown_threshold: float = 0.8
    auto_cooldown_factor: float = 0.5
    lr_floor: float = 1e-5
    warmup_steps: int = 0
    max_steps: int = 20000
    
    # ========== 梯度处理 ==========
    max_grad_norm: float = 1.0
    fc2_max_grad_norm: float = 10.0
    
    # ========== 损失函数参数 ==========
    # AIO 权重（动态残差插值）
    aio_weight: float = 0.5
    aio_warmup_epochs: int = 10
    # SDF wealth equation objective. Formal Mode B defaults to signed AiO;
    # legacy_abs_log1p remains available only as an explicit ablation choice.
    # "legacy_abs_log1p": E[log(1 + |r1*r2|)]
    # "signed_aio": E[r1*r2]
    sdf_wealth_loss_mode: str = "signed_aio"
    # Residual scale is separate from residual aggregation. The normalized-AiO
    # diagnostic experiment uses normalized_ratio. Set raw explicitly only for
    # legacy comparison.
    # "raw": A_j*w_{t+1,j}^kappa - (w_t-exp(c_t))^kappa
    # "normalized_ratio": A_j*(w_{t+1,j}/(w_t-exp(c_t)))^kappa - 1
    sdf_wealth_residual_mode: str = "normalized_ratio"
    sdf_normalized_logr_clip: float = 20.0
    sdf_gate_residual_mode: str = "normalized_ratio"
    # Signed AiO fresh-pair controls.  When enabled, wealth Euler residuals
    # use independently sampled AR(1) aggregate shocks; fixed Treatment B
    # children still supply FC1 reconstruction targets.
    sdf_fresh_pair_enabled: bool = True
    sdf_child_bank_size: int = 16
    sdf_child_bank_refresh_epochs: int = 1
    sdf_child_bank_seed: int = 12345
    sdf_child_bank_wealth_only: bool = True
    sdf_signed_aio_n_children: int = 2
    
    # 损失权重
    w_sdf: float = 1.0
    w_p0: float = 1.0
    w_pi: float = 1.0
    w_q: float = 1.0
    w_fc2: float = 1.0
    lambda_sdf: float = 1.0
    lambda_fc: float = 1.0
    lambda_warmup_epochs: int = 5
    
    # FC2 跨期一致性
    lambda_trans: float = 0.0
    lambda_trans_max: float = 1.0
    lambda_trans_warmup_epochs: int = 20
    
    # ========== 训练策略 ==========
    # 多阶段训练
    stage1_epochs: int = 30  # Q 训练
    stage2_epochs: int = 30  # P0/PI 训练
    stage3_epochs: int = 20  # bp0/bpI 训练
    stage4_epochs: int = 20  # 联合微调
    
    # L-BFGS 精调
    use_lbfgs: bool = False
    lbfgs_iters: int = 100
    lbfgs_lr: float = 1.0
    
    # 小 b 微调
    sb_steps: int = 350
    sb_margin: float = 0.07
    sb_penalty_weight: float = 10.0
    
    # ========== 稳定性保障 ==========
    # 损失爆炸防护
    loss_explosion_threshold: float = 100.0
    stage_fail_on_policy_value_explosion: bool = True
    policy_value_loss_fail_threshold: float = 1000.0
    policy_value_grad_fail_threshold: float = 1000.0
    policy_value_loss_relative_fail_multiplier: float = 10.0
    policy_value_grad_relative_fail_multiplier: float = 10.0
    policy_value_rolling_grad_fail_threshold: float = 100.0
    pv_grad_clip_norm: float = 10.0
    pv_grad_soft_threshold: float = 100.0
    pv_grad_hard_threshold: float = 1000.0
    pv_loss_hard_threshold: float = 1000.0
    pv_epoch_max_hard_spikes: int = 3
    pv_epoch_max_skip_ratio: float = 0.05
    pv_epoch_max_consecutive_soft_spikes: int = 3
    pv_epoch_max_retries: int = 1
    pv_retry_lr_decay: float = 0.3
    pv_continue_after_degraded_stage: bool = True
    pv_max_consecutive_degraded_episodes: int = 3
    pv_training_flow: str = "joint"
    pv_eval_epochs: Optional[int] = None
    bp_distill_epochs: int = 20
    bp_distill_patience: int = 3
    bp_distill_min_delta: float = 1e-4
    pv_eval_grad_clip_norm: float = 10.0
    bp_distill_grad_clip_norm: float = 10.0
    pv_rollback_on_soft_spikes: bool = False
    bp_label_cache_mode: str = "cpu"
    # P/Q value target cache validation.  "metadata" is the formal fast path:
    # validate cache length, batch ids, target/source shapes, teacher hash and
    # grid config hash without re-hashing full parent/child/M tensors.  Use
    # "full" for debug/tests when parent/child/M hashes should be recomputed.
    # "off" keeps structural checks only: cache length, batch ids, and target
    # shapes.
    pq_cache_integrity_check: str = "metadata"
    bp_distill_trainable_scope: str = "heads_only"
    # Optional Policy/Value mixture sampling for episode e>0.  When enabled,
    # refreshed SimulateTS parent groups are mixed with coverage Sample parent
    # groups after the post-refresh SDF/FC1 safety gate and before P/Q-BP
    # staged training.  Disabled by default to preserve the legacy path.
    pv_mixture_enabled: bool = False
    pv_mixture_ratio: float = 0.20
    pv_mixture_start_episode: int = 1
    pv_mixture_budget_mode: str = "fixed_total"
    pv_mixture_sampling_mode: str = "uniform"
    pv_mixture_coverage_group_size: int = 2
    pv_mixture_seed: int = 24680
    pv_mixture_stratified_validation: bool = True
    pv_mixture_preserve_rng: bool = True
    
    # NaN/Inf 检测
    nan_recovery: bool = True
    # 连续出现非有限梯度时 fail-fast，避免整轮 stage 空跑。
    nonfinite_grad_fail_after: int = 3
    nonfinite_grad_skip_step: bool = True
    
    # 早停
    early_stop_threshold: float = 0.1
    high_loss_lr_switch: float = 10.0
    high_loss_lr: float = 1e-4
    
    # ========== 数据生成参数 (INCREASED for higher memory usage) ==========
    n_samples: int = 100000
    n_paths: int = 2000
    simulate_horizon: int = 200
    # 训练数据批次是否优先走 tensor 管线（避免训练前 pandas 拼装）
    use_tensor_pipeline: bool = True
    # firm-level 训练每个 stage 最多使用多少 parent transitions；<=0 表示不截断。
    max_firm_train_units: int = 1_000_000

    # Stage2 true-state 重建损失权重：
    # (Hatc_t, LnK_t) -> (Hatc_{t+1}, LnK_{t+1})
    # 这是 Gomes 口径下识别 FC1 law of motion 的主要监督。
    sdf_training_schedule_enabled: bool = True
    fc1_only_epochs: int = 10
    sdf_true_only_epochs: int = 5
    sdf_recursive_only_epochs: int = 0
    fc1_recon_weight: float = 1.0
    # Optional forecast-state auxiliary reconstruction kept as an explicit
    # opt-in diagnostic. The primary FC1 law of motion is calculated-state
    # one-step supervision: (Hatc_t, LnK_t) -> (Hatc_{t+1}, LnK_{t+1}).
    fc1_recursive_aux_training_enabled: bool = False
    fc1_rollout_diagnostic_enabled: bool = True
    fc1_forecast_recon_weight: float = 0.0
    fc1_rollout_weight: float = 0.0
    fc1_rollout_horizon: int = 5
    # FC1 重建项内部按目标拆分权重。
    # 经验上 LnK 的原始尺度波动更大，若不单独降权，容易主导 FC1 训练并把 M 分布拉坏。
    fc1_hatc_recon_weight: float = 1.0
    fc1_lnk_recon_weight: float = 0.25
    # Forecast-state 一步增量幅度约束。
    # 不预设方向，只惩罚过大的单步跳跃，避免递推响应面把 child state 撕裂成多个 regime。
    fc1_delta_penalty_weight: float = 0.0
    fc1_delta_hatc_abs_max: float = 0.50
    fc1_delta_lnk_abs_max: float = 0.30
    # Forecast-state 响应面局部平滑约束（Jacobian penalty）。
    # 用于抑制 FC1 对 (hatcf_prev, lnkf_prev) 的过强局部敏感性，减少 child state regime splitting。
    fc1_jacobian_penalty_weight: float = 0.0
    # Jacobian penalty 涉及二阶反传，默认每 10 个 SDF/FC1 optimizer step 计算一次；<=0 表示禁用该项。
    fc1_jacobian_penalty_interval: int = 10
    # Stage2 先做若干轮 FC1 teacher forcing 预训练（使用真实 Hatc_t/LnK_t 输入）
    fc1_teacher_forcing_epochs: int = 5
    fc1_teacher_forcing_weight: float = 1.0
    # 在 stage2/joint 中，若 batch 提供真实 Hatc_t/LnK_t，优先用真实当前态驱动 FC1。
    # forecast-state 只作为辅助递推稳定项，不作为主输入口径。
    fc1_use_true_macro_state_in_stage2: bool = True
    # SDF 矩约束权重（常规阶段）
    sdf_moment_weight: float = 1.0
    # SDF 第一阶段（无 FC1 监督）专用学习率与矩约束权重
    sdf_stage1_lr: float = 1e-4
    sdf_stage1_moment_weight: float = 5.0
    # SDF 第二阶段（有 FC1 重建监督）可选专用学习率；None 表示回退到基础 lr
    sdf_stage2_lr: Optional[float] = 2e-4
    # SDF_TRUE_ONLY uses a smaller LR and dynamic parent-group batching so
    # retry validation tests local improvements instead of large epoch jumps.
    sdf_true_only_lr: float = 4e-5
    sdf_true_target_batches: int = 20
    sdf_min_parent_groups_per_batch: int = 256
    # SDF 均值锚：约束 log(E[M]) 靠近理论目标（默认 log(0.98)）
    sdf_log_mean_target: float = field(default_factory=lambda: math.log(0.98))
    sdf_log_mean_anchor_weight_stage1: float = 1.0
    sdf_log_mean_anchor_weight_stage2: float = 5.0
    # Stage2 联合训练初期，对 HJ 相关项做 warmup，避免 FC1 还未收敛时被过早牵引
    sdf_stage2_hj_warmup_epochs: int = 5
    sdf_stage2_hj_warmup_start: float = 0.2
    sdf_euler_weight: float = 1.0
    sdf_true_moment_weight: float = 5e-4
    sdf_true_anchor_weight: float = 0.05
    sdf_recursive_loss_weight: float = 0.25
    sdf_recursive_moment_weight: float = 5e-4
    sdf_recursive_anchor_weight: float = 0.05
    # SDF phase validation / recovery. These controls are diagnostic guards for
    # locating whether post-Episode-0 SDF collapse starts in FC1_ONLY or
    # SDF_TRUE_ONLY without changing the baseline optimization problem.
    sdf_epoch_validation_enabled: bool = True
    sdf_restore_best_checkpoint: bool = True
    sdf_stop_when_gate_passes: bool = True
    sdf_required_consecutive_passes: int = 1
    sdf_collapse_log_mean_error: float = 0.5
    sdf_collapse_mean_ratio: float = 0.10
    sdf_collapse_patience: int = 1
    sdf_reset_optimizer_on_true_start: bool = False
    sdf_clear_optimizer_after_restore: bool = True
    stage_parameter_invariance_check_enabled: bool = True
    stage_epochwise_validation: bool = True
    sdf_collapse_lower_ratio: float = 0.1
    sdf_collapse_upper_ratio: float = 10.0
    fc1_sdf_preserve_ratio: float = 0.5
    stage_min_improvement: float = 1e-4
    stage_lr_decay_on_reject: float = 0.1
    stage_max_retries: int = 1
    sdf_score_t_weight: float = 0.05
    sdf_score_t_cap: float = 20.0
    fc1_epochs_per_round: int = 0
    fc1_max_rounds: int = 8
    fc1_plateau_patience: int = 2
    fc1_min_relative_improvement: float = 0.01
    fc1_gate_min_pairs: int = 128
    fc1_target_std_floor: float = 1e-4
    fc1_one_step_r2_min: float = 0.0
    fc1_one_step_skill_min: float = 0.0
    fc1_one_step_hatc_rmse_abs_max: float = 0.05
    fc1_one_step_lnk_rmse_abs_max: float = 0.05
    fc1_persistence_rmse_floor: float = 1e-6
    fc1_recursive_r2_min: float = 0.0
    fc1_rmse_growth_h5_max: float = 2.0
    fc1_rollout_finite_ratio_min: float = 1.0
    sdf_log_mean_error_max: float = 0.02
    sdf_signed_t_abs_max: float = 2.0
    sdf_fc1_val_fraction: float = 0.2
    sdf_fc1_val_seed: int = 12345
    sdf_gate_m_finite_ratio_min: float = 1.0
    sdf_gate_m_p99_max: float = math.inf
    sdf_gate_m_max_max: float = math.inf
    episode0_sdf_epochs_per_round: int = 0
    episode0_sdf_max_rounds: int = 10
    episode0_sdf_log_mean_error_max: float = 0.25
    episode0_sdf_clip_low_ratio_max: float = 0.20
    episode0_sdf_finite_ratio_min: float = 1.0
    allow_in_sample_sdf_gate_for_debug: bool = False
    sdf_post_refresh_gate_enabled: bool = True
    sdf_post_refresh_gate_mode: str = "safety"
    stage_gate_required_consecutive_passes: int = 1
    allow_modea_sdf_after_pv: bool = False

    # ========== Policy/Value: Q 优先训练与形状约束 ==========
    # 在 policy/value 联合训练前先进行 q-only 预训练轮数
    q_pretrain_epochs: int = 0
    # Q 损失中对 M 的处理（先 detach 并截断，减少 SDF 噪声传导）
    q_use_detached_m: bool = True
    q_m_clamp_min: float = 0.5
    q_m_clamp_max: float = 1.5
    # P0/PI Bellman 中 M 的稳定化（防止上游 SDF 短期失稳把 P 残差推偏）
    pv_use_clipped_m: bool = True
    pv_m_clamp_min: float = 0.7
    pv_m_clamp_max: float = 1.3
    pv_sdf_clip_ratio_gate: float = 1.0
    # Q 对 b/z 的形状正则权重与区间
    q_shape_weight_z: float = 1.0
    q_shape_weight_b_low: float = 1.0
    q_shape_weight_b_high: float = 1.0
    q_shape_b_low: float = 0.2
    q_shape_b_high: float = 0.8
    # Q-only 阶段是否冻结非 Q 分支参数（保持经济方程不改写）
    q_freeze_non_q_in_pretrain: bool = True
    # Q-only 阶段可训练参数范围：'q_head_only' 或 'q_path'(share_layer+q_head)
    q_pretrain_trainable_scope: str = 'q_head_only'
    # 兼容旧配置（不再推荐）：Q-only 阶段路径解耦开关
    q_decouple_policy_in_pretrain: bool = False
    # 论文式结构化 warm-start（Q 监督预训练）
    q_warmstart_epochs: int = 0
    q_warmstart_weight: float = 1.0
    q_warm_A: float = 1.0
    q_warm_b_star: float = 0.05
    q_warm_sigma: float = 0.15
    q_warm_alpha_z: float = 0.15
    q_warm_alpha_x: float = 0.15

    # ========== Policy/Value: bp KKT 约束（P0/PI） ==========
    # bp 训练模式：
    # - target_grid: 训练时用 frozen firm target 在 bp-grid 上做经济价值 argmax，
    #   P0/PI value heads 拟合最优 Bellman backup，bp heads 拟合 grid argmax。
    # - legacy_foc_kkt: 旧路径，bp heads 直接由 FOC/KKT 训练。
    pv_bp_training_mode: str = 'target_grid'
    # Target-grid 控制搜索区间与两阶段 coarse-to-fine 网格。
    bp_grid_min: float = 0.0
    bp_grid_max: float = 1.0
    bp_grid_coarse_size: int = 21
    bp_grid_refine_enabled: bool = True
    bp_grid_fine_size: int = 9
    bp_grid_quadratic_refine: bool = False
    bp_grid_parent_chunk_size: int = 2048
    bp_grid_candidate_chunk_size: int = 0
    bp_grid_max_expanded_states: int = 65536
    # Huber losses for target-grid value backup and policy distillation.
    bp_grid_value_huber_delta: float = 1.0
    bp_grid_policy_huber_delta: float = 0.05
    bp_grid_policy_weight: float = 1.0
    bp_grid_mix_policy_weight: float = 1.0
    # BP branch policy distillation space:
    # - "output": legacy Huber(bp_pred, bp_grid_star)
    # - "logit": Huber(bp_pred_logit, logit(bp_grid_star))
    bp_grid_policy_loss_space: str = "output"
    bp_grid_logit_target_eps: float = 1e-4
    bp_grid_logit_huber_delta: float = 1.0
    # Downweight policy targets when the top-two grid values are nearly tied.
    bp_grid_margin_scale: float = 1e-3
    bp_grid_confidence_relative: bool = True
    bp_grid_confidence_min: float = 0.0
    # Diagnostic-only threshold for reporting the share of active refinancing
    # policy candidates near the lower bp boundary. This parameter does not
    # enter the policy loss or grid search.
    bp_grid_boundary_low_threshold: float = 0.05
    # Under target_grid mode, policy convergence is checked in addition to
    # Bellman residual convergence.
    bp_grid_conv_mae_thresh: float = 0.05
    bp_grid_conv_regret_p90_thresh: float = 1e-2
    bp_grid_conv_max_batches: int = 4
    bp_grid_conv_survival_eps: float = 0.05
    # Hold out tail batches for target-grid policy convergence checks.  This
    # avoids passing convergence solely on states used by optimizer steps.
    pv_target_grid_val_fraction: float = 0.10
    # 对应有界控制 0 <= bp <= 1 的一阶最优条件：
    # - 内点: FOC = 0
    # - 下界: FOC <= 0
    # - 上界: FOC >= 0
    p0_kkt_weight: float = 1.0
    pi_kkt_weight: float = 1.0
    kkt_boundary_eps: float = 0.02
    # 可选：分别设置上下边界软区宽度；None 时回退到 kkt_boundary_eps
    # 例如 upper=0.20 表示 bp>0.8 即逐步进入“上边界近邻”KKT 区
    kkt_boundary_eps_low: Optional[float] = None
    kkt_boundary_eps_high: Optional[float] = 0.20
    kkt_boundary_temp: float = 40.0
    kkt_inner_weight: float = 1.0
    kkt_boundary_weight: float = 1.0
    # 上边界(bp≈1)违反项附加权重（缓解 bp 贴边）
    kkt_high_weight: float = 3.0
    # eta 稀疏时对 bp 相关项(FOC/KKT)做条件重权重
    eta_active_reweight_enabled: bool = True
    eta_active_target_ratio: float = 0.25
    eta_active_max_reweight: float = 6.0
    # Policy/Value batch 的 eta=1 条件重采样
    pv_eta_resample_enabled: bool = True
    pv_eta_resample_active_share: float = 0.25
    # bp 项自适应权重：使 (FOC+KKT) 与 Bellman 主项同量级
    bp_adaptive_enabled: bool = False
    bp_target_main_ratio: float = 0.3
    bp_adaptive_min_scale: float = 1.0
    bp_adaptive_max_scale: float = 200.0
    # 每个 epoch 增加若干 bp-only 精修步（仅更新 bp 相关头）
    bp_refine_steps_per_epoch: int = 0
    bp_refine_batch_cap: int = 32
    # FOC/KKT 的 ∂P'/∂bp 是否使用 Phat'（避免 P=max(Phat,0) 在违约区梯度为0）
    # 注意：仅影响梯度通道；Bellman 主方程仍使用 P（含显式 P=0 违约语义）
    bp_foc_use_phat_children: bool = True

    # ========== Firm target network ==========
    # policy_value 的 target network 不进入 optimizer；Bellman RHS 使用 target no-grad 输出。
    # soft: 每个 policy_value optimizer step 后做 Polyak update；hard: 每步硬同步；
    # epoch_hard/epoch_soft: epoch 内冻结 target，epoch 末更新；none: 只保留初始化 target。
    firm_target_update: str = 'epoch_hard'
    firm_target_tau: float = 0.005
    firm_target_update_interval_steps: int = 1

    # ========== Episode 收敛判定（非 AIO Bellman 残差） ==========
    # 判定条件：Q/P0/PI 各自主残差的 mean(abs) 与 p90(abs) 同时过阈值
    bellman_conv_mean_thresh: float = 1e-3
    bellman_conv_p90_thresh: float = 5e-3
    # Bellman convergence 的 p90 只使用有界样本估计，避免超大 tensor 上 torch.quantile 崩溃。
    bellman_conv_max_samples: int = 1_000_000
    # True 时在 Trainer.train 中达到收敛后提前结束 episode 循环
    episode_stop_on_bellman_convergence: bool = True
    
    # ========== 日志与保存 ==========
    log_interval: int = 10
    save_interval: int = 5
    
    # ========== 设备 ==========
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_device(self) -> torch.device:
        """返回torch设备"""
        return torch.device(self.device)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'HyperParams':
        """从字典创建"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
