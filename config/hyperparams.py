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
    
    # ========== 优化器参数 ==========
    # SDF & FC1
    sdf_lr: float = 5e-4
    sdf_weight_decay: float = 1e-3
    fc1_lr: float = 1e-3
    fc1_weight_decay: float = 1e-4
    # 通用学习率（调度器基准值，缺省时沿用 sdf_lr）
    lr: float = 5e-4
    
    # Policy & Value
    policy_lr: float = 1e-3
    policy_weight_decay: float = 1e-6
    q_lr: float = 1e-3
    q_weight_decay: float = 1e-6
    pvbp_lr: float = 1e-3
    pvbp_weight_decay: float = 1e-6
    
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
    
    # NaN/Inf 检测
    nan_recovery: bool = True
    
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

    # Stage2 true-state 重建损失权重：
    # (Hatc_t, LnK_t) -> (Hatc_{t+1}, LnK_{t+1})
    # 默认关闭，只保留 forecast-state 闭环监督。
    fc1_recon_weight: float = 0.0
    # Stage2 额外约束 forecast-state 递推：
    # (Hatcf_t, LnKF_t) -> (Hatcf_{t+1}, LnKF_{t+1}) 也要贴近真实下一期
    fc1_forecast_recon_weight: float = 1.0
    # FC1 重建项内部按目标拆分权重。
    # 经验上 LnK 的原始尺度波动更大，若不单独降权，容易主导 FC1 训练并把 M 分布拉坏。
    fc1_hatc_recon_weight: float = 1.0
    fc1_lnk_recon_weight: float = 0.25
    # Forecast-state 一步增量幅度约束。
    # 不预设方向，只惩罚过大的单步跳跃，避免递推响应面把 child state 撕裂成多个 regime。
    fc1_delta_penalty_weight: float = 10.0
    fc1_delta_hatc_abs_max: float = 0.50
    fc1_delta_lnk_abs_max: float = 0.30
    # Forecast-state 响应面局部平滑约束（Jacobian penalty）。
    # 用于抑制 FC1 对 (hatcf_prev, lnkf_prev) 的过强局部敏感性，减少 child state regime splitting。
    fc1_jacobian_penalty_weight: float = 1.0
    # Stage2 先做若干轮 FC1 teacher forcing 预训练（使用真实 Hatc_t/LnK_t 输入）
    fc1_teacher_forcing_epochs: int = 5
    fc1_teacher_forcing_weight: float = 1.0
    # 在 stage2/joint 中，若 batch 提供真实 Hatc_t/LnK_t，是否优先用真实当前态驱动 FC1。
    # 默认关闭，joint 阶段使用 forecast-state 输入以约束递推闭环。
    fc1_use_true_macro_state_in_stage2: bool = False
    # SDF 矩约束权重（常规阶段）
    sdf_moment_weight: float = 1.0
    # SDF 第一阶段（无 FC1 监督）专用学习率与矩约束权重
    sdf_stage1_lr: float = 1e-4
    sdf_stage1_moment_weight: float = 5.0
    # SDF 第二阶段（有 FC1 重建监督）可选专用学习率；None 表示回退到基础 lr
    sdf_stage2_lr: Optional[float] = 2e-4
    # SDF 均值锚：约束 log(E[M]) 靠近理论目标（默认 log(0.98)）
    sdf_log_mean_target: float = field(default_factory=lambda: math.log(0.98))
    sdf_log_mean_anchor_weight_stage1: float = 1.0
    sdf_log_mean_anchor_weight_stage2: float = 5.0
    # Stage2 联合训练初期，对 HJ 相关项做 warmup，避免 FC1 还未收敛时被过早牵引
    sdf_stage2_hj_warmup_epochs: int = 5
    sdf_stage2_hj_warmup_start: float = 0.2

    # ========== Policy/Value: Q 优先训练与形状约束 ==========
    # 在 policy/value 联合训练前先进行 q-only 预训练轮数
    q_pretrain_epochs: int = 10
    policy_separate_q_pvbp_training: bool = True
    # 重构后默认采用两阶段训练：
    # Stage A: Q-only
    # Stage B: PV/BP-only
    q_stage_epochs: int = 100
    pvbp_stage_epochs: int = 100
    # Q 损失中对 M 的处理（先 detach 并截断，减少 SDF 噪声传导）
    q_use_detached_m: bool = True
    q_m_clamp_min: float = 0.5
    q_m_clamp_max: float = 1.5
    # P0/PI Bellman 中 M 的稳定化（防止上游 SDF 短期失稳把 P 残差推偏）
    pv_use_clipped_m: bool = True
    pv_m_clamp_min: float = 0.7
    pv_m_clamp_max: float = 1.3
    # Q 形状正则改为约束单位债价格 q_unit：
    # 1) dq_unit/dz >= 0
    # 2) dq_unit/db <= 0
    q_shape_weight_z: float = 1.0
    q_shape_weight_b_low: float = 1.0
    # 兼容旧字段；当前实现不再单独使用高 b 区权重
    q_shape_weight_b_high: float = 0.0
    q_shape_b_low: float = 0.2
    q_shape_b_high: float = 0.8
    # Q-only 阶段是否冻结非 Q 分支参数（保持经济方程不改写）
    q_freeze_non_q_in_pretrain: bool = True
    # Q-only 阶段可训练参数范围：'q_head_only' 或 'q_path'(share_layer+q_head)
    q_pretrain_trainable_scope: str = 'q_path'
    # 兼容旧配置（不再推荐）：Q-only 阶段路径解耦开关
    q_decouple_policy_in_pretrain: bool = False
    # 论文式结构化 warm-start（Q 监督预训练）
    q_warmstart_epochs: int = 10
    q_warmstart_weight: float = 1.0
    q_warm_A: float = 1.0
    q_warm_b_star: float = 0.05
    q_warm_sigma: float = 0.15
    q_warm_alpha_z: float = 0.15
    q_warm_alpha_x: float = 0.15

    # ========== Policy/Value: bp KKT 约束（P0/PI） ==========
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
    # FOC/KKT 的 ∂P'/∂bp 默认与 Bellman 主方程保持一致，直接使用 P'。
    # 这样 surrogate 与实际 payoff 对齐，避免 bp 头沿着 Phat' 在违约区继续收到与 P=0 不一致的梯度。
    # 若做对照实验，可临时改回 True，让 FOC/KKT 使用 Phat' 作为梯度通道。
    bp_foc_use_phat_children: bool = False
    # ========== bp 诊断图（默认轻量版） ==========
    bp_diag_enabled: bool = True
    # 多 episode 训练时每隔多少个 episode 生成一次 bp 诊断；最后一轮仍会强制生成
    bp_diag_every_n_episodes: int = 5
    # 逗号分隔的状态名集合：safe,mid,risky,distress
    bp_diag_states: str = "safe"
    # bp 诊断扫描网格点数；诊断只需看形状，101 通常足够
    bp_diag_grid_points: int = 101
    # 诊断图默认用有限差分近似 FOC/KKT，避免 episode 末尾额外跑 autograd.grad
    bp_diag_use_autograd_foc: bool = False
    # ========== bp 训练：child 存活区加权 ==========
    # 仅让 child 仍具继续经营意义的区域主导 bp 的 FOC/KKT 训练，
    # 避免 default 右侧局部驻点被当成正常内点最优。
    bp_survival_reweight_enabled: bool = True
    bp_survival_tau_p: float = 20.0
    bp_survival_tau_z: float = 20.0
    bp_survival_barz_threshold: float = 0.5
    # ========== bp 训练：粗网格 value supervision（GPU 上向量化） ==========
    # 仅靠 FOC/KKT 难以处理非凹、存在 regime switch 的 V(bp)。
    # 这里用小网格近似 survive-set 内的 argmax V，给 bp 一个直接的全局 value-level 信号。
    bp_value_supervision_enabled: bool = True
    bp_value_weight: float = 1.0
    bp_value_grid_points: int = 21
    bp_value_sample_cap: int = 256
    bp_value_survival_only: bool = True
    bp_value_barz_threshold: float = 0.5

    # ========== Episode 收敛判定（非 AIO Bellman 残差） ==========
    # 判定条件：Q/P0/PI 各自主残差的 mean(abs) 与 p90(abs) 同时过阈值
    bellman_conv_mean_thresh: float = 1e-3
    bellman_conv_p90_thresh: float = 5e-3
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
