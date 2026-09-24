"""
Q Loss: 债券价格损失

核心目标：拟合债券价格方程残差，叠加边界条件约束（b≤0、b≥1）与 bar_z 相关约束，
确保债券定价合理性。

支持任意数量的分支路径：
- parent: (Q_t, b_t, ...) → t 期父节点
- children: [(Qsp_{t+1}^{(j)}, bar_z_{t+1}^{(j)}, ...)] → N 条模拟路径

legacy_soft_penalty 模式（历史行为）：
    L_Q = main_q + loss3 + loss4 + loss5 + penalty_z_all

regime-aware 模式（hard / transition_band）：
    L_Q = w_survival_parent * L_Bellman + w_default_parent * (Q - R_current)^2
          + loss4 + loss5 + penalty_z_all

其中 parent regime 由 ``Phat_t`` 决定，``w_survival_parent + w_default_parent = 1``。
child default（t+1 违约）始终保留在 ``L_Bellman`` 内部，与 parent regime 无关。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

import sys
sys.path.append('..')
from config import Config

from .utils import compute_aio_residual, compute_z_penalty


# ---------------------------------------------------------------------------
# 口径解析：parent default regime 与 recovery 归一化
# ---------------------------------------------------------------------------

Q_PARENT_DEFAULT_REGIME_MODES: Tuple[str, ...] = (
    "legacy_soft_penalty",
    "hard",
    "transition_band",
)

RECOVERY_NORMALIZATION_MODES: Tuple[str, ...] = (
    # 历史行为：recovery_total = b_+ * phi * (1 - delta + exp(x + z))
    "legacy_b_times_unit",
    # main_4.tex / Gomes 口径：违约回收是资产回收，不乘债务面值 b
    # recovery_unit = phi * (1 - delta + exp(x + z))
    "asset_only",
)


def resolve_q_parent_default_regime_mode(mode: Optional[str] = None) -> str:
    value = mode if mode is not None else getattr(
        Config, "Q_PARENT_DEFAULT_REGIME_MODE", "legacy_soft_penalty"
    )
    value = str(value).strip().lower()
    if value not in Q_PARENT_DEFAULT_REGIME_MODES:
        raise ValueError(
            f"Unsupported q_parent_default_regime_mode: {value!r}; "
            f"expected one of {Q_PARENT_DEFAULT_REGIME_MODES}"
        )
    return value


def resolve_recovery_normalization_mode(mode: Optional[str] = None) -> str:
    value = mode if mode is not None else getattr(
        Config, "RECOVERY_NORMALIZATION_MODE", "legacy_b_times_unit"
    )
    value = str(value).strip().lower()
    if value not in RECOVERY_NORMALIZATION_MODES:
        raise ValueError(
            f"Unsupported recovery_normalization_mode: {value!r}; "
            f"expected one of {RECOVERY_NORMALIZATION_MODES}"
        )
    return value


def compute_recovery_unit(
    x: torch.Tensor,
    z: torch.Tensor,
    *,
    phi: float,
    delta: float,
) -> torch.Tensor:
    """归一化违约回收单位：phi * (1 - delta + exp(x + z))。"""
    return phi * (1.0 - delta + torch.exp(x + z))


def compute_recovery_target(
    b: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    *,
    phi: float,
    delta: float,
    recovery_normalization_mode: Optional[str] = None,
) -> torch.Tensor:
    """违约回收目标（与 Q 同一单位）。

    ``asset_only``：``phi * (1 - delta + exp(x + z))``（main_4.tex:344 / :968-969）
    ``legacy_b_times_unit``：``b_+ * phi * (1 - delta + exp(x + z))``
    """
    mode = resolve_recovery_normalization_mode(recovery_normalization_mode)
    unit = compute_recovery_unit(x, z, phi=phi, delta=delta)
    if mode == "asset_only":
        return unit
    return torch.clamp(b, min=0.0) * unit


def compute_parent_default_regime_weights(
    phat: torch.Tensor,
    *,
    mode: Optional[str] = None,
    eps: Optional[float] = None,
    tau: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """按当期 parent 是否已违约（``Phat_t <= 0``）构造 regime 权重。

    - ``legacy_soft_penalty``：``w_survival = 1``，``w_default = 0``
      （Bellman 覆盖全部 parent，违约侧仅由 soft ``bar_z`` 惩罚项处理）
    - ``hard``：``w_survival = 1{Phat_t > 0}``
    - ``transition_band``：``|Phat| <= eps`` 内用 ``sigmoid(Phat / tau)`` 平滑，
      带外严格取 0/1，保证深度 default 区真正 collapse 到 recovery regime
    """
    resolved = resolve_q_parent_default_regime_mode(mode)
    # regime 权重是纯 gate，必须对 Phat 梯度隔离：Q loss 不允许通过
    # w_survival(Phat) 反传到 P / Phat / value network。
    phat = phat.detach().reshape(phat.shape[0], -1)[:, :1]
    ones = torch.ones_like(phat)
    zeros = torch.zeros_like(phat)
    hard_default = (phat <= 0.0).to(phat.dtype)

    if resolved == "legacy_soft_penalty":
        survival = ones
    elif resolved == "hard":
        survival = ones - hard_default
    else:
        eps_value = float(
            eps if eps is not None else getattr(Config, "Q_PARENT_DEFAULT_EPS", 1e-2)
        )
        tau_value = float(
            tau if tau is not None else getattr(Config, "Q_PARENT_DEFAULT_TAU", 1e-2)
        )
        eps_value = max(eps_value, 0.0)
        tau_value = max(tau_value, 1e-8)
        band = torch.sigmoid(phat / tau_value)
        survival = torch.where(
            phat >= eps_value, ones, torch.where(phat <= -eps_value, zeros, band)
        )
    return {
        "parent_survival_weight": survival,
        "parent_default_weight": ones - survival,
        "parent_hard_default": hard_default,
        "parent_default_regime_mode": resolved,
    }


def build_q_polish_coverage_weights(
    phat: torch.Tensor,
    *,
    sim_share: float,
    boundary_share: float,
    default_share: float,
    eps_boundary: float,
) -> Dict[str, torch.Tensor]:
    """Q-only polishing 的 coverage 分组权重（simulated / boundary / default）。

    分组（``eps = eps_boundary``）：

        default   : Phat <  -eps
        boundary  : |Phat| <= eps
        sim       : 其余

    组权重 = share_g * N / n_g，使每组对 loss 的有效质量正比于其 share。
    deep-default 区天然样本不足（default firm 会退出），因此需要显式过采样。
    """
    phat = phat.reshape(phat.shape[0], -1)[:, :1]
    total = float(sim_share) + float(boundary_share) + float(default_share)
    if total <= 0.0:
        raise ValueError("q polish shares must sum to a positive value")
    shares = (
        float(sim_share) / total,
        float(boundary_share) / total,
        float(default_share) / total,
    )
    eps = max(float(eps_boundary), 0.0)
    is_default = phat < -eps
    is_boundary = (~is_default) & (phat.abs() <= eps)
    is_sim = ~(is_default | is_boundary)
    n_states = float(phat.shape[0])
    weights = torch.zeros_like(phat)
    for mask, share in zip((is_sim, is_boundary, is_default), shares):
        count = int(mask.sum().item())
        if count == 0:
            continue
        weights = torch.where(mask, torch.full_like(phat, share * n_states / count), weights)
    return {
        "coverage_weight": weights,
        "sim_mask": is_sim.to(phat.dtype),
        "boundary_mask": is_boundary.to(phat.dtype),
        "default_mask": is_default.to(phat.dtype),
    }


def compute_q_survival_recovery_components(
    *,
    Q: torch.Tensor,
    b: torch.Tensor,
    bar_i: torch.Tensor,
    M: torch.Tensor,
    Qsp: torch.Tensor,
    bar_z: torch.Tensor,
    x_child: torch.Tensor,
    z_child: torch.Tensor,
    g: float,
    delta: float,
    phi: float,
    recovery_normalization_mode: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    multiplier = bar_i * (g - 1) + 1
    b_nonneg = torch.clamp(b, min=0.0)
    recovery_total = compute_recovery_target(
        b_nonneg,
        x_child,
        z_child,
        phi=phi,
        delta=delta,
        recovery_normalization_mode=recovery_normalization_mode,
    )
    q_target_survival = M * (b_nonneg + Qsp * multiplier) * (1 - bar_z)
    q_target_recovery = M * recovery_total * multiplier * bar_z
    q_target_total = q_target_survival + q_target_recovery
    return {
        "q_target_survival": q_target_survival,
        "q_target_recovery": q_target_recovery,
        "q_target_total": q_target_total,
        "q_training_residual": q_target_total - Q,
        "q_issue_minus_target": Q - q_target_total,
        "q_pricing_residual": q_target_total - Q,
        "multiplier": multiplier,
        "recovery_total": recovery_total,
    }


class QLoss(nn.Module):
    """
    Q（债券价格）损失函数
    
    包含：
    - 核心债券定价残差
    - bar_z 相关约束
    - 边界条件约束（b≤0, b≥1）
    - z 值惩罚
    """
    
    def __init__(
        self,
        delta: float = None,
        phi: float = None,
        g: float = None,
        aio_weight: float = None,
        alpha_z: float = None,
        beta_z: float = None,
        z0: float = None,
        recovery_normalization_mode: Optional[str] = None,
        parent_default_regime_mode: Optional[str] = None,
        parent_default_eps: Optional[float] = None,
        parent_default_tau: Optional[float] = None,
    ):
        super().__init__()
        
        # 经济参数
        self.delta = delta if delta is not None else Config.DELTA
        self.phi = phi if phi is not None else Config.PHI
        self.g = g if g is not None else Config.G
        
        # 损失权重
        self.aio_weight = aio_weight if aio_weight is not None else Config.AIO_WEIGHT
        self.alpha_z = alpha_z if alpha_z is not None else Config.ALPHA_Z
        self.beta_z = beta_z if beta_z is not None else Config.BETA_Z
        self.z0 = z0 if z0 is not None else Config.Z0

        # 口径：recovery 归一化 + parent default regime
        self.recovery_normalization_mode = resolve_recovery_normalization_mode(
            recovery_normalization_mode
        )
        self.parent_default_regime_mode = resolve_q_parent_default_regime_mode(
            parent_default_regime_mode
        )
        self.parent_default_eps = float(
            parent_default_eps if parent_default_eps is not None
            else getattr(Config, "Q_PARENT_DEFAULT_EPS", 1e-2)
        )
        self.parent_default_tau = float(
            parent_default_tau if parent_default_tau is not None
            else getattr(Config, "Q_PARENT_DEFAULT_TAU", 1e-2)
        )
    
    def compute_recovery_value(
        self,
        x: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        计算违约时的回收价值（与 Q 同一单位）

        recovery_unit = φ * (1 - δ + exp(x+z))

        对应 main_4.tex Bondprice 公式里的 φ(1 - δ + exp(x' + z'))·(k'/k)，
        即资产回收，不含债务面值 b。
        """
        return compute_recovery_unit(x, z, phi=self.phi, delta=self.delta)

    def compute_total_recovery(
        self,
        b: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        违约回收目标（与 Q 同一单位），按 ``recovery_normalization_mode`` 解析：

        - ``asset_only``（默认，理论口径）：φ * (1 - δ + exp(x+z))
        - ``legacy_b_times_unit``：b_+ * φ * (1 - δ + exp(x+z))
        """
        return compute_recovery_target(
            b, x, z, phi=self.phi, delta=self.delta,
            recovery_normalization_mode=self.recovery_normalization_mode,
        )

    def compute_current_recovery(
        self,
        b: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """当期 parent 违约时的回收（R_current），与 ``compute_total_recovery`` 同口径。"""
        return self.compute_total_recovery(b, x, z)

    def compute_regime_weights(self, phat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """当期 parent default regime 权重（见 ``compute_parent_default_regime_weights``）。"""
        return compute_parent_default_regime_weights(
            phat,
            mode=self.parent_default_regime_mode,
            eps=self.parent_default_eps,
            tau=self.parent_default_tau,
        )

    def compute_regime_aware_q_target(
        self,
        *,
        q_target_bellman: torch.Tensor,
        recovery_current: torch.Tensor,
        phat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """训练用 Q target：surviving parent 用 Bellman，default parent 用当期回收。

            Q_target_used_for_training
                = w_survival * q_target_bellman + w_default * recovery_current

        ``legacy_soft_penalty`` 下 w_survival ≡ 1，退化为纯 Bellman target。
        """
        weights = self.compute_regime_weights(phat)
        w_survival = weights["parent_survival_weight"]
        w_default = weights["parent_default_weight"]
        target = w_survival * q_target_bellman + w_default * recovery_current
        return {
            "q_target_used_for_training": target,
            "recovery_current": recovery_current,
            **weights,
        }

    def compute_regime_aware_objective(
        self,
        *,
        residuals: List[torch.Tensor],
        Q: torch.Tensor,
        b: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
        phat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """构造 regime-aware 的 Q 主目标（sample-level gating）。

            L_Bellman  = mean(w_survival ⊙ AiO(branch residuals))
            L_recovery = mean(w_default ⊙ (R_current - Q)²)

        AiO 在 **未加 mask** 的 branch residual 上计算，再逐样本乘 regime 权重，
        避免把 default parent 的 Bellman residual 置零后污染 AiO 的乘积项。
        """
        aio_residual = compute_aio_residual(residuals, self.aio_weight)
        weights = self.compute_regime_weights(phat)
        w_survival = weights["parent_survival_weight"]
        w_default = weights["parent_default_weight"]
        recovery_current = self.compute_current_recovery(b, x, z)
        bellman_per_sample = w_survival * aio_residual
        recovery_per_sample = w_default * (recovery_current - Q).pow(2)
        return {
            "aio_residual": aio_residual,
            "bellman_per_sample": bellman_per_sample,
            "recovery_per_sample": recovery_per_sample,
            "bellman_loss": bellman_per_sample.mean(),
            "recovery_loss": recovery_per_sample.mean(),
            "recovery_current": recovery_current,
            **weights,
        }
    
    def compute_main_residual(
        self,
        Q: torch.Tensor,
        b: torch.Tensor,
        bar_i: torch.Tensor,
        M_list: List[torch.Tensor],
        Qsp_children: List[torch.Tensor],
        bar_z_children: List[torch.Tensor],
        x_children: List[torch.Tensor],
        z_children: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        计算核心债券定价残差（支持任意分支数）
        
        基于修改后债务 b' = b / (bar_i * (G-1) + 1)
        """
        # 乘数
        multiplier = bar_i * (self.g - 1) + 1
        b_nonneg = torch.clamp(b, min=0.0)
        
        residuals = []
        for M, Qsp, bar_z, x, z in zip(M_list, Qsp_children, bar_z_children, x_children, z_children):
            components = compute_q_survival_recovery_components(
                Q=Q,
                b=b_nonneg,
                bar_i=bar_i,
                M=M,
                Qsp=Qsp,
                bar_z=bar_z,
                x_child=x,
                z_child=z,
                g=self.g,
                delta=self.delta,
                phi=self.phi,
                recovery_normalization_mode=self.recovery_normalization_mode,
            )
            residual = components["q_training_residual"]
            
            residuals.append(residual)
        
        return residuals
    
    def compute_main_residual_legacy(
        self,
        Q: torch.Tensor,
        b: torch.Tensor,
        bar_i: torch.Tensor,
        M1: torch.Tensor,
        M2: torch.Tensor,
        Qsp2: torch.Tensor,
        Qsp3: torch.Tensor,
        bar_z2: torch.Tensor,
        bar_z3: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算核心债券定价残差（旧版双路径接口）
        """
        residuals = self.compute_main_residual(
            Q, b, bar_i,
            M_list=[M1, M2],
            Qsp_children=[Qsp2, Qsp3],
            bar_z_children=[bar_z2, bar_z3],
            x_children=[x2, x3],
            z_children=[z2, z3]
        )
        return residuals[0], residuals[1]
    
    def compute_bar_z_constraint(
        self,
        Q: torch.Tensor,
        b: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
        bar_z: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 bar_z 相关约束
        
        当 bar_z > 0（违约）时，Q 应接近总债价值口径回收目标
        """
        recovery_total = self.compute_total_recovery(b, x, z)
        constraint = (Q - recovery_total) * bar_z
        return constraint.pow(2)
    
    def compute_boundary_loss_low(
        self,
        Q: torch.Tensor,
        b: torch.Tensor,
        margin: float = 1e-1
    ) -> torch.Tensor:
        """
        b ≤ 0 边界条件：债券价格应接近 0
        """
        mask = (b <= margin).float()
        return Q.pow(2) * mask
    
    def compute_boundary_loss_high(
        self,
        Q: torch.Tensor,
        b: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
        margin: float = 1e-1
    ) -> torch.Tensor:
        """
        b ≥ 1 边界条件：债券价格应等于回收价值
        """
        mask = (b >= 1 - margin).float()
        recovery_total = self.compute_total_recovery(b, x, z)
        return (Q - recovery_total).pow(2) * mask
    
    def forward(
        self,
        Q: torch.Tensor,
        inputs: torch.Tensor,
        bar_i: torch.Tensor,
        M_list: List[torch.Tensor],
        Qsp_children: List[torch.Tensor],
        bar_z: torch.Tensor,
        bar_zsp_children: List[torch.Tensor],
        x_children: List[torch.Tensor],
        z_children: List[torch.Tensor],
        phat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 Q 损失（支持任意分支数）

        Args:
            Q: 当前债券价格
            inputs: 输入状态
            bar_i: 投资门槛
            M_list: List[torch.Tensor] - 各路径的 SDF
            Qsp_children: List[torch.Tensor] - 各路径的未来债券价格
            bar_z: 当期违约阈值
            bar_zsp_children: List[torch.Tensor] - 各路径的违约阈值
            x_children, z_children: 各路径的未来状态
            phat: 当期 parent 的 Phat；仅在非 legacy regime 下需要（默认 None 退化为 legacy）

        Returns:
            total_loss: 总损失
            loss_dict: 各分量损失字典
        """
        # 提取状态
        b = inputs[:, 0:1]
        z = inputs[:, 1:2]
        x = inputs[:, 4:5]

        loss_dict = {}

        # 主残差
        residuals = self.compute_main_residual(
            Q, b, bar_i, M_list, Qsp_children,
            bar_zsp_children, x_children, z_children
        )

        # 边界条件
        loss4 = self.compute_boundary_loss_low(Q, b)
        loss5 = self.compute_boundary_loss_high(Q, b, x, z)

        if self.parent_default_regime_mode == "legacy_soft_penalty" or phat is None:
            # legacy：Bellman 覆盖全部 parent，违约侧靠 soft bar_z 惩罚项
            main_q = compute_aio_residual(residuals, self.aio_weight)
            loss3 = self.compute_bar_z_constraint(Q, b, x, z, bar_z)
            penalty_z_main = compute_z_penalty(
                compute_aio_residual(residuals, self.aio_weight),
                z, self.alpha_z, self.beta_z, self.z0
            )
            penalty_z_loss3 = compute_z_penalty(
                (Q - self.compute_total_recovery(b, x, z)).pow(2) * bar_z,
                z, self.alpha_z, self.beta_z, self.z0
            )
        else:
            # regime-aware：surviving parent 走 Bellman，default parent 走当期回收
            terms = self.compute_regime_aware_objective(
                residuals=residuals, Q=Q, b=b, x=x, z=z, phat=phat,
            )
            main_q = terms["bellman_loss"]
            loss3 = terms["recovery_loss"]
            penalty_z_main = compute_z_penalty(
                terms["bellman_per_sample"], z, self.alpha_z, self.beta_z, self.z0
            )
            penalty_z_loss3 = compute_z_penalty(
                terms["recovery_per_sample"], z, self.alpha_z, self.beta_z, self.z0
            )

        total_loss = main_q + loss3 + loss4 + loss5 + penalty_z_main + penalty_z_loss3

        loss_dict = {
            'main_q': main_q,
            'loss3': loss3,
            'loss4': loss4,
            'loss5': loss5,
            'penalty_z_main': penalty_z_main,
            'penalty_z_loss3': penalty_z_loss3,
            'total_loss': total_loss
        }

        return total_loss, loss_dict
    
    def forward_legacy(
        self,
        Q: torch.Tensor,
        inputs: torch.Tensor,
        bar_i: torch.Tensor,
        M1: torch.Tensor,
        M2: torch.Tensor,
        Qsp2: torch.Tensor,
        Qsp3: torch.Tensor,
        bar_z: torch.Tensor,
        bar_z2: torch.Tensor,
        bar_z3: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        旧版接口（兼容2分支情况）
        """
        return self.forward(
            Q, inputs, bar_i,
            M_list=[M1, M2],
            Qsp_children=[Qsp2, Qsp3],
            bar_z=bar_z,
            bar_z_children=[bar_z2, bar_z3],
            x_children=[x2, x3],
            z_children=[z2, z3]
        )
    
    def forward_simplified(
        self,
        Q: torch.Tensor,
        b: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
        bar_i: torch.Tensor,
        bar_z: torch.Tensor,
        M_list: List[torch.Tensor],
        Qsp_children: List[torch.Tensor],
        bar_z_children: List[torch.Tensor],
        x_children: List[torch.Tensor],
        z_children: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        简化版 Q 损失（支持任意分支数）
        """
        # 主残差
        residuals = self.compute_main_residual(
            Q, b, bar_i, M_list, Qsp_children,
            bar_z_children, x_children, z_children
        )
        
        main_q = compute_aio_residual(residuals, self.aio_weight)
        
        # 约束和惩罚
        loss3 = self.compute_bar_z_constraint(Q, b, x, z, bar_z)
        loss4 = self.compute_boundary_loss_low(Q, b)
        loss5 = self.compute_boundary_loss_high(Q, b, x, z)
        penalty_z = compute_z_penalty(
            compute_aio_residual(residuals, self.aio_weight),
            z, self.alpha_z, self.beta_z, self.z0
        )
        
        total_loss = main_q + loss3 + loss4 + loss5 + penalty_z
        
        loss_dict = {
            'main_q': main_q,
            'loss3': loss3,
            'loss4': loss4,
            'loss5': loss5,
            'penalty_z': penalty_z,
            'total_loss': total_loss
        }
        
        return total_loss, loss_dict
    
    def forward_simplified_legacy(
        self,
        Q: torch.Tensor,
        b: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
        bar_i: torch.Tensor,
        bar_z: torch.Tensor,
        M1: torch.Tensor,
        M2: torch.Tensor,
        Qsp2: torch.Tensor,
        Qsp3: torch.Tensor,
        bar_z2: torch.Tensor,
        bar_z3: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        简化版 Q 损失（兼容2分支接口）
        """
        return self.forward_simplified(
            Q, b, z, x, bar_i, bar_z,
            M_list=[M1, M2],
            Qsp_children=[Qsp2, Qsp3],
            bar_z_children=[bar_z2, bar_z3],
            x_children=[x2, x3],
            z_children=[z2, z3]
        )
