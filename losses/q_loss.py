"""
Q Loss: 债券价格损失

核心目标：拟合债券价格方程残差，叠加高杠杆边界条件（b≥1）与 bar_z 相关约束，
确保债券定价合理性。

支持任意数量的分支路径：
- parent: (Q_t, b_t, ...) → t 期父节点
- children: [(Qsp_{t+1}^{(j)}, bar_z_{t+1}^{(j)}, ...)] → N 条模拟路径

L_Q = main_q + loss3 + loss5 + penalty_z_all
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List

import sys
sys.path.append('..')
from config import Config

from .utils import compute_aio_residual, compute_z_penalty


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
        z0: float = None
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
    
    def compute_recovery_value(
        self,
        x: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        计算违约时的回收价值
        
        recovery = φ * (1 - δ + exp(x+z))
        """
        return self.phi * (1 - self.delta + torch.exp(x + z))

    def compute_total_recovery(
        self,
        b: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        计算违约时的总回收价值。

        与论文 Bondprice 公式保持一致，违约回收来自企业可回收资产，
        而不是债务面值的线性倍数：

            recovery_total = φ * (1 - δ + exp(x+z))

        这里保留 b 参数仅为兼容现有调用链；理论上该回收值不应乘 b。
        """
        _ = b
        return self.compute_recovery_value(x, z)
    
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
            # 回收价值（总债价值口径）
            recovery_total = self.compute_total_recovery(b_nonneg, x, z)
            
            # 债券定价方程
            # loss = M * [(b + Qsp * multiplier) * (1-bar_z) + recovery_total * multiplier * bar_z] - Q
            residual = M * (
                (b_nonneg + Qsp * multiplier) * (1 - bar_z) +
                recovery_total * multiplier * bar_z
            ) - Q
            
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
        
        当 bar_z > 0（违约）时，Q 应接近论文定义的资产回收价值
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
        低杠杆边界损失已弃用。

        结构层已显式规定 Q = b * q_unit，因此 b=0 时天然满足 Q=0。
        继续对整段 b<=margin 施加 Q≈0 约束会在小正债务区错误压扁 Q。
        """
        _ = (b, margin)
        return torch.zeros_like(Q)
    
    def compute_boundary_loss_high(
        self,
        Q: torch.Tensor,
        b: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
        margin: float = 1e-1
    ) -> torch.Tensor:
        """
        b ≥ 1 边界条件：债券价格应等于资产回收价值
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
        z_children: List[torch.Tensor]
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
        
        main_q = compute_aio_residual(residuals, self.aio_weight)
        
        # bar_z 约束
        loss3 = self.compute_bar_z_constraint(Q, b, x, z, bar_z)
        
        # 边界条件
        loss4 = self.compute_boundary_loss_low(Q, b)
        loss5 = self.compute_boundary_loss_high(Q, b, x, z)
        
        # z 惩罚
        penalty_z_main = compute_z_penalty(
            compute_aio_residual(residuals, self.aio_weight), 
            z, self.alpha_z, self.beta_z, self.z0
        )
        penalty_z_loss3 = compute_z_penalty(
            (Q - self.compute_total_recovery(b, x, z)).pow(2) * bar_z,
            z, self.alpha_z, self.beta_z, self.z0
        )
        
        total_loss = main_q + loss3 + loss5 + penalty_z_main + penalty_z_loss3
        
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
        
        total_loss = main_q + loss3 + loss5 + penalty_z
        
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
