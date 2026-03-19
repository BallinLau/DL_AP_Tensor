"""
FC2 Loss: 横截面分布 → 宏观 proxy 的一致性损失

核心目标：实现 fixed-point consistency
(ĉ, ln K) = A(Policy/Value(ĉ, ln K; s_i))

L_FC2 = L_node + λ_trans * L_trans

其中：
- L_node: 当期一致性损失
- L_trans: 跨期一致性损失（可选）
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

import sys
sys.path.append('..')
from config import Config


class FC2Loss(nn.Module):
    """
    FC2 一致性损失函数
    
    让 FC2 输出的宏观量与通过 Policy/Value 聚合得到的宏观量一致
    """
    
    def __init__(
        self,
        w_c: float = 1.0,
        w_k: float = 1.0,
        lambda_trans: float = 0.0,
        delta: float = None,
        phi: float = None
    ):
        super().__init__()
        
        # 损失权重
        self.w_c = w_c
        self.w_k = w_k
        self.lambda_trans = lambda_trans
        
        # 经济参数
        self.delta = delta if delta is not None else Config.DELTA
        self.phi = phi if phi is not None else Config.PHI
    
    def compute_node_loss(
        self,
        hatc_fc2: torch.Tensor,
        lnk_fc2: torch.Tensor,
        hatc_agg: torch.Tensor,
        lnk_agg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算当期一致性损失
        
        L_node = w_c * (ĉ_fc2 - ĉ_agg)² + w_k * (lnK_fc2 - lnK_agg)²
        """
        r_c = hatc_fc2 - hatc_agg
        r_k = lnk_fc2 - lnk_agg
        
        loss_c = self.w_c * r_c.pow(2).mean()
        loss_k = self.w_k * r_k.pow(2).mean()
        
        return loss_c + loss_k, loss_c, loss_k
    
    def compute_resource_accounting(
        self,
        K: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
        bar_i: torch.Tensor,
        bar_z: torch.Tensor,
        i: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        资源核算
        
        Y = exp(x+z) * K
        Φ = (1-PHI) * (1 + exp(x+z)) * K * bar_z
        I = bar_i * K * i - bar_z * K + δ * K
        C = Y - I - Φ
        """
        # 产出
        Y = torch.exp(x + z) * K
        
        # 破产成本
        Phi = (1 - self.phi) * (1 + torch.exp(x + z)) * K * bar_z
        
        # 投资
        I = bar_i * K * i - bar_z * K + self.delta * K
        
        # 消费
        C = Y - I - Phi
        
        return Y, I, Phi, C
    
    def aggregate(
        self,
        K: torch.Tensor,
        C: torch.Tensor,
        alive_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        宏观聚合
        
        K_agg = Σ K_i
        C_agg = Σ C_i
        ln K = log(K_agg)
        ĉ = log(C_agg / K_agg + ε)
        """
        if alive_mask is not None:
            K = K * alive_mask
            C = C * alive_mask
        
        K_total = K.sum()
        C_total = C.sum().clamp(min=0)
        
        lnk_agg = torch.log(K_total + 1e-8)
        hatc_agg = torch.log(C_total / (K_total + 1e-8) + 1e-5)
        
        return hatc_agg, lnk_agg
    
    def forward(
        self,
        hatc_fc2: torch.Tensor,
        lnk_fc2: torch.Tensor,
        hatc_agg: torch.Tensor,
        lnk_agg: torch.Tensor,
        hatc_fc2_trans: Optional[torch.Tensor] = None,
        lnk_fc2_trans: Optional[torch.Tensor] = None,
        hatc_agg_trans: Optional[torch.Tensor] = None,
        lnk_agg_trans: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 FC2 损失
        
        Args:
            hatc_fc2, lnk_fc2: FC2 输出的当期宏观量
            hatc_agg, lnk_agg: 聚合得到的当期宏观量
            hatc_fc2_trans, lnk_fc2_trans: FC2 输出的下期宏观量（可选）
            hatc_agg_trans, lnk_agg_trans: 聚合得到的下期宏观量（可选）
        
        Returns:
            total_loss: 总损失
            loss_dict: 各分量损失字典
        """
        loss_dict = {}
        
        # 当期一致性损失
        loss_node, loss_c, loss_k = self.compute_node_loss(
            hatc_fc2, lnk_fc2, hatc_agg, lnk_agg
        )
        
        loss_dict['loss_node'] = loss_node
        loss_dict['loss_c'] = loss_c
        loss_dict['loss_k'] = loss_k
        
        total_loss = loss_node
        
        # 跨期一致性损失（可选）
        if (self.lambda_trans > 0 and 
            hatc_fc2_trans is not None and 
            hatc_agg_trans is not None):
            
            loss_trans, loss_c_trans, loss_k_trans = self.compute_node_loss(
                hatc_fc2_trans, lnk_fc2_trans,
                hatc_agg_trans, lnk_agg_trans
            )
            
            loss_dict['loss_trans'] = loss_trans
            loss_dict['loss_c_trans'] = loss_c_trans
            loss_dict['loss_k_trans'] = loss_k_trans
            
            total_loss = total_loss + self.lambda_trans * loss_trans
        
        # RMSE 指标（用于监控）
        rmse_hatc = (hatc_fc2 - hatc_agg).pow(2).mean().sqrt()
        rmse_lnk = (lnk_fc2 - lnk_agg).pow(2).mean().sqrt()
        
        loss_dict['rmse_hatc'] = rmse_hatc
        loss_dict['rmse_lnk'] = rmse_lnk
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict
    
    def forward_with_aggregation(
        self,
        hatc_fc2: torch.Tensor,
        lnk_fc2: torch.Tensor,
        K: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
        bar_i: torch.Tensor,
        bar_z: torch.Tensor,
        i: torch.Tensor,
        alive_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        同时进行资源核算和损失计算
        
        这个版本接受原始公司数据，自动完成聚合
        """
        # 资源核算
        Y, I, Phi, C = self.compute_resource_accounting(K, z, x, bar_i, bar_z, i)
        
        # 聚合
        hatc_agg, lnk_agg = self.aggregate(K, C, alive_mask)
        
        # 损失计算
        return self.forward(hatc_fc2, lnk_fc2, hatc_agg, lnk_agg)


class FC2TransitionLoss(nn.Module):
    """
    FC2 跨期一致性损失
    
    考虑状态转移后的一致性：
    1. 用 FC2 给出当期宏观输入
    2. 用 Policy/Value 计算控制变量
    3. 更新状态到下一期
    4. 聚合得到下期宏观量
    5. 与 FC2 对下期的预测对比
    """
    
    def __init__(
        self,
        w_c: float = 1.0,
        w_k: float = 1.0,
        g: float = None,
        delta: float = None,
        phi: float = None
    ):
        super().__init__()
        
        self.w_c = w_c
        self.w_k = w_k
        self.g = g if g is not None else Config.G
        self.delta = delta if delta is not None else Config.DELTA
        self.phi = phi if phi is not None else Config.PHI
    
    def update_capital(
        self,
        K_t: torch.Tensor,
        bar_i: torch.Tensor
    ) -> torch.Tensor:
        """
        资本更新
        
        K_{t+1} = bar_i * G * K_t + (1 - bar_i) * K_t
        """
        return bar_i * self.g * K_t + (1 - bar_i) * K_t
    
    def forward(
        self,
        hatc_fc2_t1: torch.Tensor,  # FC2 对 t+1 的预测
        lnk_fc2_t1: torch.Tensor,
        K_t: torch.Tensor,          # t 期资本
        z_t1: torch.Tensor,         # t+1 期生产率
        x_t1: torch.Tensor,         # t+1 期宏观冲击
        bar_i_t: torch.Tensor,      # t 期投资决策
        bar_z_t: torch.Tensor,      # t 期破产决策
        i_t1: torch.Tensor,         # t+1 期投资成本
        alive_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算跨期一致性损失
        """
        # 更新资本到 t+1
        K_t1 = self.update_capital(K_t, bar_i_t)
        
        # 应用存活掩码（排除破产公司）
        survival_mask = (bar_z_t < 0.5).float()
        if alive_mask is not None:
            survival_mask = survival_mask * alive_mask
        
        # 资源核算（t+1 期）
        Y_t1 = torch.exp(x_t1 + z_t1) * K_t1
        
        # 简化：假设 t+1 期决策用默认值
        bar_i_t1 = torch.zeros_like(bar_i_t)
        bar_z_t1 = torch.zeros_like(bar_z_t)
        
        Phi_t1 = (1 - self.phi) * (1 + torch.exp(x_t1 + z_t1)) * K_t1 * bar_z_t1
        I_t1 = bar_i_t1 * K_t1 * i_t1 - bar_z_t1 * K_t1 + self.delta * K_t1
        C_t1 = Y_t1 - I_t1 - Phi_t1
        
        # 聚合
        K_t1_masked = K_t1 * survival_mask
        C_t1_masked = C_t1 * survival_mask
        
        K_total_t1 = K_t1_masked.sum()
        C_total_t1 = C_t1_masked.sum().clamp(min=0)
        
        lnk_agg_t1 = torch.log(K_total_t1 + 1e-8)
        hatc_agg_t1 = torch.log(C_total_t1 / (K_total_t1 + 1e-8) + 1e-5)
        
        # 损失
        r_c = hatc_fc2_t1 - hatc_agg_t1
        r_k = lnk_fc2_t1 - lnk_agg_t1
        
        loss = self.w_c * r_c.pow(2) + self.w_k * r_k.pow(2)
        
        loss_dict = {
            'loss_trans': loss,
            'r_c': r_c,
            'r_k': r_k,
            'K_total_t1': K_total_t1,
            'C_total_t1': C_total_t1
        }
        
        return loss, loss_dict
