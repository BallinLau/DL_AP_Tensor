"""
SDF Loss: 随机贴现因子损失

核心目标：拟合欧拉方程约束，确保所有路径下 SDF 残差同时为零，
叠加 M 的矩约束惩罚，保证统计性质合理。

树状分支结构（支持任意 N 个分支）：
- parent: (w_t, k_t, c_t) → t 期父节点
- children: [(w_{t+1}^{(j)}, k_{t+1}^{(j)}, c_{t+1}^{(j)})] → N 条模拟路径
- 每条路径中 η 通过伯努利分布随机抽取

SDF 计算（对每条路径 j）：
    M^{(j)} = β^κ * exp((k_{t+1}^{(j)} - k_t)*(-γ) - κ/σ*(c_{t+1}^{(j)} - c_t)) 
              * (w_{t+1}^{(j)} / (w_t - exp(c_t)))^(κ-1)

欧拉方程残差（对每条路径 j）：
    loss^{(j)} = exp_term^{(j)} * w_{t+1}^{(j)κ} - (w_t - exp(c_t))^κ = 0

主损失：所有路径残差的乘积（要求同时为零）
L_SDF = main_loss + Σ_j (L1_{M^{(j)}} + L2_{M^{(j)}})
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List, Union

import sys
sys.path.append('..')
from config import Config

from .utils import softplus_gate


def moment_penalty(
    M: torch.Tensor,
    mu_lo: float,
    mu_hi: float,
    var_hi: float,
    delta: float = 1e-3,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算随机贴现因子 M 的矩约束惩罚项。

    对 M 的对数均值和对数方差施加约束：
    - 一阶矩（均值）：mu_lo <= log(E[M]) <= mu_hi
    - 二阶矩（方差）：log(Var[M]) <= var_hi

    Args:
        M: 随机贴现因子张量，shape [batch]
        mu_lo: 对数均值的下界
        mu_hi: 对数均值的上界
        var_hi: 对数方差的上界
        delta: 平滑阶跃函数的平滑度参数
        eps: 数值稳定项，避免 log(0)

    Returns:
        L1: 一阶矩惩罚项（均值约束）
        L2: 二阶矩惩罚项（方差约束）
    """
    # 计算均值和方差
    mu = M.mean()  # E[M]
    var = (M * M).mean() - mu * mu  # Var[M] = E[M^2] - E[M]^2

    # 平滑阶跃函数
    step = lambda x: softplus_gate(x, delta)

    # 一阶矩惩罚：当 log(E[M]) 超出 [mu_lo, mu_hi] 时施加惩罚
    log_mu = torch.log(mu + eps)
    L_low = step(mu_lo - log_mu) * 100.0 * (log_mu - mu_lo) ** 2
    L_high = step(log_mu - mu_hi) * 100.0 * (log_mu - mu_hi) ** 2
    L1 = L_low + L_high

    # 二阶矩惩罚：当 log(Var[M]) > var_hi 时施加惩罚
    log_var = torch.log(var + eps)
    L2 = step(log_var - var_hi) * 100.0 * (log_var - var_hi) ** 2

    return L1, L2


class SDFLoss(nn.Module):
    """
    SDF 损失函数（支持任意分支数）
    
    基于欧拉方程残差和矩约束
    """
    
    def __init__(
        self,
        gamma: float = None,
        kappa: float = None,
        sigma: float = None,
        beta: float = None,
        mu_lo: float = -0.025,
        mu_hi: float = 0.0,
        var_hi: float = 0.25
    ):
        super().__init__()
        
        # 从配置获取参数
        self.gamma = gamma if gamma is not None else Config.GAMMA
        self.kappa = kappa if kappa is not None else Config.KAPPA
        self.sigma = sigma if sigma is not None else Config.SIGMA
        self.beta = beta if beta is not None else Config.BETA
        self.mu_lo = mu_lo
        self.mu_hi = mu_hi
        self.var_hi = var_hi
        
        # 预计算 β^κ
        self.tmp = self.beta ** self.kappa
    
    def compute_euler_residuals(
        self,
        w_parent: torch.Tensor,
        w_children: Union[List[torch.Tensor], torch.Tensor],
        k_parent: torch.Tensor,
        k_children: Union[List[torch.Tensor], torch.Tensor],
        c_parent: torch.Tensor,
        c_children: Union[List[torch.Tensor], torch.Tensor],
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        计算所有路径的欧拉方程残差
        
        loss^{(j)} = exp_term^{(j)} * w_{t+1}^{(j)κ} - (w_t - exp(c_t))^κ
        
        Args:
            w_parent: (batch,) 父节点价值函数
            w_children: List[(batch,)] 子节点价值函数列表
            k_parent: (batch,) 父节点资本对数
            k_children: List[(batch,)] 子节点资本对数列表
            c_parent: (batch,) 父节点消费对数
            c_children: List[(batch,)] 子节点消费对数列表
        
        Returns:
            residuals: List[torch.Tensor] 或 (batch, n_children) 张量
        """
        # 张量路径：(batch, n_children) 或 (batch, n_children, 1)
        if isinstance(w_children, torch.Tensor):
            w_children_t = w_children
            k_children_t = k_children
            c_children_t = c_children

            if w_children_t.dim() == 3 and w_children_t.size(-1) == 1:
                w_children_t = w_children_t.squeeze(-1)
                k_children_t = k_children_t.squeeze(-1)
                c_children_t = c_children_t.squeeze(-1)
            if w_children_t.dim() == 1:
                w_children_t = w_children_t.unsqueeze(-1)
                k_children_t = k_children_t.unsqueeze(-1)
                c_children_t = c_children_t.unsqueeze(-1)

            w_parent_t = w_parent.squeeze(-1) if w_parent.dim() > 1 else w_parent
            k_parent_t = k_parent.squeeze(-1) if k_parent.dim() > 1 else k_parent
            c_parent_t = c_parent.squeeze(-1) if c_parent.dim() > 1 else c_parent

            current_value = (w_parent_t - torch.exp(c_parent_t)).clamp_min(1e-8)
            current_value_kappa = torch.pow(current_value, self.kappa).unsqueeze(-1)

            exp_term = torch.exp(
                (k_children_t - k_parent_t.unsqueeze(-1)) * (1.0 - self.gamma)
                - self.kappa / self.sigma * (c_children_t - c_parent_t.unsqueeze(-1))
            ) * self.tmp

            residuals = exp_term * torch.pow(w_children_t, self.kappa) - current_value_kappa
            return residuals

        # List 路径（逐分支）
        current_value = (w_parent - torch.exp(c_parent)).clamp_min(1e-8)
        current_value_kappa = torch.pow(current_value, self.kappa)

        residuals = []
        for w_child, k_child, c_child in zip(w_children, k_children, c_children):
            exp_term = torch.exp(
                (k_child - k_parent) * (1.0 - self.gamma)
                - self.kappa / self.sigma * (c_child - c_parent)
            ) * self.tmp
            residual = exp_term * torch.pow(w_child, self.kappa) - current_value_kappa
            residuals.append(residual)

        return residuals
    
    def forward(
        self,
        w_parent: torch.Tensor,
        w_children: List[torch.Tensor],
        M_list: List[torch.Tensor],
        k_parent: torch.Tensor,
        k_children: List[torch.Tensor],
        c_parent: torch.Tensor,
        c_children: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        计算 SDF 损失（支持任意分支数）
        
        Args:
            w_parent: (batch,) 父节点价值函数
            w_children: List[(batch,)] 子节点价值函数列表
            M_list: List[(batch,)] 每条路径的 SDF
            k_parent: (batch,) 父节点资本对数
            k_children: List[(batch,)] 子节点资本对数列表
            c_parent: (batch,) 父节点消费对数
            c_children: List[(batch,)] 子节点消费对数列表
        
        Returns:
            final_loss: 总损失
        """
        # 1. 计算所有路径的欧拉方程残差
        residuals = self.compute_euler_residuals(
            w_parent, w_children,
            k_parent, k_children,
            c_parent, c_children
        )
        
        # 2. 主损失：所有路径残差的乘积（要求同时为零）
        if isinstance(residuals, list):
            raw_loss = residuals[0]
            for residual in residuals[1:]:
                raw_loss = raw_loss * residual
        else:
            raw_loss = residuals.prod(dim=-1)
        
        # 使用 log1p 平滑
        main_loss = torch.mean(torch.log1p(raw_loss.abs()))
        
        # 3. 矩约束惩罚（所有路径的 M）
        moment_loss = torch.tensor(0.0, device=w_parent.device)
        if isinstance(M_list, torch.Tensor):
            M = M_list
            if M.dim() == 3 and M.size(-1) == 1:
                M = M.squeeze(-1)
            if M.dim() == 1:
                M = M.unsqueeze(-1)
            for j in range(M.shape[1]):
                L1, L2 = moment_penalty(M[:, j], self.mu_lo, self.mu_hi, self.var_hi)
                moment_loss = moment_loss + L1 + L2
        else:
            for M in M_list:
                L1, L2 = moment_penalty(M, self.mu_lo, self.mu_hi, self.var_hi)
                moment_loss = moment_loss + L1 + L2
        
        # 总损失
        final_loss = main_loss + moment_loss
        
        return final_loss
    
    def forward_legacy(
        self,
        outputs: Tuple[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # w
            torch.Tensor,  # M1
            torch.Tensor,  # M2
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # kf
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # cf
        ],
        c: None = None,
        k: None = None,
    ) -> torch.Tensor:
        """
        旧版接口，兼容2分支情况
        
        Args:
            outputs: (w, M1, M2, kf, cf) 格式
        """
        w, M1, M2, kf, cf = outputs
        w1, w2, w3 = w
        k1, k2, k3 = kf
        c1, c2, c3 = cf
        
        return self.forward(
            w_parent=w1,
            w_children=[w2, w3],
            M_list=[M1, M2],
            k_parent=k1,
            k_children=[k2, k3],
            c_parent=c1,
            c_children=[c2, c3]
        )
    
    def forward_with_details(
        self,
        w_parent: torch.Tensor,
        w_children: List[torch.Tensor],
        M_list: List[torch.Tensor],
        k_parent: torch.Tensor,
        k_children: List[torch.Tensor],
        c_parent: torch.Tensor,
        c_children: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 SDF 损失并返回详细信息
        """
        # 计算残差
        residuals = self.compute_euler_residuals(
            w_parent, w_children,
            k_parent, k_children,
            c_parent, c_children
        )
        
        # 主损失
        if isinstance(residuals, list):
            raw_loss = residuals[0]
            for residual in residuals[1:]:
                raw_loss = raw_loss * residual
        else:
            raw_loss = residuals.prod(dim=-1)
        main_loss = torch.mean(torch.log1p(raw_loss.abs()))
        
        # 矩约束惩罚
        loss_dict = {'main_loss': main_loss}
        moment_loss = torch.tensor(0.0, device=w_parent.device)
        
        if isinstance(M_list, torch.Tensor):
            M = M_list
            if M.dim() == 3 and M.size(-1) == 1:
                M = M.squeeze(-1)
            if M.dim() == 1:
                M = M.unsqueeze(-1)
            for j in range(M.shape[1]):
                L1, L2 = moment_penalty(M[:, j], self.mu_lo, self.mu_hi, self.var_hi)
                loss_dict[f'L1_M{j+1}'] = L1
                loss_dict[f'L2_M{j+1}'] = L2
                loss_dict[f'M{j+1}_mean'] = M[:, j].mean()
                moment_loss = moment_loss + L1 + L2
        else:
            for j, M in enumerate(M_list):
                L1, L2 = moment_penalty(M, self.mu_lo, self.mu_hi, self.var_hi)
                loss_dict[f'L1_M{j+1}'] = L1
                loss_dict[f'L2_M{j+1}'] = L2
                loss_dict[f'M{j+1}_mean'] = M.mean()
                moment_loss = moment_loss + L1 + L2
        
        # 残差信息
        if isinstance(residuals, list):
            for j, residual in enumerate(residuals):
                loss_dict[f'residual{j+1}_mean'] = residual.mean()
        else:
            for j in range(residuals.shape[1]):
                loss_dict[f'residual{j+1}_mean'] = residuals[:, j].mean()
        
        final_loss = main_loss + moment_loss
        
        return final_loss, loss_dict


# 便捷函数
def SDFloss(outputs, c=None, k=None):
    """函数式接口（旧版兼容）"""
    loss_fn = SDFLoss()
    return loss_fn.forward_legacy(outputs, c, k)
