"""
损失函数工具函数

包含各 Loss 共用的工具函数
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def softplus_gate(x: torch.Tensor, delta: float = 1e-3) -> torch.Tensor:
    """
    平滑阶跃函数（softplus gate）
    
    用于在约束边界处提供平滑的过渡。
    gate(x) = softplus(x/δ) / (softplus(x/δ) + softplus(-x/δ))
    
    Args:
        x: 输入张量
        delta: 平滑度参数，控制过渡的陡峭程度。delta 越小，过渡越陡峭。
    
    Returns:
        gate: 平滑的阶跃输出，范围在 [0, 1] 之间
    """
    z = x / delta
    softplus_pos = F.softplus(z)
    softplus_neg = F.softplus(-z)
    gate = softplus_pos / (softplus_pos + softplus_neg + 1e-8)
    return gate


# 别名，保持向后兼容
smooth_step = softplus_gate


def compute_z_penalty(
    residual: torch.Tensor,
    z: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 5.0,
    z0: float = 1.0
) -> torch.Tensor:
    """
    计算 z 值惩罚
    
    penalty = α * mean(sigmoid(β(z - z0)) * residual)
    
    对高 z 值区域的残差给予更高权重
    
    Args:
        residual: 残差张量
        z: z 值张量
        alpha: 惩罚权重
        beta: sigmoid 斜率
        z0: z 的阈值
    
    Returns:
        z 惩罚值
    """
    weight = torch.sigmoid(beta * (z - z0))
    penalty = alpha * (weight * residual).mean()
    return penalty


def compute_monotonicity_penalty(
    value: torch.Tensor,
    inputs: torch.Tensor,
    input_idx: int,
    direction: str = 'negative'
) -> torch.Tensor:
    """
    计算单调性惩罚
    
    通过梯度约束确保函数对某个输入的单调性
    
    Args:
        value: 函数输出值
        inputs: 输入张量（需要开启 requires_grad）
        input_idx: 要检查单调性的输入维度索引
        direction: 'negative'（应单调递减）或 'positive'（应单调递增）
    
    Returns:
        单调性惩罚
    """
    # 计算梯度
    grad = torch.autograd.grad(
        outputs=value.sum(),
        inputs=inputs,
        create_graph=True,
        retain_graph=True
    )[0]
    
    # 取指定维度的梯度
    grad_dim = grad[:, input_idx]
    
    if direction == 'negative':
        # 应该单调递减：惩罚正梯度
        penalty = F.relu(grad_dim).mean()
    else:
        # 应该单调递增：惩罚负梯度
        penalty = F.relu(-grad_dim).mean()
    
    return penalty


def compute_aio_residual(
    losses: list,
    aio_weight: float = 0.5
) -> torch.Tensor:
    """
    计算动态残差插值（AIO）- 支持任意数量的路径
    
    对于 N 条路径：
    - stable = (1/N) * Σ_j loss_j²
    - aio = |Π_j loss_j|
    residual = (1 - w) * stable + w * aio
    
    Args:
        losses: List[torch.Tensor] - 各路径的损失列表
        aio_weight: AIO 权重
    
    Returns:
        组合后的残差
    """
    if not isinstance(losses, (list, tuple)):
        # 向后兼容：如果只传入两个参数
        raise ValueError("losses must be a list of tensors")
    
    n_paths = len(losses)
    
    # stable 部分：平均平方损失
    residual_stable = sum(loss.pow(2) for loss in losses) / n_paths
    
    # AIO 部分：所有路径损失的乘积
    residual_aio = losses[0]
    for loss in losses[1:]:
        residual_aio = residual_aio * loss
    residual_aio = residual_aio.abs()
    
    return (1 - aio_weight) * residual_stable + aio_weight * residual_aio


def compute_aio_residual_legacy(
    loss1: torch.Tensor,
    loss2: torch.Tensor,
    aio_weight: float = 0.5
) -> torch.Tensor:
    """
    计算动态残差插值（AIO）- 旧版双路径接口
    
    residual = (1 - w) * stable + w * aio
    其中：
    - stable = 0.5 * (loss1² + loss2²)
    - aio = |loss1 * loss2|
    
    Args:
        loss1: 路径1的损失
        loss2: 路径2的损失
        aio_weight: AIO 权重
    
    Returns:
        组合后的残差
    """
    return compute_aio_residual([loss1, loss2], aio_weight)


def compute_moment_penalty(
    M: torch.Tensor,
    mu_lo: float = -0.025,
    mu_hi: float = 0.0,
    var_hi: float = 0.25,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 SDF 的矩约束惩罚
    
    一阶矩：log(E[M]) ∈ [μ_lo, μ_hi]
    二阶矩：log(Var[M]) ≤ var_hi
    
    Args:
        M: SDF 张量
        mu_lo: 一阶矩下界
        mu_hi: 一阶矩上界
        var_hi: 二阶矩上界
        eps: 数值稳定性常数
    
    Returns:
        L1: 一阶矩惩罚
        L2: 二阶矩惩罚
    """
    # 一阶矩
    mu = M.mean()
    log_mu = torch.log(mu + eps)
    
    # 一阶矩惩罚
    L_low = smooth_step(mu_lo - log_mu) * 100 * (log_mu - mu_lo).pow(2)
    L_high = smooth_step(log_mu - mu_hi) * 100 * (log_mu - mu_hi).pow(2)
    L1 = L_low + L_high
    
    # 二阶矩
    var = M.pow(2).mean() - mu.pow(2)
    log_var = torch.log(var + eps)
    
    # 二阶矩惩罚
    L2 = smooth_step(log_var - var_hi) * 100 * (log_var - var_hi).pow(2)
    
    return L1, L2


def compute_cashflow(
    x: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    delta: float,
    tau: float
) -> torch.Tensor:
    """
    计算税后利润
    
    prod = (exp(x+z) - δ - b) - τ * max(exp(x+z) - δ - b, 0)
    
    Args:
        x: 宏观生产率
        z: 公司生产率
        b: 杠杆
        delta: 折旧率
        tau: 税率
    
    Returns:
        税后利润
    """
    profit = torch.exp(x + z) - delta - b
    prod = profit - tau * F.relu(profit)
    return prod
