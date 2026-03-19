"""
评估指标
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


def compute_euler_residual(
    M: torch.Tensor,
    R: torch.Tensor = None
) -> Dict[str, float]:
    """
    计算欧拉方程残差
    
    E[M * R] = 1  =>  residual = |E[M * R] - 1|
    
    Args:
        M: SDF 值
        R: 收益率（如果为 None，使用 R=1）
    
    Returns:
        metrics: 残差指标字典
    """
    if R is None:
        R = torch.ones_like(M)
    
    # E[M*R]
    MR = M * R
    E_MR = MR.mean()
    
    # 残差
    residual = torch.abs(E_MR - 1.0)
    
    # 标准误
    std_MR = MR.std()
    n = len(MR)
    se = std_MR / (n ** 0.5)
    
    # t-统计量
    t_stat = (E_MR - 1.0) / (se + 1e-8)
    
    return {
        'euler_residual': residual.item(),
        'E_MR': E_MR.item(),
        'std_MR': std_MR.item(),
        't_stat': t_stat.item()
    }


def compute_resource_balance(
    Y: torch.Tensor,
    C: torch.Tensor,
    I: torch.Tensor,
    Phi: torch.Tensor = None
) -> Dict[str, float]:
    """
    计算资源平衡
    
    Y = C + I + Φ
    
    Args:
        Y: 产出
        C: 消费
        I: 投资
        Phi: 调整成本
    
    Returns:
        metrics: 资源平衡指标
    """
    if Phi is None:
        Phi = torch.zeros_like(Y)
    
    # 总量
    Y_sum = Y.sum()
    C_sum = C.sum()
    I_sum = I.sum()
    Phi_sum = Phi.sum()
    
    # 残差
    residual = Y_sum - C_sum - I_sum - Phi_sum
    
    # 比例
    metrics = {
        'Y_total': Y_sum.item(),
        'C_total': C_sum.item(),
        'I_total': I_sum.item(),
        'Phi_total': Phi_sum.item(),
        'resource_residual': residual.item(),
        'C_Y_ratio': (C_sum / Y_sum).item() if Y_sum > 0 else 0,
        'I_Y_ratio': (I_sum / Y_sum).item() if Y_sum > 0 else 0,
        'Phi_Y_ratio': (Phi_sum / Y_sum).item() if Y_sum > 0 else 0
    }
    
    return metrics


def compute_leverage_moments(
    b: torch.Tensor,
    weights: torch.Tensor = None
) -> Dict[str, float]:
    """
    计算杠杆分布的矩
    
    Args:
        b: 杠杆值
        weights: 权重（如资本权重）
    
    Returns:
        moments: 矩统计
    """
    if weights is None:
        weights = torch.ones_like(b)
    
    weights = weights / weights.sum()
    
    # 均值
    mean = (weights * b).sum()
    
    # 方差
    var = (weights * (b - mean) ** 2).sum()
    std = var ** 0.5
    
    # 偏度
    skew = (weights * ((b - mean) / (std + 1e-8)) ** 3).sum()
    
    # 峰度
    kurt = (weights * ((b - mean) / (std + 1e-8)) ** 4).sum()
    
    # 分位数
    sorted_b, sorted_idx = b.sort()
    cum_weights = weights[sorted_idx].cumsum(0)
    
    def get_quantile(p):
        idx = (cum_weights >= p).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            return sorted_b[idx[0]].item()
        return sorted_b[-1].item()
    
    return {
        'b_mean': mean.item(),
        'b_std': std.item(),
        'b_skew': skew.item(),
        'b_kurt': kurt.item(),
        'b_q05': get_quantile(0.05),
        'b_q25': get_quantile(0.25),
        'b_q50': get_quantile(0.50),
        'b_q75': get_quantile(0.75),
        'b_q95': get_quantile(0.95)
    }


def compute_model_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.1
) -> Dict[str, float]:
    """
    计算模型预测准确度
    
    Args:
        pred: 预测值
        target: 目标值
        threshold: 准确度阈值
    
    Returns:
        metrics: 准确度指标
    """
    # 误差
    error = pred - target
    abs_error = torch.abs(error)
    rel_error = abs_error / (torch.abs(target) + 1e-8)
    
    # MAE, MSE, RMSE
    mae = abs_error.mean()
    mse = (error ** 2).mean()
    rmse = mse ** 0.5
    
    # MAPE
    mape = rel_error.mean()
    
    # R²
    ss_res = (error ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    # 阈值准确度
    accuracy = (rel_error < threshold).float().mean()
    
    return {
        'mae': mae.item(),
        'mse': mse.item(),
        'rmse': rmse.item(),
        'mape': mape.item(),
        'r2': r2.item(),
        f'accuracy_{int(threshold*100)}pct': accuracy.item()
    }


def compute_bellman_residual(
    V_t: torch.Tensor,
    V_t1: torch.Tensor,
    M: torch.Tensor,
    d: torch.Tensor = None,
    beta: float = 0.96
) -> Dict[str, float]:
    """
    计算 Bellman 方程残差
    
    V_t = d_t + β * E_t[M * V_{t+1}]
    
    Args:
        V_t: 当期价值
        V_t1: 下期价值
        M: SDF
        d: 股息（如果为 None，假设为 0）
        beta: 折扣因子
    
    Returns:
        metrics: Bellman 残差指标
    """
    if d is None:
        d = torch.zeros_like(V_t)
    
    # RHS
    E_MV = M * V_t1
    rhs = d + beta * E_MV.mean()
    
    # 残差
    residual = V_t - rhs
    
    return {
        'bellman_residual_mean': residual.mean().item(),
        'bellman_residual_std': residual.std().item(),
        'bellman_residual_abs_mean': torch.abs(residual).mean().item()
    }


def compute_market_clearing(
    supply: torch.Tensor,
    demand: torch.Tensor
) -> Dict[str, float]:
    """
    计算市场出清残差
    
    Args:
        supply: 供给
        demand: 需求
    
    Returns:
        metrics: 市场出清指标
    """
    residual = supply - demand
    
    return {
        'clearing_residual': residual.sum().item(),
        'clearing_residual_rel': (residual.sum() / (supply.sum() + 1e-8)).item()
    }


def aggregate_metrics(metrics_list: list) -> Dict[str, float]:
    """
    聚合多个指标字典
    
    Args:
        metrics_list: 指标字典列表
    
    Returns:
        aggregated: 聚合后的指标
    """
    if len(metrics_list) == 0:
        return {}
    
    keys = metrics_list[0].keys()
    aggregated = {}
    
    for key in keys:
        values = [m[key] for m in metrics_list if key in m]
        if len(values) > 0:
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
    
    return aggregated
