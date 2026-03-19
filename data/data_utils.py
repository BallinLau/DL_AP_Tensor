"""
数据生成工具函数
"""

import torch
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

import sys
sys.path.append('..')
from config import Config


def sample_ar1(
    x_prev: torch.Tensor,
    rho: float,
    sigma: float,
    mean: float = 0.0
) -> torch.Tensor:
    """
    AR(1) 转移采样
    
    x_t = (1 - ρ) * μ + ρ * x_{t-1} + σ * ε
    
    Args:
        x_prev: 上期值
        rho: 自相关系数
        sigma: 创新标准差
        mean: 无条件均值
    
    Returns:
        x_t: 当期值
    """
    eps = torch.randn_like(x_prev)
    x_t = (1 - rho) * mean + rho * x_prev + sigma * eps
    return x_t


def sample_stationary_ar1(
    n_samples: int,
    rho: float,
    sigma: float,
    mean: float = 0.0,
    device: torch.device = None
) -> torch.Tensor:
    """
    从 AR(1) 稳态分布采样
    
    稳态分布：N(μ, σ² / (1 - ρ²))
    
    Args:
        n_samples: 样本数
        rho: 自相关系数
        sigma: 创新标准差
        mean: 无条件均值
        device: 设备
    
    Returns:
        samples: 稳态分布样本
    """
    if device is None:
        device = Config.DEVICE
    
    stationary_var = sigma**2 / (1 - rho**2)
    stationary_std = stationary_var ** 0.5
    
    samples = torch.randn(n_samples, device=device) * stationary_std + mean
    return samples


def sample_uniform(
    n_samples: int,
    low: float = 0.0,
    high: float = 1.0,
    device: torch.device = None
) -> torch.Tensor:
    """
    均匀分布采样
    """
    if device is None:
        device = Config.DEVICE
    
    return torch.rand(n_samples, device=device) * (high - low) + low


def sample_bernoulli(
    n_samples: int,
    p: float = 0.5,
    device: torch.device = None
) -> torch.Tensor:
    """
    伯努利分布采样
    """
    if device is None:
        device = Config.DEVICE
    
    return (torch.rand(n_samples, device=device) < p).float()


def compute_quantile_features(
    values: torch.Tensor,
    n_quantiles: int = 100
) -> torch.Tensor:
    """
    计算分位数特征
    
    Args:
        values: (n,) 输入值
        n_quantiles: 分位数数量
    
    Returns:
        quantiles: (n_quantiles,) 分位数
    """
    probs = torch.linspace(0, 0.99, n_quantiles, device=values.device)
    return torch.quantile(values, probs)


def create_triplet_structure(
    df: pd.DataFrame,
    path_col: str = 'path',
    id_col: str = 'ID',
    branch_num: int = 2
) -> pd.DataFrame:
    """
    将 DataFrame 组织为 triplet 结构
    
    每个 (path, ID) 对应 (branch_num + 1) 行：
    - 第 0 行：parent (t)
    - 第 1 行：child branch 0 (t+1)
    - 第 2 行：child branch 1 (t+1')
    
    Args:
        df: 输入 DataFrame
        path_col: path 列名
        id_col: ID 列名
        branch_num: 分支数
    
    Returns:
        重组后的 DataFrame
    """
    # 确保数据已排序
    df = df.sort_values([path_col, id_col, 't']).reset_index(drop=True)
    
    # 检查每个 (path, ID) 是否有正确的行数
    group_sizes = df.groupby([path_col, id_col]).size()
    expected_size = branch_num + 1
    
    if not (group_sizes == expected_size).all():
        raise ValueError(f"每个 (path, ID) 应有 {expected_size} 行")
    
    return df


def compute_profit(
    x: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    delta: float = None
) -> torch.Tensor:
    """
    计算利润
    
    profit = exp(x + z) - δ - b
    """
    if delta is None:
        delta = Config.DELTA
    
    return torch.exp(x + z) - delta - b


def filter_profitable(
    b: torch.Tensor,
    z: torch.Tensor,
    x: torch.Tensor,
    delta: float = None
) -> torch.Tensor:
    """
    筛选利润为正的样本
    
    返回布尔掩码
    """
    profit = compute_profit(x, z, b, delta)
    return profit > 0


def generate_macro_state(
    n_paths: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    生成宏观状态 x
    
    从 AR(1) 稳态分布采样
    """
    return sample_stationary_ar1(
        n_paths,
        Config.RHO_X,
        Config.SIGMA_X,
        Config.XBAR,
        device
    )


def generate_firm_states(
    n_firms: int,
    x: torch.Tensor,
    hatcf: Optional[torch.Tensor] = None,
    lnkf: Optional[torch.Tensor] = None,
    mode: str = 'uniform',
    device: torch.device = None
) -> Tuple[torch.Tensor, ...]:
    """
    生成公司状态
    
    Args:
        n_firms: 公司数
        x: 宏观状态（共享）
        hatcf: 宏观消费相关状态
        lnkf: 宏观资本对数
        mode: 'uniform' 或 'feasible'（利润为正的可行域）
        device: 设备
    
    Returns:
        b, z, eta, i: 公司状态
    """
    if device is None:
        device = Config.DEVICE
    
    # 基础采样
    if mode == 'uniform':
        b = sample_uniform(n_firms, 0.0, 1.0, device)
        z = sample_stationary_ar1(n_firms, Config.RHO_Z, Config.SIGMA_Z, Config.ZBAR, device)
    elif mode == 'feasible':
        # 先采样，然后筛选利润为正的
        max_attempts = 10
        valid_samples = []
        
        for _ in range(max_attempts):
            b_tmp = sample_uniform(n_firms * 2, 0.0, 1.0, device)
            z_tmp = sample_stationary_ar1(n_firms * 2, Config.RHO_Z, Config.SIGMA_Z, Config.ZBAR, device)
            
            mask = filter_profitable(b_tmp, z_tmp, x.expand(n_firms * 2))
            valid_b = b_tmp[mask]
            valid_z = z_tmp[mask]
            
            valid_samples.append((valid_b, valid_z))
            
            if sum(len(s[0]) for s in valid_samples) >= n_firms:
                break
        
        b = torch.cat([s[0] for s in valid_samples])[:n_firms]
        z = torch.cat([s[1] for s in valid_samples])[:n_firms]
        
        # 如果样本不够，用均匀采样补充
        if len(b) < n_firms:
            n_missing = n_firms - len(b)
            b = torch.cat([b, sample_uniform(n_missing, 0.0, 1.0, device)])
            z = torch.cat([z, sample_stationary_ar1(n_missing, Config.RHO_Z, Config.SIGMA_Z, Config.ZBAR, device)])
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # 其他状态
    eta = sample_bernoulli(n_firms, Config.ZETA, device)
    i = sample_uniform(n_firms, 0.0, Config.I_THRESHOLD, device)
    
    return b, z, eta, i


def build_sdf_pairs_from_macro_ts(
    df_macro_ts: pd.DataFrame,
    path_col: str = 'path',
    t_col: str = 't',
    branch_col: str = 'branch',
    parent_branch: int = -1,
    child_branch_min: int = 0,
    x_col: str = 'x',
    hatcf_col: str = 'hatcf',
    lnkf_col: str = 'lnkf',
    hatc_col: str = 'Hatc',
    lnk_col: str = 'LnK',
    include_hatc_lnk_t1: bool = True
) -> pd.DataFrame:
    """
    从 SimulateTS 的 df_macro_ts 构造 SDF/FC1 训练用的跨期配对表。

    约定：
    - parent 行：branch == parent_branch (默认 -1)
    - child 行：branch >= child_branch_min (默认 0)
    - child 的 t 对应 parent 的 t+1，因此 merge 使用 t_parent = t_child - 1

    返回列：
    - path, t, branch, x_t, x_t1, Hatcf_t, LnKF_t, Hatc_t, LnK_t
    - 如果 include_hatc_lnk_t1=True，还会包含 Hatc_t1, LnK_t1
    """
    required_parent = {path_col, t_col, branch_col, x_col, hatcf_col, lnkf_col, hatc_col, lnk_col}
    missing = required_parent - set(df_macro_ts.columns)
    if missing:
        raise KeyError(f"df_macro_ts missing columns: {sorted(missing)}")

    parent = df_macro_ts[df_macro_ts[branch_col] == parent_branch][
        [path_col, t_col, x_col, hatcf_col, lnkf_col, hatc_col, lnk_col]
    ].rename(columns={
        t_col: 't_parent',
        x_col: 'x_t',
        hatcf_col: 'Hatcf_t',
        lnkf_col: 'LnKF_t',
        hatc_col: 'Hatc_t',
        lnk_col: 'LnK_t',
    })

    child_cols = [path_col, t_col, branch_col, x_col]
    if include_hatc_lnk_t1:
        child_cols += [hatc_col, lnk_col]

    child = df_macro_ts[df_macro_ts[branch_col] >= child_branch_min][child_cols].rename(columns={
        x_col: 'x_t1',
        hatc_col: 'Hatc_t1',
        lnk_col: 'LnK_t1',
    })

    # t 必须是数值型（SimulateTS 使用 int）
    if not pd.api.types.is_numeric_dtype(child[t_col]):
        child[t_col] = pd.to_numeric(child[t_col], errors='coerce')
    if not pd.api.types.is_numeric_dtype(parent['t_parent']):
        parent['t_parent'] = pd.to_numeric(parent['t_parent'], errors='coerce')

    child['t_parent'] = child[t_col] - 1

    df_sdf = child.merge(parent, on=[path_col, 't_parent'], how='left')

    base_cols = [
        path_col, t_col, branch_col,
        'x_t', 'x_t1',
        'Hatcf_t', 'LnKF_t',
        'Hatc_t', 'LnK_t',
    ]
    if include_hatc_lnk_t1:
        base_cols += ['Hatc_t1', 'LnK_t1']

    return df_sdf[base_cols]


def generate_initial_macro_proxy(
    n_firms: int,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成初始宏观 proxy（ĉf, ln Kf）
    
    使用合理的初始值
    """
    if device is None:
        device = Config.DEVICE
    
    # 初始化为合理的默认值，加入轻微随机扰动
    hatcf = torch.randn(n_firms, device=device) * 0.1 - 2.0  # log(consumption/capital ratio)
    lnkf = torch.randn(n_firms, device=device) * 0.2 + 4.0   # log(capital) ~ log(50) with noise
    
    return hatcf, lnkf


def build_firm_state_tensor(
    b: torch.Tensor,
    z: torch.Tensor,
    eta: torch.Tensor,
    i: torch.Tensor,
    x: torch.Tensor,
    hatcf: torch.Tensor,
    lnkf: torch.Tensor
) -> torch.Tensor:
    """
    构建 firm-state 张量
    
    输出形状：(n_firms, 7)
    列顺序：(b, z, η, i, x, ĉf, ln Kf)
    """
    # 确保所有张量有正确的形状
    n = len(b)
    
    # x, hatcf, lnkf 可能需要广播
    if x.dim() == 0:
        x = x.expand(n)
    if hatcf.dim() == 0:
        hatcf = hatcf.expand(n)
    if lnkf.dim() == 0:
        lnkf = lnkf.expand(n)
    
    # 确保是 1D
    b = b.view(-1)
    z = z.view(-1)
    eta = eta.view(-1)
    i = i.view(-1)
    x = x.view(-1)[:n] if len(x) > n else x.expand(n)
    hatcf = hatcf.view(-1)[:n] if len(hatcf) > n else hatcf.expand(n)
    lnkf = lnkf.view(-1)[:n] if len(lnkf) > n else lnkf.expand(n)
    
    return torch.stack([b, z, eta, i, x, hatcf, lnkf], dim=1)


def expand_to_branches(
    parent_state: torch.Tensor,
    branch_num: int = 2
) -> torch.Tensor:
    """
    将 parent 状态扩展到多个分支
    
    Args:
        parent_state: (n_firms, 7)
        branch_num: 分支数
    
    Returns:
        expanded_state: (n_firms * (branch_num + 1), 7)
    """
    n_firms = parent_state.shape[0]
    
    # 每个 parent 扩展为 (1 + branch_num) 行
    expanded = parent_state.repeat_interleave(branch_num + 1, dim=0)
    
    return expanded
