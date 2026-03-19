# DL-AP Data Module
from .sample import Sample
from .simulate_ts import SimulateTS
from .data_utils import (
    sample_ar1,
    sample_stationary_ar1,
    compute_quantile_features,
    create_triplet_structure
)
from .tensor_data import TensorTable, TensorSimulationOutput


def create_sample_data(models=None, n_paths=1000, **kwargs):
    """
    创建 sample 模式数据（每 path 2 家公司）
    用于训练调试
    
    Returns:
        df_firm: firm-level DataFrame
    """
    return Sample(models=models, n_paths=n_paths, data_mode='sample', **kwargs)


def create_simulate_data(models=None, n_paths=100, **kwargs):
    """
    创建 simulate 模式数据（每 path 200 家公司）
    用于训练 SDF&FC1，需要宏观聚合
    
    注意：这不是 SimulateTS 的时间序列模拟，而是截面样本生成
    
    Returns:
        调用 build_df() 后返回 (df_firm, df_macro) 元组
    """
    return Sample(models=models, n_paths=n_paths, data_mode='simulate', **kwargs)


__all__ = [
    'Sample',
    'SimulateTS',
    'sample_ar1',
    'sample_stationary_ar1',
    'compute_quantile_features',
    'create_triplet_structure',
    'TensorTable',
    'TensorSimulationOutput',
    'create_sample_data',
    'create_simulate_data'
]
