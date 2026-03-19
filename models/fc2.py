"""
FC2 模型

FC2: 横截面分布 → 宏观 proxy

输入：
- b 的 100 分位点
- z 的 100 分位点
- 宏观量 x

输出：
- (ĉ, ln K) 宏观 proxy
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict
from .base import MLP, MLPWithScaler

import sys
sys.path.append('..')
from config import Config


class FC2Model(nn.Module):
    """
    FC2 模型：横截面分布 → 宏观 proxy
    
    将节点内横截面分布特征映射到宏观状态
    核心作用：实现 fixed-point consistency
    
    (ĉ, ln K) = A(Policy/Value(ĉ, ln K; s_i))
    """
    
    def __init__(
        self,
        input_dim: int = None,
        hidden_dims: List[int] = None,
        output_dim: int = 2,
        quantile_num: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 输入维度：100(b分位点) + 100(z分位点) + x = 201
        if input_dim is None:
            input_dim = 2 * quantile_num + 1
        
        if hidden_dims is None:
            hidden_dims = Config.FC2_HIDDEN_DIMS
        
        self.quantile_num = quantile_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 主网络
        self.network = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation='gelu',
            dropout=dropout
        )
    
    
    def forward(
        self, 
        phi: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            phi: (batch, 201) - [b_quantiles(100), z_quantiles(100), x]
        
        Returns:
            {'hatc': (batch,1), 'lnk': (batch,1)}
        """
        output = self.network(phi)
        # 保持原有语义：第 0 维为 lnK，第 1 维为 ĉ
        lnk = output[:, 0:1]
        hatc = output[:, 1:2]
        
        return {'hatc': hatc, 'lnk': lnk}
    


class FC2WithAggregation(nn.Module):
    """
    FC2 + 聚合算子
    
    用于训练 FC2 时同时计算：
    1. FC2 输出的宏观量
    2. 通过 Policy/Value 聚合得到的真实宏观量
    
    损失 = ||FC2_output - Aggregated_output||^2
    """
    
    def __init__(
        self,
        fc2_model: FC2Model,
        config: type = Config
    ):
        super().__init__()
        
        self.fc2 = fc2_model
        self.config = config
        
        # 经济参数
        self.delta = config.DELTA
        self.phi = config.PHI
    
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
        
        Args:
            K: (n_firms,) - 公司资本
            z: (n_firms,) - 公司生产率
            x: scalar - 宏观生产率
            bar_i: (n_firms,) - 投资决策
            bar_z: (n_firms,) - 破产决策
            i: (n_firms,) - 投资成本
        
        Returns:
            Y: 产出
            I: 投资
            Phi: 破产成本
            C: 消费
        """
        # 产出
        Y = torch.exp(x + z) * K
        
        # 破产/调整成本
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
        
        Args:
            K: (n_firms,) - 公司资本
            C: (n_firms,) - 公司消费
            alive_mask: (n_firms,) - 存活掩码
        
        Returns:
            hatc_agg: 聚合的 ĉ
            lnk_agg: 聚合的 ln K
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
        phi: torch.Tensor,
        K: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
        bar_i: torch.Tensor,
        bar_z: torch.Tensor,
        i: torch.Tensor,
        alive_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        同时计算 FC2 输出和聚合输出
        
        Returns:
            fc2_output: (hatc_fc2, lnk_fc2)
            agg_output: (hatc_agg, lnk_agg)
        """
        # FC2 输出
        fc2_out = self.fc2(phi)
        hatc_fc2 = fc2_out['hatc']
        lnk_fc2 = fc2_out['lnk']
        
        # 资源核算
        Y, I, Phi, C = self.compute_resource_accounting(
            K, z, x, bar_i, bar_z, i
        )
        
        # 聚合
        hatc_agg, lnk_agg = self.aggregate(K, C, alive_mask)
        
        return (hatc_fc2, lnk_fc2), (hatc_agg, lnk_agg)
