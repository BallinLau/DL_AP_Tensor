"""
Policy & Value 统一接口

整合 SharedModel, CombinedModel, BarzModel, BariModel
提供统一的 forward 和工具函数
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, NamedTuple
from .share_layer import ShareLayer, SharedModel, CombinedModel, BarzModel, BariModel

import sys
sys.path.append('..')
from config import Config, SIMMODEL


class PolicyValueOutput(NamedTuple):
    """Policy & Value 模型的输出"""
    Q: torch.Tensor          # 债券价值
    bp0: torch.Tensor        # 不投资时杠杆候选
    bpI: torch.Tensor        # 投资时杠杆候选
    P0: torch.Tensor         # 不投资时股票价值
    PI: torch.Tensor         # 投资时股票价值
    bar_i: torch.Tensor      # 投资门槛
    bar_z: torch.Tensor      # 破产门槛
    P: torch.Tensor          # 综合股票价值
    Phat: torch.Tensor       # P hat (中间值)
    bp: torch.Tensor         # 综合杠杆候选


class PolicyValueModel(nn.Module):
    """
    Policy & Value 统一模型
    
    整合：
    - SharedModel: Q, bp0, bpI
    - CombinedModel: P0, PI, bar_i
    - BarzModel: bar_z
    - BariModel: bar_i (value version)
    
    并提供计算 P, Phat, bp 的工具函数
    """
    
    def __init__(
        self,
        base_state_dim: int = 6,
        share_hidden_dims: Optional[list] = None,
        share_output_dim: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        
        share_layer = ShareLayer(
            input_dim=base_state_dim,
            hidden_dims=share_hidden_dims,
            output_dim=share_output_dim,
            dropout=dropout
        )
        
        # 主要模型
        self.shared_model = SharedModel(
            base_state_dim=base_state_dim,
            share_hidden_dims=share_hidden_dims,
            share_output_dim=share_output_dim,
            dropout=dropout,
            share_layer=share_layer
        )
        
        self.combined_model = CombinedModel(
            base_state_dim=base_state_dim,
            share_hidden_dims=share_hidden_dims,
            share_output_dim=share_output_dim,
            dropout=dropout,
            share_layer=share_layer
        )
        
        # 辅助模型
        self.barz_model = BarzModel(
            input_dim=base_state_dim,
            dropout=dropout
        )
        
        self.bari_model = BariModel(
            input_dim=base_state_dim,
            dropout=dropout
        )
        
        # 经济参数
        self.register_buffer('delta', torch.tensor(Config.DELTA))
        self.register_buffer('phi', torch.tensor(Config.PHI))
        self.register_buffer('g', torch.tensor(Config.G))
    
    def extract_base_state(self, firm_state: torch.Tensor) -> torch.Tensor:
        """提取 base state（不含 i）"""
        return torch.cat([
            firm_state[:, :SIMMODEL.I],
            firm_state[:, SIMMODEL.X:]
        ], dim=-1)
    
    def forward(
        self, 
        firm_state: torch.Tensor,
        return_all: bool = True
    ) -> PolicyValueOutput:
        """
        完整前向传播
        
        Args:
            firm_state: (batch, 7) - (b, z, η, i, x, ĉf, ln Kf)
            return_all: 是否返回所有输出
        
        Returns:
            PolicyValueOutput: 包含所有输出的命名元组
        """
        # SharedModel 输出
        Q, bp0, bpI = self.shared_model(firm_state)
        
        # CombinedModel 输出
        P0, PI, bar_i = self.combined_model(firm_state)

        # 计算 Phat/P，同时基于积分结果得到 bar_z（P>0 则 bar_z=0，否则=1）
        Phat, P, bar_z = self.cal_phats(firm_state, P0, PI)
        bp = self.cal_bp(bp0, bpI, bar_i)
        
        return PolicyValueOutput(
            Q=Q,
            bp0=bp0,
            bpI=bpI,
            P0=P0,
            PI=PI,
            bar_i=bar_i,
            bar_z=bar_z,
            P=P,
            Phat=Phat,
            bp=bp
        )
    
    def cal_phats(
        self,
        firm_state: torch.Tensor,
        P0: torch.Tensor,
        PI: torch.Tensor,
        simulated_i: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 Phat、P、bar_z

        按 Cal_Phats 思路：保持其他状态不变，枚举 i 的若干取值，重新计算 (P0, PI)，对 max(P0, PI)
        在 i 维度上求平均作为 Phat；再用平滑映射得到 P 与 bar_z，避免硬阈值导致梯度死区。
        """
        device = firm_state.device
        batch_size = firm_state.size(0)

        if simulated_i is None:
            # 使用少量均匀点近似积分
            simulated_i = torch.linspace(0.0, Config.I_THRESHOLD, steps=5, device=device).unsqueeze(-1)

        P0_list = []
        PI_list = []
        for i_val in simulated_i:
            modified = firm_state.clone()
            modified[:, SIMMODEL.I] = i_val.expand(batch_size)
            P0_i, PI_i, _ = self.combined_model(modified)
            P0_list.append(P0_i)
            PI_list.append(PI_i)

        P0_stack = torch.stack(P0_list, dim=0)  # (n_i, batch, 1)
        PI_stack = torch.stack(PI_list, dim=0)  # (n_i, batch, 1)
        max_vals = torch.max(P0_stack, PI_stack)
        Phat = max_vals.mean(dim=0)  # (batch, 1)

        barz_temp = float(getattr(Config, "BARZ_LOGIT_TEMP", 10.0))
        # 论文口径：P = max(0, Phat)，需要允许出现明确的 P=0 违约区域。
        P = torch.clamp_min(Phat, 0.0)
        # 使用 Phat 生成平滑违约概率代理，保留负值区域梯度
        bar_z = torch.sigmoid(-barz_temp * Phat)

        return Phat, P, bar_z
    
    def cal_bp(
        self,
        bp0: torch.Tensor,
        bpI: torch.Tensor,
        bar_i: torch.Tensor
    ) -> torch.Tensor:
        """
        计算综合杠杆候选
        
        bp = bar_i * bpI + (1 - bar_i) * bp0
        """
        return bar_i * bpI + (1 - bar_i) * bp0
    
    def update_leverage(
        self,
        b_old: torch.Tensor,
        bp: torch.Tensor,
        eta: torch.Tensor
    ) -> torch.Tensor:
        """
        更新杠杆
        
        b_new = η * bp + (1 - η) * b_old
        
        Args:
            b_old: 旧杠杆
            bp: 杠杆候选
            eta: 再融资开关
        
        Returns:
            b_new: 新杠杆
        """
        return eta * bp + (1 - eta) * b_old
    
    def update_capital(
        self,
        K_old: torch.Tensor,
        bar_i: torch.Tensor
    ) -> torch.Tensor:
        """
        更新资本
        
        K_new = bar_i * G * K_old + (1 - bar_i) * K_old
        
        Args:
            K_old: 旧资本
            bar_i: 投资决策
        
        Returns:
            K_new: 新资本
        """
        return bar_i * self.g * K_old + (1 - bar_i) * K_old
    
    def get_shared_output(
        self, 
        firm_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """只获取 SharedModel 输出"""
        return self.shared_model(firm_state)
    
    def get_combined_output(
        self, 
        firm_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """只获取 CombinedModel 输出"""
        return self.combined_model(firm_state)
    
    def get_bar_z(self, firm_state: torch.Tensor) -> torch.Tensor:
        """只获取 bar_z"""
        base_state = self.extract_base_state(firm_state)
        return self.barz_model(base_state)
    
    def get_bar_i_value(self, firm_state: torch.Tensor) -> torch.Tensor:
        """获取 bar_i 的 value version"""
        base_state = self.extract_base_state(firm_state)
        return self.bari_model(base_state)
    
    def freeze(self):
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def unfreeze(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
        self.train()
    
    def get_module_parameters(self, module_name: str):
        """
        获取指定模块的参数
        
        Args:
            module_name: 'shared', 'combined', 'barz', 'bari'
        """
        modules = {
            'shared': self.shared_model,
            'combined': self.combined_model,
            'barz': self.barz_model,
            'bari': self.bari_model
        }
        
        if module_name not in modules:
            raise ValueError(f"Unknown module: {module_name}")
        
        return modules[module_name].parameters()


class CalPhats:
    """
    计算 Phat, P, bar_z 的工具类
    
    用于在训练和模拟中统一计算逻辑
    """
    
    def __init__(self, combined_model: CombinedModel, barz_model: BarzModel):
        self.combined_model = combined_model
        self.barz_model = barz_model
    
    def __call__(
        self, 
        firm_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 bar_z, Phat, P
        
        Args:
            firm_state: (batch, 7)
        
        Returns:
            bar_z: (batch, 1)
            Phat: (batch, 1)
            P: (batch, 1)
        """
        # Combined model 输出
        P0, PI, bar_i = self.combined_model(firm_state)
        
        # bar_z
        base_state = torch.cat([
            firm_state[:, :SIMMODEL.I],
            firm_state[:, SIMMODEL.X:]
        ], dim=-1)
        bar_z = self.barz_model(base_state)
        
        # Phat 和 P
        Phat = bar_i * PI + (1 - bar_i) * P0
        P = (1 - bar_z) * Phat
        
        return bar_z, Phat, P
