"""
基础模型组件
包含 MLP 基类和带标准化功能的 MLP
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import numpy as np


class MLP(nn.Module):
    """
    基础多层感知机
    支持灵活的隐藏层配置和激活函数选择
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'relu',
        output_activation: Optional[str] = None,
        dropout: float = 0.0,
        use_batch_norm: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self._get_activation(activation))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if output_activation:
            layers.append(self._get_activation(output_activation))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'softplus': nn.Softplus(),
            'gelu': nn.GELU(),
        }
        return activations.get(activation, nn.ReLU())
    
    def _init_weights(self):
        """Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MLPWithScaler(MLP):
    """
    带标准化功能的 MLP
    用于 FC1 等需要标准化输出的模型
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        **kwargs
    ):
        super().__init__(input_dim, hidden_dims, output_dim, **kwargs)
        
        # 注册标准化参数为 buffer（不参与梯度更新但会被保存）
        self.register_buffer('y_mean_', torch.zeros(output_dim))
        self.register_buffer('y_std_', torch.ones(output_dim))
        self.register_buffer('x_mean_', torch.zeros(input_dim))
        self.register_buffer('x_std_', torch.ones(input_dim))
        self.register_buffer('fitted_', torch.tensor(False))
    
    def fit_scaler(
        self, 
        x: torch.Tensor, 
        y: Optional[torch.Tensor] = None,
        fit_x: bool = True,
        fit_y: bool = True
    ):
        """
        拟合标准化参数
        
        Args:
            x: 输入数据
            y: 输出数据（用于目标标准化）
            fit_x: 是否拟合输入标准化
            fit_y: 是否拟合输出标准化
        """
        if fit_x:
            self.x_mean_ = x.mean(dim=0)
            self.x_std_ = x.std(dim=0).clamp(min=1e-6)
        
        if fit_y and y is not None:
            self.y_mean_ = y.mean(dim=0) if y.dim() > 1 else y.mean().unsqueeze(0)
            self.y_std_ = y.std(dim=0).clamp(min=1e-6) if y.dim() > 1 else y.std().clamp(min=1e-6).unsqueeze(0)
        
        self.fitted_ = torch.tensor(True)
    
    def transform_x(self, x: torch.Tensor) -> torch.Tensor:
        """标准化输入"""
        # 确保输入与 scaler buffer 保持同一 dtype，避免 float/double 混用
        x = x.to(self.x_mean_.dtype)
        return (x - self.x_mean_) / self.x_std_
    
    def transform_y(self, y: torch.Tensor) -> torch.Tensor:
        """标准化输出目标"""
        return (y - self.y_mean_) / self.y_std_
    
    def inverse_transform(self, y_norm: torch.Tensor) -> torch.Tensor:
        """反标准化输出"""
        return y_norm * self.y_std_ + self.y_mean_
    
    def forward_normalized(self, x: torch.Tensor) -> torch.Tensor:
        """
        标准化前向传播
        输入会被标准化，输出保持在标准化空间
        """
        x_norm = self.transform_x(x)
        return super().forward(x_norm)
    
    def forward_physical(self, x: torch.Tensor) -> torch.Tensor:
        """
        物理尺度前向传播
        输入被标准化，输出被反标准化回物理尺度
        """
        y_norm = self.forward_normalized(x)
        return self.inverse_transform(y_norm)


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, dim: int, activation: str = 'relu', dropout: float = 0.0):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            self._get_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> nn.Module:
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class AttentionMLP(nn.Module):
    """
    带自注意力机制的 MLP
    用于处理变长序列或需要关注不同特征的场景
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        # 自注意力层
        self.attention = nn.MultiheadAttention(
            hidden_dims[0], 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # MLP 部分
        layers = []
        prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 投影
        h = self.input_proj(x)
        
        # 添加序列维度用于注意力
        h = h.unsqueeze(1)
        h_attn, _ = self.attention(h, h, h)
        h = h + h_attn
        h = h.squeeze(1)
        
        # MLP
        return self.mlp(h)
