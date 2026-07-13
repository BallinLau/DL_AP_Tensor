"""
ShareLayer 模型
Policy & Value 的统一架构：共享特征层 + 多头输出

关键结构约束：
- Base state (不含i): (b, z, η, x, ĉ, ln K) ∈ ℝ^6
- Investment cost: i ∈ ℝ
- No-invest heads (不吃i): (Q, bp⁰, P⁰)
- Invest heads (吃i): (bpᴵ, Pᴵ)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional
from .base import MLP

import sys
sys.path.append('..')
from config import Config, SIMMODEL


class ShareLayer(nn.Module):
    """
    共享特征层
    
    输入：base_state = (b, z, η, x, ĉ, ln K)
    输出：共享表征 h
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: List[int] = None,
        output_dim: int = 64,
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = Config.SHARE_LAYER_HIDDEN_DIMS
        
        self.network = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout=dropout
        )
        
        self.output_dim = output_dim
    
    def forward(self, base_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            base_state: (batch, 6) - (b, z, η, x, ĉ, ln K)
        Returns:
            h: (batch, output_dim) - 共享表征
        """
        return self.network(base_state)


class QHead(nn.Module):
    """
    债券价值 Q 的输出头
    不依赖投资成本 i
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dims: List[int] = None,
        output_activation: str = 'softplus'
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = Config.Q_HEAD_DIMS
        
        self.network = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            output_activation=output_activation
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, input_dim) - 共享表征
        Returns:
            Q: (batch, 1) - 债券价值
        """
        return self.network(h)


class BpHead(nn.Module):
    """
    杠杆候选 bp 的输出头
    bp⁰ 不依赖 i，bpᴵ 依赖 i
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dims: List[int] = None,
        requires_i: bool = False,
        output_activation: str = 'sigmoid'  # 约束 bp ∈ [0, 1]
    ):
        super().__init__()
        
        self.requires_i = requires_i
        actual_input_dim = input_dim + 1 if requires_i else input_dim
        
        if hidden_dims is None:
            hidden_dims = Config.BP0_HEAD_DIMS if not requires_i else Config.BPI_HEAD_DIMS
        
        self.network = MLP(
            input_dim=actual_input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            output_activation=output_activation
        )

    def _prepare_input(self, h: torch.Tensor, i: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.requires_i:
            if i is None:
                raise ValueError("Investment cost i is required for bpI head")
            h = torch.cat([h, i], dim=-1)
        return h

    def forward_logits(self, h: torch.Tensor, i: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the pre-sigmoid bp logits for diagnostics."""
        h = self._prepare_input(h, i)
        layers = list(self.network.network.children())
        if not layers or not isinstance(layers[-1], nn.Sigmoid):
            raise RuntimeError("BpHead.forward_logits() requires a sigmoid output activation.")
        for layer in layers[:-1]:
            h = layer(h)
        return h
    
    def forward(self, h: torch.Tensor, i: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            h: (batch, input_dim) - 共享表征
            i: (batch, 1) - 投资成本（仅当 requires_i=True 时使用）
        Returns:
            bp: (batch, 1) - 杠杆候选
        """
        return torch.sigmoid(self.forward_logits(h, i))


class PHead(nn.Module):
    """
    股票价值 P 的输出头
    P⁰ 不依赖 i，Pᴵ 依赖 i
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dims: List[int] = None,
        requires_i: bool = False,
        output_activation: Optional[str] = None
    ):
        super().__init__()
        
        self.requires_i = requires_i
        actual_input_dim = input_dim + 1 if requires_i else input_dim
        
        if hidden_dims is None:
            hidden_dims = Config.P0_HEAD_DIMS if not requires_i else Config.PI_HEAD_DIMS
        
        self.network = MLP(
            input_dim=actual_input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            output_activation=output_activation
        )
    
    def forward(self, h: torch.Tensor, i: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            h: (batch, input_dim) - 共享表征
            i: (batch, 1) - 投资成本（仅当 requires_i=True 时使用）
        Returns:
            P: (batch, 1) - 股票价值
        """
        if self.requires_i:
            if i is None:
                raise ValueError("Investment cost i is required for PI head")
            h = torch.cat([h, i], dim=-1)
        
        return self.network(h)


class SharedModel(nn.Module):
    """
    Shared Model: 输出 Q, bp⁰, bpᴵ
    
    结构：
    - ShareLayer: base_state → h
    - Q head: h → Q（不依赖 i）
    - bp⁰ head: h → bp⁰（不依赖 i）
    - bpᴵ head: [h, i] → bpᴵ（依赖 i）
    """
    
    def __init__(
        self,
        base_state_dim: int = 6,
        share_hidden_dims: List[int] = None,
        share_output_dim: int = 64,
        q_hidden_dims: List[int] = None,
        bp_hidden_dims: List[int] = None,
        dropout: float = 0.0,
        share_layer: Optional[ShareLayer] = None
    ):
        super().__init__()
        
        # 共享层
        if share_layer is None:
            self.share_layer = ShareLayer(
                input_dim=base_state_dim,
                hidden_dims=share_hidden_dims,
                output_dim=share_output_dim,
                dropout=dropout
            )
            actual_share_output_dim = share_output_dim
        else:
            self.share_layer = share_layer
            actual_share_output_dim = share_layer.output_dim
        
        # Q head
        self.q_head = QHead(
            input_dim=actual_share_output_dim,
            hidden_dims=q_hidden_dims
        )
        
        # bp⁰ head（不依赖 i）
        self.bp0_head = BpHead(
            input_dim=actual_share_output_dim,
            hidden_dims=bp_hidden_dims,
            requires_i=False
        )
        
        # bpᴵ head（依赖 i）
        self.bpI_head = BpHead(
            input_dim=actual_share_output_dim,
            hidden_dims=bp_hidden_dims,
            requires_i=True
        )
    
    def forward(
        self, 
        firm_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            firm_state: (batch, 7) - 完整 firm state
                (b, z, η, i, x, ĉf, ln Kf)
        Returns:
            Q: (batch, 1) - 债券价值
            bp0: (batch, 1) - 不投资时的杠杆候选
            bpI: (batch, 1) - 投资时的杠杆候选
        """
        # 提取 base state（不含 i）
        base_state = torch.cat([
            firm_state[:, :SIMMODEL.I],  # b, z, η
            firm_state[:, SIMMODEL.X:]   # x, ĉf, ln Kf
        ], dim=-1)
        
        # 提取投资成本 i
        i = firm_state[:, SIMMODEL.I:SIMMODEL.I+1]
        
        # 共享表征
        h = self.share_layer(base_state)
        
        # 各 head 输出
        # Q 采用“总债价值”口径：Q = b * q_unit，结构上保证 b=0 时 Q=0。
        q_unit = self.q_head(h)
        b_nonneg = torch.clamp(firm_state[:, SIMMODEL.B:SIMMODEL.B+1], min=0.0)
        Q = b_nonneg * q_unit
        bp0 = self.bp0_head(h)
        bpI = self.bpI_head(h, i)
        
        return Q, bp0, bpI
    
    def get_Q(self, firm_state: torch.Tensor) -> torch.Tensor:
        """只获取 Q"""
        base_state = torch.cat([
            firm_state[:, :SIMMODEL.I],
            firm_state[:, SIMMODEL.X:]
        ], dim=-1)
        h = self.share_layer(base_state)
        q_unit = self.q_head(h)
        b_nonneg = torch.clamp(firm_state[:, SIMMODEL.B:SIMMODEL.B+1], min=0.0)
        return b_nonneg * q_unit


class CombinedModel(nn.Module):
    """
    Combined Model: 输出 P⁰, Pᴵ, bar_i
    
    结构：
    - ShareLayer: base_state → h
    - P⁰ head: h → P⁰（不依赖 i）
    - Pᴵ head: [h, i] → Pᴵ（依赖 i）
    - bar_i: 由 P⁰, Pᴵ 计算得到
    """
    
    def __init__(
        self,
        base_state_dim: int = 6,
        share_hidden_dims: List[int] = None,
        share_output_dim: int = 64,
        p_hidden_dims: List[int] = None,
        dropout: float = 0.0,
        share_layer: Optional[ShareLayer] = None
    ):
        super().__init__()
        
        # 共享层
        if share_layer is None:
            self.share_layer = ShareLayer(
                input_dim=base_state_dim,
                hidden_dims=share_hidden_dims,
                output_dim=share_output_dim,
                dropout=dropout
            )
            actual_share_output_dim = share_output_dim
        else:
            self.share_layer = share_layer
            actual_share_output_dim = share_layer.output_dim
        
        # P⁰ head（不依赖 i）
        self.p0_head = PHead(
            input_dim=actual_share_output_dim,
            hidden_dims=p_hidden_dims,
            requires_i=False
        )
        
        # Pᴵ head（依赖 i）
        self.pI_head = PHead(
            input_dim=actual_share_output_dim,
            hidden_dims=p_hidden_dims,
            requires_i=True
        )
    
    def forward(
        self, 
        firm_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            firm_state: (batch, 7) - 完整 firm state
        Returns:
            P0: (batch, 1) - 不投资时的股票价值
            PI: (batch, 1) - 投资时的股票价值
            bar_i: (batch, 1) - 投资门槛（PI > P0 时为 1）
        """
        # 提取 base state
        base_state = torch.cat([
            firm_state[:, :SIMMODEL.I],
            firm_state[:, SIMMODEL.X:]
        ], dim=-1)
        
        # 提取投资成本 i
        i = firm_state[:, SIMMODEL.I:SIMMODEL.I+1]
        
        # 共享表征
        h = self.share_layer(base_state)
        
        # 各 head 输出
        P0 = self.p0_head(h)
        PI = self.pI_head(h, i)
        
        # 计算投资门槛（软化版本，用于可导训练）
        bar_i = torch.sigmoid(10 * (PI - P0))  # 平滑的指示函数
        
        return P0, PI, bar_i
    
    def get_hard_bar_i(self, firm_state: torch.Tensor) -> torch.Tensor:
        """获取硬投资门槛（用于模拟）"""
        P0, PI, _ = self.forward(firm_state)
        return (PI > P0).float()


class BarzModel(nn.Module):
    """
    Bar_z 模型：计算破产/退出阈值
    
    输入：base_state（不含 i）
    输出：bar_z ∈ [0, 1]
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: List[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]
        
        self.network = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            output_activation='sigmoid',
            dropout=dropout
        )
    
    def forward(self, base_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            base_state: (batch, 6) - (b, z, η, x, ĉ, ln K)
        Returns:
            bar_z: (batch, 1) - 破产/退出概率
        """
        return self.network(base_state)


class BariModel(nn.Module):
    """
    Bar_i 模型：价值版本的投资门槛
    用于诊断和一致性检查
    
    输入：base_state（不含 i）
    输出：bar_i_value
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: List[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]
        
        self.network = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            output_activation='sigmoid',
            dropout=dropout
        )
    
    def forward(self, base_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            base_state: (batch, 6)
        Returns:
            bar_i_value: (batch, 1)
        """
        return self.network(base_state)
