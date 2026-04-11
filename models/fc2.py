"""
FC2 模型

当前主线采用两套彻底分开的模型：

- FC2HatcModel: [b_quantiles, z_quantiles, x] -> ĉ
- FC2LnkModel:  [b_quantiles, z_quantiles, x, K_quantiles] -> ln K

保留 FC2Model 仅用于兼容旧脚本/旧 checkpoint。
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Union
from .base import MLP

import sys
sys.path.append('..')
from config import Config


class FC2ScalarModel(nn.Module):
    """单目标 FC2 标量模型。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = Config.FC2_HIDDEN_DIMS

        self.input_dim = input_dim
        self.trunk = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1],
            activation='gelu',
            dropout=dropout
        )
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        if phi.shape[-1] != self.input_dim:
            raise ValueError(f"FC2 scalar input dim mismatch: expected {self.input_dim}, got {phi.shape[-1]}")
        h = self.trunk(phi)
        return self.head(h)


class FC2HatcModel(FC2ScalarModel):
    """FC2 hatc-only 模型。"""

    def __init__(
        self,
        input_dim: int = None,
        hidden_dims: List[int] = None,
        quantile_num: int = 100,
        dropout: float = 0.1,
        use_x_baseline: bool = True,
        baseline_degree: int = 2,
    ):
        if input_dim is None:
            input_dim = 2 * quantile_num + 1
        super().__init__(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.use_x_baseline = bool(use_x_baseline)
        self.baseline_degree = int(baseline_degree)
        self.x_index = input_dim - 1
        self.register_buffer(
            "x_baseline_coef",
            torch.zeros((self.baseline_degree + 1,), dtype=torch.float32),
        )
        self.register_buffer(
            "x_baseline_mean",
            torch.zeros((1,), dtype=torch.float32),
        )
        self.register_buffer(
            "x_baseline_scale",
            torch.ones((1,), dtype=torch.float32),
        )
        self.register_buffer(
            "x_baseline_fitted",
            torch.tensor(False, dtype=torch.bool),
        )

    def _x_design(self, x: torch.Tensor) -> torch.Tensor:
        cols = [torch.ones_like(x)]
        for d in range(1, self.baseline_degree + 1):
            cols.append(x ** d)
        return torch.cat(cols, dim=1)

    def fit_x_baseline(self, phi: torch.Tensor, y: torch.Tensor) -> None:
        if not self.use_x_baseline:
            self.x_baseline_coef.zero_()
            self.x_baseline_mean.zero_()
            self.x_baseline_scale.fill_(1.0)
            self.x_baseline_fitted.fill_(False)
            return
        if phi.numel() == 0 or y.numel() == 0:
            self.x_baseline_coef.zero_()
            self.x_baseline_mean.zero_()
            self.x_baseline_scale.fill_(1.0)
            self.x_baseline_fitted.fill_(False)
            return

        x = phi[:, self.x_index:self.x_index + 1].detach()
        y_vec = y.reshape(-1, 1).detach()
        finite = torch.isfinite(x).reshape(-1) & torch.isfinite(y_vec).reshape(-1)
        if int(finite.sum().item()) < self.baseline_degree + 1:
            self.x_baseline_coef.zero_()
            self.x_baseline_mean.zero_()
            self.x_baseline_scale.fill_(1.0)
            self.x_baseline_fitted.fill_(False)
            return

        x_cpu = x[finite].to(dtype=torch.float64, device="cpu")
        y_cpu = y_vec[finite].to(dtype=torch.float64, device="cpu")
        x_mean = x_cpu.mean(dim=0, keepdim=True)
        x_scale = x_cpu.std(dim=0, keepdim=True).clamp(min=1e-6)
        x_std = (x_cpu - x_mean) / x_scale
        design = self._x_design(x_std)
        coef = torch.linalg.lstsq(design, y_cpu).solution.reshape(-1)
        self.x_baseline_coef.copy_(coef.to(device=self.x_baseline_coef.device, dtype=self.x_baseline_coef.dtype))
        self.x_baseline_mean.copy_(x_mean.reshape_as(self.x_baseline_mean).to(device=self.x_baseline_mean.device, dtype=self.x_baseline_mean.dtype))
        self.x_baseline_scale.copy_(x_scale.reshape_as(self.x_baseline_scale).to(device=self.x_baseline_scale.device, dtype=self.x_baseline_scale.dtype))
        self.x_baseline_fitted.fill_(True)

    def x_baseline(self, phi: torch.Tensor) -> torch.Tensor:
        if (not self.use_x_baseline) or (not bool(self.x_baseline_fitted.item())):
            return torch.zeros((phi.shape[0], 1), dtype=phi.dtype, device=phi.device)
        x = phi[:, self.x_index:self.x_index + 1].to(dtype=phi.dtype)
        x_mean = self.x_baseline_mean.to(device=phi.device, dtype=phi.dtype)
        x_scale = self.x_baseline_scale.to(device=phi.device, dtype=phi.dtype).clamp(min=1e-6)
        x_std = (x - x_mean) / x_scale
        design = self._x_design(x_std)
        coef = self.x_baseline_coef.to(device=phi.device, dtype=phi.dtype).unsqueeze(1)
        return design @ coef

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        residual = super().forward(phi)
        return residual + self.x_baseline(phi)

    def load_state_dict(self, state_dict, strict: bool = True):
        state = dict(state_dict)
        state.setdefault("x_baseline_coef", self.x_baseline_coef.detach().clone())
        state.setdefault("x_baseline_mean", self.x_baseline_mean.detach().clone())
        state.setdefault("x_baseline_scale", self.x_baseline_scale.detach().clone())
        state.setdefault("x_baseline_fitted", self.x_baseline_fitted.detach().clone())
        return super().load_state_dict(state, strict=strict)


class FC2LnkModel(FC2ScalarModel):
    """FC2 lnk-only 模型，额外吃一份 K quantile summary。"""

    def __init__(
        self,
        input_dim: int = None,
        hidden_dims: List[int] = None,
        quantile_num: int = 100,
        dropout: float = 0.1,
    ):
        if input_dim is None:
            input_dim = 3 * quantile_num + 1
        super().__init__(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)


class FC2Model(nn.Module):
    """
    兼容旧接口的包装器：
    - 输入 dict {'hatc': ..., 'lnk': ...}
    - 输出 dict {'hatc': ..., 'lnk': ...}
    """

    def __init__(
        self,
        input_dim: int = None,
        lnk_input_dim: int = None,
        hidden_dims: List[int] = None,
        output_dim: int = 2,
        quantile_num: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()
        if input_dim is None:
            input_dim = 2 * quantile_num + 1
        if lnk_input_dim is None:
            lnk_input_dim = input_dim + quantile_num
        self.quantile_num = quantile_num
        self.input_dim = input_dim
        self.lnk_input_dim = lnk_input_dim
        self.output_dim = output_dim
        self.hatc_model = FC2HatcModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            quantile_num=quantile_num,
            dropout=dropout,
        )
        self.lnk_model = FC2LnkModel(
            input_dim=lnk_input_dim,
            hidden_dims=hidden_dims,
            quantile_num=quantile_num,
            dropout=dropout,
        )

    def _normalize_inputs(
        self,
        phi: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(phi, dict):
            hatc_in = phi['hatc']
            lnk_in = phi.get('lnk', hatc_in)
        else:
            hatc_in = phi
            lnk_in = phi
            if phi.shape[-1] == self.input_dim and self.lnk_input_dim > self.input_dim:
                pad = torch.zeros(
                    (*phi.shape[:-1], self.lnk_input_dim - self.input_dim),
                    device=phi.device,
                    dtype=phi.dtype,
                )
                lnk_in = torch.cat([phi, pad], dim=-1)
            elif phi.shape[-1] == self.lnk_input_dim and self.input_dim < self.lnk_input_dim:
                hatc_in = phi[..., :self.input_dim]
        return hatc_in, lnk_in

    def forward(
        self,
        phi: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        hatc_in, lnk_in = self._normalize_inputs(phi)
        return {
            'hatc': self.hatc_model(hatc_in),
            'lnk': self.lnk_model(lnk_in),
        }


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
