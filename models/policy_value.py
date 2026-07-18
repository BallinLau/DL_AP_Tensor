"""
Policy & Value 统一接口

整合 SharedModel, CombinedModel, BarzModel, BariModel
提供统一的 forward 和工具函数
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, NamedTuple
from .firm_derived import FirmDerivedObjects
from .share_layer import ShareLayer, QHead, BpHead, PHead, CombinedModel, BarzModel, BariModel

import sys
sys.path.append('..')
from config import Config, SIMMODEL


def default_policy_value_model_spec() -> Dict[str, object]:
    tau_i = float(getattr(Config, "PV_TAU_I", getattr(Config, "BARI_TAU", 0.1)))
    tau_z = float(getattr(Config, "PV_TAU_Z", 1.0 / max(float(getattr(Config, "BARZ_LOGIT_TEMP", 10.0)), 1e-8)))
    return {
        "base_state_dim": int(Config.base_state_dim()),
        "share_hidden_dims": list(getattr(Config, "SHARE_LAYER_HIDDEN_DIMS", []) or []),
        "share_output_dim": 64,
        "dropout": 0.0,
        "tau_i": tau_i,
        "tau_z": tau_z,
        "i_grid_size": int(getattr(Config, "PV_I_GRID_SIZE", 11)),
        "i_threshold": float(getattr(Config, "I_THRESHOLD", 0.5)),
        "delta": float(Config.DELTA),
        "phi": float(Config.PHI),
        "g": float(Config.G),
    }


def build_policy_value_from_checkpoint_spec(
    payload: Dict[str, object],
    *,
    value_scale_mode: Optional[str] = None,
    value_scale_log_max: Optional[float] = None,
    allow_current_model_spec: bool = False,
) -> "PolicyValueModel":
    spec = payload.get("policy_value_model_spec") if isinstance(payload, dict) else None
    if spec is None:
        if not allow_current_model_spec:
            raise ValueError("checkpoint is missing policy_value_model_spec")
        spec = default_policy_value_model_spec()
    if not isinstance(spec, dict):
        raise ValueError("policy_value_model_spec must be a dictionary")
    return PolicyValueModel(
        base_state_dim=int(spec["base_state_dim"]),
        share_hidden_dims=list(spec.get("share_hidden_dims") or []),
        share_output_dim=int(spec["share_output_dim"]),
        dropout=float(spec.get("dropout", 0.0)),
        tau_i=float(spec["tau_i"]),
        tau_z=float(spec["tau_z"]),
        i_grid_size=int(spec["i_grid_size"]),
        i_threshold=float(spec["i_threshold"]),
        delta=float(spec["delta"]),
        phi=float(spec["phi"]),
        g=float(spec["g"]),
        value_scale_mode=value_scale_mode,
        value_scale_log_max=value_scale_log_max,
    )


class PolicyValueOutput(NamedTuple):
    """Policy & Value 模型的输出"""
    Q: torch.Tensor          # 债券价值
    bp0: torch.Tensor        # 不投资时杠杆候选
    bpI: torch.Tensor        # 投资时杠杆候选
    P0: torch.Tensor         # 兼容别名：V0
    PI: torch.Tensor         # 兼容别名：VI
    bar_i: torch.Tensor      # 兼容别名：bar_i_eff
    bar_z: torch.Tensor      # 破产门槛
    P: torch.Tensor          # 综合股票价值
    Phat: torch.Tensor       # P hat (中间值)
    bp: torch.Tensor         # 综合杠杆候选
    V0: torch.Tensor         # 不投资 branch value
    VI: torch.Tensor         # 投资 branch value
    bar_i_cond: torch.Tensor # 存活条件下投资概率
    bar_i_eff: torch.Tensor  # 考虑当前违约后的有效投资概率
    survival_prob: torch.Tensor
    bp_cond: torch.Tensor    # 存活条件下混合杠杆候选


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
        dropout: float = 0.0,
        value_scale_mode: Optional[str] = None,
        value_scale_log_max: Optional[float] = None,
        tau_i: Optional[float] = None,
        tau_z: Optional[float] = None,
        i_grid_size: Optional[int] = None,
        i_threshold: Optional[float] = None,
        delta: Optional[float] = None,
        phi: Optional[float] = None,
        g: Optional[float] = None,
    ):
        super().__init__()
        self.base_state_dim = int(base_state_dim)
        resolved_share_hidden_dims = list(
            share_hidden_dims if share_hidden_dims is not None else getattr(Config, "SHARE_LAYER_HIDDEN_DIMS", [])
        )
        self.share_hidden_dims = resolved_share_hidden_dims
        self.share_output_dim = int(share_output_dim)
        self.dropout = float(dropout)
        self.i_grid_size = int(i_grid_size if i_grid_size is not None else getattr(Config, "PV_I_GRID_SIZE", 11))
        self.i_threshold = float(i_threshold if i_threshold is not None else getattr(Config, "I_THRESHOLD", 0.5))
        tau_i = float(tau_i if tau_i is not None else getattr(Config, "PV_TAU_I", getattr(Config, "BARI_TAU", 0.1)))
        tau_z = float(tau_z if tau_z is not None else getattr(Config, "PV_TAU_Z", 1.0 / max(float(getattr(Config, "BARZ_LOGIT_TEMP", 10.0)), 1e-8)))
        self.tau_i = tau_i
        self.tau_z = tau_z
        self.value_scale_mode = str(value_scale_mode if value_scale_mode is not None else getattr(Config, "PV_VALUE_SCALE_MODE", "none")).lower()
        self.value_scale_log_max = float(
            value_scale_log_max if value_scale_log_max is not None else getattr(Config, "PV_VALUE_SCALE_LOG_MAX", 20.0)
        )
        self._validate_value_scale_mode()
        
        self.q_encoder = ShareLayer(
            input_dim=base_state_dim,
            hidden_dims=resolved_share_hidden_dims,
            output_dim=share_output_dim,
            dropout=dropout
        )
        self.value_encoder = ShareLayer(
            input_dim=base_state_dim,
            hidden_dims=resolved_share_hidden_dims,
            output_dim=share_output_dim,
            dropout=dropout
        )
        self.policy_encoder = ShareLayer(
            input_dim=base_state_dim,
            hidden_dims=resolved_share_hidden_dims,
            output_dim=share_output_dim,
            dropout=dropout
        )

        self.q_head = QHead(input_dim=share_output_dim)
        self.v0_head = PHead(input_dim=share_output_dim, requires_i=False)
        self.vi_head = PHead(input_dim=share_output_dim, requires_i=True)
        self.bp0_head = BpHead(input_dim=share_output_dim, requires_i=False)
        self.bpi_head = BpHead(input_dim=share_output_dim, requires_i=True)
        
        # 辅助模型
        self.barz_model = BarzModel(
            input_dim=base_state_dim,
            dropout=dropout
        )
        
        self.bari_model = BariModel(
            input_dim=base_state_dim,
            dropout=dropout
        )

        self.derived = FirmDerivedObjects(tau_i=tau_i, tau_z=tau_z)
        
        # 经济参数
        self.register_buffer('delta', torch.tensor(float(delta if delta is not None else Config.DELTA)))
        self.register_buffer('phi', torch.tensor(float(phi if phi is not None else Config.PHI)))
        self.register_buffer('g', torch.tensor(float(g if g is not None else Config.G)))

    def model_spec(self) -> Dict[str, object]:
        return {
            "base_state_dim": int(self.base_state_dim),
            "share_hidden_dims": list(self.share_hidden_dims),
            "share_output_dim": int(self.share_output_dim),
            "dropout": float(self.dropout),
            "tau_i": float(self.tau_i),
            "tau_z": float(self.tau_z),
            "i_grid_size": int(self.i_grid_size),
            "i_threshold": float(self.i_threshold),
            "delta": float(self.delta.detach().cpu().item()),
            "phi": float(self.phi.detach().cpu().item()),
            "g": float(self.g.detach().cpu().item()),
        }

    def _validate_value_scale_mode(self) -> None:
        if self.value_scale_mode not in {"none", "exp_xz"}:
            raise ValueError(f"Unsupported value_scale_mode: {self.value_scale_mode}")

    def configure_value_parameterization(
        self,
        *,
        mode: str = "none",
        log_max: float = 20.0,
    ) -> None:
        self.value_scale_mode = str(mode).lower()
        self.value_scale_log_max = float(log_max)
        self._validate_value_scale_mode()

    def value_parameterization_metadata(self) -> Dict[str, object]:
        return {
            "mode": self.value_scale_mode,
            "scale_formula": f"1+exp(clamp(x+z,max={self.value_scale_log_max:g}))" if self.value_scale_mode == "exp_xz" else "1",
            "bellman_normalization": bool(getattr(Config, "PV_BELLMAN_NORMALIZE_BY_VALUE_SCALE", False)),
            "log_max": float(self.value_scale_log_max),
        }

    def equity_value_scale(self, firm_state: torch.Tensor) -> torch.Tensor:
        if self.value_scale_mode == "none":
            return torch.ones((firm_state.shape[0], 1), dtype=firm_state.dtype, device=firm_state.device)
        log_component_raw = firm_state[:, SIMMODEL.X:SIMMODEL.X + 1] + firm_state[:, SIMMODEL.Z:SIMMODEL.Z + 1]
        log_component = torch.clamp(log_component_raw, max=float(self.value_scale_log_max))
        return 1.0 + torch.exp(log_component)

    def equity_value_scale_diagnostics(self, firm_state: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            if self.value_scale_mode == "none":
                return {"value_scale_clamp_ratio": 0.0}
            raw = firm_state[:, SIMMODEL.X:SIMMODEL.X + 1] + firm_state[:, SIMMODEL.Z:SIMMODEL.Z + 1]
            return {
                "value_scale_clamp_ratio": float((raw > float(self.value_scale_log_max)).to(torch.float32).mean().item())
            }

    def _split_state(self, firm_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_state = self.extract_base_state(firm_state)
        i = firm_state[:, SIMMODEL.I:SIMMODEL.I + 1]
        b = firm_state[:, SIMMODEL.B:SIMMODEL.B + 1]
        return base_state, i, b

    def _raw_value_outputs(self, firm_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base_state, i, _ = self._split_state(firm_state)
        h_v = self.value_encoder(base_state)
        return self.v0_head(h_v), self.vi_head(h_v, i)

    def forward_value_components(self, firm_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        V0_raw, VI_raw = self._raw_value_outputs(firm_state)
        scale = self.equity_value_scale(firm_state)
        if self.value_scale_mode == "exp_xz":
            V0_normalized = V0_raw
            VI_normalized = VI_raw
            V0_physical = scale * V0_normalized
            VI_physical = scale * VI_normalized
        else:
            V0_physical = V0_raw
            VI_physical = VI_raw
            V0_normalized = V0_raw
            VI_normalized = VI_raw
        return {
            "V0_physical": V0_physical,
            "VI_physical": VI_physical,
            "V0_normalized": V0_normalized,
            "VI_normalized": VI_normalized,
            "value_scale": scale,
        }

    def _value_outputs(self, firm_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        components = self.forward_value_components(firm_state)
        return components["V0_physical"], components["VI_physical"]

    def _q_output(self, firm_state: torch.Tensor) -> torch.Tensor:
        base_state, _, b = self._split_state(firm_state)
        h_q = self.q_encoder(base_state)
        q_unit = self.q_head(h_q)
        return torch.clamp(b, min=0.0) * q_unit

    def _policy_logits(self, firm_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base_state, i, _ = self._split_state(firm_state)
        h_pi = self.policy_encoder(base_state)
        bp0_logit = self.bp0_head.forward_logits(h_pi)
        bpI_logit = self.bpi_head.forward_logits(h_pi, i)
        return bp0_logit, bpI_logit

    def _policy_outputs(self, firm_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bp0_logit, bpI_logit = self._policy_logits(firm_state)
        return torch.sigmoid(bp0_logit), torch.sigmoid(bpI_logit)
    
    def extract_base_state(self, firm_state: torch.Tensor) -> torch.Tensor:
        """提取 base state（不含 i）"""
        return torch.cat([
            firm_state[:, :SIMMODEL.I],
            firm_state[:, SIMMODEL.X:]
        ], dim=-1)

    def forward_policy(self, firm_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lightweight policy-only forward for simulation or diagnostics.

        This returns bp0/bpI without computing Q, P0/PI, Phat/P/bar_z, or the
        internal i-grid used by cal_phats().
        """
        return self._policy_outputs(firm_state)

    def forward_policy_logits(self, firm_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return pre-sigmoid bp0/bpI logits for policy diagnostics/training."""
        return self._policy_logits(firm_state)

    def forward_equity(self, firm_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Equity-only target evaluation used by the bp-grid teacher.

        It avoids Q and policy-head work, but still computes the internal
        investment-cost grid required for Phat/P/bar_z.
        """
        Phat, P, bar_z, survival_prob = self.cal_phats(firm_state)
        return {
            "Phat": Phat,
            "P": P,
            "bar_z": bar_z,
            "survival_prob": survival_prob,
        }

    def forward_simulation(self, firm_state: torch.Tensor) -> PolicyValueOutput:
        """
        Rollout-facing forward path.

        Current simulation exports Q/P0/PI/P, default and policy fields on
        every firm row, and uses bar_i/bar_z/bp for state transitions and
        resource accounting.  Therefore this path deliberately returns the
        full PolicyValueOutput while making the simulation/training-teacher
        boundary explicit.
        """
        return self.forward(firm_state)
    
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
        Q = self._q_output(firm_state)
        bp0, bpI = self._policy_outputs(firm_state)
        V0, VI = self._value_outputs(firm_state)

        Phat, P, bar_z, survival_prob = self.cal_phats(firm_state)
        bar_i_cond = self.derived.investment_conditional(V0, VI)
        bar_i_eff = survival_prob * bar_i_cond
        bp_cond = self.cal_bp(bp0, bpI, bar_i_cond)
        bp = survival_prob * bp_cond + (1.0 - survival_prob) * bp0
        
        return PolicyValueOutput(
            Q=Q,
            bp0=bp0,
            bpI=bpI,
            P0=V0,
            PI=VI,
            bar_i=bar_i_eff,
            bar_z=bar_z,
            P=P,
            Phat=Phat,
            bp=bp,
            V0=V0,
            VI=VI,
            bar_i_cond=bar_i_cond,
            bar_i_eff=bar_i_eff,
            survival_prob=survival_prob,
            bp_cond=bp_cond,
        )
    
    def cal_phats(
        self,
        firm_state: torch.Tensor,
        simulated_i: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 Phat、P、bar_z

        按 Cal_Phats 思路：保持其他状态不变，枚举 i 的若干取值，重新计算 (P0, PI)，对 max(P0, PI)
        在 i 维度上求平均作为 Phat；再用平滑映射得到 P 与 bar_z，避免硬阈值导致梯度死区。
        """
        device = firm_state.device
        batch_size = firm_state.size(0)

        if simulated_i is None:
            simulated_i = torch.linspace(
                0.0,
                self.i_threshold,
                steps=max(int(self.i_grid_size), 2),
                device=device
            ).unsqueeze(-1)

        V0_list = []
        VI_list = []
        for i_val in simulated_i:
            modified = firm_state.clone()
            modified[:, SIMMODEL.I] = i_val.expand(batch_size)
            V0_i, VI_i = self._value_outputs(modified)
            V0_list.append(V0_i)
            VI_list.append(VI_i)

        V0_stack = torch.stack(V0_list, dim=0)  # (n_i, batch, 1)
        VI_stack = torch.stack(VI_list, dim=0)  # (n_i, batch, 1)
        max_vals = torch.max(V0_stack, VI_stack)
        Phat = max_vals.mean(dim=0)  # (batch, 1)

        P = self.derived.total_equity(Phat, hard=True)
        survival_prob = self.derived.survival(Phat)
        bar_z = 1.0 - survival_prob

        return Phat, P, bar_z, survival_prob
    
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
        Q = self._q_output(firm_state)
        bp0, bpI = self._policy_outputs(firm_state)
        return Q, bp0, bpI
    
    def get_combined_output(
        self, 
        firm_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """只获取 CombinedModel 输出"""
        V0, VI = self._value_outputs(firm_state)
        bar_i_cond = self.derived.investment_conditional(V0, VI)
        return V0, VI, bar_i_cond
    
    def get_bar_z(self, firm_state: torch.Tensor) -> torch.Tensor:
        """只获取 bar_z"""
        return self.cal_phats(firm_state)[2]
    
    def get_bar_i_value(self, firm_state: torch.Tensor) -> torch.Tensor:
        """获取 bar_i 的 value version"""
        V0, VI = self._value_outputs(firm_state)
        return self.derived.investment_conditional(V0, VI)
    
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
            module_name: 'q', 'value', 'policy', legacy names also accepted
        """
        modules = {
            'q': nn.ModuleList([self.q_encoder, self.q_head]),
            'value': nn.ModuleList([self.value_encoder, self.v0_head, self.vi_head]),
            'policy': nn.ModuleList([self.policy_encoder, self.bp0_head, self.bpi_head]),
            'shared': nn.ModuleList([self.q_encoder, self.q_head, self.policy_encoder, self.bp0_head, self.bpi_head]),
            'combined': nn.ModuleList([self.value_encoder, self.v0_head, self.vi_head]),
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
