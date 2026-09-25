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
from utils.firm_transition import apply_refinancing_policy


def default_policy_value_model_spec() -> Dict[str, object]:
    tau_i = float(getattr(Config, "PV_TAU_I", getattr(Config, "BARI_TAU", 0.1)))
    tau_z = float(getattr(Config, "PV_TAU_Z", 1.0 / max(float(getattr(Config, "BARZ_LOGIT_TEMP", 10.0)), 1e-8)))
    return {
        "base_state_dim": int(Config.base_state_dim()),
        "share_hidden_dims": list(getattr(Config, "SHARE_LAYER_HIDDEN_DIMS", []) or []),
        "share_output_dim": 64,
        "q_head_dims": list(getattr(Config, "Q_HEAD_DIMS", []) or []),
        "q_parameterization": str(getattr(Config, "Q_PARAMETERIZATION", "direct")),
        "p0_head_dims": list(getattr(Config, "P0_HEAD_DIMS", []) or []),
        "pi_head_dims": list(getattr(Config, "PI_HEAD_DIMS", []) or []),
        "bp0_head_dims": list(getattr(Config, "BP0_HEAD_DIMS", []) or []),
        "bpi_head_dims": list(getattr(Config, "BPI_HEAD_DIMS", []) or []),
        "barz_hidden_dims": [64, 32, 16],
        "bari_hidden_dims": [64, 32, 16],
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
        q_head_dims=list(spec.get("q_head_dims") or []),
        # Missing means a legacy checkpoint whose head produced q_unit and whose
        # model multiplied it by b. Never reinterpret it as direct-Q.
        q_parameterization=str(spec.get("q_parameterization", "b_times_unit")),
        p0_head_dims=list(spec.get("p0_head_dims") or []),
        pi_head_dims=list(spec.get("pi_head_dims") or []),
        bp0_head_dims=list(spec.get("bp0_head_dims") or []),
        bpi_head_dims=list(spec.get("bpi_head_dims") or []),
        barz_hidden_dims=list(spec.get("barz_hidden_dims") or [64, 32, 16]),
        bari_hidden_dims=list(spec.get("bari_hidden_dims") or [64, 32, 16]),
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
    """Policy & Value 模型的输出

    ``bp0``/``bpI``/``bp``/``bp_cond`` are the *conditional* refinancing policy
    produced by the network heads: they are only meaningful when the current
    parent state has ``eta_t = 1``. ``bp0_effective``/``bpI_effective``/
    ``bp_effective`` are the *realized/effective* next-leverage policy: they equal
    the conditional policy when ``eta_t = 1`` and fall back to ``b_t`` when
    ``eta_t = 0``, where refinancing is not available and ``bp_t`` is not a
    control. ``bp_t`` is never a control for ``eta_t = 0``.
    """
    Q: torch.Tensor          # 债券价值
    bp0: torch.Tensor        # 不投资时杠杆候选（条件策略）
    bpI: torch.Tensor        # 投资时杠杆候选（条件策略）
    P0: torch.Tensor         # 兼容别名：V0
    PI: torch.Tensor         # 兼容别名：VI
    bar_i: torch.Tensor      # 兼容别名：bar_i_eff
    bar_z: torch.Tensor      # 破产门槛
    P: torch.Tensor          # 综合股票价值
    Phat: torch.Tensor       # P hat (中间值)
    bp: torch.Tensor         # 综合杠杆候选（条件策略）
    V0: torch.Tensor         # 不投资 branch value
    VI: torch.Tensor         # 投资 branch value
    bar_i_cond: torch.Tensor # 存活条件下投资概率
    bar_i_eff: torch.Tensor  # 考虑当前违约后的有效投资概率
    survival_prob: torch.Tensor
    bp_cond: torch.Tensor    # 存活条件下混合杠杆候选（条件策略）
    bp0_effective: torch.Tensor = None  # 有效下一期杠杆策略（eta_t = 0 时为 b_t）
    bpI_effective: torch.Tensor = None  # 有效下一期杠杆策略（eta_t = 0 时为 b_t）
    bp_effective: torch.Tensor = None   # 有效下一期杠杆策略（eta_t = 0 时为 b_t）


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
        q_head_dims: Optional[list] = None,
        q_parameterization: Optional[str] = None,
        p0_head_dims: Optional[list] = None,
        pi_head_dims: Optional[list] = None,
        bp0_head_dims: Optional[list] = None,
        bpi_head_dims: Optional[list] = None,
        barz_hidden_dims: Optional[list] = None,
        bari_hidden_dims: Optional[list] = None,
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
        self.q_head_dims = list(q_head_dims if q_head_dims is not None else getattr(Config, "Q_HEAD_DIMS", []))
        self.q_parameterization = str(
            q_parameterization
            if q_parameterization is not None
            else getattr(Config, "Q_PARAMETERIZATION", "direct")
        ).lower()
        if self.q_parameterization not in {"direct", "b_times_unit", "hybrid_regime"}:
            raise ValueError(
                "q_parameterization must be 'direct', 'b_times_unit', or "
                "'hybrid_regime', got "
                f"{self.q_parameterization!r}"
            )
        self.p0_head_dims = list(p0_head_dims if p0_head_dims is not None else getattr(Config, "P0_HEAD_DIMS", []))
        self.pi_head_dims = list(pi_head_dims if pi_head_dims is not None else getattr(Config, "PI_HEAD_DIMS", []))
        self.bp0_head_dims = list(bp0_head_dims if bp0_head_dims is not None else getattr(Config, "BP0_HEAD_DIMS", []))
        self.bpi_head_dims = list(bpi_head_dims if bpi_head_dims is not None else getattr(Config, "BPI_HEAD_DIMS", []))
        self.barz_hidden_dims = list(barz_hidden_dims if barz_hidden_dims is not None else [64, 32, 16])
        self.bari_hidden_dims = list(bari_hidden_dims if bari_hidden_dims is not None else [64, 32, 16])
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

        self.q_head = QHead(
            input_dim=share_output_dim,
            hidden_dims=self.q_head_dims,
            output_activation=None if self.q_parameterization == "direct" else "softplus",
        )
        self.v0_head = PHead(input_dim=share_output_dim, hidden_dims=self.p0_head_dims, requires_i=False)
        self.vi_head = PHead(input_dim=share_output_dim, hidden_dims=self.pi_head_dims, requires_i=True)
        self.bp0_head = BpHead(input_dim=share_output_dim, hidden_dims=self.bp0_head_dims, requires_i=False)
        self.bpi_head = BpHead(input_dim=share_output_dim, hidden_dims=self.bpi_head_dims, requires_i=True)
        
        # 辅助模型
        self.barz_model = BarzModel(
            input_dim=base_state_dim,
            hidden_dims=self.barz_hidden_dims,
            dropout=dropout
        )
        
        self.bari_model = BariModel(
            input_dim=base_state_dim,
            hidden_dims=self.bari_hidden_dims,
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
            "q_head_dims": list(self.q_head_dims),
            "q_parameterization": self.q_parameterization,
            "p0_head_dims": list(self.p0_head_dims),
            "pi_head_dims": list(self.pi_head_dims),
            "bp0_head_dims": list(self.bp0_head_dims),
            "bpi_head_dims": list(self.bpi_head_dims),
            "barz_hidden_dims": list(self.barz_hidden_dims),
            "bari_hidden_dims": list(self.bari_hidden_dims),
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

    def _q_unit_output(self, firm_state: torch.Tensor) -> torch.Tensor:
        """Return the raw Q-head output.

        It is a direct total-debt value in ``direct`` mode and a nonnegative
        unit bond price in ``b_times_unit``/``hybrid_regime`` modes.
        """
        base_state, _, b = self._split_state(firm_state)
        h_q = self.q_encoder(base_state)
        return self.q_head(h_q)

    def _q_claim_output(self, firm_state: torch.Tensor) -> torch.Tensor:
        """Return the live debt-claim value used by P/BP and Q Bellman.

        In ``hybrid_regime`` this is always ``b * q_unit``.  It is conditional
        on a live firm issuing/holding the claim, so a current-state equity
        default classifier must never replace it with realized recovery.
        """
        q_unit = self._q_unit_output(firm_state)
        if self.q_parameterization in {"b_times_unit", "hybrid_regime"}:
            b = firm_state[:, SIMMODEL.B:SIMMODEL.B + 1]
            return torch.clamp(b, min=0.0) * q_unit
        return q_unit

    def _q_survival_output(self, firm_state: torch.Tensor) -> torch.Tensor:
        """Backward-compatible alias for :meth:`_q_claim_output`."""
        return self._q_claim_output(firm_state)

    def _q_recovery_output(self, firm_state: torch.Tensor) -> torch.Tensor:
        """Structural default recovery; deliberately contains no division by b."""
        x = firm_state[:, SIMMODEL.X:SIMMODEL.X + 1]
        z = firm_state[:, SIMMODEL.Z:SIMMODEL.Z + 1]
        return self.phi.to(dtype=firm_state.dtype) * (
            1.0 - self.delta.to(dtype=firm_state.dtype) + torch.exp(x + z)
        )

    def _q_effective_output(
        self,
        firm_state: torch.Tensor,
        *,
        q_claim: Optional[torch.Tensor] = None,
        phat: Optional[torch.Tensor] = None,
        default_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return realized-state Q for settlement, simulation, and plotting.

        This realized-default hard gate is not a candidate-issuance pricing
        rule. P/BP/Q Bellman code must call ``_q_claim_output`` instead. The
        public ``forward`` path supplies its already-computed ``Phat`` so it
        never evaluates equity twice.
        """
        if self.q_parameterization != "hybrid_regime":
            return self._q_claim_output(firm_state)
        if default_mask is None:
            if phat is None:
                raise ValueError(
                    "hybrid_regime effective Q requires explicit phat or default_mask"
                )
            default_mask = phat <= 0.0
        default_mask = default_mask.to(dtype=torch.bool)
        b = firm_state[:, SIMMODEL.B:SIMMODEL.B + 1]
        zero_mask = b <= 0.0
        claim_q = self._q_claim_output(firm_state) if q_claim is None else q_claim
        recovery = self._q_recovery_output(firm_state)
        return torch.where(
            zero_mask,
            torch.zeros_like(claim_q),
            torch.where(default_mask, recovery, claim_q),
        )

    def _q_output(
        self,
        firm_state: torch.Tensor,
        *,
        phat: Optional[torch.Tensor] = None,
        default_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compatibility Q API with mode-specific public semantics."""
        if self.q_parameterization != "hybrid_regime":
            return self._q_claim_output(firm_state)
        if phat is None and default_mask is None:
            phat = self.forward_equity(firm_state)["Phat"]
        return self._q_effective_output(
            firm_state,
            phat=phat,
            default_mask=default_mask,
        )

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
        q_claim = (
            self._q_claim_output(firm_state)
            if self.q_parameterization == "hybrid_regime"
            else None
        )
        Q = None if q_claim is not None else self._q_output(firm_state)
        bp0, bpI = self._policy_outputs(firm_state)
        V0, VI = self._value_outputs(firm_state)

        Phat, P, bar_z, survival_prob = self.cal_phats(firm_state)
        if Q is None:
            Q = self._q_effective_output(firm_state, q_claim=q_claim, phat=Phat)
        bar_i_cond = self.derived.investment_conditional(V0, VI)
        bar_i_eff = survival_prob * bar_i_cond
        bp_cond = self.cal_bp(bp0, bpI, bar_i_cond)
        bp = survival_prob * bp_cond + (1.0 - survival_prob) * bp0

        # ``bp0``/``bpI``/``bp`` above are the conditional policy heads and stay
        # unchanged for checkpoint/logit compatibility.  Realized next leverage
        # is governed by the CURRENT parent eta_t only:
        #     b_{t+1} = eta_t * bp_t + (1 - eta_t) * b_t
        # so the effective exports collapse to ``b_t`` whenever eta_t = 0, where
        # refinancing is unavailable and ``bp_t`` is not a control.
        b_current = firm_state[:, SIMMODEL.B:SIMMODEL.B + 1]
        eta_current = firm_state[:, SIMMODEL.ETA:SIMMODEL.ETA + 1]
        bp0_effective = apply_refinancing_policy(
            b_current=b_current, bp_candidate=bp0, eta_current=eta_current
        )
        bpI_effective = apply_refinancing_policy(
            b_current=b_current, bp_candidate=bpI, eta_current=eta_current
        )
        bp_effective = apply_refinancing_policy(
            b_current=b_current, bp_candidate=bp, eta_current=eta_current
        )

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
            bp0_effective=bp0_effective,
            bpI_effective=bpI_effective,
            bp_effective=bp_effective,
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
        eta_current: torch.Tensor
    ) -> torch.Tensor:
        """
        更新杠杆
        
        b_{t+1} = η_t * bp_t + (1 - η_t) * b_t
        
        Args:
            b_old: 旧杠杆
            bp: 杠杆候选（仅在 η_t = 1 时是真实控制变量）
            eta_current: 当前父状态的再融资实现 η_t
        
        Returns:
            b_new: 新杠杆
        """
        return apply_refinancing_policy(
            b_current=b_old,
            bp_candidate=bp,
            eta_current=eta_current,
        )
    
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
