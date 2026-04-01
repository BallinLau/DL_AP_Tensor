"""
Policy & Value wrapper with separated Q and PV/BP blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, NamedTuple, Tuple

from .share_layer import ShareLayer, QHead, BpHead, PHead, BarzModel, BariModel

import sys
sys.path.append('..')
from config import Config, SIMMODEL


class PVBPOutput(NamedTuple):
    bp0: torch.Tensor
    bpI: torch.Tensor
    V0: torch.Tensor
    VI: torch.Tensor
    Vhat: torch.Tensor
    chi: torch.Tensor
    bar_i_cond: torch.Tensor
    P0: torch.Tensor
    PI: torch.Tensor
    bar_i: torch.Tensor
    bar_z: torch.Tensor
    P: torch.Tensor
    Phat: torch.Tensor
    bp: torch.Tensor


class PolicyValueOutput(NamedTuple):
    Q: torch.Tensor
    bp0: torch.Tensor
    bpI: torch.Tensor
    V0: torch.Tensor
    VI: torch.Tensor
    Vhat: torch.Tensor
    chi: torch.Tensor
    bar_i_cond: torch.Tensor
    P0: torch.Tensor
    PI: torch.Tensor
    bar_i: torch.Tensor
    bar_z: torch.Tensor
    P: torch.Tensor
    Phat: torch.Tensor
    bp: torch.Tensor


class QModel(nn.Module):
    """
    独立的旧债价格模块。

    只负责给当前 state 中已有的 debt contract 定价，不输出 bp / P / V。
    """

    def __init__(
        self,
        base_state_dim: int = 6,
        share_hidden_dims: Optional[list] = None,
        share_output_dim: int = 64,
        q_hidden_dims: Optional[list] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.share_layer = ShareLayer(
            input_dim=base_state_dim,
            hidden_dims=share_hidden_dims,
            output_dim=share_output_dim,
            dropout=dropout,
        )
        self.q_head = QHead(
            input_dim=share_output_dim,
            hidden_dims=q_hidden_dims,
        )

    @staticmethod
    def extract_base_state(firm_state: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            firm_state[:, :SIMMODEL.I],
            firm_state[:, SIMMODEL.X:],
        ], dim=-1)

    def get_q_unit(self, firm_state: torch.Tensor) -> torch.Tensor:
        h = self.share_layer(self.extract_base_state(firm_state))
        return self.q_head(h)

    def get_Q(self, firm_state: torch.Tensor) -> torch.Tensor:
        q_unit = self.get_q_unit(firm_state)
        b_nonneg = torch.clamp(firm_state[:, SIMMODEL.B:SIMMODEL.B + 1], min=0.0)
        return b_nonneg * q_unit

    def forward(self, firm_state: torch.Tensor) -> torch.Tensor:
        return self.get_Q(firm_state)


class PVBPModel(nn.Module):
    """
    独立的 equity-side value / policy block。

    输出：
    - V0 / VI / bar_i_cond
    - bp0 / bpI
    - 派生 Vhat / chi / bar_z / P / bar_i / bp
    """

    def __init__(
        self,
        base_state_dim: int = 6,
        share_hidden_dims: Optional[list] = None,
        share_output_dim: int = 64,
        p_hidden_dims: Optional[list] = None,
        bp_hidden_dims: Optional[list] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.share_layer = ShareLayer(
            input_dim=base_state_dim,
            hidden_dims=share_hidden_dims,
            output_dim=share_output_dim,
            dropout=dropout,
        )
        self.bp0_head = BpHead(
            input_dim=share_output_dim,
            hidden_dims=bp_hidden_dims,
            requires_i=False,
        )
        self.bpI_head = BpHead(
            input_dim=share_output_dim,
            hidden_dims=bp_hidden_dims,
            requires_i=True,
        )
        self.p0_head = PHead(
            input_dim=share_output_dim,
            hidden_dims=p_hidden_dims,
            requires_i=False,
        )
        self.pI_head = PHead(
            input_dim=share_output_dim,
            hidden_dims=p_hidden_dims,
            requires_i=True,
        )
        # Keep these auxiliary nets for diagnostics/backward compatibility helpers.
        self.barz_model = BarzModel(input_dim=base_state_dim, dropout=dropout)
        self.bari_model = BariModel(input_dim=base_state_dim, dropout=dropout)
        self.chi_warmup_factor = 1.0

    @staticmethod
    def extract_base_state(firm_state: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            firm_state[:, :SIMMODEL.I],
            firm_state[:, SIMMODEL.X:],
        ], dim=-1)

    def _encode(self, firm_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base_state = self.extract_base_state(firm_state)
        i = firm_state[:, SIMMODEL.I:SIMMODEL.I + 1]
        h = self.share_layer(base_state)
        return h, i

    def get_combined_output(self, firm_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, i = self._encode(firm_state)
        V0 = self.p0_head(h)
        VI = self.pI_head(h, i)
        bar_i_cond = torch.sigmoid(10 * (VI - V0))
        return V0, VI, bar_i_cond

    def cal_phats(
        self,
        firm_state: torch.Tensor,
        V0: torch.Tensor,
        VI: torch.Tensor,
        simulated_i: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = firm_state.device
        batch_size = firm_state.size(0)
        if simulated_i is None:
            n_i_points = int(getattr(Config, "PV_I_INTEGRATION_POINTS", 11))
            n_i_points = max(3, n_i_points)
            simulated_i = torch.linspace(0.0, Config.I_THRESHOLD, steps=n_i_points, device=device).unsqueeze(-1)

        v0_list = []
        vi_list = []
        for i_val in simulated_i:
            modified = firm_state.clone()
            modified[:, SIMMODEL.I] = i_val.expand(batch_size)
            v0_i, vi_i, _ = self.get_combined_output(modified)
            v0_list.append(v0_i)
            vi_list.append(vi_i)

        v0_stack = torch.stack(v0_list, dim=0)
        vi_stack = torch.stack(vi_list, dim=0)
        max_vals = torch.max(v0_stack, vi_stack)
        Vhat = max_vals.mean(dim=0)

        p_beta = max(float(getattr(Config, "P_SOFTPLUS_BETA", 8.0)), 1e-6)
        P = F.softplus(p_beta * Vhat) / p_beta

        barz_temp = float(getattr(Config, "BARZ_LOGIT_TEMP", 3.0))
        base_chi = torch.sigmoid(barz_temp * Vhat)
        chi_warmup_factor = min(max(float(getattr(self, "chi_warmup_factor", 1.0)), 0.0), 1.0)
        if chi_warmup_factor < 1.0:
            chi = chi_warmup_factor * base_chi + (1.0 - chi_warmup_factor) * torch.ones_like(base_chi)
        else:
            chi = base_chi
        bar_z = 1.0 - chi
        return Vhat, P, chi, bar_z

    @staticmethod
    def cal_bp(bp0: torch.Tensor, bpI: torch.Tensor, bar_i: torch.Tensor) -> torch.Tensor:
        return bar_i * bpI + (1 - bar_i) * bp0

    def get_bar_z(self, firm_state: torch.Tensor) -> torch.Tensor:
        base_state = self.extract_base_state(firm_state)
        return self.barz_model(base_state)

    def get_bar_i_value(self, firm_state: torch.Tensor) -> torch.Tensor:
        base_state = self.extract_base_state(firm_state)
        return self.bari_model(base_state)

    def forward(
        self,
        firm_state: torch.Tensor,
        simulated_i: Optional[torch.Tensor] = None,
    ) -> PVBPOutput:
        h, i = self._encode(firm_state)
        bp0 = self.bp0_head(h)
        bpI = self.bpI_head(h, i)
        V0 = self.p0_head(h)
        VI = self.pI_head(h, i)
        bar_i_cond = torch.sigmoid(10 * (VI - V0))
        Vhat, P, chi, bar_z = self.cal_phats(firm_state, V0, VI, simulated_i=simulated_i)
        bar_i = chi * bar_i_cond
        bp = self.cal_bp(bp0, bpI, bar_i)
        return PVBPOutput(
            bp0=bp0,
            bpI=bpI,
            V0=V0,
            VI=VI,
            Vhat=Vhat,
            chi=chi,
            bar_i_cond=bar_i_cond,
            P0=V0,
            PI=VI,
            bar_i=bar_i,
            bar_z=bar_z,
            P=P,
            Phat=Vhat,
            bp=bp,
        )


class PolicyValueModel(nn.Module):
    """
    Backward-compatible wrapper around separate Q and PV/BP modules.
    """

    def __init__(
        self,
        base_state_dim: int = 6,
        share_hidden_dims: Optional[list] = None,
        share_output_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.q_model = QModel(
            base_state_dim=base_state_dim,
            share_hidden_dims=share_hidden_dims,
            share_output_dim=share_output_dim,
            dropout=dropout,
        )
        self.pvbp_model = PVBPModel(
            base_state_dim=base_state_dim,
            share_hidden_dims=share_hidden_dims,
            share_output_dim=share_output_dim,
            dropout=dropout,
        )

    @staticmethod
    def extract_base_state(firm_state: torch.Tensor) -> torch.Tensor:
        return PVBPModel.extract_base_state(firm_state)

    def get_q_unit(self, firm_state: torch.Tensor) -> torch.Tensor:
        return self.q_model.get_q_unit(firm_state)

    def get_shared_output(self, firm_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_model(firm_state)
        pvbp = self.pvbp_model(firm_state)
        return q, pvbp.bp0, pvbp.bpI

    def get_combined_output(self, firm_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.pvbp_model.get_combined_output(firm_state)

    def get_bar_z(self, firm_state: torch.Tensor) -> torch.Tensor:
        return self.pvbp_model.get_bar_z(firm_state)

    def get_bar_i_value(self, firm_state: torch.Tensor) -> torch.Tensor:
        return self.pvbp_model.get_bar_i_value(firm_state)

    def update_leverage(self, b_old: torch.Tensor, bp: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        return eta * bp + (1 - eta) * b_old

    def update_capital(self, K_old: torch.Tensor, bar_i: torch.Tensor) -> torch.Tensor:
        g = torch.tensor(Config.G, device=K_old.device, dtype=K_old.dtype)
        return bar_i * g * K_old + (1 - bar_i) * K_old

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        self.train()

    def get_module_parameters(self, module_name: str):
        modules = {
            'q': self.q_model,
            'pvbp': self.pvbp_model,
        }
        if module_name not in modules:
            raise ValueError(f"Unknown module: {module_name}")
        return modules[module_name].parameters()

    def forward(
        self,
        firm_state: torch.Tensor,
        return_all: bool = True,
    ) -> PolicyValueOutput:
        q = self.q_model(firm_state)
        pvbp = self.pvbp_model(firm_state)
        return PolicyValueOutput(
            Q=q,
            bp0=pvbp.bp0,
            bpI=pvbp.bpI,
            V0=pvbp.V0,
            VI=pvbp.VI,
            Vhat=pvbp.Vhat,
            chi=pvbp.chi,
            bar_i_cond=pvbp.bar_i_cond,
            P0=pvbp.P0,
            PI=pvbp.PI,
            bar_i=pvbp.bar_i,
            bar_z=pvbp.bar_z,
            P=pvbp.P,
            Phat=pvbp.Phat,
            bp=pvbp.bp,
        )


class CalPhats:
    """
    兼容旧接口：统一计算 Vhat / P / chi / bar_z。
    """

    def __init__(self, pvbp_model: PVBPModel, barz_model: Optional[BarzModel] = None):
        self.pvbp_model = pvbp_model
        self.barz_model = barz_model

    def __call__(self, firm_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        V0, VI, _ = self.pvbp_model.get_combined_output(firm_state)
        return self.pvbp_model.cal_phats(firm_state, V0, VI)
