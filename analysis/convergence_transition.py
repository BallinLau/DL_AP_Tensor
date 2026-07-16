from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .economic_config import AnalysisEconomicConfig


@dataclass(frozen=True)
class FirmStateIndex:
    B: int = 0
    Z: int = 1
    ETA: int = 2
    I: int = 3
    X: int = 4
    HATCF: int = 5
    LNKF: int = 6


@dataclass
class MacroTransitionContext:
    hatc_cal: torch.Tensor
    lnk_cal: torch.Tensor


@dataclass
class ChildExogenousBundle:
    z_next: torch.Tensor
    eta_next: torch.Tensor
    i_next: torch.Tensor
    x_next: torch.Tensor
    hatcf_next: torch.Tensor
    lnkf_next: torch.Tensor
    m_raw: torch.Tensor
    branch_weights: torch.Tensor


@dataclass
class ConvergenceShockBank:
    eps_x: torch.Tensor
    eps_z: torch.Tensor
    u_eta: torch.Tensor
    u_i: torch.Tensor
    seed: int

    @property
    def storage_numel(self) -> int:
        return int(self.eps_x.numel() + self.eps_z.numel() + self.u_eta.numel() + self.u_i.numel())

    @property
    def base_shape(self) -> tuple[int, int, int]:
        return tuple(self.eps_x.shape)

    @classmethod
    def create(
        cls,
        n_reference: int,
        n_child_shocks: int,
        *,
        seed: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "ConvergenceShockBank":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        shape = (int(n_reference), int(n_child_shocks), 1)
        eps_x = torch.randn(shape, generator=gen, dtype=dtype)
        eps_z = torch.randn(shape, generator=gen, dtype=dtype)
        u_eta = torch.rand(shape, generator=gen, dtype=dtype)
        u_i = torch.rand(shape, generator=gen, dtype=dtype)
        device = torch.device(device)
        return cls(
            eps_x=eps_x.to(device),
            eps_z=eps_z.to(device),
            u_eta=u_eta.to(device),
            u_i=u_i.to(device),
            seed=int(seed),
        )

    def gather(self, reference_index: torch.Tensor) -> "ConvergenceShockBank":
        idx = reference_index.to(device=self.eps_x.device, dtype=torch.long).reshape(-1)
        return ConvergenceShockBank(
            eps_x=self.eps_x.index_select(0, idx),
            eps_z=self.eps_z.index_select(0, idx),
            u_eta=self.u_eta.index_select(0, idx),
            u_i=self.u_i.index_select(0, idx),
            seed=self.seed,
        )


def normalize_branch_weights(
    branch_weights: Optional[torch.Tensor],
    *,
    n_parent: int,
    n_child: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if branch_weights is None:
        return torch.full((n_parent, n_child), 1.0 / float(n_child), device=device, dtype=dtype)
    weights = branch_weights.to(device=device, dtype=dtype)
    if weights.ndim == 1:
        if weights.shape[0] != n_child:
            raise ValueError(f"branch_weights length must be {n_child}, got {weights.shape[0]}")
        weights = weights.unsqueeze(0).expand(n_parent, n_child)
    elif weights.ndim == 2:
        if tuple(weights.shape) != (n_parent, n_child):
            raise ValueError(f"branch_weights shape must be {(n_parent, n_child)}, got {tuple(weights.shape)}")
    else:
        raise ValueError("branch_weights must have shape [J] or [B,J]")
    if torch.any(weights < 0):
        raise ValueError("branch_weights must be nonnegative")
    if not torch.allclose(weights.sum(dim=1), torch.ones(n_parent, device=device, dtype=dtype), atol=1e-5, rtol=1e-5):
        raise ValueError("branch_weights rows must sum to one")
    return weights


def build_child_exogenous_bundle(
    sdf_fc1_model: torch.nn.Module,
    parent_states: torch.Tensor,
    macro_context: MacroTransitionContext,
    shock_bank: ConvergenceShockBank,
    *,
    economic_config: AnalysisEconomicConfig,
    branch_weights: Optional[torch.Tensor] = None,
    return_physical: bool = True,
) -> ChildExogenousBundle:
    idx = FirmStateIndex()
    device = parent_states.device
    dtype = parent_states.dtype
    bsz = parent_states.shape[0]
    n_child = shock_bank.eps_x.shape[1]
    if shock_bank.eps_x.shape[0] != bsz:
        raise ValueError(
            f"shock bank reference dimension {shock_bank.eps_x.shape[0]} does not match parent batch {bsz}"
        )

    x = parent_states[:, idx.X:idx.X + 1]
    z = parent_states[:, idx.Z:idx.Z + 1]
    x_next = (
        (1.0 - economic_config.RHO_X) * economic_config.XBAR
        + economic_config.RHO_X * x.unsqueeze(1)
        + economic_config.SIGMA_X * shock_bank.eps_x.to(device=device, dtype=dtype)
    )
    z_next = (
        (1.0 - economic_config.RHO_Z) * economic_config.ZBAR
        + economic_config.RHO_Z * z.unsqueeze(1)
        + economic_config.SIGMA_Z * shock_bank.eps_z.to(device=device, dtype=dtype)
    )
    eta_next = (shock_bank.u_eta.to(device=device, dtype=dtype) < float(economic_config.ZETA)).to(dtype)
    i_next = float(economic_config.I_THRESHOLD) * shock_bank.u_i.to(device=device, dtype=dtype)

    hatc_cal = macro_context.hatc_cal.to(device=device, dtype=dtype)
    lnk_cal = macro_context.lnk_cal.to(device=device, dtype=dtype)
    if hatc_cal.ndim == 1:
        hatc_cal = hatc_cal.unsqueeze(-1)
    if lnk_cal.ndim == 1:
        lnk_cal = lnk_cal.unsqueeze(-1)

    _, _, m_raw, hatcf_next, lnkf_next = sdf_fc1_model.forward_step(
        x_prev=x,
        x_curr=x_next,
        hatcf_prev=hatc_cal,
        lnkf_prev=lnk_cal,
        return_physical=return_physical,
    )
    weights = normalize_branch_weights(
        branch_weights,
        n_parent=bsz,
        n_child=n_child,
        device=device,
        dtype=dtype,
    )
    return ChildExogenousBundle(
        z_next=z_next,
        eta_next=eta_next,
        i_next=i_next,
        x_next=x_next,
        hatcf_next=hatcf_next,
        lnkf_next=lnkf_next,
        m_raw=m_raw,
        branch_weights=weights,
    )
