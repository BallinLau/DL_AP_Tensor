"""
Refreshable shock bank for SDF wealth-loss double sampling.

The bank stores only aggregate AR(1) innovations.  Fixed Treatment B children
remain the source of FC1 reconstruction targets.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


def _make_generator(device: torch.device) -> torch.Generator:
    try:
        return torch.Generator(device=device)
    except RuntimeError:
        return torch.Generator()


@dataclass
class SDFShockBank:
    eps: torch.Tensor
    bank_size: int
    refresh_id: int
    base_seed: int

    @classmethod
    def create(
        cls,
        n_parents: int,
        bank_size: int,
        device: torch.device,
        base_seed: int,
        dtype: torch.dtype = torch.float32,
    ) -> "SDFShockBank":
        if bank_size < 2:
            raise ValueError("SDF shock bank requires bank_size >= 2.")
        if n_parents <= 0:
            raise ValueError("SDF shock bank requires n_parents > 0.")

        generator = _make_generator(device)
        generator.manual_seed(int(base_seed))
        eps = torch.randn(
            n_parents,
            bank_size,
            1,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        return cls(eps=eps, bank_size=bank_size, refresh_id=0, base_seed=int(base_seed))

    def refresh_(self, seed: Optional[int] = None) -> None:
        self.refresh_id += 1
        generator = _make_generator(self.eps.device)
        generator.manual_seed(int(self.base_seed + self.refresh_id if seed is None else seed))
        self.eps.normal_(mean=0.0, std=1.0, generator=generator)

    def sample_pair(
        self,
        parent_indices: torch.Tensor,
        pair_generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = int(parent_indices.numel())
        if n == 0:
            empty_idx = torch.empty(0, device=self.eps.device, dtype=torch.long)
            empty_eps = torch.empty(0, 1, device=self.eps.device, dtype=self.eps.dtype)
            return empty_eps, empty_eps, empty_idx, empty_idx

        parent_indices = parent_indices.to(device=self.eps.device, dtype=torch.long).reshape(-1)
        if parent_indices.min().item() < 0 or parent_indices.max().item() >= self.eps.shape[0]:
            raise IndexError(
                "parent_indices out of SDF shock bank bounds: "
                f"min={int(parent_indices.min().item())}, "
                f"max={int(parent_indices.max().item())}, n_parents={self.eps.shape[0]}"
            )

        j1 = torch.randint(
            0,
            self.bank_size,
            (n,),
            device=self.eps.device,
            generator=pair_generator,
        )
        offset = torch.randint(
            1,
            self.bank_size,
            (n,),
            device=self.eps.device,
            generator=pair_generator,
        )
        j2 = (j1 + offset) % self.bank_size

        eps1 = self.eps[parent_indices, j1]
        eps2 = self.eps[parent_indices, j2]
        return eps1, eps2, j1, j2


def shocks_to_x_children(
    x_parent: torch.Tensor,
    eps1: torch.Tensor,
    eps2: torch.Tensor,
    rho_x: float,
    sigma_x: float,
    xbar: float,
) -> torch.Tensor:
    conditional_mean = (1.0 - float(rho_x)) * float(xbar) + float(rho_x) * x_parent
    x1 = conditional_mean + float(sigma_x) * eps1.to(device=x_parent.device, dtype=x_parent.dtype)
    x2 = conditional_mean + float(sigma_x) * eps2.to(device=x_parent.device, dtype=x_parent.dtype)
    return torch.stack([x1, x2], dim=1)


def shock_pair_diagnostics(
    eps1: torch.Tensor,
    eps2: torch.Tensor,
    j1: torch.Tensor,
    j2: torch.Tensor,
    bank_size: int,
) -> Dict[str, float]:
    e1 = eps1.detach().reshape(-1).to(torch.float32)
    e2 = eps2.detach().reshape(-1).to(torch.float32)
    if e1.numel() == 0:
        return {
            "sdf_pair_collision_rate": 0.0,
            "sdf_pair_unique_ratio": 0.0,
            "sdf_eps1_mean": 0.0,
            "sdf_eps1_std": 0.0,
            "sdf_eps2_mean": 0.0,
            "sdf_eps2_std": 0.0,
            "sdf_eps_cross_corr": 0.0,
            "sdf_pair_negative_product_share": 0.0,
        }

    e1_centered = e1 - e1.mean()
    e2_centered = e2 - e2.mean()
    corr = (
        (e1_centered * e2_centered).mean()
        / (e1_centered.std(unbiased=False) * e2_centered.std(unbiased=False) + 1e-8)
    )
    pair_code = j1.to(torch.long) * int(bank_size) + j2.to(torch.long)
    return {
        "sdf_pair_collision_rate": float((j1 == j2).float().mean().item()),
        "sdf_pair_unique_ratio": float(torch.unique(pair_code).numel() / max(1, pair_code.numel())),
        "sdf_eps1_mean": float(e1.mean().item()),
        "sdf_eps1_std": float(e1.std(unbiased=False).item()),
        "sdf_eps2_mean": float(e2.mean().item()),
        "sdf_eps2_std": float(e2.std(unbiased=False).item()),
        "sdf_eps_cross_corr": float(corr.item()),
        "sdf_pair_negative_product_share": float(((e1 * e2) < 0).float().mean().item()),
    }
