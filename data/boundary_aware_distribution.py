"""Boundary-aware continuous firm-distribution primitives.

This module implements the two explicit representations used by the frozen-
model macro-data experiment:

``PreDecisionDistribution``
    A mixture of conditional Gaussian productivity densities.  Leverage,
    capital and timing/provenance attributes are fixed within a component;
    ``z_conditional_mean`` is a density parameter, never a realized firm
    state.

``PostDecisionTransportAtoms``
    Weighted quadrature records created *after* survival/investment/refinance
    decisions at an accepted market-clearing root.  Each record retains the
    concrete source productivity needed by the original AR(1) transition.

The helpers are intentionally optimizer-free and contain no stochastic
sampling.  They are suitable for deterministic candidate reconstruction and
candidate/commit parity checks.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch


BOUNDARY_AWARE_CONTINUUM_V1 = "boundary_aware_continuum_v1"


def _require_rank_two(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must be a rank-two [B,L] tensor")


def _tensor_digest(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in named_tensors:
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def predecision_fingerprint_from_state(state: Mapping[str, torch.Tensor]) -> str:
    """Hash every tensor that defines a boundary-aware candidate input."""
    names = (
        "component_mass",
        "b",
        "K",
        "z_conditional_mean",
        "z_conditional_std",
        "component_mask",
        "eta",
        "maintenance_eligibility",
        "component_id",
        "origin_kind",
        "origin_source_id",
        "origin_time",
        "evaluation_time",
        "cohort",
        "birth_time",
        "x",
        "hatcf",
        "lnkf",
        "economic_node_id",
    )
    return _tensor_digest((name, state[name]) for name in names if name in state)


@dataclass(frozen=True)
class PreDecisionDistribution:
    """Conditional-density mixture immediately before current decisions."""

    component_mass: torch.Tensor
    b: torch.Tensor
    capital_per_unit: torch.Tensor
    z_conditional_mean: torch.Tensor
    z_conditional_std: torch.Tensor
    component_mask: torch.Tensor
    economic_node_id: torch.Tensor
    component_id: torch.Tensor
    origin_kind: torch.Tensor
    origin_source_id: torch.Tensor
    origin_time: torch.Tensor
    evaluation_time: torch.Tensor
    cohort: torch.Tensor
    birth_time: torch.Tensor
    maintenance_eligibility: torch.Tensor
    eta_state: torch.Tensor

    def validate(self) -> "PreDecisionDistribution":
        shape = tuple(self.component_mass.shape)
        for field in fields(self):
            value = getattr(self, field.name)
            _require_rank_two(field.name, value)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"{field.name} shape {tuple(value.shape)} does not match {shape}"
                )
        if self.component_mask.dtype is not torch.bool:
            raise ValueError("component_mask must be boolean")
        if self.maintenance_eligibility.dtype is not torch.bool:
            raise ValueError("maintenance_eligibility must be boolean")
        active = self.component_mask
        if not torch.isfinite(self.component_mass[active]).all():
            raise ValueError("active component mass is non-finite")
        if (self.component_mass[active] < 0.0).any():
            raise ValueError("component mass must be non-negative")
        if (self.capital_per_unit[active] <= 0.0).any():
            raise ValueError("active component capital must be positive")
        if (self.z_conditional_std[active] <= 0.0).any():
            raise ValueError("active conditional z standard deviation must be positive")
        return self

    @property
    def batch_size(self) -> int:
        return int(self.component_mass.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.component_mass.shape[1])

    def total_mass(self) -> torch.Tensor:
        return torch.where(
            self.component_mask, self.component_mass, torch.zeros_like(self.component_mass)
        ).sum(dim=1)

    def total_capital(self) -> torch.Tensor:
        return torch.where(
            self.component_mask,
            self.component_mass * self.capital_per_unit,
            torch.zeros_like(self.component_mass),
        ).sum(dim=1)

    def aggregate_debt(self) -> torch.Tensor:
        return torch.where(
            self.component_mask,
            self.component_mass * self.capital_per_unit * self.b,
            torch.zeros_like(self.component_mass),
        ).sum(dim=1)

    def fingerprint(self) -> str:
        return _tensor_digest(
            (field.name, getattr(self, field.name)) for field in fields(self)
        )

    def as_state_tensors(self) -> Dict[str, torch.Tensor]:
        """Expose a root-controller-compatible, non-representative state view.

        ``z`` is included only as an explicitly named density mean for legacy
        metadata interfaces.  Boundary-aware evaluators must consume
        ``z_conditional_mean/std`` and are forbidden from hard-defaulting this
        alias.
        """
        return {
            "component_mass": self.component_mass,
            "firm_mass": self.component_mass,
            "b": self.b,
            "K": self.capital_per_unit,
            "z": self.z_conditional_mean,
            "z_conditional_mean": self.z_conditional_mean,
            "z_conditional_std": self.z_conditional_std,
            "alive": self.component_mask,
            "component_mask": self.component_mask,
            "economic_node_id_component": self.economic_node_id,
            "component_id": self.component_id,
            "origin_kind": self.origin_kind,
            "origin_source_id": self.origin_source_id,
            "origin_time": self.origin_time,
            "evaluation_time": self.evaluation_time,
            "cohort": self.cohort,
            "birth_time": self.birth_time,
            "maintenance_eligibility": self.maintenance_eligibility,
            "eta": self.eta_state,
        }

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor]) -> "PreDecisionDistribution":
        return cls(**{field.name: state[field.name] for field in fields(cls)}).validate()


@dataclass(frozen=True)
class PostDecisionTransportAtoms:
    """Weighted selected-state atoms used to build the next density mixture."""

    transport_weight: torch.Tensor
    source_z_current: torch.Tensor
    b_next: torch.Tensor
    capital_next_per_unit: torch.Tensor
    atom_mask: torch.Tensor
    source_component_id: torch.Tensor
    survival_region_id: torch.Tensor
    investment_region_id: torch.Tensor
    policy_branch_id: torch.Tensor
    economic_node_id: torch.Tensor
    source_time: torch.Tensor
    destination_time: torch.Tensor
    cohort: torch.Tensor
    birth_time: torch.Tensor
    maintenance_eligibility: torch.Tensor
    origin_kind: torch.Tensor
    origin_source_id: torch.Tensor

    def validate(self) -> "PostDecisionTransportAtoms":
        shape = tuple(self.transport_weight.shape)
        for field in fields(self):
            value = getattr(self, field.name)
            _require_rank_two(field.name, value)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"{field.name} shape {tuple(value.shape)} does not match {shape}"
                )
        if self.atom_mask.dtype is not torch.bool:
            raise ValueError("atom_mask must be boolean")
        active = self.atom_mask
        if not torch.isfinite(self.transport_weight[active]).all():
            raise ValueError("active transport weight is non-finite")
        if (self.transport_weight[active] < 0.0).any():
            raise ValueError("transport weights must be non-negative")
        if (self.capital_next_per_unit[active] <= 0.0).any():
            raise ValueError("active next capital must be positive")
        return self

    def total_mass(self) -> torch.Tensor:
        return torch.where(
            self.atom_mask,
            self.transport_weight,
            torch.zeros_like(self.transport_weight),
        ).sum(dim=1)

    def total_capital(self) -> torch.Tensor:
        return torch.where(
            self.atom_mask,
            self.transport_weight * self.capital_next_per_unit,
            torch.zeros_like(self.transport_weight),
        ).sum(dim=1)

    def aggregate_debt(self) -> torch.Tensor:
        return torch.where(
            self.atom_mask,
            self.transport_weight * self.capital_next_per_unit * self.b_next,
            torch.zeros_like(self.transport_weight),
        ).sum(dim=1)

    def fingerprint(self) -> str:
        return _tensor_digest(
            (field.name, getattr(self, field.name)) for field in fields(self)
        )

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, torch.Tensor]
    ) -> "PostDecisionTransportAtoms":
        return cls(**{field.name: state[field.name] for field in fields(cls)}).validate()


@dataclass(frozen=True)
class BoundaryAwareDecisionRecord:
    """Immutable selected-root ledger plus its exact transport quadrature."""

    predecision_fingerprint: str
    candidate_h: torch.Tensor
    ledger: Dict[str, torch.Tensor]
    transport_atoms: PostDecisionTransportAtoms
    provenance: Dict[str, Any]

    def detached(self) -> "BoundaryAwareDecisionRecord":
        return BoundaryAwareDecisionRecord(
            predecision_fingerprint=str(self.predecision_fingerprint),
            candidate_h=self.candidate_h.detach().clone(),
            ledger={name: value.detach().clone() for name, value in self.ledger.items()},
            transport_atoms=PostDecisionTransportAtoms(
                **{
                    field.name: getattr(self.transport_atoms, field.name).detach().clone()
                    for field in fields(self.transport_atoms)
                }
            ).validate(),
            provenance=dict(self.provenance),
        )

    def assert_ledger_parity(
        self,
        committed: Mapping[str, torch.Tensor],
        *,
        atol: float = 1.0e-10,
        rtol: float = 1.0e-10,
    ) -> None:
        for name, expected in self.ledger.items():
            if name not in committed:
                raise RuntimeError(f"committed boundary-aware ledger is missing {name}")
            torch.testing.assert_close(
                committed[name], expected, atol=atol, rtol=rtol
            )


def gauss_legendre_probability_nodes(
    lower_probability: torch.Tensor,
    upper_probability: torch.Tensor,
    points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gauss-Legendre nodes/weights on batched probability intervals.

    Bounds may have any matching shape.  Returned tensors append a quadrature
    dimension.  We never evaluate at probability zero or one, so normal
    quantiles stay finite even for infinite-support conditional densities.
    """
    if lower_probability.shape != upper_probability.shape:
        raise ValueError("probability interval bounds must have matching shapes")
    if points < 1:
        raise ValueError("quadrature points must be positive")
    if (lower_probability < 0.0).any() or (upper_probability > 1.0).any():
        raise ValueError("probability interval lies outside [0,1]")
    if (upper_probability < lower_probability).any():
        raise ValueError("probability interval has negative width")
    np_nodes, np_weights = np.polynomial.legendre.leggauss(int(points))
    nodes = torch.as_tensor(
        np_nodes, device=lower_probability.device, dtype=lower_probability.dtype
    )
    weights = torch.as_tensor(
        np_weights, device=lower_probability.device, dtype=lower_probability.dtype
    )
    midpoint = 0.5 * (lower_probability + upper_probability)
    half_width = 0.5 * (upper_probability - lower_probability)
    u = midpoint[..., None] + half_width[..., None] * nodes
    probability_weight = half_width[..., None] * weights
    tiny = torch.finfo(u.dtype).eps
    return u.clamp(min=tiny, max=1.0 - tiny), probability_weight


def normal_quantile(
    probability: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    while mean.ndim < probability.ndim:
        mean = mean.unsqueeze(-1)
        std = std.unsqueeze(-1)
    return mean + std * math.sqrt(2.0) * torch.erfinv(2.0 * probability - 1.0)


def normal_cdf(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf((value - mean) / (std * math.sqrt(2.0))))


def interval_mass_from_z_boundaries(
    component_mass: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    lower_z: torch.Tensor,
    upper_z: torch.Tensor,
) -> torch.Tensor:
    """Exact normal probability mass between possibly infinite z bounds."""
    if not (
        component_mass.shape
        == mean.shape
        == std.shape
        == lower_z.shape
        == upper_z.shape
    ):
        raise ValueError("all interval mass inputs must have matching shapes")
    lower_u = torch.where(
        torch.isneginf(lower_z), torch.zeros_like(lower_z), normal_cdf(lower_z, mean, std)
    )
    upper_u = torch.where(
        torch.isposinf(upper_z), torch.ones_like(upper_z), normal_cdf(upper_z, mean, std)
    )
    return component_mass * (upper_u - lower_u).clamp_min(0.0)


def propagate_ar1_to_predecision(
    atoms: PostDecisionTransportAtoms,
    *,
    rho_z: float,
    sigma_z: float,
    z_bar: float,
    next_component_ids: Optional[torch.Tensor] = None,
) -> PreDecisionDistribution:
    """Apply the original conditional AR(1) kernel exactly once per atom."""
    atoms.validate()
    if sigma_z <= 0.0 or abs(rho_z) >= 1.0:
        raise ValueError("invalid stationary Gaussian AR(1) parameters")
    shape = atoms.transport_weight.shape
    if next_component_ids is None:
        next_component_ids = torch.arange(
            shape[1], device=atoms.transport_weight.device, dtype=torch.long
        )[None, :].expand(shape[0], -1)
    if next_component_ids.shape != shape:
        raise ValueError("next_component_ids must match atom shape")
    mean = (1.0 - float(rho_z)) * float(z_bar) + float(rho_z) * atoms.source_z_current
    std = torch.full_like(mean, float(sigma_z))
    return PreDecisionDistribution(
        component_mass=atoms.transport_weight,
        b=atoms.b_next,
        capital_per_unit=atoms.capital_next_per_unit,
        z_conditional_mean=mean,
        z_conditional_std=std,
        component_mask=atoms.atom_mask & (atoms.transport_weight > 0.0),
        economic_node_id=atoms.economic_node_id,
        component_id=next_component_ids,
        origin_kind=atoms.origin_kind,
        origin_source_id=atoms.origin_source_id,
        origin_time=atoms.source_time,
        evaluation_time=atoms.destination_time,
        cohort=atoms.cohort,
        birth_time=atoms.birth_time,
        maintenance_eligibility=atoms.maintenance_eligibility,
        eta_state=torch.zeros_like(atoms.b_next),
    ).validate()


def propagate_ar1_with_eta_mixture(
    atoms: PostDecisionTransportAtoms,
    *,
    rho_z: float,
    sigma_z: float,
    z_bar: float,
    zeta: float,
) -> PreDecisionDistribution:
    """Advance z once and enumerate the independent current eta distribution.

    ``eta_{t+1}`` is a new discrete state.  It must not gate the already
    realized ``b_t -> b_{t+1}`` transition, so duplication occurs only after
    ``b_next`` is fixed in the transport atoms.
    """
    if not 0.0 <= float(zeta) <= 1.0:
        raise ValueError("zeta must be in [0,1]")
    base = propagate_ar1_to_predecision(
        atoms, rho_z=rho_z, sigma_z=sigma_z, z_bar=z_bar
    )

    def duplicate(value: torch.Tensor) -> torch.Tensor:
        return torch.repeat_interleave(value, 2, dim=1)

    eta_probability = torch.tensor(
        [1.0 - float(zeta), float(zeta)],
        device=base.component_mass.device,
        dtype=base.component_mass.dtype,
    )[None, None, :]
    mass = (
        base.component_mass[:, :, None] * eta_probability
    ).reshape(base.batch_size, 2 * base.capacity)
    mask = duplicate(base.component_mask) & (mass > 0.0)
    component_id = torch.arange(
        2 * base.capacity,
        device=base.component_mass.device,
        dtype=torch.long,
    )[None, :].expand(base.batch_size, -1)
    eta = torch.tensor(
        [0.0, 1.0],
        device=base.component_mass.device,
        dtype=base.b.dtype,
    )[None, None, :].expand(base.batch_size, base.capacity, 2).reshape(
        base.batch_size, 2 * base.capacity
    )
    return PreDecisionDistribution(
        component_mass=mass,
        b=duplicate(base.b),
        capital_per_unit=duplicate(base.capital_per_unit),
        z_conditional_mean=duplicate(base.z_conditional_mean),
        z_conditional_std=duplicate(base.z_conditional_std),
        component_mask=mask,
        economic_node_id=duplicate(base.economic_node_id),
        component_id=component_id,
        origin_kind=duplicate(base.origin_kind),
        origin_source_id=duplicate(base.origin_source_id),
        origin_time=duplicate(base.origin_time),
        evaluation_time=duplicate(base.evaluation_time),
        cohort=duplicate(base.cohort),
        birth_time=duplicate(base.birth_time),
        maintenance_eligibility=duplicate(base.maintenance_eligibility),
        eta_state=eta,
    ).validate()


def gaussian_mixture_moments(
    distribution: PreDecisionDistribution,
) -> Dict[str, torch.Tensor]:
    """Mass-weighted z moments without collapsing the mixture."""
    distribution.validate()
    weight = torch.where(
        distribution.component_mask,
        distribution.component_mass,
        torch.zeros_like(distribution.component_mass),
    )
    mass = weight.sum(dim=1)
    mean = (weight * distribution.z_conditional_mean).sum(dim=1) / mass.clamp_min(1e-30)
    second = (
        weight
        * (
            distribution.z_conditional_std.square()
            + distribution.z_conditional_mean.square()
        )
    ).sum(dim=1) / mass.clamp_min(1e-30)
    return {"mass": mass, "mean": mean, "variance": (second - mean.square()).clamp_min(0.0)}


def exact_merge_key(distribution: PreDecisionDistribution) -> Sequence[str]:
    """Document fields required for a lossless component merge.

    Validation code deliberately performs no approximate merge.  Callers may
    sum mass only when *all* listed state, density and provenance fields are
    bit-identical.
    """
    distribution.validate()
    return tuple(field.name for field in fields(distribution) if field.name != "component_mass")

