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


def conservative_project_transport_atoms(
    atoms: PostDecisionTransportAtoms,
    *,
    b_points: int,
    capital_points: int,
    output_points: int,
) -> tuple[PostDecisionTransportAtoms, Dict[str, torch.Tensor | int | str]]:
    """Project selected source atoms by nonnegative trilinear interpolation.

    Coordinates are ``b``, capital per unit ``k`` and output productivity
    ``u=exp(z_source)``.  Tensor-product linear interpolation preserves, atom
    by atom, mass, ``mass*k``, ``mass*b*k`` and ``mass*u*k``.  Projection is
    performed only after a policy/root is committed; it is never part of a
    candidate h evaluation.

    Discrete origin, cohort, birth and investment-region attributes are kept
    in separate grids.  This is a declared approximation to the remaining
    within-cell joint distribution, not an exact merge.
    """
    atoms.validate()
    if min(int(b_points), int(capital_points), int(output_points)) < 2:
        raise ValueError("conservative projection grids need at least two points")
    device = atoms.transport_weight.device
    dtype = atoms.b_next.dtype
    batch_size = int(atoms.transport_weight.shape[0])
    output_by_name: Dict[str, list[torch.Tensor]] = {
        field.name: [] for field in fields(atoms) if field.name != "atom_mask"
    }
    output_path: list[torch.Tensor] = []
    input_count = int(atoms.atom_mask.sum())

    def grid_and_coordinates(value: torch.Tensor, points: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lower = value.min()
        upper = value.max()
        if bool((upper - lower).abs() <= torch.finfo(value.dtype).eps * 16):
            grid = torch.linspace(
                lower - 0.5,
                lower + 0.5,
                points,
                device=device,
                dtype=value.dtype,
            )
            low = torch.full_like(value, points // 2 - 1, dtype=torch.long)
            alpha = torch.full_like(value, 0.5)
            # Recenter the bracketing pair exactly on the constant value.
            grid[points // 2 - 1] = lower
            grid[points // 2] = lower
            alpha.zero_()
            return grid, low, alpha
        grid = torch.linspace(lower, upper, points, device=device, dtype=value.dtype)
        low = torch.searchsorted(grid, value, right=True) - 1
        low = low.clamp(0, points - 2)
        left = grid[low]
        right = grid[low + 1]
        alpha = (value - left) / (right - left).clamp_min(torch.finfo(value.dtype).tiny)
        return grid, low, alpha.clamp(0.0, 1.0)

    for path in range(batch_size):
        active = atoms.atom_mask[path] & (atoms.transport_weight[path] > 0.0)
        position = torch.nonzero(active, as_tuple=False).reshape(-1)
        if position.numel() == 0:
            continue
        discrete = torch.stack(
            (
                atoms.origin_kind[path, position],
                atoms.cohort[path, position],
                atoms.birth_time[path, position],
                atoms.investment_region_id[path, position],
            ),
            dim=1,
        )
        unique_group, inverse = torch.unique(discrete, dim=0, return_inverse=True)
        for group_index in range(int(unique_group.shape[0])):
            group_position = position[inverse == group_index]
            weight = atoms.transport_weight[path, group_position]
            b = atoms.b_next[path, group_position]
            capital = atoms.capital_next_per_unit[path, group_position]
            productivity = torch.exp(atoms.source_z_current[path, group_position])
            b_grid, b_low, b_alpha = grid_and_coordinates(b, int(b_points))
            k_grid, k_low, k_alpha = grid_and_coordinates(capital, int(capital_points))
            u_grid, u_low, u_alpha = grid_and_coordinates(productivity, int(output_points))
            corner = torch.arange(8, device=device, dtype=torch.long)
            choose_b = ((corner >> 2) & 1).to(torch.bool)
            choose_k = ((corner >> 1) & 1).to(torch.bool)
            choose_u = (corner & 1).to(torch.bool)
            ib = b_low[:, None] + choose_b.to(torch.long)[None, :]
            ik = k_low[:, None] + choose_k.to(torch.long)[None, :]
            iu = u_low[:, None] + choose_u.to(torch.long)[None, :]
            wb = torch.where(choose_b[None, :], b_alpha[:, None], 1.0 - b_alpha[:, None])
            wk = torch.where(choose_k[None, :], k_alpha[:, None], 1.0 - k_alpha[:, None])
            wu = torch.where(choose_u[None, :], u_alpha[:, None], 1.0 - u_alpha[:, None])
            corner_weight = weight[:, None] * wb.to(weight.dtype) * wk.to(weight.dtype) * wu.to(weight.dtype)
            cell = (
                ib * (int(capital_points) * int(output_points))
                + ik * int(output_points)
                + iu
            ).reshape(-1)
            cell_weight = torch.zeros(
                int(b_points) * int(capital_points) * int(output_points),
                device=device,
                dtype=weight.dtype,
            )
            cell_weight.index_add_(0, cell, corner_weight.reshape(-1))
            occupied = torch.nonzero(cell_weight > 0.0, as_tuple=False).reshape(-1)
            if occupied.numel() == 0:
                continue
            ib_out = torch.div(
                occupied,
                int(capital_points) * int(output_points),
                rounding_mode="floor",
            )
            remainder = occupied % (int(capital_points) * int(output_points))
            ik_out = torch.div(remainder, int(output_points), rounding_mode="floor")
            iu_out = remainder % int(output_points)
            count = int(occupied.numel())
            exemplar = group_position[0]
            output_path.append(torch.full((count,), path, device=device, dtype=torch.long))
            output_by_name["transport_weight"].append(cell_weight[occupied])
            output_by_name["source_z_current"].append(torch.log(u_grid[iu_out].clamp_min(1e-30)))
            output_by_name["b_next"].append(b_grid[ib_out])
            output_by_name["capital_next_per_unit"].append(k_grid[ik_out])
            output_by_name["source_component_id"].append(occupied.to(torch.long))
            for name in (
                "survival_region_id",
                "investment_region_id",
                "policy_branch_id",
                "economic_node_id",
                "source_time",
                "destination_time",
                "cohort",
                "birth_time",
                "maintenance_eligibility",
                "origin_kind",
                "origin_source_id",
            ):
                value = getattr(atoms, name)[path, exemplar]
                output_by_name[name].append(value.expand(count).clone())

    if output_path:
        flat_path = torch.cat(output_path)
        flat = {name: torch.cat(parts) for name, parts in output_by_name.items()}
    else:
        flat_path = torch.empty(0, device=device, dtype=torch.long)
        flat = {}
        for field in fields(atoms):
            if field.name == "atom_mask":
                continue
            source = getattr(atoms, field.name)
            flat[field.name] = source.new_empty((0,))
    counts = torch.bincount(flat_path, minlength=batch_size)
    capacity = max(int(counts.max().item()) if counts.numel() else 0, 1)
    order = torch.argsort(
        flat_path * (flat_path.numel() + 1)
        + torch.arange(flat_path.numel(), device=device)
    ) if flat_path.numel() else flat_path
    sorted_path = flat_path[order] if flat_path.numel() else flat_path
    if flat_path.numel():
        change = torch.ones_like(sorted_path, dtype=torch.bool)
        change[1:] = sorted_path[1:] != sorted_path[:-1]
        starts = torch.nonzero(change, as_tuple=False).reshape(-1)
        lengths = torch.diff(
            torch.cat((starts, torch.tensor([flat_path.numel()], device=device)))
        )
        rank = torch.arange(flat_path.numel(), device=device) - torch.repeat_interleave(starts, lengths)
    else:
        rank = flat_path
    packed: Dict[str, torch.Tensor] = {}
    for name, value in flat.items():
        fill_value: float | int | bool
        if name == "capital_next_per_unit":
            fill_value = 1.0
        elif value.dtype == torch.bool:
            fill_value = False
        elif value.dtype.is_floating_point:
            fill_value = 0.0
        else:
            fill_value = -1
        output = torch.full(
            (batch_size, capacity), fill_value, device=device, dtype=value.dtype
        )
        if flat_path.numel():
            output[sorted_path, rank] = value[order]
        packed[name] = output
    packed["atom_mask"] = (
        torch.arange(capacity, device=device)[None, :] < counts[:, None]
    )
    projected = PostDecisionTransportAtoms(**packed).validate()
    mass_before = atoms.total_mass()
    capital_before = atoms.total_capital()
    debt_before = atoms.aggregate_debt()
    output_before = torch.where(
        atoms.atom_mask,
        atoms.transport_weight
        * atoms.capital_next_per_unit
        * torch.exp(atoms.source_z_current),
        torch.zeros_like(atoms.transport_weight),
    ).sum(dim=1)
    mass_after = projected.total_mass()
    capital_after = projected.total_capital()
    debt_after = projected.aggregate_debt()
    output_after = torch.where(
        projected.atom_mask,
        projected.transport_weight
        * projected.capital_next_per_unit
        * torch.exp(projected.source_z_current),
        torch.zeros_like(projected.transport_weight),
    ).sum(dim=1)
    scales = torch.stack(
        (
            mass_before.abs().clamp_min(1e-30),
            capital_before.abs().clamp_min(1e-30),
            debt_before.abs().clamp_min(1e-30),
            output_before.abs().clamp_min(1e-30),
        ),
        dim=1,
    )
    error = torch.stack(
        (
            (mass_after - mass_before).abs(),
            (capital_after - capital_before).abs(),
            (debt_after - debt_before).abs(),
            (output_after - output_before).abs(),
        ),
        dim=1,
    ) / scales
    audit: Dict[str, torch.Tensor | int | str] = {
        "mode": "source_atom_conservative_projection_v1",
        "input_atom_count": input_count,
        "output_atom_count": int(projected.atom_mask.sum()),
        "mass_before": mass_before,
        "mass_after": mass_after,
        "capital_before": capital_before,
        "capital_after": capital_after,
        "debt_before": debt_before,
        "debt_after": debt_after,
        "output_proxy_before": output_before,
        "output_proxy_after": output_after,
        "max_balance_relative_error": error.max(),
    }
    return projected, audit

