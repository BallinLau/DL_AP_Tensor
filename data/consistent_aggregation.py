"""Experimental internally-consistent firm-mass aggregation primitives.

The production default remains the historical finite-firm implementation.  The
objects in this module are activated only by the explicit
``continuum_mass_consistent_v1`` or ``hard_realized_reference`` modes.

They deliberately separate numerical storage slots from economic firm mass and
make a root candidate's decisions, resource ledger, and transition payload one
immutable record.  This prevents a selected market-clearing candidate from
being reinterpreted by a different soft/hard policy during commit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Dict, Mapping, Optional

import torch


LEGACY_MIXED_REPLAY = "legacy_mixed_replay"
HARD_REALIZED_REFERENCE = "hard_realized_reference"
CONTINUUM_MASS_CONSISTENT_V1 = "continuum_mass_consistent_v1"
AGGREGATION_MODES = {
    LEGACY_MIXED_REPLAY,
    HARD_REALIZED_REFERENCE,
    CONTINUUM_MASS_CONSISTENT_V1,
}


def tensor_snapshot_hash(
    state: Mapping[str, torch.Tensor],
    *,
    names: tuple[str, ...] = (
        "b", "z", "eta", "i", "K", "firm_mass", "alive", "x",
        "hatcf", "lnkf", "economic_node_id", "economic_time",
    ),
) -> str:
    """Hash the economic inputs that define one candidate evaluation."""
    digest = hashlib.sha256()
    for name in names:
        value = state.get(name)
        if value is None:
            continue
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class NodeDecisionRecord:
    """One selected candidate, including both ledger and transition decisions."""

    snapshot_hash: str
    candidate_h: torch.Tensor
    bar_i: torch.Tensor
    bar_z: torch.Tensor
    bp: torch.Tensor
    accepted_entry: torch.Tensor
    entry_bar_i: torch.Tensor
    entry_bar_z: torch.Tensor
    firm_mass: torch.Tensor
    entry_mass: torch.Tensor
    ledger: Dict[str, torch.Tensor]
    provenance: Dict[str, Any]

    def detached(self) -> "NodeDecisionRecord":
        def clone(value: torch.Tensor) -> torch.Tensor:
            return value.detach().clone()

        return NodeDecisionRecord(
            snapshot_hash=self.snapshot_hash,
            candidate_h=clone(self.candidate_h),
            bar_i=clone(self.bar_i),
            bar_z=clone(self.bar_z),
            bp=clone(self.bp),
            accepted_entry=clone(self.accepted_entry),
            entry_bar_i=clone(self.entry_bar_i),
            entry_bar_z=clone(self.entry_bar_z),
            firm_mass=clone(self.firm_mass),
            entry_mass=clone(self.entry_mass),
            ledger={name: clone(value) for name, value in self.ledger.items()},
            provenance=dict(self.provenance),
        )


@dataclass(frozen=True)
class ProjectionAudit:
    mass_before: torch.Tensor
    mass_after: torch.Tensor
    capital_before: torch.Tensor
    capital_after: torch.Tensor
    debt_before: torch.Tensor
    debt_after: torch.Tensor
    output_proxy_before: torch.Tensor
    output_proxy_after: torch.Tensor

    def maximum_relative_error(self) -> torch.Tensor:
        pairs = (
            (self.mass_before, self.mass_after),
            (self.capital_before, self.capital_after),
            (self.debt_before, self.debt_after),
            (self.output_proxy_before, self.output_proxy_after),
        )
        errors = []
        for before, after in pairs:
            scale = before.abs().clamp_min(1.0e-12)
            errors.append(((after - before).abs() / scale).amax())
        return torch.stack(errors).amax()


def equal_probability_normal_quadrature(
    points: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic normal quadrature with no discarded probability mass.

    Each node represents one equal-probability interval.  The midpoint
    quantile is the representative innovation and every interval keeps its
    exact probability weight.  This is a distribution discretization, not an
    added shock or kernel smoothing.
    """
    if int(points) < 2:
        raise ValueError("normal quadrature needs at least two points")
    probability = (torch.arange(points, device=device, dtype=dtype) + 0.5) / points
    nodes = torch.erfinv(2.0 * probability - 1.0) * (2.0 ** 0.5)
    weights = torch.full((points,), 1.0 / points, device=device, dtype=dtype)
    return nodes, weights


def uniform_midpoint_quadrature(
    points: int,
    lower: float,
    upper: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(points) < 1 or not float(lower) < float(upper):
        raise ValueError("invalid uniform quadrature specification")
    probability = (torch.arange(points, device=device, dtype=dtype) + 0.5) / points
    nodes = float(lower) + (float(upper) - float(lower)) * probability
    weights = torch.full((points,), 1.0 / points, device=device, dtype=dtype)
    return nodes, weights


def integrate_hard_investment_region(
    value_gap: torch.Tensor,
    investment_cost: torch.Tensor,
    weights: torch.Tensor,
    capital_per_unit: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Integrate the original hard ``PI >= P0`` region over cost nodes."""
    if value_gap.shape != investment_cost.shape:
        raise ValueError("value gap and investment cost must have matching shape")
    if weights.ndim != 1 or weights.numel() != value_gap.shape[-1]:
        raise ValueError("weights must index the final cost dimension")
    action = value_gap >= 0.0
    weight = weights.reshape(*([1] * (value_gap.ndim - 1)), -1)
    probability = (action.to(value_gap.dtype) * weight).sum(dim=-1)
    expected_project_cost = (
        action.to(value_gap.dtype) * investment_cost * weight
    ).sum(dim=-1) * capital_per_unit
    return {
        "hard_action": action,
        "investment_probability": probability,
        "expected_project_cost": expected_project_cost,
    }


def continuum_entry_reference(
    capital_per_unit: torch.Tensor,
    firm_mass: torch.Tensor,
    alive: torch.Tensor,
    *,
    entry_capital_ratio: float,
    entry_size_ratio: float,
) -> Dict[str, torch.Tensor]:
    """Compute entry scale from economic mass, never from numerical slots."""
    if capital_per_unit.shape != firm_mass.shape or alive.shape != firm_mass.shape:
        raise ValueError("continuum entry tensors must have matching shapes")
    active_mass = torch.where(alive, firm_mass, torch.zeros_like(firm_mass))
    total_mass = active_mass.sum(dim=1)
    total_capital = (active_mass * capital_per_unit).sum(dim=1)
    mean_capital = torch.where(
        total_mass > 0.0,
        total_capital / total_mass.clamp_min(1.0e-12),
        torch.zeros_like(total_capital),
    )
    capital_per_entrant = float(entry_size_ratio) * mean_capital
    potential_capital = float(entry_capital_ratio) * total_capital
    potential_mass = torch.where(
        capital_per_entrant > 0.0,
        potential_capital / capital_per_entrant.clamp_min(1.0e-12),
        torch.zeros_like(potential_capital),
    )
    return {
        "reference_mass": total_mass,
        "reference_capital": total_capital,
        "mean_capital_per_unit": mean_capital,
        "capital_per_entrant": capital_per_entrant,
        "potential_entry_mass": potential_mass,
        "potential_entry_capital": potential_capital,
    }


def aggregate_mass_ledger(
    *,
    mass: torch.Tensor,
    capital_per_unit: torch.Tensor,
    b: torch.Tensor,
    z: torch.Tensor,
    x: torch.Tensor,
    project_investment: torch.Tensor,
    maintenance: torch.Tensor,
    default: torch.Tensor,
    phi: float,
) -> Dict[str, torch.Tensor]:
    """Hard realized resource account with the same action in every term."""
    if any(value.shape != mass.shape for value in (
        capital_per_unit, b, z, project_investment, maintenance, default
    )):
        raise ValueError("firm ledger tensors must have matching shapes")
    while x.ndim < mass.ndim:
        x = x.unsqueeze(-1)
    productivity = torch.exp(x + z)
    output = productivity * capital_per_unit
    default_float = default.to(mass.dtype)
    liquidation_investment = -default_float * capital_per_unit
    bankruptcy_loss = (
        (1.0 - float(phi))
        * (1.0 + productivity)
        * capital_per_unit
        * default_float
    )
    investment_total = project_investment + maintenance + liquidation_investment
    consumption = output - investment_total - bankruptcy_loss
    weighted = lambda value: (mass * value).sum(dim=-1)
    return {
        "Y": weighted(output),
        "I_project": weighted(project_investment),
        "I_maintenance": weighted(maintenance),
        "I_liquidation": weighted(liquidation_investment),
        "I_oper": weighted(investment_total),
        "Liquidation": weighted(bankruptcy_loss),
        "C_oper": weighted(consumption),
        "K_total": weighted(capital_per_unit),
        "firm_mass": mass.sum(dim=-1),
        "aggregate_debt": weighted(b * capital_per_unit),
        "default_mass": weighted(default_float),
    }


def assert_candidate_commit_parity(
    candidate: NodeDecisionRecord,
    committed: Mapping[str, torch.Tensor],
    *,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-10,
) -> None:
    """Fail closed if a commit does not preserve the selected ledger."""
    for name, expected in candidate.ledger.items():
        actual = committed.get(name)
        if actual is None:
            raise RuntimeError(f"committed ledger is missing {name}")
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def attach_default_mass(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Add mass=1 to a finite-firm snapshot without changing its economics."""
    if "firm_mass" not in state:
        state["firm_mass"] = state["alive"].to(state["K"].dtype)
    return state

