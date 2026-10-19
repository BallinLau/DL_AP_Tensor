"""Boundary-aware market-clearing operator for frozen Firm models.

The class subclasses the already-audited coarse-to-fine h root controller and
changes only candidate evaluation.  A candidate is evaluated by locating all
visible Phat(z,h)=0 and PI-P0=0 boundaries in batched probability coordinates,
then integrating resources separately on the resulting policy regions.

This experimental operator never treats a conditional-density component mean
as a realized firm and never advances the distribution while testing h.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Dict

import torch

from .batched_policy_boundaries import (
    interval_bounds_from_boundaries,
    locate_all_policy_boundaries,
)
from .boundary_aware_distribution import (
    BoundaryAwareDecisionRecord,
    PostDecisionTransportAtoms,
    gauss_legendre_probability_nodes,
    normal_quantile,
    predecision_fingerprint_from_state,
)
from .entry import EntryCandidates
from .market_clearing import MarketClearingConfig, MarketClearingController, MarketClearingError
from utils.firm_transition import apply_refinancing_policy


@dataclass(frozen=True)
class BoundaryIntegrationConfig:
    z_scan_points: int = 65
    z_quadrature_points: int = 8
    i_scan_points: int = 33
    i_quadrature_points: int = 8
    boundary_refinement_rounds: int = 16
    scan_tail_probability: float = 1.0e-8
    boundary_coordinate_tolerance: float = 1.0e-9
    boundary_value_tolerance: float = 1.0e-8

    def validate(self) -> None:
        if self.z_scan_points < 5 or self.i_scan_points < 5:
            raise ValueError("boundary scans require at least five points")
        if self.z_quadrature_points < 2 or self.i_quadrature_points < 2:
            raise ValueError("boundary quadrature requires at least two points")
        if self.boundary_refinement_rounds < 1:
            raise ValueError("boundary refinement rounds must be positive")
        if not 0.0 < self.scan_tail_probability < 0.5:
            raise ValueError("scan_tail_probability must be in (0,0.5)")


def _scatter_sum(
    values: torch.Tensor,
    index: torch.Tensor,
    size: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    output = torch.zeros(size, device=values.device, dtype=dtype)
    output.index_add_(0, index, values.to(dtype))
    return output


class BoundaryAwareMarketClearingController(MarketClearingController):
    """Reuse h root search while replacing representative-firm evaluation."""

    def __init__(
        self,
        config: MarketClearingConfig,
        integration: BoundaryIntegrationConfig,
    ) -> None:
        super().__init__(config)
        integration.validate()
        self.integration = integration
        self.boundary_search_seconds = 0.0
        self.resource_integration_seconds = 0.0
        self.default_boundary_count = 0
        self.investment_boundary_count = 0
        self._validated_boundary_records: set[tuple[str, bytes]] = set()
        self.tail_probability_mass_scanned_outside = 2.0 * float(
            integration.scan_tail_probability
        )

    @staticmethod
    def _required_distribution_fields(state: Dict[str, torch.Tensor]) -> None:
        required = (
            "component_mass",
            "z_conditional_mean",
            "z_conditional_std",
            "component_mask",
            "b",
            "K",
            "eta",
        )
        missing = [name for name in required if name not in state]
        if missing:
            raise MarketClearingError(
                "boundary-aware evaluation is missing distribution fields: "
                + ", ".join(missing)
            )

    def _states_for_coordinates(
        self,
        *,
        b: torch.Tensor,
        z: torch.Tensor,
        eta: torch.Tensor,
        investment_cost: torch.Tensor,
        x: torch.Tensor,
        h: torch.Tensor,
        lnkf: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack((b, z, eta, investment_cost, x, h, lnkf), dim=-1)

    @torch.inference_mode()
    def evaluate(
        self,
        sim: Any,
        state: Dict[str, torch.Tensor],
        candidates: EntryCandidates,
        h: torch.Tensor,
        *,
        detail: bool = False,
    ) -> dict[str, torch.Tensor]:
        self._required_distribution_fields(state)
        model = sim.models.get("policy_value")
        if model is None:
            raise MarketClearingError("boundary-aware clearing requires policy_value")
        if h.ndim != 2 or h.shape[0] != state["b"].shape[0]:
            raise ValueError("market clearing h must have shape [B,H]")
        device = state["b"].device
        dtype = state["b"].dtype
        h = h.to(device=device, dtype=dtype)
        B, H = h.shape
        # Dense boundary scans combined with all root-search h candidates can
        # otherwise materialize an investment scan tensor above PyTorch's
        # INT_MAX indexing limit.  Chunk only the candidate axis; every chunk
        # still evaluates all active components and policy regions in batched
        # GPU calls.  Accepted-root detail evaluation already has H=1.
        if (
            not detail
            and H > 1
            and self.integration.z_scan_points >= 257
        ):
            pieces = [
                self.evaluate(sim, state, candidates, h[:, index : index + 1])
                for index in range(H)
            ]
            names = pieces[0].keys()
            return {
                name: torch.cat([piece[name] for piece in pieces], dim=1)
                for name in names
            }
        L = int(state["b"].shape[1])
        if candidates.mask.shape[0] != B:
            raise ValueError("entry candidates do not match distribution batch")
        self.operator_evaluations += B * H

        component_mask = state["component_mask"].bool()
        if component_mask.shape != (B, L):
            raise ValueError("component_mask must match b/K shape")
        task_path = torch.arange(B, device=device)[:, None, None].expand(B, H, L).reshape(-1)
        task_h_column = torch.arange(H, device=device)[None, :, None].expand(B, H, L).reshape(-1)
        task_component = torch.arange(L, device=device)[None, None, :].expand(B, H, L).reshape(-1)
        # A path can have zero surviving incumbents yet retain economically
        # valid potential entry mass inherited from its parent reference.  One
        # zero-mass storage slot is evaluated only to keep tensor ranks stable;
        # it is never counted, exported, or transported as an economic atom.
        evaluation_component_mask = component_mask.clone()
        entry_only_path = (~component_mask.any(dim=1)) & candidates.mask.any(dim=1)
        if bool(entry_only_path.any()):
            evaluation_component_mask[entry_only_path, 0] = True
        task_valid = evaluation_component_mask[:, None, :].expand(B, H, L).reshape(-1)
        task_position = torch.nonzero(task_valid, as_tuple=False).reshape(-1)
        if task_position.numel() == 0:
            raise MarketClearingError(
                "boundary-aware clearing received neither incumbent mass nor potential entry mass"
            )
        path = task_path[task_position]
        h_column = task_h_column[task_position]
        component = task_component[task_position]
        T = int(task_position.numel())
        parent_h_index = path * H + h_column

        b_task = state["b"][path, component]
        k_task = state["K"][path, component]
        eta_task = state["eta"][path, component]
        mass_task = state["component_mass"][path, component]
        mu_task = state["z_conditional_mean"][path, component]
        sigma_task = state["z_conditional_std"][path, component]
        x_task = state["x"][path]
        h_task = h[path, h_column]
        lnk_task = state["lnkf"][path]
        maintenance_task = state.get(
            "maintenance_eligibility", component_mask
        )[path, component].bool()

        boundary_started = time.perf_counter()
        tail = float(self.integration.scan_tail_probability)
        z_scan_u = torch.linspace(
            tail,
            1.0 - tail,
            self.integration.z_scan_points,
            device=device,
            dtype=dtype,
        )[None, :].expand(T, -1)

        def default_value(task_index: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
            z = normal_quantile(
                probability,
                mu_task[task_index],
                sigma_task[task_index],
            )
            rows = self._states_for_coordinates(
                b=b_task[task_index],
                z=z,
                eta=eta_task[task_index],
                investment_cost=torch.zeros_like(z),
                x=x_task[task_index],
                h=h_task[task_index],
                lnkf=lnk_task[task_index],
            )
            return self._firm_policy(model, rows)["Phat"]

        z_scan_task = torch.arange(T, device=device)[:, None].expand_as(z_scan_u)
        z_scan_value = default_value(z_scan_task.reshape(-1), z_scan_u.reshape(-1)).reshape_as(z_scan_u)
        default_boundaries = locate_all_policy_boundaries(
            z_scan_u,
            z_scan_value,
            default_value,
            refinement_rounds=self.integration.boundary_refinement_rounds,
            coordinate_tolerance=self.integration.boundary_coordinate_tolerance,
            zero_tolerance=self.integration.boundary_value_tolerance,
        )
        self.default_boundary_count += int(default_boundaries.boundary_mask.sum())
        z_left, z_right, z_interval_mask = interval_bounds_from_boundaries(
            default_boundaries.boundary_coordinate,
            default_boundaries.boundary_mask,
            lower=0.0,
            upper=1.0,
        )
        z_u, z_probability_weight = gauss_legendre_probability_nodes(
            z_left, z_right, self.integration.z_quadrature_points
        )
        z_valid = z_interval_mask[..., None].expand_as(z_u)
        z_task_index = torch.arange(T, device=device)[:, None, None].expand_as(z_u)[z_valid]
        z_u_flat = z_u[z_valid]
        z_probability_weight_flat = z_probability_weight[z_valid]
        z_value = normal_quantile(
            z_u_flat, mu_task[z_task_index], sigma_task[z_task_index]
        )
        z_phat = default_value(z_task_index, z_u_flat)
        z_survival = z_phat > 0.0
        z_economic_weight = mass_task[z_task_index].to(torch.float64) * z_probability_weight_flat.to(torch.float64)
        self.boundary_search_seconds += time.perf_counter() - boundary_started

        # Default regions need no cost integration because investment is not
        # executed there.  Survivor regions are split at every visible
        # PI-P0 boundary in the original uniform cost probability coordinate.
        default_z_position = torch.nonzero(~z_survival, as_tuple=False).reshape(-1)
        survivor_z_position = torch.nonzero(z_survival, as_tuple=False).reshape(-1)
        investment_boundary_probability = torch.empty(
            (0, 0), device=device, dtype=dtype
        )
        investment_boundary_mask = torch.empty(
            (0, 0), device=device, dtype=torch.bool
        )
        investment_boundary_residual = torch.empty(
            (0, 0), device=device, dtype=dtype
        )
        investment_boundary_source_task = torch.empty(
            0, device=device, dtype=torch.long
        )
        investment_boundary_source_z = torch.empty(0, device=device, dtype=dtype)
        if survivor_z_position.numel():
            S = int(survivor_z_position.numel())
            survivor_task = z_task_index[survivor_z_position]
            survivor_z = z_value[survivor_z_position]
            i_scan_v = torch.linspace(
                0.0,
                1.0,
                self.integration.i_scan_points,
                device=device,
                dtype=dtype,
            )[None, :].expand(S, -1)
            i_max = float(sim.config.I_THRESHOLD)

            def investment_gap(local_index: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
                source_task = survivor_task[local_index]
                rows = self._states_for_coordinates(
                    b=b_task[source_task],
                    z=survivor_z[local_index],
                    eta=eta_task[source_task],
                    investment_cost=float(i_max) * probability,
                    x=x_task[source_task],
                    h=h_task[source_task],
                    lnkf=lnk_task[source_task],
                )
                p0, pi = self._entry_values(model, rows)
                return pi - p0

            i_scan_task = torch.arange(S, device=device)[:, None].expand_as(i_scan_v)
            i_scan_value = investment_gap(
                i_scan_task.reshape(-1), i_scan_v.reshape(-1)
            ).reshape_as(i_scan_v)
            investment_boundaries = locate_all_policy_boundaries(
                i_scan_v,
                i_scan_value,
                investment_gap,
                refinement_rounds=self.integration.boundary_refinement_rounds,
                coordinate_tolerance=self.integration.boundary_coordinate_tolerance,
                zero_tolerance=self.integration.boundary_value_tolerance,
            )
            investment_boundary_probability = investment_boundaries.boundary_coordinate
            investment_boundary_mask = investment_boundaries.boundary_mask
            investment_boundary_residual = investment_boundaries.residual
            investment_boundary_source_task = survivor_task
            investment_boundary_source_z = survivor_z
            self.investment_boundary_count += int(investment_boundaries.boundary_mask.sum())
            i_left, i_right, i_interval_mask = interval_bounds_from_boundaries(
                investment_boundaries.boundary_coordinate,
                investment_boundaries.boundary_mask,
                lower=0.0,
                upper=1.0,
            )
            i_v, i_probability_weight = gauss_legendre_probability_nodes(
                i_left, i_right, self.integration.i_quadrature_points
            )
            i_valid = i_interval_mask[..., None].expand_as(i_v)
            i_local_index = torch.arange(S, device=device)[:, None, None].expand_as(i_v)[i_valid]
            i_v_flat = i_v[i_valid]
            i_probability_weight_flat = i_probability_weight[i_valid]
            i_gap = investment_gap(i_local_index, i_v_flat)
            survivor_atom_z_position = survivor_z_position[i_local_index]
            survivor_atom_task = z_task_index[survivor_atom_z_position]
            survivor_atom_z = z_value[survivor_atom_z_position]
            survivor_atom_i = i_v_flat * float(i_max)
            survivor_atom_weight = (
                z_economic_weight[survivor_atom_z_position]
                * i_probability_weight_flat.to(torch.float64)
            )
            survivor_atom_invest = i_gap >= 0.0
        else:
            survivor_atom_task = torch.empty(0, device=device, dtype=torch.long)
            survivor_atom_z = torch.empty(0, device=device, dtype=dtype)
            survivor_atom_i = torch.empty(0, device=device, dtype=dtype)
            survivor_atom_weight = torch.empty(0, device=device, dtype=torch.float64)
            survivor_atom_invest = torch.empty(0, device=device, dtype=torch.bool)

        default_atom_task = z_task_index[default_z_position]
        default_atom_z = z_value[default_z_position]
        default_atom_weight = z_economic_weight[default_z_position]
        atom_task = torch.cat((default_atom_task, survivor_atom_task))
        atom_z = torch.cat((default_atom_z, survivor_atom_z))
        atom_i = torch.cat((torch.zeros_like(default_atom_z), survivor_atom_i))
        atom_weight = torch.cat((default_atom_weight, survivor_atom_weight))
        atom_default = torch.cat(
            (
                torch.ones(default_atom_task.numel(), device=device, dtype=torch.bool),
                torch.zeros(survivor_atom_task.numel(), device=device, dtype=torch.bool),
            )
        )
        atom_invest = torch.cat(
            (
                torch.zeros(default_atom_task.numel(), device=device, dtype=torch.bool),
                survivor_atom_invest,
            )
        )
        atom_parent_h = parent_h_index[atom_task]
        output_size = B * H

        # Entry candidates are used only to recover the production reference
        # scale (potential capital and capital per entrant).  Their old sampled
        # z/cost coordinates are deliberately ignored.  Entry is re-integrated
        # over the model-defined stationary z density and original uniform
        # cost distribution, split at the actual PI-P0 and Phat boundaries.
        entry_atom_parent_h = torch.empty(0, device=device, dtype=torch.long)
        entry_atom_weight = torch.empty(0, device=device, dtype=torch.float64)
        entry_atom_z = torch.empty(0, device=device, dtype=dtype)
        entry_atom_i = torch.empty(0, device=device, dtype=dtype)
        entry_atom_default = torch.empty(0, device=device, dtype=torch.bool)
        entry_atom_state = torch.empty((0, 7), device=device, dtype=dtype)
        if bool(candidates.mask.any()):
            candidate_count = candidates.mask.sum(dim=1)
            k_entry = torch.where(
                candidate_count > 0,
                (candidates.K_birth * candidates.mask.to(dtype)).sum(dim=1)
                / candidate_count.to(dtype).clamp_min(1.0),
                torch.zeros(B, device=device, dtype=dtype),
            )
            potential_mass = torch.where(
                k_entry > 0.0,
                candidates.potential_capital_nominal
                / k_entry.clamp_min(torch.finfo(dtype).tiny),
                torch.zeros_like(k_entry),
            )
            entry_task_path = torch.arange(B, device=device)[:, None].expand(B, H).reshape(-1)
            entry_task_h_column = torch.arange(H, device=device)[None, :].expand(B, H).reshape(-1)
            entry_valid = potential_mass[entry_task_path] > 0.0
            entry_task_position = torch.nonzero(entry_valid, as_tuple=False).reshape(-1)
            if entry_task_position.numel():
                entry_path = entry_task_path[entry_task_position]
                entry_h_column = entry_task_h_column[entry_task_position]
                ET = int(entry_task_position.numel())
                entry_h = h[entry_path, entry_h_column]
                entry_x = state["x"][entry_path]
                entry_lnk = state["lnkf"][entry_path]
                entry_k = k_entry[entry_path]
                entry_mass_total = potential_mass[entry_path]
                entry_mu = torch.full_like(entry_h, float(sim.config.ZBAR))
                entry_sigma = torch.full_like(
                    entry_h,
                    float(sim.config.SIGMA_Z)
                    / math.sqrt(max(1.0 - float(sim.config.RHO_Z) ** 2, 1.0e-30)),
                )

                def entry_default_value(local_index: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
                    z = normal_quantile(
                        probability, entry_mu[local_index], entry_sigma[local_index]
                    )
                    rows = self._states_for_coordinates(
                        b=torch.zeros_like(z),
                        z=z,
                        eta=torch.zeros_like(z),
                        investment_cost=torch.zeros_like(z),
                        x=entry_x[local_index],
                        h=entry_h[local_index],
                        lnkf=entry_lnk[local_index],
                    )
                    return self._firm_policy(model, rows)["Phat"]

                entry_scan_u = z_scan_u.new_empty((ET, self.integration.z_scan_points))
                entry_scan_u.copy_(
                    torch.linspace(
                        tail,
                        1.0 - tail,
                        self.integration.z_scan_points,
                        device=device,
                        dtype=dtype,
                    )[None, :]
                )
                entry_scan_u = entry_scan_u.expand(ET, -1)
                entry_scan_local = torch.arange(ET, device=device)[:, None].expand_as(entry_scan_u)
                entry_scan_phat = entry_default_value(
                    entry_scan_local.reshape(-1), entry_scan_u.reshape(-1)
                ).reshape_as(entry_scan_u)
                entry_default_boundaries = locate_all_policy_boundaries(
                    entry_scan_u,
                    entry_scan_phat,
                    entry_default_value,
                    refinement_rounds=self.integration.boundary_refinement_rounds,
                    coordinate_tolerance=self.integration.boundary_coordinate_tolerance,
                    zero_tolerance=self.integration.boundary_value_tolerance,
                )
                self.default_boundary_count += int(entry_default_boundaries.boundary_mask.sum())
                entry_z_left, entry_z_right, entry_z_interval_mask = interval_bounds_from_boundaries(
                    entry_default_boundaries.boundary_coordinate,
                    entry_default_boundaries.boundary_mask,
                    lower=0.0,
                    upper=1.0,
                )
                entry_z_u, entry_z_probability_weight = gauss_legendre_probability_nodes(
                    entry_z_left, entry_z_right, self.integration.z_quadrature_points
                )
                entry_z_valid = entry_z_interval_mask[..., None].expand_as(entry_z_u)
                entry_z_local = torch.arange(ET, device=device)[:, None, None].expand_as(entry_z_u)[entry_z_valid]
                entry_z_u_flat = entry_z_u[entry_z_valid]
                entry_z_weight = entry_z_probability_weight[entry_z_valid]
                entry_z = normal_quantile(
                    entry_z_u_flat,
                    entry_mu[entry_z_local],
                    entry_sigma[entry_z_local],
                )
                entry_phat = entry_default_value(entry_z_local, entry_z_u_flat)
                EZ = int(entry_z.numel())
                entry_i_scan_v = torch.linspace(
                    0.0,
                    1.0,
                    self.integration.i_scan_points,
                    device=device,
                    dtype=dtype,
                )[None, :].expand(EZ, -1)
                entry_i_max = float(sim.effective_entry_cost_max)

                def entry_gap(z_index: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
                    local = entry_z_local[z_index]
                    rows = self._states_for_coordinates(
                        b=torch.zeros_like(probability),
                        z=entry_z[z_index],
                        eta=torch.zeros_like(probability),
                        investment_cost=float(entry_i_max) * probability,
                        x=entry_x[local],
                        h=entry_h[local],
                        lnkf=entry_lnk[local],
                    )
                    p0, pi = self._entry_values(model, rows)
                    return pi - p0

                entry_i_scan_local = torch.arange(EZ, device=device)[:, None].expand_as(entry_i_scan_v)
                entry_i_scan_gap = entry_gap(
                    entry_i_scan_local.reshape(-1), entry_i_scan_v.reshape(-1)
                ).reshape_as(entry_i_scan_v)
                entry_investment_boundaries = locate_all_policy_boundaries(
                    entry_i_scan_v,
                    entry_i_scan_gap,
                    entry_gap,
                    refinement_rounds=self.integration.boundary_refinement_rounds,
                    coordinate_tolerance=self.integration.boundary_coordinate_tolerance,
                    zero_tolerance=self.integration.boundary_value_tolerance,
                )
                entry_i_left, entry_i_right, entry_i_interval_mask = interval_bounds_from_boundaries(
                    entry_investment_boundaries.boundary_coordinate,
                    entry_investment_boundaries.boundary_mask,
                    lower=0.0,
                    upper=1.0,
                )
                entry_i_v, entry_i_probability_weight = gauss_legendre_probability_nodes(
                    entry_i_left, entry_i_right, self.integration.i_quadrature_points
                )
                entry_i_valid = entry_i_interval_mask[..., None].expand_as(entry_i_v)
                entry_i_z_index = torch.arange(EZ, device=device)[:, None, None].expand_as(entry_i_v)[entry_i_valid]
                entry_i_v_flat = entry_i_v[entry_i_valid]
                entry_i_probability_weight_flat = entry_i_probability_weight[entry_i_valid]
                accepted = entry_gap(entry_i_z_index, entry_i_v_flat) >= 0.0
                accepted_position = torch.nonzero(accepted, as_tuple=False).reshape(-1)
                if accepted_position.numel():
                    accepted_z_index = entry_i_z_index[accepted_position]
                    accepted_local = entry_z_local[accepted_z_index]
                    entry_atom_parent_h = entry_task_position[accepted_local]
                    entry_atom_weight = (
                        entry_mass_total[accepted_local].to(torch.float64)
                        * entry_z_weight[accepted_z_index].to(torch.float64)
                        * entry_i_probability_weight_flat[accepted_position].to(torch.float64)
                    )
                    entry_atom_z = entry_z[accepted_z_index]
                    entry_atom_i = entry_i_v_flat[accepted_position] * float(entry_i_max)
                    entry_atom_default = entry_phat[accepted_z_index] <= 0.0
                    entry_atom_state = self._states_for_coordinates(
                        b=torch.zeros_like(entry_atom_z),
                        z=entry_atom_z,
                        eta=torch.zeros_like(entry_atom_z),
                        investment_cost=entry_atom_i,
                        x=entry_x[accepted_local],
                        h=entry_h[accepted_local],
                        lnkf=entry_lnk[accepted_local],
                    )

        accounting_started = time.perf_counter()
        productivity = torch.exp(x_task[atom_task].to(torch.float64) + atom_z.to(torch.float64))
        capital = k_task[atom_task].to(torch.float64)
        default_float = atom_default.to(torch.float64)
        invest_float = atom_invest.to(torch.float64)
        maintenance = maintenance_task[atom_task].to(torch.float64)
        y = productivity * capital
        i_maintenance = float(sim.delta) * capital * maintenance
        i_project = invest_float * capital * atom_i.to(torch.float64)
        i_liquidation = -default_float * capital
        i_oper = i_project + i_maintenance + i_liquidation
        liquidation = (
            (1.0 - float(sim.config.PHI))
            * (1.0 + productivity)
            * capital
            * default_float
        )
        c_firm = y - i_oper - liquidation

        def aggregate(value: torch.Tensor) -> torch.Tensor:
            return _scatter_sum(atom_weight * value, atom_parent_h, output_size)

        K_total = aggregate(capital)
        Y_total = aggregate(y)
        I_oper_total = aggregate(i_oper)
        liquidation_total = aggregate(liquidation)
        maintenance_total = aggregate(i_maintenance)
        C_oper = aggregate(c_firm)
        I_entry_total = torch.zeros(output_size, device=device, dtype=torch.float64)
        entry_mass_accepted = torch.zeros_like(I_entry_total)
        entry_default_mass = torch.zeros_like(I_entry_total)
        if entry_atom_parent_h.numel():
            entry_local_path = torch.div(entry_atom_parent_h, H, rounding_mode="floor")
            entry_capital = k_entry[entry_local_path].to(torch.float64)
            entry_productivity = torch.exp(
                state["x"][entry_local_path].to(torch.float64)
                + entry_atom_z.to(torch.float64)
            )
            entry_default_float = entry_atom_default.to(torch.float64)
            entry_y = entry_productivity * entry_capital
            entry_i_liquidation = -entry_default_float * entry_capital
            entry_liquidation = (
                (1.0 - float(sim.config.PHI))
                * (1.0 + entry_productivity)
                * entry_capital
                * entry_default_float
            )
            entry_c_oper = entry_y - entry_i_liquidation - entry_liquidation
            entry_spend = entry_atom_i.to(torch.float64) * entry_capital

            def aggregate_entry(value: torch.Tensor) -> torch.Tensor:
                return _scatter_sum(
                    entry_atom_weight * value, entry_atom_parent_h, output_size
                )

            K_total += aggregate_entry(entry_capital)
            Y_total += aggregate_entry(entry_y)
            I_oper_total += aggregate_entry(entry_i_liquidation)
            liquidation_total += aggregate_entry(entry_liquidation)
            C_oper += aggregate_entry(entry_c_oper)
            I_entry_total += aggregate_entry(entry_spend)
            entry_mass_accepted += _scatter_sum(
                entry_atom_weight, entry_atom_parent_h, output_size
            )
            entry_default_mass += _scatter_sum(
                entry_atom_weight * entry_default_float,
                entry_atom_parent_h,
                output_size,
            )
        C_raw = C_oper - I_entry_total
        G = torch.exp(h.reshape(-1).to(torch.float64)) * K_total - C_raw
        scale = torch.maximum(
            torch.ones_like(G), torch.maximum(C_raw.abs(), K_total.abs())
        )
        finite = torch.isfinite(G) & torch.isfinite(C_raw) & torch.isfinite(K_total)
        feasible = finite & (C_raw > 0.0) & (K_total > 0.0)
        investment_mass = _scatter_sum(
            atom_weight * atom_invest.to(torch.float64), atom_parent_h, output_size
        )
        default_mass = _scatter_sum(
            atom_weight * atom_default.to(torch.float64), atom_parent_h, output_size
        ) + entry_default_mass
        firm_mass = _scatter_sum(atom_weight, atom_parent_h, output_size) + entry_mass_accepted

        component_default_mass = _scatter_sum(
            atom_weight * atom_default.to(torch.float64), atom_task, T
        )
        component_investment_mass = _scatter_sum(
            atom_weight * atom_invest.to(torch.float64), atom_task, T
        )
        component_default_probability = (
            component_default_mass / mass_task.to(torch.float64).clamp_min(1e-30)
        )
        component_investment_probability = (
            component_investment_mass / mass_task.to(torch.float64).clamp_min(1e-30)
        )
        bar_i = torch.zeros(B * H * L, device=device, dtype=dtype)
        bar_z = torch.zeros_like(bar_i)
        bar_i[task_position] = component_investment_probability.to(dtype)
        bar_z[task_position] = component_default_probability.to(dtype)
        bar_i = bar_i.reshape(B, H, L)
        bar_z = bar_z.reshape(B, H, L)
        positive_component = mass_task > 0.0
        investment_count_task = (component_investment_mass > 1e-14) & positive_component
        default_count_task = (component_default_mass > 1e-14) & positive_component
        investment_count = _scatter_sum(
            investment_count_task.to(torch.float64), parent_h_index, output_size
        ).to(torch.long)
        default_count = _scatter_sum(
            default_count_task.to(torch.float64), parent_h_index, output_size
        ).to(torch.long)

        def shaped(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(B, H)

        result: dict[str, torch.Tensor] = {
            "h": h,
            "G": shaped(G),
            "G_normalized": shaped(G / scale),
            "C_raw": shaped(C_raw),
            "C_oper": shaped(C_oper),
            "K_total": shaped(K_total),
            "Y_total": shaped(Y_total),
            "I_oper_total": shaped(I_oper_total),
            "I_entry_total": shaped(I_entry_total),
            "Liquidation_total": shaped(liquidation_total),
            "I_maintenance_total": shaped(maintenance_total),
            "resource_feasible": shaped(feasible),
            "finite": shaped(finite),
            "investment_count": shaped(investment_count),
            "default_count": shaped(default_count),
            "entry_count": shaped((entry_mass_accepted > 1e-14).to(torch.long)),
            "firm_count": shaped(firm_mass),
            "investment_mass": shaped(investment_mass),
            "default_mass": shaped(default_mass),
            "entry_mass": shaped(entry_mass_accepted),
            # Policy-region mass moves continuously with a refined value
            # boundary.  A nonzero change over a finite h bracket is therefore
            # not, by itself, a discrete policy jump.
            "policy_mass_continuous": torch.ones(
                (B, H), device=device, dtype=torch.bool
            ),
        }
        if detail:
            survivor_atom_mask = (~atom_default) & (atom_weight > 0.0)
            detail_state = self._states_for_coordinates(
                b=b_task[atom_task],
                z=atom_z,
                eta=eta_task[atom_task],
                investment_cost=atom_i,
                x=x_task[atom_task],
                h=h_task[atom_task],
                lnkf=lnk_task[atom_task],
            )
            policy_started = time.perf_counter()
            atom_policy = self._firm_policy(model, detail_state)
            self.policy_forward_seconds += time.perf_counter() - policy_started
            result.update(
                {
                    "accepted_entry": (entry_mass_accepted > 1e-14).reshape(B, H, 1),
                    "bar_i_firm": bar_i,
                    "bar_z_firm": bar_z,
                    "bp_firm": torch.zeros((B, H, L), device=device, dtype=dtype),
                    "P0_firm": torch.full((B, H, L), float("nan"), device=device, dtype=dtype),
                    "PI_firm": torch.full((B, H, L), float("nan"), device=device, dtype=dtype),
                    "Phat_firm": (1.0 - 2.0 * bar_z),
                    "economic_mass": state["component_mass"][:, None, :].expand(B, H, L),
                    "entry_candidate_mass": entry_mass_accepted.reshape(B, H, 1).to(dtype),
                    "firm_state": torch.stack(
                        (
                            state["b"][:, None, :].expand(B, H, L),
                            state["z_conditional_mean"][:, None, :].expand(B, H, L),
                            state["eta"][:, None, :].expand(B, H, L),
                            torch.zeros((B, H, L), device=device, dtype=dtype),
                            state["x"][:, None, None].expand(B, H, L),
                            h[:, :, None].expand(B, H, L),
                            state["lnkf"][:, None, None].expand(B, H, L),
                        ),
                        dim=-1,
                    ),
                    "transport_atom_parent_h": atom_parent_h[survivor_atom_mask],
                    "transport_atom_source_task": atom_task[survivor_atom_mask],
                    "transport_atom_weight": atom_weight[survivor_atom_mask],
                    "transport_atom_source_z": atom_z[survivor_atom_mask],
                    "transport_atom_i": atom_i[survivor_atom_mask],
                    "transport_atom_invest": atom_invest[survivor_atom_mask],
                    "transport_atom_bp_head": atom_policy["bp"][survivor_atom_mask],
                    "transport_atom_firm_state": detail_state[survivor_atom_mask],
                    "entry_transport_atom_parent_h": entry_atom_parent_h[~entry_atom_default],
                    "entry_transport_atom_weight": entry_atom_weight[~entry_atom_default],
                    "entry_transport_atom_source_z": entry_atom_z[~entry_atom_default],
                    "entry_transport_atom_i": entry_atom_i[~entry_atom_default],
                    "entry_transport_atom_firm_state": entry_atom_state[~entry_atom_default],
                    "default_boundary_probability": default_boundaries.boundary_coordinate,
                    "default_boundary_mask": default_boundaries.boundary_mask,
                    "default_boundary_residual": default_boundaries.residual,
                    "default_boundary_task_position": task_position,
                    "investment_boundary_probability": investment_boundary_probability,
                    "investment_boundary_mask": investment_boundary_mask,
                    "investment_boundary_residual": investment_boundary_residual,
                    "investment_boundary_source_task": investment_boundary_source_task,
                    "investment_boundary_source_z": investment_boundary_source_z,
                }
            )
        self._sync(device)
        self.accounting_seconds += time.perf_counter() - accounting_started
        self.resource_integration_seconds += time.perf_counter() - accounting_started
        return result

    @staticmethod
    def _pack_atoms_by_path(
        *,
        path: torch.Tensor,
        batch_size: int,
        values: Dict[str, torch.Tensor],
        fill: Dict[str, float | int | bool],
    ) -> Dict[str, torch.Tensor]:
        """Pack flat atom tensors into deterministic padded [B,A] tensors."""
        if path.ndim != 1:
            raise ValueError("atom path must be one dimensional")
        counts = torch.bincount(path, minlength=batch_size)
        capacity = max(int(counts.max().item()) if counts.numel() else 0, 1)
        if path.numel():
            order = torch.argsort(
                path * (path.numel() + 1)
                + torch.arange(path.numel(), device=path.device)
            )
            sorted_path = path[order]
            change = torch.ones_like(sorted_path, dtype=torch.bool)
            change[1:] = sorted_path[1:] != sorted_path[:-1]
            starts = torch.nonzero(change, as_tuple=False).reshape(-1)
            lengths = torch.diff(
                torch.cat(
                    (
                        starts,
                        torch.tensor([path.numel()], device=path.device),
                    )
                )
            )
            rank = torch.arange(path.numel(), device=path.device) - torch.repeat_interleave(
                starts, lengths
            )
        else:
            order = path
            sorted_path = path
            rank = path
        packed: Dict[str, torch.Tensor] = {}
        for name, value in values.items():
            output = torch.full(
                (batch_size, capacity),
                fill[name],
                device=value.device,
                dtype=value.dtype,
            )
            if path.numel():
                output[sorted_path, rank] = value[order]
            packed[name] = output
        mask = torch.arange(capacity, device=path.device)[None, :] < counts[:, None]
        packed["atom_mask"] = mask
        return packed

    @torch.inference_mode()
    def _commit_selected_decision_record(
        self,
        sim: Any,
        state: Dict[str, torch.Tensor],
        candidates: EntryCandidates,
        selected_h: torch.Tensor,
        detail: dict[str, torch.Tensor],
        *,
        t: int,
        branch: int,
    ) -> None:
        """Commit the exact survivor quadrature used by the selected root."""
        B, L = state["b"].shape
        device = state["b"].device
        dtype = state["b"].dtype
        if detail["h"].shape[1] != 1:
            raise MarketClearingError("selected decision reconstruction requires H=1")

        local_source_task = detail["transport_atom_source_task"]
        active_flat_task = detail["default_boundary_task_position"]
        flat_component = active_flat_task[local_source_task]
        incumbent_path = torch.div(flat_component, L, rounding_mode="floor")
        incumbent_component = flat_component % L
        incumbent_state = detail["transport_atom_firm_state"]
        incumbent_bp_head = detail["transport_atom_bp_head"].reshape(-1, 1)
        if incumbent_state.shape[0]:
            if int(branch) == -1 and hasattr(sim, "_resolve_bp_action"):
                component_id = state.get(
                    "component_id",
                    torch.arange(L, device=device, dtype=torch.long)[None, :].expand(B, L),
                )[incumbent_path, incumbent_component]
                node_id = state.get(
                    "economic_node_id",
                    torch.arange(B, device=device, dtype=torch.long),
                )[incumbent_path]
                bp_action = sim._resolve_bp_action(
                    sim=sim,
                    firm_state=incumbent_state,
                    model_output=None,
                    bp_head=incumbent_bp_head,
                    hatc_cal=selected_h[incumbent_path, None],
                    lnk_cal=torch.log(
                        detail["K_total"][:, 0].to(dtype).clamp_min(1.0e-30)
                    )[incumbent_path, None],
                    path_index=incumbent_path,
                    firm_id=component_id,
                    economic_node_id=node_id,
                    t=int(t),
                    branch=int(branch),
                    decision_stage="boundary_transport_parent_policy",
                    bp_execution_applicable=True,
                ).reshape(-1)
            else:
                bp_action = incumbent_bp_head.reshape(-1)
            incumbent_b_next = apply_refinancing_policy(
                state["b"][incumbent_path, incumbent_component],
                bp_action,
                state["eta"][incumbent_path, incumbent_component],
            )
            incumbent_capital_next = state["K"][incumbent_path, incumbent_component] * torch.where(
                detail["transport_atom_invest"],
                torch.full_like(detail["transport_atom_i"], float(sim.g)),
                torch.ones_like(detail["transport_atom_i"]),
            )
        else:
            incumbent_b_next = torch.empty(0, device=device, dtype=dtype)
            incumbent_capital_next = torch.empty(0, device=device, dtype=dtype)

        entry_parent_h = detail["entry_transport_atom_parent_h"]
        entry_path = torch.div(entry_parent_h, 1, rounding_mode="floor")
        entry_state = detail["entry_transport_atom_firm_state"]
        entry_count = int(entry_path.numel())
        entry_b_next = torch.zeros(entry_count, device=device, dtype=dtype)
        if entry_count:
            entry_capital = torch.where(
                candidates.mask,
                candidates.K_birth,
                torch.zeros_like(candidates.K_birth),
            ).sum(dim=1) / candidates.mask.sum(dim=1).to(dtype).clamp_min(1.0)
            entry_capital_next = entry_capital[entry_path]
        else:
            entry_capital_next = torch.empty(0, device=device, dtype=dtype)

        atom_path = torch.cat((incumbent_path, entry_path))
        weight = torch.cat(
            (
                detail["transport_atom_weight"],
                detail["entry_transport_atom_weight"],
            )
        ).to(torch.float64)
        source_z = torch.cat(
            (
                detail["transport_atom_source_z"],
                detail["entry_transport_atom_source_z"],
            )
        )
        b_next = torch.cat((incumbent_b_next, entry_b_next))
        capital_next = torch.cat((incumbent_capital_next, entry_capital_next))
        investment_region = torch.cat(
            (
                detail["transport_atom_invest"].to(torch.long),
                torch.full((entry_count,), -1, device=device, dtype=torch.long),
            )
        )
        source_component = torch.cat(
            (
                state.get(
                    "component_id",
                    torch.arange(L, device=device, dtype=torch.long)[None, :].expand(B, L),
                )[incumbent_path, incumbent_component],
                torch.full((entry_count,), -1, device=device, dtype=torch.long),
            )
        )
        origin_kind = torch.cat(
            (
                state.get("origin_kind", torch.zeros((B, L), device=device, dtype=torch.long))[
                    incumbent_path, incumbent_component
                ],
                torch.ones(entry_count, device=device, dtype=torch.long),
            )
        )
        origin_source_id = torch.cat(
            (
                state.get(
                    "origin_source_id", state.get("component_id", torch.zeros((B, L), device=device, dtype=torch.long))
                )[incumbent_path, incumbent_component],
                torch.arange(entry_count, device=device, dtype=torch.long),
            )
        )
        cohort = torch.cat(
            (
                state.get("cohort", torch.zeros((B, L), device=device, dtype=torch.long))[
                    incumbent_path, incumbent_component
                ],
                torch.full((entry_count,), int(t), device=device, dtype=torch.long),
            )
        )
        birth_time = torch.cat(
            (
                state.get("birth_time", torch.full((B, L), -1, device=device, dtype=torch.long))[
                    incumbent_path, incumbent_component
                ].to(torch.long),
                torch.full((entry_count,), int(t), device=device, dtype=torch.long),
            )
        )
        node_vector = state.get(
            "economic_node_id", torch.arange(B, device=device, dtype=torch.long)
        )
        economic_node = node_vector[atom_path]
        size = int(atom_path.numel())
        values = {
            "transport_weight": weight,
            "source_z_current": source_z,
            "b_next": b_next,
            "capital_next_per_unit": capital_next,
            "source_component_id": source_component,
            "survival_region_id": torch.zeros(size, device=device, dtype=torch.long),
            "investment_region_id": investment_region,
            "policy_branch_id": investment_region.clone(),
            "economic_node_id": economic_node,
            "source_time": torch.full((size,), int(t), device=device, dtype=torch.long),
            "destination_time": torch.full((size,), int(t + 1), device=device, dtype=torch.long),
            "cohort": cohort,
            "birth_time": birth_time,
            "maintenance_eligibility": torch.ones(size, device=device, dtype=torch.bool),
            "origin_kind": origin_kind,
            "origin_source_id": origin_source_id,
        }
        fill = {
            "transport_weight": 0.0,
            "source_z_current": 0.0,
            "b_next": 0.0,
            "capital_next_per_unit": 1.0,
            "source_component_id": -1,
            "survival_region_id": -1,
            "investment_region_id": -1,
            "policy_branch_id": -1,
            "economic_node_id": -1,
            "source_time": int(t),
            "destination_time": int(t + 1),
            "cohort": -1,
            "birth_time": -1,
            "maintenance_eligibility": False,
            "origin_kind": -1,
            "origin_source_id": -1,
        }
        packed = self._pack_atoms_by_path(
            path=atom_path, batch_size=B, values=values, fill=fill
        )
        transport = PostDecisionTransportAtoms(
            atom_mask=packed.pop("atom_mask"), **packed
        ).validate()
        ledger = {
            "C_raw": detail["C_raw"][:, 0],
            "C_oper": detail["C_oper"][:, 0],
            "K_total": detail["K_total"][:, 0],
            "Y_total": detail["Y_total"][:, 0],
            "I_oper_total": detail["I_oper_total"][:, 0],
            "I_entry_total": detail["I_entry_total"][:, 0],
            "Liquidation_total": detail["Liquidation_total"][:, 0],
            "I_maintenance_total": detail["I_maintenance_total"][:, 0],
        }
        record = BoundaryAwareDecisionRecord(
            predecision_fingerprint=predecision_fingerprint_from_state(state),
            candidate_h=selected_h,
            ledger=ledger,
            transport_atoms=transport,
            provenance={
                "t": int(t),
                "branch": int(branch),
                "representation": "boundary_aware_continuum_v1",
                "dense_candidate_h_chunk_size": (
                    1 if self.integration.z_scan_points >= 257 else None
                ),
                "root_selection": "existing_coarse_to_fine_selected_record",
                "component_mean_hard_default_used": False,
            },
        ).detached()
        state["_boundary_decision_record"] = record
        state["_selected_transport_atoms"] = record.transport_atoms
        state["hatc_cal"] = selected_h.detach().clone()
        state["lnk_cal"] = torch.log(ledger["K_total"].clamp_min(1.0e-30)).to(dtype)
        state["accounting_C_raw"] = ledger["C_raw"].to(dtype)
        state["accounting_C_oper"] = ledger["C_oper"].to(dtype)
        state["accounting_K_total"] = ledger["K_total"].to(dtype)
        state["accounting_Y_total"] = ledger["Y_total"].to(dtype)
        state["accounting_I_oper"] = ledger["I_oper_total"].to(dtype)
        state["accounting_I_entry"] = ledger["I_entry_total"].to(dtype)
        state["accounting_Liquidation_total"] = ledger["Liquidation_total"].to(dtype)
        state["accounting_I_maintenance"] = ledger["I_maintenance_total"].to(dtype)
        state["accounting_valid"] = torch.ones(B, device=device, dtype=torch.bool)

    def validate_committed(
        self,
        state: Dict[str, torch.Tensor],
        *,
        t: int,
        branch: int,
    ) -> None:
        super().validate_committed(state, t=t, branch=branch)
        record = state.get("_boundary_decision_record")
        if record is None:
            raise MarketClearingError("boundary-aware committed state lacks decision record")
        committed = {
            "C_raw": state["accounting_C_raw"].to(torch.float64),
            "C_oper": state["accounting_C_oper"].to(torch.float64),
            "K_total": state["accounting_K_total"].to(torch.float64),
            "Y_total": state["accounting_Y_total"].to(torch.float64),
            "I_oper_total": state["accounting_I_oper"].to(torch.float64),
            "I_entry_total": state["accounting_I_entry"].to(torch.float64),
            "Liquidation_total": state["accounting_Liquidation_total"].to(torch.float64),
            "I_maintenance_total": state["accounting_I_maintenance"].to(torch.float64),
        }
        record.assert_ledger_parity(committed, atol=5e-6, rtol=5e-6)
        validation_key = (
            record.predecision_fingerprint,
            record.candidate_h.detach().to(device="cpu").contiguous().numpy().tobytes(),
        )
        if validation_key not in self._validated_boundary_records:
            self._validated_boundary_records.add(validation_key)
            cleared = state.get(
                "market_root_found",
                torch.ones_like(state["accounting_C_raw"], dtype=torch.bool),
            ).bool()
            self.candidate_commit_parity_pass_count += int(cleared.sum())
        state["candidate_commit_parity_passed"] = torch.ones_like(
            state["accounting_C_raw"], dtype=torch.bool
        )

    def summary(self) -> dict[str, Any]:
        result = super().summary()
        result.update(
            {
                "representation": "boundary_aware_continuum_v1",
                "COMPONENT_MEAN_HARD_DEFAULT_USED": False,
                "default_boundary_count": int(self.default_boundary_count),
                "investment_boundary_count": int(self.investment_boundary_count),
                "boundary_search_seconds": float(self.boundary_search_seconds),
                "resource_integration_seconds": float(self.resource_integration_seconds),
                "dense_candidate_h_chunk_size": (
                    1 if self.integration.z_scan_points >= 257 else None
                ),
                "scan_tail_probability_mass": float(
                    self.tail_probability_mass_scanned_outside
                ),
                "integration": {
                    name: getattr(self.integration, name)
                    for name in self.integration.__dataclass_fields__
                },
            }
        )
        return result

