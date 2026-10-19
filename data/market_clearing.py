"""Experimental current-node market clearing for :class:`SimulateTS`.

The production simulator normally treats ``hatcf`` as an FC1 belief.  In the
``market_cleared_experimental_v1`` mode this module replaces that belief by a
root of

    exp(h) * K(h) - C_raw(h) = 0

before the economic node is committed.  Every candidate ``h`` triggers a fresh
batched Firm forward for incumbents and potential entrants.  The module is
deliberately optimizer-free and does not change any production equation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, Optional

import torch

from .entry import EntryCandidates
from .simulation_forward import forward_policy_value_for_simulation
from .consistent_aggregation import (
    NodeDecisionRecord,
    assert_candidate_commit_parity,
    tensor_snapshot_hash,
)


MARKET_CLEARING_DISABLED = "disabled"
MARKET_CLEARING_V1 = "market_cleared_experimental_v1"
STRICT_GLOBAL = "strict_global"
FC1_BOUNDED_IMPROVEMENT_PILOT = "fc1_bounded_improvement_pilot"
ROOT_ACCEPTANCE_SAFETY_FACTOR = 0.5

FAST_STATUS_STRICT_ROOT_FOUND = 0
FAST_STATUS_IMPROVED_APPROXIMATE = 1
FAST_STATUS_BASELINE_NOT_IMPROVED = 2
FAST_STATUS_NO_FEASIBLE_CANDIDATE = 3
FAST_STATUS_UNDEFINED_ECONOMIC_STATE = 4
FAST_STATUS_LABELS = {
    FAST_STATUS_STRICT_ROOT_FOUND: "STRICT_ROOT_FOUND",
    FAST_STATUS_IMPROVED_APPROXIMATE: "IMPROVED_APPROXIMATE",
    FAST_STATUS_BASELINE_NOT_IMPROVED: "BASELINE_NOT_IMPROVED",
    FAST_STATUS_NO_FEASIBLE_CANDIDATE: "NO_FEASIBLE_CANDIDATE",
    FAST_STATUS_UNDEFINED_ECONOMIC_STATE: "UNDEFINED_ECONOMIC_STATE",
}

# Per-path lifecycle codes.  These are tensors (rather than Python strings) so
# they survive GPU sub-batching without introducing device synchronizations.
# Human-readable labels are emitted only by the diagnostic/output layer.
ECONOMIC_NODE_ACTIVE = 0
ECONOMIC_NODE_EXTINCT_BEFORE_NODE = 1
ECONOMIC_NODE_EXTINCT_AFTER_DECISION = 2
ECONOMIC_NODE_NO_VALID_CLEARING_ROOT = 3
ECONOMIC_NODE_NUMERICAL_MASS_LOSS = 4
ECONOMIC_NODE_STATUS_LABELS = {
    ECONOMIC_NODE_ACTIVE: "ACTIVE_NODE",
    ECONOMIC_NODE_EXTINCT_BEFORE_NODE: "EXTINCT_BEFORE_NODE",
    ECONOMIC_NODE_EXTINCT_AFTER_DECISION: "EXTINCT_AFTER_DECISION",
    ECONOMIC_NODE_NO_VALID_CLEARING_ROOT: "NO_VALID_CLEARING_ROOT",
    ECONOMIC_NODE_NUMERICAL_MASS_LOSS: "NUMERICAL_MASS_LOSS",
}


class MarketClearingError(RuntimeError):
    """Raised when strict experimental market clearing cannot commit a node."""


@dataclass(frozen=True)
class MarketClearingConfig:
    solver_mode: str = STRICT_GLOBAL
    coarse_points: int = 10
    refinement_points: int = 20
    max_refinement_rounds: int = 4
    support_lower: float = -8.0
    support_upper: float = 2.0
    support_padding: float = 2.0
    residual_tolerance: float = 1.0e-5
    width_tolerance: float = 1.0e-5
    candidate_chunk_size: int = 262_144
    fast_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    fast_local_correction_rounds: int = 2
    fast_max_evaluations: int = 8
    fast_improvement_noise_tolerance: float = 1.0e-10

    def validate(self) -> None:
        if self.solver_mode not in {
            STRICT_GLOBAL,
            FC1_BOUNDED_IMPROVEMENT_PILOT,
        }:
            raise ValueError(f"unknown market-clearing solver mode: {self.solver_mode}")
        if self.coarse_points < 3:
            raise ValueError("market clearing coarse_points must be at least 3")
        if self.refinement_points < 3:
            raise ValueError("market clearing refinement_points must be at least 3")
        if self.max_refinement_rounds < 1:
            raise ValueError("market clearing max_refinement_rounds must be positive")
        if not self.support_lower < self.support_upper:
            raise ValueError("market clearing support must have positive width")
        if self.support_padding <= 0.0:
            raise ValueError("market clearing support_padding must be positive")
        if self.residual_tolerance <= 0.0 or self.width_tolerance <= 0.0:
            raise ValueError("market clearing tolerances must be positive")
        if self.candidate_chunk_size <= 0:
            raise ValueError("market clearing candidate_chunk_size must be positive")
        if not self.fast_alphas or any(
            not (0.0 < float(value) <= 1.0) for value in self.fast_alphas
        ):
            raise ValueError("fast_alphas must be nonempty and lie in (0,1]")
        if self.fast_local_correction_rounds < 0:
            raise ValueError("fast_local_correction_rounds must be nonnegative")
        # One initial evaluation and one final accepted-record reconstruction
        # are both real economic-operator calls and count against the budget.
        required = 2 + len(self.fast_alphas) + self.fast_local_correction_rounds
        if self.fast_max_evaluations < required:
            raise ValueError(
                "fast_max_evaluations does not cover the configured bounded search: "
                f"required={required}, budget={self.fast_max_evaluations}"
            )
        if self.fast_improvement_noise_tolerance < 0.0:
            raise ValueError("fast_improvement_noise_tolerance must be nonnegative")


def empty_entry_candidates(state: Dict[str, torch.Tensor]) -> EntryCandidates:
    """Return a zero-width candidate bank for a no-entry current node."""
    n_paths = int(state["b"].shape[0])
    device = state["b"].device
    dtype = state["b"].dtype
    empty_float = torch.empty((n_paths, 0), device=device, dtype=dtype)
    empty_bool = torch.empty((n_paths, 0), device=device, dtype=torch.bool)
    zeros = torch.zeros(n_paths, device=device, dtype=dtype)
    return EntryCandidates(
        counts=torch.zeros(n_paths, device=device, dtype=torch.long),
        mask=empty_bool,
        z=empty_float,
        entry_cost=empty_float.clone(),
        K_birth=empty_float.clone(),
        potential_capital_nominal=zeros,
        candidate_capital_realized=zeros.clone(),
    )


class MarketClearingController:
    """GPU-vectorized coarse-to-fine current-Hatc root controller."""

    def __init__(self, config: MarketClearingConfig) -> None:
        config.validate()
        self.config = config
        self.node_count = 0
        self.operator_evaluations = 0
        self.network_forward_calls = 0
        self.policy_forward_seconds = 0.0
        self.accounting_seconds = 0.0
        self.root_search_seconds = 0.0
        self.multiple_root_nodes = 0
        self.candidate_commit_parity_pass_count = 0
        self._validated_decision_records: set[tuple[str, bytes]] = set()
        self.failure_records: list[dict[str, Any]] = []
        self.root_records: list[dict[str, Any]] = []

    @property
    def _root_acceptance_tolerance(self) -> float:
        """Use a strict internal margin before the public commit gate.

        Candidate evaluation and the production replay are algebraically
        identical but need not have identical floating-point reduction order.
        Stopping at the public boundary can therefore produce a committed
        residual slightly above it.  The public economic contract remains
        ``residual_tolerance``; this tighter admission threshold ensures that
        replay must satisfy that contract without adding a tolerance slack.
        """
        return float(self.config.residual_tolerance) * ROOT_ACCEPTANCE_SAFETY_FACTOR

    @staticmethod
    def _sync(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @torch.inference_mode()
    def _entry_values(
        self, model: torch.nn.Module, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states.shape[0] == 0:
            empty = states.new_empty((0,))
            return empty, empty
        p0_parts = []
        pi_parts = []
        for start in range(0, states.shape[0], self.config.candidate_chunk_size):
            values = model.forward_value_components(
                states[start : start + self.config.candidate_chunk_size]
            )
            p0_parts.append(values["V0_physical"].reshape(-1))
            pi_parts.append(values["VI_physical"].reshape(-1))
            self.network_forward_calls += 1
        return torch.cat(p0_parts), torch.cat(pi_parts)

    @torch.inference_mode()
    def _firm_policy(
        self, model: torch.nn.Module, states: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        pieces = {
            name: [] for name in ("bar_i", "bar_z", "bp", "P0", "PI", "Phat")
        }
        for start in range(0, states.shape[0], self.config.candidate_chunk_size):
            output = forward_policy_value_for_simulation(
                model, states[start : start + self.config.candidate_chunk_size]
            )
            for name in pieces:
                value = getattr(output, name, None)
                if value is None:
                    value = torch.full_like(output.P0, float("nan"))
                pieces[name].append(value.reshape(-1))
            self.network_forward_calls += 1
        return {name: torch.cat(values) for name, values in pieces.items()}

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
        """Evaluate all ``[path, h]`` candidates without a path/firm loop."""
        model = sim.models.get("policy_value")
        if model is None:
            raise MarketClearingError("market clearing requires policy_value")
        if h.ndim != 2 or h.shape[0] != state["b"].shape[0]:
            raise ValueError("market clearing h must have shape [B,H]")
        device = state["b"].device
        dtype = state["b"].dtype
        h = h.to(device=device, dtype=dtype)
        B, H = h.shape
        N = int(state["b"].shape[1])
        E = int(candidates.mask.shape[1])
        if candidates.mask.shape[0] != B:
            raise ValueError("entry candidate path count does not match state")
        self.operator_evaluations += B * H

        policy_started = time.perf_counter()
        if E:
            entry_state = torch.stack(
                (
                    torch.zeros(B, H, E, device=device, dtype=dtype),
                    candidates.z[:, None, :].expand(B, H, E),
                    torch.zeros(B, H, E, device=device, dtype=dtype),
                    candidates.entry_cost[:, None, :].expand(B, H, E),
                    state["x"][:, None, None].expand(B, H, E),
                    h[:, :, None].expand(B, H, E),
                    state["lnkf"][:, None, None].expand(B, H, E),
                ),
                dim=-1,
            )
            entry_p0, entry_pi = self._entry_values(
                model, entry_state.reshape(B * H * E, 7)
            )
            accepted = (
                (entry_pi - entry_p0).reshape(B, H, E) >= 0.0
            ) & candidates.mask[:, None, :]
        else:
            accepted = torch.empty((B, H, 0), device=device, dtype=torch.bool)

        def incumbent(name: str) -> torch.Tensor:
            return state[name][:, None, :].expand(B, H, N)

        b = torch.cat(
            (incumbent("b"), torch.zeros(B, H, E, device=device, dtype=dtype)),
            dim=2,
        )
        z = torch.cat(
            (incumbent("z"), candidates.z[:, None, :].expand(B, H, E)), dim=2
        )
        eta = torch.cat(
            (incumbent("eta"), torch.zeros(B, H, E, device=device, dtype=dtype)),
            dim=2,
        )
        investment_cost = torch.cat(
            (
                incumbent("i"),
                candidates.entry_cost[:, None, :].expand(B, H, E),
            ),
            dim=2,
        )
        capital = torch.cat(
            (incumbent("K"), candidates.K_birth[:, None, :].expand(B, H, E)),
            dim=2,
        )
        alive = torch.cat(
            (state["alive"][:, None, :].expand(B, H, N), accepted), dim=2
        ).bool()
        incumbent_mass = state.get(
            "firm_mass", state["alive"].to(dtype)
        )[:, None, :].expand(B, H, N)
        if E:
            if getattr(sim, "aggregation_mode", "legacy_mixed_replay") == (
                "continuum_mass_consistent_v1"
            ):
                represented_capital = (
                    candidates.K_birth * candidates.mask.to(dtype)
                ).sum(dim=1)
                mass_per_candidate = torch.where(
                    represented_capital > 0.0,
                    candidates.potential_capital_nominal
                    / represented_capital.clamp_min(1.0e-12),
                    torch.zeros_like(represented_capital),
                )
                candidate_mass = (
                    mass_per_candidate[:, None] * candidates.mask.to(dtype)
                )
            else:
                candidate_mass = candidates.mask.to(dtype)
            entry_mass = candidate_mass[:, None, :].expand(B, H, E)
        else:
            entry_mass = torch.empty((B, H, 0), device=device, dtype=dtype)
        economic_mass = torch.cat((incumbent_mass, entry_mass), dim=2)
        economic_time = state["economic_time"].to(state["birth_time"].dtype)
        incumbent_birth = (
            (state["initial_cohort"] < 0.5)
            & (state["birth_time"] == economic_time[:, None])
            & state["alive"]
        )
        birth = torch.cat(
            (incumbent_birth[:, None, :].expand(B, H, N), accepted), dim=2
        )
        total_slots = N + E
        policy_state = torch.stack(
            (
                b,
                z,
                eta,
                investment_cost,
                state["x"][:, None, None].expand(B, H, total_slots),
                h[:, :, None].expand(B, H, total_slots),
                state["lnkf"][:, None, None].expand(B, H, total_slots),
            ),
            dim=-1,
        )
        policy = self._firm_policy(model, policy_state.reshape(B * H * total_slots, 7))
        policy = {
            name: value.reshape(B, H, total_slots) for name, value in policy.items()
        }
        self._sync(device)
        self.policy_forward_seconds += time.perf_counter() - policy_started

        accounting_started = time.perf_counter()
        if bool(getattr(sim, "hard_realized_aggregation", False)):
            if not torch.isfinite(policy["Phat"]).all():
                raise MarketClearingError(
                    "hard aggregation requires finite policy Phat output"
                )
            survival = policy["Phat"] > 0.0
            bar_i_policy = (
                survival & (policy["PI"] >= policy["P0"])
            ).to(dtype)
            bar_z = (~survival).to(dtype)
        else:
            bar_i_policy = policy["bar_i"]
            bar_z = policy["bar_z"]
        bar_i = torch.where(birth, torch.zeros_like(bar_i_policy), bar_i_policy)
        maintenance = alive & ~birth
        flat = B * H * total_slots
        Y, I_oper, Phi, C_firm, I_maintenance = sim._resource_accounting(
            capital.reshape(flat),
            z.reshape(flat),
            state["x"][:, None, None].expand(B, H, total_slots).reshape(flat),
            bar_i.reshape(flat),
            bar_z.reshape(flat),
            investment_cost.reshape(flat),
            maintenance_eligible=maintenance.reshape(flat),
            return_maintenance=True,
        )

        def shaped(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(B, H, total_slots)

        Y, I_oper, Phi, C_firm, I_maintenance = map(
            shaped, (Y, I_oper, Phi, C_firm, I_maintenance)
        )
        mask = alive.to(dtype) * economic_mass
        entry_spend = (
            candidates.entry_cost[:, None, :]
            * candidates.K_birth[:, None, :]
            * accepted.to(dtype)
            * entry_mass
        ).sum(dim=2)
        K_total = (capital * mask).sum(dim=2)
        Y_total = (Y * mask).sum(dim=2)
        I_oper_total = (I_oper * mask).sum(dim=2)
        liquidation_total = (Phi * mask).sum(dim=2)
        maintenance_total = (I_maintenance * mask).sum(dim=2)
        C_oper = (C_firm * mask).sum(dim=2)
        C_raw = C_oper - entry_spend
        G = torch.exp(h) * K_total - C_raw
        scale = torch.maximum(
            torch.ones_like(G), torch.maximum(C_raw.abs(), K_total.abs())
        )
        finite = torch.isfinite(G) & torch.isfinite(C_raw) & torch.isfinite(K_total)
        feasible = finite & (C_raw > 0.0) & (K_total > 0.0)
        hard_i = (bar_i >= 0.5) & alive
        hard_z = (bar_z >= 0.5) & alive
        investment_mass = (economic_mass * hard_i.to(dtype)).sum(dim=2)
        default_mass = (economic_mass * hard_z.to(dtype)).sum(dim=2)
        accepted_entry_mass = (
            entry_mass * accepted.to(dtype)
        ).sum(dim=2)
        result = {
            "h": h,
            "G": G,
            "G_normalized": G / scale,
            "C_raw": C_raw,
            "C_oper": C_oper,
            "K_total": K_total,
            "Y_total": Y_total,
            "I_oper_total": I_oper_total,
            "I_entry_total": entry_spend,
            "Liquidation_total": liquidation_total,
            "I_maintenance_total": maintenance_total,
            "resource_feasible": feasible,
            "finite": finite,
            "investment_count": hard_i.sum(dim=2),
            "default_count": hard_z.sum(dim=2),
            "entry_count": accepted.sum(dim=2),
            "firm_count": (economic_mass * alive.to(dtype)).sum(dim=2),
            "investment_mass": investment_mass,
            "default_mass": default_mass,
            "entry_mass": accepted_entry_mass,
        }
        if detail:
            result.update(
                {
                    "accepted_entry": accepted,
                    "bar_i_firm": bar_i,
                    "bar_z_firm": bar_z,
                    "bp_firm": policy["bp"],
                    "P0_firm": policy["P0"],
                    "PI_firm": policy["PI"],
                    "Phat_firm": policy["Phat"],
                    "economic_mass": economic_mass,
                    "entry_candidate_mass": entry_mass,
                    "firm_state": policy_state,
                }
            )
        self._sync(device)
        self.accounting_seconds += time.perf_counter() - accounting_started
        return result

    @staticmethod
    def _crossings(result: dict[str, torch.Tensor]) -> torch.Tensor:
        g = result["G"]
        valid = result["finite"] & (result["K_total"] > 0.0)
        crossing = (
            valid[:, :-1]
            & valid[:, 1:]
            & (torch.signbit(g[:, :-1]) != torch.signbit(g[:, 1:]))
        )
        return torch.nonzero(crossing, as_tuple=False)

    def _support(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        belief = state["hatcf"]
        # The formal experiment requires a common economically declared
        # [h_min, h_max] scan.  Belief/calculated values are anchors and root
        # selection references, not a reason to silently truncate support.
        alpha = torch.linspace(
            0.0,
            1.0,
            self.config.coarse_points,
            device=belief.device,
            dtype=belief.dtype,
        )
        grid = (
            self.config.support_lower
            + (self.config.support_upper - self.config.support_lower) * alpha
        )
        return grid[None, :].expand(belief.shape[0], -1)

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
        """Attach the exact selected hard decisions for the production commit."""
        if not bool(getattr(sim, "hard_realized_aggregation", False)):
            return
        B, N = state["b"].shape
        E = int(candidates.mask.shape[1])
        bar_i_all = detail["bar_i_firm"][:, 0]
        bar_z_all = detail["bar_z_firm"][:, 0]
        bp_all = detail["bp_firm"][:, 0].detach().clone()

        # Only a parent node commits b_t -> b_{t+1}.  Resolve its grid action
        # now and reuse it in _process_node_batched; child observation rows are
        # explicitly non-applicable and will be resolved when promoted to a
        # parent node.
        if int(branch) == -1:
            active = state["alive"]
            path_grid = torch.arange(B, device=state["b"].device)[:, None].expand(B, N)
            active_pos = torch.nonzero(active.reshape(-1), as_tuple=False).reshape(-1)
            if active_pos.numel():
                firm_state = detail["firm_state"][:, 0, :N].reshape(B * N, 7)[active_pos]
                head = bp_all[:, :N].reshape(-1)[active_pos].reshape(-1, 1)
                needs_resolver = (
                    getattr(sim, "bp_action_override", None) is not None
                    or getattr(sim, "bp_action_source", "head") == "grid"
                    or bool(getattr(sim, "bp_actual_use_trace_enabled", False))
                )
                if needs_resolver:
                    path_index = path_grid.reshape(-1)[active_pos]
                    resolved = sim._resolve_bp_action(
                        sim=sim,
                        firm_state=firm_state,
                        model_output=None,
                        bp_head=head,
                        hatc_cal=selected_h[path_index, None],
                        lnk_cal=torch.log(
                            detail["K_total"][:, 0].clamp_min(1.0e-30)
                        )[path_index, None],
                        path_index=path_index,
                        firm_id=state["firm_id"].reshape(-1)[active_pos],
                        economic_node_id=state["economic_node_id"][path_index],
                        t=int(t),
                        branch=int(branch),
                        decision_stage="parent_transition_policy",
                        bp_execution_applicable=True,
                    ).reshape(-1)
                    bp_incumbent = bp_all[:, :N].clone().reshape(-1)
                    bp_incumbent[active_pos] = resolved
                    bp_all[:, :N] = bp_incumbent.reshape(B, N)

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
        record = NodeDecisionRecord(
            snapshot_hash=tensor_snapshot_hash(state),
            candidate_h=selected_h,
            bar_i=bar_i_all[:, :N],
            bar_z=bar_z_all[:, :N],
            bp=bp_all[:, :N],
            accepted_entry=detail["accepted_entry"][:, 0],
            entry_bar_i=bar_i_all[:, N : N + E],
            entry_bar_z=bar_z_all[:, N : N + E],
            firm_mass=detail["economic_mass"][:, 0, :N],
            entry_mass=detail["entry_candidate_mass"][:, 0],
            ledger=ledger,
            provenance={
                "t": int(t),
                "branch": int(branch),
                "aggregation_mode": str(sim.aggregation_mode),
                "configured_bp_source": str(sim.bp_action_source),
                "bp_execution_applicable": int(branch) == -1,
                "root_selection": "coarse_to_fine_selected_record",
            },
        ).detached()
        state["_selected_decision_record"] = record
        state["decision_record_bar_i"] = record.bar_i
        state["decision_record_bar_z"] = record.bar_z
        state["decision_record_bp"] = record.bp
        state["decision_record_valid"] = torch.ones(
            B, device=state["b"].device, dtype=torch.bool
        )
        state["pending_entry_accepted"] = record.accepted_entry
        state["pending_entry_bar_i"] = record.entry_bar_i
        state["pending_entry_bar_z"] = record.entry_bar_z
        state["pending_entry_mass"] = record.entry_mass

    @staticmethod
    def _merge_active_paths(
        state: Dict[str, torch.Tensor],
        cleared: Dict[str, torch.Tensor],
        active_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Scatter root-clearing outputs back into the full path batch.

        The production simulator keeps extinct paths in the padded batch so
        surviving paths can continue.  Those rows are not economic nodes and
        must not be sent through the root operator, but the full-sized state is
        still needed by the vectorized transition code.  Undefined diagnostics
        therefore remain NaN/false/zero on extinct rows while active rows are
        copied exactly from the independently-cleared sub-batch.
        """
        batch_size = int(state["b"].shape[0])
        active_size = int(active_index.numel())
        undefined_float = {
            "hatcf",
            "hatc_cleared",
            "hatc_cal",
            "lnk_cal",
            "market_root_residual",
            "market_root_residual_normalized",
            "market_root_C",
            "market_root_K",
            "market_committed_residual",
            "market_committed_residual_normalized",
        }
        minus_one_integer = {"market_root_refinement_rounds"}
        belief = state["hatcf"].detach().clone()
        root_output_fields = {
            "hatcf",
            "hatc_belief",
            "hatc_cleared",
            "hatc_cal",
            "lnk_cal",
            "market_previous_root",
            "market_previous_branch_id",
            "market_root_branch_id",
            "market_root_branch_switch",
            "market_root_found",
            "market_root_count",
            "market_root_residual",
            "market_root_residual_normalized",
            "market_root_C",
            "market_root_K",
            "market_root_refinement_rounds",
            "market_root_policy_jump",
            "market_investment_flip_vs_belief",
            "market_default_flip_vs_belief",
            "market_entry_flip_vs_belief",
            "market_root_bracket_count",
            "market_root_operator_evaluations",
            "market_committed_residual",
            "market_committed_residual_normalized",
            "candidate_commit_parity_passed",
            "economic_node_status",
            "economic_state_defined",
            "decision_record_valid",
        }
        output_prefixes = (
            "accounting_",
            "decision_",
            "pending_entry_",
        )

        # Inactive rows are terminal economic paths.  Accounting flows are
        # physically zero, while market states that never existed are NaN.
        # This prevents an extinct lineage from silently inheriting a finite
        # Hatc/LnK value from the previous node.
        default_status = torch.full(
            (batch_size,),
            ECONOMIC_NODE_EXTINCT_BEFORE_NODE,
            device=active_index.device,
            dtype=torch.long,
        )
        state["economic_node_status"] = default_status
        state["economic_state_defined"] = torch.zeros(
            batch_size, device=active_index.device, dtype=torch.bool
        )

        for name, value in cleared.items():
            if not torch.is_tensor(value) or value.ndim == 0:
                continue
            if int(value.shape[0]) != active_size:
                continue
            if name not in root_output_fields and not name.startswith(output_prefixes):
                continue

            existing = state.get(name)
            preserve_existing = name in {
                "hatc_belief",
                "market_previous_root",
                "market_previous_branch_id",
                "economic_node_status",
                "economic_state_defined",
            }
            if (
                preserve_existing
                and
                torch.is_tensor(existing)
                and existing.ndim == value.ndim
                and int(existing.shape[0]) == batch_size
                and tuple(existing.shape[1:]) == tuple(value.shape[1:])
            ):
                full = existing.detach().clone()
            else:
                shape = (batch_size, *value.shape[1:])
                if name == "hatc_belief":
                    full = belief.to(dtype=value.dtype)
                elif name == "market_previous_root":
                    full = state.get("market_previous_root", belief).detach().clone()
                elif name in undefined_float and value.is_floating_point():
                    full = torch.full(
                        shape,
                        float("nan"),
                        device=value.device,
                        dtype=value.dtype,
                    )
                elif name in minus_one_integer:
                    full = torch.full(
                        shape,
                        -1,
                        device=value.device,
                        dtype=value.dtype,
                    )
                elif name.startswith("accounting_") and value.is_floating_point():
                    full = torch.zeros(shape, device=value.device, dtype=value.dtype)
                else:
                    full = torch.zeros(shape, device=value.device, dtype=value.dtype)
            full.index_copy_(0, active_index, value)
            state[name] = full
        return state

    @staticmethod
    def _mark_fully_extinct_state(
        state: Dict[str, torch.Tensor],
        *,
        t: int,
        branch: int,
    ) -> Dict[str, torch.Tensor]:
        """Return an explicit terminal batch without fabricating a macro node."""
        del t, branch
        B = int(state["b"].shape[0])
        device = state["b"].device
        dtype = state["b"].dtype
        float_nan = torch.full((B,), float("nan"), device=device, dtype=dtype)
        zeros = torch.zeros(B, device=device, dtype=dtype)
        false = torch.zeros(B, device=device, dtype=torch.bool)
        state.update(
            {
                "hatcf": float_nan.clone(),
                "hatc_belief": float_nan.clone(),
                "hatc_cleared": float_nan.clone(),
                "hatc_cal": float_nan.clone(),
                "lnk_cal": float_nan.clone(),
                "market_root_found": false.clone(),
                "market_root_count": torch.zeros(B, device=device, dtype=torch.long),
                "market_root_residual": float_nan.clone(),
                "market_root_residual_normalized": float_nan.clone(),
                "market_root_C": float_nan.clone(),
                "market_root_K": float_nan.clone(),
                "market_root_refinement_rounds": torch.full(
                    (B,), -1, device=device, dtype=torch.long
                ),
                "market_root_policy_jump": false.clone(),
                "market_root_bracket_count": torch.zeros(B, device=device, dtype=torch.long),
                "market_root_operator_evaluations": torch.zeros(
                    B, device=device, dtype=torch.long
                ),
                "accounting_C_raw": zeros.clone(),
                "accounting_C_oper": zeros.clone(),
                "accounting_K_total": zeros.clone(),
                "accounting_Y_total": zeros.clone(),
                "accounting_I_oper": zeros.clone(),
                "accounting_I_entry": zeros.clone(),
                "accounting_Liquidation_total": zeros.clone(),
                "accounting_I_maintenance": zeros.clone(),
                "accounting_valid": false.clone(),
                "decision_firm_mass": zeros.clone(),
                "decision_default_mass": zeros.clone(),
                "decision_investment_mass": zeros.clone(),
                "decision_entry_accepted_mass": zeros.clone(),
                "decision_entry_potential_mass": zeros.clone(),
                "decision_survival_mass": zeros.clone(),
                "decision_entry_transport_mass": zeros.clone(),
                "decision_transport_mass": zeros.clone(),
                "decision_transport_atom_count": torch.zeros(
                    B, device=device, dtype=torch.long
                ),
                "candidate_commit_parity_passed": false.clone(),
                "economic_node_status": torch.full(
                    (B,),
                    ECONOMIC_NODE_EXTINCT_BEFORE_NODE,
                    device=device,
                    dtype=torch.long,
                ),
                "economic_state_defined": false.clone(),
            }
        )
        return state

    @torch.inference_mode()
    def clear_state(
        self,
        sim: Any,
        state: Dict[str, torch.Tensor],
        candidates: EntryCandidates,
        *,
        t: int,
        branch: int,
    ) -> Dict[str, torch.Tensor]:
        if self.config.solver_mode == FC1_BOUNDED_IMPROVEMENT_PILOT:
            return self._clear_state_bounded_improvement(
                sim, state, candidates, t=t, branch=branch
            )
        return self._clear_state_strict(sim, state, candidates, t=t, branch=branch)

    @torch.inference_mode()
    def _clear_state_bounded_improvement(
        self,
        sim: Any,
        state: Dict[str, torch.Tensor],
        candidates: EntryCandidates,
        *,
        t: int,
        branch: int,
    ) -> Dict[str, torch.Tensor]:
        """Bounded true-operator correction around the current FC1 forecast."""

        started = time.perf_counter()
        B = int(state["b"].shape[0])
        device = state["b"].device
        dtype = state["b"].dtype
        h_fc1 = state["hatcf"].detach().clone()
        active = state["alive"].any(dim=1) | candidates.mask.any(dim=1)
        if not bool(active.all()):
            active_index = torch.nonzero(active, as_tuple=False).reshape(-1)
            if active_index.numel() == 0:
                return self._mark_fully_extinct_state(state, t=t, branch=branch)
            active_state = {
                name: value.index_select(0, active_index)
                if torch.is_tensor(value)
                and value.ndim > 0
                and int(value.shape[0]) == B
                else value
                for name, value in state.items()
            }
            active_candidates = EntryCandidates(
                **{
                    name: getattr(candidates, name).index_select(0, active_index)
                    for name in candidates.__dataclass_fields__
                }
            )
            cleared = self._clear_state_bounded_improvement(
                sim,
                active_state,
                active_candidates,
                t=t,
                branch=branch,
            )
            return self._merge_active_paths(state, cleared, active_index)

        # Evaluation 1: FC1 baseline.  Detail is retained only for matched
        # policy-movement diagnostics; the accepted record is reconstructed
        # once at the final selected h below.
        baseline = self.evaluate(
            sim, state, candidates, h_fc1[:, None], detail=True
        )
        base_residual = baseline["G_normalized"][:, 0].abs()
        base_feasible = baseline["resource_feasible"][:, 0]
        base_c = baseline["C_raw"][:, 0]
        base_k = baseline["K_total"][:, 0]
        safe_ratio = torch.where(
            base_feasible,
            base_c.to(torch.float64) / base_k.to(torch.float64),
            torch.ones_like(base_c, dtype=torch.float64),
        )
        h_implied = torch.log(safe_ratio.clamp_min(1.0e-300)).to(dtype)
        alphas = torch.as_tensor(
            self.config.fast_alphas, device=device, dtype=dtype
        )
        proposed_h = h_fc1[:, None] + alphas[None, :] * (
            h_implied - h_fc1
        )[:, None]

        # Evaluations 2..(1+A): all first-round candidates are one batched
        # operator call, while operator_evaluations records the true B*A work.
        proposed = self.evaluate(sim, state, candidates, proposed_h, detail=False)
        all_h = torch.cat((h_fc1[:, None], proposed_h), dim=1)
        all_residual = torch.cat(
            (base_residual[:, None], proposed["G_normalized"].abs()), dim=1
        )
        all_feasible = torch.cat(
            (base_feasible[:, None], proposed["resource_feasible"]), dim=1
        )
        score = torch.where(
            all_feasible,
            all_residual,
            torch.full_like(all_residual, float("inf")),
        )
        best_column = torch.argmin(score, dim=1)
        row = torch.arange(B, device=device)
        selected_h = all_h[row, best_column]
        selected_residual = score[row, best_column]
        selected_c = torch.cat(
            (base_c[:, None], proposed["C_raw"]), dim=1
        )[row, best_column]
        selected_k = torch.cat(
            (base_k[:, None], proposed["K_total"]), dim=1
        )[row, best_column]
        feasible_any = all_feasible.any(dim=1)
        noise = float(self.config.fast_improvement_noise_tolerance)
        improved = feasible_any & (
            (~base_feasible) | (selected_residual + noise < base_residual)
        )

        evaluations = torch.full(
            (B,), 1 + len(self.config.fast_alphas),
            device=device,
            dtype=torch.long,
        )
        # At most two default local updates are possible under the hard budget
        # of eight because final decision-record reconstruction counts too.
        for _round in range(self.config.fast_local_correction_rounds):
            local_implied = torch.log(
                (selected_c.to(torch.float64) / selected_k.to(torch.float64))
                .clamp_min(1.0e-300)
            ).to(dtype)
            local = self.evaluate(
                sim, state, candidates, local_implied[:, None], detail=False
            )
            evaluations += 1
            local_residual = local["G_normalized"][:, 0].abs()
            local_feasible = local["resource_feasible"][:, 0]
            take = local_feasible & (
                (~feasible_any) | (local_residual + noise < selected_residual)
            )
            selected_h = torch.where(take, local_implied, selected_h)
            selected_residual = torch.where(
                take, local_residual, selected_residual
            )
            selected_c = torch.where(take, local["C_raw"][:, 0], selected_c)
            selected_k = torch.where(take, local["K_total"][:, 0], selected_k)
            feasible_any |= local_feasible
            improved |= take

        if bool((~feasible_any).any()):
            bad = torch.nonzero(~feasible_any, as_tuple=False).reshape(-1)
            raise MarketClearingError(
                "bounded improvement found no feasible economic candidate; "
                f"t={t}, branch={branch}, paths={bad.detach().cpu().tolist()[:20]}"
            )

        # Final evaluation is deliberately counted.  It creates the exact
        # immutable decision record used by commit/transport; no interpolated
        # candidate is ever written into Macro data.
        final = self.evaluate(
            sim, state, candidates, selected_h[:, None], detail=True
        )
        evaluations += 1
        if bool((evaluations > self.config.fast_max_evaluations).any()):
            raise RuntimeError("bounded improvement exceeded its G-evaluation budget")
        final_feasible = final["resource_feasible"][:, 0]
        if not bool(final_feasible.all()):
            raise MarketClearingError(
                "bounded improvement final decision reconstruction became infeasible"
            )
        final_g = final["G"][:, 0]
        final_normalized = final["G_normalized"][:, 0]
        final_c = final["C_raw"][:, 0]
        final_k = final["K_total"][:, 0]
        h_cal = torch.log(
            (final_c.to(torch.float64) / final_k.to(torch.float64))
            .clamp_min(1.0e-300)
        ).to(dtype)
        strict = final_normalized.abs() <= self.config.residual_tolerance
        improved = improved & (
            (~base_feasible) | (final_normalized.abs() + noise < base_residual)
        )
        status = torch.full(
            (B,),
            FAST_STATUS_BASELINE_NOT_IMPROVED,
            device=device,
            dtype=torch.long,
        )
        status = torch.where(
            improved,
            torch.full_like(status, FAST_STATUS_IMPROVED_APPROXIMATE),
            status,
        )
        status = torch.where(
            strict,
            torch.full_like(status, FAST_STATUS_STRICT_ROOT_FOUND),
            status,
        )

        self._commit_selected_decision_record(
            sim, state, candidates, selected_h, final, t=t, branch=branch
        )
        baseline_i = baseline["bar_i_firm"][:, 0] >= 0.5
        final_i = final["bar_i_firm"][:, 0] >= 0.5
        baseline_z = baseline["bar_z_firm"][:, 0] >= 0.5
        final_z = final["bar_z_firm"][:, 0] >= 0.5
        policy_slots = max(int(final_i.shape[1]), 1)
        investment_flip = (baseline_i != final_i).sum(dim=1).to(dtype) / policy_slots
        default_flip = (baseline_z != final_z).sum(dim=1).to(dtype) / policy_slots
        entry_slots = max(int(final["accepted_entry"].shape[2]), 1)
        entry_flip = (
            baseline["accepted_entry"][:, 0] != final["accepted_entry"][:, 0]
        ).sum(dim=1).to(dtype) / entry_slots

        state["h_fc1_original"] = h_fc1
        state["h_policy_input"] = selected_h.detach()
        state["h_cal_from_ledger"] = h_cal.detach()
        state["hatc_belief"] = h_fc1
        state["hatc_cleared"] = selected_h.detach()
        state["hatc_cal"] = h_cal.detach()
        state["hatcf"] = selected_h.detach()
        state["market_previous_root"] = selected_h.detach()
        state["market_root_found"] = strict
        state["market_root_count"] = strict.to(torch.long)
        state["market_root_residual"] = final_g.to(dtype)
        state["market_root_residual_normalized"] = final_normalized.to(dtype)
        state["market_root_C"] = final_c.to(dtype)
        state["market_root_K"] = final_k.to(dtype)
        state["market_root_refinement_rounds"] = torch.full(
            (B,),
            self.config.fast_local_correction_rounds,
            device=device,
            dtype=torch.long,
        )
        state["market_root_policy_jump"] = (
            (investment_flip > 0) | (default_flip > 0) | (entry_flip > 0)
        )
        state["market_investment_flip_vs_belief"] = investment_flip
        state["market_default_flip_vs_belief"] = default_flip
        state["market_entry_flip_vs_belief"] = entry_flip
        state["market_root_bracket_count"] = torch.zeros(
            B, device=device, dtype=torch.long
        )
        state["market_root_operator_evaluations"] = evaluations
        state["fast_solver_status"] = status
        state["baseline_G"] = baseline["G"][:, 0].to(dtype)
        state["corrected_G"] = final_g.to(dtype)
        state["baseline_normalized_residual"] = baseline[
            "G_normalized"
        ][:, 0].abs().to(dtype)
        state["corrected_normalized_residual"] = final_normalized.abs().to(dtype)
        state["improvement_fraction"] = torch.where(
            base_residual > noise,
            (base_residual - final_normalized.abs()) / base_residual,
            torch.zeros_like(base_residual),
        ).to(dtype)
        state["h_policy_minus_h_cal"] = (selected_h - h_cal).to(dtype)
        state["G_evaluation_count"] = evaluations
        state["economic_node_status"] = torch.full(
            (B,), ECONOMIC_NODE_ACTIVE, device=device, dtype=torch.long
        )
        state["economic_state_defined"] = torch.ones(
            B, device=device, dtype=torch.bool
        )
        for path in range(B):
            self.root_records.append(
                {
                    "t": int(t),
                    "branch": int(branch),
                    "path": int(path),
                    "solver_mode": FC1_BOUNDED_IMPROVEMENT_PILOT,
                    "root_status": FAST_STATUS_LABELS[int(status[path])],
                    "h_fc1_original": float(h_fc1[path]),
                    "h_policy_input": float(selected_h[path]),
                    "h_cal_from_ledger": float(h_cal[path]),
                    "baseline_G": float(baseline["G"][path, 0]),
                    "corrected_G": float(final_g[path]),
                    "baseline_normalized_residual": float(base_residual[path]),
                    "corrected_normalized_residual": float(
                        final_normalized[path].abs()
                    ),
                    "improvement_fraction": float(
                        state["improvement_fraction"][path]
                    ),
                    "G_evaluation_count": int(evaluations[path]),
                }
            )
        self.node_count += B
        self.root_search_seconds += time.perf_counter() - started
        return state

    @torch.inference_mode()
    def _clear_state_strict(
        self,
        sim: Any,
        state: Dict[str, torch.Tensor],
        candidates: EntryCandidates,
        *,
        t: int,
        branch: int,
    ) -> Dict[str, torch.Tensor]:
        """Solve all active paths, select a deterministic root, and set ``hatcf``."""
        started = time.perf_counter()
        B = int(state["b"].shape[0])
        device = state["b"].device
        belief = state["hatcf"].detach().clone()
        active = state["alive"].any(dim=1) | candidates.mask.any(dim=1)
        if not bool(active.all()):
            active_index = torch.nonzero(active, as_tuple=False).reshape(-1)
            if active_index.numel() == 0:
                return self._mark_fully_extinct_state(state, t=t, branch=branch)
            active_state = {
                name: value.index_select(0, active_index)
                if (
                    torch.is_tensor(value)
                    and value.ndim > 0
                    and int(value.shape[0]) == B
                )
                else value
                for name, value in state.items()
            }
            active_candidates = EntryCandidates(
                **{
                    name: getattr(candidates, name).index_select(0, active_index)
                    for name in candidates.__dataclass_fields__
                }
            )
            cleared = self._clear_state_strict(
                sim,
                active_state,
                active_candidates,
                t=t,
                branch=branch,
            )
            return self._merge_active_paths(state, cleared, active_index)
        h_coarse = self._support(state)
        coarse = self.evaluate(sim, state, candidates, h_coarse)
        crossings = self._crossings(coarse)
        bracket_count = torch.bincount(crossings[:, 0], minlength=B)
        evaluations = torch.full(
            (B,), self.config.coarse_points, device=device, dtype=torch.long
        )

        candidate_parent: list[torch.Tensor] = []
        candidate_h: list[torch.Tensor] = []
        candidate_round: list[torch.Tensor] = []
        candidate_jump: list[torch.Tensor] = []
        candidate_prequalified: list[torch.Tensor] = []
        exact = torch.nonzero(
            coarse["resource_feasible"]
            & (coarse["G_normalized"].abs() <= self._root_acceptance_tolerance),
            as_tuple=False,
        )
        if exact.numel():
            candidate_parent.append(exact[:, 0])
            candidate_h.append(coarse["h"][exact[:, 0], exact[:, 1]])
            candidate_round.append(torch.zeros(len(exact), device=device, dtype=torch.long))
            candidate_jump.append(torch.zeros(len(exact), device=device, dtype=torch.bool))
            candidate_prequalified.append(
                torch.ones(len(exact), device=device, dtype=torch.bool)
            )

        active_parent = crossings[:, 0]
        active_left = coarse["h"][crossings[:, 0], crossings[:, 1]]
        active_right = coarse["h"][crossings[:, 0], crossings[:, 1] + 1]
        unresolved_discontinuity = torch.zeros(B, device=device, dtype=torch.bool)
        discontinuity_bracket_count = torch.zeros(
            B, device=device, dtype=torch.long
        )
        unresolved_bracket_count = torch.zeros(B, device=device, dtype=torch.long)
        alpha = torch.linspace(
            0.0,
            1.0,
            self.config.refinement_points,
            device=device,
            dtype=belief.dtype,
        )
        for round_index in range(1, self.config.max_refinement_rounds + 1):
            if active_parent.numel() == 0:
                break
            gathered_state = {
                name: value.index_select(0, active_parent)
                if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == B
                else value
                for name, value in state.items()
            }
            gathered_candidates = EntryCandidates(
                **{
                    name: getattr(candidates, name).index_select(0, active_parent)
                    for name in candidates.__dataclass_fields__
                }
            )
            h_refined = active_left[:, None] + (
                active_right - active_left
            )[:, None] * alpha[None, :]
            refined = self.evaluate(
                sim, gathered_state, gathered_candidates, h_refined
            )
            evaluations.scatter_add_(
                0,
                active_parent,
                torch.full_like(active_parent, self.config.refinement_points),
            )
            feasible_residual = torch.where(
                refined["resource_feasible"],
                refined["G_normalized"].abs(),
                torch.full_like(refined["G_normalized"], float("inf")),
            )
            best_residual, best_column = feasible_residual.min(dim=1)
            small = best_residual <= self._root_acceptance_tolerance
            if bool(small.any()):
                rows = torch.nonzero(small, as_tuple=False).reshape(-1)
                candidate_parent.append(active_parent[rows])
                candidate_h.append(refined["h"][rows, best_column[rows]])
                candidate_round.append(
                    torch.full((len(rows),), round_index, device=device, dtype=torch.long)
                )
                candidate_jump.append(torch.zeros(len(rows), device=device, dtype=torch.bool))
                candidate_prequalified.append(
                    torch.ones(len(rows), device=device, dtype=torch.bool)
                )

            sub = self._crossings(refined)
            if sub.numel():
                sub = sub[~small[sub[:, 0]]]
            if sub.numel() == 0:
                unresolved_parent = active_parent[~small]
                unresolved_discontinuity[unresolved_parent] = True
                unresolved_bracket_count.scatter_add_(
                    0, unresolved_parent, torch.ones_like(unresolved_parent)
                )
                active_parent = torch.empty(0, device=device, dtype=torch.long)
                break
            represented = torch.bincount(sub[:, 0], minlength=len(active_parent)) > 0
            missing = (~small) & (~represented)
            missing_parent = active_parent[missing]
            unresolved_discontinuity[missing_parent] = True
            unresolved_bracket_count.scatter_add_(
                0, missing_parent, torch.ones_like(missing_parent)
            )
            row = sub[:, 0]
            col = sub[:, 1]
            new_parent = active_parent[row]
            new_left = refined["h"][row, col]
            new_right = refined["h"][row, col + 1]
            terminal = (
                (new_right - new_left <= self.config.width_tolerance)
                | (round_index == self.config.max_refinement_rounds)
            )
            if bool(terminal.any()):
                terminal_row = row[terminal]
                terminal_col = col[terminal]
                g0 = refined["G"][terminal_row, terminal_col]
                g1 = refined["G"][terminal_row, terminal_col + 1]
                x0 = new_left[terminal]
                x1 = new_right[terminal]
                proposed = x0 - g0 * (x1 - x0) / (g1 - g0)
                mass_jump = torch.zeros_like(g0, dtype=torch.bool)
                continuous_mass = refined.get("policy_mass_continuous")
                policy_mass_is_continuous = (
                    torch.is_tensor(continuous_mass)
                    and bool(continuous_mass.to(torch.bool).all())
                )
                if not policy_mass_is_continuous:
                    for mass_name in ("investment_mass", "default_mass", "entry_mass"):
                        if mass_name in refined:
                            mass_jump |= (
                                refined[mass_name][terminal_row, terminal_col]
                                - refined[mass_name][terminal_row, terminal_col + 1]
                            ).abs() > 1.0e-10
                jump = (
                    (refined["investment_count"][terminal_row, terminal_col]
                     != refined["investment_count"][terminal_row, terminal_col + 1])
                    | (refined["default_count"][terminal_row, terminal_col]
                       != refined["default_count"][terminal_row, terminal_col + 1])
                    | (refined["entry_count"][terminal_row, terminal_col]
                       != refined["entry_count"][terminal_row, terminal_col + 1])
                    | mass_jump
                )
                jump_parent = new_parent[terminal][jump]
                discontinuity_bracket_count.scatter_add_(
                    0, jump_parent, torch.ones_like(jump_parent)
                )
                candidate_parent.append(new_parent[terminal])
                candidate_h.append(proposed)
                candidate_round.append(
                    torch.full(
                        (int(terminal.sum()),),
                        round_index,
                        device=device,
                        dtype=torch.long,
                    )
                )
                candidate_jump.append(jump)
                candidate_prequalified.append(
                    torch.zeros(int(terminal.sum()), device=device, dtype=torch.bool)
                )
            keep = ~terminal
            active_parent = new_parent[keep]
            active_left = new_left[keep]
            active_right = new_right[keep]

        selected_h = torch.full_like(belief, float("nan"))
        root_count = torch.zeros(B, device=device, dtype=torch.long)
        selected_round = torch.full((B,), -1, device=device, dtype=torch.long)
        selected_jump = torch.zeros(B, device=device, dtype=torch.bool)
        selected_branch = torch.zeros(B, device=device, dtype=torch.long)
        selected_residual = torch.full_like(belief, float("nan"))
        selected_normalized = torch.full_like(belief, float("nan"))
        selected_c = torch.full_like(belief, float("nan"))
        selected_k = torch.full_like(belief, float("nan"))
        candidate_rejection_reasons: list[list[str]] = [[] for _ in range(B)]
        if candidate_parent:
            parents = torch.cat(candidate_parent)
            values = torch.cat(candidate_h)
            rounds = torch.cat(candidate_round)
            jumps = torch.cat(candidate_jump)
            prequalified = torch.cat(candidate_prequalified)
            gathered_state = {
                name: value.index_select(0, parents)
                if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == B
                else value
                for name, value in state.items()
            }
            gathered_candidates = EntryCandidates(
                **{
                    name: getattr(candidates, name).index_select(0, parents)
                    for name in candidates.__dataclass_fields__
                }
            )
            verified = self.evaluate(
                sim, gathered_state, gathered_candidates, values[:, None], detail=True
            )
            evaluations.scatter_add_(0, parents, torch.ones_like(parents))
            replay_tolerance = torch.where(
                prequalified,
                torch.full_like(
                    verified["G_normalized"][:, 0],
                    float(self.config.residual_tolerance),
                ),
                torch.full_like(
                    verified["G_normalized"][:, 0],
                    self._root_acceptance_tolerance,
                ),
            )
            valid = (
                verified["resource_feasible"][:, 0]
                & (verified["G_normalized"][:, 0].abs() <= replay_tolerance)
            )
            rejected = ~valid
            if bool(rejected.any()):
                rejected_positions = torch.nonzero(
                    rejected, as_tuple=False
                ).reshape(-1)
                verified_finite = verified["finite"][:, 0]
                verified_feasible = verified["resource_feasible"][:, 0]
                for position in rejected_positions.detach().cpu().tolist():
                    parent = int(parents[position])
                    if not bool(verified_finite[position]):
                        reason = "NONFINITE_CANDIDATE"
                    elif not bool(verified_feasible[position]):
                        reason = "ECONOMICALLY_INFEASIBLE_CANDIDATE"
                    else:
                        reason = "RESIDUAL_TOLERANCE_FAILED"
                    candidate_rejection_reasons[parent].append(reason)

                    # A terminal policy jump is a discontinuity only when the
                    # proposed candidate itself is finite.  NaN/Inf candidates
                    # retain their numerical-failure classification instead of
                    # being relabelled as an economic discontinuity.
                    if bool(jumps[position]) and bool(verified_finite[position]):
                        unresolved_discontinuity[parent] = True
                        unresolved_bracket_count[parent] += 1
            previous = state.get("market_previous_root", belief)
            for parent in range(B):
                positions = torch.nonzero(
                    valid & (parents == parent), as_tuple=False
                ).reshape(-1)
                if positions.numel() == 0:
                    continue
                ordered = positions[torch.argsort(values[positions])]
                keep = [int(ordered[0])]
                for position in ordered[1:]:
                    position_int = int(position)
                    same_h = abs(float(values[position_int] - values[keep[-1]])) <= (
                        10.0 * self.config.width_tolerance
                    )
                    same_policy_region = all(
                        int(verified[name][position_int, 0])
                        == int(verified[name][keep[-1], 0])
                        for name in (
                            "investment_count", "default_count", "entry_count"
                        )
                    )
                    if not (same_h and same_policy_region):
                        keep.append(position_int)
                positions = torch.tensor(keep, device=device, dtype=torch.long)
                root_count[parent] = len(keep)
                distance = (values[positions] - previous[parent]).abs()
                # values are sorted, so argmin implements lower-h deterministic ties.
                selected_index = torch.argmin(distance)
                chosen = positions[selected_index]
                selected_branch[parent] = selected_index
                selected_h[parent] = values[chosen]
                selected_round[parent] = rounds[chosen]
                selected_jump[parent] = jumps[chosen]
                replay_c = verified["C_raw"][chosen, 0].to(torch.float64)
                replay_k = verified["K_total"][chosen, 0].to(torch.float64)
                replay_h = values[chosen].to(torch.float64)
                replay_g = torch.exp(replay_h) * replay_k - replay_c
                replay_scale = torch.maximum(
                    torch.ones_like(replay_g),
                    torch.maximum(replay_c.abs(), replay_k.abs()),
                )
                selected_residual[parent] = replay_g.to(selected_residual.dtype)
                selected_normalized[parent] = (
                    replay_g / replay_scale
                ).to(selected_normalized.dtype)
                selected_c[parent] = verified["C_raw"][chosen, 0]
                selected_k[parent] = verified["K_total"][chosen, 0]
                path_id = state.get("transition_path_id")
                self.root_records.append(
                    {
                        "t": int(t),
                        "branch": int(branch),
                        "path": int(
                            parent if path_id is None else path_id[parent]
                        ),
                        "h_belief": float(belief[parent]),
                        "all_valid_roots": [
                            float(values[position]) for position in positions
                        ],
                        "selected_root": float(values[chosen]),
                        "selection_reference": float(previous[parent]),
                        "selection_rule": (
                            "closest_previous_root_then_lower_h_tie"
                        ),
                        "deduplication_rule": (
                            "h_within_10_width_tol_and_same_policy_count_region"
                        ),
                        "selected_branch_id": int(selected_index),
                        "root_count": len(keep),
                        "valid_root_count": len(keep),
                        "discontinuity_bracket_count": int(
                            discontinuity_bracket_count[parent]
                        ),
                        "unresolved_bracket_count": int(
                            unresolved_bracket_count[parent]
                        ),
                        "candidate_rejection_reason": list(
                            candidate_rejection_reasons[parent]
                        ),
                        "terminal_status": "ROOT_FOUND",
                        "candidate_prequalified_at_internal_tolerance": bool(
                            prequalified[chosen]
                        ),
                        "candidate_verification_tolerance": float(
                            replay_tolerance[chosen]
                        ),
                    }
                )

        fp64_replay_failed = torch.isfinite(selected_h) & (
            ~torch.isfinite(selected_normalized)
            | (selected_normalized.abs() > self.config.residual_tolerance)
        )
        no_root = ~torch.isfinite(selected_h) | fp64_replay_failed
        if bool(no_root.any()):
            bad = torch.nonzero(no_root, as_tuple=False).reshape(-1)
            statuses = []
            for parent in bad.detach().cpu().tolist():
                if fp64_replay_failed[parent]:
                    status = "FP64_REPLAY_RESIDUAL_FAILED"
                elif unresolved_discontinuity[parent]:
                    status = "DISCONTINUITY_CROSSING"
                elif bool((~coarse["finite"][parent]).any()):
                    status = "NONFINITE_OPERATOR"
                elif not bool(coarse["resource_feasible"][parent].any()):
                    status = "INFEASIBLE_CANDIDATES"
                else:
                    status = "NO_BRACKET_ON_TESTED_GRID"
                statuses.append(status)
                self.failure_records.append(
                    {
                        "t": int(t),
                        "branch": int(branch),
                        "path": int(parent),
                        "status": status,
                        "h_belief": float(belief[parent]),
                        "bracket_count": int(bracket_count[parent]),
                        "valid_root_count": int(root_count[parent]),
                        "discontinuity_bracket_count": int(
                            discontinuity_bracket_count[parent]
                        ),
                        "unresolved_bracket_count": int(
                            unresolved_bracket_count[parent]
                        ),
                        "candidate_rejection_reason": list(
                            candidate_rejection_reasons[parent]
                        ),
                        "terminal_status": status,
                        "selected_h": (
                            float(selected_h[parent])
                            if torch.isfinite(selected_h[parent]) else None
                        ),
                        "selected_normalized_residual": (
                            float(selected_normalized[parent])
                            if torch.isfinite(selected_normalized[parent]) else None
                        ),
                        "root_candidate_acceptance_tolerance": (
                            self._root_acceptance_tolerance
                        ),
                        "public_residual_tolerance": (
                            float(self.config.residual_tolerance)
                        ),
                    }
                )
            raise MarketClearingError(
                "strict market clearing found no valid root; "
                f"t={t}, branch={branch}, paths={bad.detach().cpu().tolist()[:20]}, "
                f"statuses={statuses[:20]}, "
                f"selected_normalized_residuals="
                f"{selected_normalized[bad].detach().cpu().tolist()[:20]}, "
                f"candidate_tolerance={self._root_acceptance_tolerance}, "
                f"public_tolerance={self.config.residual_tolerance}"
            )

        # Matched belief/root replay records whether resolving the market root
        # actually changed any discrete Firm or entry decision.
        root_detail = self.evaluate(
            sim, state, candidates, selected_h[:, None], detail=True
        )
        state["economic_node_status"] = torch.full(
            (B,), ECONOMIC_NODE_ACTIVE, device=device, dtype=torch.long
        )
        state["economic_state_defined"] = torch.ones(
            B, device=device, dtype=torch.bool
        )
        self._commit_selected_decision_record(
            sim,
            state,
            candidates,
            selected_h,
            root_detail,
            t=t,
            branch=branch,
        )
        belief_detail = self.evaluate(
            sim, state, candidates, belief[:, None], detail=True
        )
        root_i = root_detail["bar_i_firm"][:, 0] >= 0.5
        belief_i = belief_detail["bar_i_firm"][:, 0] >= 0.5
        root_z = root_detail["bar_z_firm"][:, 0] >= 0.5
        belief_z = belief_detail["bar_z_firm"][:, 0] >= 0.5
        policy_slots = max(int(root_i.shape[1]), 1)
        investment_flip_share = (
            (root_i != belief_i).sum(dim=1).to(belief.dtype) / policy_slots
        )
        default_flip_share = (
            (root_z != belief_z).sum(dim=1).to(belief.dtype) / policy_slots
        )
        entry_slots = max(int(root_detail["accepted_entry"].shape[2]), 1)
        entry_flip_share = (
            (
                root_detail["accepted_entry"][:, 0]
                != belief_detail["accepted_entry"][:, 0]
            )
            .sum(dim=1)
            .to(belief.dtype)
            / entry_slots
        )
        previous_branch = state.get(
            "market_previous_branch_id", torch.zeros_like(selected_branch)
        )

        state["hatc_belief"] = belief
        state["hatc_cleared"] = selected_h.detach()
        state["hatcf"] = selected_h.detach()
        state["market_previous_root"] = selected_h.detach()
        state["market_previous_branch_id"] = selected_branch.detach()
        state["market_root_branch_id"] = selected_branch
        state["market_root_branch_switch"] = selected_branch != previous_branch
        state["market_root_found"] = torch.ones(B, device=device, dtype=torch.bool)
        state["market_root_count"] = root_count
        state["market_root_residual"] = selected_residual
        state["market_root_residual_normalized"] = selected_normalized
        state["market_root_C"] = selected_c
        state["market_root_K"] = selected_k
        state["market_root_refinement_rounds"] = selected_round
        state["market_root_policy_jump"] = selected_jump | (
            (investment_flip_share > 0)
            | (default_flip_share > 0)
            | (entry_flip_share > 0)
        )
        state["market_investment_flip_vs_belief"] = investment_flip_share
        state["market_default_flip_vs_belief"] = default_flip_share
        state["market_entry_flip_vs_belief"] = entry_flip_share
        state["market_root_bracket_count"] = bracket_count
        state["market_root_operator_evaluations"] = evaluations
        self.node_count += B
        self.multiple_root_nodes += int((root_count > 1).sum())
        self.root_search_seconds += time.perf_counter() - started
        return state

    def validate_committed(
        self,
        state: Dict[str, torch.Tensor],
        *,
        t: int,
        branch: int,
    ) -> None:
        """Require committed production accounting to reproduce the selected root."""
        required = ("accounting_K_total", "accounting_C_raw", "hatc_cleared")
        missing = [name for name in required if name not in state]
        if missing:
            raise MarketClearingError(
                f"committed market-cleared node lacks accounting fields: {missing}"
            )
        K = state["accounting_K_total"].to(torch.float64)
        C = state["accounting_C_raw"].to(torch.float64)
        h = state["hatc_cleared"].to(torch.float64)
        residual = torch.exp(h) * K - C
        scale = torch.maximum(torch.ones_like(residual), torch.maximum(K.abs(), C.abs()))
        normalized = residual / scale
        state["market_committed_residual"] = residual.detach().to(
            state["accounting_C_raw"].dtype
        )
        state["market_committed_residual_normalized"] = normalized.detach().to(
            state["accounting_C_raw"].dtype
        )
        cleared = state.get(
            "market_root_found",
            torch.ones_like(normalized, dtype=torch.bool),
        ).bool()
        accounting_valid = state.get(
            "accounting_valid", torch.ones_like(cleared, dtype=torch.bool)
        ).bool()
        cleared = cleared & accounting_valid
        invalid = cleared & (
            ~torch.isfinite(normalized)
            | (C <= 0.0)
            | (K <= 0.0)
            | (normalized.abs() > self.config.residual_tolerance)
        )
        if bool(invalid.any()):
            bad = torch.nonzero(invalid, as_tuple=False).reshape(-1)
            raise MarketClearingError(
                "market-clearing evaluator/production commit parity failed; "
                f"t={t}, branch={branch}, paths={bad.detach().cpu().tolist()[:20]}, "
                f"normalized_residual={normalized[bad].detach().cpu().tolist()[:20]}, "
                f"tolerance={self.config.residual_tolerance}"
            )
        record = state.get("_selected_decision_record")
        if record is not None:
            committed = {
                "C_raw": state["accounting_C_raw"],
                "C_oper": state["accounting_C_oper"],
                "K_total": state["accounting_K_total"],
                "Y_total": state["accounting_Y_total"],
                "I_oper_total": state["accounting_I_oper"],
                "I_entry_total": state["accounting_I_entry"],
                "Liquidation_total": state["accounting_Liquidation_total"],
                "I_maintenance_total": state["accounting_I_maintenance"],
            }
            try:
                assert_candidate_commit_parity(
                    record,
                    committed,
                    atol=5.0e-6,
                    rtol=5.0e-6,
                )
            except AssertionError as exc:
                raise MarketClearingError(
                    "selected DecisionRecord differs from committed ledger; "
                    f"t={t}, branch={branch}"
                ) from exc
            state["candidate_commit_parity_passed"] = torch.ones_like(
                cleared, dtype=torch.bool
            )
            validation_key = (
                record.snapshot_hash,
                record.candidate_h.detach().to(device="cpu").contiguous().numpy().tobytes(),
            )
            if validation_key not in self._validated_decision_records:
                self._validated_decision_records.add(validation_key)
                self.candidate_commit_parity_pass_count += int(cleared.sum())

    def summary(self) -> dict[str, Any]:
        return {
            "mode": MARKET_CLEARING_V1,
            "solver_mode": self.config.solver_mode,
            "VECTORIZED_EVALUATOR": True,
            "VECTORIZED_COARSE_SEARCH": True,
            "VECTORIZED_REFINEMENT": True,
            "ECONOMIC_EQUATIONS_CHANGED": False,
            "TRAINING_PERFORMED": False,
            "node_count": int(self.node_count),
            "multiple_root_nodes": int(self.multiple_root_nodes),
            "candidate_commit_parity_pass_count": int(
                self.candidate_commit_parity_pass_count
            ),
            "total_operator_evaluations": int(self.operator_evaluations),
            "network_forward_calls": int(self.network_forward_calls),
            "policy_forward_seconds": float(self.policy_forward_seconds),
            "accounting_seconds": float(self.accounting_seconds),
            "root_search_seconds": float(self.root_search_seconds),
            "failure_count": len(self.failure_records),
            "root_candidate_acceptance_tolerance": (
                self._root_acceptance_tolerance
            ),
            "root_records": list(self.root_records),
            "failure_records": list(self.failure_records),
            "config": {
                name: getattr(self.config, name)
                for name in self.config.__dataclass_fields__
            },
        }
