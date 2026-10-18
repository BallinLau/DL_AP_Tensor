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


MARKET_CLEARING_DISABLED = "disabled"
MARKET_CLEARING_V1 = "market_cleared_experimental_v1"
ROOT_ACCEPTANCE_SAFETY_FACTOR = 0.5


class MarketClearingError(RuntimeError):
    """Raised when strict experimental market clearing cannot commit a node."""


@dataclass(frozen=True)
class MarketClearingConfig:
    coarse_points: int = 10
    refinement_points: int = 20
    max_refinement_rounds: int = 4
    support_lower: float = -8.0
    support_upper: float = 2.0
    support_padding: float = 2.0
    residual_tolerance: float = 1.0e-5
    width_tolerance: float = 1.0e-5
    candidate_chunk_size: int = 262_144

    def validate(self) -> None:
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
        pieces = {name: [] for name in ("bar_i", "bar_z", "bp", "P0", "PI")}
        for start in range(0, states.shape[0], self.config.candidate_chunk_size):
            output = forward_policy_value_for_simulation(
                model, states[start : start + self.config.candidate_chunk_size]
            )
            for name in pieces:
                pieces[name].append(getattr(output, name).reshape(-1))
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
        bar_i = torch.where(birth, torch.zeros_like(policy["bar_i"]), policy["bar_i"])
        bar_z = policy["bar_z"]
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
        mask = alive.to(dtype)
        entry_spend = (
            candidates.entry_cost[:, None, :]
            * candidates.K_birth[:, None, :]
            * accepted.to(dtype)
        ).sum(dim=2)
        K_total = (capital * mask).sum(dim=2)
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
        result = {
            "h": h,
            "G": G,
            "G_normalized": G / scale,
            "C_raw": C_raw,
            "K_total": K_total,
            "resource_feasible": feasible,
            "finite": finite,
            "investment_count": hard_i.sum(dim=2),
            "default_count": hard_z.sum(dim=2),
            "entry_count": accepted.sum(dim=2),
            "firm_count": alive.sum(dim=2),
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
    def clear_state(
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
            bad = torch.nonzero(~active, as_tuple=False).reshape(-1)
            for parent in bad.detach().cpu().tolist():
                self.failure_records.append(
                    {
                        "t": int(t),
                        "branch": int(branch),
                        "path": int(parent),
                        "status": "NO_ACTIVE_FIRM_OR_ENTRY_CANDIDATE",
                        "h_belief": float(belief[parent]),
                        "bracket_count": 0,
                    }
                )
            raise MarketClearingError(
                "strict market clearing cannot define an economic node without "
                "an incumbent or entry candidate; "
                f"t={t}, branch={branch}, paths={bad.detach().cpu().tolist()[:20]}"
            )
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

        active_parent = crossings[:, 0]
        active_left = coarse["h"][crossings[:, 0], crossings[:, 1]]
        active_right = coarse["h"][crossings[:, 0], crossings[:, 1] + 1]
        unresolved_discontinuity = torch.zeros(B, device=device, dtype=torch.bool)
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

            sub = self._crossings(refined)
            if sub.numel():
                sub = sub[~small[sub[:, 0]]]
            if sub.numel() == 0:
                unresolved_discontinuity[active_parent[~small]] = True
                active_parent = torch.empty(0, device=device, dtype=torch.long)
                break
            represented = torch.bincount(sub[:, 0], minlength=len(active_parent)) > 0
            missing = (~small) & (~represented)
            unresolved_discontinuity[active_parent[missing]] = True
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
                jump = (
                    (refined["investment_count"][terminal_row, terminal_col]
                     != refined["investment_count"][terminal_row, terminal_col + 1])
                    | (refined["default_count"][terminal_row, terminal_col]
                       != refined["default_count"][terminal_row, terminal_col + 1])
                    | (refined["entry_count"][terminal_row, terminal_col]
                       != refined["entry_count"][terminal_row, terminal_col + 1])
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
        if candidate_parent:
            parents = torch.cat(candidate_parent)
            values = torch.cat(candidate_h)
            rounds = torch.cat(candidate_round)
            jumps = torch.cat(candidate_jump)
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
            valid = (
                verified["resource_feasible"][:, 0]
                & (
                    verified["G_normalized"][:, 0].abs()
                    <= self._root_acceptance_tolerance
                )
            )
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
                    }
                )

        fp64_replay_failed = torch.isfinite(selected_h) & (
            ~torch.isfinite(selected_normalized)
            | (selected_normalized.abs() > self._root_acceptance_tolerance)
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
                    }
                )
            raise MarketClearingError(
                "strict market clearing found no valid root; "
                f"t={t}, branch={branch}, paths={bad.detach().cpu().tolist()[:20]}, "
                f"statuses={statuses[:20]}"
            )

        # Matched belief/root replay records whether resolving the market root
        # actually changed any discrete Firm or entry decision.
        root_detail = self.evaluate(
            sim, state, candidates, selected_h[:, None], detail=True
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

    def summary(self) -> dict[str, Any]:
        return {
            "mode": MARKET_CLEARING_V1,
            "VECTORIZED_EVALUATOR": True,
            "VECTORIZED_COARSE_SEARCH": True,
            "VECTORIZED_REFINEMENT": True,
            "ECONOMIC_EQUATIONS_CHANGED": False,
            "TRAINING_PERFORMED": False,
            "node_count": int(self.node_count),
            "multiple_root_nodes": int(self.multiple_root_nodes),
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
