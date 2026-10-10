"""Shared economic entry primitives for simulation rollouts.

The functions in this module are deliberately independent of the simulation
storage layout.  They define the value-cost entry experiment's reference
scale, candidate draws, equity-value screen, resource account, and capital
decomposition without changing the policy/value training equations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .simulation_forward import forward_policy_value_for_simulation


ENTRY_SPEC_VERSION = "value_cost_v1"
INVESTMENT_COMPARE_SPEC_VERSION = "investment_compare_v1"


class EntryEconomicInfeasibilityError(RuntimeError):
    """The stated entry economy produced a finite but infeasible resource node."""


class EntryAccountingError(RuntimeError):
    """Independent entry/resource reconstruction disagrees with the node ledger."""


def entry_configuration_snapshot(
    *,
    entry_mode: str,
    entry_spec_version: str,
    entry_capital_ratio: float,
    entry_size_ratio: float,
    entry_cost_max: float,
    entry_dummy_i: float,
    entry_inference_chunk_size: int,
    entry_rng_seed: int,
    consumption_aggregation_mode: str,
    node_accounting_mode: str,
    transition_rng_mode: str,
    economic_config: Any,
) -> Dict[str, Any]:
    """Canonical entry configuration used by output/cache provenance."""
    snapshot = {
        "entry_mode": str(entry_mode),
        "entry_specification_version": str(entry_spec_version),
        "entry_capital_ratio": float(entry_capital_ratio),
        "entry_size_ratio": float(entry_size_ratio),
        "entry_cost_max": float(entry_cost_max),
        "entry_dummy_i": float(entry_dummy_i),
        "entry_inference_chunk_size": int(entry_inference_chunk_size),
        "entry_rng_seed": int(entry_rng_seed),
        "consumption_aggregation_mode": str(consumption_aggregation_mode),
        "node_accounting_mode": str(node_accounting_mode),
        "transition_rng_mode": str(transition_rng_mode),
        "reference_stage": "parent_pre_transition_alive_firms",
        "entry_timing": "child_node_creation_cost_paid_before_operation",
        "eta_information_timing": (
            "birth_eta_fixed_zero_and_birth_i_equals_entry_cost"
            if str(entry_mode) == "investment_compare"
            else "eta_and_i_drawn_after_entry_acceptance"
        ),
        "economic": {
            "zeta": float(economic_config.ZETA),
            "rho_z": float(economic_config.RHO_Z),
            "sigma_z": float(economic_config.SIGMA_Z),
            "zbar": float(economic_config.ZBAR),
            "i_threshold": float(economic_config.I_THRESHOLD),
            "delta": float(economic_config.DELTA),
            "phi": float(economic_config.PHI),
            "g": float(economic_config.G),
        },
    }
    if str(entry_mode) == "investment_compare":
        snapshot.update(
            {
                "entry_criterion": "PI_ge_P0",
                "birth_b": 0.0,
                "birth_eta": 0.0,
                "cost_distribution_source": "ordinary_investment_cost",
                "effective_entry_cost_max": float(entry_cost_max),
                "birth_i_equals_entry_cost": True,
                "birth_extra_expansion": False,
                "entry_dummy_i_effective": False,
            }
        )
    return snapshot


def entry_configuration_fingerprint(snapshot: Dict[str, Any]) -> str:
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EntryReference:
    reference_K: torch.Tensor
    reference_N: torch.Tensor
    mean_K_ref: torch.Tensor
    K_per_entrant: torch.Tensor
    potential_capital_nominal: torch.Tensor
    n_star: torch.Tensor
    extinct: torch.Tensor


@dataclass(frozen=True)
class EntryCandidates:
    counts: torch.Tensor
    mask: torch.Tensor
    z: torch.Tensor
    entry_cost: torch.Tensor
    K_birth: torch.Tensor
    potential_capital_nominal: torch.Tensor
    candidate_capital_realized: torch.Tensor


def make_entry_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def compute_entry_reference(
    K: torch.Tensor,
    alive: torch.Tensor,
    *,
    entry_capital_ratio: float,
    entry_size_ratio: float,
) -> EntryReference:
    if K.shape != alive.shape or K.ndim != 2:
        raise ValueError("K and alive must be matching two-dimensional tensors")
    if entry_capital_ratio < 0.0:
        raise ValueError("entry_capital_ratio must be nonnegative")
    if not 0.0 < entry_size_ratio <= 1.0:
        raise ValueError("entry_size_ratio must be in (0, 1]")
    if not torch.isfinite(K).all() or (K < 0.0).any():
        raise ValueError("reference firm capital must be finite and nonnegative")

    active_K = torch.where(alive, K, torch.zeros_like(K))
    reference_K = active_K.sum(dim=1)
    reference_N = alive.sum(dim=1).to(torch.long)
    extinct = reference_N == 0
    mean_K_ref = torch.where(
        extinct,
        torch.zeros_like(reference_K),
        reference_K / reference_N.to(reference_K.dtype).clamp_min(1.0),
    )
    K_per_entrant = float(entry_size_ratio) * mean_K_ref
    potential_capital_nominal = float(entry_capital_ratio) * reference_K
    n_star = torch.where(
        extinct | (K_per_entrant <= 0.0),
        torch.zeros_like(reference_K),
        potential_capital_nominal / K_per_entrant,
    )
    return EntryReference(
        reference_K=reference_K,
        reference_N=reference_N,
        mean_K_ref=mean_K_ref,
        K_per_entrant=K_per_entrant,
        potential_capital_nominal=potential_capital_nominal,
        n_star=n_star,
        extinct=extinct,
    )


def stochastic_round_counts(
    n_star: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    if n_star.ndim != 1 or not torch.isfinite(n_star).all() or (n_star < 0.0).any():
        raise ValueError("n_star must be a finite nonnegative vector")
    values = n_star.detach().to(device="cpu", dtype=torch.float64)
    base = torch.floor(values)
    fractional = values - base
    draws = torch.rand(values.shape, generator=generator, dtype=torch.float64)
    return (base + (draws < fractional).to(torch.float64)).to(
        device=n_star.device, dtype=torch.long
    )


def _cpu_randn(shape, generator: torch.Generator, *, device, dtype) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float64).to(
        device=device, dtype=dtype
    )


def _cpu_rand(shape, generator: torch.Generator, *, device, dtype) -> torch.Tensor:
    return torch.rand(shape, generator=generator, dtype=torch.float64).to(
        device=device, dtype=dtype
    )


def draw_value_cost_candidates(
    reference: EntryReference,
    *,
    rho_z: float,
    sigma_z: float,
    zbar: float,
    entry_cost_max: float,
    generator: torch.Generator,
) -> EntryCandidates:
    if entry_cost_max <= 0.0:
        raise ValueError("entry_cost_max must be positive")
    if abs(float(rho_z)) >= 1.0 or sigma_z < 0.0:
        raise ValueError("stationary z draw requires abs(rho_z)<1 and sigma_z>=0")
    counts = stochastic_round_counts(reference.n_star, generator=generator)
    max_count = int(counts.max().item()) if counts.numel() else 0
    n_paths = int(counts.numel())
    shape = (n_paths, max_count)
    positions = torch.arange(max_count, device=counts.device).unsqueeze(0)
    mask = positions < counts.unsqueeze(1)
    dtype = reference.reference_K.dtype
    device = reference.reference_K.device
    stationary_std = float(sigma_z) / max((1.0 - float(rho_z) ** 2) ** 0.5, 1e-12)
    z = float(zbar) + stationary_std * _cpu_randn(
        shape, generator, device=device, dtype=dtype
    )
    entry_cost = float(entry_cost_max) * _cpu_rand(
        shape, generator, device=device, dtype=dtype
    )
    K_birth = reference.K_per_entrant.unsqueeze(1).expand(shape).clone()
    zero = torch.zeros((), device=device, dtype=dtype)
    z = torch.where(mask, z, zero)
    entry_cost = torch.where(mask, entry_cost, zero)
    K_birth = torch.where(mask, K_birth, zero)
    candidate_capital_realized = K_birth.sum(dim=1)
    return EntryCandidates(
        counts=counts,
        mask=mask,
        z=z,
        entry_cost=entry_cost,
        K_birth=K_birth,
        potential_capital_nominal=reference.potential_capital_nominal,
        candidate_capital_realized=candidate_capital_realized,
    )


def evaluate_entry_cutoff(
    policy_value_model: Any,
    candidates: EntryCandidates,
    *,
    x: torch.Tensor,
    hatcf: torch.Tensor,
    lnkf: torch.Tensor,
    zeta: float,
    dummy_i: float = 0.0,
    chunk_size: int = 65536,
) -> Dict[str, torch.Tensor]:
    """Evaluate the economic-scale equity cutoff before entrant eta/i draws."""
    if policy_value_model is None:
        raise RuntimeError("entry_mode='value_cost' requires policy_value model")
    if not 0.0 <= zeta <= 1.0:
        raise ValueError("zeta must be in [0, 1]")
    if chunk_size <= 0:
        raise ValueError("entry inference chunk_size must be positive")
    if candidates.mask.numel() == 0:
        empty = candidates.z.clone()
        return {
            "P_eta0": empty,
            "P_eta1": empty,
            "entry_cutoff": empty,
            "entry_net_value_per_capital": empty,
            "entry_net_value_total": empty,
            "accepted": candidates.mask.clone(),
        }

    path_index, candidate_index = torch.nonzero(
        candidates.mask, as_tuple=True
    )
    z = candidates.z[path_index, candidate_index]
    n = int(z.numel())
    b = torch.zeros_like(z)
    i_dummy = torch.full_like(z, float(dummy_i))
    x_rows = x[path_index]
    hatcf_rows = hatcf[path_index]
    lnkf_rows = lnkf[path_index]
    p0_parts = []
    p1_parts = []
    modes = {
        module: module.training
        for module in policy_value_model.modules()
    }
    parameter_versions = [parameter._version for parameter in policy_value_model.parameters()]
    buffer_versions = [buffer._version for buffer in policy_value_model.buffers()]
    try:
        policy_value_model.eval()
        with torch.no_grad():
            for start in range(0, n, int(chunk_size)):
                stop = min(start + int(chunk_size), n)
                common = torch.stack(
                    [
                        b[start:stop],
                        z[start:stop],
                        torch.zeros_like(z[start:stop]),
                        i_dummy[start:stop],
                        x_rows[start:stop],
                        hatcf_rows[start:stop],
                        lnkf_rows[start:stop],
                    ],
                    dim=1,
                )
                eta0 = common
                eta1 = common.clone()
                eta1[:, 2] = 1.0
                p0_parts.append(
                    forward_policy_value_for_simulation(policy_value_model, eta0).P.reshape(-1)
                )
                p1_parts.append(
                    forward_policy_value_for_simulation(policy_value_model, eta1).P.reshape(-1)
                )
    finally:
        for module, was_training in modes.items():
            module.train(was_training)
    for before, after in zip(parameter_versions, policy_value_model.parameters()):
        if before != after._version:
            raise RuntimeError("entry cutoff forward modified policy model parameters")
    for before, after in zip(buffer_versions, policy_value_model.buffers()):
        if before != after._version:
            raise RuntimeError("entry cutoff forward modified policy model buffers")

    p_eta0_flat = torch.cat(p0_parts)
    p_eta1_flat = torch.cat(p1_parts)
    if not torch.isfinite(p_eta0_flat).all() or not torch.isfinite(p_eta1_flat).all():
        raise FloatingPointError("entry equity cutoff contains non-finite P values")
    cutoff_flat = (1.0 - float(zeta)) * p_eta0_flat + float(zeta) * p_eta1_flat
    cost_flat = candidates.entry_cost[path_index, candidate_index]
    k_flat = candidates.K_birth[path_index, candidate_index]
    net_flat = cutoff_flat - cost_flat
    accepted_flat = net_flat > 0.0

    def scatter(values: torch.Tensor, *, fill: float = 0.0) -> torch.Tensor:
        result = torch.full_like(candidates.z, float(fill))
        result[path_index, candidate_index] = values
        return result

    accepted = torch.zeros_like(candidates.mask)
    accepted[path_index, candidate_index] = accepted_flat
    return {
        "P_eta0": scatter(p_eta0_flat),
        "P_eta1": scatter(p_eta1_flat),
        "entry_cutoff": scatter(cutoff_flat),
        "entry_net_value_per_capital": scatter(net_flat),
        "entry_net_value_total": scatter(k_flat * net_flat),
        "accepted": accepted,
    }


def evaluate_investment_compare_entry(
    policy_value_model: Any,
    candidates: EntryCandidates,
    *,
    x: torch.Tensor,
    hatcf: torch.Tensor,
    lnkf: torch.Tensor,
    chunk_size: int = 65536,
) -> Dict[str, torch.Tensor]:
    """Screen entrants by the existing physical-value investment comparison.

    Each valid candidate is evaluated at ``b=0``, ``eta=0`` and
    ``i=entry_cost``. The cost is already an input to ``PI`` and is therefore
    never subtracted again from ``PI-P0``.
    """
    if policy_value_model is None:
        raise RuntimeError(
            "entry_mode='investment_compare' requires a policy_value model"
        )
    if chunk_size <= 0:
        raise ValueError("entry inference chunk_size must be positive")
    if candidates.mask.numel() == 0:
        empty = candidates.z.clone()
        return {
            "entry_P0": empty,
            "entry_PI": empty,
            "entry_value_gap": empty,
            "accepted": candidates.mask.clone(),
        }
    value_forward = getattr(policy_value_model, "forward_value_components", None)
    if not callable(value_forward):
        raise RuntimeError(
            "entry_mode='investment_compare' requires "
            "policy_value.forward_value_components()"
        )

    path_index, candidate_index = torch.nonzero(candidates.mask, as_tuple=True)
    z = candidates.z[path_index, candidate_index]
    cost = candidates.entry_cost[path_index, candidate_index]
    n = int(z.numel())
    x_rows = x[path_index]
    hatcf_rows = hatcf[path_index]
    lnkf_rows = lnkf[path_index]
    candidate_inputs = (z, cost, x_rows, hatcf_rows, lnkf_rows)
    if any(not torch.isfinite(value).all() for value in candidate_inputs):
        raise FloatingPointError(
            "investment-compare entry contains non-finite candidate state or cost"
        )

    p0_parts = []
    pi_parts = []
    modes = {module: module.training for module in policy_value_model.modules()}
    parameter_versions = [
        parameter._version for parameter in policy_value_model.parameters()
    ]
    buffer_versions = [buffer._version for buffer in policy_value_model.buffers()]
    try:
        policy_value_model.eval()
        with torch.no_grad():
            for start in range(0, n, int(chunk_size)):
                stop = min(start + int(chunk_size), n)
                state = torch.stack(
                    [
                        torch.zeros_like(z[start:stop]),
                        z[start:stop],
                        torch.zeros_like(z[start:stop]),
                        cost[start:stop],
                        x_rows[start:stop],
                        hatcf_rows[start:stop],
                        lnkf_rows[start:stop],
                    ],
                    dim=1,
                )
                values = value_forward(state)
                try:
                    p0 = values["V0_physical"].reshape(-1)
                    pi = values["VI_physical"].reshape(-1)
                except (KeyError, TypeError) as exc:
                    raise RuntimeError(
                        "forward_value_components() must expose physical-scale "
                        "V0_physical and VI_physical"
                    ) from exc
                p0_parts.append(p0)
                pi_parts.append(pi)
    finally:
        for module, was_training in modes.items():
            module.train(was_training)

    for before, after in zip(parameter_versions, policy_value_model.parameters()):
        if before != after._version:
            raise RuntimeError("entry branch-value forward modified model parameters")
    for before, after in zip(buffer_versions, policy_value_model.buffers()):
        if before != after._version:
            raise RuntimeError("entry branch-value forward modified model buffers")

    p0_flat = torch.cat(p0_parts)
    pi_flat = torch.cat(pi_parts)
    if not torch.isfinite(p0_flat).all() or not torch.isfinite(pi_flat).all():
        raise FloatingPointError(
            "investment-compare entry contains non-finite physical P0/PI values"
        )
    gap_flat = pi_flat - p0_flat
    if not torch.isfinite(gap_flat).all():
        raise FloatingPointError(
            "investment-compare entry contains non-finite value gaps"
        )

    def scatter(values: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(candidates.z)
        result[path_index, candidate_index] = values
        return result

    accepted = torch.zeros_like(candidates.mask)
    accepted[path_index, candidate_index] = gap_flat >= 0.0
    return {
        "entry_P0": scatter(p0_flat),
        "entry_PI": scatter(pi_flat),
        "entry_value_gap": scatter(gap_flat),
        "accepted": accepted,
    }


def summarize_candidate_values(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Per-path candidate distribution summary; empty paths remain NaN."""
    if values.shape != mask.shape or values.ndim != 2:
        raise ValueError("candidate values and mask must be matching matrices")
    n_paths = int(values.shape[0])
    nan = torch.full(
        (n_paths,), float("nan"), device=values.device, dtype=values.dtype
    )
    result = {name: nan.clone() for name in ("mean", "p10", "p50", "p90")}
    for path in range(n_paths):
        selected = values[path][mask[path]]
        if selected.numel() == 0:
            continue
        if not torch.isfinite(selected).all():
            raise FloatingPointError("candidate distribution contains non-finite values")
        result["mean"][path] = selected.mean()
        quantiles = torch.quantile(
            selected, selected.new_tensor([0.10, 0.50, 0.90])
        )
        result["p10"][path], result["p50"][path], result["p90"][path] = quantiles
    return result


def aggregate_node_resources(
    operating_contribution: torch.Tensor,
    path_index: torch.Tensor,
    *,
    n_paths: int,
    I_entry: torch.Tensor,
    mode: str,
) -> Dict[str, torch.Tensor]:
    if mode not in {"legacy_per_firm_clamp", "raw"}:
        raise ValueError(f"unknown consumption aggregation mode: {mode!r}")
    if I_entry.shape != (n_paths,):
        raise ValueError("I_entry must contain one value per path")
    contribution = (
        operating_contribution.clamp_min(0.0)
        if mode == "legacy_per_firm_clamp"
        else operating_contribution
    )
    C_oper = torch.zeros(n_paths, device=contribution.device, dtype=contribution.dtype)
    C_oper.index_add_(0, path_index, contribution)
    C_raw = C_oper - I_entry
    raw_operating = torch.zeros_like(C_oper)
    raw_operating.index_add_(0, path_index, operating_contribution)
    return {
        "C_oper": C_oper,
        "I_entry": I_entry,
        "C_raw": C_raw,
        "legacy_clamp_adjustment": C_oper - raw_operating,
        "resource_feasible": torch.isfinite(C_raw) & (C_raw > 0.0),
    }


def validate_node_resource_account(
    *,
    Y: torch.Tensor,
    I_oper: torch.Tensor,
    Phi: torch.Tensor,
    path_index: torch.Tensor,
    birth_mask: torch.Tensor,
    entry_cost: torch.Tensor,
    K_birth: torch.Tensor,
    K_current: torch.Tensor,
    n_paths: int,
    I_entry_ledger: torch.Tensor,
    C_reported: torch.Tensor,
    mode: str,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    strict: bool = False,
) -> Dict[str, torch.Tensor]:
    """Independently rebuild entry spending and node resources from details.

    The reported and rebuilt totals use separate GPU ``index_add_`` reductions.
    Float32 atomic accumulation order can differ, so the comparison uses
    PyTorch's conventional float32 relative tolerance while still rejecting
    economically meaningful ledger discrepancies.
    """
    if mode not in {"legacy_per_firm_clamp", "raw"}:
        raise ValueError(f"unknown consumption aggregation mode: {mode!r}")
    vectors = (
        Y, I_oper, Phi, path_index, birth_mask, entry_cost, K_birth, K_current
    )
    if any(value.ndim != 1 for value in vectors):
        raise ValueError("resource-account detail inputs must be one-dimensional")
    if len({int(value.numel()) for value in vectors}) != 1:
        raise ValueError("resource-account detail inputs must have matching lengths")
    if I_entry_ledger.shape != (n_paths,) or C_reported.shape != (n_paths,):
        raise ValueError("reported resource totals must contain one value per path")

    dtype = C_reported.dtype
    device = C_reported.device
    operating_detail = Y - I_oper - Phi
    operating_used = (
        operating_detail.clamp_min(0.0)
        if mode == "legacy_per_firm_clamp"
        else operating_detail
    )
    C_oper_rebuilt = torch.zeros(n_paths, device=device, dtype=dtype)
    C_oper_raw_rebuilt = torch.zeros_like(C_oper_rebuilt)
    I_entry_rebuilt = torch.zeros_like(C_oper_rebuilt)
    C_oper_rebuilt.index_add_(0, path_index, operating_used)
    C_oper_raw_rebuilt.index_add_(0, path_index, operating_detail)
    birth_spend = torch.where(
        birth_mask,
        entry_cost * K_birth,
        torch.zeros_like(entry_cost),
    )
    I_entry_rebuilt.index_add_(0, path_index, birth_spend)
    C_rebuilt = C_oper_rebuilt - I_entry_rebuilt
    entry_spend_residual = I_entry_ledger - I_entry_rebuilt
    resource_residual = C_reported - C_rebuilt
    entry_scale = torch.maximum(I_entry_ledger.abs(), I_entry_rebuilt.abs())
    resource_scale = torch.maximum(C_reported.abs(), C_rebuilt.abs())
    entry_tolerance = float(atol) + float(rtol) * entry_scale
    resource_tolerance = float(atol) + float(rtol) * resource_scale
    finite_detail = (
        torch.isfinite(Y)
        & torch.isfinite(I_oper)
        & torch.isfinite(Phi)
        & torch.isfinite(entry_cost)
        & torch.isfinite(K_birth)
        & torch.isfinite(K_current)
    )
    finite_by_path = torch.ones(n_paths, device=device, dtype=torch.bool)
    if finite_detail.numel():
        bad_paths = path_index[~finite_detail]
        if bad_paths.numel():
            finite_by_path[bad_paths.unique()] = False
    finite_totals = (
        torch.isfinite(I_entry_ledger)
        & torch.isfinite(I_entry_rebuilt)
        & torch.isfinite(C_reported)
        & torch.isfinite(C_rebuilt)
        & finite_by_path
    )
    accounting_valid = (
        finite_totals
        & (entry_spend_residual.abs() <= entry_tolerance)
        & (resource_residual.abs() <= resource_tolerance)
    )
    resource_feasible = finite_totals & (C_reported > 0.0)
    result = {
        "I_entry_rebuilt": I_entry_rebuilt,
        "entry_spend_residual": entry_spend_residual,
        "C_oper_rebuilt": C_oper_rebuilt,
        "C_oper_raw_rebuilt": C_oper_raw_rebuilt,
        "legacy_clamp_adjustment": C_oper_rebuilt - C_oper_raw_rebuilt,
        "C_rebuilt": C_rebuilt,
        "resource_accounting_residual": resource_residual,
        "accounting_valid": accounting_valid,
        "resource_feasible": resource_feasible,
    }
    if strict and not bool(accounting_valid.all()):
        bad = torch.nonzero(~accounting_valid, as_tuple=False).reshape(-1).tolist()
        raise EntryAccountingError(
            "independent node resource reconstruction failed; "
            f"paths={bad[:20]}"
        )
    if strict and not bool(resource_feasible.all()):
        bad = torch.nonzero(~resource_feasible, as_tuple=False).reshape(-1).tolist()
        raise EntryEconomicInfeasibilityError(
            "raw node resource account is infeasible or non-finite; "
            f"paths={bad[:20]}"
        )
    return result


def capital_growth_decomposition(
    previous_ids: torch.Tensor,
    previous_K: torch.Tensor,
    next_ids: torch.Tensor,
    next_K: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Exact one-path endpoint decomposition using real firm identities."""
    if previous_ids.ndim != 1 or next_ids.ndim != 1:
        raise ValueError("firm identities must be one-dimensional")
    if previous_K.shape != previous_ids.shape or next_K.shape != next_ids.shape:
        raise ValueError("firm capital and identity vectors must have matching shapes")
    if not torch.isfinite(previous_K).all() or not torch.isfinite(next_K).all():
        raise ValueError("capital decomposition requires finite firm capital")
    previous_id_list = [int(value) for value in previous_ids.tolist()]
    next_id_list = [int(value) for value in next_ids.tolist()]
    if len(set(previous_id_list)) != len(previous_id_list):
        raise ValueError("previous firm identities must be unique")
    if len(set(next_id_list)) != len(next_id_list):
        raise ValueError("next firm identities must be unique")
    previous = {int(i): previous_K[pos] for pos, i in enumerate(previous_id_list)}
    nxt = {int(i): next_K[pos] for pos, i in enumerate(next_id_list)}
    shared = sorted(set(previous) & set(nxt))
    exited = sorted(set(previous) - set(nxt))
    entered = sorted(set(nxt) - set(previous))
    zero = previous_K.new_zeros(()) if previous_K.numel() else next_K.new_zeros(())
    incumbent = sum((nxt[i] - previous[i] for i in shared), zero)
    entry = sum((nxt[i] for i in entered), zero)
    exit_old = sum((previous[i] for i in exited), zero)
    K_previous = previous_K.sum()
    K_next = next_K.sum()
    residual = (K_next - K_previous) - incumbent - entry + exit_old
    return {
        "deltaK_incumbent": incumbent,
        "K_entry_endpoint": entry,
        "K_exit_old": exit_old,
        "K_previous": K_previous,
        "K_next": K_next,
        "K_decomposition_start": K_previous,
        "K_decomposition_end": K_next,
        "capital_accounting_residual": residual,
    }
