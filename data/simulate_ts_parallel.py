from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from .data_utils import sample_ar1, sample_bernoulli, sample_stationary_ar1, sample_uniform
from .entry import (
    EntryCandidates,
    EntryAccountingError,
    EntryEconomicInfeasibilityError,
    aggregate_node_resources,
    capital_growth_decomposition,
    compute_entry_reference,
    draw_value_cost_candidates,
    evaluate_entry_cutoff,
    evaluate_investment_compare_entry,
    identify_birth_at_node,
    scan_investment_compare_hatc,
    slice_entry_candidates,
    summarize_candidate_values,
    validate_node_resource_account,
)
from .simulation_forward import forward_policy_value_for_simulation
from .tensor_data import TensorSimulationOutput, TensorTable, cat_rows
from .market_clearing import empty_entry_candidates
from .consistent_aggregation import (
    CONTINUUM_MASS_CONSISTENT_V1,
    continuum_entry_reference,
    equal_probability_normal_quadrature,
    uniform_midpoint_quadrature,
)
from utils.firm_transition import apply_refinancing_policy


def _module_state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _write_entry_hatc_failure_scan(
    sim,
    state: Dict[str, torch.Tensor],
    candidates: EntryCandidates,
    validation: Dict[str, torch.Tensor],
    *,
    path: int,
    t: int,
    branch_k: int,
) -> Dict[str, object]:
    """Scan one failed node without redrawing candidates or replaying history."""
    output_root = getattr(sim, "entry_hatc_failure_scan_output_dir", None)
    if not output_root:
        raise RuntimeError("entry Hatc failure scan output directory is not configured")
    model = sim.models.get("policy_value")
    if model is None:
        raise RuntimeError("entry Hatc failure scan requires policy_value")

    path = int(path)
    alive = state["alive"][path]
    same_node_entry = state["entry"][path] > 0.5
    incumbent_mask = alive & ~same_node_entry
    economic_time = state["economic_time"][path]
    incumbent_birth = (
        (state["initial_cohort"][path][incumbent_mask] < 0.5)
        & (
            state["birth_time"][path][incumbent_mask]
            == economic_time.to(state["birth_time"].dtype)
        )
    )
    center = float(state["hatcf"][path].detach().cpu())
    points = int(getattr(sim, "entry_hatc_failure_scan_points", 81))
    half_width = float(
        getattr(sim, "entry_hatc_failure_scan_half_width", 2.0)
    )
    hatc_values = torch.linspace(
        center - half_width,
        center + half_width,
        points,
        device=state["b"].device,
        dtype=state["b"].dtype,
    )
    hatc_values[points // 2] = center
    path_candidates = slice_entry_candidates(candidates, path)
    scan = scan_investment_compare_hatc(
        model,
        path_candidates,
        hatc_values=hatc_values,
        incumbent_b=state["b"][path][incumbent_mask],
        incumbent_z=state["z"][path][incumbent_mask],
        incumbent_eta=state["eta"][path][incumbent_mask],
        incumbent_i=state["i"][path][incumbent_mask],
        incumbent_K=state["K"][path][incumbent_mask],
        incumbent_birth_at_node=incumbent_birth,
        x=state["x"][path],
        lnkf=state["lnkf"][path],
        resource_accounting=sim._resource_accounting,
        chunk_size=int(sim.entry_inference_chunk_size),
    )
    scan_cpu = {name: value.detach().cpu() for name, value in scan.items()}
    residual = scan_cpu["clearing_residual"]
    feasible = scan_cpu["aggregate_resource_balance"] > 0.0
    feasible_pos = torch.nonzero(feasible, as_tuple=False).reshape(-1)
    if feasible_pos.numel():
        local = residual[feasible_pos].abs().argmin()
        best_pos = int(feasible_pos[local].item())
    else:
        best_pos = int(residual.abs().argmin().item())
    sign_change = (
        (residual[:-1] == 0.0)
        | (residual[1:] == 0.0)
        | ((residual[:-1] < 0.0) != (residual[1:] < 0.0))
    )

    episode = int(getattr(sim, "entry_hatc_failure_episode_id", -1))
    stem = f"ep{episode}_t{int(t)}_branch{int(branch_k)}_path{path}"
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}_hatc_scan.csv"
    fieldnames = list(scan_cpu)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in range(points):
            writer.writerow(
                {name: float(values[row].item()) for name, values in scan_cpu.items()}
            )

    accounting_fields = (
        "I_entry_reported",
        "I_entry_rebuilt",
        "entry_spend_residual",
        "entry_spend_tolerance",
        "C_reported",
        "C_rebuilt",
        "resource_accounting_residual",
        "resource_accounting_tolerance",
        "I_maintenance_reported",
        "I_maintenance_rebuilt",
        "maintenance_accounting_residual",
        "maintenance_accounting_tolerance",
        "I_oper_reported",
        "I_oper_rebuilt",
        "I_oper_accounting_residual",
        "I_oper_accounting_tolerance",
        "finite_totals",
    )
    accounting = {
        name: (
            bool(validation[name][path].item())
            if validation[name].dtype == torch.bool
            else float(validation[name][path].detach().cpu())
        )
        for name in accounting_fields
    }
    performed = validation.get("independent_validation_performed")
    accounting["independent_validation_performed"] = (
        True
        if performed is None
        else bool(performed[path].item())
    )
    summary = {
        "episode": episode,
        "t": int(t),
        "branch": int(branch_k),
        "path": path,
        "hatc_source": "current FC1 forecast input",
        "hatc_center": center,
        "hatc_min": float(hatc_values[0].detach().cpu()),
        "hatc_max": float(hatc_values[-1].detach().cpu()),
        "hatc_points": points,
        "fixed_incumbent_count": int(incumbent_mask.sum().item()),
        "fixed_candidate_count": int(path_candidates.counts[0].item()),
        "model_state_hash": _module_state_hash(model),
        "scan_csv": str(csv_path),
        "positive_resource_balance_points": int(feasible.sum().item()),
        "clearing_sign_change_intervals": int(sign_change.sum().item()),
        "best_positive_resource_point": {
            name: float(values[best_pos].item())
            for name, values in scan_cpu.items()
        },
        "original_accounting_failure": accounting,
    }
    summary_path = output_dir / f"{stem}_hatc_scan_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    snapshot_fields = (
        "b",
        "z",
        "eta",
        "i",
        "K",
        "alive",
        "entry",
        "initial_cohort",
        "birth_time",
        "entry_cost",
        "K_birth",
        "firm_id",
    )
    snapshot = {
        "metadata": summary,
        "x": state["x"][path].detach().cpu(),
        "hatcf": state["hatcf"][path].detach().cpu(),
        "lnkf": state["lnkf"][path].detach().cpu(),
        "economic_time": economic_time.detach().cpu(),
        "node_state": {
            name: state[name][path].detach().cpu() for name in snapshot_fields
        },
        "candidates": {
            name: getattr(path_candidates, name).detach().cpu()
            for name in path_candidates.__dataclass_fields__
        },
    }
    snapshot_path = output_dir / f"{stem}_node_snapshot.pt"
    torch.save(snapshot, snapshot_path)
    summary["snapshot"] = str(snapshot_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _draw_transition_shock(sim, child_t: int, branch_index: int, tag: int, draw):
    """Optionally isolate one transition shock stream for matched smoke runs."""
    base_seed = getattr(sim, "common_transition_seed", None)
    if base_seed is None:
        return draw()
    device = sim.device
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index or torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        seed = int(base_seed) + int(child_t) * 1_000_003 + int(branch_index) * 10_007 + int(tag)
        torch.manual_seed(seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(seed)
        return draw()


_HASH_MODULUS = 2_147_483_647


def _keyed_uniform(
    *,
    seed: int,
    path_id: torch.Tensor,
    economic_time: int | torch.Tensor,
    branch_id: int | torch.Tensor,
    identity: torch.Tensor,
    shock_type: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Vectorized deterministic uniforms keyed by canonical economic identity."""
    time_key = torch.as_tensor(
        economic_time,
        dtype=torch.int64,
        device=path_id.device,
    )
    while time_key.ndim < path_id.ndim:
        time_key = time_key.unsqueeze(-1)
    branch_key = torch.as_tensor(
        branch_id,
        dtype=torch.int64,
        device=path_id.device,
    )
    while branch_key.ndim < path_id.ndim:
        branch_key = branch_key.unsqueeze(-1)
    path_key, identity_key, time_key, branch_key = torch.broadcast_tensors(
        path_id.to(torch.int64),
        identity.to(torch.int64),
        time_key,
        branch_key,
    )
    mixed = torch.full_like(path_key, int(seed) % _HASH_MODULUS)
    for value, salt in (
        (path_key, 1_000_003),
        (time_key, 97_409),
        (branch_key, 65_537),
        (identity_key, 32_771),
        (torch.full_like(path_key, int(shock_type)), 8_191),
    ):
        mixed = torch.remainder(
            mixed * 48_271 + (value + 104_729) * salt,
            _HASH_MODULUS,
        )
    uniform = (mixed.to(torch.float64) + 0.5) / float(_HASH_MODULUS)
    return uniform.to(dtype=dtype)


def _keyed_normal(**kwargs) -> torch.Tensor:
    uniform = _keyed_uniform(**kwargs).clamp(1e-7, 1.0 - 1e-7)
    return torch.erfinv(2.0 * uniform - 1.0) * (2.0 ** 0.5)


def _stable_transition_draws(sim, state, child_t: int, branch_index: int):
    """Draw macro and firm innovations without depending on slots or capacity."""
    n_paths, n_firms = state["b"].shape
    path = state.get(
        "transition_path_id",
        torch.arange(n_paths, device=sim.device, dtype=torch.long),
    ).to(torch.long)
    path_2d = path.unsqueeze(1).expand(n_paths, n_firms)
    firm_identity = state["firm_id"].to(torch.long)
    macro_identity = torch.zeros_like(path)
    next_economic_time = state["economic_time"].to(torch.long) + 1
    common = {
        "seed": int(sim.common_transition_seed),
        "branch_id": int(branch_index),
        "dtype": state["x"].dtype,
    }
    eps_x = _keyed_normal(
        **common,
        path_id=path,
        economic_time=next_economic_time,
        identity=macro_identity,
        shock_type=1,
    )
    eps_z = _keyed_normal(
        **common,
        path_id=path_2d,
        economic_time=next_economic_time,
        identity=firm_identity,
        shock_type=2,
    )
    u_eta = _keyed_uniform(
        **common,
        path_id=path_2d,
        economic_time=next_economic_time,
        identity=firm_identity,
        shock_type=3,
    )
    u_i = _keyed_uniform(
        **common,
        path_id=path_2d,
        economic_time=next_economic_time,
        identity=firm_identity,
        shock_type=4,
    )
    x_next = (
        (1.0 - float(sim.config.RHO_X)) * float(sim.config.XBAR)
        + float(sim.config.RHO_X) * state["x"]
        + float(sim.config.SIGMA_X) * eps_x
    )
    z_next = (
        (1.0 - float(sim.config.RHO_Z)) * float(sim.config.ZBAR)
        + float(sim.config.RHO_Z) * state["z"]
        + float(sim.config.SIGMA_Z) * eps_z
    )
    eta_next = (u_eta < float(sim.config.ZETA)).to(state["eta"].dtype)
    i_next = u_i.to(state["i"].dtype) * float(sim.config.I_THRESHOLD)
    return x_next, z_next, eta_next, i_next, eps_x, eps_z, u_eta, u_i


def _stable_entry_candidates(sim, state, reference) -> EntryCandidates:
    """Draw entrant candidates from economic identity, not call order."""
    path = state["transition_path_id"].to(torch.long)
    economic_time = state["economic_time"].to(torch.long)
    branch = state.get(
        "transition_branch_id", torch.zeros_like(path, dtype=torch.long)
    ).to(torch.long)
    zero_identity = torch.zeros_like(path)
    rounding = _keyed_uniform(
        seed=int(sim.entry_rng_seed),
        path_id=path,
        economic_time=economic_time,
        branch_id=branch,
        identity=zero_identity,
        shock_type=101,
        dtype=torch.float64,
    )
    n_star = reference.n_star.to(torch.float64)
    base = torch.floor(n_star)
    counts = (base + (rounding < (n_star - base)).to(torch.float64)).to(torch.long)
    max_count = int(counts.max().item()) if counts.numel() else 0
    candidate = torch.arange(max_count, device=path.device, dtype=torch.long)
    candidate = candidate.unsqueeze(0).expand(path.numel(), -1)
    mask = candidate < counts.unsqueeze(1)
    path_grid = path.unsqueeze(1).expand_as(candidate)
    time_grid = economic_time.unsqueeze(1).expand_as(candidate)
    branch_grid = branch.unsqueeze(1).expand_as(candidate)
    dtype = reference.reference_K.dtype
    stationary_std = float(sim.config.SIGMA_Z) / max(
        (1.0 - float(sim.config.RHO_Z) ** 2) ** 0.5,
        1e-12,
    )
    z = float(sim.config.ZBAR) + stationary_std * _keyed_normal(
        seed=int(sim.entry_rng_seed),
        path_id=path_grid,
        economic_time=time_grid,
        branch_id=branch_grid,
        identity=candidate,
        shock_type=102,
        dtype=dtype,
    )
    entry_cost = float(sim.effective_entry_cost_max) * _keyed_uniform(
        seed=int(sim.entry_rng_seed),
        path_id=path_grid,
        economic_time=time_grid,
        branch_id=branch_grid,
        identity=candidate,
        shock_type=103,
        dtype=dtype,
    )
    K_birth = reference.K_per_entrant.unsqueeze(1).expand_as(z).clone()
    zero = torch.zeros((), device=path.device, dtype=dtype)
    z = torch.where(mask, z, zero)
    entry_cost = torch.where(mask, entry_cost, zero)
    K_birth = torch.where(mask, K_birth, zero)
    return EntryCandidates(
        counts=counts,
        mask=mask,
        z=z,
        entry_cost=entry_cost,
        K_birth=K_birth,
        potential_capital_nominal=reference.potential_capital_nominal,
        candidate_capital_realized=K_birth.sum(dim=1),
    )


def _ensure_entry_state_fields(state: Dict[str, torch.Tensor]) -> None:
    """Add zero-cost provenance fields to legacy/test states in place."""
    b = state["b"]
    n_paths = int(b.shape[0]) if b.ndim == 2 else 1
    device = b.device
    if b.ndim != 2:
        return
    state.setdefault("firm_mass", state["alive"].to(state["K"].dtype))
    state.setdefault("initial_cohort", state["alive"].to(torch.float32))
    state.setdefault("birth_time", torch.full_like(b, -1.0))
    state.setdefault("entry_cost", torch.zeros_like(b))
    state.setdefault("K_birth", torch.where(state["alive"], state["K"], torch.zeros_like(state["K"])))
    for name in ("entry_P0", "entry_PI", "entry_value_gap"):
        state.setdefault(name, torch.full_like(b, float("nan")))
    state.setdefault("economic_node_id", torch.zeros(n_paths, dtype=torch.long, device=device))
    state.setdefault("economic_time", torch.zeros(n_paths, dtype=torch.long, device=device))
    state.setdefault(
        "transition_path_id",
        torch.arange(n_paths, dtype=torch.long, device=device),
    )
    state.setdefault(
        "transition_branch_id",
        torch.full((n_paths,), -1, dtype=torch.long, device=device),
    )
    state.setdefault("accounting_valid", torch.zeros(n_paths, dtype=torch.bool, device=device))
    state.setdefault("transition_eps_x", torch.full((n_paths,), float("nan"), device=device))
    for name in ("transition_eps_z", "transition_u_eta", "transition_u_i"):
        state.setdefault(name, torch.full_like(b, float("nan")))
    for name in (
        "event_I_entry", "event_potential_count", "event_accepted_count",
        "event_reference_K", "event_reference_N", "event_mean_K_ref",
        "event_K_per_entrant", "event_potential_capital_nominal",
        "event_candidate_capital_realized", "event_K_entry_gross",
        "event_same_node_entrant_exit_count", "event_K_entry_same_node_exit",
        "event_K_entry_surviving",
        "event_entry_cutoff_mean", "event_entry_cutoff_p10",
        "event_entry_cutoff_p50", "event_entry_cutoff_p90",
        "event_entry_cost_mean", "event_entry_cost_p10",
        "event_entry_cost_p50", "event_entry_cost_p90",
    ):
        state.setdefault(name, torch.zeros(n_paths, device=device))
    for name in (
        "event_entry_P0_mean", "event_entry_P0_p10",
        "event_entry_P0_p50", "event_entry_P0_p90",
        "event_entry_PI_mean", "event_entry_PI_p10",
        "event_entry_PI_p50", "event_entry_PI_p90",
        "event_entry_value_gap_mean", "event_entry_value_gap_p10",
        "event_entry_value_gap_p50", "event_entry_value_gap_p90",
    ):
        state.setdefault(
            name, torch.full((n_paths,), float("nan"), device=device)
        )


def _initialize_continuum_cost_distribution(
    sim,
    state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Replace each initial slot by equal-mass original-cost quadrature units."""
    if bool(state.get("continuum_cost_distribution_initialized", False)):
        return state
    _ensure_entry_state_fields(state)
    B, N = state["b"].shape
    nodes, weights = uniform_midpoint_quadrature(
        sim.continuum_i_points,
        0.0,
        float(sim.config.I_THRESHOLD),
        device=state["b"].device,
        dtype=state["b"].dtype,
    )
    Ji = int(nodes.numel())
    original_mass = state.get("firm_mass", state["alive"].to(state["K"].dtype))
    for name, value in list(state.items()):
        if not torch.is_tensor(value) or value.ndim != 2 or value.shape != (B, N):
            continue
        state[name] = value[:, :, None].expand(B, N, Ji).reshape(B, N * Ji).clone()
    state["i"] = nodes[None, None, :].expand(B, N, Ji).reshape(B, N * Ji).clone()
    state["firm_mass"] = (
        original_mass[:, :, None] * weights[None, None, :]
    ).reshape(B, N * Ji)
    state["firm_id"] = torch.arange(
        N * Ji, device=state["b"].device, dtype=torch.long
    )[None, :].expand(B, -1).clone()
    state["next_firm_id"] = torch.full(
        (B,), N * Ji, device=state["b"].device, dtype=torch.long
    )
    state["continuum_cost_distribution_initialized"] = True
    return state


def simulate_tensor_parallel(sim) -> TensorSimulationOutput:
    """
    Simulate all paths in parallel on device.

    Time remains sequential because the state transition is recursive, but the
    path and firm dimensions are advanced as batched tensors.
    """
    device = sim.device
    n_potential = max(20, int(sim.group_size * 0.1))
    # Legacy keeps its historical static bound. Value-cost entry expands on
    # demand, so tensor capacity is never an economic quota.
    max_firms = (
        sim.group_size + sim.horizon * n_potential
        if sim.entry_mode == "legacy"
        else sim.group_size
    )
    initial_state = getattr(sim, "initial_state", None)
    if initial_state is None:
        state = _initialize_batched_state(sim, max_firms)
    else:
        from .state_bank import clone_state

        state = clone_state(initial_state, device=device)
        _ensure_entry_state_fields(state)
        max_firms = int(state["b"].shape[1])
    if getattr(sim, "aggregation_mode", "legacy_mixed_replay") == (
        CONTINUUM_MASS_CONSISTENT_V1
    ):
        state = _initialize_continuum_cost_distribution(sim, state)
        max_firms = int(state["b"].shape[1])
    sim._max_firms_observed = max(int(sim._max_firms_observed), int(max_firms))

    firm_rows: List[torch.Tensor] = []
    macro_rows: List[torch.Tensor] = []
    stopped_early = False
    stop_time = None
    extinct_paths: List[int] = []
    path_stop_times: List[int | None] = [None] * int(sim.n_paths)
    known_extinct = torch.zeros(
        int(sim.n_paths), dtype=torch.bool, device=device
    )
    completed_parent_steps = 0

    for t in tqdm(range(sim.horizon), desc="Simulating steps (tensor)"):
        state_bank_capture = getattr(sim, "state_bank_capture", None)
        market_controller = getattr(sim, "market_clearing_controller", None)
        if state_bank_capture is not None and market_controller is None:
            # State Bank v1 contract: capture the complete parent-ready state
            # before any node processing mutates policy/accounting fields.
            state_bank_capture.capture_parent_ready(state, time_index=t)
        committed = state.get("market_node_committed")
        if market_controller is not None and (
            committed is None or not bool(committed.all())
        ):
            state = market_controller.clear_state(
                sim,
                state,
                empty_entry_candidates(state),
                t=t,
                branch=-1,
            )
        if state_bank_capture is not None and market_controller is not None:
            # Experimental v1 captures the root-cleared parent-ready state.
            state_bank_capture.capture_parent_ready(state, time_index=t)
        parent_firm, parent_macro = _process_node_batched(sim, state, t, branch_k=-1)
        if market_controller is not None:
            market_controller.validate_committed(state, t=t, branch=-1)
            state["market_node_committed"] = torch.ones(
                state["b"].shape[0], dtype=torch.bool, device=device
            )
        firm_rows.append(parent_firm)
        macro_rows.append(parent_macro)
        completed_parent_steps += 1

        branch_states = _expand_branches_batched(sim, state, child_t=t + 1)
        _annotate_parent_next_leverage_rows(
            sim, parent_firm, state, branch_states[sim.main_branch]
        )
        for branch_k, branch_state in enumerate(branch_states):
            market_candidates = None
            if market_controller is not None:
                market_candidates = _prepare_market_entry_candidates(
                    sim, branch_state, enable_entry=bool(sim.enable_entry)
                )
                branch_state = market_controller.clear_state(
                    sim,
                    branch_state,
                    market_candidates,
                    t=t + 1,
                    branch=branch_k,
                )
                if sim.enable_entry:
                    sim._market_entry_candidates_override = market_candidates
            if sim.enable_entry:
                branch_state = _apply_entry_batched(sim, branch_state, n_potential)
                branch_states[branch_k] = branch_state

            branch_firm, branch_macro = _process_node_batched(sim, branch_state, t + 1, branch_k=branch_k)
            if market_controller is not None:
                market_controller.validate_committed(
                    branch_state, t=t + 1, branch=branch_k
                )
                branch_state["market_node_committed"] = torch.ones(
                    branch_state["b"].shape[0], dtype=torch.bool, device=device
                )
            firm_rows.append(branch_firm)
            macro_rows.append(branch_macro)

        state = branch_states[sim.main_branch]
        if sim.enable_exit:
            state = _apply_exit_batched(state)
            active_after_exit = state["alive"].any(dim=1)
            newly_extinct = ~active_after_exit & ~known_extinct
            extinct = torch.nonzero(newly_extinct, as_tuple=False).reshape(-1)
            if extinct.numel() > 0:
                new_paths = [
                    int(path) for path in extinct.detach().cpu().tolist()
                ]
                for path in new_paths:
                    path_stop_times[path] = int(t + 1)
                extinct_paths.extend(new_paths)
                known_extinct = known_extinct | ~active_after_exit
                print(
                    "Stopping extinct paths while continuing survivors: "
                    f"paths={new_paths[:20]} at t={t + 1}."
                )
            if not bool(active_after_exit.any()):
                stopped_early = True
                stop_time = t + 1
                print(
                    "Stopping simulation early: no surviving firms remain in "
                    f"any path at t={stop_time}."
                )
                break

    firm_tensor = cat_rows(firm_rows, len(sim.FIRM_COLUMNS), device)
    macro_tensor = cat_rows(macro_rows, len(sim.MACRO_COLUMNS), device)
    return TensorSimulationOutput(
        firm=TensorTable(firm_tensor, sim.FIRM_COLUMNS),
        macro=TensorTable(macro_tensor, sim.MACRO_COLUMNS),
        meta={
            "n_paths": sim.n_paths,
            "horizon": sim.horizon,
            "branch_num": sim.branch_num,
            "simulation_mode": "tensor_path_parallel",
            "max_firms_initial": max_firms,
            "max_firms": int(sim._max_firms_observed),
            "firm_id_float32_exact_guard": int(2 ** 24),
            "completed_parent_steps": completed_parent_steps,
            "stopped_early": stopped_early,
            "stop_reason": "no_surviving_firms" if stopped_early else None,
            "stop_time": stop_time,
            "extinct_paths": extinct_paths,
            "n_extinct_paths": len(extinct_paths),
            "path_stop_times": path_stop_times,
        },
    )


def _prepare_market_entry_candidates(sim, state, *, enable_entry: bool) -> EntryCandidates:
    """Freeze the exact candidate draw used by root search and node commit."""
    if not enable_entry:
        return empty_entry_candidates(state)
    if getattr(sim, "entry_mode", "legacy") != "investment_compare":
        raise RuntimeError(
            "experimental market clearing supports only investment_compare entry"
        )
    if getattr(sim, "entry_rng_mode", "legacy_sequential") != "stable_economic_identity":
        raise RuntimeError(
            "experimental market clearing requires stable economic entry identity"
        )
    if getattr(sim, "aggregation_mode", "legacy_mixed_replay") == (
        CONTINUUM_MASS_CONSISTENT_V1
    ):
        previous_K = state["transition_previous_K"]
        previous_alive = state["transition_previous_alive"]
        previous_mass = state.get(
            "transition_previous_mass", previous_alive.to(previous_K.dtype)
        )
        reference = continuum_entry_reference(
            previous_K,
            previous_mass,
            previous_alive,
            entry_capital_ratio=sim.entry_capital_ratio,
            entry_size_ratio=sim.entry_size_ratio,
        )
        z_nodes, _ = equal_probability_normal_quadrature(
            sim.continuum_z_points,
            device=previous_K.device,
            dtype=previous_K.dtype,
        )
        i_nodes, _ = uniform_midpoint_quadrature(
            sim.continuum_i_points,
            0.0,
            float(sim.effective_entry_cost_max),
            device=previous_K.device,
            dtype=previous_K.dtype,
        )
        stationary_std = float(sim.config.SIGMA_Z) / max(
            (1.0 - float(sim.config.RHO_Z) ** 2) ** 0.5, 1.0e-12
        )
        z_grid, i_grid = torch.meshgrid(z_nodes, i_nodes, indexing="ij")
        z_flat = float(sim.config.ZBAR) + stationary_std * z_grid.reshape(-1)
        i_flat = i_grid.reshape(-1)
        E = int(z_flat.numel())
        B = int(previous_K.shape[0])
        active = reference["potential_entry_mass"] > 0.0
        mask = active[:, None].expand(B, E).clone()
        z = z_flat[None, :].expand(B, E).clone()
        entry_cost = i_flat[None, :].expand(B, E).clone()
        K_birth = reference["capital_per_entrant"][:, None].expand(B, E).clone()
        zero = previous_K.new_zeros(())
        z = torch.where(mask, z, zero)
        entry_cost = torch.where(mask, entry_cost, zero)
        K_birth = torch.where(mask, K_birth, zero)
        return EntryCandidates(
            counts=torch.where(
                active,
                torch.full((B,), E, device=previous_K.device, dtype=torch.long),
                torch.zeros(B, device=previous_K.device, dtype=torch.long),
            ),
            mask=mask,
            z=z,
            entry_cost=entry_cost,
            K_birth=K_birth,
            potential_capital_nominal=reference["potential_entry_capital"],
            candidate_capital_realized=reference["potential_entry_capital"],
        )
    reference = compute_entry_reference(
        state["transition_previous_K"],
        state["transition_previous_alive"],
        entry_capital_ratio=sim.entry_capital_ratio,
        entry_size_ratio=sim.entry_size_ratio,
    )
    return _stable_entry_candidates(sim, state, reference)


def _initialize_batched_state(sim, max_firms: int) -> Dict[str, torch.Tensor]:
    # Macro-state semantics:
    # - hatcf/lnkf are current-node FC1 forecasts used by Policy/Value.
    # - hatc_cal/lnk_cal are filled after node processing from realized firm
    #   aggregation and are the FC1 inputs for child-node forecasts.
    device = sim.device
    n_paths = sim.n_paths
    n0 = sim.group_size

    x = sample_stationary_ar1(n_paths, sim.config.RHO_X, sim.config.SIGMA_X, sim.config.XBAR, device)
    z0 = sample_stationary_ar1(n_paths * n0, sim.config.RHO_Z, sim.config.SIGMA_Z, sim.config.ZBAR, device).view(n_paths, n0)
    b0 = sim._sample_initial_b(n_paths * n0, z0.reshape(-1), device).view(n_paths, n0)
    eta0 = sample_bernoulli(n_paths * n0, sim.config.ZETA, device).view(n_paths, n0)
    i0 = sample_uniform(n_paths * n0, 0.0, sim.config.I_THRESHOLD, device).view(n_paths, n0)

    lnkf = torch.normal(mean=4.0, std=1.0, size=(n_paths,), device=device)
    weights = torch.rand(n_paths, n0, device=device)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    K0 = torch.exp(lnkf).unsqueeze(1) * weights
    hatcf = torch.normal(mean=-2.0, std=1.0, size=(n_paths,), device=device)

    b = torch.zeros(n_paths, max_firms, device=device)
    z = torch.zeros(n_paths, max_firms, device=device)
    eta = torch.zeros(n_paths, max_firms, device=device)
    i = torch.zeros(n_paths, max_firms, device=device)
    K = torch.zeros(n_paths, max_firms, device=device)
    alive = torch.zeros(n_paths, max_firms, dtype=torch.bool, device=device)
    entry = torch.zeros(n_paths, max_firms, dtype=torch.float32, device=device)
    firm_id = torch.full((n_paths, max_firms), -1, dtype=torch.long, device=device)

    b[:, :n0] = b0
    z[:, :n0] = z0
    eta[:, :n0] = eta0
    i[:, :n0] = i0
    K[:, :n0] = K0
    alive[:, :n0] = True
    entry[:, :n0] = 1.0 if sim.entry_mode == "legacy" else 0.0
    firm_id[:, :n0] = torch.arange(n0, device=device, dtype=torch.long).unsqueeze(0).expand(n_paths, -1)

    return {
        "x": x,
        "b": b,
        "z": z,
        "eta": eta,
        "i": i,
        "K": K,
        "firm_mass": alive.to(K.dtype),
        "hatcf": hatcf,
        "lnkf": lnkf,
        "M": torch.ones(n_paths, device=device),
        "alive": alive,
        "entry": entry,
        "initial_cohort": alive.to(torch.float32),
        "birth_time": torch.full_like(K, -1.0),
        "entry_cost": torch.zeros_like(K),
        "K_birth": torch.where(alive, K, torch.zeros_like(K)),
        "entry_P0": torch.full_like(K, float("nan")),
        "entry_PI": torch.full_like(K, float("nan")),
        "entry_value_gap": torch.full_like(K, float("nan")),
        "firm_id": firm_id,
        "next_firm_id": torch.full((n_paths,), n0, dtype=torch.long, device=device),
        "bar_i": torch.zeros_like(b),
        "bar_z": torch.zeros_like(b),
        "bp": b.clone(),
        "economic_node_id": torch.zeros(n_paths, dtype=torch.long, device=device),
        "economic_time": torch.zeros(n_paths, dtype=torch.long, device=device),
        "transition_path_id": torch.arange(n_paths, dtype=torch.long, device=device),
        "transition_branch_id": torch.full(
            (n_paths,), -1, dtype=torch.long, device=device
        ),
        "accounting_valid": torch.zeros(n_paths, dtype=torch.bool, device=device),
        "transition_eps_x": torch.full((n_paths,), float("nan"), device=device),
        "transition_eps_z": torch.full_like(K, float("nan")),
        "transition_u_eta": torch.full_like(K, float("nan")),
        "transition_u_i": torch.full_like(K, float("nan")),
        "event_I_entry": torch.zeros(n_paths, device=device),
        "event_potential_count": torch.zeros(n_paths, device=device),
        "event_accepted_count": torch.zeros(n_paths, device=device),
        "event_reference_K": torch.zeros(n_paths, device=device),
        "event_reference_N": torch.zeros(n_paths, device=device),
        "event_mean_K_ref": torch.zeros(n_paths, device=device),
        "event_K_per_entrant": torch.zeros(n_paths, device=device),
        "event_potential_capital_nominal": torch.zeros(n_paths, device=device),
        "event_candidate_capital_realized": torch.zeros(n_paths, device=device),
        "event_K_entry_gross": torch.zeros(n_paths, device=device),
        "event_same_node_entrant_exit_count": torch.zeros(n_paths, device=device),
        "event_K_entry_same_node_exit": torch.zeros(n_paths, device=device),
        "event_K_entry_surviving": torch.zeros(n_paths, device=device),
        "event_entry_cutoff_mean": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_cutoff_p10": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_cutoff_p50": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_cutoff_p90": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_P0_mean": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_P0_p10": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_P0_p50": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_P0_p90": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_PI_mean": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_PI_p10": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_PI_p50": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_PI_p90": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_value_gap_mean": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_value_gap_p10": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_value_gap_p50": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_value_gap_p90": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_cost_mean": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_cost_p10": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_cost_p50": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_cost_p90": torch.full((n_paths,), float("nan"), device=device),
    }


def _process_node_batched(sim, state: Dict[str, torch.Tensor], t: int, branch_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if getattr(sim, "entry_mode", "legacy") == "investment_compare":
        missing = [
            name
            for name in ("initial_cohort", "birth_time", "economic_time")
            if name not in state
        ]
        if missing:
            raise RuntimeError(
                "investment_compare maintenance accounting requires explicit "
                f"birth provenance fields: missing={missing}"
            )
    _ensure_entry_state_fields(state)
    device = sim.device
    n_paths, n_firms = state["b"].shape
    alive = state["alive"]
    alive_flat = alive.reshape(-1)
    alive_pos = torch.nonzero(alive_flat, as_tuple=False).squeeze(-1)
    n_alive_per_path = alive.sum(dim=1)
    active_paths = n_alive_per_path > 0

    path_grid = torch.arange(n_paths, device=device, dtype=torch.long).unsqueeze(1).expand(n_paths, n_firms)

    if alive_pos.numel() == 0:
        if getattr(sim, "entry_mode", "legacy") in {
            "value_cost", "investment_compare"
        }:
            raise RuntimeError(
                f"{sim.entry_mode} entry encountered an extinct economic node at t={t}; "
                "no synthetic firms or macro floor are created"
            )
        state["hatc_cal"] = torch.full((n_paths,), -10.0, device=device)
        state["lnk_cal"] = torch.full((n_paths,), -10.0, device=device)
        legacy_prefix = [
                torch.arange(n_paths, device=device, dtype=torch.float32),
                torch.full((n_paths,), float(t), device=device),
                torch.full((n_paths,), float(branch_k), device=device),
                torch.zeros(n_paths, device=device),
                torch.zeros(n_paths, device=device),
                torch.full((n_paths,), -10.0, device=device),
                torch.full((n_paths,), -10.0, device=device),
                torch.zeros(n_paths, device=device),
                state["M"].to(torch.float32),
                state["x"].to(torch.float32),
                state["hatcf"].to(torch.float32),
                state["lnkf"].to(torch.float32),
        ]
        macro_rows = torch.stack(
            legacy_prefix
            + [torch.zeros(n_paths, device=device) for _ in sim.MACRO_COLUMNS[len(legacy_prefix):]],
            dim=1,
        ).to(torch.float32)
        firm_empty = torch.empty((0, len(sim.FIRM_COLUMNS)), device=device, dtype=torch.float32)
        return firm_empty, macro_rows

    path_idx = path_grid.reshape(-1)[alive_flat]
    b = state["b"].reshape(-1)[alive_flat]
    z = state["z"].reshape(-1)[alive_flat]
    eta = state["eta"].reshape(-1)[alive_flat]
    i = state["i"].reshape(-1)[alive_flat]
    K = state["K"].reshape(-1)[alive_flat]
    firm_mass = state.get(
        "firm_mass", state["alive"].to(state["K"].dtype)
    ).reshape(-1)[alive_flat]
    entry = state["entry"].reshape(-1)[alive_flat]
    initial_cohort = state["initial_cohort"].reshape(-1)[alive_flat]
    birth_time = state["birth_time"].reshape(-1)[alive_flat]
    entry_cost = state["entry_cost"].reshape(-1)[alive_flat]
    K_birth = state["K_birth"].reshape(-1)[alive_flat]
    entry_p0 = state["entry_P0"].reshape(-1)[alive_flat]
    entry_pi = state["entry_PI"].reshape(-1)[alive_flat]
    entry_value_gap = state["entry_value_gap"].reshape(-1)[alive_flat]
    firm_id = state["firm_id"].reshape(-1)[alive_flat].to(torch.float32)
    x = state["x"][path_idx]
    hatcf = state["hatcf"][path_idx]
    lnkf = state["lnkf"][path_idx]
    m_vec = state["M"][path_idx]

    firm_state = torch.stack([b, z, eta, i, x, hatcf, lnkf], dim=1)
    pv_model = sim.models.get("policy_value")
    output = None
    if pv_model is not None:
        with torch.no_grad():
            output = forward_policy_value_for_simulation(pv_model, firm_state)
        q = output.Q.reshape(-1)
        p0 = output.P0.reshape(-1)
        pi = output.PI.reshape(-1)
        bar_i_model = output.bar_i.reshape(-1)
        if bool(getattr(sim, "hard_realized_aggregation", False)):
            committed_record = state.get("decision_record_valid")
            if committed_record is not None and bool(committed_record.all()):
                bar_z = state["decision_record_bar_z"].reshape(-1)[alive_flat]
                bar_i_hard = state["decision_record_bar_i"].reshape(-1)[alive_flat]
            else:
                hard_survival = output.Phat.reshape(-1) > 0.0
                bar_z = (~hard_survival).to(b.dtype)
                bar_i_hard = (
                    hard_survival
                    & (output.PI.reshape(-1) >= output.P0.reshape(-1))
                ).to(b.dtype)
        else:
            bar_z = output.bar_z.reshape(-1)
            bar_i_hard = bar_i_model
        p = output.P.reshape(-1)
        bp0 = output.bp0.reshape(-1)
        bpI = output.bpI.reshape(-1)
        bp = output.bp.reshape(-1)
    else:
        q = torch.zeros_like(b)
        p0 = torch.zeros_like(b)
        pi = torch.zeros_like(b)
        bar_i_model = torch.zeros_like(b)
        bar_z = torch.zeros_like(b)
        p = torch.zeros_like(b)
        bp0 = b.clone()
        bpI = b.clone()
        bp = b.clone()
        bar_i_hard = bar_i_model

    birth_at_node = identify_birth_at_node(
        initial_cohort,
        birth_time,
        state["economic_time"],
        path_idx,
    )
    investment_compare = (
        getattr(sim, "entry_mode", "legacy") == "investment_compare"
    )
    birth_node_mask = investment_compare & birth_at_node
    maintenance_eligible = (
        ~birth_at_node
        if investment_compare
        else torch.ones_like(birth_at_node)
    )
    bar_i_executed = torch.where(
        birth_node_mask, torch.zeros_like(bar_i_hard), bar_i_hard
    )
    # Keep the historical Bar_i export as the action actually used by the
    # resource account and K transition. The model diagnostic is exported
    # separately as Bar_i_model.
    bar_i = bar_i_executed

    b_next_p0 = torch.full_like(b, float("nan"))
    b_next_pi = torch.full_like(b, float("nan"))
    b_next_policy = torch.full_like(b, float("nan"))

    Y, I, Phi, C, I_maintenance = sim._resource_accounting(
        K,
        z,
        x,
        bar_i,
        bar_z,
        i,
        maintenance_eligible=maintenance_eligible,
        return_maintenance=True,
    )

    K_total_computed = torch.zeros(n_paths, device=device)
    Y_total_computed = torch.zeros(n_paths, device=device)
    Phi_total_computed = torch.zeros(n_paths, device=device)
    I_oper_computed = torch.zeros(n_paths, device=device)
    K_maintenance_base_computed = torch.zeros(n_paths, device=device)
    K_birth_current_node_computed = torch.zeros(n_paths, device=device)
    I_maintenance_computed = torch.zeros(n_paths, device=device)
    K_total_computed.index_add_(0, path_idx, K * firm_mass)
    Y_total_computed.index_add_(0, path_idx, Y * firm_mass)
    Phi_total_computed.index_add_(0, path_idx, Phi * firm_mass)
    I_oper_computed.index_add_(0, path_idx, I * firm_mass)
    K_maintenance_base_computed.index_add_(
        0, path_idx, K * firm_mass * maintenance_eligible.to(K.dtype)
    )
    K_birth_current_node_computed.index_add_(
        0, path_idx, K * firm_mass * birth_at_node.to(K.dtype)
    )
    I_maintenance_computed.index_add_(0, path_idx, I_maintenance * firm_mass)
    I_entry_event = state.get("event_I_entry", torch.zeros(n_paths, device=device))
    resources = aggregate_node_resources(
        C * firm_mass,
        path_idx,
        n_paths=n_paths,
        I_entry=I_entry_event,
        mode=getattr(sim, "consumption_aggregation_mode", "legacy_per_firm_clamp"),
    )
    firm_mass_by_path = torch.zeros(n_paths, device=device, dtype=K.dtype)
    firm_mass_by_path.index_add_(0, path_idx, firm_mass)
    aggregate_debt = torch.zeros(n_paths, device=device, dtype=K.dtype)
    aggregate_debt.index_add_(0, path_idx, firm_mass * b * K)
    use_node_ledger = (
        getattr(sim, "node_accounting_mode", "legacy_recompute")
        == "economic_node_ledger"
    )
    cache_valid = state.get(
        "accounting_valid", torch.zeros(n_paths, dtype=torch.bool, device=device)
    )
    if use_node_ledger and bool(cache_valid.any()) and not bool(cache_valid.all()):
        raise RuntimeError("economic-node accounting cache must be path-complete")
    reused_account = use_node_ledger and bool(cache_valid.all())
    if reused_account:
        K_total = state["accounting_K_total"]
        C_raw_total = state["accounting_C_raw"]
        C_total = state.get("accounting_C_effective", C_raw_total)
        C_oper_total = state["accounting_C_oper"]
        I_oper_total = state["accounting_I_oper"]
        I_entry_total = state["accounting_I_entry"]
        resource_residual = state["accounting_resource_residual"]
        resource_feasible = state["accounting_resource_feasible"]
        I_entry_rebuilt = state["accounting_I_entry_rebuilt"]
        entry_spend_residual = state["accounting_entry_spend_residual"]
        C_rebuilt = state["accounting_C_rebuilt"]
        legacy_clamp_adjustment = state["accounting_legacy_clamp_adjustment"]
        LnK = state["accounting_LnK"]
        Hatc = state["accounting_Hatc"]
        n_firms_accounted = state["accounting_n_firms"]
        K_maintenance_base = state["accounting_K_maintenance_base"]
        K_birth_current_node = state["accounting_K_birth_current_node"]
        I_maintenance_total = state["accounting_I_maintenance"]
        I_maintenance_rebuilt = state["accounting_I_maintenance_rebuilt"]
        maintenance_residual = state["accounting_maintenance_residual"]
        birth_maintenance_exemption = state[
            "accounting_birth_maintenance_exemption"
        ]
        consumption_floor_applied = state.get(
            "accounting_consumption_floor_applied",
            torch.zeros(n_paths, dtype=torch.bool, device=device),
        )
    else:
        K_total = K_total_computed
        C_raw_total = resources["C_raw"]
        C_oper_total = resources["C_oper"]
        I_oper_total = I_oper_computed
        I_entry_total = resources["I_entry"]
        birth_mask = birth_at_node
        K_maintenance_base = K_maintenance_base_computed
        K_birth_current_node = K_birth_current_node_computed
        I_maintenance_total = I_maintenance_computed
        independent_validation_enabled = bool(
            getattr(sim, "entry_independent_resource_validation_enabled", False)
        )
        if independent_validation_enabled:
            validation = validate_node_resource_account(
                Y=Y,
                I_oper=I,
                Phi=Phi,
                path_index=path_idx,
                birth_mask=birth_mask,
                entry_cost=entry_cost,
                K_birth=K_birth,
                K_current=K,
                n_paths=n_paths,
                I_entry_ledger=I_entry_total,
                C_reported=C_raw_total,
                mode=getattr(
                    sim, "consumption_aggregation_mode", "legacy_per_firm_clamp"
                ),
                bar_i_executed=bar_i,
                bar_z=bar_z,
                investment_cost=i,
                initial_cohort=initial_cohort,
                birth_time=birth_time,
                economic_time=state["economic_time"],
                delta=float(sim.delta),
                maintenance_reported=I_maintenance,
                birth_maintenance_exempt=investment_compare,
            )
            validation["independent_validation_performed"] = torch.ones(
                n_paths, dtype=torch.bool, device=device
            )
            I_entry_rebuilt = validation["I_entry_rebuilt"]
            entry_spend_residual = validation["entry_spend_residual"]
            C_rebuilt = validation["C_rebuilt"]
            legacy_clamp_adjustment = validation["legacy_clamp_adjustment"]
            resource_residual = validation["resource_accounting_residual"]
            resource_feasible = validation["resource_feasible"]
            I_maintenance_rebuilt = validation["I_maintenance_rebuilt"]
            maintenance_residual = validation["maintenance_accounting_residual"]
            birth_maintenance_exemption = validation[
                "birth_maintenance_exemption"
            ]
            invalid_accounting = active_paths & ~validation["accounting_valid"]
            if bool(invalid_accounting.any()):
                bad = torch.nonzero(
                    invalid_accounting, as_tuple=False
                ).reshape(-1).tolist()
                if (
                    getattr(sim, "entry_mode", "legacy") == "investment_compare"
                    and bool(getattr(sim, "entry_hatc_failure_scan_enabled", False))
                ):
                    pending_candidates = getattr(
                        sim, "_pending_entry_candidates", None
                    )
                    if pending_candidates is None:
                        print(
                            "Entry Hatc failure scan skipped: fixed candidate draws "
                            "are unavailable."
                        )
                    else:
                        for bad_path in bad[:20]:
                            try:
                                diagnostic = _write_entry_hatc_failure_scan(
                                    sim,
                                    state,
                                    pending_candidates,
                                    validation,
                                    path=bad_path,
                                    t=t,
                                    branch_k=branch_k,
                                )
                                print(
                                    "Entry Hatc failure scan saved: "
                                    f"{diagnostic['scan_csv']}"
                                )
                            except Exception as diagnostic_exc:
                                print(
                                    "Entry Hatc failure scan failed without replacing "
                                    "the original accounting error: "
                                    f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
                                )
                raise EntryAccountingError(
                    "independent node resource reconstruction failed; "
                    f"t={t}, branch={branch_k}, paths={bad[:20]}"
                )
        else:
            # Use the authoritative first-pass ledger directly. These aliases
            # preserve the output schema but are not an independent validation.
            zero_by_path = torch.zeros_like(C_raw_total)
            finite_totals = (
                torch.isfinite(C_raw_total)
                & torch.isfinite(I_entry_total)
                & torch.isfinite(I_oper_total)
                & torch.isfinite(I_maintenance_total)
            )
            resource_feasible = resources["resource_feasible"]
            I_entry_rebuilt = I_entry_total
            entry_spend_residual = zero_by_path
            C_rebuilt = C_raw_total
            legacy_clamp_adjustment = resources["legacy_clamp_adjustment"]
            resource_residual = zero_by_path
            I_maintenance_rebuilt = I_maintenance_total
            maintenance_residual = zero_by_path
            birth_maintenance_exemption = (
                float(sim.delta) * K_birth_current_node
                if investment_compare
                else zero_by_path
            )
            validation = {
                "I_entry_reported": I_entry_total,
                "I_entry_rebuilt": I_entry_rebuilt,
                "entry_spend_residual": entry_spend_residual,
                "entry_spend_tolerance": zero_by_path,
                "C_reported": C_raw_total,
                "C_rebuilt": C_rebuilt,
                "resource_accounting_residual": resource_residual,
                "resource_accounting_tolerance": zero_by_path,
                "I_maintenance_reported": I_maintenance_total,
                "I_maintenance_rebuilt": I_maintenance_rebuilt,
                "maintenance_accounting_residual": maintenance_residual,
                "maintenance_accounting_tolerance": zero_by_path,
                "I_oper_reported": I_oper_total,
                "I_oper_rebuilt": I_oper_total,
                "I_oper_accounting_residual": zero_by_path,
                "I_oper_accounting_tolerance": zero_by_path,
                "finite_totals": finite_totals,
                "accounting_valid": finite_totals,
                "resource_feasible": resource_feasible,
                "independent_validation_performed": torch.zeros(
                    n_paths, dtype=torch.bool, device=device
                ),
            }
        n_firms_accounted = firm_mass_by_path.to(torch.float32)
        consumption_floor = float(getattr(sim, "consumption_floor", 0.0))
        raw_mode = (
            getattr(sim, "consumption_aggregation_mode", "legacy_per_firm_clamp")
            == "raw"
        )
        raw_finite = torch.isfinite(C_raw_total)
        invalid = active_paths & ~raw_finite
        if raw_mode and consumption_floor <= 0.0:
            invalid = invalid | (active_paths & (C_raw_total <= 0.0))
        if raw_mode and bool(invalid.any()):
            bad = torch.nonzero(invalid, as_tuple=False).reshape(-1).tolist()
            if (
                getattr(sim, "entry_mode", "legacy") == "investment_compare"
                and bool(getattr(sim, "entry_hatc_failure_scan_enabled", False))
            ):
                pending_candidates = getattr(
                    sim, "_pending_entry_candidates", None
                )
                if pending_candidates is not None:
                    for bad_path in bad[:20]:
                        try:
                            diagnostic = _write_entry_hatc_failure_scan(
                                sim,
                                state,
                                pending_candidates,
                                validation,
                                path=bad_path,
                                t=t,
                                branch_k=branch_k,
                            )
                            print(
                                "Entry Hatc infeasibility scan saved: "
                                f"{diagnostic['scan_csv']}"
                            )
                        except Exception as diagnostic_exc:
                            print(
                                "Entry Hatc infeasibility scan failed without "
                                "replacing the original error: "
                                f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
                            )
            raise EntryEconomicInfeasibilityError(
                "raw node resource account is infeasible or non-finite (C_raw<=0); "
                f"t={t}, branch={branch_k}, paths={bad[:20]}, "
                f"C_oper={[float(C_oper_total[index]) for index in bad[:20]]}, "
                f"I_entry={[float(I_entry_total[index]) for index in bad[:20]]}, "
                f"C_raw={[float(C_raw_total[index]) for index in bad[:20]]}, "
                f"accepted_count={[float(state.get('event_accepted_count', torch.zeros(n_paths, device=device))[index]) for index in bad[:20]]}, "
                "no positive floor was applied"
            )
        sim._pending_entry_candidates = None
        if raw_mode and consumption_floor > 0.0:
            C_total = C_raw_total.clamp_min(consumption_floor)
            K_for_macro = K_total.clamp_min(consumption_floor)
            consumption_floor_applied = (
                (C_raw_total < consumption_floor)
                | (K_total < consumption_floor)
            )
            LnK = torch.log(K_for_macro)
            Hatc = torch.log(C_total / K_for_macro)
        elif not raw_mode:
            C_total = C_raw_total
            consumption_floor_applied = torch.zeros(
                n_paths, dtype=torch.bool, device=device
            )
            LnK = torch.where(
                active_paths,
                torch.log(K_total + 1e-8),
                torch.full_like(K_total, float("nan")),
            )
            # Preserve the exact historical macro transform in legacy mode.
            Hatc = torch.where(
                active_paths,
                torch.log(C_total / (K_total + 1e-8) + 1e-5),
                torch.full_like(C_total, -10.0),
            )
        else:
            C_total = C_raw_total
            consumption_floor_applied = torch.zeros(
                n_paths, dtype=torch.bool, device=device
            )
            LnK = torch.log(K_total + 1e-8)
            Hatc = torch.where(
                resource_feasible,
                torch.log(C_total / (K_total + 1e-8)),
                torch.full_like(C_total, float("nan")),
            )
        if use_node_ledger:
            state["accounting_valid"] = torch.ones(n_paths, dtype=torch.bool, device=device)
            state["accounting_K_total"] = K_total.detach()
            state["accounting_Y_total"] = Y_total_computed.detach()
            state["accounting_Liquidation_total"] = Phi_total_computed.detach()
            state["accounting_C_raw"] = C_raw_total.detach()
            state["accounting_C_effective"] = C_total.detach()
            state["accounting_C_oper"] = C_oper_total.detach()
            state["accounting_I_oper"] = I_oper_total.detach()
            state["accounting_I_entry"] = I_entry_total.detach()
            state["accounting_I_entry_rebuilt"] = I_entry_rebuilt.detach()
            state["accounting_entry_spend_residual"] = entry_spend_residual.detach()
            state["accounting_C_rebuilt"] = C_rebuilt.detach()
            state["accounting_legacy_clamp_adjustment"] = legacy_clamp_adjustment.detach()
            state["accounting_resource_residual"] = resource_residual.detach()
            state["accounting_resource_feasible"] = resource_feasible.detach()
            state["accounting_LnK"] = LnK.detach()
            state["accounting_Hatc"] = Hatc.detach()
            state["accounting_n_firms"] = n_firms_accounted.detach()
            state["accounting_K_maintenance_base"] = K_maintenance_base.detach()
            state["accounting_K_birth_current_node"] = K_birth_current_node.detach()
            state["accounting_I_maintenance"] = I_maintenance_total.detach()
            state["accounting_I_maintenance_rebuilt"] = (
                I_maintenance_rebuilt.detach()
            )
            state["accounting_maintenance_residual"] = maintenance_residual.detach()
            state["accounting_birth_maintenance_exemption"] = (
                birth_maintenance_exemption.detach()
            )
            state["accounting_consumption_floor_applied"] = (
                consumption_floor_applied.detach()
            )
    state["hatc_cal"] = torch.where(
        active_paths,
        Hatc.detach(),
        state.get("hatc_cal", Hatc.detach()),
    )
    state["lnk_cal"] = torch.where(
        active_paths,
        LnK.detach(),
        state.get("lnk_cal", LnK.detach()),
    )

    # Endpoint capital uses the post-default survivor set. Once a child node is
    # promoted to parent, its full-node account remains authoritative even
    # though the live firm view has already removed same-node exits.
    if reused_account:
        deltaK_incumbent = state["accounting_deltaK_incumbent"]
        K_entry_endpoint = state["accounting_K_entry_endpoint"]
        K_exit_old = state["accounting_K_exit_old"]
        capital_residual = state["accounting_capital_residual"]
        K_node_pre_exit = state["accounting_K_node_pre_exit"]
        K_endpoint_post_exit = state["accounting_K_endpoint_post_exit"]
        K_decomposition_start = state["accounting_K_decomposition_start"]
        K_decomposition_end = state["accounting_K_decomposition_end"]
        decomposition_start_stage = state["accounting_decomposition_start_stage"]
        decomposition_end_stage = state["accounting_decomposition_end_stage"]
        decomposition_start_node_id = state["accounting_decomposition_start_node_id"]
        decomposition_end_node_id = state["accounting_decomposition_end_node_id"]
        decomposition_transition_valid = state[
            "accounting_decomposition_transition_valid"
        ]
        entry_capital_event_residual = state[
            "accounting_entry_capital_event_residual"
        ]
        state["event_same_node_entrant_exit_count"] = state[
            "accounting_same_node_entrant_exit_count"
        ]
        state["event_K_entry_same_node_exit"] = state[
            "accounting_K_entry_same_node_exit"
        ]
        state["event_K_entry_surviving"] = state[
            "accounting_K_entry_surviving"
        ]
    else:
        full_bar_z_preview = torch.zeros_like(state["b"])
        full_bar_z_preview.reshape(-1)[alive_pos] = bar_z.detach()
        endpoint_alive = (
            state["alive"] & (full_bar_z_preview < 0.5)
            if bool(getattr(sim, "enable_exit", True))
            else state["alive"].clone()
        )
        same_node_exit = state["alive"] & (state["entry"] > 0.5) & ~endpoint_alive
        entry_survives = state["alive"] & (state["entry"] > 0.5) & endpoint_alive
        state_mass = state.get(
            "firm_mass", state["alive"].to(state["K"].dtype)
        )
        state["event_same_node_entrant_exit_count"] = (
            state_mass * same_node_exit.to(state_mass.dtype)
        ).sum(dim=1).to(torch.float32)
        state["event_K_entry_same_node_exit"] = torch.where(
            same_node_exit,
            state["K"] * state_mass,
            torch.zeros_like(state["K"]),
        ).sum(dim=1)
        state["event_K_entry_surviving"] = torch.where(
            entry_survives,
            state["K"] * state_mass,
            torch.zeros_like(state["K"]),
        ).sum(dim=1)
        K_node_pre_exit = K_total_computed
        K_endpoint_post_exit = torch.where(
            endpoint_alive,
            state["K"] * state_mass,
            torch.zeros_like(state["K"]),
        ).sum(dim=1)
        entry_capital_event_residual = (
            state.get("event_K_entry_gross", torch.zeros(n_paths, device=device))
            - state["event_K_entry_surviving"]
            - state["event_K_entry_same_node_exit"]
        )
        capital_rows = []
        for path in range(n_paths):
            prev_alive = state.get("transition_previous_alive")
            if prev_alive is None:
                nan = K_total.new_full((), float("nan"))
                zero = K_total.new_zeros(())
                capital_rows.append((zero, zero, zero, nan, nan, nan, False))
                continue
            if getattr(sim, "aggregation_mode", "legacy_mixed_replay") == (
                CONTINUUM_MASS_CONSISTENT_V1
            ):
                previous_mass = state.get(
                    "transition_previous_mass",
                    prev_alive.to(state["transition_previous_K"].dtype),
                )[path]
                start_capital = (
                    previous_mass
                    * state["transition_previous_K"][path]
                    * prev_alive[path].to(previous_mass.dtype)
                ).sum()
                end_capital = K_endpoint_post_exit[path]
                entry_capital = state["event_K_entry_surviving"][path]
                exit_capital = K_node_pre_exit[path] - end_capital
                delta_incumbent = (
                    K_node_pre_exit[path] - start_capital
                    - state.get("event_K_entry_gross", torch.zeros_like(K_total))[path]
                )
                residual = (
                    start_capital + delta_incumbent
                    + state.get("event_K_entry_gross", torch.zeros_like(K_total))[path]
                    - exit_capital - end_capital
                )
                capital_rows.append((
                    delta_incumbent,
                    entry_capital,
                    exit_capital,
                    residual,
                    start_capital,
                    end_capital,
                    True,
                ))
                continue
            previous_mask = prev_alive[path]
            next_mask = endpoint_alive[path]
            decomposition = capital_growth_decomposition(
                state["transition_previous_ids"][path][previous_mask],
                state["transition_previous_K"][path][previous_mask],
                state["firm_id"][path][next_mask],
                state["K"][path][next_mask],
            )
            capital_rows.append((
                decomposition["deltaK_incumbent"],
                decomposition["K_entry_endpoint"],
                decomposition["K_exit_old"],
                decomposition["capital_accounting_residual"],
                decomposition["K_decomposition_start"],
                decomposition["K_decomposition_end"],
                True,
            ))
        deltaK_incumbent = torch.stack([row[0] for row in capital_rows])
        K_entry_endpoint = torch.stack([row[1] for row in capital_rows])
        K_exit_old = torch.stack([row[2] for row in capital_rows])
        capital_residual = torch.stack([row[3] for row in capital_rows])
        K_decomposition_start = torch.stack([row[4] for row in capital_rows])
        K_decomposition_end = torch.stack([row[5] for row in capital_rows])
        decomposition_transition_valid = torch.tensor(
            [row[6] for row in capital_rows], device=device, dtype=torch.bool
        )
        decomposition_start_stage = torch.where(
            decomposition_transition_valid,
            torch.zeros(n_paths, device=device),
            torch.full((n_paths,), -1.0, device=device),
        )
        decomposition_end_stage = torch.where(
            decomposition_transition_valid,
            torch.ones(n_paths, device=device),
            torch.full((n_paths,), -1.0, device=device),
        )
        decomposition_start_node_id = state.get(
            "transition_previous_node_id",
            torch.full((n_paths,), -1, device=device, dtype=torch.long),
        ).to(torch.float32)
        decomposition_start_node_id = torch.where(
            decomposition_transition_valid,
            decomposition_start_node_id,
            torch.full_like(decomposition_start_node_id, -1.0),
        )
        decomposition_end_node_id = torch.where(
            decomposition_transition_valid,
            state["economic_node_id"].to(torch.float32),
            torch.full((n_paths,), -1.0, device=device),
        )
        if use_node_ledger:
            state["accounting_deltaK_incumbent"] = deltaK_incumbent.detach()
            state["accounting_K_entry_endpoint"] = K_entry_endpoint.detach()
            state["accounting_K_exit_old"] = K_exit_old.detach()
            state["accounting_capital_residual"] = capital_residual.detach()
            state["accounting_K_node_pre_exit"] = K_node_pre_exit.detach()
            state["accounting_K_endpoint_post_exit"] = K_endpoint_post_exit.detach()
            state["accounting_K_decomposition_start"] = K_decomposition_start.detach()
            state["accounting_K_decomposition_end"] = K_decomposition_end.detach()
            state["accounting_decomposition_start_stage"] = decomposition_start_stage.detach()
            state["accounting_decomposition_end_stage"] = decomposition_end_stage.detach()
            state["accounting_decomposition_start_node_id"] = decomposition_start_node_id.detach()
            state["accounting_decomposition_end_node_id"] = decomposition_end_node_id.detach()
            state["accounting_decomposition_transition_valid"] = decomposition_transition_valid.detach()
            state["accounting_entry_capital_event_residual"] = entry_capital_event_residual.detach()
            state["accounting_same_node_entrant_exit_count"] = state[
                "event_same_node_entrant_exit_count"
            ].detach()
            state["accounting_K_entry_same_node_exit"] = state[
                "event_K_entry_same_node_exit"
            ].detach()
            state["accounting_K_entry_surviving"] = state[
                "event_K_entry_surviving"
            ].detach()

    firm_rows = torch.stack(
        [
            path_idx.to(torch.float32),
            torch.full_like(path_idx, float(t), dtype=torch.float32),
            torch.full_like(path_idx, float(branch_k), dtype=torch.float32),
            firm_id,
            entry.to(torch.float32),
            b,
            z,
            eta,
            i,
            x,
            hatcf,
            lnkf,
            K,
            m_vec,
            q,
            p0,
            pi,
            bar_i,
            bar_i_model,
            bar_i_executed,
            bar_z,
            p,
            bp0,
            bpI,
            bp,
            b_next_p0,
            b_next_pi,
            b_next_policy,
            Y,
            I,
            Phi,
            C,
            initial_cohort,
            birth_time,
            entry_cost,
            K_birth,
            entry_p0,
            entry_pi,
            entry_value_gap,
            state["economic_node_id"][path_idx].to(torch.float32),
            torch.full_like(b, 0.0 if branch_k == -1 else 1.0),
            torch.full_like(b, 1.0 if reused_account else 0.0),
            state["transition_eps_z"].reshape(-1)[alive_flat],
            state["transition_u_eta"].reshape(-1)[alive_flat],
            state["transition_u_i"].reshape(-1)[alive_flat],
            birth_at_node.to(torch.float32),
            maintenance_eligible.to(torch.float32),
            I_maintenance,
            firm_mass,
        ],
        dim=1,
    ).to(torch.float32)

    use_resolved_bp = (
        getattr(sim, "bp_action_override", None) is not None
        or getattr(sim, "bp_action_source", "head") == "grid"
        or bool(getattr(sim, "bp_actual_use_trace_enabled", False))
    )
    if use_resolved_bp:
        if output is None:
            raise RuntimeError(
                "resolved BP action requires a policy_value model output"
            )
        committed_record = state.get("decision_record_valid")
        reuse_record_bp = (
            bool(getattr(sim, "hard_realized_aggregation", False))
            and committed_record is not None
            and bool(committed_record.all())
            and int(branch_k) == -1
        )
        if reuse_record_bp:
            overridden = state["decision_record_bp"].reshape(-1)[alive_flat].reshape(-1, 1)
        else:
            overridden = sim._resolve_bp_action(
                sim=sim,
                firm_state=firm_state.detach(),
                model_output=output,
                bp_head=bp.detach().reshape(-1, 1),
                hatc_cal=Hatc[path_idx].detach().reshape(-1, 1),
                lnk_cal=LnK[path_idx].detach().reshape(-1, 1),
                path_index=path_idx.detach(),
                firm_id=firm_id.detach(),
                economic_node_id=state["economic_node_id"][path_idx].detach(),
                t=int(t),
                branch=int(branch_k),
                decision_stage=(
                    "parent_transition_policy"
                    if int(branch_k) == -1
                    else "child_observation_policy"
                ),
                bp_execution_applicable=(int(branch_k) == -1),
            )
        if not torch.is_tensor(overridden):
            raise TypeError("resolved BP action must return a torch.Tensor")
        overridden = overridden.to(device=bp.device, dtype=bp.dtype).reshape(-1)
        if overridden.shape != bp.shape:
            raise ValueError(
                "resolved BP action returned shape "
                f"{tuple(overridden.shape)}, expected {tuple(bp.shape)}"
            )
        if not torch.isfinite(overridden).all():
            raise FloatingPointError("resolved BP action returned non-finite values")
        bp = overridden
        firm_rows[:, sim.FIRM_COLUMNS.index("bp")] = bp.to(firm_rows.dtype)

    potential_count = state.get("event_potential_count", torch.zeros(n_paths, device=device))
    accepted_count = state.get("event_accepted_count", torch.zeros(n_paths, device=device))
    acceptance_rate = torch.where(
        potential_count > 0,
        accepted_count / potential_count,
        torch.zeros_like(potential_count),
    )
    macro_rows = torch.stack(
        [
            torch.arange(n_paths, device=device, dtype=torch.float32),
            torch.full((n_paths,), float(t), device=device),
            torch.full((n_paths,), float(branch_k), device=device),
            K_total,
            C_total,
            LnK,
            Hatc,
            n_firms_accounted,
            state["M"].to(torch.float32),
            state["x"].to(torch.float32),
            state["hatcf"].to(torch.float32),
            state["lnkf"].to(torch.float32),
            state["economic_node_id"].to(torch.float32),
            torch.full((n_paths,), 0.0 if branch_k == -1 else 1.0, device=device),
            torch.full((n_paths,), 1.0 if reused_account else 0.0, device=device),
            I_oper_total,
            I_entry_total,
            I_oper_total + I_entry_total,
            C_oper_total,
            C_raw_total,
            resource_residual,
            resource_feasible.to(torch.float32),
            state.get("event_reference_K", torch.zeros(n_paths, device=device)),
            state.get("event_reference_N", torch.zeros(n_paths, device=device)),
            state.get("event_mean_K_ref", torch.zeros(n_paths, device=device)),
            state.get("event_K_per_entrant", torch.zeros(n_paths, device=device)),
            potential_count,
            accepted_count,
            acceptance_rate,
            state.get("event_entry_cutoff_mean", torch.full((n_paths,), float("nan"), device=device)),
            state.get("event_entry_cutoff_p10", torch.full((n_paths,), float("nan"), device=device)),
            state.get("event_entry_cutoff_p50", torch.full((n_paths,), float("nan"), device=device)),
            state.get("event_entry_cutoff_p90", torch.full((n_paths,), float("nan"), device=device)),
            state.get("event_entry_P0_mean", torch.full((n_paths,), float("nan"), device=device)),
            state.get("event_entry_P0_p10", torch.full((n_paths,), float("nan"), device=device)),
            state.get("event_entry_P0_p50", torch.full((n_paths,), float("nan"), device=device)),
            state.get("event_entry_P0_p90", torch.full((n_paths,), float("nan"), device=device)),
            state.get("event_entry_PI_mean", torch.full((n_paths,), float("nan"), device=device)),
            state.get("event_entry_PI_p10", torch.full((n_paths,), float("nan"), device=device)),
            state.get("event_entry_PI_p50", torch.full((n_paths,), float("nan"), device=device)),
            state.get("event_entry_PI_p90", torch.full((n_paths,), float("nan"), device=device)),
            state.get("event_entry_value_gap_mean", torch.full((n_paths,), float("nan"), device=device)),
            state.get("event_entry_value_gap_p10", torch.full((n_paths,), float("nan"), device=device)),
            state.get("event_entry_value_gap_p50", torch.full((n_paths,), float("nan"), device=device)),
            state.get("event_entry_value_gap_p90", torch.full((n_paths,), float("nan"), device=device)),
            state.get("event_entry_cost_mean", torch.full((n_paths,), float("nan"), device=device)),
            state.get("event_entry_cost_p10", torch.full((n_paths,), float("nan"), device=device)),
            state.get("event_entry_cost_p50", torch.full((n_paths,), float("nan"), device=device)),
            state.get("event_entry_cost_p90", torch.full((n_paths,), float("nan"), device=device)),
            state.get("event_potential_capital_nominal", torch.zeros(n_paths, device=device)),
            state.get("event_candidate_capital_realized", torch.zeros(n_paths, device=device)),
            state.get("event_K_entry_gross", torch.zeros(n_paths, device=device)),
            state.get("event_same_node_entrant_exit_count", torch.zeros(n_paths, device=device)),
            state.get("event_K_entry_same_node_exit", torch.zeros(n_paths, device=device)),
            state.get("event_K_entry_surviving", torch.zeros(n_paths, device=device)),
            deltaK_incumbent,
            K_entry_endpoint,
            K_exit_old,
            capital_residual,
            torch.full((n_paths,), float(state["b"].shape[1]), device=device),
            I_entry_rebuilt,
            entry_spend_residual,
            C_rebuilt,
            legacy_clamp_adjustment,
            K_node_pre_exit,
            K_endpoint_post_exit,
            K_decomposition_start,
            K_decomposition_end,
            decomposition_start_stage,
            decomposition_end_stage,
            decomposition_start_node_id,
            decomposition_end_node_id,
            decomposition_transition_valid.to(torch.float32),
            entry_capital_event_residual,
            state["transition_eps_x"],
            K_maintenance_base,
            K_birth_current_node,
            I_maintenance_total,
            I_maintenance_rebuilt,
            maintenance_residual,
            birth_maintenance_exemption,
            torch.full(
                (n_paths,), float(getattr(sim, "consumption_floor", 0.0)),
                device=device,
            ),
            consumption_floor_applied.to(torch.float32),
            state.get("hatc_belief", state["hatcf"]).to(torch.float32),
            state.get("hatc_cleared", state["hatcf"]).to(torch.float32),
            state["hatcf"].to(torch.float32),
            state["lnkf"].to(torch.float32),
            LnK.to(torch.float32),
            state.get(
                "market_root_found",
                torch.zeros(n_paths, dtype=torch.bool, device=device),
            ).to(torch.float32),
            state.get(
                "market_root_count", torch.zeros(n_paths, device=device)
            ).to(torch.float32),
            state.get(
                "market_root_residual",
                torch.full((n_paths,), float("nan"), device=device),
            ).to(torch.float32),
            state.get(
                "market_root_residual_normalized",
                torch.full((n_paths,), float("nan"), device=device),
            ).to(torch.float32),
            state.get(
                "market_root_bracket_count", torch.zeros(n_paths, device=device)
            ).to(torch.float32),
            state.get(
                "market_root_refinement_rounds",
                torch.full((n_paths,), -1, device=device),
            ).to(torch.float32),
            state.get(
                "market_root_branch_id", torch.zeros(n_paths, device=device)
            ).to(torch.float32),
            state.get(
                "market_root_branch_switch",
                torch.zeros(n_paths, dtype=torch.bool, device=device),
            ).to(torch.float32),
            state.get(
                "market_root_policy_jump",
                torch.zeros(n_paths, dtype=torch.bool, device=device),
            ).to(torch.float32),
            state.get(
                "market_investment_flip_vs_belief",
                torch.zeros(n_paths, device=device),
            ).to(torch.float32),
            state.get(
                "market_default_flip_vs_belief",
                torch.zeros(n_paths, device=device),
            ).to(torch.float32),
            state.get(
                "market_entry_flip_vs_belief",
                torch.zeros(n_paths, device=device),
            ).to(torch.float32),
            state.get(
                "market_root_operator_evaluations",
                torch.zeros(n_paths, device=device),
            ).to(torch.float32),
            (
                torch.exp(state.get("hatc_cleared", state["hatcf"]))
                * K_total
                - C_raw_total
                if getattr(sim, "market_clearing_controller", None) is not None
                else torch.full((n_paths,), float("nan"), device=device)
            ).to(torch.float32),
            (
                (
                    torch.exp(state.get("hatc_cleared", state["hatcf"]))
                    * K_total
                    - C_raw_total
                )
                / torch.maximum(
                    torch.ones_like(K_total),
                    torch.maximum(K_total.abs(), C_raw_total.abs()),
                )
                if getattr(sim, "market_clearing_controller", None) is not None
                else torch.full((n_paths,), float("nan"), device=device)
            ).to(torch.float32),
            n_alive_per_path.to(torch.float32),
            firm_mass_by_path.to(torch.float32),
            aggregate_debt.to(torch.float32),
            state.get(
                "distribution_projection_error",
                torch.zeros(n_paths, device=device),
            ).to(torch.float32),
        ],
        dim=1,
    ).to(torch.float32)
    macro_rows = macro_rows[active_paths]

    full_bar_i = torch.zeros_like(state["b"])
    full_bar_z = torch.zeros_like(state["b"])
    full_bp = state["b"].clone()
    full_bar_i.reshape(-1)[alive_pos] = bar_i.detach()
    full_bar_z.reshape(-1)[alive_pos] = bar_z.detach()
    full_bp.reshape(-1)[alive_pos] = bp.detach()
    state["bar_i"] = full_bar_i
    state["bar_z"] = full_bar_z
    state["bp"] = full_bp

    return firm_rows, macro_rows


def _annotate_parent_next_leverage_rows(
    sim,
    parent_rows: torch.Tensor,
    parent_state: Dict[str, torch.Tensor],
    main_child_state: Dict[str, torch.Tensor],
) -> None:
    """Fill parent diagnostics from the current parent eta (``eta_t``).

    Realized leverage depends only on the parent eta, so sibling branches that
    differ only in the child eta ``eta_{t+1}`` share the same ``b_next``.
    """
    if parent_rows.numel() == 0:
        return
    columns = {name: idx for idx, name in enumerate(sim.FIRM_COLUMNS)}
    eta_current = parent_rows[:, columns["ETA"]]
    for bp_name, out_name in (
        ("bp0", "b_next_p0"),
        ("bpI", "b_next_pi"),
        ("bp", "b_next_policy"),
    ):
        parent_rows[:, columns[out_name]] = apply_refinancing_policy(
            b_current=parent_rows[:, columns["b"]],
            bp_candidate=parent_rows[:, columns[bp_name]],
            eta_current=eta_current,
        )


def _expand_branches_batched(
    sim,
    state: Dict[str, torch.Tensor],
    *,
    child_t: int = 1,
) -> List[Dict[str, torch.Tensor]]:
    _ensure_entry_state_fields(state)
    device = sim.device
    alive_any = state["alive"].any(dim=1)
    next_economic_time = state["economic_time"].to(torch.long) + 1
    branches: List[Dict[str, torch.Tensor]] = []
    b_prev = state["b"]
    bp_prev = state.get("bp", b_prev)
    continuum = getattr(sim, "aggregation_mode", "legacy_mixed_replay") == (
        CONTINUUM_MASS_CONSISTENT_V1
    )
    if continuum:
        continuum_reference = continuum_entry_reference(
            state["K"],
            state.get("firm_mass", state["alive"].to(state["K"].dtype)),
            state["alive"],
            entry_capital_ratio=float(getattr(sim, "entry_capital_ratio", 0.10)),
            entry_size_ratio=float(getattr(sim, "entry_size_ratio", 1.0)),
        )
        entry_reference = None
    else:
        entry_reference = compute_entry_reference(
            state["K"],
            state["alive"],
            entry_capital_ratio=float(getattr(sim, "entry_capital_ratio", 0.10)),
            entry_size_ratio=float(getattr(sim, "entry_size_ratio", 1.0)),
        )
    for branch_index in range(sim.branch_num):
        if getattr(sim, "transition_rng_mode", "legacy_position") == "stable_firm_identity":
            (
                x_next, z_next, eta_next, i_next,
                eps_x, eps_z, u_eta, u_i,
            ) = _stable_transition_draws(sim, state, child_t, branch_index)
        else:
            x_next = _draw_transition_shock(
                sim, child_t, branch_index, 1,
                lambda: sample_ar1(state["x"], sim.config.RHO_X, sim.config.SIGMA_X, sim.config.XBAR),
            )
            z_next = _draw_transition_shock(
                sim, child_t, branch_index, 2,
                lambda: sample_ar1(state["z"], sim.config.RHO_Z, sim.config.SIGMA_Z, sim.config.ZBAR),
            )
            eta_next = _draw_transition_shock(
                sim, child_t, branch_index, 3,
                lambda: sample_bernoulli(state["eta"].numel(), sim.config.ZETA, device).view_as(state["eta"]),
            )
            i_next = _draw_transition_shock(
                sim, child_t, branch_index, 4,
                lambda: sample_uniform(state["i"].numel(), 0.0, sim.config.I_THRESHOLD, device).view_as(state["i"]),
            )
            eps_x = torch.full_like(state["x"], float("nan"))
            eps_z = torch.full_like(state["z"], float("nan"))
            u_eta = torch.full_like(state["eta"], float("nan"))
            u_i = torch.full_like(state["i"], float("nan"))

        # Realize b_{t+1} from the current parent eta (eta_t). The child
        # eta_{t+1} is drawn below and never gates b_t -> b_{t+1}.
        b_next = apply_refinancing_policy(
            b_current=b_prev,
            bp_candidate=bp_prev,
            eta_current=state["eta"],
        )

        k_prev = state["K"]
        bar_i_prev = state.get("bar_i")
        if bar_i_prev is not None:
            K_next = bar_i_prev * sim.g * k_prev + (1.0 - bar_i_prev) * k_prev
        else:
            K_next = k_prev.clone()

        if sim.models.get("sdf_fc1") is not None:
            hatcf_next, lnkf_next, M_next = _predict_macro_fc1_batched(
                sim,
                state["x"],
                x_next,
                state["hatc_cal"],
                state["lnk_cal"],
            )
            hatcf_out = torch.where(alive_any, hatcf_next, state["hatcf"])
            lnkf_out = torch.where(alive_any, lnkf_next, state["lnkf"])
            M_out = torch.where(alive_any, M_next, state["M"])
        else:
            hatcf_out = state["hatcf"].clone()
            lnkf_out = state["lnkf"].clone()
            M_out = state["M"].clone()

        active2d = alive_any.unsqueeze(1)
        branch_state = {
                "x": torch.where(alive_any, x_next, state["x"]),
                "b": torch.where(active2d, b_next, state["b"]),
                "z": torch.where(active2d, z_next, state["z"]),
                "eta": torch.where(active2d, eta_next, state["eta"]),
                "i": torch.where(active2d, i_next, state["i"]),
                "K": torch.where(active2d, K_next, state["K"]),
                "firm_mass": state.get(
                    "firm_mass", state["alive"].to(state["K"].dtype)
                ).clone(),
                "hatcf": hatcf_out,
                "lnkf": lnkf_out,
                "M": M_out,
                "market_previous_root": state.get(
                    "hatc_cleared", state["hatcf"]
                ).clone(),
                "alive": state["alive"].clone(),
                "entry": torch.zeros_like(state["entry"]),
                "initial_cohort": state["initial_cohort"].clone(),
                "birth_time": state["birth_time"].clone(),
                "entry_cost": state["entry_cost"].clone(),
                "K_birth": state["K_birth"].clone(),
                "entry_P0": state["entry_P0"].clone(),
                "entry_PI": state["entry_PI"].clone(),
                "entry_value_gap": state["entry_value_gap"].clone(),
                "firm_id": state["firm_id"].clone(),
                "next_firm_id": state["next_firm_id"].clone(),
                "transition_path_id": state["transition_path_id"].clone(),
                "transition_branch_id": torch.full_like(
                    state["transition_path_id"], int(branch_index)
                ),
                "bar_i": state.get("bar_i", torch.zeros_like(state["b"])) .clone(),
                "bar_z": state.get("bar_z", torch.zeros_like(state["b"])) .clone(),
                "bp": state.get("bp", state["b"]).clone(),
                "economic_node_id": torch.where(
                    alive_any,
                    next_economic_time * int(sim.branch_num + 1)
                    + int(branch_index + 1),
                    state["economic_node_id"],
                ),
                "economic_time": torch.where(
                    alive_any,
                    next_economic_time,
                    state["economic_time"],
                ),
                "accounting_valid": torch.zeros(
                    state["b"].shape[0], dtype=torch.bool, device=device
                ),
                "transition_eps_x": eps_x,
                "transition_eps_z": eps_z,
                "transition_u_eta": u_eta,
                "transition_u_i": u_i,
                "event_I_entry": torch.zeros(state["b"].shape[0], device=device),
                "event_potential_count": torch.zeros(state["b"].shape[0], device=device),
                "event_accepted_count": torch.zeros(state["b"].shape[0], device=device),
                "event_reference_K": (
                    continuum_reference["reference_capital"].clone()
                    if continuum else entry_reference.reference_K.clone()
                ),
                "event_reference_N": (
                    continuum_reference["reference_mass"].to(torch.float32)
                    if continuum else entry_reference.reference_N.to(torch.float32)
                ),
                "event_mean_K_ref": (
                    continuum_reference["mean_capital_per_unit"].clone()
                    if continuum else entry_reference.mean_K_ref.clone()
                ),
                "event_K_per_entrant": (
                    continuum_reference["capital_per_entrant"].clone()
                    if continuum else entry_reference.K_per_entrant.clone()
                ),
                "event_potential_capital_nominal": (
                    continuum_reference["potential_entry_capital"].clone()
                    if continuum else entry_reference.potential_capital_nominal.clone()
                ),
                "event_candidate_capital_realized": torch.zeros(state["b"].shape[0], device=device),
                "event_K_entry_gross": torch.zeros(state["b"].shape[0], device=device),
                "event_same_node_entrant_exit_count": torch.zeros(state["b"].shape[0], device=device),
                "event_K_entry_same_node_exit": torch.zeros(state["b"].shape[0], device=device),
                "event_K_entry_surviving": torch.zeros(state["b"].shape[0], device=device),
                "event_entry_cutoff_mean": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_cutoff_p10": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_cutoff_p50": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_cutoff_p90": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_P0_mean": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_P0_p10": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_P0_p50": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_P0_p90": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_PI_mean": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_PI_p10": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_PI_p50": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_PI_p90": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_value_gap_mean": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_value_gap_p10": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_value_gap_p50": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_value_gap_p90": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_cost_mean": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_cost_p10": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_cost_p50": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_cost_p90": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "transition_previous_ids": state["firm_id"].clone(),
                "transition_previous_K": state["K"].clone(),
                "transition_previous_alive": state["alive"].clone(),
                "transition_previous_mass": state.get(
                    "firm_mass", state["alive"].to(state["K"].dtype)
                ).clone(),
                "transition_previous_node_id": state["economic_node_id"].clone(),
            }
        if continuum:
            branch_state = _expand_and_project_continuum_distribution(
                sim, branch_state
            )
        branches.append(branch_state)
    return branches


def _expand_and_project_continuum_distribution(
    sim,
    state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Integrate idiosyncratic z/eta/i and conservatively project the result.

    The common macro shock ``x`` is already fixed separately for this branch.
    Every surviving distribution unit is expanded over the original AR(1)
    innovation, Bernoulli refinancing state, and uniform investment-cost
    distributions.  Projection preserves mass, capital, debt, and current
    output exactly up to floating-point reduction error.
    """
    device = state["b"].device
    dtype = state["b"].dtype
    B, N = state["b"].shape
    z_nodes, z_weights = equal_probability_normal_quadrature(
        sim.continuum_z_points, device=device, dtype=dtype
    )
    i_nodes, i_weights = uniform_midpoint_quadrature(
        sim.continuum_i_points,
        0.0,
        float(sim.config.I_THRESHOLD),
        device=device,
        dtype=dtype,
    )
    eta_nodes = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
    eta_weights = torch.tensor(
        [1.0 - float(sim.config.ZETA), float(sim.config.ZETA)],
        device=device,
        dtype=dtype,
    )
    Jz, Ji = int(z_nodes.numel()), int(i_nodes.numel())
    factor = Jz * 2 * Ji

    base_alive = state["alive"]
    base_mass = state.get("firm_mass", base_alive.to(dtype))
    weight = (
        z_weights[:, None, None]
        * eta_weights[None, :, None]
        * i_weights[None, None, :]
    ).reshape(1, 1, factor)
    mass = (base_mass[:, :, None] * weight).reshape(B, N * factor)
    alive = base_alive[:, :, None].expand(B, N, factor).reshape(B, N * factor)
    mass = torch.where(alive, mass, torch.zeros_like(mass))

    def repeat(value: torch.Tensor) -> torch.Tensor:
        return value[:, :, None].expand(B, N, factor).reshape(B, N * factor)

    b = repeat(state["b"])
    K = repeat(state["K"])
    z_mean = (
        (1.0 - float(sim.config.RHO_Z)) * float(sim.config.ZBAR)
        + float(sim.config.RHO_Z) * state["z"]
    )
    z = (
        z_mean[:, :, None, None, None]
        + float(sim.config.SIGMA_Z)
        * z_nodes[None, None, :, None, None]
    ).expand(B, N, Jz, 2, Ji).reshape(B, N * factor)
    eta = eta_nodes[None, None, None, :, None].expand(
        B, N, Jz, 2, Ji
    ).reshape(B, N * factor)
    investment_cost = i_nodes[None, None, None, None, :].expand(
        B, N, Jz, 2, Ji
    ).reshape(B, N * factor)
    eps_z = z_nodes[None, None, :, None, None].expand(
        B, N, Jz, 2, Ji
    ).reshape(B, N * factor)
    u_eta_nodes = torch.tensor(
        [0.5 * (1.0 + float(sim.config.ZETA)), 0.5 * float(sim.config.ZETA)],
        device=device,
        dtype=dtype,
    )
    u_eta = u_eta_nodes[None, None, None, :, None].expand(
        B, N, Jz, 2, Ji
    ).reshape(B, N * factor)
    u_i = investment_cost / max(float(sim.config.I_THRESHOLD), 1.0e-12)

    # Fixed bins bound the number of storage units.  The representatives are
    # conservative moments rather than unweighted sample averages.
    b_bins, z_bins, k_bins = 6, 6, 3
    stationary_std = float(sim.config.SIGMA_Z) / max(
        (1.0 - float(sim.config.RHO_Z) ** 2) ** 0.5, 1.0e-12
    )
    z_lo = float(sim.config.ZBAR) - 4.0 * stationary_std
    z_hi = float(sim.config.ZBAR) + 4.0 * stationary_std
    projected_rows: list[dict[str, torch.Tensor]] = []
    projection_error = torch.zeros(B, device=device, dtype=dtype)

    for path in range(B):
        valid = alive[path] & (mass[path] > 0.0)
        if not bool(valid.any()):
            projected_rows.append({})
            continue
        m = mass[path, valid]
        bp = b[path, valid]
        zp = z[path, valid]
        etap = eta[path, valid]
        ip = investment_cost[path, valid]
        kp = K[path, valid]
        initial = repeat(state["initial_cohort"])[path, valid]
        birth = repeat(state["birth_time"])[path, valid]
        klog = torch.log(kp.clamp_min(1.0e-30))
        kmin, kmax = klog.min(), klog.max()
        kspan = (kmax - kmin).clamp_min(1.0e-12)
        b_key = torch.clamp((bp * b_bins).floor().to(torch.long), 0, b_bins - 1)
        z_key = torch.clamp(
            (((zp - z_lo) / max(z_hi - z_lo, 1.0e-12)) * z_bins)
            .floor()
            .to(torch.long),
            0,
            z_bins - 1,
        )
        k_key = torch.clamp(
            (((klog - kmin) / kspan) * k_bins).floor().to(torch.long),
            0,
            k_bins - 1,
        )
        eta_key = (etap > 0.5).to(torch.long)
        i_key = torch.clamp(
            (ip / max(float(sim.config.I_THRESHOLD), 1.0e-12) * Ji)
            .floor()
            .to(torch.long),
            0,
            Ji - 1,
        )
        birth_key = torch.where(
            initial >= 0.5,
            torch.zeros_like(birth, dtype=torch.long),
            birth.round().to(torch.long).clamp_min(0) + 1,
        )
        birth_base = max(int(birth_key.max().item()) + 1, 2)
        key = (((((b_key * z_bins + z_key) * k_bins + k_key) * 2 + eta_key)
                 * Ji + i_key) * birth_base + birth_key)
        unique, inverse = torch.unique(key, sorted=True, return_inverse=True)
        groups = int(unique.numel())

        def reduce_sum(value: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(groups, device=device, dtype=dtype)
            out.scatter_add_(0, inverse, value)
            return out

        m_out = reduce_sum(m)
        capital_total = reduce_sum(m * kp)
        debt_total = reduce_sum(m * kp * bp)
        output_proxy = reduce_sum(m * kp * torch.exp(zp))
        k_out = capital_total / m_out.clamp_min(1.0e-30)
        b_out = debt_total / capital_total.clamp_min(1.0e-30)
        z_out = torch.log(output_proxy / capital_total.clamp_min(1.0e-30))
        eta_out = reduce_sum(m * etap) / m_out.clamp_min(1.0e-30)
        i_out = reduce_sum(m * ip) / m_out.clamp_min(1.0e-30)
        initial_out = reduce_sum(m * initial) / m_out.clamp_min(1.0e-30)
        birth_out = reduce_sum(m * birth) / m_out.clamp_min(1.0e-30)
        eps_out = reduce_sum(m * eps_z[path, valid]) / m_out.clamp_min(1.0e-30)
        ueta_out = reduce_sum(m * u_eta[path, valid]) / m_out.clamp_min(1.0e-30)
        ui_out = reduce_sum(m * u_i[path, valid]) / m_out.clamp_min(1.0e-30)

        before = torch.stack(
            (m.sum(), (m * kp).sum(), (m * kp * bp).sum(),
             (m * kp * torch.exp(zp)).sum())
        )
        after = torch.stack(
            (m_out.sum(), capital_total.sum(), debt_total.sum(), output_proxy.sum())
        )
        projection_error[path] = ((after - before).abs() / before.abs().clamp_min(1.0e-12)).max()
        projected_rows.append(
            {
                "b": b_out,
                "z": z_out,
                "eta": eta_out,
                "i": i_out,
                "K": k_out,
                "firm_mass": m_out,
                "initial_cohort": initial_out,
                "birth_time": birth_out,
                "transition_eps_z": eps_out,
                "transition_u_eta": ueta_out,
                "transition_u_i": ui_out,
            }
        )

    max_groups = max((int(row.get("b", torch.empty(0)).numel()) for row in projected_rows), default=0)
    if max_groups == 0:
        return state
    projected_names = tuple(next(row for row in projected_rows if row).keys())
    for name in projected_names:
        fill = float("nan") if name.startswith("transition_") else 0.0
        target = torch.full((B, max_groups), fill, device=device, dtype=dtype)
        for path, row in enumerate(projected_rows):
            if row:
                target[path, : row[name].numel()] = row[name]
        state[name] = target
    alive_projected = state["firm_mass"] > 0.0
    state["alive"] = alive_projected
    state["entry"] = torch.zeros((B, max_groups), device=device, dtype=dtype)
    state["entry_cost"] = torch.zeros((B, max_groups), device=device, dtype=dtype)
    state["K_birth"] = torch.where(alive_projected, state["K"], torch.zeros_like(state["K"]))
    for name in ("entry_P0", "entry_PI", "entry_value_gap"):
        state[name] = torch.full((B, max_groups), float("nan"), device=device, dtype=dtype)
    state["firm_id"] = torch.arange(
        max_groups, device=device, dtype=torch.long
    )[None, :].expand(B, -1).clone()
    state["next_firm_id"] = torch.full(
        (B,), max_groups, device=device, dtype=torch.long
    )
    state["bar_i"] = torch.zeros((B, max_groups), device=device, dtype=dtype)
    state["bar_z"] = torch.zeros((B, max_groups), device=device, dtype=dtype)
    state["bp"] = state["b"].clone()
    state["distribution_projection_error"] = projection_error
    projection_tolerance = float(
        getattr(sim, "continuum_projection_tolerance", 1.0e-6)
    )
    if bool((projection_error > projection_tolerance).any()):
        bad = torch.nonzero(
            projection_error > projection_tolerance,
            as_tuple=False,
        ).reshape(-1)
        raise RuntimeError(
            "continuum conservative projection exceeded tolerance; "
            f"paths={bad.detach().cpu().tolist()[:20]}, "
            f"errors={projection_error[bad].detach().cpu().tolist()[:20]}, "
            f"tolerance={sim.continuum_projection_tolerance}"
        )
    return state


def _predict_macro_fc1_batched(
    sim,
    x_t: torch.Tensor,
    x_t1: torch.Tensor,
    hatcf_t: torch.Tensor,
    lnkf_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model = sim.models["sdf_fc1"]
    with torch.no_grad():
        _, _, M, hatcf_t1, lnkf_t1 = model.forward_step(
            x_prev=x_t.to(device=sim.device, dtype=torch.float32),
            x_curr=x_t1.to(device=sim.device, dtype=torch.float32),
            hatcf_prev=hatcf_t.to(device=sim.device, dtype=torch.float32),
            lnkf_prev=lnkf_t.to(device=sim.device, dtype=torch.float32),
            return_physical=True,
        )
    return hatcf_t1.reshape(-1), lnkf_t1.reshape(-1), M.reshape(-1)


def _apply_exit_batched(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if "bar_z" not in state:
        return state
    state["alive"] = state["alive"] & (state["bar_z"] < 0.5)
    if "firm_mass" in state:
        state["firm_mass"] = torch.where(
            state["alive"], state["firm_mass"], torch.zeros_like(state["firm_mass"])
        )
    return state


def _apply_entry_batched(sim, state: Dict[str, torch.Tensor], n_potential: int) -> Dict[str, torch.Tensor]:
    if getattr(sim, "entry_mode", "legacy") in {
        "value_cost", "investment_compare"
    }:
        return _apply_value_cost_entry_batched(sim, state)
    if getattr(sim, "preserve_global_rng_around_entry", False):
        cuda_devices = []
        if state["b"].device.type == "cuda":
            cuda_devices = [state["b"].device.index or torch.cuda.current_device()]
        with torch.random.fork_rng(devices=cuda_devices):
            event_seed = int(sim.entry_rng_seed) + int(sim._entry_event_counter) * 100003
            torch.manual_seed(event_seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(event_seed)
            result = _apply_legacy_entry_batched(sim, state, n_potential)
        sim._entry_event_counter += 1
        return result
    return _apply_legacy_entry_batched(sim, state, n_potential)


def _apply_legacy_entry_batched(
    sim,
    state: Dict[str, torch.Tensor],
    n_potential: int,
) -> Dict[str, torch.Tensor]:
    device = sim.device
    n_paths, max_firms = state["b"].shape

    z_new = sample_stationary_ar1(
        n_paths * n_potential,
        sim.config.RHO_Z,
        sim.config.SIGMA_Z,
        sim.config.ZBAR,
        device,
    ).view(n_paths, n_potential)
    i_new = sample_uniform(n_paths * n_potential, 0.0, sim.config.I_THRESHOLD, device).view(n_paths, n_potential)
    x = state["x"].unsqueeze(1)
    profit = torch.exp(x + z_new) - sim.delta
    entry_value = 1.0 + profit.clamp(min=0.0) - i_new
    enter_mask = entry_value > 0
    enter_counts = enter_mask.sum(dim=1)

    state["entry"] = torch.zeros_like(state["entry"], dtype=torch.float32)
    previous_alive = state.get("transition_previous_alive", state["alive"])
    previous_K = state.get("transition_previous_K", state["K"])
    legacy_reference_K = torch.where(
        previous_alive, previous_K, torch.zeros_like(previous_K)
    ).sum(dim=1)
    legacy_reference_N = previous_alive.sum(dim=1).to(torch.float32)
    state["event_reference_K"] = legacy_reference_K
    state["event_reference_N"] = legacy_reference_N
    state["event_mean_K_ref"] = torch.where(
        legacy_reference_N > 0,
        legacy_reference_K / legacy_reference_N.clamp_min(1.0),
        torch.zeros_like(legacy_reference_K),
    )
    state["event_K_per_entrant"] = torch.ones(n_paths, device=device)
    state["event_potential_count"] = torch.full(
        (n_paths,), float(n_potential), device=device
    )
    state["event_accepted_count"] = enter_counts.to(torch.float32)
    state["event_potential_capital_nominal"] = torch.full(
        (n_paths,), float(n_potential), device=device
    )
    state["event_candidate_capital_realized"] = torch.full(
        (n_paths,), float(n_potential), device=device
    )
    state["event_K_entry_gross"] = enter_counts.to(torch.float32)
    state["event_I_entry"] = torch.zeros(n_paths, device=device)
    if int(enter_counts.sum().item()) == 0:
        return state

    b_new = sample_uniform(n_paths * n_potential, sim.config.ENTRY_B_MIN, sim.config.ENTRY_B_MAX, device).view(n_paths, n_potential)
    eta_new = sample_bernoulli(n_paths * n_potential, sim.config.ZETA, device).view(n_paths, n_potential)
    K_new = torch.ones(n_paths, n_potential, device=device)

    rank = torch.cumsum(enter_mask.to(torch.long), dim=1) - 1
    dest = state["next_firm_id"].unsqueeze(1) + rank
    valid = enter_mask & (dest < max_firms)
    if not bool(valid.any()):
        return state

    path_idx = torch.arange(n_paths, device=device, dtype=torch.long).unsqueeze(1).expand(n_paths, n_potential)[valid]
    slot_idx = dest[valid].to(torch.long)
    new_id = (state["next_firm_id"].unsqueeze(1) + rank)[valid].to(torch.long)

    state["b"][path_idx, slot_idx] = b_new[valid]
    state["z"][path_idx, slot_idx] = z_new[valid]
    state["eta"][path_idx, slot_idx] = eta_new[valid]
    state["i"][path_idx, slot_idx] = i_new[valid]
    state["K"][path_idx, slot_idx] = K_new[valid]
    state["alive"][path_idx, slot_idx] = True
    state["entry"][path_idx, slot_idx] = 1.0
    state["initial_cohort"][path_idx, slot_idx] = 0.0
    state["birth_time"][path_idx, slot_idx] = state["economic_time"][path_idx].to(
        state["birth_time"].dtype
    )
    state["entry_cost"][path_idx, slot_idx] = 0.0
    state["K_birth"][path_idx, slot_idx] = K_new[valid]
    state["firm_id"][path_idx, slot_idx] = new_id
    for key in ("transition_eps_z", "transition_u_eta", "transition_u_i"):
        if key in state:
            state[key][path_idx, slot_idx] = float("nan")

    for key in ["bar_i", "bar_z", "bp"]:
        if key in state:
            fill_val = 0.0 if key != "bp" else state["b"][path_idx, slot_idx]
            state[key][path_idx, slot_idx] = fill_val

    state["next_firm_id"] = torch.clamp(state["next_firm_id"] + enter_counts, max=max_firms)
    return state


def _expand_firm_capacity(
    sim,
    state: Dict[str, torch.Tensor],
    required_capacity: int,
) -> Dict[str, torch.Tensor]:
    current = int(state["b"].shape[1])
    if required_capacity <= current:
        return state
    new_capacity = max(required_capacity, max(current * 2, 1))
    if new_capacity >= 2 ** 24:
        raise RuntimeError(
            "firm identity exceeds exact float32 export range; refusing invisible ID precision loss"
        )
    for key, value in list(state.items()):
        if not torch.is_tensor(value) or value.ndim != 2 or value.shape[1] != current:
            continue
        fill = (
            False
            if value.dtype == torch.bool
            else -1
            if key == "firm_id"
            else float("nan")
            if key in {
                "transition_eps_z", "transition_u_eta", "transition_u_i",
                "entry_P0", "entry_PI", "entry_value_gap",
            }
            else 0
        )
        expanded = torch.full(
            (value.shape[0], new_capacity),
            fill,
            dtype=value.dtype,
            device=value.device,
        )
        expanded[:, :current] = value
        state[key] = expanded
    sim._max_firms_observed = max(int(sim._max_firms_observed), new_capacity)
    return state


def _apply_value_cost_entry_batched(sim, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    n_paths = int(state["b"].shape[0])
    device = state["b"].device
    continuum = getattr(sim, "aggregation_mode", "legacy_mixed_replay") == (
        CONTINUUM_MASS_CONSISTENT_V1
    )
    if continuum:
        reference_mass = continuum_entry_reference(
            state["transition_previous_K"],
            state.get(
                "transition_previous_mass",
                state["transition_previous_alive"].to(
                    state["transition_previous_K"].dtype
                ),
            ),
            state["transition_previous_alive"],
            entry_capital_ratio=sim.entry_capital_ratio,
            entry_size_ratio=sim.entry_size_ratio,
        )
        reference = None
    else:
        reference = compute_entry_reference(
            state["transition_previous_K"],
            state["transition_previous_alive"],
            entry_capital_ratio=sim.entry_capital_ratio,
            entry_size_ratio=sim.entry_size_ratio,
        )
    override = getattr(sim, "_market_entry_candidates_override", None)
    if override is not None:
        candidates = override
        sim._market_entry_candidates_override = None
    elif getattr(sim, "entry_rng_mode", "legacy_sequential") == "stable_economic_identity":
        if continuum:
            raise RuntimeError(
                "continuum entry candidates must be prepared once by the "
                "market-clearing controller"
            )
        candidates = _stable_entry_candidates(sim, state, reference)
    else:
        candidates = draw_value_cost_candidates(
            reference,
            rho_z=float(sim.config.RHO_Z),
            sigma_z=float(sim.config.SIGMA_Z),
            zbar=float(sim.config.ZBAR),
            entry_cost_max=sim.effective_entry_cost_max,
            generator=sim.entry_generator,
        )
    # Retain this event's exact draws only until the corresponding node has
    # passed accounting validation.  This enables an in-place failure scan
    # without redrawing candidates or accumulating history-sized state.
    sim._pending_entry_candidates = candidates
    if sim.entry_mode == "investment_compare":
        screened = evaluate_investment_compare_entry(
            sim.models.get("policy_value"),
            candidates,
            x=state["x"],
            hatcf=state["hatcf"],
            lnkf=state["lnkf"],
            chunk_size=sim.entry_inference_chunk_size,
        )
    else:
        screened = evaluate_entry_cutoff(
            sim.models.get("policy_value"),
            candidates,
            x=state["x"],
            hatcf=state["hatcf"],
            lnkf=state["lnkf"],
            zeta=float(sim.config.ZETA),
            dummy_i=sim.entry_dummy_i,
            chunk_size=sim.entry_inference_chunk_size,
        )
    accepted = screened["accepted"] & candidates.mask
    pending_accepted = state.get("pending_entry_accepted")
    if pending_accepted is not None:
        if pending_accepted.shape != accepted.shape:
            raise RuntimeError("selected DecisionRecord entry shape changed before commit")
        if not torch.equal(pending_accepted, accepted):
            raise RuntimeError(
                "selected DecisionRecord entry acceptance changed before commit"
            )
        accepted = pending_accepted
    accepted_counts = accepted.sum(dim=1).to(torch.long)
    if continuum:
        represented_capital = (
            candidates.K_birth * candidates.mask.to(candidates.K_birth.dtype)
        ).sum(dim=1)
        mass_per_candidate = torch.where(
            represented_capital > 0.0,
            candidates.potential_capital_nominal
            / represented_capital.clamp_min(1.0e-12),
            torch.zeros_like(represented_capital),
        )
        candidate_mass = (
            mass_per_candidate[:, None]
            * candidates.mask.to(candidates.K_birth.dtype)
        )
        accepted_mass = (candidate_mass * accepted.to(candidate_mass.dtype)).sum(dim=1)
    else:
        candidate_mass = candidates.mask.to(candidates.K_birth.dtype)
        accepted_mass = accepted_counts.to(candidates.K_birth.dtype)
    required = int((state["next_firm_id"] + accepted_counts).max().item()) if n_paths else 0
    _expand_firm_capacity(sim, state, required)

    state["entry"].zero_()
    state["event_potential_count"] = (
        reference_mass["potential_entry_mass"].to(torch.float32)
        if continuum else candidates.counts.to(torch.float32)
    )
    state["event_accepted_count"] = accepted_mass.to(torch.float32)
    state["event_reference_K"] = (
        reference_mass["reference_capital"] if continuum else reference.reference_K
    )
    state["event_reference_N"] = (
        reference_mass["reference_mass"].to(torch.float32)
        if continuum else reference.reference_N.to(torch.float32)
    )
    state["event_mean_K_ref"] = (
        reference_mass["mean_capital_per_unit"] if continuum else reference.mean_K_ref
    )
    state["event_K_per_entrant"] = (
        reference_mass["capital_per_entrant"] if continuum else reference.K_per_entrant
    )
    state["event_potential_capital_nominal"] = candidates.potential_capital_nominal
    state["event_candidate_capital_realized"] = candidates.candidate_capital_realized
    if sim.entry_mode == "investment_compare":
        for field in ("entry_P0", "entry_PI", "entry_value_gap"):
            summary = summarize_candidate_values(screened[field], candidates.mask)
            for statistic, values in summary.items():
                state[f"event_{field}_{statistic}"] = values
    else:
        cutoff_summary = summarize_candidate_values(
            screened["entry_cutoff"], candidates.mask
        )
        for statistic, values in cutoff_summary.items():
            state[f"event_entry_cutoff_{statistic}"] = values
    cost_summary = summarize_candidate_values(
        candidates.entry_cost, candidates.mask
    )
    for statistic, values in cost_summary.items():
        state[f"event_entry_cost_{statistic}"] = values
    state["event_I_entry"] = torch.where(
        accepted,
        candidates.entry_cost * candidates.K_birth * candidate_mass,
        torch.zeros_like(candidates.entry_cost),
    ).sum(dim=1)
    state["event_K_entry_gross"] = torch.where(
        accepted,
        candidates.K_birth * candidate_mass,
        torch.zeros_like(candidates.K_birth),
    ).sum(dim=1)
    if not bool(accepted.any()):
        return state

    path_grid = torch.arange(n_paths, device=device).unsqueeze(1).expand_as(accepted)
    rank = torch.cumsum(accepted.to(torch.long), dim=1) - 1
    destination = state["next_firm_id"].unsqueeze(1) + rank
    path_idx = path_grid[accepted]
    slot_idx = destination[accepted]
    z_new = candidates.z[accepted]
    K_new = candidates.K_birth[accepted]
    cost_new = candidates.entry_cost[accepted]
    n_accepted = int(path_idx.numel())
    if sim.entry_mode == "investment_compare":
        eta_draw = torch.zeros(
            n_accepted, device=device, dtype=state["eta"].dtype
        )
        i_draw = cost_new.to(device=device, dtype=state["i"].dtype)
    else:
        # Historical value_cost_v1 timing: eta and ordinary investment cost are
        # drawn only after the aggregate-equity entry screen.
        if getattr(sim, "entry_rng_mode", "legacy_sequential") == "stable_economic_identity":
            candidate_grid = torch.arange(
                accepted.shape[1], device=device, dtype=torch.long
            ).unsqueeze(0).expand_as(accepted)
            candidate_id = candidate_grid[accepted]
            keyed_common = {
                "seed": int(sim.entry_rng_seed),
                "path_id": state["transition_path_id"][path_idx],
                "economic_time": state["economic_time"][path_idx],
                "branch_id": state["transition_branch_id"][path_idx],
                "identity": candidate_id,
                "dtype": torch.float64,
            }
            eta_draw = (
                _keyed_uniform(**keyed_common, shock_type=104)
                < float(sim.config.ZETA)
            ).to(device=device, dtype=state["eta"].dtype)
            i_draw = (
                _keyed_uniform(**keyed_common, shock_type=105)
                * float(sim.config.I_THRESHOLD)
            ).to(device=device, dtype=state["i"].dtype)
        else:
            eta_draw = (
                torch.rand(n_accepted, generator=sim.entry_generator, dtype=torch.float64)
                < float(sim.config.ZETA)
            ).to(device=device, dtype=state["eta"].dtype)
            i_draw = (
                torch.rand(n_accepted, generator=sim.entry_generator, dtype=torch.float64)
                * float(sim.config.I_THRESHOLD)
            ).to(device=device, dtype=state["i"].dtype)
    new_id = destination[accepted].to(torch.long)

    state["b"][path_idx, slot_idx] = 0.0
    state["z"][path_idx, slot_idx] = z_new
    state["eta"][path_idx, slot_idx] = eta_draw
    state["i"][path_idx, slot_idx] = i_draw
    state["K"][path_idx, slot_idx] = K_new
    state["firm_mass"][path_idx, slot_idx] = candidate_mass[accepted]
    state["alive"][path_idx, slot_idx] = True
    state["entry"][path_idx, slot_idx] = 1.0
    state["initial_cohort"][path_idx, slot_idx] = 0.0
    state["birth_time"][path_idx, slot_idx] = state["economic_time"][path_idx].to(
        state["birth_time"].dtype
    )
    state["entry_cost"][path_idx, slot_idx] = cost_new
    state["K_birth"][path_idx, slot_idx] = K_new
    if sim.entry_mode == "investment_compare":
        state["entry_P0"][path_idx, slot_idx] = screened["entry_P0"][accepted]
        state["entry_PI"][path_idx, slot_idx] = screened["entry_PI"][accepted]
        state["entry_value_gap"][path_idx, slot_idx] = screened[
            "entry_value_gap"
        ][accepted]
    state["firm_id"][path_idx, slot_idx] = new_id
    state["bar_i"][path_idx, slot_idx] = 0.0
    state["bar_z"][path_idx, slot_idx] = 0.0
    state["bp"][path_idx, slot_idx] = 0.0
    if "decision_record_bar_i" in state:
        state["decision_record_bar_i"][path_idx, slot_idx] = state[
            "pending_entry_bar_i"
        ][accepted]
        state["decision_record_bar_z"][path_idx, slot_idx] = state[
            "pending_entry_bar_z"
        ][accepted]
        # Entry BP is not an executed control at birth (eta=0); keep b=0.
        state["decision_record_bp"][path_idx, slot_idx] = 0.0
    state["transition_eps_z"][path_idx, slot_idx] = float("nan")
    state["transition_u_eta"][path_idx, slot_idx] = float("nan")
    state["transition_u_i"][path_idx, slot_idx] = float("nan")
    state["next_firm_id"] = state["next_firm_id"] + accepted_counts
    return state
