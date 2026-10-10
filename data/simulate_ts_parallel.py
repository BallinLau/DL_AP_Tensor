from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from .data_utils import sample_ar1, sample_bernoulli, sample_stationary_ar1, sample_uniform
from .entry import (
    EntryAccountingError,
    EntryEconomicInfeasibilityError,
    aggregate_node_resources,
    capital_growth_decomposition,
    compute_entry_reference,
    draw_value_cost_candidates,
    evaluate_entry_cutoff,
    summarize_candidate_values,
    validate_node_resource_account,
)
from .simulation_forward import forward_policy_value_for_simulation
from .tensor_data import TensorSimulationOutput, TensorTable, cat_rows
from utils.firm_transition import apply_refinancing_policy


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
    economic_time: int,
    branch_id: int,
    identity: torch.Tensor,
    shock_type: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Vectorized deterministic uniforms keyed by canonical economic identity."""
    path_key, identity_key = torch.broadcast_tensors(
        path_id.to(torch.int64), identity.to(torch.int64)
    )
    mixed = torch.full_like(path_key, int(seed) % _HASH_MODULUS)
    for value, salt in (
        (path_key, 1_000_003),
        (torch.full_like(path_key, int(economic_time)), 97_409),
        (torch.full_like(path_key, int(branch_id)), 65_537),
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
    path = torch.arange(n_paths, device=sim.device, dtype=torch.long)
    path_2d = path.unsqueeze(1).expand(n_paths, n_firms)
    firm_identity = state["firm_id"].to(torch.long)
    macro_identity = torch.zeros_like(path)
    common = {
        "seed": int(sim.common_transition_seed),
        "economic_time": int(child_t),
        "branch_id": int(branch_index),
        "dtype": state["x"].dtype,
    }
    eps_x = _keyed_normal(
        **common, path_id=path, identity=macro_identity, shock_type=1
    )
    eps_z = _keyed_normal(
        **common, path_id=path_2d, identity=firm_identity, shock_type=2
    )
    u_eta = _keyed_uniform(
        **common, path_id=path_2d, identity=firm_identity, shock_type=3
    )
    u_i = _keyed_uniform(
        **common, path_id=path_2d, identity=firm_identity, shock_type=4
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


def _ensure_entry_state_fields(state: Dict[str, torch.Tensor]) -> None:
    """Add zero-cost provenance fields to legacy/test states in place."""
    b = state["b"]
    n_paths = int(b.shape[0]) if b.ndim == 2 else 1
    device = b.device
    if b.ndim != 2:
        return
    state.setdefault("initial_cohort", state["alive"].to(torch.float32))
    state.setdefault("birth_time", torch.full_like(b, -1.0))
    state.setdefault("entry_cost", torch.zeros_like(b))
    state.setdefault("K_birth", torch.where(state["alive"], state["K"], torch.zeros_like(state["K"])))
    state.setdefault("economic_node_id", torch.zeros(n_paths, dtype=torch.long, device=device))
    state.setdefault("economic_time", torch.zeros(n_paths, dtype=torch.long, device=device))
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
    state = _initialize_batched_state(sim, max_firms)
    sim._max_firms_observed = max(int(sim._max_firms_observed), int(max_firms))

    firm_rows: List[torch.Tensor] = []
    macro_rows: List[torch.Tensor] = []

    for t in tqdm(range(sim.horizon), desc="Simulating steps (tensor)"):
        parent_firm, parent_macro = _process_node_batched(sim, state, t, branch_k=-1)
        firm_rows.append(parent_firm)
        macro_rows.append(parent_macro)

        branch_states = _expand_branches_batched(sim, state, child_t=t + 1)
        _annotate_parent_next_leverage_rows(
            sim, parent_firm, state, branch_states[sim.main_branch]
        )
        for branch_k, branch_state in enumerate(branch_states):
            if sim.enable_entry:
                branch_state = _apply_entry_batched(sim, branch_state, n_potential)
                branch_states[branch_k] = branch_state

            branch_firm, branch_macro = _process_node_batched(sim, branch_state, t + 1, branch_k=branch_k)
            firm_rows.append(branch_firm)
            macro_rows.append(branch_macro)

        state = branch_states[sim.main_branch]
        if sim.enable_exit:
            state = _apply_exit_batched(state)

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
        },
    )


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
        "hatcf": hatcf,
        "lnkf": lnkf,
        "M": torch.ones(n_paths, device=device),
        "alive": alive,
        "entry": entry,
        "initial_cohort": alive.to(torch.float32),
        "birth_time": torch.full_like(K, -1.0),
        "entry_cost": torch.zeros_like(K),
        "K_birth": torch.where(alive, K, torch.zeros_like(K)),
        "firm_id": firm_id,
        "next_firm_id": torch.full((n_paths,), n0, dtype=torch.long, device=device),
        "bar_i": torch.zeros_like(b),
        "bar_z": torch.zeros_like(b),
        "bp": b.clone(),
        "economic_node_id": torch.zeros(n_paths, dtype=torch.long, device=device),
        "economic_time": torch.zeros(n_paths, dtype=torch.long, device=device),
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
        "event_entry_cost_mean": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_cost_p10": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_cost_p50": torch.full((n_paths,), float("nan"), device=device),
        "event_entry_cost_p90": torch.full((n_paths,), float("nan"), device=device),
    }


def _process_node_batched(sim, state: Dict[str, torch.Tensor], t: int, branch_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    _ensure_entry_state_fields(state)
    device = sim.device
    n_paths, n_firms = state["b"].shape
    alive = state["alive"]
    alive_flat = alive.reshape(-1)
    alive_pos = torch.nonzero(alive_flat, as_tuple=False).squeeze(-1)
    n_alive_per_path = alive.sum(dim=1)

    path_grid = torch.arange(n_paths, device=device, dtype=torch.long).unsqueeze(1).expand(n_paths, n_firms)

    if alive_pos.numel() == 0:
        if getattr(sim, "entry_mode", "legacy") == "value_cost":
            raise RuntimeError(
                f"value_cost entry encountered an extinct economic node at t={t}; "
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
    entry = state["entry"].reshape(-1)[alive_flat]
    initial_cohort = state["initial_cohort"].reshape(-1)[alive_flat]
    birth_time = state["birth_time"].reshape(-1)[alive_flat]
    entry_cost = state["entry_cost"].reshape(-1)[alive_flat]
    K_birth = state["K_birth"].reshape(-1)[alive_flat]
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
        bar_i = output.bar_i.reshape(-1)
        bar_z = output.bar_z.reshape(-1)
        p = output.P.reshape(-1)
        bp0 = output.bp0.reshape(-1)
        bpI = output.bpI.reshape(-1)
        bp = output.bp.reshape(-1)
    else:
        q = torch.zeros_like(b)
        p0 = torch.zeros_like(b)
        pi = torch.zeros_like(b)
        bar_i = torch.zeros_like(b)
        bar_z = torch.zeros_like(b)
        p = torch.zeros_like(b)
        bp0 = b.clone()
        bpI = b.clone()
        bp = b.clone()

    b_next_p0 = torch.full_like(b, float("nan"))
    b_next_pi = torch.full_like(b, float("nan"))
    b_next_policy = torch.full_like(b, float("nan"))

    Y, I, Phi, C = sim._resource_accounting(K, z, x, bar_i, bar_z, i)

    K_total_computed = torch.zeros(n_paths, device=device)
    I_oper_computed = torch.zeros(n_paths, device=device)
    K_total_computed.index_add_(0, path_idx, K)
    I_oper_computed.index_add_(0, path_idx, I)
    I_entry_event = state.get("event_I_entry", torch.zeros(n_paths, device=device))
    resources = aggregate_node_resources(
        C,
        path_idx,
        n_paths=n_paths,
        I_entry=I_entry_event,
        mode=getattr(sim, "consumption_aggregation_mode", "legacy_per_firm_clamp"),
    )
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
        C_total = state["accounting_C_raw"]
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
    else:
        K_total = K_total_computed
        C_total = resources["C_raw"]
        C_oper_total = resources["C_oper"]
        I_oper_total = I_oper_computed
        I_entry_total = resources["I_entry"]
        birth_mask = (
            (entry > 0.5)
            & (birth_time == state["economic_time"][path_idx].to(birth_time.dtype))
            & (initial_cohort < 0.5)
        )
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
            C_reported=C_total,
            mode=getattr(sim, "consumption_aggregation_mode", "legacy_per_firm_clamp"),
        )
        I_entry_rebuilt = validation["I_entry_rebuilt"]
        entry_spend_residual = validation["entry_spend_residual"]
        C_rebuilt = validation["C_rebuilt"]
        legacy_clamp_adjustment = validation["legacy_clamp_adjustment"]
        resource_residual = validation["resource_accounting_residual"]
        resource_feasible = validation["resource_feasible"]
        if not bool(validation["accounting_valid"].all()):
            bad = torch.nonzero(
                ~validation["accounting_valid"], as_tuple=False
            ).reshape(-1).tolist()
            raise EntryAccountingError(
                "independent node resource reconstruction failed; "
                f"t={t}, branch={branch_k}, paths={bad[:20]}"
            )
        n_firms_accounted = n_alive_per_path.to(torch.float32)
        if getattr(sim, "consumption_aggregation_mode", "legacy_per_firm_clamp") == "raw" and not bool(resource_feasible.all()):
            bad = torch.nonzero(~resource_feasible, as_tuple=False).reshape(-1).tolist()
            raise EntryEconomicInfeasibilityError(
                "raw node resource account is infeasible or non-finite (C_raw<=0); "
                f"t={t}, branch={branch_k}, paths={bad[:20]}, "
                "no positive floor was applied"
            )
        alive_any = n_alive_per_path > 0
        LnK = torch.where(
            alive_any,
            torch.log(K_total + 1e-8),
            torch.full_like(K_total, float("nan")),
        )
        if getattr(sim, "consumption_aggregation_mode", "legacy_per_firm_clamp") == "legacy_per_firm_clamp":
            # Preserve the exact historical macro transform in legacy mode.
            Hatc = torch.where(
                alive_any,
                torch.log(C_total / (K_total + 1e-8) + 1e-5),
                torch.full_like(C_total, -10.0),
            )
        else:
            Hatc = torch.where(
                resource_feasible,
                torch.log(C_total / (K_total + 1e-8)),
                torch.full_like(C_total, float("nan")),
            )
        if use_node_ledger:
            state["accounting_valid"] = torch.ones(n_paths, dtype=torch.bool, device=device)
            state["accounting_K_total"] = K_total.detach()
            state["accounting_C_raw"] = C_total.detach()
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
    state["hatc_cal"] = Hatc.detach()
    state["lnk_cal"] = LnK.detach()

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
        state["event_same_node_entrant_exit_count"] = same_node_exit.sum(dim=1).to(torch.float32)
        state["event_K_entry_same_node_exit"] = torch.where(
            same_node_exit, state["K"], torch.zeros_like(state["K"])
        ).sum(dim=1)
        state["event_K_entry_surviving"] = torch.where(
            entry_survives, state["K"], torch.zeros_like(state["K"])
        ).sum(dim=1)
        K_node_pre_exit = K_total_computed
        K_endpoint_post_exit = torch.where(
            endpoint_alive, state["K"], torch.zeros_like(state["K"])
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
            state["economic_node_id"][path_idx].to(torch.float32),
            torch.full_like(b, 0.0 if branch_k == -1 else 1.0),
            torch.full_like(b, 1.0 if reused_account else 0.0),
            state["transition_eps_z"].reshape(-1)[alive_flat],
            state["transition_u_eta"].reshape(-1)[alive_flat],
            state["transition_u_i"].reshape(-1)[alive_flat],
        ],
        dim=1,
    ).to(torch.float32)

    use_resolved_bp = (
        getattr(sim, "bp_action_override", None) is not None
        or getattr(sim, "bp_action_source", "head") == "grid"
    )
    if use_resolved_bp:
        if output is None:
            raise RuntimeError(
                "resolved BP action requires a policy_value model output"
            )
        overridden = sim._resolve_bp_action(
            sim=sim,
            firm_state=firm_state.detach(),
            model_output=output,
            bp_head=bp.detach().reshape(-1, 1),
            hatc_cal=Hatc[path_idx].detach().reshape(-1, 1),
            lnk_cal=LnK[path_idx].detach().reshape(-1, 1),
            path_index=path_idx.detach(),
            firm_id=firm_id.detach(),
            t=int(t),
            branch=int(branch_k),
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
            C_total,
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
        ],
        dim=1,
    ).to(torch.float32)

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
    branches: List[Dict[str, torch.Tensor]] = []
    b_prev = state["b"]
    bp_prev = state.get("bp", b_prev)
    entry_reference = compute_entry_reference(
        state["K"],
        state["alive"],
        entry_capital_ratio=float(getattr(sim, "entry_capital_ratio", 0.10)),
        entry_size_ratio=float(getattr(sim, "entry_size_ratio", 0.10)),
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
        branches.append(
            {
                "x": torch.where(alive_any, x_next, state["x"]),
                "b": torch.where(active2d, b_next, state["b"]),
                "z": torch.where(active2d, z_next, state["z"]),
                "eta": torch.where(active2d, eta_next, state["eta"]),
                "i": torch.where(active2d, i_next, state["i"]),
                "K": torch.where(active2d, K_next, state["K"]),
                "hatcf": hatcf_out,
                "lnkf": lnkf_out,
                "M": M_out,
                "alive": state["alive"].clone(),
                "entry": torch.zeros_like(state["entry"]),
                "initial_cohort": state["initial_cohort"].clone(),
                "birth_time": state["birth_time"].clone(),
                "entry_cost": state["entry_cost"].clone(),
                "K_birth": state["K_birth"].clone(),
                "firm_id": state["firm_id"].clone(),
                "next_firm_id": state["next_firm_id"].clone(),
                "bar_i": state.get("bar_i", torch.zeros_like(state["b"])) .clone(),
                "bar_z": state.get("bar_z", torch.zeros_like(state["b"])) .clone(),
                "bp": state.get("bp", state["b"]).clone(),
                "economic_node_id": torch.full(
                    (state["b"].shape[0],),
                    int(child_t * (sim.branch_num + 1) + branch_index + 1),
                    dtype=torch.long,
                    device=device,
                ),
                "economic_time": torch.full(
                    (state["b"].shape[0],),
                    int(child_t),
                    dtype=torch.long,
                    device=device,
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
                "event_reference_K": entry_reference.reference_K.clone(),
                "event_reference_N": entry_reference.reference_N.to(torch.float32),
                "event_mean_K_ref": entry_reference.mean_K_ref.clone(),
                "event_K_per_entrant": entry_reference.K_per_entrant.clone(),
                "event_potential_capital_nominal": entry_reference.potential_capital_nominal.clone(),
                "event_candidate_capital_realized": torch.zeros(state["b"].shape[0], device=device),
                "event_K_entry_gross": torch.zeros(state["b"].shape[0], device=device),
                "event_same_node_entrant_exit_count": torch.zeros(state["b"].shape[0], device=device),
                "event_K_entry_same_node_exit": torch.zeros(state["b"].shape[0], device=device),
                "event_K_entry_surviving": torch.zeros(state["b"].shape[0], device=device),
                "event_entry_cutoff_mean": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_cutoff_p10": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_cutoff_p50": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_cutoff_p90": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_cost_mean": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_cost_p10": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_cost_p50": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "event_entry_cost_p90": torch.full((state["b"].shape[0],), float("nan"), device=device),
                "transition_previous_ids": state["firm_id"].clone(),
                "transition_previous_K": state["K"].clone(),
                "transition_previous_alive": state["alive"].clone(),
                "transition_previous_node_id": state["economic_node_id"].clone(),
            }
        )
    return branches


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
    return state


def _apply_entry_batched(sim, state: Dict[str, torch.Tensor], n_potential: int) -> Dict[str, torch.Tensor]:
    if getattr(sim, "entry_mode", "legacy") == "value_cost":
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
            if key in {"transition_eps_z", "transition_u_eta", "transition_u_i"}
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
    reference = compute_entry_reference(
        state["transition_previous_K"],
        state["transition_previous_alive"],
        entry_capital_ratio=sim.entry_capital_ratio,
        entry_size_ratio=sim.entry_size_ratio,
    )
    candidates = draw_value_cost_candidates(
        reference,
        rho_z=float(sim.config.RHO_Z),
        sigma_z=float(sim.config.SIGMA_Z),
        zbar=float(sim.config.ZBAR),
        entry_cost_max=sim.entry_cost_max,
        generator=sim.entry_generator,
    )
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
    accepted_counts = accepted.sum(dim=1).to(torch.long)
    required = int((state["next_firm_id"] + accepted_counts).max().item()) if n_paths else 0
    _expand_firm_capacity(sim, state, required)

    state["entry"].zero_()
    state["event_potential_count"] = candidates.counts.to(torch.float32)
    state["event_accepted_count"] = accepted_counts.to(torch.float32)
    state["event_reference_K"] = reference.reference_K
    state["event_reference_N"] = reference.reference_N.to(torch.float32)
    state["event_mean_K_ref"] = reference.mean_K_ref
    state["event_K_per_entrant"] = reference.K_per_entrant
    state["event_potential_capital_nominal"] = candidates.potential_capital_nominal
    state["event_candidate_capital_realized"] = candidates.candidate_capital_realized
    cutoff_summary = summarize_candidate_values(
        screened["entry_cutoff"], candidates.mask
    )
    cost_summary = summarize_candidate_values(
        candidates.entry_cost, candidates.mask
    )
    for statistic, values in cutoff_summary.items():
        state[f"event_entry_cutoff_{statistic}"] = values
    for statistic, values in cost_summary.items():
        state[f"event_entry_cost_{statistic}"] = values
    state["event_I_entry"] = torch.where(
        accepted,
        candidates.entry_cost * candidates.K_birth,
        torch.zeros_like(candidates.entry_cost),
    ).sum(dim=1)
    state["event_K_entry_gross"] = torch.where(
        accepted, candidates.K_birth, torch.zeros_like(candidates.K_birth)
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
    # Draw eta and ordinary investment cost only after the entry screen.
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
    state["alive"][path_idx, slot_idx] = True
    state["entry"][path_idx, slot_idx] = 1.0
    state["initial_cohort"][path_idx, slot_idx] = 0.0
    state["birth_time"][path_idx, slot_idx] = state["economic_time"][path_idx].to(
        state["birth_time"].dtype
    )
    state["entry_cost"][path_idx, slot_idx] = cost_new
    state["K_birth"][path_idx, slot_idx] = K_new
    state["firm_id"][path_idx, slot_idx] = new_id
    state["bar_i"][path_idx, slot_idx] = 0.0
    state["bar_z"][path_idx, slot_idx] = 0.0
    state["bp"][path_idx, slot_idx] = 0.0
    state["transition_eps_z"][path_idx, slot_idx] = float("nan")
    state["transition_u_eta"][path_idx, slot_idx] = float("nan")
    state["transition_u_i"][path_idx, slot_idx] = float("nan")
    state["next_firm_id"] = state["next_firm_id"] + accepted_counts
    return state
