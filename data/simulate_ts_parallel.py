from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from .data_utils import sample_ar1, sample_bernoulli, sample_stationary_ar1, sample_uniform
from .tensor_data import TensorSimulationOutput, TensorTable, cat_rows


def simulate_tensor_parallel(sim) -> TensorSimulationOutput:
    """
    Simulate all paths in parallel on device.

    Time remains sequential because the state transition is recursive, but the
    path and firm dimensions are advanced as batched tensors.
    """
    device = sim.device
    n_potential = max(20, int(sim.group_size * 0.1))
    max_firms = sim.group_size + sim.horizon * n_potential
    state = _initialize_batched_state(sim, max_firms)

    firm_rows: List[torch.Tensor] = []
    macro_rows: List[torch.Tensor] = []

    for t in tqdm(range(sim.horizon), desc="Simulating steps (tensor)"):
        parent_firm, parent_macro = _process_node_batched(sim, state, t, branch_k=-1)
        firm_rows.append(parent_firm)
        macro_rows.append(parent_macro)

        branch_states = _expand_branches_batched(sim, state)
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
            "max_firms": max_firms,
        },
    )


def _initialize_batched_state(sim, max_firms: int) -> Dict[str, torch.Tensor]:
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
    entry[:, :n0] = 1.0
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
        "firm_id": firm_id,
        "next_firm_id": torch.full((n_paths,), n0, dtype=torch.long, device=device),
        "bar_i": torch.zeros_like(b),
        "bar_z": torch.zeros_like(b),
        "bp": b.clone(),
    }


def _process_node_batched(sim, state: Dict[str, torch.Tensor], t: int, branch_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    device = sim.device
    n_paths, n_firms = state["b"].shape
    alive = state["alive"]
    alive_flat = alive.reshape(-1)
    alive_pos = torch.nonzero(alive_flat, as_tuple=False).squeeze(-1)
    n_alive_per_path = alive.sum(dim=1)

    path_grid = torch.arange(n_paths, device=device, dtype=torch.long).unsqueeze(1).expand(n_paths, n_firms)

    if alive_pos.numel() == 0:
        macro_rows = torch.stack(
            [
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
            ],
            dim=1,
        )
        firm_empty = torch.empty((0, len(sim.FIRM_COLUMNS)), device=device, dtype=torch.float32)
        return firm_empty, macro_rows

    path_idx = path_grid.reshape(-1)[alive_flat]
    b = state["b"].reshape(-1)[alive_flat]
    z = state["z"].reshape(-1)[alive_flat]
    eta = state["eta"].reshape(-1)[alive_flat]
    i = state["i"].reshape(-1)[alive_flat]
    K = state["K"].reshape(-1)[alive_flat]
    entry = state["entry"].reshape(-1)[alive_flat]
    firm_id = state["firm_id"].reshape(-1)[alive_flat].to(torch.float32)
    x = state["x"][path_idx]
    hatcf = state["hatcf"][path_idx]
    lnkf = state["lnkf"][path_idx]
    m_vec = state["M"][path_idx]

    firm_state = torch.stack([b, z, eta, i, x, hatcf, lnkf], dim=1)
    pv_model = sim.models.get("policy_value")
    if pv_model is not None:
        with torch.no_grad():
            output = pv_model(firm_state)
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

    Y, I, Phi, C = sim._resource_accounting(K, z, x, bar_i, bar_z, i)

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
            Y,
            I,
            Phi,
            C,
        ],
        dim=1,
    ).to(torch.float32)

    K_total = torch.zeros(n_paths, device=device)
    C_total = torch.zeros(n_paths, device=device)
    K_total.index_add_(0, path_idx, K)
    C_total.index_add_(0, path_idx, C.clamp(min=0.0))
    alive_any = n_alive_per_path > 0
    LnK = torch.where(alive_any, torch.log(K_total + 1e-8), torch.full_like(K_total, -10.0))
    Hatc_raw = torch.log(C_total / (K_total + 1e-8) + 1e-5)
    Hatc = torch.where(alive_any, Hatc_raw, torch.full_like(C_total, -10.0))

    macro_rows = torch.stack(
        [
            torch.arange(n_paths, device=device, dtype=torch.float32),
            torch.full((n_paths,), float(t), device=device),
            torch.full((n_paths,), float(branch_k), device=device),
            K_total,
            C_total,
            LnK,
            Hatc,
            n_alive_per_path.to(torch.float32),
            state["M"].to(torch.float32),
            state["x"].to(torch.float32),
            state["hatcf"].to(torch.float32),
            state["lnkf"].to(torch.float32),
        ],
        dim=1,
    ).to(torch.float32)

    state["hatcf"] = torch.where(alive_any, Hatc.detach(), state["hatcf"])
    state["lnkf"] = torch.where(alive_any, LnK.detach(), state["lnkf"])

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


def _expand_branches_batched(sim, state: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    device = sim.device
    alive_any = state["alive"].any(dim=1)
    branches: List[Dict[str, torch.Tensor]] = []

    for _ in range(sim.branch_num):
        x_next = sample_ar1(state["x"], sim.config.RHO_X, sim.config.SIGMA_X, sim.config.XBAR)
        z_next = sample_ar1(state["z"], sim.config.RHO_Z, sim.config.SIGMA_Z, sim.config.ZBAR)
        eta_next = sample_bernoulli(state["eta"].numel(), sim.config.ZETA, device).view_as(state["eta"])
        i_next = sample_uniform(state["i"].numel(), 0.0, sim.config.I_THRESHOLD, device).view_as(state["i"])

        b_prev = state["b"]
        bp_prev = state.get("bp", b_prev)
        b_next = eta_next * bp_prev + (1.0 - eta_next) * b_prev

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
                state["hatcf"],
                state["lnkf"],
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
                "firm_id": state["firm_id"].clone(),
                "next_firm_id": state["next_firm_id"].clone(),
                "bar_i": state.get("bar_i", torch.zeros_like(state["b"])) .clone(),
                "bar_z": state.get("bar_z", torch.zeros_like(state["b"])) .clone(),
                "bp": state.get("bp", state["b"]).clone(),
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
    state["firm_id"][path_idx, slot_idx] = new_id

    for key in ["bar_i", "bar_z", "bp"]:
        if key in state:
            fill_val = 0.0 if key != "bp" else state["b"][path_idx, slot_idx]
            state[key][path_idx, slot_idx] = fill_val

    state["next_firm_id"] = torch.clamp(state["next_firm_id"] + enter_counts, max=max_firms)
    return state
