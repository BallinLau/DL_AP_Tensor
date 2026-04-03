from __future__ import annotations

from typing import Optional, Tuple

import torch

from .data_utils import (
    generate_initial_macro_proxy,
    sample_ar1,
    sample_bernoulli,
    sample_stationary_ar1,
    sample_uniform,
)
from .tensor_data import TensorTable


PV_COLUMNS = [
    'path', 'firm', 'branch', 'Entry',
    'b', 'z', 'ETA', 'i', 'x', 'Hatcf', 'LnKF', 'M', 'K'
]

MACRO_COLUMNS = [
    'path', 't_code', 'branch', 'x', 'x_t', 'x_t1',
    'n_firms', 'n_entrants', 'Hatcf', 'LnKF', 'Hatcf_t', 'LnKF_t', 'M'
]


def build_policy_value_tables_parallel(sample, include_macro: bool = False) -> Tuple[TensorTable, Optional[TensorTable]]:
    if sample.sampling_mode != 'uniform':
        raise NotImplementedError('parallel sample builder currently supports sampling_mode=uniform only')

    device = sample.device
    n_paths = sample.n_paths
    group_size = sample.group_size
    branch_num = sample.branch_num

    macro_parent = sample._sample_parent_macro_state(n_paths, device)
    if macro_parent is not None:
        x_t, hatcf_t, lnkf_t = macro_parent
    else:
        x_t = sample_stationary_ar1(n_paths, sample.config.RHO_X, sample.config.SIGMA_X, sample.config.XBAR, device)
        hatcf_t, lnkf_t = generate_initial_macro_proxy(n_paths, device)
    b = sample_uniform(n_paths * group_size, 0.0, 1.0, device).view(n_paths, group_size)
    z = sample_stationary_ar1(
        n_paths * group_size, sample.config.RHO_Z, sample.config.SIGMA_Z, sample.config.ZBAR, device
    ).view(n_paths, group_size)
    eta = sample_bernoulli(n_paths * group_size, sample.config.ZETA, device).view(n_paths, group_size)
    i = sample_uniform(n_paths * group_size, 0.0, sample.config.I_THRESHOLD, device).view(n_paths, group_size)
    if sample.data_mode == 'simulate':
        weights = torch.rand(n_paths, group_size, device=device)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        K = torch.exp(lnkf_t).unsqueeze(1) * weights
    else:
        K = torch.ones(n_paths, group_size, device=device)

    parent_firm = (torch.arange(n_paths, device=device, dtype=torch.long).unsqueeze(1) * group_size)
    parent_firm = parent_firm + torch.arange(group_size, device=device, dtype=torch.long).unsqueeze(0)

    path_parent = torch.arange(n_paths, device=device, dtype=torch.float32).unsqueeze(1).expand(n_paths, group_size)
    parent_rows = torch.stack(
        [
            path_parent.reshape(-1),
            parent_firm.reshape(-1).to(torch.float32),
            torch.zeros(n_paths * group_size, device=device),
            torch.ones(n_paths * group_size, device=device),
            b.reshape(-1),
            z.reshape(-1),
            eta.reshape(-1),
            i.reshape(-1),
            x_t.unsqueeze(1).expand(n_paths, group_size).reshape(-1),
            hatcf_t.unsqueeze(1).expand(n_paths, group_size).reshape(-1),
            lnkf_t.unsqueeze(1).expand(n_paths, group_size).reshape(-1),
            torch.ones(n_paths * group_size, device=device),
            K.reshape(-1),
        ],
        dim=1,
    ).to(torch.float32)

    x_t1 = sample_ar1(
        x_t.unsqueeze(1).expand(n_paths, branch_num).reshape(-1),
        sample.config.RHO_X,
        sample.config.SIGMA_X,
        sample.config.XBAR,
    ).view(n_paths, branch_num)
    z_t1 = sample_ar1(
        z.unsqueeze(1).expand(n_paths, branch_num, group_size).reshape(-1),
        sample.config.RHO_Z,
        sample.config.SIGMA_Z,
        sample.config.ZBAR,
    ).view(n_paths, branch_num, group_size)
    eta_t1 = sample_bernoulli(n_paths * branch_num * group_size, sample.config.ZETA, device).view(n_paths, branch_num, group_size)
    i_t1 = sample_uniform(
        n_paths * branch_num * group_size, 0.0, sample.config.I_THRESHOLD, device
    ).view(n_paths, branch_num, group_size)

    if sample.models.get('sdf_fc1') is not None:
        with torch.no_grad():
            _, _, M_t1, hatcf_t1, lnkf_t1 = sample.models['sdf_fc1'].forward_step(
                x_prev=x_t.unsqueeze(1).expand(n_paths, branch_num).reshape(-1).to(torch.float32),
                x_curr=x_t1.reshape(-1).to(torch.float32),
                hatcf_prev=hatcf_t.unsqueeze(1).expand(n_paths, branch_num).reshape(-1).to(torch.float32),
                lnkf_prev=lnkf_t.unsqueeze(1).expand(n_paths, branch_num).reshape(-1).to(torch.float32),
                return_physical=True,
            )
        M_branch = M_t1.reshape(n_paths, branch_num)
        hatcf_branch = hatcf_t1.reshape(n_paths, branch_num)
        lnkf_branch = lnkf_t1.reshape(n_paths, branch_num)
    else:
        M_branch = torch.ones(n_paths, branch_num, device=device)
        hatcf_branch = hatcf_t.unsqueeze(1).expand(n_paths, branch_num)
        lnkf_branch = lnkf_t.unsqueeze(1).expand(n_paths, branch_num)

    path_child = torch.arange(n_paths, device=device, dtype=torch.float32).view(n_paths, 1, 1).expand(n_paths, branch_num, group_size)
    firm_child = parent_firm.unsqueeze(1).expand(n_paths, branch_num, group_size).to(torch.float32)
    branch_child = (torch.arange(branch_num, device=device, dtype=torch.float32).view(1, branch_num, 1) + 1).expand(n_paths, branch_num, group_size)

    child_rows = torch.stack(
        [
            path_child.reshape(-1),
            firm_child.reshape(-1),
            branch_child.reshape(-1),
            torch.zeros(n_paths * branch_num * group_size, device=device),
            b.unsqueeze(1).expand(n_paths, branch_num, group_size).reshape(-1),
            z_t1.reshape(-1),
            eta_t1.reshape(-1),
            i_t1.reshape(-1),
            x_t1.unsqueeze(-1).expand(n_paths, branch_num, group_size).reshape(-1),
            hatcf_branch.unsqueeze(-1).expand(n_paths, branch_num, group_size).reshape(-1),
            lnkf_branch.unsqueeze(-1).expand(n_paths, branch_num, group_size).reshape(-1),
            M_branch.unsqueeze(-1).expand(n_paths, branch_num, group_size).reshape(-1),
            K.unsqueeze(1).expand(n_paths, branch_num, group_size).reshape(-1),
        ],
        dim=1,
    ).to(torch.float32)

    rows = [parent_rows, child_rows]
    macro_table = None

    entrants_per_branch = None
    if sample.data_mode == 'simulate' and sample.enable_entry:
        n_potential = max(10, int(group_size * sample.entry_rate))
        z_potential = sample_stationary_ar1(
            n_paths * branch_num * n_potential,
            sample.config.RHO_Z,
            sample.config.SIGMA_Z,
            sample.config.ZBAR,
            device,
        ).view(n_paths, branch_num, n_potential)
        i_potential = sample_uniform(
            n_paths * branch_num * n_potential, 0.0, sample.config.I_THRESHOLD, device
        ).view(n_paths, branch_num, n_potential)
        profit = torch.exp(x_t1.unsqueeze(-1) + z_potential) - sample.config.DELTA
        entry_value = 1.0 + profit - i_potential
        enter_mask = entry_value > 0.0
        entrants_per_branch = enter_mask.sum(dim=-1)

        if bool(enter_mask.any()):
            eta_enter = sample_bernoulli(n_paths * branch_num * n_potential, sample.config.ZETA, device).view(n_paths, branch_num, n_potential)
            entryK = K.mean(dim=1, keepdim=True).unsqueeze(1).expand(n_paths, branch_num, n_potential)
            path_grid = torch.arange(n_paths, device=device, dtype=torch.long).view(n_paths, 1, 1).expand(n_paths, branch_num, n_potential)
            branch_grid = torch.arange(branch_num, device=device, dtype=torch.long).view(1, branch_num, 1).expand(n_paths, branch_num, n_potential)
            valid_idx = torch.nonzero(enter_mask.reshape(-1), as_tuple=False).squeeze(-1)
            n_valid = int(valid_idx.numel())
            entrant_ids = torch.arange(n_paths * group_size, n_paths * group_size + n_valid, device=device, dtype=torch.float32)

            entrant_rows = torch.stack(
                [
                    path_grid.reshape(-1)[valid_idx].to(torch.float32),
                    entrant_ids,
                    (branch_grid.reshape(-1)[valid_idx].to(torch.float32) + 1.0),
                    torch.ones(n_valid, device=device),
                    torch.zeros(n_valid, device=device),
                    z_potential.reshape(-1)[valid_idx],
                    eta_enter.reshape(-1)[valid_idx],
                    i_potential.reshape(-1)[valid_idx],
                    x_t1.unsqueeze(-1).expand(n_paths, branch_num, n_potential).reshape(-1)[valid_idx],
                    hatcf_branch.unsqueeze(-1).expand(n_paths, branch_num, n_potential).reshape(-1)[valid_idx],
                    lnkf_branch.unsqueeze(-1).expand(n_paths, branch_num, n_potential).reshape(-1)[valid_idx],
                    M_branch.unsqueeze(-1).expand(n_paths, branch_num, n_potential).reshape(-1)[valid_idx],
                    entryK.reshape(-1)[valid_idx],
                ],
                dim=1,
            ).to(torch.float32)
            rows.append(entrant_rows)

    firm_table = TensorTable(torch.cat(rows, dim=0), PV_COLUMNS)

    if include_macro and sample.data_mode == 'simulate':
        n_entrants_total = entrants_per_branch.sum(dim=1) if entrants_per_branch is not None else torch.zeros(n_paths, device=device, dtype=torch.long)
        parent_macro = torch.stack(
            [
                torch.arange(n_paths, device=device, dtype=torch.float32),
                torch.zeros(n_paths, device=device),
                torch.full((n_paths,), -1.0, device=device),
                x_t,
                x_t,
                torch.full((n_paths,), float('nan'), device=device),
                torch.full((n_paths,), float(group_size), device=device),
                n_entrants_total.to(torch.float32),
                hatcf_t,
                lnkf_t,
                hatcf_t,
                lnkf_t,
                torch.ones(n_paths, device=device),
            ],
            dim=1,
        ).to(torch.float32)

        child_macro = torch.stack(
            [
                torch.arange(n_paths, device=device, dtype=torch.float32).view(n_paths, 1).expand(n_paths, branch_num).reshape(-1),
                (torch.arange(branch_num, device=device, dtype=torch.float32).view(1, branch_num).expand(n_paths, branch_num) + 1.0).reshape(-1),
                torch.arange(branch_num, device=device, dtype=torch.float32).view(1, branch_num).expand(n_paths, branch_num).reshape(-1),
                x_t1.reshape(-1),
                x_t.unsqueeze(1).expand(n_paths, branch_num).reshape(-1),
                x_t1.reshape(-1),
                torch.full((n_paths * branch_num,), float(group_size), device=device),
                (entrants_per_branch.reshape(-1).to(torch.float32) if entrants_per_branch is not None else torch.zeros(n_paths * branch_num, device=device)),
                hatcf_branch.reshape(-1),
                lnkf_branch.reshape(-1),
                hatcf_t.unsqueeze(1).expand(n_paths, branch_num).reshape(-1),
                lnkf_t.unsqueeze(1).expand(n_paths, branch_num).reshape(-1),
                M_branch.reshape(-1),
            ],
            dim=1,
        ).to(torch.float32)
        macro_table = TensorTable(torch.cat([parent_macro, child_macro], dim=0), MACRO_COLUMNS)

    return firm_table, macro_table
