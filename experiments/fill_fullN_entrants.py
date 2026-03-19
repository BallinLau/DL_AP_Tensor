"""
Utility to fill each (path, branch) slice of an existing firm-level DataFrame
up to `full_N` firms by sampling additional entrants using the same logic as
`Sample._generate_entrants`.

- Only branches > 0 (t+1 layers) are expanded; entrants have no parent rows.
- Existing firms are kept; new firms are appended with `Entry=1`.

Example
-------
python -m experiments.fill_fullN_entrants \
    --pkl ../cachedir/20260203_1522/data/outputs/ep0_stage_pv.pkl \
    --out ../cachedir/20260203_1522/data/outputs/ep0_stage_pv_full.pkl \
    --full-N 2000
"""
from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
import torch

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from data.data_utils import (  # noqa: E402
    sample_stationary_ar1,
    sample_uniform,
    sample_bernoulli,
)


def _fast_sample_branch(
    need: int,
    meta: Tuple[int, int, float, float, float, float, float],
    cfg: type,
    device: torch.device,
    entry_num: Optional[int] = None,
) -> List[Dict]:
    """
    Faster entrant sampler for one (path, branch) using a single oversized draw.
    meta = (path_idx, branch, x_t1, hatcf_t1, lnkf_t1, M_t1, entry_k)
    entry_num: number of entrants to flag with Entry=1 (randomly); the rest get 0.
               If None, all generated entrants are flagged 1.
    """
    path_idx, branch, x_t1, hatcf_t1, lnkf_t1, M_t1, entry_k = meta
    branch_k = branch - 1  # branch label for t string

    # Oversample once; fallback to a second smaller batch only if strictly needed
    def draw(batch_need: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_potential = max(10, batch_need * 4)  # aggressive oversample to avoid loops
        z = sample_stationary_ar1(n_potential, cfg.RHO_Z, cfg.SIGMA_Z, cfg.ZBAR, device)
        i_cost = sample_uniform(n_potential, 0.0, cfg.I_THRESHOLD, device)
        profit = torch.exp(torch.tensor(x_t1, device=device) + z) - cfg.DELTA
        entry_value = 1 + profit - i_cost
        mask = entry_value > 0
        return z[mask], i_cost[mask], torch.full((mask.sum(),), entry_k, device=device)

    z_acc, i_acc, k_acc = draw(need)
    if len(z_acc) < need:
        extra_z, extra_i, extra_k = draw(need - len(z_acc))
        z_acc = torch.cat([z_acc, extra_z], dim=0)
        i_acc = torch.cat([i_acc, extra_i], dim=0)
        k_acc = torch.cat([k_acc, extra_k], dim=0)

    # Truncate to exact need
    z_acc = z_acc[:need]
    i_acc = i_acc[:need]
    k_acc = k_acc[:need]
    eta_acc = sample_bernoulli(need, cfg.ZETA, device)

    # Randomly choose which entrants are flagged as Entry=1
    if entry_num is None:
        entry_flags = torch.ones(need, dtype=torch.int64, device=device)
    else:
        k = max(0, min(entry_num, need))
        perm = torch.randperm(need, device=device)
        entry_flags = torch.zeros(need, dtype=torch.int64, device=device)
        entry_flags[perm[:k]] = 1

    entrants = [
        {
            'path': path_idx,
            'ID': uuid.uuid4().hex,
            't': f't+1_{branch_k}',
            'branch': branch,
            'b': 0.0,
            'z': float(z_acc[j]),
            'ETA': float(eta_acc[j]),
            'i': float(i_acc[j]),
            'x': float(x_t1),
            'Hatcf': float(hatcf_t1),
            'LnKF': float(lnkf_t1),
            'M': float(M_t1),
            'K': float(k_acc[j]),
            'Entry': int(entry_flags[j]),
        }
        for j in range(need)
    ]
    return entrants


def fill_df_to_fullN(
    df: pd.DataFrame,
    full_N: int,
    cfg: type = Config,
    device: torch.device | None = None,
    entry_num: Optional[int] = None,
) -> pd.DataFrame:
    """Ensure each path has exactly `full_N` companies shared across parent and all child branches.

    - IDs are aligned across branch=0 and all child branches.
    - Existing rows are preserved; missing rows per (ID, branch) are created.
    - New companies are sampled to reach `full_N` per path; parent + child rows are added.
    - `entry_num` (optional): for each child branch, randomly flag at most `entry_num`
      of the newly added companies as Entry=1 (others set 0). Existing child rows keep
      their original Entry flags.
    """
    device = device or cfg.DEVICE

    required_cols = {'path', 'branch', 'x', 'Hatcf', 'LnKF'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame缺少必要字段: {missing}")

    child_branches = sorted(b for b in df['branch'].unique() if b > 0)
    if not child_branches:
        raise ValueError("DataFrame中没有 child branches (branch > 0)")

    new_rows: List[Dict] = []

    # Precompute parent mean K per path
    parent_mean_K = (
        df[df['branch'] == 0]
        .groupby('path')['K']
        .mean()
        .to_dict()
    )

    for path_idx in sorted(df['path'].unique()):
        df_path = df[df['path'] == path_idx]

        # Parent macros (assume one parent x/hatcf/lnkf per path)
        parent_rows = df_path[df_path['branch'] == 0]
        if parent_rows.empty:
            raise ValueError(f"path {path_idx} 缺少 parent 行 (branch=0)")
        parent_x = parent_rows['x'].iloc[0]
        parent_hatcf = parent_rows['Hatcf'].iloc[0]
        parent_lnkf = parent_rows['LnKF'].iloc[0]
        entry_k = parent_mean_K.get(path_idx, 1.0)

        # Branch-level macro states
        branch_macro = {}
        for b in child_branches:
            gb = df_path[df_path['branch'] == b]
            if gb.empty:
                raise ValueError(f"path {path_idx} 缺少 branch={b} 数据行以提供宏观状态")
            branch_macro[b] = {
                'x': gb['x'].iloc[0],
                'Hatcf': gb['Hatcf'].iloc[0],
                'LnKF': gb['LnKF'].iloc[0],
                'M': gb['M'].iloc[0] if 'M' in gb else 1.0,
            }

        # Existing IDs (from parent rows)
        existing_ids = list(parent_rows['ID'].unique())
        n_existing = len(existing_ids)

        # Fill missing child rows for existing IDs (sampled, Entry=0)
        for b in child_branches:
            # IDs missing this branch
            have_ids = set(df_path[df_path['branch'] == b]['ID'])
            missing_ids = [fid for fid in existing_ids if fid not in have_ids]
            if not missing_ids:
                continue
            bm = branch_macro[b]
            m = len(missing_ids)
            z_child = sample_stationary_ar1(m, cfg.RHO_Z, cfg.SIGMA_Z, cfg.ZBAR, device)
            i_child = sample_uniform(m, 0.0, cfg.I_THRESHOLD, device)
            eta_child = sample_bernoulli(m, cfg.ZETA, device)
            for j, firm_id in enumerate(missing_ids):
                new_rows.append(
                    {
                        'path': path_idx,
                        'ID': firm_id,
                        't': f't+1_{b-1}',
                        'branch': b,
                        'b': 0.0,
                        'z': float(z_child[j]),
                        'ETA': float(eta_child[j]),
                        'i': float(i_child[j]),
                        'x': bm['x'],
                        'Hatcf': bm['Hatcf'],
                        'LnKF': bm['LnKF'],
                        'M': bm['M'],
                        'K': entry_k,
                        'Entry': 0,
                    }
                )

        # Need to add new companies?
        need_new = max(0, full_N - n_existing)
        if need_new == 0:
            continue

        # Generate new IDs
        ids_new = [uuid.uuid4().hex for _ in range(need_new)]

        # Add parent rows for new companies (only ID kept; others NaN)
        for firm_id in ids_new:
            new_rows.append(
                {
                    'path': path_idx,
                    'ID': firm_id,
                    't': 't',
                    'branch': 0,
                    'b': np.nan,
                    'z': np.nan,
                    'ETA': np.nan,
                    'i': np.nan,
                    'x': np.nan,
                    'Hatcf': np.nan,
                    'LnKF': np.nan,
                    'M': np.nan,
                    'K': np.nan,
                    'Entry': np.nan,
                }
            )

        # Add child rows for new companies
        for b in child_branches:
            bm = branch_macro[b]

            z_child = sample_stationary_ar1(need_new, cfg.RHO_Z, cfg.SIGMA_Z, cfg.ZBAR, device)
            i_child = sample_uniform(need_new, 0.0, cfg.I_THRESHOLD, device)
            eta_child = sample_bernoulli(need_new, cfg.ZETA, device)

            if entry_num is None:
                entry_flags = torch.ones(need_new, dtype=torch.int64, device=device)
            else:
                k = max(0, min(entry_num, need_new))
                perm = torch.randperm(need_new, device=device)
                entry_flags = torch.zeros(need_new, dtype=torch.int64, device=device)
                entry_flags[perm[:k]] = 1

            for j, firm_id in enumerate(ids_new):
                new_rows.append(
                    {
                        'path': path_idx,
                        'ID': firm_id,
                        't': f't+1_{b-1}',
                        'branch': b,
                        'b': 0.0,
                        'z': float(z_child[j]),
                        'ETA': float(eta_child[j]),
                        'i': float(i_child[j]),
                        'x': bm['x'],
                        'Hatcf': bm['Hatcf'],
                        'LnKF': bm['LnKF'],
                        'M': bm['M'],
                        'K': entry_k,
                        'Entry': int(entry_flags[j]),
                    }
                )

    if not new_rows:
        return df

    df_new = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df_new


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Fill each path-branch with entrants to reach full_N.'
    )
    parser.add_argument('--pkl', type=Path, required=True, help='Input pickle path')
    parser.add_argument('--out', type=Path, required=True, help='Output pickle path')
    parser.add_argument('--full-N', dest='full_N', type=int, default=2000)
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Torch device to use (default: cpu)',
    )
    parser.add_argument(
        '--entry-num',
        type=int,
        default=None,
        help='Number of entrants per branch to flag Entry=1 (random). '
             'Default: all newly added entrants are Entry=1.',
    )
    args = parser.parse_args()

    df = pd.read_pickle(args.pkl)
    device = torch.device(args.device)
    df_filled = fill_df_to_fullN(
        df,
        full_N=args.full_N,
        device=device,
        entry_num=args.entry_num,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_filled.to_pickle(args.out)
    print(
        f"Saved filled DataFrame to {args.out} with shape {df_filled.shape}, "
        f"added {len(df_filled) - len(df)} rows."
    )


if __name__ == '__main__':
    main()
