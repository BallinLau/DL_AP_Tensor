"""
FC2 Pipeline.

Phase A goal:
- keep the economic logic unchanged
- add a tensor-native input path so FC2 can consume simulation tensors directly
"""
from __future__ import annotations

from typing import Optional, Dict, Sequence, List, Tuple

import numpy as np
import pandas as pd
import torch

from config import Config
from data.tensor_data import TensorTable
from experiments.fill_fullN_entrants import fill_df_to_fullN

# Helpers

def compute_quantile_features(b_vals: torch.Tensor, z_vals: torch.Tensor, quantiles: torch.Tensor):
    b_quantiles = torch.quantile(b_vals, quantiles)
    z_quantiles = torch.quantile(z_vals, quantiles)
    quantile_features = torch.cat([b_quantiles, z_quantiles], dim=0)
    return quantile_features


def compute_fit_stats_t(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    y_true = y_true.reshape(-1).to(torch.float32)
    y_pred = y_pred.reshape(-1).to(torch.float32)
    mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.numel() == 0:
        return {
            'n': 0.0,
            'mean_true': float('nan'),
            'mean_pred': float('nan'),
            'mean_resid': float('nan'),
            'mae_resid': float('nan'),
            'rmse_resid': float('nan'),
            'std_true': float('nan'),
            'std_pred': float('nan'),
            'std_ratio': float('nan'),
            'corr': float('nan'),
            'slope': float('nan'),
            'intercept': float('nan'),
        }

    resid = y_true - y_pred
    mean_true = float(y_true.mean().item())
    mean_pred = float(y_pred.mean().item())
    mean_resid = float(resid.mean().item())
    mae_resid = float(resid.abs().mean().item())
    rmse_resid = float(torch.sqrt((resid.pow(2)).mean()).item())
    std_true_t = y_true.std(unbiased=False)
    std_pred_t = y_pred.std(unbiased=False)
    std_true = float(std_true_t.item())
    std_pred = float(std_pred_t.item())
    std_ratio = float(std_pred / std_true) if std_true > 1e-12 else float('nan')

    if y_true.numel() < 2 or std_true <= 1e-12 or std_pred <= 1e-12:
        corr = float('nan')
        slope = float('nan')
        intercept = float('nan')
    else:
        y_true_center = y_true - y_true.mean()
        y_pred_center = y_pred - y_pred.mean()
        denom = torch.sqrt((y_true_center.pow(2)).sum() * (y_pred_center.pow(2)).sum()).clamp_min(1e-12)
        corr = float((y_true_center * y_pred_center).sum().div(denom).item())
        var_true = y_true_center.pow(2).mean().clamp_min(1e-12)
        cov = (y_true_center * y_pred_center).mean()
        slope = float((cov / var_true).item())
        intercept = float((y_pred.mean() - slope * y_true.mean()).item())

    return {
        'n': float(y_true.numel()),
        'mean_true': mean_true,
        'mean_pred': mean_pred,
        'mean_resid': mean_resid,
        'mae_resid': mae_resid,
        'rmse_resid': rmse_resid,
        'std_true': std_true,
        'std_pred': std_pred,
        'std_ratio': std_ratio,
        'corr': corr,
        'slope': slope,
        'intercept': intercept,
    }

class FC2Pipeline:
    """
    OOP pipeline aligned with losses/FC2losspipe.py
    """
    def __init__(
        self,
        pkl_path=None,
        df=None,
        firm_table: Optional[TensorTable] = None,
        macro_table: Optional[TensorTable] = None,
        full_N=None,
        entry_num=None,
        device='cpu'
    ):
        self.device = torch.device(device)
        self.full_N = full_N
        self.branch_num = 2
        self.phi = Config.PHI
        self.delta = Config.DELTA
        self.G = Config.G
        self.quantiles = torch.linspace(0, 1, steps=100, device=self.device)
        self.firm_table = firm_table.to(self.device) if firm_table is not None else None
        self.macro_table = macro_table.to(self.device) if macro_table is not None else None
        self.tensor_native = self.firm_table is not None

        if self.tensor_native:
            self._build_ragged_from_tensor_tables()
            return

        if df is None:
            if pkl_path is None:
                raise ValueError('pkl_path or df required')
            df = pd.read_pickle(pkl_path)
        if 'K' not in df.columns:
            df = df.copy(); df['K'] = 1.0

        df_filled = fill_df_to_fullN(df, full_N=full_N, device=self.device, entry_num=entry_num)
        df_filled.sort_values(by=['path', 'ID', 'branch'], inplace=True)
        self.df_filled = df_filled
        

        parent = df_filled[df_filled.branch == 0].reset_index(drop=True)
        child1 = df_filled[df_filled.branch == 1].reset_index(drop=True)
        child2 = df_filled[df_filled.branch == 2].reset_index(drop=True)

        df1 = pd.merge(parent, child1, on=['ID'], suffixes=('', '_child1'), how='outer')
        df2 = pd.merge(df1, child2, on=['ID'], suffixes=('', '_child2'), how='outer')

        df2 = df2.sort_values(
            ['path', 'b'],
            ascending=[True, True],
            na_position='last',
            kind='mergesort',
        ).reset_index(drop=True)

        self.df2 = df2
        self._build_tensors()

    @staticmethod
    def _col_index(columns: Sequence[str]) -> Dict[str, int]:
        return {name: idx for idx, name in enumerate(columns)}

    def _safe_quantile_features(
        self,
        b_vals: torch.Tensor,
        z_vals: torch.Tensor,
    ) -> torch.Tensor:
        if b_vals.numel() == 0 or z_vals.numel() == 0:
            return torch.zeros((200,), dtype=torch.float32, device=self.device)
        return compute_quantile_features(
            b_vals.to(torch.float32),
            z_vals.to(torch.float32),
            quantiles=self.quantiles,
        )

    def _select_fc2_window(
        self,
        rows: torch.Tensor,
        cols: Dict[str, int],
        path_val: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = torch.round(rows[:, cols['path']]).to(torch.long)
        t = torch.round(rows[:, cols['t']]).to(torch.long)
        branch = torch.round(rows[:, cols['branch']]).to(torch.long)
        mask_path = path == path_val
        parent_ts = t[mask_path & (branch == -1)]
        if parent_ts.numel() == 0:
            return (
                rows.new_zeros((0, rows.shape[1])),
                rows.new_zeros((0, rows.shape[1])),
                rows.new_zeros((0, rows.shape[1])),
            )
        parent_t = int(parent_ts.max().item())
        child_t = parent_t + 1
        parent_rows = rows[mask_path & (t == parent_t) & (branch == -1)]
        child0_rows = rows[mask_path & (t == child_t) & (branch == 0)]
        child1_rows = rows[mask_path & (t == child_t) & (branch == 1)]
        return parent_rows, child0_rows, child1_rows

    def _build_ragged_from_tensor_tables(self):
        rows = self.firm_table.data.to(device=self.device, dtype=torch.float32)
        cols = self._col_index(self.firm_table.columns)

        path_all = torch.round(rows[:, cols['path']]).to(torch.long)
        path_values = torch.unique(path_all, sorted=True)
        selected = []
        max_ids = 0

        for path_val_t in path_values:
            path_val = int(path_val_t.item())
            parent_rows, child0_rows, child1_rows = self._select_fc2_window(rows, cols, path_val)
            id_parts = []
            for part in (parent_rows, child0_rows, child1_rows):
                if part.numel() > 0:
                    id_parts.append(torch.round(part[:, cols['ID']]).to(torch.long))
            if not id_parts:
                continue
            ids = torch.unique(torch.cat(id_parts), sorted=True)
            max_ids = max(max_ids, int(ids.numel()))
            selected.append((path_val, parent_rows, child0_rows, child1_rows, ids))

        self.path_num = len(selected)
        self.N = max_ids
        self.parent_states_list: List[torch.Tensor] = []
        self.parent_K_list: List[torch.Tensor] = []
        self.parent_alive_list: List[torch.Tensor] = []
        self.child_states_list: List[torch.Tensor] = []
        self.child_K_list: List[torch.Tensor] = []
        self.child_present_list: List[torch.Tensor] = []
        self.entry_mask_list: List[torch.Tensor] = []
        if self.path_num == 0 or self.N == 0:
            return

        state_cols = ['b', 'z', 'ETA', 'i', 'x']
        state_idx = [cols[name] for name in state_cols]

        for i, (_, parent_rows, child0_rows, child1_rows, ids) in enumerate(selected):
            n_i = int(ids.numel())
            slot_map = {int(fid.item()): j for j, fid in enumerate(ids)}
            parent_state = torch.zeros((n_i, 5), dtype=torch.float32, device=self.device)
            parent_K = torch.zeros((n_i, 1), dtype=torch.float32, device=self.device)
            parent_alive = torch.zeros((n_i,), dtype=torch.bool, device=self.device)
            child_states = torch.zeros((n_i, self.branch_num, 5), dtype=torch.float32, device=self.device)
            child_K = torch.zeros((n_i, self.branch_num, 1), dtype=torch.float32, device=self.device)
            child_present = torch.zeros((n_i, self.branch_num), dtype=torch.bool, device=self.device)
            entry_mask = torch.zeros((n_i, self.branch_num), dtype=torch.bool, device=self.device)

            if parent_rows.numel() > 0:
                parent_ids = torch.round(parent_rows[:, cols['ID']]).to(torch.long)
                parent_slots = torch.tensor(
                    [slot_map[int(fid.item())] for fid in parent_ids],
                    device=self.device,
                    dtype=torch.long,
                )
                parent_state[parent_slots, :] = parent_rows[:, state_idx]
                parent_K[parent_slots, 0] = parent_rows[:, cols['K']]
                parent_alive[parent_slots] = True

            for child_j, child_rows in enumerate((child0_rows, child1_rows)):
                if child_rows.numel() == 0:
                    continue
                child_ids = torch.round(child_rows[:, cols['ID']]).to(torch.long)
                child_slots = torch.tensor(
                    [slot_map[int(fid.item())] for fid in child_ids],
                    device=self.device,
                    dtype=torch.long,
                )
                child_states[child_slots, child_j, :] = child_rows[:, state_idx]
                child_K[child_slots, child_j, 0] = child_rows[:, cols['K']]
                child_present[child_slots, child_j] = True
                if 'entry' in cols:
                    entry_mask[child_slots, child_j] = child_rows[:, cols['entry']] > 0.5

            child_states[:, :, 0] = 0
            self.parent_states_list.append(parent_state)
            self.parent_K_list.append(parent_K)
            self.parent_alive_list.append(parent_alive)
            self.child_states_list.append(child_states)
            self.child_K_list.append(child_K)
            self.child_present_list.append(child_present)
            self.entry_mask_list.append(entry_mask)

    def _gather_parent_policy_rows(self, parent_macro_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        flat_inputs = []
        flat_k = []
        path_idx = []
        slot_idx = []
        for i in range(self.path_num):
            alive_idx = torch.nonzero(self.parent_alive_list[i], as_tuple=False).squeeze(-1)
            if alive_idx.numel() == 0:
                continue
            macro_i = parent_macro_pred[i].unsqueeze(0).expand(alive_idx.numel(), -1)
            flat_inputs.append(torch.cat([self.parent_states_list[i][alive_idx], macro_i], dim=-1))
            flat_k.append(self.parent_K_list[i][alive_idx])
            path_idx.append(torch.full((alive_idx.numel(),), i, dtype=torch.long, device=self.device))
            slot_idx.append(alive_idx)
        if not flat_inputs:
            return {
                'firm_state': torch.zeros((0, 7), dtype=torch.float32, device=self.device),
                'K': torch.zeros((0, 1), dtype=torch.float32, device=self.device),
                'path_idx': torch.zeros((0,), dtype=torch.long, device=self.device),
                'slot_idx': torch.zeros((0,), dtype=torch.long, device=self.device),
            }
        return {
            'firm_state': torch.cat(flat_inputs, dim=0),
            'K': torch.cat(flat_k, dim=0),
            'path_idx': torch.cat(path_idx, dim=0),
            'slot_idx': torch.cat(slot_idx, dim=0),
        }

    def _scatter_parent_slot_values(self, flat_values: torch.Tensor, path_idx: torch.Tensor, slot_idx: torch.Tensor) -> List[torch.Tensor]:
        outputs = [torch.zeros((self.parent_states_list[i].shape[0], flat_values.shape[-1]), dtype=flat_values.dtype, device=flat_values.device) for i in range(self.path_num)]
        if flat_values.numel() == 0:
            return outputs
        for i in range(self.path_num):
            mask = path_idx == i
            if mask.any():
                outputs[i][slot_idx[mask]] = flat_values[mask]
        return outputs

    def _gather_child_policy_rows(
        self,
        children_states_list: List[torch.Tensor],
        child_macro_pred: torch.Tensor,
        alive_mask_list: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        flat_inputs = []
        flat_k = []
        flat_base_alive = []
        path_idx = []
        branch_idx = []
        slot_idx = []
        for i in range(self.path_num):
            for j in range(self.branch_num):
                alive_ij = alive_mask_list[i][:, j] > 0
                idx = torch.nonzero(alive_ij, as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    continue
                macro_ij = child_macro_pred[i, j].unsqueeze(0).expand(idx.numel(), -1)
                flat_inputs.append(torch.cat([children_states_list[i][idx, j, :], macro_ij], dim=-1))
                flat_k.append(self.child_K_list[i][idx, j, :])
                flat_base_alive.append(alive_mask_list[i][idx, j].unsqueeze(-1))
                path_idx.append(torch.full((idx.numel(),), i, dtype=torch.long, device=self.device))
                branch_idx.append(torch.full((idx.numel(),), j, dtype=torch.long, device=self.device))
                slot_idx.append(idx)
        if not flat_inputs:
            zero_idx = torch.zeros((0,), dtype=torch.long, device=self.device)
            return {
                'firm_state': torch.zeros((0, 7), dtype=torch.float32, device=self.device),
                'K': torch.zeros((0, 1), dtype=torch.float32, device=self.device),
                'base_alive': torch.zeros((0, 1), dtype=torch.float32, device=self.device),
                'path_idx': zero_idx,
                'branch_idx': zero_idx,
                'slot_idx': zero_idx,
            }
        return {
            'firm_state': torch.cat(flat_inputs, dim=0),
            'K': torch.cat(flat_k, dim=0),
            'base_alive': torch.cat(flat_base_alive, dim=0),
            'path_idx': torch.cat(path_idx, dim=0),
            'branch_idx': torch.cat(branch_idx, dim=0),
            'slot_idx': torch.cat(slot_idx, dim=0),
        }

    def _scatter_child_slot_values(
        self,
        flat_values: torch.Tensor,
        path_idx: torch.Tensor,
        branch_idx: torch.Tensor,
        slot_idx: torch.Tensor,
    ) -> List[torch.Tensor]:
        outputs = [torch.zeros((self.child_states_list[i].shape[0], self.branch_num, flat_values.shape[-1]), dtype=flat_values.dtype, device=flat_values.device) for i in range(self.path_num)]
        if flat_values.numel() == 0:
            return outputs
        for i in range(self.path_num):
            mask_i = path_idx == i
            if not mask_i.any():
                continue
            slots_i = slot_idx[mask_i]
            branches_i = branch_idx[mask_i]
            values_i = flat_values[mask_i]
            outputs[i][slots_i, branches_i, :] = values_i
        return outputs

    def _build_tensors(self):
        df2 = self.df2
        self.path_num = df2['path'].nunique()
        self.N = df2.groupby('path')['ID'].nunique().max()

        P_s = torch.zeros((self.path_num, self.N, 5), dtype=torch.float32, device=self.device)
        K_parent = torch.ones((self.path_num, self.N, 1), dtype=torch.float32, device=self.device)
        alive_mask = torch.zeros((self.path_num, self.N), dtype=torch.bool, device=self.device)
        entry_mask = torch.zeros((self.path_num, self.N, self.branch_num), dtype=torch.bool, device=self.device)

        for i, (path, group) in enumerate(df2.groupby('path')):
            vals = torch.tensor(group[['b', 'z', 'ETA', 'i', 'x']].to_numpy(dtype=np.float32), device=self.device)
            P_s[i, : len(group)] = vals
            K_parent[i, : len(group), 0] = torch.tensor(group['K'].to_numpy(dtype=np.float32), device=self.device)
            alive_mask[i, : len(group)] = torch.tensor(group['b'].notna().to_numpy(), device=self.device)
            entry_mask[i, : len(group), 0] = torch.tensor(group['Entry_child1'].to_numpy(dtype=np.float32), device=self.device)
            entry_mask[i, : len(group), 1] = torch.tensor(group['Entry_child2'].to_numpy(dtype=np.float32), device=self.device)

        Children_s = torch.zeros((self.path_num, self.N, self.branch_num, 5), dtype=torch.float32, device=self.device)
        K_children = torch.zeros((self.path_num, self.N, self.branch_num, 1), dtype=torch.float32, device=self.device)
        for i, (path, group) in enumerate(df2.groupby('path')):
            vals_child1 = torch.tensor(group[['b_child1', 'z_child1', 'ETA_child1', 'i_child1', 'x_child1']].to_numpy(dtype=np.float32), device=self.device)
            vals_child2 = torch.tensor(group[['b_child2', 'z_child2', 'ETA_child2', 'i_child2', 'x_child2']].to_numpy(dtype=np.float32), device=self.device)
            Children_s[i, : len(group), 0, :] = vals_child1
            Children_s[i, : len(group), 1, :] = vals_child2
            K_children[i, : len(group), 0, 0] = torch.tensor(group['K_child1'].to_numpy(dtype=np.float32), device=self.device)
            K_children[i, : len(group), 1, 0] = torch.tensor(group['K_child2'].to_numpy(dtype=np.float32), device=self.device)

        Children_s[:, :, :, 0] = 0

        if P_s.shape[1] == self.full_N:
            self.P_s_full = P_s
            self.K_parent_full = K_parent
            self.Children_s_full = Children_s
            self.K_children_full = K_children
        else:
            self.P_s_full = torch.zeros((self.path_num, self.full_N, 5), dtype=torch.float32, device=self.device)
            self.P_s_full[:, :P_s.shape[1], :] = P_s
            self.K_parent_full = torch.zeros((self.path_num, self.full_N, 1), dtype=torch.float32, device=self.device)
            self.K_parent_full[:, :K_parent.shape[1], :] = K_parent

            self.Children_s_full = torch.zeros((self.path_num, self.full_N, 2, 5), dtype=torch.float32, device=self.device)
            self.Children_s_full[:, :Children_s.shape[1], :, :] = Children_s
            self.K_children_full = torch.zeros((self.path_num, self.full_N, 2, 1), dtype=torch.float32, device=self.device)
            self.K_children_full[:, :K_children.shape[1], :, :] = K_children

        self.alive_mask = alive_mask
        self.entry_mask = entry_mask

    def build_fc2_input_parent(self):
        if self.tensor_native:
            FC2_input_parent = torch.zeros((self.path_num, 201), dtype=torch.float32, device=self.device)
            for i in range(self.path_num):
                alive_i = self.parent_alive_list[i]
                b_vals = self.parent_states_list[i][alive_i, 0]
                z_vals = self.parent_states_list[i][alive_i, 1]
                quantile_features = self._safe_quantile_features(b_vals, z_vals)
                if alive_i.any():
                    x_feature = self.parent_states_list[i][alive_i, 4].mean().unsqueeze(0)
                else:
                    x_feature = torch.zeros((1,), dtype=torch.float32, device=self.device)
                FC2_input_parent[i, :200] = quantile_features
                FC2_input_parent[i, 200] = x_feature
            return FC2_input_parent
        FC2_input_parent = torch.zeros((self.path_num, 201), dtype=torch.float32, device=self.device)
        for i in range(self.path_num):
            b_vals = self.P_s_full[i, self.alive_mask[i], 0]
            z_vals = self.P_s_full[i, self.alive_mask[i], 1]
            quantile_features = self._safe_quantile_features(b_vals, z_vals)
            if self.alive_mask[i].any():
                x_feature = self.P_s_full[i, self.alive_mask[i], 4].mean().unsqueeze(0)
            else:
                x_feature = torch.zeros((1,), dtype=torch.float32, device=self.device)
            FC2_input_parent[i, :200] = quantile_features
            FC2_input_parent[i, 200] = x_feature
        return FC2_input_parent

    def build_fc2_input_children(self, children_s_full, alive_mask):
        if self.tensor_native and isinstance(children_s_full, list):
            FC2_input_children = torch.zeros((self.path_num, 2, 201), dtype=torch.float32, device=self.device)
            for i in range(self.path_num):
                for j in range(2):
                    alive_ij = alive_mask[i][:, j] > 0
                    b_vals = children_s_full[i][alive_ij, j, 0]
                    z_vals = children_s_full[i][alive_ij, j, 1]
                    quantile_features = self._safe_quantile_features(b_vals, z_vals)
                    if alive_ij.any():
                        x_feature = children_s_full[i][alive_ij, j, 4].mean().unsqueeze(0)
                    else:
                        x_feature = torch.zeros((1,), dtype=torch.float32, device=self.device)
                    FC2_input_children[i, j, :200] = quantile_features
                    FC2_input_children[i, j, 200] = x_feature
            return FC2_input_children
        FC2_input_children = torch.zeros((self.path_num, 2, 201), dtype=torch.float32, device=self.device)
        for i in range(self.path_num):
            for j in range(2):
                b_vals = children_s_full[i, alive_mask[i, :, j].bool(), j, 0]
                z_vals = children_s_full[i, alive_mask[i, :, j].bool(), j, 1]
                quantile_features = self._safe_quantile_features(b_vals, z_vals)
                if alive_mask[i, :, j].any():
                    x_feature = children_s_full[i, alive_mask[i, :, j].bool(), j, 4].mean().unsqueeze(0)
                else:
                    x_feature = torch.zeros((1,), dtype=torch.float32, device=self.device)
                FC2_input_children[i, j, :200] = quantile_features
                FC2_input_children[i, j, 200] = x_feature
        return FC2_input_children

    def _pv_forward(self, pv_model: torch.nn.Module, firm_state: torch.Tensor):
        """
        PolicyValueModel expects (batch, 7). Flatten/reshape to support [path, N, 7] or [path, N, 2, 7].
        Skip rows containing NaN/Inf so they are never fed into the model.
        """
        orig_shape = firm_state.shape[:-1]
        flat = firm_state.reshape(-1, firm_state.shape[-1])
        valid = torch.isfinite(flat).all(dim=-1)

        def _scatter(v: torch.Tensor):
            out_full = torch.zeros(flat.shape[0], v.shape[-1], device=v.device, dtype=v.dtype)
            out_full[valid] = v
            return out_full.reshape(*orig_shape, -1)

        if valid.any():
            out = pv_model(flat[valid])
        else:
            out = pv_model(torch.zeros(1, flat.shape[-1], device=flat.device, dtype=flat.dtype))
            def _zero_like(v: torch.Tensor):
                z = torch.zeros(flat.shape[0], v.shape[-1], device=v.device, dtype=v.dtype)
                return z.reshape(*orig_shape, -1)
            if isinstance(out, dict):
                return {k: _zero_like(v) for k, v in out.items()}
            if hasattr(out, "_fields"):
                return out.__class__(**{k: _zero_like(getattr(out, k)) for k in out._fields})
            return _zero_like(out)

        if isinstance(out, dict):
            return {k: _scatter(v) for k, v in out.items()}
        if hasattr(out, "_fields"):
            return out.__class__(**{k: _scatter(getattr(out, k)) for k in out._fields})
        return _scatter(out)


    def _get_out(self, out, name: str, idx: int) -> torch.Tensor:
        if isinstance(out, dict):
            return out[name]
        if hasattr(out, name):
            return getattr(out, name)
        return out[..., idx:idx + 1]

    def _forward_parent_fc2(self, fc2_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        FC2_input_parent = self.build_fc2_input_parent()
        fc2_out_parent = fc2_model(FC2_input_parent)
        hatc_parent_pred = fc2_out_parent['hatc']
        lnk_parent_pred = fc2_out_parent['lnk']
        fc2_output_parent = torch.cat([lnk_parent_pred, hatc_parent_pred], dim=-1)
        return {
            'fc2_input': FC2_input_parent,
            'hatc_pred': hatc_parent_pred,
            'lnk_pred': lnk_parent_pred,
            'macro_pred': fc2_output_parent,
        }

    def _forward_parent_policy(self, pv_model: torch.nn.Module, parent_macro_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.tensor_native:
            gathered = self._gather_parent_policy_rows(parent_macro_pred)
            PV_input = gathered['firm_state']
            if PV_input.shape[0] == 0:
                zero_parent = [torch.zeros((self.parent_states_list[i].shape[0], 1), dtype=torch.float32, device=self.device) for i in range(self.path_num)]
                updated_alive = [zp.clone() for zp in zero_parent]
                alive_mask = [torch.zeros((self.child_states_list[i].shape[0], self.branch_num), dtype=torch.float32, device=self.device) for i in range(self.path_num)]
                return {
                    'pv_input': PV_input,
                    'pv_output': None,
                    'bar_z': zero_parent,
                    'bar_i': zero_parent,
                    'bp': zero_parent,
                    'updated_alive': updated_alive,
                    'alive_mask': alive_mask,
                    'gathered': gathered,
                }

            PV_output = pv_model(PV_input)
            bar_z_flat = self._get_out(PV_output, 'bar_z', 6)
            bar_i_flat = self._get_out(PV_output, 'bar_i', 5)
            bp_flat = self._get_out(PV_output, 'bp', 9)

            bar_z = self._scatter_parent_slot_values(bar_z_flat, gathered['path_idx'], gathered['slot_idx'])
            bar_i = self._scatter_parent_slot_values(bar_i_flat, gathered['path_idx'], gathered['slot_idx'])
            bp = self._scatter_parent_slot_values(bp_flat, gathered['path_idx'], gathered['slot_idx'])

            updated_alive = []
            alive_mask = []
            for i in range(self.path_num):
                alive_prob_i = bar_z[i].clamp(0, 1)
                base_alive_i = self.parent_alive_list[i].float().unsqueeze(-1)
                updated_alive_i = torch.nan_to_num(base_alive_i * alive_prob_i, nan=0.0)
                presence_i = self.child_present_list[i].float()
                alive_mask_i = (updated_alive_i.expand(-1, self.branch_num) + self.entry_mask_list[i].float()) * presence_i
                updated_alive.append(updated_alive_i)
                alive_mask.append(alive_mask_i)
            return {
                'pv_input': PV_input,
                'pv_output': PV_output,
                'bar_z': bar_z,
                'bar_i': bar_i,
                'bp': bp,
                'updated_alive': updated_alive,
                'alive_mask': alive_mask,
                'gathered': gathered,
            }
        fc2_output_parent = parent_macro_pred.unsqueeze(1).expand(-1, self.full_N, -1) * self.alive_mask.unsqueeze(-1)
        PV_input = torch.cat((self.P_s_full, fc2_output_parent), dim=-1)

        PV_output = self._pv_forward(pv_model, PV_input)
        bar_z = self._get_out(PV_output, 'bar_z', 6)
        bar_i = self._get_out(PV_output, 'bar_i', 5)
        bp = self._get_out(PV_output, 'bp', 9)

        alive_prob = bar_z.clamp(0, 1).squeeze(-1)
        base_alive = self.alive_mask.float()
        updated_alive = base_alive * alive_prob
        updated_alive = torch.nan_to_num(updated_alive, nan=0.0)
        updated_alive = updated_alive.unsqueeze(-1).expand(-1, -1, 2)
        entry_flag = self.entry_mask.float()
        alive_mask = updated_alive + entry_flag
        return {
            'pv_input': PV_input,
            'pv_output': PV_output,
            'bar_z': bar_z,
            'bar_i': bar_i,
            'bp': bp,
            'updated_alive': updated_alive,
            'alive_mask': alive_mask,
        }

    def _compute_parent_aggregates(
        self,
        pv_input: torch.Tensor,
        bar_i: torch.Tensor,
        bar_z: torch.Tensor,
        updated_alive: torch.Tensor,
        lnk_parent_pred: torch.Tensor,
        hatc_parent_pred: torch.Tensor,
        gathered: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.tensor_native:
            assert gathered is not None
            if gathered['firm_state'].shape[0] == 0:
                zero = torch.zeros((self.path_num, 1), dtype=torch.float32, device=self.device)
                return {
                    'Y': torch.zeros((0, 1), dtype=torch.float32, device=self.device),
                    'Phi': torch.zeros((0, 1), dtype=torch.float32, device=self.device),
                    'I': torch.zeros((0, 1), dtype=torch.float32, device=self.device),
                    'C': torch.zeros((0, 1), dtype=torch.float32, device=self.device),
                    'lnk_actual': zero,
                    'hatc_actual': zero,
                    'loss': torch.mean((lnk_parent_pred - zero) ** 2) + torch.mean((hatc_parent_pred - zero) ** 2),
                }
            # Rebuild the flat policy outputs in the same row order as pv_input.
            bar_i_flat = []
            bar_z_flat = []
            updated_alive_flat = []
            for i in range(self.path_num):
                alive_idx = torch.nonzero(self.parent_alive_list[i], as_tuple=False).squeeze(-1)
                if alive_idx.numel() == 0:
                    continue
                bar_i_flat.append(bar_i[i][alive_idx])
                bar_z_flat.append(bar_z[i][alive_idx])
                updated_alive_flat.append(updated_alive[i][alive_idx])
            bar_i_flat = torch.cat(bar_i_flat, dim=0)
            bar_z_flat = torch.cat(bar_z_flat, dim=0)
            updated_alive_flat = torch.cat(updated_alive_flat, dim=0)
            Y = torch.exp(pv_input[:, 4:5] + pv_input[:, 1:2]) * gathered['K']
            Phi = (1 - self.phi) * (1 + torch.exp(pv_input[:, 1:2] + pv_input[:, 3:4])) * gathered['K'] * bar_z_flat
            I = bar_i_flat * gathered['K'] * pv_input[:, 3:4] - bar_z_flat * gathered['K'] + self.delta * gathered['K']
            C = torch.abs(Y - I - Phi)
            K_sum = torch.zeros((self.path_num, 1), dtype=torch.float32, device=self.device)
            C_sum = torch.zeros((self.path_num, 1), dtype=torch.float32, device=self.device)
            K_sum.index_add_(0, gathered['path_idx'], gathered['K'] * updated_alive_flat)
            C_sum.index_add_(0, gathered['path_idx'], C * updated_alive_flat)
            lnk_parent = torch.log(K_sum + 1e-8)
            hatc_parent = torch.log(C_sum / (K_sum + 1e-8))
            loss_parent = torch.mean((lnk_parent - lnk_parent_pred) ** 2) + torch.mean((hatc_parent - hatc_parent_pred) ** 2)
            return {
                'Y': Y,
                'Phi': Phi,
                'I': I,
                'C': C,
                'lnk_actual': lnk_parent,
                'hatc_actual': hatc_parent,
                'loss': loss_parent,
            }
        Y = torch.exp(pv_input[..., 4:5] + pv_input[..., 1:2]) * self.K_parent_full
        Phi = (1 - self.phi) * (1 + torch.exp(pv_input[..., 1:2] + pv_input[..., 3:4])) * self.K_parent_full * bar_z
        I = bar_i * self.K_parent_full * pv_input[..., 3:4] - bar_z * self.K_parent_full + self.delta * self.K_parent_full
        C = torch.abs(Y - I - Phi)

        masked_K = torch.where(
            updated_alive[..., 0:1] > 0,
            self.K_parent_full,
            torch.zeros_like(self.K_parent_full)
        )
        lnk_parent = torch.log(masked_K.sum(dim=1) + 1e-8)

        masked_C = torch.where(
            updated_alive[..., 0:1] > 0,
            C,
            torch.zeros_like(C)
        )
        hatc_parent = torch.log(torch.sum(masked_C, dim=1) / (torch.sum(masked_K, dim=1) + 1e-8))

        loss_parent = torch.mean((lnk_parent - lnk_parent_pred)**2) + torch.mean((hatc_parent - hatc_parent_pred)**2)
        return {
            'Y': Y,
            'Phi': Phi,
            'I': I,
            'C': C,
            'lnk_actual': lnk_parent,
            'hatc_actual': hatc_parent,
            'loss': loss_parent,
        }

    def _update_children_state(
        self,
        updated_alive: torch.Tensor,
        bp: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.tensor_native:
            children_states_list = []
            for i in range(self.path_num):
                child_state = self.child_states_list[i].clone()
                masked_b = torch.where(
                    updated_alive[i][..., 0] > 0,
                    self.parent_states_list[i][:, 0],
                    torch.zeros_like(self.parent_states_list[i][:, 0])
                )
                masked_bp = torch.where(
                    updated_alive[i][..., 0] > 0,
                    bp[i][..., 0],
                    torch.zeros_like(bp[i][..., 0])
                )
                for j in range(self.branch_num):
                    eta_j = child_state[:, j, 2]
                    child_state[:, j, 0] = masked_bp * eta_j + masked_b * (1 - eta_j)
                children_states_list.append(child_state)
            return {'children_s_full': children_states_list}
        children_s_full = self.Children_s_full.clone()
        masked_b = torch.where(
            updated_alive[..., 0] > 0,
            self.P_s_full[:, :, 0],
            torch.zeros_like(self.P_s_full[:, :, 0])
        )
        masked_bp = torch.where(
            updated_alive[..., 0] > 0,
            bp[..., 0],
            torch.zeros_like(bp[..., 0])
        )
        child0 = children_s_full[:, :, 0, :]
        child1 = children_s_full[:, :, 1, :]
        new_b0 = masked_bp * child0[:, :, 2] + masked_b * (1 - child0[:, :, 2])
        new_b1 = masked_bp * child1[:, :, 2] + masked_b * (1 - child1[:, :, 2])
        child0 = torch.cat([new_b0.unsqueeze(-1), child0[:, :, 1:]], dim=-1)
        child1 = torch.cat([new_b1.unsqueeze(-1), child1[:, :, 1:]], dim=-1)
        children_s_full = torch.stack([child0, child1], dim=2)
        return {'children_s_full': children_s_full}

    def _forward_children_fc2(
        self,
        fc2_model: torch.nn.Module,
        children_s_full: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        FC2_input_children = self.build_fc2_input_children(children_s_full, alive_mask)
        flat_children = FC2_input_children.reshape(-1, FC2_input_children.shape[-1])
        fc2_out_children = fc2_model(flat_children)
        hatc_children_pred = fc2_out_children['hatc'].view(self.path_num, 2, 1)
        lnk_children_pred = fc2_out_children['lnk'].view(self.path_num, 2, 1)
        FC2_output_children1 = torch.cat([lnk_children_pred, hatc_children_pred], dim=-1)
        return {
            'fc2_input': FC2_input_children,
            'hatc_pred': hatc_children_pred,
            'lnk_pred': lnk_children_pred,
            'macro_pred': FC2_output_children1,
        }

    def _forward_children_policy(
        self,
        pv_model: torch.nn.Module,
        children_s_full: torch.Tensor,
        child_macro_pred: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.tensor_native:
            gathered = self._gather_child_policy_rows(children_s_full, child_macro_pred, alive_mask)
            CV_input = gathered['firm_state']
            if CV_input.shape[0] == 0:
                zero_child = [torch.zeros((self.child_states_list[i].shape[0], self.branch_num, 1), dtype=torch.float32, device=self.device) for i in range(self.path_num)]
                return {
                    'cv_input': CV_input,
                    'cv_output': None,
                    'bar_z': zero_child,
                    'bar_i': zero_child,
                    'alive_mask_children': zero_child,
                    'gathered': gathered,
                }
            CV_output = pv_model(CV_input)
            bar_z_flat = self._get_out(CV_output, 'bar_z', 6)
            bar_i_flat = self._get_out(CV_output, 'bar_i', 5)
            alive_prob_children = bar_z_flat.clamp(0, 1)
            updated_alive_flat = gathered['base_alive'] * alive_prob_children
            bar_z_children = self._scatter_child_slot_values(bar_z_flat, gathered['path_idx'], gathered['branch_idx'], gathered['slot_idx'])
            bar_i_children = self._scatter_child_slot_values(bar_i_flat, gathered['path_idx'], gathered['branch_idx'], gathered['slot_idx'])
            alive_mask_children = self._scatter_child_slot_values(updated_alive_flat, gathered['path_idx'], gathered['branch_idx'], gathered['slot_idx'])
            return {
                'cv_input': CV_input,
                'cv_output': CV_output,
                'bar_z': bar_z_children,
                'bar_i': bar_i_children,
                'alive_mask_children': alive_mask_children,
                'gathered': gathered,
            }
        fc2_output_children = child_macro_pred.unsqueeze(1).expand(-1, self.full_N, -1, -1) * alive_mask.unsqueeze(-1)
        CV_input = torch.cat((children_s_full, fc2_output_children), dim=-1)

        CV_output = self._pv_forward(pv_model, CV_input)
        bar_z_children = self._get_out(CV_output, 'bar_z', 6)
        bar_i_children = self._get_out(CV_output, 'bar_i', 5)

        alive_mask_children = alive_mask.clone()
        alive_prob_children = bar_z_children.clamp(0, 1).squeeze(-1)
        base_alive_children = alive_mask.float()
        updated_alive_children = base_alive_children * alive_prob_children
        alive_mask_children = updated_alive_children.unsqueeze(-1)
        return {
            'cv_input': CV_input,
            'cv_output': CV_output,
            'bar_z': bar_z_children,
            'bar_i': bar_i_children,
            'alive_mask_children': alive_mask_children,
        }

    def _compute_children_aggregates(
        self,
        cv_input: torch.Tensor,
        bar_i_children: torch.Tensor,
        bar_z_children: torch.Tensor,
        alive_mask_children: torch.Tensor,
        lnk_children_pred: torch.Tensor,
        hatc_children_pred: torch.Tensor,
        gathered: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.tensor_native:
            assert gathered is not None
            if cv_input.shape[0] == 0:
                zero = torch.zeros((self.path_num, self.branch_num, 1), dtype=torch.float32, device=self.device)
                return {
                    'Y': torch.zeros((0, 1), dtype=torch.float32, device=self.device),
                    'Phi': torch.zeros((0, 1), dtype=torch.float32, device=self.device),
                    'I': torch.zeros((0, 1), dtype=torch.float32, device=self.device),
                    'C': torch.zeros((0, 1), dtype=torch.float32, device=self.device),
                    'K_children_full': torch.zeros((0, 1), dtype=torch.float32, device=self.device),
                    'lnk_actual': zero,
                    'hatc_actual': zero,
                    'loss': torch.mean((lnk_children_pred - zero) ** 2) + torch.mean((hatc_children_pred - zero) ** 2),
                }
            flat_k = gathered['K']
            flat_bar_i = []
            flat_bar_z = []
            flat_alive = []
            for i in range(self.path_num):
                for j in range(self.branch_num):
                    alive_ij = (alive_mask_children[i][:, j, 0] if alive_mask_children[i].dim() == 3 else alive_mask_children[i][:, j]) > 0
                    idx = torch.nonzero(alive_ij, as_tuple=False).squeeze(-1)
                    if idx.numel() == 0:
                        continue
                    flat_bar_i.append(bar_i_children[i][idx, j, :])
                    flat_bar_z.append(bar_z_children[i][idx, j, :])
                    flat_alive.append(alive_mask_children[i][idx, j, :])
            flat_bar_i = torch.cat(flat_bar_i, dim=0)
            flat_bar_z = torch.cat(flat_bar_z, dim=0)
            flat_alive = torch.cat(flat_alive, dim=0)
            K_children_flat = flat_k * (1 + self.G * flat_bar_i)
            Y = torch.exp(cv_input[:, 4:5] + cv_input[:, 1:2]) * K_children_flat
            Phi = (1 - self.phi) * (1 + torch.exp(cv_input[:, 1:2] + cv_input[:, 3:4])) * K_children_flat * flat_bar_z
            I = flat_bar_i * K_children_flat * cv_input[:, 3:4] - flat_bar_z * K_children_flat + self.delta * K_children_flat
            C = torch.abs(Y - I - Phi)
            K_sum = torch.zeros((self.path_num * self.branch_num, 1), dtype=torch.float32, device=self.device)
            C_sum = torch.zeros((self.path_num * self.branch_num, 1), dtype=torch.float32, device=self.device)
            linear_idx = gathered['path_idx'] * self.branch_num + gathered['branch_idx']
            K_sum.index_add_(0, linear_idx, K_children_flat * flat_alive)
            C_sum.index_add_(0, linear_idx, C * flat_alive)
            K_sum = K_sum.view(self.path_num, self.branch_num, 1)
            C_sum = C_sum.view(self.path_num, self.branch_num, 1)
            lnk_children = torch.log(K_sum + 1e-8)
            hatc_children = torch.log(C_sum / (K_sum + 1e-8))
            loss_children = torch.mean((lnk_children - lnk_children_pred) ** 2) + torch.mean((hatc_children - hatc_children_pred) ** 2)
            return {
                'Y': Y,
                'Phi': Phi,
                'I': I,
                'C': C,
                'K_children_full': K_children_flat,
                'lnk_actual': lnk_children,
                'hatc_actual': hatc_children,
                'loss': loss_children,
            }
        bari = bar_i_children
        K_children_full = self.K_children_full * (1 + self.G * bari)

        Y = torch.exp(cv_input[..., 4:5] + cv_input[..., 1:2]) * K_children_full
        Phi = (1 - self.phi) * (1 + torch.exp(cv_input[..., 1:2] + cv_input[..., 3:4])) * K_children_full * bar_z_children
        I = bar_i_children * K_children_full * cv_input[..., 3:4] - bar_z_children * K_children_full + self.delta * K_children_full
        C = torch.abs(Y - I - Phi)
        lnk_children = torch.log(torch.sum(K_children_full * alive_mask_children[..., 0:1], dim=1) + 1e-8)
        hatc_children = torch.log(torch.sum(C * alive_mask_children[..., 0:1], dim=1) / (torch.sum(K_children_full * alive_mask_children[..., 0:1], dim=1) + 1e-8))

        loss_children = torch.mean((lnk_children - lnk_children_pred)**2) + torch.mean((hatc_children - hatc_children_pred)**2)
        return {
            'Y': Y,
            'Phi': Phi,
            'I': I,
            'C': C,
            'K_children_full': K_children_full,
            'lnk_actual': lnk_children,
            'hatc_actual': hatc_children,
            'loss': loss_children,
        }

    def forward(self, fc2_model: torch.nn.Module, pv_model: torch.nn.Module):
        parent_fc2 = self._forward_parent_fc2(fc2_model)
        parent_policy = self._forward_parent_policy(pv_model, parent_fc2['macro_pred'])
        parent_agg = self._compute_parent_aggregates(
            parent_policy['pv_input'],
            parent_policy['bar_i'],
            parent_policy['bar_z'],
            parent_policy['updated_alive'],
            parent_fc2['lnk_pred'],
            parent_fc2['hatc_pred'],
            parent_policy.get('gathered'),
        )

        child_state = self._update_children_state(parent_policy['updated_alive'], parent_policy['bp'])
        children_fc2 = self._forward_children_fc2(fc2_model, child_state['children_s_full'], parent_policy['alive_mask'])
        children_policy = self._forward_children_policy(
            pv_model,
            child_state['children_s_full'],
            children_fc2['macro_pred'],
            parent_policy['alive_mask'],
        )
        children_agg = self._compute_children_aggregates(
            children_policy['cv_input'],
            children_policy['bar_i'],
            children_policy['bar_z'],
            children_policy['alive_mask_children'],
            children_fc2['lnk_pred'],
            children_fc2['hatc_pred'],
            children_policy.get('gathered'),
        )

        total_loss = parent_agg['loss'] + children_agg['loss']
        return {
            'parent': {
                'fc2_input': parent_fc2['fc2_input'],
                'macro_pred': parent_fc2['macro_pred'],
                'hatc_pred': parent_fc2['hatc_pred'],
                'lnk_pred': parent_fc2['lnk_pred'],
                'pv_input': parent_policy['pv_input'],
                'pv_output': parent_policy['pv_output'],
                'updated_alive': parent_policy['updated_alive'],
                'alive_mask': parent_policy['alive_mask'],
                'macro_actual': {
                    'hatc': parent_agg['hatc_actual'],
                    'lnk': parent_agg['lnk_actual'],
                },
                'loss': parent_agg['loss'],
            },
            'children': {
                'children_s_full': child_state['children_s_full'],
                'fc2_input': children_fc2['fc2_input'],
                'macro_pred': children_fc2['macro_pred'],
                'hatc_pred': children_fc2['hatc_pred'],
                'lnk_pred': children_fc2['lnk_pred'],
                'cv_input': children_policy['cv_input'],
                'cv_output': children_policy['cv_output'],
                'alive_mask_children': children_policy['alive_mask_children'],
                'macro_actual': {
                    'hatc': children_agg['hatc_actual'],
                    'lnk': children_agg['lnk_actual'],
                },
                'loss': children_agg['loss'],
            },
            'total': {
                'loss_total': total_loss,
            },
            'FC2_input_parent': parent_fc2['fc2_input'],
            'fc2_output_parent': parent_fc2['macro_pred'],
            'PV_input': parent_policy['pv_input'],
            'PV_output': parent_policy['pv_output'],
            'updated_alive': parent_policy['updated_alive'],
            'alive_mask': parent_policy['alive_mask'],
            'children_s_full': child_state['children_s_full'],
            'FC2_input_children': children_fc2['fc2_input'],
            'FC2_output_children1': children_fc2['macro_pred'],
            'CV_input': children_policy['cv_input'],
            'CV_output': children_policy['cv_output'],
            'alive_mask_children': children_policy['alive_mask_children'],
            'lnk_parent': parent_agg['lnk_actual'],
            'hatc_parent': parent_agg['hatc_actual'],
            'lnk_children': children_agg['lnk_actual'],
            'hatc_children': children_agg['hatc_actual'],
            'loss_parent': parent_agg['loss'],
            'loss_children': children_agg['loss'],
            'loss_total': total_loss,
        }

    def loss(self, fc2_model: torch.nn.Module, pv_model: torch.nn.Module):
        outputs = self.forward(fc2_model, pv_model)
        outputs['diagnostics'] = {
            'parent_hatc': compute_fit_stats_t(outputs['parent']['macro_actual']['hatc'], outputs['parent']['hatc_pred']),
            'parent_lnk': compute_fit_stats_t(outputs['parent']['macro_actual']['lnk'], outputs['parent']['lnk_pred']),
            'children_hatc': compute_fit_stats_t(outputs['children']['macro_actual']['hatc'], outputs['children']['hatc_pred']),
            'children_lnk': compute_fit_stats_t(outputs['children']['macro_actual']['lnk'], outputs['children']['lnk_pred']),
            'loss_parent': float(outputs['loss_parent'].detach().item()),
            'loss_children': float(outputs['loss_children'].detach().item()),
            'loss_total': float(outputs['loss_total'].detach().item()),
        }
        return outputs['loss_total'], outputs


# Backward-compatible alias
FC2LossPipe = FC2Pipeline
