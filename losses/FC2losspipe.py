"""
FC2 Pipeline (synced from experiments/FC2train2.ipynb)
"""
from __future__ import annotations

from typing import Optional, Dict

import numpy as np
import pandas as pd
import torch

from config import Config
from experiments.fill_fullN_entrants import fill_df_to_fullN

# Helpers

def compute_quantile_features(b_vals: torch.Tensor, z_vals: torch.Tensor, quantiles: torch.Tensor):
    b_quantiles = torch.quantile(b_vals, quantiles)
    z_quantiles = torch.quantile(z_vals, quantiles)
    quantile_features = torch.cat([b_quantiles, z_quantiles], dim=0)
    return quantile_features

class FC2Pipeline:
    """
    OOP pipeline aligned with losses/FC2losspipe.py
    """
    def __init__(self, pkl_path=None, df=None, full_N=2000, entry_num=None, device='cpu'):
        self.device = torch.device(device)
        self.full_N = full_N
        self.branch_num = 2
        self.phi = Config.PHI
        self.delta = Config.DELTA
        self.G = Config.G

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
        FC2_input_parent = torch.zeros((self.path_num, 201), dtype=torch.float32, device=self.device)
        for i in range(self.path_num):
            b_vals = self.P_s_full[i, self.alive_mask[i], 0]
            z_vals = self.P_s_full[i, self.alive_mask[i], 1]
            quantile_features = compute_quantile_features(b_vals, z_vals, quantiles=torch.linspace(0, 1, steps=100, device=self.device))
            x_feature = self.P_s_full[i, self.alive_mask[i], 4].mean().unsqueeze(0)
            FC2_input_parent[i, :200] = quantile_features
            FC2_input_parent[i, 200] = x_feature
        return FC2_input_parent

    def build_fc2_input_children(self, children_s_full, alive_mask):
        FC2_input_children = torch.zeros((self.path_num, 2, 201), dtype=torch.float32, device=self.device)
        for i in range(self.path_num):
            for j in range(2):
                b_vals = children_s_full[i, alive_mask[i, :, j].bool(), j, 0]
                z_vals = children_s_full[i, alive_mask[i, :, j].bool(), j, 1]
                quantile_features = compute_quantile_features(b_vals, z_vals, quantiles=torch.linspace(0, 1, steps=100, device=self.device))
                x_feature = children_s_full[i, alive_mask[i, :, j].bool(), j, 4].mean().unsqueeze(0)
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

    def forward(self, fc2_model: torch.nn.Module, pv_model: torch.nn.Module):
        FC2_input_parent = self.build_fc2_input_parent()
        fc2_out_parent = fc2_model(FC2_input_parent)
        hatc_parent_pred = fc2_out_parent['hatc']
        lnk_parent_pred = fc2_out_parent['lnk']
        fc2_output_parent = torch.cat([lnk_parent_pred, hatc_parent_pred], dim=-1)

        FC2_output_parent = fc2_output_parent.unsqueeze(1).expand(-1, self.full_N, -1) * self.alive_mask.unsqueeze(-1)
        PV_input = torch.cat((self.P_s_full, FC2_output_parent), dim=-1)

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

        Y = torch.exp(PV_input[..., 4:5] + PV_input[..., 1:2]) * self.K_parent_full
        Phi = (1 - self.phi) * (1 + torch.exp(PV_input[..., 1:2] + PV_input[..., 3:4])) * self.K_parent_full * bar_z
        I = bar_i * self.K_parent_full * PV_input[..., 3:4] - bar_z * self.K_parent_full + self.delta * self.K_parent_full
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


        FC2_input_children = self.build_fc2_input_children(children_s_full, alive_mask)

        flat_children = FC2_input_children.reshape(-1, FC2_input_children.shape[-1])
        fc2_out_children = fc2_model(flat_children)
        hatc_children_pred = fc2_out_children['hatc'].view(self.path_num, 2, 1)

        lnk_children_pred = fc2_out_children['lnk'].view(self.path_num, 2, 1)

        FC2_output_children1 = torch.cat([lnk_children_pred, hatc_children_pred], dim=-1)

        FC2_output_children = FC2_output_children1.unsqueeze(1).expand(-1, self.full_N, -1, -1) * alive_mask.unsqueeze(-1)

        CV_input = torch.cat((children_s_full, FC2_output_children), dim=-1)

        CV_output = self._pv_forward(pv_model, CV_input)
        bar_z_children = self._get_out(CV_output, 'bar_z', 6)
        bar_i_children = self._get_out(CV_output, 'bar_i', 5)

        alive_mask_children = alive_mask.clone()
        alive_prob_children = bar_z_children.clamp(0, 1).squeeze(-1)
        base_alive_children = alive_mask.float()
        updated_alive_children = base_alive_children * alive_prob_children
        alive_mask_children = updated_alive_children.unsqueeze(-1)

        bari = bar_i_children
        K_children_full = self.K_children_full * (1 + self.G * bari)

        Y = torch.exp(CV_input[..., 4:5] + CV_input[..., 1:2]) * K_children_full
        Phi = (1 - self.phi) * (1 + torch.exp(CV_input[..., 1:2] + CV_input[..., 3:4])) * K_children_full * bar_z_children
        I = bar_i_children * K_children_full * CV_input[..., 3:4] - bar_z_children * K_children_full + self.delta * K_children_full
        C = torch.abs(Y - I - Phi)
        lnk_children = torch.log(torch.sum(K_children_full * alive_mask_children[..., 0:1], dim=1) + 1e-8)
        hatc_children = torch.log(torch.sum(C * alive_mask_children[..., 0:1], dim=1) / (torch.sum(K_children_full * alive_mask_children[..., 0:1], dim=1) + 1e-8))

        loss_children = torch.mean((lnk_children - lnk_children_pred)**2) + torch.mean((hatc_children - hatc_children_pred)**2)

        return {
            'FC2_input_parent': FC2_input_parent,
            'fc2_output_parent': fc2_output_parent,
            'PV_input': PV_input,
            'PV_output': PV_output,
            'updated_alive': updated_alive,
            'alive_mask': alive_mask,
            'children_s_full': children_s_full,
            'FC2_input_children': FC2_input_children,
            'FC2_output_children1': FC2_output_children1,
            'CV_input': CV_input,
            'CV_output': CV_output,
            'alive_mask_children': alive_mask_children,
            'lnk_parent': lnk_parent,
            'hatc_parent': hatc_parent,
            'lnk_children': lnk_children,
            'hatc_children': hatc_children,
            'loss_parent': loss_parent,
            'loss_children': loss_children,
            'loss_total': loss_parent + loss_children,
        }

    def loss(self, fc2_model: torch.nn.Module, pv_model: torch.nn.Module):
        outputs = self.forward(fc2_model, pv_model)
        return outputs['loss_total'], outputs


# Backward-compatible alias
FC2LossPipe = FC2Pipeline
