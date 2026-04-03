"""
Sample 类：训练数据生成器

给定模型 → 生成可直接训练的 DataFrame

特点：
- (b,z) 主要来自均匀抽样或利润为正的可行域抽样
- 每条 path 公司数较少（用于稳定训练与调试）
- 结构对齐 SDF/FC1/Policy&Value 的输入格式（按三元组组织）
"""

import torch
import pandas as pd
import numpy as np
import uuid
from typing import Dict, Optional, Tuple, List, Union
from tqdm import tqdm

import sys
sys.path.append('..')
from config import Config, SIMMODEL

from .data_utils import (
    sample_ar1,
    sample_stationary_ar1,
    sample_uniform,
    sample_bernoulli,
    generate_macro_state,
    generate_firm_states,
    generate_initial_macro_proxy,
    build_firm_state_tensor
)
from .tensor_data import TensorTable
from .sample_parallel import build_policy_value_tables_parallel


class Sample:
    """
    训练数据生成器
    
    给定 models（可包含 SDF/FC1/Policy&Value/FC2 的子集），
    生成一个按 path 切分、按 (path, ID) 组织为分支元组的 DataFrame
    
    支持两种数据模式：
    - sample: 每 path 少量公司（默认2家），用于稳定训练与调试
    - simulate: 每 path 大量公司（默认200家），用于更真实的宏观聚合
    
    注意：这里的 simulate 模式不同于 SimulateTS 类的树状时间序列模拟，
    它仍然是"训练用截面样本"，只是公司数量更多。
    """

    
    # 预设的公司数量
    DEFAULT_GROUP_SIZE = {
        'sample': 2,      # 训练调试用：少量公司
        'simulate': 200   # 截面聚合用：大量公司
    }
    SDF_COLUMNS = ['path', 'branch', 'x_t', 'x_t1', 'Hatcf_t', 'LnKF_t']
    PV_COLUMNS = [
        'path', 'firm', 'branch', 'Entry',
        'b', 'z', 'ETA', 'i', 'x', 'Hatcf', 'LnKF', 'M', 'K'
    ]
    
    def __init__(
        self,
        models: Optional[Dict] = None,
        config: type = Config,
        n_samples: int = 10000,
        n_paths: int = 1000,
        group_size: int = None,
        branch_num: int = 2,
        data_mode: str = 'sample',
        sampling_mode: str = 'uniform',
        enable_entry: bool = None,
        entry_rate: float = 0.1,
        macro_source_df: Optional[pd.DataFrame] = None,
        device: torch.device = None
    ):
        """
        Args:
            models: 模型字典，可包含：
                - 'sdf_fc1': SDFFC1Combined
                - 'policy_value': PolicyValueModel
                - 'fc2': FC2Model
            config: 配置类
            n_samples: 总样本数
            n_paths: path 数量
            group_size: 每条 path 的公司数（如果为 None，则根据 data_mode 自动设置）
            branch_num: 分支数
            data_mode: 数据模式
                - 'sample': 每 path 2 家公司（训练调试用）
                - 'simulate': 每 path 200 家公司（截面聚合用）
            sampling_mode: 采样模式 ('uniform', 'feasible', 'realbz')
            enable_entry: 是否启用公司进入（None 时 simulate 模式自动启用）
            entry_rate: 进入率（潜在进入者占现有公司数的比例）
            device: 设备
        """
        self.models = models or {}
        self.config = config
        self.n_samples = n_samples
        self.n_paths = n_paths
        self.branch_num = branch_num
        self.data_mode = data_mode
        self.sampling_mode = sampling_mode
        self.entry_rate = entry_rate
        self.device = device or config.DEVICE
        self.macro_source_df = macro_source_df.copy() if macro_source_df is not None else None
        
        # simulate 模式默认启用 entry
        if enable_entry is None:
            self.enable_entry = (data_mode == 'simulate')
        else:
            self.enable_entry = enable_entry
        
        # 根据 data_mode 设置 group_size
        if group_size is not None:
            self.group_size = group_size
        else:
            self.group_size = self.DEFAULT_GROUP_SIZE.get(data_mode, 2)
        
        # 计算实际需要的样本数
        # 每个 path 有 group_size 家公司，每家公司有 (branch_num + 1) 行
        self.rows_per_path = self.group_size * (branch_num + 1)
        
        # 如果指定了 n_samples，调整 n_paths
        if n_samples is not None:
            self.n_paths = n_samples // self.rows_per_path

    def _sample_parent_macro_state(
        self,
        n_paths: int,
        device: Optional[torch.device] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        从上一轮 simulate 的 parent macro 表中抽样当前节点宏观状态。

        返回：
        - x_t
        - Hatc_t
        - LnK_t

        如果当前没有可用的 macro source，则返回 None。
        """
        if self.macro_source_df is None or self.macro_source_df.empty:
            return None

        use_df = self.macro_source_df
        if 'branch' in use_df.columns:
            parent_df = use_df[use_df['branch'] == -1]
            if not parent_df.empty:
                use_df = parent_df

        required = {'x', 'Hatc', 'LnK'}
        if not required.issubset(use_df.columns):
            return None

        n = int(n_paths)
        replace = len(use_df) < n
        sampled = use_df.sample(n=n, replace=replace)
        tgt_device = device or self.device
        x_t = torch.tensor(sampled['x'].to_numpy(), device=tgt_device, dtype=torch.float32)
        hatc_t = torch.tensor(sampled['Hatc'].to_numpy(), device=tgt_device, dtype=torch.float32)
        lnk_t = torch.tensor(sampled['LnK'].to_numpy(), device=tgt_device, dtype=torch.float32)
        return x_t, hatc_t, lnk_t
    
    def build_df(self) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        构建训练 DataFrame
        
        Returns:
            - sample 模式: df_firm (firm-level DataFrame)
            - simulate 模式: (df_firm, df_macro) 元组
              - df_firm: firm-level DataFrame
              - df_macro: macro-level DataFrame (按 path 聚合)
        """
        if self.sampling_mode == 'uniform':
            firm_table, macro_table = build_policy_value_tables_parallel(
                self, include_macro=(self.data_mode == 'simulate')
            )
            df_firm = self._firm_tensor_table_to_df(firm_table)
            if self.data_mode == 'simulate':
                df_macro = self._macro_tensor_table_to_df(macro_table) if macro_table is not None else pd.DataFrame()
                if 'policy_value' in self.models and self.models['policy_value'] is not None:
                    df_firm = self.fill_fc1(df_firm, df_macro)
                    df_firm, df_macro = self.fill_policy_value(df_firm, df_macro)
                    df_macro = self._update_macro_from_policy(df_firm, df_macro)
                return df_firm, df_macro
            return df_firm

        all_firm_data = []
        all_macro_data = []
        
        for path_idx in tqdm(range(self.n_paths), desc=f"Generating {self.data_mode} data"):
            firm_data, macro_data = self._generate_path(path_idx)
            all_firm_data.append(firm_data)
            if macro_data is not None:
                # macro_data may be a single dict or a list of dicts
                if isinstance(macro_data, list):
                    all_macro_data.extend(macro_data)
                else:
                    all_macro_data.append(macro_data)
        
        # 合并所有 path
        df_firm = pd.concat(all_firm_data, ignore_index=True)
        
        # simulate 模式返回 (df_firm, df_macro)，优先用 FC1/Policy 填充并基于微观聚合宏观
        if self.data_mode == 'simulate':
            df_macro = pd.DataFrame(all_macro_data)
            if 'policy_value' in self.models and self.models['policy_value'] is not None:
                df_firm = self.fill_fc1(df_firm, df_macro)
                df_firm, df_macro = self.fill_policy_value(df_firm, df_macro)
                df_macro = self._update_macro_from_policy(df_firm, df_macro)
            return df_firm, df_macro
        
        # sample 模式只返回 df_firm
        return df_firm

    def _firm_tensor_table_to_df(self, table: TensorTable) -> pd.DataFrame:
        df = table.to_dataframe()
        for col in ['path', 'firm', 'branch']:
            df[col] = np.rint(df[col]).astype(np.int64)
        df['ID'] = df['firm'].astype(str)
        df['t'] = np.where(
            df['branch'] == 0,
            't',
            't+1_' + (df['branch'] - 1).astype(np.int64).astype(str)
        )
        keep_cols = [
            'path', 'ID', 't', 'branch', 'b', 'z', 'ETA', 'i', 'x',
            'Hatcf', 'LnKF', 'M', 'K', 'Entry'
        ]
        return df[keep_cols]

    def _macro_tensor_table_to_df(self, table: TensorTable) -> pd.DataFrame:
        df_macro = table.to_dataframe()
        for col in ['path', 't_code', 'branch', 'n_firms', 'n_entrants']:
            if col in df_macro.columns:
                df_macro[col] = np.rint(df_macro[col]).astype(np.int64)
        df_macro['t'] = np.where(
            df_macro['branch'] == -1,
            't',
            't+1_' + df_macro['branch'].astype(np.int64).astype(str)
        )
        if 't_code' in df_macro.columns:
            df_macro = df_macro.drop(columns=['t_code'])
        return df_macro

    def build_sdf_fc1_df(self) -> pd.DataFrame:
        """
        构建 SDF/FC1 训练数据（仅宏观跨期变量）
        
        列包含：x_t, Hatcf_t, LnKF_t, x_t1 （t+1 期）
        这里的 x_t1 是直接从 AR(1) 演化的，而不是通过模型预测的，主要用于训练 SDF/FC1 的宏观跨期关系，并且不再展示parent的状态，因为parent其实就是x_t, Hatcf_t, LnKF_t
        """
        return self.build_sdf_fc1_tensor().to_dataframe()

    def build_sdf_fc1_tensor(self) -> TensorTable:
        """
        构建 SDF/FC1 训练数据（tensor 版本，末端可选转 DataFrame）。
        """
        n_paths = self.n_paths
        bnum = self.branch_num
        device = self.device

        macro_parent = self._sample_parent_macro_state(n_paths, device)
        if macro_parent is not None:
            x_t, hatcf_t, lnkf_t = macro_parent
        else:
            x_t = sample_stationary_ar1(
                n_paths, self.config.RHO_X, self.config.SIGMA_X, self.config.XBAR, device
            )
            hatcf_t, lnkf_t = generate_initial_macro_proxy(n_paths, device)

        # 每个 path 复制到各个分支
        x_t_rep = x_t.repeat_interleave(bnum)
        hatcf_rep = hatcf_t.repeat_interleave(bnum)
        lnkf_rep = lnkf_t.repeat_interleave(bnum)
        x_t1 = sample_ar1(x_t_rep, self.config.RHO_X, self.config.SIGMA_X, self.config.XBAR)

        path_idx = torch.arange(n_paths, device=device, dtype=torch.float32).repeat_interleave(bnum)
        branch_idx = torch.arange(bnum, device=device, dtype=torch.float32).repeat(n_paths)
        rows = torch.stack([path_idx, branch_idx, x_t_rep, x_t1, hatcf_rep, lnkf_rep], dim=1).to(torch.float32)
        return TensorTable(rows, self.SDF_COLUMNS)

    def build_policy_value_tensor(self) -> TensorTable:
        """
        构建 Policy/Value 训练数据（tensor 版本）。

        说明：
        - uniform 采样模式下，走 path-parallel GPU 批量构造
        - 其他采样模式保留旧实现以保持兼容
        """
        if self.sampling_mode == 'uniform':
            firm_table, _ = build_policy_value_tables_parallel(self, include_macro=False)
            return firm_table

        device = self.device
        rows: List[torch.Tensor] = []
        firm_offset = 0

        for path_idx in tqdm(range(self.n_paths), desc="Generating policy/value tensor data"):
            x_t = sample_stationary_ar1(
                1, self.config.RHO_X, self.config.SIGMA_X, self.config.XBAR, device
            ).reshape(())
            b, z, eta, i = generate_firm_states(
                self.group_size, x_t.reshape(1), mode=self.sampling_mode, device=device
            )
            hatcf_t, lnkf_t = generate_initial_macro_proxy(1, device)
            hatcf_t = hatcf_t.reshape(())
            lnkf_t = lnkf_t.reshape(())

            if self.data_mode == 'simulate':
                weights = torch.rand(self.group_size, device=device)
                weights = weights / weights.sum().clamp(min=1e-8)
                K = torch.exp(lnkf_t) * weights
            else:
                K = torch.ones(self.group_size, device=device)

            # parent 行：branch=0，M=1
            firm_id = torch.arange(firm_offset, firm_offset + self.group_size, device=device, dtype=torch.float32)
            firm_offset += self.group_size
            parent_rows = torch.stack(
                [
                    torch.full((self.group_size,), float(path_idx), device=device),
                    firm_id,
                    torch.zeros(self.group_size, device=device),   # branch
                    torch.ones(self.group_size, device=device),    # Entry
                    b,
                    z,
                    eta,
                    i,
                    x_t.expand(self.group_size),
                    hatcf_t.expand(self.group_size),
                    lnkf_t.expand(self.group_size),
                    torch.ones(self.group_size, device=device),    # M
                    K,
                ],
                dim=1
            )
            rows.append(parent_rows)

            # children 行：branch=1..B
            for branch_k in range(self.branch_num):
                x_t1 = sample_ar1(x_t.reshape(1), self.config.RHO_X, self.config.SIGMA_X, self.config.XBAR).reshape(())
                z_t1 = sample_ar1(z, self.config.RHO_Z, self.config.SIGMA_Z, self.config.ZBAR)
                eta_t1 = sample_bernoulli(self.group_size, self.config.ZETA, device)
                i_t1 = sample_uniform(self.group_size, 0.0, self.config.I_THRESHOLD, device)

                if 'sdf_fc1' in self.models and self.models['sdf_fc1'] is not None:
                    with torch.no_grad():
                        _, _, M_t1, hatcf_t1, lnkf_t1 = self.models['sdf_fc1'].forward_step(
                            x_prev=x_t.reshape(1).to(torch.float32),
                            x_curr=x_t1.reshape(1).to(torch.float32),
                            hatcf_prev=hatcf_t.reshape(1).to(torch.float32),
                            lnkf_prev=lnkf_t.reshape(1).to(torch.float32),
                            return_physical=True
                        )
                    m_child = M_t1.reshape(()).expand(self.group_size)
                    hatcf_child = hatcf_t1.reshape(()).expand(self.group_size)
                    lnkf_child = lnkf_t1.reshape(()).expand(self.group_size)
                else:
                    m_child = torch.ones(self.group_size, device=device)
                    hatcf_child = hatcf_t.expand(self.group_size)
                    lnkf_child = lnkf_t.expand(self.group_size)

                child_rows = torch.stack(
                    [
                        torch.full((self.group_size,), float(path_idx), device=device),
                        firm_id,
                        torch.full((self.group_size,), float(branch_k + 1), device=device),
                        torch.zeros(self.group_size, device=device),  # Entry
                        b,
                        z_t1,
                        eta_t1,
                        i_t1,
                        x_t1.expand(self.group_size),
                        hatcf_child,
                        lnkf_child,
                        m_child,
                        K,
                    ],
                    dim=1
                )
                rows.append(child_rows)

        if rows:
            data = torch.cat(rows, dim=0).to(torch.float32)
        else:
            data = torch.empty((0, len(self.PV_COLUMNS)), device=device, dtype=torch.float32)
        return TensorTable(data, self.PV_COLUMNS)

    def build_policy_value_df(self) -> pd.DataFrame:
        """
        构建 Policy/Value 训练数据（微观截面，含 t 与 t+1）。

        新实现：先在 GPU 上生成 tensor，再在末端转换为 DataFrame。
        """
        return self._firm_tensor_table_to_df(self.build_policy_value_tensor())

    def build_fc2_df(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        构建 FC2 训练数据（仅 t 期的 firm-level + macro 聚合）
        """
        if self.data_mode != 'simulate':
            raise ValueError("FC2 data requires data_mode='simulate' ")
        df_firm, df_macro = self.build_df()
        return df_firm, df_macro
    
    def _generate_path(self, path_idx: int) -> Tuple[pd.DataFrame, Optional[Dict]]:
        """
        生成单个 path 的数据
        
        Returns:
            firm_data: firm-level DataFrame
            macro_data: macro-level dict (仅 simulate 模式返回)
        """
        data = []
        
        # Step 1: 生成宏观状态 x_t
        macro_parent = self._sample_parent_macro_state(1, self.device)
        if macro_parent is not None:
            x_t = float(macro_parent[0].reshape(-1)[0].item())
            hatcf_t = float(macro_parent[1].reshape(-1)[0].item())
            lnkf_t = float(macro_parent[2].reshape(-1)[0].item())
        else:
            x_t = sample_stationary_ar1(
                1, self.config.RHO_X, self.config.SIGMA_X,
                self.config.XBAR, self.device
            ).item()
            hatcf_t_tensor, lnkf_t_tensor = generate_initial_macro_proxy(1, self.device)
            hatcf_t, lnkf_t = hatcf_t_tensor.item(), lnkf_t_tensor.item()
        
        # Step 2: 生成公司状态
        b, z, eta, i = generate_firm_states(
            self.group_size,
            torch.tensor(x_t, device=self.device),
            mode=self.sampling_mode,
            device=self.device
        )
        
        # 初始化资本（simulate 模式下用随机资本并用权重聚合）
        if self.data_mode == 'simulate':
            weights = torch.rand(self.group_size, device=self.device)
            weights = weights / weights.sum()
            K = torch.exp(torch.tensor(lnkf_t, device=self.device)) * weights

            entryK_t = (K).mean().item()
        else:
            K = torch.ones(self.group_size, device=self.device)
        
        # 记录每个 branch 的 t+1 期宏观状态（用于进入者判断）
        branch_states = {}
        for branch_k in range(self.branch_num):
            x_t1 = sample_ar1(
                torch.tensor(x_t, device=self.device),
                self.config.RHO_X,
                self.config.SIGMA_X,
                self.config.XBAR
            ).item()
            
            # FC1 预测宏观状态（如果有模型）
            if 'sdf_fc1' in self.models and self.models['sdf_fc1'] is not None:
                hatcf_t1, lnkf_t1, M_t1 = self._predict_macro(x_t, x_t1, hatcf_t, lnkf_t)
            else:
                # 简单演化
                hatcf_t1 = hatcf_t + np.random.randn() * 0.01
                lnkf_t1 = lnkf_t + np.random.randn() * 0.01
                M_t1 = None
            
            branch_states[branch_k] = {
                'x_t1': x_t1,
                'hatcf_t1': hatcf_t1,
                'lnkf_t1': lnkf_t1,
                'M': M_t1
            }
        
        # Step 4: 为每家公司创建 triplet，同时收集 branch 层面的 z 以便宏观聚合
        branch_z_values = {k: [] for k in range(self.branch_num)}
        for firm_idx in range(self.group_size):
            firm_id = uuid.uuid4().hex
            
            # Parent (t)
            parent_row = {
                'path': path_idx,
                'ID': firm_id,
                't': 't',
                'branch': 0,
                'b': b[firm_idx].item(),
                'z': z[firm_idx].item(),
                'ETA': eta[firm_idx].item(),
                'i': i[firm_idx].item(),
                'x': x_t,
                'Hatcf': hatcf_t,
                'LnKF': lnkf_t,
                'M': 1.0,  # parent M 默认取 1
                'K': K[firm_idx].item(),
                'Entry': 1  # 现有公司
            }
            data.append(parent_row)
            
            # Children (t+1^k)
            for branch_k in range(self.branch_num):
                branch_info = branch_states[branch_k]
                x_t1 = branch_info['x_t1']
                hatcf_t1 = branch_info['hatcf_t1']
                lnkf_t1 = branch_info['lnkf_t1']
                M_t1 = branch_info['M']
                
                z_t1 = sample_ar1(
                    z[firm_idx],
                    self.config.RHO_Z,
                    self.config.SIGMA_Z,
                    self.config.ZBAR
                ).item()
                
                eta_t1 = sample_bernoulli(1, self.config.ZETA, self.device).item()
                i_t1 = sample_uniform(1, 0.0, self.config.I_THRESHOLD, self.device).item()
                
                child_row = {
                    'path': path_idx,
                    'ID': firm_id,
                    't': f't+1_{branch_k}',
                    'branch': branch_k + 1,
                    'b': parent_row['b'],  # 初始与 parent 相同，后续可能更新
                    'z': z_t1,
                    'ETA': eta_t1,
                    'i': i_t1,
                    'x': x_t1,
                    'Hatcf': hatcf_t1,
                    'LnKF': lnkf_t1, # 初始与 parent 相同，后续可能更新
                    'M': M_t1,
                    'K': K[firm_idx].item(),
                    'Entry': 0  # 现有公司
                }
                data.append(child_row)
                branch_z_values[branch_k].append(z_t1)
        
        # Step 5: 生成进入者（如果启用）
        # 进入者在 t+1 期才进入，每个 branch 独立判断
        total_entrants = 0
        branch_entrants = {k: 0 for k in range(self.branch_num)}
        if self.enable_entry:

            for branch_k in range(self.branch_num):
                branch_info = branch_states.get(branch_k)
                if branch_info is None:
                    continue
                
                entry_data, n_entrants = self._generate_entrants(
                    path_idx,
                    x_t1=branch_info['x_t1'],
                    hatcf_t1=branch_info['hatcf_t1'],
                    lnkf_t1=branch_info['lnkf_t1'],
                    branch_k=branch_k,
                    enrty_k = entryK_t
                )
                data.extend(entry_data)
                total_entrants += n_entrants
                branch_entrants[branch_k] += n_entrants
        
        # 构建 firm DataFrame
        df_firm = pd.DataFrame(data)
        
        
        # simulate 模式：计算宏观聚合（包含 t=0 的父节点和 t+1 的各分支）
        macro_data = None
        if self.data_mode == 'simulate':
            # 合并现有公司和进入者的资本
            # 注意：进入者在 t+1 期进入，所以 t 期的宏观聚合只包含现有公司

            total_firms_t = self.group_size  # t 期只有现有公司

            
            macro_rows = []
            macro_rows.append({
                'path': path_idx,
                't': "t",
                'branch': -1,
                'x': x_t,
                'n_firms': total_firms_t,
                'n_entrants': total_entrants,
                'Hatcf': hatcf_t,
                'LnKF': lnkf_t,
                'M': 1
            })

            # 各分支 (t=1, branch = k)
            for branch_k, branch_info in branch_states.items():
                z_vals = branch_z_values.get(branch_k, [])
                n_branch = len(z_vals)

                x_t1 = branch_info['x_t1']
                M_t1 = branch_info.get('M')




                macro_rows.append({
                    'path': path_idx,
                    't': "t+1_" + str(branch_k),
                    'branch': branch_k,
                    'x': x_t1,
                    'x_t': x_t,
                    'x_t1': x_t1,
                    'n_firms': n_branch,
                    'n_entrants': branch_entrants.get(branch_k, 0),
                    'Hatcf': branch_info['hatcf_t1'],
                    'LnKF': branch_info['lnkf_t1'],
                    'Hatcf_t': hatcf_t,
                    'LnKF_t': lnkf_t,
                    'M': M_t1
                })

            macro_data = macro_rows
        
        return df_firm, macro_data
    
    def _generate_entrants(
        self,
        path_idx: int,
        x_t1: float,
        hatcf_t1: float,
        lnkf_t1: float,
        branch_k: int,
        enrty_k = None
    ) -> Tuple[List[Dict], int]:
        """
        生成 t+1 期的潜在进入者并筛选
        
        注意：进入者在 t+1 期才进入，所以：
        1. 进入判断基于 x_{t+1} 和 z（潜在进入者的生产率）
        2. 进入者只有 t+1 期的数据行（作为 child），没有 t 期的 parent
        
        进入条件：entry_value > 0
        entry_value = 1 + max(profit, 0) - i
        profit = exp(x_{t+1} + z) - δ
        
        Args:
            path_idx: 路径索引
            x_t1: t+1 期宏观状态（进入时刻）
            hatcf_t1: t+1 期 ĉf
            lnkf_t1: t+1 期 ln Kf
            branch_k: 分支编号
            base_firm_id: 进入者 ID 起始值
        
        Returns:
            entry_data: 进入者数据列表（只有 t+1 期的行）
            n_entrants: 实际进入者数量
        """
        entry_data = []
        
        # 生成潜在进入者数量
        n_potential = max(10, int(self.group_size * self.entry_rate))
        
        # 潜在进入者的状态（b=0，z 从稳态分布抽样）
        z_potential = sample_stationary_ar1(
            n_potential, self.config.RHO_Z, self.config.SIGMA_Z,
            self.config.ZBAR, self.device
        )
        i_potential = sample_uniform(n_potential, 0.0, self.config.I_THRESHOLD, self.device)
        
        # 计算进入价值（基于 t+1 期的宏观状态 x_{t+1}）
        delta = self.config.DELTA
        x_tensor = torch.tensor(x_t1, device=self.device)
        profit = torch.exp(x_tensor + z_potential) - delta
        entry_value = 1 + profit - i_potential
        
        # 筛选进入者
        enter_mask = entry_value > 0
        n_entrants = enter_mask.sum().item()
        
        if n_entrants == 0:
            return [], 0
        
        # 提取进入者状态
        z_entrants = z_potential[enter_mask]
        i_entrants = i_potential[enter_mask]
        if enrty_k is None:
            K_entrants = torch.ones(n_entrants, device=self.device)
        else:
            K_entrants = torch.full((n_entrants,), float(enrty_k), device=self.device)
        
        # 为每个进入者创建 t+1 期的数据行（进入者没有 t 期的 parent）
        for ent_idx in range(n_entrants):
            firm_id = uuid.uuid4().hex
            
            # 进入者在 t+1 期进入，只有这一行
            entry_row = {
                'path': path_idx,
                'ID': firm_id,
                't': f't+1_{branch_k}',
                'branch': branch_k + 1,
                'b': 0.0,  # 进入者初始杠杆为 0
                'z': z_entrants[ent_idx].item(),
                'ETA': sample_bernoulli(1, self.config.ZETA, self.device).item(),
                'i': i_entrants[ent_idx].item(),
                'x': x_t1,
                'Hatcf': hatcf_t1,
                'LnKF': lnkf_t1,
                'K': K_entrants[ent_idx].item(),
                'Entry': 1  # 标记为进入者
            }
            entry_data.append(entry_row)
        
        return entry_data, n_entrants
    
    def _predict_macro(
        self, 
        x_t: float, 
        x_t1: float, 
        hatcf_t: float, 
        lnkf_t: float
    ) -> Tuple[float, float, Optional[float]]:
        """
        使用 FC1 预测下期宏观状态
        """
        model = self.models['sdf_fc1']
        model.eval()
        # comment: 同样的，这里也需要有M
        with torch.no_grad():
            _, _, M, hatcf_t1, lnkf_t1 = model.forward_step(
                x_prev=torch.tensor([x_t], device=self.device, dtype=torch.float32),
                x_curr=torch.tensor([x_t1], device=self.device, dtype=torch.float32),
                hatcf_prev=torch.tensor([hatcf_t], device=self.device, dtype=torch.float32),
                lnkf_prev=torch.tensor([lnkf_t], device=self.device, dtype=torch.float32),
                return_physical=True
            )
        
        return hatcf_t1.item(), lnkf_t1.item(), M.squeeze().item()
    
    def fill_fc1(self, df: pd.DataFrame, df_macro: pd.DataFrame = None) -> pd.DataFrame:
        """
        使用 FC1 填充宏观状态：
        - 优先使用 df_macro 中的 Hatc/LnK 直接合并（simulate 模式下已有宏观表）
        - 否则回退到 sdf_fc1 模型逐行预测
        """
        df = df.copy()

        # 如果有宏观表，直接按 path + branch 对齐填充
        if df_macro is not None:
            df_macro_local = df_macro.copy()
            # 兼容不同命名（Hatcf/LnKF 或 Hatc/LnK）
            rename_map = {}
            if 'Hatc' in df_macro_local.columns:
                rename_map['Hatc'] = 'Hatcf_macro'
            elif 'Hatcf' in df_macro_local.columns:
                rename_map['Hatcf'] = 'Hatcf_macro'
            if 'LnK' in df_macro_local.columns:
                rename_map['LnK'] = 'LnKF_macro'
            elif 'LnKF' in df_macro_local.columns:
                rename_map['LnKF'] = 'LnKF_macro'
            df_macro_local = df_macro_local.rename(columns=rename_map)
            # macro 中 parent 为 branch=-1，child 为分支编号；firm parent 为 0，child 为 1,2,...
            df['macro_branch'] = np.where(df['branch'] > 0, df['branch'] - 1, -1)
            macro_merge_cols = ['path', 'branch'] + [c for c in ['Hatcf_macro', 'LnKF_macro'] if c in df_macro_local.columns]
            df = df.merge(
                df_macro_local[macro_merge_cols],
                left_on=['path', 'macro_branch'],
                right_on=['path', 'branch'],
                how='left',
                suffixes=('', '_drop')
            )
            if 'Hatcf_macro' in df.columns:
                df['Hatcf'] = df['Hatcf_macro'].fillna(df['Hatcf'])
            if 'LnKF_macro' in df.columns:
                df['LnKF'] = df['LnKF_macro'].fillna(df['LnKF'])
            df = df.drop(columns=[c for c in ['Hatcf_macro', 'LnKF_macro', 'branch_drop'] if c in df.columns])
            df = df.drop(columns=['macro_branch'])
            return df

        # 没有宏观表且没有模型，直接返回
        if 'sdf_fc1' not in self.models or self.models['sdf_fc1'] is None:
            return df

        # 回退：逐行用 sdf_fc1 预测 child 的 Hatcf/LnKF
        model = self.models['sdf_fc1']
        model.eval()
        child_mask = df['branch'] > 0
        with torch.no_grad():
            for idx in df[child_mask].index:
                path = df.loc[idx, 'path']
                firm_id = df.loc[idx, 'ID']
                parent_idx = df[(df['path'] == path) & (df['ID'] == firm_id) & (df['branch'] == 0)].index[0]
                
                x_t = df.loc[parent_idx, 'x']
                x_t1 = df.loc[idx, 'x']
                hatcf_t = df.loc[parent_idx, 'Hatcf']
                lnkf_t = df.loc[parent_idx, 'LnKF']
                
                hatcf_t1, lnkf_t1, M_t1 = self._predict_macro(x_t, x_t1, hatcf_t, lnkf_t)
                
                df.loc[idx, 'Hatcf'] = hatcf_t1
                df.loc[idx, 'LnKF'] = lnkf_t1
                df.loc[idx, 'M'] = M_t1
        
        return df
    
    def fill_policy_value(
        self, 
        df: pd.DataFrame,
        df_macro: pd.DataFrame = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        使用 Policy&Value 填充决策和价值变量
        
        Args:
            df: firm-level DataFrame
            df_macro: macro-level DataFrame (simulate 模式需要)
        
        Returns:
            - sample 模式: df_firm
            - simulate 模式: (df_firm, df_macro)
        """
        if 'policy_value' not in self.models or self.models['policy_value'] is None:
            if df_macro is not None:
                return df, df_macro
            return df
        
        model = self.models['policy_value']
        model.eval()
        
        df = df.copy()

        # 先仅用 parent 行填充输出
        input_cols = ['b', 'z', 'ETA', 'i', 'x', 'Hatcf', 'LnKF']
        parent_mask = df['branch'] == 0
        X_parent = torch.tensor(df.loc[parent_mask, input_cols].values, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            out_parent = model(X_parent)
        for col, val in [
            ('Q', out_parent.Q),
            ('bp0', out_parent.bp0),
            ('bpI', out_parent.bpI),
            ('P0', out_parent.P0),
            ('PI', out_parent.PI),
            ('Bar_i_cond', out_parent.bar_i_cond),
            ('Bar_i', out_parent.bar_i),
            ('Chi', out_parent.chi),
            ('Bar_z', out_parent.bar_z),
            ('P', out_parent.P),
            ('Phat', out_parent.Phat),
            ('bp', out_parent.bp),
        ]:
            df.loc[parent_mask, col] = val.cpu().numpy()

        # 基于 parent 的 bp 与 child 的 ETA(eta_{t+1}) 更新 child 杠杆
        self._update_child_leverage(df)

        # 更新 child 输入后再跑一次模型获取 child 输出
        child_mask = df['branch'] > 0
        if child_mask.any():
            X_child = torch.tensor(df.loc[child_mask, input_cols].values, device=self.device, dtype=torch.float32)
            with torch.no_grad():
                out_child = model(X_child)
            for col, val in [
                ('Q', out_child.Q),
                ('bp0', out_child.bp0),
                ('bpI', out_child.bpI),
                ('P0', out_child.P0),
                ('PI', out_child.PI),
                ('Bar_i_cond', out_child.bar_i_cond),
                ('Bar_i', out_child.bar_i),
                ('Chi', out_child.chi),
                ('Bar_z', out_child.bar_z),
                ('P', out_child.P),
                ('Phat', out_child.Phat),
                ('bp', out_child.bp),
            ]:
                df.loc[child_mask, col] = val.cpu().numpy()
        
        # simulate 模式：更新宏观数据
        if df_macro is not None:
            df_macro = self._update_macro_from_policy(df, df_macro)
            return df, df_macro
        
        return df
    
    def _update_macro_from_policy(
        self, 
        df: pd.DataFrame, 
        df_macro: pd.DataFrame
    ) -> pd.DataFrame:
        """
        根据 Policy&Value 输出更新宏观数据
        
        C_t = Σ Y_i - Σ I_i - Σ Φ_i
        """
        df_macro = df_macro.copy()
        
        delta = self.config.DELTA
        phi = self.config.PHI
        
        for idx, row in df_macro.iterrows():
            path = row['path']
            
            # 获取该 path 的 parent 行（t 时刻）
            path_df = df[(df['path'] == path) & (df['branch'] == 0)]
            
            if len(path_df) == 0:
                continue
            
            # 计算资源核算
            x = row['x']
            z = path_df['z'].values
            b = path_df['b'].values
            bar_i = path_df['Bar_i'].values if 'Bar_i' in path_df.columns else np.zeros_like(z)
            bar_z = path_df['Bar_z'].values if 'Bar_z' in path_df.columns else np.zeros_like(z)
            i_cost = path_df['i'].values
            
            # 假设 K=1 for simplicity，或从 df 中获取
            K = np.ones_like(z)
            
            # Y = exp(x+z) * K
            Y = np.exp(x + z) * K
            Y_total = Y.sum()
            
            # I = bar_i * K * i - bar_z * K + δ * K
            I = bar_i * K * i_cost - bar_z * K + delta * K
            I_total = I.sum()
            
            # Φ = (1-PHI) * (1 + exp(x+z)) * K * bar_z
            Phi = (1 - phi) * (1 + np.exp(x + z)) * K * bar_z
            Phi_total = Phi.sum()
            
            # C = Y - I - Φ
            C_total = Y_total - I_total - Phi_total
            K_total = K.sum()
            
            # 更新宏观数据
            # comment: 这里要有M
            df_macro.loc[idx, 'Y'] = Y_total
            df_macro.loc[idx, 'C'] = max(C_total, 1e-6)  # 避免负消费
            df_macro.loc[idx, 'K'] = K_total
            df_macro.loc[idx, 'I'] = I_total
            df_macro.loc[idx, 'Phi'] = Phi_total
            df_macro.loc[idx, 'LnK'] = np.log(K_total + 1e-8)
            df_macro.loc[idx, 'Hatc'] = np.log(max(C_total, 1e-6) / (K_total + 1e-8))
            if 'M' in df.columns:
                if row['branch'] == -1:
                    df_macro.loc[idx, 'M'] = 1.0
                else:
                    target_branch = row['branch'] + 1  # firm-child 分支 = macro 分支 + 1
                    m_values = df[(df['path'] == path) & (df['branch'] == target_branch)]['M'].dropna().values
                    df_macro.loc[idx, 'M'] = float(m_values.mean()) if len(m_values) > 0 else None
        
        return df_macro
        
        return df
    
    def _update_child_leverage(self, df: pd.DataFrame):
        """
        根据 parent 的决策更新 child 的杠杆
        
        b_{t+1} = η_{t+1} * bp_t + (1 - η_{t+1}) * b_t
        """
        if 'bp' not in df.columns:
            return
        for path in df['path'].unique():
            for firm_id in df[df['path'] == path]['ID'].unique():
                parent_rows = df[(df['path'] == path) & (df['ID'] == firm_id) & (df['branch'] == 0)]
                child_rows = df[(df['path'] == path) & (df['ID'] == firm_id) & (df['branch'] > 0)]
                if len(parent_rows) == 0:
                    continue
                bp_parent = parent_rows['bp'].values[0]
                b_parent = parent_rows['b'].values[0]
                eta_child = child_rows['ETA'].values
                df.loc[child_rows.index, 'b'] = eta_child * bp_parent + (1 - eta_child) * b_parent
    
    def diagnose(self, df: pd.DataFrame) -> Dict:
        """
        诊断数据质量
        """
        diag = {}
        
        # 基本统计
        diag['n_rows'] = len(df)
        diag['n_paths'] = df['path'].nunique()
        diag['n_firms'] = df['ID'].nunique()
        
        # NaN/Inf 检查
        for col in df.select_dtypes(include=[np.number]).columns:
            n_nan = df[col].isna().sum()
            n_inf = np.isinf(df[col]).sum() if not df[col].isna().all() else 0
            if n_nan > 0 or n_inf > 0:
                diag[f'{col}_nan'] = n_nan
                diag[f'{col}_inf'] = n_inf
        
        # 分布统计
        for col in ['b', 'z', 'x']:
            if col in df.columns:
                diag[f'{col}_mean'] = df[col].mean()
                diag[f'{col}_std'] = df[col].std()
                diag[f'{col}_min'] = df[col].min()
                diag[f'{col}_max'] = df[col].max()
        
        # Triplet 对齐检查
        group_sizes = df.groupby(['path', 'ID']).size()
        expected_size = self.branch_num + 1
        misaligned = (group_sizes != expected_size).sum()
        diag['misaligned_triplets'] = misaligned
        
        return diag
    
    def to_tensor(self, df: pd.DataFrame) -> torch.Tensor:
        """
        将 DataFrame 转换为训练用张量
        
        Returns:
            X: (n_samples, 7) firm-state 张量
        """
        input_cols = ['b', 'z', 'ETA', 'i', 'x', 'Hatcf', 'LnKF']
        X = torch.tensor(df[input_cols].values, device=self.device, dtype=torch.float32)
        return X
    
    def get_triplet_tensors(
        self, 
        df: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将 DataFrame 转换为 triplet 格式的张量
        
        Returns:
            parent: (n_units, 7)
            child0: (n_units, 7)
            child1: (n_units, 7)
        """
        X = self.to_tensor(df)
        n = len(X)
        
        # 按 triplet 拆分
        parent = X[0::self.branch_num + 1]
        children = []
        for k in range(self.branch_num):
            children.append(X[k + 1::self.branch_num + 1])
        
        return parent, *children
