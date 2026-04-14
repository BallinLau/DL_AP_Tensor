"""
Episode 类：训练周期管理

一个 Episode 包含：
1. 数据生成（Sample 或 SimulateTS）
2. FC1 填充
3. Policy/Value 填充
4. 各模块的训练循环
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append('..')
from config import Config, HyperParams, SIMMODEL
from data import Sample, SimulateTS, TensorTable, TensorSimulationOutput
from data.data_utils import compute_quantile_features
from losses import SDFLoss, P0Loss, PILoss, QLoss, FC2Loss
from losses.FC2losspipe import FC2LossPipe, compute_fit_stats_t
from losses.utils import compute_z_penalty, compute_aio_residual, compute_monotonicity_penalty
from losses.sdf_loss import moment_penalty
from data.data_utils import build_sdf_pairs_from_macro_ts
from .gradient_utils import gradient_protection, compute_gradient_norm
from .scheduler import LossWeightScheduler, LearningRateScheduler
from utils.gpu_monitor import GPUMonitor, print_memory_summary


logger = logging.getLogger(__name__)


@dataclass
class FC2SupervisedBundle:
    x_hatc: torch.Tensor
    x_lnk: torch.Tensor
    y: torch.Tensor
    path_ids: torch.Tensor


def convert_tree_fast(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['path_ori'] = df['path']
    df['t_ori'] = df['t']
    df['branch_ori'] = df['branch']

    parent_t = np.where(
        df['branch'] == -1,
        df['t'],
        df['t'] - 1
    )

    state_keys = list(zip(df['path_ori'], parent_t))
    df['path'] = pd.factorize(state_keys)[0].astype('int32')

    df['branch'] = df['branch'].map({-1: 0, 0: 1, 1: 2}).astype('int8')

    df['t'] = np.select(
        [
            df['branch'] == 0,
            df['branch'] == 1,
            df['branch'] == 2
        ],
        ['t', 't+1_0', 't+1_1']
    )

    df = df.sort_values(['path', 'branch'], kind='mergesort')

    return df


def trim_child_only_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop child-only firms (IDs that do not appear in parent branch=0).
    This enforces parent-aligned IDs before fill_df_to_fullN.
    """
    df = df.copy()
    allowed = df[df['branch'] == 0][['path', 'ID']].drop_duplicates()
    allowed['__keep__'] = True
    df = df.merge(allowed, on=['path', 'ID'], how='left')
    df = df[df['__keep__'].fillna(False)].drop(columns='__keep__')
    return df


class Episode:
    """
    训练 Episode
    
    管理单个训练周期的数据生成、填充和训练流程
    """

    def __init__(
        self,
        models: Dict[str, nn.Module],
        optimizers: Dict[str, torch.optim.Optimizer],
        config: type = Config,
        hyperparams: HyperParams = None,
        device: torch.device = None,
        episode_id: int = 0,
        gpu_monitor = None
    ):
        """
        Args:
            models: 模型字典
                - 'sdf_fc1': SDFFC1Combined
                - 'policy_value': PolicyValueModel
                - 'fc2_hatc': FC2HatcModel (可选)
                - 'fc2_lnk': FC2LnkModel (可选)
                - 'fc2': FC2Model 兼容旧接口（可选）
            optimizers: 优化器字典
            config: 配置类
            hyperparams: 超参数
            device: 设备
            episode_id: Episode 编号
            gpu_monitor: GPU 监控器（可选，用于共享监控数据）
        """
        self.models = models
        self.optimizers = optimizers
        self.config = config
        self.hyperparams = hyperparams or HyperParams()
        self.device = device or config.DEVICE
        self.episode_id = episode_id
        
        # GPU 监控器（使用外部传入的或创建新的）
        self.gpu_monitor = gpu_monitor if gpu_monitor is not None else GPUMonitor(self.device, log_interval=10)
        self.gpu_monitor.reset_peak_stats()
        
        # 损失函数
        self.loss_fns = self._init_loss_functions()
        
        # 损失权重调度器
        self.weight_scheduler = self._init_weight_scheduler()
        
        # 学习率调度器
        self.lr_schedulers = self._init_lr_schedulers()
        
        # 数据
        self.df = None
        self.df_macro = None
        self.df_sdf = None
        self.macro_source_df: Optional[pd.DataFrame] = None
        self.tensor_firm: Optional[TensorTable] = None
        self.tensor_macro: Optional[TensorTable] = None
        self.tensor_sdf: Optional[TensorTable] = None
        
        # 训练状态
        self.step_count = 0
        self.loss_history = {}
        self.add_FC1loss = False
        self.train_mode = '2time'
        self._latest_sdf_diag = {}
        self._latest_sdf_terms = {}
        self._latest_p0_terms = {}
        self._latest_pi_terms = {}
        self._latest_q_terms = {}
        self._latest_fc2_diag = {}
        self._latest_fc2_fit_df: Optional[pd.DataFrame] = None
        self._latest_fc2_outer_df: Optional[pd.DataFrame] = None
        self._latest_fc2_current_macro_panel_df: Optional[pd.DataFrame] = None
        self._latest_fc2_outer_macro_panel_df: Optional[pd.DataFrame] = None
        self._sdf_base_lr_backup = None
        self._current_epoch_idx = 0
        self._q_only_stage = False
        self._q_refresh_stage = False
        self._bp_only_stage = False
        self._value_only_stage = False
        self._pvbp_only_stage = False
        self._fc1_teacher_forcing_stage = False
        self._policy_q_freeze_active = False
        self._policy_value_grad_backup = {}
        self._policy_bp_freeze_active = False
        self._policy_bp_grad_backup = {}
        self._policy_value_only_freeze_active = False
        self._policy_pvbp_freeze_active = False
        self._sdf_fc1_teacher_freeze_active = False
        self._sdf_fc1_grad_backup = {}
    
    def _init_loss_functions(self) -> Dict:
        """
        初始化损失函数
        """
        return {
            'sdf': SDFLoss(),
            'p0': P0Loss(),
            'pi': PILoss(),
            'q': QLoss(),
            'fc2': FC2Loss() if self._has_fc2_models() else None
        }

    def _get_fc2_models(self) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
        hatc_model = self.models.get('fc2_hatc')
        lnk_model = self.models.get('fc2_lnk')
        if hatc_model is not None and lnk_model is not None:
            return hatc_model, lnk_model
        return None, None

    def _has_fc2_models(self) -> bool:
        hatc_model, lnk_model = self._get_fc2_models()
        return hatc_model is not None and lnk_model is not None

    def _get_fc2_model_keys(self) -> List[str]:
        if 'fc2_hatc' in self.models and 'fc2_lnk' in self.models:
            return ['fc2_hatc', 'fc2_lnk']
        return []

    def _get_fc2_optimizer_keys(self) -> List[str]:
        keys = []
        if 'fc2_hatc' in self.optimizers and 'fc2_lnk' in self.optimizers:
            keys.extend(['fc2_hatc', 'fc2_lnk'])
        return keys

    def _get_fc2_scheduler_keys(self) -> List[str]:
        keys = []
        if 'fc2_hatc' in self.lr_schedulers and 'fc2_lnk' in self.lr_schedulers:
            keys.extend(['fc2_hatc', 'fc2_lnk'])
        return keys

    def _get_fc2_lr(self) -> float:
        for key in self._get_fc2_scheduler_keys():
            scheduler = self.lr_schedulers.get(key)
            if scheduler is not None:
                return float(scheduler.get_lr())
        return 0.0

    def _fit_fc2_hatc_x_baseline(
        self,
        sup_source: Optional[Any],
    ) -> Dict[str, float]:
        hatc_model, _ = self._get_fc2_models()
        if hatc_model is None or sup_source is None:
            return {}
        if not hasattr(hatc_model, "fit_x_baseline"):
            return {}
        tensors = self._extract_fc2_supervised_tensors(sup_source)
        if tensors is None:
            return {}
        x_hatc_all, _, y_all = tensors
        if x_hatc_all.shape[0] == 0:
            return {}
        hatc_target = y_all[:, 1:2]
        hatc_model.fit_x_baseline(x_hatc_all, hatc_target)
        diag: Dict[str, float] = {}
        coef = getattr(hatc_model, "x_baseline_coef", None)
        if coef is not None:
            coef_flat = coef.detach().cpu().reshape(-1).tolist()
            for idx, val in enumerate(coef_flat):
                diag[f"fc2_hatc_x_baseline_coef_{idx}"] = float(val)
        x_mean = getattr(hatc_model, "x_baseline_mean", None)
        if x_mean is not None:
            diag["fc2_hatc_x_baseline_mean"] = float(x_mean.detach().cpu().reshape(-1)[0].item())
        x_scale = getattr(hatc_model, "x_baseline_scale", None)
        if x_scale is not None:
            diag["fc2_hatc_x_baseline_scale"] = float(x_scale.detach().cpu().reshape(-1)[0].item())
        fitted = getattr(hatc_model, "x_baseline_fitted", None)
        if fitted is not None:
            diag["fc2_hatc_x_baseline_fitted"] = float(bool(fitted.item()))
        return diag

    def _extract_fc2_supervised_tensors(
        self,
        sup_source: Optional[Any],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if sup_source is None:
            return None
        if isinstance(sup_source, FC2SupervisedBundle):
            return sup_source.x_hatc, sup_source.x_lnk, sup_source.y
        if isinstance(sup_source, TensorDataset):
            if len(sup_source.tensors) < 3:
                return None
            return sup_source.tensors[0], sup_source.tensors[1], sup_source.tensors[2]
        return None

    def _fc2_bundle_to_dataset(self, bundle: FC2SupervisedBundle) -> TensorDataset:
        return TensorDataset(bundle.x_hatc, bundle.x_lnk, bundle.y)

    def _fc2_index_bundle(
        self,
        bundle: FC2SupervisedBundle,
        index: torch.Tensor,
    ) -> FC2SupervisedBundle:
        return FC2SupervisedBundle(
            x_hatc=bundle.x_hatc[index].detach(),
            x_lnk=bundle.x_lnk[index].detach(),
            y=bundle.y[index].detach(),
            path_ids=bundle.path_ids[index].detach(),
        )

    def _split_fc2_supervised_bundle(
        self,
        bundle: FC2SupervisedBundle,
    ) -> Tuple[FC2SupervisedBundle, FC2SupervisedBundle]:
        val_frac = float(getattr(self.hyperparams, 'fc2_pretrain_val_frac', 0.2))
        split_seed = int(getattr(self.hyperparams, 'fc2_pretrain_split_seed', 42))
        path_ids = bundle.path_ids.detach().cpu().numpy().astype(np.int64)
        uniq_paths = np.unique(path_ids)
        if uniq_paths.shape[0] <= 1 or val_frac <= 0.0:
            full_index = torch.arange(bundle.y.shape[0], device=bundle.y.device, dtype=torch.long)
            full_bundle = self._fc2_index_bundle(bundle, full_index)
            return full_bundle, full_bundle

        rng = np.random.default_rng(split_seed)
        perm = rng.permutation(uniq_paths)
        n_val = min(max(1, int(round(perm.shape[0] * val_frac))), perm.shape[0] - 1)
        val_paths = perm[:n_val]
        train_paths = perm[n_val:]

        path_tensor = bundle.path_ids
        train_mask = torch.zeros_like(path_tensor, dtype=torch.bool)
        val_mask = torch.zeros_like(path_tensor, dtype=torch.bool)
        for path in train_paths.tolist():
            train_mask |= (path_tensor == int(path))
        for path in val_paths.tolist():
            val_mask |= (path_tensor == int(path))

        train_index = torch.nonzero(train_mask, as_tuple=False).reshape(-1)
        val_index = torch.nonzero(val_mask, as_tuple=False).reshape(-1)
        if train_index.numel() == 0 or val_index.numel() == 0:
            full_index = torch.arange(bundle.y.shape[0], device=bundle.y.device, dtype=torch.long)
            full_bundle = self._fc2_index_bundle(bundle, full_index)
            return full_bundle, full_bundle

        return self._fc2_index_bundle(bundle, train_index), self._fc2_index_bundle(bundle, val_index)

    def _expand_train_model_keys(self, train_modules: List[str]) -> List[str]:
        expanded: List[str] = []
        for name in train_modules:
            if name == 'fc2':
                expanded.extend(self._get_fc2_model_keys())
            elif name in self.models:
                expanded.append(name)
        return list(dict.fromkeys(expanded))

    @staticmethod
    def _macro_fit_stats_np(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Rich consistency stats for aggregate law diagnostics."""
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() < 2:
            return {}
        yt = y_true[mask]
        yp = y_pred[mask]
        resid = yt - yp
        yt_mean = float(yt.mean())
        yp_mean = float(yp.mean())
        y_std = float(np.std(yt))
        p_std = float(np.std(yp))
        sst = float(np.sum((yt - yt_mean) ** 2))
        sse = float(np.sum(resid ** 2))
        out: Dict[str, float] = {
            'n_obs': float(mask.sum()),
            'mean_true': yt_mean,
            'mean_pred': yp_mean,
            'mean_resid': float(resid.mean()),
            'mae_resid': float(np.mean(np.abs(resid))),
            'rmse_resid': float(np.sqrt(np.mean(resid ** 2))),
            'std_true': y_std,
            'std_pred': p_std,
            'std_ratio': float(p_std / y_std) if y_std > 1e-12 else float('nan'),
            'r2': float('nan') if sst <= 1e-12 else float(1.0 - sse / sst),
        }
        if y_std > 1e-12 and p_std > 1e-12:
            out['corr'] = float(np.corrcoef(yp, yt)[0, 1])
        else:
            out['corr'] = float('nan')
        x_centered = yp - yp_mean
        denom = float(np.sum(x_centered ** 2))
        if denom > 1e-12:
            slope = float(np.sum(x_centered * (yt - yt_mean)) / denom)
            intercept = float(yt_mean - slope * yp_mean)
        else:
            slope = float('nan')
            intercept = float('nan')
        out['slope'] = slope
        out['intercept'] = intercept
        return out

    @staticmethod
    def _macro_fit_stats_t(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
        mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
        if int(mask.sum().item()) < 2:
            return {}
        yt = y_true[mask]
        yp = y_pred[mask]
        resid = yt - yp
        yt_mean_t = yt.mean()
        yp_mean_t = yp.mean()
        y_centered = yt - yt_mean_t
        p_centered = yp - yp_mean_t
        y_std_t = yt.std(unbiased=False)
        p_std_t = yp.std(unbiased=False)
        sst = (y_centered ** 2).sum()
        sse = (resid ** 2).sum()
        out: Dict[str, float] = {
            'n_obs': float(mask.sum().item()),
            'mean_true': float(yt_mean_t.item()),
            'mean_pred': float(yp_mean_t.item()),
            'mean_resid': float(resid.mean().item()),
            'mae_resid': float(resid.abs().mean().item()),
            'rmse_resid': float(torch.sqrt((resid ** 2).mean()).item()),
            'std_true': float(y_std_t.item()),
            'std_pred': float(p_std_t.item()),
            'std_ratio': float((p_std_t / y_std_t).item()) if float(y_std_t.item()) > 1e-12 else float('nan'),
            'r2': float('nan') if float(sst.item()) <= 1e-12 else float((1.0 - sse / sst).item()),
        }
        if float(y_std_t.item()) > 1e-12 and float(p_std_t.item()) > 1e-12:
            corr = (y_centered * p_centered).mean() / (y_std_t * p_std_t)
            out['corr'] = float(corr.item())
        else:
            out['corr'] = float('nan')
        denom = (p_centered ** 2).sum()
        if float(denom.item()) > 1e-12:
            slope = (p_centered * y_centered).sum() / denom
            intercept = yt_mean_t - slope * yp_mean_t
            out['slope'] = float(slope.item())
            out['intercept'] = float(intercept.item())
        else:
            out['slope'] = float('nan')
            out['intercept'] = float('nan')
        return out

    @staticmethod
    def _binned_response_loss_t(
        x: torch.Tensor,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        n_bins: int,
    ) -> torch.Tensor:
        """
        Compare conditional responses E[y | x-bin] between truth and prediction.
        This targets shape alignment of the aggregate law, not just pointwise MSE.
        """
        mask = torch.isfinite(x) & torch.isfinite(y_true) & torch.isfinite(y_pred)
        if int(mask.sum().item()) < max(4, n_bins):
            return torch.zeros((), device=x.device, dtype=x.dtype)
        x_use = x[mask]
        yt = y_true[mask]
        yp = y_pred[mask]
        x_min = x_use.min()
        x_max = x_use.max()
        if float((x_max - x_min).abs().item()) < 1e-8:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        edges = torch.linspace(
            float(x_min.item()),
            float(x_max.item()),
            steps=max(2, int(n_bins)) + 1,
            device=x.device,
            dtype=x.dtype,
        )
        bin_idx = torch.bucketize(x_use, edges[1:-1], right=False)
        losses = []
        for b in range(max(2, int(n_bins))):
            mk = bin_idx == b
            if int(mk.sum().item()) == 0:
                continue
            losses.append((yp[mk].mean() - yt[mk].mean()).pow(2))
        if not losses:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        return torch.stack(losses).mean()

    @staticmethod
    def _macro_forecast_r2(df_macro: Optional[pd.DataFrame]) -> Dict[str, float]:
        """
        计算宏观 law consistency / forecast fit 诊断。
        兼容列名：realized(Hatc/LnK), forecast(hatcf/lnkf 或 Hatcf/LnKF)。
        """
        if df_macro is None or df_macro.empty:
            return {}

        def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        use_df = df_macro
        hatc_true_col = _pick_col(use_df, ['Hatc', 'hatc'])
        lnk_true_col = _pick_col(use_df, ['LnK', 'lnk'])
        hatc_pred_col = _pick_col(use_df, ['hatcf', 'Hatcf'])
        lnk_pred_col = _pick_col(use_df, ['lnkf', 'LnKF'])

        if hatc_true_col is None or lnk_true_col is None:
            return {}

        out: Dict[str, float] = {'n_t': float(len(use_df)), 'n_obs': float(len(use_df))}
        if hatc_pred_col is not None:
            hatc_stats = Episode._macro_fit_stats_np(
                use_df[hatc_true_col].to_numpy(),
                use_df[hatc_pred_col].to_numpy()
            )
            out.update({f"{k}_hatc": v for k, v in hatc_stats.items() if k != 'n_obs'})
            if 'branch' in use_df.columns:
                by_branch_hatc: Dict[str, float] = {}
                for br, grp in use_df.groupby('branch'):
                    key = f"{int(br)}" if float(br).is_integer() else str(br)
                    by_branch_hatc[key] = Episode._macro_fit_stats_np(
                        grp[hatc_true_col].to_numpy(),
                        grp[hatc_pred_col].to_numpy()
                    ).get('r2', float('nan'))
                out['r2_hatc_by_branch'] = by_branch_hatc
                branch_mask = use_df['branch'].isin([0, 1])
                if branch_mask.any():
                    hatc_b01 = Episode._macro_fit_stats_np(
                        use_df.loc[branch_mask, hatc_true_col].to_numpy(),
                        use_df.loc[branch_mask, hatc_pred_col].to_numpy()
                    )
                    out.update({f"{k}_hatc_branch01": v for k, v in hatc_b01.items() if k != 'n_obs'})
        if lnk_pred_col is not None:
            lnk_stats = Episode._macro_fit_stats_np(
                use_df[lnk_true_col].to_numpy(),
                use_df[lnk_pred_col].to_numpy()
            )
            out.update({f"{k}_lnk": v for k, v in lnk_stats.items() if k != 'n_obs'})
            if 'branch' in use_df.columns:
                by_branch_lnk: Dict[str, float] = {}
                for br, grp in use_df.groupby('branch'):
                    key = f"{int(br)}" if float(br).is_integer() else str(br)
                    by_branch_lnk[key] = Episode._macro_fit_stats_np(
                        grp[lnk_true_col].to_numpy(),
                        grp[lnk_pred_col].to_numpy()
                    ).get('r2', float('nan'))
                out['r2_lnk_by_branch'] = by_branch_lnk
                branch_mask = use_df['branch'].isin([0, 1])
                if branch_mask.any():
                    lnk_b01 = Episode._macro_fit_stats_np(
                        use_df.loc[branch_mask, lnk_true_col].to_numpy(),
                        use_df.loc[branch_mask, lnk_pred_col].to_numpy()
                    )
                    out.update({f"{k}_lnk_branch01": v for k, v in lnk_b01.items() if k != 'n_obs'})
        if 'branch' in use_df.columns:
            out['n_obs_by_branch'] = {
                (f"{int(br)}" if float(br).is_integer() else str(br)): float(len(grp))
                for br, grp in use_df.groupby('branch')
            }
        return out

    def _macro_consumption_diagnostics(
        self,
        macro_panel: Optional[pd.DataFrame | TensorTable],
        prefix: str,
    ) -> Dict[str, float]:
        if macro_panel is None:
            return {}
        if isinstance(macro_panel, TensorTable):
            df_macro = self._table_to_dataframe(macro_panel)
        else:
            df_macro = macro_panel.copy()
        if df_macro is None or df_macro.empty:
            return {}
        df_macro = self._select_fc2_macro_window_df(df_macro)
        if df_macro is None or df_macro.empty:
            return {}

        required_cols = {
            'C_raw',
            'C_firmclip',
            'C_aggclip',
            'Hatc_firmclip',
            'Hatc_aggclip',
            'C_raw_over_K',
            'C_pos_mass',
            'C_neg_mass',
            'neg_c_mass_share',
            'feasible_raw',
        }
        if not required_cols.issubset(df_macro.columns):
            return {}

        use_df = df_macro

        out: Dict[str, float] = {
            f'{prefix}_n_rows': float(len(use_df)),
            f'{prefix}_feasible_path_share': float(pd.to_numeric(use_df['feasible_raw'], errors='coerce').mean()),
            f'{prefix}_mean_c_raw_over_k': float(pd.to_numeric(use_df['C_raw_over_K'], errors='coerce').mean()),
            f'{prefix}_mean_neg_c_mass_share': float(pd.to_numeric(use_df['neg_c_mass_share'], errors='coerce').mean()),
            f'{prefix}_p90_neg_c_mass_share': float(pd.to_numeric(use_df['neg_c_mass_share'], errors='coerce').quantile(0.9)),
            f'{prefix}_mean_hatc_firmclip': float(pd.to_numeric(use_df['Hatc_firmclip'], errors='coerce').mean()),
            f'{prefix}_mean_hatc_aggclip': float(pd.to_numeric(use_df['Hatc_aggclip'], errors='coerce').mean()),
            f'{prefix}_mean_hatc_clip_gap': float(
                (pd.to_numeric(use_df['Hatc_firmclip'], errors='coerce') - pd.to_numeric(use_df['Hatc_aggclip'], errors='coerce')).mean()
            ),
            f'{prefix}_mean_c_raw': float(pd.to_numeric(use_df['C_raw'], errors='coerce').mean()),
            f'{prefix}_mean_c_firmclip': float(pd.to_numeric(use_df['C_firmclip'], errors='coerce').mean()),
            f'{prefix}_mean_c_aggclip': float(pd.to_numeric(use_df['C_aggclip'], errors='coerce').mean()),
        }
        return out

    def _select_fc2_macro_window_df(
        self,
        macro_panel: Optional[pd.DataFrame | TensorTable],
    ) -> Optional[pd.DataFrame]:
        if macro_panel is None:
            return None
        if isinstance(macro_panel, TensorTable):
            df_macro = self._table_to_dataframe(macro_panel)
        else:
            df_macro = macro_panel.copy()
        if df_macro is None or df_macro.empty:
            return None
        required = {'path', 't', 'branch'}
        if not required.issubset(df_macro.columns):
            return df_macro

        selected: List[pd.DataFrame] = []
        use_df = df_macro.copy()
        use_df['path'] = pd.to_numeric(use_df['path'], errors='coerce').astype('Int64')
        use_df['t'] = pd.to_numeric(use_df['t'], errors='coerce').astype('Int64')
        use_df['branch'] = pd.to_numeric(use_df['branch'], errors='coerce').astype('Int64')

        for _, grp in use_df.groupby('path', sort=True):
            parent = grp[grp['branch'] == -1]
            if parent.empty:
                continue
            parent_t = int(parent['t'].max())
            child_t = parent_t + 1
            keep = grp[
                ((grp['t'] == parent_t) & (grp['branch'] == -1))
                | ((grp['t'] == child_t) & (grp['branch'].isin([0, 1])))
            ]
            if not keep.empty:
                selected.append(keep)

        if not selected:
            return use_df.sort_values(['path', 't', 'branch']).reset_index(drop=True)
        out = pd.concat(selected, ignore_index=True)
        return out.sort_values(['path', 't', 'branch']).reset_index(drop=True)

    @staticmethod
    def _macro_forecast_r2_tensor(macro_table: Optional[TensorTable]) -> Dict[str, float]:
        """
        tensor 版本宏观 law consistency / forecast fit 诊断。
        """
        if macro_table is None or macro_table.data.numel() == 0:
            return {}
        col = {name: i for i, name in enumerate(macro_table.columns)}
        for k in ['Hatc', 'LnK', 'hatcf', 'lnkf']:
            if k not in col:
                return {}
        data = macro_table.data
        y_hatc = data[:, col['Hatc']]
        y_lnk = data[:, col['LnK']]
        p_hatc = data[:, col['hatcf']]
        p_lnk = data[:, col['lnkf']]

        out: Dict[str, Any] = {
            'n_t': float(data.shape[0]),
            'n_obs': float(data.shape[0]),
        }
        hatc_stats = Episode._macro_fit_stats_t(y_hatc, p_hatc)
        lnk_stats = Episode._macro_fit_stats_t(y_lnk, p_lnk)
        out.update({f"{k}_hatc": v for k, v in hatc_stats.items() if k != 'n_obs'})
        out.update({f"{k}_lnk": v for k, v in lnk_stats.items() if k != 'n_obs'})
        if 'branch' in col:
            br = data[:, col['branch']].long()
            br_vals = torch.unique(br)
            r2_hatc_by_branch: Dict[str, float] = {}
            r2_lnk_by_branch: Dict[str, float] = {}
            n_obs_by_branch: Dict[str, float] = {}
            for b in br_vals:
                mk = br == b
                key = f"{int(b.item())}"
                r2_hatc_by_branch[key] = Episode._macro_fit_stats_t(y_hatc[mk], p_hatc[mk]).get('r2', float('nan'))
                r2_lnk_by_branch[key] = Episode._macro_fit_stats_t(y_lnk[mk], p_lnk[mk]).get('r2', float('nan'))
                n_obs_by_branch[key] = float(mk.sum().item())
            out['r2_hatc_by_branch'] = r2_hatc_by_branch
            out['r2_lnk_by_branch'] = r2_lnk_by_branch
            out['n_obs_by_branch'] = n_obs_by_branch
            mk01 = (br == 0) | (br == 1)
            if int(mk01.sum().item()) >= 2:
                hatc_b01 = Episode._macro_fit_stats_t(y_hatc[mk01], p_hatc[mk01])
                lnk_b01 = Episode._macro_fit_stats_t(y_lnk[mk01], p_lnk[mk01])
                out.update({f"{k}_hatc_branch01": v for k, v in hatc_b01.items() if k != 'n_obs'})
                out.update({f"{k}_lnk_branch01": v for k, v in lnk_b01.items() if k != 'n_obs'})
        return out

    def _use_tensor_pipeline(self) -> bool:
        return bool(getattr(self.hyperparams, "use_tensor_pipeline", True))

    def _table_to_dataframe(self, table: Optional[TensorTable]) -> Optional[pd.DataFrame]:
        if table is None:
            return None
        df = table.to_dataframe()
        for col in ['path', 't', 'branch', 'ID', 'firm', 'n_firms']:
            if col in df.columns:
                df[col] = np.rint(df[col]).astype(np.int64)
        if 'ID' in df.columns:
            df['ID'] = df['ID'].astype(str)
        return df

    @staticmethod
    def _encode_int_keys(
        path: torch.Tensor,
        ident: torch.Tensor,
        t: torch.Tensor,
        max_id: Optional[int] = None,
        max_t: Optional[int] = None
    ) -> torch.Tensor:
        """
        Encode (path, id, t) tuples into unique int64 keys.
        """
        path = path.long()
        ident = ident.long()
        t = t.long()
        if max_id is None:
            max_id = int(ident.max().item()) + 1 if ident.numel() > 0 else 1
        if max_t is None:
            max_t = int(t.max().item()) + 2 if t.numel() > 0 else 2
        return ((path * max_id) + ident) * max_t + t

    @staticmethod
    def _match_keys(parent_keys: torch.Tensor, child_keys: torch.Tensor, child_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Match parent keys to child rows.
        Returns:
            valid: (N_parent,) bool
            matched_child_idx: (N_parent,) long
        """
        if child_keys.numel() == 0:
            n = parent_keys.shape[0]
            return torch.zeros(n, dtype=torch.bool, device=parent_keys.device), torch.zeros(
                n, dtype=torch.long, device=parent_keys.device
            )
        order = torch.argsort(child_keys)
        sorted_keys = child_keys[order]
        sorted_child_idx = child_indices[order]
        pos = torch.searchsorted(sorted_keys, parent_keys)
        pos_clamped = pos.clamp(max=max(sorted_keys.numel() - 1, 0))
        valid = (pos < sorted_keys.numel()) & (sorted_keys[pos_clamped] == parent_keys)
        return valid, sorted_child_idx[pos_clamped]

    def _build_batches_from_parent_children(
        self,
        parent: torch.Tensor,
        children: List[torch.Tensor],
        batch_size: int,
        eta_resample: bool = True
    ) -> List[Dict[str, torch.Tensor]]:
        if parent is None or parent.numel() == 0:
            return []
        n_units = int(parent.shape[0])
        if n_units == 0:
            return []

        n_batches = (n_units + batch_size - 1) // batch_size
        indices = torch.arange(n_units, device=parent.device)

        # eta 稀疏时，对 Policy/Value 批次进行条件重采样，增强 eta=1 信号。
        resample_enabled = bool(getattr(self.hyperparams, "pv_eta_resample_enabled", True))
        if eta_resample and resample_enabled and n_units > 1 and len(children) > 0:
            eta_child_stack = torch.stack([c[:, 2:3] for c in children], dim=1)  # (B, N, 1)
            active_mask = (eta_child_stack.max(dim=1).values.squeeze(-1) > 0.5)
            active_idx = torch.where(active_mask)[0]
            inactive_idx = torch.where(~active_mask)[0]
            if active_idx.numel() > 0 and inactive_idx.numel() > 0:
                target_active_share = float(getattr(self.hyperparams, "pv_eta_resample_active_share", 0.25))
                target_active_share = min(max(target_active_share, 1e-3), 1.0 - 1e-3)
                n_active = int(round(n_units * target_active_share))
                n_active = min(max(1, n_active), n_units - 1)
                n_inactive = n_units - n_active
                active_pick = active_idx[torch.randint(0, active_idx.numel(), (n_active,), device=active_idx.device)]
                inactive_pick = inactive_idx[
                    torch.randint(0, inactive_idx.numel(), (n_inactive,), device=inactive_idx.device)
                ]
                indices = torch.cat([active_pick, inactive_pick], dim=0)
                indices = indices[torch.randperm(indices.numel(), device=indices.device)]
            else:
                indices = indices[torch.randperm(n_units, device=indices.device)]
        else:
            indices = indices[torch.randperm(n_units, device=indices.device)]

        batches = []
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_units)
            idx = indices[start:end]
            batch = {
                'parent': parent[idx],
                'children': [c[idx] for c in children],
                'child0': children[0][idx] if len(children) > 0 else None,
                'child1': children[1][idx] if len(children) > 1 else None
            }
            batches.append(batch)
        return batches

    def _create_firm_batches_from_tensor(
        self,
        table: TensorTable,
        batch_size: int = 1024,
        n_branches: int = 2,
        eta_resample: bool = True
    ) -> List[Dict[str, torch.Tensor]]:
        """
        从 firm-level TensorTable 创建训练批次（兼容 Sample/SimulateTS tensor 输出）。
        """
        data = table.data
        if data.numel() == 0:
            return []
        col = {name: i for i, name in enumerate(table.columns)}
        required = ['path', 'branch', 'b', 'z', 'ETA', 'i', 'x', 'Hatcf', 'LnKF']
        missing = [k for k in required if k not in col]
        if missing:
            raise ValueError(f"Tensor firm table missing columns: {missing}")
        id_col = 'firm' if 'firm' in col else ('ID' if 'ID' in col else None)
        if id_col is None:
            raise ValueError("Tensor firm table missing id column ('firm' or 'ID').")

        path = data[:, col['path']].long()
        ident = data[:, col[id_col]].long()
        branch = data[:, col['branch']].long()
        has_t = 't' in col
        t = data[:, col['t']].long() if has_t else torch.zeros_like(path)
        parent_branch = -1 if int(branch.min().item()) < 0 else 0
        key_max_id = int(ident.max().item()) + 1 if ident.numel() > 0 else 1
        key_max_t = int(t.max().item()) + 2 if t.numel() > 0 else 2

        parent_mask = (branch == parent_branch)
        parent_idx = torch.nonzero(parent_mask, as_tuple=False).squeeze(-1)
        if parent_idx.numel() == 0:
            return []
        parent_path = path[parent_idx]
        parent_id = ident[parent_idx]
        parent_t = t[parent_idx] if (has_t and parent_branch < 0) else torch.zeros_like(parent_path)
        parent_key = self._encode_int_keys(parent_path, parent_id, parent_t, max_id=key_max_id, max_t=key_max_t)

        active = torch.ones(parent_idx.shape[0], dtype=torch.bool, device=data.device)
        child_selected: List[torch.Tensor] = []
        for k in range(n_branches):
            child_branch = k if parent_branch < 0 else (k + 1)
            child_mask = (branch == child_branch)
            child_idx = torch.nonzero(child_mask, as_tuple=False).squeeze(-1)
            child_path = path[child_idx]
            child_id = ident[child_idx]
            if has_t and parent_branch < 0:
                child_t_parent = t[child_idx] - 1
            else:
                child_t_parent = torch.zeros_like(child_path)
            child_key = self._encode_int_keys(
                child_path, child_id, child_t_parent, max_id=key_max_id, max_t=key_max_t
            )
            valid, matched_idx = self._match_keys(parent_key, child_key, child_idx)
            active = active & valid
            child_selected.append(matched_idx)

        if active.sum().item() == 0:
            return []

        parent_take = parent_idx[active]
        child_take = [idx[active] for idx in child_selected]

        feat_names = ['b', 'z', 'ETA', 'i', 'x', 'Hatcf', 'LnKF']
        feat_idx = [col[nm] for nm in feat_names]
        parent_feat = data[parent_take][:, feat_idx]
        children_feat = [data[idx][:, feat_idx] for idx in child_take]
        if 'M' in col:
            parent_m = data[parent_take][:, col['M']:col['M'] + 1]
            children_m = [data[idx][:, col['M']:col['M'] + 1] for idx in child_take]
        else:
            parent_m = torch.ones(parent_feat.shape[0], 1, device=data.device, dtype=data.dtype)
            children_m = [torch.ones(c.shape[0], 1, device=data.device, dtype=data.dtype) for c in children_feat]
        parent = torch.cat([parent_feat, parent_m], dim=1).to(torch.float32)
        children = [torch.cat([c, m], dim=1).to(torch.float32) for c, m in zip(children_feat, children_m)]

        return self._build_batches_from_parent_children(
            parent=parent,
            children=children,
            batch_size=batch_size,
            eta_resample=eta_resample
        )

    def _build_sdf_pairs_from_macro_tensor(self, macro_table: TensorTable) -> TensorTable:
        """
        从 SimulateTS 的 macro TensorTable 构造 SDF/FC1 跨期配对表（tensor 版本）。
        """
        data = macro_table.data
        col = {name: i for i, name in enumerate(macro_table.columns)}
        required = ['path', 't', 'branch', 'x', 'hatcf', 'lnkf', 'Hatc', 'LnK']
        missing = [k for k in required if k not in col]
        if missing:
            raise ValueError(f"Tensor macro table missing columns: {missing}")
        path = data[:, col['path']].long()
        t = data[:, col['t']].long()
        branch = data[:, col['branch']].long()
        parent_mask = (branch == -1)
        child_mask = (branch >= 0)

        parent_idx = torch.nonzero(parent_mask, as_tuple=False).squeeze(-1)
        child_idx = torch.nonzero(child_mask, as_tuple=False).squeeze(-1)
        key_max_t = int(t.max().item()) + 2 if t.numel() > 0 else 2
        parent_key = self._encode_int_keys(
            path[parent_idx], torch.zeros_like(path[parent_idx]), t[parent_idx], max_id=1, max_t=key_max_t
        )
        child_parent_t = t[child_idx] - 1
        child_key = self._encode_int_keys(
            path[child_idx], torch.zeros_like(path[child_idx]), child_parent_t, max_id=1, max_t=key_max_t
        )
        valid, matched_parent_idx = self._match_keys(child_key, parent_key, parent_idx)
        if valid.sum().item() == 0:
            empty = torch.empty((0, 11), device=self.device, dtype=torch.float32)
            return TensorTable(
                data=empty,
                columns=['path', 't', 'branch', 'x_t', 'x_t1', 'Hatcf_t', 'LnKF_t', 'Hatc_t', 'LnK_t', 'Hatc_t1', 'LnK_t1']
            )

        child_take = child_idx[valid]
        parent_take = matched_parent_idx[valid]

        out = torch.stack(
            [
                data[child_take, col['path']],    # path
                data[child_take, col['t']],       # t (child t)
                data[child_take, col['branch']],  # child branch
                data[parent_take, col['x']],      # x_t
                data[child_take, col['x']],       # x_t1
                data[parent_take, col['hatcf']],  # Hatcf_t
                data[parent_take, col['lnkf']],   # LnKF_t
                data[parent_take, col['Hatc']],   # Hatc_t
                data[parent_take, col['LnK']],    # LnK_t
                data[child_take, col['Hatc']],    # Hatc_t1
                data[child_take, col['LnK']],     # LnK_t1
            ],
            dim=1
        ).to(torch.float32)
        return TensorTable(
            data=out,
            columns=['path', 't', 'branch', 'x_t', 'x_t1', 'Hatcf_t', 'LnKF_t', 'Hatc_t', 'LnK_t', 'Hatc_t1', 'LnK_t1']
        )

    def _create_sdf_batches_from_macro_tensor(
        self,
        sdf_table: TensorTable,
        batch_size: int = 1024,
        n_branches: int = 2
    ) -> List[Dict[str, torch.Tensor]]:
        """
        从宏观跨期 TensorTable 创建 SDF 批次。
        """
        data = sdf_table.data
        if data.numel() == 0:
            return []
        col = {name: i for i, name in enumerate(sdf_table.columns)}
        needed = ['path', 'branch', 'x_t', 'x_t1', 'Hatcf_t', 'LnKF_t']
        if any(k not in col for k in needed):
            raise ValueError(f"SDF tensor table columns mismatch, need {needed}")

        if self.add_FC1loss and self.train_mode != '2time' and 't' in col:
            keep = data[:, col['t']] > 2.0
            data = data[keep]
            if data.numel() == 0:
                return []

        path = data[:, col['path']].long()
        branch = data[:, col['branch']].long()
        if self.add_FC1loss and 't' in col:
            t = data[:, col['t']].long()
            key = self._encode_int_keys(path, torch.zeros_like(path), t)
        else:
            key = self._encode_int_keys(path, torch.zeros_like(path), torch.zeros_like(path))

        unique_keys = torch.unique(key)
        parent_rows: List[torch.Tensor] = []
        child_rows: List[List[torch.Tensor]] = [[] for _ in range(n_branches)]

        for gk in unique_keys:
            gm = key == gk
            group = data[gm]
            g_branch = group[:, col['branch']].long()
            ok = True
            group_children = []
            for k in range(n_branches):
                mk = g_branch == k
                if mk.sum().item() == 0:
                    ok = False
                    break
                group_children.append(group[mk][0])
            if not ok:
                continue

            x_t = group_children[0][col['x_t']]
            hatcf_t = group_children[0][col['Hatcf_t']]
            lnkf_t = group_children[0][col['LnKF_t']]
            if self.add_FC1loss:
                if 'Hatc_t' not in col or 'LnK_t' not in col:
                    continue
                hatc_t = group_children[0][col['Hatc_t']]
                lnk_t = group_children[0][col['LnK_t']]
                p = torch.stack([
                    torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device),
                    torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device),
                    x_t, hatcf_t, lnkf_t, hatc_t, lnk_t
                ])
            else:
                p = torch.stack([
                    torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device),
                    torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device),
                    x_t, hatcf_t, lnkf_t
                ])
            parent_rows.append(p)

            for k in range(n_branches):
                x_t1 = group_children[k][col['x_t1']]
                if self.add_FC1loss:
                    if 'Hatc_t1' not in col or 'LnK_t1' not in col:
                        ok = False
                        break
                    hatc_t1 = group_children[k][col['Hatc_t1']]
                    lnk_t1 = group_children[k][col['LnK_t1']]
                    c = torch.stack([
                        torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device),
                        torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device),
                        x_t1,
                        torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device),
                        hatc_t1, lnk_t1
                    ])
                else:
                    c = torch.stack([
                        torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device),
                        torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device),
                        x_t1,
                        torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
                    ])
                child_rows[k].append(c)
            if not ok:
                parent_rows.pop()
                for k in range(n_branches):
                    if child_rows[k]:
                        child_rows[k].pop()

        if not parent_rows:
            return []
        parent = torch.stack(parent_rows, dim=0).to(torch.float32)
        children = [torch.stack(rows, dim=0).to(torch.float32) for rows in child_rows]
        return self._build_batches_from_parent_children(parent, children, batch_size=batch_size, eta_resample=False)
    
    def _init_weight_scheduler(self) -> LossWeightScheduler:
        """
        初始化损失权重调度器
        """
        initial_weights = {
            'sdf': self.hyperparams.w_sdf,
            'p0': self.hyperparams.w_p0,
            'pi': self.hyperparams.w_pi,
            'q': self.hyperparams.w_q,
            'fc2': self.hyperparams.w_fc2
        }
        
        return LossWeightScheduler(
            initial_weights,
            schedule_type='fixed',
            warmup_steps=self.hyperparams.warmup_steps,
            total_steps=self.hyperparams.max_steps
        )
    
    def _init_lr_schedulers(self) -> Dict:
        """
        初始化学习率调度器
        """
        schedulers = {}
        
        for name, optimizer in self.optimizers.items():
            schedulers[name] = LearningRateScheduler(
                optimizer,
                base_lr=self.hyperparams.lr,
                warmup_steps=self.hyperparams.warmup_steps,
                decay_type='cosine',
                total_steps=self.hyperparams.max_steps,
                min_lr=self.hyperparams.min_lr
            )
        
        return schedulers

    def _compute_bp_kkt_penalty(
        self,
        bp: torch.Tensor,
        foc_residuals: List[torch.Tensor],
        eta_children: Optional[List[torch.Tensor]] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        对有界控制变量 bp ∈ [0,1] 施加 KKT 条件：
        - 内点: FOC = 0
        - 下边界(bp≈0): FOC <= 0
        - 上边界(bp≈1): FOC >= 0
        """
        if not foc_residuals:
            z = torch.tensor(0.0, device=self.device)
            return z, {
                'kkt_inner': 0.0,
                'kkt_low': 0.0,
                'kkt_high': 0.0,
                'kkt_foc_abs_mean': 0.0,
                'kkt_w_inner_mean': 0.0,
                'kkt_w_low_mean': 0.0,
                'kkt_w_high_mean': 0.0,
                'kkt_high_weight': 0.0,
                'kkt_active_ratio': 0.0,
            }

        foc_stack = torch.stack([r if r.dim() == 2 else r.unsqueeze(-1) for r in foc_residuals], dim=1)  # (B, N, 1)
        if eta_children is not None and len(eta_children) == foc_stack.shape[1]:
            eta_stack = torch.stack(
                [e if e.dim() == 2 else e.unsqueeze(-1) for e in eta_children], dim=1
            ).to(foc_stack.dtype)
            eta_stack = eta_stack.clamp(min=0.0, max=1.0)
            eta_count = eta_stack.sum(dim=1)  # (B,1)
            active_mask = (eta_count > 0).to(foc_stack.dtype)
            foc_mean = (eta_stack * foc_stack).sum(dim=1) / (eta_count + 1e-6)  # (B,1), signed
        else:
            active_mask = torch.ones_like(bp)
            foc_mean = foc_stack.mean(dim=1)  # (B, 1), signed
        if sample_weight is not None:
            active_mask = active_mask * sample_weight.clamp(min=0.0, max=1.0)
        active_ratio = float(active_mask.mean().item())
        if active_ratio <= 1e-8:
            z = torch.tensor(0.0, device=self.device)
            return z, {
                'kkt_inner': 0.0,
                'kkt_low': 0.0,
                'kkt_high': 0.0,
                'kkt_foc_abs_mean': 0.0,
                'kkt_w_inner_mean': 0.0,
                'kkt_w_low_mean': 0.0,
                'kkt_w_high_mean': 0.0,
                'kkt_high_weight': float(getattr(self.hyperparams, "kkt_high_weight", 3.0)),
                'kkt_active_ratio': 0.0,
            }

        eps_default = float(getattr(self.hyperparams, "kkt_boundary_eps", 0.02))
        eps_low = getattr(self.hyperparams, "kkt_boundary_eps_low", None)
        eps_high = getattr(self.hyperparams, "kkt_boundary_eps_high", None)
        eps_low = eps_default if eps_low is None else float(eps_low)
        eps_high = eps_default if eps_high is None else float(eps_high)
        temp = float(getattr(self.hyperparams, "kkt_boundary_temp", 40.0))
        eps_low = min(max(eps_low, 1e-6), 0.49)
        eps_high = min(max(eps_high, 1e-6), 0.49)
        temp = max(temp, 1.0)

        # 软区域权重，避免硬切分导致不连续
        w_low = torch.sigmoid(temp * (eps_low - bp)) * active_mask
        w_high = torch.sigmoid(temp * (bp - (1.0 - eps_high))) * active_mask
        w_inner = (1.0 - w_low) * (1.0 - w_high)

        def _wmean(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            return (v * w).sum() / (w.sum() + 1e-6)

        # 违反 KKT 的量
        # 内点要求 FOC=0；边界分别要求符号
        viol_inner = foc_mean.pow(2)
        viol_low = torch.relu(foc_mean)      # bp≈0 时应 <=0
        viol_high = torch.relu(-foc_mean)    # bp≈1 时应 >=0

        loss_inner = _wmean(viol_inner, w_inner)
        loss_low = _wmean(viol_low, w_low)
        loss_high = _wmean(viol_high, w_high)

        w_inner_cfg = float(getattr(self.hyperparams, "kkt_inner_weight", 1.0))
        w_bound_cfg = float(getattr(self.hyperparams, "kkt_boundary_weight", 1.0))
        w_high_cfg = float(getattr(self.hyperparams, "kkt_high_weight", 3.0))
        penalty = w_inner_cfg * loss_inner + w_bound_cfg * (loss_low + w_high_cfg * loss_high)

        with torch.no_grad():
            diag = {
                'kkt_inner': float(loss_inner.item()),
                'kkt_low': float(loss_low.item()),
                'kkt_high': float(loss_high.item()),
                'kkt_foc_abs_mean': float(foc_mean.abs().mean().item()),
                'kkt_w_inner_mean': float(w_inner.mean().item()),
                'kkt_w_low_mean': float(w_low.mean().item()),
                'kkt_w_high_mean': float(w_high.mean().item()),
                'kkt_eps_low': float(eps_low),
                'kkt_eps_high': float(eps_high),
                'kkt_high_weight': float(w_high_cfg),
                'kkt_active_ratio': float(active_mask.mean().item()),
            }
        return penalty, diag

    def _compute_bp_survival_weight(
        self,
        P_children: List[torch.Tensor],
        bar_z_children: List[torch.Tensor],
        eta_children: Optional[List[torch.Tensor]] = None,
    ) -> Optional[torch.Tensor]:
        """
        基于 child 的总股权价值与存活门，构造 bp surrogate 的软存活权重。

        w_surv ≈ sigmoid(tau_p * P_{t+1}) * sigmoid(tau_z * (z_th - bar_z_{t+1}))
        再按 eta-active 分支做平均，使 default 区右侧的局部驻点不再主导 bp 训练。
        """
        if not bool(getattr(self.hyperparams, "bp_survival_reweight_enabled", True)):
            return None
        if not P_children or not bar_z_children:
            return None

        tau_p = float(getattr(self.hyperparams, "bp_survival_tau_p", 20.0))
        tau_z = float(getattr(self.hyperparams, "bp_survival_tau_z", 20.0))
        z_th = float(getattr(self.hyperparams, "bp_survival_barz_threshold", 0.5))
        tau_p = max(1.0, tau_p)
        tau_z = max(1.0, tau_z)

        branch_weights = []
        for p_child, bar_z_child in zip(P_children, bar_z_children):
            w_p = torch.sigmoid(tau_p * p_child)
            w_z = torch.sigmoid(tau_z * (z_th - bar_z_child))
            branch_weights.append(w_p * w_z)
        w_stack = torch.stack(
            [w if w.dim() == 2 else w.unsqueeze(-1) for w in branch_weights], dim=1
        )  # (B,N,1)

        if eta_children is not None and len(eta_children) == w_stack.shape[1]:
            eta_stack = torch.stack(
                [e if e.dim() == 2 else e.unsqueeze(-1) for e in eta_children], dim=1
            ).to(w_stack.dtype).clamp(min=0.0, max=1.0)
            eta_count = eta_stack.sum(dim=1)
            w_surv = (eta_stack * w_stack).sum(dim=1) / (eta_count + 1e-6)
        else:
            w_surv = w_stack.mean(dim=1)
        return w_surv.clamp(min=0.0, max=1.0)

    def _compute_bp_value_supervision_loss(
        self,
        *,
        parent_state: torch.Tensor,
        parent_Q: torch.Tensor,
        bp_pred: torch.Tensor,
        M_list: List[torch.Tensor],
        eta_children: List[torch.Tensor],
        branch_kind: str,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        用粗网格近似 argmax V(bp)，给 bp 一个直接的全局 value-level 训练信号。

        关键约束：
        - 目标搜索在 `torch.no_grad()` 下执行，不额外保留计算图
        - 小网格 + sample cap，保证速度
        - 全程在 GPU 上向量化，避免 Python 循环扫点
        """
        enabled = bool(getattr(self.hyperparams, "bp_value_supervision_enabled", False))
        weight = float(getattr(self.hyperparams, "bp_value_weight", 0.0))
        if (not enabled) or weight <= 0.0 or parent_state.numel() == 0:
            z = torch.tensor(0.0, device=self.device)
            return z, {
                'bp_value_enabled': 0.0,
                'bp_value_weight': float(weight),
                'bp_value_target_mean': 0.0,
                'bp_value_pred_mean': 0.0,
                'bp_value_feasible_ratio': 0.0,
                'bp_value_sample_n': 0.0,
            }

        sample_cap = int(getattr(self.hyperparams, "bp_value_sample_cap", 256))
        grid_points = int(getattr(self.hyperparams, "bp_value_grid_points", 21))
        survival_only = bool(getattr(self.hyperparams, "bp_value_survival_only", True))
        barz_threshold = float(getattr(self.hyperparams, "bp_value_barz_threshold", 0.5))
        sample_cap = max(1, sample_cap)
        grid_points = max(5, grid_points)

        batch_n = parent_state.shape[0]
        if batch_n <= sample_cap:
            sample_idx = torch.arange(batch_n, device=self.device)
        else:
            sample_idx = torch.linspace(
                0, batch_n - 1, steps=sample_cap, device=self.device
            ).round().long().unique(sorted=True)

        parent_sub = parent_state.index_select(0, sample_idx)
        q_parent_sub = parent_Q.index_select(0, sample_idx)
        bp_pred_sub = bp_pred.index_select(0, sample_idx)
        m_sub_list = [m.index_select(0, sample_idx) for m in M_list]
        eta_sub_list = [eta.index_select(0, sample_idx) for eta in eta_children]

        S = parent_sub.shape[0]
        G = grid_points
        bp_grid = torch.linspace(0.0, 1.0, steps=G, device=self.device).view(1, G, 1)

        x_grid = parent_sub[:, 4:5].unsqueeze(1).expand(S, G, 1)
        z_grid = parent_sub[:, 1:2].unsqueeze(1).expand(S, G, 1)
        b_parent_grid = parent_sub[:, 0:1].unsqueeze(1).expand(S, G, 1)
        i_grid = parent_sub[:, 3:4].unsqueeze(1).expand(S, G, 1)
        q_parent_grid = q_parent_sub.unsqueeze(1).expand(S, G, 1)

        pv_model = self.models['policy_value']
        p0_loss = self.loss_fns['p0']
        pi_loss = self.loss_fns['pi']

        with torch.no_grad():
            branch_values = []
            branch_P = []
            branch_barz = []
            for eta_sub, m_sub in zip(eta_sub_list, m_sub_list):
                eta_grid = eta_sub.unsqueeze(1).expand(S, G, 1)
                child_state = parent_sub.unsqueeze(1).expand(S, G, parent_sub.shape[1]).clone()
                # Align bp value supervision with the Bellman continuation object:
                # the continuation debt argument should stay at the chosen contract b',
                # not the eta'-mixed realized debt.
                child_state[:, :, 0:1] = bp_grid
                child_out = pv_model(child_state.reshape(S * G, -1))

                q_child = child_out.Q.reshape(S, G, 1)
                p_child = child_out.P.reshape(S, G, 1)
                bar_z_child = child_out.bar_z.reshape(S, G, 1)
                m_grid = m_sub.unsqueeze(1).expand(S, G, 1)

                if branch_kind == 'p0':
                    cf = p0_loss.compute_cashflow_p0(
                        x_grid, z_grid, b_parent_grid, q_parent_grid, q_child, eta_grid
                    )
                    cont = m_grid * p_child * (1.0 - bar_z_child)
                else:
                    cf = pi_loss.compute_cashflow_pi(
                        x_grid, z_grid, b_parent_grid, i_grid, q_parent_grid, q_child, eta_grid
                    )
                    cont = Config.G * m_grid * p_child * (1.0 - bar_z_child)

                branch_values.append((cf + cont).squeeze(-1))
                branch_P.append(p_child.squeeze(-1))
                branch_barz.append(bar_z_child.squeeze(-1))

            value_grid = torch.stack(branch_values, dim=1).mean(dim=1)  # (S,G)
            p_grid = torch.stack(branch_P, dim=1).mean(dim=1)  # (S,G)
            barz_grid = torch.stack(branch_barz, dim=1).mean(dim=1)  # (S,G)

            if survival_only:
                feasible = (p_grid > 0.0) & (barz_grid < barz_threshold)
                masked_value = value_grid.masked_fill(~feasible, -1e12)
                best_idx = masked_value.argmax(dim=1)
                has_feasible = feasible.any(dim=1)
                best_idx = torch.where(
                    has_feasible,
                    best_idx,
                    torch.zeros_like(best_idx),
                )
                feasible_ratio = float(has_feasible.float().mean().item())
            else:
                best_idx = value_grid.argmax(dim=1)
                feasible_ratio = 1.0

            bp_target = bp_grid.view(G).index_select(0, best_idx).unsqueeze(-1)

        loss = (bp_pred_sub - bp_target).pow(2).mean()
        with torch.no_grad():
            diag = {
                'bp_value_enabled': 1.0,
                'bp_value_weight': float(weight),
                'bp_value_target_mean': float(bp_target.mean().item()),
                'bp_value_pred_mean': float(bp_pred_sub.mean().item()),
                'bp_value_feasible_ratio': float(feasible_ratio),
                'bp_value_sample_n': float(S),
            }
        return loss, diag

    def _compute_conditional_signed_foc_terms(
        self,
        foc_residuals: List[torch.Tensor],
        eta_children: List[torch.Tensor],
        z_parent: torch.Tensor,
        alpha_z: float,
        beta_z: float,
        z0: float,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        FOC 采用“条件在再融资事件上的 signed moment”：
            E[FOC | eta=1] = 0
        其中 FOC 保留符号，再最小化该条件矩的平方。
        """
        if not foc_residuals:
            z = torch.tensor(0.0, device=self.device)
            return z, z, {
                'foc_active_ratio': 0.0,
                'foc_signed_moment': 0.0,
                'foc_cond_abs_mean': 0.0,
            }

        foc_stack = torch.stack([r if r.dim() == 2 else r.unsqueeze(-1) for r in foc_residuals], dim=1)  # (B,N,1)
        eta_stack = torch.stack(
            [e if e.dim() == 2 else e.unsqueeze(-1) for e in eta_children], dim=1
        ).to(foc_stack.dtype).clamp(min=0.0, max=1.0)

        eta_count = eta_stack.sum(dim=1)  # (B,1)
        active_mask = (eta_count > 0).to(foc_stack.dtype)
        active_bool = active_mask.squeeze(-1) > 0.5
        active_n = int(active_bool.sum().item())
        if active_n == 0:
            z = torch.tensor(0.0, device=self.device)
            return z, z, {
                'foc_active_ratio': 0.0,
                'foc_signed_moment': 0.0,
                'foc_cond_abs_mean': 0.0,
                'foc_active_n': 0.0,
            }

        foc_cond_signed = (eta_stack * foc_stack).sum(dim=1) / (eta_count + 1e-6)  # (B,1), signed
        foc_cond_abs = (eta_stack * foc_stack.abs()).sum(dim=1) / (eta_count + 1e-6)  # (B,1), non-negative
        if sample_weight is not None:
            sw = sample_weight[active_bool].clamp(min=0.0, max=1.0)
        else:
            sw = torch.ones_like(foc_cond_signed[active_bool])
        sw_sum = sw.sum().clamp_min(1e-6)
        foc_signed_moment = (sw * foc_cond_signed[active_bool]).sum() / sw_sum
        loss_foc = foc_signed_moment.pow(2)

        # 仅在 eta 活跃子样本上评估 z-penalty，避免被 eta=0 样本稀释。
        z_weight = torch.sigmoid(beta_z * (z_parent[active_bool] - z0))
        penalty_z_foc = alpha_z * ((sw * z_weight * foc_cond_abs[active_bool]).sum() / sw_sum)

        with torch.no_grad():
            diag = {
                'foc_active_ratio': float(active_mask.mean().item()),
                'foc_signed_moment': float(foc_signed_moment.item()),
                'foc_cond_abs_mean': float(foc_cond_abs[active_bool].mean().item()),
                'foc_active_n': float(active_n),
                'foc_survival_weight_mean': float(sample_weight.mean().item()) if sample_weight is not None else 1.0,
            }
        return loss_foc, penalty_z_foc, diag

    def _compute_eta_active_boost(self, active_ratio: float) -> float:
        """
        eta 稀疏时，对 bp 相关项(FOC/KKT)做条件重权重。
        """
        if not bool(getattr(self.hyperparams, "eta_active_reweight_enabled", True)):
            return 1.0
        target = float(getattr(self.hyperparams, "eta_active_target_ratio", 0.25))
        max_boost = float(getattr(self.hyperparams, "eta_active_max_reweight", 6.0))
        target = min(max(target, 1e-4), 1.0)
        max_boost = max(1.0, max_boost)
        ar = max(float(active_ratio), 1e-6)
        return float(min(max_boost, max(1.0, target / ar)))

    def _compute_bp_adaptive_scale(
        self,
        main_loss: torch.Tensor,
        bp_terms_after_eta: torch.Tensor
    ) -> float:
        """
        自适应放大 bp 相关损失，使其与 Bellman 主项达到可比量级。
        """
        if not bool(getattr(self.hyperparams, "bp_adaptive_enabled", True)):
            return 1.0
        ratio = float(getattr(self.hyperparams, "bp_target_main_ratio", 0.3))
        min_scale = float(getattr(self.hyperparams, "bp_adaptive_min_scale", 1.0))
        max_scale = float(getattr(self.hyperparams, "bp_adaptive_max_scale", 200.0))
        min_scale = max(1.0, min_scale)
        max_scale = max(min_scale, max_scale)

        main_v = float(main_loss.detach().abs().item())
        bp_v = float(bp_terms_after_eta.detach().abs().item())
        target_v = max(1e-10, ratio * main_v)
        raw_scale = target_v / (bp_v + 1e-10)
        return float(min(max_scale, max(min_scale, raw_scale)))

    def _configure_sdf_lr_for_phase(self):
        """
        在 SDF 第一阶段/第二阶段按子模块设置学习率。
        """
        if 'sdf_fc1' not in self.optimizers or 'sdf_fc1' not in self.lr_schedulers:
            return

        opt = self.optimizers['sdf_fc1']
        scheduler = self.lr_schedulers['sdf_fc1']
        sdf_lr = float(getattr(self.hyperparams, "sdf_stage1_lr", getattr(self.hyperparams, "sdf_lr", scheduler.base_lr)))
        if self.add_FC1loss:
            stage2_lr = getattr(self.hyperparams, "sdf_stage2_lr", None)
            if stage2_lr is not None:
                sdf_lr = float(stage2_lr)
        fc1_lr = float(getattr(self.hyperparams, "fc1_lr", sdf_lr))

        group_base_lrs = []
        for group in opt.param_groups:
            group_name = str(group.get('group_name', ''))
            target_lr = fc1_lr if group_name == 'fc1' else sdf_lr
            group['lr'] = target_lr
            group_base_lrs.append(target_lr)

        scheduler.group_base_lrs = list(group_base_lrs)
        scheduler.base_lr = max(group_base_lrs) if group_base_lrs else sdf_lr
        scheduler.current_lr = scheduler.base_lr

    def _set_policy_q_only_freeze(self, enable: bool):
        """
        Q-only 阶段：仅允许 Q block 训练。
        """
        if 'policy_value' not in self.models:
            return

        model = self.models['policy_value']
        if not bool(getattr(self.hyperparams, "q_freeze_non_q_in_pretrain", True)):
            enable = False

        for p in model.parameters():
            p.requires_grad = not enable

        if enable:
            scope = str(getattr(self.hyperparams, "q_pretrain_trainable_scope", "q_path")).lower()
            if scope not in {"q_head_only", "q_path"}:
                scope = "q_path"
            if scope == "q_path":
                for p in model.q_model.parameters():
                    p.requires_grad = True
            else:
                for p in model.q_model.q_head.parameters():
                    p.requires_grad = True
        self._policy_q_freeze_active = enable

    def _set_policy_pvbp_only_freeze(self, enable: bool):
        """
        PV/BP 阶段：仅允许 PVBP block 训练，Q block 保持冻结。
        """
        if 'policy_value' not in self.models:
            return

        model = self.models['policy_value']
        for p in model.parameters():
            p.requires_grad = not enable
        if enable:
            for p in model.pvbp_model.parameters():
                p.requires_grad = True
        self._policy_pvbp_freeze_active = enable

    def _set_policy_bp_only_freeze(self, enable: bool):
        """
        bp-only 精修阶段：仅更新 PVBP block 中的 bp0/bpI 头参数。
        """
        if 'policy_value' not in self.models:
            return

        model = self.models['policy_value']
        for p in model.parameters():
            p.requires_grad = not enable
        if enable:
            for p in model.pvbp_model.bp0_head.parameters():
                p.requires_grad = True
            for p in model.pvbp_model.bpI_head.parameters():
                p.requires_grad = True
        self._policy_bp_freeze_active = enable

    def _set_policy_value_only_freeze(self, enable: bool):
        """
        value-only 阶段：仅更新 PVBP block 中除 bp0/bpI 外的参数，
        让 value/gate 面先对齐，再单独交给 bp 头做策略跟随。
        """
        if 'policy_value' not in self.models:
            return

        model = self.models['policy_value']
        for p in model.parameters():
            p.requires_grad = not enable
        if enable:
            for p in model.pvbp_model.parameters():
                p.requires_grad = True
            for p in model.pvbp_model.bp0_head.parameters():
                p.requires_grad = False
            for p in model.pvbp_model.bpI_head.parameters():
                p.requires_grad = False
        self._policy_value_only_freeze_active = enable

    def _resolve_optimizer_keys(
        self,
        train_modules: List[str],
        policy_loss_terms: Optional[List[str]] = None,
    ) -> List[str]:
        keys: List[str] = []
        for name in train_modules:
            if name == 'fc2':
                keys.extend(self._get_fc2_optimizer_keys())
                continue
            if name != 'policy_value':
                if name in self.optimizers:
                    keys.append(name)
                continue
            terms = set(policy_loss_terms or [])
            if terms == {'q'} and len(policy_loss_terms or []) == 1:
                if 'policy_value_q' in self.optimizers:
                    keys.append('policy_value_q')
            elif terms == {'p0', 'pi'} and len(policy_loss_terms or []) == 2:
                if 'policy_value_pvbp' in self.optimizers:
                    keys.append('policy_value_pvbp')
            else:
                if 'policy_value_q' in self.optimizers:
                    keys.append('policy_value_q')
                if 'policy_value_pvbp' in self.optimizers:
                    keys.append('policy_value_pvbp')
        return keys

    def _set_sdf_fc1_teacher_only_freeze(self, enable: bool):
        """
        FC1 teacher forcing 阶段：仅更新 FC1_C / FC1_K，不更新 SDF/value。
        """
        if 'sdf_fc1' not in self.models:
            return

        model = self.models['sdf_fc1']
        if enable:
            if self._sdf_fc1_teacher_freeze_active:
                return
            self._sdf_fc1_grad_backup = {
                name: p.requires_grad for name, p in model.named_parameters()
            }
            for p in model.parameters():
                p.requires_grad = False
            for p in model.fc1_model.parameters():
                p.requires_grad = True
            self._sdf_fc1_teacher_freeze_active = True
        else:
            if not self._sdf_fc1_teacher_freeze_active:
                return
            for name, p in model.named_parameters():
                if name in self._sdf_fc1_grad_backup:
                    p.requires_grad = self._sdf_fc1_grad_backup[name]
            self._sdf_fc1_grad_backup = {}
            self._sdf_fc1_teacher_freeze_active = False

    def _compute_stage2_hj_warmup_factor(self) -> float:
        """
        Stage2 联合训练初期，线性放大 HJ 相关项权重，避免 FC1 被过早牵引。
        """
        if not self.add_FC1loss or bool(getattr(self, "_fc1_teacher_forcing_stage", False)):
            return 1.0
        warmup_epochs = max(0, int(getattr(self.hyperparams, "sdf_stage2_hj_warmup_epochs", 0)))
        if warmup_epochs <= 0:
            return 1.0
        start = float(getattr(self.hyperparams, "sdf_stage2_hj_warmup_start", 0.2))
        start = min(max(start, 0.0), 1.0)
        if self._current_epoch_idx >= warmup_epochs:
            return 1.0
        progress = float(self._current_epoch_idx + 1) / float(max(1, warmup_epochs))
        return float(start + (1.0 - start) * progress)

    def _compute_pvbp_anti_collapse_warmup_factor(self, q_stage_epochs: int) -> float:
        """
        PVBP 阶段前几轮，保持 chi 更接近 1，避免 P/bar_z 过早塌到全默认吸收态。
        """
        if not bool(getattr(self, "_pvbp_only_stage", False)):
            return 1.0
        warmup_epochs = max(0, int(getattr(self.hyperparams, "pvbp_anti_collapse_warmup_epochs", 0)))
        if warmup_epochs <= 0:
            return 1.0
        start = float(getattr(self.hyperparams, "pvbp_anti_collapse_start", 0.25))
        start = min(max(start, 0.0), 1.0)
        local_epoch = int(self._current_epoch_idx) - int(q_stage_epochs)
        if local_epoch >= warmup_epochs:
            return 1.0
        progress = float(local_epoch + 1) / float(max(1, warmup_epochs))
        return float(start + (1.0 - start) * progress)

    def _set_policy_runtime_controls(self, q_stage_epochs: int):
        if 'policy_value' not in self.models:
            return
        model = self.models['policy_value']
        pvbp_model = getattr(model, 'pvbp_model', None)
        if pvbp_model is None:
            return
        pvbp_model.chi_warmup_factor = self._compute_pvbp_anti_collapse_warmup_factor(q_stage_epochs)

    def _compute_value_gate_monotonicity_penalty(
        self,
        parent_state: torch.Tensor,
        value: torch.Tensor,
        chi: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        约束 value/gate 的基本经济方向：
        - value 对 b 递减、对 z 递增
        - chi 对 b 递减、对 z 递增
        等价地，bar_z 对 b 递增、对 z 递减。
        """
        zero = torch.tensor(0.0, device=parent_state.device)
        if not parent_state.requires_grad:
            return zero, {
                'mono_value_b': 0.0,
                'mono_value_z': 0.0,
                'mono_chi_b': 0.0,
                'mono_chi_z': 0.0,
                'mono_total': 0.0,
            }

        mono_value_b = compute_monotonicity_penalty(value, parent_state, 0, 'negative')
        mono_value_z = compute_monotonicity_penalty(value, parent_state, 1, 'positive')
        mono_chi_b = compute_monotonicity_penalty(chi, parent_state, 0, 'negative')
        mono_chi_z = compute_monotonicity_penalty(chi, parent_state, 1, 'positive')

        w_value_b = float(getattr(self.hyperparams, "pv_mono_weight_b", 1.0))
        w_value_z = float(getattr(self.hyperparams, "pv_mono_weight_z", 1.0))
        w_chi_b = float(getattr(self.hyperparams, "chi_mono_weight_b", 0.5))
        w_chi_z = float(getattr(self.hyperparams, "chi_mono_weight_z", 0.5))
        w_barz_b_high = float(getattr(self.hyperparams, "barz_mono_weight_b_high", 1.0))

        total = (
            w_value_b * mono_value_b +
            w_value_z * mono_value_z +
            w_chi_b * mono_chi_b +
            w_chi_z * mono_chi_z
        )
        return total, {
            'mono_value_b': float(mono_value_b.item()),
            'mono_value_z': float(mono_value_z.item()),
            'mono_chi_b': float(mono_chi_b.item()),
            'mono_chi_z': float(mono_chi_z.item()),
            'mono_total': float(total.item()),
        }
    
    def generate_data(
        self,
        mode: str = 'sample',
        n_samples: int = 10000,
        n_paths: int = 100,
        group_size: int = 100,
        **kwargs
    ) -> pd.DataFrame:
        """
        生成训练数据
        
        Args:
            mode: 'sample' 或 'simulate'
            n_samples: 样本数
            n_paths: path 数量
            group_size: 每条 path 的公司数
        
        Returns:
            df: 生成的 DataFrame
        """
        
        if mode == 'sample':
            sampler = Sample(
                models=self.models,
                config=self.config,
                n_samples=None,
                n_paths=n_paths,
                group_size=group_size,
                macro_source_df=self.macro_source_df,
                **kwargs
            )
            self.df = sampler.build_df()
            
        elif mode == 'simulate':
            simulator = SimulateTS(
                models=self.models,
                config=self.config,
                n_paths=n_paths,
                group_size=group_size,
                **kwargs
            )
            self.df, self.df_macro = simulator.simulate()
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return self.df
    
    def fill_fc1(self):
        """
        使用 FC1 填充宏观状态
        """
        if self.df is None:
            raise RuntimeError("先调用 generate_data()")
        
        sampler = Sample(models=self.models, config=self.config)
        self.df = sampler.fill_fc1(self.df)
    
    def fill_policy_value(self):
        """
        使用 Policy/Value 填充决策和价值变量
        """
        if self.df is None:
            raise RuntimeError("先调用 generate_data()")
        
        sampler = Sample(models=self.models, config=self.config)
        self.df = sampler.fill_policy_value(self.df)
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        train_modules: List[str] = None,
        policy_loss_terms: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            batch: 数据批次
            train_modules: 要训练的模块列表
        
        Returns:
            losses: 各损失值字典
        """
        if train_modules is None:
            train_modules = ['sdf_fc1', 'policy_value']
        if policy_loss_terms is None:
            policy_loss_terms = ['p0', 'pi', 'q']

        # 先恢复，再按当前 step 规则决定是否冻结
        self._set_policy_q_only_freeze(False)
        self._set_policy_pvbp_only_freeze(False)
        self._set_policy_bp_only_freeze(False)
        self._set_policy_value_only_freeze(False)
        self._set_sdf_fc1_teacher_only_freeze(False)
        q_only_step = (
            'policy_value' in train_modules and
            set(policy_loss_terms) == {'q'} and
            len(policy_loss_terms) == 1
        )
        pvbp_only_step = (
            'policy_value' in train_modules and
            set(policy_loss_terms) == {'p0', 'pi'} and
            len(policy_loss_terms) == 2
        )
        bp_only_step = (
            pvbp_only_step and
            bool(getattr(self, "_bp_only_stage", False))
        )
        value_only_step = (
            pvbp_only_step and
            bool(getattr(self, "_value_only_stage", False))
        )
        fc1_teacher_step = (
            'sdf_fc1' in train_modules and
            bool(getattr(self, "_fc1_teacher_forcing_stage", False))
        )
        fc1_forecast_only_step = (
            'sdf_fc1' in train_modules and
            bool(getattr(self.hyperparams, "fc1_forecast_only_ablation", False))
        )
        self._set_policy_q_only_freeze(q_only_step)
        if bp_only_step:
            self._set_policy_bp_only_freeze(True)
        elif value_only_step:
            self._set_policy_value_only_freeze(True)
        elif pvbp_only_step:
            self._set_policy_pvbp_only_freeze(True)
        elif not q_only_step:
            self._set_policy_bp_only_freeze(bp_only_step)
        self._set_sdf_fc1_teacher_only_freeze(fc1_teacher_step or fc1_forecast_only_step)
        active_optimizer_keys = self._resolve_optimizer_keys(train_modules, policy_loss_terms)
        
        losses = {}
        
        # 设置训练模式
        for name in self._expand_train_model_keys(train_modules):
            if name in self.models:
                self.models[name].train()
        
        # 清零梯度
        for opt_key in active_optimizer_keys:
            self.optimizers[opt_key].zero_grad()
        
        try:
            # 计算损失
            total_loss = torch.tensor(0.0, device=self.device)
            
            # SDF Loss
            if 'sdf_fc1' in train_modules and 'sdf_fc1' in self.models:
                sdf_loss = self._compute_sdf_loss(batch)
                losses['sdf'] = sdf_loss.item()
                losses.update(self._latest_sdf_terms)
                losses.update(self._latest_sdf_diag)
                total_loss = total_loss + self.weight_scheduler['sdf'] * sdf_loss
            
            # P0 Loss
            if 'policy_value' in train_modules and 'policy_value' in self.models and 'p0' in policy_loss_terms:
                p0_loss = self._compute_p0_loss(batch)
                losses['p0'] = p0_loss.item()
                losses.update(self._latest_p0_terms)
                total_loss = total_loss + self.weight_scheduler['p0'] * p0_loss
            
            # PI Loss
            if 'policy_value' in train_modules and 'policy_value' in self.models and 'pi' in policy_loss_terms:
                pi_loss = self._compute_pi_loss(batch)
                losses['pi'] = pi_loss.item()
                losses.update(self._latest_pi_terms)
                total_loss = total_loss + self.weight_scheduler['pi'] * pi_loss
            
            # Q Loss
            if 'policy_value' in train_modules and 'policy_value' in self.models and 'q' in policy_loss_terms:
                q_loss = self._compute_q_loss(batch)
                losses['q'] = q_loss.item()
                losses.update(self._latest_q_terms)
                total_loss = total_loss + self.weight_scheduler['q'] * q_loss
            
            # FC2 Loss
            if 'fc2' in train_modules and self._has_fc2_models():
                if self.episode_id > 0 and isinstance(batch, pd.DataFrame):
                    batch.to_csv(f"fc2_input_episode{self.episode_id}_ori.csv", index=False)
                    batch = convert_tree_fast(batch)
                    batch = trim_child_only_ids(batch)
                    batch.to_csv(f"fc2_input_episode{self.episode_id}.csv", index=False)
                
                fc2_loss = self._compute_fc2_loss(batch)
                losses['fc2'] = fc2_loss.item()
                losses.update(self._latest_fc2_diag)
                total_loss = total_loss + self.weight_scheduler['fc2'] * fc2_loss
            
            losses['total'] = total_loss.item()
            
            # 反向传播
            if total_loss.requires_grad:
                total_loss.backward()
                
                # 梯度保护
                for name in self._expand_train_model_keys(train_modules):
                    if name in self.models:
                        grad_norm, had_nan = gradient_protection(
                            self.models[name].parameters(),
                            max_norm=self.hyperparams.max_grad_norm
                        )
                        losses[f'{name}_grad_norm'] = grad_norm
                        if had_nan:
                            logger.warning(f"NaN gradient detected in {name}")

                if 'fc2_hatc_grad_norm' in losses or 'fc2_lnk_grad_norm' in losses:
                    losses['fc2_grad_norm'] = float(max(
                        losses.get('fc2_hatc_grad_norm', 0.0),
                        losses.get('fc2_lnk_grad_norm', 0.0),
                    ))
                
                # 优化器步骤
                for opt_key in active_optimizer_keys:
                    self.optimizers[opt_key].step()
        finally:
            self._set_policy_q_only_freeze(False)
            self._set_policy_pvbp_only_freeze(False)
            self._set_policy_bp_only_freeze(False)
            self._set_policy_value_only_freeze(False)
            self._set_sdf_fc1_teacher_only_freeze(False)

        # 更新调度器
        self.weight_scheduler.step(losses)
        for key in active_optimizer_keys:
            scheduler = self.lr_schedulers.get(key)
            if scheduler is not None:
                scheduler.step()
        
        self.step_count += 1
        
        # 记录历史
        for k, v in losses.items():
            if k not in self.loss_history:
                self.loss_history[k] = []
            self.loss_history[k].append(v)
        
        return losses
    
    def _compute_sdf_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算 SDF 损失（支持任意 N 分支）
        """
        model = self.models['sdf_fc1']
        loss_fn = self.loss_fns['sdf']
        
        # 提取输入
        parent = batch['parent']
        children = batch.get('children', [])  # List of child tensors
        
        # 兼容旧的 child0/child1 格式
        if not children:
            child0 = batch.get('child0')
            child1 = batch.get('child1')
            if child0 is not None and child1 is not None:
                children = [child0, child1]

        if not children or len(children) < 2:
            raise ValueError("Need at least two children in batch for SDF loss")

        # 仅使用 child0 / child1
        if len(children) > 2:
            children = children[:2]

        children_t = torch.stack(children, dim=1)  # (batch, 2, feat)

        use_true_prev_macro = bool(
            self.add_FC1loss and
            parent.shape[1] >= 9 and
            getattr(self.hyperparams, "fc1_use_true_macro_state_in_stage2", True)
        )
        if self._fc1_teacher_forcing_stage and parent.shape[1] >= 9:
            use_true_prev_macro = True
        c_prev_input = parent[:, 7:8] if use_true_prev_macro else parent[:, 5:6]
        k_prev_input = parent[:, 8:9] if use_true_prev_macro else parent[:, 6:7]

        # 前向传播：一次性处理两条子路径
        w_parent, w_children, M, c_children, k_children = model.forward_step(
            x_prev=parent[:, 4:5],
            x_curr=children_t[:, :, 4:5],
            hatcf_prev=c_prev_input,
            lnkf_prev=k_prev_input,
            return_physical=True
        )
        
        # 提取父节点状态并计算 w
        c_parent = c_prev_input
        k_parent = k_prev_input
        
        # 计算损失
        # 构造残差并按 parent 聚合
        residuals = loss_fn.compute_euler_residuals(
            w_parent.squeeze(-1), w_children.squeeze(-1),
            k_parent.squeeze(-1), k_children.squeeze(-1),
            c_parent.squeeze(-1), c_children.squeeze(-1)
        )  # (batch, n_children)
        combined = residuals.prod(dim=-1)  # (batch,)
        main_loss = torch.log1p(combined.abs()).mean()

        moment_loss = torch.tensor(0.0, device=self.device)
        M_use = M.squeeze(-1) if M.dim() == 3 else M
        if M_use.dim() == 1:
            M_use = M_use.unsqueeze(-1)
        for j in range(M_use.shape[1]):
            L1, L2 = moment_penalty(M_use[:, j], loss_fn.mu_lo, loss_fn.mu_hi, loss_fn.var_hi)
            moment_loss = moment_loss + L1 + L2

        # 第一阶段（add_FC1loss=False）加强矩约束权重
        if self.add_FC1loss:
            moment_weight = getattr(self.hyperparams, "sdf_moment_weight", 1.0)
        else:
            moment_weight = getattr(self.hyperparams, "sdf_stage1_moment_weight", 5.0)

        # 对 log(E[M]) 增加显式锚，避免 SDF 均值在两阶段切换后漂移
        mean_anchor_loss = torch.tensor(0.0, device=self.device)
        mean_anchor_target = getattr(self.hyperparams, "sdf_log_mean_target", None)
        if mean_anchor_target is not None:
            log_mu_for_anchor = torch.log(M_use.mean().clamp_min(1e-8))
            mean_anchor_loss = (log_mu_for_anchor - float(mean_anchor_target)) ** 2
        if self.add_FC1loss:
            mean_anchor_weight = float(
                getattr(self.hyperparams, "sdf_log_mean_anchor_weight_stage2", 5.0)
            )
        else:
            mean_anchor_weight = float(
                getattr(self.hyperparams, "sdf_log_mean_anchor_weight_stage1", 1.0)
            )

        # 可选：FC1 输出与真实 hatcf / lnkf 的重建误差
        recon_weight = getattr(self.hyperparams, "fc1_recon_weight", 0.0)
        forecast_recon_weight = float(
            getattr(self.hyperparams, "fc1_forecast_recon_weight", 0.0)
        )
        hatc_law_consistency_weight = float(
            getattr(self.hyperparams, "fc1_hatc_law_consistency_weight", 1.0)
        )
        lnk_law_consistency_weight = float(
            getattr(self.hyperparams, "fc1_lnk_law_consistency_weight", 0.25)
        )
        hatc_x_response_weight = float(
            getattr(self.hyperparams, "fc1_hatc_x_response_weight", 0.0)
        )
        lnk_x_response_weight = float(
            getattr(self.hyperparams, "fc1_lnk_x_response_weight", 0.0)
        )
        x_response_bins = int(
            getattr(self.hyperparams, "fc1_x_response_bins", 7)
        )
        hatc_recon_inner_weight = float(
            getattr(self.hyperparams, "fc1_hatc_recon_weight", 1.0)
        )
        lnk_recon_inner_weight = float(
            getattr(self.hyperparams, "fc1_lnk_recon_weight", 1.0)
        )
        delta_penalty_weight = float(
            getattr(self.hyperparams, "fc1_delta_penalty_weight", 0.0)
        )
        jacobian_penalty_weight = float(
            getattr(self.hyperparams, "fc1_jacobian_penalty_weight", 0.0)
        )
        delta_hatc_abs_max = float(
            getattr(self.hyperparams, "fc1_delta_hatc_abs_max", float("inf"))
        )
        delta_lnk_abs_max = float(
            getattr(self.hyperparams, "fc1_delta_lnk_abs_max", float("inf"))
        )
        recon_loss = torch.tensor(0.0, device=self.device)
        recon_loss_forecast = torch.tensor(0.0, device=self.device)
        recon_loss_hatc = torch.tensor(0.0, device=self.device)
        recon_loss_lnk = torch.tensor(0.0, device=self.device)
        recon_loss_forecast_hatc = torch.tensor(0.0, device=self.device)
        recon_loss_forecast_lnk = torch.tensor(0.0, device=self.device)
        law_consistency_loss_hatc = torch.tensor(0.0, device=self.device)
        law_consistency_loss_lnk = torch.tensor(0.0, device=self.device)
        law_consistency_loss = torch.tensor(0.0, device=self.device)
        x_response_loss_hatc = torch.tensor(0.0, device=self.device)
        x_response_loss_lnk = torch.tensor(0.0, device=self.device)
        x_response_loss = torch.tensor(0.0, device=self.device)
        delta_penalty = torch.tensor(0.0, device=self.device)
        delta_penalty_hatc = torch.tensor(0.0, device=self.device)
        delta_penalty_lnk = torch.tensor(0.0, device=self.device)
        jacobian_penalty = torch.tensor(0.0, device=self.device)
        jacobian_penalty_hatc = torch.tensor(0.0, device=self.device)
        jacobian_penalty_lnk = torch.tensor(0.0, device=self.device)
        if self.add_FC1loss:
            hatcf_pred = c_children  # (batch, 2, 1)
            lnkf_pred = k_children   # (batch, 2, 1)
            # 与当前 SDF macro-batch 布局保持一致：
            # [..., x, Hatcf, LnKF, Hatc_true, LnK_true]
            if children_t.shape[-1] >= 10:
                # 带 M 列时，真值位于 8/9 列
                hatcf_true = children_t[:, :, 8:9]
                lnkf_true = children_t[:, :, 9:10]
            elif children_t.shape[-1] >= 9:
                # 旧布局（无 M 列）时，真值位于 7/8 列
                hatcf_true = children_t[:, :, 7:8]
                lnkf_true = children_t[:, :, 8:9]
            else:
                # 兼容旧批次格式（无完整 FC1 真值列）时跳过重建损失，避免空切片导致 NaN
                logger.warning(
                    "FC1 recon targets missing in SDF batch (need >=9 cols, got %d). "
                    "Skip recon loss for this step.",
                    children_t.shape[-1]
                )
                recon_loss = torch.tensor(0.0, device=self.device)
                hatcf_true = None
                lnkf_true = None

            if hatcf_true is not None and lnkf_true is not None:
                recon_loss_hatc = (hatcf_pred - hatcf_true).pow(2).mean()
                recon_loss_lnk = (lnkf_pred - lnkf_true).pow(2).mean()
                recon_loss = (
                    hatc_recon_inner_weight * recon_loss_hatc
                    + lnk_recon_inner_weight * recon_loss_lnk
                )

            # 当前口径统一为：
            # (x_t, x_{t+1}, Hatc_t, LnK_t) -> (Hatc_{t+1}, LnK_{t+1})
            # 因此这里的 law consistency 直接使用真实当前 macro state 作为 FC1 输入。
            if (
                forecast_recon_weight > 0.0
                and parent.shape[1] >= 7
                and children_t.shape[-1] >= 9
            ):
                hatcf_prev_forecast = c_prev_input.detach().clone().requires_grad_(True)
                lnkf_prev_forecast = k_prev_input.detach().clone().requires_grad_(True)
                _, _, _, c_children_forecast, k_children_forecast = model.forward_step(
                    x_prev=parent[:, 4:5],
                    x_curr=children_t[:, :, 4:5],
                    hatcf_prev=hatcf_prev_forecast,
                    lnkf_prev=lnkf_prev_forecast,
                    return_physical=True
                )
                law_consistency_loss_hatc = (c_children_forecast - hatcf_true).pow(2).mean()
                law_consistency_loss_lnk = (k_children_forecast - lnkf_true).pow(2).mean()
                recon_loss_forecast_hatc = law_consistency_loss_hatc
                recon_loss_forecast_lnk = law_consistency_loss_lnk
                law_consistency_loss = (
                    hatc_law_consistency_weight * law_consistency_loss_hatc
                    + lnk_law_consistency_weight * law_consistency_loss_lnk
                )
                recon_loss_forecast = law_consistency_loss
                x_curr_flat = children_t[:, :, 4:5].reshape(-1)
                x_response_loss_hatc = self._binned_response_loss_t(
                    x_curr_flat,
                    hatcf_true.reshape(-1),
                    c_children_forecast.reshape(-1),
                    n_bins=x_response_bins,
                )
                x_response_loss_lnk = self._binned_response_loss_t(
                    x_curr_flat,
                    lnkf_true.reshape(-1),
                    k_children_forecast.reshape(-1),
                    n_bins=x_response_bins,
                )
                x_response_loss = (
                    hatc_x_response_weight * x_response_loss_hatc
                    + lnk_x_response_weight * x_response_loss_lnk
                )
                d_hatcf_forecast = c_children_forecast - c_prev_input.unsqueeze(1)
                d_lnkf_forecast = k_children_forecast - k_prev_input.unsqueeze(1)
                delta_penalty_hatc = torch.relu(
                    d_hatcf_forecast.abs() - delta_hatc_abs_max
                ).pow(2).mean()
                delta_penalty_lnk = torch.relu(
                    d_lnkf_forecast.abs() - delta_lnk_abs_max
                ).pow(2).mean()
                delta_penalty = (
                    hatc_recon_inner_weight * delta_penalty_hatc
                    + lnk_recon_inner_weight * delta_penalty_lnk
                )

                if jacobian_penalty_weight > 0.0:
                    grad_hat_wrt_hat = torch.autograd.grad(
                        c_children_forecast.sum(),
                        hatcf_prev_forecast,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    grad_hat_wrt_lnk = torch.autograd.grad(
                        c_children_forecast.sum(),
                        lnkf_prev_forecast,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    grad_lnk_wrt_hat = torch.autograd.grad(
                        k_children_forecast.sum(),
                        hatcf_prev_forecast,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    grad_lnk_wrt_lnk = torch.autograd.grad(
                        k_children_forecast.sum(),
                        lnkf_prev_forecast,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    jacobian_penalty_hatc = (
                        grad_hat_wrt_hat.pow(2).mean()
                        + grad_hat_wrt_lnk.pow(2).mean()
                    )
                    jacobian_penalty_lnk = (
                        grad_lnk_wrt_hat.pow(2).mean()
                        + grad_lnk_wrt_lnk.pow(2).mean()
                    )
                    jacobian_penalty = (
                        hatc_recon_inner_weight * jacobian_penalty_hatc
                        + lnk_recon_inner_weight * jacobian_penalty_lnk
                    )
        if not torch.isfinite(recon_loss):
            logger.warning("Non-finite recon loss detected. Replace with 0.0 for stability.")
            recon_loss = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(recon_loss_forecast):
            logger.warning("Non-finite forecast recon loss detected. Replace with 0.0 for stability.")
            recon_loss_forecast = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(recon_loss_hatc):
            recon_loss_hatc = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(recon_loss_lnk):
            recon_loss_lnk = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(recon_loss_forecast_hatc):
            recon_loss_forecast_hatc = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(recon_loss_forecast_lnk):
            recon_loss_forecast_lnk = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(law_consistency_loss_hatc):
            law_consistency_loss_hatc = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(law_consistency_loss_lnk):
            law_consistency_loss_lnk = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(law_consistency_loss):
            law_consistency_loss = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(x_response_loss_hatc):
            x_response_loss_hatc = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(x_response_loss_lnk):
            x_response_loss_lnk = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(x_response_loss):
            x_response_loss = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(delta_penalty):
            delta_penalty = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(delta_penalty_hatc):
            delta_penalty_hatc = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(delta_penalty_lnk):
            delta_penalty_lnk = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(jacobian_penalty):
            jacobian_penalty = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(jacobian_penalty_hatc):
            jacobian_penalty_hatc = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(jacobian_penalty_lnk):
            jacobian_penalty_lnk = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(mean_anchor_loss):
            logger.warning("Non-finite mean-anchor loss detected. Replace with 0.0 for stability.")
            mean_anchor_loss = torch.tensor(0.0, device=self.device)

        hj_warmup_factor = self._compute_stage2_hj_warmup_factor()
        moment_weight_eff = float(moment_weight) * hj_warmup_factor
        mean_anchor_weight_eff = float(mean_anchor_weight) * hj_warmup_factor
        fc1_forecast_only_ablation = bool(
            getattr(self.hyperparams, "fc1_forecast_only_ablation", False)
        )

        if (
            bool(getattr(self, "_fc1_teacher_forcing_stage", False))
            and self.add_FC1loss
        ):
            teacher_weight = float(getattr(self.hyperparams, "fc1_teacher_forcing_weight", 1.0))
            total_sdf_loss = teacher_weight * (
                recon_loss
                + forecast_recon_weight * law_consistency_loss
                + x_response_loss
                + delta_penalty_weight * delta_penalty
                + jacobian_penalty_weight * jacobian_penalty
            )
            moment_weight_eff = 0.0
            mean_anchor_weight_eff = 0.0
        elif fc1_forecast_only_ablation and self.add_FC1loss:
            total_sdf_loss = (
                recon_weight * recon_loss
                + forecast_recon_weight * law_consistency_loss
                + x_response_loss
                + delta_penalty_weight * delta_penalty
                + jacobian_penalty_weight * jacobian_penalty
            )
            moment_weight_eff = 0.0
            mean_anchor_weight_eff = 0.0
        else:
            total_sdf_loss = (
                main_loss
                + moment_weight_eff * moment_loss
                + recon_weight * recon_loss
                + forecast_recon_weight * law_consistency_loss
                + x_response_loss
                + delta_penalty_weight * delta_penalty
                + jacobian_penalty_weight * jacobian_penalty
                + mean_anchor_weight_eff * mean_anchor_loss
            )

        # 诊断：每步记录 M 的矩和 FC1 跨期增量分布
        with torch.no_grad():
            mu = M_use.mean().clamp_min(1e-8)
            var = ((M_use - mu) ** 2).mean().clamp_min(1e-8)
            d_hatcf = (c_children - c_parent.unsqueeze(1)).reshape(-1)
            d_lnkf = (k_children - k_parent.unsqueeze(1)).reshape(-1)

            def _q(v: torch.Tensor, q: float) -> float:
                return float(torch.quantile(v, q).item()) if v.numel() > 0 else 0.0

            self._latest_sdf_terms = {
                'sdf_main_loss': float(main_loss.detach().item()),
                'sdf_moment_loss': float(moment_loss.detach().item()),
                'sdf_recon_loss': float(recon_loss.detach().item()),
                'sdf_recon_loss_hatc': float(recon_loss_hatc.detach().item()),
                'sdf_recon_loss_lnk': float(recon_loss_lnk.detach().item()),
                'sdf_recon_loss_forecast': float(recon_loss_forecast.detach().item()),
                'sdf_recon_loss_forecast_hatc': float(recon_loss_forecast_hatc.detach().item()),
                'sdf_recon_loss_forecast_lnk': float(recon_loss_forecast_lnk.detach().item()),
                'sdf_law_consistency_loss': float(law_consistency_loss.detach().item()),
                'sdf_law_consistency_loss_hatc': float(law_consistency_loss_hatc.detach().item()),
                'sdf_law_consistency_loss_lnk': float(law_consistency_loss_lnk.detach().item()),
                'sdf_x_response_loss': float(x_response_loss.detach().item()),
                'sdf_x_response_loss_hatc': float(x_response_loss_hatc.detach().item()),
                'sdf_x_response_loss_lnk': float(x_response_loss_lnk.detach().item()),
                'sdf_delta_penalty': float(delta_penalty.detach().item()),
                'sdf_delta_penalty_hatc': float(delta_penalty_hatc.detach().item()),
                'sdf_delta_penalty_lnk': float(delta_penalty_lnk.detach().item()),
                'sdf_jacobian_penalty': float(jacobian_penalty.detach().item()),
                'sdf_jacobian_penalty_hatc': float(jacobian_penalty_hatc.detach().item()),
                'sdf_jacobian_penalty_lnk': float(jacobian_penalty_lnk.detach().item()),
                'sdf_mean_anchor_loss': float(mean_anchor_loss.detach().item()),
                'sdf_mean_anchor_weight': float(mean_anchor_weight_eff),
                'sdf_mean_anchor_target': (
                    float(mean_anchor_target) if mean_anchor_target is not None else float('nan')
                ),
                'sdf_moment_weight': float(moment_weight_eff),
                'sdf_forecast_recon_weight': float(forecast_recon_weight),
                'sdf_hatc_law_consistency_weight': float(hatc_law_consistency_weight),
                'sdf_lnk_law_consistency_weight': float(lnk_law_consistency_weight),
                'sdf_hatc_x_response_weight': float(hatc_x_response_weight),
                'sdf_lnk_x_response_weight': float(lnk_x_response_weight),
                'sdf_x_response_bins': float(x_response_bins),
                'sdf_hatc_recon_inner_weight': float(hatc_recon_inner_weight),
                'sdf_lnk_recon_inner_weight': float(lnk_recon_inner_weight),
                'sdf_delta_penalty_weight': float(delta_penalty_weight),
                'sdf_delta_hatc_abs_max': float(delta_hatc_abs_max),
                'sdf_delta_lnk_abs_max': float(delta_lnk_abs_max),
                'sdf_jacobian_penalty_weight': float(jacobian_penalty_weight),
                'sdf_hj_warmup_factor': float(hj_warmup_factor),
                'sdf_teacher_forcing_stage': float(1.0 if self._fc1_teacher_forcing_stage else 0.0),
                'sdf_forecast_only_ablation': float(1.0 if fc1_forecast_only_ablation else 0.0),
                'sdf_use_true_prev_macro': float(1.0 if use_true_prev_macro else 0.0),
            }
            self._latest_sdf_diag = {
                'sdf_log_mean_M': float(torch.log(mu).item()),
                'sdf_log_var_M': float(torch.log(var).item()),
                'sdf_dhatcf_mean': float(d_hatcf.mean().item()),
                'sdf_dhatcf_p10': _q(d_hatcf, 0.10),
                'sdf_dhatcf_p50': _q(d_hatcf, 0.50),
                'sdf_dhatcf_p90': _q(d_hatcf, 0.90),
                'sdf_dlnkf_mean': float(d_lnkf.mean().item()),
                'sdf_dlnkf_p10': _q(d_lnkf, 0.10),
                'sdf_dlnkf_p50': _q(d_lnkf, 0.50),
                'sdf_dlnkf_p90': _q(d_lnkf, 0.90),
            }

        return total_sdf_loss
    
    def _compute_p0_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算 P0 损失（接口名保留；语义上对应 conditional value V0）
        """
        model = self.models['policy_value']
        loss_fn = self.loss_fns['p0']
        
        parent = batch['parent']
        children = batch.get('children', [])
        strip_extra = lambda x: x[:, :7] if x.shape[1] > 7 else x
        
        # 兼容旧的 child0/child1 格式
        if not children:
            child0 = batch.get('child0')
            child1 = batch.get('child1')
            if child0 is not None and child1 is not None:
                children = [child0, child1]
        
        if not children:
            raise ValueError("No children data in batch")
        
        # 获取 SDF（优先用 batch 内的 M，避免重复计算）
        if parent.shape[1] > 7:
            raw_M_list = [child[:, 7:8] for child in children]
        else:
            raw_M_list = [torch.ones(parent.shape[0], 1, device=self.device) for _ in children]
        if bool(getattr(self.hyperparams, "pv_use_clipped_m", True)):
            m_lo = float(getattr(self.hyperparams, "pv_m_clamp_min", 0.7))
            m_hi = float(getattr(self.hyperparams, "pv_m_clamp_max", 1.3))
            M_list = [m.clamp(m_lo, m_hi) for m in raw_M_list]
        else:
            M_list = raw_M_list
        
        # 前向传播
        parent_state = strip_extra(parent).clone().detach().requires_grad_(True)
        output_t = model(parent_state)

        def _get_out(out, name: str, idx: int) -> torch.Tensor:
            alias = {
                'P0': 'V0',
                'PI': 'VI',
                'Phat': 'Vhat',
            }
            if isinstance(out, dict):
                if name in out:
                    return out[name]
                if name in alias and alias[name] in out:
                    return out[alias[name]]
                raise KeyError(name)
            if hasattr(out, name):
                return getattr(out, name)
            if name in alias and hasattr(out, alias[name]):
                return getattr(out, alias[name])
            return out[:, idx:idx + 1]

        bp0_t = _get_out(output_t, 'bp0', 1)
        bpI_t = _get_out(output_t, 'bpI', 2)
        bar_i_cond_t = _get_out(output_t, 'bar_i_cond', 4)
        chi_t = _get_out(output_t, 'chi', 3)
        bar_i_t = _get_out(output_t, 'bar_i', 4)
        bp_t = _get_out(output_t, 'bp', -1)
        if bp_t.shape != bp0_t.shape:
            bp_t = bar_i_t * bpI_t + (1 - bar_i_t) * bp0_t
        # P0 分支使用不投资场景的杠杆候选 bp0
        bp_for_p0 = bp0_t

        output_children = []
        eta_children = []

        for child in children:
            child_state_raw = strip_extra(child)
            eta_child = child[:, 2:3]
            child_state = child_state_raw.clone()
            # 按 main_4.tex 的 Bellman 口径，continuation 中的 debt argument
            # 应保持为当期选定的 contract b'，而不是 eta' 混合后的 realized debt。
            child_state[:, 0:1] = bp_for_p0
            output_children.append(model(child_state))
            eta_children.append(eta_child)
            
        childp0_state = parent_state.clone()
        childp0_state[:, 0:1] = bp_for_p0
        outputp0_children = model(childp0_state)
        
        # 左边对象按理论口径解释为 conditional no-invest value V0；
        # Bellman RHS 继续使用 child 的总股权价值 P_{t+1}。
        P0 = _get_out(output_t, 'P0', 3)
        Vhat_t = _get_out(output_t, 'Phat', 8)
        P_children = [_get_out(out, 'P', 7) for out in output_children]
        # FOC/KKT 梯度通道可选用 Phat，避免 P=max(Phat,0) 在违约区产生大面积零梯度
        use_phat_for_bp_foc = bool(getattr(self.hyperparams, "bp_foc_use_phat_children", True))
        P_children_for_foc = [
            _get_out(out, 'Phat', 8) if use_phat_for_bp_foc else _get_out(out, 'P', 7)
            for out in output_children
        ]
        bar_z_children = [_get_out(out, 'bar_z', 6) for out in output_children]
        
        # Q 值
        Q = _get_out(output_t, 'Q', 0)
        Qp = _get_out(outputp0_children, 'Q', 0)

        
        # 计算现金流与残差（逐 parent × child 对齐）
        CF0p = [
            loss_fn.compute_cashflow_p0(
                parent_state[:, 4:5],  # x
                parent_state[:, 1:2],  # z
                parent_state[:, 0:1],  # b
                Q, Qp,
                eta_j
            )
            for eta_j in eta_children
        ]
        residuals = loss_fn.compute_bellman_residual(
            P0, CF0p, M_list, P_children, bar_z_children
        )
        bellman_residual = compute_aio_residual(residuals, loss_fn.aio_weight)
        main_loss = bellman_residual.mean()

        penalty_z = compute_z_penalty(
            bellman_residual, parent_state[:, 1:2],
            loss_fn.alpha_z, loss_fn.beta_z, loss_fn.z0
        )

        foc_residuals = loss_fn.compute_foc_residual_from_bp(
            CF0p=CF0p,
            M_list=M_list,
            P_children=P_children_for_foc,
            bar_z_children=bar_z_children,
            bp=bp_for_p0,
            eta=eta_children
        )
        bp_surv_weight = self._compute_bp_survival_weight(
            P_children=P_children,
            bar_z_children=bar_z_children,
            eta_children=eta_children,
        )
        loss_foc, penalty_z_foc, foc_diag = self._compute_conditional_signed_foc_terms(
            foc_residuals=foc_residuals,
            eta_children=eta_children,
            z_parent=parent_state[:, 1:2],
            alpha_z=loss_fn.alpha_z,
            beta_z=loss_fn.beta_z,
            z0=loss_fn.z0,
            sample_weight=bp_surv_weight,
        )
        kkt_penalty_base, kkt_diag = self._compute_bp_kkt_penalty(
            bp_for_p0, foc_residuals, eta_children=eta_children, sample_weight=bp_surv_weight
        )
        p0_kkt_w = float(getattr(self.hyperparams, "p0_kkt_weight", 1.0))
        kkt_penalty = p0_kkt_w * kkt_penalty_base
        eta_active_boost = self._compute_eta_active_boost(foc_diag.get('foc_active_ratio', 0.0))
        bp_terms_base = loss_foc + penalty_z_foc + kkt_penalty
        bp_terms_after_eta = eta_active_boost * bp_terms_base
        bp_adapt_scale = self._compute_bp_adaptive_scale(main_loss, bp_terms_after_eta)
        bp_terms = bp_adapt_scale * bp_terms_after_eta
        bp_value_loss, bp_value_diag = self._compute_bp_value_supervision_loss(
            parent_state=parent_state,
            parent_Q=Q,
            bp_pred=bp_for_p0,
            M_list=M_list,
            eta_children=eta_children,
            branch_kind='p0',
        )
        bp_value_weight = float(getattr(self.hyperparams, "bp_value_weight", 1.0))

        mono_penalty, mono_diag = self._compute_value_gate_monotonicity_penalty(
            parent_state=parent_state,
            value=P0,
            chi=chi_t,
        )

        total_loss = main_loss + penalty_z + mono_penalty + bp_terms + bp_value_weight * bp_value_loss
        with torch.no_grad():
            raw_m = torch.cat([m.reshape(-1) for m in raw_M_list], dim=0)
            use_m = torch.cat([m.reshape(-1) for m in M_list], dim=0)
            self._latest_p0_terms = {
                'p0_main': float(main_loss.item()),
                'p0_foc': float(loss_foc.item()),
                'p0_penalty_z': float(penalty_z.item()),
                'p0_penalty_z_foc': float(penalty_z_foc.item()),
                'p0_kkt': float(kkt_penalty.item()),
                'p0_bp_terms_base': float(bp_terms_base.item()),
                'p0_bp_terms_after_eta': float(bp_terms_after_eta.item()),
                'p0_bp_terms': float(bp_terms.item()),
                'p0_eta_active_boost': float(eta_active_boost),
                'p0_bp_adapt_scale': float(bp_adapt_scale),
                'p0_kkt_inner': float(kkt_diag['kkt_inner']),
                'p0_kkt_low': float(kkt_diag['kkt_low']),
                'p0_kkt_high': float(kkt_diag['kkt_high']),
                'p0_kkt_foc_abs_mean': float(kkt_diag['kkt_foc_abs_mean']),
                'p0_kkt_active_ratio': float(kkt_diag['kkt_active_ratio']),
                'p0_kkt_high_weight': float(kkt_diag.get('kkt_high_weight', 0.0)),
                'p0_foc_active_ratio': float(foc_diag['foc_active_ratio']),
                'p0_foc_signed_moment': float(foc_diag['foc_signed_moment']),
                'p0_foc_cond_abs_mean': float(foc_diag['foc_cond_abs_mean']),
                'p0_foc_active_n': float(foc_diag.get('foc_active_n', 0.0)),
                'p0_foc_survival_weight_mean': float(foc_diag.get('foc_survival_weight_mean', 1.0)),
                'p0_bp_value_loss': float(bp_value_loss.item()),
                'p0_bp_value_weight': float(bp_value_weight),
                'p0_mono_penalty': float(mono_penalty.item()),
                'p0_log_mean_M_raw': float(torch.log(raw_m.mean().clamp_min(1e-8)).item()),
                'p0_log_mean_M_used': float(torch.log(use_m.mean().clamp_min(1e-8)).item()),
                'p0_M_raw_p90': float(torch.quantile(raw_m, 0.90).item()),
                'p0_M_used_p90': float(torch.quantile(use_m, 0.90).item()),
                'p0_bp_foc_use_phat': float(1.0 if use_phat_for_bp_foc else 0.0),
                'p0_bar_i_cond_mean': float(bar_i_cond_t.mean().item()),
                'p0_bar_i_eff_mean': float(bar_i_t.mean().item()),
                'p0_chi_mean': float(chi_t.mean().item()),
                'p0_vhat_mean': float(Vhat_t.mean().item()),
            }
            self._latest_p0_terms.update({f'p0_{k}': v for k, v in mono_diag.items()})
            self._latest_p0_terms.update(getattr(loss_fn, 'latest_foc_diag', {}))
            self._latest_p0_terms.update({f'p0_{k}': v for k, v in bp_value_diag.items() if k != 'bp_value_weight'})
        return total_loss
    
    def _compute_pi_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算 PI 损失（接口名保留；语义上对应 conditional value VI）
        """
        model = self.models['policy_value']
        loss_fn = self.loss_fns['pi']
        
        parent = batch['parent']
        children = batch.get('children', [])
        strip_extra = lambda x: x[:, :7] if x.shape[1] > 7 else x
        
        # 兼容旧的 child0/child1 格式
        if not children:
            child0 = batch.get('child0')
            child1 = batch.get('child1')
            if child0 is not None and child1 is not None:
                children = [child0, child1]
        
        if not children:
            raise ValueError("No children data in batch")
        
        # 获取 SDF（优先用 batch 内的 M，避免重复计算）
        if parent.shape[1] > 7:
            raw_M_list = [child[:, 7:8] for child in children]
        else:
            raw_M_list = [torch.ones(parent.shape[0], 1, device=self.device) for _ in children]
        if bool(getattr(self.hyperparams, "pv_use_clipped_m", True)):
            m_lo = float(getattr(self.hyperparams, "pv_m_clamp_min", 0.7))
            m_hi = float(getattr(self.hyperparams, "pv_m_clamp_max", 1.3))
            M_list = [m.clamp(m_lo, m_hi) for m in raw_M_list]
        else:
            M_list = raw_M_list
        
        # 前向传播
        parent_state = strip_extra(parent).clone().detach().requires_grad_(True)
        output_t = model(parent_state)

        def _get_out(out, name: str, idx: int) -> torch.Tensor:
            alias = {
                'P0': 'V0',
                'PI': 'VI',
                'Phat': 'Vhat',
            }
            if isinstance(out, dict):
                if name in out:
                    return out[name]
                if name in alias and alias[name] in out:
                    return out[alias[name]]
                raise KeyError(name)
            if hasattr(out, name):
                return getattr(out, name)
            if name in alias and hasattr(out, alias[name]):
                return getattr(out, alias[name])
            return out[:, idx:idx + 1]


        bp0_t = _get_out(output_t, 'bp0', 1)
        bpI_t = _get_out(output_t, 'bpI', 2)
        bar_i_cond_t = _get_out(output_t, 'bar_i_cond', 4)
        chi_t = _get_out(output_t, 'chi', 3)
        bar_i_t = _get_out(output_t, 'bar_i', 4)
        bp_t = _get_out(output_t, 'bp', -1)
        if bp_t.shape != bp0_t.shape:
            bp_t = bar_i_t * bpI_t + (1 - bar_i_t) * bp0_t
        # PI 分支使用投资场景的杠杆候选 bpI
        bp_for_pi = bpI_t

        output_children = []
        eta_children = []

        for child in children:
            child_state_raw = strip_extra(child)
            eta_child = child[:, 2:3]
            child_state = child_state_raw.clone()
            # 按 main_4.tex 的 Bellman 口径，continuation 中的 debt argument
            # 应保持为当期选定的 contract b'，而不是 eta' 混合后的 realized debt。
            child_state[:, 0:1] = bp_for_pi
            output_children.append(model(child_state))
            eta_children.append(eta_child)
            
        childpI_state = parent_state.clone()
        childpI_state[:, 0:1] = bp_for_pi
        outputpI_children = model(childpI_state)
        
        # 左边对象按理论口径解释为 conditional invest value VI；
        # Bellman RHS 继续使用 child 的总股权价值 P_{t+1}。
        Q = _get_out(output_t, 'Q', 0)
        PI = _get_out(output_t, 'PI', 4)
        Vhat_t = _get_out(output_t, 'Phat', 8)
        P_children = [_get_out(out, 'P', 7) for out in output_children]
        # FOC/KKT 梯度通道可选用 Phat，避免 P=max(Phat,0) 在违约区产生大面积零梯度
        use_phat_for_bp_foc = bool(getattr(self.hyperparams, "bp_foc_use_phat_children", True))
        P_children_for_foc = [
            _get_out(out, 'Phat', 8) if use_phat_for_bp_foc else _get_out(out, 'P', 7)
            for out in output_children
        ]
        bar_z_children = [_get_out(out, 'bar_z', 6) for out in output_children]
        QpI = _get_out(outputpI_children, 'Q', 0)
        
        # 提取 z 和 b
        z = parent[:, 1:2]
        b = parent[:, 0:1]
        
        # CFip (现金流）
        CFip = output_t.get('CFip', torch.zeros_like(PI)) if isinstance(output_t, dict) else torch.zeros_like(PI)
        
        # 计算现金流与残差（逐 parent × child 对齐）
        CFip = [
            loss_fn.compute_cashflow_pi(
                parent_state[:, 4:5],  # x
                parent_state[:, 1:2],  # z
                parent_state[:, 0:1],  # b
                parent_state[:, 3:4],  # i
                Q, QpI,
                eta_j
            )
            for eta_j in eta_children
        ]
        residuals = loss_fn.compute_bellman_residual(
            PI, CFip, M_list, P_children, bar_z_children
        )  # List[(batch,1)]
        bellman_residual = compute_aio_residual(residuals, loss_fn.aio_weight)
        main_loss = bellman_residual.mean()

        penalty_z = compute_z_penalty(
            bellman_residual, parent_state[:, 1:2],
            loss_fn.alpha_z, loss_fn.beta_z, loss_fn.z0
        )
        penalty_b = loss_fn.b_penalty_weight * loss_fn.compute_b_penalty(PI, parent_state[:, 0:1]).mean()

        foc_residuals = loss_fn.compute_foc_residual_from_bp(
            CFip=CFip,
            M_list=M_list,
            P_children=P_children_for_foc,
            bar_z_children=bar_z_children,
            bp=bp_for_pi,
            eta=eta_children
        )
        bp_surv_weight = self._compute_bp_survival_weight(
            P_children=P_children,
            bar_z_children=bar_z_children,
            eta_children=eta_children,
        )
        loss_foc, penalty_z_foc, foc_diag = self._compute_conditional_signed_foc_terms(
            foc_residuals=foc_residuals,
            eta_children=eta_children,
            z_parent=parent_state[:, 1:2],
            alpha_z=loss_fn.alpha_z,
            beta_z=loss_fn.beta_z,
            z0=loss_fn.z0,
            sample_weight=bp_surv_weight,
        )
        kkt_penalty_base, kkt_diag = self._compute_bp_kkt_penalty(
            bp_for_pi, foc_residuals, eta_children=eta_children, sample_weight=bp_surv_weight
        )
        pi_kkt_w = float(getattr(self.hyperparams, "pi_kkt_weight", 1.0))
        kkt_penalty = pi_kkt_w * kkt_penalty_base
        eta_active_boost = self._compute_eta_active_boost(foc_diag.get('foc_active_ratio', 0.0))
        bp_terms_base = loss_foc + penalty_z_foc + kkt_penalty
        bp_terms_after_eta = eta_active_boost * bp_terms_base
        bp_adapt_scale = self._compute_bp_adaptive_scale(main_loss, bp_terms_after_eta)
        bp_terms = bp_adapt_scale * bp_terms_after_eta
        bp_value_loss, bp_value_diag = self._compute_bp_value_supervision_loss(
            parent_state=parent_state,
            parent_Q=Q,
            bp_pred=bp_for_pi,
            M_list=M_list,
            eta_children=eta_children,
            branch_kind='pi',
        )
        bp_value_weight = float(getattr(self.hyperparams, "bp_value_weight", 1.0))

        mono_penalty, mono_diag = self._compute_value_gate_monotonicity_penalty(
            parent_state=parent_state,
            value=PI,
            chi=chi_t,
        )

        total_loss = main_loss + penalty_z + penalty_b + mono_penalty + bp_terms + bp_value_weight * bp_value_loss
        with torch.no_grad():
            raw_m = torch.cat([m.reshape(-1) for m in raw_M_list], dim=0)
            use_m = torch.cat([m.reshape(-1) for m in M_list], dim=0)
            self._latest_pi_terms = {
                'pi_main': float(main_loss.item()),
                'pi_foc': float(loss_foc.item()),
                'pi_penalty_z': float(penalty_z.item()),
                'pi_penalty_b': float(penalty_b.item()),
                'pi_penalty_z_foc': float(penalty_z_foc.item()),
                'pi_kkt': float(kkt_penalty.item()),
                'pi_bp_terms_base': float(bp_terms_base.item()),
                'pi_bp_terms_after_eta': float(bp_terms_after_eta.item()),
                'pi_bp_terms': float(bp_terms.item()),
                'pi_eta_active_boost': float(eta_active_boost),
                'pi_bp_adapt_scale': float(bp_adapt_scale),
                'pi_kkt_inner': float(kkt_diag['kkt_inner']),
                'pi_kkt_low': float(kkt_diag['kkt_low']),
                'pi_kkt_high': float(kkt_diag['kkt_high']),
                'pi_kkt_foc_abs_mean': float(kkt_diag['kkt_foc_abs_mean']),
                'pi_kkt_active_ratio': float(kkt_diag['kkt_active_ratio']),
                'pi_kkt_high_weight': float(kkt_diag.get('kkt_high_weight', 0.0)),
                'pi_foc_active_ratio': float(foc_diag['foc_active_ratio']),
                'pi_foc_signed_moment': float(foc_diag['foc_signed_moment']),
                'pi_foc_cond_abs_mean': float(foc_diag['foc_cond_abs_mean']),
                'pi_foc_active_n': float(foc_diag.get('foc_active_n', 0.0)),
                'pi_foc_survival_weight_mean': float(foc_diag.get('foc_survival_weight_mean', 1.0)),
                'pi_bp_value_loss': float(bp_value_loss.item()),
                'pi_bp_value_weight': float(bp_value_weight),
                'pi_mono_penalty': float(mono_penalty.item()),
                'pi_log_mean_M_raw': float(torch.log(raw_m.mean().clamp_min(1e-8)).item()),
                'pi_log_mean_M_used': float(torch.log(use_m.mean().clamp_min(1e-8)).item()),
                'pi_M_raw_p90': float(torch.quantile(raw_m, 0.90).item()),
                'pi_M_used_p90': float(torch.quantile(use_m, 0.90).item()),
                'pi_bp_foc_use_phat': float(1.0 if use_phat_for_bp_foc else 0.0),
                'pi_bar_i_cond_mean': float(bar_i_cond_t.mean().item()),
                'pi_bar_i_eff_mean': float(bar_i_t.mean().item()),
                'pi_chi_mean': float(chi_t.mean().item()),
                'pi_vhat_mean': float(Vhat_t.mean().item()),
            }
            self._latest_pi_terms.update({f'pi_{k}': v for k, v in mono_diag.items()})
            self._latest_pi_terms.update(getattr(loss_fn, 'latest_foc_diag', {}))
            self._latest_pi_terms.update({f'pi_{k}': v for k, v in bp_value_diag.items() if k != 'bp_value_weight'})
        return total_loss
    
    def _compute_q_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算 Q 损失（支持任意 N 分支）
        """
        model = self.models['policy_value']
        loss_fn = self.loss_fns['q']

        parent = batch['parent']
        children = batch.get('children', [])
        strip_extra = lambda x: x[:, :7] if x.shape[1] > 7 else x

        # 兼容旧的 child0/child1 格式
        if not children:
            child0 = batch.get('child0')
            child1 = batch.get('child1')
            if child0 is not None and child1 is not None:
                children = [child0, child1]

        if not children:
            raise ValueError("No children data in batch")

        # Q-only 阶段控制：用于“冻结非 Q 参数 + warm-start”
        q_only_stage = bool(getattr(self, "_q_only_stage", False))
        q_freeze_mode = q_only_stage and bool(
            getattr(self.hyperparams, "q_freeze_non_q_in_pretrain", True)
        )
        current_epoch = int(getattr(self, "_current_epoch_idx", 0))
        warm_epochs = max(0, int(getattr(self.hyperparams, "q_warmstart_epochs", 0)))
        use_warmstart = current_epoch < warm_epochs

        # 获取 SDF（优先用 batch 内的 M，避免重复计算）
        if parent.shape[1] > 7:
            raw_M_list = [child[:, 7:8] for child in children]
        else:
            raw_M_list = [torch.ones(parent.shape[0], 1, device=self.device) for _ in children]
        if getattr(self.hyperparams, "q_use_detached_m", True):
            m_lo = float(getattr(self.hyperparams, "q_m_clamp_min", 0.5))
            m_hi = float(getattr(self.hyperparams, "q_m_clamp_max", 1.5))
            M_list = [m.clamp(m_lo, m_hi).detach() for m in raw_M_list]
        else:
            M_list = raw_M_list

        # 前向传播（Q 形状正则需要对输入求梯度）
        parent_state = strip_extra(parent).clone().detach().requires_grad_(True)
        output_t = model(parent_state)

        def _get_out(out, name: str, idx: int) -> torch.Tensor:
            alias = {
                'P0': 'V0',
                'PI': 'VI',
                'Phat': 'Vhat',
            }
            if isinstance(out, dict):
                if name in out:
                    return out[name]
                if name in alias and alias[name] in out:
                    return out[alias[name]]
                raise KeyError(name)
            if hasattr(out, name):
                return getattr(out, name)
            if name in alias and hasattr(out, alias[name]):
                return getattr(out, alias[name])
            return out[:, idx:idx + 1]

        bp0_t = _get_out(output_t, 'bp0', 1)
        bpI_t = _get_out(output_t, 'bpI', 2)
        bar_i_cond_t = _get_out(output_t, 'bar_i_cond', 4)
        chi_t = _get_out(output_t, 'chi', 3)
        bar_i_t = _get_out(output_t, 'bar_i', 5)
        bp_t = _get_out(output_t, 'bp', -1)
        if bp_t.shape != bp0_t.shape:
            bp_t = bar_i_t * bpI_t + (1 - bar_i_t) * bp0_t
        b_parent = parent_state[:, 0:1]
        bar_z_t = _get_out(output_t, 'bar_z', 6)
        bar_i_use = bar_i_t
        bp_use = bp_t
        bar_z_use = bar_z_t
        # 与 q_loss 主方程保持一致：Qsp 输入使用 b' = b / (bar_i*(G-1)+1)
        g_val = float(getattr(loss_fn, "g", 1.0))
        multiplier = bar_i_use * (g_val - 1.0) + 1.0
        b_sp = b_parent / multiplier.clamp_min(1e-6)

        output_children = []
        outputsp_children = []
        for child in children:
            child_state_raw = strip_extra(child)
            eta_child = child[:, 2:3]
            child_state = child_state_raw.clone()
            child_state[:, 0:1] = eta_child * bp_use + (1 - eta_child) * b_parent
            output_children.append(model(child_state))

            childsp_state = child_state_raw.clone()
            childsp_state[:, 0:1] = b_sp
            outputsp_children.append(model(childsp_state))

        # 提取 Q 和所需变量
        Q = _get_out(output_t, 'Q', 0)
        Qsp_children = [_get_out(out, 'Q', 0) for out in outputsp_children]
        bar_zsp_children = [_get_out(out, 'bar_z', 6) for out in outputsp_children]
        x_children = [child[:, 4:5] for child in children]
        z_children = [child[:, 1:2] for child in children]

        # 构造分支残差：使用 AIO 稳定组合（而非纯乘积）
        z_parent = parent[:, 1:2]
        x_parent = parent[:, 4:5]

        residuals = loss_fn.compute_main_residual(
            Q, b_parent, bar_i_use, M_list, Qsp_children,
            bar_zsp_children, x_children, z_children
        )  # List[(batch,1)]
        aio_residual = compute_aio_residual(residuals, loss_fn.aio_weight)
        main_loss = aio_residual.mean()

        # 其余约束在 episode 层做聚合
        loss3 = loss_fn.compute_bar_z_constraint(Q, b_parent, x_parent, z_parent, bar_z_use).mean()
        loss4 = loss_fn.compute_boundary_loss_low(Q, b_parent).mean()
        loss5 = loss_fn.compute_boundary_loss_high(Q, b_parent, x_parent, z_parent).mean()
        penalty_z_main = compute_z_penalty(
            aio_residual, z_parent,
            loss_fn.alpha_z, loss_fn.beta_z, loss_fn.z0
        )
        penalty_z_loss3 = compute_z_penalty(
            (Q - loss_fn.compute_total_recovery(b_parent, x_parent, z_parent)).pow(2) * bar_z_use,
            z_parent, loss_fn.alpha_z, loss_fn.beta_z, loss_fn.z0
        )

        # q_unit 形状正则：
        # 1) dq_unit/dz > 0
        # 2) dq_unit/db < 0
        if hasattr(model, "get_q_unit"):
            q_unit = model.get_q_unit(parent_state)
        else:
            raise AttributeError(
                "PolicyValueModel is missing get_q_unit; please sync models/policy_value.py."
            )
        q_grads = torch.autograd.grad(
            outputs=q_unit.sum(),
            inputs=parent_state,
            create_graph=True,
            retain_graph=True
        )[0]
        dq_db = q_grads[:, 0:1]
        dq_dz = q_grads[:, 1:2]
        q_shape_z = torch.relu(-dq_dz).mean()
        q_shape_b = torch.relu(dq_db).mean()
        w_shape_z = float(getattr(self.hyperparams, "q_shape_weight_z", 1.0))
        w_shape_b_low = float(getattr(self.hyperparams, "q_shape_weight_b_low", 1.0))
        q_shape_penalty = (
            w_shape_z * q_shape_z +
            w_shape_b_low * q_shape_b
        )

        physics_loss = (
            main_loss + loss3 + loss4 + loss5 +
            penalty_z_main + penalty_z_loss3 + q_shape_penalty
        )
        warm_loss = torch.tensor(0.0, device=self.device)
        warm_weight = 0.0
        if use_warmstart:
            warm_weight = float(getattr(self.hyperparams, "q_warmstart_weight", 1.0))
            A = float(getattr(self.hyperparams, "q_warm_A", 1.0))
            b_star = float(getattr(self.hyperparams, "q_warm_b_star", 0.05))
            sigma = max(1e-6, float(getattr(self.hyperparams, "q_warm_sigma", 0.15)))
            alpha_z = float(getattr(self.hyperparams, "q_warm_alpha_z", 0.15))
            alpha_x = float(getattr(self.hyperparams, "q_warm_alpha_x", 0.15))
            b_nonneg = torch.clamp(b_parent, min=0.0)
            gaussian_peak = torch.exp(-0.5 * ((b_parent - b_star) / sigma).pow(2))
            risk_term = torch.exp((alpha_z * z_parent + alpha_x * x_parent).clamp(-10.0, 10.0))
            q_warm_target = (A * b_nonneg * gaussian_peak * risk_term).detach()
            warm_loss = (Q - q_warm_target).pow(2).mean()

        total_loss = physics_loss + warm_weight * warm_loss
        with torch.no_grad():
            self._latest_q_terms = {
                'q_main': float(main_loss.item()),
                'q_bdry_low': float(loss4.item()),
                'q_bdry_high': float(loss5.item()),
                'q_shape_z': float(q_shape_z.item()),
                'q_shape_b_low': float(q_shape_b.item()),
                'q_shape_b_high': 0.0,
                'q_physics': float(physics_loss.item()),
                'q_warmstart': float(warm_loss.item()),
                'q_warm_weight': float(warm_weight),
                'q_pretrain_mode': float(1.0 if q_only_stage else 0.0),
                'q_freeze_mode': float(1.0 if q_freeze_mode else 0.0),
                'q_bar_i_cond_mean': float(bar_i_cond_t.mean().item()),
                'q_bar_i_eff_mean': float(bar_i_t.mean().item()),
                'q_chi_mean': float(chi_t.mean().item()),
                'q_unit_mean': float(q_unit.mean().item()),
                'dq_unit_db_mean': float(dq_db.mean().item()),
                'dq_unit_dz_mean': float(dq_dz.mean().item()),
            }
        return total_loss
    
    def _compute_fc2_loss(self, batch) -> torch.Tensor:
        """
        计算 FC2 损失（使用 FC2train2.ipynb pipeline，batch 为 DataFrame）
        """
        hatc_model, lnk_model = self._get_fc2_models()
        if hatc_model is None or lnk_model is None:
            return torch.tensor(0.0, device=self.device)

        if 'policy_value' not in self.models or self.models['policy_value'] is None:
            return torch.tensor(0.0, device=self.device)

        pv_model = self.models['policy_value']
        default_full_n = getattr(self.hyperparams, 'fc2_full_n', 1000)
        self._latest_fc2_diag = {}

        if isinstance(batch, dict) and 'firm_table' in batch:
            pipe = FC2LossPipe(
                firm_table=batch['firm_table'],
                macro_table=batch.get('macro_table'),
                full_N=batch.get('full_N', default_full_n),
                entry_num=batch.get('entry_num', None),
                pv_chunk_size=getattr(self.hyperparams, 'fc2_pv_chunk_size', 10000),
                device=self.device,
            )
        elif isinstance(batch, pd.DataFrame):
            df = batch.copy()
            full_N = default_full_n
            entry_num = None
            if df['branch'].min() < 0:
                df['branch'] = df['branch'] + 1
            pipe = FC2LossPipe(
                df=df,
                full_N=full_N,
                entry_num=entry_num,
                pv_chunk_size=getattr(self.hyperparams, 'fc2_pv_chunk_size', 10000),
                device=self.device,
            )
        elif isinstance(batch, dict) and 'df' in batch:
            df = batch['df'].copy()
            full_N = batch.get('full_N', default_full_n)
            entry_num = batch.get('entry_num', None)
            if df['branch'].min() < 0:
                df['branch'] = df['branch'] + 1
            pipe = FC2LossPipe(
                df=df,
                full_N=full_N,
                entry_num=entry_num,
                pv_chunk_size=getattr(self.hyperparams, 'fc2_pv_chunk_size', 10000),
                device=self.device,
            )
        else:
            return torch.tensor(0.0, device=self.device)
        # keep for inspection/debugging
        self._last_fc2_pipe = pipe
        fc2_loss = pipe.loss(hatc_model, lnk_model, pv_model)
        if isinstance(fc2_loss, tuple):
            fc2_value, fc2_outputs = fc2_loss
            self._last_fc2_outputs = fc2_outputs
            diagnostics = fc2_outputs.get('diagnostics', {}) if isinstance(fc2_outputs, dict) else {}
            flat_diag: Dict[str, float] = {}
            for section_name in ('parent_hatc', 'parent_lnk', 'children_hatc', 'children_lnk'):
                section = diagnostics.get(section_name)
                if isinstance(section, dict):
                    prefix = f'fc2_{section_name}'
                    for k, v in section.items():
                        try:
                            flat_diag[f'{prefix}_{k}'] = float(v)
                        except (TypeError, ValueError):
                            continue
            for k in ('loss_parent', 'loss_children', 'loss_total'):
                if k in diagnostics:
                    try:
                        flat_diag[f'fc2_{k}'] = float(diagnostics[k])
                    except (TypeError, ValueError):
                        pass
            self._latest_fc2_diag = flat_diag
            fc2_loss = fc2_value
        return fc2_loss

    def _build_fc2_pipe(self, batch) -> Optional[FC2LossPipe]:
        default_full_n = getattr(self.hyperparams, 'fc2_full_n', 1000)
        if isinstance(batch, dict) and 'firm_table' in batch:
            return FC2LossPipe(
                firm_table=batch['firm_table'],
                macro_table=batch.get('macro_table'),
                full_N=batch.get('full_N', default_full_n),
                entry_num=batch.get('entry_num', None),
                pv_chunk_size=getattr(self.hyperparams, 'fc2_pv_chunk_size', 10000),
                device=self.device,
            )
        if isinstance(batch, pd.DataFrame):
            df = batch.copy()
            if df['branch'].min() < 0:
                df['branch'] = df['branch'] + 1
            return FC2LossPipe(
                df=df,
                full_N=default_full_n,
                entry_num=None,
                pv_chunk_size=getattr(self.hyperparams, 'fc2_pv_chunk_size', 10000),
                device=self.device,
            )
        if isinstance(batch, dict) and 'df' in batch:
            df = batch['df'].copy()
            if df['branch'].min() < 0:
                df['branch'] = df['branch'] + 1
            return FC2LossPipe(
                df=df,
                full_N=batch.get('full_N', default_full_n),
                entry_num=batch.get('entry_num', None),
                pv_chunk_size=getattr(self.hyperparams, 'fc2_pv_chunk_size', 10000),
                device=self.device,
            )
        return None

    def _compute_fc2_supervised_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        hatc_model, lnk_model = self._get_fc2_models()
        if hatc_model is None or lnk_model is None:
            return torch.tensor(0.0, device=self.device), {}
        pipe = self._build_fc2_pipe(batch)
        if pipe is None or pipe.macro_table is None:
            return torch.tensor(0.0, device=self.device), {}

        parent_x = pipe.build_fc2_input_parent_dual()
        parent_y = pipe.build_supervised_targets_parent()
        child_x = pipe.build_fc2_input_children_dual(pipe.child_states_list, pipe.child_present_list)
        child_y = pipe.build_supervised_targets_children()

        child_valid = torch.tensor(
            [
                bool(pipe.child_present_list[i][:, j].any().item())
                for i in range(pipe.path_num)
                for j in range(pipe.branch_num)
            ],
            device=self.device,
            dtype=torch.bool,
        )

        x_hatc_parts = [parent_x['hatc']]
        x_lnk_parts = [parent_x['lnk']]
        y_parts = [parent_y]
        child_x_hatc_flat = child_x['hatc'].reshape(-1, child_x['hatc'].shape[-1])
        child_x_lnk_flat = child_x['lnk'].reshape(-1, child_x['lnk'].shape[-1])
        child_y_flat = child_y.reshape(-1, child_y.shape[-1])
        if child_valid.any():
            x_hatc_parts.append(child_x_hatc_flat[child_valid])
            x_lnk_parts.append(child_x_lnk_flat[child_valid])
            y_parts.append(child_y_flat[child_valid])
        x_hatc_all = torch.cat(x_hatc_parts, dim=0)
        x_lnk_all = torch.cat(x_lnk_parts, dim=0)
        y_all = torch.cat(y_parts, dim=0)

        hatc_pred = hatc_model(x_hatc_all)
        lnk_pred = lnk_model(x_lnk_all)
        lnk_target = y_all[:, 0:1]
        hatc_target = y_all[:, 1:2]
        hatc_loss = nn.functional.mse_loss(hatc_pred, hatc_target)
        lnk_loss = nn.functional.mse_loss(lnk_pred, lnk_target)

        hatc_w = float(getattr(self.hyperparams, 'fc2_supervised_hatc_weight', 1.0))
        lnk_w = float(getattr(self.hyperparams, 'fc2_supervised_lnk_weight', 1.0))
        total = hatc_w * hatc_loss + lnk_w * lnk_loss

        with torch.no_grad():
            diag = {
                'fc2_pretrain_loss': float(total.item()),
                'fc2_pretrain_hatc_loss': float(hatc_loss.item()),
                'fc2_pretrain_lnk_loss': float(lnk_loss.item()),
            }
            for prefix, y_true, y_pred in (
                ('fc2_pretrain_hatc', hatc_target, hatc_pred),
                ('fc2_pretrain_lnk', lnk_target, lnk_pred),
            ):
                stats = compute_fit_stats_t(y_true, y_pred)
                for k, v in stats.items():
                    try:
                        diag[f'{prefix}_{k}'] = float(v)
                    except (TypeError, ValueError):
                        continue
        return total, diag

    def _record_manual_losses(self, losses: Dict[str, float]) -> None:
        self.step_count += 1
        for k, v in losses.items():
            if k not in self.loss_history:
                self.loss_history[k] = []
            self.loss_history[k].append(v)

    def _build_fc2_supervised_bundle_from_tables(
        self,
        firm_table: Optional[TensorTable],
        macro_table: Optional[TensorTable],
    ) -> Optional[FC2SupervisedBundle]:
        if firm_table is None or macro_table is None:
            return None
        pipe = FC2LossPipe(
            firm_table=firm_table,
            macro_table=macro_table,
            pv_chunk_size=getattr(self.hyperparams, 'fc2_pv_chunk_size', 10000),
            device=self.device,
        )
        parent_x = pipe.build_fc2_input_parent_dual()
        parent_y = pipe.build_supervised_targets_parent()
        child_x = pipe.build_fc2_input_children_dual(pipe.child_states_list, pipe.child_present_list)
        child_y = pipe.build_supervised_targets_children()

        child_valid = torch.tensor(
            [
                bool(pipe.child_present_list[i][:, j].any().item())
                for i in range(pipe.path_num)
                for j in range(pipe.branch_num)
            ],
            device=self.device,
            dtype=torch.bool,
        )

        parent_path_ids = torch.arange(pipe.path_num, device=self.device, dtype=torch.long)
        child_path_ids = torch.tensor(
            [i for i in range(pipe.path_num) for _ in range(pipe.branch_num)],
            device=self.device,
            dtype=torch.long,
        )
        x_hatc_parts = [parent_x['hatc']]
        x_lnk_parts = [parent_x['lnk']]
        y_parts = [parent_y]
        path_parts = [parent_path_ids]
        child_x_hatc_flat = child_x['hatc'].reshape(-1, child_x['hatc'].shape[-1])
        child_x_lnk_flat = child_x['lnk'].reshape(-1, child_x['lnk'].shape[-1])
        child_y_flat = child_y.reshape(-1, child_y.shape[-1])
        if child_valid.any():
            x_hatc_parts.append(child_x_hatc_flat[child_valid])
            x_lnk_parts.append(child_x_lnk_flat[child_valid])
            y_parts.append(child_y_flat[child_valid])
            path_parts.append(child_path_ids[child_valid])
        x_hatc_all = torch.cat(x_hatc_parts, dim=0)
        x_lnk_all = torch.cat(x_lnk_parts, dim=0)
        y_all = torch.cat(y_parts, dim=0)
        path_all = torch.cat(path_parts, dim=0)
        return FC2SupervisedBundle(
            x_hatc=x_hatc_all.detach(),
            x_lnk=x_lnk_all.detach(),
            y=y_all.detach(),
            path_ids=path_all.detach(),
        )

    def _build_fc2_supervised_dataset_from_tables(
        self,
        firm_table: Optional[TensorTable],
        macro_table: Optional[TensorTable],
    ) -> Optional[TensorDataset]:
        bundle = self._build_fc2_supervised_bundle_from_tables(firm_table, macro_table)
        if bundle is None:
            return None
        return self._fc2_bundle_to_dataset(bundle)

    def _build_fc2_supervised_bundle(self) -> Optional[FC2SupervisedBundle]:
        if not self._use_tensor_pipeline() or self.tensor_firm is None or self.tensor_macro is None:
            return None
        return self._build_fc2_supervised_bundle_from_tables(self.tensor_firm, self.tensor_macro)

    def _build_fc2_supervised_dataset(self) -> Optional[TensorDataset]:
        bundle = self._build_fc2_supervised_bundle()
        if bundle is None:
            return None
        return self._fc2_bundle_to_dataset(bundle)

    def _evaluate_fc2_dataset_fit(
        self,
        sup_dataset: Optional[TensorDataset],
        prefix: str,
        store_attr: Optional[str] = None,
    ) -> Optional[Dict[str, float]]:
        hatc_model, lnk_model = self._get_fc2_models()
        if hatc_model is None or lnk_model is None:
            return None
        if sup_dataset is None or len(sup_dataset) == 0:
            return None

        x_hatc_all, x_lnk_all, y_all = sup_dataset.tensors
        hatc_target = y_all[:, 1:2]
        lnk_target = y_all[:, 0:1]
        with torch.no_grad():
            hatc_pred = hatc_model(x_hatc_all)
            lnk_pred = lnk_model(x_lnk_all)

        if store_attr is not None:
            pred_df = pd.DataFrame(
                {
                    'hatc_true': hatc_target.detach().cpu().numpy().reshape(-1),
                    'hatc_pred': hatc_pred.detach().cpu().numpy().reshape(-1),
                    'lnk_true': lnk_target.detach().cpu().numpy().reshape(-1),
                    'lnk_pred': lnk_pred.detach().cpu().numpy().reshape(-1),
                }
            )
            setattr(self, store_attr, pred_df)

        diag: Dict[str, float] = {
            f'{prefix}_hatc_loss': float(nn.functional.mse_loss(hatc_pred, hatc_target).item()),
            f'{prefix}_lnk_loss': float(nn.functional.mse_loss(lnk_pred, lnk_target).item()),
        }
        for target_name, y_true, y_pred in (
            ('hatc', hatc_target, hatc_pred),
            ('lnk', lnk_target, lnk_pred),
        ):
            stats = compute_fit_stats_t(y_true, y_pred)
            for k, v in stats.items():
                try:
                    diag[f'{prefix}_{target_name}_{k}'] = float(v)
                except (TypeError, ValueError):
                    continue
        return diag

    def _evaluate_fc2_fit_after_train(
        self,
        sup_dataset: Optional[TensorDataset] = None,
    ) -> Optional[Dict[str, float]]:
        if sup_dataset is None:
            sup_dataset = self._build_fc2_supervised_dataset()
        return self._evaluate_fc2_dataset_fit(
            sup_dataset,
            prefix='fc2_fit_after_train',
            store_attr='_latest_fc2_fit_df',
        )

    def _run_fc2_supervised_pretrain_probe_lite(
        self,
        sup_bundle: FC2SupervisedBundle,
        n_epochs: int,
        log_interval: int,
    ) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
        hatc_model, lnk_model = self._get_fc2_models()
        if hatc_model is None or lnk_model is None or sup_bundle.y.shape[0] == 0:
            return [], {}

        train_bundle, val_bundle = self._split_fc2_supervised_bundle(sup_bundle)
        hatc_baseline_diag = self._fit_fc2_hatc_x_baseline(train_bundle)
        train_loader = DataLoader(
            self._fc2_bundle_to_dataset(train_bundle),
            batch_size=max(int(getattr(self.hyperparams, 'fc2_pretrain_batch_size', 4096)), 1),
            shuffle=True,
        )
        x_hatc_val = val_bundle.x_hatc
        x_lnk_val = val_bundle.x_lnk
        y_val = val_bundle.y
        hatc_w = float(getattr(self.hyperparams, 'fc2_supervised_hatc_weight', 1.0))
        lnk_w = float(getattr(self.hyperparams, 'fc2_supervised_lnk_weight', 1.0))
        patience = max(int(getattr(self.hyperparams, 'fc2_pretrain_patience', 30)), 1)
        lr = float(getattr(self.hyperparams, 'fc2_pretrain_lr', 1e-3))
        weight_decay = float(getattr(self.hyperparams, 'fc2_pretrain_weight_decay', self.hyperparams.fc2_weight_decay))
        disable_dropout = bool(getattr(self.hyperparams, 'fc2_pretrain_disable_dropout', True))

        hatc_optimizer = torch.optim.AdamW(hatc_model.parameters(), lr=lr, weight_decay=weight_decay)
        lnk_optimizer = torch.optim.AdamW(lnk_model.parameters(), lr=lr, weight_decay=weight_decay)

        prev_hatc_mode = hatc_model.training
        prev_lnk_mode = lnk_model.training
        best_val = float('inf')
        best_epoch = -1
        best_states: Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = None
        pretrain_epoch_losses: List[Dict[str, float]] = []

        try:
            for _ in tqdm(range(n_epochs), desc='FC2 Supervised Pretrain'):
                if disable_dropout:
                    hatc_model.eval()
                    lnk_model.eval()
                else:
                    hatc_model.train()
                    lnk_model.train()

                batch_losses = []
                for x_hatc_batch, x_lnk_batch, y_batch in train_loader:
                    hatc_optimizer.zero_grad(set_to_none=True)
                    lnk_optimizer.zero_grad(set_to_none=True)
                    hatc_pred = hatc_model(x_hatc_batch)
                    lnk_pred = lnk_model(x_lnk_batch)
                    lnk_target = y_batch[:, 0:1]
                    hatc_target = y_batch[:, 1:2]
                    hatc_loss = nn.functional.mse_loss(hatc_pred, hatc_target)
                    lnk_loss = nn.functional.mse_loss(lnk_pred, lnk_target)
                    sup_loss = hatc_w * hatc_loss + lnk_w * lnk_loss
                    sup_loss.backward()

                    hatc_grad_norm, hatc_had_nan = gradient_protection(
                        hatc_model.parameters(),
                        max_norm=getattr(self.hyperparams, 'fc2_max_grad_norm', self.hyperparams.max_grad_norm)
                    )
                    lnk_grad_norm, lnk_had_nan = gradient_protection(
                        lnk_model.parameters(),
                        max_norm=getattr(self.hyperparams, 'fc2_max_grad_norm', self.hyperparams.max_grad_norm)
                    )
                    if hatc_had_nan:
                        logger.warning("NaN gradient detected in fc2_hatc supervised pretrain")
                    if lnk_had_nan:
                        logger.warning("NaN gradient detected in fc2_lnk supervised pretrain")

                    hatc_optimizer.step()
                    lnk_optimizer.step()

                    batch_log = {
                        'fc2_pretrain_loss': float(sup_loss.item()),
                        'fc2_pretrain_hatc_loss': float(hatc_loss.item()),
                        'fc2_pretrain_lnk_loss': float(lnk_loss.item()),
                    }
                    with torch.no_grad():
                        for prefix, y_true, y_pred in (
                            ('fc2_pretrain_hatc', hatc_target, hatc_pred),
                            ('fc2_pretrain_lnk', lnk_target, lnk_pred),
                        ):
                            stats = compute_fit_stats_t(y_true, y_pred)
                            for k, v in stats.items():
                                try:
                                    batch_log[f'{prefix}_{k}'] = float(v)
                                except (TypeError, ValueError):
                                    continue
                    batch_log['fc2_hatc_grad_norm'] = hatc_grad_norm
                    batch_log['fc2_lnk_grad_norm'] = lnk_grad_norm
                    batch_log['fc2_grad_norm'] = float(max(hatc_grad_norm, lnk_grad_norm))
                    batch_log['total'] = float(sup_loss.item())
                    batch_log.update(hatc_baseline_diag)
                    self._latest_fc2_diag = batch_log
                    self._record_manual_losses(batch_log)
                    batch_losses.append(batch_log)

                hatc_model.eval()
                lnk_model.eval()
                with torch.no_grad():
                    hatc_pred_val = hatc_model(x_hatc_val)
                    lnk_pred_val = lnk_model(x_lnk_val)
                    lnk_target_val = y_val[:, 0:1]
                    hatc_target_val = y_val[:, 1:2]
                    hatc_val_loss = nn.functional.mse_loss(hatc_pred_val, hatc_target_val)
                    lnk_val_loss = nn.functional.mse_loss(lnk_pred_val, lnk_target_val)
                    val_total = hatc_w * hatc_val_loss + lnk_w * lnk_val_loss

                epoch_loss = {
                    k: float(np.mean([l[k] for l in batch_losses if k in l]))
                    for k in batch_losses[0].keys()
                }
                epoch_loss['fc2_pretrain_val_loss'] = float(val_total.item())
                epoch_loss['fc2_pretrain_val_hatc_loss'] = float(hatc_val_loss.item())
                epoch_loss['fc2_pretrain_val_lnk_loss'] = float(lnk_val_loss.item())
                with torch.no_grad():
                    for prefix, y_true, y_pred in (
                        ('fc2_pretrain_val_hatc', hatc_target_val, hatc_pred_val),
                        ('fc2_pretrain_val_lnk', lnk_target_val, lnk_pred_val),
                    ):
                        stats = compute_fit_stats_t(y_true, y_pred)
                        for k, v in stats.items():
                            try:
                                epoch_loss[f'{prefix}_{k}'] = float(v)
                            except (TypeError, ValueError):
                                continue
                pretrain_epoch_losses.append(epoch_loss)

                if val_total.item() < best_val:
                    best_val = float(val_total.item())
                    best_epoch = len(pretrain_epoch_losses) - 1
                    best_states = (
                        {k: v.detach().cpu().clone() for k, v in hatc_model.state_dict().items()},
                        {k: v.detach().cpu().clone() for k, v in lnk_model.state_dict().items()},
                    )
                elif (len(pretrain_epoch_losses) - 1) - best_epoch >= patience:
                    break

                if self.step_count % log_interval == 0:
                    avg_loss = np.mean([l['total'] for l in pretrain_epoch_losses[-log_interval:]])
                    logger.info(
                        "FC2 Supervised Pretrain Step %d: train_loss=%.6f, val_loss=%.6f, lr=%.2e",
                        self.step_count,
                        avg_loss,
                        float(val_total.item()),
                        lr,
                    )
        finally:
            hatc_model.train(prev_hatc_mode)
            lnk_model.train(prev_lnk_mode)

        if best_states is not None:
            hatc_model.load_state_dict(best_states[0])
            lnk_model.load_state_dict(best_states[1])
            hatc_model.train(prev_hatc_mode)
            lnk_model.train(prev_lnk_mode)

        if pretrain_epoch_losses:
            hatc_baseline_diag['fc2_pretrain_best_val_loss'] = float(best_val)
            hatc_baseline_diag['fc2_pretrain_best_epoch'] = float(best_epoch)
            hatc_baseline_diag['fc2_pretrain_train_paths'] = float(train_bundle.path_ids.unique().numel())
            hatc_baseline_diag['fc2_pretrain_val_paths'] = float(val_bundle.path_ids.unique().numel())
        return pretrain_epoch_losses, hatc_baseline_diag

    def _compute_fc2_outer_after_resim(
        self,
        n_paths: int,
        group_size: int,
        n_branches: int,
        horizon: int,
        simulate_kwargs: Dict,
    ) -> Optional[Dict[str, float]]:
        if not bool(getattr(self.hyperparams, 'fc2_as_main_macro_state', False)):
            return None
        if not self._has_fc2_models():
            return None
        sim_kwargs = dict(simulate_kwargs)
        sim_kwargs.setdefault('fc2_as_main_macro_state', True)
        simulator = SimulateTS(
            models=self.models,
            config=self.config,
            n_paths=n_paths,
            group_size=group_size,
            branch_num=n_branches,
            horizon=int(horizon),
            **sim_kwargs
        )
        out: TensorSimulationOutput = simulator.simulate_tensor()
        self._latest_fc2_outer_macro_panel_df = self._select_fc2_macro_window_df(out.macro)
        sup_dataset = self._build_fc2_supervised_dataset_from_tables(out.firm, out.macro)
        diag = self._evaluate_fc2_dataset_fit(
            sup_dataset,
            prefix='fc2_outer_after_resim',
            store_attr='_latest_fc2_outer_df',
        )
        if diag is None:
            return None
        diag.update(
            self._macro_consumption_diagnostics(
                self._latest_fc2_outer_macro_panel_df,
                prefix='fc2_outer_after_resim_macro_consumption',
            )
        )
        return diag

    def _capture_current_fc2_macro_panel(self) -> Dict[str, float]:
        macro_panel: Optional[pd.DataFrame | TensorTable]
        if self.df_macro is not None and not self.df_macro.empty:
            macro_panel = self.df_macro
        else:
            macro_panel = self.tensor_macro
        self._latest_fc2_current_macro_panel_df = self._select_fc2_macro_window_df(macro_panel)
        return self._macro_consumption_diagnostics(
            self._latest_fc2_current_macro_panel_df,
            prefix='fc2_current_macro_consumption',
        )
    
    def create_batches(
        self,
        batch_size: int = 1024,
        n_branches: int = 2
    ) -> List[Dict[str, torch.Tensor]]:
        """
        从 DataFrame 创建训练批次（支持任意 N 分支）
        
        Args:
            batch_size: 批大小
            n_branches: 分支数量（默认2）
        
        数据组织：每 (1 + n_branches) 行为一组
            - 第 0 行: parent
            - 第 1 ~ n_branches 行: children
        """
        if self._use_tensor_pipeline() and self.tensor_firm is not None:
            return self._create_firm_batches_from_tensor(
                self.tensor_firm, batch_size=batch_size, n_branches=n_branches
            )
        if self.df is None:
            raise RuntimeError("先调用 generate_data()")
        return self._create_firm_batches_from_df(
            self.df, batch_size=batch_size, n_branches=n_branches
        )

    def _create_sdf_batches_from_macro_df(
        self,
        df_sdf: pd.DataFrame,
        batch_size: int = 1024,
        n_branches: int = 2
    ) -> List[Dict[str, torch.Tensor]]:
        """
        从宏观跨期 DataFrame 创建 SDF 训练批次
        
        df_sdf 列要求：x_t, x_t1, Hatcf_t, LnKF_t, path, branch
        """
        parent_rows = []
        children_rows = [[] for _ in range(n_branches)]

        if self.add_FC1loss:
            if self.train_mode != '2time' and 't' in df_sdf.columns:
                df_sdf = df_sdf[df_sdf['t'] > 2]
            group_keys = ['path', 't'] if 't' in df_sdf.columns else ['path']
        else:
            group_keys = ['path']

        for _, group in df_sdf.groupby(group_keys):
            group = group.sort_values('branch')
            if len(group) < n_branches:
                continue
            x_t = group['x_t'].iloc[0]
            hatcf_t = group['Hatcf_t'].iloc[0]
            lnkf_t = group['LnKF_t'].iloc[0]
            
            if self.add_FC1loss:
                hatc_t = group['Hatc_t'].loc[group.branch == 0].iloc[0]
                lnk_t = group['LnK_t'].loc[group.branch == 0].iloc[0]
            
            # add_FC1loss=True 时保持 9 列布局：
            # [..., x, Hatcf, LnKF, Hatc_true, LnK_true]
            parent_rows.append(
                [0.0, 0.0, 0.0, 0.0, x_t, hatcf_t, lnkf_t, hatc_t, lnk_t]
                if self.add_FC1loss else
                [0.0, 0.0, 0.0, 0.0, x_t, hatcf_t, lnkf_t]
            )
            for k in range(n_branches):
                x_t1 = group['x_t1'].loc[group.branch == k].iloc[0]
                if self.add_FC1loss:
                    hatc_t1 = group['Hatc_t1'].loc[group.branch == k].iloc[0]
                    lnk_t1 = group['LnK_t1'].loc[group.branch == k].iloc[0]
                children_rows[k].append(
                    [0.0, 0.0, 0.0, 0.0, x_t1, 0.0, 0.0, hatc_t1, lnk_t1]
                    if self.add_FC1loss else
                    [0.0, 0.0, 0.0, 0.0, x_t1, 0.0, 0.0]
                )
        
        if not parent_rows:
            return []
        
        parent = torch.tensor(parent_rows, device=self.device, dtype=torch.float32)
        children = [
            torch.tensor(rows, device=self.device, dtype=torch.float32)
            for rows in children_rows
        ]
        
        n_units = len(parent)
        n_batches = (n_units + batch_size - 1) // batch_size
        indices = torch.randperm(n_units)
        batches = []
        
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_units)
            idx = indices[start:end]
            batch = {
                'parent': parent[idx],
                'children': [c[idx] for c in children],
                'child0': children[0][idx] if len(children) > 0 else None,
                'child1': children[1][idx] if len(children) > 1 else None
            }
            batches.append(batch)
        
        return batches

    def _create_firm_batches_from_df(
        self,
        df: pd.DataFrame,
        batch_size: int = 1024,
        n_branches: int = 2,
        eta_resample: bool = True
    ) -> List[Dict[str, torch.Tensor]]:
        """
        从 firm-level DataFrame 创建训练批次（兼容 sample 与 simulateTS）
        """
        input_cols = ['b', 'z', 'ETA', 'i', 'x', 'Hatcf', 'LnKF']
        if 'M' in df.columns:
            input_cols.append('M')
        if 't' not in df.columns or 'branch' not in df.columns:
            raise ValueError("DataFrame missing required columns: 't' and 'branch'")
        
        # Pandas string extension dtype (`string`) is not equal to `object`.
        # Use value-aware detection so sample-mode labels ('t', 't+1_k') are handled.
        t_non_null = df['t'].dropna()
        t_first = t_non_null.iloc[0] if len(t_non_null) > 0 else None
        t_is_str = pd.api.types.is_string_dtype(df['t']) or isinstance(t_first, str)
        if t_is_str:
            parent_df = df[df['t'] == 't'].copy()
            child_dfs = [
                df[df['t'] == f't+1_{k}'].copy() for k in range(n_branches)
            ]
            
            parent_df = parent_df.set_index(['path', 'ID'])
            child_dfs = [c.set_index(['path', 'ID']) for c in child_dfs]
            
            common_index = parent_df.index
            for child_df in child_dfs:
                common_index = common_index.intersection(child_df.index)
            
            parent_df = parent_df.loc[common_index]
            child_dfs = [c.loc[common_index] for c in child_dfs]
        else:
            parent_df = df[df['branch'] == -1].copy()
            child_df = df[df['branch'] >= 0].copy()
            child_index = child_df.set_index(['path', 'ID', 't', 'branch'])
            
            parent_rows = []
            child_rows = [[] for _ in range(n_branches)]
            for _, row in parent_df.iterrows():
                t_next = row['t'] + 1
                rows_for_parent = []
                for k in range(n_branches):
                    key = (row['path'], row['ID'], t_next, k)
                    if key not in child_index.index:
                        rows_for_parent = []
                        break
                    hit = child_index.loc[key]
                    if isinstance(hit, pd.DataFrame):
                        hit = hit.iloc[0]
                    hit_dict = hit.to_dict()
                    hit_dict['path'] = row['path']
                    hit_dict['ID'] = row['ID']
                    hit_dict['t'] = t_next
                    hit_dict['branch'] = k
                    rows_for_parent.append(hit_dict)
                if not rows_for_parent:
                    continue
                parent_rows.append(row)
                for k, hit in enumerate(rows_for_parent):
                    child_rows[k].append(hit)
            
            if not parent_rows:
                return []
            
            parent_df = pd.DataFrame(parent_rows).set_index(['path', 'ID'])
            child_dfs = [pd.DataFrame(rows).set_index(['path', 'ID']) for rows in child_rows]
        
        parent = torch.tensor(parent_df[input_cols].values, device=self.device, dtype=torch.float32)
        children = [
            torch.tensor(c[input_cols].values, device=self.device, dtype=torch.float32)
            for c in child_dfs
        ]
        
        n_units = len(parent)
        n_batches = (n_units + batch_size - 1) // batch_size
        indices = torch.arange(n_units, device=parent.device)

        # eta 稀疏时，对 Policy/Value 批次进行条件重采样，增强 eta=1 信号。
        resample_enabled = bool(getattr(self.hyperparams, "pv_eta_resample_enabled", True))
        if eta_resample and resample_enabled and n_units > 1 and len(children) > 0:
            eta_child_stack = torch.stack([c[:, 2:3] for c in children], dim=1)  # (B, N, 1)
            active_mask = (eta_child_stack.max(dim=1).values.squeeze(-1) > 0.5)
            active_idx = torch.where(active_mask)[0]
            inactive_idx = torch.where(~active_mask)[0]
            if active_idx.numel() > 0 and inactive_idx.numel() > 0:
                target_active_share = float(getattr(self.hyperparams, "pv_eta_resample_active_share", 0.25))
                target_active_share = min(max(target_active_share, 1e-3), 1.0 - 1e-3)
                n_active = int(round(n_units * target_active_share))
                n_active = min(max(1, n_active), n_units - 1)
                n_inactive = n_units - n_active

                active_pick = active_idx[torch.randint(0, active_idx.numel(), (n_active,), device=active_idx.device)]
                inactive_pick = inactive_idx[
                    torch.randint(0, inactive_idx.numel(), (n_inactive,), device=inactive_idx.device)
                ]
                indices = torch.cat([active_pick, inactive_pick], dim=0)
                indices = indices[torch.randperm(indices.numel(), device=indices.device)]
            else:
                indices = indices[torch.randperm(n_units, device=indices.device)]

        batches = []
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_units)
            idx = indices[start:end]
            batch = {
                'parent': parent[idx],
                'children': [c[idx] for c in children],
                'child0': children[0][idx] if len(children) > 0 else None,
                'child1': children[1][idx] if len(children) > 1 else None
            }
            batches.append(batch)
        
        return batches

    def _create_fc2_batches(
        self,
        df_firm: pd.DataFrame,
        batch_size: int = 1024,
        quantile_num: int = 100,
        n_branches: int = 2
    ) -> List[pd.DataFrame]:
        """
        从 firm-level DataFrame 创建 FC2 训练批次（每个 batch 是一个 DataFrame）
        batch_size 解释为每个 batch 的 path 数量。
        """
        if df_firm is None or df_firm.empty:
            return []
        if 'path' not in df_firm.columns:
            return [df_firm]

        paths = sorted(df_firm['path'].unique())
        if batch_size is None or batch_size <= 0:
            batch_size = len(paths)

        batches = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            batches.append(df_firm[df_firm['path'].isin(batch_paths)].copy())
        return batches

    def _create_fc2_batches_from_tensor(
        self,
        firm_table: TensorTable,
        macro_table: Optional[TensorTable] = None,
        batch_size: int = 256,
        shuffle: bool = True,
    ) -> List[Dict[str, TensorTable]]:
        """
        从 tensor-native simulate 输出创建 FC2 训练批次。
        batch_size 解释为每个 batch 的 path 数量。
        """
        if firm_table is None or len(firm_table) == 0:
            return []

        firm_cols = {name: i for i, name in enumerate(firm_table.columns)}
        if 'path' not in firm_cols:
            return [{'firm_table': firm_table, 'macro_table': macro_table}]

        firm_paths_all = torch.round(firm_table.data[:, firm_cols['path']]).to(torch.long)
        unique_paths = torch.unique(firm_paths_all, sorted=True)
        if unique_paths.numel() == 0:
            return []

        if shuffle and unique_paths.numel() > 1:
            perm = torch.randperm(unique_paths.numel(), device=unique_paths.device)
            unique_paths = unique_paths[perm]

        if batch_size is None or batch_size <= 0:
            batch_size = int(unique_paths.numel())

        macro_cols = {name: i for i, name in enumerate(macro_table.columns)} if macro_table is not None else {}
        macro_paths_all = None
        if macro_table is not None and 'path' in macro_cols:
            macro_paths_all = torch.round(macro_table.data[:, macro_cols['path']]).to(torch.long)

        batches: List[Dict[str, TensorTable]] = []
        for start in range(0, int(unique_paths.numel()), int(batch_size)):
            batch_paths = unique_paths[start:start + int(batch_size)]
            firm_mask = torch.isin(firm_paths_all, batch_paths)
            if not torch.any(firm_mask):
                continue
            firm_batch = TensorTable(
                data=firm_table.data[firm_mask],
                columns=list(firm_table.columns),
            )
            macro_batch = None
            if macro_table is not None:
                if macro_paths_all is None:
                    macro_batch = macro_table
                else:
                    macro_mask = torch.isin(macro_paths_all, batch_paths)
                    macro_batch = TensorTable(
                        data=macro_table.data[macro_mask],
                        columns=list(macro_table.columns),
                    )
            batches.append({
                'firm_table': firm_batch,
                'macro_table': macro_batch,
            })
        return batches

    def _get_policy_children(self, batch: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        统一获取 policy/value 所需 children 列表。
        """
        children = batch.get('children', [])
        if not children:
            child0 = batch.get('child0')
            child1 = batch.get('child1')
            if child0 is not None and child1 is not None:
                children = [child0, child1]
        return children

    @staticmethod
    def _policy_strip_extra(x: torch.Tensor) -> torch.Tensor:
        return x[:, :7] if x.shape[1] > 7 else x

    @staticmethod
    def _policy_get_out(out, name: str, idx: int) -> torch.Tensor:
        alias = {
            'P0': 'V0',
            'PI': 'VI',
            'Phat': 'Vhat',
        }
        if isinstance(out, dict):
            if name in out:
                return out[name]
            if name in alias and alias[name] in out:
                return out[alias[name]]
            raise KeyError(name)
        if hasattr(out, name):
            return getattr(out, name)
        if name in alias and hasattr(out, alias[name]):
            return getattr(out, alias[name])
        return out[:, idx:idx + 1]

    @staticmethod
    def _flatten_abs_residuals(residuals) -> torch.Tensor:
        """
        将分支残差压平为 |residual| 的一维向量（非 AIO 口径）。
        """
        chunks: List[torch.Tensor] = []
        if isinstance(residuals, (list, tuple)):
            for r in residuals:
                if r is None:
                    continue
                rr = torch.abs(r).reshape(-1)
                if rr.numel() > 0:
                    chunks.append(rr)
        elif torch.is_tensor(residuals):
            rr = torch.abs(residuals).reshape(-1)
            if rr.numel() > 0:
                chunks.append(rr)
        if not chunks:
            return torch.empty(0, device='cpu', dtype=torch.float32)
        return torch.cat(chunks, dim=0)

    def _compute_p0_bellman_abs_residual(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        model = self.models['policy_value']
        loss_fn = self.loss_fns['p0']

        parent = batch['parent']
        children = self._get_policy_children(batch)
        if not children:
            return torch.empty(0, device=self.device)

        if parent.shape[1] > 7:
            M_list = [child[:, 7:8] for child in children]
        else:
            M_list = [torch.ones(parent.shape[0], 1, device=self.device) for _ in children]

        parent_state = self._policy_strip_extra(parent)
        output_t = model(parent_state)

        bp0_t = self._policy_get_out(output_t, 'bp0', 1)
        bpI_t = self._policy_get_out(output_t, 'bpI', 2)
        bar_i_t = self._policy_get_out(output_t, 'bar_i', 4)
        bp_t = self._policy_get_out(output_t, 'bp', -1)
        if bp_t.shape != bp0_t.shape:
            bp_t = bar_i_t * bpI_t + (1 - bar_i_t) * bp0_t

        bp_for_p0 = bp0_t
        output_children = []
        eta_children = []
        for child in children:
            child_state_raw = self._policy_strip_extra(child)
            eta_child = child[:, 2:3]
            child_state = child_state_raw.clone()
            child_state[:, 0:1] = bp_for_p0
            output_children.append(model(child_state))
            eta_children.append(eta_child)

        childp0_state = parent_state.clone()
        childp0_state[:, 0:1] = bp_for_p0
        outputp0_children = model(childp0_state)

        P0 = self._policy_get_out(output_t, 'P0', 3)
        P_children = [self._policy_get_out(out, 'P', 7) for out in output_children]
        bar_z_children = [self._policy_get_out(out, 'bar_z', 6) for out in output_children]
        Q = self._policy_get_out(output_t, 'Q', 0)
        Qp = self._policy_get_out(outputp0_children, 'Q', 0)

        CF0p = [
            loss_fn.compute_cashflow_p0(
                parent_state[:, 4:5],
                parent_state[:, 1:2],
                parent_state[:, 0:1],
                Q,
                Qp,
                eta_j
            )
            for eta_j in eta_children
        ]
        residuals = loss_fn.compute_bellman_residual(P0, CF0p, M_list, P_children, bar_z_children)
        return self._flatten_abs_residuals(residuals)

    def _compute_pi_bellman_abs_residual(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        model = self.models['policy_value']
        loss_fn = self.loss_fns['pi']

        parent = batch['parent']
        children = self._get_policy_children(batch)
        if not children:
            return torch.empty(0, device=self.device)

        if parent.shape[1] > 7:
            M_list = [child[:, 7:8] for child in children]
        else:
            M_list = [torch.ones(parent.shape[0], 1, device=self.device) for _ in children]

        parent_state = self._policy_strip_extra(parent)
        output_t = model(parent_state)

        bp0_t = self._policy_get_out(output_t, 'bp0', 1)
        bpI_t = self._policy_get_out(output_t, 'bpI', 2)
        bar_i_t = self._policy_get_out(output_t, 'bar_i', 4)
        bp_t = self._policy_get_out(output_t, 'bp', -1)
        if bp_t.shape != bp0_t.shape:
            bp_t = bar_i_t * bpI_t + (1 - bar_i_t) * bp0_t

        bp_for_pi = bpI_t
        output_children = []
        eta_children = []
        for child in children:
            child_state_raw = self._policy_strip_extra(child)
            eta_child = child[:, 2:3]
            child_state = child_state_raw.clone()
            child_state[:, 0:1] = bp_for_pi
            output_children.append(model(child_state))
            eta_children.append(eta_child)

        childpI_state = parent_state.clone()
        childpI_state[:, 0:1] = bp_for_pi
        outputpI_children = model(childpI_state)

        Q = self._policy_get_out(output_t, 'Q', 0)
        PI = self._policy_get_out(output_t, 'PI', 4)
        P_children = [self._policy_get_out(out, 'P', 7) for out in output_children]
        bar_z_children = [self._policy_get_out(out, 'bar_z', 6) for out in output_children]
        QpI = self._policy_get_out(outputpI_children, 'Q', 0)

        CFip = [
            loss_fn.compute_cashflow_pi(
                parent_state[:, 4:5],
                parent_state[:, 1:2],
                parent_state[:, 0:1],
                parent_state[:, 3:4],
                Q,
                QpI,
                eta_j
            )
            for eta_j in eta_children
        ]
        residuals = loss_fn.compute_bellman_residual(PI, CFip, M_list, P_children, bar_z_children)
        return self._flatten_abs_residuals(residuals)

    def _compute_q_bellman_abs_residual(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        model = self.models['policy_value']
        loss_fn = self.loss_fns['q']

        parent = batch['parent']
        children = self._get_policy_children(batch)
        if not children:
            return torch.empty(0, device=self.device)

        if parent.shape[1] > 7:
            raw_M_list = [child[:, 7:8] for child in children]
        else:
            raw_M_list = [torch.ones(parent.shape[0], 1, device=self.device) for _ in children]
        if getattr(self.hyperparams, "q_use_detached_m", True):
            m_lo = float(getattr(self.hyperparams, "q_m_clamp_min", 0.5))
            m_hi = float(getattr(self.hyperparams, "q_m_clamp_max", 1.5))
            M_list = [m.clamp(m_lo, m_hi) for m in raw_M_list]
        else:
            M_list = raw_M_list

        parent_state = self._policy_strip_extra(parent)
        output_t = model(parent_state)

        bp0_t = self._policy_get_out(output_t, 'bp0', 1)
        bpI_t = self._policy_get_out(output_t, 'bpI', 2)
        bar_i_t = self._policy_get_out(output_t, 'bar_i', 5)
        bp_t = self._policy_get_out(output_t, 'bp', -1)
        if bp_t.shape != bp0_t.shape:
            bp_t = bar_i_t * bpI_t + (1 - bar_i_t) * bp0_t

        b_parent = parent_state[:, 0:1]
        bar_i_use = bar_i_t
        g_val = float(getattr(loss_fn, "g", 1.0))
        multiplier = bar_i_use * (g_val - 1.0) + 1.0
        b_sp = b_parent / multiplier.clamp_min(1e-6)

        outputsp_children = []
        for child in children:
            child_state_raw = self._policy_strip_extra(child)
            childsp_state = child_state_raw.clone()
            childsp_state[:, 0:1] = b_sp
            outputsp_children.append(model(childsp_state))

        Q = self._policy_get_out(output_t, 'Q', 0)
        Qsp_children = [self._policy_get_out(out, 'Q', 0) for out in outputsp_children]
        bar_zsp_children = [self._policy_get_out(out, 'bar_z', 6) for out in outputsp_children]
        x_children = [child[:, 4:5] for child in children]
        z_children = [child[:, 1:2] for child in children]

        residuals = loss_fn.compute_main_residual(
            Q, b_parent, bar_i_use, M_list, Qsp_children,
            bar_zsp_children, x_children, z_children
        )
        return self._flatten_abs_residuals(residuals)

    def evaluate_bellman_convergence(
        self,
        batches: List[Dict[str, torch.Tensor]],
        mean_threshold: Optional[float] = None,
        p90_threshold: Optional[float] = None
    ) -> Dict:
        """
        评估 P0/PI/Q 的 Bellman 主残差收敛（非 AIO 口径）。
        """
        mean_thr = float(
            mean_threshold
            if mean_threshold is not None
            else getattr(self.hyperparams, "bellman_conv_mean_thresh", 1e-3)
        )
        p90_thr = float(
            p90_threshold
            if p90_threshold is not None
            else getattr(self.hyperparams, "bellman_conv_p90_thresh", 5e-3)
        )

        if not batches or 'policy_value' not in self.models or self.models['policy_value'] is None:
            return {
                'enabled': False,
                'passed': False,
                'thresholds': {'mean': mean_thr, 'p90': p90_thr},
                'equations': {}
            }

        model = self.models['policy_value']
        was_training = model.training
        model.eval()

        p0_chunks: List[torch.Tensor] = []
        pi_chunks: List[torch.Tensor] = []
        q_chunks: List[torch.Tensor] = []

        with torch.no_grad():
            for idx, batch in enumerate(batches):
                try:
                    p0_abs = self._compute_p0_bellman_abs_residual(batch)
                    pi_abs = self._compute_pi_bellman_abs_residual(batch)
                    q_abs = self._compute_q_bellman_abs_residual(batch)
                except Exception as exc:
                    logger.warning("Bellman convergence eval skip batch %d due to error: %s", idx, exc)
                    continue
                if p0_abs.numel() > 0:
                    p0_chunks.append(p0_abs.detach())
                if pi_abs.numel() > 0:
                    pi_chunks.append(pi_abs.detach())
                if q_abs.numel() > 0:
                    q_chunks.append(q_abs.detach())

        if was_training:
            model.train()

        def _summarize(name: str, chunks: List[torch.Tensor]) -> Dict:
            if not chunks:
                return {
                    'enabled': False,
                    'n': 0,
                    'mean': float('nan'),
                    'p90': float('nan'),
                    'passed': False
                }
            vals = torch.cat(chunks, dim=0).reshape(-1).to(torch.float32)
            finite_mask = torch.isfinite(vals)
            vals = vals[finite_mask]
            if vals.numel() == 0:
                return {
                    'enabled': False,
                    'n': 0,
                    'mean': float('nan'),
                    'p90': float('nan'),
                    'passed': False
                }
            mean_v = float(vals.mean().item())
            p90_v = float(torch.quantile(vals, 0.9).item())
            passed = bool(mean_v < mean_thr and p90_v < p90_thr)
            logger.info(
                "Bellman convergence [%s] | mean(abs)=%.6e, p90(abs)=%.6e, pass=%s",
                name,
                mean_v,
                p90_v,
                str(passed)
            )
            return {
                'enabled': True,
                'n': int(vals.numel()),
                'mean': mean_v,
                'p90': p90_v,
                'passed': passed
            }

        equations = {
            'p0': _summarize('p0', p0_chunks),
            'pi': _summarize('pi', pi_chunks),
            'q': _summarize('q', q_chunks)
        }

        enabled_eq = [m for m in equations.values() if m.get('enabled', False)]
        all_passed = bool(enabled_eq) and all(m.get('passed', False) for m in enabled_eq)
        summary = {
            'enabled': True,
            'thresholds': {'mean': mean_thr, 'p90': p90_thr},
            'equations': equations,
            'passed': all_passed
        }
        logger.info(
            "Bellman convergence summary | mean<%.3e, p90<%.3e, passed=%s",
            mean_thr,
            p90_thr,
            str(all_passed)
        )
        return summary

    def _run_batches(
        self,
        batches: List[Dict[str, torch.Tensor]],
        n_epochs: int,
        log_interval: int,
        train_modules: List[str],
        desc_prefix: str = '',
        policy_stage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict:
        """
        使用预生成的 batches 执行训练循环
        """
        if 'sdf_fc1' in train_modules:
            self._configure_sdf_lr_for_phase()

        policy_staged_training = bool(
            'policy_value' in train_modules and
            getattr(self.hyperparams, "policy_separate_q_pvbp_training", True)
        )
        if policy_staged_training:
            q_stage_cfg = max(
                int(getattr(self.hyperparams, "q_stage_epochs", 100)),
                int(getattr(self.hyperparams, "q_pretrain_epochs", 0)),
                int(getattr(self.hyperparams, "q_warmstart_epochs", 0)),
            )
            pvbp_stage_cfg = int(getattr(self.hyperparams, "pvbp_stage_epochs", 100))
            q_refresh_cfg = int(getattr(self.hyperparams, "q_refresh_stage_epochs", 0))
            q_stage_epochs = 0 if q_stage_cfg <= 0 else max(100, q_stage_cfg)
            pvbp_stage_epochs = 0 if pvbp_stage_cfg <= 0 else max(100, pvbp_stage_cfg)
            q_refresh_stage_epochs = 0 if q_refresh_cfg <= 0 else q_refresh_cfg
            total_epochs = max(n_epochs, q_stage_epochs + pvbp_stage_epochs + q_refresh_stage_epochs)
        else:
            q_stage_epochs = 0
            pvbp_stage_epochs = 0
            q_refresh_stage_epochs = 0
            total_epochs = n_epochs

        stage_summaries: Dict[str, Dict[str, Any]] = {}
        for epoch in range(total_epochs):
            self._current_epoch_idx = epoch
            self._q_only_stage = bool(policy_staged_training and q_stage_epochs > 0 and epoch < q_stage_epochs)
            self._pvbp_only_stage = bool(
                policy_staged_training and
                pvbp_stage_epochs > 0 and
                q_stage_epochs <= epoch < (q_stage_epochs + pvbp_stage_epochs)
            )
            self._q_refresh_stage = bool(
                policy_staged_training and
                q_refresh_stage_epochs > 0 and
                epoch >= (q_stage_epochs + pvbp_stage_epochs)
            )
            self._set_policy_runtime_controls(q_stage_epochs)
            policy_loss_terms = None
            if policy_staged_training:
                if self._q_only_stage or self._q_refresh_stage:
                    policy_loss_terms = ['q']
                elif self._pvbp_only_stage:
                    policy_loss_terms = ['p0', 'pi']
            epoch_losses = []
            pvbp_alternating = bool(
                self._pvbp_only_stage and
                getattr(self.hyperparams, "pvbp_alternating_enabled", True)
            )

            def _run_epoch_pass(
                pass_batches: List[Dict[str, torch.Tensor]],
                pass_desc: str,
                pass_train_modules: List[str],
                pass_policy_loss_terms: Optional[List[str]],
                value_only: bool = False,
                bp_only: bool = False,
            ) -> List[Dict[str, float]]:
                pass_losses: List[Dict[str, float]] = []
                prev_value_only = self._value_only_stage
                prev_bp_only = self._bp_only_stage
                self._value_only_stage = value_only
                self._bp_only_stage = bp_only
                try:
                    for batch in tqdm(pass_batches, desc=pass_desc):
                        losses = self.train_step(
                            batch,
                            pass_train_modules,
                            policy_loss_terms=pass_policy_loss_terms
                        )
                        pass_losses.append(losses)
                        epoch_losses.append(losses)

                        if self.step_count % log_interval == 0:
                            recent = epoch_losses[-min(log_interval, len(epoch_losses)):]
                            avg_loss = np.mean([l['total'] for l in recent])
                            lr = None
                            optimizer_keys = self._resolve_optimizer_keys(
                                pass_train_modules,
                                pass_policy_loss_terms
                            )
                            for name in optimizer_keys:
                                lr = self.lr_schedulers.get(name)
                                if lr is not None:
                                    break
                            current_lr = lr.get_lr() if lr else 0
                            logger.info(
                                f"Step {self.step_count}: "
                                f"loss={avg_loss:.6f}, lr={current_lr:.2e}"
                            )
                finally:
                    self._value_only_stage = prev_value_only
                    self._bp_only_stage = prev_bp_only
                return pass_losses

            if pvbp_alternating:
                value_steps = max(1, int(getattr(self.hyperparams, "pvbp_value_steps_per_epoch", 1)))
                policy_steps = max(1, int(getattr(self.hyperparams, "pvbp_policy_steps_per_epoch", 1)))
                for v in range(value_steps):
                    _run_epoch_pass(
                        batches,
                        f"{desc_prefix}PVBP value {v+1}/{value_steps} (epoch {epoch+1}/{total_epochs})",
                        train_modules,
                        ['p0', 'pi'],
                        value_only=True,
                        bp_only=False,
                    )
                for p in range(policy_steps):
                    _run_epoch_pass(
                        batches,
                        f"{desc_prefix}PVBP policy {p+1}/{policy_steps} (epoch {epoch+1}/{total_epochs})",
                        train_modules,
                        ['p0', 'pi'],
                        value_only=False,
                        bp_only=True,
                    )
            else:
                _run_epoch_pass(
                    batches,
                    f"{desc_prefix}Epoch {epoch+1}/{total_epochs}",
                    train_modules,
                    policy_loss_terms,
                )

            # 每个 epoch 追加 bp-only 精修：只优化 P0/PI 且仅更新 bp 头。
            bp_refine_steps = max(0, int(getattr(self.hyperparams, "bp_refine_steps_per_epoch", 0)))
            bp_refine_cap = int(getattr(self.hyperparams, "bp_refine_batch_cap", 32))
            if (
                'policy_value' in train_modules and
                not self._q_only_stage and
                not pvbp_alternating and
                bp_refine_steps > 0 and
                len(batches) > 0
            ):
                if bp_refine_cap > 0:
                    refine_batches = batches[:min(bp_refine_cap, len(batches))]
                else:
                    refine_batches = batches
                self._bp_only_stage = True
                try:
                    for r in range(bp_refine_steps):
                        refine_losses = []
                        for batch in tqdm(
                            refine_batches,
                            desc=f"{desc_prefix}BP refine {r+1}/{bp_refine_steps} (epoch {epoch+1})"
                        ):
                            losses = self.train_step(
                                batch,
                                train_modules=['policy_value'],
                                policy_loss_terms=['p0', 'pi']
                            )
                            refine_losses.append(losses)
                            epoch_losses.append(losses)
                        if refine_losses:
                            refine_avg = {
                                k: np.mean([l[k] for l in refine_losses if k in l])
                                for k in refine_losses[0].keys()
                            }
                            logger.info(
                                "%sBP refine %d/%d finished: %s",
                                desc_prefix,
                                r + 1,
                                bp_refine_steps,
                                refine_avg
                            )
                finally:
                    self._bp_only_stage = False
                
            avg_losses = {
                k: np.mean([l[k] for l in epoch_losses if k in l])
                for k in epoch_losses[0].keys()
            }
            logger.info(f"{desc_prefix}Epoch {epoch+1} finished: {avg_losses}")
            if 'sdf_log_mean_M' in avg_losses:
                logger.info(
                    f"{desc_prefix}SDF Diagnostics | "
                    f"logE[M]={avg_losses['sdf_log_mean_M']:.4f}, "
                    f"logVar[M]={avg_losses['sdf_log_var_M']:.4f}, "
                    f"dHatcf[p10,p50,p90]=({avg_losses['sdf_dhatcf_p10']:.4f}, "
                    f"{avg_losses['sdf_dhatcf_p50']:.4f}, {avg_losses['sdf_dhatcf_p90']:.4f}), "
                    f"dLnKF[p10,p50,p90]=({avg_losses['sdf_dlnkf_p10']:.4f}, "
                    f"{avg_losses['sdf_dlnkf_p50']:.4f}, {avg_losses['sdf_dlnkf_p90']:.4f})"
                )
            if (
                policy_staged_training and
                epoch + 1 == q_stage_epochs
            ):
                q_only_summary = {
                    'phase': 'q_stage_end',
                    'epoch': epoch + 1,
                    'n_epochs': total_epochs,
                    'q_stage_epochs': q_stage_epochs,
                    'pvbp_stage_epochs': pvbp_stage_epochs,
                    'q_refresh_stage_epochs': q_refresh_stage_epochs,
                    'final_losses': dict(avg_losses),
                }
                stage_summaries['q_stage_end'] = q_only_summary
                if policy_stage_callback is not None:
                    policy_stage_callback(q_only_summary)
            if (
                policy_staged_training and
                pvbp_stage_epochs > 0 and
                epoch + 1 == q_stage_epochs + pvbp_stage_epochs
            ):
                pvbp_summary = {
                    'phase': 'pvbp_stage_end',
                    'epoch': epoch + 1,
                    'n_epochs': total_epochs,
                    'q_stage_epochs': q_stage_epochs,
                    'pvbp_stage_epochs': pvbp_stage_epochs,
                    'q_refresh_stage_epochs': q_refresh_stage_epochs,
                    'final_losses': dict(avg_losses),
                }
                stage_summaries['pvbp_stage_end'] = pvbp_summary
                if policy_stage_callback is not None:
                    policy_stage_callback(pvbp_summary)
            if (
                policy_staged_training and
                q_refresh_stage_epochs > 0 and
                epoch + 1 == q_stage_epochs + pvbp_stage_epochs + q_refresh_stage_epochs
            ):
                q_refresh_summary = {
                    'phase': 'q_refresh_end',
                    'epoch': epoch + 1,
                    'n_epochs': total_epochs,
                    'q_stage_epochs': q_stage_epochs,
                    'pvbp_stage_epochs': pvbp_stage_epochs,
                    'q_refresh_stage_epochs': q_refresh_stage_epochs,
                    'final_losses': dict(avg_losses),
                }
                stage_summaries['q_refresh_end'] = q_refresh_summary
                if policy_stage_callback is not None:
                    policy_stage_callback(q_refresh_summary)
        self._q_only_stage = False
        self._q_refresh_stage = False
        self._pvbp_only_stage = False
        self._bp_only_stage = False
        self._value_only_stage = False
        self._set_policy_runtime_controls(0)
        convergence = None
        if 'policy_value' in train_modules and 'policy_value' in self.models:
            convergence = self.evaluate_bellman_convergence(batches)

        result = {
            'final_losses': avg_losses
        }
        if convergence is not None:
            result['convergence'] = convergence
        if 'policy_value' in train_modules:
            if policy_staged_training and q_refresh_stage_epochs > 0 and total_epochs > (q_stage_epochs + pvbp_stage_epochs):
                final_phase = 'q_refresh_end'
            elif policy_staged_training and total_epochs > q_stage_epochs:
                final_phase = 'pvbp_stage_end'
            elif policy_staged_training:
                final_phase = 'q_stage_end'
            else:
                final_phase = 'policy_end'
            final_summary = {
                'phase': final_phase,
                'epoch': total_epochs,
                'n_epochs': total_epochs,
                'q_stage_epochs': q_stage_epochs,
                'pvbp_stage_epochs': pvbp_stage_epochs,
                'q_refresh_stage_epochs': q_refresh_stage_epochs,
                'final_losses': dict(avg_losses),
            }
            if convergence is not None:
                final_summary['convergence'] = convergence
            if final_phase in stage_summaries:
                stage_summaries[final_phase].update(final_summary)
            else:
                stage_summaries[final_phase] = final_summary
                if policy_stage_callback is not None:
                    policy_stage_callback(final_summary)
        if stage_summaries:
            result['stage_summaries'] = stage_summaries
        return result

    def _simulate_df(
        self,
        n_paths: int,
        group_size: int,
        n_branches: int,
        horizon: int,
        simulate_kwargs: Dict
    ) -> None:
        sim_kwargs = dict(simulate_kwargs)
        sim_kwargs.setdefault('fc2_as_main_macro_state', getattr(self.hyperparams, 'fc2_as_main_macro_state', False))
        simulator = SimulateTS(
            models=self.models,
            config=self.config,
            n_paths=n_paths,
            group_size=group_size,
            branch_num=n_branches,
            horizon=int(horizon),
            **sim_kwargs
        )
        self.df, self.df_macro = simulator.simulate()
        self.tensor_firm = None
        self.tensor_macro = None

    def _simulate_tensor(
        self,
        n_paths: int,
        group_size: int,
        n_branches: int,
        horizon: int,
        simulate_kwargs: Dict,
        export_df: bool = False
    ) -> None:
        sim_kwargs = dict(simulate_kwargs)
        sim_kwargs.setdefault('fc2_as_main_macro_state', getattr(self.hyperparams, 'fc2_as_main_macro_state', False))
        simulator = SimulateTS(
            models=self.models,
            config=self.config,
            n_paths=n_paths,
            group_size=group_size,
            branch_num=n_branches,
            horizon=int(horizon),
            **sim_kwargs
        )
        out: TensorSimulationOutput = simulator.simulate_tensor()
        self.tensor_firm = out.firm
        self.tensor_macro = out.macro
        if export_df:
            self.df, self.df_macro = out.to_dataframes()
            for col in ['path', 't', 'branch']:
                if col in self.df.columns:
                    self.df[col] = np.rint(self.df[col]).astype(np.int64)
                if col in self.df_macro.columns:
                    self.df_macro[col] = np.rint(self.df_macro[col]).astype(np.int64)
            if 'ID' in self.df.columns:
                self.df['ID'] = np.rint(self.df['ID']).astype(np.int64).astype(str)
            if 'n_firms' in self.df_macro.columns:
                self.df_macro['n_firms'] = np.rint(self.df_macro['n_firms']).astype(np.int64)
        else:
            self.df = None
            self.df_macro = None

    def _run_fc2_epochs(
        self,
        n_epochs: int,
        log_interval: int
    ) -> Optional[Dict]:
        if not self._has_fc2_models():
            return None
        pretrain_epoch_losses = []
        epoch_losses = []
        pv_model = self.models.get('policy_value')
        freeze_pv = bool(getattr(self.hyperparams, 'fc2_freeze_pv_during_epochs', True))
        fc2_path_batch_size = int(getattr(self.hyperparams, 'fc2_path_batch_size', 256))
        pretrain_epochs = int(getattr(self.hyperparams, 'fc2_supervised_pretrain_epochs', 0))
        pretrain_only = bool(getattr(self.hyperparams, 'fc2_supervised_pretrain_only', False))
        probe_lite_pretrain = bool(getattr(self.hyperparams, 'fc2_probe_lite_pretrain', True))
        hatc_model, lnk_model = self._get_fc2_models()
        fc2_optimizer_keys = self._get_fc2_optimizer_keys()
        sup_dataset: Optional[TensorDataset] = None
        sup_bundle: Optional[FC2SupervisedBundle] = None
        hatc_baseline_diag: Dict[str, float] = {}

        if (
            pretrain_epochs > 0
            and fc2_optimizer_keys
            and self._use_tensor_pipeline()
            and self.tensor_firm is not None
        ):
            sup_bundle = self._build_fc2_supervised_bundle()
            if sup_bundle is not None:
                sup_dataset = self._fc2_bundle_to_dataset(sup_bundle)
            if sup_dataset is None or len(sup_dataset) == 0:
                pretrain_epochs = 0
            elif probe_lite_pretrain and sup_bundle is not None:
                pretrain_epoch_losses, hatc_baseline_diag = self._run_fc2_supervised_pretrain_probe_lite(
                    sup_bundle=sup_bundle,
                    n_epochs=pretrain_epochs,
                    log_interval=log_interval,
                )
            else:
                hatc_baseline_diag = self._fit_fc2_hatc_x_baseline(sup_dataset)
                sup_loader = DataLoader(
                    sup_dataset,
                    batch_size=max(int(getattr(self.hyperparams, 'batch_size', 4096)), 1),
                    shuffle=True,
                )
                for _ in tqdm(range(pretrain_epochs), desc='FC2 Supervised Pretrain'):
                    batch_losses = []
                    for x_hatc_batch, x_lnk_batch, y_batch in sup_loader:
                        for opt_key in fc2_optimizer_keys:
                            self.optimizers[opt_key].zero_grad(set_to_none=True)
                        hatc_pred = hatc_model(x_hatc_batch)
                        lnk_pred = lnk_model(x_lnk_batch)
                        lnk_target = y_batch[:, 0:1]
                        hatc_target = y_batch[:, 1:2]
                        hatc_loss = nn.functional.mse_loss(hatc_pred, hatc_target)
                        lnk_loss = nn.functional.mse_loss(lnk_pred, lnk_target)
                        hatc_w = float(getattr(self.hyperparams, 'fc2_supervised_hatc_weight', 1.0))
                        lnk_w = float(getattr(self.hyperparams, 'fc2_supervised_lnk_weight', 1.0))
                        sup_loss = hatc_w * hatc_loss + lnk_w * lnk_loss
                        sup_loss.backward()
                        hatc_grad_norm, hatc_had_nan = gradient_protection(
                            hatc_model.parameters(),
                            max_norm=getattr(self.hyperparams, 'fc2_max_grad_norm', self.hyperparams.max_grad_norm)
                        )
                        lnk_grad_norm, lnk_had_nan = gradient_protection(
                            lnk_model.parameters(),
                            max_norm=getattr(self.hyperparams, 'fc2_max_grad_norm', self.hyperparams.max_grad_norm)
                        )
                        if hatc_had_nan:
                            logger.warning("NaN gradient detected in fc2_hatc supervised pretrain")
                        if lnk_had_nan:
                            logger.warning("NaN gradient detected in fc2_lnk supervised pretrain")
                        for opt_key in fc2_optimizer_keys:
                            self.optimizers[opt_key].step()
                        for sched_key in self._get_fc2_scheduler_keys():
                            scheduler = self.lr_schedulers.get(sched_key)
                            if scheduler is not None:
                                scheduler.step()
                        batch_log = {
                            'fc2_pretrain_loss': float(sup_loss.item()),
                            'fc2_pretrain_hatc_loss': float(hatc_loss.item()),
                            'fc2_pretrain_lnk_loss': float(lnk_loss.item()),
                        }
                        with torch.no_grad():
                            for prefix, y_true, y_pred in (
                                ('fc2_pretrain_hatc', hatc_target, hatc_pred),
                                ('fc2_pretrain_lnk', lnk_target, lnk_pred),
                            ):
                                stats = compute_fit_stats_t(y_true, y_pred)
                                for k, v in stats.items():
                                    try:
                                        batch_log[f'{prefix}_{k}'] = float(v)
                                    except (TypeError, ValueError):
                                        continue
                        batch_log['fc2_hatc_grad_norm'] = hatc_grad_norm
                        batch_log['fc2_lnk_grad_norm'] = lnk_grad_norm
                        batch_log['fc2_grad_norm'] = float(max(hatc_grad_norm, lnk_grad_norm))
                        batch_log['total'] = float(sup_loss.item())
                        batch_log.update(hatc_baseline_diag)
                        self._latest_fc2_diag = batch_log
                        self._record_manual_losses(batch_log)
                        batch_losses.append(batch_log)

                    epoch_loss = {
                        k: float(np.mean([l[k] for l in batch_losses if k in l]))
                        for k in batch_losses[0].keys()
                    }
                    pretrain_epoch_losses.append(epoch_loss)
                    if self.step_count % log_interval == 0:
                        avg_loss = np.mean([l['total'] for l in pretrain_epoch_losses[-log_interval:]])
                        current_lr = self._get_fc2_lr()
                        logger.info(
                            "FC2 Supervised Pretrain Step %d: loss=%.6f, lr=%.2e",
                            self.step_count,
                            avg_loss,
                            current_lr
                        )
        if pretrain_only:
            if not pretrain_epoch_losses:
                logger.warning("fc2_supervised_pretrain_only enabled but no supervised pretrain epochs were run")
                return None
            final_pretrain = dict(pretrain_epoch_losses[-1])
            fit_after_train = self._evaluate_fc2_fit_after_train(sup_dataset)
            logger.info("FC2 Supervised Pretrain Only finished: %s", final_pretrain)
            result = {
                'mode': 'supervised_pretrain_only',
                'final_losses': final_pretrain,
                'pretrain_epoch_losses': pretrain_epoch_losses,
            }
            if hatc_baseline_diag:
                result['hatc_x_baseline'] = hatc_baseline_diag
            if fit_after_train:
                result['fit_after_train'] = fit_after_train
            return result

        if freeze_pv and pv_model is not None:
            pv_model.freeze()
        try:
            for _ in tqdm(range(n_epochs), desc='FC2 Epochs'):
                if self._use_tensor_pipeline() and self.tensor_firm is not None:
                    fc2_batches = self._create_fc2_batches_from_tensor(
                        firm_table=self.tensor_firm,
                        macro_table=self.tensor_macro,
                        batch_size=fc2_path_batch_size,
                        shuffle=True,
                    )
                else:
                    if (self.df is None or self.df.empty) and self.tensor_firm is not None:
                        self.df = self._table_to_dataframe(self.tensor_firm)
                    if self.df is None or self.df.empty:
                        return None
                    fc2_batches = self._create_fc2_batches(
                        self.df,
                        batch_size=fc2_path_batch_size,
                    )
                if not fc2_batches:
                    return None

                batch_losses = []
                for fc2_batch in fc2_batches:
                    losses = self.train_step(fc2_batch, ['fc2'])
                    batch_losses.append(losses)
                epoch_loss = {
                    k: float(np.mean([l[k] for l in batch_losses if k in l]))
                    for k in batch_losses[0].keys()
                }
                epoch_losses.append(epoch_loss)
                if self.step_count % log_interval == 0:
                    avg_loss = np.mean([l['total'] for l in epoch_losses[-log_interval:]])
                    current_lr = self._get_fc2_lr()
                    logger.info(
                        "FC2 Step %d: loss=%.6f, lr=%.2e",
                        self.step_count,
                        avg_loss,
                        current_lr
                    )
        finally:
            if freeze_pv and pv_model is not None:
                pv_model.unfreeze()
        if not epoch_losses:
            return None
        avg_losses = {
            k: np.mean([l[k] for l in epoch_losses if k in l])
            for k in epoch_losses[0].keys()
        }
        logger.info("FC2 Epochs finished: %s", avg_losses)
        result = {'final_losses': avg_losses}
        if hatc_baseline_diag:
            result['hatc_x_baseline'] = hatc_baseline_diag
        if pretrain_epoch_losses:
            result['pretrain_final_losses'] = {
                k: np.mean([l[k] for l in pretrain_epoch_losses if k in l])
                for k in pretrain_epoch_losses[0].keys()
            }
        fit_after_train = self._evaluate_fc2_fit_after_train(sup_dataset)
        if fit_after_train:
            result['fit_after_train'] = fit_after_train
        return result

    def _run_sdf_recon_from_macro(
        self,
        module_summaries: Dict,
        n_epochs: int,
        batch_size: int,
        log_interval: int,
        n_branches: int
    ) -> None:
        macro_df = self.df_macro
        if (macro_df is None or macro_df.empty) and self.tensor_macro is not None:
            macro_df = self._table_to_dataframe(self.tensor_macro)
        if not self._use_tensor_pipeline() and (macro_df is None or macro_df.empty):
            return
        if macro_df is not None and not macro_df.empty:
            self.df_macro = macro_df

        if self._use_tensor_pipeline() and self.tensor_macro is not None:
            sdf_table = self._build_sdf_pairs_from_macro_tensor(self.tensor_macro)
        else:
            if macro_df is None or macro_df.empty:
                return
            df_macro_sdf = build_sdf_pairs_from_macro_ts(macro_df.copy(), include_hatc_lnk_t1=True)
            sdf_table = TensorTable(
                data=torch.tensor(df_macro_sdf.values, device=self.device, dtype=torch.float32),
                columns=list(df_macro_sdf.columns)
            )

        prev_flag = self.add_FC1loss
        prev_teacher_flag = self._fc1_teacher_forcing_stage
        self.add_FC1loss = True
        try:
            sdf_batches = self._create_sdf_batches_from_macro_tensor(
                sdf_table, batch_size=batch_size, n_branches=n_branches
            )
            if sdf_batches:
                tf_epochs = max(0, int(getattr(self.hyperparams, "fc1_teacher_forcing_epochs", 0)))
                if tf_epochs > 0:
                    self._fc1_teacher_forcing_stage = True
                    module_summaries['sdf_fc1_teacher_forcing'] = self._run_batches(
                        sdf_batches, tf_epochs, log_interval, ['sdf_fc1'], desc_prefix='SDF/FC1(tf) '
                    )
                    self._fc1_teacher_forcing_stage = False
                module_summaries['sdf_fc1_stage2'] = self._run_batches(
                    sdf_batches, n_epochs, log_interval, ['sdf_fc1'], desc_prefix='SDF/FC1(stage2) '
                )
                # keep backward compatibility for consumers expecting a single sdf_fc1 key
                module_summaries['sdf_fc1'] = module_summaries['sdf_fc1_stage2']
        finally:
            self.add_FC1loss = prev_flag
            self._fc1_teacher_forcing_stage = prev_teacher_flag

    def _resolve_episode_mode(self, episode_mode: Optional[str]) -> str:
        if episode_mode is None:
            return 'mode0' if self.episode_id == 0 else 'modeb'
        token = str(episode_mode).strip().lower()
        mapping = {
            'auto': 'mode0' if self.episode_id == 0 else 'modeb',
            'mode0': 'mode0',
            'modea': 'modea',
            'modeb': 'modeb',
            '0': 'mode0',
            'a': 'modea',
            'b': 'modeb',
        }
        if token not in mapping:
            raise ValueError(f"Unknown episode_mode: {episode_mode}")
        mode = mapping[token]
        if self.episode_id == 0 and mode != 'mode0':
            logger.warning("episode_id=0 uses %s (not mode0); this is allowed but not recommended.", mode)
        return mode

    def run_episode(
        self,
        n_epochs: int = 10,
        batch_size: int = 256,
        log_interval: int = 100,
        n_samples: int = 10000,
        n_paths: int = 100,
        group_size: int = 100,
        sample_group_size: Optional[int] = None,
        simulate_group_size: Optional[int] = None,
        n_branches: int = 2,
        train_mode: str = '2time',
        train_modules: Optional[List[str]] = None,
        simulate_kwargs: Optional[Dict] = None,
        episode_mode: Optional[str] = None,
        policy_stage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict:
        """
        按 Episode 逻辑执行训练（三模式）：
        - mode0: Sample(sample_group_size) 训 SDF/PV，再 SimulateTS(simulate_group_size, h=1) 训 SDF 二阶段（可选 FC2）
        - modeA: Sample(sample_group_size) 训 PV，再 SimulateTS(simulate_group_size, h=1) 训 SDF 二阶段（可选 FC2）
        - modeB: SimulateTS(simulate_group_size, h=T) 直接训练 PV/SDF（可选 FC2）
        """
        simulate_kwargs = dict(simulate_kwargs or {})
        train_modules = train_modules or ['sdf_fc1', 'policy_value', 'fc2']
        self.train_mode = train_mode
        self.add_FC1loss = False

        sample_group_size = int(group_size if sample_group_size is None else sample_group_size)
        simulate_group_size = int(group_size if simulate_group_size is None else simulate_group_size)

        horizon_mode1 = int(simulate_kwargs.pop('horizon_mode1', 1))
        horizon_modeb = int(simulate_kwargs.pop('horizon', getattr(self.hyperparams, 'simulate_horizon', 10)))
        mode = self._resolve_episode_mode(episode_mode)
        tensor_pipeline = self._use_tensor_pipeline()

        module_summaries = {}
        self.tensor_firm = None
        self.tensor_macro = None
        self.tensor_sdf = None
        self._latest_fc2_fit_df = None
        self._latest_fc2_outer_df = None
        self._latest_fc2_current_macro_panel_df = None
        self._latest_fc2_outer_macro_panel_df = None
        use_sdf_fc1 = 'sdf_fc1' in train_modules and 'sdf_fc1' in self.models
        use_policy_value = 'policy_value' in train_modules and 'policy_value' in self.models
        use_fc2 = 'fc2' in train_modules and self._has_fc2_models()

        # 记录 episode 开始时的 GPU 显存
        logger.info(f"Episode {self.episode_id} starting - GPU Memory:")
        mem_info = self.gpu_monitor.log_memory("episode_start")
        print_memory_summary(mem_info, prefix=f"  [Episode {self.episode_id}] ")

        try:
            if mode == 'mode0':
                if use_sdf_fc1 or use_policy_value:
                    sampler = Sample(
                        models=self.models,
                        config=self.config,
                        n_samples=n_samples,
                        n_paths=n_paths,
                        group_size=sample_group_size,
                        branch_num=n_branches,
                        macro_source_df=self.macro_source_df,
                    )
                else:
                    sampler = None

                if use_sdf_fc1 and sampler is not None:
                    if tensor_pipeline:
                        self.tensor_sdf = sampler.build_sdf_fc1_tensor()
                        self.df_sdf = None
                        sdf_batches = self._create_sdf_batches_from_macro_tensor(
                            self.tensor_sdf, batch_size=batch_size, n_branches=n_branches
                        )
                    else:
                        self.df_sdf = sampler.build_sdf_fc1_df()
                        self.tensor_sdf = None
                        sdf_batches = self._create_sdf_batches_from_macro_df(
                            self.df_sdf, batch_size=batch_size, n_branches=n_branches
                        )
                    if sdf_batches:
                        module_summaries['sdf_fc1_stage1'] = self._run_batches(
                            sdf_batches, n_epochs, log_interval, ['sdf_fc1'], desc_prefix='SDF/FC1(stage1) '
                        )

                if use_policy_value and sampler is not None:
                    if tensor_pipeline:
                        self.tensor_firm = sampler.build_policy_value_tensor()
                        self.df = None
                        pv_batches = self._create_firm_batches_from_tensor(
                            self.tensor_firm, batch_size=batch_size, n_branches=n_branches
                        )
                    else:
                        self.df = sampler.build_policy_value_df()
                        self.tensor_firm = None
                        pv_batches = self._create_firm_batches_from_df(
                            self.df, batch_size=batch_size, n_branches=n_branches
                        )
                    if pv_batches:
                        module_summaries['policy_value'] = self._run_batches(
                            pv_batches,
                            n_epochs,
                            log_interval,
                            ['policy_value'],
                            desc_prefix='Policy/Value ',
                            policy_stage_callback=policy_stage_callback,
                        )

                if use_sdf_fc1 or use_fc2:
                    if tensor_pipeline:
                        self._simulate_tensor(
                            n_paths=n_paths,
                            group_size=simulate_group_size,
                            n_branches=n_branches,
                            horizon=horizon_mode1,
                            simulate_kwargs=simulate_kwargs,
                            export_df=use_fc2
                        )
                    else:
                        self._simulate_df(
                            n_paths=n_paths,
                            group_size=simulate_group_size,
                            n_branches=n_branches,
                            horizon=horizon_mode1,
                            simulate_kwargs=simulate_kwargs
                        )

                if use_fc2:
                    current_macro_diag = self._capture_current_fc2_macro_panel()
                    fc2_summary = self._run_fc2_epochs(n_epochs=n_epochs, log_interval=log_interval)
                    if fc2_summary is not None:
                        if current_macro_diag:
                            fc2_summary['current_macro_consumption_diag'] = current_macro_diag
                        outer_diag = self._compute_fc2_outer_after_resim(
                            n_paths=n_paths,
                            group_size=simulate_group_size,
                            n_branches=n_branches,
                            horizon=horizon_mode1,
                            simulate_kwargs=simulate_kwargs,
                        )
                        if outer_diag:
                            fc2_summary['outer_after_resim'] = outer_diag
                        module_summaries['fc2'] = fc2_summary

                if use_sdf_fc1:
                    self._run_sdf_recon_from_macro(
                        module_summaries=module_summaries,
                        n_epochs=n_epochs,
                        batch_size=batch_size,
                        log_interval=log_interval,
                        n_branches=n_branches
                    )

            elif mode == 'modea':
                if use_policy_value:
                    sampler = Sample(
                        models=self.models,
                        config=self.config,
                        n_samples=n_samples,
                        n_paths=n_paths,
                        group_size=sample_group_size,
                        branch_num=n_branches,
                        macro_source_df=self.macro_source_df,
                    )
                    if tensor_pipeline:
                        self.tensor_firm = sampler.build_policy_value_tensor()
                        self.df = None
                        pv_batches = self._create_firm_batches_from_tensor(
                            self.tensor_firm, batch_size=batch_size, n_branches=n_branches
                        )
                    else:
                        self.df = sampler.build_policy_value_df()
                        self.tensor_firm = None
                        pv_batches = self._create_firm_batches_from_df(
                            self.df, batch_size=batch_size, n_branches=n_branches
                        )
                    if pv_batches:
                        module_summaries['policy_value'] = self._run_batches(
                            pv_batches,
                            n_epochs,
                            log_interval,
                            ['policy_value'],
                            desc_prefix='Policy/Value ',
                            policy_stage_callback=policy_stage_callback,
                        )

                if use_sdf_fc1 or use_fc2:
                    if tensor_pipeline:
                        self._simulate_tensor(
                            n_paths=n_paths,
                            group_size=simulate_group_size,
                            n_branches=n_branches,
                            horizon=horizon_mode1,
                            simulate_kwargs=simulate_kwargs,
                            export_df=use_fc2
                        )
                    else:
                        self._simulate_df(
                            n_paths=n_paths,
                            group_size=simulate_group_size,
                            n_branches=n_branches,
                            horizon=horizon_mode1,
                            simulate_kwargs=simulate_kwargs
                        )

                if use_fc2:
                    current_macro_diag = self._capture_current_fc2_macro_panel()
                    fc2_summary = self._run_fc2_epochs(n_epochs=n_epochs, log_interval=log_interval)
                    if fc2_summary is not None:
                        if current_macro_diag:
                            fc2_summary['current_macro_consumption_diag'] = current_macro_diag
                        outer_diag = self._compute_fc2_outer_after_resim(
                            n_paths=n_paths,
                            group_size=simulate_group_size,
                            n_branches=n_branches,
                            horizon=horizon_mode1,
                            simulate_kwargs=simulate_kwargs,
                        )
                        if outer_diag:
                            fc2_summary['outer_after_resim'] = outer_diag
                        module_summaries['fc2'] = fc2_summary

                if use_sdf_fc1:
                    self._run_sdf_recon_from_macro(
                        module_summaries=module_summaries,
                        n_epochs=n_epochs,
                        batch_size=batch_size,
                        log_interval=log_interval,
                        n_branches=n_branches
                    )

            elif mode == 'modeb':
                if tensor_pipeline:
                    self._simulate_tensor(
                        n_paths=n_paths,
                        group_size=simulate_group_size,
                        n_branches=n_branches,
                        horizon=horizon_modeb,
                        simulate_kwargs=simulate_kwargs,
                        export_df=use_fc2
                    )
                else:
                    self._simulate_df(
                        n_paths=n_paths,
                        group_size=simulate_group_size,
                        n_branches=n_branches,
                        horizon=horizon_modeb,
                        simulate_kwargs=simulate_kwargs
                    )

                if use_policy_value:
                    if tensor_pipeline and self.tensor_firm is not None:
                        pv_batches = self._create_firm_batches_from_tensor(
                            self.tensor_firm, batch_size=batch_size, n_branches=n_branches
                        )
                    else:
                        pv_batches = self._create_firm_batches_from_df(
                            self.df, batch_size=batch_size, n_branches=n_branches
                        )
                    if pv_batches:
                        module_summaries['policy_value'] = self._run_batches(
                            pv_batches,
                            n_epochs,
                            log_interval,
                            ['policy_value'],
                            desc_prefix='Policy/Value ',
                            policy_stage_callback=policy_stage_callback,
                        )

                if use_sdf_fc1:
                    self.add_FC1loss = False
                    if tensor_pipeline and self.tensor_firm is not None:
                        sdf_batches = self._create_firm_batches_from_tensor(
                            self.tensor_firm, batch_size=batch_size, n_branches=n_branches, eta_resample=False
                        )
                    else:
                        sdf_batches = self._create_firm_batches_from_df(
                            self.df, batch_size=batch_size, n_branches=n_branches, eta_resample=False
                        )
                    if sdf_batches:
                        module_summaries['sdf_fc1'] = self._run_batches(
                            sdf_batches, n_epochs, log_interval, ['sdf_fc1'], desc_prefix='SDF/FC1 '
                        )

                if use_fc2:
                    current_macro_diag = self._capture_current_fc2_macro_panel()
                    fc2_summary = self._run_fc2_epochs(n_epochs=n_epochs, log_interval=log_interval)
                    if fc2_summary is not None:
                        if current_macro_diag:
                            fc2_summary['current_macro_consumption_diag'] = current_macro_diag
                        outer_diag = self._compute_fc2_outer_after_resim(
                            n_paths=n_paths,
                            group_size=simulate_group_size,
                            n_branches=n_branches,
                            horizon=horizon_modeb,
                            simulate_kwargs=simulate_kwargs,
                        )
                        if outer_diag:
                            fc2_summary['outer_after_resim'] = outer_diag
                        module_summaries['fc2'] = fc2_summary
            else:
                raise ValueError(f"Unknown episode mode: {mode}")
        finally:
            self.add_FC1loss = False

        if 'sdf_fc1' not in module_summaries and 'sdf_fc1_stage1' in module_summaries:
            module_summaries['sdf_fc1'] = module_summaries['sdf_fc1_stage1']

        # 训练结束后再导出 DataFrame，兼容现有实验脚本的可视化/落盘逻辑
        if self.df is None and self.tensor_firm is not None:
            self.df = self._table_to_dataframe(self.tensor_firm)
        if self.df_macro is None and self.tensor_macro is not None:
            self.df_macro = self._table_to_dataframe(self.tensor_macro)
        if self.df_sdf is None and self.tensor_sdf is not None:
            self.df_sdf = self._table_to_dataframe(self.tensor_sdf)

        # 记录 episode 结束时的 GPU 显存
        logger.info(f"Episode {self.episode_id} completed - GPU Memory:")
        mem_info = self.gpu_monitor.log_memory("episode_end")
        logger.info(f"GPU Memory at episode end: {mem_info}")
        
        summary = {
            'episode_id': self.episode_id,
            'episode_mode': mode,
            'total_steps': self.step_count,
            'module_summaries': module_summaries,
            'loss_history': self.loss_history,
            'gpu_memory': self.gpu_monitor.get_summary()
        }
        if 'policy_value' in module_summaries and isinstance(module_summaries['policy_value'], dict):
            summary['convergence'] = module_summaries['policy_value'].get('convergence')
        
        return summary
    
    def run(
        self,
        n_epochs: int = 10,
        batch_size: int = 1024,
        log_interval: int = 100,
        train_modules: List[str] = None
    ) -> Dict:
        """
        运行完整的 Episode 训练
        
        Args:
            n_epochs: 轮数
            batch_size: 批大小
            log_interval: 日志间隔
            train_modules: 要训练的模块
        
        Returns:
            summary: 训练摘要
        """
        train_modules = train_modules or ['sdf_fc1', 'policy_value']

        logger.info(f"Episode {self.episode_id}: Starting training")
        batches: List[Dict[str, torch.Tensor]] = []
        for epoch in range(n_epochs):
            self._current_epoch_idx = epoch
            self._q_only_stage = False
            batches = self.create_batches(batch_size)
            
            epoch_losses = []
            for batch in tqdm(batches, desc=f"Epoch {epoch+1}/{n_epochs}"):
                losses = self.train_step(batch, train_modules)
                epoch_losses.append(losses)
                
                if self.step_count % log_interval == 0:
                    avg_loss = np.mean([l['total'] for l in epoch_losses[-log_interval:]])
                    lr = self.lr_schedulers.get('sdf_fc1', self.lr_schedulers.get('policy_value'))
                    current_lr = lr.get_lr() if lr else 0
                    
                    logger.info(
                        f"Step {self.step_count}: "
                        f"loss={avg_loss:.6f}, lr={current_lr:.2e}"
                    )
            
            # Epoch 结束统计
            avg_losses = {
                k: np.mean([l[k] for l in epoch_losses if k in l])
                for k in epoch_losses[0].keys()
            }
            logger.info(f"Epoch {epoch+1} finished: {avg_losses}")
        convergence = None
        if 'policy_value' in train_modules and 'policy_value' in self.models and batches:
            convergence = self.evaluate_bellman_convergence(batches)
        
        summary = {
            'episode_id': self.episode_id,
            'total_steps': self.step_count,
            'final_losses': avg_losses,
            'loss_history': self.loss_history
        }
        if convergence is not None:
            summary['convergence'] = convergence
        
        return summary
    
    def get_metrics(self) -> Dict:
        """
        获取训练指标
        """
        metrics = {}
        
        for k, v in self.loss_history.items():
            if len(v) > 0:
                metrics[f'{k}_mean'] = np.mean(v)
                metrics[f'{k}_std'] = np.std(v)
                metrics[f'{k}_min'] = np.min(v)
                metrics[f'{k}_max'] = np.max(v)
                metrics[f'{k}_last'] = v[-1]
        
        return metrics
