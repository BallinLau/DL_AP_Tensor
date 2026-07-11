"""
Episode 类：训练周期管理

一个 Episode 包含：
1. 数据生成（Sample 或 SimulateTS）
2. FC1 填充
3. Policy/Value 填充
4. 各模块的训练循环
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from copy import deepcopy
from enum import Enum
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import logging

import sys
import warnings
sys.path.append('..')
from config import Config, HyperParams, SIMMODEL
from data import Sample, SimulateTS, TensorTable, TensorSimulationOutput
from data.data_utils import compute_quantile_features
from losses import SDFLoss, P0Loss, PILoss, QLoss, FC2Loss
from losses.FC2losspipe import FC2LossPipe
from losses.utils import compute_z_penalty, compute_aio_residual
from losses.sdf_loss import moment_penalty
from data.data_utils import build_sdf_pairs_from_macro_ts
from .gradient_utils import gradient_protection, compute_gradient_norm
from .scheduler import LossWeightScheduler, LearningRateScheduler
from .sdf_shock_bank import (
    SDFShockBank,
    _make_generator,
    shock_pair_diagnostics,
    shocks_to_x_children,
)
from .bp_grid_teacher import BPGridTeacher
from .target_utils import hard_update, soft_update
from utils.gpu_monitor import GPUMonitor, print_memory_summary


logger = logging.getLogger(__name__)


class NumericalStageFailure(RuntimeError):
    """Raised when a training stage repeatedly produces non-finite gradients."""


class SDFTrainingPhase(str, Enum):
    EPISODE0_BOOTSTRAP = "episode0_bootstrap"
    FC1_ONLY = "fc1_only"
    SDF_TRUE_ONLY = "sdf_true_only"
    SDF_RECURSIVE_ONLY = "sdf_recursive_only"
    JOINT_DISABLED = "joint_disabled"


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
        gpu_monitor = None,
        firm_target: Optional[nn.Module] = None
    ):
        """
        Args:
            models: 模型字典
                - 'sdf_fc1': SDFFC1Combined
                - 'policy_value': PolicyValueModel
                - 'fc2': FC2Model (可选)
            optimizers: 优化器字典
            config: 配置类
            hyperparams: 超参数
            device: 设备
            episode_id: Episode 编号
            gpu_monitor: GPU 监控器（可选，用于共享监控数据）
            firm_target: 冻结的 policy_value target network（可选）
        """
        self.models = models
        self.optimizers = optimizers
        self.config = config
        self.hyperparams = hyperparams or HyperParams()
        self._validate_sdf_fresh_pair_config()
        self.device = device or config.DEVICE
        self.episode_id = episode_id
        self.firm_target = self._init_firm_target(firm_target)
        
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
        self.tensor_firm: Optional[TensorTable] = None
        self.tensor_macro: Optional[TensorTable] = None
        self.tensor_sdf: Optional[TensorTable] = None
        
        # 训练状态
        self.step_count = 0
        self.sdf_fc1_step_count = 0
        self.sdf_training_phase = (
            SDFTrainingPhase.EPISODE0_BOOTSTRAP
            if int(self.episode_id) == 0
            else SDFTrainingPhase.SDF_TRUE_ONLY
        )
        self.loss_history = {}
        self.add_FC1loss = False
        self.train_mode = '2time'
        self._latest_sdf_diag = {}
        self._latest_sdf_terms = {}
        self._latest_p0_terms = {}
        self._latest_pi_terms = {}
        self._latest_q_terms = {}
        self._sdf_base_lr_backup = None
        self._current_epoch_idx = 0
        self._q_only_stage = False
        self._bp_only_stage = False
        self._fc1_teacher_forcing_stage = False
        self._policy_q_freeze_active = False
        self._policy_value_grad_backup = {}
        self._policy_bp_freeze_active = False
        self._policy_bp_grad_backup = {}
        self._sdf_fc1_teacher_freeze_active = False
        self._sdf_fc1_grad_backup = {}
        self._nonfinite_grad_streak = 0
        self._nonfinite_grad_total = 0
        self._last_nonfinite_grad_params: Dict[str, List[str]] = {}
        self._last_policy_value_stage_summary: Optional[Dict[str, float]] = None
        self._last_policy_value_gate_context: Dict[str, float] = {}
        self._sdf_shock_bank: Optional[SDFShockBank] = None
        self._sdf_shock_bank_n_parents: int = 0
        self._sdf_shock_bank_epoch: Optional[int] = None
        self._sdf_shock_bank_episode_id: Optional[int] = None
        self._sdf_shock_bank_key: Optional[Tuple[Any, ...]] = None
        self._sdf_pair_generator: Optional[torch.Generator] = None

    def set_sdf_training_phase(self, phase: str | SDFTrainingPhase) -> None:
        self.sdf_training_phase = SDFTrainingPhase(phase)

    @staticmethod
    def _normalize_metric_value(key: str, value: Any) -> Any:
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise TypeError(
                    f"Metric {key!r} must be scalar, got tensor shape={tuple(value.shape)}."
                )
            return value.detach().item()
        return value

    @staticmethod
    def _is_numeric_metric_value(value: Any) -> bool:
        return (
            isinstance(value, (Number, np.number))
            and not isinstance(value, (str, bytes))
        )

    @classmethod
    def _aggregate_metric_records(
        cls,
        records: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        if not records:
            return {}, {}

        numeric_metrics: Dict[str, float] = {}
        metadata: Dict[str, Any] = {}
        keys = set().union(*(record.keys() for record in records))

        for key in keys:
            values = [
                cls._normalize_metric_value(key, record[key])
                for record in records
                if key in record
            ]
            if not values:
                continue
            if all(cls._is_numeric_metric_value(value) for value in values):
                numeric_metrics[key] = float(np.mean(values))
                continue

            first = values[0]
            if any(value != first for value in values[1:]):
                raise RuntimeError(
                    f"Non-numeric metric {key!r} changed within one epoch: {values!r}"
                )
            metadata[key] = first

        return numeric_metrics, metadata

    def reset_sdf_shock_bank(self) -> None:
        """Drop cached fresh-pair shock bank when episode/data stage changes."""
        self._sdf_shock_bank = None
        self._sdf_shock_bank_n_parents = 0
        self._sdf_shock_bank_epoch = None
        self._sdf_shock_bank_episode_id = int(self.episode_id)
        self._sdf_shock_bank_key = None
        self._sdf_pair_generator = None

    def _validate_sdf_fresh_pair_config(self) -> None:
        mode = str(getattr(self.hyperparams, "sdf_wealth_loss_mode", "legacy_abs_log1p")).lower()
        fresh = bool(getattr(self.hyperparams, "sdf_fresh_pair_enabled", False))
        if mode == "signed_aio" and not fresh:
            raise ValueError("signed_aio requires sdf_fresh_pair_enabled=True for valid double sampling.")
        if mode != "signed_aio" and fresh:
            warnings.warn(
                "sdf_fresh_pair_enabled=True with legacy SDF wealth loss is allowed, "
                "but fresh pair bank is primarily designed for signed_aio.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _init_firm_target(self, firm_target: Optional[nn.Module] = None) -> Optional[nn.Module]:
        """
        Initialize or attach a frozen policy/value target network.
        """
        online = self.models.get('policy_value')
        if online is None:
            return None
        target = firm_target if firm_target is not None else deepcopy(online)
        target.to(self.device)
        target.eval()
        target.requires_grad_(False)
        if firm_target is None:
            hard_update(target, online)
        return target

    def refresh_firm_target(self) -> None:
        """Hard-copy the online policy/value model into the target model."""
        if self.firm_target is not None and self.models.get('policy_value') is not None:
            hard_update(self.firm_target, self.models['policy_value'])
            self.firm_target.eval()
            self.firm_target.requires_grad_(False)

    def _update_firm_target_now(self, mode: str) -> None:
        if self.firm_target is None or self.models.get('policy_value') is None:
            return
        mode = mode.lower()
        if mode in {"hard", "epoch_hard"}:
            hard_update(self.firm_target, self.models['policy_value'])
        elif mode in {"soft", "epoch_soft"}:
            tau = float(getattr(self.hyperparams, "firm_target_tau", 0.005))
            soft_update(self.firm_target, self.models['policy_value'], tau=tau)
        else:
            raise ValueError(f"Unknown firm_target_update mode: {mode}")
        self.firm_target.eval()
        self.firm_target.requires_grad_(False)

    def _maybe_update_firm_target(self, train_modules: List[str]) -> None:
        """
        Update firm target after online optimizer steps.
        """
        if 'policy_value' not in train_modules:
            return
        if self.firm_target is None or self.models.get('policy_value') is None:
            return
        mode = str(getattr(self.hyperparams, "firm_target_update", "soft")).lower()
        if mode in {"none", "off", "disabled"}:
            return
        if mode in {"epoch_hard", "epoch_soft"}:
            return
        interval = max(1, int(getattr(self.hyperparams, "firm_target_update_interval_steps", 1)))
        if interval > 1 and (self.step_count + 1) % interval != 0:
            return
        self._update_firm_target_now(mode)

    def _maybe_update_firm_target_epoch(self, train_modules: List[str]) -> None:
        if 'policy_value' not in train_modules:
            return
        mode = str(getattr(self.hyperparams, "firm_target_update", "soft")).lower()
        if mode in {"epoch_hard", "epoch_soft"}:
            self._update_firm_target_now(mode)

    def _target_policy_value(self) -> nn.Module:
        target = self.firm_target
        if target is None:
            target = self.models['policy_value']
        target.eval()
        return target

    def _pv_bp_training_mode(self) -> str:
        return str(getattr(self.hyperparams, "pv_bp_training_mode", "legacy_foc_kkt")).lower()

    def _pv_use_target_grid_bp(self) -> bool:
        mode = self._pv_bp_training_mode()
        return mode in {"target_grid", "grid", "grid_b", "full_grid"}

    def _ablation_mode(self) -> str:
        return str(getattr(self.hyperparams, "ablation_mode", "baseline")).lower()

    def _policy_value_bellman_only(self) -> bool:
        return bool(getattr(self.hyperparams, "policy_value_bellman_only", False)) or self._ablation_mode() == "bellman_only"

    def _pv_use_fixed_sdf(self) -> bool:
        return bool(getattr(self.hyperparams, "pv_fixed_sdf", False)) or self._ablation_mode() == "fixed_sdf"

    def _pv_use_fixed_policy(self) -> bool:
        return bool(getattr(self.hyperparams, "pv_fixed_policy", False)) or self._ablation_mode() == "fixed_policy"

    def _apply_policy_ablation(self, bp: torch.Tensor, parent_b: torch.Tensor) -> torch.Tensor:
        if not self._pv_use_fixed_policy():
            return bp
        mode = str(getattr(self.hyperparams, "pv_fixed_policy_mode", "parent_b")).lower()
        if mode == "zero":
            fixed = torch.zeros_like(parent_b)
        elif mode == "one":
            fixed = torch.ones_like(parent_b)
        else:
            fixed = parent_b.clamp(0.0, 1.0)
        return fixed.detach().clone().requires_grad_(bp.requires_grad)

    def _build_policy_m_lists(self, parent: torch.Tensor, children: List[torch.Tensor], clamp_min: float, clamp_max: float) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if parent.shape[1] > 7:
            raw_M_list = [child[:, 7:8] for child in children]
        else:
            raw_M_list = [torch.ones(parent.shape[0], 1, device=self.device) for _ in children]
        if self._pv_use_fixed_sdf():
            fixed = float(getattr(self.hyperparams, "pv_fixed_sdf_value", 0.98))
            M_list = [torch.full_like(m, fixed) for m in raw_M_list]
        elif bool(getattr(self.hyperparams, "pv_use_clipped_m", True)):
            M_list = [m.clamp(clamp_min, clamp_max) for m in raw_M_list]
        else:
            M_list = raw_M_list
        return raw_M_list, M_list

    @staticmethod
    def _safe_quantile(v: torch.Tensor, q: float) -> float:
        vv = v.detach().reshape(-1)
        vv = vv[torch.isfinite(vv)]
        if vv.numel() == 0:
            return float('nan')
        return float(torch.quantile(vv.to(torch.float32), q).item())

    def _m_diagnostics(self, prefix: str, raw_m: torch.Tensor, use_m: torch.Tensor, lo: float, hi: float) -> Dict[str, float]:
        raw = raw_m.detach().reshape(-1)
        used = use_m.detach().reshape(-1)
        raw_finite = raw[torch.isfinite(raw)]
        used_finite = used[torch.isfinite(used)]
        if raw_finite.numel() == 0 or used_finite.numel() == 0:
            return {
                f'{prefix}_M_raw_finite_ratio': 0.0,
                f'{prefix}_M_clip_low_ratio': float('nan'),
                f'{prefix}_M_clip_high_ratio': float('nan'),
            }
        return {
            f'{prefix}_log_mean_M_raw': float(torch.log(raw_finite.mean().clamp_min(1e-8)).item()),
            f'{prefix}_log_mean_M_used': float(torch.log(used_finite.mean().clamp_min(1e-8)).item()),
            f'{prefix}_M_raw_p50': self._safe_quantile(raw_finite, 0.50),
            f'{prefix}_M_raw_p90': self._safe_quantile(raw_finite, 0.90),
            f'{prefix}_M_raw_p99': self._safe_quantile(raw_finite, 0.99),
            f'{prefix}_M_raw_max': float(raw_finite.max().item()),
            f'{prefix}_M_used_p50': self._safe_quantile(used_finite, 0.50),
            f'{prefix}_M_used_p90': self._safe_quantile(used_finite, 0.90),
            f'{prefix}_M_used_p99': self._safe_quantile(used_finite, 0.99),
            f'{prefix}_M_used_max': float(used_finite.max().item()),
            f'{prefix}_M_raw_finite_ratio': float(raw_finite.numel() / max(raw.numel(), 1)),
            f'{prefix}_M_clip_low_ratio': float((raw_finite < lo).float().mean().item()),
            f'{prefix}_M_clip_high_ratio': float((raw_finite > hi).float().mean().item()),
        }

    def _tensor_tail_diagnostics(self, prefix: str, value: torch.Tensor) -> Dict[str, float]:
        flat = value.detach().reshape(-1)
        finite = flat[torch.isfinite(flat)]
        if finite.numel() == 0:
            return {
                f'{prefix}_finite_ratio': 0.0,
                f'{prefix}_mean': float('nan'),
                f'{prefix}_abs_p50': float('nan'),
                f'{prefix}_abs_p90': float('nan'),
                f'{prefix}_abs_p99': float('nan'),
                f'{prefix}_abs_p999': float('nan'),
                f'{prefix}_abs_max': float('nan'),
            }
        abs_finite = finite.abs()
        return {
            f'{prefix}_finite_ratio': float(finite.numel() / max(flat.numel(), 1)),
            f'{prefix}_mean': float(finite.mean().item()),
            f'{prefix}_abs_p50': self._safe_quantile(abs_finite, 0.50),
            f'{prefix}_abs_p90': self._safe_quantile(abs_finite, 0.90),
            f'{prefix}_abs_p99': self._safe_quantile(abs_finite, 0.99),
            f'{prefix}_abs_p999': self._safe_quantile(abs_finite, 0.999),
            f'{prefix}_abs_max': float(abs_finite.max().item()),
        }

    @staticmethod
    def _module_grad_norm(module: nn.Module) -> float:
        vals = []
        for p in module.parameters():
            if p.grad is not None:
                vals.append(float(p.grad.detach().norm(2).item()) ** 2)
        return float(sum(vals) ** 0.5) if vals else 0.0

    def _policy_value_grad_group_norms(self) -> Dict[str, float]:
        model = self.models.get('policy_value')
        if model is None:
            return {}
        names = [
            'value_encoder', 'v0_head', 'vi_head',
            'q_encoder', 'q_head',
            'policy_encoder', 'bp0_head', 'bpi_head',
        ]
        out = {}
        for name in names:
            module = getattr(model, name, None)
            if module is not None:
                out[f'{name}_grad_norm'] = self._module_grad_norm(module)
        return out

    def _policy_value_gate_limits(self) -> Tuple[float, float]:
        loss_abs = float(getattr(self.hyperparams, "policy_value_loss_fail_threshold", 1000.0))
        grad_abs = float(getattr(self.hyperparams, "policy_value_grad_fail_threshold", 1000.0))
        prev = self._last_policy_value_stage_summary or {}
        prev_loss = prev.get('total')
        prev_grad = prev.get('policy_value_grad_norm')
        if prev_loss is not None and np.isfinite(prev_loss):
            mult = float(getattr(self.hyperparams, "policy_value_loss_relative_fail_multiplier", 10.0))
            loss_abs = max(loss_abs, float(prev_loss) * mult)
        if prev_grad is not None and np.isfinite(prev_grad):
            mult = float(getattr(self.hyperparams, "policy_value_grad_relative_fail_multiplier", 10.0))
            rolling_abs = float(getattr(self.hyperparams, "policy_value_rolling_grad_fail_threshold", 100.0))
            grad_abs = min(grad_abs, max(rolling_abs, float(prev_grad) * mult))
        return loss_abs, grad_abs

    def _check_policy_value_gate(self, losses: Dict[str, float], context: str) -> None:
        if not bool(getattr(self.hyperparams, "stage_fail_on_policy_value_explosion", True)):
            return
        total_v = float(losses.get('total', 0.0))
        grad_v = float(losses.get('policy_value_grad_norm', 0.0))
        loss_thr, grad_thr = self._policy_value_gate_limits()
        clip_gate = float(getattr(self.hyperparams, "pv_sdf_clip_ratio_gate", 0.05))
        clip_high = max(
            float(losses.get('p0_M_clip_high_ratio', 0.0)),
            float(losses.get('pi_M_clip_high_ratio', 0.0)),
        )
        clip_low = max(
            float(losses.get('p0_M_clip_low_ratio', 0.0)),
            float(losses.get('pi_M_clip_low_ratio', 0.0)),
        )
        failed = (
            (not np.isfinite(total_v))
            or (not np.isfinite(grad_v))
            or total_v > loss_thr
            or grad_v > grad_thr
            or clip_high > clip_gate
            or clip_low > clip_gate
        )
        if failed:
            self._last_policy_value_gate_context = {
                'total': total_v,
                'policy_value_grad_norm': grad_v,
                'loss_threshold': loss_thr,
                'grad_threshold': grad_thr,
                'clip_high_ratio': clip_high,
                'clip_low_ratio': clip_low,
                'clip_ratio_gate': clip_gate,
                'context': context,
                'losses': dict(losses),
            }
            raise NumericalStageFailure(
                f"Policy/value stage failed at {context}: "
                f"total={total_v:.6g} (thr={loss_thr:.6g}), "
                f"grad_norm={grad_v:.6g} (thr={grad_thr:.6g}), "
                f"clip_high={clip_high:.6g} (thr={clip_gate:.6g}), "
                f"clip_low={clip_low:.6g} (thr={clip_gate:.6g})"
            )

    @staticmethod
    def _nonfinite_gradient_params(model: nn.Module, limit: int = 20) -> List[str]:
        """
        Return names of parameters whose gradients contain NaN or Inf.
        """
        bad: List[str] = []
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                bad.append(name)
                if len(bad) >= limit:
                    break
        return bad
    
    def _init_loss_functions(self) -> Dict:
        """
        初始化损失函数
        """
        return {
            'sdf': SDFLoss(
                wealth_loss_mode=getattr(self.hyperparams, "sdf_wealth_loss_mode", "legacy_abs_log1p")
            ),
            'p0': P0Loss(),
            'pi': PILoss(),
            'q': QLoss(),
            'fc2': FC2Loss() if 'fc2' in self.models else None
        }

    @staticmethod
    def _macro_forecast_r2(df_macro: Optional[pd.DataFrame]) -> Dict[str, float]:
        """
        计算宏观预测(FC1 proxy)与实现值(realized)的 R^2。
        兼容列名：realized(Hatc/LnK), forecast(hatcf/lnkf 或 Hatcf/LnKF)。
        """
        if df_macro is None or df_macro.empty:
            return {}

        def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if mask.sum() < 2:
                return float('nan')
            yt = y_true[mask]
            yp = y_pred[mask]
            sst = float(np.sum((yt - yt.mean()) ** 2))
            if sst <= 1e-12:
                return float('nan')
            sse = float(np.sum((yt - yp) ** 2))
            return 1.0 - sse / sst

        use_df = df_macro
        hatc_true_col = _pick_col(use_df, ['Hatc', 'hatc'])
        lnk_true_col = _pick_col(use_df, ['LnK', 'lnk'])
        hatc_pred_col = _pick_col(use_df, ['hatcf', 'Hatcf'])
        lnk_pred_col = _pick_col(use_df, ['lnkf', 'LnKF'])

        if hatc_true_col is None or lnk_true_col is None:
            return {}

        out: Dict[str, float] = {'n_t': float(len(use_df)), 'n_obs': float(len(use_df))}
        if hatc_pred_col is not None:
            out['r2_hatc'] = _safe_r2(
                use_df[hatc_true_col].to_numpy(),
                use_df[hatc_pred_col].to_numpy()
            )
            if 'branch' in use_df.columns:
                by_branch_hatc: Dict[str, float] = {}
                for br, grp in use_df.groupby('branch'):
                    key = f"{int(br)}" if float(br).is_integer() else str(br)
                    by_branch_hatc[key] = _safe_r2(
                        grp[hatc_true_col].to_numpy(),
                        grp[hatc_pred_col].to_numpy()
                    )
                out['r2_hatc_by_branch'] = by_branch_hatc
        if lnk_pred_col is not None:
            out['r2_lnk'] = _safe_r2(
                use_df[lnk_true_col].to_numpy(),
                use_df[lnk_pred_col].to_numpy()
            )
            if 'branch' in use_df.columns:
                by_branch_lnk: Dict[str, float] = {}
                for br, grp in use_df.groupby('branch'):
                    key = f"{int(br)}" if float(br).is_integer() else str(br)
                    by_branch_lnk[key] = _safe_r2(
                        grp[lnk_true_col].to_numpy(),
                        grp[lnk_pred_col].to_numpy()
                    )
                out['r2_lnk_by_branch'] = by_branch_lnk
        if 'branch' in use_df.columns:
            out['n_obs_by_branch'] = {
                (f"{int(br)}" if float(br).is_integer() else str(br)): float(len(grp))
                for br, grp in use_df.groupby('branch')
            }
        return out

    @staticmethod
    def _macro_forecast_r2_tensor(macro_table: Optional[TensorTable]) -> Dict[str, float]:
        """
        tensor 版本宏观 R² 诊断，避免训练前 DataFrame 依赖。
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

        def _safe_r2_t(yt: torch.Tensor, yp: torch.Tensor) -> float:
            mask = torch.isfinite(yt) & torch.isfinite(yp)
            if int(mask.sum().item()) < 2:
                return float('nan')
            yt = yt[mask]
            yp = yp[mask]
            sst = ((yt - yt.mean()) ** 2).sum()
            if float(sst.item()) <= 1e-12:
                return float('nan')
            sse = ((yt - yp) ** 2).sum()
            return float((1.0 - sse / sst).item())

        out: Dict[str, Any] = {
            'n_t': float(data.shape[0]),
            'n_obs': float(data.shape[0]),
            'r2_hatc': _safe_r2_t(y_hatc, p_hatc),
            'r2_lnk': _safe_r2_t(y_lnk, p_lnk),
        }
        if 'branch' in col:
            br = data[:, col['branch']].long()
            br_vals = torch.unique(br)
            r2_hatc_by_branch: Dict[str, float] = {}
            r2_lnk_by_branch: Dict[str, float] = {}
            n_obs_by_branch: Dict[str, float] = {}
            for b in br_vals:
                mk = br == b
                key = f"{int(b.item())}"
                r2_hatc_by_branch[key] = _safe_r2_t(y_hatc[mk], p_hatc[mk])
                r2_lnk_by_branch[key] = _safe_r2_t(y_lnk[mk], p_lnk[mk])
                n_obs_by_branch[key] = float(mk.sum().item())
            out['r2_hatc_by_branch'] = r2_hatc_by_branch
            out['r2_lnk_by_branch'] = r2_lnk_by_branch
            out['n_obs_by_branch'] = n_obs_by_branch
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

    def _capture_rng_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
        }
        if torch.cuda.is_available():
            state['cuda'] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            return
        if 'torch' in state:
            torch.set_rng_state(state['torch'])
        if 'numpy' in state:
            np.random.set_state(state['numpy'])
        if 'cuda' in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda'])

    @staticmethod
    def _selected_frame_snapshot(
        table: Optional[TensorTable],
        dataframe: Optional[pd.DataFrame],
        columns: List[str],
        key_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Materialize only selected diagnostic columns, preferring tensor data.
        This avoids copying the full firm table while still supporting df mode.
        """
        key_columns = key_columns or []
        wanted: List[str] = []
        for name in key_columns + columns:
            if name not in wanted:
                wanted.append(name)
        if table is not None and table.data.numel() > 0:
            col = {name: i for i, name in enumerate(table.columns)}
            have = [name for name in wanted if name in col]
            if not have:
                return pd.DataFrame()
            idx = [col[name] for name in have]
            arr = table.data[:, idx].detach().cpu().numpy()
            out = pd.DataFrame(arr, columns=have)
        elif dataframe is not None and not dataframe.empty:
            have = [name for name in wanted if name in dataframe.columns]
            if not have:
                return pd.DataFrame()
            out = dataframe.loc[:, have].copy()
        else:
            return pd.DataFrame()

        for name in key_columns:
            if name in out.columns:
                if name == 'ID':
                    out[name] = out[name].astype(str)
                else:
                    out[name] = np.rint(pd.to_numeric(out[name], errors='coerce')).astype('Int64')
        return out

    @staticmethod
    def _snapshot_stats(prefix: str, frame: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if frame is None or frame.empty:
            return out
        columns = columns or list(frame.columns)
        for name in columns:
            if name not in frame.columns:
                continue
            v = pd.to_numeric(frame[name], errors='coerce').to_numpy(dtype=np.float64)
            finite = v[np.isfinite(v)]
            key = f'{prefix}_{name}'
            out[f'{key}_n'] = float(finite.size)
            if finite.size == 0:
                continue
            out[f'{key}_mean'] = float(np.mean(finite))
            out[f'{key}_std'] = float(np.std(finite))
            out[f'{key}_p50'] = float(np.quantile(finite, 0.50))
            out[f'{key}_p90'] = float(np.quantile(finite, 0.90))
            out[f'{key}_p99'] = float(np.quantile(finite, 0.99))
        return out

    @staticmethod
    def _keyed_snapshot_gap(
        prefix: str,
        old: pd.DataFrame,
        new: pd.DataFrame,
        key_columns: List[str],
        value_columns: List[str]
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not key_columns:
            out[f'{prefix}_keyed'] = 0.0
            out[f'{prefix}_common_rows'] = 0.0
            return out
        if old is None or new is None or old.empty or new.empty:
            out[f'{prefix}_old_rows'] = float(0 if old is None else len(old))
            out[f'{prefix}_new_rows'] = float(0 if new is None else len(new))
            out[f'{prefix}_common_rows'] = 0.0
            return out
        if any(k not in old.columns or k not in new.columns for k in key_columns):
            out[f'{prefix}_keyed'] = 0.0
            return out

        old_keyed = old.dropna(subset=key_columns).drop_duplicates(subset=key_columns)
        new_keyed = new.dropna(subset=key_columns).drop_duplicates(subset=key_columns)
        old_keys = old_keyed.loc[:, key_columns]
        new_keys = new_keyed.loc[:, key_columns]
        common_keys = old_keys.merge(new_keys, on=key_columns, how='inner')
        out[f'{prefix}_keyed'] = 1.0
        out[f'{prefix}_old_rows'] = float(len(old_keyed))
        out[f'{prefix}_new_rows'] = float(len(new_keyed))
        out[f'{prefix}_common_rows'] = float(len(common_keys))
        out[f'{prefix}_old_only_rows'] = float(max(len(old_keyed) - len(common_keys), 0))
        out[f'{prefix}_new_only_rows'] = float(max(len(new_keyed) - len(common_keys), 0))

        have_values = [
            name for name in value_columns
            if name in old_keyed.columns and name in new_keyed.columns
        ]
        if not have_values or common_keys.empty:
            return out
        merged = old_keyed.loc[:, key_columns + have_values].merge(
            new_keyed.loc[:, key_columns + have_values],
            on=key_columns,
            how='inner',
            suffixes=('_old', '_new')
        )
        for name in have_values:
            a = pd.to_numeric(merged[f'{name}_old'], errors='coerce').to_numpy(dtype=np.float64)
            b = pd.to_numeric(merged[f'{name}_new'], errors='coerce').to_numpy(dtype=np.float64)
            mask = np.isfinite(a) & np.isfinite(b)
            key = f'{prefix}_{name}'
            out[f'{key}_n_common'] = float(mask.sum())
            if not mask.any():
                continue
            d = b[mask] - a[mask]
            out[f'{key}_mae'] = float(np.mean(np.abs(d)))
            out[f'{key}_mean_delta'] = float(np.mean(d))
            out[f'{key}_rmse'] = float(np.sqrt(np.mean(d ** 2)))
        return out

    @staticmethod
    def _firm_economic_moments(prefix: str, frame: pd.DataFrame) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if frame is None or frame.empty:
            out[f'{prefix}_n_rows'] = 0.0
            return out
        out[f'{prefix}_n_rows'] = float(len(frame))
        key_cols = [k for k in ['path', 't', 'branch', 'ID'] if k in frame.columns]
        if key_cols:
            out[f'{prefix}_n_keys'] = float(len(frame.dropna(subset=key_cols).drop_duplicates(subset=key_cols)))
        if 'entry' in frame.columns:
            entry = pd.to_numeric(frame['entry'], errors='coerce').to_numpy(dtype=np.float64)
            entry = entry[np.isfinite(entry)]
            out[f'{prefix}_entry_rate'] = float(np.mean(entry > 0.5)) if entry.size else float('nan')
        if 'Bar_z' in frame.columns:
            bar_z = pd.to_numeric(frame['Bar_z'], errors='coerce').to_numpy(dtype=np.float64)
            bar_z = bar_z[np.isfinite(bar_z)]
            out[f'{prefix}_exit_rule_rate_bar_z_ge_0p5'] = float(np.mean(bar_z >= 0.5)) if bar_z.size else float('nan')
        if 'Bar_i' in frame.columns:
            bar_i = pd.to_numeric(frame['Bar_i'], errors='coerce').to_numpy(dtype=np.float64)
            bar_i = bar_i[np.isfinite(bar_i)]
            out[f'{prefix}_investment_rule_rate_bar_i_ge_0p5'] = float(np.mean(bar_i >= 0.5)) if bar_i.size else float('nan')
        if 'b' in frame.columns:
            b = pd.to_numeric(frame['b'], errors='coerce').to_numpy(dtype=np.float64)
            b = b[np.isfinite(b)]
            out[f'{prefix}_mean_leverage_b'] = float(np.mean(b)) if b.size else float('nan')
        for name in ['P', 'Q']:
            if name in frame.columns:
                v = pd.to_numeric(frame[name], errors='coerce').to_numpy(dtype=np.float64)
                v = v[np.isfinite(v)]
                out[f'{prefix}_mean_{name}'] = float(np.mean(v)) if v.size else float('nan')
        return out

    @staticmethod
    def _prefixed_delta(prefix: str, old_stats: Dict[str, float], new_stats: Dict[str, float], old_prefix: str, new_prefix: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for old_key, old_value in old_stats.items():
            if not old_key.startswith(old_prefix):
                continue
            suffix = old_key[len(old_prefix):]
            new_key = f'{new_prefix}{suffix}'
            new_value = new_stats.get(new_key)
            if new_value is None:
                continue
            if np.isfinite(old_value) and np.isfinite(new_value):
                out[f'{prefix}{suffix}_delta'] = float(new_value - old_value)
        return out

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
        eta_resample: bool = True,
        extra_tensors: Optional[Dict[str, torch.Tensor]] = None
    ) -> List[Dict[str, torch.Tensor]]:
        if parent is None or parent.numel() == 0:
            return []
        n_units = int(parent.shape[0])
        if n_units == 0:
            return []

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

        max_units = int(getattr(self.hyperparams, "max_firm_train_units", 0))
        if max_units > 0 and indices.numel() > max_units:
            logger.warning(
                "Capping firm training units from %d to %d before batching",
                indices.numel(),
                max_units
            )
            indices = indices[:max_units]

        selected_indices = indices
        compact_indices = torch.arange(selected_indices.numel(), device=selected_indices.device)
        n_batches = (indices.numel() + batch_size - 1) // batch_size
        batches = []
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, indices.numel())
            idx = selected_indices[start:end]
            batch = {
                'parent': parent[idx],
                'children': [c[idx] for c in children],
                'child0': children[0][idx] if len(children) > 0 else None,
                'child1': children[1][idx] if len(children) > 1 else None,
                'parent_index': compact_indices[start:end],
                'parent_source_index': idx,
            }
            if extra_tensors:
                for name, value in extra_tensors.items():
                    batch[name] = value[idx]
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
        rollout_initial_x_rows: List[torch.Tensor] = []
        rollout_initial_state_rows: List[torch.Tensor] = []
        rollout_future_x_rows: List[torch.Tensor] = []
        rollout_target_state_rows: List[torch.Tensor] = []
        rollout_horizon = max(0, int(getattr(self.hyperparams, "fc1_rollout_horizon", 5)))
        rollout_train_enabled = (
            bool(getattr(self.hyperparams, "fc1_recursive_aux_training_enabled", False))
            and float(getattr(self.hyperparams, "fc1_rollout_weight", 0.0)) > 0.0
        )
        rollout_diag_enabled = bool(
            getattr(self.hyperparams, "fc1_rollout_diagnostic_enabled", True)
        )
        need_rollout = bool(
            self.add_FC1loss
            and rollout_horizon > 0
            and (rollout_train_enabled or rollout_diag_enabled)
        )
        branch0_lookup: Dict[Tuple[int, int], torch.Tensor] = {}
        if need_rollout:
            if 't' not in col or 'Hatc_t1' not in col or 'LnK_t1' not in col:
                raise RuntimeError(
                    "FC1 rollout enabled but SDF macro table lacks t/Hatc_t1/LnK_t1 columns."
                )
            branch0_mask = branch == 0
            branch0_idx = torch.nonzero(branch0_mask, as_tuple=False).squeeze(-1)
            for ridx in branch0_idx.tolist():
                branch0_lookup[(int(path[ridx].item()), int(data[ridx, col['t']].item()))] = data[ridx]

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

            rollout_future_x = None
            rollout_target_states = None
            if need_rollout:
                path_id = int(group_children[0][col['path']].item())
                first_child_t = int(group_children[0][col['t']].item())
                future_x_parts = []
                target_state_parts = []
                for h in range(rollout_horizon):
                    row = branch0_lookup.get((path_id, first_child_t + h))
                    if row is None:
                        ok = False
                        break
                    future_x_parts.append(row[col['x_t1']].reshape(1))
                    target_state_parts.append(torch.stack([row[col['Hatc_t1']], row[col['LnK_t1']]]))
                if not ok:
                    continue
                rollout_future_x = torch.stack(future_x_parts, dim=0)
                rollout_target_states = torch.stack(target_state_parts, dim=0)
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
                continue
            if need_rollout:
                rollout_initial_x_rows.append(x_t.reshape(1))
                rollout_initial_state_rows.append(torch.stack([hatc_t, lnk_t]))
                rollout_future_x_rows.append(rollout_future_x)
                rollout_target_state_rows.append(rollout_target_states)

        if not parent_rows:
            return []
        parent = torch.stack(parent_rows, dim=0).to(torch.float32)
        children = [torch.stack(rows, dim=0).to(torch.float32) for rows in child_rows]
        extra_tensors = None
        if need_rollout:
            if len(rollout_future_x_rows) != parent.shape[0]:
                raise RuntimeError("FC1 rollout construction produced misaligned sequence tensors.")
            extra_tensors = {
                'fc1_rollout_initial_x': torch.stack(rollout_initial_x_rows, dim=0).to(torch.float32),
                'fc1_rollout_initial_state': torch.stack(rollout_initial_state_rows, dim=0).to(torch.float32),
                'fc1_rollout_future_x': torch.stack(rollout_future_x_rows, dim=0).to(torch.float32),
                'fc1_rollout_target_states': torch.stack(rollout_target_state_rows, dim=0).to(torch.float32),
            }
        return self._build_batches_from_parent_children(
            parent,
            children,
            batch_size=batch_size,
            eta_resample=False,
            extra_tensors=extra_tensors
        )

    def _split_sdf_table_by_path(
        self,
        sdf_table: TensorTable,
    ) -> Tuple[TensorTable, TensorTable, Dict[str, Any]]:
        col = {name: i for i, name in enumerate(sdf_table.columns)}
        if sdf_table.data.numel() == 0 or 'path' not in col:
            return sdf_table, sdf_table, {
                'sdf_fc1_holdout_active': False,
                'sdf_fc1_holdout_reason': 'empty_or_missing_path',
            }
        val_fraction = float(getattr(self.hyperparams, "sdf_fc1_val_fraction", 0.2))
        val_fraction = min(max(val_fraction, 0.0), 0.5)
        path = sdf_table.data[:, col['path']].long()
        unique_paths = torch.unique(path)
        allow_in_sample = bool(getattr(self.hyperparams, "allow_in_sample_sdf_gate_for_debug", False))
        if val_fraction <= 0.0:
            if not allow_in_sample:
                raise RuntimeError(
                    "Path-level SDF validation is disabled by sdf_fc1_val_fraction=0. "
                    "Set allow_in_sample_sdf_gate_for_debug=True only for debug runs."
                )
            return sdf_table, sdf_table, {
                'sdf_fc1_holdout_active': False,
                'sdf_fc1_holdout_reason': 'disabled_debug_in_sample',
                'sdf_fc1_holdout_n_paths': int(unique_paths.numel()),
            }
        if unique_paths.numel() < 2:
            if not allow_in_sample:
                raise RuntimeError(
                    "Path-level SDF validation requires at least two paths. "
                    "Set allow_in_sample_sdf_gate_for_debug=True only for debug runs."
                )
            return sdf_table, sdf_table, {
                'sdf_fc1_holdout_active': False,
                'sdf_fc1_holdout_reason': 'insufficient_paths_debug_in_sample',
                'sdf_fc1_holdout_n_paths': int(unique_paths.numel()),
            }
        seed = int(getattr(self.hyperparams, "sdf_fc1_val_seed", 12345))
        generator = torch.Generator(device=unique_paths.device)
        generator.manual_seed(seed)
        order = torch.randperm(unique_paths.numel(), generator=generator, device=unique_paths.device)
        n_val = int(round(unique_paths.numel() * val_fraction))
        n_val = min(max(1, n_val), unique_paths.numel() - 1)
        val_paths = unique_paths[order[:n_val]]
        train_paths = unique_paths[order[n_val:]]
        val_mask = torch.isin(path, val_paths)
        train_mask = torch.isin(path, train_paths)
        train_table = TensorTable(data=sdf_table.data[train_mask], columns=list(sdf_table.columns))
        val_table = TensorTable(data=sdf_table.data[val_mask], columns=list(sdf_table.columns))
        return train_table, val_table, {
            'sdf_fc1_holdout_active': True,
            'sdf_fc1_holdout_seed': seed,
            'sdf_fc1_holdout_fraction': val_fraction,
            'sdf_fc1_train_paths': int(train_paths.numel()),
            'sdf_fc1_val_paths': int(val_paths.numel()),
            'sdf_fc1_train_rows': int(train_table.data.shape[0]),
            'sdf_fc1_val_rows': int(val_table.data.shape[0]),
        }
    
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
        eta_children: Optional[List[torch.Tensor]] = None
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

    def _compute_conditional_signed_foc_terms(
        self,
        foc_residuals: List[torch.Tensor],
        eta_children: List[torch.Tensor],
        z_parent: torch.Tensor,
        alpha_z: float,
        beta_z: float,
        z0: float
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

        foc_signed_moment = foc_cond_signed[active_bool].mean()
        loss_foc = foc_signed_moment.pow(2)

        # 仅在 eta 活跃子样本上评估 z-penalty，避免被 eta=0 样本稀释。
        penalty_z_foc = compute_z_penalty(
            foc_cond_abs[active_bool],
            z_parent[active_bool],
            alpha_z,
            beta_z,
            z0
        )

        with torch.no_grad():
            diag = {
                'foc_active_ratio': float(active_mask.mean().item()),
                'foc_signed_moment': float(foc_signed_moment.item()),
                'foc_cond_abs_mean': float(foc_cond_abs[active_bool].mean().item()),
                'foc_active_n': float(active_n),
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
        Q-only 阶段冻结非 Q 参数，保持损失方程结构不变。
        """
        if 'policy_value' not in self.models:
            return

        model = self.models['policy_value']
        if not bool(getattr(self.hyperparams, "q_freeze_non_q_in_pretrain", True)):
            enable = False

        if enable:
            if self._policy_q_freeze_active:
                return
            self._policy_value_grad_backup = {
                name: p.requires_grad for name, p in model.named_parameters()
            }
            for p in model.parameters():
                p.requires_grad = False
            # Q 头始终可训练
            for p in model.q_head.parameters():
                p.requires_grad = True
            scope = str(getattr(self.hyperparams, "q_pretrain_trainable_scope", "q_path")).lower()
            if scope not in {"q_head_only", "q_path"}:
                scope = "q_path"
            # q_path: 允许共享表征与 Q 头联合适配
            if scope == "q_path":
                for p in model.q_encoder.parameters():
                    p.requires_grad = True
            self._policy_q_freeze_active = True
        else:
            if not self._policy_q_freeze_active:
                return
            for name, p in model.named_parameters():
                if name in self._policy_value_grad_backup:
                    p.requires_grad = self._policy_value_grad_backup[name]
            self._policy_value_grad_backup = {}
            self._policy_q_freeze_active = False

    def _set_policy_bp_only_freeze(self, enable: bool):
        """
        bp-only 精修阶段：仅更新 bp0/bpI 头参数。
        """
        if 'policy_value' not in self.models:
            return

        model = self.models['policy_value']
        if enable:
            if self._policy_bp_freeze_active:
                return
            self._policy_bp_grad_backup = {
                name: p.requires_grad for name, p in model.named_parameters()
            }
            for p in model.parameters():
                p.requires_grad = False
            for p in model.policy_encoder.parameters():
                p.requires_grad = True
            for p in model.bp0_head.parameters():
                p.requires_grad = True
            for p in model.bpi_head.parameters():
                p.requires_grad = True
            self._policy_bp_freeze_active = True
        else:
            if not self._policy_bp_freeze_active:
                return
            for name, p in model.named_parameters():
                if name in self._policy_bp_grad_backup:
                    p.requires_grad = self._policy_bp_grad_backup[name]
            self._policy_bp_grad_backup = {}
            self._policy_bp_freeze_active = False

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

    def _set_sdf_training_phase_freeze(self) -> None:
        """
        Explicit SDF/FC1 phase freeze.

        FC1_ONLY trains only FC1; SDF phases freeze FC1 and train SDF/value.
        """
        if 'sdf_fc1' not in self.models:
            return
        model = self.models['sdf_fc1']
        phase = getattr(self, "sdf_training_phase", SDFTrainingPhase.EPISODE0_BOOTSTRAP)
        phase = SDFTrainingPhase(phase)

        for p in model.parameters():
            p.requires_grad = False

        if phase == SDFTrainingPhase.FC1_ONLY:
            if not hasattr(model, "fc1_model"):
                raise RuntimeError("SDF/FC1 model has no fc1_model for FC1_ONLY phase.")
            for p in model.fc1_model.parameters():
                p.requires_grad = True
        elif phase in {
            SDFTrainingPhase.EPISODE0_BOOTSTRAP,
            SDFTrainingPhase.SDF_TRUE_ONLY,
            SDFTrainingPhase.SDF_RECURSIVE_ONLY,
        }:
            if not hasattr(model, "sdf_model") or not hasattr(model, "value_model"):
                raise RuntimeError("SDF/FC1 model must expose sdf_model and value_model for SDF-only phases.")
            for p in model.sdf_model.parameters():
                p.requires_grad = True
            for p in model.value_model.parameters():
                p.requires_grad = True
        elif phase == SDFTrainingPhase.JOINT_DISABLED:
            raise RuntimeError(
                "Joint FC1/SDF training is disabled until separate-stage validation has passed."
            )
        else:
            raise ValueError(f"Unknown SDF training phase: {phase}")

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

    def _sdf_fresh_pair_enabled(self) -> bool:
        return bool(getattr(self.hyperparams, "sdf_fresh_pair_enabled", False))

    def _sdf_child_bank_seed(self, epoch: int) -> int:
        base_seed = int(getattr(self.hyperparams, "sdf_child_bank_seed", 12345))
        return int(base_seed + 10000 * int(self.episode_id) + int(epoch))

    def _ensure_sdf_bank_capacity(
        self,
        parent_index: torch.Tensor,
        *,
        dtype: torch.dtype,
        epoch: Optional[int] = None,
    ) -> None:
        if parent_index.numel() == 0:
            return
        bank_size = int(getattr(self.hyperparams, "sdf_child_bank_size", 16))
        if bank_size < 2:
            raise ValueError("sdf_child_bank_size must be >= 2 when fresh pairs are enabled.")
        n_children = int(getattr(self.hyperparams, "sdf_signed_aio_n_children", 2))
        if n_children != 2:
            raise ValueError("Only sdf_signed_aio_n_children=2 is currently supported.")

        epoch_i = int(self._current_epoch_idx if epoch is None else epoch)
        device = self.device
        parent_index = parent_index.to(device=device, dtype=torch.long).reshape(-1)
        required = int(parent_index.max().item()) + 1
        seed = self._sdf_child_bank_seed(epoch_i)
        prior_key = self._sdf_shock_bank_key
        needs_create = (
            self._sdf_shock_bank is None
            or self._sdf_shock_bank.eps.shape[0] < required
            or self._sdf_shock_bank.bank_size != bank_size
            or self._sdf_shock_bank.eps.device != device
            or self._sdf_shock_bank.eps.dtype != dtype
            or self._sdf_shock_bank_episode_id != int(self.episode_id)
        )
        key_capacity = required if needs_create else int(self._sdf_shock_bank.eps.shape[0])
        key = (
            int(self.episode_id),
            epoch_i,
            key_capacity,
            bank_size,
            str(device),
            str(dtype),
        )
        if needs_create:
            self._sdf_shock_bank = SDFShockBank.create(
                n_parents=required,
                bank_size=bank_size,
                device=device,
                base_seed=seed,
                dtype=dtype,
            )
            self._sdf_shock_bank_n_parents = required
            self._sdf_shock_bank_epoch = epoch_i
            self._sdf_shock_bank_episode_id = int(self.episode_id)
            self._sdf_shock_bank_key = key
        else:
            refresh_epochs = max(1, int(getattr(self.hyperparams, "sdf_child_bank_refresh_epochs", 1)))
            if epoch_i % refresh_epochs == 0 and self._sdf_shock_bank_key != key:
                self._sdf_shock_bank.refresh_(seed=seed)
                self._sdf_shock_bank_epoch = epoch_i
                self._sdf_shock_bank_key = key

        if self._sdf_pair_generator is None or prior_key != key:
            pair_seed = seed + 7919
            self._sdf_pair_generator = _make_generator(device)
            self._sdf_pair_generator.manual_seed(pair_seed)
        self._sdf_shock_bank_key = key

    def _prepare_sdf_shock_bank_for_epoch(
        self,
        batches: List[Dict[str, torch.Tensor]],
        epoch: int,
        train_modules: List[str],
    ) -> None:
        if 'sdf_fc1' not in train_modules or not self._sdf_fresh_pair_enabled():
            return
        if not batches:
            return

        max_parent_index = -1
        first_parent = None
        for batch in batches:
            if first_parent is None and 'parent' in batch:
                first_parent = batch['parent']
            parent_index = batch.get('parent_index')
            if parent_index is None:
                raise ValueError("SDF fresh pair sampling requires batch['parent_index'].")
            if parent_index.numel() > 0:
                max_parent_index = max(max_parent_index, int(parent_index.max().item()))
        if max_parent_index < 0 or first_parent is None:
            return

        dtype = first_parent.dtype
        all_parent_index = torch.cat([batch['parent_index'].reshape(-1) for batch in batches], dim=0)
        self._ensure_sdf_bank_capacity(all_parent_index, dtype=dtype, epoch=epoch)
        logger.info(
            "SDF shock bank ready | episode=%s epoch=%s n_parents=%s bank_size=%s refresh_id=%s seed=%s",
            self.episode_id,
            epoch,
            self._sdf_shock_bank.eps.shape[0] if self._sdf_shock_bank is not None else 0,
            self._sdf_shock_bank.bank_size if self._sdf_shock_bank is not None else -1,
            self._sdf_shock_bank.refresh_id if self._sdf_shock_bank is not None else -1,
            self._sdf_child_bank_seed(epoch),
        )

    def _sample_sdf_fresh_x_children(
        self,
        parent: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        if not self._sdf_fresh_pair_enabled():
            return None, {
                'sdf_fresh_pair_enabled': 0.0,
            }

        wealth_only = bool(getattr(self.hyperparams, "sdf_child_bank_wealth_only", True))
        if self.add_FC1loss and not wealth_only:
            raise ValueError(
                "sdf_child_bank_wealth_only=False would replace FC1 reconstruction children "
                "without true targets; keep it True for Treatment B semantics."
            )

        parent_index = batch.get('parent_index')
        if parent_index is None:
            raise ValueError("SDF fresh pair sampling requires batch['parent_index'].")
        self._ensure_sdf_bank_capacity(parent_index, dtype=parent.dtype, epoch=self._current_epoch_idx)

        eps1, eps2, j1, j2 = self._sdf_shock_bank.sample_pair(
            parent_index.to(self.device),
            self._sdf_pair_generator,
        )
        x_children = shocks_to_x_children(
            x_parent=parent[:, 4:5],
            eps1=eps1,
            eps2=eps2,
            rho_x=self.config.RHO_X,
            sigma_x=self.config.SIGMA_X,
            xbar=self.config.XBAR,
        )
        diag = shock_pair_diagnostics(
            eps1,
            eps2,
            j1,
            j2,
            self._sdf_shock_bank.bank_size,
            parent_index=parent_index,
        )
        diag.update({
            'sdf_fresh_pair_enabled': 1.0,
            'sdf_fresh_pair_requested': 1.0,
            'sdf_fresh_pair_used': 1.0,
            'sdf_bank_size': float(self._sdf_shock_bank.bank_size),
            'sdf_bank_refresh_id': float(self._sdf_shock_bank.refresh_id),
            'sdf_bank_wealth_only': float(1.0 if wealth_only else 0.0),
            'sdf_bank_n_parents': float(self._sdf_shock_bank.eps.shape[0]),
        })
        return x_children, diag
    
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
        self._set_policy_bp_only_freeze(False)
        q_only_step = (
            'policy_value' in train_modules and
            set(policy_loss_terms) == {'q'} and
            len(policy_loss_terms) == 1
        )
        bp_only_step = (
            'policy_value' in train_modules and
            set(policy_loss_terms) == {'p0', 'pi'} and
            len(policy_loss_terms) == 2 and
            bool(getattr(self, "_bp_only_stage", False))
        )
        self._set_policy_q_only_freeze(q_only_step)
        if not q_only_step:
            self._set_policy_bp_only_freeze(bp_only_step)
        if 'sdf_fc1' in train_modules:
            self._set_sdf_training_phase_freeze()
        
        losses = {}
        
        # 设置训练模式
        for name in train_modules:
            if name in self.models:
                self.models[name].train()
        
        # 清零梯度
        for name in train_modules:
            if name in self.optimizers:
                self.optimizers[name].zero_grad()
        
        stepped_modules = set()
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
            if 'fc2' in train_modules and 'fc2' in self.models:
                if self.episode_id > 0 and isinstance(batch, pd.DataFrame):
                    batch.to_csv(f"fc2_input_episode{self.episode_id}_ori.csv", index=False)
                    batch = convert_tree_fast(batch)
                    batch = trim_child_only_ids(batch)
                    batch.to_csv(f"fc2_input_episode{self.episode_id}.csv", index=False)
                
                fc2_loss = self._compute_fc2_loss(batch)
                losses['fc2'] = fc2_loss.item()
                total_loss = total_loss + self.weight_scheduler['fc2'] * fc2_loss
            
            losses['total'] = total_loss.item()
            
            # 反向传播
            if total_loss.requires_grad:
                total_loss.backward()

                bad_grad_params: Dict[str, List[str]] = {}
                for name in train_modules:
                    if name in self.models:
                        bad = self._nonfinite_gradient_params(self.models[name])
                        if bad:
                            bad_grad_params[name] = bad

                if bad_grad_params:
                    self._nonfinite_grad_streak += 1
                    self._nonfinite_grad_total += 1
                    self._last_nonfinite_grad_params = bad_grad_params
                    losses['nonfinite_grad'] = 1.0
                    losses['nonfinite_grad_streak'] = float(self._nonfinite_grad_streak)
                    logger.warning(
                        "Non-finite gradient detected at episode=%s step=%s streak=%d modules=%s",
                        self.episode_id,
                        self.step_count,
                        self._nonfinite_grad_streak,
                        {k: v[:5] for k, v in bad_grad_params.items()}
                    )
                    fail_after = int(getattr(self.hyperparams, "nonfinite_grad_fail_after", 3))
                    if fail_after > 0 and self._nonfinite_grad_streak >= fail_after:
                        raise NumericalStageFailure(
                            f"Repeated non-finite gradients for {self._nonfinite_grad_streak} "
                            f"consecutive steps at episode={self.episode_id}; "
                            f"bad params={bad_grad_params}"
                        )
                    skip_step = bool(getattr(self.hyperparams, "nonfinite_grad_skip_step", True))
                    if skip_step:
                        for name in train_modules:
                            if name in self.optimizers:
                                self.optimizers[name].zero_grad(set_to_none=True)
                    else:
                        for name in train_modules:
                            if name in self.models:
                                gradient_protection(
                                    self.models[name].parameters(),
                                    max_norm=self.hyperparams.max_grad_norm,
                                    nan_to_num=True
                                )
                        for name in train_modules:
                            if name in self.optimizers:
                                self.optimizers[name].step()
                                stepped_modules.add(name)
                        self._maybe_update_firm_target(train_modules)
                else:
                    self._nonfinite_grad_streak = 0
                    losses['nonfinite_grad'] = 0.0

                    # 梯度保护
                    for name in train_modules:
                        if name in self.models:
                            grad_norm, had_nan = gradient_protection(
                                self.models[name].parameters(),
                                max_norm=self.hyperparams.max_grad_norm,
                                nan_to_num=False
                            )
                            losses[f'{name}_grad_norm'] = grad_norm

                            if had_nan:
                                logger.warning(f"NaN gradient detected in {name}")
                    if 'policy_value' in train_modules and 'policy_value' in self.models:
                        losses.update(self._policy_value_grad_group_norms())

                    # 优化器步骤
                    for name in train_modules:
                        if name in self.optimizers:
                            self.optimizers[name].step()
                            stepped_modules.add(name)
                    self._maybe_update_firm_target(train_modules)
        finally:
            self._set_policy_q_only_freeze(False)
            self._set_policy_bp_only_freeze(False)
        
        # 更新调度器
        self.weight_scheduler.step(losses)
        for scheduler in self.lr_schedulers.values():
            scheduler.step()

        if 'sdf_fc1' in stepped_modules:
            self.sdf_fc1_step_count = int(getattr(self, "sdf_fc1_step_count", 0)) + 1
            losses['sdf_fc1_step_count'] = float(self.sdf_fc1_step_count)

        self.step_count += 1
        
        # 记录历史
        for k, v in losses.items():
            v_norm = self._normalize_metric_value(k, v)
            if not self._is_numeric_metric_value(v_norm):
                continue
            if k not in self.loss_history:
                self.loss_history[k] = []
            self.loss_history[k].append(float(v_norm))
        
        return losses

    def _extract_fc1_targets(
        self,
        children_t: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if children_t.shape[-1] >= 10:
            return children_t[:, :, 8:9], children_t[:, :, 9:10]
        if children_t.shape[-1] >= 9:
            return children_t[:, :, 7:8], children_t[:, :, 8:9]
        return None, None

    def _compute_fc1_rollout_loss(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Optional multi-step FC1 rollout loss.

        This intentionally requires explicit same-path time-series tensors and
        never treats child branches as a time sequence.
        """
        required = {
            "fc1_rollout_initial_state",
            "fc1_rollout_initial_x",
            "fc1_rollout_future_x",
            "fc1_rollout_target_states",
        }
        missing = required.difference(batch.keys())
        if missing:
            raise RuntimeError(f"FC1 rollout enabled but batch is missing: {sorted(missing)}")

        initial_state = batch["fc1_rollout_initial_state"].to(self.device)
        initial_x = batch["fc1_rollout_initial_x"].to(self.device)
        future_x = batch["fc1_rollout_future_x"].to(self.device)
        target_states = batch["fc1_rollout_target_states"].to(self.device)
        horizon = min(
            int(getattr(self.hyperparams, "fc1_rollout_horizon", 5)),
            int(future_x.shape[1]),
            int(target_states.shape[1]),
        )
        if horizon <= 0:
            return torch.tensor(0.0, device=self.device)

        hatc = initial_state[:, 0:1]
        lnk = initial_state[:, 1:2]
        x_prev = initial_x
        weights = torch.ones(horizon, device=self.device, dtype=future_x.dtype) / float(horizon)
        losses = []
        for h in range(horizon):
            x_curr = future_x[:, h, :]
            _, _, _, hatc_next, lnk_next = model.forward_step(
                x_prev=x_prev,
                x_curr=x_curr.unsqueeze(1),
                hatcf_prev=hatc,
                lnkf_prev=lnk,
                return_physical=True,
            )
            hatc = hatc_next[:, 0, :]
            lnk = lnk_next[:, 0, :]
            pred = torch.cat([hatc, lnk], dim=1)
            target = target_states[:, h, :]
            losses.append(weights[h] * (pred - target).pow(2).mean())
            x_prev = x_curr
        return torch.stack(losses).sum()

    def _compute_fc1_only_loss(
        self,
        batch: Dict[str, torch.Tensor],
        parent: torch.Tensor,
        children_t: torch.Tensor,
    ) -> torch.Tensor:
        model = self.models['sdf_fc1']
        device = self.device
        hatc_true, lnk_true = self._extract_fc1_targets(children_t)
        zero = torch.tensor(0.0, device=device)

        recon_loss = zero
        recon_loss_hatc = zero
        recon_loss_lnk = zero
        recon_loss_dlnk = zero
        recon_loss_forecast = zero
        recon_loss_forecast_hatc = zero
        recon_loss_forecast_lnk = zero
        recon_loss_forecast_dlnk = zero
        delta_penalty = zero
        delta_penalty_hatc = zero
        delta_penalty_lnk = zero
        jacobian_penalty = zero
        jacobian_penalty_hatc = zero
        jacobian_penalty_lnk = zero
        rollout_loss = zero
        jacobian_penalty_active = False

        hatc_w = float(getattr(self.hyperparams, "fc1_hatc_recon_weight", 1.0))
        lnk_w = float(getattr(self.hyperparams, "fc1_lnk_recon_weight", 0.25))
        recon_weight = float(getattr(self.hyperparams, "fc1_recon_weight", 1.0))
        forecast_weight = float(getattr(self.hyperparams, "fc1_forecast_recon_weight", 0.25))
        rollout_weight = float(getattr(self.hyperparams, "fc1_rollout_weight", 0.5))
        delta_weight = float(getattr(self.hyperparams, "fc1_delta_penalty_weight", 1.0))
        jac_weight = float(getattr(self.hyperparams, "fc1_jacobian_penalty_weight", 0.0))
        recursive_aux_enabled = bool(
            getattr(self.hyperparams, "fc1_recursive_aux_training_enabled", False)
        )
        jac_interval = int(getattr(self.hyperparams, "fc1_jacobian_penalty_interval", 10))
        sdf_step = int(getattr(self, "sdf_fc1_step_count", 0))
        compute_jac = (
            recursive_aux_enabled
            and jac_weight > 0.0
            and jac_interval > 0
            and sdf_step % jac_interval == 0
        )
        delta_hatc_abs_max = float(getattr(self.hyperparams, "fc1_delta_hatc_abs_max", 0.50))
        delta_lnk_abs_max = float(getattr(self.hyperparams, "fc1_delta_lnk_abs_max", 0.30))

        if hatc_true is None or lnk_true is None or parent.shape[1] < 9:
            raise RuntimeError(
                "FC1_ONLY phase requires parent true Hatc/LnK and child true Hatc/LnK targets."
            )
        else:
            _, _, _, hatc_pred, lnk_pred = model.forward_step(
                x_prev=parent[:, 4:5],
                x_curr=children_t[:, :, 4:5],
                hatcf_prev=parent[:, 7:8],
                lnkf_prev=parent[:, 8:9],
                return_physical=True,
            )
            recon_loss_hatc = (hatc_pred - hatc_true).pow(2).mean()
            recon_loss_lnk = (lnk_pred - lnk_true).pow(2).mean()
            recon_loss_dlnk = ((lnk_pred - parent[:, 8:9].unsqueeze(1)) - (lnk_true - parent[:, 8:9].unsqueeze(1))).pow(2).mean()
            recon_loss = hatc_w * recon_loss_hatc + lnk_w * recon_loss_lnk

            if recursive_aux_enabled:
                hatcf_prev = parent[:, 5:6].detach().clone()
                lnkf_prev = parent[:, 6:7].detach().clone()
                if compute_jac:
                    hatcf_prev.requires_grad_(True)
                    lnkf_prev.requires_grad_(True)
                _, _, _, hatc_forecast, lnk_forecast = model.forward_step(
                    x_prev=parent[:, 4:5],
                    x_curr=children_t[:, :, 4:5],
                    hatcf_prev=hatcf_prev,
                    lnkf_prev=lnkf_prev,
                    return_physical=True,
                )
                recon_loss_forecast_hatc = (hatc_forecast - hatc_true).pow(2).mean()
                recon_loss_forecast_lnk = (lnk_forecast - lnk_true).pow(2).mean()
                recon_loss_forecast_dlnk = ((lnk_forecast - lnkf_prev.unsqueeze(1)) - (lnk_true - lnkf_prev.unsqueeze(1))).pow(2).mean()
                recon_loss_forecast = hatc_w * recon_loss_forecast_hatc + lnk_w * recon_loss_forecast_lnk
                d_hatcf = hatc_forecast - parent[:, 5:6].unsqueeze(1)
                d_lnkf = lnk_forecast - parent[:, 6:7].unsqueeze(1)
                delta_penalty_hatc = torch.relu(d_hatcf.abs() - delta_hatc_abs_max).pow(2).mean()
                delta_penalty_lnk = torch.relu(d_lnkf.abs() - delta_lnk_abs_max).pow(2).mean()
                delta_penalty = hatc_w * delta_penalty_hatc + lnk_w * delta_penalty_lnk

                if compute_jac:
                    jacobian_penalty_active = True
                    grad_hat_wrt_hat = torch.autograd.grad(hatc_forecast.sum(), hatcf_prev, create_graph=True, retain_graph=True)[0]
                    grad_hat_wrt_lnk = torch.autograd.grad(hatc_forecast.sum(), lnkf_prev, create_graph=True, retain_graph=True)[0]
                    grad_lnk_wrt_hat = torch.autograd.grad(lnk_forecast.sum(), hatcf_prev, create_graph=True, retain_graph=True)[0]
                    grad_lnk_wrt_lnk = torch.autograd.grad(lnk_forecast.sum(), lnkf_prev, create_graph=True, retain_graph=True)[0]
                    jacobian_penalty_hatc = grad_hat_wrt_hat.pow(2).mean() + grad_hat_wrt_lnk.pow(2).mean()
                    jacobian_penalty_lnk = grad_lnk_wrt_hat.pow(2).mean() + grad_lnk_wrt_lnk.pow(2).mean()
                    jacobian_penalty = hatc_w * jacobian_penalty_hatc + lnk_w * jacobian_penalty_lnk

            if recursive_aux_enabled and rollout_weight > 0.0:
                rollout_loss = self._compute_fc1_rollout_loss(model, batch)

        forecast_weight_eff = float(forecast_weight) if recursive_aux_enabled else 0.0
        rollout_weight_eff = float(rollout_weight) if recursive_aux_enabled else 0.0
        delta_weight_eff = float(delta_weight) if recursive_aux_enabled else 0.0
        jac_weight_eff = float(jac_weight) if recursive_aux_enabled else 0.0

        total = recon_weight * recon_loss
        if recursive_aux_enabled:
            total = (
                total
                + forecast_weight_eff * recon_loss_forecast
                + rollout_weight_eff * rollout_loss
                + delta_weight_eff * delta_penalty
                + jac_weight_eff * jacobian_penalty
            )
        self._latest_sdf_terms = {
            'sdf_training_phase': SDFTrainingPhase.FC1_ONLY.value,
            'sdf_main_loss': 0.0,
            'sdf_total_loss': float(total.detach().item()),
            'sdf_main_weight_effective': 0.0,
            'sdf_moment_weight_effective': 0.0,
            'sdf_anchor_weight_effective': 0.0,
            'fc1_recon_weight_effective': float(recon_weight),
            'fc1_forecast_weight_effective': float(forecast_weight_eff),
            'fc1_rollout_weight_effective': float(rollout_weight_eff),
            'fc1_delta_weight_effective': float(delta_weight_eff),
            'fc1_jacobian_weight_effective': float(jac_weight_eff if jacobian_penalty_active else 0.0),
            'fc1_recursive_aux_training_enabled': float(1.0 if recursive_aux_enabled else 0.0),
            'sdf_recon_loss': float(recon_loss.detach().item()),
            'sdf_fc1_true_recon': float(recon_loss.detach().item()),
            'sdf_recon_loss_hatc': float(recon_loss_hatc.detach().item()),
            'sdf_recon_loss_lnk': float(recon_loss_lnk.detach().item()),
            'sdf_recon_loss_dlnk': float(recon_loss_dlnk.detach().item()),
            'sdf_recon_loss_forecast': float(recon_loss_forecast.detach().item()),
            'sdf_fc1_forecast_recon': float(recon_loss_forecast.detach().item()),
            'sdf_recon_loss_forecast_hatc': float(recon_loss_forecast_hatc.detach().item()),
            'sdf_recon_loss_forecast_lnk': float(recon_loss_forecast_lnk.detach().item()),
            'sdf_recon_loss_forecast_dlnk': float(recon_loss_forecast_dlnk.detach().item()),
            'sdf_fc1_rollout_loss': float(rollout_loss.detach().item()),
            'sdf_delta_penalty': float(delta_penalty.detach().item()),
            'sdf_delta_penalty_hatc': float(delta_penalty_hatc.detach().item()),
            'sdf_delta_penalty_lnk': float(delta_penalty_lnk.detach().item()),
            'sdf_jacobian_penalty': float(jacobian_penalty.detach().item()),
            'sdf_jacobian_penalty_hatc': float(jacobian_penalty_hatc.detach().item()),
            'sdf_jacobian_penalty_lnk': float(jacobian_penalty_lnk.detach().item()),
            'sdf_jacobian_penalty_active': float(1.0 if jacobian_penalty_active else 0.0),
            'sdf_jacobian_penalty_interval': float(jac_interval),
            'sdf_fc1_step_count': float(sdf_step),
            'sdf_fresh_pair_requested': 0.0,
            'sdf_fresh_pair_used': 0.0,
            'sdf_fresh_pair_enabled': 0.0,
            'sdf_use_true_prev_macro': 1.0,
        }
        self._latest_sdf_diag = {}
        return total
    
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
        phase = SDFTrainingPhase(getattr(self, "sdf_training_phase", SDFTrainingPhase.EPISODE0_BOOTSTRAP))
        if phase == SDFTrainingPhase.FC1_ONLY:
            return self._compute_fc1_only_loss(batch, parent, children_t)
        if phase == SDFTrainingPhase.JOINT_DISABLED:
            raise RuntimeError("Joint FC1/SDF loss is disabled by explicit SDF training phase.")

        use_true_prev_macro = False
        if phase == SDFTrainingPhase.EPISODE0_BOOTSTRAP:
            if int(getattr(self, "episode_id", -1)) != 0:
                raise RuntimeError("EPISODE0_BOOTSTRAP requires episode_id == 0.")
            if bool(getattr(self, "add_FC1loss", False)):
                raise RuntimeError("Episode 0 bootstrap must not train FC1.")
            if parent.shape[1] < 7:
                raise RuntimeError("Episode 0 bootstrap requires x_t, Hatcf_t and LnKF_t.")
            use_true_prev_macro = False
        elif phase == SDFTrainingPhase.SDF_TRUE_ONLY:
            if parent.shape[1] < 9:
                raise RuntimeError("SDF_TRUE_ONLY requires true Hatc_t and LnK_t in the parent batch.")
            use_true_prev_macro = True
        elif phase == SDFTrainingPhase.SDF_RECURSIVE_ONLY:
            if parent.shape[1] < 9:
                raise RuntimeError("SDF_RECURSIVE_ONLY requires true Hatc_t/LnK_t for its true-state baseline.")
            use_true_prev_macro = False
        elif self._fc1_teacher_forcing_stage and parent.shape[1] >= 9:
            use_true_prev_macro = True
        elif self.add_FC1loss:
            use_true_prev_macro = bool(
                parent.shape[1] >= 9 and
                getattr(self.hyperparams, "fc1_use_true_macro_state_in_stage2", True)
            )
        c_prev_input = parent[:, 7:8] if use_true_prev_macro else parent[:, 5:6]
        k_prev_input = parent[:, 8:9] if use_true_prev_macro else parent[:, 6:7]

        fixed_children_x = children_t[:, :, 4:5]
        fresh_pair_requested = self._sdf_fresh_pair_enabled()
        use_fresh_wealth = fresh_pair_requested and not bool(
            getattr(self, "_fc1_teacher_forcing_stage", False)
        )
        if use_fresh_wealth:
            x_children_fresh, fresh_pair_diag = self._sample_sdf_fresh_x_children(parent, batch)
        else:
            x_children_fresh = None
            fresh_pair_diag = {
                'sdf_fresh_pair_requested': float(1.0 if fresh_pair_requested else 0.0),
                'sdf_fresh_pair_used': 0.0,
                'sdf_fresh_pair_enabled': float(1.0 if fresh_pair_requested else 0.0),
            }
        if use_fresh_wealth:
            if self.add_FC1loss:
                x_children_all = torch.cat([x_children_fresh, fixed_children_x], dim=1)
                wealth_slice = slice(0, 2)
                recon_slice = slice(2, 4)
            else:
                x_children_all = x_children_fresh
                wealth_slice = slice(0, 2)
                recon_slice = slice(0, 2)
        else:
            x_children_all = fixed_children_x
            wealth_slice = slice(0, 2)
            recon_slice = slice(0, 2)

        # 前向传播：fresh wealth pair 与固定 recon pair 共享 parent forward
        w_parent, w_children_all, M_all, c_children_all, k_children_all = model.forward_step(
            x_prev=parent[:, 4:5],
            x_curr=x_children_all,
            hatcf_prev=c_prev_input,
            lnkf_prev=k_prev_input,
            return_physical=True
        )
        w_children_wealth = w_children_all[:, wealth_slice]
        M_wealth = M_all[:, wealth_slice]
        c_children_wealth = c_children_all[:, wealth_slice]
        k_children_wealth = k_children_all[:, wealth_slice]
        c_children_recon = c_children_all[:, recon_slice]
        k_children_recon = k_children_all[:, recon_slice]
        
        # 提取父节点状态并计算 w
        c_parent = c_prev_input
        k_parent = k_prev_input
        
        residual_mode = str(
            getattr(self.hyperparams, "sdf_wealth_residual_mode", "raw")
        ).lower()
        normalized_logr_clip = float(
            getattr(self.hyperparams, "sdf_normalized_logr_clip", 20.0)
        )

        def _select_wealth_residuals(residual_pack: Dict[str, torch.Tensor]) -> torch.Tensor:
            if residual_mode == "raw":
                return residual_pack["raw"]
            if residual_mode == "normalized_ratio":
                return residual_pack["normalized"]
            raise ValueError(f"Unknown sdf_wealth_residual_mode={residual_mode!r}")

        residual_pack = loss_fn.compute_wealth_residuals(
            w_parent=w_parent.squeeze(-1),
            w_children=w_children_wealth.squeeze(-1),
            k_parent=k_parent.squeeze(-1),
            k_children=k_children_wealth.squeeze(-1),
            c_parent=c_parent.squeeze(-1),
            c_children=c_children_wealth.squeeze(-1),
            normalized_logr_clip=normalized_logr_clip,
        )
        raw_residuals = residual_pack["raw"]
        normalized_residuals = residual_pack["normalized"]
        residuals = _select_wealth_residuals(residual_pack)
        main_loss, wealth_main_details = loss_fn.compute_wealth_main_loss(residuals)
        _, raw_wealth_details = loss_fn.compute_wealth_main_loss(raw_residuals)
        _, normalized_wealth_details = loss_fn.compute_wealth_main_loss(normalized_residuals)
        true_state_main_loss = main_loss
        if phase == SDFTrainingPhase.SDF_RECURSIVE_ONLY:
            w_parent_true, w_children_true, _, c_children_true, k_children_true = model.forward_step(
                x_prev=parent[:, 4:5],
                x_curr=fixed_children_x,
                hatcf_prev=parent[:, 7:8],
                lnkf_prev=parent[:, 8:9],
                return_physical=True,
            )
            residual_pack_true = loss_fn.compute_wealth_residuals(
                w_parent=w_parent_true.squeeze(-1),
                w_children=w_children_true.squeeze(-1),
                k_parent=parent[:, 8:9].squeeze(-1),
                k_children=k_children_true.squeeze(-1),
                c_parent=parent[:, 7:8].squeeze(-1),
                c_children=c_children_true.squeeze(-1),
                normalized_logr_clip=normalized_logr_clip,
            )
            residuals_true = _select_wealth_residuals(residual_pack_true)
            true_state_main_loss, _ = loss_fn.compute_wealth_main_loss(residuals_true)

        moment_loss = torch.tensor(0.0, device=self.device)
        M_use = M_wealth.squeeze(-1) if M_wealth.dim() == 3 else M_wealth
        if M_use.dim() == 1:
            M_use = M_use.unsqueeze(-1)
        for j in range(M_use.shape[1]):
            L1, L2 = moment_penalty(M_use[:, j], loss_fn.mu_lo, loss_fn.mu_hi, loss_fn.var_hi)
            moment_loss = moment_loss + L1 + L2

        euler_weight = 1.0
        recursive_euler_weight = 0.0
        # Explicit SDF phases use phase-specific weights to avoid legacy stage1/stage2 ambiguity.
        if phase == SDFTrainingPhase.EPISODE0_BOOTSTRAP:
            euler_weight = 1.0
            recursive_euler_weight = 0.0
            moment_weight = float(getattr(self.hyperparams, "sdf_stage1_moment_weight", 5.0))
        elif phase == SDFTrainingPhase.SDF_TRUE_ONLY:
            euler_weight = float(getattr(self.hyperparams, "sdf_euler_weight", 1.0))
            moment_weight = float(getattr(self.hyperparams, "sdf_true_moment_weight", 5e-4))
        elif phase == SDFTrainingPhase.SDF_RECURSIVE_ONLY:
            euler_weight = 1.0
            recursive_euler_weight = float(getattr(self.hyperparams, "sdf_recursive_loss_weight", 0.25))
            moment_weight = float(getattr(self.hyperparams, "sdf_recursive_moment_weight", 5e-4))
        elif self.add_FC1loss:
            moment_weight = float(getattr(self.hyperparams, "sdf_moment_weight", 1.0))
        else:
            moment_weight = float(getattr(self.hyperparams, "sdf_stage1_moment_weight", 5.0))

        # 对 log(E[M]) 增加显式锚，避免 SDF 均值在两阶段切换后漂移
        mean_anchor_loss = torch.tensor(0.0, device=self.device)
        mean_anchor_target = getattr(self.hyperparams, "sdf_log_mean_target", None)
        if mean_anchor_target is not None:
            log_mu_for_anchor = torch.log(M_use.mean().clamp_min(1e-8))
            mean_anchor_loss = (log_mu_for_anchor - float(mean_anchor_target)) ** 2
        if phase == SDFTrainingPhase.EPISODE0_BOOTSTRAP:
            mean_anchor_weight = float(getattr(self.hyperparams, "sdf_log_mean_anchor_weight_stage1", 1.0))
        elif phase == SDFTrainingPhase.SDF_TRUE_ONLY:
            mean_anchor_weight = float(getattr(self.hyperparams, "sdf_true_anchor_weight", 0.05))
        elif phase == SDFTrainingPhase.SDF_RECURSIVE_ONLY:
            mean_anchor_weight = float(getattr(self.hyperparams, "sdf_recursive_anchor_weight", 0.05))
        elif self.add_FC1loss:
            mean_anchor_weight = float(getattr(self.hyperparams, "sdf_log_mean_anchor_weight_stage2", 5.0))
        else:
            mean_anchor_weight = float(
                getattr(self.hyperparams, "sdf_log_mean_anchor_weight_stage1", 1.0)
            )

        # 可选：FC1 输出与真实 hatcf / lnkf 的重建误差
        recon_weight = getattr(self.hyperparams, "fc1_recon_weight", 0.0)
        forecast_recon_weight = float(
            getattr(self.hyperparams, "fc1_forecast_recon_weight", 0.0)
        )
        recursive_aux_enabled = bool(
            getattr(self.hyperparams, "fc1_recursive_aux_training_enabled", False)
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
        jacobian_penalty_interval = int(
            getattr(self.hyperparams, "fc1_jacobian_penalty_interval", 10)
        )
        sdf_fc1_step_count = int(getattr(self, "sdf_fc1_step_count", 0))
        compute_jacobian_penalty = (
            jacobian_penalty_weight > 0.0
            and jacobian_penalty_interval > 0
            and sdf_fc1_step_count % jacobian_penalty_interval == 0
        )
        jacobian_penalty_active = False
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
        recon_loss_dlnk = torch.tensor(0.0, device=self.device)
        recon_loss_forecast_hatc = torch.tensor(0.0, device=self.device)
        recon_loss_forecast_lnk = torch.tensor(0.0, device=self.device)
        recon_loss_forecast_dlnk = torch.tensor(0.0, device=self.device)
        delta_penalty = torch.tensor(0.0, device=self.device)
        delta_penalty_hatc = torch.tensor(0.0, device=self.device)
        delta_penalty_lnk = torch.tensor(0.0, device=self.device)
        jacobian_penalty = torch.tensor(0.0, device=self.device)
        jacobian_penalty_hatc = torch.tensor(0.0, device=self.device)
        jacobian_penalty_lnk = torch.tensor(0.0, device=self.device)
        if self.add_FC1loss:
            hatcf_pred = c_children_recon  # fixed Treatment B children
            lnkf_pred = k_children_recon
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
                dlnkf_pred = lnkf_pred - k_prev_input.unsqueeze(1)
                dlnkf_true = lnkf_true - k_prev_input.unsqueeze(1)
                recon_loss_dlnk = (dlnkf_pred - dlnkf_true).pow(2).mean()
                recon_loss = (
                    hatc_recon_inner_weight * recon_loss_hatc
                    + lnk_recon_inner_weight * recon_loss_lnk
                )

            # 额外加一条 forecast-state 闭环监督：
            # 用 (Hatcf_t, LnKF_t) 做当前态输入，直接约束下一期预测贴近真实值。
            # 这条项补上“递推口径”目标，而不仅是 true-state teacher-forcing 口径。
            if (
                recursive_aux_enabled
                and
                forecast_recon_weight > 0.0
                and parent.shape[1] >= 7
                and children_t.shape[-1] >= 9
            ):
                hatcf_prev_forecast = parent[:, 5:6].detach().clone()
                lnkf_prev_forecast = parent[:, 6:7].detach().clone()
                if compute_jacobian_penalty:
                    hatcf_prev_forecast.requires_grad_(True)
                    lnkf_prev_forecast.requires_grad_(True)
                _, _, _, c_children_forecast, k_children_forecast = model.forward_step(
                    x_prev=parent[:, 4:5],
                    x_curr=children_t[:, :, 4:5],
                    hatcf_prev=hatcf_prev_forecast,
                    lnkf_prev=lnkf_prev_forecast,
                    return_physical=True
                )
                recon_loss_forecast_hatc = (c_children_forecast - hatcf_true).pow(2).mean()
                recon_loss_forecast_lnk = (k_children_forecast - lnkf_true).pow(2).mean()
                dlnkf_forecast_pred = k_children_forecast - lnkf_prev_forecast.unsqueeze(1)
                dlnkf_forecast_true = lnkf_true - lnkf_prev_forecast.unsqueeze(1)
                recon_loss_forecast_dlnk = (dlnkf_forecast_pred - dlnkf_forecast_true).pow(2).mean()
                recon_loss_forecast = (
                    hatc_recon_inner_weight * recon_loss_forecast_hatc
                    + lnk_recon_inner_weight * recon_loss_forecast_lnk
                )
                d_hatcf_forecast = c_children_forecast - parent[:, 5:6].unsqueeze(1)
                d_lnkf_forecast = k_children_forecast - parent[:, 6:7].unsqueeze(1)
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

                if compute_jacobian_penalty:
                    jacobian_penalty_active = True
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
        if not torch.isfinite(recon_loss_dlnk):
            recon_loss_dlnk = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(recon_loss_forecast_hatc):
            recon_loss_forecast_hatc = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(recon_loss_forecast_lnk):
            recon_loss_forecast_lnk = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(recon_loss_forecast_dlnk):
            recon_loss_forecast_dlnk = torch.tensor(0.0, device=self.device)
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

        hj_warmup_factor = 1.0
        if phase not in {
            SDFTrainingPhase.EPISODE0_BOOTSTRAP,
            SDFTrainingPhase.SDF_TRUE_ONLY,
            SDFTrainingPhase.SDF_RECURSIVE_ONLY,
        }:
            hj_warmup_factor = self._compute_stage2_hj_warmup_factor()
        moment_weight_eff = float(moment_weight) * hj_warmup_factor
        mean_anchor_weight_eff = float(mean_anchor_weight) * hj_warmup_factor
        recon_weight_eff = float(recon_weight)
        forecast_recon_weight_eff = float(forecast_recon_weight)
        delta_penalty_weight_eff = float(delta_penalty_weight)
        jacobian_penalty_weight_eff = float(jacobian_penalty_weight)
        if not recursive_aux_enabled:
            forecast_recon_weight_eff = 0.0
            delta_penalty_weight_eff = 0.0
            jacobian_penalty_weight_eff = 0.0
        if phase in {
            SDFTrainingPhase.EPISODE0_BOOTSTRAP,
            SDFTrainingPhase.SDF_TRUE_ONLY,
            SDFTrainingPhase.SDF_RECURSIVE_ONLY,
        }:
            recon_weight_eff = 0.0
            forecast_recon_weight_eff = 0.0
            delta_penalty_weight_eff = 0.0
            jacobian_penalty_weight_eff = 0.0

        if bool(getattr(self, "_fc1_teacher_forcing_stage", False)) and self.add_FC1loss:
            teacher_weight = float(getattr(self.hyperparams, "fc1_teacher_forcing_weight", 1.0))
            total_sdf_loss = teacher_weight * (
                recon_loss
                + forecast_recon_weight_eff * recon_loss_forecast
                + delta_penalty_weight_eff * delta_penalty
                + jacobian_penalty_weight_eff * jacobian_penalty
            )
            moment_weight_eff = 0.0
            mean_anchor_weight_eff = 0.0
        else:
            total_sdf_loss = (
                euler_weight * true_state_main_loss
                + recursive_euler_weight * main_loss
                + moment_weight_eff * moment_loss
                + recon_weight_eff * recon_loss
                + forecast_recon_weight_eff * recon_loss_forecast
                + delta_penalty_weight_eff * delta_penalty
                + jacobian_penalty_weight_eff * jacobian_penalty
                + mean_anchor_weight_eff * mean_anchor_loss
            )

        # 诊断：每步记录 M 的矩和 FC1 跨期增量分布
        with torch.no_grad():
            mu = M_use.mean().clamp_min(1e-8)
            var = ((M_use - mu) ** 2).mean().clamp_min(1e-8)
            d_hatcf = (c_children_wealth - c_parent.unsqueeze(1)).reshape(-1)
            d_lnkf = (k_children_wealth - k_parent.unsqueeze(1)).reshape(-1)
            d_hatcf_recon = (c_children_recon - c_parent.unsqueeze(1)).reshape(-1)
            d_lnkf_recon = (k_children_recon - k_parent.unsqueeze(1)).reshape(-1)

            def _q(v: torch.Tensor, q: float) -> float:
                return float(torch.quantile(v, q).item()) if v.numel() > 0 else 0.0

            eps_diag = 1e-8
            w_parent_diag = w_parent.detach().squeeze(-1).reshape(-1)
            w_child_diag = w_children_wealth.detach().squeeze(-1).reshape(-1)
            c_child_diag = c_children_wealth.detach().squeeze(-1)
            surplus_parent_raw_diag = (
                w_parent.detach().squeeze(-1) - torch.exp(c_parent.detach().squeeze(-1))
            ).reshape(-1)
            surplus_parent_diag = residual_pack["surplus_parent"].detach().reshape(-1)
            surplus_child_diag = (
                w_children_wealth.detach().squeeze(-1) - torch.exp(c_child_diag)
            ).clamp_min(eps_diag).reshape(-1)
            wealth_ratio_diag = residual_pack["wealth_ratio"].detach().reshape(-1)
            log_wealth_ratio_diag = torch.log(wealth_ratio_diag.clamp_min(eps_diag))

            value_scale_diag = {
                'sdf_w_parent_p01': _q(w_parent_diag, 0.01),
                'sdf_w_parent_p50': _q(w_parent_diag, 0.50),
                'sdf_w_parent_p99': _q(w_parent_diag, 0.99),
                'sdf_w_child_p01': _q(w_child_diag, 0.01),
                'sdf_w_child_p50': _q(w_child_diag, 0.50),
                'sdf_w_child_p99': _q(w_child_diag, 0.99),
                'sdf_surplus_parent_p01': _q(surplus_parent_diag, 0.01),
                'sdf_surplus_parent_p50': _q(surplus_parent_diag, 0.50),
                'sdf_surplus_parent_p99': _q(surplus_parent_diag, 0.99),
                'sdf_surplus_child_p01': _q(surplus_child_diag, 0.01),
                'sdf_surplus_child_p50': _q(surplus_child_diag, 0.50),
                'sdf_surplus_child_p99': _q(surplus_child_diag, 0.99),
                'sdf_wealth_ratio_p01': _q(wealth_ratio_diag, 0.01),
                'sdf_wealth_ratio_p50': _q(wealth_ratio_diag, 0.50),
                'sdf_wealth_ratio_p99': _q(wealth_ratio_diag, 0.99),
                'sdf_log_wealth_ratio_p01': _q(log_wealth_ratio_diag, 0.01),
                'sdf_log_wealth_ratio_p50': _q(log_wealth_ratio_diag, 0.50),
                'sdf_log_wealth_ratio_p99': _q(log_wealth_ratio_diag, 0.99),
                'sdf_surplus_parent_floor_share': float(
                    (surplus_parent_raw_diag <= eps_diag).to(torch.float32).mean().item()
                ),
            }

            wealth_diag = {
                f"sdf_{key if key != 'signed_aio' else 'signed_aio_main'}": float(
                    value.detach().item()
                )
                for key, value in wealth_main_details.items()
                if torch.is_tensor(value) and value.numel() == 1
            }
            raw_wealth_diag = {
                f"sdf_raw_{key if key != 'signed_aio' else 'signed_aio_main'}": float(
                    value.detach().item()
                )
                for key, value in raw_wealth_details.items()
                if torch.is_tensor(value) and value.numel() == 1
            }
            normalized_wealth_diag = {
                f"sdf_normalized_{key if key != 'signed_aio' else 'signed_aio_main'}": float(
                    value.detach().item()
                )
                for key, value in normalized_wealth_details.items()
                if torch.is_tensor(value) and value.numel() == 1
            }
            self._latest_sdf_terms = {
                'sdf_training_phase': phase.value,
                'sdf_main_loss': float(main_loss.detach().item()),
                'sdf_true_state_main_loss': float(true_state_main_loss.detach().item()),
                'sdf_total_loss': float(total_sdf_loss.detach().item()),
                'sdf_main_weight_effective': float(euler_weight),
                'sdf_recursive_main_weight_effective': float(recursive_euler_weight),
                'sdf_moment_weight_effective': float(moment_weight_eff),
                'sdf_anchor_weight_effective': float(mean_anchor_weight_eff),
                'fc1_recon_weight_effective': float(recon_weight_eff),
                'fc1_forecast_weight_effective': float(forecast_recon_weight_eff),
                'fc1_rollout_weight_effective': 0.0,
                'fc1_delta_weight_effective': float(delta_penalty_weight_eff),
                'fc1_jacobian_weight_effective': float(jacobian_penalty_weight_eff if jacobian_penalty_active else 0.0),
                'sdf_wealth_loss_mode_signed_aio': float(
                    1.0 if getattr(loss_fn, "wealth_loss_mode", "legacy_abs_log1p") == "signed_aio" else 0.0
                ),
                'sdf_wealth_residual_mode_normalized_ratio': float(
                    1.0 if residual_mode == "normalized_ratio" else 0.0
                ),
                'sdf_normalized_logr_clip': float(normalized_logr_clip),
                'sdf_normalized_logr_clip_share': float(residual_pack["log_R_clip_share"].detach().item()),
                'sdf_normalized_logr_abs_mean': float(residual_pack["log_R"].detach().abs().mean().item()),
                'sdf_wealth_ratio_mean': float(residual_pack["wealth_ratio"].detach().mean().item()),
                'sdf_surplus_parent_mean': float(residual_pack["surplus_parent"].detach().mean().item()),
                'sdf_moment_loss': float(moment_loss.detach().item()),
                'sdf_recon_loss': float(recon_loss.detach().item()),
                'sdf_fc1_true_recon': float(recon_loss.detach().item()),
                'sdf_recon_loss_hatc': float(recon_loss_hatc.detach().item()),
                'sdf_recon_loss_lnk': float(recon_loss_lnk.detach().item()),
                'sdf_recon_loss_dlnk': float(recon_loss_dlnk.detach().item()),
                'sdf_recon_loss_forecast': float(recon_loss_forecast.detach().item()),
                'sdf_fc1_forecast_recon': float(recon_loss_forecast.detach().item()),
                'sdf_recon_loss_forecast_hatc': float(recon_loss_forecast_hatc.detach().item()),
                'sdf_recon_loss_forecast_lnk': float(recon_loss_forecast_lnk.detach().item()),
                'sdf_recon_loss_forecast_dlnk': float(recon_loss_forecast_dlnk.detach().item()),
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
                'sdf_forecast_recon_weight': float(forecast_recon_weight_eff),
                'sdf_hatc_recon_inner_weight': float(hatc_recon_inner_weight),
                'sdf_lnk_recon_inner_weight': float(lnk_recon_inner_weight),
                'sdf_delta_penalty_weight': float(delta_penalty_weight_eff),
                'sdf_delta_hatc_abs_max': float(delta_hatc_abs_max),
                'sdf_delta_lnk_abs_max': float(delta_lnk_abs_max),
                'sdf_jacobian_penalty_weight': float(jacobian_penalty_weight_eff),
                'sdf_jacobian_penalty_interval': float(jacobian_penalty_interval),
                'sdf_jacobian_penalty_active': float(1.0 if jacobian_penalty_active else 0.0),
                'sdf_fc1_step_count': float(sdf_fc1_step_count),
                'sdf_hj_warmup_factor': float(hj_warmup_factor),
                'sdf_teacher_forcing_stage': float(1.0 if self._fc1_teacher_forcing_stage else 0.0),
                'sdf_use_true_prev_macro': float(1.0 if use_true_prev_macro else 0.0),
            }
            self._latest_sdf_terms.update(wealth_diag)
            self._latest_sdf_terms.update(raw_wealth_diag)
            self._latest_sdf_terms.update(normalized_wealth_diag)
            self._latest_sdf_terms.update(value_scale_diag)
            self._latest_sdf_terms.update(fresh_pair_diag)
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
                'sdf_recon_dhatcf_mean': float(d_hatcf_recon.mean().item()),
                'sdf_recon_dlnkf_mean': float(d_lnkf_recon.mean().item()),
            }

        return total_sdf_loss
    
    def _huber_element(self, pred: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
        diff = (pred - target).abs()
        beta = float(beta)
        if beta <= 0:
            return diff
        return torch.where(diff < beta, 0.5 * diff.pow(2) / beta, diff - 0.5 * beta)

    def _grid_diag_terms(
        self,
        prefix: str,
        grid: Dict[str, torch.Tensor],
        bp_pred: torch.Tensor,
        value_pred_online: torch.Tensor,
        value_loss_elem: torch.Tensor,
        policy_loss_elem: torch.Tensor,
        policy_loss: torch.Tensor,
        policy_weight: float,
        value_loss: torch.Tensor,
        penalty_z: torch.Tensor,
        extra_terms: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        bp_err = (bp_pred.detach() - grid["bp_star"]).abs()
        terms = {
            f'{prefix}_main': float(value_loss.item()),
            f'{prefix}_foc': 0.0,
            f'{prefix}_penalty_z': float(penalty_z.item()),
            f'{prefix}_penalty_z_foc': 0.0,
            f'{prefix}_kkt': 0.0,
            f'{prefix}_bp_terms_base': float(policy_loss.item()),
            f'{prefix}_bp_terms_after_eta': float(policy_loss.item()),
            f'{prefix}_bp_terms': float((policy_weight * policy_loss).item()),
            f'{prefix}_eta_active_boost': 1.0,
            f'{prefix}_bp_adapt_scale': 1.0,
            f'{prefix}_kkt_inner': 0.0,
            f'{prefix}_kkt_low': 0.0,
            f'{prefix}_kkt_high': 0.0,
            f'{prefix}_kkt_foc_abs_mean': 0.0,
            f'{prefix}_kkt_active_ratio': 0.0,
            f'{prefix}_kkt_high_weight': 0.0,
            f'{prefix}_foc_active_ratio': 0.0,
            f'{prefix}_foc_signed_moment': 0.0,
            f'{prefix}_foc_cond_abs_mean': 0.0,
            f'{prefix}_foc_active_n': 0.0,
            f'{prefix}_bp_foc_use_phat': 0.0,
            f'{prefix}_bellman_only': 0.0,
            f'{prefix}_fixed_sdf': float(1.0 if self._pv_use_fixed_sdf() else 0.0),
            f'{prefix}_fixed_policy': float(1.0 if self._pv_use_fixed_policy() else 0.0),
            f'{prefix}_target_grid': 1.0,
            f'{prefix}_grid_value_loss': float(value_loss.item()),
            f'{prefix}_grid_value_loss_elem_mean': float(value_loss_elem.detach().mean().item()),
            f'{prefix}_grid_policy_loss': float(policy_loss.item()),
            f'{prefix}_grid_policy_loss_elem_mean': float(policy_loss_elem.detach().mean().item()),
            f'{prefix}_grid_policy_weight': float(policy_weight),
            f'{prefix}_grid_bp_mae': float(bp_err.mean().item()),
            f'{prefix}_grid_bp_err_p90': self._safe_quantile(bp_err, 0.90),
            f'{prefix}_grid_regret_mean': float(grid["regret"].mean().item()),
            f'{prefix}_grid_regret_p90': self._safe_quantile(grid["regret"], 0.90),
            f'{prefix}_grid_top2_margin_mean': float(grid["top2_margin"].mean().item()),
            f'{prefix}_grid_fine_top2_margin_mean': float(grid["fine_top2_margin"].mean().item()),
            f'{prefix}_grid_top2_margin_p10': self._safe_quantile(grid["top2_margin"], 0.10),
            f'{prefix}_grid_confidence_mean': float(grid["confidence"].mean().item()),
            f'{prefix}_grid_boundary_low_share': float(grid["boundary_low"].mean().item()),
            f'{prefix}_grid_boundary_high_share': float(grid["boundary_high"].mean().item()),
            f'{prefix}_grid_bp_star_mean': float(grid["bp_star"].mean().item()),
            f'{prefix}_grid_bp_star_p50': self._safe_quantile(grid["bp_star"], 0.50),
            f'{prefix}_grid_bp_star_p90': self._safe_quantile(grid["bp_star"], 0.90),
            f'{prefix}_grid_value_star_mean': float(grid["value_star"].mean().item()),
            f'{prefix}_grid_value_pred_mean': float(grid["value_pred"].mean().item()),
            f'{prefix}_grid_value_online_mean': float(value_pred_online.detach().mean().item()),
            f'{prefix}_grid_default_at_star_mean': float(grid["default_at_star"].mean().item()),
            f'{prefix}_grid_p_child_at_star_mean': float(grid["p_child_at_star"].mean().item()),
            f'{prefix}_grid_q_issue_at_star_mean': float(grid["q_issue_at_star"].mean().item()),
            f'{prefix}_grid_argmax_index_mean': float(grid["argmax_index"].to(torch.float32).mean().item()),
            f'{prefix}_grid_value_low_bp_mean': float(grid["coarse_value_grid"][:, 0:1].mean().item()),
            f'{prefix}_grid_value_high_bp_mean': float(grid["coarse_value_grid"][:, -1:].mean().item()),
            f'{prefix}_grid_default_low_bp_mean': float(grid["coarse_default_grid_mean"][:, 0:1].mean().item()),
            f'{prefix}_grid_default_high_bp_mean': float(grid["coarse_default_grid_mean"][:, -1:].mean().item()),
            f'{prefix}_grid_p_child_low_bp_mean': float(grid["coarse_p_child_grid_mean"][:, 0:1].mean().item()),
            f'{prefix}_grid_p_child_high_bp_mean': float(grid["coarse_p_child_grid_mean"][:, -1:].mean().item()),
            f'{prefix}_grid_q_issue_low_bp_mean': float(grid["coarse_q_issue_grid"][:, 0:1].mean().item()),
            f'{prefix}_grid_q_issue_high_bp_mean': float(grid["coarse_q_issue_grid"][:, -1:].mean().item()),
            f'{prefix}_grid_local_value_left_mean': float(grid["local_value_left"].mean().item()),
            f'{prefix}_grid_local_value_right_mean': float(grid["local_value_right"].mean().item()),
        }
        if extra_terms:
            terms.update(extra_terms)
        terms.update(self._tensor_tail_diagnostics(f'{prefix}_grid_value_star', grid["value_star"]))
        terms.update(self._tensor_tail_diagnostics(f'{prefix}_grid_regret', grid["regret"]))
        return terms

    def _grid_policy_only_diag_terms(
        self,
        prefix: str,
        grid: Dict[str, torch.Tensor],
        bp_pred: torch.Tensor,
        policy_loss_elem: torch.Tensor,
        policy_loss: torch.Tensor,
        policy_weight: float,
    ) -> Dict[str, float]:
        bp_err = (bp_pred.detach() - grid["bp_star"]).abs()
        return {
            f'{prefix}_grid_policy_loss': float(policy_loss.item()),
            f'{prefix}_grid_policy_loss_elem_mean': float(policy_loss_elem.detach().mean().item()),
            f'{prefix}_grid_policy_weight': float(policy_weight),
            f'{prefix}_grid_bp_mae': float(bp_err.mean().item()),
            f'{prefix}_grid_bp_err_p90': self._safe_quantile(bp_err, 0.90),
            f'{prefix}_grid_regret_mean': float(grid["regret"].mean().item()),
            f'{prefix}_grid_regret_p90': self._safe_quantile(grid["regret"], 0.90),
            f'{prefix}_grid_top2_margin_mean': float(grid["top2_margin"].mean().item()),
            f'{prefix}_grid_fine_top2_margin_mean': float(grid["fine_top2_margin"].mean().item()),
            f'{prefix}_grid_top2_margin_p10': self._safe_quantile(grid["top2_margin"], 0.10),
            f'{prefix}_grid_confidence_mean': float(grid["confidence"].mean().item()),
            f'{prefix}_grid_boundary_low_share': float(grid["boundary_low"].mean().item()),
            f'{prefix}_grid_boundary_high_share': float(grid["boundary_high"].mean().item()),
            f'{prefix}_grid_bp_star_mean': float(grid["bp_star"].mean().item()),
            f'{prefix}_grid_bp_star_p50': self._safe_quantile(grid["bp_star"], 0.50),
            f'{prefix}_grid_bp_star_p90': self._safe_quantile(grid["bp_star"], 0.90),
            f'{prefix}_grid_value_star_mean': float(grid["value_star"].mean().item()),
            f'{prefix}_grid_value_pred_mean': float(grid["value_pred"].mean().item()),
            f'{prefix}_grid_default_at_star_mean': float(grid["default_at_star"].mean().item()),
            f'{prefix}_grid_p_child_at_star_mean': float(grid["p_child_at_star"].mean().item()),
            f'{prefix}_grid_q_issue_at_star_mean': float(grid["q_issue_at_star"].mean().item()),
            f'{prefix}_grid_argmax_index_mean': float(grid["argmax_index"].to(torch.float32).mean().item()),
            f'{prefix}_grid_value_low_bp_mean': float(grid["coarse_value_grid"][:, 0:1].mean().item()),
            f'{prefix}_grid_value_high_bp_mean': float(grid["coarse_value_grid"][:, -1:].mean().item()),
            f'{prefix}_grid_default_low_bp_mean': float(grid["coarse_default_grid_mean"][:, 0:1].mean().item()),
            f'{prefix}_grid_default_high_bp_mean': float(grid["coarse_default_grid_mean"][:, -1:].mean().item()),
            f'{prefix}_grid_p_child_low_bp_mean': float(grid["coarse_p_child_grid_mean"][:, 0:1].mean().item()),
            f'{prefix}_grid_p_child_high_bp_mean': float(grid["coarse_p_child_grid_mean"][:, -1:].mean().item()),
            f'{prefix}_grid_q_issue_low_bp_mean': float(grid["coarse_q_issue_grid"][:, 0:1].mean().item()),
            f'{prefix}_grid_q_issue_high_bp_mean': float(grid["coarse_q_issue_grid"][:, -1:].mean().item()),
            f'{prefix}_grid_local_value_left_mean': float(grid["local_value_left"].mean().item()),
            f'{prefix}_grid_local_value_right_mean': float(grid["local_value_right"].mean().item()),
        }

    def _target_investment_conditional(
        self,
        target_model: nn.Module,
        parent_state: torch.Tensor,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            value_fn = getattr(target_model, "_value_outputs", None)
            derived = getattr(target_model, "derived", None)
            if callable(value_fn) and derived is not None:
                v0_t, vi_t = value_fn(parent_state.detach())
                return derived.investment_conditional(v0_t, vi_t).detach()
            out = target_model(parent_state.detach())
            if isinstance(out, dict) and 'bar_i_cond' in out:
                return out['bar_i_cond'].detach()
            if hasattr(out, 'bar_i_cond'):
                return getattr(out, 'bar_i_cond').detach()
            return fallback.detach()

    def _target_survival_probability(
        self,
        target_model: nn.Module,
        parent_state: torch.Tensor,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            equity_fn = getattr(target_model, "forward_equity", None)
            if callable(equity_fn):
                out = equity_fn(parent_state.detach())
                if isinstance(out, dict) and 'survival_prob' in out:
                    return out['survival_prob'].detach()
            out = target_model(parent_state.detach())
            if isinstance(out, dict) and 'survival_prob' in out:
                return out['survival_prob'].detach()
            if hasattr(out, 'survival_prob'):
                return getattr(out, 'survival_prob').detach()
            return fallback.detach()

    def _mixed_policy_conditional_bp(
        self,
        output_t,
        bp0_t: torch.Tensor,
        bpI_t: torch.Tensor,
        parent_b: torch.Tensor,
        fallback_bar_i: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(output_t, dict):
            bar_i_cond = output_t.get('bar_i_cond', fallback_bar_i)
        else:
            bar_i_cond = getattr(output_t, 'bar_i_cond', fallback_bar_i)
        bp0_for_mix = self._apply_policy_ablation(bp0_t, parent_b)
        bpI_for_mix = self._apply_policy_ablation(bpI_t, parent_b)
        bar_i_policy = bar_i_cond.detach()
        return bar_i_policy * bpI_for_mix + (1.0 - bar_i_policy) * bp0_for_mix

    def _compute_target_grid_pv_loss(
        self,
        *,
        branch: str,
        parent: torch.Tensor,
        children: List[torch.Tensor],
        parent_state: torch.Tensor,
        target_model: nn.Module,
        value_pred: torch.Tensor,
        bp_pred: torch.Tensor,
        m_list: List[torch.Tensor],
        raw_m_list: List[torch.Tensor],
        m_lo: float,
        m_hi: float,
        loss_fn,
        mix_weight: Optional[torch.Tensor] = None,
        bp_mix_pred: Optional[torch.Tensor] = None,
        mix_policy_sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        branch = branch.lower()
        prefix = 'p0' if branch == 'p0' else 'pi'
        teacher = BPGridTeacher.from_hyperparams(
            target_model,
            self.loss_fns['p0'],
            self.loss_fns['pi'],
            self.hyperparams,
        )
        grid = teacher.compute(
            parent_state=parent_state,
            children=children,
            m_list=m_list,
            branch=branch,
            bp_pred=bp_pred,
        )

        value_delta = float(getattr(self.hyperparams, "bp_grid_value_huber_delta", 1.0))
        policy_delta = float(getattr(self.hyperparams, "bp_grid_policy_huber_delta", 0.05))
        policy_weight = float(getattr(self.hyperparams, "bp_grid_policy_weight", 1.0))

        value_loss_elem = self._huber_element(value_pred, grid["value_star"], value_delta)
        value_loss = value_loss_elem.mean()
        penalty_z = compute_z_penalty(
            value_loss_elem,
            parent_state[:, 1:2],
            loss_fn.alpha_z,
            loss_fn.beta_z,
            loss_fn.z0,
        )
        policy_loss_elem = self._huber_element(bp_pred, grid["bp_star"], policy_delta)
        policy_loss = (grid["confidence"] * policy_loss_elem).mean()
        total_loss = value_loss + penalty_z + policy_weight * policy_loss

        extra_terms: Dict[str, float] = {}
        if branch == 'pi':
            penalty_b = loss_fn.b_penalty_weight * loss_fn.compute_b_penalty(value_pred, parent_state[:, 0:1]).mean()
            total_loss = total_loss + penalty_b
            extra_terms['pi_penalty_b'] = float(penalty_b.item())

        mix_terms: Dict[str, float] = {}
        if branch == 'pi' and mix_weight is not None and bp_mix_pred is not None:
            mix_grid = teacher.compute(
                parent_state=parent_state,
                children=children,
                m_list=m_list,
                branch='mix',
                bp_pred=bp_mix_pred,
                mix_weight=mix_weight,
            )
            mix_policy_weight = float(getattr(self.hyperparams, "bp_grid_mix_policy_weight", 1.0))
            mix_policy_loss_elem = self._huber_element(bp_mix_pred, mix_grid["bp_star"], policy_delta)
            mix_sample_weight = mix_grid["confidence"]
            if mix_policy_sample_weight is not None:
                mix_sample_weight = mix_sample_weight * mix_policy_sample_weight.detach().clamp(0.0, 1.0)
            mix_policy_loss = (mix_sample_weight * mix_policy_loss_elem).mean()
            total_loss = total_loss + mix_policy_weight * mix_policy_loss
            mix_terms = self._grid_policy_only_diag_terms(
                'mix',
                mix_grid,
                bp_mix_pred,
                mix_policy_loss_elem,
                mix_policy_loss,
                mix_policy_weight,
            )
            mix_terms['mix_grid_target_survival_weight_mean'] = float(
                mix_policy_sample_weight.detach().mean().item()
                if mix_policy_sample_weight is not None
                else 1.0
            )

        with torch.no_grad():
            raw_m = torch.cat([m.reshape(-1) for m in raw_m_list], dim=0)
            use_m = torch.cat([m.reshape(-1) for m in m_list], dim=0)
            terms = self._grid_diag_terms(
                prefix,
                grid,
                bp_pred,
                value_pred,
                value_loss_elem,
                policy_loss_elem,
                policy_loss,
                policy_weight,
                value_loss,
                penalty_z,
                extra_terms=extra_terms,
            )
            terms.update(self._m_diagnostics(prefix, raw_m, use_m, m_lo, m_hi))
            if branch == 'p0':
                self._latest_p0_terms = terms
            else:
                terms.update(mix_terms)
                self._latest_pi_terms = terms
        return total_loss

    def _compute_p0_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算 P0 损失（支持任意 N 分支）
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
        
        m_lo = float(getattr(self.hyperparams, "pv_m_clamp_min", 0.7))
        m_hi = float(getattr(self.hyperparams, "pv_m_clamp_max", 1.3))
        raw_M_list, M_list = self._build_policy_m_lists(parent, children, m_lo, m_hi)
        
        # 前向传播
        parent_state = strip_extra(parent)
        target_model = self._target_policy_value()
        output_t = model(parent_state)

        def _get_out(out, name: str, idx: int) -> torch.Tensor:
            if isinstance(out, dict):
                return out[name]
            if hasattr(out, name):
                return getattr(out, name)
            return out[:, idx:idx + 1]

        bp0_t = _get_out(output_t, 'bp0', 1)
        bpI_t = _get_out(output_t, 'bpI', 2)
        bar_i_t = _get_out(output_t, 'bar_i', 4)
        bp_t = _get_out(output_t, 'bp', -1)
        if bp_t.shape != bp0_t.shape:
            bp_t = bar_i_t * bpI_t + (1 - bar_i_t) * bp0_t
        # P0 分支使用不投资场景的杠杆候选 bp0
        b_parent = parent_state[:, 0:1]
        bp_for_p0 = self._apply_policy_ablation(bp0_t, b_parent)

        if self._pv_use_target_grid_bp() and not self._policy_value_bellman_only():
            return self._compute_target_grid_pv_loss(
                branch='p0',
                parent=parent,
                children=children,
                parent_state=parent_state,
                target_model=target_model,
                value_pred=_get_out(output_t, 'P0', 3),
                bp_pred=bp_for_p0,
                m_list=M_list,
                raw_m_list=raw_M_list,
                m_lo=m_lo,
                m_hi=m_hi,
                loss_fn=loss_fn,
            )

        output_children = []
        output_children_target = []
        eta_children = []

        for child in children:
            child_state_raw = strip_extra(child)
            eta_child = child[:, 2:3]
            child_state = child_state_raw.clone()
            child_state[:, 0:1] = eta_child * bp_for_p0 + (1 - eta_child) * b_parent
            output_children.append(model(child_state))
            with torch.no_grad():
                output_children_target.append(target_model(child_state.detach()))
            eta_children.append(eta_child)
            
        childp0_state = parent_state.clone()
        childp0_state[:, 0:1] = bp_for_p0
        outputp0_children = model(childp0_state)
        with torch.no_grad():
            outputp0_children_target = target_model(childp0_state.detach())
        
        # 提取 P0 和所需变量
        P0 = _get_out(output_t, 'P0', 3)
        P_children = [_get_out(out, 'P', 7).detach() for out in output_children_target]
        # FOC/KKT 梯度通道可选用 Phat，避免 P=max(Phat,0) 在违约区产生大面积零梯度
        use_phat_for_bp_foc = bool(getattr(self.hyperparams, "bp_foc_use_phat_children", True))
        P_children_for_foc = [
            _get_out(out, 'Phat', 8) if use_phat_for_bp_foc else _get_out(out, 'P', 7)
            for out in output_children
        ]
        bar_z_children = [_get_out(out, 'bar_z', 6).detach() for out in output_children_target]
        bar_z_children_for_foc = [_get_out(out, 'bar_z', 6) for out in output_children]
        
        # Q 值
        Q = _get_out(output_t, 'Q', 0)
        Qp = _get_out(outputp0_children_target, 'Q', 0).detach()
        Qp_for_foc = _get_out(outputp0_children, 'Q', 0)

        
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
        CF0p_for_foc = [
            loss_fn.compute_cashflow_p0(
                parent_state[:, 4:5],  # x
                parent_state[:, 1:2],  # z
                parent_state[:, 0:1],  # b
                Q, Qp_for_foc,
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
        bellman_only = self._policy_value_bellman_only()
        if bellman_only:
            zero = torch.tensor(0.0, device=self.device)
            loss_foc = zero
            penalty_z_foc = zero
            kkt_penalty = zero
            bp_terms_base = zero
            bp_terms_after_eta = zero
            bp_terms = zero
            eta_active_boost = 1.0
            bp_adapt_scale = 1.0
            foc_diag = {'foc_active_ratio': 0.0, 'foc_signed_moment': 0.0, 'foc_cond_abs_mean': 0.0, 'foc_active_n': 0.0}
            kkt_diag = {'kkt_inner': 0.0, 'kkt_low': 0.0, 'kkt_high': 0.0, 'kkt_foc_abs_mean': 0.0, 'kkt_active_ratio': 0.0, 'kkt_high_weight': 0.0}
            total_loss = main_loss
        else:
            foc_residuals = loss_fn.compute_foc_residual_from_bp(
                CF0p=CF0p_for_foc,
                M_list=M_list,
                P_children=P_children_for_foc,
                bar_z_children=bar_z_children_for_foc,
                bp=bp_for_p0,
                eta=eta_children
            )
            loss_foc, penalty_z_foc, foc_diag = self._compute_conditional_signed_foc_terms(
                foc_residuals=foc_residuals,
                eta_children=eta_children,
                z_parent=parent_state[:, 1:2],
                alpha_z=loss_fn.alpha_z,
                beta_z=loss_fn.beta_z,
                z0=loss_fn.z0
            )
            kkt_penalty_base, kkt_diag = self._compute_bp_kkt_penalty(
                bp_for_p0, foc_residuals, eta_children=eta_children
            )
            p0_kkt_w = float(getattr(self.hyperparams, "p0_kkt_weight", 1.0))
            kkt_penalty = p0_kkt_w * kkt_penalty_base
            eta_active_boost = self._compute_eta_active_boost(foc_diag.get('foc_active_ratio', 0.0))
            bp_terms_base = loss_foc + penalty_z_foc + kkt_penalty
            bp_terms_after_eta = eta_active_boost * bp_terms_base
            bp_adapt_scale = self._compute_bp_adaptive_scale(main_loss, bp_terms_after_eta)
            bp_terms = bp_adapt_scale * bp_terms_after_eta
            total_loss = main_loss + penalty_z + bp_terms
        with torch.no_grad():
            raw_m = torch.cat([m.reshape(-1) for m in raw_M_list], dim=0)
            use_m = torch.cat([m.reshape(-1) for m in M_list], dim=0)
            target_y = torch.cat(
                [(cf + m * p).reshape(-1) for cf, m, p in zip(CF0p, M_list, P_children)],
                dim=0,
            )
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
                'p0_bp_foc_use_phat': float(1.0 if use_phat_for_bp_foc else 0.0),
                'p0_bellman_only': float(1.0 if bellman_only else 0.0),
                'p0_fixed_sdf': float(1.0 if self._pv_use_fixed_sdf() else 0.0),
                'p0_fixed_policy': float(1.0 if self._pv_use_fixed_policy() else 0.0),
            }
            self._latest_p0_terms.update(self._m_diagnostics('p0', raw_m, use_m, m_lo, m_hi))
            self._latest_p0_terms.update(self._tensor_tail_diagnostics('p0_target_y', target_y))
            self._latest_p0_terms.update(getattr(loss_fn, 'latest_foc_diag', {}))
        return total_loss
    
    def _compute_pi_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算 PI 损失（支持任意 N 分支）
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
        
        m_lo = float(getattr(self.hyperparams, "pv_m_clamp_min", 0.7))
        m_hi = float(getattr(self.hyperparams, "pv_m_clamp_max", 1.3))
        raw_M_list, M_list = self._build_policy_m_lists(parent, children, m_lo, m_hi)
        
        # 前向传播
        parent_state = strip_extra(parent)
        target_model = self._target_policy_value()
        output_t = model(parent_state)

        def _get_out(out, name: str, idx: int) -> torch.Tensor:
            if isinstance(out, dict):
                return out[name]
            if hasattr(out, name):
                return getattr(out, name)
            return out[:, idx:idx + 1]


        bp0_t = _get_out(output_t, 'bp0', 1)
        bpI_t = _get_out(output_t, 'bpI', 2)
        bar_i_t = _get_out(output_t, 'bar_i', 4)
        bp_t = _get_out(output_t, 'bp', -1)
        if bp_t.shape != bp0_t.shape:
            bp_t = bar_i_t * bpI_t + (1 - bar_i_t) * bp0_t
        # PI 分支使用投资场景的杠杆候选 bpI
        b_parent = parent_state[:, 0:1]
        bp_for_pi = self._apply_policy_ablation(bpI_t, b_parent)

        if self._pv_use_target_grid_bp() and not self._policy_value_bellman_only():
            if isinstance(output_t, dict):
                bar_i_cond_online = output_t.get('bar_i_cond', bar_i_t)
                survival_online = output_t.get('survival_prob', torch.ones_like(bar_i_t))
            else:
                bar_i_cond_online = getattr(output_t, 'bar_i_cond', bar_i_t)
                survival_online = getattr(output_t, 'survival_prob', torch.ones_like(bar_i_t))
            mix_weight_target = self._target_investment_conditional(
                target_model,
                parent_state,
                fallback=bar_i_cond_online,
            )
            mix_survival_target = self._target_survival_probability(
                target_model,
                parent_state,
                fallback=survival_online,
            )
            bp_mix_cond_pred = self._mixed_policy_conditional_bp(
                output_t,
                bp0_t,
                bpI_t,
                b_parent,
                fallback_bar_i=bar_i_cond_online,
            )
            return self._compute_target_grid_pv_loss(
                branch='pi',
                parent=parent,
                children=children,
                parent_state=parent_state,
                target_model=target_model,
                value_pred=_get_out(output_t, 'PI', 4),
                bp_pred=bp_for_pi,
                m_list=M_list,
                raw_m_list=raw_M_list,
                m_lo=m_lo,
                m_hi=m_hi,
                loss_fn=loss_fn,
                mix_weight=mix_weight_target,
                bp_mix_pred=bp_mix_cond_pred,
                mix_policy_sample_weight=mix_survival_target,
            )

        output_children = []
        output_children_target = []
        eta_children = []

        for child in children:
            child_state_raw = strip_extra(child)
            eta_child = child[:, 2:3]
            child_state = child_state_raw.clone()
            child_state[:, 0:1] = eta_child * bp_for_pi + (1 - eta_child) * b_parent
            output_children.append(model(child_state))
            with torch.no_grad():
                output_children_target.append(target_model(child_state.detach()))
            eta_children.append(eta_child)
            
        childpI_state = parent_state.clone()
        childpI_state[:, 0:1] = bp_for_pi
        outputpI_children = model(childpI_state)
        with torch.no_grad():
            outputpI_children_target = target_model(childpI_state.detach())
        
        # 提取 PI 和所需变量
        Q = _get_out(output_t, 'Q', 0)
        PI = _get_out(output_t, 'PI', 4)
        P_children = [_get_out(out, 'P', 7).detach() for out in output_children_target]
        # FOC/KKT 梯度通道可选用 Phat，避免 P=max(Phat,0) 在违约区产生大面积零梯度
        use_phat_for_bp_foc = bool(getattr(self.hyperparams, "bp_foc_use_phat_children", True))
        P_children_for_foc = [
            _get_out(out, 'Phat', 8) if use_phat_for_bp_foc else _get_out(out, 'P', 7)
            for out in output_children
        ]
        bar_z_children = [_get_out(out, 'bar_z', 6).detach() for out in output_children_target]
        bar_z_children_for_foc = [_get_out(out, 'bar_z', 6) for out in output_children]
        QpI = _get_out(outputpI_children_target, 'Q', 0).detach()
        QpI_for_foc = _get_out(outputpI_children, 'Q', 0)
        
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
        CFip_for_foc = [
            loss_fn.compute_cashflow_pi(
                parent_state[:, 4:5],  # x
                parent_state[:, 1:2],  # z
                parent_state[:, 0:1],  # b
                parent_state[:, 3:4],  # i
                Q, QpI_for_foc,
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
        bellman_only = self._policy_value_bellman_only()
        if bellman_only:
            zero = torch.tensor(0.0, device=self.device)
            loss_foc = zero
            penalty_z_foc = zero
            kkt_penalty = zero
            bp_terms_base = zero
            bp_terms_after_eta = zero
            bp_terms = zero
            eta_active_boost = 1.0
            bp_adapt_scale = 1.0
            foc_diag = {'foc_active_ratio': 0.0, 'foc_signed_moment': 0.0, 'foc_cond_abs_mean': 0.0, 'foc_active_n': 0.0}
            kkt_diag = {'kkt_inner': 0.0, 'kkt_low': 0.0, 'kkt_high': 0.0, 'kkt_foc_abs_mean': 0.0, 'kkt_active_ratio': 0.0, 'kkt_high_weight': 0.0}
            total_loss = main_loss
        else:
            foc_residuals = loss_fn.compute_foc_residual_from_bp(
                CFip=CFip_for_foc,
                M_list=M_list,
                P_children=P_children_for_foc,
                bar_z_children=bar_z_children_for_foc,
                bp=bp_for_pi,
                eta=eta_children
            )
            loss_foc, penalty_z_foc, foc_diag = self._compute_conditional_signed_foc_terms(
                foc_residuals=foc_residuals,
                eta_children=eta_children,
                z_parent=parent_state[:, 1:2],
                alpha_z=loss_fn.alpha_z,
                beta_z=loss_fn.beta_z,
                z0=loss_fn.z0
            )
            kkt_penalty_base, kkt_diag = self._compute_bp_kkt_penalty(
                bp_for_pi, foc_residuals, eta_children=eta_children
            )
            pi_kkt_w = float(getattr(self.hyperparams, "pi_kkt_weight", 1.0))
            kkt_penalty = pi_kkt_w * kkt_penalty_base
            eta_active_boost = self._compute_eta_active_boost(foc_diag.get('foc_active_ratio', 0.0))
            bp_terms_base = loss_foc + penalty_z_foc + kkt_penalty
            bp_terms_after_eta = eta_active_boost * bp_terms_base
            bp_adapt_scale = self._compute_bp_adaptive_scale(main_loss, bp_terms_after_eta)
            bp_terms = bp_adapt_scale * bp_terms_after_eta
            total_loss = main_loss + penalty_z + penalty_b + bp_terms
        with torch.no_grad():
            raw_m = torch.cat([m.reshape(-1) for m in raw_M_list], dim=0)
            use_m = torch.cat([m.reshape(-1) for m in M_list], dim=0)
            target_y = torch.cat(
                [(cf + m * p).reshape(-1) for cf, m, p in zip(CFip, M_list, P_children)],
                dim=0,
            )
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
                'pi_bp_foc_use_phat': float(1.0 if use_phat_for_bp_foc else 0.0),
                'pi_bellman_only': float(1.0 if bellman_only else 0.0),
                'pi_fixed_sdf': float(1.0 if self._pv_use_fixed_sdf() else 0.0),
                'pi_fixed_policy': float(1.0 if self._pv_use_fixed_policy() else 0.0),
            }
            self._latest_pi_terms.update(self._m_diagnostics('pi', raw_m, use_m, m_lo, m_hi))
            self._latest_pi_terms.update(self._tensor_tail_diagnostics('pi_target_y', target_y))
            self._latest_pi_terms.update(getattr(loss_fn, 'latest_foc_diag', {}))
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
        if self._pv_use_fixed_sdf():
            fixed = float(getattr(self.hyperparams, "pv_fixed_sdf_value", 0.98))
            M_list = [torch.full_like(m, fixed).detach() for m in raw_M_list]
        elif getattr(self.hyperparams, "q_use_detached_m", True):
            m_lo = float(getattr(self.hyperparams, "q_m_clamp_min", 0.5))
            m_hi = float(getattr(self.hyperparams, "q_m_clamp_max", 1.5))
            M_list = [m.clamp(m_lo, m_hi).detach() for m in raw_M_list]
        else:
            M_list = raw_M_list

        # 前向传播（Q 形状正则需要对输入求梯度）
        parent_state = strip_extra(parent).clone().detach().requires_grad_(True)
        target_model = self._target_policy_value()
        output_t = model(parent_state)
        with torch.no_grad():
            output_t_target = target_model(parent_state.detach())

        def _get_out(out, name: str, idx: int) -> torch.Tensor:
            if isinstance(out, dict):
                return out[name]
            if hasattr(out, name):
                return getattr(out, name)
            return out[:, idx:idx + 1]

        bp0_t = _get_out(output_t_target, 'bp0', 1).detach()
        bpI_t = _get_out(output_t_target, 'bpI', 2).detach()
        bar_i_t = _get_out(output_t_target, 'bar_i', 5).detach()
        bp_t = _get_out(output_t_target, 'bp', -1).detach()
        if bp_t.shape != bp0_t.shape:
            bp_t = bar_i_t * bpI_t + (1 - bar_i_t) * bp0_t
        b_parent = parent_state[:, 0:1]
        bar_z_t = _get_out(output_t_target, 'bar_z', 6).detach()
        bar_i_use = bar_i_t
        if self._pv_use_fixed_policy():
            bar_i_use = torch.zeros_like(bar_i_t)
        bp_use = bp_t
        bar_z_use = bar_z_t
        # 与 q_loss 主方程保持一致：Qsp 输入使用 b' = b / (bar_i*(G-1)+1)
        g_val = float(getattr(loss_fn, "g", 1.0))
        multiplier = bar_i_use * (g_val - 1.0) + 1.0
        b_sp = b_parent / multiplier.clamp_min(1e-6)

        outputsp_children = []
        for child in children:
            child_state_raw = strip_extra(child)
            childsp_state = child_state_raw.clone()
            childsp_state[:, 0:1] = b_sp
            with torch.no_grad():
                outputsp_children.append(target_model(childsp_state.detach()))

        # 提取 Q 和所需变量
        Q = _get_out(output_t, 'Q', 0)
        Qsp_children = [_get_out(out, 'Q', 0).detach() for out in outputsp_children]
        bar_zsp_children = [_get_out(out, 'bar_z', 6).detach() for out in outputsp_children]
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

        # Q 形状正则：
        # 1) dQ/dz > 0
        # 2) 低杠杆区 dQ/db > 0
        # 3) 高杠杆区 dQ/db < 0
        q_grads = torch.autograd.grad(
            outputs=Q.sum(),
            inputs=parent_state,
            create_graph=True,
            retain_graph=True
        )[0]
        dQ_db = q_grads[:, 0:1]
        dQ_dz = q_grads[:, 1:2]
        b_low = float(getattr(self.hyperparams, "q_shape_b_low", 0.2))
        b_high = float(getattr(self.hyperparams, "q_shape_b_high", 0.8))
        low_mask = (b_parent <= b_low).float()
        high_mask = (b_parent >= b_high).float()

        def _masked_mean(v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return (v * mask).sum() / (mask.sum() + 1e-6)

        q_shape_z = torch.relu(-dQ_dz).mean()
        q_shape_b_low = _masked_mean(torch.relu(-dQ_db), low_mask)
        q_shape_b_high = _masked_mean(torch.relu(dQ_db), high_mask)
        w_shape_z = float(getattr(self.hyperparams, "q_shape_weight_z", 1.0))
        w_shape_b_low = float(getattr(self.hyperparams, "q_shape_weight_b_low", 1.0))
        w_shape_b_high = float(getattr(self.hyperparams, "q_shape_weight_b_high", 1.0))
        q_shape_penalty = (
            w_shape_z * q_shape_z +
            w_shape_b_low * q_shape_b_low +
            w_shape_b_high * q_shape_b_high
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
                'q_shape_b_low': float(q_shape_b_low.item()),
                'q_shape_b_high': float(q_shape_b_high.item()),
                'q_physics': float(physics_loss.item()),
                'q_warmstart': float(warm_loss.item()),
                'q_warm_weight': float(warm_weight),
                'q_pretrain_mode': float(1.0 if q_only_stage else 0.0),
                'q_freeze_mode': float(1.0 if q_freeze_mode else 0.0),
            }
        return total_loss
    
    def _compute_fc2_loss(self, batch) -> torch.Tensor:
        """
        计算 FC2 损失（使用 FC2train2.ipynb pipeline，batch 为 DataFrame）
        """
        model = self.models.get('fc2')
        if model is None:
            return torch.tensor(0.0, device=self.device)

        if 'policy_value' not in self.models or self.models['policy_value'] is None:
            return torch.tensor(0.0, device=self.device)

        pv_model = self.models['policy_value']

        if isinstance(batch, pd.DataFrame):
            df = batch
            full_N = 1000
            entry_num = None
        elif isinstance(batch, dict) and 'df' in batch:
            df = batch['df']
            full_N = batch.get('full_N', 1000)
            entry_num = batch.get('entry_num', None)
        else:
            return torch.tensor(0.0, device=self.device)
        if df['branch'].min() < 0:
            df['branch'] = df['branch'] + 1

        pipe = FC2LossPipe(
            df=df,
            full_N=full_N,
            entry_num=entry_num,
            device=self.device,
        )
        # keep for inspection/debugging
        self._last_fc2_pipe = pipe
        fc2_loss = pipe.loss(model, pv_model)
        if isinstance(fc2_loss, tuple):
            fc2_loss = fc2_loss[0]
        return fc2_loss
    
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
        selected_indices = indices
        compact_indices = torch.arange(selected_indices.numel(), device=selected_indices.device)
        batches = []
        
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_units)
            idx = selected_indices[start:end]
            batch = {
                'parent': parent[idx],
                'children': [c[idx] for c in children],
                'child0': children[0][idx] if len(children) > 0 else None,
                'child1': children[1][idx] if len(children) > 1 else None,
                'parent_index': compact_indices[start:end],
                'parent_source_index': idx,
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

        max_units = int(getattr(self.hyperparams, "max_firm_train_units", 0))
        if max_units > 0 and indices.numel() > max_units:
            logger.warning(
                "Capping firm training units from %d to %d before batching",
                indices.numel(),
                max_units
            )
            indices = indices[:max_units]

        selected_indices = indices
        compact_indices = torch.arange(selected_indices.numel(), device=selected_indices.device)
        n_batches = (indices.numel() + batch_size - 1) // batch_size
        batches = []
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, indices.numel())
            idx = selected_indices[start:end]
            batch = {
                'parent': parent[idx],
                'children': [c[idx] for c in children],
                'child0': children[0][idx] if len(children) > 0 else None,
                'child1': children[1][idx] if len(children) > 1 else None,
                'parent_index': compact_indices[start:end],
                'parent_source_index': idx,
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
        if isinstance(out, dict):
            return out[name]
        if hasattr(out, name):
            return getattr(out, name)
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
        b_parent = parent_state[:, 0:1]
        output_children = []
        eta_children = []
        for child in children:
            child_state_raw = self._policy_strip_extra(child)
            eta_child = child[:, 2:3]
            child_state = child_state_raw.clone()
            child_state[:, 0:1] = eta_child * bp_for_p0 + (1 - eta_child) * b_parent
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
        b_parent = parent_state[:, 0:1]
        output_children = []
        eta_children = []
        for child in children:
            child_state_raw = self._policy_strip_extra(child)
            eta_child = child[:, 2:3]
            child_state = child_state_raw.clone()
            child_state[:, 0:1] = eta_child * bp_for_pi + (1 - eta_child) * b_parent
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

    def evaluate_target_grid_policy_convergence(self, batches: List[Dict[str, torch.Tensor]]) -> Dict:
        if not self._pv_use_target_grid_bp():
            return {'enabled': False, 'passed': True, 'policies': {}}
        if not batches or 'policy_value' not in self.models or self.models['policy_value'] is None:
            return {'enabled': False, 'passed': False, 'policies': {}}

        mae_thr = float(getattr(self.hyperparams, "bp_grid_conv_mae_thresh", 0.05))
        regret_thr = float(getattr(self.hyperparams, "bp_grid_conv_regret_p90_thresh", 1e-2))
        max_batches = int(getattr(self.hyperparams, "bp_grid_conv_max_batches", 4))
        max_batches = max(1, max_batches)
        survival_eps = float(getattr(self.hyperparams, "bp_grid_conv_survival_eps", 0.05))

        class _PolicyAccumulator:
            def __init__(self):
                self.errs: List[torch.Tensor] = []
                self.regrets: List[torch.Tensor] = []
                self.active_weights: List[torch.Tensor] = []

            def update(
                self,
                pred: torch.Tensor,
                grid: Dict[str, torch.Tensor],
                active_weight: Optional[torch.Tensor] = None,
            ) -> None:
                err = (pred.detach() - grid["bp_star"]).abs().reshape(-1)
                regret = grid["regret"].detach().reshape(-1)
                self.errs.append(err.cpu())
                self.regrets.append(regret.cpu())
                if active_weight is not None:
                    self.active_weights.append(active_weight.detach().reshape(-1).cpu().to(torch.float32))

            def summarize(self) -> Dict:
                if not self.errs:
                    return {
                        'enabled': False,
                        'n': 0,
                        'mae': float('nan'),
                        'mae_p90': float('nan'),
                        'regret_mean': float('nan'),
                        'regret_p90': float('nan'),
                        'mae_all': float('nan'),
                        'regret_p90_all': float('nan'),
                        'survival_active_share': float('nan'),
                        'passed': False,
                    }
                err = torch.cat(self.errs).to(torch.float32)
                regret = torch.cat(self.regrets).to(torch.float32)
                finite = torch.isfinite(err) & torch.isfinite(regret)
                if not bool(finite.any()):
                    return {
                        'enabled': True,
                        'n': 0,
                        'mae': float('nan'),
                        'mae_p90': float('nan'),
                        'regret_mean': float('nan'),
                        'regret_p90': float('nan'),
                        'mae_all': float('nan'),
                        'regret_p90_all': float('nan'),
                        'survival_active_share': float('nan'),
                        'passed': False,
                    }
                weight = None
                if self.active_weights:
                    weight = torch.cat(self.active_weights).to(torch.float32)
                    weight = weight[finite]
                err = err[finite]
                regret = regret[finite]
                mae_all = float(err.mean().item())
                mae_p90_all = float(torch.quantile(err, 0.9).item())
                regret_mean_all = float(regret.mean().item())
                regret_p90_all = float(torch.quantile(regret, 0.9).item())
                active_share = float('nan')
                if weight is not None:
                    active = torch.isfinite(weight) & (weight > survival_eps)
                    active_share = float(active.to(torch.float32).mean().item()) if active.numel() else 0.0
                    if bool(active.any()):
                        err_eval = err[active]
                        regret_eval = regret[active]
                    else:
                        err_eval = err
                        regret_eval = regret
                else:
                    err_eval = err
                    regret_eval = regret
                mae = float(err_eval.mean().item())
                mae_p90 = float(torch.quantile(err_eval, 0.9).item())
                regret_mean = float(regret_eval.mean().item())
                regret_p90 = float(torch.quantile(regret_eval, 0.9).item())
                return {
                    'enabled': True,
                    'n': int(err.numel()),
                    'n_active': int(err_eval.numel()),
                    'mae': mae,
                    'mae_p90': mae_p90,
                    'regret_mean': regret_mean,
                    'regret_p90': regret_p90,
                    'mae_all': mae_all,
                    'mae_p90_all': mae_p90_all,
                    'regret_mean_all': regret_mean_all,
                    'regret_p90_all': regret_p90_all,
                    'survival_active_share': active_share,
                    'passed': bool(mae < mae_thr and regret_p90 < regret_thr),
                }

        p0_acc = _PolicyAccumulator()
        pi_acc = _PolicyAccumulator()
        mix_acc = _PolicyAccumulator()
        model = self.models['policy_value']
        target_model = self._target_policy_value()
        teacher = BPGridTeacher.from_hyperparams(
            target_model,
            self.loss_fns['p0'],
            self.loss_fns['pi'],
            self.hyperparams,
        )
        m_lo = float(getattr(self.hyperparams, "pv_m_clamp_min", 0.7))
        m_hi = float(getattr(self.hyperparams, "pv_m_clamp_max", 1.3))

        with torch.no_grad():
            for batch in batches[:max_batches]:
                parent = batch['parent']
                children = self._get_policy_children(batch)
                if not children:
                    continue
                raw_M_list, M_list = self._build_policy_m_lists(parent, children, m_lo, m_hi)
                del raw_M_list
                parent_state = self._policy_strip_extra(parent)
                output_t = model(parent_state)
                bp0_t = self._policy_get_out(output_t, 'bp0', 1)
                bpI_t = self._policy_get_out(output_t, 'bpI', 2)
                bar_i_t = self._policy_get_out(output_t, 'bar_i', 4)
                bp_t = self._policy_get_out(output_t, 'bp', -1)
                if bp_t.shape != bp0_t.shape:
                    bp_t = bar_i_t * bpI_t + (1 - bar_i_t) * bp0_t
                if isinstance(output_t, dict):
                    bar_i_cond_online = output_t.get('bar_i_cond', bar_i_t)
                else:
                    bar_i_cond_online = getattr(output_t, 'bar_i_cond', bar_i_t)
                mix_weight_target = self._target_investment_conditional(
                    target_model,
                    parent_state,
                    fallback=bar_i_cond_online,
                )
                if isinstance(output_t, dict):
                    survival_online = output_t.get('survival_prob', torch.ones_like(bar_i_t))
                else:
                    survival_online = getattr(output_t, 'survival_prob', torch.ones_like(bar_i_t))
                mix_survival_target = self._target_survival_probability(
                    target_model,
                    parent_state,
                    fallback=survival_online,
                )
                bp_mix_cond = self._mixed_policy_conditional_bp(
                    output_t,
                    bp0_t,
                    bpI_t,
                    parent_state[:, 0:1],
                    fallback_bar_i=bar_i_cond_online,
                )

                p0_grid = teacher.compute(parent_state, children, M_list, branch='p0', bp_pred=bp0_t)
                pi_grid = teacher.compute(parent_state, children, M_list, branch='pi', bp_pred=bpI_t)
                mix_grid = teacher.compute(
                    parent_state,
                    children,
                    M_list,
                    branch='mix',
                    bp_pred=bp_mix_cond,
                    mix_weight=mix_weight_target,
                )
                p0_acc.update(bp0_t, p0_grid)
                pi_acc.update(bpI_t, pi_grid)
                mix_acc.update(bp_mix_cond, mix_grid, active_weight=mix_survival_target)

        policies = {
            'bp0': p0_acc.summarize(),
            'bpI': pi_acc.summarize(),
            'mix': mix_acc.summarize(),
        }
        enabled = [v for v in policies.values() if v.get('enabled', False)]
        passed = bool(enabled) and all(v.get('passed', False) for v in enabled)
        logger.info(
            "Target-grid policy convergence | mae<%.3e, regret_p90<%.3e, passed=%s",
            mae_thr,
            regret_thr,
            str(passed),
        )
        return {
            'enabled': True,
            'thresholds': {'mae': mae_thr, 'regret_p90': regret_thr, 'survival_eps': survival_eps},
            'max_batches': max_batches,
            'policies': policies,
            'passed': passed,
        }

    def evaluate_bellman_convergence(
        self,
        batches: List[Dict[str, torch.Tensor]],
        validation_batches: Optional[List[Dict[str, torch.Tensor]]] = None,
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

        max_q_samples = int(getattr(self.hyperparams, "bellman_conv_max_samples", 1_000_000))
        max_q_samples = max(1, max_q_samples)

        class _ResidualAccumulator:
            def __init__(self, max_samples: int):
                self.max_samples = max_samples
                self.n_total = 0
                self.n_finite = 0
                self.abs_sum = 0.0
                self.samples: List[torch.Tensor] = []
                self.n_sampled = 0

            def update(self, values: torch.Tensor) -> None:
                vals = values.detach().reshape(-1)
                self.n_total += int(vals.numel())
                if vals.numel() == 0:
                    return
                vals = vals[torch.isfinite(vals)]
                self.n_finite += int(vals.numel())
                if vals.numel() == 0:
                    return
                vals = vals.abs().to(torch.float32)
                self.abs_sum += float(vals.sum().item())
                remaining = self.max_samples - self.n_sampled
                if remaining <= 0:
                    return
                if vals.numel() > remaining:
                    idx = torch.randperm(vals.numel(), device=vals.device)[:remaining]
                    vals = vals[idx]
                self.samples.append(vals.cpu())
                self.n_sampled += int(vals.numel())

            def summarize(self, name: str, mean_thr: float, p90_thr: float) -> Dict:
                if self.n_finite == 0:
                    return {
                        'enabled': False,
                        'n': 0,
                        'n_total': self.n_total,
                        'n_finite': 0,
                        'n_used_for_p90': 0,
                        'nonfinite_ratio': 1.0 if self.n_total > 0 else 0.0,
                        'mean': float('nan'),
                        'p90': float('nan'),
                        'passed': False
                    }
                mean_v = self.abs_sum / max(self.n_finite, 1)
                if self.samples:
                    sample = torch.cat(self.samples, dim=0)
                    p90_v = float(torch.quantile(sample, 0.9).item())
                else:
                    p90_v = float('nan')
                nonfinite_ratio = 1.0 - (self.n_finite / max(self.n_total, 1))
                passed = bool(mean_v < mean_thr and p90_v < p90_thr and nonfinite_ratio == 0.0)
                logger.info(
                    "Bellman convergence [%s] | mean(abs)=%.6e, p90(abs)=%.6e, "
                    "n_total=%d, n_used=%d, nonfinite=%.3e, pass=%s",
                    name,
                    mean_v,
                    p90_v,
                    self.n_total,
                    self.n_sampled,
                    nonfinite_ratio,
                    str(passed)
                )
                return {
                    'enabled': True,
                    'n': self.n_finite,
                    'n_total': self.n_total,
                    'n_finite': self.n_finite,
                    'n_used_for_p90': self.n_sampled,
                    'nonfinite_ratio': nonfinite_ratio,
                    'mean': mean_v,
                    'p90': p90_v,
                    'passed': passed
                }

        p0_acc = _ResidualAccumulator(max_q_samples)
        pi_acc = _ResidualAccumulator(max_q_samples)
        q_acc = _ResidualAccumulator(max_q_samples)

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
                    p0_acc.update(p0_abs)
                if pi_abs.numel() > 0:
                    pi_acc.update(pi_abs)
                if q_abs.numel() > 0:
                    q_acc.update(q_abs)

        if was_training:
            model.train()

        equations = {
            'p0': p0_acc.summarize('p0', mean_thr, p90_thr),
            'pi': pi_acc.summarize('pi', mean_thr, p90_thr),
            'q': q_acc.summarize('q', mean_thr, p90_thr)
        }
        policy_train = self.evaluate_target_grid_policy_convergence(batches)
        policy_val = (
            self.evaluate_target_grid_policy_convergence(validation_batches)
            if validation_batches
            else {'enabled': False, 'passed': True, 'policies': {}}
        )
        policy_convergence = policy_val if policy_val.get('enabled', False) else policy_train

        enabled_eq = [m for m in equations.values() if m.get('enabled', False)]
        bellman_passed = bool(enabled_eq) and all(m.get('passed', False) for m in enabled_eq)
        policy_passed = (
            bool(policy_convergence.get('passed', False))
            if policy_convergence.get('enabled', False)
            else True
        )
        all_passed = bool(bellman_passed and policy_passed)
        summary = {
            'enabled': True,
            'thresholds': {'mean': mean_thr, 'p90': p90_thr},
            'max_quantile_samples': max_q_samples,
            'equations': equations,
            'policy': policy_convergence,
            'policy_train': policy_train,
            'policy_val': policy_val,
            'bellman_passed': bellman_passed,
            'policy_passed': policy_passed,
            'passed': all_passed
        }
        logger.info(
            "Bellman convergence summary | mean<%.3e, p90<%.3e, bellman_passed=%s, policy_passed=%s, passed=%s",
            mean_thr,
            p90_thr,
            str(bellman_passed),
            str(policy_passed),
            str(all_passed)
        )
        return summary

    def _run_batches(
        self,
        batches: List[Dict[str, torch.Tensor]],
        n_epochs: int,
        log_interval: int,
        train_modules: List[str],
        desc_prefix: str = ''
    ) -> Dict:
        """
        使用预生成的 batches 执行训练循环
        """
        if 'sdf_fc1' in train_modules:
            self._configure_sdf_lr_for_phase()

        q_pretrain_epochs = 0
        if 'policy_value' in train_modules:
            q_pretrain_epochs = max(0, int(getattr(self.hyperparams, "q_pretrain_epochs", 0)))
            q_warmstart_epochs = max(0, int(getattr(self.hyperparams, "q_warmstart_epochs", 0)))
            q_only_epochs = max(q_pretrain_epochs, q_warmstart_epochs)
        else:
            q_warmstart_epochs = 0
            q_only_epochs = 0

        pv_train_batches = batches
        validation_batches: List[Dict[str, torch.Tensor]] = []
        if 'policy_value' in train_modules and self._pv_use_target_grid_bp() and len(batches) > 1:
            val_fraction = float(getattr(self.hyperparams, "pv_target_grid_val_fraction", 0.10))
            val_fraction = min(max(val_fraction, 0.0), 0.5)
            n_val = int(round(len(batches) * val_fraction))
            if val_fraction > 0.0:
                n_val = max(1, n_val)
            n_val = min(n_val, len(batches) - 1)
            if n_val > 0:
                pv_train_batches = batches[:-n_val]
                validation_batches = batches[-n_val:]
                logger.info(
                    "%sTarget-grid PV validation split: train_batches=%d, val_batches=%d",
                    desc_prefix,
                    len(pv_train_batches),
                    len(validation_batches),
                )
        joint_module_split = (
            bool(validation_batches)
            and 'policy_value' in train_modules
            and any(m != 'policy_value' for m in train_modules)
        )

        epoch_offset = int(getattr(self, "_run_batches_epoch_offset", 0))
        for epoch in range(n_epochs):
            effective_epoch = epoch_offset + epoch
            self._current_epoch_idx = effective_epoch
            self._prepare_sdf_shock_bank_for_epoch(batches, effective_epoch, train_modules)
            self._q_only_stage = bool(
                'policy_value' in train_modules and q_only_epochs > 0 and epoch < q_only_epochs
            )
            policy_loss_terms = None
            if 'policy_value' in train_modules and q_only_epochs > 0:
                policy_loss_terms = ['q'] if self._q_only_stage else ['p0', 'pi', 'q']
            epoch_losses = []
            if joint_module_split:
                non_pv_modules = [m for m in train_modules if m != 'policy_value']
                if non_pv_modules:
                    for batch in tqdm(batches, desc=f"{desc_prefix}Epoch {epoch+1}/{n_epochs} non-PV"):
                        losses = self.train_step(batch, non_pv_modules, policy_loss_terms=policy_loss_terms)
                        epoch_losses.append(losses)
                        if self.step_count % log_interval == 0:
                            avg_loss = np.mean([l['total'] for l in epoch_losses[-log_interval:]])
                            logger.info(f"Step {self.step_count}: loss={avg_loss:.6f}")
                for batch in tqdm(pv_train_batches, desc=f"{desc_prefix}Epoch {epoch+1}/{n_epochs} PV"):
                    losses = self.train_step(
                        batch,
                        ['policy_value'],
                        policy_loss_terms=policy_loss_terms
                    )
                    epoch_losses.append(losses)
                    self._check_policy_value_gate(
                        losses,
                        context=f"episode={self.episode_id}, epoch={epoch + 1}, batch={len(epoch_losses)}"
                    )
                    if self.step_count % log_interval == 0:
                        avg_loss = np.mean([l['total'] for l in epoch_losses[-log_interval:]])
                        logger.info(f"Step {self.step_count}: loss={avg_loss:.6f}")
            else:
                train_loop_batches = pv_train_batches if 'policy_value' in train_modules else batches
                for batch in tqdm(train_loop_batches, desc=f"{desc_prefix}Epoch {epoch+1}/{n_epochs}"):
                    losses = self.train_step(
                        batch,
                        train_modules,
                        policy_loss_terms=policy_loss_terms
                    )
                    epoch_losses.append(losses)
                    if 'policy_value' in train_modules and 'policy_value' in self.models:
                        self._check_policy_value_gate(
                            losses,
                            context=f"episode={self.episode_id}, epoch={epoch + 1}, batch={len(epoch_losses)}"
                        )

                    if self.step_count % log_interval == 0:
                        avg_loss = np.mean([l['total'] for l in epoch_losses[-log_interval:]])
                        lr = None
                        for name in train_modules:
                            lr = self.lr_schedulers.get(name)
                            if lr is not None:
                                break
                        current_lr = lr.get_lr() if lr else 0

                        logger.info(
                            f"Step {self.step_count}: "
                            f"loss={avg_loss:.6f}, lr={current_lr:.2e}"
                        )

            # 每个 epoch 追加 bp-only 精修：只优化 P0/PI 且仅更新 bp 头。
            bp_refine_steps = max(0, int(getattr(self.hyperparams, "bp_refine_steps_per_epoch", 0)))
            bp_refine_cap = int(getattr(self.hyperparams, "bp_refine_batch_cap", 32))
            if (
                'policy_value' in train_modules and
                not self._q_only_stage and
                bp_refine_steps > 0 and
                len(pv_train_batches) > 0
            ):
                if bp_refine_cap > 0:
                    refine_batches = pv_train_batches[:min(bp_refine_cap, len(pv_train_batches))]
                else:
                    refine_batches = pv_train_batches
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
                            refine_avg, refine_metadata = self._aggregate_metric_records(refine_losses)
                            logger.info(
                                "%sBP refine %d/%d finished: %s",
                                desc_prefix,
                                r + 1,
                                bp_refine_steps,
                                {**refine_avg, **refine_metadata}
                            )
                finally:
                    self._bp_only_stage = False
                
            avg_losses, epoch_metadata = self._aggregate_metric_records(epoch_losses)
            if 'policy_value' in train_modules and 'policy_value' in self.models:
                self._check_policy_value_gate(
                    avg_losses,
                    context=f"episode={self.episode_id}, epoch={epoch + 1}"
                )
            epoch_summary = {**avg_losses, **epoch_metadata}
            logger.info(f"{desc_prefix}Epoch {epoch+1} finished: {epoch_summary}")
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
            self._maybe_update_firm_target_epoch(train_modules)
        self._q_only_stage = False
        self._bp_only_stage = False
        convergence = None
        if 'policy_value' in train_modules and 'policy_value' in self.models:
            convergence = self.evaluate_bellman_convergence(pv_train_batches, validation_batches=validation_batches)

        result = {
            'final_losses': avg_losses
        }
        if epoch_metadata:
            result['metadata'] = epoch_metadata
        if convergence is not None:
            result['convergence'] = convergence
            result['target_grid_validation_batches'] = len(validation_batches)
        if 'policy_value' in train_modules:
            self._last_policy_value_stage_summary = dict(avg_losses)
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
        if 'fc2' not in self.models:
            return None
        if (self.df is None or self.df.empty) and self.tensor_firm is not None:
            self.df = self._table_to_dataframe(self.tensor_firm)
        if self.df is None or self.df.empty:
            return None
        epoch_losses = []
        for _ in tqdm(range(n_epochs), desc='FC2 Epochs'):
            losses = self.train_step(self.df, ['fc2'])
            epoch_losses.append(losses)
            if self.step_count % log_interval == 0:
                avg_loss = np.mean([l['total'] for l in epoch_losses[-log_interval:]])
                lr = self.lr_schedulers.get('fc2')
                current_lr = lr.get_lr() if lr else 0
                logger.info(
                    "FC2 Step %d: loss=%.6f, lr=%.2e",
                    self.step_count,
                    avg_loss,
                    current_lr
                )
        if not epoch_losses:
            return None
        avg_losses, metadata = self._aggregate_metric_records(epoch_losses)
        logger.info("FC2 Epochs finished: %s", {**avg_losses, **metadata})
        result = {'final_losses': avg_losses}
        if metadata:
            result['metadata'] = metadata
        return result

    def _evaluate_sdf_fc1_batches(
        self,
        batches: List[Dict[str, torch.Tensor]],
        prefix: str,
        max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        if not batches or 'sdf_fc1' not in self.models:
            return {}
        model = self.models['sdf_fc1']
        was_training = bool(getattr(model, 'training', False))
        model.eval()
        max_batches = len(batches) if max_batches is None else min(max_batches, len(batches))

        hatc_true_parts: List[torch.Tensor] = []
        hatc_pred_parts: List[torch.Tensor] = []
        primary_hatc_baseline_parts: List[torch.Tensor] = []
        lnk_true_parts: List[torch.Tensor] = []
        lnk_pred_parts: List[torch.Tensor] = []
        primary_lnk_baseline_parts: List[torch.Tensor] = []
        dlnk_true_parts: List[torch.Tensor] = []
        dlnk_pred_parts: List[torch.Tensor] = []
        current_hatc_forecast_parts: List[torch.Tensor] = []
        current_hatc_true_parts: List[torch.Tensor] = []
        current_lnk_forecast_parts: List[torch.Tensor] = []
        current_lnk_true_parts: List[torch.Tensor] = []
        recursive_hatc_true_parts: List[torch.Tensor] = []
        recursive_hatc_pred_parts: List[torch.Tensor] = []
        recursive_lnk_true_parts: List[torch.Tensor] = []
        recursive_lnk_pred_parts: List[torch.Tensor] = []
        recursive_dlnk_true_parts: List[torch.Tensor] = []
        recursive_dlnk_pred_parts: List[torch.Tensor] = []
        primary_m_parts: List[torch.Tensor] = []
        recursive_m_parts: List[torch.Tensor] = []
        primary_signed_parts: List[torch.Tensor] = []
        recursive_signed_parts: List[torch.Tensor] = []
        primary_raw_signed_parts: List[torch.Tensor] = []
        primary_normalized_signed_parts: List[torch.Tensor] = []
        recursive_raw_signed_parts: List[torch.Tensor] = []
        recursive_normalized_signed_parts: List[torch.Tensor] = []
        rollout_rmse_parts: Dict[int, List[torch.Tensor]] = {}

        try:
            with torch.no_grad():
                for batch in batches[:max_batches]:
                    parent = batch['parent']
                    children = batch.get('children', [])
                    if not children:
                        child0 = batch.get('child0')
                        child1 = batch.get('child1')
                        if child0 is not None and child1 is not None:
                            children = [child0, child1]
                    if not children or len(children) < 2:
                        continue
                    children_t = torch.stack(children[:2], dim=1)
                    has_true_prev_macro = parent.shape[1] >= 9
                    c_prev_true = parent[:, 7:8] if has_true_prev_macro else parent[:, 5:6]
                    k_prev_true = parent[:, 8:9] if has_true_prev_macro else parent[:, 6:7]
                    c_prev_forecast = parent[:, 5:6]
                    k_prev_forecast = parent[:, 6:7]
                    w_parent_primary, w_children_primary, M_primary, c_children_primary, k_children_primary = model.forward_step(
                        x_prev=parent[:, 4:5],
                        x_curr=children_t[:, :, 4:5],
                        hatcf_prev=c_prev_true,
                        lnkf_prev=k_prev_true,
                        return_physical=True
                    )
                    w_parent_recursive, w_children_recursive, M_recursive, c_children_recursive, k_children_recursive = model.forward_step(
                        x_prev=parent[:, 4:5],
                        x_curr=children_t[:, :, 4:5],
                        hatcf_prev=c_prev_forecast,
                        lnkf_prev=k_prev_forecast,
                        return_physical=True
                    )
                    loss_fn = getattr(self, "loss_fns", {}).get("sdf") if hasattr(self, "loss_fns") else None
                    if loss_fn is not None:
                        normalized_logr_clip = float(
                            getattr(self.hyperparams, "sdf_normalized_logr_clip", 20.0)
                        )
                        residual_pack_primary = loss_fn.compute_wealth_residuals(
                            w_parent=w_parent_primary.squeeze(-1),
                            w_children=w_children_primary.squeeze(-1),
                            k_parent=k_prev_true.squeeze(-1),
                            k_children=k_children_primary.squeeze(-1),
                            c_parent=c_prev_true.squeeze(-1),
                            c_children=c_children_primary.squeeze(-1),
                            normalized_logr_clip=normalized_logr_clip,
                        )
                        residual_pack_recursive = loss_fn.compute_wealth_residuals(
                            w_parent=w_parent_recursive.squeeze(-1),
                            w_children=w_children_recursive.squeeze(-1),
                            k_parent=k_prev_forecast.squeeze(-1),
                            k_children=k_children_recursive.squeeze(-1),
                            c_parent=c_prev_forecast.squeeze(-1),
                            c_children=c_children_recursive.squeeze(-1),
                            normalized_logr_clip=normalized_logr_clip,
                        )
                        residual_mode = str(
                            getattr(self.hyperparams, "sdf_gate_residual_mode", "normalized_ratio")
                        ).lower()
                        residuals_primary = (
                            residual_pack_primary["normalized"]
                            if residual_mode == "normalized_ratio"
                            else residual_pack_primary["raw"]
                        )
                        residuals_recursive = (
                            residual_pack_recursive["normalized"]
                            if residual_mode == "normalized_ratio"
                            else residual_pack_recursive["raw"]
                        )
                        if residual_pack_primary["raw"].shape[1] >= 2:
                            primary_raw_signed_parts.append(
                                (
                                    residual_pack_primary["raw"][:, 0]
                                    * residual_pack_primary["raw"][:, 1]
                                ).detach().reshape(-1).cpu()
                            )
                            primary_normalized_signed_parts.append(
                                (
                                    residual_pack_primary["normalized"][:, 0]
                                    * residual_pack_primary["normalized"][:, 1]
                                ).detach().reshape(-1).cpu()
                            )
                            primary_signed_parts.append(
                                (residuals_primary[:, 0] * residuals_primary[:, 1]).detach().reshape(-1).cpu()
                            )
                        if residual_pack_recursive["raw"].shape[1] >= 2:
                            recursive_raw_signed_parts.append(
                                (
                                    residual_pack_recursive["raw"][:, 0]
                                    * residual_pack_recursive["raw"][:, 1]
                                ).detach().reshape(-1).cpu()
                            )
                            recursive_normalized_signed_parts.append(
                                (
                                    residual_pack_recursive["normalized"][:, 0]
                                    * residual_pack_recursive["normalized"][:, 1]
                                ).detach().reshape(-1).cpu()
                            )
                            recursive_signed_parts.append(
                                (residuals_recursive[:, 0] * residuals_recursive[:, 1]).detach().reshape(-1).cpu()
                            )
                    if children_t.shape[-1] >= 10:
                        hatcf_true = children_t[:, :, 8:9]
                        lnkf_true = children_t[:, :, 9:10]
                    elif children_t.shape[-1] >= 9:
                        hatcf_true = children_t[:, :, 7:8]
                        lnkf_true = children_t[:, :, 8:9]
                    else:
                        hatcf_true = None
                        lnkf_true = None
                    if hatcf_true is not None and lnkf_true is not None:
                        hatc_true_parts.append(hatcf_true.detach().reshape(-1).cpu())
                        hatc_pred_parts.append(c_children_primary.detach().reshape(-1).cpu())
                        primary_hatc_baseline_parts.append(
                            c_prev_true.unsqueeze(1).expand_as(hatcf_true).detach().reshape(-1).cpu()
                        )
                        lnk_true_parts.append(lnkf_true.detach().reshape(-1).cpu())
                        lnk_pred_parts.append(k_children_primary.detach().reshape(-1).cpu())
                        primary_lnk_baseline_parts.append(
                            k_prev_true.unsqueeze(1).expand_as(lnkf_true).detach().reshape(-1).cpu()
                        )
                        dlnk_true_parts.append((lnkf_true - k_prev_true.unsqueeze(1)).detach().reshape(-1).cpu())
                        dlnk_pred_parts.append((k_children_primary - k_prev_true.unsqueeze(1)).detach().reshape(-1).cpu())
                        if has_true_prev_macro:
                            current_hatc_forecast_parts.append(c_prev_forecast.detach().reshape(-1).cpu())
                            current_hatc_true_parts.append(c_prev_true.detach().reshape(-1).cpu())
                            current_lnk_forecast_parts.append(k_prev_forecast.detach().reshape(-1).cpu())
                            current_lnk_true_parts.append(k_prev_true.detach().reshape(-1).cpu())
                            recursive_hatc_true_parts.append(hatcf_true.detach().reshape(-1).cpu())
                            recursive_hatc_pred_parts.append(c_children_recursive.detach().reshape(-1).cpu())
                            recursive_lnk_true_parts.append(lnkf_true.detach().reshape(-1).cpu())
                            recursive_lnk_pred_parts.append(k_children_recursive.detach().reshape(-1).cpu())
                            recursive_dlnk_true_parts.append(
                                (lnkf_true - k_prev_true.unsqueeze(1)).detach().reshape(-1).cpu()
                            )
                            recursive_dlnk_pred_parts.append(
                                (k_children_recursive - k_prev_forecast.unsqueeze(1)).detach().reshape(-1).cpu()
                            )
                    primary_m_parts.append(M_primary.detach().reshape(-1).cpu())
                    recursive_m_parts.append(M_recursive.detach().reshape(-1).cpu())

                    rollout_required = {
                        "fc1_rollout_initial_state",
                        "fc1_rollout_initial_x",
                        "fc1_rollout_future_x",
                        "fc1_rollout_target_states",
                    }
                    if rollout_required.issubset(batch.keys()):
                        initial_state = batch["fc1_rollout_initial_state"]
                        x_prev_roll = batch["fc1_rollout_initial_x"]
                        future_x = batch["fc1_rollout_future_x"]
                        target_states = batch["fc1_rollout_target_states"]
                        hatc_roll = initial_state[:, 0:1]
                        lnk_roll = initial_state[:, 1:2]
                        horizon = min(
                            int(getattr(self.hyperparams, "fc1_rollout_horizon", future_x.shape[1])),
                            int(future_x.shape[1]),
                            int(target_states.shape[1]),
                        )
                        for h in range(horizon):
                            x_curr = future_x[:, h, :]
                            _, _, _, hatc_next, lnk_next = model.forward_step(
                                x_prev=x_prev_roll,
                                x_curr=x_curr.unsqueeze(1),
                                hatcf_prev=hatc_roll,
                                lnkf_prev=lnk_roll,
                                return_physical=True,
                            )
                            hatc_roll = hatc_next[:, 0, :]
                            lnk_roll = lnk_next[:, 0, :]
                            pred_state = torch.cat([hatc_roll, lnk_roll], dim=1)
                            target_state = target_states[:, h, :]
                            se = (pred_state - target_state).pow(2).mean(dim=1).detach().cpu()
                            rollout_rmse_parts.setdefault(h + 1, []).append(se)
                            x_prev_roll = x_curr
        finally:
            if was_training:
                model.train()

        out: Dict[str, float] = {}

        def _safe_forecast_metrics(
            name: str,
            true_parts: List[torch.Tensor],
            pred_parts: List[torch.Tensor],
            baseline_parts: Optional[List[torch.Tensor]] = None,
        ) -> None:
            if not true_parts or not pred_parts:
                return
            y_raw = torch.cat(true_parts).to(torch.float32)
            p_raw = torch.cat(pred_parts).to(torch.float32)
            total_n = int(y_raw.numel())
            target_mask = torch.isfinite(y_raw)
            pred_mask = torch.isfinite(p_raw)
            mask = target_mask & pred_mask
            target_finite_n = int(target_mask.sum().item())
            pred_finite_n = int(pred_mask.sum().item())
            finite_n = int(mask.sum().item())
            out[f'{prefix}_{name}_total_n'] = float(total_n)
            out[f'{prefix}_{name}_target_finite_n'] = float(target_finite_n)
            out[f'{prefix}_{name}_target_finite_ratio'] = (
                float(target_mask.to(torch.float32).mean().item()) if total_n > 0 else float('nan')
            )
            out[f'{prefix}_{name}_pred_finite_n'] = float(pred_finite_n)
            out[f'{prefix}_{name}_pred_finite_ratio'] = (
                float(pred_mask.to(torch.float32).mean().item()) if total_n > 0 else float('nan')
            )
            out[f'{prefix}_{name}_finite_n'] = float(finite_n)
            out[f'{prefix}_{name}_joint_finite_ratio'] = (
                float(mask.to(torch.float32).mean().item()) if total_n > 0 else float('nan')
            )
            out[f'{prefix}_{name}_finite_ratio'] = out[f'{prefix}_{name}_joint_finite_ratio']
            out[f'{prefix}_{name}_n'] = float(finite_n)
            if int(mask.sum().item()) < 2:
                out[f'{prefix}_{name}_r2'] = float('nan')
                out[f'{prefix}_{name}_rmse'] = float('nan')
                return
            y = y_raw[mask]
            p = p_raw[mask]
            err = p - y
            target_mean = y.mean()
            target_std = y.std(unbiased=True)
            pred_mean = p.mean()
            pred_std = p.std(unbiased=True)
            sst = (y - target_mean).pow(2).sum()
            sse = err.pow(2).sum()
            out[f'{prefix}_{name}_target_mean'] = float(target_mean.item())
            out[f'{prefix}_{name}_target_std'] = float(target_std.item())
            out[f'{prefix}_{name}_target_min'] = float(y.min().item())
            out[f'{prefix}_{name}_target_max'] = float(y.max().item())
            out[f'{prefix}_{name}_pred_mean'] = float(pred_mean.item())
            out[f'{prefix}_{name}_pred_std'] = float(pred_std.item())
            out[f'{prefix}_{name}_bias'] = float(err.mean().item())
            out[f'{prefix}_{name}_mae'] = float(err.abs().mean().item())
            out[f'{prefix}_{name}_r2'] = float((1.0 - sse / sst).item()) if float(sst.item()) > 1e-12 else float('nan')
            out[f'{prefix}_{name}_rmse'] = float(torch.sqrt(err.pow(2).mean()).item())
            if baseline_parts:
                b_raw = torch.cat(baseline_parts).to(torch.float32)
                baseline_mask = mask & torch.isfinite(b_raw)
                if int(baseline_mask.sum().item()) >= 2:
                    y_b = y_raw[baseline_mask]
                    p_b = p_raw[baseline_mask]
                    b_b = b_raw[baseline_mask]
                    model_mse = (p_b - y_b).pow(2).mean()
                    baseline_mse = (b_b - y_b).pow(2).mean()
                    out[f'{prefix}_{name}_baseline_rmse'] = float(torch.sqrt(baseline_mse).item())
                    out[f'{prefix}_{name}_innovation_std'] = float((y_b - b_b).std(unbiased=True).item())
                    out[f'{prefix}_{name}_skill_vs_persistence'] = (
                        float((1.0 - model_mse / baseline_mse).item())
                        if float(baseline_mse.item()) > 1e-12
                        else float('nan')
                    )

        _safe_forecast_metrics('hatc', hatc_true_parts, hatc_pred_parts, primary_hatc_baseline_parts)
        _safe_forecast_metrics('lnk', lnk_true_parts, lnk_pred_parts, primary_lnk_baseline_parts)
        _safe_forecast_metrics(
            'primary_true_state_hatc_next',
            hatc_true_parts,
            hatc_pred_parts,
            primary_hatc_baseline_parts,
        )
        _safe_forecast_metrics(
            'primary_true_state_lnk_next',
            lnk_true_parts,
            lnk_pred_parts,
            primary_lnk_baseline_parts,
        )
        _safe_forecast_metrics('primary_true_state_dlnk_next', dlnk_true_parts, dlnk_pred_parts)
        _safe_forecast_metrics('recursive_forecast_state_hatc_next', recursive_hatc_true_parts, recursive_hatc_pred_parts)
        _safe_forecast_metrics('recursive_forecast_state_lnk_next', recursive_lnk_true_parts, recursive_lnk_pred_parts)
        _safe_forecast_metrics('recursive_forecast_state_dlnk_next', recursive_dlnk_true_parts, recursive_dlnk_pred_parts)

        def _safe_gap_metrics(name: str, true_parts: List[torch.Tensor], pred_parts: List[torch.Tensor]) -> None:
            if not true_parts or not pred_parts:
                return
            y = torch.cat(true_parts).to(torch.float32)
            p = torch.cat(pred_parts).to(torch.float32)
            mask = torch.isfinite(y) & torch.isfinite(p)
            out[f'{prefix}_{name}_n'] = float(mask.sum().item())
            if int(mask.sum().item()) == 0:
                return
            d = p[mask] - y[mask]
            out[f'{prefix}_{name}_mean_error'] = float(d.mean().item())
            out[f'{prefix}_{name}_mae'] = float(d.abs().mean().item())
            out[f'{prefix}_{name}_rmse'] = float(torch.sqrt(d.pow(2).mean()).item())
            out[f'{prefix}_{name}_p50_abs'] = float(torch.quantile(d.abs(), 0.50).item())
            out[f'{prefix}_{name}_p90_abs'] = float(torch.quantile(d.abs(), 0.90).item())

        _safe_gap_metrics('current_belief_hatc_vs_realized', current_hatc_true_parts, current_hatc_forecast_parts)
        _safe_gap_metrics('current_belief_lnk_vs_realized', current_lnk_true_parts, current_lnk_forecast_parts)

        def _safe_m_metrics(name: str, parts: List[torch.Tensor]) -> None:
            if not parts:
                return
            raw_m = torch.cat(parts).to(torch.float32)
            finite_mask = torch.isfinite(raw_m)
            out[f'{prefix}_{name}_total_n'] = float(raw_m.numel())
            out[f'{prefix}_{name}_finite_n'] = float(finite_mask.sum().item())
            out[f'{prefix}_{name}_finite_ratio'] = (
                float(finite_mask.to(torch.float32).mean().item()) if raw_m.numel() > 0 else float('nan')
            )
            out[f'{prefix}_{name}_nonfinite_n'] = float((~finite_mask).sum().item())
            m = raw_m[finite_mask]
            out[f'{prefix}_{name}_n'] = float(m.numel())
            if m.numel() > 0:
                out[f'{prefix}_{name}_mean'] = float(m.mean().item())
                out[f'{prefix}_{name}_p50'] = float(torch.quantile(m, 0.50).item())
                out[f'{prefix}_{name}_p90'] = float(torch.quantile(m, 0.90).item())
                out[f'{prefix}_{name}_p99'] = float(torch.quantile(m, 0.99).item())
                out[f'{prefix}_{name}_max'] = float(m.max().item())
                out[f'{prefix}_{name}_lt_0p7_rate'] = float((m < 0.7).to(torch.float32).mean().item())
                out[f'{prefix}_{name}_gt_1p3_rate'] = float((m > 1.3).to(torch.float32).mean().item())

        _safe_m_metrics('primary_true_state_M', primary_m_parts)
        _safe_m_metrics('recursive_forecast_state_M', recursive_m_parts)

        def _safe_signed_t(name: str, parts: List[torch.Tensor]) -> None:
            if not parts:
                return
            v = torch.cat(parts).to(torch.float32)
            v = v[torch.isfinite(v)]
            out[f'{prefix}_{name}_n'] = float(v.numel())
            if v.numel() < 2:
                out[f'{prefix}_{name}_mean'] = float('nan')
                out[f'{prefix}_{name}_std'] = float('nan')
                out[f'{prefix}_{name}_se'] = float('nan')
                out[f'{prefix}_{name}_t'] = float('nan')
                return
            mean = v.mean()
            std = v.std(unbiased=True)
            se = std.clamp_min(1e-12) / float(v.numel()) ** 0.5
            out[f'{prefix}_{name}_mean'] = float(mean.item())
            out[f'{prefix}_{name}_std'] = float(std.item())
            out[f'{prefix}_{name}_se'] = float(se.item())
            out[f'{prefix}_{name}_t'] = float((mean / se).item())

        _safe_signed_t('primary_true_state_signed_aio', primary_signed_parts)
        _safe_signed_t('recursive_forecast_state_signed_aio', recursive_signed_parts)
        _safe_signed_t('primary_true_state_raw_signed_aio', primary_raw_signed_parts)
        _safe_signed_t('primary_true_state_normalized_signed_aio', primary_normalized_signed_parts)
        _safe_signed_t('recursive_forecast_state_raw_signed_aio', recursive_raw_signed_parts)
        _safe_signed_t('recursive_forecast_state_normalized_signed_aio', recursive_normalized_signed_parts)

        rollout_rmse_by_h: Dict[int, float] = {}
        for h, parts in rollout_rmse_parts.items():
            if not parts:
                continue
            se = torch.cat(parts).to(torch.float32)
            se = se[torch.isfinite(se)]
            if se.numel() == 0:
                continue
            rmse = float(torch.sqrt(se.mean()).item())
            rollout_rmse_by_h[h] = rmse
            out[f'{prefix}_fc1_rollout_h{h}_rmse'] = rmse
        if 1 in rollout_rmse_by_h:
            target_h = min(max(rollout_rmse_by_h), int(getattr(self.hyperparams, "fc1_rollout_horizon", 5)))
            growth = float(rollout_rmse_by_h[target_h] / (rollout_rmse_by_h[1] + 1e-8))
            out[f'{prefix}_fc1_rollout_actual_horizon'] = float(target_h)
            out[f'{prefix}_fc1_rmse_growth_h{target_h}'] = growth
            out[f'{prefix}_fc1_rmse_growth_terminal'] = growth
        return out

    def _fc1_data_viability_passed(self, eval_metrics: Dict[str, float], prefix: str) -> Tuple[bool, Dict[str, Any]]:
        min_pairs = int(getattr(self.hyperparams, "fc1_gate_min_pairs", 128))
        finite_min = float(getattr(self.hyperparams, "fc1_rollout_finite_ratio_min", 1.0))
        std_floor = float(getattr(self.hyperparams, "fc1_target_std_floor", 1e-4))
        variables = {
            "hatc": "primary_true_state_hatc_next",
            "lnk": "primary_true_state_lnk_next",
        }
        details = {}
        passed_all = True
        for short_name, metric_name in variables.items():
            n = eval_metrics.get(f"{prefix}_{metric_name}_target_finite_n", float("nan"))
            finite_ratio = eval_metrics.get(f"{prefix}_{metric_name}_target_finite_ratio", float("nan"))
            target_std = eval_metrics.get(f"{prefix}_{metric_name}_target_std", float("nan"))
            valid = (
                np.isfinite(n)
                and np.isfinite(finite_ratio)
                and np.isfinite(target_std)
                and float(n) >= min_pairs
                and float(finite_ratio) >= finite_min
            )
            low_variance = np.isfinite(target_std) and float(target_std) < std_floor
            details[short_name] = {
                "n": float(n),
                "finite_ratio": float(finite_ratio),
                "target_std": float(target_std),
                "low_variance": bool(low_variance),
                "valid": bool(valid),
            }
            passed_all = passed_all and bool(valid)
        return bool(passed_all), {
            "passed": bool(passed_all),
            "failure_source": None if passed_all else "rollout_data_invalid",
            "min_pairs": min_pairs,
            "finite_ratio_min": finite_min,
            "target_std_floor": std_floor,
            "variables": details,
        }

    def _fc1_one_step_gate_passed(self, eval_metrics: Dict[str, float], prefix: str) -> Tuple[bool, Dict[str, Any]]:
        std_floor = float(getattr(self.hyperparams, "fc1_target_std_floor", 1e-4))
        pred_finite_min = float(getattr(self.hyperparams, "fc1_rollout_finite_ratio_min", 1.0))
        persistence_floor = float(getattr(self.hyperparams, "fc1_persistence_rmse_floor", 1e-6))
        r2_min = float(getattr(self.hyperparams, "fc1_one_step_r2_min", 0.0))
        skill_min = float(getattr(self.hyperparams, "fc1_one_step_skill_min", 0.0))
        configs = {
            "hatc": {
                "name": "primary_true_state_hatc_next",
                "abs_rmse_max": float(getattr(self.hyperparams, "fc1_one_step_hatc_rmse_abs_max", 0.05)),
            },
            "lnk": {
                "name": "primary_true_state_lnk_next",
                "abs_rmse_max": float(getattr(self.hyperparams, "fc1_one_step_lnk_rmse_abs_max", 0.05)),
            },
        }
        checks = {}
        passed_all = True
        for variable, cfg in configs.items():
            name = cfg["name"]
            target_std = eval_metrics.get(f"{prefix}_{name}_target_std", float("nan"))
            r2 = eval_metrics.get(f"{prefix}_{name}_r2", float("nan"))
            rmse = eval_metrics.get(f"{prefix}_{name}_rmse", float("nan"))
            skill = eval_metrics.get(f"{prefix}_{name}_skill_vs_persistence", float("nan"))
            pred_finite_ratio = eval_metrics.get(f"{prefix}_{name}_pred_finite_ratio", float("nan"))
            baseline_rmse = eval_metrics.get(f"{prefix}_{name}_baseline_rmse", float("nan"))
            innovation_std = eval_metrics.get(f"{prefix}_{name}_innovation_std", float("nan"))
            pred_finite_passed = np.isfinite(pred_finite_ratio) and float(pred_finite_ratio) >= pred_finite_min
            low_variance = np.isfinite(target_std) and float(target_std) < std_floor
            near_perfect_persistence = np.isfinite(baseline_rmse) and float(baseline_rmse) <= persistence_floor
            if not pred_finite_passed:
                variable_passed = False
                gate_mode = "prediction_nonfinite"
                failure_source = "fc1_prediction_nonfinite"
            elif near_perfect_persistence:
                variable_passed = np.isfinite(rmse) and float(rmse) <= cfg["abs_rmse_max"]
                gate_mode = "near_perfect_persistence"
                failure_source = "fc1_one_step_underfit"
            elif low_variance:
                variable_passed = np.isfinite(rmse) and float(rmse) <= cfg["abs_rmse_max"]
                gate_mode = "absolute_rmse_low_target_variance"
                failure_source = "fc1_one_step_underfit"
            else:
                variable_passed = (
                    np.isfinite(r2)
                    and np.isfinite(skill)
                    and float(r2) >= r2_min
                    and float(skill) >= skill_min
                )
                gate_mode = "r2_and_persistence_skill"
                failure_source = "fc1_one_step_underfit"
            checks[variable] = {
                "passed": bool(variable_passed),
                "gate_mode": gate_mode,
                "target_std": float(target_std),
                "pred_finite_ratio": float(pred_finite_ratio),
                "pred_finite_ratio_min": pred_finite_min,
                "baseline_rmse": float(baseline_rmse),
                "persistence_rmse_floor": persistence_floor,
                "innovation_std": float(innovation_std),
                "r2": float(r2),
                "skill_vs_persistence": float(skill),
                "rmse": float(rmse),
                "rmse_abs_max": cfg["abs_rmse_max"],
                "failure_source": None if variable_passed else failure_source,
            }
            passed_all = passed_all and bool(variable_passed)
        failure_sources = [
            check.get("failure_source")
            for check in checks.values()
            if check.get("failure_source") is not None
        ]
        failure_source = (
            None if passed_all else (
                "fc1_prediction_nonfinite"
                if "fc1_prediction_nonfinite" in failure_sources
                else "fc1_one_step_underfit"
            )
        )
        return bool(passed_all), {
            "passed": bool(passed_all),
            "failure_source": failure_source,
            "r2_min": r2_min,
            "skill_min": skill_min,
            "checks": checks,
        }

    def _evaluate_fc1_recursive_diagnostic(self, eval_metrics: Dict[str, float], prefix: str) -> Tuple[bool, Dict[str, Any]]:
        std_floor = float(getattr(self.hyperparams, "fc1_target_std_floor", 1e-4))
        pred_finite_min = float(getattr(self.hyperparams, "fc1_rollout_finite_ratio_min", 1.0))
        min_r2 = float(getattr(self.hyperparams, "fc1_recursive_r2_min", 0.0))
        growth_max = float(getattr(self.hyperparams, "fc1_rmse_growth_h5_max", 2.0))
        horizon = int(getattr(self.hyperparams, "fc1_rollout_horizon", 5))
        actual_horizon = eval_metrics.get(f"{prefix}_fc1_rollout_actual_horizon", float("nan"))
        growth = eval_metrics.get(f"{prefix}_fc1_rmse_growth_terminal", float("nan"))
        passed_all = np.isfinite(growth) and float(growth) <= growth_max
        variables = {}
        for variable, metric_name, abs_rmse_max in [
            ("hatc", "recursive_forecast_state_hatc_next", float(getattr(self.hyperparams, "fc1_one_step_hatc_rmse_abs_max", 0.05))),
            ("lnk", "recursive_forecast_state_lnk_next", float(getattr(self.hyperparams, "fc1_one_step_lnk_rmse_abs_max", 0.05))),
        ]:
            target_std = eval_metrics.get(f"{prefix}_{metric_name}_target_std", float("nan"))
            r2 = eval_metrics.get(f"{prefix}_{metric_name}_r2", float("nan"))
            rmse = eval_metrics.get(f"{prefix}_{metric_name}_rmse", float("nan"))
            pred_finite_ratio = eval_metrics.get(f"{prefix}_{metric_name}_pred_finite_ratio", float("nan"))
            pred_finite_passed = np.isfinite(pred_finite_ratio) and float(pred_finite_ratio) >= pred_finite_min
            low_variance = np.isfinite(target_std) and float(target_std) < std_floor
            if not pred_finite_passed:
                variable_passed = False
                gate_mode = "prediction_nonfinite"
                failure_source = "fc1_prediction_nonfinite"
            elif low_variance:
                variable_passed = np.isfinite(rmse) and float(rmse) <= abs_rmse_max
                gate_mode = "absolute_rmse_low_target_variance"
                failure_source = "fc1_recursive_instability"
            else:
                variable_passed = np.isfinite(r2) and float(r2) >= min_r2
                gate_mode = "recursive_r2"
                failure_source = "fc1_recursive_instability"
            variables[variable] = {
                "passed": bool(variable_passed),
                "gate_mode": gate_mode,
                "target_std": float(target_std),
                "pred_finite_ratio": float(pred_finite_ratio),
                "pred_finite_ratio_min": pred_finite_min,
                "r2": float(r2),
                "rmse": float(rmse),
                "rmse_abs_max": abs_rmse_max,
                "failure_source": None if variable_passed else failure_source,
            }
            passed_all = passed_all and bool(variable_passed)
        failure_sources = [
            check.get("failure_source")
            for check in variables.values()
            if check.get("failure_source") is not None
        ]
        failure_source = (
            None if passed_all else (
                "fc1_prediction_nonfinite"
                if "fc1_prediction_nonfinite" in failure_sources
                else "fc1_recursive_instability"
            )
        )
        return bool(passed_all), {
            "passed": bool(passed_all),
            "failure_source": failure_source,
            "min_r2": min_r2,
            "configured_horizon": horizon,
            "actual_horizon": float(actual_horizon),
            "rmse_growth": float(growth),
            "rmse_growth_max": growth_max,
            "max_rmse_growth": growth_max,
            "variables": variables,
        }

    @staticmethod
    def _fc1_gate_violation_score(one_step_diag: Dict[str, Any]) -> float:
        # Recursive forecast diagnostics are intentionally excluded from the hard
        # FC1 gate score. FC1 acceptance is based on calculated-state one-step
        # performance; recursive rollout remains a reported stability diagnostic.
        penalty = 1e6

        def _lower(value: Any, threshold: Any) -> float:
            try:
                value_f = float(value)
                threshold_f = float(threshold)
            except (TypeError, ValueError):
                return penalty
            if not np.isfinite(value_f) or not np.isfinite(threshold_f):
                return penalty
            return max(0.0, threshold_f - value_f)

        def _upper(value: Any, threshold: Any) -> float:
            try:
                value_f = float(value)
                threshold_f = float(threshold)
            except (TypeError, ValueError):
                return penalty
            if not np.isfinite(value_f) or not np.isfinite(threshold_f):
                return penalty
            return max(0.0, value_f - threshold_f)

        def _rmse_violation(check: Dict[str, Any]) -> float:
            rmse = check.get("rmse", float("nan"))
            limit = check.get("rmse_abs_max", float("nan"))
            try:
                rmse_f = float(rmse)
                limit_f = float(limit)
            except (TypeError, ValueError):
                return penalty
            if not np.isfinite(rmse_f) or not np.isfinite(limit_f) or limit_f <= 0:
                return penalty
            return max(0.0, rmse_f / limit_f - 1.0)

        score = 0.0
        for check in one_step_diag.get("checks", {}).values():
            mode = check.get("gate_mode")
            if mode in {"absolute_rmse_low_target_variance", "near_perfect_persistence"}:
                score += _rmse_violation(check)
            elif mode == "prediction_nonfinite":
                score += penalty
            else:
                score += _lower(check.get("r2"), one_step_diag.get("r2_min", 0.0))
                score += _lower(check.get("skill_vs_persistence"), one_step_diag.get("skill_min", 0.0))

        return float(score)

    def _run_fc1_until_gates(
        self,
        train_batches: List[Dict[str, torch.Tensor]],
        val_batches: List[Dict[str, torch.Tensor]],
        log_interval: int,
    ) -> Dict[str, Any]:
        configured_epochs_per_round = int(getattr(self.hyperparams, "fc1_epochs_per_round", 0))
        epochs_per_round = (
            configured_epochs_per_round
            if configured_epochs_per_round > 0
            else int(getattr(self.hyperparams, "fc1_only_epochs", 10))
        )
        epochs_per_round = max(1, epochs_per_round)
        max_rounds = max(1, int(getattr(self.hyperparams, "fc1_max_rounds", 8)))
        patience = max(1, int(getattr(self.hyperparams, "fc1_plateau_patience", 2)))
        min_improvement = float(getattr(self.hyperparams, "fc1_min_relative_improvement", 0.01))
        eval_batches = int(getattr(self.hyperparams, "sdf_fc1_eval_max_batches", 0))
        eval_batches_arg = eval_batches if eval_batches > 0 else None

        before_eval = self._evaluate_sdf_fc1_batches(
            val_batches,
            prefix="fc1_before",
            max_batches=eval_batches_arg,
        )
        data_passed, data_diag = self._fc1_data_viability_passed(before_eval, prefix="fc1_before")
        if not data_passed:
            return {
                "passed": False,
                "failed_stage": "rollout_data_viability",
                "failure_source": "rollout_data_invalid",
                "data_viability": data_diag,
                "before_eval_metrics": before_eval,
                "rounds": [],
            }

        rounds: List[Dict[str, Any]] = []
        best_score = float("inf")
        stale_rounds = 0
        for round_idx in range(max_rounds):
            display_round = round_idx + 1
            train_summary = self._run_batches(
                train_batches,
                epochs_per_round,
                log_interval,
                ["sdf_fc1"],
                desc_prefix=f"SDF/FC1(fc1 r{display_round}/{max_rounds}) ",
            )
            prefix = f"after_fc1_round{display_round}"
            eval_metrics = self._evaluate_sdf_fc1_batches(
                val_batches,
                prefix=prefix,
                max_batches=eval_batches_arg,
            )
            one_step_passed, one_step_diag = self._fc1_one_step_gate_passed(eval_metrics, prefix=prefix)
            _, recursive_diag = self._evaluate_fc1_recursive_diagnostic(eval_metrics, prefix=prefix)
            score = self._fc1_gate_violation_score(one_step_diag)
            round_record = {
                "round": display_round,
                "train_summary": train_summary,
                "eval_metrics": eval_metrics,
                "one_step_gate": one_step_diag,
                "recursive_diagnostic": recursive_diag,
                "score": score,
            }
            rounds.append(round_record)
            if one_step_passed:
                return {
                    "passed": True,
                    "failed_stage": None,
                    "failure_source": None,
                    "data_viability": data_diag,
                    "before_eval_metrics": before_eval,
                    "rounds_completed": display_round,
                    "final_train_summary": train_summary,
                    "final_eval_metrics": eval_metrics,
                    "final_one_step_gate": one_step_diag,
                    "final_recursive_diagnostic": recursive_diag,
                    "rounds": rounds,
                }

            relative_improvement = (
                (best_score - score) / max(abs(best_score), 1e-12)
                if np.isfinite(best_score)
                else float("inf")
            )
            if score < best_score:
                best_score = score
            if relative_improvement < min_improvement:
                stale_rounds += 1
            else:
                stale_rounds = 0
            if stale_rounds >= patience:
                failure_source = one_step_diag.get("failure_source", "fc1_one_step_underfit")
                return {
                    "passed": False,
                    "failed_stage": "fc1_plateau",
                    "failure_source": failure_source,
                    "reason": "fc1_metrics_plateaued",
                    "data_viability": data_diag,
                    "before_eval_metrics": before_eval,
                    "rounds_completed": display_round,
                    "final_eval_metrics": eval_metrics,
                    "final_one_step_gate": one_step_diag,
                    "final_recursive_diagnostic": recursive_diag,
                    "rounds": rounds,
                }

        final_round = rounds[-1]
        failure_source = final_round["one_step_gate"].get("failure_source", "fc1_one_step_underfit")
        return {
            "passed": False,
            "failed_stage": "fc1_max_rounds",
            "failure_source": failure_source,
            "reason": "fc1_failed_after_max_rounds",
            "data_viability": data_diag,
            "before_eval_metrics": before_eval,
            "rounds_completed": max_rounds,
            "final_eval_metrics": final_round["eval_metrics"],
            "final_one_step_gate": final_round["one_step_gate"],
            "final_recursive_diagnostic": final_round["recursive_diagnostic"],
            "rounds": rounds,
        }

    def _sdf_gate_passed(
        self,
        eval_metrics: Dict[str, float],
        prefix: str,
        stage: SDFTrainingPhase
    ) -> Tuple[bool, Dict[str, Any]]:
        max_log_mean_error = float(getattr(self.hyperparams, "sdf_log_mean_error_max", 0.02))
        max_t = float(getattr(self.hyperparams, "sdf_signed_t_abs_max", 2.0))
        min_finite_ratio = float(getattr(self.hyperparams, "sdf_gate_m_finite_ratio_min", 1.0))
        p99_max = float(getattr(self.hyperparams, "sdf_gate_m_p99_max", float("inf")))
        max_max = float(getattr(self.hyperparams, "sdf_gate_m_max_max", float("inf")))
        tail_gate_active = bool(np.isfinite(p99_max) or np.isfinite(max_max))
        target = getattr(self.hyperparams, "sdf_log_mean_target", None)
        target = float(target) if target is not None else 0.0
        gate_residual_mode = str(
            getattr(self.hyperparams, "sdf_gate_residual_mode", "normalized_ratio")
        ).lower()
        if gate_residual_mode == "normalized_ratio":
            signed_suffix = "normalized_signed_aio"
        elif gate_residual_mode == "raw":
            signed_suffix = "raw_signed_aio"
        else:
            raise ValueError(f"Unknown sdf_gate_residual_mode={gate_residual_mode!r}")
        if stage in (SDFTrainingPhase.EPISODE0_BOOTSTRAP, SDFTrainingPhase.SDF_TRUE_ONLY):
            m_prefix = f"{prefix}_primary_true_state_M"
            signed_prefix = f"{prefix}_primary_true_state_{signed_suffix}"
        else:
            m_prefix = f"{prefix}_recursive_forecast_state_M"
            signed_prefix = f"{prefix}_recursive_forecast_state_{signed_suffix}"
        t_key = f"{signed_prefix}_t"
        m_key = f"{m_prefix}_mean"
        m_mean = eval_metrics.get(m_key, float("nan"))
        finite_ratio = eval_metrics.get(f"{m_prefix}_finite_ratio", float("nan"))
        m_p99 = eval_metrics.get(f"{m_prefix}_p99", float("nan"))
        m_max = eval_metrics.get(f"{m_prefix}_max", float("nan"))
        signed_t = eval_metrics.get(t_key, float("nan"))
        if not np.isfinite(signed_t):
            legacy_prefix = signed_prefix.replace(f"_{signed_suffix}", "_signed_aio")
            signed_t = eval_metrics.get(f"{legacy_prefix}_t", float("nan"))
        log_mean_error = abs(np.log(max(float(m_mean), 1e-12)) - target) if np.isfinite(m_mean) else float("nan")
        passed = (
            np.isfinite(log_mean_error)
            and np.isfinite(signed_t)
            and np.isfinite(finite_ratio)
            and np.isfinite(m_p99)
            and np.isfinite(m_max)
            and log_mean_error <= max_log_mean_error
            and abs(float(signed_t)) <= max_t
            and float(finite_ratio) >= min_finite_ratio
            and float(m_p99) <= p99_max
            and float(m_max) <= max_max
        )
        diag = {
            "passed": bool(passed),
            "stage": stage.value,
            "m_mean": float(m_mean),
            "m_finite_ratio": float(finite_ratio),
            "m_finite_ratio_min": min_finite_ratio,
            "m_p99": float(m_p99),
            "m_p99_max": p99_max,
            "m_max": float(m_max),
            "m_max_max": max_max,
            "tail_gate_active": tail_gate_active,
            "log_mean_target": target,
            "log_mean_error": float(log_mean_error),
            "max_log_mean_error": max_log_mean_error,
            "gate_residual_mode": gate_residual_mode,
            "signed_aio_t": float(signed_t),
            "max_signed_t_abs": max_t,
        }
        return bool(passed), diag

    def _episode0_sdf_safety_gate_passed(
        self,
        eval_metrics: Dict[str, float],
        prefix: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        target = getattr(self.hyperparams, "sdf_log_mean_target", None)
        target = float(target) if target is not None else 0.0
        max_log_mean_error = float(getattr(self.hyperparams, "episode0_sdf_log_mean_error_max", 0.25))
        max_clip_low = float(getattr(self.hyperparams, "episode0_sdf_clip_low_ratio_max", 0.20))
        min_finite_ratio = float(getattr(self.hyperparams, "episode0_sdf_finite_ratio_min", 1.0))
        m_prefix = f"{prefix}_primary_true_state_M"
        m_mean = eval_metrics.get(f"{m_prefix}_mean", float("nan"))
        finite_ratio = eval_metrics.get(f"{m_prefix}_finite_ratio", float("nan"))
        clip_low_ratio = eval_metrics.get(f"{m_prefix}_lt_0p7_rate", float("nan"))
        clip_high_ratio = eval_metrics.get(f"{m_prefix}_gt_1p3_rate", float("nan"))
        m_p99 = eval_metrics.get(f"{m_prefix}_p99", float("nan"))
        m_max = eval_metrics.get(f"{m_prefix}_max", float("nan"))
        gate_residual_mode = str(
            getattr(self.hyperparams, "sdf_gate_residual_mode", "normalized_ratio")
        ).lower()
        if gate_residual_mode == "normalized_ratio":
            signed_suffix = "normalized_signed_aio"
        elif gate_residual_mode == "raw":
            signed_suffix = "raw_signed_aio"
        else:
            raise ValueError(f"Unknown sdf_gate_residual_mode={gate_residual_mode!r}")
        signed_prefix = f"{prefix}_primary_true_state_{signed_suffix}"
        signed_mean = eval_metrics.get(f"{signed_prefix}_mean", float("nan"))
        signed_std = eval_metrics.get(f"{signed_prefix}_std", float("nan"))
        signed_se = eval_metrics.get(f"{signed_prefix}_se", float("nan"))
        signed_t = eval_metrics.get(f"{signed_prefix}_t", float("nan"))
        if not np.isfinite(signed_t):
            legacy_prefix = f"{prefix}_primary_true_state_signed_aio"
            signed_mean = eval_metrics.get(f"{legacy_prefix}_mean", float("nan"))
            signed_std = eval_metrics.get(f"{legacy_prefix}_std", float("nan"))
            signed_se = eval_metrics.get(f"{legacy_prefix}_se", float("nan"))
            signed_t = eval_metrics.get(f"{legacy_prefix}_t", float("nan"))
        log_mean_error = abs(np.log(max(float(m_mean), 1e-12)) - target) if np.isfinite(m_mean) else float("nan")
        passed = (
            np.isfinite(m_mean)
            and np.isfinite(log_mean_error)
            and np.isfinite(finite_ratio)
            and np.isfinite(clip_low_ratio)
            and float(finite_ratio) >= min_finite_ratio
            and log_mean_error <= max_log_mean_error
            and float(clip_low_ratio) <= max_clip_low
        )
        diag = {
            "passed": bool(passed),
            "gate_type": "episode0_sdf_safety",
            "stage": SDFTrainingPhase.EPISODE0_BOOTSTRAP.value,
            "m_mean": float(m_mean),
            "m_finite_ratio": float(finite_ratio),
            "m_finite_ratio_min": min_finite_ratio,
            "m_lt_0p7_rate": float(clip_low_ratio),
            "m_lt_0p7_rate_max": max_clip_low,
            "m_gt_1p3_rate": float(clip_high_ratio),
            "m_p99": float(m_p99),
            "m_max": float(m_max),
            "log_mean_target": target,
            "log_mean_error": float(log_mean_error),
            "max_log_mean_error": max_log_mean_error,
            "gate_residual_mode": gate_residual_mode,
            "signed_aio_mean": float(signed_mean),
            "signed_aio_std": float(signed_std),
            "signed_aio_se": float(signed_se),
            "signed_aio_t": float(signed_t),
            "signed_aio_t_observed": float(signed_t),
            "signed_aio_t_binding": False,
        }
        return bool(passed), diag

    def _evaluate_post_refresh_sdf_gate(
        self,
        module_summaries: Dict[str, Any],
        batch_size: int,
        n_branches: int,
    ) -> Dict[str, Any]:
        if self._use_tensor_pipeline() and self.tensor_macro is not None:
            sdf_table = self._build_sdf_pairs_from_macro_tensor(self.tensor_macro)
        else:
            macro_df = self.df_macro
            if (macro_df is None or macro_df.empty) and self.tensor_macro is not None:
                macro_df = self._table_to_dataframe(self.tensor_macro)
            if macro_df is None or macro_df.empty:
                raise RuntimeError("Post-refresh SDF gate requires non-empty refreshed macro data.")
            df_macro_sdf = build_sdf_pairs_from_macro_ts(macro_df.copy(), include_hatc_lnk_t1=True)
            sdf_table = TensorTable(
                data=torch.tensor(df_macro_sdf.values, device=self.device, dtype=torch.float32),
                columns=list(df_macro_sdf.columns)
            )

        prev_flag = self.add_FC1loss
        prev_phase = getattr(self, "sdf_training_phase", SDFTrainingPhase.SDF_TRUE_ONLY)
        self.add_FC1loss = True
        try:
            _, val_table, holdout_diag = self._split_sdf_table_by_path(sdf_table)
            val_batches = self._create_sdf_batches_from_macro_tensor(
                val_table,
                batch_size=batch_size,
                n_branches=n_branches,
            )
            if not val_batches:
                raise RuntimeError("Post-refresh SDF gate could not build holdout validation batches.")
            eval_batches = int(getattr(self.hyperparams, "sdf_fc1_eval_max_batches", 0))
            eval_batches_arg = eval_batches if eval_batches > 0 else None
            refresh_eval = self._evaluate_sdf_fc1_batches(
                val_batches,
                prefix="post_refresh",
                max_batches=eval_batches_arg,
            )
            fc1_data_passed, fc1_data_diag = self._fc1_data_viability_passed(
                refresh_eval,
                prefix="post_refresh",
            )
            fc1_one_step_passed, fc1_one_step_diag = self._fc1_one_step_gate_passed(
                refresh_eval,
                prefix="post_refresh",
            )
            _, fc1_recursive_diag = self._evaluate_fc1_recursive_diagnostic(
                refresh_eval,
                prefix="post_refresh",
            )
            fc1_passed = bool(fc1_data_passed and fc1_one_step_passed)
            sdf_passed, sdf_diag = self._sdf_gate_passed(
                refresh_eval,
                prefix="post_refresh",
                stage=SDFTrainingPhase.SDF_RECURSIVE_ONLY,
            )
            result = {
                "passed": bool(fc1_passed and sdf_passed),
                "failed_stage": None if (fc1_passed and sdf_passed) else "post_refresh",
                "holdout_split": holdout_diag,
                "fc1_gate": {
                    "passed": bool(fc1_passed),
                    "failure_source": None if fc1_passed else (
                        fc1_data_diag.get("failure_source") or
                        fc1_one_step_diag.get("failure_source", "fc1_one_step_underfit")
                    ),
                    "data_viability": fc1_data_diag,
                    "one_step_gate": fc1_one_step_diag,
                    "recursive_diagnostic": fc1_recursive_diag,
                },
                "sdf_recursive_gate": sdf_diag,
            }
            module_summaries["sdf_fc1_post_refresh_eval"] = refresh_eval
            module_summaries["sdf_fc1_post_refresh_gate"] = result
            return result
        finally:
            self.add_FC1loss = prev_flag
            self.set_sdf_training_phase(prev_phase)

    def _run_sdf_recon_from_macro(
        self,
        module_summaries: Dict,
        n_epochs: int,
        batch_size: int,
        log_interval: int,
        n_branches: int
    ) -> Dict[str, Any]:
        if self.episode_id <= 0:
            raise RuntimeError("FC1/SDF Stage2 is only valid after Episode 0.")
        macro_df = self.df_macro
        macro_r2_diag: Dict[str, Any] = {}
        if self._use_tensor_pipeline() and self.tensor_macro is not None:
            macro_r2_diag = self._macro_forecast_r2_tensor(self.tensor_macro)
        else:
            if (macro_df is None or macro_df.empty) and self.tensor_macro is not None:
                macro_df = self._table_to_dataframe(self.tensor_macro)
            if macro_df is None or macro_df.empty:
                raise RuntimeError("FC1/SDF Stage2 requires non-empty macro simulation data.")
            self.df_macro = macro_df
            macro_r2_diag = self._macro_forecast_r2(macro_df)

        if macro_r2_diag:
            logger.info(
                "Macro R2 before SDF recon | n_t=%.0f, R2(Hatc)=%.6f, R2(LnK)=%.6f",
                macro_r2_diag.get('n_t', float('nan')),
                macro_r2_diag.get('r2_hatc', float('nan')),
                macro_r2_diag.get('r2_lnk', float('nan')),
            )
            module_summaries['macro_diag_before_sdf2'] = macro_r2_diag

        if self._use_tensor_pipeline() and self.tensor_macro is not None:
            sdf_table = self._build_sdf_pairs_from_macro_tensor(self.tensor_macro)
        else:
            if macro_df is None or macro_df.empty:
                raise RuntimeError("FC1/SDF Stage2 requires non-empty macro simulation data.")
            df_macro_sdf = build_sdf_pairs_from_macro_ts(macro_df.copy(), include_hatc_lnk_t1=True)
            sdf_table = TensorTable(
                data=torch.tensor(df_macro_sdf.values, device=self.device, dtype=torch.float32),
                columns=list(df_macro_sdf.columns)
            )

        prev_flag = self.add_FC1loss
        prev_teacher_flag = self._fc1_teacher_forcing_stage
        prev_phase = getattr(self, "sdf_training_phase", SDFTrainingPhase.SDF_TRUE_ONLY)
        self.add_FC1loss = True
        try:
            train_table, val_table, holdout_diag = self._split_sdf_table_by_path(sdf_table)
            module_summaries['sdf_fc1_holdout_split'] = holdout_diag
            train_batches = self._create_sdf_batches_from_macro_tensor(
                train_table, batch_size=batch_size, n_branches=n_branches
            )
            val_batches = self._create_sdf_batches_from_macro_tensor(
                val_table, batch_size=batch_size, n_branches=n_branches
            )
            if not train_batches:
                raise RuntimeError("FC1/SDF Stage2 could not build non-empty SDF macro batches.")
            if not val_batches:
                raise RuntimeError("FC1/SDF Stage2 could not build non-empty holdout validation batches.")
            eval_batches = int(getattr(self.hyperparams, "sdf_fc1_eval_max_batches", 0))
            eval_batches_arg = eval_batches if eval_batches > 0 else None
            gate_result: Dict[str, Any] = {"passed": True, "failed_stage": None}
            before_eval = self._evaluate_sdf_fc1_batches(
                val_batches,
                prefix='before',
                max_batches=eval_batches_arg
            )
            if before_eval:
                module_summaries['sdf_fc1_fixed_batch_eval_before'] = before_eval

            def _fail(stage_name: str, diag: Dict[str, Any]) -> Dict[str, Any]:
                result = {
                    "passed": False,
                    "failed_stage": stage_name,
                    "failure_source": diag.get("failure_source") if isinstance(diag, dict) else None,
                    "diagnostics": diag,
                }
                module_summaries['sdf_fc1_gate'] = result
                logger.warning("FC1/SDF gate failed at %s: %s", stage_name, diag)
                return result

            if bool(getattr(self.hyperparams, "sdf_training_schedule_enabled", True)):
                fc1_epochs = max(0, int(getattr(self.hyperparams, "fc1_only_epochs", 10)))
                true_epochs = max(0, int(getattr(self.hyperparams, "sdf_true_only_epochs", 20)))
                recursive_epochs = max(0, int(getattr(self.hyperparams, "sdf_recursive_only_epochs", 10)))

                if fc1_epochs > 0:
                    self.set_sdf_training_phase(SDFTrainingPhase.FC1_ONLY)
                    fc1_result = self._run_fc1_until_gates(
                        train_batches=train_batches,
                        val_batches=val_batches,
                        log_interval=log_interval,
                    )
                    module_summaries['sdf_fc1_fc1_continuation'] = fc1_result
                    module_summaries['sdf_fc1_fc1_only'] = fc1_result.get('final_train_summary', {})
                    if fc1_result.get('final_eval_metrics'):
                        module_summaries['sdf_fc1_eval_after_fc1_only'] = fc1_result['final_eval_metrics']
                    module_summaries['sdf_fc1_gate_fc1_only'] = fc1_result
                    if not fc1_result.get("passed", False):
                        return _fail(
                            fc1_result.get("failed_stage", SDFTrainingPhase.FC1_ONLY.value),
                            fc1_result,
                        )
                if true_epochs > 0:
                    self.set_sdf_training_phase(SDFTrainingPhase.SDF_TRUE_ONLY)
                    module_summaries['sdf_fc1_sdf_true_only'] = self._run_batches(
                        train_batches, true_epochs, log_interval, ['sdf_fc1'], desc_prefix='SDF/FC1(sdf-true) '
                    )
                    true_eval = self._evaluate_sdf_fc1_batches(
                        val_batches, prefix='after_sdf_true', max_batches=eval_batches_arg
                    )
                    module_summaries['sdf_fc1_eval_after_sdf_true_only'] = true_eval
                    passed, diag = self._sdf_gate_passed(
                        true_eval, prefix='after_sdf_true', stage=SDFTrainingPhase.SDF_TRUE_ONLY
                    )
                    module_summaries['sdf_fc1_gate_sdf_true_only'] = diag
                    if not passed:
                        return _fail(SDFTrainingPhase.SDF_TRUE_ONLY.value, diag)
                if recursive_epochs > 0:
                    self.set_sdf_training_phase(SDFTrainingPhase.SDF_RECURSIVE_ONLY)
                    module_summaries['sdf_fc1_sdf_recursive_only'] = self._run_batches(
                        train_batches, recursive_epochs, log_interval, ['sdf_fc1'], desc_prefix='SDF/FC1(sdf-recursive) '
                    )
                    recursive_eval = self._evaluate_sdf_fc1_batches(
                        val_batches, prefix='after_sdf_recursive', max_batches=eval_batches_arg
                    )
                    module_summaries['sdf_fc1_eval_after_sdf_recursive_only'] = recursive_eval
                    passed, diag = self._sdf_gate_passed(
                        recursive_eval,
                        prefix='after_sdf_recursive',
                        stage=SDFTrainingPhase.SDF_RECURSIVE_ONLY,
                    )
                    module_summaries['sdf_fc1_gate_sdf_recursive_only'] = diag
                    if not passed:
                        return _fail(SDFTrainingPhase.SDF_RECURSIVE_ONLY.value, diag)
                module_summaries['sdf_fc1_stage2'] = module_summaries.get(
                    'sdf_fc1_sdf_recursive_only',
                    module_summaries.get('sdf_fc1_sdf_true_only', module_summaries.get('sdf_fc1_fc1_only', {}))
                )
            else:
                tf_epochs = max(0, int(getattr(self.hyperparams, "fc1_teacher_forcing_epochs", 0)))
                if tf_epochs > 0:
                    self.set_sdf_training_phase(SDFTrainingPhase.FC1_ONLY)
                    self._fc1_teacher_forcing_stage = True
                    module_summaries['sdf_fc1_teacher_forcing'] = self._run_batches(
                        train_batches, tf_epochs, log_interval, ['sdf_fc1'], desc_prefix='SDF/FC1(tf) '
                    )
                    self._fc1_teacher_forcing_stage = False
                self.set_sdf_training_phase(SDFTrainingPhase.SDF_RECURSIVE_ONLY)
                module_summaries['sdf_fc1_stage2'] = self._run_batches(
                    train_batches, n_epochs, log_interval, ['sdf_fc1'], desc_prefix='SDF/FC1(stage2) '
                )
            # keep backward compatibility for consumers expecting a single sdf_fc1 key
            module_summaries['sdf_fc1'] = module_summaries['sdf_fc1_stage2']
            after_eval = self._evaluate_sdf_fc1_batches(
                val_batches,
                prefix='after',
                max_batches=eval_batches_arg
            )
            if after_eval:
                module_summaries['sdf_fc1_fixed_batch_eval_after'] = after_eval
            module_summaries['sdf_fc1_gate'] = gate_result
            return gate_result
        finally:
            self.add_FC1loss = prev_flag
            self._fc1_teacher_forcing_stage = prev_teacher_flag
            self.set_sdf_training_phase(prev_phase)

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

    def _split_sdf_dataframe_by_path(
        self,
        sdf_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        if sdf_df is None or sdf_df.empty or 'path' not in sdf_df.columns:
            return sdf_df, sdf_df, {
                'sdf_fc1_holdout_active': False,
                'sdf_fc1_holdout_reason': 'empty_or_missing_path',
            }
        val_fraction = float(getattr(self.hyperparams, "sdf_fc1_val_fraction", 0.2))
        val_fraction = min(max(val_fraction, 0.0), 0.5)
        unique_paths = np.asarray(pd.unique(sdf_df['path']))
        allow_in_sample = bool(getattr(self.hyperparams, "allow_in_sample_sdf_gate_for_debug", False))
        if val_fraction <= 0.0:
            if not allow_in_sample:
                raise RuntimeError(
                    "Path-level Episode 0 SDF validation is disabled by sdf_fc1_val_fraction=0. "
                    "Set allow_in_sample_sdf_gate_for_debug=True only for debug runs."
                )
            return sdf_df, sdf_df, {
                'sdf_fc1_holdout_active': False,
                'sdf_fc1_holdout_reason': 'disabled_debug_in_sample',
                'sdf_fc1_holdout_n_paths': int(unique_paths.size),
            }
        if unique_paths.size < 2:
            if not allow_in_sample:
                raise RuntimeError(
                    "Path-level Episode 0 SDF validation requires at least two paths. "
                    "Set allow_in_sample_sdf_gate_for_debug=True only for debug runs."
                )
            return sdf_df, sdf_df, {
                'sdf_fc1_holdout_active': False,
                'sdf_fc1_holdout_reason': 'insufficient_paths_debug_in_sample',
                'sdf_fc1_holdout_n_paths': int(unique_paths.size),
            }
        seed = int(getattr(self.hyperparams, "sdf_fc1_val_seed", 12345))
        rng = np.random.default_rng(seed)
        order = rng.permutation(unique_paths.size)
        n_val = int(round(unique_paths.size * val_fraction))
        n_val = min(max(1, n_val), unique_paths.size - 1)
        val_paths = set(unique_paths[order[:n_val]].tolist())
        val_mask = sdf_df['path'].isin(val_paths)
        train_df = sdf_df.loc[~val_mask].copy()
        val_df = sdf_df.loc[val_mask].copy()
        return train_df, val_df, {
            'sdf_fc1_holdout_active': True,
            'sdf_fc1_holdout_seed': seed,
            'sdf_fc1_holdout_fraction': val_fraction,
            'sdf_fc1_train_paths': int(unique_paths.size - n_val),
            'sdf_fc1_val_paths': int(n_val),
            'sdf_fc1_train_rows': int(len(train_df)),
            'sdf_fc1_val_rows': int(len(val_df)),
        }

    def _run_episode0_sdf_bootstrap_until_gate(
        self,
        train_batches: List[Dict[str, torch.Tensor]],
        val_batches: List[Dict[str, torch.Tensor]],
        n_epochs: int,
        log_interval: int,
    ) -> Dict[str, Any]:
        if int(self.episode_id) != 0:
            raise RuntimeError("Episode 0 SDF bootstrap continuation is only valid for episode_id == 0.")
        if not train_batches:
            raise RuntimeError("Episode 0 SDF bootstrap requires non-empty training batches.")
        if not val_batches:
            raise RuntimeError("Episode 0 SDF bootstrap requires non-empty validation batches.")

        epochs_per_round = int(getattr(self.hyperparams, "episode0_sdf_epochs_per_round", 0))
        if epochs_per_round <= 0:
            epochs_per_round = int(n_epochs)
        epochs_per_round = max(1, epochs_per_round)
        max_rounds = max(1, int(getattr(self.hyperparams, "episode0_sdf_max_rounds", 10)))
        eval_batches = int(getattr(self.hyperparams, "sdf_fc1_eval_max_batches", 0))
        eval_batches_arg = eval_batches if eval_batches > 0 else None

        rounds: List[Dict[str, Any]] = []
        for round_idx in range(max_rounds):
            display_round = round_idx + 1
            prev_epoch_offset = getattr(self, "_run_batches_epoch_offset", 0)
            self._run_batches_epoch_offset = round_idx * epochs_per_round
            try:
                train_summary = self._run_batches(
                    train_batches,
                    epochs_per_round,
                    log_interval,
                    ['sdf_fc1'],
                    desc_prefix=f'SDF/FC1(ep0-sdf r{display_round}/{max_rounds}) '
                )
            finally:
                self._run_batches_epoch_offset = prev_epoch_offset
            prefix = f'episode0_sdf_round{display_round}'
            eval_metrics = self._evaluate_sdf_fc1_batches(
                val_batches,
                prefix=prefix,
                max_batches=eval_batches_arg
            )
            passed, gate_diag = self._episode0_sdf_safety_gate_passed(
                eval_metrics,
                prefix=prefix,
            )
            gate_diag.update({
                'round': display_round,
                'max_rounds': max_rounds,
                'epochs_per_round': epochs_per_round,
                'total_bootstrap_epochs': display_round * epochs_per_round,
                'sdf_acceptance_gate_applied': True,
            })
            round_record = {
                'round': display_round,
                'train_summary': train_summary,
                'eval_metrics': eval_metrics,
                'gate': gate_diag,
            }
            rounds.append(round_record)
            if passed:
                return {
                    'active': True,
                    'sdf_acceptance_gate_applied': True,
                    'passed': True,
                    'rounds_completed': display_round,
                    'epochs_per_round': epochs_per_round,
                    'total_bootstrap_epochs': display_round * epochs_per_round,
                    'max_rounds': max_rounds,
                    'final_train_summary': train_summary,
                    'final_eval_metrics': eval_metrics,
                    'final_gate': gate_diag,
                    'rounds': rounds,
                }
            logger.warning("Episode 0 SDF bootstrap gate failed at round %s/%s: %s", display_round, max_rounds, gate_diag)

        final_record = rounds[-1]
        return {
            'active': True,
            'sdf_acceptance_gate_applied': True,
            'passed': False,
            'rounds_completed': max_rounds,
            'epochs_per_round': epochs_per_round,
            'total_bootstrap_epochs': max_rounds * epochs_per_round,
            'max_rounds': max_rounds,
            'final_train_summary': final_record['train_summary'],
            'final_eval_metrics': final_record['eval_metrics'],
            'final_gate': final_record['gate'],
            'rounds': rounds,
            'reason': 'episode0_sdf_gate_failed_after_max_rounds',
        }

    def run_episode(
        self,
        n_epochs: int = 10,
        batch_size: int = 256,
        log_interval: int = 100,
        n_samples: int = 10000,
        n_paths: int = 100,
        group_size: int = 100,
        n_branches: int = 2,
        train_mode: str = '2time',
        train_modules: Optional[List[str]] = None,
        simulate_kwargs: Optional[Dict] = None,
        episode_mode: Optional[str] = None
    ) -> Dict:
        """
        按 Episode 逻辑执行训练（三模式）：
        - mode0: Sample 训 SDF/PV，再 SimulateTS(h=1) 训 SDF 二阶段（可选 FC2）
        - modeA: Sample 训 PV，再 SimulateTS(h=1) 训 SDF 二阶段（可选 FC2）
        - modeB: SimulateTS(h=T) 直接训练 PV/SDF（可选 FC2）
        """
        simulate_kwargs = dict(simulate_kwargs or {})
        train_modules = train_modules or ['sdf_fc1', 'policy_value', 'fc2']
        self.train_mode = train_mode
        self.add_FC1loss = False
        if int(self.episode_id) == 0:
            self.set_sdf_training_phase(SDFTrainingPhase.EPISODE0_BOOTSTRAP)
        else:
            self.set_sdf_training_phase(SDFTrainingPhase.JOINT_DISABLED)
        self.reset_sdf_shock_bank()

        horizon_mode1 = int(simulate_kwargs.pop('horizon_mode1', 1))
        horizon_modeb = int(simulate_kwargs.pop('horizon', getattr(self.hyperparams, 'simulate_horizon', 20)))
        modeb_resimulate_after_pv = bool(simulate_kwargs.pop('modeb_resimulate_after_pv', False))
        mode = self._resolve_episode_mode(episode_mode)
        tensor_pipeline = self._use_tensor_pipeline()

        module_summaries = {}
        self.tensor_firm = None
        self.tensor_macro = None
        self.tensor_sdf = None
        use_sdf_fc1 = 'sdf_fc1' in train_modules and 'sdf_fc1' in self.models
        use_policy_value = 'policy_value' in train_modules and 'policy_value' in self.models
        use_fc2 = 'fc2' in train_modules and 'fc2' in self.models
        if (
            mode == 'modea'
            and self.episode_id > 0
            and use_sdf_fc1
            and use_policy_value
            and not bool(getattr(self.hyperparams, "allow_modea_sdf_after_pv", False))
        ):
            raise ValueError(
                "Mode A trains Policy/Value before FC1/SDF gate and is disabled for Episode>0 "
                "when both sdf_fc1 and policy_value are active. Use modeb or set "
                "allow_modea_sdf_after_pv=True for legacy experiments."
            )

        # 记录 episode 开始时的 GPU 显存
        logger.info(f"Episode {self.episode_id} starting - GPU Memory:")
        mem_info = self.gpu_monitor.log_memory("episode_start")
        print_memory_summary(mem_info, prefix=f"  [Episode {self.episode_id}] ")

        try:
            if mode == 'mode0':
                module_summaries['episode0_bootstrap_policy_training'] = {
                    'active': bool(self.episode_id == 0 and use_policy_value),
                    'sdf_acceptance_gate_applied': bool(self.episode_id == 0 and use_sdf_fc1),
                    'reason': 'awaiting_episode0_sdf_gate' if self.episode_id == 0 and use_sdf_fc1 else 'no_sdf_bootstrap_gate',
                }
                if use_sdf_fc1 or use_policy_value:
                    sampler = Sample(
                        models=self.models,
                        config=self.config,
                        n_samples=n_samples,
                        n_paths=n_paths,
                        group_size=group_size,
                        branch_num=n_branches
                    )
                else:
                    sampler = None

                if use_sdf_fc1 and sampler is not None:
                    if tensor_pipeline:
                        self.tensor_sdf = sampler.build_sdf_fc1_tensor()
                        self.df_sdf = None
                        train_table, val_table, holdout_diag = self._split_sdf_table_by_path(self.tensor_sdf)
                        sdf_batches = self._create_sdf_batches_from_macro_tensor(
                            train_table, batch_size=batch_size, n_branches=n_branches
                        )
                        sdf_val_batches = self._create_sdf_batches_from_macro_tensor(
                            val_table, batch_size=batch_size, n_branches=n_branches
                        )
                    else:
                        self.df_sdf = sampler.build_sdf_fc1_df()
                        self.tensor_sdf = None
                        train_df, val_df, holdout_diag = self._split_sdf_dataframe_by_path(self.df_sdf)
                        sdf_batches = self._create_sdf_batches_from_macro_df(
                            train_df, batch_size=batch_size, n_branches=n_branches
                        )
                        sdf_val_batches = self._create_sdf_batches_from_macro_df(
                            val_df, batch_size=batch_size, n_branches=n_branches
                        )
                    module_summaries['episode0_sdf_holdout_split'] = holdout_diag
                    if not sdf_batches:
                        raise RuntimeError("Episode 0 SDF bootstrap could not build non-empty training batches.")
                    if sdf_batches:
                        if int(self.episode_id) != 0:
                            raise RuntimeError("EPISODE0_BOOTSTRAP stage1 is only valid for episode_id == 0.")
                        prev_phase = getattr(self, "sdf_training_phase", SDFTrainingPhase.EPISODE0_BOOTSTRAP)
                        self.set_sdf_training_phase(SDFTrainingPhase.EPISODE0_BOOTSTRAP)
                        try:
                            episode0_sdf_gate = self._run_episode0_sdf_bootstrap_until_gate(
                                train_batches=sdf_batches,
                                val_batches=sdf_val_batches,
                                n_epochs=n_epochs,
                                log_interval=log_interval,
                            )
                            module_summaries['episode0_sdf_bootstrap_gate'] = episode0_sdf_gate
                            module_summaries['sdf_fc1_stage1'] = episode0_sdf_gate['final_train_summary']
                            module_summaries['episode0_bootstrap_policy_training'].update({
                                'sdf_acceptance_gate_applied': True,
                                'sdf_acceptance_gate_passed': bool(episode0_sdf_gate.get('passed', False)),
                                'sdf_rounds_completed': episode0_sdf_gate.get('rounds_completed'),
                                'reason': (
                                    'episode0_sdf_gate_passed'
                                    if episode0_sdf_gate.get('passed', False)
                                    else 'episode0_sdf_gate_failed_after_max_rounds'
                                ),
                            })
                            if not episode0_sdf_gate.get('passed', False):
                                total_epochs = episode0_sdf_gate.get('total_bootstrap_epochs')
                                raise NumericalStageFailure(
                                    f"Episode 0 SDF failed after {total_epochs} bootstrap epochs; "
                                    "skip Policy/Value to avoid training on invalid M."
                                )
                        finally:
                            self.set_sdf_training_phase(prev_phase)

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
                            pv_batches, n_epochs, log_interval, ['policy_value'], desc_prefix='Policy/Value '
                        )

                if use_sdf_fc1 or use_fc2:
                    if tensor_pipeline:
                        self._simulate_tensor(
                            n_paths=n_paths,
                            group_size=group_size,
                            n_branches=n_branches,
                            horizon=horizon_mode1,
                            simulate_kwargs=simulate_kwargs,
                            export_df=use_fc2
                        )
                    else:
                        self._simulate_df(
                            n_paths=n_paths,
                            group_size=group_size,
                            n_branches=n_branches,
                            horizon=horizon_mode1,
                            simulate_kwargs=simulate_kwargs
                        )

                if use_fc2:
                    fc2_summary = self._run_fc2_epochs(n_epochs=n_epochs, log_interval=log_interval)
                    if fc2_summary is not None:
                        module_summaries['fc2'] = fc2_summary

                if use_sdf_fc1 and self.episode_id > 0:
                    gate_result = self._run_sdf_recon_from_macro(
                        module_summaries=module_summaries,
                        n_epochs=n_epochs,
                        batch_size=batch_size,
                        log_interval=log_interval,
                        n_branches=n_branches
                    )
                    if not gate_result.get("passed", False):
                        raise NumericalStageFailure(
                            f"FC1/SDF validation failed in {gate_result.get('failed_stage')}; skip downstream training."
                        )

            elif mode == 'modea':
                if use_policy_value:
                    sampler = Sample(
                        models=self.models,
                        config=self.config,
                        n_samples=n_samples,
                        n_paths=n_paths,
                        group_size=group_size,
                        branch_num=n_branches
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
                            pv_batches, n_epochs, log_interval, ['policy_value'], desc_prefix='Policy/Value '
                        )

                if use_sdf_fc1 or use_fc2:
                    if tensor_pipeline:
                        self._simulate_tensor(
                            n_paths=n_paths,
                            group_size=group_size,
                            n_branches=n_branches,
                            horizon=horizon_mode1,
                            simulate_kwargs=simulate_kwargs,
                            export_df=use_fc2
                        )
                    else:
                        self._simulate_df(
                            n_paths=n_paths,
                            group_size=group_size,
                            n_branches=n_branches,
                            horizon=horizon_mode1,
                            simulate_kwargs=simulate_kwargs
                        )

                if use_fc2:
                    fc2_summary = self._run_fc2_epochs(n_epochs=n_epochs, log_interval=log_interval)
                    if fc2_summary is not None:
                        module_summaries['fc2'] = fc2_summary

                if use_sdf_fc1 and self.episode_id > 0:
                    gate_result = self._run_sdf_recon_from_macro(
                        module_summaries=module_summaries,
                        n_epochs=n_epochs,
                        batch_size=batch_size,
                        log_interval=log_interval,
                        n_branches=n_branches
                    )
                    if not gate_result.get("passed", False):
                        raise NumericalStageFailure(
                            f"FC1/SDF validation failed in {gate_result.get('failed_stage')}; skip downstream training."
                        )

            elif mode == 'modeb':
                modeb_rng_before_first_sim = self._capture_rng_state()
                if tensor_pipeline:
                    self._simulate_tensor(
                        n_paths=n_paths,
                        group_size=group_size,
                        n_branches=n_branches,
                        horizon=horizon_modeb,
                        simulate_kwargs=simulate_kwargs,
                        export_df=use_fc2
                    )
                else:
                    self._simulate_df(
                        n_paths=n_paths,
                        group_size=group_size,
                        n_branches=n_branches,
                        horizon=horizon_modeb,
                        simulate_kwargs=simulate_kwargs
                    )

                macro_key_columns = ['path', 't', 'branch']
                macro_diag_columns = ['Hatc', 'LnK', 'hatcf', 'lnkf', 'M', 'n_firms', 'K', 'C']
                firm_key_columns = ['path', 't', 'branch', 'ID']
                firm_diag_columns = ['bp', 'Bar_i', 'Bar_z', 'entry', 'b', 'z', 'K', 'P', 'Q', 'M', 'Hatcf', 'LnKF']
                old_macro_snapshot = self._selected_frame_snapshot(
                    self.tensor_macro,
                    self.df_macro,
                    macro_diag_columns,
                    key_columns=macro_key_columns
                )
                old_firm_snapshot = self._selected_frame_snapshot(
                    self.tensor_firm,
                    self.df,
                    firm_diag_columns,
                    key_columns=firm_key_columns
                )
                old_firm_stats = self._snapshot_stats('old_firm', old_firm_snapshot, firm_diag_columns)
                old_firm_stats.update(self._firm_economic_moments('old_firm', old_firm_snapshot))
                old_firm_keys = old_firm_snapshot.loc[
                    :,
                    [k for k in firm_key_columns if k in old_firm_snapshot.columns]
                ].copy() if not old_firm_snapshot.empty else pd.DataFrame()
                modeb_old_diag: Dict[str, Any] = {
                    'resimulate_after_pv': bool(modeb_resimulate_after_pv),
                    'macro_r2': (
                        self._macro_forecast_r2_tensor(self.tensor_macro)
                        if tensor_pipeline and self.tensor_macro is not None
                        else self._macro_forecast_r2(self.df_macro)
                    ),
                }
                modeb_old_diag.update(self._snapshot_stats('old_macro', old_macro_snapshot, macro_diag_columns))
                modeb_old_diag.update(old_firm_stats)
                module_summaries['modeb_pre_pv_simulation_diag'] = modeb_old_diag

                if use_sdf_fc1 and self.episode_id > 0:
                    gate_result = self._run_sdf_recon_from_macro(
                        module_summaries=module_summaries,
                        n_epochs=n_epochs,
                        batch_size=batch_size,
                        log_interval=log_interval,
                        n_branches=n_branches
                    )
                    if not gate_result.get("passed", False):
                        raise NumericalStageFailure(
                            f"FC1/SDF validation failed in {gate_result.get('failed_stage')}; skip Q/P/bp."
                        )

                if use_policy_value and use_sdf_fc1 and self.episode_id > 0:
                    rng_after_sdf_training = self._capture_rng_state()
                    self._restore_rng_state(modeb_rng_before_first_sim)
                    if tensor_pipeline:
                        self._simulate_tensor(
                            n_paths=n_paths,
                            group_size=group_size,
                            n_branches=n_branches,
                            horizon=horizon_modeb,
                            simulate_kwargs=simulate_kwargs,
                            export_df=use_fc2
                        )
                    else:
                        self._simulate_df(
                            n_paths=n_paths,
                            group_size=group_size,
                            n_branches=n_branches,
                            horizon=horizon_modeb,
                            simulate_kwargs=simulate_kwargs
                        )
                    self._restore_rng_state(rng_after_sdf_training)
                    refreshed_macro_snapshot = self._selected_frame_snapshot(
                        self.tensor_macro,
                        self.df_macro,
                        macro_diag_columns,
                        key_columns=macro_key_columns
                    )
                    refreshed_firm_snapshot = self._selected_frame_snapshot(
                        self.tensor_firm,
                        self.df,
                        firm_diag_columns,
                        key_columns=firm_key_columns
                    )
                    refreshed_firm_stats = self._snapshot_stats(
                        'refreshed_firm',
                        refreshed_firm_snapshot,
                        firm_diag_columns
                    )
                    refreshed_firm_stats.update(
                        self._firm_economic_moments('refreshed_firm', refreshed_firm_snapshot)
                    )
                    module_summaries['modeb_pre_pv_sdf_refresh_diag'] = {
                        'resimulated_after_sdf_gate': True,
                        'rng_state_replayed': True,
                        'policy_value_uses_refreshed_sdf_data': True,
                        **self._snapshot_stats('refreshed_macro', refreshed_macro_snapshot, macro_diag_columns),
                        **refreshed_firm_stats,
                        **self._keyed_snapshot_gap(
                            'macro_old_to_sdf_refreshed',
                            old_macro_snapshot,
                            refreshed_macro_snapshot,
                            key_columns=macro_key_columns,
                            value_columns=macro_diag_columns
                        ),
                    }
                    refreshed_firm_keys = refreshed_firm_snapshot.loc[
                        :,
                        [k for k in firm_key_columns if k in refreshed_firm_snapshot.columns]
                    ].copy() if not refreshed_firm_snapshot.empty else pd.DataFrame()
                    module_summaries['modeb_pre_pv_sdf_refresh_diag'].update(
                        self._keyed_snapshot_gap(
                            'firm_old_to_sdf_refreshed_keys',
                            old_firm_keys,
                            refreshed_firm_keys,
                            key_columns=[
                                k for k in firm_key_columns
                                if k in old_firm_keys.columns and k in refreshed_firm_keys.columns
                            ],
                            value_columns=[]
                        )
                    )
                    module_summaries['modeb_pre_pv_sdf_refresh_diag'].update(
                        self._keyed_snapshot_gap(
                            'firm_old_to_sdf_refreshed',
                            old_firm_snapshot,
                            refreshed_firm_snapshot,
                            key_columns=[
                                k for k in firm_key_columns
                                if k in old_firm_snapshot.columns and k in refreshed_firm_snapshot.columns
                            ],
                            value_columns=['M', 'Hatcf', 'LnKF']
                        )
                    )
                    if bool(getattr(self.hyperparams, "sdf_post_refresh_gate_enabled", True)):
                        post_refresh_gate = self._evaluate_post_refresh_sdf_gate(
                            module_summaries=module_summaries,
                            batch_size=batch_size,
                            n_branches=n_branches,
                        )
                        if not post_refresh_gate.get("passed", False):
                            raise NumericalStageFailure(
                                "Post-refresh FC1/SDF gate failed; skip Q/P/bp."
                            )
                else:
                    module_summaries['modeb_pre_pv_sdf_refresh_diag'] = {
                        'resimulated_after_sdf_gate': False,
                        'rng_state_replayed': False,
                        'policy_value_uses_refreshed_sdf_data': bool(not use_sdf_fc1),
                    }

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
                            pv_batches, n_epochs, log_interval, ['policy_value'], desc_prefix='Policy/Value '
                        )

                if modeb_resimulate_after_pv and use_policy_value:
                    rng_after_pv_training = self._capture_rng_state()
                    self._restore_rng_state(modeb_rng_before_first_sim)
                    if tensor_pipeline:
                        self._simulate_tensor(
                            n_paths=n_paths,
                            group_size=group_size,
                            n_branches=n_branches,
                            horizon=horizon_modeb,
                            simulate_kwargs=simulate_kwargs,
                            export_df=use_fc2
                        )
                    else:
                        self._simulate_df(
                            n_paths=n_paths,
                            group_size=group_size,
                            n_branches=n_branches,
                            horizon=horizon_modeb,
                            simulate_kwargs=simulate_kwargs
                        )
                    self._restore_rng_state(rng_after_pv_training)
                    new_macro_snapshot = self._selected_frame_snapshot(
                        self.tensor_macro,
                        self.df_macro,
                        macro_diag_columns,
                        key_columns=macro_key_columns
                    )
                    new_firm_snapshot = self._selected_frame_snapshot(
                        self.tensor_firm,
                        self.df,
                        firm_diag_columns,
                        key_columns=firm_key_columns
                    )
                    new_firm_stats = self._snapshot_stats('new_firm', new_firm_snapshot, firm_diag_columns)
                    new_firm_stats.update(self._firm_economic_moments('new_firm', new_firm_snapshot))
                    new_firm_keys = new_firm_snapshot.loc[
                        :,
                        [k for k in firm_key_columns if k in new_firm_snapshot.columns]
                    ].copy() if not new_firm_snapshot.empty else pd.DataFrame()
                    modeb_refresh_diag: Dict[str, Any] = {
                        'resimulated_after_pv': True,
                        'rng_state_replayed': True,
                        'macro_r2': (
                            self._macro_forecast_r2_tensor(self.tensor_macro)
                            if tensor_pipeline and self.tensor_macro is not None
                            else self._macro_forecast_r2(self.df_macro)
                        ),
                    }
                    modeb_refresh_diag.update(self._snapshot_stats('new_macro', new_macro_snapshot, macro_diag_columns))
                    modeb_refresh_diag.update(new_firm_stats)
                    modeb_refresh_diag.update(
                        self._keyed_snapshot_gap(
                            'macro_old_to_new',
                            old_macro_snapshot,
                            new_macro_snapshot,
                            key_columns=macro_key_columns,
                            value_columns=macro_diag_columns
                        )
                    )
                    modeb_refresh_diag.update(
                        self._keyed_snapshot_gap(
                            'firm_old_to_new_keys',
                            old_firm_keys,
                            new_firm_keys,
                            key_columns=[k for k in firm_key_columns if k in old_firm_keys.columns and k in new_firm_keys.columns],
                            value_columns=[]
                        )
                    )
                    modeb_refresh_diag.update(
                        self._prefixed_delta(
                            'firm_new_minus_old',
                            old_firm_stats,
                            new_firm_stats,
                            old_prefix='old_firm',
                            new_prefix='new_firm'
                        )
                    )
                    module_summaries['modeb_post_pv_resimulation_diag'] = modeb_refresh_diag
                else:
                    module_summaries['modeb_post_pv_resimulation_diag'] = {
                        'resimulated_after_pv': False,
                        'rng_state_replayed': False,
                    }

                if use_sdf_fc1 and self.episode_id <= 0:
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
                    fc2_summary = self._run_fc2_epochs(n_epochs=n_epochs, log_interval=log_interval)
                    if fc2_summary is not None:
                        module_summaries['fc2'] = fc2_summary

                if tensor_pipeline and self.tensor_macro is not None:
                    macro_r2_diag = self._macro_forecast_r2_tensor(self.tensor_macro)
                else:
                    macro_df = self.df_macro
                    if (macro_df is None or macro_df.empty) and self.tensor_macro is not None:
                        macro_df = self._table_to_dataframe(self.tensor_macro)
                    macro_r2_diag = self._macro_forecast_r2(macro_df)
                if macro_r2_diag:
                    module_summaries['macro_diag_modeb'] = macro_r2_diag
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
            avg_losses, epoch_metadata = self._aggregate_metric_records(epoch_losses)
            logger.info(f"Epoch {epoch+1} finished: { {**avg_losses, **epoch_metadata} }")
        convergence = None
        if 'policy_value' in train_modules and 'policy_value' in self.models and batches:
            convergence = self.evaluate_bellman_convergence(batches)
        
        summary = {
            'episode_id': self.episode_id,
            'total_steps': self.step_count,
            'final_losses': avg_losses,
            'loss_history': self.loss_history
        }
        if epoch_metadata:
            summary['metadata'] = epoch_metadata
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
