"""
Trainer 类：完整训练流程管理

管理多个 Episode 的训练、模型保存、日志记录
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import logging
import json
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy

import sys
sys.path.append('..')
from config import Config, HyperParams

from .episode import Episode
from .scheduler import EpisodeScheduler
from .target_utils import hard_update


logger = logging.getLogger(__name__)


class Trainer:
    """
    完整训练流程管理器
    
    支持：
    - 多 Episode 训练
    - 分阶段训练（先 SDF/FC1，后 Policy/Value）
    - 模型保存/加载
    - 日志记录和可视化
    """
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        config: type = Config,
        hyperparams: HyperParams = None,
        save_dir: str = './checkpoints',
        log_dir: str = './logs',
        device: torch.device = None
    ):
        """
        Args:
            models: 模型字典
            config: 配置类
            hyperparams: 超参数
            save_dir: 模型保存目录
            log_dir: 日志目录
            device: 设备
        """
        self.models = models
        self.config = config
        self.hyperparams = hyperparams or HyperParams()
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        self.device = device or config.DEVICE
        
        # 创建目录
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 优化器
        self.optimizers = self._init_optimizers()
        self.firm_target = self._init_firm_target()
        
        # 训练状态
        self.current_episode = 0
        self.total_steps = 0
        self.best_loss = float('inf')
        self.history = []
        
        # 设置日志
        self._setup_logging()

    def _init_firm_target(self) -> Optional[nn.Module]:
        """
        Create a frozen policy/value target network that is not added to optimizers.
        """
        online = self.models.get('policy_value')
        if online is None:
            return None
        target = deepcopy(online).to(self.device)
        hard_update(target, online)
        target.eval()
        target.requires_grad_(False)
        return target
    
    def _init_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """
        初始化优化器
        """
        optimizers = {}
        
        for name, model in self.models.items():
            if model is not None:
                optimizers[name] = torch.optim.AdamW(
                    model.parameters(),
                    lr=self.hyperparams.lr,
                    weight_decay=self.hyperparams.weight_decay,
                    betas=(self.hyperparams.beta1, self.hyperparams.beta2)
                )
        
        return optimizers
    
    def _setup_logging(self):
        """
        设置日志记录
        """
        log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
    
    def train(
        self,
        n_episodes: int = 10,
        epochs_per_episode: int = 10,
        batch_size: int = 256,
        data_mode: str = 'sample',
        n_samples: int = 10000,
        train_schedule: Optional[List[Dict]] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict:
        """
        完整训练流程
        
        Args:
            n_episodes: Episode 数量
            epochs_per_episode: 每个 Episode 的 epoch 数
            batch_size: 批大小
            data_mode: 数据生成模式
            n_samples: 每个 Episode 的样本数
            train_schedule: 训练调度（每个 episode 训练哪些模块）
            callbacks: 回调函数列表
        
        Returns:
            summary: 训练摘要
        """
        logger.info(f"Starting training: {n_episodes} episodes")
        
        # 默认训练调度
        if train_schedule is None:
            train_schedule = [
                {'modules': ['sdf_fc1', 'policy_value']} for _ in range(n_episodes)
            ]
        
        callbacks = callbacks or []
        # comment: 这里的逻辑不对，在第0个episode的时候，要做的事首先生成宏观模拟数据，然后训练 sdf_fc1，使用sdfloss，然后使用训练好的sdf_fc1 和 微观模拟数据去训练policy_value，使用policy_value loss 
        for ep in range(n_episodes):
            logger.info(f"\n{'='*50}")
            logger.info(f"Episode {ep + 1}/{n_episodes}")
            logger.info(f"{'='*50}")
            
            # 获取本 episode 的训练模块
            ep_config = train_schedule[min(ep, len(train_schedule) - 1)]
            train_modules = ep_config.get('modules', ['sdf_fc1', 'policy_value'])
            
            # 创建 Episode
            episode = Episode(
                models=self.models,
                optimizers=self.optimizers,
                config=self.config,
                hyperparams=self.hyperparams,
                device=self.device,
                episode_id=ep,
                firm_target=self.firm_target
            )
            
            # 生成数据
            logger.info("Generating data...")
            episode.generate_data(mode=data_mode, n_samples=n_samples)
            
            # FC1 填充
            if 'sdf_fc1' in self.models:
                logger.info("Filling FC1...")
                episode.fill_fc1()
            
            # Policy/Value 填充
            if 'policy_value' in self.models:
                logger.info("Filling Policy/Value...")
                episode.fill_policy_value()
            
            # 训练
            summary = episode.run(
                n_epochs=epochs_per_episode,
                batch_size=batch_size,
                train_modules=train_modules
            )
            
            # 记录历史
            self.history.append(summary)
            self.total_steps += summary['total_steps']
            self.current_episode = ep + 1
            
            # 检查是否是最佳模型
            current_loss = summary['final_losses'].get('total', float('inf'))
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_checkpoint('best')
                logger.info(f"New best model saved: loss={current_loss:.6f}")
            
            # 定期保存
            if (ep + 1) % 5 == 0:
                self.save_checkpoint(f'episode_{ep + 1}')
            
            # 回调
            for callback in callbacks:
                callback(self, episode, summary)

            # Bellman 残差收敛后可提前停止 episode 循环
            convergence = summary.get('convergence') if isinstance(summary, dict) else None
            if (
                getattr(self.hyperparams, "episode_stop_on_bellman_convergence", True)
                and isinstance(convergence, dict)
                and bool(convergence.get('passed', False))
            ):
                logger.info(
                    "Early stop episodes at %d due to Bellman convergence: %s",
                    ep + 1,
                    convergence
                )
                break
        
        # 保存最终模型
        self.save_checkpoint('final')
        
        # 保存历史
        self.save_history()
        
        return {
            'total_episodes': n_episodes,
            'total_steps': self.total_steps,
            'best_loss': self.best_loss,
            'history': self.history
        }
    
    def train_staged(
        self,
        stage1_episodes: int = 5,
        stage2_episodes: int = 5,
        **kwargs
    ) -> Dict:
        """
        分阶段训练
        
        Stage 1: 训练 SDF/FC1
        Stage 2: 训练 Policy/Value（固定 SDF/FC1）
        """
        logger.info("Starting staged training")
        
        # Stage 1
        logger.info("\n" + "="*60)
        logger.info("STAGE 1: Training SDF/FC1")
        logger.info("="*60)
        
        schedule1 = [{'modules': ['sdf_fc1']} for _ in range(stage1_episodes)]
        self.train(n_episodes=stage1_episodes, train_schedule=schedule1, **kwargs)
        
        # 冻结 SDF/FC1
        if 'sdf_fc1' in self.models:
            for param in self.models['sdf_fc1'].parameters():
                param.requires_grad = False
        
        # Stage 2
        logger.info("\n" + "="*60)
        logger.info("STAGE 2: Training Policy/Value")
        logger.info("="*60)
        
        schedule2 = [{'modules': ['policy_value']} for _ in range(stage2_episodes)]
        summary = self.train(n_episodes=stage2_episodes, train_schedule=schedule2, **kwargs)
        
        # 解冻
        if 'sdf_fc1' in self.models:
            for param in self.models['sdf_fc1'].parameters():
                param.requires_grad = True
        
        return summary
    
    def train_alternating(
        self,
        n_episodes: int = 10,
        **kwargs
    ) -> Dict:
        """
        交替训练
        
        Episode 奇数: SDF/FC1
        Episode 偶数: Policy/Value
        """
        schedule = []
        for i in range(n_episodes):
            if i % 2 == 0:
                schedule.append({'modules': ['sdf_fc1']})
            else:
                schedule.append({'modules': ['policy_value']})
        
        return self.train(n_episodes=n_episodes, train_schedule=schedule, **kwargs)
    
    def save_checkpoint(self, name: str):
        """
        保存检查点
        """
        checkpoint = {
            'models': {},
            'optimizers': {},
            'current_episode': self.current_episode,
            'total_steps': self.total_steps,
            'best_loss': self.best_loss,
            'hyperparams': self.hyperparams.__dict__
        }
        
        for model_name, model in self.models.items():
            if model is not None:
                checkpoint['models'][model_name] = model.state_dict()
        if self.firm_target is not None:
            checkpoint['models']['firm_target'] = self.firm_target.state_dict()
        
        for opt_name, opt in self.optimizers.items():
            checkpoint['optimizers'][opt_name] = opt.state_dict()
        
        path = self.save_dir / f'{name}.pt'
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, name: str):
        """
        加载检查点
        """
        path = self.save_dir / f'{name}.pt'
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        for model_name, state_dict in checkpoint['models'].items():
            if model_name in self.models and self.models[model_name] is not None:
                self.models[model_name].load_state_dict(state_dict)
            elif model_name == 'firm_target' and self.firm_target is not None:
                self.firm_target.load_state_dict(state_dict)

        if self.firm_target is not None and 'firm_target' not in checkpoint.get('models', {}):
            hard_update(self.firm_target, self.models['policy_value'])
        
        for opt_name, state_dict in checkpoint['optimizers'].items():
            if opt_name in self.optimizers:
                self.optimizers[opt_name].load_state_dict(state_dict)
        
        self.current_episode = checkpoint.get('current_episode', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        logger.info(f"Checkpoint loaded: {path}")
    
    def save_history(self):
        """
        保存训练历史
        """
        def _to_serializable(obj):
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_serializable(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_to_serializable(v) for v in obj)
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            return obj

        # 转换为可序列化格式
        history_serializable = []
        for h in self.history:
            h_copy = h.copy()
            if 'loss_history' in h_copy:
                h_copy['loss_history'] = {
                    k: [float(v) for v in vals]
                    for k, vals in h_copy['loss_history'].items()
                }
            if 'final_losses' in h_copy:
                h_copy['final_losses'] = {
                    k: float(v) for k, v in h_copy['final_losses'].items()
                }
            if 'convergence' in h_copy:
                h_copy['convergence'] = _to_serializable(h_copy['convergence'])
            history_serializable.append(h_copy)
        
        path = self.log_dir / 'history.json'
        with open(path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        logger.info(f"History saved: {path}")
    
    def load_history(self):
        """
        加载训练历史
        """
        path = self.log_dir / 'history.json'
        
        if path.exists():
            with open(path, 'r') as f:
                self.history = json.load(f)
        
        return self.history
    
    def evaluate(
        self,
        df: pd.DataFrame = None,
        n_samples: int = 5000
    ) -> Dict:
        """
        评估模型
        """
        logger.info("Evaluating model...")
        
        # 设置为评估模式
        for model in self.models.values():
            if model is not None:
                model.eval()
        
        # 生成评估数据
        if df is None:
            episode = Episode(
                models=self.models,
                optimizers={},
                config=self.config,
                device=self.device
            )
            df = episode.generate_data(mode='sample', n_samples=n_samples)
            episode.fill_fc1()
            episode.fill_policy_value()
        
        # 计算各项指标
        metrics = {}
        
        # 杠杆分布
        b = df['b'].values
        metrics['b_mean'] = float(np.mean(b))
        metrics['b_std'] = float(np.std(b))
        metrics['b_q05'] = float(np.percentile(b, 5))
        metrics['b_q95'] = float(np.percentile(b, 95))
        
        # 如果有 Policy 输出
        if 'P0' in df.columns:
            metrics['P0_mean'] = float(df['P0'].mean())
            metrics['PI_mean'] = float(df['PI'].mean())
            metrics['bar_z_mean'] = float(df.get('Bar_z', pd.Series([0])).mean())
        
        # 资源核算（如果有）
        if 'Y' in df.columns:
            K = df['K'].values
            Y = df['Y'].values
            C = df['C'].values
            
            metrics['Y_K_ratio'] = float(np.mean(Y) / np.mean(K))
            metrics['C_Y_ratio'] = float(np.mean(C) / np.mean(Y))
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def diagnose(self) -> Dict:
        """
        诊断训练状态
        """
        diag = {
            'current_episode': self.current_episode,
            'total_steps': self.total_steps,
            'best_loss': self.best_loss
        }
        
        # 模型参数统计
        for name, model in self.models.items():
            if model is not None:
                n_params = sum(p.numel() for p in model.parameters())
                n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                diag[f'{name}_params'] = n_params
                diag[f'{name}_trainable'] = n_trainable
                
                # 检查 NaN/Inf
                has_nan = any(torch.isnan(p).any() for p in model.parameters())
                has_inf = any(torch.isinf(p).any() for p in model.parameters())
                
                diag[f'{name}_has_nan'] = has_nan
                diag[f'{name}_has_inf'] = has_inf
        
        # 损失趋势
        if len(self.history) > 0:
            losses = [h['final_losses'].get('total', 0) for h in self.history]
            diag['loss_trend'] = 'decreasing' if len(losses) > 1 and losses[-1] < losses[0] else 'increasing'
            diag['loss_first'] = losses[0]
            diag['loss_last'] = losses[-1]
        
        return diag
