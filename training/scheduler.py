"""
学习率和损失权重调度器
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Callable


class LossWeightScheduler:
    """
    损失权重调度器
    
    支持多种调度策略：
    - 固定权重
    - 线性增长/衰减
    - 阶梯式变化
    - 自适应调整（基于损失值）
    """
    
    def __init__(
        self,
        initial_weights: Dict[str, float],
        schedule_type: str = 'fixed',
        warmup_steps: int = 0,
        total_steps: int = 10000,
        target_weights: Optional[Dict[str, float]] = None,
        milestone_steps: Optional[List[int]] = None,
        milestone_weights: Optional[List[Dict[str, float]]] = None
    ):
        """
        Args:
            initial_weights: 初始权重字典
            schedule_type: 调度类型 ('fixed', 'linear', 'step', 'adaptive')
            warmup_steps: 预热步数（在此期间线性增长到初始值）
            total_steps: 总步数
            target_weights: 目标权重（用于 'linear' 类型）
            milestone_steps: 里程碑步数列表（用于 'step' 类型）
            milestone_weights: 里程碑权重列表（用于 'step' 类型）
        """
        self.initial_weights = initial_weights.copy()
        self.current_weights = initial_weights.copy()
        self.schedule_type = schedule_type
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.target_weights = target_weights or initial_weights.copy()
        self.milestone_steps = milestone_steps or []
        self.milestone_weights = milestone_weights or []
        
        self.current_step = 0
        
        # 用于自适应调度
        self.loss_history = {k: [] for k in initial_weights}
        self.adaptive_window = 100
    
    def step(self, losses: Optional[Dict[str, float]] = None):
        """
        更新一步
        
        Args:
            losses: 当前的损失值字典（用于自适应调度）
        """
        self.current_step += 1
        
        if self.schedule_type == 'fixed':
            pass
        elif self.schedule_type == 'linear':
            self._linear_step()
        elif self.schedule_type == 'step':
            self._step_step()
        elif self.schedule_type == 'adaptive':
            if losses is not None:
                self._adaptive_step(losses)
    
    def _linear_step(self):
        """
        线性调度
        """
        if self.current_step <= self.warmup_steps:
            # 预热阶段
            alpha = self.current_step / max(1, self.warmup_steps)
            for k in self.current_weights:
                self.current_weights[k] = self.initial_weights[k] * alpha
        else:
            # 线性插值到目标
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            
            for k in self.current_weights:
                self.current_weights[k] = (
                    self.initial_weights[k] * (1 - progress) +
                    self.target_weights[k] * progress
                )
    
    def _step_step(self):
        """
        阶梯式调度
        """
        # 找到当前应该使用的里程碑
        for i, milestone in enumerate(self.milestone_steps):
            if self.current_step >= milestone:
                if i < len(self.milestone_weights):
                    self.current_weights = self.milestone_weights[i].copy()
    
    def _adaptive_step(self, losses: Dict[str, float]):
        """
        自适应调度
        
        根据损失值动态调整权重，使各损失项贡献平衡
        """
        # 记录损失历史
        for k, v in losses.items():
            if k in self.loss_history:
                self.loss_history[k].append(v)
                if len(self.loss_history[k]) > self.adaptive_window:
                    self.loss_history[k] = self.loss_history[k][-self.adaptive_window:]
        
        # 计算平均损失
        avg_losses = {}
        for k, history in self.loss_history.items():
            if len(history) > 0:
                avg_losses[k] = np.mean(history)
        
        if len(avg_losses) == 0:
            return
        
        # 归一化权重使得各损失项贡献相近
        total_loss = sum(avg_losses.values())
        if total_loss > 0:
            for k in self.current_weights:
                if k in avg_losses and avg_losses[k] > 0:
                    # 损失越大，权重越小
                    self.current_weights[k] = self.initial_weights[k] * total_loss / avg_losses[k]
        
        # 归一化权重和
        weight_sum = sum(self.current_weights.values())
        initial_sum = sum(self.initial_weights.values())
        
        if weight_sum > 0:
            for k in self.current_weights:
                self.current_weights[k] *= initial_sum / weight_sum
    
    def get_weights(self) -> Dict[str, float]:
        """
        获取当前权重
        """
        return self.current_weights.copy()
    
    def __getitem__(self, key: str) -> float:
        """
        支持下标访问
        """
        return self.current_weights.get(key, 0.0)
    
    def state_dict(self) -> Dict:
        """
        保存状态
        """
        return {
            'current_step': self.current_step,
            'current_weights': self.current_weights.copy()
        }
    
    def load_state_dict(self, state: Dict):
        """
        加载状态
        """
        self.current_step = state['current_step']
        self.current_weights = state['current_weights'].copy()


class LearningRateScheduler:
    """
    学习率调度器包装器
    
    支持预热和多种衰减策略
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        warmup_steps: int = 0,
        decay_type: str = 'cosine',
        total_steps: int = 10000,
        min_lr: float = 1e-6,
        decay_rate: float = 0.1,
        decay_steps: Optional[List[int]] = None
    ):
        """
        Args:
            optimizer: 优化器
            base_lr: 基础学习率
            warmup_steps: 预热步数
            decay_type: 衰减类型 ('cosine', 'linear', 'step', 'exponential')
            total_steps: 总步数
            min_lr: 最小学习率
            decay_rate: 衰减率（用于 step 和 exponential）
            decay_steps: 衰减步数列表（用于 step）
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.decay_type = decay_type
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps or []
        
        self.current_step = 0
        self.current_lr = base_lr
    
    def step(self):
        """
        更新学习率
        """
        self.current_step += 1
        self.current_lr = self._compute_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
    
    def _compute_lr(self) -> float:
        """
        计算当前学习率
        """
        if self.current_step <= self.warmup_steps:
            # 线性预热
            return self.base_lr * self.current_step / max(1, self.warmup_steps)
        
        # 衰减阶段
        progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        
        if self.decay_type == 'cosine':
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        elif self.decay_type == 'linear':
            lr = self.base_lr * (1 - progress) + self.min_lr * progress
        
        elif self.decay_type == 'step':
            lr = self.base_lr
            for step in self.decay_steps:
                if self.current_step >= step:
                    lr *= self.decay_rate
            lr = max(lr, self.min_lr)
        
        elif self.decay_type == 'exponential':
            lr = self.base_lr * (self.decay_rate ** progress)
            lr = max(lr, self.min_lr)
        
        else:
            lr = self.base_lr
        
        return lr
    
    def get_lr(self) -> float:
        """
        获取当前学习率
        """
        return self.current_lr
    
    def state_dict(self) -> Dict:
        """
        保存状态
        """
        return {
            'current_step': self.current_step,
            'current_lr': self.current_lr
        }
    
    def load_state_dict(self, state: Dict):
        """
        加载状态
        """
        self.current_step = state['current_step']
        self.current_lr = state['current_lr']


class EpisodeScheduler:
    """
    Episode 级别的调度器
    
    管理 Episode 之间的参数变化
    """
    
    def __init__(
        self,
        episode_configs: List[Dict],
        default_config: Optional[Dict] = None
    ):
        """
        Args:
            episode_configs: 每个 episode 的配置列表
            default_config: 默认配置
        """
        self.episode_configs = episode_configs
        self.default_config = default_config or {}
        self.current_episode = 0
    
    def get_config(self, episode: Optional[int] = None) -> Dict:
        """
        获取指定 episode 的配置
        """
        if episode is None:
            episode = self.current_episode
        
        if episode < len(self.episode_configs):
            config = self.default_config.copy()
            config.update(self.episode_configs[episode])
            return config
        
        return self.default_config.copy()
    
    def step(self):
        """
        推进到下一个 episode
        """
        self.current_episode += 1
    
    def reset(self):
        """
        重置
        """
        self.current_episode = 0


def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    创建带预热的余弦退火调度器
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
