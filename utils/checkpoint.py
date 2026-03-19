"""
检查点管理
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, List
import json
from datetime import datetime
import shutil


class CheckpointManager:
    """
    检查点管理器
    
    功能：
    - 保存/加载模型和优化器状态
    - 管理多个检查点
    - 自动清理旧检查点
    """
    
    def __init__(
        self,
        save_dir: str = './checkpoints',
        max_keep: int = 5,
        save_best_only: bool = False
    ):
        """
        Args:
            save_dir: 保存目录
            max_keep: 最大保留数量
            save_best_only: 是否只保存最佳模型
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_keep = max_keep
        self.save_best_only = save_best_only
        
        self.best_metric = float('inf')
        self.checkpoints = []
        
        # 加载已有检查点信息
        self._load_checkpoint_info()
    
    def _load_checkpoint_info(self):
        """
        加载已有检查点信息
        """
        info_file = self.save_dir / 'checkpoints.json'
        
        if info_file.exists():
            with open(info_file, 'r') as f:
                data = json.load(f)
                self.checkpoints = data.get('checkpoints', [])
                self.best_metric = data.get('best_metric', float('inf'))
    
    def _save_checkpoint_info(self):
        """
        保存检查点信息
        """
        info_file = self.save_dir / 'checkpoints.json'
        
        with open(info_file, 'w') as f:
            json.dump({
                'checkpoints': self.checkpoints,
                'best_metric': self.best_metric
            }, f, indent=2)
    
    def save(
        self,
        models: Dict[str, nn.Module],
        optimizers: Dict[str, torch.optim.Optimizer] = None,
        epoch: int = 0,
        step: int = 0,
        metric: float = None,
        extra: dict = None,
        name: str = None
    ) -> Optional[Path]:
        """
        保存检查点
        
        Args:
            models: 模型字典
            optimizers: 优化器字典
            epoch: 当前 epoch
            step: 当前 step
            metric: 评估指标（用于判断是否是最佳）
            extra: 额外信息
            name: 检查点名称
        
        Returns:
            保存路径，如果不保存则返回 None
        """
        is_best = metric is not None and metric < self.best_metric
        
        # 如果只保存最佳且不是最佳，跳过
        if self.save_best_only and not is_best:
            return None
        
        # 更新最佳指标
        if is_best:
            self.best_metric = metric
        
        # 生成文件名
        if name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = f'checkpoint_epoch{epoch}_step{step}_{timestamp}'
        
        # 构建检查点
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'metric': metric,
            'is_best': is_best,
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'optimizers': {}
        }
        
        # 保存模型状态
        for model_name, model in models.items():
            if model is not None:
                checkpoint['models'][model_name] = model.state_dict()
        
        # 保存优化器状态
        if optimizers:
            for opt_name, opt in optimizers.items():
                if opt is not None:
                    checkpoint['optimizers'][opt_name] = opt.state_dict()
        
        # 保存额外信息
        if extra:
            checkpoint['extra'] = extra
        
        # 保存文件
        path = self.save_dir / f'{name}.pt'
        torch.save(checkpoint, path)
        
        # 更新检查点列表
        self.checkpoints.append({
            'name': name,
            'path': str(path),
            'epoch': epoch,
            'step': step,
            'metric': metric,
            'is_best': is_best
        })
        
        # 如果是最佳，创建软链接
        if is_best:
            best_path = self.save_dir / 'best.pt'
            if best_path.exists():
                best_path.unlink()
            shutil.copy(path, best_path)
        
        # 清理旧检查点
        self._cleanup()
        
        # 保存信息
        self._save_checkpoint_info()
        
        return path
    
    def load(
        self,
        models: Dict[str, nn.Module],
        optimizers: Dict[str, torch.optim.Optimizer] = None,
        name: str = 'best',
        device: torch.device = None
    ) -> Dict:
        """
        加载检查点
        
        Args:
            models: 模型字典
            optimizers: 优化器字典
            name: 检查点名称或 'best' / 'latest'
            device: 设备
        
        Returns:
            检查点信息
        """
        # 确定路径
        if name == 'best':
            path = self.save_dir / 'best.pt'
        elif name == 'latest':
            if len(self.checkpoints) == 0:
                raise FileNotFoundError("No checkpoints found")
            path = Path(self.checkpoints[-1]['path'])
        else:
            path = self.save_dir / f'{name}.pt'
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        # 加载
        checkpoint = torch.load(path, map_location=device)
        
        # 恢复模型
        for model_name, state_dict in checkpoint['models'].items():
            if model_name in models and models[model_name] is not None:
                models[model_name].load_state_dict(state_dict)
        
        # 恢复优化器
        if optimizers and 'optimizers' in checkpoint:
            for opt_name, state_dict in checkpoint['optimizers'].items():
                if opt_name in optimizers and optimizers[opt_name] is not None:
                    optimizers[opt_name].load_state_dict(state_dict)
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'step': checkpoint.get('step', 0),
            'metric': checkpoint.get('metric'),
            'extra': checkpoint.get('extra', {})
        }
    
    def _cleanup(self):
        """
        清理旧检查点
        """
        # 保留最佳和最近的 max_keep 个
        non_best = [c for c in self.checkpoints if not c.get('is_best', False)]
        
        while len(non_best) > self.max_keep - 1:  # -1 为最佳保留位置
            oldest = non_best.pop(0)
            path = Path(oldest['path'])
            if path.exists():
                path.unlink()
            self.checkpoints.remove(oldest)
    
    def list_checkpoints(self) -> List[Dict]:
        """
        列出所有检查点
        """
        return self.checkpoints.copy()
    
    def get_best(self) -> Optional[Dict]:
        """
        获取最佳检查点信息
        """
        for c in reversed(self.checkpoints):
            if c.get('is_best', False):
                return c
        return None
