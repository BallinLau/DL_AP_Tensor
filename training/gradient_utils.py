"""
梯度工具函数
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


def compute_gradient_norm(
    parameters,
    norm_type: float = 2.0
) -> float:
    """
    计算参数梯度的范数
    
    Args:
        parameters: 模型参数迭代器
        norm_type: 范数类型
    
    Returns:
        grad_norm: 梯度范数
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    
    if len(parameters) == 0:
        return 0.0
    
    total_norm = 0.0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def gradient_protection(
    parameters,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
    nan_to_num: bool = True
) -> Tuple[float, bool]:
    """
    梯度保护：裁剪和 NaN 处理
    
    Args:
        parameters: 模型参数迭代器
        max_norm: 最大梯度范数
        norm_type: 范数类型
        nan_to_num: 是否将 NaN 替换为 0
    
    Returns:
        grad_norm: 裁剪前的梯度范数
        had_nan: 是否存在 NaN
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    
    if len(parameters) == 0:
        return 0.0, False
    
    # 检查并处理 NaN
    had_nan = False
    if nan_to_num:
        for p in parameters:
            if torch.isnan(p.grad).any():
                had_nan = True
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 计算梯度范数
    grad_norm = compute_gradient_norm(parameters, norm_type)
    
    # 梯度裁剪
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)
    
    return grad_norm, had_nan


def freeze_parameters(module: nn.Module):
    """
    冻结模块参数
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_parameters(module: nn.Module):
    """
    解冻模块参数
    """
    for param in module.parameters():
        param.requires_grad = True


def get_parameter_groups(
    model: nn.Module,
    base_lr: float,
    layer_decay: float = 1.0
) -> list:
    """
    获取分层学习率的参数组
    
    Args:
        model: 模型
        base_lr: 基础学习率
        layer_decay: 层级学习率衰减因子
    
    Returns:
        param_groups: 参数组列表
    """
    if layer_decay == 1.0:
        return [{'params': model.parameters(), 'lr': base_lr}]
    
    param_groups = []
    
    # 对于 ShareLayer 模型
    if hasattr(model, 'share_layer'):
        param_groups.append({
            'params': model.share_layer.parameters(),
            'lr': base_lr * layer_decay
        })
    
    # 其他子模块
    for name, child in model.named_children():
        if name != 'share_layer':
            param_groups.append({
                'params': child.parameters(),
                'lr': base_lr
            })
    
    return param_groups


def compute_loss_gradient(
    loss: torch.Tensor,
    model: nn.Module,
    retain_graph: bool = False,
    create_graph: bool = False
) -> Dict[str, torch.Tensor]:
    """
    计算损失对模型参数的梯度
    
    Args:
        loss: 损失张量
        model: 模型
        retain_graph: 是否保留计算图
        create_graph: 是否创建梯度的计算图
    
    Returns:
        grads: 参数名到梯度的字典
    """
    grads = {}
    
    # 清除现有梯度
    model.zero_grad()
    
    # 反向传播
    loss.backward(retain_graph=retain_graph, create_graph=create_graph)
    
    # 收集梯度
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.clone()
    
    return grads


def gradient_conflict_detection(
    grads1: Dict[str, torch.Tensor],
    grads2: Dict[str, torch.Tensor]
) -> float:
    """
    检测两个梯度之间的冲突程度
    
    返回：余弦相似度（-1 到 1，负数表示冲突）
    """
    cos_sims = []
    
    for name in grads1:
        if name in grads2:
            g1 = grads1[name].flatten()
            g2 = grads2[name].flatten()
            
            cos_sim = torch.nn.functional.cosine_similarity(
                g1.unsqueeze(0), g2.unsqueeze(0)
            ).item()
            
            cos_sims.append(cos_sim)
    
    if len(cos_sims) == 0:
        return 0.0
    
    return sum(cos_sims) / len(cos_sims)


class GradientAccumulator:
    """
    梯度累积器
    
    用于小 batch 训练时累积梯度
    """
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
    
    def should_step(self) -> bool:
        """
        是否应该执行优化器 step
        """
        self.step_count += 1
        return self.step_count >= self.accumulation_steps
    
    def reset(self):
        """
        重置计数器
        """
        self.step_count = 0
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        缩放损失以实现梯度累积
        """
        return loss / self.accumulation_steps
