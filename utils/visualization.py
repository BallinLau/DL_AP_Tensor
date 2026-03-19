"""
可视化工具
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def plot_training_curves(
    history: List[Dict],
    metrics: List[str] = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    绘制训练曲线
    
    Args:
        history: 训练历史列表
        metrics: 要绘制的指标列表
        save_path: 保存路径
        figsize: 图片大小
    """
    if metrics is None:
        # 自动检测指标
        if len(history) > 0 and 'final_losses' in history[0]:
            metrics = list(history[0]['final_losses'].keys())
        else:
            metrics = ['total']
    
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    
    episodes = list(range(1, len(history) + 1))
    
    for idx, metric in enumerate(metrics):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]
        
        values = []
        for h in history:
            if 'final_losses' in h and metric in h['final_losses']:
                values.append(h['final_losses'][metric])
            elif 'loss_history' in h and metric in h['loss_history']:
                values.append(np.mean(h['loss_history'][metric]))
            else:
                values.append(0)
        
        ax.plot(episodes, values, 'b-o', markersize=4)
        ax.set_xlabel('Episode')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} vs Episode')
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(n_metrics, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_leverage_distribution(
    b_values: np.ndarray,
    title: str = 'Leverage Distribution',
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    绘制杠杆分布
    
    Args:
        b_values: 杠杆值数组
        title: 标题
        save_path: 保存路径
        figsize: 图片大小
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 直方图
    ax1 = axes[0]
    ax1.hist(b_values, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(np.mean(b_values), color='red', linestyle='--', label=f'Mean: {np.mean(b_values):.3f}')
    ax1.axvline(np.median(b_values), color='green', linestyle='--', label=f'Median: {np.median(b_values):.3f}')
    ax1.set_xlabel('Leverage (b)')
    ax1.set_ylabel('Density')
    ax1.set_title('Histogram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # CDF
    ax2 = axes[1]
    sorted_b = np.sort(b_values)
    cdf = np.arange(1, len(sorted_b) + 1) / len(sorted_b)
    ax2.plot(sorted_b, cdf, 'b-')
    ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(np.median(b_values), color='green', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Leverage (b)')
    ax2.set_ylabel('CDF')
    ax2.set_title('Cumulative Distribution')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_policy_functions(
    b_grid: np.ndarray,
    policy_values: Dict[str, np.ndarray],
    z_fixed: float = 0.0,
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    绘制策略函数
    
    Args:
        b_grid: 杠杆网格
        policy_values: 策略值字典 {name: values}
        z_fixed: 固定的 z 值
        save_path: 保存路径
        figsize: 图片大小
    """
    n_policies = len(policy_values)
    n_cols = min(3, n_policies)
    n_rows = (n_policies + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx, (name, values) in enumerate(policy_values.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]
        
        ax.plot(b_grid, values, 'b-', linewidth=2)
        ax.set_xlabel('Leverage (b)')
        ax.set_ylabel(name)
        ax.set_title(f'{name}(b | z={z_fixed:.2f})')
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(n_policies, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_sdf_surface(
    x_grid: np.ndarray,
    hatc_grid: np.ndarray,
    M_values: np.ndarray,
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    绘制 SDF 曲面
    
    Args:
        x_grid: x 网格
        hatc_grid: ĉ 网格
        M_values: SDF 值 (len(x_grid), len(hatc_grid))
        save_path: 保存路径
        figsize: 图片大小
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(x_grid, hatc_grid)
    
    ax.plot_surface(X, Y, M_values.T, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('x')
    ax.set_ylabel('ĉ')
    ax.set_zlabel('M')
    ax.set_title('Stochastic Discount Factor')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_resource_balance(
    df,
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 5)
):
    """
    绘制资源核算图
    
    Args:
        df: 包含 Y, I, C, Phi 的 DataFrame
        save_path: 保存路径
        figsize: 图片大小
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Y 分解
    ax1 = axes[0]
    if all(col in df.columns for col in ['Y', 'I', 'C']):
        Y_mean = df['Y'].mean()
        I_mean = df['I'].mean()
        C_mean = df['C'].mean()
        Phi_mean = df.get('Phi', 0).mean() if 'Phi' in df.columns else 0
        
        components = ['I', 'C', 'Φ']
        values = [I_mean, C_mean, Phi_mean]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        ax1.pie(values, labels=components, colors=colors, autopct='%1.1f%%')
        ax1.set_title(f'Output Decomposition\nY = {Y_mean:.2f}')
    
    # 时间序列（如果有 t 列）
    ax2 = axes[1]
    if 't' in df.columns:
        df_grouped = df.groupby('t').agg({'Y': 'sum', 'C': 'sum', 'K': 'sum'}).reset_index()
        ax2.plot(df_grouped.index, df_grouped['Y'], label='Y')
        ax2.plot(df_grouped.index, df_grouped['C'], label='C')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Aggregate')
        ax2.set_title('Aggregate Dynamics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 截面分布
    ax3 = axes[2]
    if 'K' in df.columns:
        ax3.hist(np.log(df['K'].clip(lower=1e-6)), bins=50, alpha=0.7)
        ax3.set_xlabel('log(K)')
        ax3.set_ylabel('Count')
        ax3.set_title('Capital Distribution')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
