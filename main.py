"""
DL-AP: Deep Learning Asset Pricing

主入口文件
"""

import argparse
import torch
import logging
from pathlib import Path

from config import Config, HyperParams
from models import (
    SDFFC1Combined,
    PolicyValueModel,
    FC2Model
)
from training import Trainer
from utils import setup_logger, CheckpointManager


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='DL-AP: Deep Learning Asset Pricing')
    
    # 训练配置
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'simulate'],
                        help='运行模式')
    parser.add_argument('--n_episodes', type=int, default=10,
                        help='训练 Episode 数量')
    parser.add_argument('--epochs_per_episode', type=int, default=10,
                        help='每个 Episode 的 epoch 数')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='批大小')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='每个 Episode 的样本数')
    
    # 超参数
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    
    # 模型配置
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='隐藏层维度')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='层数')
    
    # 训练策略
    parser.add_argument('--train_mode', type=str, default='joint',
                        choices=['joint', 'staged', 'alternating'],
                        help='训练策略')
    
    # 路径
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    
    # 设备
    parser.add_argument('--device', type=str, default='auto',
                        help='设备 (auto/cpu/cuda/mps)')
    
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """
    获取计算设备
    """
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_str)


def build_models(config, hyperparams, device):
    """
    构建模型
    """
    models = {}
    
    # SDF & FC1 Combined Model
    models['sdf_fc1'] = SDFFC1Combined(
        sdf_input_dim=config.SDF_INPUT_DIM,
        fc1_input_dim=config.FC1_INPUT_DIM,
        hidden_dim=hyperparams.hidden_dim,
        n_layers=hyperparams.n_layers
    ).to(device)
    
    # Policy & Value Model
    models['policy_value'] = PolicyValueModel(
        state_dim=config.FIRM_STATE_DIM,
        hidden_dim=hyperparams.hidden_dim,
        n_layers=hyperparams.n_layers
    ).to(device)
    
    # FC2 Model (optional)
    models['fc2'] = FC2Model(
        n_quantiles=config.FC2_N_QUANTILES,
        hidden_dim=hyperparams.hidden_dim,
        n_layers=hyperparams.n_layers
    ).to(device)
    
    return models


def train(args):
    """
    训练流程
    """
    # 设置日志
    logger = setup_logger('DL-AP', args.log_dir)
    logger.info(f"Starting training with args: {args}")
    
    # 设备
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # 配置
    Config.DEVICE = device
    hyperparams = HyperParams(
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers
    )
    
    # 构建模型
    models = build_models(Config, hyperparams, device)
    logger.info("Models built successfully")
    
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  {name}: {n_params:,} parameters")
    
    # 创建 Trainer
    trainer = Trainer(
        models=models,
        config=Config,
        hyperparams=hyperparams,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        device=device
    )
    
    # 恢复训练
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    if args.train_mode == 'joint':
        summary = trainer.train(
            n_episodes=args.n_episodes,
            epochs_per_episode=args.epochs_per_episode,
            batch_size=args.batch_size,
            n_samples=args.n_samples
        )
    elif args.train_mode == 'staged':
        summary = trainer.train_staged(
            stage1_episodes=args.n_episodes // 2,
            stage2_episodes=args.n_episodes // 2,
            epochs_per_episode=args.epochs_per_episode,
            batch_size=args.batch_size,
            n_samples=args.n_samples
        )
    elif args.train_mode == 'alternating':
        summary = trainer.train_alternating(
            n_episodes=args.n_episodes,
            epochs_per_episode=args.epochs_per_episode,
            batch_size=args.batch_size,
            n_samples=args.n_samples
        )
    
    logger.info(f"Training completed: {summary}")
    
    return summary


def evaluate(args):
    """
    评估流程
    """
    logger = setup_logger('DL-AP', args.log_dir)
    
    device = get_device(args.device)
    Config.DEVICE = device
    
    hyperparams = HyperParams(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers
    )
    
    models = build_models(Config, hyperparams, device)
    
    # 创建 Trainer
    trainer = Trainer(
        models=models,
        config=Config,
        hyperparams=hyperparams,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        device=device
    )
    
    # 加载模型
    checkpoint_name = args.resume or 'best'
    trainer.load_checkpoint(checkpoint_name)
    
    # 评估
    metrics = trainer.evaluate(n_samples=args.n_samples)
    
    logger.info(f"Evaluation results: {metrics}")
    
    return metrics


def simulate(args):
    """
    模拟流程
    """
    from data import SimulateTS
    
    logger = setup_logger('DL-AP', args.log_dir)
    
    device = get_device(args.device)
    Config.DEVICE = device
    
    hyperparams = HyperParams(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers
    )
    
    models = build_models(Config, hyperparams, device)
    
    # 加载模型
    ckpt_manager = CheckpointManager(args.save_dir)
    checkpoint_name = args.resume or 'best'
    ckpt_manager.load(models, name=checkpoint_name, device=device)
    
    # 创建模拟器
    simulator = SimulateTS(
        models=models,
        config=Config,
        n_paths=100,
        group_size=200,
        horizon=20,
        device=device
    )
    
    # 运行模拟
    df_firm, df_macro = simulator.simulate()
    
    # 保存结果
    output_dir = Path(args.save_dir) / 'simulation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_firm.to_csv(output_dir / 'firm_panel.csv', index=False)
    df_macro.to_csv(output_dir / 'macro_panel.csv', index=False)
    
    logger.info(f"Simulation completed. Results saved to {output_dir}")
    
    return df_firm, df_macro


def main():
    """
    主函数
    """
    args = parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args)
    elif args.mode == 'simulate':
        simulate(args)


if __name__ == '__main__':
    main()
