"""
日志工具
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = 'DL-AP',
    log_dir: str = './logs',
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志器名称
        log_dir: 日志目录
        level: 日志级别
        console: 是否输出到控制台
        file: 是否输出到文件
    
    Returns:
        logger: 配置好的日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除现有处理器
    logger.handlers = []
    
    # 格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if file:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{name}_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'DL-AP') -> logging.Logger:
    """
    获取已存在的日志器
    """
    return logging.getLogger(name)


class LoggerContext:
    """
    日志上下文管理器
    
    用于临时修改日志级别
    """
    
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.new_level = level
        self.old_level = logger.level
    
    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)
        return False


class ProgressLogger:
    """
    进度日志记录器
    
    定期记录训练进度
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        total_steps: int,
        log_interval: int = 100
    ):
        self.logger = logger
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.current_step = 0
        self.metrics = {}
    
    def step(self, metrics: dict = None):
        """
        记录一步
        """
        self.current_step += 1
        
        if metrics:
            for k, v in metrics.items():
                if k not in self.metrics:
                    self.metrics[k] = []
                self.metrics[k].append(v)
        
        if self.current_step % self.log_interval == 0:
            self._log_progress()
    
    def _log_progress(self):
        """
        输出进度日志
        """
        progress = self.current_step / self.total_steps * 100
        
        msg = f"Step {self.current_step}/{self.total_steps} ({progress:.1f}%)"
        
        if self.metrics:
            # 计算最近 log_interval 步的平均值
            recent_metrics = {}
            for k, v in self.metrics.items():
                recent = v[-self.log_interval:]
                if recent:
                    recent_metrics[k] = sum(recent) / len(recent)
            
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in recent_metrics.items())
            msg += f" | {metrics_str}"
        
        self.logger.info(msg)
    
    def reset(self):
        """
        重置
        """
        self.current_step = 0
        self.metrics = {}
