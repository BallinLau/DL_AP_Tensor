"""
GPU Memory Monitor - Monitor GPU memory usage during training
Outputs results in JSON format
"""

import torch
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path


class GPUMonitor:
    """GPU 显存监控器"""
    
    def __init__(self, device: torch.device, log_interval: int = 10):
        """
        Args:
            device: PyTorch device
            log_interval: Log every N calls to log_memory()
        """
        self.device = device
        self.log_interval = log_interval
        self.call_count = 0
        self.memory_history = []
        self.enabled = device.type == 'cuda' and torch.cuda.is_available()
        
    def get_memory_info(self) -> Dict[str, Any]:
        """获取当前 GPU 显存信息"""
        if not self.enabled:
            return {"enabled": False, "device": str(self.device)}
        
        torch.cuda.synchronize(self.device)
        
        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)  # MB
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)  # MB
        max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)  # MB
        
        # Get total memory
        total_memory = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 2)  # MB
        
        return {
            "enabled": True,
            "device": str(self.device),
            "device_name": torch.cuda.get_device_name(self.device),
            "allocated_mb": round(allocated, 2),
            "reserved_mb": round(reserved, 2),
            "max_allocated_mb": round(max_allocated, 2),
            "total_mb": round(total_memory, 2),
            "utilization_percent": round((allocated / total_memory) * 100, 2),
            "timestamp": time.time()
        }
    
    def log_memory(self, context: str = "") -> Dict[str, Any]:
        """记录显存使用情况"""
        self.call_count += 1
        
        info = self.get_memory_info()
        info["context"] = context
        info["call_count"] = self.call_count
        
        # Only store every Nth call to save memory
        if self.call_count % self.log_interval == 0 or context in ["episode_start", "episode_end"]:
            self.memory_history.append(info)
        
        return info
    
    def reset_peak_stats(self):
        """重置峰值统计"""
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取显存使用摘要"""
        if not self.memory_history:
            return {"enabled": self.enabled, "history_count": 0}
        
        allocated_values = [h["allocated_mb"] for h in self.memory_history if h.get("enabled")]
        
        if not allocated_values:
            return {"enabled": self.enabled, "history_count": len(self.memory_history)}
        
        return {
            "enabled": True,
            "history_count": len(self.memory_history),
            "max_allocated_mb": max(allocated_values),
            "min_allocated_mb": min(allocated_values),
            "avg_allocated_mb": round(sum(allocated_values) / len(allocated_values), 2),
            "final_allocated_mb": allocated_values[-1],
            "device_name": self.memory_history[0].get("device_name", "Unknown")
        }
    
    def save_to_json(self, filepath: Path):
        """保存监控历史到 JSON 文件"""
        data = {
            "summary": self.get_summary(),
            "history": self.memory_history
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath


def print_memory_summary(info: Dict[str, Any], prefix: str = ""):
    """打印显存摘要信息"""
    if not info.get("enabled"):
        print(f"{prefix}GPU monitoring disabled (CPU mode)")
        return
    
    print(f"{prefix}GPU: {info.get('device_name', 'Unknown')}")
    print(f"{prefix}  Allocated: {info['allocated_mb']:.1f} MB / {info['total_mb']:.1f} MB ({info['utilization_percent']:.1f}%)")
    print(f"{prefix}  Reserved: {info['reserved_mb']:.1f} MB")
    print(f"{prefix}  Max Allocated: {info['max_allocated_mb']:.1f} MB")


# Global monitor instance
_global_monitor: Optional[GPUMonitor] = None


def get_monitor(device: torch.device = None, log_interval: int = 10) -> GPUMonitor:
    """获取或创建全局监控器"""
    global _global_monitor
    
    if _global_monitor is None and device is not None:
        _global_monitor = GPUMonitor(device, log_interval)
    
    return _global_monitor


def reset_monitor():
    """重置全局监控器"""
    global _global_monitor
    _global_monitor = None
