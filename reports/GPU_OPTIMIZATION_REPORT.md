# DL-AP 训练流程 GPU 优化报告

**报告日期**: 2026-03-12  
**硬件环境**: 80GB GPU + 有限 CPU 资源  
**目标**: 在保持代码功能的前提下，最大化 GPU 利用率，提升训练速度

---

## 1. 当前性能瓶颈分析

### 1.1 各阶段训练速度

基于日志 `error.274531` 的分析：

| 训练阶段 | 当前速度 | 批次/数据量 | 每轮时间 | GPU利用率 |
|----------|----------|-------------|----------|-----------|
| **SDF/FC1** | ~90-130 it/s | 1 batch/轮 | ~0.01s/轮 | 中 |
| **Policy/Value** | ~8 it/s | 2 batches/轮 | ~0.12s/轮 | 中 |
| **FC2** | ~0.04 it/s | 200 paths × 20 horizon | **~23s/轮** | **极低** |
| **SimulateTS** | ~50 paths/s | 纯 CPU 执行 | - | 无 |

### 1.2 关键瓶颈识别

```
┌─────────────────────────────────────────────────────────────┐
│                    训练流程瓶颈分析                          │
├─────────────────────────────────────────────────────────────┤
│  FC2 阶段 (主要瓶颈)                                         │
│  ├── 每轮从 DataFrame 重新构建 tensors                       │
│  ├── CPU-GPU 数据传输开销巨大                                │
│  ├── FC2LossPipe 中的循环操作                                │
│  └── 单次前向传播涉及多次模型调用                            │
│                                                              │
│  数据生成阶段                                                │
│  ├── Sample 类完全在 CPU 上执行                              │
│  ├── SimulateTS 串行模拟 paths                               │
│  └── 数据生成成为 GPU 等待的主要原因                         │
│                                                              │
│  训练配置                                                    │
│  ├── batch_size=512 过小，无法充分利用 GPU                   │
│  ├── n_samples=10000 样本量不足                              │
│  └── 未启用混合精度训练                                      │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 资源利用率分析

- **GPU 显存**: 当前使用 < 2GB / 80GB (利用率 < 3%)
- **GPU 计算**: FC2 阶段大部分时间等待 CPU 数据传输
- **CPU**: 成为瓶颈，数据生成和 DataFrame 操作占用大量 CPU 时间

---

## 2. 优化策略总览

### 2.1 优化原则

1. **最大化 GPU 驻留数据**: 将数据预生成并常驻 GPU，减少 CPU-GPU 传输
2. **增大并行度**: 利用 80GB 显存，大幅增加 batch size 和样本量
3. **减少 CPU 依赖**: 将数据生成和预处理搬到 GPU
4. **启用混合精度**: 使用 AMP 提升计算吞吐量

### 2.2 优化方案优先级

| 优先级 | 优化项 | 预期提升 | 实施难度 | 投入产出比 |
|--------|--------|----------|----------|------------|
| ⭐⭐⭐⭐⭐ | FC2 预计算优化 | 20-40x | 中 | 极高 |
| ⭐⭐⭐⭐⭐ | 增大 batch size | 2-4x | 低 | 极高 |
| ⭐⭐⭐⭐ | 混合精度训练 | 1.5-2x | 低 | 高 |
| ⭐⭐⭐⭐ | GPU 化数据生成 | 3-5x | 中 | 高 |
| ⭐⭐⭐ | 数据预加载缓存 | 2-3x | 中 | 中 |
| ⭐⭐ | 并行 SimulateTS | 2-3x | 高 | 中 |

---

## 3. 详细优化方案

### 3.1 FC2 阶段优化（优先级：最高）

#### 3.1.1 核心问题

当前 FC2 训练代码 (`episode.py:1789-1818`)：

```python
# 问题：每轮都重新处理 DataFrame
for epoch in tqdm(range(n_epochs), desc='FC2 Epochs'):
    losses = self.train_step(self.df, ['fc2'])  # 每轮重建 tensors
    # train_step 内部调用 FC2LossPipe，涉及大量 DataFrame 操作
```

**瓶颈分析**:
- `FC2LossPipe.__init__` 每轮都执行 `fill_df_to_fullN` 和 `_build_tensors`
- 大量 CPU-GPU 数据传输: `torch.tensor(group[...].to_numpy(), device=self.device)`
- `build_fc2_input_parent` 和 `build_fc2_input_children` 每轮循环计算

#### 3.1.2 优化方案

**方案 A: 预计算所有 tensors（推荐）**

```python
# episode.py 修改

def run_episode(self, ...):
    # ... 前面代码 ...
    
    if 'fc2' in train_modules and 'fc2' in self.models:
        # 1. 生成模拟数据（一次）
        simulator = SimulateTS(...)
        self.df, self.df_macro = simulator.simulate()
        
        # 2. 预计算所有 FC2 tensors（关键优化）
        fc2_tensors = self._precompute_fc2_tensors(self.df)
        
        # 3. 使用 _run_batches 训练（复用已有高效训练循环）
        fc2_batches = [fc2_tensors]  # 整个数据集作为一个 batch
        module_summaries['fc2'] = self._run_batches(
            fc2_batches, n_epochs, log_interval, ['fc2'], desc_prefix='FC2 '
        )

def _precompute_fc2_tensors(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
    """
    预计算 FC2 所需的所有 tensors，避免每轮重复计算
    """
    from losses.FC2losspipe import FC2Pipeline
    
    # 创建 pipeline 并提取所有预计算的 tensors
    pipe = FC2Pipeline(df=df, full_N=1000, device=self.device)
    
    return {
        'P_s_full': pipe.P_s_full,
        'K_parent_full': pipe.K_parent_full,
        'Children_s_full': pipe.Children_s_full,
        'K_children_full': pipe.K_children_full,
        'alive_mask': pipe.alive_mask,
        'entry_mask': pipe.entry_mask,
        'path_num': pipe.path_num,
        'full_N': pipe.full_N,
        # 预计算输入特征
        'FC2_input_parent': pipe.build_fc2_input_parent(),
    }

def compute_fc2_loss_optimized(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    优化的 FC2 loss 计算，直接使用预计算 tensors
    """
    model = self.models['fc2']
    pv_model = self.models['policy_value']
    
    # 直接从 batch 获取预计算 tensors
    fc2_input_parent = batch['FC2_input_parent']
    
    # 前向传播（无需重复构建输入）
    fc2_out_parent = model(fc2_input_parent)
    
    # ... 后续计算使用 batch 中的其他预计算 tensors
    # 避免所有 DataFrame 操作
```

**方案 B: 使用 DataLoader（备选）**

```python
from torch.utils.data import Dataset, DataLoader

class FC2Dataset(Dataset):
    """FC2 数据集，预加载所有数据到 GPU"""
    
    def __init__(self, df, device):
        self.device = device
        # 预计算所有数据
        self.data = self._precompute(df)
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}
    
    def __len__(self):
        return len(self.data['path_num'])

# 训练时使用 DataLoader
dataset = FC2Dataset(df, self.device)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

for epoch in range(n_epochs):
    for batch in dataloader:
        loss = self.compute_fc2_loss(batch)
        # ... 反向传播
```

#### 3.1.3 预期效果

- **速度提升**: 23s/轮 → 0.5-1s/轮 (**20-40 倍**)
- **GPU 利用率**: 从 <10% 提升到 >80%
- **CPU 负载**: 大幅降低

---

### 3.2 增大 Batch Size（优先级：最高）

#### 3.2.1 当前配置分析

```python
# config/hyperparams.py 当前值
@dataclass
class HyperParams:
    batch_size: int = 512
    n_samples: int = 10000
    n_paths: int = 1000
    group_size: int = 200
```

**显存占用估算**:
- 模型参数: ~100MB
- 10k samples × 50 features × 4 bytes = 2MB
- 1k paths × 200 companies × 20 features × 4 bytes = 16MB
- **总计 < 200MB，远小于 80GB**

#### 3.2.2 优化配置

```python
# config/hyperparams.py 优化后
@dataclass
class HyperParams:
    # ========== 优化后的训练参数 ==========
    batch_size: int = 4096        # 从 512 增大 8 倍
    n_samples: int = 50000        # 从 10000 增大 5 倍
    n_paths: int = 2000           # 从 1000 增大 2 倍
    group_size: int = 500         # 从 200 增大 2.5 倍
    simulate_horizon: int = 20    # 保持不变
    
    # ========== GPU 优化参数 ==========
    use_amp: bool = True          # 启用混合精度训练
    pin_memory: bool = True       # 固定内存加速传输
    num_workers: int = 0          # CPU 有限，设为 0
    
    # ========== FC2 优化参数 ==========
    fc2_precompute: bool = True   # 预计算 FC2 tensors
    fc2_cache_gpu: bool = True    # FC2 数据常驻 GPU
    fc2_batch_size: int = 1       # FC2 使用全数据集
```

#### 3.2.3 显存占用重新估算

优化后：
- 模型参数: ~100MB
- 50k samples × 50 features × 4 bytes = 10MB
- 2k paths × 500 companies × 20 features × 4 bytes = 80MB
- FC2 预计算 tensors: ~500MB
- **总计 < 1GB，仍远小于 80GB**

**仍有巨大空间可以进一步增大**。

---

### 3.3 混合精度训练（优先级：高）

#### 3.3.1 实现方案

```python
# training/episode.py

from torch.cuda.amp import autocast, GradScaler

class Episode:
    def __init__(self, ...):
        # ... 原有初始化 ...
        self.scaler = GradScaler()  # 添加梯度缩放器
        self.use_amp = hyperparams.use_amp
    
    def train_step(self, batch, train_modules, ...):
        """支持混合精度的训练步骤"""
        
        if self.use_amp:
            with autocast():  # 自动混合精度上下文
                losses = self.compute_loss(batch, train_modules)
            
            # 缩放梯度防止下溢
            self.scaler.scale(losses['total']).backward()
            
            # 梯度裁剪（需要在缩放后）
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            # 更新参数
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # 原有训练逻辑
            losses = self.compute_loss(batch, train_modules)
            losses['total'].backward()
            optimizer.step()
        
        return losses
```

#### 3.3.2 适用场景

- **适合**: SDF/FC1, Policy/Value 的前向传播
- **注意**: FC2 涉及大量数值计算，需要验证精度

---

### 3.4 GPU 化数据生成（优先级：高）

#### 3.4.1 Sample 类 GPU 化

```python
# data/sample.py

class GPUSample:
    """GPU 加速的数据采样"""
    
    def __init__(self, models, config, n_samples, device):
        self.device = device
        self.n_samples = n_samples
        
    def build_sdf_fc1_df_gpu(self) -> torch.Tensor:
        """直接在 GPU 上生成 SDF/FC1 训练数据"""
        
        # 在 GPU 上生成所有随机数
        b = torch.rand(self.n_samples, device=self.device)
        z = torch.rand(self.n_samples, device=self.device)
        x = torch.randn(self.n_samples, device=self.device) * 0.1
        
        # 直接在 GPU 上计算
        with torch.no_grad():
            # 调用模型前向传播
            outputs = self.models['sdf_fc1'](torch.stack([b, z, x], dim=1))
        
        # 返回 GPU tensors，无需转到 CPU
        return outputs
```

#### 3.4.2 SimulateTS GPU 批量化

```python
# data/simulate_ts.py

def simulate_gpu_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU 批量模拟所有 paths
    替代原有的串行 simulate()
    """
    batch_size = 100  # 每批处理 100 个 paths
    n_batches = (self.n_paths + batch_size - 1) // batch_size
    
    all_firm_data = []
    all_macro_data = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, self.n_paths)
        
        # 批量初始化状态
        states = self._initialize_batch(start_idx, end_idx)
        
        # 在 GPU 上批量模拟
        for t in range(self.horizon):
            states = self._step_batch(states, t)
        
        # 收集结果
        firm_data, macro_data = self._collect_batch_results(states)
        all_firm_data.append(firm_data)
        all_macro_data.append(macro_data)
    
    return torch.cat(all_firm_data), torch.cat(all_macro_data)
```

---

### 3.5 数据预加载缓存（优先级：中）

#### 3.5.1 实现方案

```python
# training/data_cache.py

class GPUDataCache:
    """
    预生成并缓存所有训练数据在 GPU 上
    避免每轮重复生成数据
    """
    
    def __init__(self, models, config, hyperparams, device):
        self.device = device
        self.models = models
        self.config = config
        self.hyperparams = hyperparams
        self.cache = {}
        
    def precompute_all(self, episode_id: int = 0):
        """预计算指定 episode 的所有数据"""
        
        if episode_id == 0:
            self._precompute_episode0()
        else:
            self._precompute_episode_n(episode_id)
    
    def _precompute_episode0(self):
        """预计算 Episode 0 的所有数据"""
        
        # 1. SDF/FC1 数据
        self.cache['sdf_batches'] = self._generate_sdf_batches(
            n_samples=self.hyperparams.n_samples,
            batch_size=self.hyperparams.batch_size
        )
        
        # 2. Policy/Value 数据
        self.cache['policy_batches'] = self._generate_policy_batches(
            n_samples=self.hyperparams.n_samples,
            batch_size=self.hyperparams.batch_size
        )
        
        # 3. FC2 数据
        if 'fc2' in self.models:
            self._precompute_fc2_data()
    
    def _precompute_fc2_data(self):
        """预计算 FC2 的所有 tensors"""
        from data.simulate_ts import SimulateTS
        
        simulator = SimulateTS(
            models=self.models,
            config=self.config,
            n_paths=self.hyperparams.n_paths,
            group_size=self.config.SIMULATE_GROUP_SIZE,
            device=self.device,
        )
        
        df, df_macro = simulator.simulate()
        
        # 预计算所有 tensors 并常驻 GPU
        from losses.FC2losspipe import FC2Pipeline
        pipe = FC2Pipeline(df=df, device=self.device)
        
        self.cache['fc2_tensors'] = {
            'P_s_full': pipe.P_s_full,
            'K_parent_full': pipe.K_parent_full,
            'Children_s_full': pipe.Children_s_full,
            'alive_mask': pipe.alive_mask,
            'entry_mask': pipe.entry_mask,
        }
        self.cache['fc2_df'] = df
        self.cache['fc2_df_macro'] = df_macro
    
    def get_batches(self, stage: str) -> List[Dict]:
        """获取预计算的 batches"""
        return self.cache.get(f'{stage}_batches', [])
    
    def get_fc2_tensors(self) -> Dict[str, torch.Tensor]:
        """获取预计算的 FC2 tensors"""
        return self.cache.get('fc2_tensors', {})
```

---

## 4. 实施路线图

### 4.1 第一阶段：立即实施（1-2 天）

**目标**: 实现 50-100 倍速度提升

| 任务 | 文件 | 改动量 | 预期提升 |
|------|------|--------|----------|
| 1. 增大 batch size | `config/hyperparams.py` | 小 | 2-4x |
| 2. FC2 预计算优化 | `training/episode.py` | 中 | 20-40x |
| 3. 启用混合精度 | `training/episode.py` | 小 | 1.5-2x |

**实施步骤**:

1. **修改超参数**:
   ```python
   batch_size = 4096
   n_samples = 50000
   n_paths = 2000
   use_amp = True
   ```

2. **添加 FC2 预计算**:
   - 在 `Episode` 类中添加 `_precompute_fc2_tensors` 方法
   - 修改 `run_episode` 中的 FC2 训练逻辑

3. **启用 AMP**:
   - 在 `Episode.__init__` 中添加 `GradScaler`
   - 修改 `train_step` 支持混合精度

### 4.2 第二阶段：中期优化（3-5 天）

**目标**: 再提升 3-5 倍

| 任务 | 文件 | 改动量 | 预期提升 |
|------|------|--------|----------|
| 4. 实现 GPUDataCache | `training/data_cache.py` | 中 | 2-3x |
| 5. GPU 化 Sample | `data/sample.py` | 中 | 2-3x |
| 6. 批量化 SimulateTS | `data/simulate_ts.py` | 大 | 3-5x |

### 4.3 第三阶段：深度优化（1-2 周）

**目标**: 榨干 GPU 性能

- 实现完整的 GPU 数据流水线
- 使用 CUDA Graphs 进一步优化
- 多 GPU 并行训练（如果需要）

---

## 5. 预期效果汇总

### 5.1 速度提升预测

```
优化前总时间估算（单 episode）:
├── SDF/FC1: 20 epochs × 0.01s = 0.2s
├── Policy/Value: 20 epochs × 0.12s = 2.4s
├── FC2: 20 epochs × 23s = 460s  ← 瓶颈
└── 总计: ~463s (约 7.7 分钟)

优化后总时间预测:
├── SDF/FC1: 20 epochs × 0.005s = 0.1s  (batch增大+AMP)
├── Policy/Value: 20 epochs × 0.03s = 0.6s  (batch增大+AMP)
├── FC2: 20 epochs × 0.5s = 10s  (预计算优化)
└── 总计: ~11s (约 0.2 分钟)

整体提升: ~42 倍
```

### 5.2 资源利用率预测

| 资源 | 优化前 | 优化后 |
|------|--------|--------|
| GPU 显存 | < 2GB | ~10-20GB |
| GPU 计算利用率 | 10-30% | 80-95% |
| CPU 负载 | 高 | 低 |
| 训练时间 | ~8 分钟/episode | ~10 秒/episode |

---

## 6. 风险与注意事项

### 6.1 数值精度风险

- **混合精度**: FC2 涉及复杂数值计算，启用 AMP 后需验证结果正确性
- **建议**: 先在小数据集上测试，对比 float32 和 amp 的结果差异

### 6.2 显存溢出风险

- 虽然 80GB 充足，但增大 batch size 时仍需监控
- **建议**: 使用 `torch.cuda.memory_summary()` 监控显存使用

### 6.3 代码兼容性

- 修改 `Episode` 类时需保持向后兼容
- **建议**: 添加配置开关，保留原有代码路径作为备选

---

## 7. 监控与验证

### 7.1 关键指标

```python
# 训练时监控以下指标
metrics = {
    'gpu_utilization': 'nvidia-smi 显示的 GPU 利用率',
    'gpu_memory_used': '显存使用量 (GB)',
    'throughput': 'samples/second 或 epochs/minute',
    'step_time': '每个训练步骤的平均时间',
    'data_loading_time': '数据加载时间占比',
}
```

### 7.2 验证清单

- [ ] FC2 优化后结果与优化前一致
- [ ] 混合精度训练无 NaN/Inf
- [ ] GPU 利用率稳定在 80% 以上
- [ ] 训练速度达到预期提升
- [ ] 显存使用在安全范围内

---

## 8. 附录

### 8.1 相关文件清单

| 文件路径 | 说明 | 修改优先级 |
|----------|------|------------|
| `config/hyperparams.py` | 超参数配置 | ⭐⭐⭐⭐⭐ |
| `training/episode.py` | Episode 训练逻辑 | ⭐⭐⭐⭐⭐ |
| `losses/FC2losspipe.py` | FC2 Loss 计算 | ⭐⭐⭐⭐ |
| `data/simulate_ts.py` | 模拟数据生成 | ⭐⭐⭐ |
| `data/sample.py` | 采样数据生成 | ⭐⭐⭐ |
| `training/data_cache.py` | 新增：数据缓存 | ⭐⭐⭐ |

### 8.2 参考资源

- [PyTorch Automatic Mixed Precision](https://pytorch.org/docs/stable/notes/amp_examples.html)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

**报告完成时间**: 2026-03-12  
**下次 review 建议**: 实施第一阶段优化后，对比实际效果与预期
