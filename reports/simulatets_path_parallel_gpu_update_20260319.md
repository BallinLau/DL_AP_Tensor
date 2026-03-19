# SimulateTS Path 并行 GPU 化修改报告

**日期**: 2026-03-19

## 修改目标

原先 `DL_AP_Tensor/data/simulate_ts.py` 的 `simulate_tensor()` 仍然是“单 path 张量化”，即：

1. `time` 用循环推进
2. `path` 也用 Python 循环逐条跑
3. 只有 path 内部的 firm 计算使用 tensor

这会导致 GPU 只能吃到单条 path 的工作量，无法把 `n_paths` 这个天然独立的维度并行起来。

本次修改的目标是：

1. 保留 `time` 递推逻辑
2. 去掉 `path` 维度的 Python 循环
3. 把所有 path 在 GPU 上批量推进
4. 最终仍然输出与原接口兼容的 `TensorSimulationOutput`

## 为什么要这样改

状态转移的真正依赖结构是：

\[
S_{t+1} = \mathcal{T}(S_t, \varepsilon_{t+1})
\]

其中 `time` 维度有因果依赖，通常必须顺序推进；但不同 `path` 在给定当前状态后彼此独立，因此完全可以并行。

所以正确的并行结构应该是：

1. `for t in range(T)` 保留
2. `path` 维度改成批量张量
3. `firm` 横截面也作为批量张量一起处理

## 本次具体修改

### 1. 新增 path 并行 helper 模块

新增文件：
- [simulate_ts_parallel.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/data/simulate_ts_parallel.py)

新增主入口：
- `simulate_tensor_parallel(sim)`

职责：
- 统一管理 `[n_paths, max_firms]` 的 batched state
- 保留时间循环
- 在每个时间步同时推进所有 paths

### 2. `simulate_tensor()` 切到并行实现

修改文件：
- [simulate_ts.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/data/simulate_ts.py)

行为变化：
- 旧实现：`for path_idx in range(self.n_paths)` 逐条 path 模拟
- 新实现：直接调用 `simulate_tensor_parallel(self)`

接口不变：
- 仍返回 `TensorSimulationOutput`
- `simulate()` 仍只在最后转 DataFrame

### 3. batched state 设计

核心状态从“单 path dict”改成“全路径大张量 dict”：

- `b, z, eta, i, K, alive, entry, firm_id, bar_i, bar_z, bp`: `[n_paths, max_firms]`
- `x, hatcf, lnkf, M, next_firm_id`: `[n_paths]`

其中：
- `max_firms = group_size + horizon * n_potential_entry`
- `alive` 控制当前存活公司
- `next_firm_id` 保证每条 path 的进入者按顺序附加，不复用旧槽位

### 4. batched node 处理

新增逻辑：
- `_process_node_batched(...)`

做的事：
1. 用 `alive` mask 把所有 path 的活跃 firm 一次性拉平成大 batch
2. 一次性跑 `policy_value`
3. 一次性做资源核算
4. 用 `index_add_` 聚合出每条 path 的宏观量
5. 再把 `bar_i / bar_z / bp` scatter 回完整状态矩阵

这样 GPU 每次看到的是“所有 path 的所有活跃 firm”的大 batch，而不是单条 path 的小 batch。

### 5. batched branch 扩展

新增逻辑：
- `_expand_branches_batched(...)`

做的事：
1. 对所有 path 同时采样 `x_{t+1}, z_{t+1}, eta_{t+1}, i_{t+1}`
2. 同时更新 `b_{t+1}` 和 `K_{t+1}`
3. 如果存在 `sdf_fc1`，则同时对所有 path 跑一次 FC1/SDF 宏观更新

### 6. batched entry / exit

新增逻辑：
- `_apply_entry_batched(...)`
- `_apply_exit_batched(...)`

进入：
- 每条 path 同时生成 `n_potential` 个候选进入者
- 用 `enter_mask` 判断进入
- 用 `cumsum` 计算每条 path 的 append 位置
- 直接 scatter 到 `[path, firm_slot]`

退出：
- 直接对整块张量做 `alive &= (bar_z < 0.5)`

## 本次改动的工程含义

### 改进想法

目标不是把所有循环都删掉，而是只保留真正无法并行的 `time` 循环，把本来可并行的 `path` 和 `firm` 两个维度合并成大张量批量算。

### 改进行为

从原来的：
- `time` 循环
- `path` 循环
- `firm` 张量化

变成：
- `time` 循环
- `path` 并行
- `firm` 并行

### 改进原因

原先 GPU 吞吐低的根本原因之一，是 `n_paths` 这个最自然的并行维度被 Python 串行掉了。即使 firm 内部是 tensor，也只能让 GPU 看到单条 path 的横截面，无法形成大的工作集。

## 预期效果

1. 当 `n_paths` 较大时，GPU 利用率应明显上升
2. 模拟阶段不再随着 `n_paths` 线性地被 Python path 循环拖慢
3. 后续 FC2 若重写为 tensor-first，可以直接复用这种 batched state 结构

## 仍然保留的限制

1. `time` 维度仍是顺序递推，这是经济学状态转移本身决定的
2. 输出收集仍然是“每个时间步 append rows”，最后再 `cat_rows`
3. 还没有继续做 `torch.compile`、AMP、CUDA graph 等进一步优化

## 后续建议

下一步如果继续压榨吞吐，优先顺序应该是：

1. 用实际服务器数据测 `simulate_tensor()` 的 wall time 与 GPU 峰值
2. 评估 `n_paths × group_size × horizon` 的最优工作集
3. 再决定是否把输出 buffer 也预分配，进一步减少 append/cat 开销
