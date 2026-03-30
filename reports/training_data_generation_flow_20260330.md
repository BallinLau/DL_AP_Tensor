# 训练数据生成流程梳理（2026-03-30）

## 本次口径修正

默认配置现在统一为：

- `Sample` 默认 `group_size = 2`
- `SimulateTS` 默认 `group_size = 200`
- 默认 `simulate_horizon = 10`

对应代码位置：

- [`config/constants.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/config/constants.py)
  - `GROUP_SIZE = 2`
  - `SIMULATE_GROUP_SIZE = 200`
- [`config/hyperparams.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/config/hyperparams.py)
  - `simulate_horizon = 10`

同时，主训练入口 [`experiments/run_multi_episode_job.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/experiments/run_multi_episode_job.py) 和旧入口 [`experiments/run_multi_episode.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/experiments/run_multi_episode.py) 已改成按“数据源”传递 `group_size`：

- `Sample` 路径使用 `sample_group_size`
- `SimulateTS` 路径使用 `simulate_group_size`

这避免了之前 `modea` 在 `episode > 0` 时错误继承 `simulate_group_size = 200` 去生成 `Sample` 数据的问题。

## 三种 mode 的数据来源

### mode0

训练顺序：

1. 用 [`Sample`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/data/sample.py) 生成截面训练数据
2. 用这批 `Sample` 数据训练 `sdf_fc1(stage1)` 和 `policy_value`
3. 再用 [`SimulateTS`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/data/simulate_ts.py) 跑一次 `h=1` 的时间模拟
4. 用这次 `SimulateTS(h=1)` 的数据训练 `sdf_fc1(stage2)`，可选训练 `fc2`

默认数据规模：

- `Sample`: `n_paths x 2 firms/path x (1 + 2 branches)`
- `SimulateTS(h=1)`: `n_paths x 200 firms/path`

### modea

训练顺序：

1. 用 [`Sample`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/data/sample.py) 生成 `policy_value` 训练数据
2. 只用这批 `Sample` 数据训练 `policy_value`
3. 再用 [`SimulateTS`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/data/simulate_ts.py) 跑一次 `h=1`
4. 用这次 `SimulateTS(h=1)` 的结果训练 `sdf_fc1`，可选训练 `fc2`

关键点：

- `modea` 的 `policy_value` 数据来源是 `Sample`
- 不是 `SimulateTS`
- 因此 `policy_value` 的默认 firm 数应是 `2/path`，不是 `200/path`

### modeb

训练顺序：

1. 直接用 [`SimulateTS`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/data/simulate_ts.py) 跑 `h = simulate_horizon`
2. 从整段时间序列 panel 中提取 parent-child 单元
3. 用这批 panel 数据训练 `policy_value`
4. 再从同一批 panel 数据训练 `sdf_fc1`
5. 可选训练 `fc2`

关键点：

- `modeb` 的 `policy_value` 和 `sdf_fc1` 都来自 `SimulateTS(h=T)`
- 这里的样本量是时间维展开后的 panel 规模，不再是单次 `Sample` 的小截面

## `Sample` 数据是怎么生成的

见 [`data/sample.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/data/sample.py) 和 [`data/sample_parallel.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/data/sample_parallel.py)。

对每条 path：

1. 先生成当前期 `parent` firms
2. 每家 firm 复制出 `branch_num` 个 `t+1` children
3. 组成 `(parent, child_0, child_1, ...)` 对齐结构

默认参数下：

- `group_size = 2`
- `branch_num = 2`

因此每条 path 原始 firm rows 数是：

```text
rows_per_path = group_size x (1 + branch_num) = 2 x 3 = 6
```

如果 `n_paths = 1000`，那么 `Sample` firm rows 大约是：

```text
1000 x 6 = 6000 rows
```

而可训练 `policy_value` 单元数是 parent firms 数：

```text
1000 x 2 = 2000 units
```

所以在 `Sample` 路径下，`1000 path` 绝不会对应上百万个 `policy_value` 训练单元。

## `SimulateTS` 数据是怎么生成的

见 [`data/simulate_ts_parallel.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/data/simulate_ts_parallel.py)。

对 `modeb`，流程是：

1. 初始化 `n_paths x simulate_group_size` 家 firm
2. 每个时间步先记录一次 parent 节点
3. 再扩展 `branch_num` 个分支节点
4. 每个分支上执行：
   - 宏观状态推进
   - 债务状态推进
   - 资本状态推进
   - 进入/退出
5. 进入下一时间步，重复直到 `horizon`

默认参数下：

- `simulate_group_size = 200`
- `branch_num = 2`
- `simulate_horizon = 10`

忽略进入退出时，单看原始节点规模：

```text
每期节点数 = 1 parent + 2 branches = 3
每期 firm rows ≈ n_paths x 200 x 3
总 firm rows ≈ n_paths x 200 x 3 x horizon
```

如果 `n_paths = 1000`、`horizon = 10`，粗量级就是：

```text
1000 x 200 x 3 x 10 = 6,000,000 rows
```

这还是不含 entry 带来的新增 firms。

因此在 `modeb` 下，`policy_value` 训练数据达到百万级单元是正常现象。

## `policy_value` 训练单元是怎么从数据里抽出来的

见 [`training/episode.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/training/episode.py)：

- DataFrame 路径：`_create_firm_batches_from_df`
- Tensor 路径：`_create_firm_batches_from_tensor`

核心逻辑都是：

1. 找到 parent rows
2. 找到每个 parent 对应的全部 branch children
3. 只保留 child 齐全的 parent
4. 生成训练单元：
   - `parent`
   - `child0`
   - `child1`

所以 `policy_value` 的训练样本数不是 raw row 数，而是：

```text
matched parent-child units
```

这也是为什么：

- `Sample` 下通常只有几千个 unit
- `modeb` 的 `SimulateTS(h=T)` 下会到几十万甚至上百万个 unit

## 当前正确的直觉

- `mode0` / `modea` 的 `policy_value` 数据是小截面 `Sample`
- `modeb` 的 `policy_value` 数据是长时间 `SimulateTS` panel
- 所以同样 `n_paths = 1000`
  - 在 `Sample` 路径下，数据量是千级到万级
  - 在 `modeb` 下，数据量是百万级，很正常

## 建议

如果目标是控制 `modeb` 的训练规模，优先调这三个量：

- `--n-paths`
- `--simulate-group-size`
- `--simulate-horizon`

其中：

- `Sample` 的训练规模主要由 `n_paths` 和 `sample_group_size` 决定
- `SimulateTS` 的训练规模主要由 `n_paths x simulate_group_size x horizon` 决定
