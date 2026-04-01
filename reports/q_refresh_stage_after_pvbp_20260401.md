# 在 PVBP 后增加 Q Refresh Stage

日期：2026-04-01

## 背景

最新一轮结果表明：

1. `Q stage` 末尾的 `Q` 通常已经比较健康。
2. 但经过 `PVBP stage` 后，`chi / bar_z / bar_i` 被明显改写。
3. `Q` 方程右侧依赖这些 gate，所以冻结的 `Q` 会和更新后的 `PVBP` 失配。

典型表现是：

- `q_stage_end` 的 `q_main` 较低
- 但最终 `pvbp_stage_end` 的 `q` Bellman residual 又重新变大

这说明问题不只是 `Q` 训练不够，而是 `Q` 与后续更新后的 `PVBP` 缺少重新对齐。

## 修改思路

把原来的两阶段：

1. `Q stage`
2. `PVBP stage`

改成三阶段：

1. `Q stage`
2. `PVBP stage`
3. `Q refresh stage`

其中第三阶段只更新 `Q` 路径，目的是让 `Q` 在 `PVBP` 已经收敛到的新 gate/value 面上重新定价。

## 实现

在 [`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/training/episode.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/training/episode.py) 的 `_run_batches(...)` 中：

- 新增 `q_refresh_stage_epochs`
- 阶段顺序改为：
  - `epoch < q_stage_epochs`: `q_only`
  - `q_stage_epochs <= epoch < q_stage_epochs + pvbp_stage_epochs`: `pvbp`
  - `epoch >= q_stage_epochs + pvbp_stage_epochs`: `q_refresh`

`q_refresh` 使用与 `q_stage` 相同的 `policy_loss_terms=['q']`，因此仍然只更新 `Q`。

## 默认设置

在 [`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/config/hyperparams.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/config/hyperparams.py) 中新增：

- `q_refresh_stage_epochs = 20`

即默认训练顺序是：

1. `Q stage`: 100 epoch
2. `PVBP stage`: 100 epoch
3. `Q refresh stage`: 20 epoch

## 新的阶段落盘

现在 `policy stage summaries` 会额外记录：

- `q_stage_end`
- `pvbp_stage_end`
- `q_refresh_end`

这样可以直接比较：

1. `Q` 在第一阶段是否学好
2. `PVBP` 更新后是否把 `Q` 弄失配
3. `Q refresh` 是否把这种失配重新拉回去

## 预期观察点

如果这个改动有效，应看到：

1. `q_refresh_end` 的 `q_main` 低于 `pvbp_stage_end`
2. 最终 `q` convergence 优于没有 refresh 的版本
3. `Q surface` 在高 `z` 区不再明显过于乐观
4. 不破坏已经改善的 `bp` 对齐结果
