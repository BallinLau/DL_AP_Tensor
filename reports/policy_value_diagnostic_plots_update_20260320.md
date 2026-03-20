# `PI/P0` 诊断图与最终模拟 `b` 分布图补充说明（2026-03-20）

## 修改背景

在当前训练结果下，出现了两个需要进一步拆解的现象：

1. `bar_i` 热图偏向高 `b`、低 `z`
2. 低 `b`、高 `z` 的安全区域反而投资较弱

仅看 `bar_i` 本身，无法判断这是：

- 当期现金流差导致的
- continuation 差导致的
- 还是两者共同作用

因此需要把 `PI - P0` 进一步分解成可视化诊断图。

## 本次代码修改

涉及文件：

- [experiments/run_utils.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/experiments/run_utils.py)
- [experiments/run_multi_episode.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/experiments/run_multi_episode.py)
- [experiments/run_multi_episode_job.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/experiments/run_multi_episode_job.py)
- [experiments/run_episode0_full.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/experiments/run_episode0_full.py)

### 1. `plot_surfaces()` 新增三类诊断图

在原有 `p0/pi/p/bar_i/bar_z/bp/q` 截面图之外，新增：

#### `pidiff`

```math
PI - P0
```

直接对应 `bar_i = sigmoid(10(PI-P0))` 的驱动项。

#### `cfdiff`

```math
CFip - CF0p
```

表示在当前期现金流口径下，投资相对不投资的增量价值。

#### `contdiff`

```math
g M P'_I (1-\bar z'_I) - M P'_0 (1-\bar z'_0)
```

表示 continuation 项在投资与不投资之间的差异。

### 2. 诊断图的用途

这三张图用于回答：

- 高 `b`、低 `z` 的 `bar_i` 是否主要由 `cfdiff` 驱动
- 低 `b`、高 `z` 的“不投资”是否因为 `cfdiff < 0` 且 `contdiff` 不够大

也就是说，不再只看 `bar_i` 最终结果，而是把它拆回：

```math
PI - P0
=
(CFip - CF0p)
+
\left[g M P'_I (1-\bar z'_I) - M P'_0 (1-\bar z'_0)\right]
```

## 训练后最终模拟的 `b` 分布图

除 `(b,z)` 诊断图外，本次还补充了训练结束后 `horizon=200` 最终模拟的中段债务分布图：

- 文件名：
  `final_b_hist_t50_t150_parent.png`

口径为：

- 使用 final simulate 的 `firm panel`
- 只取 `parent firm states`
- 只保留 `50 <= t <= 150`

这样做的目的，是避免：

- 把 child forecast 分支混入 `b` 分布
- 被最初期和最后期的过渡状态干扰

从而更接近“中段稳定时点”的债务横截面。

## 为什么用 `parent firm states`

`b` 的中段分布如果混入 child branch，会受到 forecast 分支展开的重复影响。

因此这里选择：

- parent-only
- 中段时点窗口

作为更稳健的分布口径。

## 预期收益

本次修改不改变训练目标，只增强诊断能力：

1. 把 `bar_i` 的异常区域拆解为“当期现金流差”与“continuation 差”
2. 单独观察最终模拟中段时点的 `b` 分布
3. 为下一步判断应该改 `PI/P0` 哪一段、还是继续改 `M/FC1` 提供直接证据
