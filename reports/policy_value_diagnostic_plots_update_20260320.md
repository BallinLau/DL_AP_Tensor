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
g\,M\,P_{t+1}(bp_I)\,(1-\bar z_{t+1}(bp_I))
-
M\,P_{t+1}(bp_0)\,(1-\bar z_{t+1}(bp_0))
```

这里的 `P_{t+1}` 是下一期总股权价值 `P`，不是 `P^I_{t+1}` 或 `P^0_{t+1}` 分头。

也就是说，这张图比较的是：

- 投资分支对应的 child state（由 `bp_I` 诱导）下的下一期总股权 continuation
- 不投资分支对应的 child state（由 `bp_0` 诱导）下的下一期总股权 continuation

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
\left[
g\,M\,P_{t+1}(bp_I)\,(1-\bar z_{t+1}(bp_I))
-
M\,P_{t+1}(bp_0)\,(1-\bar z_{t+1}(bp_0))
\right]
```

## 理论口径补充：为什么不是乘当前期 `(1-\bar z_t)`

理论里的 `P^0` 和 `P^I` Bellman 方程都是**条件于当前期存活**的价值。

因此：

- 当前期是否 default，由当期总股权价值
  ```math
  P_t = \max\{0, \int \max(P_t^0, P_t^I)\,dH(i)\}
  ```
  这一层处理；
- Bellman continuation 里乘的是**下一期 survival**，也就是
  ```math
  (1-\bar z_{t+1})
  ```
  而不是当前期 `(1-\bar z_t)`。

这与理论文稿和当前实现是一致的。

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

## 后续可视化修正：只显示当前期存活区域

进一步讨论后确认：

- 高 `b`、低 `z` 区域中有不少状态在当前期其实已经应当 default
- 这些状态虽然能在静态网格上算出 `bar_i`，但在真实时序模拟中并不会进入下一期继续执行投资决策

因此，当前 `(b,z)` 投资相关截面图已增加 current-survival mask。

具体来说，以下图形现在只在当前存活区域显示：

- `bari`
- `bp`
- `pidiff`
- `cfdiff`
- `contdiff`

mask 条件为：

```math
P_t > 0
\quad\text{and}\quad
\bar z_t < 0.5
```

这样可以避免把“当前期已经 default 的状态”错误解读为真实投资区。
