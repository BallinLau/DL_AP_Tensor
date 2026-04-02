# Macro Law Consistency Diagnostic Definition

## 背景

当前项目中的 `sdf_fc1` 不只是一个一般意义上的预测网络，而是外层 fixed-point 迭代中给定的 aggregate consumption law 近似器。

因此，`macro_hatc` 诊断的核心问题不是：

- “普通 one-step 预测是否有高 R^2”

而是：

- “给定当前固定的 `sdf_fc1` aggregate consumption law，firm-side `policy/value + simulate` 是否能生成一个与该 law 一致的 implied aggregate transition”

这属于 **aggregate law-of-motion consistency** 检验，而不是两条连续时间 SDE 的比较。

## 要检验的对象

记：

- `g_theta(X_t)`：当前固定 `sdf_fc1` 在状态 `X_t` 下给出的 aggregate consumption law 预测
- `hatc'_{sim}`：在该 law 下求解 firm problem 并 simulation 后得到的下一期 implied aggregate consumption object

我们真正关心的是：

```text
hatc'_{sim} ?= g_theta(X_t)
```

更严格地说，关心的是：

```text
E[hatc'_{sim} - g_theta(X_t) | X_t] = 0
```

如果成立，说明当前固定的 aggregate law 与 firm-side response 是自洽的。

## 为什么单独看 R^2 不够

现有 `R^2(Hatc)` 会把以下几种误差混在一起：

- law 方向本身不一致
- 斜率错误（slope != 1）
- 截距错误（intercept != 0）
- 预测方差塌缩（std ratio << 1）
- 少量极端点带来的 SSE 爆炸

因此，`R^2` 可以保留，但只能作为一个粗诊断，不应单独作为 outer convergence metric。

## 新诊断口径

对 `Hatc` 和 `LnK`，同时输出：

- `r2`
- `corr`
- `slope`
- `intercept`
- `mean_resid`
- `mae_resid`
- `rmse_resid`
- `std_ratio`

其中：

- `resid = realized - forecast`
- `slope/intercept` 来自回归 `realized = intercept + slope * forecast`
- `std_ratio = std(forecast) / std(realized)`

这些统计量用于区分：

1. law 本身偏离
2. 尺度/方差塌缩
3. 线性响应过强或过弱

## branch 口径

为了避免 parent branch `-1` 混入口径，额外输出：

- `*_branch01`

这表示只在 simulated child branches `0/1` 上计算的统计量。

这样可以区分：

- 全样本口径下的总体拟合
- 纯 simulation-implied aggregate transition 口径下的一致性

## 解释方式

### 情况 A：`corr` 高，但 `std_ratio` 很低

说明方向大致对，但预测方差塌缩；这时 `R^2` 可能依然很差。

### 情况 B：`corr` 高，`slope` 显著偏离 1

说明 law 的响应强度不对，可能是 aggregate block 尺度或灵敏度错误。

### 情况 C：`corr` 低，`rmse` 高

说明当前固定的 aggregate law 与 simulation implied law 在结构上不一致。

## 与收敛判断的关系

这个诊断回答的是：

- “固定当前 `sdf_fc1` law 时，firm-side implied aggregate 是否与其一致”

它是 outer fixed-point 诊断的重要组成部分，但仍然不是完整的收敛标准。

完整 outer convergence 后续仍应结合：

- Bellman residual
- episode 间 function drift
- simulated moments drift
- 必要时的 cross-seed robustness

## 代码落点

- `training/episode.py`
  - `_macro_forecast_r2`
  - `_macro_forecast_r2_tensor`
- `experiments/run_utils.py`
  - `plot_macro_series`

本轮修改保留原有 `R^2`，并追加上述 richer diagnostics。
