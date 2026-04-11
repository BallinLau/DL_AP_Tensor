# FC2 `sum C` 诊断接入说明

## 1. 目的

这轮修改的目标不是继续调 `FC2` 网络，而是先把 `hatc` 背后的资源对象看清楚。

核心问题是当前主线里一直在用：

\[
C_j = Y_j - I_j - \Phi_j
\]

然后再构造：

\[
C^{firmclip} = \sum_j \max(C_j, 0)
\]

以及：

\[
\hat c = \log\left(\frac{C^{firmclip}}{K + 10^{-8}} + 10^{-5}\right)
\]

如果要判断这个对象是否合理，第一步必须直接看：

\[
C^{raw} = \sum_j C_j
\]

也就是逐 path / node 的原始 `sum C` 分布，而不是只盯 `hatc`。

---

## 2. 新增诊断对象

这次在 simulate 和 FC2 评估链上统一加入了三套 consumption aggregation：

1. 原始聚合：

\[
C^{raw} = \sum_j C_j
\]

2. 现有主线使用的 firm-level clipping：

\[
C^{firmclip} = \sum_j \max(C_j, 0)
\]

3. aggregate-level clipping：

\[
C^{aggclip} = \max\left(\sum_j C_j, 0\right)
\]

并新增：

\[
\hat c^{firmclip}
=
\log\left(\frac{C^{firmclip}}{K + 10^{-8}} + 10^{-5}\right)
\]

\[
\hat c^{aggclip}
=
\log\left(\frac{C^{aggclip}}{K + 10^{-8}} + 10^{-5}\right)
\]

额外记录：

- `C_raw_over_K`
- `C_pos_mass`
- `C_neg_mass`
- `neg_c_mass_share`
- `feasible_raw = 1{C_raw > 0}`

---

## 3. 接入位置

### 3.1 simulate 输出

已在以下文件中加入新列：

- `data/simulate_ts.py`
- `data/simulate_ts_parallel.py`

`MACRO_COLUMNS` 现在包含：

- `C_raw`
- `C_firmclip`
- `C_aggclip`
- `Hatc_firmclip`
- `Hatc_aggclip`
- `C_raw_over_K`
- `C_pos_mass`
- `C_neg_mass`
- `neg_c_mass_share`
- `feasible_raw`

当前 `Hatc` 主列仍保持为旧主线口径：

\[
Hatc \equiv Hatc^{firmclip}
\]

这样不会直接打断现有训练和评估。

### 3.2 FC2 summary / diagnostics

已在以下文件中接入：

- `losses/FC2losspipe.py`
- `training/episode.py`

新增两套 summary 诊断：

1. `current_macro_consumption_diag`
   - 对应 `FC2` 训练前、当前 episode simulate 出来的 macro panel

2. `outer_after_resim` 内的 consumption diagnostics
   - 对应训练后再跑一遍 resim 的 macro panel

其中最关键的指标是：

- `fc2_current_macro_consumption_feasible_path_share`
- `fc2_current_macro_consumption_mean_c_raw_over_k`
- `fc2_current_macro_consumption_mean_neg_c_mass_share`
- `fc2_outer_after_resim_macro_consumption_feasible_path_share`

---

## 4. 新增图

已在以下文件中接入自动画图：

- `experiments/run_utils.py`
- `experiments/run_multi_episode.py`
- `experiments/run_multi_episode_job.py`

每个 episode 现在会自动画：

1. `ep{ep}_fc2_before_train_c_raw_distribution.png`
2. `ep{ep}_fc2_outer_after_resim_c_raw_distribution.png`

这两张图都画的是：

\[
C^{raw} = \sum_j C_j
\]

并且分成：

- `parent`
- `children`

两个面板，避免把 parent 和 child 混在一起看不清。

图上同时标：

- `n`
- `mean`
- `p05 / p50 / p95`
- `feasible`
- `neg_share`

---

## 5. 这轮修改的意义

这轮修改不试图直接回答“`hatc` 应该怎么学”，而是先把更基础的问题可视化：

1. `sum C` 在 path-level 上到底经常是正还是负；
2. 当前 `firm-level clip` 和 `aggregate-level clip` 差距有多大；
3. `hatc` 难学到底是网络问题，还是目标对象本身已经被 clipping 改形了。

如果后续发现：

\[
P(C^{raw} > 0)
\]

非常低，那么主问题就不在 `FC2`，而在当前 policy / aggregate 组合下产生了大量 aggregate-infeasible path。

---

## 6. 当前策略

这次先做“诊断接入”，不直接切换主线定义。

也就是说：

- 训练主线暂时仍用 `Hatc_firmclip`
- 但所有关键诊断已经能同时看到：
  - `raw`
  - `firmclip`
  - `aggclip`

后续再决定是否把主线 target 从 `firmclip` 切到 `aggclip`。
