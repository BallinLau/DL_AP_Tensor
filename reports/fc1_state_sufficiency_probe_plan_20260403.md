# FC1 状态充分性 Probe 实验计划（2026-04-03）

## 背景

当前 `FC1` 在系统中的角色是：

- 低维、前馈的 branch-conditioned expectation block
- 输入信息集大致为：
  - `x_t`
  - `x_{t+1}`
  - `hatcf_t`
  - `lnkf_t`
- 输出：
  - `hatcf_{t+1}`
  - `lnkf_{t+1}`

目前出现的问题是：

- `x -> hatcf_pred`
- `x -> Hatc_true`

之间仍然明显割裂。

但此时还不能直接断言“状态不足”，因为至少还混着两类可能性：

1. `FC1` 的状态确实不充分。
2. `FC1` 在系统内和 `SDF`、`policy/value`、`simulate` 的耦合导致训练目标冲突。

因此，继续在主系统里直接改 `FC1`，很难得到干净结论。

---

## 核心想法

先做一个 **系统外的 standalone probe**，只回答一个问题：

> 在足够大的 simulate 横截面上，当前 `FC1` 的低维 state 是否已经足够预测下一期 aggregate outcome？

也就是把问题从：

```text
FC1 + SDF + policy/value + simulate
```

拆开成一个单独的监督学习识别实验。

---

## 要识别的问题

这次 probe 只想区分：

### 假设 A：状态不足

如果在同一批 simulate 数据上，

- 低维输入 probe
- 分布增强输入 probe

相比之下，后者明显更好，

则说明：

- `FC1` 当前的低维 state 不是充分统计量
- 分布 summary 确实包含额外信息

### 假设 B：状态不是主问题

如果加入分布 summary 后提升很小，
则更支持：

- `FC1` 的主要问题不是 state 不足
- 而是系统内训练耦合 / 损失冲突 / 其他实现问题

---

## 实验设计

### 数据来源

只使用 **simulate 产生的较大横截面数据**。

不使用 sample 小 group 数据做这个 probe。

原因：

- sample 里的 group 很小
- 用它构造 `b_mean/std`、`z_mean/std` 容易噪声过大
- 会污染“状态充分性”判断

因此，这个 probe 的训练数据应该来自：

- `SimulateTS`
- 或 episode 过程中保存的 `df_macro / df_firm`
- 并且优先用 group 足够大的 simulate 结果

当前已有的 stage 输出就可以直接用，例如：

- `ep30_stage_modea.pkl`
- `ep30_stage_modea_macro.pkl`

只要它们来自 `modea` / `SimulateTS` 阶段，而不是 sample 小 group 阶段。

具体口径：

- `macro.pkl` 用来构造跨期 pair
- `firm.pkl` 只用 parent 节点横截面来计算 `t` 时点 summary

---

## 两个 probe

### Probe 1：低维 baseline

输入：

```text
[x_t, x_{t+1}, hatc_t, lnk_t]
```

输出：

```text
[Hatc_{t+1}, LnK_{t+1}]
```

用途：

- 复现当前 `FC1` 的低维信息集能力上限

### Probe 2：分布增强版

输入：

```text
[x_t, x_{t+1}, hatc_t, lnk_t, summary_t]
```

其中 `summary_t` 第一版建议只用最小集合：

```text
[b_mean_t, b_std_t, z_mean_t, z_std_t]
```

输出同样是：

```text
[Hatc_{t+1}, LnK_{t+1}]
```

用途：

- 检验这些当前时点分布 summary 是否提供了显著额外信息

---

## 为什么这是更干净的识别

这个 probe **不进入主系统闭环**，因此不会混入下面这些因素：

- `SDF` Euler / moment 目标
- `policy/value` 的 Bellman/FOC/KKT 目标
- `FC1` 在 stage1/stage2 的口径切换
- sample 小截面 summary 噪声
- “修改 FC1 会不会改变其经济角色边界”的争议

也就是说，它只回答：

> 对于 aggregate law 预测本身，分布 summary 有没有信息增量？

---

## 评价指标

两个 probe 的比较，至少看这几类：

### 1. 标量指标

- `corr_hatc`
- `slope_hatc`
- `std_ratio_hatc`
- `rmse_hatc`
- `r2_hatc`

- `corr_lnk`
- `slope_lnk`
- `std_ratio_lnk`
- `rmse_lnk`
- `r2_lnk`

### 2. 条件响应

看：

- `x -> pred`
- `x -> true`

必要时也看：

- `summary_t -> residual`

### 3. 条件残差

定义：

```text
resid_hatc = Hatc_true - Hatc_pred
resid_lnk  = LnK_true  - LnK_pred
```

检查：

- `E[resid | x_t]`
- `E[resid | x_{t+1}]`
- `E[resid | hatc_t]`
- `E[resid | summary_t]`

如果 baseline probe 的残差对 `summary_t` 有系统模式，而 augmented probe 明显减弱，
这就是“状态不足”的强证据。

---

## 判别逻辑

### 若 augmented probe 明显优于 baseline probe

可得出：

- 当前 `FC1` 的低维状态不充分
- 分布 summary 对 aggregate law 预测确实有边际信息

这时后续再决定：

1. 是否要扩充 `FC1`
2. 还是直接推进 `FC2`

### 若 augmented probe 提升很小

更支持：

- 状态不足不是主问题
- 当前 `FC1` 的主要问题来自系统内训练耦合或目标冲突

这时更应优先考虑：

- 继续做 `FC1-SDF` 解耦识别
- 或者把 aggregate law 主线转交给 `FC2`

---

## 与主系统的关系

这个 probe **不是** 要替代主系统，也不是直接改 `FC1`。

它的作用是：

- 在不污染主系统设计的前提下
- 单独识别“状态变量是否充分”

所以它的结论是一个 **设计决策依据**，不是最终模型组件。

---

## 下一步实现建议

### 第一步

实现一个独立脚本，例如：

```text
experiments/run_fc1_state_probe.py
```

输入：

- 某一轮 episode 的 simulate 结果
- `df_macro`
- `df_firm`

功能：

1. 构造 baseline 数据集
2. 构造 augmented 数据集
3. 训练两个小 MLP probe
4. 输出指标与图

### 第二步

先跑单个 episode / 单个 simulate dump，
不接入训练主循环。

### 第三步

如果 probe 结果支持“状态不足”，
再回到系统层讨论：

- 扩充 `FC1`
- 或使用 `FC2`

---

## 当前结论

在现阶段，最干净的做法不是继续直接改系统内 `FC1`，
而是先做这个 standalone probe。

因为只有这样，才能把：

- 状态不足
- 训练耦合
- 闭环污染

这三件事分开。
