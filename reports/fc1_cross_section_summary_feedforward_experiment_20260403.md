# FC1 横截面 Summary 前馈实验计划（2026-04-03）

## 目标

做一个最小实验，检验下面这个问题：

> `FC1` 现在效果差，是否部分来自输入 state 过弱？

这轮实验**不改变 `FC1` 的功能定位**：

- 仍然把 `FC1` 看成前馈的 branch-conditioned expectation block
- 不把它改成闭环/固定点求解器
- 不把 `FC2` 的职责塞进 `FC1`

## 实验原则

### 保持 `FC1` 仍然是前馈器

`FC1` 继续做：

```text
current aggregate state + next branch macro shock + current summary
    -> next-node macro proxy
```

而不是做：

```text
proxy = Aggregate(PolicyValue(proxy ; state))
```

后者仍然属于 `FC2` / 闭环块的职责。

### 只加最小 summary，不做“大 FC2 化”

这轮只加 4 个当前时点横截面 summary：

- `b_mean_t`
- `b_std_t`
- `z_mean_t`
- `z_std_t`

原因：

- 都是当前节点可观测信息
- 不直接使用 policy-derived summary
- 足够小，便于识别“state 不足”这个假设

## 输入的时期口径

这次 `FC1` 的输入扩展为：

```text
[x_prev, x_curr, hatcf_prev, lnkf_prev, b_mean_t, b_std_t, z_mean_t, z_std_t]
```

它们对应的时期和含义如下：

| 输入名 | 时期 | 含义 |
|---|---|---|
| `x_prev` | `t` | 当前 parent 节点的宏观 shock，等价于 `x_t` |
| `x_curr` | `t+1` | 当前 child branch 上已知的下一期宏观 shock，等价于 `x_{t+1}` |
| `hatcf_prev` | `t` | 当前节点已有的宏观 proxy `\hat c_t` |
| `lnkf_prev` | `t` | 当前节点已有的宏观 proxy `\ln K_t` |
| `b_mean_t` | `t` | 当前 parent 节点横截面中 `b` 的均值 |
| `b_std_t` | `t` | 当前 parent 节点横截面中 `b` 的标准差 |
| `z_mean_t` | `t` | 当前 parent 节点横截面中 `z` 的均值 |
| `z_std_t` | `t` | 当前 parent 节点横截面中 `z` 的标准差 |

所以这次实验对应的信息集是：

```text
(x_t, x_{t+1}, \hat c_t, \ln K_t, summary_t)
```

其中：

```text
summary_t = (b_mean_t, b_std_t, z_mean_t, z_std_t)
```

而 `FC1` 预测的对象仍然是：

```text
(\hat c_{t+1}, \ln K_{t+1})
```

## 代码层修改思路

### 1. 扩展 FC1 输入维度，但保持前馈结构不变

`FC1` 输入从：

```text
[x_prev, x_curr, hatcf_prev, lnkf_prev]
```

变成：

```text
[x_prev, x_curr, hatcf_prev, lnkf_prev, b_mean_t, b_std_t, z_mean_t, z_std_t]
```

说明：

- `FC1` 网络结构仍然是 MLP 前馈
- 新增 summary 只是额外输入维度
- 如果某条数据流拿不到 summary，就自动补零

### 2. 在 SDF/FC1 训练 batch 中单独携带 `fc1_summary_prev`

不去打乱现有：

- `parent`
- `children`

的列布局。

而是在 batch dict 里新增：

- `fc1_summary_prev`

这样：

- 现有 `Hatc_t / LnK_t / Hatc_t1 / LnK_t1` 列位置不变
- `_compute_sdf_loss()` 只额外读取一个可选字段

### 3. 训练和推理统一口径

不仅训练阶段 `_compute_sdf_loss()` 会把 `summary_prev` 传给 `FC1`，下面这些用 `FC1` 做下一期 macro proxy 的地方也同步改：

- `data/sample_parallel.py`
- `data/simulate_ts_parallel.py`
- `data/simulate_ts.py`

这样避免出现：

- 训练时 `FC1` 看到了 summary
- simulate 时却没看到

## Summary 的来源

### 训练时

优先从当前时点 parent firm set 构造 summary：

- tensor pipeline：从 `self.tensor_firm`
- dataframe pipeline：从 `self.df`

### simulate/sample 推理时

直接从当前状态里现有的 firm arrays 计算：

- `b`
- `z`
- `alive mask`（如果有）

## 这轮不做什么

- 不给 `FC1` 加 policy-derived summary
- 不引入 `bar_z / bar_i / bp / Q / P` 等输入
- 不改 `FC1` 的 loss 设计
- 不把 `FC1` 改成闭环模块
- 不碰 `FC2`

## 判断标准

这轮如果有效，最先应该改善的是：

- `macro_hatc_branch01`
  - `corr_hatc`
  - `slope_hatc`
  - `std_ratio_hatc`
- `macro_hatc_vs_x`
  - `pred` 曲线更接近 `true`

不要求第一轮就把 `R²` 拉高很多，但至少要看到：

- `x -> hatcf_pred` 的方向和振幅更接近 `x -> Hatc_true`

## 如果这轮无效，说明什么

如果加了当前横截面 summary 之后，`FC1` 仍几乎没有改善，那么更支持：

- 不是简单的 state 不足
- 而是：
  - `FC1` 与 `SDF` 的目标冲突更严重
  - 或者 `FC1` 作为低维前馈器本身就不适合承担这条 law

那时就更有理由把 aggregate law closure 主线彻底转给 `FC2`。
