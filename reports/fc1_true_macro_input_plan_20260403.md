# FC1 Current-State Contract Update Plan (2026-04-03)

## 背景

前面的 standalone probe 说明：

- 用低维当前 aggregate state  
  `x_t, x_{t+1}, Hatc_t, LnK_t`
  预测下一期 `Hatc_{t+1}, LnK_{t+1}` 已经足够好。
- 再加入简单横截面 summary  
  `b_mean_t, b_std_t, z_mean_t, z_std_t`
  只带来极小改进。

因此，当前 `FC1` 的主要问题不再像是“状态维度不够”，而更像是：

- 系统里递归使用了 `hatcf_t, lnkf_t` 这组 proxy current state
- 这会把 proxy 误差继续往下传

## 新的目标

把 `FC1` 的 current input 从：

```text
[x_t, x_{t+1}, hatcf_t, lnkf_t]
```

改成：

```text
[x_t, x_{t+1}, Hatc_t, LnK_t]
```

输出仍然是：

```text
[hatcf_{t+1}, lnkf_{t+1}]
```

含义：

- `Hatc_t, LnK_t` 是当前节点真实 aggregate state
- `hatcf_{t+1}, lnkf_{t+1}` 是下一节点 macro proxy

## 关键原则

### 1. sample 不生成真实 aggregate

`sample_group_size=2` 保留不动。

sample 的职责是：

- 提供 firm-side 训练样本
- 不负责从 2 家公司里聚合“真实” `Hatc/LnK`

因此 sample 中 `policy_value` 需要的 current macro input，不应来自小组内聚合。

### 2. sample 读取当前 outer loop 的 node-level macro state

sample 阶段 parent 当前 macro state 的来源改为：

- 上一轮 simulate 导出的 parent macro 表
- 按 node 抽样得到 `(x_t, Hatc_t, LnK_t)`

也就是说，sample 是“读取当前宏观状态”，不是“重新生成当前宏观状态”。

### 3. simulate 才是 realized macro 的来源

realized `Hatc, LnK` 继续由 simulate 大截面产生。

它承担两件事：

- 为下一轮 sample 提供 current macro lookup
- 为 `FC1` stage2 提供训练 pair

## 具体修改

### A. sample 阶段

在 `Sample` 中新增可选的 `macro_source_df`。

当 `macro_source_df` 可用时：

- 从其中抽样 parent 节点（`branch == -1`）
- 读取该节点的：
  - `x`
  - `Hatc`
  - `LnK`
- 把它们写进 sample parent 的 current macro 槽位

这样 sample parent 的 `Hatcf/LnKF` 列，虽然名字暂时不变，但内容解释为：

```text
current node macro state = (Hatc_t, LnK_t)
```

children 仍由 `FC1` 给出下一节点 proxy：

```text
FC1(x_t, x_{t+1}, Hatc_t, LnK_t) -> (hatcf_{t+1}, lnkf_{t+1})
```

### B. FC1 stage2

在 SDF stage2 / macro-pair 训练里，当前态输入统一切到真实 current macro：

```text
(x_t, x_{t+1}, Hatc_t, LnK_t) -> (hatcf_{t+1}, lnkf_{t+1})
```

不再默认用 `(Hatcf_t, LnKF_t)` 做 current input。

### C. episode 间 macro lookup

每轮 episode 结束后，把本轮 simulate 导出的 `df_macro` 保存为下一轮 sample 的 `macro_source_df`。

于是：

- `ep0`：没有上一轮 macro source，继续用初始化 guess
- `ep1+`：sample 读取 `ep-1` 的 simulate parent macro

### D. simulate 递归口径也统一

在 simulate 树里，节点 `t` 经过 `policy/value + aggregation` 后会得到 realized：

- `Hatc_t`
- `LnK_t`

下一次 branch expansion 时，`FC1` 的 current input 不再用上一轮 proxy：

```text
[x_t, x_{t+1}, hatcf_t, lnkf_t]
```

而改成当前节点实现出来的：

```text
[x_t, x_{t+1}, Hatc_t, LnK_t]
```

同时仍然保留节点上记录的 `hatcf/lnkf`，用于宏观诊断和比较，不把它们覆盖成真值。

## 本轮不改的东西

- 不改 `sample_group_size`
- 不让 sample 从小组内聚合真实 `Hatc/LnK`
- 不改 `policy_value` 输入接口名字
- 不动 `FC2`

## 预期结果

如果这个方向对，应该看到：

- `macro_hatc` 的 `corr / slope / std_ratio` 改善
- `x -> hatcf_pred` 更接近 `x -> Hatc_true`
- 不再需要继续往 `FC1` 里塞横截面 summary

## 仍然保留的 caveat

这一步优先修的是：

- current macro input 的口径

它还不能单独回答：

- `FC1` 和 `SDF` 的联合目标是否仍然冲突
- `FC2` 是否更适合承担 aggregate closure
