# Q 与 P 训练拆分方案（基于当前理论口径）

Date: 2026-03-30

## 1. 结论

根据当前理论文档与实验结果，`Q` 不应与 `P / V / bp` 做联合训练。

更准确地说：

- `Q` 应作为独立的债券定价对象单独训练
- `V0 / VI / bar_i_cond / bp0 / bpI` 可以归为同一组训练
- `Vhat / chi / bar_z / P / bar_i` 更适合作为 value block 的派生量，而不是再与 `Q` 并列做联合学习

因此，当前 `PolicyValueModel` 里把 `Q` 与 `bp0 / bpI` 放在同一个 `SharedModel`、再把 `Q` 与 `V0 / VI / bar_i_cond` 放进同一个总模型统一反传，这在理论上和训练上都不稳。

---

## 2. 理论依据

### 2.1 `Q` 是独立的债券价格对象

`Q` 对应的是论文中的债券定价对象：

```math
B(k,\tilde b,z,S)
```

这里第二个自变量 `\tilde b` 是当期选择的债务 contract。

在论文口径下：

- 当前时点定价的是同一个 contract
- future continuation 里延续定价的也应是同一个 contract
- `\eta'` 只影响 future integration / default region，不应改写 contract 本身

但当前实现的 `Q` 对象并不统一：

1. 当前 `Q_t` 绑定在 `b_parent`
2. 普通 child state 使用 `\eta bp + (1-\eta)b_parent`
3. `Qsp_children` 使用 `b_parent / (\bar i(g-1)+1)`

因此当前 `Q` loss 学的不是单一理论对象，而是混合 debt mapping 下的数值自洽物。

这意味着：

- `Q` 本身已经是一个需要单独理顺对象定义的模块
- 它不适合再与股权价值块共享训练路径

---

### 2.2 `V0 / VI / bar_i_cond` 属于同一个股权条件价值块

按当前 value refactor 文档：

- `V0`: 当前仍存活条件下，不投资价值
- `VI`: 当前仍存活条件下，投资价值
- `bar_i_cond = sigmoid(10 * (VI - V0))`

这三者描述的是同一层经济对象：

```text
给定“当前仍存活”这一条件下，企业应当如何比较投/不投两条股权 continuation value
```

因此它们可以一起训练，而且应该共享表征。

---

### 2.3 `Vhat / chi / bar_z / P / bar_i` 是派生量，不宜再作为独立主对象与 `Q` 混训

当前理论口径已经把这些量写成 value block 的导出对象：

- `Vhat = E_i[max(V0, VI)]`
- `chi`: 当前期软存活门
- `bar_z = 1 - chi`
- `P`: 无条件总股权价值
- `bar_i = chi * bar_i_cond`

这说明它们本质上属于 survival / equity block，而不是 bond-pricing block。

尤其是：

- `chi` 和 `bar_z` 属于 default / survival gating
- `P` 属于由 conditional value 导出的总股权值
- `bar_i` 属于执行层投资权重

这些对象都不应反过来去主导 `Q` 的共享表示。

---

### 2.4 `bp0 / bpI` 属于股权决策块，而不是债券价格块

`bp0 / bpI` 是企业在不投资/投资两种情形下的杠杆选择候选。

从经济上看，它们回答的问题是：

```text
在给定定价环境下，企业想选什么杠杆 contract
```

而 `Q` 回答的是：

```text
给定某个 contract，这个 contract 的市场价值是多少
```

所以：

- `bp0 / bpI` 是 choice / policy objects
- `Q` 是 pricing object

二者当然有相互作用，但不应在同一个共享表征里同时学习。

更合理的关系是：

- `bp0 / bpI` 在训练时读取一个固定的 `Q(bp)`
- 而不是让 `bp` 的优化目标反过来改写 `Q` 的表示

---

## 3. 哪些可以一起训练

### 3.1 可以放在一个 value-policy block 的对象

以下对象可以一起训练：

- `V0`
- `VI`
- `bar_i_cond`
- `bp0`
- `bpI`

理由：

- `V0 / VI / bar_i_cond` 是同一个 survival-conditioned value comparison block
- `bp0 / bpI` 是这一股权块下的最优杠杆选择
- 它们共享的是“企业股权侧在给定价格环境中的最优决策”这层对象

但这里有一个前提：

```text
Q 已经固定，或者至少 Q 的 encoder 不再被这一组 loss 反传更新
```

否则股权侧 loss 会继续污染 `Q`。

---

### 3.2 可以只做派生、不单独立头主训的对象

以下对象更适合作为派生量：

- `Vhat`
- `chi`
- `bar_z`
- `P`
- `bar_i`

理由：

- 它们都可以从 `V0 / VI / bar_i_cond` 的输出再加工得到
- 若再把它们当并列训练目标，容易引入额外冲突
- 当前理论文档本身就是朝“对象语义分层”走，而不是多头混训

---

## 4. 哪些应该独立训练

### 4.1 `Q` 应独立训练

`Q` 应该单独训练，最好拥有独立 encoder 和独立 head。

最少也要做到：

- `Q` 不与 `bp0 / bpI` 共用 trunk
- `Q` 不与 `V0 / VI / bar_i_cond` 共用 trunk
- value/policy block 只能读取 `Q`，不能回传修改 `Q`

这是当前最重要的结构性调整。

---

### 4.2 若保留独立 `bar_z` head，也应归到 value-survival block，而不是 `Q` block

如果未来不再用 `bar_z = 1 - chi` 的纯派生方式，而想保留独立 `bar_z` 网络，那么它也应该与 `V0 / VI / chi / P` 同组，而不应放在 `Q` 路径上。

原因是它描述的是 survival / default boundary，而不是债券 contract 的市场定价。

---

## 5. 推荐训练阶段

### Stage A: Q-only

训练对象：

- `Q encoder`
- `q_head`

此阶段目标：

- 先把 `Q(bp)` 学成稳定、可解释的价格曲线
- 先把 `q_unit(bp)` 学成合理形状
- 不让任何 equity-side loss 干扰 `Q`

---

### Stage B: Value-only / Value-policy

冻结：

- `Q encoder`
- `q_head`

训练对象：

- `V0`
- `VI`
- `bar_i_cond`
- `bp0`
- `bpI`

派生：

- `Vhat`
- `chi`
- `bar_z`
- `P`
- `bar_i`

此阶段目标：

- 在固定 `Q` 环境下学股权侧 continuation value 与 policy
- 避免 `Q` 被股权块重新拖偏

---

### Stage C: 可选的轻量交替更新

如果后续发现固定 `Q` 后 value block 仍有系统偏差，可以考虑 very light alternating：

1. 固定 value-policy block，更新少量 `Q`
2. 固定 `Q`，再更新 value-policy block

但这里仍不建议做真正意义上的 end-to-end joint backward。

更不建议恢复“共享 trunk + 多头一起拉”的旧结构。

---

## 6. 对当前代码结构的直接含义

当前结构里最不合理的一点是：

- `SharedModel` 中 `Q / bp0 / bpI` 共用同一个 `share_layer`
- `PolicyValueModel` 中 `shared_model` 与 `combined_model` 又通过同一个 `share_layer` 相连

这意味着：

```text
Q、杠杆候选、股权条件价值、投资门槛
都在争夺同一套底层表征
```

而你的理论文档已经说明：

- `Q` 是一个独立的 pricing object
- `V0 / VI / bar_i_cond / bp` 是 equity-side decision block

所以当前共享方式本身就违反对象分层。

---

## 7. 最终建议

最合理的分组是：

### Group 1: Bond Pricing Block

- `Q`

单独训练，单独 encoder。

### Group 2: Equity Value / Policy Block

- `V0`
- `VI`
- `bar_i_cond`
- `bp0`
- `bpI`

可以一起训练，但建立在 `Q` 固定的前提下。

### Group 3: Derived Objects

- `Vhat`
- `chi`
- `bar_z`
- `P`
- `bar_i`

尽量由 Group 2 派生，不单独与 `Q` 混训。

---

## 8. 一句话版本

按当前理论，能够一起训练的是：

```text
V0 / VI / bar_i_cond / bp0 / bpI
```

应该独立训练的是：

```text
Q
```

更适合作为派生量的是：

```text
Vhat / chi / bar_z / P / bar_i
```

核心原则不是“所有头都拆开”，而是：

```text
债券定价对象与股权决策对象必须分层；
Q 负责 pricing，P/V/bp 负责 equity-side decision。
```
