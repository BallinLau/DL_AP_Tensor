# 当前阶段总结（2026-03-26）

## 1. 当前已经确认的问题

### 1.1 `M` 分布不合理

最开始观察到：

- `M` 在图上大量堆积在 `0` 附近
- 同时存在 `1` 附近的尖峰和右侧长尾

后续排查确认：

- 最早一部分异常来自可视化口径问题：
  - `firm panel` 重复计数
  - `parent` 与 `child` 混画
- 修正口径后，真正的问题仍然存在：
  - `child macro states` 的 `M` 分布本身就不合理
  - 并不是围绕 `1` 附近的单峰正值分布

### 1.2 `bar_i` 出现在高 `b`、低 `z` 区域

当前结果里：

- `bar_i` 偏向高杠杆、低生产率区域
- 低杠杆、高生产率的安全区域反而投资偏弱

这与正常“低 `b`、高 `z` 更容易投资”的理论直觉不一致。

### 1.3 `bp` 偏大且容易贴边

`bp` 长期偏高、靠近边界，说明当前 `PI/P0` 的比较中混入了较强的融资操作效应，而不只是纯粹的投资价值比较。

### 1.4 当前实现没有清楚区分 conditional / unconditional value

这是目前最核心的理论问题。

现有代码里：

- `P0 / PI` 一边承担 Bellman 递推
- 一边又通过 `Phat -> P / bar_z` 进入当前期 default 逻辑

因此对象定义混在了一起，没有严格区分：

1. 当前期存活条件下的 continuation value
2. 考虑当前期 default 后的总股权价值

## 2. 目前已经完成的修正与诊断

### 2.1 可视化口径修正

已经完成：

- `M` 直方图改为优先使用 `df_macro["M"]`
- `M` 主图只画 `child macro states`
- `parent macro states` 单独成图
- `bp` 只画 parent states
- 新增 `PI/P0` 诊断图：
  - `pidiff`
  - `cfdiff`
  - `contdiff`
- 新增 final simulate 的中段 `b` 分布图：
  - `final_b_hist_t50_t150_parent.png`

此外，投资相关 `(b,z)` 图现在已加 current-survival mask：

```math
P_t > 0 \quad \text{and} \quad \bar z_t < 0.5
```

也就是说：

- `bari`
- `bp`
- `pidiff`
- `cfdiff`
- `contdiff`

只显示当前期仍存活的区域，避免把当期已经 default 的状态误画进投资区。

### 2.2 `SimulateTS` 递推口径修正

已经修正：

- 不再把 `state["hatcf"] / state["lnkf"]` 回写成 realized `Hatc / LnK`
- `M` 只沿 forecast-state 递推

这样后，`child M` 的异常可以更明确地解释为 forecast-state 一步映射问题，而不是 realized 宏观量回写带来的口径混杂。

### 2.3 FC1 稳定性修正

已经做过的稳定性修改包括：

- 去掉 FC1 scaler 依赖
- stage2 以 forecast-state 为主进行重建
- `LnK` 重建项降权
- 增加 forecast-state 单步增量惩罚
- 增加 Jacobian / 局部平滑惩罚

结果上：

- forecast-state 递推的极端跳跃有所收敛
- `M` 分布有一定改善
- 但仍然没有恢复为理论期望的单峰形状

### 2.4 关于 `PI/P0` 诊断的关键结论

通过新增图形，目前看到：

1. `pidiff` 在高 `b`、低 `z` 区域大于 0
2. `cfdiff` 全局小于 0
3. `contdiff` 全局大于 0

这说明：

- 当期现金流口径下，投资几乎 everywhere 都吃亏
- 投资分支之所以还能在某些区域赢过不投资，主要靠 continuation 优势

但这里需要特别注意：

- continuation 用的是下一期**总股权价值** `P_{t+1}`
- 不是 `P_{t+1}^0 / P_{t+1}^I` 的分头 continuation

更准确地说，当前实现诊断的是：

```math
g\,M_t\,P_{t+1}(bp_I)\,(1-\bar z_{t+1}(bp_I))
-
M_t\,P_{t+1}(bp_0)\,(1-\bar z_{t+1}(bp_0))
```

## 3. 当前最重要的理论澄清

### 3.1 Bellman 方程本身是 conditional-on-survival 的

理论文稿里：

- `P^0`
- `P^I`

本质上是“当前期存活条件下”的价值。

当前期是否 default，不应只乘在 continuation 上，而应在总股权价值这一层处理。

因此：

- 当前期若已破产，未来当然不应再有股权价值
- 但这个逻辑应由当前期 survival gate 作用在整个价值对象上，而不是只作用在未来项上

### 3.2 这正是当前代码最根本的混淆点

现在代码里：

- `P0 / PI` 同时承担了 Bellman 对象和总值对象的语义

这会导致：

- `bar_i = sigmoid(PI - P0)` 在当前其实已 default 的状态上也有数值
- 必须靠后处理 mask 才能把图画对

也就是说，图已经尽量修正了，但模型对象定义本身还没有完全理顺。

## 4. 当前最新方案

下一阶段最推荐的方案，是把对象分成两层：

### 第一层：survival-conditioned values

定义：

- `V0`：当前期存活条件下，不投资价值
- `VI`：当前期存活条件下，投资价值

Bellman 训练对象改为：

```math
V_t^0 = CF_t^0 + E_t[M_{t,t+1} P_{t+1}]
```

```math
V_t^I = CF_t^I + g E_t[M_{t,t+1} P_{t+1}]
```

这里右边继续用下一期总股权价值 `P_{t+1}`，这一点与理论一致。

### 第二层：unconditional total equity value

定义：

```math
\hat V_t = \int \max(V_t^0, V_t^I)\, dH(i)
```

```math
\chi_t = \text{当前期软存活门}
```

```math
P_t = \chi_t \hat V_t
```

数值上最小改法可以先保留：

```math
P_t = \max(0,\hat V_t)
```

同时把 `\chi_t` 与 `bar_z_t` 联系起来。

### 投资边界也分两层

推荐拆成：

```math
\bar i_t^{cond} = \sigma\big(\tau_i (V_t^I - V_t^0)\big)
```

```math
\bar i_t^{eff} = \chi_t \cdot \bar i_t^{cond}
```

含义：

- `bar_i_cond`：如果当前还活着，投不投资
- `bar_i_eff`：真正会被执行的投资决策

这样：

- parent 已经 default 时，`bar_i_eff = 0`
- 但仍保留 `bar_i_cond` 作为理论上的条件比较对象

## 5. 为什么这是目前最好的下一步

因为它同时解决了目前最核心的几个理论问题：

1. current default 与 continuation value 的层次被分开
2. `bar_i` 不再在已 default 区域拥有经济意义
3. `P0/PI` 不再同时承担 conditional 和 unconditional 两种语义
4. 当前图上看到的很多“反常投资区”，可以更清楚地区分究竟来自：
   - conditional investment incentive
   - 还是 current survival gate

## 6. 当前版本管理建议

不建议直接复制整套项目目录作为新版本管理方式。

更推荐：

1. 先把当前状态完整提交并推到 GitHub
2. 在 Git 里新建一个专门用于 conditional/unconditional 分层改造的分支
3. 如果你希望物理上分开两个工作目录，使用 `git worktree`

原因：

- 比直接复制目录更不容易丢失历史
- 更方便比较差异
- 更方便随时 cherry-pick 或回滚
- 不会出现两套目录独立漂移、难以同步的问题

如果需要物理隔离，建议用：

- 当前目录保留为稳定版本
- 新建一个 worktree 目录专门做结构改造

这比手工复制整个项目更适合作为科研代码的版本管理方式。
