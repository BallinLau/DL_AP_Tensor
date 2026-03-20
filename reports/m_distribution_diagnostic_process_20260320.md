# `M` 分布异常诊断过程记录（2026-03-20）

## 问题背景

在多次 `modea` 训练后，`child macro states` 的 `M` 直方图表现出明显异常：

- 不是围绕 `1` 附近的小幅波动
- 存在低 `M` 堆积
- 同时又有较厚的中高值区域和长尾

这与理论上期望的“单峰、均值接近 `0.98` 的正值分布”不一致。

## 诊断步骤

### 1. 排除画图口径问题

先后修正了以下可视化问题：

1. `M` 直方图不再混入 firm panel 的重复计数，改为直接使用 `df_macro["M"]`
2. `M` 主图与 `parent/child` 分开
3. `bp` 只画真正的 parent states

修正后，异常分布依然存在，因此问题不在画图。

### 2. 确认 `SimulateTS` 中 `M` 的递推口径

检查发现，模拟里原先会把 `state["hatcf"] / state["lnkf"]` 回写成 realized `Hatc/LnK`。

这会导致：

- `M` 不再纯粹来自 forecast-state 递推
- child `M` 的解释混杂了 forecast state 与 realized macro

后续已改为：

- `M` 只来自 `state["hatcf"] / state["lnkf"]`
- realized `Hatc/LnK` 只写入输出表，不回写到状态

### 3. 用 `ep3_stage_modea_macro.pkl` 做 child-state 分组

对 [ep3_stage_modea_macro.pkl](/Users/ballinliu/Desktop/ep3_stage_modea_macro.pkl) 的 child macro states 按 `M` 分组后，发现：

- `M` 的均值并不一定错，但形状明显裂开
- 不同 `M` 组的 `hatcf / lnkf` 呈现明显不同的 regime

### 4. 配对 parent-child，检查一步增量

将 child macro state 与对应 parent state 配对后，发现：

- 低 `M` 区域的 `ΔlnK` 偏大
- 高 `M` 区域的 `Δhatcf` 偏大
- 整体不是一个平滑的小扰动递推，而是一个明显 state-dependent 的分裂映射

### 5. 直接扫描 `ep3_sdf_fc1.pt` 的响应面

固定 `x_t, x_{t+1}` 在均值附近，对 `(hatcf_prev, lnkf_prev)` 网格做 one-step 扫描：

- 当 `hatcf_prev` 较高、`lnkf_prev` 较低时，会出现
  - `Δhatcf < 0`
  - `Δlnkf` 偏大
  - 指数项很负
  - `M` 被压到接近 0

- 当 `hatcf_prev` 很低、`lnkf_prev` 较高时，会出现
  - `Δhatcf` 很大且为正
  - `M` 被推高

## 关键结论

### 结论 1：问题不在 parent 初始分布

`parent` 的整体分布仍然接近初始化时的高斯分布：

- `hatcf ~ N(-2, 1)` 附近
- `lnkf ~ N(4, 1)` 附近

因此，不是 parent 被“中途改坏”。

### 结论 2：问题在 `parent -> child` 的 forecast-state 映射

同一个接近高斯的 parent 分布，经过当前 `FC1/SDF` 一步映射后，被送进了多个不同的 child regime。

也就是说，异常的不是初始分布，而是响应面本身。

### 结论 3：`M` 的主导项是指数项，不是 `ratio`

将 `M` 分解后发现：

- `corr(M, exponent_term)` 高于 `corr(M, ratio_term)`
- `ratio` 有作用，但不是第一主因

因此，child `M` 的异常主要来自：

```math
\exp(-4 \Delta \ln K + 3 \Delta \hat c)
```

这部分在不同 parent 区域上给出了完全不同的动态。

## 由此得到的修正方向

不再优先从“均值锚”或“分布图展示”入手，而是直接约束一步 forecast-state 递推：

1. 对 `Hatc/LnK` 重建项拆分并下调 `LnK` 的内部权重
2. 对 `forecast-state` 的一步增量幅度加约束

第二条的目标不是硬编码经济方向，而是避免单步递推过大，把 child 状态撕裂成多个 regime。

## 为什么又从“步长约束”升级到“响应面平滑”

在加入 forecast-state 单步增量惩罚后，`child M` 分布出现了**一定改善，但改善不足**：

- 极端低 `M` 堆积有所收缩
- 但整体仍然不是单峰分布
- 仍然存在明显的 regime splitting

这说明仅仅限制：

```math
|\Delta \hat c|,\ |\Delta \ln K|
```

的绝对幅度，并不能完全修正问题。原因在于：

1. 步长约束只能压制“过大的跳跃”
2. 但当前更深层的问题是：`FC1` 对 `(hatcf_prev, lnkf_prev)` 的局部响应过于敏感
3. 于是即使跳跃幅度有所下降，不同 parent 区域仍会被映射到不同 child regime

因此，下一步需要从“限制步长”升级到“限制响应面陡峭度”，也就是增加 Jacobian / 局部平滑约束。

## 后续联动诊断：为什么继续检查 `bar_i`

在 `child M` 分布依然不理想的情况下，后续又观察到：

- `bar_i` 热图偏向高 `b`、低 `z` 区域
- 而低 `b`、高 `z` 的安全区域反而投资很弱

这说明问题已经不只是 `M` 分布本身，还影响到了 `PI - P0` 的空间形状。

因此，后续诊断转向把 `PI - P0` 拆成两部分：

1. 当期现金流差
   ```math
   CFip - CF0p
   ```
2. continuation 差
   ```math
   g\,M\,P_{t+1}(bp_I)\,(1-\bar z_{t+1}(bp_I))
   -
   M\,P_{t+1}(bp_0)\,(1-\bar z_{t+1}(bp_0))
   ```

这样可以区分：

- 是当期融资/投资成本项把 `bar_i` 推偏
- 还是 continuation 项把投资区推偏

## 本次新增的可视化诊断

为支持上述分解，当前代码的 `(b,z)` 截面图新增了三类诊断图：

1. `pidiff`
   ```math
   PI - P0
   ```
2. `cfdiff`
   ```math
   CFip - CF0p
   ```
3. `contdiff`
   ```math
   g\,M\,P_{t+1}(bp_I)\,(1-\bar z_{t+1}(bp_I))
   -
   M\,P_{t+1}(bp_0)\,(1-\bar z_{t+1}(bp_0))
   ```

这些图的作用是：

- 判断 `bar_i` 在高 `b` 低 `z` 区域究竟是由哪一部分驱动
- 检查低 `b` 高 `z` 区域“完全不投资”是否主要是投资成本 `-i` 压过了 continuation

此外，训练结束后的最终 `horizon=200` 模拟，还新增了：

- `t=50..150` 的 `parent firm states` 上 `b` 分布图

这样可以单独检查中段时点的债务分布，而不是只看全样本平均。

## 进一步修正：投资相关图只显示当前存活区域

在后续讨论中又确认了一点：

- `P^0 / P^I` 的 Bellman 方程是条件于当前期存活的价值
- 当前期已经 default 的状态，虽然在静态网格上仍可前向计算 `bar_i`，但在真实模拟里并不会继续进入下一期

因此，若不遮罩当前 default 区，会把本来“不会执行投资决策”的状态误画进投资区。

为此，投资相关 `(b,z)` 图已统一加入 current-survival mask：

```math
P_t > 0
\quad\text{and}\quad
\bar z_t < 0.5
```

目前受此 mask 约束的图包括：

- `bari`
- `bp`
- `pidiff`
- `cfdiff`
- `contdiff`

这样后续再看高 `b`、低 `z` 区域时，能更准确地区分：

- 真正会继续存活并执行投资决策的状态
- 以及本来当期就应该退出的状态
