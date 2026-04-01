# PVBP / Q 下一轮修改方案

日期：2026-04-01

## 目标

当前训练已经从“完全失真”进入“方向基本可解释”的阶段，但仍有三类残余问题：

1. 高杠杆区的 default discipline 不够强，`bar_z` 对高 `b` 的抬升不足。
2. `Q` 虽然满足 `dq_unit/db <= 0`，但总量 `Q(b) = b * q_unit(b)` 在高 `b` 区仍可能继续上升。
3. `PVBP` 当前还是单阶段一起训练，`value` 和 `policy` 之间仍可能互相拖动。

本轮修改不再继续大改模型架构，而是集中在：

- 第一组：增强高杠杆违约纪律
- 第二组：补上 `Q` 总量斜率约束
- 第三组：将 `PVBP` 改成 `value / policy` 交替训练

---

## 1. 第一组：高杠杆违约纪律

### 1.1 不再使用固定 `b_cut`

当前没有数值解，因此不应人为指定一个“从这里开始才算高杠杆”的绝对阈值。

本轮不采用：

- 固定 `b_cut = 0.8`
- 固定 `b_cut = 0.6`

这种硬阈值做法。

改为使用连续加权：

```text
w(b) = b^p
```

其中 `p` 初始可取：

- `p = 2`
- 或 `p = 3`

含义是：

- 低 `b` 区的违约纪律仍然存在，但较弱
- `b` 越高，违约纪律越强
- 不需要数值解来定义“高杠杆区”

### 1.2 约束对象

当前已有的方向约束应保留：

- `∂V/∂b <= 0`
- `∂V/∂z >= 0`
- `∂chi/∂b <= 0`
- `∂chi/∂z >= 0`

本轮要加强的是 `b` 向违约纪律，并显式加入 `bar_z`：

- `∂chi/∂b <= 0`
- `∂bar_z/∂b >= 0`

其中 `bar_z = 1 - chi`，但仍建议直接对 `bar_z` 也写一条约束，
原因是这样在日志和调试图里更容易定位问题。

### 1.3 建议损失形式

可写成：

```text
L_chi_b = E[ w(b) * relu(dchi/db)^2 ]
L_barz_b = E[ w(b) * relu(-dbar_z/db)^2 ]
```

这样做的目的不是硬性规定某一段必须违约，
而是让“高杠杆越不该继续保持高存活概率”这件事在训练里逐步显现。

---

## 2. 第二组：`Q` 总量约束

### 2.1 为什么现有 `dq_unit/db <= 0` 不够

当前模型中：

```text
Q(b) = b * q_unit(b)
```

虽然已经约束：

```text
dq_unit/db <= 0
```

但总量导数是：

```text
dQ/db = q_unit + b * dq_unit/db
```

因此：

- `q_unit` 单调下降
- 不能推出 `Q` 在高 `b` 区下降

只要 `|dq_unit/db|` 不够大，`q_unit` 这项仍会让 `dQ/db > 0`。

这正是当前图里经常出现的情况：

- `q_unit(bp)` 看起来合理
- 但 `Q(bp)` 高 `b` 区仍继续抬升

### 2.2 本轮修改思路

保留现有 `dq_unit/db <= 0` 约束，
再额外加入对总量 `Q` 的正斜率惩罚：

```text
L_Q_slope = E[ w(b) * relu(dQ/db)^2 ]
```

其中同样使用连续加权：

```text
w(b) = b^p
```

而不是使用固定 `q_shape_b_high`。

### 2.3 可选相对尾部约束

如果只靠局部斜率惩罚还不够，
可进一步加入一个相对的尾部约束，但仍不依赖绝对阈值。

做法：

- 将 `b` 网格按分位数分成中段和尾部
- 中段：例如 `40% ~ 60%`
- 尾部：例如 `80% ~ 100%`

然后加弱约束：

```text
mean(Q_tail) <= mean(Q_mid)
```

或：

```text
mean(dQ/db in tail) <= mean(dQ/db in mid)
```

第一轮实现时，建议先只上连续加权的 `dQ/db` 惩罚；
尾部相对约束可作为第二步。

---

## 3. 第三组：`PVBP` 的 value / policy 交替训练

### 3.1 为什么需要交替训练

当前 `PVBP` 虽然已经与 `Q` 分开，
但内部仍是同一阶段里同时训练：

- `V0 / VI / chi / bar_z / P`
- `bp0 / bpI`

这容易出现两类问题：

1. `value` 还没稳定，`bp` 头已经在追逐一个会移动的目标。
2. `bp` 的短期变化又会反过来扰动 `value` 面。

因此本轮不建议回到完全 joint，
而是采用 block coordinate 风格的交替训练。

### 3.2 拟采用的训练节奏

`Q` 仍然保持独立训练：

```text
Q stage: 100 epoch
```

其后 `PVBP` 内部拆成两块：

#### A. value stage

冻结：

- `bp0_head`
- `bpI_head`

只训练：

- `V0`
- `VI`
- `chi / bar_z / P`

主要损失：

- `p0_main`
- `pi_main`
- monotonicity / default discipline

#### B. policy stage

冻结：

- `V0 / VI / chi / bar_z / P`

只训练：

- `bp0_head`
- `bpI_head`

主要损失：

- `FOC`
- `KKT`
- `bp value`

### 3.3 初始建议节奏

可先试：

```text
Q stage: 100 epoch
Value stage: 30 epoch
Policy stage: 20 epoch
Value stage: 20 epoch
Policy stage: 20 epoch
```

先用这个最小交替版本判断：

- `value` 是否更稳
- `bp` 是否不再把 `P` 面拖歪

如果有效，再考虑把它推广成循环 schedule。

---

## 4. 本轮不修改的内容

本轮不继续动：

- 模型总架构
- `modea / modeb` 数据流
- `SDF`
- `I_THRESHOLD`

原因是当前主要矛盾已经收缩到：

- 高 `b` 区 default discipline 不够
- `Q` 总量高 `b` 端惩罚不足
- `PVBP` 内部 value / policy 互相拖动

继续改其它模块会让诊断变得不干净。

---

## 5. 实施顺序

建议按以下顺序实施并做短实验：

1. 第一组：连续加权的 `chi / bar_z` 高杠杆纪律
2. 第二组：连续加权的 `dQ/db` 惩罚
3. 第三组：`PVBP` 改为 value / policy 交替训练

每一步都先跑短程对照，再决定是否保留。

---

## 6. 观察指标

每轮短实验固定看：

- `Q surface`
- `P surface`
- `bar_z heatmap`
- `PIDIFF heatmap`
- `bp diagnostics (safe state)`

判据如下：

### 第一组有效

- 高 `b` 区 `bar_z` 明显抬升
- 高 `b` 区 `chi` 明显下降
- `P` 不再在高 `b` 区过度乐观

### 第二组有效

- `Q` 高 `b` 端不再继续抬升
- `q_unit` 仍保持单调下降
- `Q` 在中高 `b` 区出现更合理的平台或回落

### 第三组有效

- `V0 / VI / P / bar_z` 面更稳定
- `bp0 / bpI` 不再出现明显抖动
- Bellman 主残差下降，而不是只靠 FOC/KKT 改善

---

## 7. 一句话总结

本轮修改不再依赖任何固定杠杆阈值。

核心思想是：

1. 用连续加权而不是硬 cutoff 来增强高杠杆违约纪律；
2. 保留 `dq_unit/db <= 0`，同时补上 `dQ/db` 的总量约束；
3. 让 `PVBP` 从单阶段联合训练改成 `value / policy` 交替训练。

这三步的目标是把当前“方向基本对，但高杠杆区纪律不足”的状态，
推进到“高杠杆 default / pricing / investment 能同时协调”的状态。
