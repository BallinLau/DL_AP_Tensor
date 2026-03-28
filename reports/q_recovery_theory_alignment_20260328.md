# Q 违约回收口径与论文对齐说明（2026-03-28）

## 1. 结论

论文里的违约回收价值不应乘债务面值 `b`。

理论口径是：

```math
\text{recovery}_{total}(x,z,k)=\phi(1-\delta+\exp(x+z))k
```

若当前实现已把资本口径归一到 `k=1`，则代码里应写成：

```math
\text{recovery}_{total}(x,z)=\phi(1-\delta+\exp(x+z))
```

而不是：

```math
\text{recovery}_{total}(b,x,z)=b\cdot\phi(1-\delta+\exp(x+z))
```

## 2. 论文证据

来自 [main_4.tex](/Users/ballinliu/Desktop/PHD/Project1/DL_Equilibrium/main_4.tex)：

- [main_4.tex:344](/Users/ballinliu/Desktop/PHD/Project1/DL_Equilibrium/main_4.tex#L344)
  明确写道，违约时债权人收到
  `\phi(1-\delta+\exp(x+z))k`

- [main_4.tex:349](/Users/ballinliu/Desktop/PHD/Project1/DL_Equilibrium/main_4.tex#L349)
  和 [main_4.tex:350](/Users/ballinliu/Desktop/PHD/Project1/DL_Equilibrium/main_4.tex#L350)
  的债券递推公式中，default payoff 也是
  `\phi(1-\delta+\exp(x'+z'))k'`

论文公式里没有 `b` 或 `\tilde b` 乘在回收项前面。

## 3. 之前代码的问题

修改前的 [q_loss.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/losses/q_loss.py) 用的是：

```math
\text{recovery}_{total}=b_+\cdot\phi(1-\delta+\exp(x+z))
```

这会带来两个直接后果：

1. 高杠杆区的总债价值 `Q(bp)` 被系统性托高。
2. 由于 `CF` 中的净发债收入项依赖 `Q(bp)`，会进一步把 `bp` 推向高位。

这和论文里“回收来自资产价值，而不是债务面值线性放大”的经济含义不一致。

## 4. 本次修改

文件：[q_loss.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/losses/q_loss.py)

改动点：

- `compute_total_recovery(...)`
  - 从 `b * recovery_unit` 改成仅返回 `recovery_unit`

- `compute_main_residual(...)`
  - default 项继续使用 `recovery_total * multiplier * bar_z`
  - 但其中的 `recovery_total` 不再线性依赖 `b`

- `compute_bar_z_constraint(...)`
  - `Q` 的违约约束改为贴近资产回收值，而非 `b * recovery`

- `compute_boundary_loss_high(...)`
  - 高 `b` 边界条件同样改为贴近资产回收值

- `compute_boundary_loss_low(...)`
  - 不再对整段 `b<=0.1` 施加 `Q≈0` 约束
  - 原因是结构层已经硬编码 `Q=b*q_unit`，因此 `b=0 => Q=0` 已天然成立
  - 若继续用区间型 low-boundary loss，会把小正债务区的 `Q` 错误压成接近 0

## 5. 为什么这样改是合理的

### 数学上

当前债券总价值可写成：

```math
Q(b)=b\cdot q_{unit}(b)
```

若把违约总回收写成 `b \cdot recovery_unit`，则单位债价格 `q_{unit}(b)` 会下降过慢，
从而让 `Q(b)` 的峰值出现在很高的 `b` 区域。

改回资产回收后，违约区域不再因为债务面值更大而机械抬高总回收，
`q_{unit}(b)` 会在高杠杆区更快下降，`Q(b)` 的峰值也应更靠左。

同时，去掉原先 `b<=0.1 => Q≈0` 的 low-boundary loss 后，
不会再把 `b=0.10` 这类小正债务状态也误判成“接近零债务边界”，
从而避免将整条 `Q(bp)` 曲线压到 `1e-5` 一类不合理量级。

### 经济上

违约回收应来自企业资产可清算价值，而不是因为债务写得更大、总回收也同比例变大。
否则模型会人为制造“多借债也不会显著降低总债价值”的假象，从而鼓励疯狂借债。

## 6. 兼容性说明

`compute_total_recovery(...)` 仍保留了 `b` 参数，以避免大面积改调用链；
但该参数现在仅用于接口兼容，理论上不再进入回收值计算。
