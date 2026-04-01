# High-b Gate And Q Total Slope Fix

日期：2026-04-01

## 背景

在上一轮分析中，当前模型虽然已经具备基本可解释性，但仍存在两类残余问题：

1. 高杠杆区域的 default discipline 不够强，`bar_z` 对高 `b` 的抬升不足。
2. `q_unit(bp)` 已经满足单调下降，但总量 `Q(bp) = bp * q_unit(bp)` 在高 `b` 区仍可能继续抬升。

用户进一步指出：

- 当前没有数值解，不能人为指定一个固定的 `b_cut`
- 同理，也不应拍一个固定的 `q_shape_b_high`

因此本轮修复改用连续加权，而不是硬阈值。

---

## 问题 1：为什么不能用固定 `b_cut`

在没有数值解的情况下，无法严谨地说：

- `b > 0.6` 就一定属于高杠杆区
- `b > 0.8` 才应该更强 default

如果直接写固定 cutoff：

```text
1{b > b_cut}
```

那这个 cutoff 只是人为拍脑袋选择，既不稳，也不利于后续解释。

因此本轮不使用固定 `b_cut`，改成：

```text
w(b) = 1 + scale * b^p
```

这样约束会随着 `b` 平滑增强：

- 低 `b` 区也有纪律，但较弱
- `b` 越高，违约纪律越强
- 不需要先知道理论上的精确阈值

---

## 问题 2：为什么 `dq_unit/db <= 0` 不够

当前模型里：

```text
Q(b) = b * q_unit(b)
```

现有约束要求：

```text
dq_unit/db <= 0
```

这对“单位债价格随杠杆升高而下降”是合理的，但它并不能推出总量 `Q` 在高 `b` 区下降。

因为总量导数是：

```text
dQ/db = q_unit + b * dq_unit/db
```

因此即使：

- `dq_unit/db < 0`

只要：

- `|dq_unit/db|` 不够大

仍然可能出现：

- `dQ/db > 0`

这正是之前图里常见的现象：

- `q_unit(bp)` 下降
- 但 `Q(bp)` 在高 `b` 区仍继续抬升

所以本轮修复不是替换掉原约束，而是：

1. 保留 `dq_unit/db <= 0`
2. 新增对 `dQ/db > 0` 的连续加权惩罚

---

## 本轮修改

### 1. 连续加权的 high-b default discipline

修改文件：

- [`config/hyperparams.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/config/hyperparams.py)
- [`training/episode.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/training/episode.py)

新增超参数：

- `gate_high_b_weight_power = 2.0`
- `gate_high_b_weight_scale = 2.0`
- `barz_mono_weight_b_high = 1.0`

在 `value/gate` 单调性惩罚中，新增：

```text
mono_barz_b_high
```

其形式为：

```text
E[ w(b) * relu(-dbar_z/db) ]
```

其中：

```text
w(b) = 1 + scale * b^power
```

这会在不使用固定阈值的情况下，让高杠杆区域更强地满足：

```text
dbar_z/db >= 0
```

也就是：

- 杠杆越高，违约阈值越该抬升

### 2. 新增 `Q` 总量正斜率惩罚

修改文件：

- [`config/hyperparams.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/config/hyperparams.py)
- [`training/episode.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/training/episode.py)

新增超参数：

- `q_shape_weight_b_total = 1.0`
- `q_total_slope_weight_power = 2.0`
- `q_total_slope_weight_scale = 2.0`

在 `q_loss` 中，新增：

```text
dQ/db
```

并加入惩罚：

```text
E[ w(b) * relu(dQ/db) ]
```

同样使用连续加权：

```text
w(b) = 1 + scale * b^power
```

这条约束不要求整个 `Q(bp)` 到处下降，
但会越来越不允许它在高杠杆区继续抬升。

原有的：

```text
dq_unit/db <= 0
```

仍然保留。

---

## 日志与诊断输出同步调整

本轮还同步更新了训练日志诊断项：

- `mono_barz_b_high`
- `q_shape_b_high` 现在记录的是新的 `Q` 总量斜率惩罚
- `dQ_db_mean`

这样后续看日志时可以区分：

- `q_unit` 的单位价格斜率
- `Q` 的总量斜率

---

## 本轮未做的事

这轮没有实现：

- `PVBP` 的 `value / policy` 交替训练

原因是这属于更大一步的训练调度改动，
不适合和本轮的约束修复混在一起做归因。

本轮只聚焦：

1. high-b default discipline
2. `Q` 总量高杠杆斜率

---

## 预期结果

如果修复有效，应看到：

1. `bar_z` 在高 `b` 区更明显抬升。
2. `P` 在高 `b` 区不再过于乐观。
3. `Q(bp)` 高 `b` 端更容易平台化或回落。
4. `PIDIFF` 右上角仍可保留高 `z` 的投资动力，但不会再因为高 `b` 缺乏惩罚而持续扩大。

---

## 一句话总结

本轮修复不依赖任何固定杠杆阈值。

做法是：

1. 用 `b` 的连续加权增强 high-b default discipline；
2. 保留 `dq_unit/db <= 0`，并额外补上 `dQ/db` 的总量约束。

目标是把当前“高杠杆区纪律不足”的问题，转化成训练中可直接被惩罚的对象。
