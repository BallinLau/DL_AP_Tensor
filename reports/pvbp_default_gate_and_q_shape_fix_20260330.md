# PVBP Default Gate And Q Shape Fix

日期：2026-03-30

## 背景

在拆分 `QModel` 与 `PVBPModel` 之后，曾先后出现两类相反但相关的问题：

1. `P=0, bar_z=1` 的全默认吸收态。
2. 修复吸收态后，`P` 表面看起来恢复，但 `chi` 过高、`bar_z` 过低，导致高杠杆区域几乎不违约；与此同时 `Q` 面沿 `b` 近似单调上升，不再体现高杠杆压价。

用户给出的最新图属于第二类。

## 原因分析

### 1. PVBP anti-collapse warmup 过强

在 [`models/policy_value.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/models/policy_value.py) 中，
`chi` 由 `sigmoid(BARZ_LOGIT_TEMP * Vhat)` 给出，`bar_z = 1 - chi`。

此前为避免 `P=0` 吸收态，引入了 `chi_warmup_factor`，并在 [`training/episode.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/training/episode.py) 的 PVBP 阶段前若干轮将 `chi` 向 1 混合。

旧默认值：

- `pvbp_anti_collapse_warmup_epochs = 20`
- `pvbp_anti_collapse_start = 0.25`

这意味着训练初期会显式鼓励“更高存活、更低违约”。当 `Vhat` 本身已经被推正时，这个 warmup 会进一步把 `chi` 顶高、`bar_z` 压低。

### 2. P0/PI 主训练链没有真正接入 value/default gate 的形状约束

虽然 [`losses/p0_loss.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/losses/p0_loss.py) 和
[`losses/pi_loss.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/losses/pi_loss.py) 内部写了单调性惩罚，
但主训练实际走的是 [`training/episode.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/training/episode.py) 自己拼装的 loss。

在这条主训练链里，之前只包含：

- Bellman 主残差
- z penalty
- FOC / KKT
- bp value supervision

没有把以下经济方向显式接回：

- `V0 / VI` 对 `b` 递减
- `V0 / VI` 对 `z` 递增
- `chi` 对 `b` 递减
- `chi` 对 `z` 递增

结果就是：

- 高杠杆区域的 value 不一定被有效压低
- default gate 没有被训练成“高 b 更易违约、低 z 更易违约”
- `Q` 又把这个过度乐观的 `bar_z` 当外生条件读进去，于是违约回收分支失声

## 本次修复

### 1. 削弱 PVBP anti-collapse warmup

修改文件：

- [`config/hyperparams.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/config/hyperparams.py)

默认值由：

- `pvbp_anti_collapse_warmup_epochs = 20`
- `pvbp_anti_collapse_start = 0.25`

改为：

- `pvbp_anti_collapse_warmup_epochs = 5`
- `pvbp_anti_collapse_start = 0.85`

这会把 warmup 从“强行存活”改成“仅短暂缓冲”，避免 default gate 被长时间压平。

### 2. 在 P0/PI 主训练链显式加入单调性约束

修改文件：

- [`training/episode.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/training/episode.py)

新增 `_compute_value_gate_monotonicity_penalty(...)`，并在 `P0` / `PI` loss 中接入：

- value 单调性：
  - `∂V/∂b <= 0`
  - `∂V/∂z >= 0`
- gate 单调性：
  - `∂chi/∂b <= 0`
  - `∂chi/∂z >= 0`

这等价于让：

- 高 `b` 更容易 default
- 低 `z` 更容易 default

同时保留原有：

- Bellman 主残差
- z penalty
- FOC / KKT
- bp value supervision

### 3. 新增可调权重

修改文件：

- [`config/hyperparams.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/config/hyperparams.py)

新增：

- `pv_mono_weight_b = 1.0`
- `pv_mono_weight_z = 1.0`
- `chi_mono_weight_b = 0.5`
- `chi_mono_weight_z = 0.5`

## 预期结果

如果修复有效，应看到：

1. `P` 不再轻易掉回全 0。
2. `chi` 不再在大部分 `(b,z)` 区域都接近 1。
3. `bar_z` 在高 `b`、低 `z` 区域应明显抬升。
4. `Q` 在高 `b` 区域不再持续线性上升，而会重新体现高杠杆压价。

## 风险

这次修复仍然是“补回经济方向约束”，不是最终结构解。

若后续仍出现：

- `Q` 高杠杆端持续上升
- `bar_z` 仍过低
- `P` 在 `z` 维方向反转

下一步应继续考虑：

1. 给 `Q` 总量形状增加直接约束，而不只约束 `q_unit`。
2. 对 `V0/VI` 增加更强的结构化参数化，而不只是裸 MLP 标量输出。
