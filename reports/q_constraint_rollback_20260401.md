# Q 约束回退说明 2026-04-01

## 回退原因

在提交 `11d6f2d` 之后，新的 `Q` 总量斜率约束会把 `Q` 在 `q_stage` 中快速压到近零常数解。

从训练日志可以直接看到：

- `q_unit_mean` 很快掉到 `1e-12`
- `dQ_db_mean` 很快掉到 `1e-12`
- `policy_value_grad_norm` 在 `q_stage_end` 基本接近 `0`
- 但 `q_main` 仍然明显大于 `0`

这说明问题不是 `Q` 方程被满足，而是 `Q` 头掉进了近零死区。

## 回退内容

本次只回退上一轮新增的两部分代码：

1. 回退 `Q` 总量斜率约束

- 删除 `dQ/db > 0` 的连续加权惩罚
- 恢复为只约束 `dq_unit/db <= 0` 和 `dq_unit/dz >= 0`

2. 回退高 `b` 连续加权的 gate/bar_z 单调惩罚

- 删除 `mono_barz_b_high`
- 恢复为上一版的 `value/chi` 基础单调性约束

3. 回退对应超参数

- 删除：
  - `gate_high_b_weight_power`
  - `gate_high_b_weight_scale`
  - `barz_mono_weight_b_high`
  - `q_shape_weight_b_total`
  - `q_total_slope_weight_power`
  - `q_total_slope_weight_scale`

## 保留内容

以下内容保持不变：

- `Q` 与 `PVBP` 模块拆分
- `Q stage -> PVBP stage` 训练顺序
- `I_THRESHOLD = 0.2`
- 之前已经加入且未直接导致 `Q` 数值塌缩的稳定化修复

## 回退后的判断重点

回退后优先观察：

- `q_stage_end` 的 `q_unit_mean` 是否恢复到正常量级
- `Q surface` 是否不再在训练初期塌成近常数
- `q_main` 是否开始真实下降，而不是伴随梯度消失停滞

如果回退后 `Q` 恢复，说明问题主要来自上一轮新增的总量斜率约束过强，而不是更底层的 `Q` 架构错误。
