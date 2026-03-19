# Q 的 b*recovery 语义改造报告（2026-03-12）

## 1) 改造目标

按你确认的口径，引入“总债价值”语义：

- `Q` 表示总债价值（不是单位债价格）；
- `b=0` 时应严格满足 `Q=0`；
- 违约约束与边界条件按 `b*recovery` 口径对齐。

---

## 2) 具体改动

### A. 结构层：Q 输出改成 `Q = b * q_unit`

文件：`models/share_layer.py`

- `SharedModel.forward(...)`
  - 原来：`Q = self.q_head(h)`
  - 现在：
    - `q_unit = self.q_head(h)`
    - `b_nonneg = clamp(b, min=0)`
    - `Q = b_nonneg * q_unit`

- `SharedModel.get_Q(...)` 同步改为相同口径。

效果：结构上硬满足 `b=0 => Q=0`。

### B. 损失层：违约目标改为 `b*recovery`

文件：`losses/q_loss.py`

新增函数：
- `compute_total_recovery(b, x, z) = b_+ * phi*(1-delta+exp(x+z))`

并改动：
- `compute_main_residual(...)`
  - 默认违约项从 `recovery` 改为 `recovery_total = b*recovery`。
- `compute_bar_z_constraint(...)`
  - 签名增加 `b`；约束从 `Q≈recovery` 改为 `Q≈b*recovery`。
- `compute_boundary_loss_high(...)`
  - 目标从 `recovery` 改为 `b*recovery`。
- `forward(...) / forward_simplified(...)`
  - `loss3` 与 `penalty_z_loss3` 全部切换到 `b*recovery` 口径。

### C. Episode 对接

文件：`training/episode.py`

- `_compute_q_loss(...)`
  - `compute_bar_z_constraint` 调用新增 `b_parent` 参数；
  - `penalty_z_loss3` 使用 `compute_total_recovery(b_parent, x_parent, z_parent)`。

---

## 3) 一致性说明

当前实现满足以下一致性：

1. 结构层 `Q = b*q_unit`，保证 `b=0` 不会出现正债值。  
2. 损失层违约目标与边界条件统一为“总债价值口径”。  
3. Q 方程主残差、bar_z 违约约束、高杠杆边界使用同一 recovery 语义。

---

## 4) 校验

已执行：

```bash
python -m py_compile models/share_layer.py losses/q_loss.py training/episode.py
```

结果：通过。

> 注：本环境无法稳定运行完整训练（OpenMP SHM 限制），行为验证请在本机 notebook 重跑。
