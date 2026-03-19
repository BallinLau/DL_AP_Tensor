# Qsp 输入杠杆口径修复报告（2026-03-12）

## 背景

你指出 `Qsp` 的输入杠杆与 `q_loss` 主方程注释口径不一致：

- 方程口径：`b' = b / (bar_i * (G-1) + 1)`（见 `losses/q_loss.py` 注释）
- 实现口径（旧）：`childsp_state[:,0] = b_parent`

这会导致 Bellman 残差里 continuation 项的状态输入与缩放项不匹配。

## 本次修改

文件：`training/episode.py`

在 `_compute_q_loss` 中，新增：

```python
g_val = float(getattr(loss_fn, "g", 1.0))
multiplier = bar_i_use * (g_val - 1.0) + 1.0
b_sp = b_parent / multiplier.clamp_min(1e-6)
```

并将：

```python
childsp_state[:, 0:1] = b_parent
```

改为：

```python
childsp_state[:, 0:1] = b_sp
```

## 预期影响

1. `Qsp` 的输入状态与 `q_loss.compute_main_residual(...)` 的经济方程一致。  
2. 降低 Q 形状学习中的“错口径补偿”，有助于恢复 `Q-b` 曲线的合理弯曲（倒 U 识别）。  
3. 减少 `bar_i` 与 Q 斜率学习的错误耦合。

## 校验

已执行：

```bash
python -m py_compile training/episode.py
```

结果：通过。

> 说明：当前环境无法稳定运行完整训练（OpenMP SHM 限制），行为验证请在本机 notebook 复跑。
