# `bp_value_loss` 语义对齐修复（2026-04-01）

## 问题

`P0/PI` 主 Bellman 链已经改成了论文口径：continuation 的 debt argument 直接使用当期选定的 contract `b'`。

但 `bp_value_loss` 里的 coarse-grid supervision 仍然在用旧口径：

```python
child_state[:, :, 0:1] = eta_grid * bp_grid + (1.0 - eta_grid) * b_parent_grid
```

这表示监督目标仍在评价“`eta'` 实现后的 realized debt”，而不是“选定的 contract `b'`”。

于是 `bp` 头同时收到两套不一致信号：

- 主 Bellman / FOC / KKT：固定 `b'`
- `bp_value_loss`：混合 realized debt

这会导致：

- `bp0` 与 `bpI` 难以分开
- `bp_value_target_mean` 与 `bp_value_pred_mean` 长期偏离
- `bp` 容易停在 `P(bp)` 崖边附近的折中位置

## 修复

将 `bp_value_loss` 中 child debt 的构造改为与主 Bellman 一致：

```python
child_state[:, :, 0:1] = bp_grid
```

## 预期影响

1. `bp_value_target` 与主训练语义重新一致。
2. `bp0` / `bpI` 更容易体现 branch-specific 差异。
3. `bp_value_target_mean` 与 `bp_value_pred_mean` 应更接近。
4. `bp` 选点应更少落在 `P(bp)` 快速塌缩的边界附近。

## 后续观察指标

- `p0_bp_value_target_mean` vs `p0_bp_value_pred_mean`
- `pi_bp_value_target_mean` vs `pi_bp_value_pred_mean`
- `p0_kkt_active_ratio` / `pi_kkt_active_ratio`
- `bp0*`, `bpI*`, `bp*` 在 `bp_diag_safe` 中的相对位置
