# P/Q 时点口径回滚（eta_{t+1}）与 Notebook 报错修复报告

日期：2026-03-12

## 1. 目标

按你的要求，把 P/Q 训练与模拟中的杠杆更新口径统一为：

\[
b_{t+1} = \eta_{t+1} b'_t + (1-\eta_{t+1}) b_t
\]

并同步修复：
1. 状态转移改回 child eta（eta_{t+1}）。
2. FOC 的 eta 权重改为分支 eta（child eta）。
3. Bellman 现金流债务调整项按分支 eta 逐分支计算，避免“CF 用 parent eta、continuation 用 child eta”的混时点。
4. 修复 `tests/sdf_fc1_two_modes_test.ipynb` 在 `SimulateTS.simulate()` 处的 `zero-dimensional tensor cannot be concatenated` 报错。

---

## 2. 代码改动

### A. 状态转移：统一到 eta_{t+1}

- `data/sample.py`
  - `_update_child_leverage` 改回使用 `child_rows['ETA']`。
  - 注释更新为 `b_{t+1} = η_{t+1} * bp_t + (1 - η_{t+1}) * b_t`。

- `data/simulate_ts.py`
  - `_expand_branches` 的 `new_state['b']` 改回由 `new_state['eta']`（即 child eta）更新。
  - 保留 pad/truncate 对齐逻辑，但先 `reshape(-1)`，避免单公司时 0 维张量问题。

- `training/episode.py`
  - `_compute_p0_loss` / `_compute_pi_loss` / `_compute_q_loss` 构造 child state 时，全部使用 `eta_child` 更新 b。

### B. Bellman 现金流与 FOC：按分支 child eta 计算

- `training/episode.py`
  - `_compute_p0_loss`：
    - `CF0p` 从单个 tensor 改为分支列表 `CF0p[j]`，每个分支用自己的 `eta_child[j]`。
    - `compute_foc_residual_from_bp(..., eta=eta_children)`，FOC 传入分支 eta 列表。
  - `_compute_pi_loss`：同上，`CFip` 与 FOC 都改为分支 eta 口径。

- `losses/p0_loss.py` / `losses/pi_loss.py`
  - `compute_bellman_residual` 支持 `CF` 为 tensor 或 list，list 场景按分支对齐。
  - `compute_foc_residual` 支持 `cf_grad` 与 `eta` 为 tensor 或 list，list 场景逐分支计算。
  - `compute_foc_residual_from_bp` 支持 `CF` 为 list，逐分支求 `∂CF_j/∂bp` 并与分支 eta 配对。
  - 新增分支数量一致性检查，防止 `zip` 静默截断。

### C. Notebook 报错修复（0 维张量拼接）

报错现象：
- `RuntimeError: zero-dimensional tensor (at position 0) cannot be concatenated`
- 位置：`tests/sdf_fc1_two_modes_test.ipynb` 的 `SimulateTS.simulate()` 调用单元。

根因修复：
- `data/simulate_ts.py`
  - `_process_node` 中 `state['bar_i'] / state['bar_z'] / state['bp']` 改为 `reshape(-1)`，防止单公司时 `squeeze()` 变成 0 维。
  - `_expand_branches` 中 `b/K/bp/bar_i/eta` 参与拼接前统一 `reshape(-1)`。
  - `_apply_entry` 前增加 0 维张量规范化（若 dim==0 则 reshape(1)）。

- `tests/sdf_fc1_two_modes_test.ipynb`
  - mode2 policy-value 模拟单元（cell 23）加入 `importlib.reload(data.simulate_ts)`，避免 notebook 复跑仍引用旧类定义。
  - 清空该单元旧错误输出。

---

## 3. 影响与预期

1. `bp -> b_{t+1}` 的状态转移与分支 shock (`eta_{t+1}`) 一致。  
2. P0/PI 的 Bellman 与 FOC 均以“同一分支时点口径”计算，避免梯度和经济含义错配。  
3. mode2 `simulate()` 在单公司/低存活状态下不再因 0 维张量拼接崩溃。  

---

## 4. 校验

已完成静态语法检查：

```bash
python -m py_compile data/sample.py data/simulate_ts.py losses/p0_loss.py losses/pi_loss.py training/episode.py
```

结果：通过。

说明：本环境无法稳定跑完整 torch 训练（OpenMP SHM 限制），行为验证请在你本机 notebook 复跑确认。
