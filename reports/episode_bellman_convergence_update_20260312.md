# Episode 收敛判定更新报告（2026-03-12）

## 本次目标
按用户要求将收敛判定固定为：
- `mean(abs residual) < 0.001`
- `p90(abs residual) < 0.005`
并将判定接入训练流程。

## 关键改动

### 1) 超参数新增（默认即用户指定阈值）
- 文件：`config/hyperparams.py`
- 新增字段：
  - `bellman_conv_mean_thresh: float = 1e-3`
  - `bellman_conv_p90_thresh: float = 5e-3`
  - `episode_stop_on_bellman_convergence: bool = True`

### 2) Episode 内新增非 AIO Bellman 残差评估
- 文件：`training/episode.py`
- 新增方法：
  - `evaluate_bellman_convergence(...)`
  - `_compute_p0_bellman_abs_residual(...)`
  - `_compute_pi_bellman_abs_residual(...)`
  - `_compute_q_bellman_abs_residual(...)`
  - 以及批次 children/输出提取与残差 flatten 辅助函数。

实现口径：
- 直接使用 **原始 branch residual**（非 AIO 聚合）。
- 对各方程（`p0/pi/q`）将所有 branch+sample 的 `|residual|` 展平后统计：
  - `mean`
  - `p90`
- 方程通过条件：`mean < mean_thresh` 且 `p90 < p90_thresh`。
- 总通过条件：`p0/pi/q` 所有启用方程都通过。

### 3) 在训练返回结果中写入 convergence 摘要
- 文件：`training/episode.py`
- 在 `_run_batches(...)` 训练结束后：
  - 若训练模块包含 `policy_value`，自动计算并返回 `convergence`。
- 在 `run(...)` 返回 `summary` 中也加入 `convergence`。
- 在 `run_episode(...)` 返回 `summary` 中透传 `module_summaries['policy_value']['convergence']`。

### 4) Trainer 侧接入 episode 早停
- 文件：`training/trainer.py`
- 在 `train(...)` 的 episode 循环中：
  - 若 `summary['convergence']['passed'] == True` 且 `episode_stop_on_bellman_convergence=True`，提前结束后续 episode。

### 5) 历史保存兼容
- 文件：`training/trainer.py`
- `save_history()` 增加对 `convergence` 的可序列化转换，避免 `numpy` 标量导致 JSON 序列化问题。

## 代码检查
已执行：
- `python3 -m py_compile config/hyperparams.py training/episode.py training/trainer.py`
- 结果：通过，无语法错误。

## 结果说明
当前工程默认收敛阈值已变为：
- `mean < 0.001`
- `p90 < 0.005`
且采用用户要求的 Bellman **非 AIO** 口径进行判定，并可在 episode 级触发自动停止。
