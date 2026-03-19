Debug change log (recent edits) — updated 2026-03-11 14:30:00

## 2026-03-11: P0/PI FOC 接线修复（episode + loss）

修改原因
- 方法论要求 P0/PI 包含 FOC 条件；此前训练主路径在 `training/episode.py` 只使用 Bellman 相关项，FOC 未接入总损失。
- 需要将 FOC 从“接口存在”变成“训练中实际生效”，并统一到当前多分支（parent-children）框架。

前后对比
- P0 (`training/episode.py::_compute_p0_loss`)
  - 修改前：`total_loss = main_loss + penalty_z`（仅 Bellman）
  - 修改后：`total_loss = main_loss + penalty_z + loss_foc + penalty_z_foc`
  - 关键实现：新增 `P0Loss.compute_foc_residual_from_bp(...)`，通过 autograd 计算 `dCF0p/dbp` 与 `dP_child/dbp`，再构造多分支 FOC 残差。
- PI (`training/episode.py::_compute_pi_loss`)
  - 修改前：`total_loss = main_loss + penalty_z + penalty_b`（无 FOC）
  - 修改后：`total_loss = main_loss + penalty_z + penalty_b + loss_foc + penalty_z_foc`
  - 关键实现：新增 `PILoss.compute_foc_residual_from_bp(...)`，通过 autograd 计算 `dCFip/dbp` 与 `dP_child/dbp`，再构造多分支 FOC 残差。

涉及文件
- `losses/p0_loss.py`
- `losses/pi_loss.py`
- `training/episode.py`
- `losses/README.md`

- data/data_utils.py
  - Added small random noise to initial macro proxy: `hatcf = N(-2, 0.1)`, `lnkf = N(4, 0.2)`.

- data/simulate_ts.py
  - Forward-step tensors in _expand_branches now forced to float32 for M prediction.
  - bp/bar_i padding to match firm count when entry/exit changes length.
  - bar_z padding in _apply_exit to match alive_mask length.
  - Child bp update now guards length mismatch.

- training/episode.py
  - _create_firm_batches_from_df: when t/branch numeric, child rows now reattach path/ID/t/branch before set_index to avoid missing columns.

- experiments/run_episode0_full.py
  - Staged training (sdf1 → pv → sdf2 → fc2) with SimulateTS for FC2; reduced simulate size.
  - Saves stage data, models, policy/value plots (heatmap+3D), M/bp histograms.

- experiments/run_multi_episode.py
  - Multi-episode staged runner (ep0 sample, ep≥1 simulate) with per-episode checkpoints, stage data dumps, plots (heatmap+3D) and macro series plots (Hatc, LnK for branch=0).
  - Added helpers for dirs, saving models/data, plotting surfaces/distributions/macro series.
  - Hyperparams reduced for quicker runs (n_samples=2000, n_paths=200, epochs=2) but can be tuned.

- experiments/inspect_stage_data.ipynb
  - Notebook to load and display saved stage DataFrames.

- experiments/exp_quick_train.py
  - Uses Config dims, handles optional Qp/QpI, strips extra columns, etc. (earlier fixes).

Notes
- FC2 stages now use SimulateTS-generated data with current sdf_fc1/policy_value weights.
- For quick completion, run with fewer epochs/paths; unsandboxed execution required for PyTorch.
