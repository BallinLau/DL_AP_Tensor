# bp 修复后 P/M 异常的稳定化修改（2026-03-13）

## 1. 问题背景

你观察到：

1. `bp` 修复后明显回落；
2. 但随后 `P` 和 `M` 表现异常（`M` 偏高、`P` 易被顶偏）。

根据当前代码链路，`P0/PI` 的 Bellman 与 FOC 都直接乘 `M`，因此 `M` 的短期漂移会被放大传导到 `P`。

---

## 2. 本次修改目标

在不回退 `bp` 修复的前提下，增强 SDF 第二阶段稳定性，降低 `M` 漂移对 `P` 联立系统的冲击。

---

## 3. 代码修改

## 3.1 SDF 第二阶段增加独立学习率

文件：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/config/hyperparams.py`

新增超参数：

- `sdf_stage2_lr: Optional[float] = 2e-4`

文件：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/training/episode.py`

- `_configure_sdf_lr_for_phase(...)` 在 `add_FC1loss=True`（SDF 第二阶段）时，优先使用 `sdf_stage2_lr`，否则回退到基础 `lr`。

目的：

- 第二阶段不再默认回到较大的基础学习率，减少 `M` 在阶段切换后的抖动。

## 3.2 新增 `log(E[M])` 显式锚定损失

文件：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/config/hyperparams.py`

新增超参数：

- `sdf_log_mean_target = log(0.98)`
- `sdf_log_mean_anchor_weight_stage1 = 1.0`
- `sdf_log_mean_anchor_weight_stage2 = 5.0`

文件：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/training/episode.py`

在 `_compute_sdf_loss(...)` 中新增：

- `mean_anchor_loss = (log(E[M]) - target)^2`
- 总损失改为：
  - `main_loss + moment_weight*moment_loss + recon_weight*recon_loss + mean_anchor_weight*mean_anchor_loss`

并增加非有限值保护：

- `mean_anchor_loss` 非有限时自动置零并告警。

目的：

- 过去仅靠区间矩约束，`M` 在某些阶段可漂移到区间边缘或外侧；
- 显式锚把 SDF 均值直接拉向论文目标附近。

## 3.3 增加 SDF 锚定项诊断输出

文件：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/training/episode.py`

`_latest_sdf_terms` 新增：

- `sdf_mean_anchor_loss`
- `sdf_mean_anchor_weight`
- `sdf_mean_anchor_target`

目的：

- 能区分“矩约束主导”与“均值锚主导”的训练状态。

## 3.4 CLI 默认训练参数同步

文件：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/experiments/run_utils.py`

`build_hyperparams()` 默认值更新：

- `sdf_stage2_lr = 2e-4`
- `sdf_moment_weight = 5.0`（第二阶段加强矩约束）
- `sdf_log_mean_anchor_weight_stage1 = 1.0`
- `sdf_log_mean_anchor_weight_stage2 = 5.0`

说明：

- 让 notebook 与 CLI 的稳定性口径一致，避免“同代码不同入口”导致结论不一致。

---

## 4. 校验

已执行：

- `python3 -m py_compile config/hyperparams.py training/episode.py experiments/run_utils.py models/policy_value.py`

结果：通过。

说明：

- 本次只做了静态编译校验，尚未重新跑完整 episode 训练回归。

---

## 5. 预期效果

1. SDF 第二阶段 `log(E[M])` 波动收敛更快；
2. `P0/PI` 的 Bellman continuation 项受 `M` 异常放大概率降低；
3. `bp` 已修复的前提下，`P` 与违约区形态更稳定。
