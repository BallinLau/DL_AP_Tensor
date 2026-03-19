# P 无违约区 + M 偏高联动问题修复（2026-03-13）

## 1. 问题复述

你反馈当前训练出现：

1. `bp` 已经较正常；
2. 但 `P` 基本没有 `=0` 区域（破产区消失）；
3. 同时 `M` 偏高，进而把 `P0/PI` Bellman continuation 顶高。

这三者在现有实现里确实是同方向联动的。

---

## 2. 根因（代码层）

## 2.1 `P` 的结构约束导致“很难出现 0 区域”

在 `PolicyValueModel.cal_phats` 中，之前使用：

- `P = softplus(Phat)`
- `bar_z = sigmoid(-temp * Phat)`

对应代码位置：

- `models/policy_value.py`（本次修改前）

`softplus` 会把 `P` 强制为严格正值（仅在非常负时趋近于 0），这与论文中“`P=max(0, \hat P)` 形成明确违约区”不一致，视觉上也会表现为很难看到“清晰的 `P=0` 区域”。

## 2.2 `P0/PI` 损失直接吃到上游偏高 `M`，会把 `P` 残差推向偏正

`_compute_p0_loss` / `_compute_pi_loss` 中，Bellman 残差含：

- `P0 - CF0p - M * P' * (1-bar_z')`
- `PI - CFip - g * M * P' * (1-bar_z')`

当样本里的 `M` 偏高时，continuation 项会被放大，网络会更倾向把 `P0/PI` 抬高来匹配残差，进一步压缩违约区。

---

## 3. 本次修改

## 3.1 将 `P` 改回论文口径：`P=max(0, Phat)`

修改：

- `models/policy_value.py:173-177`

从：

- `P = softplus(Phat)`

改为：

- `P = torch.clamp_min(Phat, 0.0)`

并保留：

- `bar_z = sigmoid(-temp * Phat)`

目的：

1. 允许出现明确 `P=0` 的违约区域；
2. `bar_z` 仍由 `Phat` 平滑生成，保留边界学习信号。

## 3.2 在 `P0/PI` 路径引入可配置的 `M` clip

新增超参数：

- `config/hyperparams.py:126-129`
  - `pv_use_clipped_m`
  - `pv_m_clamp_min`
  - `pv_m_clamp_max`

在损失中接入：

- `training/episode.py:909-919`（P0）
- `training/episode.py:1070-1080`（PI）

默认行为：

- `M` 在 `P0/PI` Bellman 与 FOC 计算中裁剪到 `[0.7, 1.3]`。

作用：

- 防止上游 SDF 短期失稳时，`M` 异常值直接把 `P` 联立系统“顶偏”。

## 3.3 增加 `P0/PI` 的 `M` 诊断指标

新增日志项：

- P0：`p0_log_mean_M_raw`, `p0_log_mean_M_used`, `p0_M_raw_p90`, `p0_M_used_p90`
  - `training/episode.py:1041-1044`
- PI：`pi_log_mean_M_raw`, `pi_log_mean_M_used`, `pi_M_raw_p90`, `pi_M_used_p90`
  - `training/episode.py:1210-1213`

作用：

- 明确区分“原始 M 分布异常”还是“裁剪后依然导致 P 异常”。

## 3.4 命令行默认训练超参数同步到稳定版

修改：

- `experiments/run_utils.py:82-90`

新增默认：

- `fc1_recon_weight=1.0`
- `sdf_stage1_lr=1e-4`
- `sdf_stage1_moment_weight=5.0`
- `sdf_moment_weight=3.0`
- `pv_use_clipped_m=True`
- `pv_m_clamp_min=0.7, pv_m_clamp_max=1.3`

目的：

- 避免 notebook 与 CLI 跑法的超参数口径不一致，重复出现 “M 高 -> P 不违约”。

---

## 4. 编译校验

已执行：

- `python3 -m py_compile models/policy_value.py training/episode.py config/hyperparams.py experiments/run_utils.py`

结果：通过。

---

## 5. 你下一次运行时建议重点看

1. `sdf_log_mean_M`（SDF 自身）是否回到接近 `log(0.98)` 附近；
2. `p0/pi` 的 `*_log_mean_M_raw` 与 `*_log_mean_M_used` 差距是否明显；
3. `P` 热力图是否出现稳定 `P=0` 区域；
4. `bar_z` 是否从“几乎全 0”恢复到有边界结构。

若 `P=0` 区域恢复但 `M_raw` 仍偏高，下一步应继续收紧 SDF 第二阶段（moment 权重或 lr），而不是再改 `P` 方程。
