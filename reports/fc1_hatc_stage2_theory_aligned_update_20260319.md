# FC1/Hatc Stage2 Theory-Aligned Update

日期：2026-03-19

## 背景

当前训练里，`M` 的均值约束和矩约束具有明确的经济学意义，可以视为 HJ bound / admissibility 的一部分；问题不在于这些约束“是否存在”，而在于 `FC1/Hatc` 尚未学稳时，`SDF/HJ` 项已经开始主导训练，导致：

- `Hatc` 预测长期失真，`R2(Hatc)` 显著为负
- `M` 的分布被较强地拉回到“均值合理但动态未必合理”的区域
- 下游 `P/Q` 会继续在错误的宏观状态上定价

这次修改的目标不是弱化经济学约束，而是把训练顺序改成更接近理论识别逻辑：

1. 先把 macro transition 尤其是 `Hatc` 学对
2. 再让 `SDF/HJ` 约束在联合阶段接管

## 本次落地的 5 项修改

### 1. Stage2 前增加 FC1 teacher-forcing 预训练

在 `SDF/FC1(stage2)` 联合训练前，新增若干轮 `FC1` 专项预训练：

- 仅更新 `FC1_C / FC1_K`
- 冻结 `SDF/value` 子网络
- 损失只使用 `Hatc_{t+1} / LnK_{t+1}` 的重建误差

实现位置：

- `training/episode.py`
  - `self._fc1_teacher_forcing_stage`
  - `_set_sdf_fc1_teacher_only_freeze()`
  - `_run_sdf_recon_from_macro()`

新增超参数：

- `fc1_teacher_forcing_epochs`
- `fc1_teacher_forcing_weight`

### 2. Stage2 有真值时，用真实 `Hatc_t / LnK_t` 作为 FC1 当前态输入

以前 stage2 batch 虽然已经带了真实的 `Hatc_t / LnK_t`，但前向递推时仍然把 `Hatcf_t / LnKF_t` 当作当前态输入，导致 FC1 在训练时就处于误差递推模式。

现在改为：

- 当 `stage2` batch 提供真实 `Hatc_t / LnK_t`
- 且 `fc1_use_true_macro_state_in_stage2=True`
- FC1 前向优先使用真实当前态

这样训练时识别的是 one-step law of motion，而不是“带着自己上一步误差继续滚”。

实现位置：

- `training/episode.py`
  - `_compute_sdf_loss()`

新增超参数：

- `fc1_use_true_macro_state_in_stage2`

### 3. Stage2 联合训练里，对 HJ 相关项做 warmup

为了避免 FC1 刚进入联合阶段时，`moment loss` 和 `log(E[M])` anchor 立刻把训练方向拉回 SDF 约束，这次加入了 warmup：

- `sdf_moment_weight`
- `sdf_log_mean_anchor_weight_stage2`

在 stage2 前若干个 epoch 内按线性因子从较小值放大到原设定。

这不是删除 HJ 约束，而是把它放到 FC1 已经学到基础 macro transition 之后再逐步接管。

实现位置：

- `training/episode.py`
  - `_compute_stage2_hj_warmup_factor()`
  - `_compute_sdf_loss()`

新增超参数：

- `sdf_stage2_hj_warmup_epochs`
- `sdf_stage2_hj_warmup_start`

### 4. `sdf_fc1` 优化器拆成 FC1 与 SDF/value 两组学习率

之前常用入口里，`sdf_fc1` 整个组合模型实际上共用一个 `AdamW(lr=1e-3)`；配置里的 `fc1_lr / sdf_lr` 名义上存在，但在这些入口没有真正生效。

这次改成：

- `fc1_model` 使用 `fc1_lr / fc1_weight_decay`
- `sdf_model + value_model` 使用 `sdf_lr / sdf_weight_decay`

并同步修改学习率调度器，使其支持 param-group base lr，而不是每步把所有 group 覆盖成同一个 lr。

实现位置：

- `experiments/run_utils.py`
- `experiments/run_multi_episode_job.py`
- `experiments/run_multi_episode.py`
- `experiments/run_episode0_full.py`
- `experiments/exp_quick_train.py`
- `training/scheduler.py`
- `training/episode.py::_configure_sdf_lr_for_phase()`

### 5. 文档与训练接口同步更新

本文件即为本次训练逻辑更新的说明文档，避免后续日志分析仍按旧逻辑理解：

- stage2 现在包含 `teacher forcing` 预训练子阶段
- stage2 joint 使用真实当前宏观态作为 FC1 输入
- HJ 约束在 stage2 早期采用 warmup
- `sdf_fc1` 优化器已改为分组学习率

## 关键文件

- `config/hyperparams.py`
- `training/episode.py`
- `training/scheduler.py`
- `experiments/run_utils.py`
- `experiments/run_multi_episode_job.py`
- `experiments/run_multi_episode.py`
- `experiments/run_episode0_full.py`
- `experiments/exp_quick_train.py`

## 预期影响

预期首先改善的是：

- `R2(Hatc)` 不再长期处于大幅负值
- `dHatcf` 的分布更接近真实 one-step transition
- `M` 不再主要由“先满足均值锚、再让 FC1 迁就”驱动

短期内不保证立刻改善的量：

- `Q`
- `P`
- firm-level 直方图里重复展开后的 `M`

这些量仍然依赖 stage2 联合训练是否把 `Hatc` 这条线真正拉回稳定区域。

## 备注

这次修改没有删除 `HJ bound` 相关的 `M` 均值锚和矩约束；修改的是训练顺序、输入口径和优化器分组，使这些经济学约束建立在更可靠的 `FC1/Hatc` 过渡规律之上。
