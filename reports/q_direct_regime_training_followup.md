# Direct-Q regime 训练 follow-up（第二轮修复）

基准 commit：`82814c5ede095cdafbd7d59ead10791b9fce7a45`
（branch `exp/equity-value-xz-scaling`）

本轮只处理四块，主体经济语义未动。

---

## 1. Episode-0 direct-Q cold-start bootstrap

### 问题

原顺序是 `P stage -> freeze P -> Q0 -> QD -> QS -> polish -> BP`。fresh run 的
episode 0 里 `q_encoder/q_head` 是随机初始化的；P stage 虽然不更新 Q 参数，但 P 的
cashflow/target 会消费 Q 的数值（`CF0p = prod + ((1 - kappa_b) Qp - Q) eta`），于是

$$Q_{\rm random}^{(0)}\rightarrow P^{(1)}\rightarrow \widehat P^{T}\rightarrow \mathcal D/\mathcal S$$

随机 Q 先污染第一轮 P，再污染之后被冻结的 `Phat` 与 default region。

### 实现

`training/episode.py`：

- `_q_bootstrap_enabled()`：仅当 `q_parameterization == "direct"` **且** `episode_id == 0`
  **且** 未加载已训练的 direct-Q（`q_checkpoint_loaded=False`）时触发。
- `_run_q_bootstrap_stage(batches)`：在 `_run_policy_value_staged` 中**位于第一次
  P stage 之前**调用。只更新 `q_encoder + q_head`，其他模块全部冻结（若检测到非 Q
  参数变化则直接 `RuntimeError`）。目标函数

  $$L_{\rm boot}=\mathbb E\big[(Q-Q^\*)^2\big]+\lambda_+\;\mathbb E\big[{\rm ReLU}(-Q)^2\big]$$

  其中 `constant_unit` 模式（当前唯一实现）

  $$Q^\*=\begin{cases}0,& b=0\\ b\cdot q_{\rm boot},& b>0\end{cases}$$

  刻意**不**使用未训练 P 的 default classification，也不使用 recovery / default label。
- 日志字段：`q_bootstrap_loss`、`q_bootstrap_Q_mean/std`、`q_bootstrap_target_mean/std`、
  `q_bootstrap_negative_share`、`q_bootstrap_optimizer_steps`，外加 `q_raw_grad_norm` /
  `q_clipped_grad_norm`；summary 另含 `non_q_parameter_max_change`。
- `q_bootstrap_mode` 只支持 `"constant_unit"`；`"legacy_distill"` 需要目前不存在的
  迁移工具，因此显式 `ValueError`，不伪造实现。

bootstrap 数值失败（non-finite loss/grad）会回滚到 staged 起点并返回
`policy_value_stage_status = "failed_q_bootstrap"`。

## 2. Q0/QD/QS required-phase gate

### 问题

单个 phase 可以返回 `skipped_no_samples`，但 `_run_q_regime_training` 只把
`rejected_numerical` 当失败，因此「QS 完全没有 Bellman optimizer step」仍会整体
`accepted` 并继续 BP。

### 实现

- `_run_q_regime_phase` 现在额外返回 `coverage` 诊断：
  QD 为 `default_candidates_generated / default_candidates_selected / min_phat /
  max_phat / fraction_phat_le_0 / fraction_phat_le_minus_eps`；
  QS 为 `survival_parent_count`。用于区分「sampler 有问题」「frozen P 根本没有破产
  区域」「default region 太小」。
- `_run_q_regime_training` 增加 gate，拒绝原因互斥且优先级明确：

  | 条件 | 状态 |
  |---|---|
  | 任一 phase `rejected_numerical` | `rejected_numerical` |
  | `q_require_zero_phase` 且 Q0 steps `< q_min_zero_optimizer_steps` | `rejected_insufficient_zero_boundary` |
  | `q_require_default_phase` 且（QD steps 或 selected 样本不足） | `rejected_insufficient_default_coverage` |
  | `q_require_survival_phase` 且（QS steps 或 parent 数不足） | `rejected_no_survival_bellman` |

  返回中新增 `q_stage_required_gate_passed`、`q_stage_rejection_reason`、
  `q_zero/default/survival_optimizer_steps`、`q_polish_status`、`q_default_coverage`、
  `q_survival_coverage`。**polish 不是 required phase**。
- `_run_policy_value_staged`：Q stage 未 `accepted` 时立即回滚并返回
  `policy_value_stage_status = "failed_q"`，`bp_distillation_stage.status =
  "skipped_q_stage_rejected"` —— **不进入 BP**。`stages_successful` 也改为要求
  `q_summary.status == "accepted"`（不再接受 `skipped_missing_q_training_batches`）。

## 3. checkpoint / model-spec semantic metadata

### 问题

formal runner 只写 `epX_policy_value.pt` / `epX_sdf_fc1.pt` / `epX_fc2.pt` 三个裸
`state_dict`；而 `b_times_unit` 与 direct-Q 的 q-head 张量 shape 相同，
`load_state_dict(strict=True)` 无法发现「旧权重表示 $q_{\rm unit}$，新模型解释为 $Q$」。

### 实现

`experiments/run_utils.py`：

- 新增 `POLICY_VALUE_SPEC_FILENAME` / `load_policy_value_model_spec(ckpt_dir, prefix)`，
  在 `ckpt_dir`、`ckpt_dir/../metadata/` 下查找 `policy_value_model_spec.json`。
- `build_models(..., ckpt_dir=...)`：
  - 有 spec → 校验 `spec["q_parameterization"]` 与当前 `Config.Q_PARAMETERIZATION`
    一致，不一致直接 `ValueError`（`Explicit migration is required`）。
  - 无 spec（裸 state_dict）且当前目标是 `direct` → **默认拒绝**，除非显式传
    `allow_unsafe_raw_checkpoint=True`（此时打印强 warning，按 legacy/unknown 处理）。
  - 无 spec 且当前目标是 `b_times_unit` → 语义唯一，正常加载。
- 新增 `save_checkpoint_metadata(...)`；`save_models(...)` 保留原签名（新增可选
  keyword）并额外写：

  ```
  metadata/hyperparams.json
  metadata/config_snapshot.json
  metadata/policy_value_model_spec.json
  checkpoints/epX_combined.pt     # models + hyperparams + config_snapshot + spec + value_parameterization + git_commit
  ```

`experiments/run_multi_episode_job.py`：

- `save_models(..., hyperparams=hyperparams, extra_models={"firm_target": ...})`，
  combined checkpoint 因此包含 `firm_target`。

既有诊断/导出脚本（`export_bp_fixed_state_cross_episode.py`、`run_bp_fixed_teacher_refit_probe.py`、
`run_bp_recovery_probes.py`、`export_continuation_surface.py`、`export_target_grid_decomposition.py`、
`export_bp_deep_diagnostics.py`）读取的是旧 run root 的裸 checkpoint，已显式传入
`allow_unsafe_raw_checkpoint=True` 并注明原因；**正式 Slurm/run 不开这个 bypass**。

## 4. Q shape prior 默认置 0

`config/hyperparams.py`：`q_shape_weight_z / q_shape_weight_b_low /
q_shape_weight_b_high` 默认由 `1.0` 改为 `0.0`（代码保留，仅权重置零），
`training/episode.py` 的 `getattr` fallback 同步改为 `0.0`。
CLI 新增 `--q-shape-weight-z / --q-shape-weight-b-low / --q-shape-weight-b-high`，
Slurm 新增 `Q_SHAPE_WEIGHT_{Z,B_LOW,B_HIGH}`（默认 `0`），run 启动时打印生效值：

```
[q-baseline] q_shape_weight_z=0 q_shape_weight_b_low=0 q_shape_weight_b_high=0 (shape priors are OFF only when all three are 0)
```

形状诊断量 `q_shape_z / q_shape_b_low / q_shape_b_high` 仍然照常记录，只是不再进入 loss。

---

## 新增 / 修改的配置

| 名称 | 默认 | 说明 |
|---|---|---|
| `q_bootstrap_epochs` | `5` | episode-0 cold-start bootstrap 轮数 |
| `q_bootstrap_mode` | `"constant_unit"` | 唯一实现；`b=0 -> 0`，`b>0 -> b * unit` |
| `q_bootstrap_unit_value` | `1.0` | bootstrap target 单位值 |
| `q_bootstrap_nonnegative_weight` | `1.0` | `ReLU(-Q)^2` 权重 |
| `q_require_zero_phase` | `True` | Q0 无 optimizer step 即失败 |
| `q_require_default_phase` | `True` | QD 无 coverage 即失败 |
| `q_require_survival_phase` | `True` | QS 无 Bellman step 即失败 |
| `q_min_zero_optimizer_steps` | `1` | Q0 门槛 |
| `q_min_default_samples` | `1` | QD 选中样本门槛 |
| `q_min_default_optimizer_steps` | `1` | QD step 门槛 |
| `q_min_survival_samples` | `1` | QS parent 数门槛 |
| `q_min_survival_optimizer_steps` | `1` | QS step 门槛 |
| `q_shape_weight_{z,b_low,b_high}` | `0.0` | 默认关闭，可 override |

新增 CLI：`--q-shape-weight-{z,b-low,b-high}`、`--q-bootstrap-{epochs,mode,unit-value,nonnegative-weight}`、
`--q-require-{zero,default,survival}-phase`。

---

## 测试

新增 `tests/test_q_checkpoint_semantics.py`（11 个），并在
`tests/test_q_direct_regime_training.py` 追加 10 个：

- A：bootstrap 调用顺序（`bootstrap -> p_stage -> q_regime -> bp`）；bootstrap 只改
  `q_encoder/q_head`；target 为 `b * unit` 且 `b=0 -> 0`
- B：`episode_id > 0` 或已加载 direct-Q 时不 bootstrap（`q_bootstrap_optimizer_steps == 0`）
- C：QS 无样本 → `rejected_no_survival_bellman`；被拒时 BP 不执行（`failed_q`）
- D：QD 无覆盖 → `rejected_insufficient_default_coverage`，并保留 coverage 诊断
- E：polish `skipped_no_samples` 不导致拒绝；required-phase flag 可关闭 gate
- F：裸 checkpoint 在 direct-Q 目标下被拒；显式 opt-in 可加载；legacy 目标下语义唯一；
  spec 的 `q_parameterization` 不一致被拒；spec 一致时无需 bypass
- G：`save_models` 落盘 `metadata/` 三件套 + combined ckpt；combined 含
  `firm_target`；保存后可经 `build_models` 完整 round-trip
- H：shape weight 默认 0 且可 override

命令与结果：

```
python -m compileall -q losses evaluation experiments training config tests analysis   -> OK
git diff --check                                                                        -> 无告警
pytest tests/ -p no:randomly -q                                                         -> 569 passed
```

（基线为 `548 passed`；本轮 +21。）

---

## 仍未验证 / 风险

- `constant_unit` 只是把 direct-Q 放到有限、平滑、非随机的位置，**不是**均衡定价解；
  真正定价仍由 Q0/QD/QS/polish 完成。
- `q_bootstrap_unit_value = 1.0` 是量级占位值，尚未用 GPU run 校准；若 P stage 对 Q
  量级敏感，需要用 `q_bootstrap_Q_mean` 与 P loss 一起检查。
- required-phase gate 会新增失败模式：若某次 run 的 frozen P 覆盖不足，会得到
  `failed_q` 而不是静默 accepted。这是有意的，但需要观察正式 run 的失败率。
- `build_models` 的 guard 是**校验型**（比对 spec 的 `q_parameterization`），不是
  从 spec 重建模型；架构层面的不一致仍由 `load_state_dict(strict=True)` 负责。
- shape prior 置 0 后，Q 的 b/z 单调性不再被显式约束，需要用 `q_shape_*` 诊断量与
  evaluator 的 `Q_peak_diagnostics.csv` 观察是否出现非经济形状。
