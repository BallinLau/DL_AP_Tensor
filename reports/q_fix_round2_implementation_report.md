# Q 修复第二轮实施报告（2026-03-12）

## 1. 本轮目标
- 把 Q-only 阶段从“只关损失项”升级为“参数冻结控制”。
- 增加结构化 warm-start，避免 Q 早期塌缩到近零。
- 修复 SDF recon 目标列位不一致导致的 NaN 风险。

## 2. 代码改动

### 2.1 Q-only 冻结 + warm-start 接线
- 文件：`training/episode.py`
- 关键改动：
  - 增加阶段状态：`_current_epoch_idx`, `_q_only_stage`。
  - `_run_batches` 中 Q-only 阶段长度改为：
    - `max(q_pretrain_epochs, q_warmstart_epochs)`
  - 在 `train_step` 中，若处于 Q-only 且 `q_freeze_non_q_in_pretrain=True`：
    - 保持 `bar_i/bp/bar_z` 为模型输出，不改写方程
    - 冻结非 Q 参数
    - 仅开放 Q 路径参数（`q_head_only` 或 `q_path`）
  - warm-start：
    - `epoch < q_warmstart_epochs` 时添加
      - `q_warmstart_weight * MSE(Q, Q_warm_target(b,z,x))`
  - 新增诊断项：
    - `q_physics`, `q_warmstart`, `q_warm_weight`
    - `q_pretrain_mode`, `q_freeze_mode`

### 2.2 超参数扩展
- 文件：`config/hyperparams.py`
- 新增：
  - `q_freeze_non_q_in_pretrain`
  - `q_pretrain_trainable_scope`
  - `q_warmstart_epochs`, `q_warmstart_weight`
  - `q_warm_A`, `q_warm_b_star`, `q_warm_sigma`, `q_warm_alpha_z`, `q_warm_alpha_x`

### 2.3 SDF recon 列位兼容修复
- 文件：`training/episode.py`
- 逻辑：
  - 若 child 列数 `>=10`：使用 `8:9` 和 `9:10`
  - 若 child 列数 `>=9`：使用 `7:8` 和 `8:9`
  - 否则跳过 recon 并告警

## 3. Notebook 同步
- 文件：`tests/sdf_fc1_two_modes_test.ipynb`
- 在 policy/value 参数单元新增：
  - `q_freeze_non_q_in_pretrain = True`
  - `q_pretrain_trainable_scope = "q_head_only"`
  - `q_warmstart_epochs = 10`
  - `q_warmstart_weight = 1.0`
  - `q_warm_A/q_warm_b_star/q_warm_sigma/q_warm_alpha_z/q_warm_alpha_x`

## 4. 文档同步
- 更新：
  - `training/README.md`
  - `reports/policy_value_issue_analysis.md`
  - `reports/sdf_fc1_stability_update.md`

## 5. 最小验证结果
- 语法检查：
  - `python3 -m py_compile training/episode.py config/hyperparams.py` 通过。
- smoke（policy-only, 3 epochs）：
  - `q` loss 为有限值。
  - 阶段标志符合预期：
    - `q_pretrain_mode = [1.0, 1.0, 0.0]`
    - `q_freeze_mode = [1.0, 1.0, 0.0]`
    - `q_warm_weight = [1.0, 1.0, 0.0]`
- 严格冻结校验（`q_pretrain_trainable_scope=\"q_head_only\"`）：
  - 单步 `q-only` 更新后，`q_head` 参数有变化；
  - `share_layer / p0_head / pI_head / barz` 参数变化为 0。

## 6. 建议下一步实验
- 固定本轮代码，做 3 组对比：
  1. `q_freeze_non_q_in_pretrain=False` vs `True`
  2. `warmstart_epochs=0` vs `10/20`
  3. 高杠杆重采样关闭 vs 开启
- 主要看：
  - `Q(b,z)` 是否恢复“b 倒 U + z 递增”
  - `b≈0` 时 `Q` 是否接近 0
  - `P/P0/PI` 是否出现合理违约区间（非全域正值）

## 7. 方案修订说明
- 早期版本曾在 Q-only 阶段直接改写 `bar_i/bp/bar_z`，该做法可能偏离经济方程原义。
- 已按反馈修订为“冻结参数方案”：
  - 保留方程结构不变
  - 通过参数冻结隔离训练通道
  - notebook 默认改为 `q_pretrain_trainable_scope=\"q_head_only\"`（严格冻结）
