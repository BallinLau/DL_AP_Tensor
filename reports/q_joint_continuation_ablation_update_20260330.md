# Q-only -> Joint Continuation 对照实验接线说明

日期：2026-03-30

## 目标

把下面这个最小对照实验直接接入主运行入口：

1. 先在同一份 `policy_value` 训练数据上跑 `Q-only` 若干 epoch；
2. 不重采样、不重建模型，直接从该状态继续跑 `joint` 若干 epoch；
3. 分别导出：
   - `Q-only end`
   - `joint end`

这样可以直接回答：

- `Q-only` 学出来的 `Q(bp)` 是否会在后续 joint 训练中被破坏；
- 若会，破坏是否发生在同一份数据、同一模型、同一条训练路径的阶段切换之后。

## 改动思路

遵循最小改动原则，没有重写训练逻辑，而是复用现有 `Episode._run_batches(...)` 中已经存在的阶段切换机制：

- 前 `q_pretrain_epochs` / `q_warmstart_epochs` 走 `Q-only`
- 后续 epoch 自动切到 `['p0', 'pi', 'q']` joint

本次新增的只是“阶段边界可观测性”：

1. 在 `policy_value` 训练循环里暴露阶段快照回调；
2. 在 runner 层收到回调后，立即保存该时点模型和图；
3. 用 `tag` 区分同一 episode 内的不同阶段产物，避免覆盖。

## 代码改动

### 1. `training/episode.py`

新增 `policy_stage_callback` 接口，并贯穿到 `policy_value` 的 `_run_batches(...)` 调用。

在 `policy_value` 训练中新增两个阶段快照：

- `q_only_end`
  - 触发条件：第 `q_only_epochs` 个 epoch 结束
- `joint_end`
  - 触发条件：整个 `policy_value` 训练结束，且存在 joint continuation

同时把阶段摘要写入返回结构：

- `module_summaries['policy_value']['stage_summaries']['q_only_end']`
- `module_summaries['policy_value']['stage_summaries']['joint_end']`

阶段摘要包含：

- `phase`
- `epoch`
- `n_epochs`
- `q_only_epochs`
- `joint_epochs`
- `final_losses`
- `convergence`（仅最终阶段有）

### 2. `experiments/run_utils.py`

新增 tag-aware artifact 命名支持。

#### 新增辅助函数

- `build_policy_ref_state(df)`
- `_episode_prefix(ep, tag)`
- `_episode_title_prefix(ep, tag)`

#### 改动的 helper

- `save_models(..., tag=None)`
- `plot_surfaces(..., tag=None)`
- `plot_bp_diagnostic_curves(..., hyperparams=None, tag=None)`
- `plot_distributions(..., tag=None)`

效果：

- 默认行为不变；
- 指定 `tag` 时，同一 episode 内可并存多套 checkpoint/figures。

示例命名：

- `ep0_qonly_end_policy_value.pt`
- `ep0_joint_end_policy_value.pt`
- `ep0_qonly_end_bp_diag_safe.png`
- `ep0_joint_end_bp_diag_safe.png`

另外把 `plot_bp_diagnostic_curves(...)` 改成可显式接收 `hyperparams`，避免函数内部重新构造默认超参数后覆盖运行时设置。

### 3. `experiments/run_multi_episode_job.py`

新增 CLI 开关：

- `--q-joint-continuation-ablation`
- `--q-only-epochs`
- `--joint-epochs`

约束：

- 不能和 `--q-only-ablation` 同时开启。

该模式下的行为：

1. `train_modules` 强制为 `['policy_value']`
2. `hyperparams.q_pretrain_epochs = q_only_epochs`
3. `hyperparams.q_warmstart_epochs = q_only_epochs`
4. `hyperparams.epochs = q_only_epochs + joint_epochs`
5. `policy_stage_callback` 在阶段边界导出 artifacts

runner 侧新增：

- `materialize_episode_outputs(...)`
- `make_json_safe(...)`
- `export_policy_stage_artifacts(...)`

阶段导出内容包括：

- tagged checkpoints
- tagged surface plots
- tagged bp diagnostic plots
- tagged distribution plots
- `data/outputs/ep{ep}_{tag}_summary.json`

## 为什么这样设计

### 1. 不重采样

这是这次实验最关键的控制点。

如果 `Q-only` 和 `joint` 分别在不同数据上训练，就没法把 `Q` 的退化明确归因到 joint stage 本身。

本次实现中，两个阶段都发生在同一轮 `Episode.run_episode(...)` 的同一份 `pv_batches` 上。

### 2. 不复制训练器

现有训练循环已经支持阶段切换，重新写两套 runner 或两次单独训练只会增加状态漂移和维护成本。

因此这次只增加“阶段快照”，不改训练数学对象。

### 3. 导出逻辑放在 runner 层

训练代码负责：

- 何时到达阶段边界

runner 代码负责：

- 保存模型
- 画图
- 写 summary

这样训练逻辑和实验落盘逻辑保持解耦。

## 使用方式

推荐最小实验命令：

```bash
python experiments/run_multi_episode_job.py \
  --n-episodes 1 \
  --post0-mode mode0 \
  --q-joint-continuation-ablation \
  --q-only-epochs 10 \
  --joint-epochs 10
```

如果想做快速 smoke test：

```bash
python experiments/run_multi_episode_job.py \
  --n-episodes 1 \
  --quick-test \
  --q-joint-continuation-ablation \
  --q-only-epochs 2 \
  --joint-epochs 2
```

## 预期产物

假设是 `episode 0`，将新增：

### Checkpoints

- `checkpoints/ep0_qonly_end_policy_value.pt`
- `checkpoints/ep0_joint_end_policy_value.pt`

### Figures

- `experiments/figs/ep0_qonly_end_q_heatmap.png`
- `experiments/figs/ep0_joint_end_q_heatmap.png`
- `experiments/figs/ep0_qonly_end_bp_diag_safe.png`
- `experiments/figs/ep0_joint_end_bp_diag_safe.png`

以及对应的 `surface` / `boundary` / `distribution` 图。

### Stage summaries

- `data/outputs/ep0_qonly_end_summary.json`
- `data/outputs/ep0_joint_end_summary.json`

## 说明

这次改动没有碰：

- `losses/q_loss.py`
- `losses/p0_loss.py`
- `losses/pi_loss.py`
- `models/policy_value.py`

也就是说，这次提交只是在现有训练定义不变的前提下，把“Q-only 结束”和“joint 结束”两个时点显式暴露并落盘，服务于因果更干净的对照实验。
