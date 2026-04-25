# Policy Schedule Realignment 2026-04-25

## 背景

在 `policy_loss_residual_alignment_20260425_141230` 这次 raw-M audit 中：

- `q_stage` 主要改善 `Q`
- `pvbp_stage` 主要改善 `P0/PI`
- `q_refresh` 主要继续压低 `Q`

但同时也看到：

- `P0` 最优大约在 `epoch 60`
- `PI` 最优大约在 `epoch 65`
- 再往后继续跑 `pvbp_stage`，`P0/PI` 没有继续改善，反而开始回退

因此，原来的默认 policy schedule：

- `q_stage = 100`
- `pvbp_stage = 100`
- `q_refresh = 20`

明显过长，尤其是 `pvbp_stage`。

## 目标

将默认 policy schedule 调整为：

- `q_stage = 40`
- `pvbp_stage = 60`
- `q_refresh = 10`

并保证这组数值在以下入口中 **真实生效**，而不是只改表面默认值：

- `HyperParams` 默认值
- `build_hyperparams()` 运行入口默认值
- `Episode.train_with_batches()` 内部阶段切换
- `run_multi_episode_job.py` 命令行覆盖逻辑
- 对应 slurm 脚本
- alignment audit / policy datagen compare 的默认参数

## 发现的问题

之前存在两个“表面改了、实际没改”的问题：

### 1. 训练内部有硬下限

在 `training/episode.py` 中，policy staged training 会做：

```python
q_stage_epochs = 0 if q_stage_cfg <= 0 else max(100, q_stage_cfg)
pvbp_stage_epochs = 0 if pvbp_stage_cfg <= 0 else max(100, pvbp_stage_cfg)
```

这意味着即使外部把 schedule 改成 `40 / 60 / 10`，
实际训练仍然会跑成至少 `100 / 100 / 10`。

### 2. 主训练入口不会显式传 q_refresh

`run_fc2_main_macro_state_80g.slurm` 之前没有把：

- `--q-stage-epochs`
- `--pvbp-stage-epochs`
- `--q-refresh-stage-epochs`

显式传给 `run_multi_episode_job.py`。

同时 `run_multi_episode_job.py` 也没有 `--q-refresh-stage-epochs` 参数，
而且 stage override 主要只在 `q_joint_continuation_ablation` 分支里处理。

结果就是：

- 一部分默认值来自 `build_hyperparams()`
- 一部分又被入口覆盖
- 一部分还会被 `Episode` 内部 `max(100, ...)` 再次覆盖

这会导致“代码默认值”和“实际运行 schedule”不一致。

## 本次修正

### 1. 默认值统一到 40 / 60 / 10

修改：

- `config/hyperparams.py`
- `experiments/run_utils.py`
- `experiments/run_policy_loss_residual_alignment_audit.py`
- `experiments/run_policy_datagen_compare.py`

统一默认：

- `q_stage_epochs = 40`
- `pvbp_stage_epochs = 60`
- `q_refresh_stage_epochs = 10`

并把：

- `q_pretrain_epochs = 40`
- `q_warmstart_epochs = 40`

同步到 `q_stage` 长度，避免 Q 阶段相关控制和主 stage 长度脱节。

### 2. 删除 Episode 内部的 100 epoch 硬下限

修改 `training/episode.py`：

从：

```python
q_stage_epochs = 0 if q_stage_cfg <= 0 else max(100, q_stage_cfg)
pvbp_stage_epochs = 0 if pvbp_stage_cfg <= 0 else max(100, pvbp_stage_cfg)
```

改成：

```python
q_stage_epochs = 0 if q_stage_cfg <= 0 else q_stage_cfg
pvbp_stage_epochs = 0 if pvbp_stage_cfg <= 0 else pvbp_stage_cfg
```

这样 schedule 才会按外部配置真实执行。

### 3. 主训练入口显式支持 q_refresh override

修改 `experiments/run_multi_episode_job.py`：

- 新增 `--q-refresh-stage-epochs`
- 允许 policy stage override 在主训练入口直接生效
- `hyperparams.epochs` 改为：

```python
q_stage + pvbp_stage + q_refresh_stage
```

而不是以前只算前两段

### 4. slurm 显式传递 policy schedule

修改：

- `slurm/run_fc2_main_macro_state_80g.slurm`
- `slurm/run_policy_loss_residual_alignment_audit_80g.slurm`

统一增加：

```bash
Q_STAGE_EPOCHS=40
PVBP_STAGE_EPOCHS=60
Q_REFRESH_STAGE_EPOCHS=10
```

并显式传参给 Python 入口。

## 现在的结论

本次修改之后：

- `40 / 60 / 10` 不再只是“配置文件里的意图”
- 它已经成为主训练、audit、policy compare 这几条主链路上的 **真实运行 schedule**

换句话说，之后如果再看到 `policy` 行为不符合预期，
至少可以排除“入口默认值、内部 hardcode、slurm 传参三者不一致”这个问题。

## 当前建议

后续所有相关实验，都先默认使用：

- `q_stage = 40`
- `pvbp_stage = 60`
- `q_refresh = 10`

只有当新的 alignment audit 明确显示：

- `P0/PI` 最优点显著早于 60
- 或显著晚于 60

才再继续调 schedule。
