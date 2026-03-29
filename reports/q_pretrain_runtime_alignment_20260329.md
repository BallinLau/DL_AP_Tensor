# Q 预训练运行入口对齐说明

## 背景

在 `Q(bp)` 诊断图中，`q_unit(bp)` 与 `Q(bp)` 长期停留在 `1e-8` 到 `1e-5` 量级，同时训练日志里的：

- `q_pretrain_mode = 0`
- `q_warmstart = 0`

持续为零。

这说明问题不只是 `Q` 头学得差，更关键的是：实际运行时并没有进入 `Q-only` 预训练与 warm-start 阶段。

## 原因

当前训练逻辑在 [episode.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/training/episode.py) 中使用：

```python
q_pretrain_epochs = max(0, int(getattr(self.hyperparams, "q_pretrain_epochs", 0)))
q_warmstart_epochs = max(0, int(getattr(self.hyperparams, "q_warmstart_epochs", 0)))
q_only_epochs = max(q_pretrain_epochs, q_warmstart_epochs)
```

只有当 `q_only_epochs > 0` 时，`q_pretrain_mode` 才会变成 `1`。

但运行入口此前仍然读到：

- `q_pretrain_epochs = 0`
- `q_warmstart_epochs = 0`
- `q_pretrain_trainable_scope = 'q_head_only'`

因此即便理论上希望先把 `Q` 救起来，实际运行时并没有执行这一路径。

## 本次修改

### 1. 修改全局默认值

在 [hyperparams.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/config/hyperparams.py) 中改为：

- `q_pretrain_epochs = 10`
- `q_warmstart_epochs = 10`
- `q_pretrain_trainable_scope = 'q_path'`

### 2. 修改运行入口显式覆盖

在 [run_utils.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/experiments/run_utils.py) 的 `build_hyperparams()` 中再次显式设置：

- `hp.q_pretrain_epochs = 10`
- `hp.q_warmstart_epochs = 10`
- `hp.q_pretrain_trainable_scope = "q_path"`

这样做是为了避免后续再次出现：

- 配置文件看起来已修改
- 但实际运行入口仍沿用旧默认值

## 修改理由

当前 `Q` 已经塌到几乎常数小值，继续在：

- `recovery` 规格
- `bp` surrogate
- `CF` 分解

这些下游对象上做分析，信息量已经有限。

在这种情况下，先确保 `Q` 真正经历：

1. `Q-only` 预训练
2. warm-start 监督
3. `share_layer + q_head` 联合适配

是最必要的一步。

## 预期检查项

修正后运行日志里应能看到：

- `q_pretrain_mode = 1`
- `q_warmstart > 0`

如果这两项仍然没有出现，则说明服务器运行的仍不是最新代码版本，或者启动脚本没有拉到最新提交。
