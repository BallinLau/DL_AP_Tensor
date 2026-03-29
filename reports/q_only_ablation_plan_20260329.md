# Q-only Ablation Plan (2026-03-29)

## 目的

这个实验用于回答一个更基础的问题：

- `Q` 是否能在 **不受 `P0/PI/bp` 联合训练干扰** 的情况下先被训练好？
- 还是说 `Q` 从一开始在 `Q-only` 阶段就已经训练失败？

如果 `Q-only` 阶段都学不出正常的 `Q` surface / safe-state `Q(bp)`，那么后续 joint stage 不是主因。

## 入口修改

已在主训练入口加入：

```bash
--q-only-ablation
```

文件：

- `/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/experiments/run_multi_episode_job.py`

## 这个开关会做什么

### 1. 只训练 `policy_value`

训练模块改为：

```python
train_modules = ["policy_value"]
```

也就是不训练：

- `sdf_fc1`
- `fc2`

### 2. 整个 run 都停留在 `Q-only` 阶段

会把：

```python
q_pretrain_epochs = epochs
q_warmstart_epochs = epochs
```

因此在 `_run_batches(...)` 里，整个 `policy_value` 训练都会满足：

```python
policy_loss_terms = ['q']
```

而不会切回：

```python
['p0', 'pi', 'q']
```

### 3. 保持诊断图输出

为了实验结束后立即判断结果：

- `bp_diag_enabled = True`
- `bp_diag_every_n_episodes = 1`
- `bp_diag_states = "safe"`

所以每个 episode 结束后仍会输出：

- `Q surface`
- `safe bp diagnostics`

## 这个实验能回答什么

它能直接区分两种情况：

### A. `Q-only` 本身就失败

如果在这个 ablation 下仍然看到：

- safe 区 `Q(bp)` 近零
- `q_unit(bp)` 量级异常
- `Q surface` 只有局部脊线或大面积塌缩

那么问题主要在：

- `q_loss` 对象定义
- `Q` 方程实现
- `Q` 训练数值性质

而不是 joint stage 把它训坏。

### B. `Q-only` 其实能学好

如果在这个 ablation 下：

- safe 区 `Q(bp)` 量级正常
- `q_unit(bp)` 形状正常
- `Q surface` 比 joint 训练下明显更合理

那么后续 joint stage 的确是主要破坏来源。

## 这个实验不能回答什么

它不能直接证明：

- `Q` 的经济对象已经正确定义
- `Q` 和 `bp/P` 的联动已经正确

因为它只是把 joint stage 干扰拿掉，不会自动修复“债务对象错位”。

## 推荐运行方式

建议先跑一个最小版本，例如：

```bash
python3 -u experiments/run_multi_episode_job.py \
  --q-only-ablation \
  --n-episodes 1 \
  --epochs 10 \
  --n-paths 200 \
  --simulate-horizon 50
```

然后重点看：

1. `Q surface`
2. `safe state` 的 `Q(bp)`
3. `safe state` 的 `q_unit(bp)`

## 解释优先级

若结果仍然坏：

1. 优先怀疑 `Q` 的对象定义和主方程实现
2. 再看 state-space 覆盖 / loss 分布
3. 不优先归因给 joint stage
