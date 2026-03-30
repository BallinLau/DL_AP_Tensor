# PVBP 全默认塌缩修复说明（2026-03-30）

## 现象

拆分 `QModel + PVBPModel` 之后，`PVBP` 阶段出现了明显的全默认塌缩：

- `P = 0`
- `chi = 0`
- `bar_z = 1`
- continuation term 全部为 0

图上的直接表现是：

- `P_{t+1}(bp)` 全部贴在 0
- `bar_z_{t+1}(bp)` 全部贴在 1
- `V0/VI` 退化成纯当前现金流
- safe state 也被判成 `default`

## 根因

问题不在 `Q`，而在 `PVBP` block 内部 `P / chi / bar_z` 的参数化过硬，模型容易掉进错误但稳定的吸收态。

旧实现逻辑：

```python
Vhat = max_vals.mean(dim=0)
chi = sigmoid(temp * Vhat)
P = clamp_min(Vhat, 0)
bar_z = 1 - chi
```

一旦训练早期 `Vhat < 0`：

1. `P = 0`
2. `chi ≈ 0`
3. `bar_z ≈ 1`
4. continuation `P_child * (1 - bar_z_child)` 消失

之后 Bellman RHS 退化成纯当前现金流，训练就很难把 `P` 再拉回正区。

## 修复内容

### 1. `P` 从硬截断改为平滑正值映射

文件：

- [`models/policy_value.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/models/policy_value.py)

修改：

```python
P = softplus(beta * Vhat) / beta
```

替代原来的：

```python
P = clamp_min(Vhat, 0)
```

作用：

- `Vhat < 0` 时仍保留梯度
- 避免 `P` 直接贴死到 0
- 保住 continuation 通道

### 2. 降低 `chi/bar_z` 温度

文件：

- [`config/constants.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/config/constants.py)

修改：

```python
BARZ_LOGIT_TEMP: 10.0 -> 3.0
```

作用：

- 减少 `chi = sigmoid(temp * Vhat)` 的过快饱和
- 避免 `bar_z` 在训练早期迅速贴到 1

### 3. 为 PVBP 阶段加入 anti-collapse warmup

文件：

- [`config/hyperparams.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/config/hyperparams.py)
- [`training/episode.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/training/episode.py)

新增超参数：

```python
pvbp_anti_collapse_warmup_epochs = 20
pvbp_anti_collapse_start = 0.25
```

实现方式：

- 仅在 `PVBP` 阶段生效
- 最初若干轮把 `chi` 向 `1` 做线性 warmup 混合
- 让 `bar_z = 1 - chi` 在训练初期不要立刻贴到 1

具体形式：

```python
chi = warmup_factor * sigmoid(temp * Vhat) + (1 - warmup_factor) * 1
```

作用：

- 让 continuation 在 `PVBP` 训练初期仍有信号
- 打断 `P=0, bar_z=1` 的全默认吸收态

## 预期效果

修复后，合理预期是：

- `P(bp)` 不再整段贴在 0
- `bar_z(bp)` 不再整段贴在 1
- continuation term 恢复为非零
- `V0/VI` 不再完全退化成纯当前现金流

## 涉及文件

- [`config/constants.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/config/constants.py)
- [`config/hyperparams.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/config/hyperparams.py)
- [`models/policy_value.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/models/policy_value.py)
- [`training/episode.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/training/episode.py)
