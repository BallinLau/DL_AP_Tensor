# Conditional / Unconditional Value Refactor Step 1

Date: 2026-03-26

## Background

在当前实现里，`P0 / PI` 同时承担了两层含义：

1. 当前期仍存活时的 continuation value
2. 当前期 default 之后的总股权价值

这会导致几个解释问题：

- `bar_i = sigmoid(PI - P0)` 会在当前期已经应该 default 的区域也给出“投资倾向”
- `P0 / PI` 与 `P / bar_z` 的经济语义混杂
- 讨论 Bellman 时，很难区分 conditional value 和 unconditional value

## Theory Target

本轮改造采用最小分层：

- `V0`: 当前仍存活条件下，不投资价值
- `VI`: 当前仍存活条件下，投资价值
- `Vhat = E_i[max(V0, VI)]`: 条件总价值
- `chi`: 当前期软存活门
- `P`: 无条件总股权价值

投资决策也拆成两层：

- `bar_i_cond = sigmoid(10 * (VI - V0))`
- `bar_i = chi * bar_i_cond`

其中：

- `bar_i_cond` 只表示“如果当前还活着，投不投资”
- `bar_i` 表示真正进入模拟执行层的有效投资权重

## Step 1 Implementation

本轮只做模型语义层改造，不改 Bellman loss 主体。

### 1. `CombinedModel`

`models/share_layer.py`

- 保留原有两个 value head
- 但语义改成：
  - `V0`
  - `VI`
  - `bar_i_cond`

### 2. `PolicyValueOutput`

`models/policy_value.py`

新增字段：

- `V0`
- `VI`
- `Vhat`
- `chi`
- `bar_i_cond`

同时保留旧字段兼容现有训练/日志代码：

- `P0 == V0`
- `PI == VI`
- `Phat == Vhat`

### 3. `PolicyValueModel.forward`

当前逻辑改成：

1. `combined_model(firm_state)` 输出 `V0, VI, bar_i_cond`
2. `cal_phats(...)` 构造：
   - `Vhat`
   - `chi`
   - `bar_z = 1 - chi`
   - `P = clamp_min(Vhat, 0)`
3. 实际执行投资权重：
   - `bar_i = chi * bar_i_cond`
4. `bp = bar_i * bpI + (1 - bar_i) * bp0`

### 4. Sample / Simulate / Diagnostic Outputs

为了让后续诊断不再把“条件投资比较”和“真正执行的投资权重”混在一起，本轮也把外层输出补齐了：

- `data/sample.py`
- `data/simulate_ts.py`
- `data/simulate_ts_parallel.py`
- `experiments/run_utils.py`
- `experiments/run_episode0_full.py`

新增显式记录：

- `Bar_i_cond`
- `Chi`

这样后面看：

- sample 数据
- simulate firm panel
- `(b, z)` heatmap

时，可以直接对比：

- `bar_i_cond`: 条件投资边界
- `bar_i`: 真正执行的 gated 投资权重
- `chi`: 当前期软存活门

### 5. Training-Layer Compatibility

`training/episode.py` 本轮没有改 Bellman 主方程，但做了两件兼容工作：

1. `P0 / PI / Phat` 读取增加了别名支持：
   - `P0 -> V0`
   - `PI -> VI`
   - `Phat -> Vhat`

2. policy/value 诊断项新增：
   - `*_bar_i_cond_mean`
   - `*_bar_i_eff_mean`
   - `*_chi_mean`
   - `*_vhat_mean`（P0 / PI 两个头里）

这样训练日志里可以开始区分：

- 条件投资比较是否合理
- 当前期存活门是否把执行层关掉
- 条件总值 `Vhat` 和最终 `P` 是否仍然混淆

### 6. Bellman Semantics Clarified In Code

本轮进一步把理论口径写进了 loss / episode 注释中，但**没有改变数值公式本身**。

涉及：

- `losses/p0_loss.py`
- `losses/pi_loss.py`
- `training/episode.py`

明确写清：

- 左边 `P0 / PI` 在现阶段应解释为：
  - `V0 / VI`
  - 即当前期 survival-conditioned values
- 右边 `P_children` 继续解释为：
  - child state 上的下一期总股权价值 `P_{t+1}`

也就是本轮先做到：

```math
V_t^0 = CF_t^0 + E_t[M_{t,t+1} P_{t+1}]
```

```math
V_t^I = CF_t^I + g E_t[M_{t,t+1} P_{t+1}]
```

其中 `P_{t+1}` 仍然是当前实现里的 child total equity value。

这一步的目标不是重写 Bellman，而是先让：

- 模型输出语义
- 训练 helper
- loss 文档

三者一致。

## Why Keep `P = clamp_min(Vhat, 0)` For Now

理论上更彻底的无条件写法是：

```math
P = \chi \cdot Vhat
```

但本轮先保留：

```math
P = \max(0, Vhat)
```

原因：

- 这是最小改动，避免一次性冲击所有现有 loss 和下游逻辑
- 先把对象语义理顺
- 第二步再决定是否把 `P` 完全切成 gated 版本

## What Changed Economically

最关键的变化不是数值公式，而是对象解释：

- 现在 `V0 / VI` 明确是 conditional value
- 现在 `bar_i_cond` 明确是 conditional investment comparison
- 现在模拟真正执行的是 `bar_i = chi * bar_i_cond`

因此：

- 若当前 parent 已经在 default 区，`chi` 会接近 0
- 则有效投资权重 `bar_i` 也会接近 0
- 这比原先直接用 `sigmoid(PI - P0)` 更符合经济学时序

## Not Yet Changed

以下部分本轮还没有改：

- `P0 / PI` 的 Bellman loss 命名与语义
- `Q` loss
- `bar_z` 是否完全由 `Vhat` 派生，还是继续保留独立头
- `P = chi * Vhat` 的彻底切换
- `training/episode.py` 中日志命名是否整体切换成 `V0 / VI / Vhat`

这些留到 Step 2。

## Validation

已通过静态检查：

- `python3 -m py_compile models/share_layer.py`
- `python3 -m py_compile models/policy_value.py`
- `python3 -m py_compile data/sample.py`
- `python3 -m py_compile data/simulate_ts.py`
- `python3 -m py_compile data/simulate_ts_parallel.py`
- `python3 -m py_compile training/episode.py`
- `python3 -m py_compile losses/p0_loss.py`
- `python3 -m py_compile losses/pi_loss.py`
- `python3 -m py_compile experiments/run_utils.py`
- `python3 -m py_compile experiments/run_episode0_full.py`
- `python3 -m py_compile experiments/compare_local_tensor_consistency.py`

`README.md` 只做了文档更新，不参与 `py_compile`。
