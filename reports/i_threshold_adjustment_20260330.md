# I Threshold Adjustment

日期：2026-03-30

## 背景

在加入 `bp diagnostic` 的多 `i` 叠加图后，可以直接看到：

- 在 `safe state` 下，较低的 `i`（例如 `0.0`、`0.125`）时，`VI` 明显高于 `V0`
- 当 `i` 上升到 `0.25` 及以上时，`VI` 被整体下压，并逐渐低于 `V0`

这说明当前“几乎不投资”的主要原因不是 `PI` 分支完全失真，而是投资成本上界过高。

## 修改

修改文件：

- [`config/constants.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/config/constants.py)

将：

- `I_THRESHOLD = 0.5`

改为：

- `I_THRESHOLD = 0.2`

## 原因

当前模型里投资收益端主要来自：

- 债务融资收益 `eta * ((1-kappa_b) * g * QpI - Q)`
- continuation 放大项 `g * M * P_{t+1}`

而成本端是直接减去 `i`。  
在现有参数下，`g - 1 = 0.14`，若 `i` 允许到 `0.5`，则成本端显著大于典型的投资增益量级，容易把 `VI` 系统性压到 `V0` 下方。

多 `i` 叠加图已经显示，临界区域大致落在 `0.2 ~ 0.25`。
因此先把 `I_THRESHOLD` 降到 `0.2`，属于最小且最有针对性的修正。

## 预期

若判断正确，修改后应看到：

1. `bar_i_cond` 明显抬升
2. `PIDIFF` 的正值区域扩大
3. simulate 中的投资发生率上升
4. `VI` 不再系统性低于 `V0`
