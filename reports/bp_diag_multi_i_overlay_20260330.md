# BP Diagnostic Multi-i Overlay

日期：2026-03-30

## 修改目的

当前 `bp diagnostic` 图中的第三行第一列原本只画：

- `CF0(bp)`
- `CFI(bp)`
- `V0 diag(bp)`
- `VI diag(bp)`

其中投资分支只使用单个参考状态里的固定 `i` 值。  
这样只能看到“在当前这个 `i` 下，投资值是否高于不投资值”，但不能判断：

- 是不是因为当前 `i` 偏高，才导致 `VI < V0`
- 如果 `i` 更低，投资分支是否会恢复合理区域

因此新增了“多 `i` 叠加”诊断。

## 修改内容

修改文件：

- [`experiments/run_utils.py`](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/experiments/run_utils.py)

在 `plot_bp_diagnostic_curves(...)` 中，第三行第一列
`Current cash flow and one-step V`
这张图现在改为：

- 保留 `CF0(bp)` 与 `V0 diag(bp)`
- 对投资分支叠加多条不同 `i` 下的：
  - `CFI(bp)`
  - `VI diag(bp)`

默认 `i` 取：

- `torch.linspace(0.0, Config.I_THRESHOLD, steps=5)`

也就是当前默认参数下的 5 个点：

- `0.000`
- `0.125`
- `0.250`
- `0.375`
- `0.500`

同时会自动标出最接近当前 `ref_state["i"]` 的那条曲线，并在图例里加 `[ref]`。

## 解释方式

这张图现在可以直接回答：

1. 当 `i` 从高往低变化时，`CFI` 是否明显抬升。
2. `VI` 是否在较低 `i` 时开始超过 `V0`。
3. 当前“几乎不投资”究竟是：
   - 投资成本 `i` 太高
   - 还是 continuation / Q 融资收益本身仍然不够强

## 预期现象

如果问题主要来自 `i` 偏高，那么应看到：

- 低 `i` 曲线下，`CFI` 和 `VI` 明显上移
- 部分 `bp` 区间里 `VI(i_{low}) > V0`

如果即使在很低 `i` 下，`VI` 仍普遍低于 `V0`，则问题更可能来自：

- `Q` 形状仍不对
- continuation 项偏弱
- `PI` 分支的收益放大机制还没有学出来
