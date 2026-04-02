# Macro Law Mismatch Decomposition Plots

## 背景

`ep*_macro_hatc.png` 显示：

- `true = Hatc`（simulate implied aggregate）
- `pred = hatcf`（`sdf_fc1` aggregate law）

点云呈现明显竖带结构，说明 mismatch 不只是 `R2` 低，而是：

- `true` 和 `pred` 的波动尺度可能不同
- branch `-1` 可能影响整体判断
- `x -> hatcf` 与 `x -> Hatc` 的响应可能不一致

## 新增图

在 `experiments/run_utils.py` 中新增：

- `ep*_macro_hatc_branch01.png`
- `ep*_macro_lnk_branch01.png`

用于只看 simulated child branches `0/1` 的散点图。

以及：

- `ep*_macro_hatc_vs_x.png`
- `ep*_macro_lnk_vs_x.png`

用于直接比较：

- `x -> realized macro object`
- `x -> fc1 law output`

从而区分：

1. aggregate realize 端是否被聚合压平
2. `sdf_fc1` law 是否对 `x` 过度敏感

## 目的

把“law 没闭合”进一步拆成：

- branch mixing 问题
- 波动尺度错配问题
- `x` 响应错配问题
