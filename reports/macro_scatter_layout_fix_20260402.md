# Macro Scatter Layout Fix

## 问题

`ep*_macro_hatc.png` 和 `ep*_macro_lnk.png` 在加入更多 law-consistency 统计量后，标题过长，导致：

- 标题被截断
- 关键信息难以阅读
- 图面拥挤

## 修改

在 `experiments/run_utils.py` 中：

- 主标题只保留短标题
- 将 `R2 / corr / slope / std_ratio` 移到图上方单独一行
- 略微增大图尺寸
- 为统计行预留顶部留白

## 目的

让宏观散点图继续保留 richer diagnostics，同时避免标题挤压导致的信息不可读问题。
