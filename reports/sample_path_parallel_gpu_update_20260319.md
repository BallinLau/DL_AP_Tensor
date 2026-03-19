# Sample Path 并行 GPU 化修改报告

**日期**: 2026-03-19

## 修改目标

沿用 `SimulateTS` 的思路，把 `Sample` 中原本按 `path` 逐条生成的主路径改成批量 GPU 张量生成。

这次优先覆盖：

1. `build_policy_value_tensor()`
2. `build_policy_value_df()`
3. `build_df()` 的 uniform 主路径

## 为什么这样改

原先 `Sample` 的主要瓶颈在于：

1. `build_policy_value_tensor()` 里逐 path 循环生成 parent/children
2. `build_df()` 里逐 path 调 `_generate_path()`
3. 每条 path 只生成很小一块数据，GPU 一直在吃碎片化工作量

这和之前 `SimulateTS` 的问题本质一样：
- `time` 不存在问题，因为 `Sample` 本来就是一步截面构造
- 真正该并行的是 `path` 和 `firm` 两个维度

## 本次改动

### 1. 新增并行 helper

新增文件：
- [sample_parallel.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/data/sample_parallel.py)

新增主函数：
- `build_policy_value_tables_parallel(sample, include_macro=False)`

职责：
- 一次性生成所有 path 的 parent firm 状态
- 一次性生成所有 branch child 状态
- 可选地在 `simulate` 模式下批量生成进入者和宏观表

### 2. `build_policy_value_tensor()` 改为优先走并行路径

修改文件：
- [sample.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/data/sample.py)

行为：
- 当 `sampling_mode == 'uniform'` 时，直接调用并行 helper
- 其他采样模式暂时保留旧实现，避免一次性改动过大

### 3. `build_policy_value_df()` 不再自己重复做 DataFrame 转换

新增内部转换函数：
- `_firm_tensor_table_to_df()`
- `_macro_tensor_table_to_df()`

作用：
- 保持 `path/branch/firm` 的整数化
- 恢复旧接口里的 `ID` 和 `t` 字段格式

### 4. `build_df()` 的 uniform 主路径切到并行生成

行为：
- `sampling_mode == 'uniform'` 时：
  - 直接调用并行 helper
  - 在末端转成 `df_firm/df_macro`
  - 再复用原来的 `fill_fc1 / fill_policy_value / _update_macro_from_policy`
- 非 uniform 时：
  - 回退到旧的逐 path 实现

## 改进原因

本次改动的目标不是追求“所有模式一次性完全重写”，而是先把当前真正常用、也是训练主路径的 uniform/tensor-first 部分改成 path 并行 GPU 版本。

这样做的原因是：

1. 风险更低
2. 训练收益最大
3. 不会把 `feasible/realbz` 这些次要兼容分支一起打断

## 改进行为

从原来的：
- path 串行
- firm 小批量
- DataFrame 风格思维主导

变成：
- path 并行
- firm 并行
- 末端才转 DataFrame

## 当前边界

本次并行化当前**优先支持**：
- `sampling_mode='uniform'`

对于：
- `sampling_mode='feasible'`
- `sampling_mode='realbz'`

当前仍保留旧逻辑回退。

## 预期效果

1. `Sample.build_policy_value_tensor()` 在大 `n_paths` 下应明显更快
2. `build_df()` 的 uniform 主路径不再被逐 path Python 循环拖慢
3. 训练入口中的 `Sample -> tensor batch -> episode` 会更接近真正的 GPU-first 流程

## 后续建议

下一步如果继续推进：

1. 把 `feasible/realbz` 也补成 batched 实现
2. 评估 `build_df(simulate)` 是否值得进一步完全去 DataFrame 化
3. 用服务器实测 `Sample` 与 `SimulateTS` 两部分的 wall time 和 GPU 峰值
