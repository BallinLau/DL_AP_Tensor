# Sample + SimulateTS Tensor 化（Phase 1）

日期：2026-03-13  
项目：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor`

## 目标
- 将 `Sample` 与 `SimulateTS` 的数据生成改为 Tensor-first。
- 在模拟/采样阶段尽量保持 CUDA 张量，不在中间流程落地 DataFrame。
- 仅在末端（需要保存/画图时）再转换为 DataFrame。

## 本次已完成

### 1) 新增统一 Tensor 数据容器
文件：`data/tensor_data.py`

- `TensorTable(data, columns)`：
  - 约束二维张量 + 显式列名；
  - 支持 `.to(device)`；
  - 支持末端 `.to_dataframe()`。
- `TensorSimulationOutput(firm, macro, meta)`：
  - 统一承载 firm 面板与 macro 面板；
  - 一次性 `to_dataframes()` 导出。
- `cat_rows(...)`：
  - 安全拼接 ragged path 结果（空结果可处理）。

### 2) SimulateTS 增加 tensor-native 主路径
文件：`data/simulate_ts.py`

- 新增列定义：
  - `FIRM_COLUMNS`
  - `MACRO_COLUMNS`
- 新增 `simulate_tensor()`：
  - 返回 `TensorSimulationOutput`；
  - 逐 path 调用 `_simulate_path_tensor`；
  - 全程在 device 上拼接。
- `simulate()` 改为：
  - 内部调用 `simulate_tensor()`；
  - 仅末端 `to_dataframes()`。
- 新增 tensor 子流程：
  - `_initialize_path_tensor`
  - `_simulate_path_tensor`
  - `_process_node_tensor`
  - `_expand_branches_tensor`
  - `_predict_macro_fc1_tensor`
  - `_apply_entry_tensor`
- 资源核算函数 `_resource_accounting` 改为同时兼容 `float` 与 `tensor` 标量。

### 3) Sample 增加 tensor-native 构建接口
文件：`data/sample.py`

- 新增列定义：
  - `SDF_COLUMNS`
  - `PV_COLUMNS`
- 新增 `build_sdf_fc1_tensor()`（全张量生成）。
- `build_sdf_fc1_df()` 改为末端转换：
  - `build_sdf_fc1_tensor().to_dataframe()`
- 新增 `build_policy_value_tensor()`：
  - 生成 parent + children 的训练面板；
  - 全程 tensor，不经 dict/DataFrame 中间态。
- `build_policy_value_df()` 改为：
  - `build_policy_value_tensor()` 末端转 DataFrame；
  - 保留训练所需字段（`path/ID/t/branch/b/z/ETA/i/x/Hatcf/LnKF/M/K/Entry`）。

### 4) 模块导出更新
文件：`data/__init__.py`

- 导出 `TensorTable` 与 `TensorSimulationOutput`。

## 验证
- 已执行语法检查：
  - `python3 -m py_compile data/tensor_data.py data/simulate_ts.py data/sample.py`
  - 结果：通过。
- 已执行最小烟雾测试（CPU）：
  - `Sample.build_sdf_fc1_tensor()` 正常返回张量；
  - `Sample.build_policy_value_df()` 正常输出 `t / t+1_0 / t+1_1`；
  - `SimulateTS.simulate_tensor()` 正常返回 firm/macro 张量。

## 当前边界与下一步

### 当前边界（Phase 1）
- `Sample.build_df()` 的旧 DataFrame 流程仍保留（为了兼容旧逻辑）。
- tensor 版优先覆盖 SDF/PV 训练数据构建；FC2 兼容未作为本阶段目标。
- `SimulateTS` 旧 `_simulate_path`/`_process_node` 仍在文件中保留，便于回溯对照。

### 下一步（Phase 2 建议）
- 在 `training/episode.py` 增加 `use_tensor_pipeline`，直接吃 tensor batch（不经 df）。
- 将 `data_utils.py` 中 SDF pair 与 batch 组织逻辑补齐 tensor 版本，替代 pandas merge/groupby。
- 统一随机数生成器与 dtype（建议 bfloat16/fp16 + 关键量 fp32）。
