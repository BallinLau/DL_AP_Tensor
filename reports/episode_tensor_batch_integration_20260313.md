# Episode Tensor Batch 接入（去掉训练前 pandas 拼装）

日期：2026-03-13  
项目：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor`

## 目标
- 在 `training/episode.py` 中把训练批次构建切到 tensor 管线。
- `mode0/modeA/modeB` 的 SDF 与 Policy/Value 训练前不再依赖 DataFrame 拼装。
- 保留 DataFrame fallback（兼容旧调用与 FC2）。

## 代码改动

### 1) 超参数开关
文件：`config/hyperparams.py`

- 新增：
  - `use_tensor_pipeline: bool = True`

含义：Episode 优先使用 tensor 数据流；关闭后回退旧 DataFrame 流程。

### 2) Episode 新增 tensor 数据状态
文件：`training/episode.py`

- 新增成员：
  - `self.tensor_firm: Optional[TensorTable]`
  - `self.tensor_macro: Optional[TensorTable]`
  - `self.tensor_sdf: Optional[TensorTable]`

### 3) 新增 tensor 批次与配对工具
文件：`training/episode.py`

- `_use_tensor_pipeline()`
- `_table_to_dataframe()`（仅兼容导出）
- `_encode_int_keys()` / `_match_keys()`（GPU 键匹配）
- `_build_batches_from_parent_children()`（统一 batch 打包+eta 重采样）
- `_create_firm_batches_from_tensor()`
  - 同时兼容 `Sample.build_policy_value_tensor` 与 `SimulateTS.simulate_tensor` 输出
  - 使用共享 key 基数对齐 parent/children，避免存在 entrant（child 侧新 ID）时错配
- `_build_sdf_pairs_from_macro_tensor()`
  - 从 macro tensor 直接构造 SDF stage2 配对
- `_create_sdf_batches_from_macro_tensor()`
  - 直接产出 SDF 训练 batch（含 add_FC1loss 两种布局）

### 4) SimulateTS 调用改造
- 新增 `_simulate_tensor(...)`
  - 默认只保留 tensor 输出
  - `export_df=True` 时才转 DataFrame（给 FC2 用）
- `_simulate_df(...)` 保留旧行为。

### 5) run_episode 三种模式接入 tensor
- `mode0`：
  - SDF stage1：`Sample.build_sdf_fc1_tensor()` -> `_create_sdf_batches_from_macro_tensor`
  - PV：`Sample.build_policy_value_tensor()` -> `_create_firm_batches_from_tensor`
  - SimulateTS：`_simulate_tensor(h=1)`（FC2 开启时才导出 df）
- `modeA`：
  - PV 用 tensor batch
  - SimulateTS 用 tensor
  - SDF stage2 从 macro tensor 直接配对
- `modeB`：
  - SimulateTS(h=T) 用 tensor
  - PV/SDF 都从 tensor 直接组 batch

### 6) 诊断避免训练前 pandas 依赖
- 新增 `_macro_forecast_r2_tensor()`，直接在 tensor 上算 R²（含分支统计）。
- `_run_sdf_recon_from_macro()` 与 `modeB` 诊断优先使用 tensor R²。

### 7) 兼容性
- FC2 流程仍基于 DataFrame。
- `_run_fc2_epochs()` 若当前只有 `tensor_firm`，会自动末端转一次 DataFrame。
- 旧 `_create_*_from_df` 与 `build_sdf_pairs_from_macro_ts` 仍保留 fallback。
- `run_episode` 结束时会把 `tensor_firm/tensor_macro/tensor_sdf` 导出到 `df/df_macro/df_sdf`，保证现有 `run_multi_episode*.py` 的画图与存档不需要改。

## 验证

1. 语法检查：
- `python3 -m py_compile training/episode.py config/hyperparams.py`

2. smoke test（已执行）：
- `Sample.build_policy_value_tensor` -> `_create_firm_batches_from_tensor` 正常
- `Sample.build_sdf_fc1_tensor` -> `_create_sdf_batches_from_macro_tensor` 正常
- `SimulateTS.simulate_tensor` -> `_build_sdf_pairs_from_macro_tensor` -> stage2 batch 正常
- `run_episode(modeB, train_modules=[])` 调度可运行，且可输出 `macro_diag_modeb`
- 含 `entry=True` 的 SimulateTS 数据下，firm/sdf 对齐仍正常（key 编码修复验证）

## 当前边界
- FC2 训练仍需要 DataFrame（你已明确该模块后续会重写）。
- `Trainer.generate_data/fill_fc1/fill_policy_value` 旧接口仍是 df-first；主改造针对 `run_episode` 训练主流程。
