# FC2 Phase A 接口清理 TodoList

## 1. 目标

这份文档用于跟踪 `FC2` 第一阶段（Phase A）的代码修改进度。

当前阶段只做一件事：

> **把 `FC2` 从“黑箱 pipeline”改造成“tensor-native、输入清楚、递归调用可解释、聚合诊断可解释”的模块。**

这一阶段**不改经济逻辑**，也**不切换训练主线**。  
重点是先把 `FC2` 的接口和数据流理清楚，为后续把 `FC2` 升级为主 aggregate law block 做准备。

---

## 2. 当前问题

当前 `FC2` 路径主要有以下问题：

1. `FC2` 输入构造、`policy_value` 调用、children 更新、聚合、loss 计算全部混在 `losses/FC2losspipe.py` 的 `forward()` 中。
2. `FC2` 输入 summary 的来源不够清楚，和 `DataFrame -> fill_df_to_fullN -> rebuild tensor` 这条旧路径纠缠在一起。
3. parent 与 children 的逻辑没有显式分层，后续很难单独判断：
   - parent law 是否合理；
   - children law 是否合理；
   - 还是 `policy_value + aggregation` 的闭环出了问题。
4. `training/episode.py` 中 `_compute_fc2_loss()` 仍走 DataFrame 路径：
   - branch 原地改写；
   - `full_N = 1000` 硬编码；
   - 先把 tensor 状态转回 DataFrame，再在 FC2 pipe 中重新 fill / rebuild tensor；
   - 结构化输出暴露不够。

---

## 2.1 当前已经可直接使用的 tensor state

根据当前 `simulate_tensor_parallel` 的实现，`FC2` 在训练前已经可以拿到这些 GPU 上的 node-level 张量状态：

- `state["x"]`
- `state["b"]`
- `state["z"]`
- `state["eta"]`
- `state["i"]`
- `state["K"]`
- `state["alive"]`
- `state["entry"]`
- `state["hatcf"]`
- `state["lnkf"]`
- `state["hatc_curr"]`
- `state["lnk_curr"]`
- `state["M"]`
- `state["bar_i"]`
- `state["bar_z"]`
- `state["bp"]`

并且 `simulate_tensor_parallel` 在每个时间步上已经显式产生：

- parent node rows
- child branch rows
- parent macro rows
- child macro rows

因此，`FC2` 的 Phase A 应该优先改造成：

> **直接从 simulate 的 tensor state / node view 构造 FC2 输入，而不是走 DataFrame 中转。**

---

## 3. 本阶段不做什么

本阶段明确**不做**以下事情：

- 不修改 `FC1`。
- 不让 `FC2` 立即接管整个训练主线。
- 不删除 `full_N` / padding 机制，但不再让 FC2 summary 主路径依赖 `df -> fill -> tensor`。
- 不重写 `policy_value`。
- 不修改 `FC2` 网络结构和输入维度。
- 不在本阶段引入新的经济状态变量。

---

## 4. Todo Checklist

### A. 文档与接口定义

- [x] A1. 在代码中固定 `FC2` 的 contract：
  - 输入：node-level cross-sectional summary + node-level `x`
  - 输出：node-level `(\hat c, \ln K)`
- [x] A2. 在 `FC2` 路径中明确区分：
  - clean summary representation
  - tensor bookkeeping / padded representation

### B. tensor contract 梳理

- [x] B1. 明确 `FC2` 训练时直接消费哪些 tensor state。
- [x] B2. 明确 parent node tensor view 的 contract：
  - `b_parent`
  - `z_parent`
  - `x_parent`
  - `K_parent`
  - `alive_parent`
- [x] B3. 明确 children node tensor view 的 contract：
  - `b_child`
  - `z_child`
  - `x_child`
  - `K_child`
  - `alive_child`

### C. `losses/FC2losspipe.py` 拆分（tensor-native）

- [x] C1. 新增 tensor summary 构造函数：
  - 从 parent tensor view 直接提取 `b/z` quantiles 与 `x`。
- [x] C2. 新增 children tensor summary 构造函数：
  - 从 children tensor view 直接提取 `b/z` quantiles 与 `x`。
- [ ] C3. 新增 `build_fc2_input_from_summary(...)`。
- [x] C4. 改写 `build_fc2_input_parent()`：
  - 从 parent tensor summary 构造输入。
- [x] C5. 改写 `build_fc2_input_children()`：
  - 从 children tensor summary 构造输入。
- [x] C6. 去掉 FC2 训练主路径中的：
  - `fill_df_to_fullN`
  - `df merge`
  - `_build_tensors()` rebuild tensor
  - `full_N` padded tensor 作为 FC2 tensor-native 主路径内部表示

### D. `FC2Pipeline.forward()` 显式分层

- [x] D1. 拆出 parent FC2 前向步骤。
- [x] D2. 拆出 parent `policy_value` 调用步骤。
- [x] D3. 拆出 parent aggregate 计算步骤。
- [x] D4. 拆出 child state update 步骤。
- [x] D5. 拆出 children FC2 前向步骤。
- [x] D6. 拆出 children `policy_value` 调用步骤。
- [x] D7. 拆出 children aggregate 计算步骤。
- [x] D8. 将 `forward()` 返回值改为结构化 dict：
  - `parent`
  - `children`
  - `total`

### E. `loss()` 与 diagnostics

- [x] E1. 改写 `loss()`，让 parent/children diagnostics 直接读取结构化输出。
- [x] E2. 明确 parent 和 children 的：
  - `corr`
  - `slope`
  - `std_ratio`
  - `rmse`
- [x] E3. 保留 `loss_parent / loss_children / loss_total`，但显式挂到结构化结果里。

### F. `training/episode.py` 调用侧清理

- [x] F1. `_compute_fc2_loss()` 改成优先接收 tensor-native FC2 state，而不是 DataFrame。
- [x] F2. `_compute_fc2_loss()` 中不再原地改 `df['branch']`。
- [x] F3. 去掉 `full_N = 1000` 的硬编码，改为从 config / hyperparams 读取。
- [x] F4. 保留 `self._last_fc2_pipe`，同时新增 `self._last_fc2_outputs`，便于调试。
- [x] F5. 保证训练日志能读到结构化的 parent/children diagnostics。

### G. 验证

- [x] G1. 通过 `py_compile` 静态检查。
- [x] G2. 确认 FC2 loss 路径仍可运行。
- [x] G3. 确认 parent/children diagnostics 仍能正常输出。

---

## 5. 执行顺序

建议按下面顺序执行：

1. 先确认 `episode.py` 在 FC2 训练前可直接拿到的 tensor state。
2. 再改 `losses/FC2losspipe.py` 为 tensor-native summary 输入。
3. 再改 `training/episode.py` 的调用侧。
4. 最后做静态检查和最小运行检查。

---

## 6. 变更日志

### 2026-04-09

- 新建本 Todo 文档。
- 将 Phase A 计划修正为 tensor-native 路线。
- 已确认 `simulate_tensor_parallel` 当前 state 中已经持有 FC2 所需的大部分 GPU 张量状态。
- 已完成第一刀：
  - `FC2LossPipe` 新增 `TensorTable` 输入路径；
  - `FC2` 训练在 tensor pipeline 下优先直接消费 `self.tensor_firm / self.tensor_macro`；
  - `_compute_fc2_loss()` 不再原地修改 DataFrame；
  - `full_N` 默认值改为从 hyperparams 读取；
  - `py_compile` 已通过。
- 已完成第二刀：
  - `FC2Pipeline.forward()` 已拆成 parent / children 的显式子步骤；
  - `forward()` 已返回结构化的 `parent / children / total` dict；
  - `loss()` diagnostics 已改为读取结构化输出；
  - 兼容旧键名的返回值仍然保留；
  - `py_compile` 再次通过。
- 已完成第三刀：
  - `Episode` 新增 `_latest_fc2_diag`；
  - `FC2` 的 parent/children diagnostics 已挂到训练日志输出；
  - `py_compile` 再次通过。
- 已完成第四刀：
  - `FC2LossPipe` 的 tensor-native 主路径已改成 ragged per-path 表示，不再在 pipe 内部构造 `full_N` padded parent/children tensors；
  - `policy_value` 在 FC2 loss 中只对真实 parent/child rows 前向，不再对 `n_paths * full_N` 的 padded rows 全量前向；
  - `episode._run_fc2_epochs()` 的 tensor-native FC2 batch 不再传递 `full_N`；
  - `full_N` 现在只保留在 df fallback 路径中。
- 已完成最小运行验证：
  - 使用合成 `TensorTable` + dummy `FC2` / `policy_value`，`FC2LossPipe.loss(...)` 可直接跑通；
  - `parent / children` diagnostics 可正常生成；
  - `Episode._compute_fc2_loss(...)` 可正常写出 `_latest_fc2_diag`，包含 `fc2_children_hatc_*` 等结构化日志字段。
- 当前状态：Phase A 的核心改造已经完成，后续可以进入真实训练验证或开始 Phase B。
