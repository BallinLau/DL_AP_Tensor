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
- [x] F6. `FC2` epochs 改成按 path mini-batches 训练，不再每个 epoch 对整轮 paths 一次性建图。

### G. 验证

- [x] G1. 通过 `py_compile` 静态检查。
- [x] G2. 确认 FC2 loss 路径仍可运行。
- [x] G3. 确认 parent/children diagnostics 仍能正常输出。
- [x] G4. 新增独立 supervised probe，脱离 closure 检查当前 FC2 summary 对 `hatc / lnk` 的直接可预测性。

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
- 已完成第五刀：
  - `FC2LossPipe._pv_forward()` 已支持按 row chunk 分批调用 `policy_value`，避免一次性对全部真实 firms rows 建图；
  - `Episode._run_fc2_epochs()` 在 FC2 epochs 期间会临时冻结 `policy_value` 参数，并在结束后恢复；
  - 新增了 `fc2_pv_chunk_size` 与 `fc2_freeze_pv_during_epochs` 两个超参数用于控制 FC2 阶段的显存占用。
- 已完成第六刀：
  - 修正 tensor-native ragged parent/children 分支，确保其 `policy_value` 调用也走 `_pv_forward()` 的 chunked 路径，而不是直接整批 `pv_model(...)`；
  - 将 `fc2_pv_chunk_size` 的默认值从 `50000` 下调到 `10000`，降低 80G 训练时的峰值显存。
- 已完成第七刀：
  - `Episode._run_fc2_epochs()` 改成按 path mini-batches 训练；
  - 新增 `fc2_path_batch_size` 超参数，默认按 `1024` 个 paths 组成一个 FC2 batch；
  - tensor-native FC2 路径会在每个 epoch 内重新按 path 切分 `firm_table / macro_table`，而不是整轮 `15000` 条 paths 一次性建图。
- 已完成第八刀：
  - 修正 `FC2LossPipe._update_children_state()` 在 tensor-native 路径下对 `child_state` 的原地写入；
  - 之前的写法先取 `eta_j = child_state[:, j, 2]` 这个 view，再对同一 `child_state[:, j, 0]` 原地赋值，会触发 backward 的 version mismatch；
  - 现已改为一次性函数式构造 `new_b` 和新的 `updated_child_state`，避免 inplace autograd 冲突。
- 已完成第九刀：
  - 新增 `fc2_supervised_pretrain_only` 开关，可只运行 `FC2 Supervised Pretrain` 并跳过 closure finetune；
  - `run_multi_episode.py` 与 `run_multi_episode_job.py` 已新增对应 CLI 开关。
- 已完成第十刀：
  - `FC2` 的 `hatc` / `lnk` 输入路径已显式拆开；
  - `hatc` 仍使用 `[b_quantiles, z_quantiles, x]`；
  - `lnk` 改为额外吃一份 `K` 的 quantile summary，即 `[b_quantiles, z_quantiles, x, K_quantiles]`；
  - tensor-native pretrain / closure 路径已统一到这套分头输入 contract。
- 已完成第十一刀：
  - 新增独立 slurm 脚本 `run_fc2_supervised_pretrain_only_80g.slurm`；
  - 可直接运行 “只保留 FC2 Supervised Pretrain” 的版本，不再手动拼接 CLI 参数。
- 已完成第九刀：
  - 按 `simulate_ts_parallel.py` 的定义重新对齐 `FC2LossPipe` 的 parent / transition / child operator；
  - parent 聚合不再用 `bar_z` 对 `K` 加权，`C` 改回 `max(C, 0)`，并补齐 `+1e-5` 的 `hatc` 口径；
  - child 当前节点改为使用 `child_present` 作为 alive 集合，而不是混入 parent `bar_z`；
  - child incumbent 的 `K` 更新改为只在 transition 段做一次：`K_{t+1}=K_t[1+(g-1)\\bar i_t]`；
  - child aggregate 阶段不再重复额外乘一次投资更新因子。
- 已完成第十刀：
  - 新增独立脚本 `experiments/run_fc2_supervised_probe.py`；
  - probe 不走 `policy_value + aggregation` 的 closure 训练，只测试当前 `FC2` summary `[b-quantiles, z-quantiles, x]` 对节点级 `hatc / lnk` 的直接预测能力；
  - 支持 `hatc-only`、`lnk-only`、`joint` 三种任务；
  - 默认使用 `SimulateTS.simulate_tensor()` 生成 `TensorTable`，再通过 `FC2Pipeline` 复用当前 summary 构造逻辑；
  - `FC2Pipeline` 额外暴露 `path_values`，确保 probe 对 summary 和 macro target 的 path 对齐是显式可检查的。
- 已完成第十一步：
  - 新增 `slurm/run_fc2_supervised_probe_80g.slurm`，用于在 80G 单卡上直接跑 `FC2` supervised probe；
  - slurm 脚本支持通过环境变量覆盖 `CKPT_DIR / CKPT_PREFIX / OUT_DIR / N_PATHS / GROUP_SIZE / HORIZON / EPOCHS / BATCH_SIZE`；
  - 同时显式设置 `MPLCONFIGDIR` 到可写目录，避免 probe 首次启动时卡在 matplotlib 字体缓存。
- 已完成第十二刀：
  - `FC2Model` 从单个共享输出层改为共享 trunk + `hatc/lnk` 双 head，减少两个目标在最后一层的硬耦合；
  - `FC2Pipeline` 新增 `build_supervised_targets_parent()` 与 `build_supervised_targets_children()`，把节点级 `(lnk, hatc)` 真值 target 暴露为可复用接口；
  - `Episode._run_fc2_epochs()` 现在支持 `FC2 supervised pretrain -> closure finetune` 两阶段训练；
  - supervised pretrain 默认使用当前 tensor simulate 的 path mini-batches，直接拟合 `FC2` summary 到节点级 `hatc/lnk`；
  - 新增超参数：
    - `fc2_supervised_pretrain_epochs`
    - `fc2_supervised_hatc_weight`
    - `fc2_supervised_lnk_weight`
- 已完成第十三刀：
  - `FC2 supervised pretrain` 改为一次性缓存 supervised dataset，再用标准 `DataLoader` 训练；
  - 不再在每个 pretrain epoch 内重复：
    - path 切 batch
    - `FC2Pipe` 重建
    - ragged path 解析
    - target 重新抽取；
  - pretrain 的主要耗时从“重复数据工程”收缩为“单次 dataset 构建 + 普通 MLP mini-batch 训练”。
- 已完成第十四刀：
  - `PVBPModel.cal_phats()` 改为并行批量计算所有 `i` 积分点；
  - `share_layer(base_state)` 与 `V0=p0_head(h)` 在 `forward()` 中只计算一次，并显式传入 `cal_phats()`；
  - 不再对每个 `i` 点重复：
    - `firm_state.clone()`
    - `_encode()`
    - `p0_head(h)`；
  - 现在仅对 `pI_head(h, i)` 做大 batch 并行评估，再回收成 `Vhat / P / chi / bar_z`。
- 已完成最小运行验证：
  - 使用合成 `TensorTable` + dummy `FC2` / `policy_value`，`FC2LossPipe.loss(...)` 可直接跑通；
  - `parent / children` diagnostics 可正常生成；
  - `Episode._compute_fc2_loss(...)` 可正常写出 `_latest_fc2_diag`，包含 `fc2_children_hatc_*` 等结构化日志字段。
- 当前状态：Phase A 的核心改造已经完成，后续可以进入真实训练验证或开始 Phase B。
