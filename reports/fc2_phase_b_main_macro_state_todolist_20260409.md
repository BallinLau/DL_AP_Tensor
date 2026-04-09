# FC2 Phase B 主 Macro State 接管 TodoList

## 1. 目标

Phase B 的最小目标是：

> **让 `FC2` 在递归张量主路径中成为 parent / children 节点的主 macro state generator。**

这一步只切换主递归路径中的 node-level macro state 来源：

- 旧逻辑：`FC1` 生成 `hatcf / lnkf`，再喂给 `policy_value`
- 新逻辑：`FC2` 直接从当前 node 的 cross-sectional tensor summary 生成 `hatcf / lnkf`

同时暂时保留：

- `FC1` 继续给 `M` 服务
- `FC1` 继续作为 auxiliary block 存在

---

## 2. 本阶段只做什么

本阶段只做以下事情：

1. 增加一个显式开关：
   - `fc2_as_main_macro_state`
2. 在 `simulate_ts_parallel.py` 中新增 tensor-native 的 node summary → FC2 预测路径。
3. 在 node processing 中让 `policy_value` 优先读取 `FC2` 生成的 node-level macro state。
4. 保持 `FC1` 仍可用于生成 `M`。

---

## 3. 本阶段不做什么

- 不删除 `FC1`
- 不重构 `policy_value`
- 不修改 `FC2` 网络输入维度
- 不修改旧的 DataFrame/非主路径逻辑，除非为兼容必须补小改动
- 不直接改 outer convergence 评价体系

---

## 4. Todo Checklist

### A. 配置与开关

- [x] A1. 在 `HyperParams` 中新增 `fc2_as_main_macro_state: bool = False`
- [x] A2. 在 `Episode._simulate_tensor()` 中把该开关传入 `SimulateTS`

### B. `simulate_ts_parallel.py`

- [x] B1. 新增 batched node summary 构造函数
- [x] B2. 新增 `_predict_macro_fc2_batched(...)`
- [x] B3. 在 `_process_node_batched(...)` 中，当开关打开且 `fc2` 存在时，优先用 `FC2` 生成当前 node 的 `hatcf / lnkf`
- [x] B4. 保证 parent 和 child node 都走同一套 FC2 current-node 逻辑
- [x] B5. 保留 `FC1` 对 `M` 的生成路径

### C. 验证

- [x] C1. `py_compile` 通过
- [x] C2. 最小张量路径验证：
  - `FC2-main=False` 旧逻辑不坏
  - `FC2-main=True` 时 node macro state 能正常生成

---

## 5. 变更日志

### 2026-04-09

- 新建本 Todo 文档。
- 已完成第一刀：
  - 新增 `fc2_as_main_macro_state` 开关；
  - `Episode._simulate_df/_simulate_tensor` 会把该开关传给 `SimulateTS`。
- 已完成第二刀：
  - `SimulateTS` 已接收 `fc2_as_main_macro_state`；
  - `simulate_ts_parallel.py` 已新增 batched FC2 summary/input 生成函数；
  - `_process_node_batched(...)` 在开关打开时会优先用 `FC2` 生成当前 node 的 `hatcf / lnkf`；
  - `FC1` 仍保留在 branch expansion 中用于生成 `M`。
- 已完成最小验证：
  - `py_compile` 已通过；
  - `FC2-main=False` 与 `FC2-main=True` 的张量模拟路径都可正常运行；
  - 在 `FC2-main=True` 时，macro panel 中记录的 `hatcf / lnkf` 已切换为 FC2 生成的 node-level macro state。
- 已补齐真实训练入口：
  - 命令行新增 `--fc2-as-main-macro-state`
  - 已新增 `slurm/run_fc2_main_macro_state_80g.slurm`
- 当前状态：Phase B 的最小切换已经完成，代码已具备真实训练验证条件；下一步是实跑并比较 `FC2-main=False/True`。
