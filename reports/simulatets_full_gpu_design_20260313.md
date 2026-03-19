# SimulateTS 全 GPU 化设计方案（仅末端转 DataFrame）

## 1. 目标

将 `SimulateTS` 改造成“计算期全程 GPU 张量化”，仅在模拟完成后一次性转为 `df_firm/df_macro`：

1. 模拟阶段不创建 `dict/list/uuid/DataFrame`。
2. 所有状态演化、Policy/Value 前向、SDF 前向、资源核算、宏观聚合都用 `torch` 在 GPU 上完成。
3. 末端统一 `tensor -> numpy -> pandas` 一次落盘。

---

## 2. 当前瓶颈（为何慢）

现有 `SimulateTS` 的主要开销在 CPU/Python 侧：

1. 按 path、按时间、按公司逐行构造 Python 字典。
2. 频繁维护 `ids`、`alive`、`entry` 的动态列表逻辑。
3. 模拟中间过程不断隐式进行 CPU <-> GPU 切换。
4. 早期就转 DataFrame，导致大量对象分配与复制。

这些会抵消 GPU 对神经网络前向和大规模算子的加速收益。

---

## 3. 目标架构（张量优先）

## 3.1 核心思想

采用“固定槽位（slot-based）+ 掩码”而非动态增删公司：

- 设最大公司槽位 `N_max`（如 `group_size + entry_buffer`）。
- 每个 path 在每个时点都维持 `[N_max]` 规模张量。
- `alive_mask` 表示槽位是否有效。
- `entry_mask` 仅标识本期新进入槽位。

这样可以将所有逻辑写成批量张量算子。

## 3.2 状态张量定义

建议统一状态容器（全部在 GPU）：

1. `b, z, eta, i, K, alive, entry`
- 形状：`[P, N]`

2. `x, hatcf, lnkf, M`
- 形状：`[P]`

3. 模拟输出缓存（预分配）
- `firm_buffer`：`[T, B, P, N, F_firm]`
- `macro_buffer`：`[T, B, P, F_macro]`
- 其中 `B = branch_num + 1`（含 parent 记录位）

说明：
- parent 记录位可固定为 `branch_index=0`，children 为 `1..branch_num`。
- 最终转换 DataFrame 时再映射为现有约定的 `branch=-1,0,1,...`。

---

## 4. 关键计算流程（全 GPU）

## 4.1 初始化

1. 一次性采样 `x0`、`z0`、`b0`、`eta0`、`i0`、`K0`。
2. 写入 parent 状态张量。
3. 初始化 `alive_mask=1`（有效槽位）与 `entry_mask=0`。

## 4.2 每期 parent 处理

输入 `state_t`（`[P,N]`）后：

1. 构造 `firm_state_t = stack([b,z,eta,i,x,hatcf,lnkf])`，形状 `[P,N,7]`。
2. reshape 为 `[P*N,7]` 喂给 `policy_value`（单次大 batch）。
3. 得到 `Q/P0/PI/bar_i/bar_z/bp`，再 reshape 回 `[P,N]`。
4. 资源核算：`Y/I/Phi/C` 全张量计算。
5. 宏观聚合：按 `alive_mask` 对公司维求和，得到 `K_t/C_t/LnK_t/Hatc_t`。
6. 将 parent 结果写入 `firm_buffer/macro_buffer`。

## 4.3 分支扩展（children）

对每个分支 `k in [0, branch_num-1]`：

1. 采样 `x_{t+1}^k, z_{t+1}^k, eta_{t+1}^k, i_{t+1}^k`（全 GPU）。
2. 用 **child eta** 更新杠杆（与你当前理论一致）：
\[
b_{t+1}^{k} = \eta_{t+1}^{k} \cdot bp_t + (1-\eta_{t+1}^{k})\cdot b_t.
\]
3. 资本转移：
\[
K_{t+1}^{k}=\bar i_t G K_t + (1-\bar i_t)K_t.
\]
4. 用 `sdf_fc1.forward_step` 计算 `M_{t+1}^k` 与宏观代理 `(hatcf,lnkf)` 演化。
5. 执行 entry/exit（见 4.4），并写入 children 的 buffer。

## 4.4 进入与退出（GPU 掩码版）

退出：
- `alive_next = alive_prev & (bar_z < threshold)`（阈值与现有代码保持一致）。

进入：
1. 为每个 path 生成 `n_potential` 候选（GPU）。
2. 计算进入条件 `entry_value > 0`，得到 `enter_mask_candidate`。
3. 从 `alive_next=0` 的空槽中填充进入者状态（`scatter`/索引写入）。
4. 更新 `alive_next` 与 `entry_mask_next`。

注意：
- 不再新增 Python ID；仅维护 `slot_id`。
- 若最终需要 `ID`，末端可按 `(path, slot)` 规则生成稳定字符串。

---

## 5. 输出设计（仅末端转 DataFrame）

## 5.1 firm buffer 列映射

建议 `F_firm` 对应当前训练依赖列：

1. `path, t, branch, slot`
2. `entry, b, z, ETA, i, x, Hatcf, LnKF, K, M`
3. `Q, P0, PI, Bar_i, Bar_z, P, bp0, bpI, bp`
4. `Y, I, Phi, C`
5. `alive`

## 5.2 macro buffer 列映射

建议包含：

1. `path, t, branch`
2. `K, C, LnK, Hatc, n_firms`
3. `M, x, hatcf, lnkf`

并保证 `build_sdf_pairs_from_macro_ts(...)` 所需列完整（`path/t/branch/x/Hatc/LnK/hatcf/lnkf`）。

## 5.3 一次性转换

模拟结束后：

1. `tensor.detach().cpu().numpy()`
2. 向量化 `reshape` 成二维数组
3. 过滤 `alive==0` 且非调试所需行
4. 单次 `pd.DataFrame(...)`

---

## 6. 与现有工程对接方案

## 6.1 新类建议

新增：`data/simulate_ts_gpu.py`

- `class SimulateTSGPU`，接口尽量对齐现有 `SimulateTS`：
  - `simulate() -> (df_firm, df_macro)`（对上层保持兼容）
  - `simulate_tensor() -> (firm_tensors, macro_tensors)`（供未来无 DataFrame 流水线）

## 6.2 在 Episode 中切换

在 `Episode._simulate_df(...)` 增加开关：

- `simulate_backend='cpu'|'gpu'`（默认 `cpu`）
- `gpu` 时调用 `SimulateTSGPU`。

这样不破坏已有 notebook/脚本。

---

## 7. 开发分阶段（建议）

### Phase 1（最小可用）

1. 先实现无 entry/exit 的全 GPU 版本。
2. 跑通 parent + children + 主分支推进。
3. 与现有 `SimulateTS` 在小样本上做统计对齐（均值/分位）。

### Phase 2（完整行为）

1. 加入 exit（bar_z）。
2. 加入 entry（空槽填充）。
3. 对齐 `df_firm/df_macro` 字段与现有下游。

### Phase 3（性能优化）

1. `torch.compile`（PyTorch 2.x）。
2. 减少 reshape/reindex 次数。
3. 按 H100 显存设定 `P,N,T` 分块策略。

---

## 8. 正确性与一致性检查

每次改动后建议固定做 4 组校验：

1. 维度校验：所有核心张量 shape 不漂移。
2. 经济约束校验：`b_{t+1}` 使用 child eta、`Q(b=0)≈0`、`K/C` 非负。
3. 分布校验：与旧版 CPU `SimulateTS` 对比 `bp/Q/P/Bar_z` 的均值与 p10/p50/p90。
4. 下游兼容校验：`Episode.run_episode(...)` 三模式都可运行。

---

## 9. 风险与注意事项

1. entry/exit 语义如果用掩码重写，最容易发生“计数对但公司错位”。
2. `alive` 与 `bar_z` 的语义要与现有代码保持一致，否则 `P/Q` 曲面会明显偏移。
3. 末端 DataFrame 列名必须严格保持兼容（尤其 `branch/t/path` 和 macro 列）。
4. 大规模 `P*T*N` 下 buffer 可能很大，必要时采用“分块写盘 + 汇总”。

---

## 10. 本设计对应你的需求

你提出的是：

- “SimulateTS 完全在 GPU 上实现”
- “只在最后得到模拟结果后再转成 DataFrame”

本方案完全满足该要求，并且保持了与当前训练框架（Episode / run_multi_episode_job）的兼容迁移路径。
