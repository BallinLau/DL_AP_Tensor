# Market-Cleared Full Nested：经济接口一致性审查

## 1. 方法概览

本审查只回答正式集成前的两个接口问题：`bar_i` 在资源账户与资本转移中的经济口径是否一致，以及 `LnKF_t / LnK_cal_t / K_total_t` 的时点是否允许在固定资本预测下进行一维当前 `Hatc` 求根。审查对象是 `exp/macro-parent-one-step` 的生产 tensor simulation；尚未接入的 market-clearing 机制不得判定为已实现。

---

## 2. 组件映射

| 方法组件 | 期望功能 | 代码位置 | 状态 |
|---|---|---|---|
| `bar_i` 经济含义 | 连续条件投资对象及违约调整后的有效投资权重 | `models/policy_value.py:509-515` | Implemented |
| 资源账户投资支出 | 使用与政策输出相同的连续 `bar_i_eff` | `data/simulate_ts.py:1155-1190`；`data/simulate_ts_parallel.py:815-835` | Implemented |
| 下一期资本转移 | 使用同一个连续 `bar_i`，而非 0.5 阈值后的 hard indicator | `models/policy_value.py:637-654`；`data/simulate_ts_parallel.py:1592-1597` | Implemented |
| hard investment 的作用域 | 只作为分类/诊断，不控制 production 资本转移 | production transition 无 hard threshold；root diagnostic 中仅计数 | Implemented |
| 维护性投资与折旧 | 支付 `delta*K` 维持未投资资本基数，资本 law 不再显式减一次折旧 | `data/simulate_ts.py:1184-1190`；`data/simulate_ts_parallel.py:1592-1597` | Implemented |
| 当前 `K_total_t` | 在当期政策评估前由当前 firm slots 聚合；进入可使当期 K 随 h 改变 | `data/simulate_ts_parallel.py:838-840` 及 entry ledger | Implemented |
| `LnK_cal_t` | 由当期 production ledger 的 `K_total_t` 计算并写回 | `data/simulate_ts_parallel.py:1142-1167` | Implemented |
| `LnKF_t` | 当前节点给 Firm 的 FC1 belief，而非当期会计真值 | `data/simulate_ts_parallel.py:589-608` | Implemented |
| `cal_t -> forecast_{t+1}` | child FC1 使用 parent 的 realized `hatc_cal/lnk_cal` | `data/simulate_ts_parallel.py:1599-1609` | Implemented |
| 一维 h root 的解释 | 固定外生状态与 `LnKF_t` belief，在允许 entry 响应时求当前资源一致性；不是联合 `(h,K)` 固定点 | 尚未接入 production loop | Partial |

---

## 3. 详细分析

### 3.1 Soft Investment 与资本转移

期望：资源账户中的投资支出与下一期资本增长必须采用同一政策口径。

代码证据：

- `PolicyValueModel` 先构造条件投资权重 `bar_i_cond`，再乘 survival probability 得到 `bar_i_eff`，并把它作为公开的 `bar_i`。
- simulation 对当期出生的 investment-comparison entrants 执行维护/普通投资豁免后，将同一 `bar_i_executed` 送入 `_resource_accounting`。
- 普通投资支出是 `bar_i*K*i`；下一期资本是 `bar_i*g*K + (1-bar_i)*K`。
- production 资本更新没有 `bar_i >= 0.5` 的 hard threshold。所谓 hard investment 只是把连续权重按 0.5 分类的诊断量。

结论：**Implemented / 口径一致。** 当前模型是连续统或条件概率意义的期望政策，而不是逐企业 Bernoulli 实现。资源支出与资本转移都使用同一个连续有效权重，因此不能为了修复负消费而改成 hard investment。

偏差或风险：`bar_z` 同时承担连续 liquidation/resource 权重与 0.5 hard exit 的角色，这属于既有模型近似；本实验不修改它，但报告必须避免把 firm count 的 hard exit 与连续 liquidation 支出混为同一对象。

### 3.2 当前资本时点与一维 h root

期望：明确 `LnKF_t`、`LnK_cal_t` 与 `K_total_t` 的时点，避免把部分条件根称为完整二维固定点。

代码证据：

- 节点开始时，firm slots 中的 `K_t` 已由上一期 `bar_i_{t-1}` 转移得到。
- 当期 ledger 聚合这些 incumbents，并把当期接受进入者的 `K_birth` 纳入 `K_total_t`。
- `LnKF_t` 是 FC1 提供给 Firm policy 的当前节点 belief；`LnK_cal_t` 是完成当期 entry/policy/accounting 后得到的 realized aggregate。
- child forecast 使用 parent 的 `hatc_cal_t, lnk_cal_t`，因此时间对齐是 `cal_t -> f_{t+1}`。

结论：**一维 h root 在明确的条件意义下可实施，但不是完整二维宏观固定点。** 它固定 `LnKF_t` belief 与 pre-decision cross-section，允许 entry acceptance、政策和当前 ledger 随候选 h 重算，并求 `exp(h)K_t(h)-C_t(h)=0`。因为当前 entry 可改变 `K_total_t(h)`，这里甚至不是严格的“固定物理 K”求根；但 `LnKF_t` 仍未作为第二个未知量联合求解。

偏差或风险：正式报告必须称其为 `market_cleared_experimental_v1 conditional current-Hatc equilibrium`，不能宣称已经求解完整 `(Hatc, LnK)` 二维一般均衡。

---

## 4. 缺失组件

1. Production `SimulateTS` 尚无逐节点 root-clearing mode。
2. 尚无将 belief、selected root、firm input 与 calculated macro 同时保留的 schema。
3. 尚无多根跨期 branch tracking 与确定性 selection provenance。
4. 现有 full nested runner 的 Firm bank 仍来自旧 direct-aggregation transition，需 bootstrap root-cleared matched transitions。
5. 两个 child branch 尚未共同通过 root-cleared operator 后再进入 SDF pairing。

---

## 5. 修复建议

1. 新增实验性 `MarketClearingController`，默认关闭，严格保持 legacy behavior。
2. 在 tensor simulation 中把 root search 放在 entry acceptance 与 `_process_node_batched` 之前；选根后以同一 h 重放 entry、policy、ledger 和 transition。
3. 增加 `Hatc_belief/Hatc_cleared/Hatc_firm_input/LnK_belief/LnK_cal` 及 root provenance 字段。
4. 对 parent path batch 使用 GPU vectorized coarse/refinement；两个 child 均执行相同流程。
5. Tiny smoke 必须同时验证 root policy、accounting、transition、CRN、two-child pairing 和 optimizer 更新。

---

## 6. 实现评分

- Coverage: 65%
- Implemented: 9
- Partial: 1
- Missing: 5（列于第 4 节，属于完整实验集成层）

接口审计结论：既有 investment/K 经济口径没有发现需要先修改的确定矛盾；允许在不改变经济方程的前提下继续实现实验性一维 current-Hatc clearing，但必须保留其条件固定点解释范围。

---

## 7. 实现后复核与首次 Slurm 预检失败

本节记录实现后的状态；前六节保留为实现前的审计基线，不能用其 65% 分数描述当前代码。

### 7.1 已接入组件

- `data/market_clearing.py` 已实现 GPU 向量化 coarse-to-fine 搜索、全部 bracket 保留、逐根完整经济节点重放、跳跃交叉分类、确定性多根选择和 FP64 最终残差复核。
- `SimulateTS` / `SimulateTSParallel` 仅在显式 `market_cleared_experimental_v1` 下，对 parent 与两个 child 节点分别清算；默认 `disabled` 的 legacy 路径保持不变。
- Firm transition bank 使用同一 production simulation 的 root-cleared parent 与两个 matched child，不用人为 B-only child 替代。
- full nested runner 按 Firm -> root-cleared simulation -> FC1 -> root-cleared re-simulation -> SDF 顺序运行，并保留 FC2 禁用、hash invariant、同一当前 SDF 的 P/Q parity 以及 legacy-direct 对照。
- stage gate、checkpoint、CSV/JSON 报告、固定 probe surfaces、代表路径图和 GPU profiling 已接入 Tiny -> Small -> Formal 链。

### 7.2 Job 695282 的失败证据与根因

首次服务器提交（source commit `bcd2d808bc7321152174345d52a810f9e3ee0b1b`）在进入训练前即停止，错误为：

```text
ValueError: market_cleared_experimental_v1 requires entry_mode='investment_compare'
```

这证明严格入口 gate 正常工作，也证明链式 runner 当时只打开了 market-clearing mode，却没有把生产经济配置同步为已审计的 Gomes/root-diagnostic 配置。该失败没有产生可解释的训练结果，也没有污染后续 checkpoint；job 和日志应永久保留为失败现场。

### 7.3 配置修复

market-clearing mode 现在显式锁定以下生产口径：

| 项目 | 固定值 |
|---|---|
| entry | `investment_compare / investment_compare_v1` |
| entry capital ratio | `0.05` |
| entry size ratio / max cost / dummy i | `1.0 / 0.5 / 0.0` |
| consumption | `raw`, floor `0.0` |
| accounting | `economic_node_ledger` |
| BP action | production `grid`, 2 child shocks |
| RNG identity | stable entry and stable firm transition identity |

下游 simulation helper 不再硬编码 `bp_action_source="head"`；它读取同一 hyperparameter，并在 grid 模式下构造 production grid resolver。新增回归测试覆盖精确经济配置和 BP source 传播。

### 7.4 当前判定

代码层面缺失的五个集成组件已经补齐，但数值方法仍只应称为 `conditional current-Hatc equilibrium`。完整实验是否通过，只能由新的、隔离输出目录中的 Tiny 数值 gate 决定；在 Tiny 完成前，不得把实现完成等同于经济/数值验证完成。
