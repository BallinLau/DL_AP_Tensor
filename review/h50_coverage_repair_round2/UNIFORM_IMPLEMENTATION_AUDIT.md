# Phase A — Existing Uniform implementation audit

## 状态与范围

基于 f48b2e2 新建 `exp/h50-coverage-repair-round2-20261021`；未改变已推送的 db27985 分支，也未触及旧 H50 worktree 未提交修改。主目录 bf639c2 存在大量不相关未跟踪文件，保持原样。

只读审查的旧实现：`/private/tmp/DL_AP_Tensor_hstar_h50_20261021`（c725c69 + 未提交 overlay）。其中既有标准Uniform，也有cost01变体。本轮不采用cost01的经济参数变更，不把它与标准Uniform混合。

## A1–A10

| 问题 | 审计结论与证据 | 状态 |
| --- | --- | --- |
| A1 新宏观确实进入Firm输入？ | `data/finite_direct_uniform_firm_macro.py:15–49` clone [N,7] 后向量化写cols5/6，范围[-6,2]/[-2,5]；cols0–4保持 | CONFIRMED_BY_CODE |
| A2 所有inner？ | `run_uniform_macro_frozen_firm.py:46–53,142–169` 先锁同一Uniform Bank，再运行3套P×3/Q×3/BP | CONFIRMED_BY_CODE |
| A3 P/Q/BP都覆盖？ | `ProductionBackend._execute` :285–315读取同一Bank；`_bp` :372–390使用metadata-aware parent adapter | CONFIRMED_BY_CODE |
| A4 Child宏观怎么来？ | `conditional_child_bank` :214–242以新transition_macro_context构造MacroTransitionContext，再调用现有FC1 exogenous builder | CONFIRMED_BY_CODE |
| A5 对应新Parent？ | `firm_transition_context` :195–212要求synthetic context与Parent cols5/6逐值一致；BP使用同一接口 | CONFIRMED_BY_CODE |
| A6 M重算？ | 原builder返回m_raw，重新按既有P/Q clamp生成Child第8列，存raw_m；没有常数替代 | CONFIRMED_BY_CODE |
| A7 旧target复用？ | Parent fingerprint改变；child parent_fingerprint与Macro hash绑定；P/Q在每个block调用生产target builder；BP重建grid teacher | CONFIRMED_BY_CODE |
| A8 Macro真冻结？ | runner guard_macro_optimizers阻止相关optimizer.step，逐block比hash；仅参数冻结不等于全部running state已证明，另核验最终sdf_fc1全state | CODE + ARTIFACT CHECK |
| A9 AFTER checkpoint？ | THU r2/firm_complete.pt存在，2.6MB，SHA d26b9004210eef0be224c79d1f1faef6e59692e48ceba45050e616865ef2d325；已只读下载临时副本 | CONFIRMED_BY_ARTIFACT |
| A10 图可复现？ | 存在checkpoint、后模拟脚本和图源CSV，可重建；旧图eta不匹配不能作公平因果对照。本轮需重画同条件eta0/1 | PARTIAL |

历史summary记录batch2048、3inner、8192/2048 donor rows、标准成本版本、所有P/Q/BP阶段完成，Macro hashes与Outer11一致。旧r1在CSV导出发生重复stage字段错误；r2显式导入已提交前缀，不能把97秒r2耗时当成全部训练耗时。

## 不可照搬的部分

旧H50 overlay无条件把全部Firm Bank改为Uniform，并改默认batch2048。新合同要求显式启用，默认关闭；Hybrid配比显式给定，剩余ON_POLICY/CALCULATED保持50:50并输出实际整数配比。旧独立runner的优化流程应复用，不另造不同公式trainer。

## 训练链路（静态；本轮不执行）

1. direct call：`run_uniform_macro_frozen_firm.run:110–218` → `uniform_macro_bank:15–49`；输出新7D Parent。
2. direct call：`FirmBackend._conditional:103–108` → `conditional_child_bank:214–242`；输出新Conditional Children/raw M。
3. nested call：`ProductionBackend._execute:285–315` → `run_full_nested_equilibrium._p_policy_outer:791–853` / `_q_picard_outer:854–915` / `_bp`。
4. scalar loss与优化sink复用原Adam/LBFGS frozen-policy/Picard及BP distillation；本轮将追踪并测试接口，但不调用其执行路径。

## 方法评分（只衡量旧代码，不是经济正确性）

10项：Implemented7、Partial3、Missing0，Coverage85%。运行参数、完整状态冻结与图形公平性仍须artifact/同条件冻结验证。未改经济语义，未启动训练、未创建批准、未提交Slurm。
