# H50 Joint Coverage：审查契约

本轮基线 c725c69bffecfde0673f8c77423abc340d172234。用户已撤回切换至 97a0181 的指令。

## 授权边界

仅代码、CPU 非优化测试、最多 64 候选的冻结探针及本地提交。不得训练、Slurm、GPU、生成 approved=true、push 或 merge。

## 配置和批准

`run_initial_hstar_h50_broad_macro.py` 默认 plan-only。未来生产执行仍需要显式 `--execute-approved` 和用户批准文件。本次不创建该文件。

V5 绑定除原 source/checkpoint/sampling/transaction SHA 外，新增 `joint_coverage_fingerprint`，完整 coverage 配置也参与 `plan_fingerprint`。因此旧 H50 批准不能授权本轮新计划。绝对部署路径不参与经验池身份；绑定源 Bank 内容 SHA、Pool manifest SHA 和 Outer。

`--coverage-config FILE.json` 可配置 families 参数、K tiers、rank modes、bins、bounded candidate budget；未提供时采用代码中锁定的 JointCoverageConfig。Slurm 模板的 COVERAGE_CONFIG_JSON 仅透传相同参数，不代表已经提交。

## 不改变的经济规则

原 STRICT_GLOBAL t0 h-star、残差容差、Grid BP、资本/资源账本、Parent Exit-v2、Child Entry、AR(1) 校准、Firm Bellman、FC1 production loss、SDF/AiO/AL 未修改。t>=1 不添加 root search。

## 覆盖与资格

Beta 七类分布；K tiers 为现有初始资本的 0.25/0.5/1/2/4 倍，保持相对企业规模及 Entry reference 的尺度一致。b 的 rank permutation 不改变 z/K 的边际分布。h_cal 不是 proposal 标签，只能来自原严格求根及独立生产账本。

初始目标 Empirical 50%；仅使用已提交、Train、历史 Outer、有限合法分布、同节点账本已认证且不超过 lineage reuse cap 的模板。未知账本不升级为 True。不足由真实合格 Beta 填补；仍不足则明确 underfilled。FC1 按合法 Parent-time、SDF 按完整 Parent-time 两 Child 独立计算容量；不得用初始行数代替下游容量。

## 诊断

所有数据阶段统一由 `_store` 写 `diagnostics/`，避免重复扫描完整 shard。主干固定 Child0；Sibling 不复活主干。连续可推进路径、经济训练资格、单 Child 行、完整 SDF Pair 分开记录。t50 是右删失，不是灭绝。

真实生产 FC1 forward 使用 Parent 的 calculated h/lnK。因此 TRUE_STATE_ONE_STEP 与 ON_POLICY_BELIEF_GAP 的样本资格不同，但后者不能冒称额外的纯递归 forecast-state 实验。该额外实验标记 NOT_TESTED。

D0/D1/D2 比较要求相同固定 Validation Bank/IDs/CRN/经济配置/根配置/runtime/bins；Firm 或 SDF 变化时不能把差异单独归因 FC1。没有真实阶段数据只输出缺失标记，不合成训练后曲线。

## 测试与证据身份

定向 CPU 测试包含真实 ledger reductions、原 H50/continuation 回归、配置拒绝、来源资格、根失败隔离、bounded selection、resume 内容哈希及无数据指标。冻结探针强制拦截 Adam/AdamW/LBFGS.step 和 Tensor.backward，保存模型前后 hash、FC2 calls、生成 commit、代码 SHA、dirty 状态。

已有开发探针不冒称最终 commit 的数据。最终 review report 应另列代码生成 commit 与报告 commit。
