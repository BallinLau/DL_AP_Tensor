# H50 Beta Joint Macro Coverage 审查报告

本轮基于 c725c69bffecfde0673f8c77423abc340d172234。用户已撤回改用 97a0181 为基线的指令。独立分支 `exp/h50-joint-macro-coverage-20261021`，没有覆盖主目录或旧 H50 worktree 的其他修改。

代码生成提交：`f48b2e2a7af02435ded2a43b47e0f30005cd7500`。

## 1. 结论与边界

本轮实现的是原 H50 Parent Bank 的联合 proposal/selection 扩展和统一诊断，不是新的经济模拟器或训练实验。h_cal 不来自任意赋值，而来自原 STRICT_GLOBAL 求根和独立生产账本。Root 容差、BP Grid、Exit-v2、Entry、经济冲击和训练目标没有修改。

本轮仅运行 CPU 非优化测试与冻结探针。没有训练、GPU benchmark、Slurm、正式批准、push 或 merge。初始覆盖扩大不等于中后期数据可用，更不等于 FC1/SDF/完整均衡收敛。

## 2. 基线问题（CONFIRMED_BY_CODE）

原 `finite_direct_sampling.py` 的 Beta 参数主要为 alpha=1.8–3.2、beta=4.8–7.8，K 总量45–85。它有资本权重和 b–K 排列变化，但不是完整七类 b 家族覆盖，也没有按真实 (h_cal,lnK_cal) 稀缺格选择。

原初始50:50是来源数量，不是合法 FC1 行或完整 SDF Pair 的50:50。历史模板的 `post_decision_accounting_valid=NOT_YET_EVALUATED` 不能作为已通过节点账本的证据。本轮不追认该标记。

## 3. 生成与选择

`data/finite_direct_joint_coverage.py`：沿用现有 Bank state schema。保持每 Parent200个有限企业槽位，非连续统企业质量重构。

| family | 分布 |
| --- | --- |
| low | Beta(2,12) |
| medium | Beta(8,8) |
| high | Beta(12,2) |
| broad | Uniform=Beta(1,1) |
| u_shaped | Beta(0.4,0.4) |
| bimodal | low/high各50%混合 |
| upper_tail | low85%/high15%混合 |

K 为源 Parent 实际 K 的0.25/0.5/1/2/4倍，保留相对资本权重；不是将 lnK 标签平移后不改企业资本。K_birth 与物理 K 同步，Entry reference 从生产函数重新计算。z 与 x 使用现有校准 stationary law，eta/i 使用原分布。x 采用固定0.01/0.25/0.5/0.75/0.99分位 tiers，属于显式非人口权重覆盖采样。b 可独立、与 K 正/负排序、与 z 正/负排序；排序不改变边际 z/K/b。

固定8×8网格：h边界[-2.2,0.4]，步长0.325；ell边界[2.3,6]，步长0.4625。边界外单独记overflow，不伪装成区间内覆盖。

默认最多32768候选，batch16，bin cap256；最终 Train target8192。候选按 shard 原子保存、内容 SHA 校验，可恢复；内存保留有界候选引用，最终才物化入选 Bank。所有 h-dependent policy 均由原 operator 重新执行。候选根/账本失败保留原因，不生成伪 h。

初始 Empirical 目标4096：要求 committed Train 历史 Outer、合法有限 b-CDF、同节点账本通过、lineage reuse≤1。Recent/Archive目标70:30，不足在合法容量内分配；不足50%由真实通过 Root/账本/经济资格的 Beta 填补，若仍不足则 underfilled。候选与初始 Bank 的补齐不能被称作训练数据补齐。

FC1 sampler 单独按合法 Parent-time 容量配置来源和时间段，SDF sampler 仅接受完整双 Child。8192预算单位为 Parent-time group，最多产生16384 Child 行，不是8192个独立初始经济体。Firm ON_POLICY/CALCULATED 的50:50是另一维度，不与 Beta/Empirical 混淆。

## 4. 数据与 loss 调用链

下列为静态 direct/nested 调用证据，本轮没有执行 backward。

| 类型 | 调用边/位置 | 数据或输出 |
| --- | --- | --- |
| direct | H50 `_store` → `generate_long_horizon_dataset` | 原 Parent/two-Child ledger shards |
| direct | `finite_direct_long_horizon_sampling.py:42–113` `sample_macro_rows` | 合法 FC1行或完整 SDF Pair，不伪造失败 suffix |
| direct | `run_full_nested_finite_direct_outer4.py:390–405` `_fc1_epoch` → `xy` | fit/dev按 Parent ID分离 |
| direct | `run_fc1_self_consistent_macro.py:319–324` `xy` | X=(x_t,x_t1,Hatc_cal_t,LnK_cal_t)，Y=(Hatc_cal_t1,LnK_cal_t1) |
| nested | `run_fc1_self_consistent_macro.py:353–358` `production_loss` | Parent cols7/8为真实宏观，Child cols8/9为真实标签 |
| nested | `training/episode.py:3928–3938` `_compute_fc1_only_loss` | true-state forward；加权 MSE_h/MSE_K |
| nested | `training/episode.py:3981–3988` | scalar total=recon_weight×recon_loss；recursive辅助在本 setup关闭 |
| sink（未运行） | `run_full_nested_finite_direct_outer4.py:405–407` | zero_grad → loss.backward → fc1.step |

没有改动 FC1 production loss、Firm P0/PI/Q 或 SDF Signed AiO/Wealth/AL。

## 5. H50 survival 和误差定义

`finite_direct_long_horizon_runtime.node_diagnostics` 只做已有张量 reductions，不增加 NN forward，不改状态。记录 predecision alive、hard default、post-exit alive、Entry候选/接受及资本事件。

`analysis/h50_joint_diagnostics.write_dashboard` 区分：可继续推进的 Child0 主干、宏观定义性、至少一个合法 Child、完整两 Child Pair。C/K定义但当前 hard survivors为0时，仍保留原下一期 Entry机会；不让 sibling重新接续已终止主干。t0为Root节点，t≥1为Child到达时点，incoming Parent-time=t−1。t50可推进者标记右删失。

A TRUE_STATE_ONE_STEP 只统计合法 FC1节点；B ON_POLICY_BELIEF_GAP 统计真实轨迹中有定义账本节点上的 belief−calculated。C纯递归 forecast-state诊断在原 setup没有启用，因此 NOT_TESTED。原 `_expand_branches_batched` 已从真实 Parent calculated h/lnK调用 FC1；不能把B误称成C。

逐时间、宏观bin、family给出 MAE/RMSE/bias/绝对误差分位数和有效样本数、原初始分母及无效/终止数。晚期无数据为NaN，不是零误差。误差与下一期 hard default仅作关联，不作因果结论。

D0/D1/D2比较校验固定 Validation Bank内容/IDs/CRN、operator、经济/Root配置、runtime及bins，并保存所有模型hash。缺少实际阶段只显示 NO_ACTUAL_STAGE_DATA。没有训练后快照，因此本轮不声称观测了 FC1改善。

## 6. 测试与冻结证据

77项定向CPU测试通过；`bash -n`、`git diff --check`通过。覆盖原H50/continuation回归、七类分布、rank边际、K/Entry reference尺度、资格/lineage上限、根失败隔离、缓存内容哈希/恢复、hard计数只读、主干不等于Pair、晚期无数据NaN、固定Validation CRN拒绝、endpoint分位数及缺失阶段不伪造。

开发16候选（dirty工作树，仅辅助证据）：16/16 Root+ledger通过，14当前经济有效；7个families有入选，9/64初始macro bins；48合法Child行、24完整Pair，最后合法t6，t10–50无合法数据。CPU总84.48秒，峰值3.398GiB。该数据不能冒称提交后的结果。

最终32候选、已提交源码的冻结结果及精确文件位置见同目录 `frozen_probe_results.md`。冻结driver拦截 optimizer.step/backward，验证模型hash不变和FC2 calls=0。CPU结果不代表A800吞吐、显存或收敛性能。

## 7. 方法一致性审查

| 要求 | 状态 | 解释 |
| --- | --- | --- |
| 真实Root/账本后选择joint bins | Implemented | 不预先赋h标签 |
| bounded family/K/rank覆盖 | Implemented | 采样权重不是population权重 |
| Empirical短缺Beta fallback | Implemented | 初始/FC1/SDF各自核算，不保证存在足量数据 |
| H50 one spine与Exit/Entry | Implemented | 原operator保持，插桩只读 |
| D0/D1/D2统一记录/固定Validation比较 | Implemented in code / not run with trained snapshots | 本轮无optimizer |
| 纯递归forecast-state指标 | Not tested / not added | 既有setup关闭，避免伪命名 |
| A800性能和长期覆盖充分性 | Not tested | CPU小样本不得外推 |
| 后期真实训练改善 | Not tested | 未获训练授权 |

此表不是模型科学就绪评分，也不把代码实现等同于经济改善。

## 8. 研究问题 Q1–Q7

Q1（CONFIRMED_BY_CODE）：原Bank既缺真实(h,lnK)联合选择，也缺现在这七类完整b覆盖。不能说原来已经有足够丰富b、唯一问题只是macro范围。

Q2（CONFIRMED_BY_CODE；因果分解NOT_TESTED）：物理K及其Entry reference同步改变，lnK域因此扩大；若模型具有规模齐次性，纯K缩放可能主要平移lnK而不明显改变h。当前多维proposal共同变化，不能用其总体h范围证明K单独扩大h。

Q3（CONFIRMED_BY_FROZEN_SAMPLE仅指实际范围；机制HYPOTHESIS）：小样本确实产生多个合法h-star与joint bins；b/z/K关联改变与h变化的受控因果归因未完成。不是生成任何指定h都可合法。

Q4（CONFIRMED_BY_FROZEN_SAMPLE + HYPOTHESIS）：初始Root合法不保证动态存活。开发探针t10后为空；Default、Exit、资本/杠杆及Entry反馈决定后期可用性。不能把“没有晚期FC1行”归为初始Root未求好，也不能凭单一关联认定具体网络错误。

Q5（CONFIRMED_BY_CODE，完整预算充分性NOT_TESTED）：fallback在下游按实际合法Parent-time/完整Pair容量重新执行；它不重复无效路径或造数据。因此有能力用合法Beta替代不足Empirical，但若Beta晚期也灭绝，仍必须报告不足。

Q6（CONFIRMED_BY_CODE；实际D0→D2改善NOT_TESTED）：固定Validation/CRN及模型hash检查已接入；本轮没有训练后快照，不能提供改善结论。Firm/SDF也变化时只能作系统比较。

Q7（HYPOTHESIS）：若严格匹配(x,h,lnK)与共同冲击后，不同横截面仍稳定产生不同下一期真值，则支持FC1遗漏分布信息。当前小样本family/bin关联不是该识别实验，x/z/K形状与来源等混杂尚未排除；未扩大网络输入。

## 9. 审查停止状态

仅本地提交。没有approved=true、没有训练、没有Slurm、没有push/merge。当前风险是中后期合法样本不足，而非“代码测试通过所以可直接认定长期覆盖已解决”。任何后续objective、经济参数或网络输入修改都需独立proposal及用户授权。
