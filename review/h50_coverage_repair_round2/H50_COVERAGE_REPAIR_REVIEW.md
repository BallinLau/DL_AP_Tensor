# H50 Joint Macro Coverage 第二轮综合审查

## 1. 执行范围与版本

独立分支 `exp/h50-coverage-repair-round2-20261021`，严格从f48b2e2建立；未覆盖旧Uniform未提交改动，未修改已推送的db27985分支。

按阶段本地提交：ea8ad13审计；13bde27显式Uniform/Hybrid与同节点模板证据；6063949容量/overflow/ESS及匹配评价；9a495b5、7976ed2修正诊断工程问题；b0d8d73锁定最终匹配数据生成；54ddf42锁定真实下一Bank冻结验证。最终报告commit另由交付manifest记录。

最终匹配经济数据来自b0d8d73，下一Bank数据来自54ddf42；不能用最终报告commit冒充生成commit。本轮无新训练、无Slurm/GPU、无approved=true、无push/merge/PR。

## 2. 主要结果

1. 复用已有Uniform trainer/context逻辑，显式STANDALONE与HYBRID，默认关闭，fraction不设新正式默认。
2. Stage1保bin/family多样性；Stage2只从已认证不同Parent有限refill。14bin、8192合法fixture可精确区分3584容量与4608cap损失，不把它叫经济不可行；真实8192生成未测试。
3. 原23条合法overflow拆成21条H_IN/ELL_LOW、2条H_HIGH/ELL_LOW，edges固定不变。
4. 原197行来自18lineages，ESS3.585、最大占50.76%；中晚期各仅1条lineage。
5. 两checkpoint真同Macro；1944相同7D状态下两个eta组P/Q residual均改善，但融资组BP regret恶化。
6. 相同4个初始经济体、独立严格Root、同CRN：中期路径数改善，t50仍同为1条。
7. 新真实模板可在下一Outer构造合法Bank；初始2Emp/2Beta，但合法106行/53Pair全部来自Empirical，不能声称有效来源50:50。

## 3. 研究问题

| 问题 | 判断 | 证据与限制 |
| --- | --- | --- |
| Q1 原Firm宽域approximation error？ | CONFIRMED_BY_FROZEN_SAMPLE | 固定网格P/Q residual较大；不是全域数学结论 |
| Q2 Uniform后同状态residual改善？ | CONFIRMED_BY_FROZEN_SAMPLE | P/Q均值改善，BP regret不改善；不能概括全面收敛 |
| Q3 eta0/1都改善？ | CONFIRMED_BY_FROZEN_SAMPLE / qualified | 最终网格两组P/Q改善；局部背景和BP可能反向 |
| Q4 仅因多3inner？ | NOT_TESTED | B额外原分布训练checkpoint缺失；已备等预算接口，未擅自训练 |
| Q5 仅改Firm可改善H50？ | CONFIRMED_BY_FROZEN_SAMPLE | 同4路径中期改善，终点未改善，不外推总体 |
| Q6 direct default还是宏观避险？ | HYPOTHESIS / NOT_TESTED decomposition | 同状态输出确有变化，但Root/内生路径也变，未量化分解 |
| Q7 8192合法Beta足够？ | NOT_TESTED | 有条件容量修复，不等于真实候选一定足够 |
| Q8 underfill哪一类？ | CONFIRMED_BY_TEST / bounded evidence | 14bin fixture为cap；旧32候选实际29经济有效，预算内差3，不证明经济全域不可能 |
| Q9 低K合法数据正确分层？ | CONFIRMED_BY_CODE/TEST/FROZEN_SAMPLE | 保留方向overflow，NaN拒绝，23行精确复算 |
| Q10 Empirical能恢复？ | CONFIRMED_BY_FROZEN_SAMPLE | 真实模板→下一Bank→合法FC1/SDF，不再必然0%；大规模容量未知 |
| Q11 长期数据多少lineage？ | CONFIRMED_BY_FROZEN_SAMPLE | 原Middle/Late各1条，ESS1，详见分段CSV |
| Q12 即使Firm改善，低K FC1仍可能崩溃？ | HYPOTHESIS supported by risk evidence | 宽域91.67%在观测D2轴包络外且M同等比例clipping；未证明单一因果 |

## 4. 未解决的阻塞与风险

原分布额外3inner控制尚未训练，无法识别Uniform与步数效应；冻结Macro宽域可信度差（raw M约1e-8至2.49e5）；大规模独立lineage与合法数据容量未验证。Hybrid fraction也尚未由用户指定，示例中的null会被校验拒绝，不是新的正式默认。新代码可以准备未来实验，但不足以建议直接启动正式Outer。

```ini
READY_FOR_FORMAL_TRAINING = FALSE
TRAINING_PERFORMED = FALSE
SLURM_JOB_SUBMITTED = FALSE
APPROVAL_CREATED = FALSE
GPU_PERFORMANCE = NOT_TESTED
ECONOMIC_CONVERGENCE = NOT_TESTED
CURRENT_PHASE = WAITING_FOR_USER_REVIEW
```

## 5. 工程错误与修复记录

首次regret诊断把bp_pred传给value-only接口，TypeError；改用现有完整Grid接口，额外数值parity=0。ESS首次处理终止suffix缺列失败；仅为无效审计行补NaN schema，未填训练label。第二次匹配评价在inference_mode中创建model，生产Entry读取参数version counter报错；改no_grad，保留Entry原保护。另修正summary根文件路径为真实initial_hstar_audit.csv。失败输出目录保留，不拿部分结果当完成。

## 6. 测试、性能与经济语义

最终CPU测试数量与日志见test_results.txt，含原H50/Exit-v2/Entry与StageJournal回归；不降低原assertion。定向overflow期望值的改变是用更强的方向分类替代旧collapsed语义，不放宽数值门槛。

匹配冻结49.42秒、峰值RSS0.710GiB；真实下一Bank/H50约19.37秒。未运行A800 benchmark。Uniform采样、Child与网络前向batched；reservoir occupancy增量，候选状态shard streaming。所有统计CPU只读reduction，不逐企业NN调用。

经济结构参数、Default/Exit/Entry公式、Bellman、FC1主loss、SDF/AiO/AL、Root算法与1e-5容差未改变。改变的是显式collocation上下文、覆盖选择/overflow分类、模板资格证据和诊断；这些不应笼统写成“什么都没变”。

## 7. 组件一致性评分（非科学就绪）

8项：审计、Uniform接口、容量/overflow、下一Pool、ESS、匹配评价、阶段真实性7项Implemented；控制B接口已备但训练未授权，列Partial1/Missing0，接口审查Coverage93.75%。这是工作项评分，不是模型科学就绪评分；代码支持不等于训练完成，绝不据此给经济收敛评分。

建议下一阶段先由用户审查：是否授权等预算B/C控制及如何评估冻结Macro的off-support风险。未经批准不扩张网络输入、不改objective/校准、不启动新的训练。
