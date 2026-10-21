# H50 生产级修正与早期经济灭绝诊断

## 1. 结论先行

**本轮工程修正完成，但不建议启动正式 Outer12–15 H50 训练。**

认证后的 Job702636 Outer11 完整学习状态，与最终 StageJournal committed backend 完全一致。旧 H50 的 Train 初始化确有来源问题：没有激活 source state 中的 Outer11 Dynamic Bank。新版明确沿用经过认证的 Outer11 Bank/schedule，不冒充新独立样本，也不回退 Job699894 t0。

64 个分层真实 Parent 全部找到严格初始根，但计划 6,400 条 Child 行只有 **145 条合法、68 对完整 SDF pairs**。最晚合法 Child 到 t=5；t≥6 没有合法 Child，更没有 Middle/Late 覆盖。H50 接口存在，不等于长期数据已存在。

最早可直接观察到的失稳是 **t=0 Firm hard default**：12,800 家初始企业中 9,266 家触发原退出条件，占 **72.390625%**。64 个 Parent 中 61 个在 t=0 已超过 50% hard default；此时 h 已严格出清、lnK 使用真实资本，下一期 FC1 尚未执行。因此不支持将这些路径的首发事件描述成“未来 FC1 先预测错误，然后才开始退出”。历史训练的 Macro 对 Firm 定价的影响、以及后续预测误差的放大作用，仍不能仅凭这一次冻结路径排除。

Entry 没有被笼统跳过：实际观察到 1,725 个候选，9 个获准，全部发生在 t=1。随后既有无候选，也有价值比较拒绝。零 hard survivors 但账本有定义时，仍执行了下一期 Entry；128 个 Child 在 Entry 后仍灭绝，其中 56 个无候选、72 个有候选但全拒绝。不能将这些都写成未调用 Entry。

本轮没有训练、没有 backward、没有 Slurm、没有批准文件、没有 push/merge；经济方程、退出、进入、融资、资本增长和根接受门槛均未更改。

## 2. 版本、来源与证据边界

- 审查基线：`241968c7425c7ee3efe13303c1ef7ee9ea24a446`。
- **实际生成最终冻结路径的代码 commit：`576e8045da0580b62f950555471bff5c751db383`**，生成时 tracked_dirty=false。
- 最终实现另有 `a0e74d64ee465a9dea170b8181528b50b8e8ae89`：在生产入口硬性核验固定 Validation Bank/schedule 的原始 SHA，附新增测试；它不改变模拟或经济计算。实际诊断的两项 Validation SHA 已独立核验通过，不将该后续保护提交冒充数据生成提交。
- 源训练：Job702636；Slurm 只读查询 COMPLETED / 0:0。
- 源训练代码：`97a018118e917fc5bc5d6b7ed3dd662890560595`。
- 源 stage：`outer_011/outer_commit`；Exit-v2。
- 源完整 checkpoint 重新计算的 SHA：`9ad7610a1851d06d80c8b2d08052a01532923f7a2808e3a006f5538b404aa94d`。
- 最终指针：checkpoint_00907.pt；重新核验 SHA：`a9156eb0a120f144d1d6c3063e98ac17730b4e3b87821c0e0dd57f4311763cf2`。

完整 checkpoint 与最终事务 backend 的 transaction、fresh、state 类型递归哈希完全一致，而非只比网络权重。各项 optimizer、scheduler、AL、RNG 摘要见三个 source audit JSON；live optimizer 的动量/groups/LR/参数顺序比较另见 live_optimizer_restore_audit.json。

SDF 和 FC1 conditional optimizer 分别有 8 和 16 个非空 state entries；源 policy_value 和 fc2 optimizer entries 为 0，照实保留，不伪造动量。源中没有独立 FC1 conditional scheduler；原 Episode schedulers 完整保留。源 AL 的 lambda_var_high 约 9.749e11、rho=2，这只是来源事实，不据此单独判定本次灭绝因果。

CPU 与 CUDA RNG 状态格式不同。一次复跑在将 16-byte CUDA fresh-pair 状态设置到 CPU generator 时失败，错误为 `Expected a CPUGeneratorImplState of size 5056 ... size 16`。修正仅限 CPU 诊断：原 CUDA 字节被认证并保留，不转换、不用于 CPU SDF 随机训练。冻结 rollout 使用原身份键控 simulation CRN，且完全不抽 SDF training pairs。正式 A800 的 restore_fresh 没有更改。不宣称 CPU 是 CUDA 随机训练的逐位 resume。

## 3. 方法与代码对应

| 组件 | 代码证据 | 结果 |
|---|---|---|
| 严格 source 认证 | analysis/h50_source_certification.py:20 certify | Implemented，真实认证通过 |
| Outer11 动态 Train 继承 | analysis/run_initial_hstar_h50_broad_macro.py H50Backend.__init__ | Implemented，不回退 t0 |
| Bank 不可变引用 | h50_source_certification.py:95 / :110 | Implemented，实际序列化测量及篡改拒绝测试 |
| StageJournal 恢复 | H50Backend.snapshot/restore + 原 StageJournal | Implemented，committed stage 不重复 |
| quota/身份统计 | finite_direct_long_horizon_sampling.py:28 / :42 | Implemented，空层不填造 |
| calculated children/M/RHS | 原 conditional_child_bank + real parity JSON | Implemented，8 行真实 P0/PI parity |
| 只在 t0 严格求根 | finite_direct_initial_hstar.py | Implemented，64/64，原门槛 |
| 早期失稳时间线 | run_h50_frozen_extinction_review.py Observer / summarize_h50_frozen_review.py | Implemented，完整记录失败后缀 |
| 长期合法数据覆盖 | coverage_by_time.csv | Missing：本 checkpoint/Bank/CRN 下 Middle/Late 为零 |
| A800 成本实测 | gpu_memory_and_io_budget.md | Partial：只有 CPU/磁盘证据，GPU 未申请 |

按该清单仅衡量实现/证据覆盖为 8 Implemented、1 Partial、1 Missing（85%）；**这不是经济收敛分数，更不是训练批准**。

## 4. 动态 Bank 修正的具体含义

源 active_dynamic_outer=11，实际 Bank 的 SHA 为 `c1bca9220b563729819219105416fafb1599afea75ab442c7297e73379536e81`；state fingerprint 为 `89a51a4b59c26331b601620b8242f7e6fb4dfdb35b9a3f7a957d421a3568f922`。

源 Pool fingerprint 为 `6cb8f3727def6b63ed3ffa37d90d7ace8c2baf78899b9db411186be26daa5f93`；Pool 文件 SHA 为 `df17c7326bdd8baff9e3284f6e959432734c7a3e237062d65b1b73043aea1312`。schedule SHA 为 `3214e19f79e12d211f629b40f7c769b45e6ce19b286e69b0b5f63bc3fe8cad5a`。

Outer12 起始规则是直接沿用认证 Bank，不额外抽样。因此新旧逻辑 Bank 的 state digest 相同，不声称“新一轮独立 Parents”。引用文件包装会有自己的 SHA，与源单独 Bank 文件 SHA 分开记录。后续新 H50 已提交动态来源按既有 Recent/Archive 规则推进；源 Pool 的 archive/recent 用作初始历史 Archive。

固定 Validation SHA 保持 `b7be82344320aecf75bf7c57a1580e28b2cd8282f1426bc1363fe5a8dceddab8`；Validation schedule 也按真实源 SHA 校验。Train 动态状态不得覆盖 Validation。

## 5. 冻结试验设计

先完成 32 Parent，再因 CPU 成本可接受扩大到 64；最终报告以锁定 commit 的 64 Parent 为准。两次只是嵌套的覆盖探针，不把重复 Parent 计为 96 个独立样本。

64 个 Parent：32 Train + 32 Validation，每个 split 为 16 BETA + 16 EMPIRICAL。使用初始 mean b、b P90、K、x、b–K correlation 的 rank-space farthest-point 选取，seed=20261021；保留原始 parent_id、企业 ID、shock schedule。未观察未来结果后筛选；未替换失败路径。

这是确定性覆盖样本，不是人口随机样本。因此 64/64 不能外推为全量 10,240 Parent 根成功率，2.265625% 合法行比例也不是未经加权的人口有效率估计。

所有网络 eval、inference_mode，optimizer.step 与 Tensor.backward 在诊断进程显式禁用。全程模型 hash 不变，FC2 calls=0。t0 Root/production ledger/实际 Grid BP 使用原实现；t>0 无 Root。每个节点 Entry 一次，Child0 promotion 只执行应执行的 parent financing BP，不重做投资、违约或账本。

## 6. 严格根与长期数据

64/64 找到根，各 Parent 有 1 个有效根，原 residual tolerance 不变，独立 production ledger 校验通过。不能由此断言每个未来状态都有根或一定能存活。

| Child time | 合法 Child / 128 | 完整 SDF pairs | 有合法 Child 的初始 lineage |
|---|---:|---:|---:|
| 1 | 98 | 47 | 51 |
| 2 | 25 | 11 | 14 |
| 3 | 16 | 7 | 9 |
| 4 | 4 | 2 | 2 |
| 5 | 2 | 1 | 1 |
| 6–50 | 0 | 0 | 0 |

合计 145/6400 条合法 Child、68 对 pairs。主干 last_valid_time 的中位数为 1，范围 0–5；0 表示只有初始根节点合法。Middle=10–29、Late=30–49 的 parent-time 数据均为零。分母始终是 2×64，没有改成“仍存活样本数”。

原始 Child 失败标签包含 Parent validity 传播，因此必须结合 economic_timing_events.csv 看节点自身原因：实际已尝试节点中 145 合法、129 ZERO_SURVIVING_MASS、128 ECONOMICALLY_EXTINCT_AFTER_ENTRY。不能将传播后的 PARENT_ZERO_SURVIVING_MASS 当作 Child 自身唯一机制。

## 7. 首发机制与一个真实路径

初始 hard default 平均 144.78125/200，范围 57–183。初始全部资本合计 4110.6450；按同一 hard exit 条件保留的旧资本合计 1085.3941，约 73.60% 旧资本将退出。这里不是将 default_mass 当整数公司数；soft gate 加权质量另列。

实际 default 来源：PolicyValue.cal_phats 对原 i-grid 的 max(P0,PI) 取均值形成 Phat；FirmDerivedObjects.survival 使用原 sigmoid(Phat/tau_z)；退出按原 bar_z>=0.5。这次 9266 个 Phat<=0 与 9266 个 hard defaults 一致。单个实际 i 上 max(P0,PI)>0 并不等价于整段 i-grid 平均 Phat>0（本样本仅 10 家出现这种差异）。不能凭旧的某一宏观背景二维图替代实际七维输入核验。

真实 Train Parent **2006294**，主干 branch0：

| t | 决策前公司 | hard default | 退出前 K | hard 退出后旧 K | Entry候选/接受 | 预测 lnK / 实际 lnK |
|---|---:|---:|---:|---:|---:|---|
| 0 | 200 | 137 | 76.5119 | 22.1277 | 未执行初始 Entry | 4.33745 / 4.33745 |
| 1 | 63 | 44 | 22.4514 | 7.5099 | 10 / 0 | 3.01627 / 3.11135 |
| 2 | 19 | 19 | 7.5737 | 0 | 3 / 0 | 1.32115 / 2.02468 |
| 3 | 0 | 0 | 0 | 0 | 1 / 0 | -1.05246 / 未定义 |

t0 严格 h 约 -0.996933，预测与账本误差接近根精度；先退出 137 家。t1 真实资本下降，FC1 lnK 误差约 -0.095，而非资本已崩之前先出现巨大误差；t2 误差扩大到约 -0.704，并同时出现剩余 19 家违约，可能产生进一步反馈。t2 本期账本仍有定义，所以仍尝试 t3 原 Entry；候选 1 家被拒绝后 C/K 无定义，才停止，不捏造新的 FC1 输入。

以探索性阈值 |forecast error|≥ln2、hard default≥50%、K下降≥25% 制作 first_failure_timeline。阈值只用于描述事件先后，不是改动经济 gate，也不是看结果后放松 Root 标准。61 条路径的大规模违约先在 t0 发生；39 条有观察到上述宏观误差阈值，其余在停止前没有可定义的同等事件，不能将缺失解释为一直预测准确。

这些证据支持“当前学习到的 Firm 决策先大量退出 → 资本下降 → 后续 Macro 偏离/反馈”，但**不证明当前 P/Q 是准确经济最优解，也不证明结构模型必然灭绝**。尚未用此批节点的独立 Bellman residual 定量判定已学价值函数误差；不能跳过这一步宣布结构经济不可行。

## 8. Entry 与合法继续资格

已执行 Child 节点：311 ALL_REJECTED、82 NO_CANDIDATES、9 POSITIVE_ENTRY。共 1725 个候选，9 接受（约0.522%），接受全在 t1。entry_candidates.csv 保留每个候选原 P0、PI、PI−P0、i、z、K_birth。没有将 Entry 规则改成与零比较。

原潜在数量由 reference N/K、entry_capital_ratio=.05、size_ratio=1 和身份键控随机舍入决定。企业数量缩小时潜在数也变少；这是原有限企业规则，不增设最小 entrant。

128 个 ECONOMICALLY_EXTINCT_AFTER_ENTRY 都是 entry_attempted=true；其中 56 个 potential=0，72 个 potential>0 但 accepted=0。证明零 hard survivors 并未一律让下一期 Entry 消失。

所有已尝试 Child 的 gross entry K 合计约 3.09259，按该节点 hard survival 保留的 entry K 约2.45813。这是跨时、跨两个 branches 的诊断合计，不与某单个经济体期初 K 当作同一账本直接相减。每节点精确资本桥见 POST_EXIT 的 K_exit_old 与 delta_K_incumbent。

未执行后缀在 macro_path_timeseries / all_child_rows 中保留 NaN 与 failure reason；entry_attempted=false，不能填 accepted_entry_count=0 冒充一次拒绝。

## 9. 训练接口能得到什么，不能得到什么

完整合法 FC1 行：Train83、Validation62；有效 parent-time：Train44、Validation33。SDF complete pairs：Train39、Validation29，始终同 Parent-time 双 branch。空 Middle/Late 不产生样本。

Firm 审查各抽 512 行，包含真实 t>0 的合法 source states：Train可到t5，Validation可到t3。但所有状态仍在 Early，不能称为“宽广长期覆盖”。重复 draw 与 unique source firm-time-branch 分开统计；不把512等同512个独立状态。

Calculated collocation 的真实8行检查为4 on-policy+4 calculated，来源含t0/t1/t3。重建 children、当前 FC1/SDF context/M，再调用原 P0/PI production RHS；最大 target parity 误差 P0约2.98e-8、PI约7.45e-9，原1e-5门槛通过。这是 target plumbing parity，不是 P/Q Bellman residual 已收敛。Q 使用原条件 Child/RHS 接口，没有复用旧标签；本轮没有额外宣称独立真实 Q 数值 parity 实测。

本轮不是训练，因而没有 FC1/SDF 优化结果。训练数据类型仍是有限企业 direct simulation，不是连续统质量块。

## 10. 存储、性能与测试

同一真实学习状态进行实际序列化：旧 inline Bank snapshot 160,239,905 bytes；新 reference snapshot 12,822,117 bytes。每次少写147,417,788 bytes。不可变 Bank+schedule 包装文件只需150,097,309 bytes一次（每个新 Bank 版本另存一次）。拟议449个stage若沿用同一体积，避免约66.19GB重复写入；这只是同体积工程情景，不是未来总磁盘保证。

模型、optimizer、fresh shock bank、history/probe 等仍然存在，不能宣称checkpoint已无大对象。真实64-Parent冻结循环约111秒，含额外诊断 forward/CSV；初始 source 认证与软件初始化不在该计时内，更不能线性外推A800全量训练时间。GPU峰值/吞吐为A800_COST_NOT_MEASURED。

定向 CPU 测试 **141 passed，6 skipped**。6项缺少历史Job701387特定fixture，未计为通过。本轮真实 Job702636 来源/optimizer恢复/冻结前向证据独立存在。bash -n、git diff --check通过。

## 11. 必须回答的15项

1. 241968c实际小样本用的是正确Outer11模型；但生产入口只检查transaction/fresh/state，认证不足。新版补齐严格认证。
2. 旧H50 Train初始化未激活Outer11动态Bank，确有回退locked分布风险；固定Validation试验不能排除该bug。
3. 新Outer12直接复用已认证Outer11实际Train Bank/schedule；不声称独立重抽样。
4. 64个真实分层Parent严格根成功64，失败0；不外推全量。
5. 初始节点都合法；主干最后合法时间0–5，中位1；任意Child最晚t5。
6. 首发普遍是t0大量hard default；原Entry接受少，随规模缩小候选减少；继续机会用尽后经济状态无定义。
7. 61/64大规模违约先在t0，后续FC1误差可能放大，不证明历史Macro训练无关。
8. 三种Entry情况都已区分；停止后的未执行也明确保留。非一律“没调用”。
9. Middle/Late没有合法h-lnK数据。
10. Firm Bank有合法t>0，但只有Early覆盖。
11. Calculated children/M/RHS重建，真实P0/PI target parity通过；不将此说成残差收敛。
12. SDF双Child配对同split/Parent/time，未删一边补另一边。
13. 大Bank改引用；实际序列化证明阶段重复写入减少。
14. 全量成本主要是初始Grid/Root、随存活规模变化的双Child推进、shard扫描/conditional-bank材料化、模型optimizer checkpoint；GPU成本未实测。
15. **目前不值得启动正式H50 Outer12–15**。最小后续证据应是当前真实初始/早期失败节点的Firm Bellman residual与实际collocation支持对照，辨别价值函数误差与经济机制。未获授权不执行新训练，不先改退出/Entry，也不靠更多Epoch或重复早期样本冒充长期覆盖。

## 12. 停止状态

SOURCE_CHECKPOINT_VERIFIED=TRUE；OUTER11_DYNAMIC_BANK_VERIFIED=TRUE；OUTER12_BANK_CONTINUITY=PASS。

MIDDLE_LATE_COVERAGE=NOT_ACHIEVED；FIRM/FC1长期多期覆盖=FAIL；SDF_PAIR_TIME_IDENTITY=PASS。

ECONOMIC_EQUATIONS_CHANGED=FALSE；MODEL_TRAINING_PERFORMED=FALSE；OPTIMIZER_STEPS=0；SLURM_JOB_SUBMITTED=FALSE；APPROVAL_CREATED=FALSE；READY_FOR_FORMAL_TRAINING=FALSE。

CURRENT_PHASE=WAITING_FOR_USER_REVIEW。ZIP只含轻量审查文档、CSV、JSON和图；不含模型checkpoint或全部50期状态tensor。
