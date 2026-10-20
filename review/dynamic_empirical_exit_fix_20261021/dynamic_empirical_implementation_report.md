# Finite Direct 退出时序修正与动态 Empirical Pool：审查交付

日期：2026-10-21（Asia/Shanghai）。基线 bcb2c5a（其生产基线为 788fad7）。本报告与代码、测试同一次本地提交；实际提交 SHA 见 git log 与交付清单。

## 1. 方法与授权

本轮先修复 Parent 已违约企业仍进入 Child 的 P0 执行缺陷，再实现 Outer4–11 动态 Train 经验 CDF。仅代码、CPU 非优化测试及本地 commit；没有运行 Firm/FC1/SDF 优化，没有 Slurm、服务器访问、push 或 merge，也没有生成 approved=true 文件。

使用 methodology-audit 的组件映射标准：区分调用链已接入、局部前向验证和未经执行的正式训练。不能将单元测试通过写成 A800 训练完成。

## 2. 组件映射

| 组件 | 真实调用位置 | 状态 |
|---|---|---|
| Parent default → Child exit | data/finite_direct_runtime.py:expand_after_parent_exit / generate_dataset | Implemented，生产 CPU 前向验证 |
| 原融资、资本、Entry reference | 复用 _expand_branches_batched、_apply_exit_batched、_apply_entry_batched | Implemented，原函数不改 |
| Child pre-decision realized-b capture | capture_templates，在 Entry 后、process_node 前 | Implemented，无事后成功筛选 |
| 滚动 CDF、Parent cluster、70/30 | data/finite_direct_dynamic_empirical.py | Implemented，确定性 CPU 测试 |
| Outer 真正切换 Train / 固定 Validation | ContinuationBackend._activate_dynamic / outer_start | Implemented，引用恢复测试 |
| Source → Pool 原子提交 | stages / _empirical_source / _pool_commit / StageJournal | Implemented，无训练的 journal 测试 |
| cache/operator/Approval v4 | cache_provenance / finite_direct_dynamic_approval.py | Implemented，旧授权与旧缓存拒绝 |
| 历史解释、统计与曲线 | plan / write_histories / finite_direct_dynamic_diagnostics.py | Implemented，正式结果尚不存在 |

结构覆盖8/8（100%）仅表示本次声明组件有实现和真实接线，不表示正式经济收敛或 GPU 性能已验证。

## 3. P0 修正：保留 reference，不让违约企业进入 Child 政策

实际顺序如下：

1. Parent 完整政策与资源账本完成，bar_z/bar_i/执行 BP 写回。
2. 调用原生产 branch expansion。它锁定转移前 Parent K/alive/ID，作为 Entry reference 与资本桥接历史；按 eta_t 实现 b，按原 bar_i/G 转移资本，生成固定身份冲击和 Child 宏观预测。
3. 在每个 Child 尚未执行政策、bar_z 仍属于 Parent 时，调用原 `_apply_exit_batched()`，移除 Parent 违约企业并把其 mass 置零。
4. 调用原 Child Entry，且仅一次。reference 继续来自保存的 transition_previous_K/alive，不擅自改为 post-exit reference。
5. 若为 post-SDF Train source，立即捕获 pre-decision b/K 截面。
6. Child 当前政策、资源核算执行，禁止再次 Entry。

退出阈值仍是生产 alive & (bar_z<0.5)。融资仍是 b_next=eta_t*bp+(1-eta_t)*b。资本公式仍是 K_next=[1+(G-1)*bar_i]*K。退出企业的死槽位可保留历史数值，但 alive=false、mass=0，不进入 Child 网络或资源加总。原 Parent 账本不被更改，也不重复清算。

为什么不是先在 Parent 上直接改 alive？因为这样会改变原 Entry reference 的规模，且破坏资本起点历史。现在原 reference 保留、Child 经济参与集合修正，资本桥接仍根据真实 ID 将退出列入 K_exit_old。测试覆盖了 pre-reference N=2、Child survivors N=1，reference 与原始展开完全一致，资本桥接残差为零。这不是采用了新的 post-exit Entry 经济设定。

## 4. 动态分布

Outer4 初始 Bank 与 reviewed 固定 Bank 的数值完全一致；初始经验池明确为 Job699894 t0，未宣称使用 Job701387 的动态快照。

Outer k 的 Firm source、D0、D1、D2 使用 Bank k。SDF 两个 chunk 完成并完成固定 Validation 重模拟后，新增无优化的 POST_SDF_TRAIN_EMPIRICAL_SOURCE：仅对 Train 作真实 finite-direct 模拟，在两个 Child 各自 Entry 后、当前政策前提取紧凑模板。既不改 D2，也不训练任何模型。

模板记录 realized b、资本条件 CDF、来源 Parent/Outer/branch/time、模型 hash、默认退出统计。只拒绝空截面或不合法 b/K，不依据 Child 当前消费/灭绝/账本成功选择。当前 Child 将要违约的 entrant 也必须保留在 pre-decision 截面。

下一 Bank 的4096 Empirical Parent 精确分配为 Recent2867、Archive1229。先均匀选 source Parent lineage cluster，再均匀选其可用 snapshot-child；Archive中同一lineage跨Outer共用一个cluster权重，避免多个时期/两个sibling被当成独立Parent加倍计权。Recent 来自本轮（即目标下一轮的上一已执行轮）；Archive 保留更早 committed source 与原始 Train t0。模板不足允许重复使用，但重新抽 b 并报告 reuse；完全没有 Recent 则明确失败，不伪装成静态池。

新 Parent 保留独立生产初始化中已锁定的资本/z/eta/i/ID/CRN，只重新抽 Empirical b 并同步初始化 bp。按资本四分位使用离散经验 CDF；组内少于8点沿用总体 CDF fallback。无 jitter、核平滑或新冲击。Beta 数值不变，Validation Bank 与 CRN 不变。这只保留粗粒度 b-K 关联，不是完整 b-z-K-eta 联合分布重建。

## 5. Atomic commit 与恢复

独立 EMPIRICAL_POOL_COMMIT 位于 source 后、cross-block 前。内容寻址文件保存 pool/bank/schedule，journal 中只存路径、SHA、版本和紧凑抽样诊断，不复制8192×200完整 Bank。下轮 outer_start 才激活下一 Bank，本轮不会提前切换。

若 cross-block 未完成，即使下一池文件已落盘，也不会越过 Outer commit 启动下一轮。Resume 恢复 journal 中 active_outer 对应的 bank，并逐文件校验；已提交池不重新抽样。未提交更新使用相同 source/model/seed，重放得到同样文件路径与 SHA。最后 Outer11 只保存最终池统计，不生成 Bank12。

## 6. 版本与历史解释

新算子：finite_direct_parent_exit_before_child_v2。旧算子：finite_direct_production_ledger_v1。

这是经济转移算子的修正，不是“仅加诊断”。经济方程、阈值与校准不变，但修正后的企业截面、消费、资本、训练标签和有效率可能与旧结果不同。旧 simulation shards、缓存、Journal 合同和 Approval v3 不能直接用于新训练。

Job701387 Outer3 仍可作为完整学习状态初始化（模型、optimizer、AL、RNG 保留），不是修正后模型的直接收敛证据。新 run identity 带 dynamic_exit_v2，需新目录、重新生成数据、重新评价初始 residual。历史文件保存在历史目录，只用于来源审计。

Approval v4 绑定 exact commit、plan、source terminal SHA、完整 initial checkpoint SHA、动态算法配置和种子、初始 pool SHA、initial Train SHA、固定 Validation SHA、50:50/70:30 与新 operator。旧 v3 明确拒绝。prepare 工具可以生成初始 pool，但绝不会生成批准实例。

## 7. 成本与限制

每 Outer 增加一次 Train-only source，8192 Parent、每个两 child，128物理批量时64个 parent batches。网络仍为批量 GPU 实现；source 使用 inference_mode，不增加逐企业网络调用。CDF/分位数/归档在CPU处理，模板只存 b CDF 和资本统计，不存完整历史7D状态。

运行时记录 source_seconds、pool_update_seconds、bank_construction_seconds、artifact_bytes、peak_gpu_memory_gib（进程高水位，不是独占增量）。本轮未用 GPU、未运行正式规模 source，不能提供 A800 秒数/显存或训练性能保证。Archive 在最多8轮内保留历史模板，磁盘会增长；实际字节数由每次更新输出。

新增四幅跨轮 b/coverage 图由实际 Bank 生成，不制造训练结果。Train 指标标明变化分布；固定 Validation 指标仍按固定状态比较。

## 8. 交付回答

动态 pool 已实现并接入真实 Runner；realized b 的 eta_t、退出与 Entry/pre-decision 时序通过 CPU 生产函数测试；Outer4 诚实沿用 t0，Outer5 用 Outer4 post-SDF source，Outer6 的 Recent 用 Outer5 source。Validation 固定；恢复不重抽已提交池。未修改经济方程，但经济转移 operator 明确升级。仍保留 Job701387 完整状态初始化通道。

此版本仅 READY_FOR_USER_REVIEW，未获新正式训练批准。请先审查 economic_timing_audit.md、测试结果和 Approval v4 再决定部署。
