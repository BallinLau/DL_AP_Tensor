# 动态 Empirical Distribution：经济时序阻塞审查

日期：2026-10-21（Asia/Shanghai）

基线：788fad7efef6c748f48193f30e357e812c4921b3。
本次只增加 CPU 缺陷复现、非优化证据测试及审查文档；没有实现可运行的动态训练机制。

## 1. 方法概览与结论

目标是每个 Outer 以已完成上一轮 Train 的 realized pre-decision b 更新经验 CDF，固定 Validation，并按 Parent 保持 Beta/Empirical 50:50、Recent/Archive 70:30。

当前唯一直接 blocker：真实 finite-direct adapter 在 Parent 决策后，没有应用上一期 default 的退出，就进入 Child。已有 Child capture 还发生在 Child 当期政策评价之后。因此当前快照不能认证为符合要求的已退出、可达、pre-decision 截面。

附件 §3.2 明确要求：发现生产转移漏退出后，不得私自修正经济执行时序，必须报告并等待确认。本次因此停止后续动态接线，不把一个未接入的函数宣称为实现完成，也不生成新批准文件。

## 2. 组件映射

| 组件 | 代码位置 | 状态 |
|---|---|---|
| realized b 的 eta_t 融资规则 | utils/firm_transition.py:apply_refinancing_policy；simulate_ts_parallel.py:2000 | Implemented，最小 forward 验证 |
| 上一期 default 在 Child 前退出 | finite_direct_runtime.py:110–122；simulate_ts_parallel.py:2046 | Missing，已复现 |
| Child 入场、当期决策和捕获 | run_fc1_self_consistent_macro.py:194；finite_direct_runtime.py:122 | Partial，entry 存在但 capture 在决策后 |
| 每轮 Train 版本切换、固定 Validation | ProductionBackend.__init__ / _generate | Missing，目前两者皆固定 |
| 动态 Recent/Archive pool 原子提交 | StageJournal / continuation stages | Missing |
| 动态 Approval v4 和 cache/resume | continuation_state.approval_binding | Missing，仍为 v3 |

结构覆盖（不是验收得分）：Implemented 1、Partial 1、Missing 4，Coverage=(1+0.5)/6=25%。

## 3. 缺陷复现

合成 Parent 有企业 ID 10、11、12：10/11 alive，12 已退出；Parent bar_z 为 [1,0,0]，即 10 当期判违约。调用真实 `_expand_branches_batched()`，两个 Child 的 alive 都仍为 [true,true,false]，企业 10 未退出。

Parent b=[0.6,0.4,0.8]、bp=[0.2,0.9,0.3]、eta=[1,0,1]；Child realized b=[0.2,0.4,0.3]。融资 eta_t 规则本身正确，第二家没有错误使用 bp=0.9。问题不是把 BP head 当成 realized b。

对独立副本调用现有 `_apply_exit_batched()` 后，alive 才变为 [false,true,false]。该副本只用于证明退出函数的效果，未插入真实 Runner，没有授权经济修正。

复现仅使用 1 Parent × 3 slots × 2 branches，CPU、无网络、无 optimizer；它证明调用链存在确定的反例，不证明真实 Job 每个 Parent 都发生了该问题，也不量化旧结果的偏差。

## 4. 真实调用链核对

1. `ProductionBackend.__init__` 保存固定 self.banks/self.schedules；恢复只恢复学习状态与 self.state，没有动态 Train Bank 加载。
2. `load_locked_sampling()` 验证 sampling signature、文件 SHA、固定 schedule、初始银行 fingerprint 以及 train/validation ID 不重叠。
3. `firm_bank` 使用 `_generate(...,'firm_source')`；D0/D1/D2 同样使用 `_generate`，目前确实来自同一个固定初始 Bank。
4. `generate_dataset()` 先 process Parent，再筛有效 Parent，直接 expand，然后 process Child（其中先 entry 再当前政策），最后 capture。没有 `_apply_exit_batched()`。
5. `_process_node_batched()` 写回 bar_i/bar_z/bp；endpoint_alive 是资源分解用局部变量，并没有写回 state['alive']。
6. `_expand_branches_batched()` 实现 eta_t 融资与资本转移，但复制 alive，不以 Parent bar_z 删除违约企业。
7. 常规 `simulate_tensor_parallel()` 虽然有 exit 调用，但位于 Child 当期政策评价、选择 main branch 之后；不能把该调用说成 finite-direct adapter 已经执行的退出。
8. StageJournal snapshot 持久化模型/optimizer/RNG/backend.state，contract 必须相等；当前没有动态池版本或独立 pool commit stage。
9. Approval v3 绑定固定初始采样与初始 checkpoint；不能批准未来动态算法。新 v4 仅给出设计 schema，未接入、未生成批准实例。

## 5. 十二项回答

1. 每轮动态 Pool 是否实现？否，被退出时序 gate 阻塞。
2. realized b 还是 target？原转移中的 b 是 eta_t 门控后的 realized b，已验证；新模板提取未实现。
3. 前一期退出企业是否处理正确？当前 finite-direct Child 调用链否，已给出反例。
4. 新进入企业？原 entry 写入存在（investment_compare b=0、K=K_birth、eta=0、i=entry cost）；新 pre-decision hook 未接入，不能声明完整时序通过。
5. 下一版模板来自哪个阶段？设计锁定 POST_SDF_TRAIN_EMPIRICAL_SOURCE；尚未执行/接入。
6. Outer4 模板？当前 788fad7 仍是 Job699894 t0；本次未从 Job701387 恢复任何动态 source，不能冒充 dynamic。
7. Outer5 如何更新？设计为读取 Outer4 已提交 source/pool/bank；尚未实现。
8. Validation？本次没有更改固定 Bank、ID 或 CRN；动态机制下的验收尚未执行。
9. Resume 同一动态 pool？尚未实现，不能标 PASS；原 StageJournal 未修改。
10. 经济方程？未修改；生产执行代码也完全未修改。
11. 增量成本？设计每 Outer 增加一次 Train-only 两 child source forward；8192 个 Parent，128 batch 即 64 个 parent batch。不能将其当作秒数/显存实测。仅保存模板、CDF、bank SHA，不重复复制大 bank 到每个 checkpoint；实际成本 NOT_MEASURED。
12. Job701387 Outer3 完整状态续训？原 full-state 通道保留；动态计划尚不能启动，必须新 commit、新计划和 v4 批准，不能套旧 v3。

## 6. 建议与停止点

请用户确认是否授权修正 finite-direct Parent-default → Child pre-decision 的退出执行契约。确认时还应锁定退出相对于 entry reference、资本/债务转移和 child 宏观预测的先后，不能仅盲加一行 exit（可能影响 entry reference 和总资本）。之后才完成 read-only hook、rolling pool、同 Outer bank、journal/cache/v4 接线及全部 22 项测试。

无训练、无 GPU、无 Slurm、无服务器访问、无 push/merge、未修改/取消现有作业。使用 methodology-audit 的组件映射与证据标准，明确区分反例测试通过与功能验收通过。
