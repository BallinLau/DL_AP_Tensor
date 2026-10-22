# H10 + H100 Long-run Macro Replay — 实施与运行证据

日期：2026-10-22（Asia/Shanghai）。这是独立实验，不是原 H50 的恢复或替换。

## 1. 状态边界

原 H50 Job **711731** 最近一次实查为 **RUNNING / g53**，源码
`1e05ecde9f6981c54f89deeadde1a55bc5c48296`。本任务没有对它执行 cancel、
requeue、scontrol 修改、源码编辑、checkpoint 写入或 Bank/Journal 修改。
集群与 a01 分区 PreemptMode 均为 OFF；所有新 Smoke 排除 g53。

新系统已经实现、完成 CPU 回归，并实际执行过两轮 **缩小预算** 的 GPU
流水线 Smoke。不能据此声称四轮正式训练完成。正式执行保留独立 V6
审批闸门，不复用 H50 的 V5，也不自动创建 approved=true 文件。

## 2. 独立源码与目录

- 分支：`exp/h10-h100-longrun-macro-replay-20261022`。
- 本地工作树：`/private/tmp/DL_AP_Tensor_h10_h100_20261022`。
- 已通过第一版端到端 Smoke 的实际源码：`1f46cbca6e9b168b3a90040408b2060e0e61dea7`。
- 包含最终采样边界、CRN 元数据、bin 配额和审查工具的源码：
  `b291c645e0b821f2256ce9a5ee30f08e264cdb24`。
- 最终源码服务器目录：`/home/fit/zhuyingz/WORK/LiuHao/h10_h100_source_b291c64`。
- 计划正式输出：`/home/fit/zhuyingz/WORK/LiuHao/cachedir/h10_h100_longrun_macro_replay_20261022`。
- 计划正式 StageJournal：该目录下 `transactions/latest.json`。
- 待审查材料：`/home/fit/zhuyingz/WORK/LiuHao/h10_h100_final_review_b291c64`。

以上新路径均不在原 H50 源码/输出目录内。不自动 push 或 merge。
文档提交可能晚于实际数据生成提交；请以实际作业使用的源码 SHA 为准。

## 3. 初始完整学习状态

只读输入：

```
/home/fit/zhuyingz/WORK/LiuHao/cachedir/finite_direct_dynamic_outer4_11_v2_97a0181_r1/outer_011/outer_complete.pt
```

THU 实际 SHA-256：
`9ad7610a1851d06d80c8b2d08052a01532923f7a2808e3a006f5538b404aa94d`。

认证沿用 Job702636 原 `transactions/latest.json` 和 `checkpoint_00907.pt`，
核对导出与 committed backend 状态一致，并恢复模型、持久 optimizer、scheduler、
AL、global RNG 和 Fresh Pair CUDA RNG。没有从当前 H50 正在写入的 checkpoint 启动。
CUDA Fresh Pair 字节不伪装成 CPU RNG。Smoke 在 A800 实际经过完整恢复路径。

环境沿用只读 `DL_HL` 二进制，进程、GPU allocation、学习状态、数据、RNG 和 Journal
均独立。没有宣称复制出另一套物理 Conda 环境，也没有修改原环境的软件包。

## 4. 正式计划

独立编号 Outer12–15，共四轮，均为 Firm-first。每轮：

Initial Joint Bank → Firm Source H10 → Hybrid Firm Bank → 原 P/Q 优化 →
D0 H10 → FC1 Inner0 → D1 H10 → FC1 Inner1 → D2 H10 → 原 SDF/Wealth →
Post-SDF Validation H10 → 模型 Commit → H100 → Replay Pool Commit。

原动态 Initial Empirical Pool 仍独立维护，不能把 H100 宏观行直接当作 Initial Parent。
经验模板不足由通过原严格根与账本认证的 Beta 补齐，实际来源比例如实保存。

| 项目 | 正式配置 |
|---|---|
| H10 Initial Train / Validation | 4096 / 1024，200 firms，2 Child，10 次转移 |
| Firm 宏观 collocation | 50% Uniform + 25% ON_POLICY + 25% CALCULATED |
| Uniform 范围 | h[-6,2]，ell[-2,5] |
| Firm batch / inner | 2048 / 3，P Adam200、LBFGS100、Q Picard3、fit300 |
| BP Head | 不专项训练；实际模拟继续 Grid BP |
| FC1 | 2 Inner，各25–40 epochs，原 loss/optimizer/patience |
| FC1 目标 | 4096 unique Parent-time Groups；2048 Replay + 2048 Coverage |
| SDF | 当前 D2 真正同父双 Child，2 chunks × 50 epochs |
| H100 | 200 paths × 200 initial firms × 100 转移，单 Child0 |
| H100 初始设计 | 100 root-consistent + 100 wide-belief，Sobol 联合分层 |
| Burn-in | 排除 parent t0–19；只允许合法 t20→21 至 t99→100 |

上限预算不等于已执行预算。合法容量不足时不重复 Parent-time 伪造样本量。

## 5. 经济与数据契约

没有修改 default、Entry、融资、投资、资本、资源账户、Signed AiO、wealth recursion
或 AL 方程，也没有消费下界、伪根或新冲击。依然使用 Exit-v2。

H100 只执行 Child0，但保持生产双分支 identity namespace。这样才能保持相同 Child0
冲击下的节点 ID、Entry CRN、资本/融资与政策结果一致。CPU 测试覆盖真实生产转移、
Entry 和 policy 后的逐张量一致性，而不是仅比较冲击均值。

Wide-belief 的 ell 同步缩放真实资本；h 只作 belief。所有监督标签来自实际账本。
20 期 burn-in 不等于稳态或均衡证明。每行保留 initial mode、实际 seed/CRN identity、
原 lineage、时间、source Outer 和模型哈希。Replay 明确标记历史政策，不冒充当前政策。

H100 Pool 是独立 transaction：模型已 Commit 后才开始生成；只有完整 `commit.json`
及 CSV SHA 通过验证才能被下一轮读取。空池是合法 `EMPTY_LONGRUN_POOL`；下一轮
Coverage-only。H100 从不进入 SDF 配对。

二维选择器冻结 Train-only bins，保留方向性 overflow，按 bin/time/family 优先级和
lineage/bin 有界配额选择独立 Parent-time。最大边界属于末格，超过最大边界才是
high-overflow。输出 requested/eligible/selected/underfilled、ESS、集中度、范围、
visited-bin coverage 和分 bin FC1 MAE/RMSE。Validation 不决定 bins 或 scaler。

## 6. 实际测试与性能

CPU 定向回归涵盖：原 H50、Exit-v2、StageJournal、H10/H100、单 Child0 parity、
Sparse/Dense 合法行 parity、空池/非空 fixture、burn-in、重复行、非法标签、Validation
隔离、hash 损坏拒绝、V5 拒绝、初始真实资本与 Sobol 设计等。最终测试输出单独留档。

真实 Outer11 checkpoint 的四路径冻结回放：H10 与 H100 执行成功，模型哈希不变；
没有 post20 合法样本，正确提交空池并激活 Coverage-only。不是训练结果，也不是
200-path 代表性结论。H100 实际只写7个时间 shard，仍保留400条 scheduled child
分母和372个 batch 终止后 scheduled parent-times，未填造空后缀标签。

20,480 行 CPU 资格扫描：原约1.997秒，向量化约0.01566秒，FC1/SDF index 完全一致。
约127.5倍只适用于该扫描，不适用于整体训练。D0/D1/D2 计划307,200条 Child，
比原计划3,072,000减少90%；不承诺端到端快90%。详见 `performance.json`。

## 7. GPU Smoke 作业记录

1. **713290**：`sbatch --wrap` 的 `/bin/sh` 不支持 source；未进入 Python/优化。
   修复为显式 Bash Slurm 脚本，保留旧日志。
2. **713291**：完成16-parent 初始 Bank，Firm Bank 时局部 import 遮蔽 module_hash。
   已修复并加入作用域回归测试；未执行 P/Q 优化。
3. **713307**：源码 `1f46cbc`；`COMPLETED 0:0`。两轮小预算完整流水线通过。
   汇总：P/Q 各2个实际优化 block、FC1 8步、SDF 12步；两轮都有模型与 Pool Commit。
   循环部分约81.8秒（不包含全部启动/认证时间）。SDF 日志显示两轮实际 batch 优化。
4. **713318**：最终源码 `b291c64` 的复验，输出目录：
   `/home/fit/zhuyingz/WORK/LiuHao/cachedir/h10_h100_20261022_r4_pipeline_smoke`。
   实查 **COMPLETED 0:0 / summary PASS**；P/Q 各2个优化 block、FC1 8步、
   SDF 12步，两轮都有完整模型与 Pool Commit，循环部分约72.61秒。
5. **713328**：同源码已完成任务的 resume 验证；比较原 Journal 指针，避免把恢复
   日志中重打印的 cached stage result 当作新执行的 optimizer step。
   实查 **COMPLETED 0:0**，恢复前后指针 SHA 均为
   `e7091afb068f3adf288519e7a558a73c3272b645da88435429148402b800000a`，
   没有推进 Journal 或重复优化。

Smoke 使用16 Parents/分割、每轮4条H100、短优化预算，两轮仅用于验证连接和下一轮
Pool 激活。**四轮正式完成数仍为0。** Smoke 不能证明长期均衡，也不能证明非空真实
post20 replay 的统计覆盖。非空 Replay 数据契约由 fixture 测试覆盖，不能混称实证数据。

## 8. 正式审批与恢复

正式 Slurm：`slurm/run_h10_h100_longrun_macro_a800.slurm`。必要环境键、命令和
恢复方式见同目录 README 与服务器待审查 `planned_environment.json`。
准备工具只生成 `approved=false` 的 `approval_required.json`；该文件不能执行训练。
THU 输入认证实查通过，待批准材料已生成并下载到本地。Plan fingerprint：
`10f564ba7f2c6e147c6a4f724a1f1789bb1267d5af7a3b8f23ec460cf5202e05`。

V6 绑定：新 source SHA、完整 plan fingerprint、初始 checkpoint SHA、sampling signature、
实际输入 artifact SHA、原 transaction/pointer SHA、dynamic source artifacts 和 runtime
partition。V5 不被接受，也不能只换 CLI 目录来替换输入。

正式批准后才可创建有效 V6 文件并提交新 Job。恢复使用同一源码、同一批准和输入，
先核验旧 job 已结束，再 `RESUME=1 sbatch ...`。不得重跑 committed 阶段。

## 9. 结论与不能宣称的内容

已经保护旧 H50、建立独立系统并实际跑通小预算训练链；没有自动提交四轮正式任务。
最终源码复验、恢复验证和 THU 输入认证均通过，剩余正式闸门是新的用户 V6 审查批准。
最终只读复核：H50 711731 仍 RUNNING / g53，源码1e05ecd且工作树干净。
不会因为空 Long-run Pool、覆盖窄或 residual 较高擅自改经济标准或停止合法训练；
但不会把 Smoke、已提交作业或中间 loss 当作四轮正式完成或均衡收敛。

最终状态：

```ini
OLD_H50_JOB = RUNNING
OLD_H50_UNMODIFIED = TRUE
NEW_H10_H100_JOB = PREPARED
FINAL_SOURCE_PIPELINE_SMOKE = PASS
RESUME_CHECK = PASS
FORMAL_TRAINING_STARTED = FALSE
FORMAL_SLURM_JOB_SUBMITTED = FALSE
FORMAL_OUTERS_COMPLETED = 0
CURRENT_PHASE = WAITING_FOR_USER_V6_REVIEW
```
