# Initial h-star + H50 + Broad Macro：实现与审查报告
日期：2026-10-21（Asia/Shanghai）

## 结论
基于 97a018118e917fc5bc5d6b7ed3dd662890560595，在独立分支 exp/initial-hstar-h50-broad-macro-20261021 实现新的审查门控 Runner。没有训练、Slurm 提交、自动批准或 push。
CPU 接口与少量真实冻结前向已执行；这不是全量数据准备完成，也不是 GPU 性能验收完成。

## 已核验的来源
GitHub 审查分支 exp/dynamic-empirical-exit-fix-20261021 指向 97a0181。
THU sacct：Job702636 COMPLETED，0:0；最终完整学习状态为 Outer11，而不是历史 Job701387 Outer3。
Checkpoint：
/home/fit/zhuyingz/WORK/LiuHao/cachedir/finite_direct_dynamic_outer4_11_v2_97a0181_r1/outer_011/outer_complete.pt
SHA256：9ad7610a1851d06d80c8b2d08052a01532923f7a2808e3a006f5538b404aa94d。
固定 Validation 初始 bank SHA256：
b7be82344320aecf75bf7c57a1580e28b2cd8282f1426bc1363fe5a8dceddab8。

## 改动
1. 初始 h-star 直接复用 MarketClearingController STRICT_GLOBAL，10/20/4、1e-5；t0 不 Entry，不进行二维求根。
2. t0 接受根后独立调用真实 process_node 回放资源账本及 Parent Grid BP。K/C 和 normalized G 都须通过，缓存有完整内容身份。
3. 50 个时间片 streaming shards：Child0 主分支，Child1 条件 sibling。晋升只执行此时才适用的 Parent Grid BP，不重做 Entry、投资/违约提交或账本。
4. Firm 不再过滤 source_time==0。On-policy 与 calculated collocation 分开标注，默认提议各50%；从新 bank 重建 Macro children/M，再走原 Bellman RHS。
5. CPU 分层读取覆盖多个时间、lineage、h/lnK。只把批准的 bounded subset 送进原 P/Q/BP/FC1/SDF 优化逻辑。
6. 新 V5 批准绑定精确源码、最新 checkpoint、实际输入 bank、horizon、求根参数、预算和完整执行顺序。没有生成 approved=true。
7. Train Empirical 仍 50:50，Recent/Archive 2867/1229；从已提交的 H50 source store 按 lineage→time→snapshot 取 realized b CDF；初始 Archive 永久保留。Validation 不变。

## 重要的经济时序发现
原 process_node 的 ZERO_SURVIVING_MASS 同时被当作训练无效和推进停止条件。但 Exit-v2 的生产 Entry reference 保留退出前 K/alive。
因此“当期政策后零存活”不等于“下一期 Entry 已无可能”。
新管线将 continuation eligibility 与 training eligibility 分开：当前 K/C/log 状态有定义时，仍执行原下一期 Exit→Entry；失败训练标签保持失败，不被洗白。
若原 Entry 已执行且仍无企业，则记录 ECONOMICALLY_EXTINCT_AFTER_ENTRY，不调用会抛出灭绝错误的 policy ledger，不造零值标签。
没有添加复活机制、冲击、消费下界或改融资/资本方程。

## 真实 CPU 小样本
使用 Outer11 模型及固定 Validation 前4条真实路径；全局 RNG/模型冻结，无 optimizer step。
4/4 初始根通过，h* 约 -0.955403、-0.807478、-1.070035、-0.989166；normalized G 约1e-6–2e-6。
计划400个 Child observations 全部留存；合法 Child t1=6、t2=2、t3=2，后续为0。
这说明初始一致性不保证长期存活，不能宣传已取得50期有效宏观覆盖。
继续执行合法 Entry 后，部分失败路径确实推进到更晚节点，但没有使本小样本产生 t>=4 的合法训练 Child。
早期原型遇到全空 Child 的生产保护异常，现已在 Entry 执行后显式分类处理；原型输出保留，不作为最终成功证据。
这4条不是随机总体估计，不能声称全8192根可行性或长期覆盖已通过。

## 接入与预算
新入口 analysis/run_initial_hstar_h50_broad_macro.py 默认 plan-only；H50Backend 已连接原 ProductionBackend 的优化器路径。
提议从最新完整学习状态开始新 Outer12–15；此编号/四轮只是待批准计划，不是已获训练批准。
每阶段保存全部可用50期来源，但拟每个 Macro epoch 使用锁定的 Train8192 / Validation2048 个 parent-time 上限，防止隐性50倍 optimizer 扩张。
Firm 为8192/2048行；原 Adam/LBFGS、Q Picard、BP distillation、FC1 loss、Signed AiO/AL 预算保留。
细节见 planned_training_config.json 和各合同。

## 未完成的实测与限制
- A800 峰值显存、吞吐、全量 h-star 可行性及 wallclock：未测。
- 没有正式 FC1/SDF/Firm 优化，不能给训练后 residual 或收敛结论。
- 原始经济分布上的完整 Bellman residual 热图尚无数值，明确 NOT_AVAILABLE；已有接口保留 training-bank / fixed-7D-probe，不能冒充完整内生分布 residual。
- 当前真实小样本没有 Middle/Late 合法数据；sampler 不补造空层。
- 单元测试不能替代新算子的全量 A800 部署验收。
- 代码改变的是数据生成/采样与允许原 Entry 机会的推进合同，不是经济参数。历史一期结果不是新 H50 算子下的收敛证据。

## 状态
CURRENT_PHASE = WAITING_FOR_USER_REVIEW
FORMAL_TRAINING_STARTED = FALSE
OPTIMIZER_STEPS = 0
SLURM_JOB_SUBMITTED = FALSE
APPROVAL_CREATED = FALSE
GPU_PERFORMANCE = NOT_MEASURED
