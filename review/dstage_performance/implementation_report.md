# H10 D0/D1/D2 优化阶段报告

## 结论：局部修改与CPU测试完成，GPU验证阻塞，不能部署

基线 `b291c645e0b821f2256ce9a5ee30f08e264cdb24`。
本地实现提交 `d8c1ec0cd54e7c3379fc6719687894eae63e6fcf`。
本报告是后续文档提交，不是已经生成GPU benchmark数据的commit。
分支 `exp/h10-h100-dstage-performance-20261022`。
工作树 `/private/tmp/DL_AP_Tensor_h10_dstage_perf_20261022`。

未修改H50 Job711731、H10 Job718682的源码、参数、输出、Approval、Journal，
没有取消、重排、重启或提交新的正式训练。没有push。

## 改动与接口

1. `MarketClearingConfig.lightweight_hard_candidates=False`：显式可选，
   仅在hard_realized_aggregation且detail=False且具体模型为PolicyValueModel时，
   使用forward_value_components + forward_equity，跳过Q与BP head。
   未知模型与子类、soft聚合、最终detail=True全部保留完整生产forward。
2. `MarketClearingConfig.profile_root_timing=False`：候选评估不再无条件执行
   两次CUDA synchronize。CPU或显式profiling才累计这两项耗时。
   未启用时GPU policy_forward_seconds/accounting_seconds不能解释为完整kernel时间；
   旧record_gpu_memory同步与必要控制流同步尚未优化。
3. `generate_long_horizon_dataset(..., reuse_shard_digest=True)`：新写入不可变PT
   计算一次SHA，同时用于digest sidecar和manifest。已存在shard仍重新读取、
   验证SHA及provenance。False仅供冻结benchmark复现原重复hash开销。
   原子写入、恢复验证、损坏拒绝保留。未改变文件布局与sample selection。
4. 添加冻结128-Parent benchmark脚本，优化器step/backward显式禁止。
   ORIGINAL、轻量候选开关、batch64/128、IO和组合各使用相同实际128Parents。
   这是待运行工具，不是已通过的测试结果；当前版本还未提供完整分项profiler、
   原始全版本的独立source对照与Root-only-cache A/B实现。

## 重要的模式核对

实际H10 simulator设置aggregation_mode=legacy_mixed_replay。
SimulateTS只对hard_realized_reference或continuum_mass_consistent_v1设置hard flag。
因此本次实现的硬决策最小前向对当前H10配置不适用，必须fallback。
不能为达到速度指标擅自把当前经济算子切换成hard模式。
后续若要优化soft路径，需要单独证明其bar_i/bar_z的原有计算完全保留。

## Root-only缓存：未实现、未启用

静态call graph显示当前候选账本无直接FC1/SDF读取，实际D0/D1/D2审计中
Firm、初始Bank与CRN相同，而FC1改变。此证据仍不足以完成强制主动扰动测试。
现有cache保存完整processed state，包含实际Grid融资，禁止跨D复用。
需要完成FC1/SDF参数主动变化后的Root及两Child全路径A/B，才能新增Root-only缓存。
当前状态 `BLOCKED_PENDING_DEPENDENCY_AND_FULL_PATH_AB`，不能声称经济等价通过。

## 已执行验证

命令（本地CPU，无优化器）：

```sh
MPLCONFIGDIR=/private/tmp/mpl-hstar /Users/ballinliu/anaconda3/bin/python -m pytest -q \
  tests/test_dstage_performance.py tests/test_h10_h100_replay.py \
  tests/test_market_clearing_classification.py
```

结果：30 passed in 2.57s。
另通过py_compile及git diff --check。
覆盖三种Q parameterization、两种value scale的P0/PI/Phat bitwise一致，
硬决策一致，未知模型fallback，同模式完整candidate账本与final-detail精确一致，
原H10/Child0 CRN及根分类定向回归。
这些是合成/CPU测试，不证明真实trained-checkpoint的GPU轨迹、root选择或性能。
测试开发中一次失败为fixture实际B=1、h误写B=2；修正测试shape后通过，
未因此调整任何经济参数、root tolerance或生产公式。

## 真实基线

见D_STAGE_PERFORMANCE_BASELINE.md。
D0/D1/D2文件窗口Train+Validation分别576/599/606秒，非完整Journal总耗时。
Root+生产replay+审计约85秒/阶段；其余约86%不能仅从旧日志精确归因。
新写shard一次SHA可确定减少一次完整PT读取，但尚无端到端速度实测。
CPU后处理仍有多次sidecar扫描，未在缺少分项profiling时进行大范围Collector重构。

## GPU与部署阻塞

只读sinfo确认a01有MIXED节点，两个正式任务分别占g53/g55。
未进入它们的GPU。准备上传独立Git bundle时安全审查拒绝了scp，原因是
用户未明确批准这份包含内部代码、测试、报告的具体payload外传到THU。
遵守拒绝：未使用SSH管道或其他途径绕过，未上传、未提交benchmark Slurm。
本地bundle：`/private/tmp/dstage_perf_d8c1ec0.bundle`。
需要用户明确批准该bundle上传到zhuyingz@166.111.143.25的独立目录后，
才能继续独立A800验证。不得修改现有训练。

## 验收状态（严格区分局部测试与整体）

```ini
PERFORMANCE_IMPROVED = NOT_TESTED
ECONOMIC_PARITY = NOT_TESTED
CPU_TARGETED_PARITY = PASS
SAFE_FOR_NEW_TRAINING = FALSE
ROOT_CACHE_ENABLED = FALSE
GPU_BATCH_32_64_128 = NOT_TESTED
OLD_H50_UNMODIFIED = TRUE
RUNNING_H10_UNMODIFIED = TRUE
NEW_FORMAL_TRAINING_SUBMITTED = FALSE
BENCHMARK_JOB_SUBMITTED = FALSE
PUSH_PERFORMED = FALSE
```

未来切换必须完成真实数据/GPU全路径parity与性能验证，锁定新source/config，
从完整committed Outer通过独立certified continuation与新V6批准启动。
不得热替换当前Job源码、复用旧Approval或混用StageJournal。
