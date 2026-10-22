# H10 D-stage GPU/CPU Optimization Round 3

## 1. 当前结论

完成限定范围的本地修改、静态审查与CPU非优化测试。
未执行GPU benchmark，未修改、取消、重启、热替换或重新提交任何正式训练。
当前直接阻塞为 **BLOCKED_TRANSFER_PERMISSION**，不是训练或经济错误。
本轮没有再次尝试上传bundle，也没有使用SSH管道、Git fetch或其他方式绕过旧拒绝。
不存在D0/D1/D2的本轮CUDA profiler实测数据；不能宣布整体加速或部署安全。

|最终验收项|结果|
|---|---|
|ECONOMIC_PARITY|NOT_TESTED（真实checkpoint两Child全路径）；局部CPU回归PASS|
|BP_GRID_HOTSPOT_CONFIRMED|NOT_TESTED|
|D_STAGE_SPEEDUP_MEASURED|FALSE|
|D_STAGE_SPEEDUP_FACTOR|NOT_TESTED|
|BEST_PARENT_BATCH|NOT_TESTED|
|SAFE_FOR_FUTURE_TRAINING|FALSE|
|OLD_H50_UNMODIFIED|TRUE|
|RUNNING_H10_UNMODIFIED|TRUE|
|FORMAL_TRAINING_SUBMITTED|FALSE|
|BENCHMARK_SUBMITTED|FALSE|
|ROOT_ONLY_CACHE_IMPLEMENTED|FALSE|
|PUSH_PERFORMED|FALSE|

## 2. 源码与隔离

真实生产基线：`b291c645e0b821f2256ce9a5ee30f08e264cdb24`。
Round2基线：`3ff44d2ae4149f2488cb276042ee1ec77d72fc2c`。
本轮独立分支：`exp/h10-h100-dstage-gpu-hotpath-20261022`。
工作树：`/private/tmp/DL_AP_Tensor_h10_dstage_round3_20261022`。
未修改主工作区的已有未跟踪文件，未修改旧性能分支或两套正式源码。
实际提交SHA见本报告旁的`git log -1`；本轮未生成实验训练数据。

2026-10-22只读保护检查：

|项目|H50|H10+H100|
|---|---|---|
|Job|711731|718682|
|squeue/sacct|RUNNING / 0:0|RUNNING / 0:0|
|节点|g53|g55|
|源码|1e05ecde9f6981c54f89deeadde1a55bc5c48296|b291c645e0b821f2256ce9a5ee30f08e264cdb24|
|Git状态|clean|clean|
|观察到的最新阶段|Outer13 FC1 epoch40 committed|Outer14 post_sdf_validation committed|
|最后指针|checkpoint_00177.pt|checkpoint_00336.pt|
|指针SHA|f4895cb57c3959a4dd91047f0e04e7b8d72ff2d27799e8e4b75e605712809e54|1f5852be7088d4a270043587c636946ed59b4930649512073472068dd9aa85a1|

以上是检查瞬间，不是任务最终状态。仍在运行的作业会正常推进。
源码目录位于`/home/fit/zhuyingz/WORK/LiuHao/`：
`h50_formal_source_1e05ecd`、`h10_h100_source_b291c64`。
RUN_ROOT分别为该父目录下`cachedir/h50_joint_coverage_train_outer12_20_20261021_r2`
和`cachedir/h10_h100_longrun_macro_replay_20261022`。

## 3. 可使用的真实性能基线及限制

继承已完成Outer12文件级证据：D0/D1/D2 Train+Validation生成窗口576/599/606秒，
每阶段初始Root+生产重放+审计约85秒。这些不是本轮GPU测量，也不是纯Root kernel时间。
约85%的剩余wall time未被分项归因。不能直接宣布BP Grid为最大热点。
即使完全消除这85秒，其它开销不变的理想上限约为1.16–1.17倍；
Root-only缓存本轮不开发，hard候选轻量路径默认关闭，legacy模式不触发。

## 4. BP Grid调用链与重复计算审查

`_process_node_batched`从online policy_value生成output，并传给`_resolve_bp_action`。
`promote_child`也先生成完整online output，再调用同一Grid resolver。
`GridBPSimulationPolicy.__call__ -> evaluate_actions`忽略context中的model_output，
对`self.target_model(firm_state)`重新前向。存在可研究的重复全模型forward。

但不能直接删除第二次forward：

- online模型与target模型是不同对象，传入state有detach视图；
- H10 `_store`检查两者module hash freshness，但不能替代拟议复用接口逐batch的
  dtype、device、state顺序、版本、eval及输出来源校验；
- 尚未实施或验证安全的precomputed-output凭据，**本轮不启用output复用**；
- 不放宽target freshness，也不把online输出静默冒充target输出。

`evaluate_actions`已经只调用一次`_build_transition`，其children、M、weights同时传给
P0和mixed teacher.compute。**这部分Child/SDF计算已经共享**。
两个branch的Grid目标、fine refinement候选不同；不能不经证明合并。
`_evaluate_grid`还按branch及candidate chunk处理Q与child状态，没有引入全局缓存。
现有parent/candidate chunk hard cap保留，没有调整任何网格点或shock数。

Child observation Grid完全保留。`promote_child`后续实际融资同样保留，
没有用Head、bp=b、固定BP或跳过融资代替。

## 5. 实际实现：可选延迟诊断同步

`GridBPSimulationPolicy(..., defer_diagnostic_sync=False)`新增兼容参数。
默认False保持原API和行为。True时把refinancing-active行数累积为单个GPU long scalar，
直到明确调用`instrumentation()`再复制至host并刷新累计统计；重复读取不会重复计数。

未改变refinancing mask、teacher.compute、P0/mix目标、survival mixing、argmax、
fine refinement、eta精确积分、SDF M或执行action。只移走诊断行数的逐node `.item()`。
每次evaluate_actions理论减少一次显式诊断 `.item()`；实际同步总次数和总wall收益未测量。
teacher中必要all/any控制流、state序列化/日志host复制没有擅自删改。
不使用逐企业Python网络调用。

## 6. Benchmark修正

重写`analysis/benchmark_dstage_performance.py`及support模块：

- 默认只写计划，只有`--execute-frozen`才执行无优化器worker。
- BASELINE_PRODUCTION从真实b291c64 source-root导入模块并核对SHA；
  ROUND2_BASELINE从真实3ff44d2导入，拒绝dirty源码及baseline import污染。
- NO_TIMING_SYNC_ONLY、LIGHTWEIGHT_ROOT_ONLY、SHARD_DIGEST_REUSE_ONLY、
  BP_GRID_LOCAL_ONLY、BATCH64_ONLY、BATCH128_ONLY各只改变一个因子。
- COMBINED未自动执行；没有真实全路径parity证书时直接拒绝该worker。
- 不开发Dynamic transition局部优化来凑variant；此项明确NOT_IMPLEMENTED。
- 同一记录Bank的前128/512/1024Parents；按Bank顺序查Schedule，不随机造Parent。
  重算schedule.fingerprint=stable_json_hash(parents)、bank.schedule_fingerprint、
  bank.fingerprint，并拒绝缺失/重复映射。
- 初始Root cache只读加载配置/identity证据，不复用processed state、BP、标签或Child。
  加载实际hyperparameters并严格比对经济hash、当前模型hash及原Root配置。
- 各variant/repeat独立进程，独立warmup16Parents和独立输出；保留运行顺序，
  不声称已经制造冷文件系统cache。RSS是独立进程含warmup的高水位。
- 正常计时只在实验边界同步；可选`--profile`是独立插桩实验，不能混作无插桩speedup。
- 计数包含Root operator evaluations、BP calls/candidates、SHA bytes、实际轻量调用。
  Firm_forward_calls目前仅统计whole-model forward，不冒充所有value-helper或GEMM调用。
- 当前legacy_mixed_replay的lightweight_forward_actual_calls预期为0，但需运行后确认，
  不能把它的理论收益当作H10实测。
- worker禁止Adam/AdamW/LBFGS step及backward，核对model、AL、global RNG、FC2。

该工具尚未经真实GPUworker验证，CLI/计划测试不等于可用于正式训练。
worker输出economic_rows供核对，但**尚未实现或认证全状态、两Child、SDF完整pair的
跨源自动比较器**，result始终明确full_path_economic_parity=NOT_TESTED。
不会因为标量CSV看起来相同就自动启用Combined。

## 7. Profiler范围与当前缺口

新增bounded profiler，在region内不调用synchronize，延迟解析CUDA events。
记录host调用窗口与CUDA stream span；嵌套窗口不能求和当作独占CPU/kernel时间。
可选torch.profiler trace用于实际kernel及同步事件归因。
已准备节点全调用、Parent/Child BP、Parent Exit、Child0/1 Shock、Entry、FC1、
promotion、PT serialization标记。尚无实际trace。

Child Firm与accounting独立kernel归因、各阶段active Firms统计、CSV/JSON/fsync
细分尚未全部接通，报告不伪造这些数字。
本轮gpu_profile.json是明确的NOT_TESTED占位状态，不是伪profiler输出。

## 8. 动态转移与I/O

stable_firm_identity与stable_economic_identity RNG未修改；Parent Exit、entry reference、
Child entry、b/K与Child观察时序未修改。
promote_child的完整forward尚未删除。没有新增全局cache。

Round2 digest reuse保持原样；worker可独立关闭它以测量旧重复SHA行为。
本轮未改shard schema、原子提交、恢复验证、损坏拒绝、StageJournal。
未合并CSV/JSON或重写Collector：缺乏CPU热点占比证据时不做大范围改造。
重复sidecar扫描仍存在，精确占比NOT_TESTED。

## 9. CPU验证

定向测试：59 passed、1 skipped。包含真实BP teacher单元回归、Grid模拟语义、
H10/Child0 CRN、Root分类、轻量前向、subset身份、单因子计划及延迟诊断计数。
跳过项是要求CUDA硬件的测试，不能声称GPU测试通过。
具体命令和输出见test_results.txt；py_compile与git diff --check通过。
未执行真实optimizer、模拟训练或大型独立调参实验。

## 10. 下一步与安全切换

需要明确批准具体Round3代码/测试/报告payload上传THU独立目录后，才可执行独立A800
frozen benchmark；不得绕过传输限制。届时排除g53/g55且使用独立allocation，
先测128，再测512；profile与wall对照分开，OOM保留失败证据，不改变经济精度。
先补齐全路径parity与profile，再决定是否启用forward复用或更大batch。

当前不能推荐64或128为最佳，也不能推荐迁移正在运行任务。
未来采用优化版必须新source/config provenance与新Approval，通过已完成Outer的
certified continuation独立启动，不可热替换、混用Journal或复用旧V6。

**本轮停止于本地代码与审查交付；不push、不merge、不部署、不提交训练。**
