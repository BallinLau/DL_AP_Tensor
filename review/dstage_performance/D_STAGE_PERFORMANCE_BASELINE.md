# H10 D-stage 性能基线（2026-10-22，只读）

## 范围和保护

独立分支 `exp/h10-h100-dstage-performance-20261022`，基线
`b291c645e0b821f2256ce9a5ee30f08e264cdb24`。
正式 H50 Job 711731（g53）与 H10 Job 718682（g55）未修改。
源目录分别为 `h50_formal_source_1e05ecd`、`h10_h100_source_b291c64`，
均位于 THU `/home/fit/zhuyingz/WORK/LiuHao/`，读取时 Git clean。
独立测试不得使用 g53/g55，不修改任何原训练预算、Approval 或输出。

## 已完成 Outer12 的文件级基线

数据根目录：
`/home/fit/zhuyingz/WORK/LiuHao/cachedir/h10_h100_longrun_macro_replay_20261022/outer_012/`。

|阶段|Split|生成文件窗口(s)|Root+生产重放+审计(s，批次末累计)|shards|PT bytes|合法Child rows|
|---|---|---:|---:|---:|---:|---:|
|D0|Train|464|68.438|1280|1633744512|66434|
|D0|Validation|112|17.082|320|457802560|19553|
|D1|Train|482|68.288|1280|1728705280|72585|
|D1|Validation|117|17.083|320|467202432|20352|
|D2|Train|485|68.236|1280|1730940800|72740|
|D2|Validation|121|17.152|320|467105920|20327|

窗口为首个 initial cache mtime 到最终 manifest mtime，文件系统秒级时间，
不是完整 StageJournal stage runtime，也不包含之前的初始数据准备。
Root计时从 calibrate_initial 的 clear_state 前开始，包含 selected evaluation、
生产 process_node 和每行审计构造，不能称为纯 root kernel time。
runtime按初始 bank 顺序、physical_parent_batch=32依次追加 audit；
按连续32行恢复物理batch，每batch只取最后一行累计时间，绝不将4096行直接相加。
每组最后一行还含极少量先前行CPU审计开销。
Train 128 batches、Validation 32 batches。
Root合计约85秒/阶段，占上述窗口约14%，因此缓存Root也不能据此承诺2倍整体加速。

## 旧日志无法恢复的分项

INITIAL_HSTAR_ROOT 与 INITIAL_ROOT_PRODUCTION_REPLAY 没有分开计时；
FIRM_FORWARD、BP_GRID_ACTION、CHILD_TRANSITION、PARENT_EXIT、CHILD_ENTRY、
CHILD_RESOURCE_ACCOUNTING、FC1_FORECAST_FORWARD、GPU_CPU_TRANSFER、
PT_SERIALIZATION、SHA_HASHING、CSV_SIDECAR、CPU_DIAGNOSTICS、
FC1_ELIGIBILITY_AND_SAMPLING 的独立GPU-aware耗时均为 NOT_AVAILABLE。
不能用文件mtime或异步Python调用时长伪造这些分项。
GPU利用率与各阶段峰值需独立bounded profiler补测。

## 初步依赖审查，不是Cache通过证书

market_clearing.evaluate的当期资源候选使用policy_value与固定snapshot；
未直接读取FC1/SDF。t0无进入候选。
calibrate_initial随后调用process_node，实际Grid融资依赖当前SDF/FC1。
因此现有完整processed-state缓存绝对不能跨D复用。
Root-only复用仍需实际FC1/SDF扰动和完整两Child轨迹A/B证明；完成前禁用。
实际D0/D1/D2 Firm hash一致，FC1 hash不同，bank和CRN一致，
但这些被动记录不能替代主动参数扰动测试。

## 当前验证边界

首批CPU测试29 passed，含值前向三种Q参数化和两种价值尺度的bitwise parity。
这不是生产checkpoint/GPU/full-rollout经济一致性证书。
性能提升尚未测量；不得部署到现有训练。
