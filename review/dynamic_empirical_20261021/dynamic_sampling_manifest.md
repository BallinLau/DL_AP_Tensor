# Sampling 设计清单（待实现）

Train8192：4096 固定设计 Beta +4096 动态 Empirical；Validation2048：1024/1024，整个4–11固定。

Beta alpha[1.8,3.2]、beta[4.8,7.8]、原资本初始化、z/eta/i/ID生成不变。只替换 Train Empirical 的 b；不能复制完整历史微观状态。

Outer4 默认明确沿用已锁定 Job699894 t0；本次没有对 Job701387 动态微观快照进行安全恢复，不声称已使用它们。

每个新版本应记录 outer,parent_id,sampling_source,template_version,template_id,sampling_seed,source_cluster,source_branch,pool SHA,bank SHA,CRN SHA。种子规则和算法版本必须在可执行实现及未来 Approval v4 中一起锁定；目前未实现，禁止运行草案。

诊断应含 b mean/std/P10/P50/P90、high-b、capital weighted mean、b-K correlation、source firm count/default、unique parent、reuse/fallback、pool/bank fingerprint；不得从无数据状态制造曲线。
