# CPU证据与A800/I-O预算

本轮无GPU申请、无benchmark、无训练。A800_COST_NOT_MEASURED，不能承诺全量小时数或显存峰值。

实际64 Parent初始根：平均1.008秒/Parent CPU（0.795–1.249秒），平均87.875次根operator evaluations；含独立复核为90.875次；平均8.844个root batched network calls。根失败0/64，仅是分层覆盖样本，不推断全量无根率。冻结循环约110.98秒，额外诊断向量forward/CSV包括在内；source认证与初始化不在该计时中。不将CPU数字线性外推A800。

保存3200个batch-time pt shard（诊断物理batch1），payload合计21,238,516 bytes；初始root cache共4,988,608 bytes。文件系统实际分配还含4KB对齐和sidecar/provenance，约41MiB。单shard均值约6.6KB，但多数是失败后缀；存活早期状态shard明显更大。这不是50期全部存活时的预算。

仅作格式容量情景：若按该batch1、相同早期灭绝形态和每Parent平均bytes套到10,240 Parent，一次D约4.20GB（含初始cache），D0/D1/D2约12.6GB。不能当作正式预测：production batch128摊薄header/文件数，后续模型改变可能提高存活并大幅增加状态。正式batch128时一次全量D约80个parent batches ×50 =4000个时间shard，而不是512000个单Parent文件。另有sidecar/root/manifest和firm_source、post-SDF、empirical source等多次generation，必须单独预留空间。

Firm parents 8192×7×FP32仅约0.22MiB原始7D输入；conditional children为8192×J_child×8×FP32，另有shock/context/metadata；必须按chunk_size处理。SDF材料化上限是complete parent-time pairs，不把firm层行当宏观样本。样本不足不重复补造独立pairs。

活跃GPU状态只保留当前bounded parent batch和当前双Child/网格工作集；inference_mode，无50步autograd graph；CPU streaming shards。Root保留batched policy/grid实现，诊断singleton隔离不改变正式GPU批量算法。合法h-invariant缓存保留，模型/算子/Bank/CRN/Root/source哈希改变则失效；不复用已改变模型下的root。

同一Bank不再随449个stage重复写；剩余checkpoint开销见checkpoint_storage_audit.md。数据抽样为lazy多遍扫描，其代价主要是SHA、反序列化、索引；并未声称消除所有重复读取。将来需要A800性能数据时，必须先由用户另行批准。
