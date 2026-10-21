# 成本预算（未运行A800）
## 规模
Train上限409600 parent-time、819200 child rows；Validation上限102400 parent-time、204800 child rows。
这些是计划上限，不是有效样本量。现实小样本只得到10个合法Child，不能按完整50期存活吞吐外推。

## 优化预算
原Firm P Adam200/LBFGS100、3x3 P blocks；Q3x300、BP1000；FC1两轮25–40epochs；SDF2x50保持。
拟Macro optimizer subset为8192/2048 parent-time；SDF最多64 train batches/epoch、6400 primal batches/Outer；另计AL/Validation。
FC1按initial lineage成批，batch128，不把50期全部rows隐式变成50倍step。

## 显存
Parent physical batch128，初始coarse10，初始N200；policy flat rows为256000。网络forward按candidate_chunk_size262144分块。
Refinement R个active brackets的逻辑规模R*20*N；仍复用原controller的candidate chunks。
Rollout只驻留一批主Parent和两个Children；全部历史和next-state checkpoint写CPU。
额外Firm bank8192*7*4约0.22MiB，仅输入，不含子状态/网络/RHS缓存。
这不是峰值显存估计；原Grid候选/child shock tensor及Entry导致的slot增长仍需实测。
GPU_MEMORY_ESTIMATE=NOT_CERTIFIED_ANALYTICAL_ONLY；PEAK_GPU_MEMORY=NOT_MEASURED。
不能拿历史boundary-aware的49GiB当作本Finite-Direct实现的峰值。

## 时间
每Outer：Firm-source + D0/D1/D2 + post-SDF Validation + post-SDF Train source。
每个stage初始h-star以当时模型hash重新求根；不同Macro/SDF哈希会失效，不盲目复用。
A800_RUNTIME_ESTIMATE=NOT_MEASURED；可核验预算式：
sum(stage initial_roots*measured_root_seconds + attempted_nodes*measured_node_seconds) + 原优化器时间 + IO。
CPU4Parent一次真实小流程约5.7秒，但迅速失效，不能换算全量A800时间。

## 磁盘
主成本是各时间shard的next-state resume快照，以及Firm状态capture。假定N200、每slot约40个FP32等价字段，Train每时间约250MiB，50期约12GiB/stage，仅状态部分。
再加sibling capture、索引、Validation及多个stage，单Outer可能达到数十至上百GiB；Entry增长还会增加。
应在正式部署前测量单shard真实bytes并检查持久盘。不得放/dev/shm或默默删除状态。
