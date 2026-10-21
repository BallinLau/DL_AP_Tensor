# Resume
沿用原StageJournal原子指针和RESTART_UNCOMMITTED_BLOCK，不重新发明Journal。
新contract绑定V5精确源码和全部输入身份、H50、root config、执行顺序。
snapshot恢复模型/optimizers/schedulers/AL/global RNG/fresh-pair RNG、stage进度、active Train bank、pool sources及样本元数据。
H50Backend.restore核验已经提交的数据manifest、observations、shard及Firm bank哈希。
root缓存另有原state hash、模型hash、经济/CRN/root config/adapter源码身份和完整payload digest。
已存在合法shard直接恢复next-state，不重抽冲击、不重做Entry或当前policy。
尚未提交部分可以在锁定CRN下重放；不会覆盖不匹配provenance。
StageJournal自身并未被重写。
