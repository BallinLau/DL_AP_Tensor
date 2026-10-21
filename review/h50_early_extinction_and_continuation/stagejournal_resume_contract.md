# StageJournal 与不可变 Bank

H50.snapshot 不再内联 h50_train_bank。它保存 h50_train_bank_reference 与 h50_source_bank_reference：文件路径、文件 SHA、类型递归内容 digest、活动 Outer、bank_version、Pool version。

Bank+schedule 仅在初始化/Outer 切换时写到 immutable_artifacts/train_bank_<content_digest>.pt。相同内容不得覆写；已存在则校验。每个训练阶段 checkpoint 不重复写 Bank。

restore 先用原 ProductionBackend 恢复学习状态，再检查 reference Outer 与 h50_active_outer 相等，并校验文件 SHA、payload digest、Pool、state fingerprint，恢复当前 Train Bank/schedule 和初始 source Bank/schedule。旧内联 H50 snapshot 不隐式迁移。

StageJournal 原子 pointer、committed stage 不重做、RESTART_UNCOMMITTED_BLOCK 契约不改。数据、Firm conditional bank、Pool source 的 manifest/shard SHA 继续校验。固定 Validation 与 CRN 不改。

仍然存在的大对象：模型+optimizer+fresh shock bank；state 中数据路径、函数 probe 快照、metrics/history。此次只去掉完整 Train Bank 的阶段重复序列化，不能声称解决所有 checkpoint/磁盘开销。
