# Journal 与恢复

初始化：dynamic_initialize提交初始Pool/Bank4引用；baseline在修正operator上重新评价旧Outer3权重。

每轮：outer_start激活已提交Bank → Firm → D0/FC1/D1/FC1/D2 → SDF → 固定Validation → empirical_source → EMPIRICAL_POOL_COMMIT → cross_block → outer_commit。

Pool commit先持久化不可变内容寻址文件，再由原StageJournal发布checkpoint/pointer。Pool阶段不激活下一Bank。仅下一轮outer_start切换，防止D0/D1/D2中途换分布。

Resume校验引用文件SHA、pool/schedule指纹、bank状态hash、版本、CRN。已经committed的阶段不执行；未committed阶段从上一个journal学习状态/source/seed重放。测试使用真实_pool_commit、save/load和StageJournal，证明重试得到相同Bank记录及SHA，而不是只测试计数器。

旧operator/source commit合同不可恢复为新run。不能使用allow_code_mismatch。新训练必须新输出目录与v4批准。
