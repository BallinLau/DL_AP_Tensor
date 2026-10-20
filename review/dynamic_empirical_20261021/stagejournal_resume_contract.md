# StageJournal / Resume 待实现契约

当前 StageJournal 不变；当前没有 EMPIRICAL_POOL_COMMIT，也没有动态 pool 恢复功能。

未来需独立 EMPIRICAL_POOL_COMMIT stage：完整 source → 不可变 pool/bank/schedule/CRN files → 校验 hashes → journal发布pointer。训练 checkpoint只存引用，不复制全bank。

下一轮 outer_start 才切换到已提交 Bank；同轮任何 stage 不换 bank。恢复加载已提交引用并校验文件SHA，绝不重新抽已提交版本。未提交更新用已锁定 source/checkpoint/seed 重放，临时文件不是提交证据。

cache 绑定 outer/train_bank/pool/signature/all_model_hashes/economics/CRN，任何变化 STALE_CACHE_REJECTED。不能用 code-mismatch 绕过。上述实现和动态恢复测试均尚未完成。
