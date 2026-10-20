# 动态部署契约核验

本次未改7074dca实现。

- operator：finite_direct_parent_exit_before_child_v2。原Exit阈值、eta_t融资、bar_i/G资本、pre-transition Entry reference不变。
- Parent账本后，两个Child在Entry/当期决策前移除上一期default；模板位于Child Entry后、当期决策前，使用realized b。
- Outer4锁定历史t0；Outer k的post-SDF Train source形成Pool k+1，经过EMPIRICAL_POOL_COMMIT；下一轮outer_start激活。
- 同Outer的Firm source/D0/D1/D2不切换Bank。
- Train8192：4096Beta/4096Empirical；Recent2867/Archive1229；Validation2048、1024/1024与CRN固定。
- source严格Train-only，不用Validation，不按Child事后经济成功选模板，不复制历史全状态、不使用bp target替代realized b。
- Parent lineage/sibling权重与CDF/fallback规则沿用7074dca，无新采样设计。
- content-addressed pool/bank/schedule；journal只存不可变引用及SHA，resume核验版本。旧v1缓存或不同bank/pool/signature不可复用。
- 不生成Bank12；Outer11只留最终池统计。

回归继续覆盖生产CPU Exit/Entry/accounting前向、实际Pool提交方法及恢复。正式GPU模拟、数据成功率与经济收敛本轮未测试。
