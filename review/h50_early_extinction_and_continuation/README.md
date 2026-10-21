# 审查入口

先读 [完整报告](implementation_report.md)，再看 diagnostic_summary.json、coverage_by_time.csv、first_failure_timeline.csv 与 figures/。

主要结果：64/64 初始严格根；145/6400 合法 Child；68 complete SDF pairs；最晚合法 Child t5；Middle/Late 无覆盖。初始 hard default 9266/12800。未开始训练，当前不建议正式 H50。

实际生成冻结经济路径的 commit：576e8045da0580b62f950555471bff5c751db383。
最终实现增加固定 Validation SHA 保护的 commit：a0e74d64ee465a9dea170b8181528b50b8e8ae89。
随后审查文件归档的 commit 不冒充数据来源。

轻量审查 ZIP 不含模型 checkpoint、source Bank/Pool 或全部50期状态tensor。完整冻结原始输出本地另存：

/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/cachedir/h50_early_extinction_continuation_raw_576e804_20261021

最终可点击审查副本：

/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/cachedir/h50_early_extinction_continuation_review_20261021

economic_timing_events.csv 保留 POST_EXIT / POST_ENTRY_PRE_DECISION / POST_DECISION / NEXT_PARENT_BP_STAGE，不把这些相同节点的阶段行当作独立经济样本。
macro_path_timeseries.csv 是主干 t0–50，包括未执行后缀；all_child_rows.csv 保留两Child全部计划观测。
entry_timeseries.csv 和 entry_candidates.csv 是真实已尝试的事件/候选；未执行后缀见完整主干/Child表，不将其填成零次拒绝。
forecast-vs-actual 图包括账本仍定义但hard survivors为零的已观察节点；训练是否可用必须看economic_state_valid，不能仅按图上有点判定。

sampling_quota_audit.json 明确unused quota；Firm512行是重复抽样draw，不是512个独立状态。测试141通过、6历史fixture跳过；通过不表示正式训练已批准。
