# 待实现契约（不是已接入功能）

前置硬 gate：确认并修复 Parent default 在 Child pre-decision 前退出的生产执行缺陷。当前 BLOCKED。

- source：仅上一已完成 Outer 的 Train；推荐 POST_SDF_TRAIN_EMPIRICAL_SOURCE、无梯度/优化器、保存最终模型 hashes。
- 抽样单位 Parent-cluster：先均匀选择 source Parent，再在其可用的两个 child 中按声明权重选 branch；不能两 child 各算独立 Parent。
- 4096 Empirical 精确分配 Recent=2867、Archive=1229，固定整数规则，不按结果改动。
- Recent 必须来自 k-1；Archive 来自更早已 committed Outer，可含原始 t0 Train。完全缺 Recent 必须失败或等待用户明确规则，不冒充动态；不足则确定性重复模板、重新抽 b，记录复用率。
- 只按 pre-decision reachability/finite b/K 判断，不以当前 Child 消费或后续成功挑选。
- 条件 CDF 按资本四分位，保持原 min_count=8 与总体 CDF fallback，无 jitter、平滑和新冲击。不能声称保留完整 b-z-K-eta 联合分布。
- pool 文件不可变，以 source outer/stage/model/checkpoint/seed/config/content SHA 标识。
- 同轮 Firm source、D0/D1/D2 使用同一个 Bank k；pool k+1 提交后不可在本轮切换。
- Outer11 仅保存最终模板统计，不生成 Bank12。

以上未实现/未测试，不产生批准。
