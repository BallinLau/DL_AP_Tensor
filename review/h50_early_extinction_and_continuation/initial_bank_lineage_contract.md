# Outer12 初始 Bank 的明确来源

旧 241968c 的 H50Backend 恢复 transaction/fresh，但初始化 self.initial_train/self.initial_schedule 直接来自 load_locked_sampling。因此它未激活 source state 的 dynamic_banks['11']，可静默退回 locked t0 Empirical 分布。此前四 Parent CPU 试验用的是固定 Validation，并非证明 Train 来源正确。

新版选择明确规则：REUSE_CERTIFIED_OUTER11_BANK_NOT_INDEPENDENT_RESAMPLE。

Outer12 沿用 Outer11 实际使用、已提交的 Train Bank 与 schedule，不声称是新独立抽样。其 bank_version 仍为 outer_011；活动 H50 Outer 为 12；两者是不同概念。SHA、state fingerprint、schedule、CRN、Pool fingerprint 都按 committed record 认证。源 Pool 是构建 Outer11 Bank 的 Pool，而非把最终统计 JSON 当作一个已抽好的 Bank12。

Outer12 以前不重新抽 Train，不回退 Job699894 t0。后续 H50 Outer 的 Recent 来自上一轮已提交 POST_SDF_TRAIN_EMPIRICAL_SOURCE，Archive 包含源 Outer11 Pool 的 archive/recent 与较早已提交 H50 来源。既有 50:50 Beta/Empirical、70:30 Recent/Archive 不改。

固定 Validation 文件保持原始 SHA b7be82344320aecf75bf7c57a1580e28b2cd8282f1426bc1363fe5a8dceddab8，不能被动态 Train 替换。冻结诊断从 Train/Validation 各取一半，分层后保留原 parent_id、split、shock schedule；不根据未来成功率选样。
