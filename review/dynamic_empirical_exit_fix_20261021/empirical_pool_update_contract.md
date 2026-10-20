# Pool 更新契约

- 算法 train_dynamic_realized_b_parent_cluster_v1，完整 CONFIG 见 data/finite_direct_dynamic_empirical.py。
- 初始 pool：锁定 Job699894 t0 的 Train 模板；Outer4 原 b 样本保持不变。
- 目标 Outer k+1 Recent：仅 Outer k 的 post-SDF Train pre-decision 模板；Archive：更早 committed 模板及t0。
- 精确2867 Recent、1229 Archive；模板不足有放回，b 使用新确定性分位数重新抽样。完全无Recent硬失败。
- Parent cluster均匀→可用child均匀，重复 parent/branch记录拒绝。不能将sibling当独立Parent。
- 动态来源cluster按continuation Train Parent lineage跨Outer归组（ID不带Outer），模板ID仍带Outer/branch；Archive中同一lineage多个历史截面先共享一个cluster权重，再在可用snapshot-child中选取，避免重复出现的lineage被加倍计数。
- 固定种子 seed_base202610214；outer seed=base+outer*100003；每Parent seed再加parent_id*97。来源不按G/违约/验证R²筛选。
- 条件CDF按资本四分位；少于8样本 fallback总体离散CDF，无平滑、无插值。
- pool记录源outer/model/source manifest SHA，Bank和schedule包含pool fingerprint、outer版本和CRN。
- 无合法模板只记失败，不制造企业；不把事后accounting validity当可达性。
