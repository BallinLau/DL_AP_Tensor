# Phase C/D — Capacity and directional overflow

原cap256、8192目标若全Beta至少需32个充分填充bins；14bin的小样本不能证明总体只有14bin，但说明风险。纯CPU元数据测试：8192合法fixture仅占14bins时，Stage1严格保留3584，缺口4608；不是Root/经济不可行。

新Stage1保持原bin/family多样性和cap；occupancy_by_bin/occupancy_by_bin_family/total_count增量维护，不再每个candidate扫描全部buckets重算Counter。

新Stage2从额外有界refill reservoir（至多2×target引用）选择已认证、未入选的不同Parent，优先稀缺bin；显式selection_stage=CAPACITY_REFILL。没有新抽样、放松经济资格或无限重复。单bin20个合法候选、target16、cap1的fixture中Stage1=1，refill15，最终16个不同Parent；resume结果一致。

输出capacity_audit区分eligible/retained per-bin、cap rejection、root failure、economic invalid、原cap reachable capacity、实际Bank/target、capacity/economic shortfall、budget exhaustion。经济资格不足仍UNDERFILLED，未做8192真实经济生成。

原8×8 edges不变。每轴使用-1表示LOW、n_bins表示HIGH，另一轴保留具体bin；通过bin_direction输出H_LOW/H_HIGH/H_IN和ELL_LOW/ELL_HIGH/ELL_IN组合。NaN不进入bin，合法极端值保留为overflow；热图不把overflow挤进内部或复制到多bins。

本轮未把共同支持域自动扩大。低ell频繁出现提示下一次审查可选择锁定更宽edges，但那需要新的config/provenance，所有Validation和D0/D1/D2一致。数值标签和Root容差未变。

风险：refill降低相对于理想均匀网格的覆盖平衡，必须看actual occupancy和lineage ESS，而不能将填满Bank等同丰富覆盖。已锁新的Bank版本V2，旧proposal cache不得复用。
