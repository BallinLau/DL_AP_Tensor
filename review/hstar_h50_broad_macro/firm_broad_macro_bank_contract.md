# Firm Broad Macro Bank
父状态7维不变。来源是所有合法t=0..49主干及条件sibling的pre-decision输入。
ON_POLICY保存实际hatcf/lnkf。CALCULATED_MACRO_COLLOCATION复制同一企业(b,z,eta,i,x)，仅替换为该node真实calculated h/lnK。
默认提议50:50，Train8192/Val2048行；这是优化分布，不是实体经济分布。
equal nonempty time-band→lineage→macro-bin→capture→firm，metadata记录原path/firm/time/source kind与true macro context。
两遍流式读取只保留group counts和有限抽样行，不保留全部企业历史到GPU。
固定评价probe保持FIXED_7D_PROBE标识；完整原经济分布上的Bellman residual数值尚未生成，不得混同。
