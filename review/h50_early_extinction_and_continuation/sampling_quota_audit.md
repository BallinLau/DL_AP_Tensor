# 多期抽样与身份

完整原始 Child 行保留失败与未执行后缀，不只报告成功观测。FC1 允许合法的单 Child；SDF 仅使用同 split / parent_id / t 的两条不同 branch，不能跨时或跨 Parent 配对。

sample_macro_rows 两遍 lazy 扫描，按实际 available parent-time 分配容量。空 strata 配额为 0；某 strata 不足时将未用容量分给仍有未选独立 parent-time 的 strata；不超过总预算，不复制观测填满。输出实际 unique child rows、parent-times、lineages、宏观 bins、unused quota 和三层 available。

Firm collocation 可以有 replacement draws，但 metadata 分别记录 draws 与 unique source firm-time-branch 数；不把重复抽样当新增独立状态。记录 source_time/source_branch/firm_id/path，计算态替换仅改变宏观输入，不使用旧 RHS。

Calculated states 路径仍是 conditional_child_bank → 当前 FC1/SDF Macro context/M → 原 P0/PI/Q RHS。冻结真实小样本另做 production RHS parity，不运行 optimizer。

512 行/拆分 Firm 抽样只用于本轮 CPU 数据语义审查，不改变拟议正式 8192/2048 预算。实际中后期覆盖看 coverage_by_time.csv，不能从 horizon=50 或预算字段推断。
