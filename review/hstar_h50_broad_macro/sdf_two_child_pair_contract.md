# SDF TRUE_ONLY配对
key=(split,initial_parent_id,child_time)，parent_time=child_time-1。
沿用production_sdf_tensors按key配branch0/1；不能跨时间配，也不能用单child。
Signed AiO、wealth recursion、normalized residual、AL dual/rho与M clamp均不修改。
拟每epoch最多8192完整Train pairs；使用独立的complete_pairs_only索引，不从FC1抽样子集倒推SDF数据，避免合法完整pairs被FC1单child抽样挤掉。D2保存独立sdf_observations.csv及内容SHA。2chunks*50epochs，physical pair batch128，至多6400 primal optimizer batches/Outer，另有原AL/validation前向。
这只是待批准预算；实际pair数、optimizer步数必须以执行日志为准。
来源store保存全部合法时间；bounded索引不会永远只取t0，空Middle/Late不能宣称使用了。
