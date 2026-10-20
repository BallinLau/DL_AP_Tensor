# 文件与修改范围

- data/finite_direct_runtime.py：修正Child退出、Entry只执行一次、pre-decision hook、operator/cache升级。
- data/finite_direct_dynamic_empirical.py：Train模板、滚动池、cluster sampling、条件CDF、内容寻址Bank与恢复。
- analysis/run_finite_direct_large_continuation.py：实际stage接线、激活当前Bank、post-SDF Train source、pool commit、恢复与新批准门。
- analysis/finite_direct_dynamic_approval.py：v4内容绑定、初始pool准备，绝不创建批准实例。
- analysis/finite_direct_dynamic_diagnostics.py：CPU统计、四幅实际Bank分布图。
- analysis/prepare_finite_direct_continuation.py：准备初始pool并标记旧operator历史不是新收敛证据。
- slurm/run_finite_direct_large_continuation_outer4_11_a800.slurm：文档/必需批准提示改为v4，未执行。
- tests/test_dynamic_empirical_exit_fix.py：新非优化测试。
- tests/test_dynamic_empirical_timing_blocker.py、tests/test_finite_direct_large_continuation.py：保留旧反例、更新修正接线与新stage预期。
- 本目录：审查报告、契约、schema、测试结果、acceptance。
- README.md：同步新operator、动态采样、Approval v4与旧结果不可作新收敛证据的迁移说明。

未修改Firm方程、BP Grid经济、Default/Investment/融资/Entry公式、资本增长、资源公式、FC1 loss/结构、SDF/AiO/wealth/AL或冲击过程。
