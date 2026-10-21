# 修改文件
- analysis/run_full_nested_finite_direct_outer4.py：默认legacy行为不变，仅允许显式新adapter绕开旧“所有root均禁用”的全局guard。
- analysis/run_initial_hstar_h50_broad_macro.py：V5门控及H50Backend接入。
- analysis/check_initial_hstar_cpu.py：真实小样本CPU冻结检查，禁止optimizer step。
- analysis/h50_diagnostics.py：CPU流式覆盖、误差与完整H50经济统计。
- data/finite_direct_initial_hstar.py：已有STRICT_GLOBAL的t0适配与缓存。
- data/finite_direct_long_horizon_runtime.py：双Child单主干、节点晋升、shards。
- data/finite_direct_long_horizon_sampling.py：bounded时序/宏观分层抽样。
- data/finite_direct_long_horizon_empirical.py：lazy多期Train CDF来源。
- tests/test_initial_hstar_h50.py：非优化定向测试。
- slurm/run_initial_hstar_h50_broad_macro_a800.slurm：仅模板，未提交。
- review/hstar_h50_broad_macro/：审查合同、计划和真实CPU证据说明。
没有修改Firm方程、FC1 loss、Signed AiO、AL、原资源或资本公式。
