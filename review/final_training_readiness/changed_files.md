# 修改清单

1. analysis/run_finite_direct_large_continuation.py：仅批准留档文件名v3→v4。
2. analysis/build_finite_direct_continuation_review.py：旧v3材料生成器对动态计划fail-fast，防止输出过期批准说明。
3. slurm/run_finite_direct_large_continuation_outer4_11_a800.slurm：增加metadata/approval_v4.json注释；参数/资源不变，未执行。
4. README.md：链接最终准备报告、说明v4留档路径。
5. tests/test_final_training_readiness.py：命名、拒绝旧授权、输入绑定、plan-only、预算、真实Slurm argparse及历史生成器保护的非优化回归。
6. review/final_training_readiness/：本次报告、实际本地source审计、真实plan配置、契约、未来命令、测试与验收。

没有修改Approval内容绑定算法、模型/经济/采样算法、StageJournal或历史Job审计文件。
