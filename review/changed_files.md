# Changes

## Latest limited patch on bdb58fc

- `analysis/run_full_nested_finite_direct_outer4.py`: approval schema v2 content binding; one validation-only post-SDF stage; CPU history recording hooks; fixed-probe label.
- `analysis/finite_direct_outer_metrics.py`: CPU grouped summaries and five report figures from saved ledgers/committed results.
- `analysis/prepare_finite_direct_outer4_review.py`: one plan label for the new post-SDF evaluation stage, no training code.
- `tests/test_finite_direct_approval_metrics_patch.py`: scoped non-optimization regressions.
- `tests/test_finite_direct_outer4_runner.py`: invalid fixture now writes approved=false, never a usable approval.
- `slurm/run_full_nested_finite_direct_outer4_a800.slurm`: schema-v2 comment only; budgets/CLI unchanged.
- `review/approval_schema.json`, `review/approval_metrics_patch_report.md`: new schema and exact metric/approval contracts.
- Updated `review/acceptance.json`, `review/planned_slurm_command.txt`, `review/test_results.txt`, `review/implementation_report.md`, this list.

StageJournal source and all production economic/training equations are unchanged.

## Prior integration baseline

新增正式独立runner、StageJournal、finite-direct运行/观察/conditional bank适配。
新增四类本地测试文件与未来A800脚本、完整review目录。
`data/finite_direct_sampling.py`仅标准化经验来源语义标签。
`analysis/prepare_finite_direct_outer4_review.py`更新已接通runner的review状态，不增加任何训练调用。

没有改 `training/episode.py`、`training/inner_fixed_point.py`、损失函数、SimulateTS生产代码或root solver。
旧runner模式分派不动。最小适配集中在新模块；实际git文件清单与commit见交付 `git_info.txt`。
