# Changes

新增正式独立runner、StageJournal、finite-direct运行/观察/conditional bank适配。
新增四类本地测试文件与未来A800脚本、完整review目录。
`data/finite_direct_sampling.py`仅标准化经验来源语义标签。
`analysis/prepare_finite_direct_outer4_review.py`更新已接通runner的review状态，不增加任何训练调用。

没有改 `training/episode.py`、`training/inner_fixed_point.py`、损失函数、SimulateTS生产代码或root solver。
旧runner模式分派不动。最小适配集中在新模块；实际git文件清单与commit见交付 `git_info.txt`。
