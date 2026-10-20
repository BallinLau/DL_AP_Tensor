# Checkpoint 与 Resume

`StageJournal`每个block保存完整backend transaction，写入临时文件并原子替换，回读检查后计算SHA。
随后发布stage metadata，最后原子更新唯一latest pointer。旧Outer/旧阶段checkpoint保留。
只有pointer引用且SHA通过的文件可恢复；半写文件、未发布的更高generation不算完成。

合同绑定clean source commit、economic config、finite-direct operator、完整预算plan fingerprint、sampling signature、
initial checkpoint hash与运行分块配置。合同不一致即拒绝，无allow_code_mismatch。
各data stage另绑定全部网络hash、Parent bank、CRN、outer/stage，消费时验证manifest与observation SHA。

恢复内容：所有model、持久optimizer、scheduler、weight scheduler、AL lambdas/rho、phase、continuation counters、
Python/NumPy/Torch/CUDA RNG、SDF fresh eps/bank identity/pair generator、epoch与optimizer计数、当前data版本。
Simulation Grid的Firm target在每个已提交Firm块与online权重相同；恢复后从恢复的online权重确定性hard-refresh，不恢复一个滞后EMA策略。

- P：每个完整policy-outer块；Q：每个Picard块；BP：完整distillation块。
  这些原函数内部的局部optimizer不跨block持久化；未提交block从其起点重做。
- FC1：每完整epoch提交，last权重+同epoch optimizer，best仅诊断；跨条件阶段继续同一FC1 optimizer。
- SDF：每完整50epoch chunk提交；未提交chunk恢复原optimizer、scheduler、AL、fresh RNG后整chunk重做。
- 已提交块跳过；没有声称未提交块具有精确step-level continuation。

`RESTART_UNCOMMITTED_BLOCK`明确记录在checkpoint与interruption证据。
采样/transition身份与CRN不重抽，已完整且SHA匹配的CPU simulation shard可以复用。
Outer0是新run warm start，不是旧root事务resume；新run首次进入optimizer合法初始化，此后按新checkpoint恢复。

本地中断测试对FC1 epoch、SDF chunk、P block注入异常，核对fake模型/optimizer/AL/data/RNG恢复和已提交块不重跑。
另用真实SDFShockBank验证eps、pair RNG与下一次pair draw逐项一致。未进行跨进程A800优化器恢复实测。
