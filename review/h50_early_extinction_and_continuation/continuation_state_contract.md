# Outer11 → H50 学习状态继承

本轮不执行训练。生产入口仍默认 plan-only。

认证入口：`analysis/h50_source_certification.py:certify`。固定 Job702636、97a0181、Outer11 committed stage、Exit-v2，以及重新计算的完整 checkpoint SHA。最终指针引用的事务文件也校验 SHA；其 backend 必须与 outer_complete.pt 的 transaction/fresh/state 类型递归哈希完全一致。不同则 STOP_AND_REPORT。

完整保留：Firm/FC1/SDF/Wealth/FC2 参数、所有已序列化 optimizer groups/state、Episode scheduler/weight scheduler、AL、全局 RNG、fresh shock bank、fresh-pair RNG。已提交 Outer4–11 必须全部存在。

原始 policy_value、fc2 optimizer state entries 为 0，不能称为已恢复不存在的 Adam 动量；Firm P/Q 使用原来的阶段局部求解器。SDF 和 FC1 conditional 的持久 optimizer state 非空，并在恢复 live 对象后逐项核验动量、groups、学习率及参数顺序。源中没有独立的 FC1 conditional scheduler，不伪造一个；源中真实存在的 Episode schedulers 原样保留。

CPU 限制：CUDA fresh-pair RNG 是 16-byte 状态，CPU generator 需要另一种格式。冻结诊断只认证并保留源字节，不能转换/套用到 CPU。原 SDF fresh bank/counters 可恢复；冻结模拟不消费 SDF training-pair RNG，使用原 identity-keyed simulation CRN。不宣称 CPU 等价重放 CUDA 的随机训练轨迹。正式 A800 恢复仍使用原 restore_fresh，未更改算法或随机规则。

新 H50 数据、初始 residual、conditional children/M/RHS 必须重建；旧学习状态用于初始化，不把旧 finite-direct 的收敛结果冒充新 H50 证据。
