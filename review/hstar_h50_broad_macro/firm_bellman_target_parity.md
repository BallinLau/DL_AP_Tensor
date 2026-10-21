# Calculated Collocation的目标重算
没有持久化Bellman labels，也不修改原on-policy targets后复用。
新broad bank→当前冻结Macro下的conditional_child_bank→原build_child_exogenous_bundle→FC1 child context、raw/used M→原P/Q Bellman RHS和BP Grid teacher。
metadata parent_fingerprint绑定新的7D输入。CPU非优化测试实际调用Macro子模块，验证calculated bank fingerprint与新child provenance，以及finite child/M。
单个企业仍是price taker：不因候选b'重求整个经济市场出清。
另执行真实Outer11模型的8行小型Bellman forward，包含4行calculated与4行on-policy，实际调用原build_problem/evaluate_problem/initial_parity。P0/PI动态RHS与production_target的最大误差约5.96e-8，通过原1e-5阈值。normalized residual均值约0.03324，这是小型分层probe，不是总体收敛证据。
本次未执行Adam/LBFGS/Picard/distillation；“接通原RHS”不是“已完成Firm训练”。
