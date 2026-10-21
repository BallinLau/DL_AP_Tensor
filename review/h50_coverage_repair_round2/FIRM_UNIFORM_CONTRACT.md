# Phase B — Firm Uniform contract

复用已审计旧 `uniform_macro_bank`、独立 Firm-only runner、生产 P/Q/BP block；没有新增另一套Bellman公式或优化器。删除cost01分支，不改成本校准。

默认 FirmMacroCoverageConfig.enabled=false、fraction=null；原H50默认batch512、既有ON_POLICY/CALCULATED 50:50保持。显式启用HYBRID必须给fraction，剩余行按50:50分配（奇数最多差1行），Uniform单列来源。STANDALONE使用独立runner，默认配置复现2048 batch、3inner、8192/2048行、[-6,2]×[-2,5]；依然plan-only，缺批准不得执行。

Uniform从真实微观donors保留(b,z,eta,i,x)，按eta/b/z占据strata交错选择，再向量化生成h/ell。无h-star筛选；不把合成宏观当calculated标签。报告eta、b/z、h/ell覆盖、实际来源与独立donor数量；同一微观donor可出现在不同collocation类型，显式记录，不声称独立经济体。

`firm_transition_context`统一Conditional Children/M/BP parent rows。原两类observed行继续用同节点true context；Uniform行用新Parent cols5/6，并逐值验证。`conditional_child_bank`依原FC1预测和SDF重建raw/used M，Parent fingerprint、Macro hash和Child fingerprint更新；P/Q/BP在生产block重建targets。无旧target复用。

配置参与plan/Approval fingerprint；本轮不生成approved=true。batch/采样budget只在显式启用时采用新值。3inner规则与原预算相同。原StageJournal语义不改。

测试假设：新输入确实进入Child与M；原宏观状态不被篡改；默认关闭；缺fraction拒绝；Hybrid实际48/48/32例子准确；eta0/1都有；Macro完整state不变。定向套件当前88项通过。

风险：Uniform可以远离冻结Macro可信支持域；不裁剪，记录M尾部/clipping和target finite结果。parent finite不等于target finite。困难state不能自动删掉来完成训练。模型经济收敛未测试。

数据类型明确区分FIRM_UNIFORM_MACRO_COLLOCATION、FIRM_ON_POLICY、FIRM_CALCULATED_MACRO_COLLOCATION与BETA_JOINT_ECONOMIC_COVERAGE/DYNAMIC_EMPIRICAL，前3类是Firm输入，后2类是经济Parent来源。
