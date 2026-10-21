# Firm Target and Loss Trace

## 1. Overview

只静态追踪生产训练路径；本轮仅运行冻结评价，所有optimizer/backward均被guard禁止。合成Firm宏观不进入FC1标签。

## 2. Entry Point

Mode A：`analysis/run_uniform_macro_frozen_firm.py:109–197` run。Mode B：`analysis/run_initial_hstar_h50_broad_macro.py` 的H50Backend firm_bank阶段，经显式FirmMacroCoverageConfig控制。默认plan-only。

## 3. Data Input Pipeline

`data/finite_direct_uniform_firm_macro.py:17–49` uniform_macro_bank与`:90–142` configured_macro_bank保留真实(b,z,eta,i,x)，更新h_f/ell_f。默认关闭时保持原Bank；Hybrid原两类剩余50:50。source composition与physical donor身份单列，不能将重复抽样索引视为新微观企业。

## 4. High-Level Call Graph

真实micro donor → synthetic h_f/ell_f → 同Parent context → FC1 conditional child → SDF raw/used M → 新P/Q/BP targets → 原生产scalar loss → 原optimizer（本轮未运行）。

## 5. Detailed Function Trace

| edge type | 函数与精确定位 | 输出如何被使用 |
| --- | --- | --- |
| direct call | `data/finite_direct_runtime.py:195–211` firm_transition_context / firm_bp_parent_rows | 同一context送到Child与BP adapter；synthetic逐值校验 |
| direct call | `data/finite_direct_runtime.py:214–237` conditional_child_bank | existing build_child_exogenous_bundle输出FC1预测、raw M和新Child tensor |
| nested call | `analysis/convergence_transition.py:120–185` build_child_exogenous_bundle | 原AR(1)与eta/i冲击，传入同一hypothetical context |
| nested call | `models/sdf_fc1.py:471–550` SDFFC1Combined.forward_step | FC1→Child h/ell；value_model→Parent/Child wealth；sdf_model.get_M→fresh M |
| direct call | `analysis/run_full_nested_finite_direct_outer4.py:285–315` ProductionBackend._execute | 同一个Bank进入P/Q/BP |
| nested call | `analysis/run_full_nested_equilibrium.py:791–846` _p_policy_outer | 每次build_problem，冻结策略solve后再算full-greedy residual |
| nested call | `analysis/run_neural_policy_iteration.py:619–672` _solve_frozen_policy | 解析导入run_adam/run_lbfgs_chunked至test_frozen_policy_root_solve，输出solved model与history |
| nested call | `analysis/run_full_nested_equilibrium.py:854–898` _q_picard_outer | current_sdf_child + fresh Picard targets，拒绝非Q参数变化 |
| nested call | `analysis/run_full_nested_finite_direct_outer4.py:371–388` _bp | 用metadata-aware parent rows重建BP target cache |

## 6. Loss Function Tracing

P：`analysis/test_frozen_policy_root_solve.py:561–571` root_loss = mean(p0_normalized²)+mean(pi_normalized²)。normalized residual来自当前Parent、新Child和生产RHS，不是旧macro label。

Q：`analysis/run_q_root_solve_fixed_macro.py:227–241` q_root_loss = mean(normalized²)，取current_sdf_child及每轮刷新target；default/zero/survival语义不改。

BP：`training/episode.py:9479–9555` `_compute_bp_cache_loss`依现有配置选择logit/level目标distillation，包含原confidence/sample/active weights；scalar total=bp0_total+bpi_total+mix_total。eta0没有融资控制，不能把未适用head误差当经济执行错误。

## 7. Loss Aggregation and Optimization

P Adam sink：`analysis/test_frozen_policy_root_solve.py:670–674` loss.backward→step；LBFGS closure `714–735`。Q sink：`analysis/run_q_root_solve_fixed_macro.py:442–449`，LBFGS `501–508`。

BP sink：`training/episode.py:9857–9883` zero_grad→total.backward→原gradient guards/clipping→optimizer.step。新模块不重写这些loss系数、现金流或scheduler。

## 8. Unresolved / Ambiguous Points

没有新训练，故未观测本轮优化趋势。现有M采用原common clamp和各operator既有处理；本轮不改变它们。宽域raw M极端且大量clipping，有限target不等于Macro可信。

旧MacroTransitionContext容器的字段名叫hatc_cal/lnk_cal，但此处仅复用其条件转移接口；synthetic语义在Bank metadata显式标注，没有把这些值写入FC1 calculated监督标签。

## 9. Final Summary

关键修复是重新构造新宏观条件下的Child/M/target，并给BP同一个context，不是仅改Parent cols5/6。训练接口复用旧实现；经济公式未变。

## 10. Structured Trace Table

以上表格映射数据阶段；P/Q/BP三个scalar及各backward sink已列出。冻结对照额外启用的regret teacher与原value-only target在首batch的P0/PI/Q误差均0；该只读开关默认false，不改变训练默认路径。
