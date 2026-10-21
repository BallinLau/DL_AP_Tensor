# 基线审查（c725c69）

主目录 HEAD bf639c2 与目标分支不同；旧 H50 worktree HEAD 正确但有其他未提交 Uniform Macro 修改。本轮从 c725c69 新建独立 exp/h50-joint-macro-coverage-20261021，不覆盖它们。

## A. Beta 生成

data/finite_direct_sampling.py:117–251：SamplingConfig/fixed_schedule/construct_parent_bank。每 Parent200 firms；Sobol6选alpha1.8–3.2、beta4.8–7.8、K总量45–85、资本权重幂0.5–2、正/负/独立b–K排序。没有U形/双峰/高杠杆等完整family覆盖，也没有b–z rank coupling。

data/simulate_ts_parallel.py:757–829 的生产initializer：x/z来自校准AR(1)stationary分布，eta原Bernoulli(ZETA)，i原Uniform(0,I_THRESHOLD)。原hatcf为Normal(-2,1)，原lnkf随机Normal(4,1)，但construct_parent_bank随后改写lnkf=log实际K总量；H50再在t0调用原STRICT_GLOBAL h-star，不将随机hatcf当calculated标签。尚无h/lnK联合稀缺bin选择。

## B. 混合

fixed_schedule初始Parent精确50:50，旧resample_from_sources强制4096/4096、2867Recent/1229Archive；旧Empirical可反复抽相同lineage/template，无eligible capacity fallback。Firm Bank的50:50指ON_POLICY/CALCULATED，不是Beta/Empirical。FC1/SDF sampler按真实资格抽parent-time，来源比例随存活改变。source IDs隔离Train/Validation，H50固定Validation内容SHA。

历史Pool template的post_decision_accounting_valid=NOT_YET_EVALUATED不能当成已认证的节点账本。本轮新eligible规则将明确区分finite b/K模板合法性与实际节点经济认证；不伪造旧模板有效标记。

## C. H50与FC1真实链路

generate_long_horizon_dataset→calibrate_initial(STRICT_GLOBAL)→process_node；t>=1 expand_after_parent_exit→原exit→原entry→process_node，Child0 promotion仅融资BP，不重复账本，Child1不成为主干。continuable_nodes按C/K/log是否定义决定原Entry继续机会，不等同training eligibility。

FC1 direct call：ProductionBackend._fc1_epoch (analysis/run_full_nested_finite_direct_outer4.py:390–421)→xy (run_fc1_self_consistent_macro.py:319–324)→production_loss (:353–358)→Episode._compute_fc1_only_loss (training/episode.py:3874–4027)。X=[x_t,x_t1,Hatc_cal_t,LnK_cal_t]，Y=[Hatc_cal_t1,LnK_cal_t1]；parent cols7/8真实宏观；网络forward_step使用同样真实输入。总loss为recon_weight*(hatc_weight*MSE_h+lnk_weight*MSE_K)，recursive auxiliary在setup关闭。backward/FC1.step位于runner:407。不修改这条训练路径或权重。

SDF使用select_training_rows/production_sdf_tensors，同split/parent/time完整两Child；Firm calculated states继续重建conditional children/M/RHS。

## D. 插桩缺口

旧write_diagnostics没有完整t0–50主干、hard计数、Entry时点、固定Validation阶段比较；它主要统计Child eligibility，不能代替continuation survival。上轮Observer有细节但只用于独立CPU probe。此次需把轻量节点事件与true-state预测纳入production-shape streaming shards，在所有_store阶段统一产表，而非另建模拟器。

后续正式trace报告按实际新代码行号更新；此处只说明已解析的基线direct calls，不表示运行了任何优化。
