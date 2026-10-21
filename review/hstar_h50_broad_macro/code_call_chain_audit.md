# 生产调用链审查
CURRENT_MACRO_HORIZON=1；TARGET_MACRO_HORIZON=50。
CURRENT_FIRM_SOURCE=T0_ONLY；TARGET_FIRM_SOURCE=MULTI_TIME_MACRO_STATES。

|问题|基线97a0181证据与新实现|
|---|---|
|Parent/Child|finite_direct_runtime.generate_dataset：t0 process_node→expand_after_parent_exit→两个child Entry/process_node|
|Default/Exit|expand_after_parent_exit 先生产 expand 保存 reference，再 _apply_exit_batched，之后 Child Entry|
|FC1 输入|simulate_ts_parallel._predict_macro_fc1_batched：x_t,x_child,hatc_cal,lnk_cal；forward_step 用 FC1|
|初始化|_initialize_batched_state：hatcf~Normal(-2,1)；construct_parent_bank 覆盖资本后 lnkf=log(sum K)|
|Firm Bank|firm_bank_from_captures 明确 source_time==0；新 broad_firm_bank 读全部合法 source times|
|FC1 rows|select_training_rows：真实 parent cal 与 child cal；新 rows t=1..50，parent time=t-1|
|TRUE_ONLY SDF|production_sdf_tensors 按 split,parent_candidate_id,t 排序并保留同父同时间branch0/1|
|Dynamic b|capture_templates 位于 child Entry 后、当前policy前，使用 state.b/K，不用 BP head|
|Strict root兼容性|t0无entry，既有 evaluate 资源公式与真实node回放一致；Exit-v2仍在下一期边界，新适配器独立认证|
|Journal/Approval|旧合同只有一期；V5 plan、source SHA、horizon/root config以及所有新执行阶段均被绑定|

## 组件映射与评分
Implemented：初始严格根适配、H50单主干、双Child配对、流式存储、跨时间Firm采样、calculated重建children/M、FC1/SDF bounded来源、动态Train b池、V5门控、CPU覆盖图，共10项。
Partial：GPU性能验证、完整原经济分布Bellman residual评价，共2项。
Missing：正式训练结果（本轮明确禁止，不计实现缺陷）。
实现覆盖评分=(10+0.5*2)/12=91.7%。这不是科学结论可信度或正式部署通过率。
