# Finite Direct 四轮 Runner 集成审查

最新限定审查修复见 `approval_metrics_patch_report.md`：approval schema v2额外绑定输入artifacts，
每轮最后SDF chunk后增加固定Validation-only post-SDF重模拟，并从现有账本生成跨Outer历史与图。
下文49项是原bdb58fc集成基线测试，不代替新补丁的定向测试结果。

## 结论与授权

基于 `2a456f000ab57e3a55c3956a7e2e0d430aa0254e` 补齐独立执行器。
**本轮只有代码、本地非生产优化测试和审查资料，没有正式训练、Slurm、push 或 merge。**
`RUNNER_READY_FOR_USER_REVIEW` 仅表示代码接口已连接且可审查，不是 A800 训练验收，更不是均衡收敛。

## 1. 正式入口与调用链

入口 `analysis/run_full_nested_finite_direct_outer4.py`，默认只输出计划并退出。
`--execute-approved` 必须同时提供与 clean commit、plan fingerprint 一致的独立人工批准文件；本轮没有生成批准文件。
实际调用关系、旧路径排除理由见 `finite_direct_runner_call_graph.md`。

## 2. Outer0 的复用

从已有组合 checkpoint 加载原 Firm、φ8 FC1、原 SDF/Wealth；校验 format、组件 hash。
Outer0 只有 `reuse_firm`，不运行 P/Q/BP。标记 ROOT_CLEARED→FINITE_FIRM_DIRECT_AGGREGATION 的过渡轮。
本地使用真实已保存 checkpoint 初始化 backend 成功，优化步0、FC2调用0。
这不是从旧 Root run 的事务恢复，不导入旧 candidate shard 或旧 target。

## 3. Outer1–3 的 Firm 数据

每轮使用上轮 committed Macro 执行真实 finite-direct production simulator。
观察现成的同节点7D企业输入与该节点的 calculated h/LnK，不跨节点拼边际列。
Train/Validation 分别取独立来源，path级交集必须为空。
通用 `build_collocation_bank` 会混合边际、通用 `build_child_bank` 默认拿 perceived macro 作上下文，因此不直接使用。
改为复用其底层 `build_child_exogenous_bundle`，显式传入同节点 true macro context。
条件 child M 来自当前 Macro、按现有 common P clamp [.7,1.3] 形成 used M，raw M 另存；P/Q使用同一used M。
后续 exact eta integration 保留生产实现。固定训练行预算8192/2048，不替换4096/1024个宏观Parent规模。

## 4. P/Q/BP 的实际生产入口

P复用 `_p_policy_outer`→`build_problem`→现有 frozen-policy Adam/LBFGS。
Q复用 `_q_picard_outer`→`build_q_survival_problem(m_mode=current_sdf_child)`→原Q Adam。
BP复用 Episode 的 target cache 与 `_run_bp_distillation_stage`；其 parent附同节点真实macro列，避免7列表把M退回1。
P/Q/BP每块核对Macro不变；原P/Q内部进一步检查非目标网络hash。BP另核对P/Q前缀不变。
BP没有合法active行/没有optimizer执行时为硬错误，不伪称完成。
另显式在已完成P/Q/BP块后hard-refresh Episode的simulation Grid target；这些现有求解器不走Episode.train_step，
不会自动触发原epoch-hard刷新。模拟前校验Grid target hash等于当前committed Firm，防止Outer1–3继续执行旧Firm target。

## 5. Macro与FC1

`D0 → FC1 0 → D1 → FC1 1 → D2`，每次都消费已锁定初始Parent与CRN。
Simulator复用已执行过的 `simulator/process_node` 与 production entry/transition/ledger。
不调用Root、FAST、Boundary integration；安装Root/FC2失败即抛错的guard。
一个合法child可进入FC1；失败行不删除，coverage同时记录attempted Parent、合法行和完整pair数。
输入真实 `(x_t,x_next,h_cal_t,lnK_cal_t)`，目标真实child calculated h/LnK，不用forecast冒充。

FC1实际调用 production `_compute_fc1_only_loss`，原C/K权重1/.25。
优化至少25、最多40epochs，训练来源固定每10个Parent取1个development组，余下做fit，Validation不参与选best。
同一FC1 optimizer跨macro阶段与Outer连续；epoch checkpoint包含匹配的last模型及optimizer。
best只作为带匹配optimizer的诊断文件，不用best权重与last moments混装。

## 6. SDF/Wealth

只有同一有效Parent的完整双child可进入SDF，parent_index与经济ID分开。
真实执行 `_run_batches`→`train_step`→`_compute_sdf_loss(SDF_TRUE_ONLY)`；并调用生产pooled AL estimator和dual update。
保留Signed AiO、normalized wealth residual、原AL约束；不修改公式。
每epoch用真实生产验证入口记录Validation；每5个primal epochs按原公式更新dual。
两个50epoch chunk之间不重置optimizer、scheduler、AL或fresh shock RNG。
四轮保持AL连续；每Outer首次SDF入口显式配置该phase LR，optimizer moments不清空。
convergence只报告，不沿用旧validated phase中因gate通过/未通过而停止整个Outer的逻辑。

本地forward-only测试用真实模型/Episode确认TRUE_ONLY读取parent true列、dummy child M列不被误用，且原AL estimator可运行。
Formal fresh pairs仍按原SDF条件AR1抽样；固定模拟pair保留用于监督/固定验证/AL pooled估计。

## 7. 跨Outer

每轮D0记录pre-Macro Firm残差，SDF后使用同一固定probe/CRN重算当前Macro下P/Q残差。
Outer0建立的独立probe Parent行不再换样，M/FC1随当前Macro刷新。
`outer_commit`保存独立完整checkpoint；下一轮验证当前组件hash等于上一轮committed hash后开始Firm。
不因为P/Q残差、R²、G非零或AL未达阈值而提前终止四轮计划。
非有限参数/梯度、损坏provenance、非法pair或无合法训练样本必须停，不造数据。

## 8. 恢复与缓存

见 `checkpoint_resume_contract.md`。P/Q/BP按块，FC1按完整epoch，SDF按完整chunk提交。
未提交块从上个commit的完整状态重做，明确 `RESTART_UNCOMMITTED_BLOCK`，不声称step-exact。
模型、所有持久optimizer/scheduler、AL、RNG、fresh-bank eps/identity与pair RNG、阶段计数、data版本同时保存。
采样bank与CRN固定，不重抽。cache绑定所有Firm/FC1/SDF/Wealth hash及经济/算子/采样/Parent/CRN/stage版本。
仅成功发布latest pointer的文件才是已提交阶段；中断残留不算完成。

## 9. 经验模板

当前仍是699894的真实t0保存数据，标准标签 `RECORDED_INITIAL_PARENT_DISTRIBUTION`。
默认启用的状态捕获还会记录已发生t≥1节点，标为 `ENDOGENOUS_SIMULATED_PARENT_DISTRIBUTION`。
捕获只读取现成状态，无额外政策forward，不自动升级模板、重分配Parent或根据child成功筛选。
过去审查包的旧verbose标签可作为旧artifact provenance保留；新生成模板使用新标准标签。

## 10. 测试、发现及限制

49项targeted tests通过，详见 `test_results.txt`。真实生产SDF/FC1损失只forward，无生产optimizer step。
四Outer顺序和中断恢复使用fake backend/optimizer状态；不是四轮真实训练。
CPU小规模production ledger/entry/branch接口测试显式使用已有Grid CPU reference选项，正式Grid CUDA guard不变。
另使用真实warm-start checkpoint完成CPU backend构造/hash核验，未模拟正式数据、未优化。

未测A800峰值/吞吐和全预算数值稳定性；没有跑全仓中可能触发optimizer的测试。
因此不承诺四轮可无硬错误跑完，也不报告训练完成或均衡收敛。
没有已知未连接的关键P0接口；实机运行风险需在用户批准后才能检验。

## 11. 审查后停止

完整预算与CLI见 `planned_training_config.json`、`planned_slurm_command.txt`。
Slurm脚本只准备、不提交。本地commit后停止；没有自动push/merge、后台运行或定时续训。
最终本地commit见交付目录的 `git_info.txt`，训练数据生成commit为NONE（本轮无正式训练/数据生成）。
