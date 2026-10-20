# Finite Direct 四轮训练：实现审查包（尚未完整接通）

## 结论与授权边界

本次只实施 Phase A：本地代码、单元测试、CPU 初始状态构造与已保存权重组合。
没有执行新的 Firm/FC1/SDF optimizer step，没有提交 Slurm，没有联系 4090，没有 push/merge。
第二份补充指令覆盖第一份 Prompt 中“立即正式训练”的要求。

**这是部分实现，不是已经可以提交的四轮训练 runner。**
已完成抽样模块、宏观记录资格检查、CPU 组合 checkpoint 和审查准备命令。
完整 Outer 执行器、有限企业 Firm bank 捕获及阶段内严格恢复尚未完成，不能宣称实现全部完成。

## 1. 分支与版本

- 独立 worktree：`/private/tmp/DL_AP_Tensor_finite_direct_outer4_review_20261020`
- 分支：`exp/full-nested-finite-direct-outer4-review-20261020`
- 基线 commit：`3575abd44dd2b7e197662f2b053db036ff18f76e`
- 最终本地提交以交付的 `git_info.txt` 为准；没有远端 push。
- 主工作区原有修改未覆盖。
- CPU 准备结果的 `preparation_source_manifest.json` 记录实际准备时的 commit 与 dirty 状态。
  本次没有新的模拟或训练数据生成 commit；不能用最终文档 commit 冒充训练源码版本。

## 2. 实际修改文件与 diff 范围

新增：

- `data/finite_direct_sampling.py`：有来源记录的 Beta/经验 CDF 抽样、计划锁定和恢复校验。
- `data/finite_direct_macro_contract.py`：FC1 单合法 child / SDF 完整双 child 的不同资格与张量接口。
- `analysis/prepare_finite_direct_outer4_review.py`：仅 CPU 准备，无训练执行入口。
- `tests/test_finite_direct_sampling.py`
- `tests/test_finite_direct_macro_contract.py`
- `tests/test_finite_direct_review_preparation.py`
- 本报告。

没有改动 `training/episode.py`、P/Q Bellman、Signed AiO/AL、production resource ledger、SimulateTS 或 root solver。
新 sampler 尚未连接正式四轮执行器；不能以“已有 prepare 命令调用 sampler”代替该集成验收。

## 3. 抽样设计

按 Parent 而不是企业分配 50:50，计划在模拟之前固定，不根据 child 成败重分配。

| Split | 总 Parent | Beta | Empirical | 每 Parent 企业数 |
|---|---:|---:|---:|---:|
| Train | 4096 | 2048 | 2048 | 200 |
| Validation | 1024 | 512 | 512 | 200 |

Beta 使用 Sobol 参数，alpha ∈ [1.8,3.2]，beta ∈ [4.8,7.8]，b ∈ [0,1]。
资本 K 总量支持沿用 [45,85]，资本权重使用既有初始设计。z/x/eta/i 和初始 h 的抽样调用现有 production initializer。
此准备过程不运行企业政策、进入、退出或账本。

Empirical：先按模板中资本排名划四组，保存各组 b 的离散经验 CDF；
新 Parent 独立生成资本、按新资本排名分组，再从对应 CDF 抽 b。
组样本数小于事先锁定的 8 时使用同一模板总体 CDF，并记录 fallback；本次实际 fallback 为 0。
没有 jitter、插值、核平滑或额外冲击。经验 b 支持由实际 source 样本决定。

资本条件 CDF 保留的是 b 与资本分位组的粗粒度依赖，不保证精确重建原连续 b–K copula。
不保留完整 b–z、b–eta 联合分布；这不是高维联合分布生成器。
Firm identity 是 `(new parent_id, local firm_id)`，局部编号可相同，但全局身份与来源不同；没有复制旧完整企业状态。

## 4. 经验数据 provenance 与限制

来源是已完成 Job 699894 的真实保存文件：

`cachedir/fc1_self_consistent_macro_fixed_point_20261020/banks/train.pt`

`cachedir/fc1_self_consistent_macro_fixed_point_20261020/banks/validation.pt`

source commit = `3575abd44dd2b7e197662f2b053db036ff18f76e`。
Firm hash = `fce66108a07152375c3e4a84950b001d3555c10e2a2c9c7c3540afefa5a93820`。

来源口径明确为 **RECORDED_INITIAL_PARENT_BANK_PRE_DECISION / t=0**。
这些是历史 finite-direct 模拟实际使用的初始企业横截面，不是演化后 on-policy 分布。
历史动态文件只有 aggregate 输出，不能从这些摘要恢复动态逐企业 b。
没有根据未来成功的 child、验证 R² 或资源账户结果筛选模板。

模板总量为 Train4096 / Validation1024；本次分别选择2048 / 512个，不重复使用。
除了带来源的模板 fingerprint，也记录只基于 CDF 值的 shape fingerprint，避免用不同 template ID 虚报分布差异。
选中经验 CDF shape 数为2048 / 512，模板复用率均为0。

## 5. 独立性与 CRN

来源 `(source_run,source_path)` 的 train/validation 重叠为0；模板ID重叠为0。
新 Parent IDs：train从2000000开始、validation从3000000开始，不覆盖来源ID。
两个 child 的键保存 transition seed、entry seed、parent/path ID、time=1 和 branch=0/1。
这里只核验键的设计；没有在本次执行 production 双 child 模拟，不能宣称动态独立性已经端到端实测。

## 6. Frozen checkpoint 兼容性

Firm 原文件 SHA-256：

`25183836fc856915448444e32bca997f2ec0e5c894e48f730180efe3a88c2110`

已在 CPU 验证原文件、Firm 权重 hash 和 φ8 中 frozen Firm hash。
只将 `fc1_model.*` 从 `fc1_iteration_008_complete.pt` 导入原始 Macro state dict，
核对 FC1 参数名、shape、dtype 以及每个组件 hash。
SDF、Wealth 非 FC1 tensor 与原始 checkpoint 逐项相同。
详细值见生成的 `checkpoint_manifest.json`。

这是 **新 run 权重 warm start**，不是旧 root-cleared transaction 的等价 resume。
不导入旧训练 bank、target 或 simulation dataset；不将旧 operator optimizer 状态当成新阶段 optimizer。
Outer0 计划复用已完成 Firm，不重新优化；Outer1–3 必须重新生成 finite-direct Firm bank。

## 7. FC1 / SDF 接口

FC1：允许有效单 child，输入 `(x_t,x_next,Hatc_cal_t,LnK_cal_t)`，目标为真实 child calculated h/LnK。
SDF：只有同一有效 parent 的两个有效 child 才有资格，invalid 行完整保留在 audit。
检查 parent lineage、时间/branch 标识、重复行、兄弟 parent macro context、非有限值以及 child h/LnK 与 raw C/K 对应关系。
TRUE_ONLY parent tensor 的 true h/LnK 位于7/8列，child true h/LnK位于8/9列；连续 `parent_index` 与全局经济ID分开。
没有 parent forecast 来源时写 NaN，而非造一个零预测供递归模式使用。

**限制**：当前是数据契约及布局测试，不是 production `Episode._compute_sdf_loss` 的端到端训练验收。
尚需接通 SDF fresh shock-bank、parent_index、epoch pair RNG 与 AL checkpoint；未接通前不得正式启动。

## 8. Resume 与缓存

当前已验证的是 sampling resume：

- 配置 / 模板 fingerprint / split / parent schedule 不同即拒绝。
- 每个采样 artifact 的 SHA-256 锁定，tampering 即拒绝。
- bank tensor hash、coverage中bank hash 与 manifest 唯一性核对。
- 原始输出目录非空却没有有效签名时拒绝覆盖；不自动删掉部分结果。
- 组合 checkpoint 再次读取时核对完整 source manifest 和权重 hash。

**尚未实现**训练阶段内 P/LBFGS/Q/BP/FC1/SDF 的完整 optimizer-step 恢复事务。
不能把 sampling resume 宣称为四轮训练 resume。
旧 FC1-only simulator cache 主要按 FC1 hash 识别，完整 Outer 集成时还必须加入 Firm/SDF/operator/sampling hash，禁止复用旧 cache。

## 9. 实际测试

以下命令仅 CPU 单元测试，无优化：

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/fc1_mpl_cache PYTHONPATH=. \
  /Users/ballinliu/anaconda3/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_finite_direct_sampling.py \
  tests/test_finite_direct_macro_contract.py \
  tests/test_finite_direct_review_preparation.py \
  tests/test_fc1_self_consistent_macro.py
```

结果：**32 passed**。
包括真实准备命令之外的合成 fixture 测试；fixture 明确标记 TEST_ONLY，不作为经验实验来源。
checkpoint composition 的测试故意令 φ8 的 SDF/Wealth 不同，验证它们不会被导入。
prepare 命令的 AST 检查确认无 backward、optimizer step 或训练提交调用。

补充指令12项中的第11项“完整 Outer runner 调用新 sampler”尚未通过；
第12项当前只覆盖 sampling，不覆盖训练优化器恢复。

## 10. 实际生成的文件

本地最终准备目录：

`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/cachedir/full_nested_finite_direct_outer4_20261020_review/v2/`

包括：

- `sampling/macro_initial_distribution_manifest.csv`
- `sampling/empirical_template_bank.pt`
- `sampling/empirical_template_manifest.csv`
- `sampling/train_parent_sampling_schedule.json`
- `sampling/validation_parent_sampling_schedule.json`
- `sampling/sampling_coverage_summary.json`
- `sampling/train_initial_bank.pt` / `validation_initial_bank.pt`
- `sampling/sampling_signature.json`
- `initial_combined_checkpoint.pt`
- `checkpoint_manifest.json`
- `training_plan.json`
- `preparation_source_manifest.json`
- `review_status.json`

这些只有初始状态与已有权重；没有新的宏观标签、Bellman target、训练历史或 Outer checkpoint。
前一版准备输出保留，未覆盖旧实验。

## 11. 四轮预算与执行顺序（计划，不是执行证据）

Outer0：reuse verified Firm → D0 → FC1 → D1 → FC1 → D2 → SDF chunk0/1 → cross-block → checkpoint。
Outer1–3：finite-direct Firm bank → P0/PI → Q → BP，再执行相同 Macro 链。

Firm inner≤3、P policy outer≤3、Adam200、LBFGS100、Q Picard≤3、Q fit300、BP≤1000。
每轮FC1两次，最少25/最多40epochs，patience10，原C/K权重1/.25；SDF两chunk、各50epochs。
计划以固定训练开发子集选FC1 best，不以Validation选模型；固定独立Validation和probe跨轮比较。
非收敛只报告，不作为Outer3之前整体早停理由；无合法数据/非有限值等硬错误必须停止。

## 12. Slurm 命令与资源

**目前没有可供正式执行的 exact Slurm command。** 完整 runner 未接通，不能提供一个看似可执行但实为占位符的 sbatch 命令。
没有创建提交后自动启动、后台继续或自动恢复任务。
计划资源为 THU a01 / 1×A80080GB / 8CPU / 128GB host memory / walltime最多7天。
这只是申请预算，不是已测runtime/显存承诺；本次未运行A800，不能给出四轮可靠ETA或峰值。

## 13. 剩余风险与下一步

优先完成同一件事：把新的 locked finite-direct banks 接入完整阶段事务执行器，
包括 Outers1–3 Firm collocation/child的真实宏观上下文，以及全部 optimizer/AL/fresh-shock RNG恢复。
然后运行不优化模型的本地阶段编排/冻结范围/断点恢复接口测试，再提交完整审查包。
尚未完成这些集成，不应要求用户批准正式训练。

本次只新增抽样、契约、CPU准备代码；没有更改 Firm P/Q、SDF loss 或资源会计。
但训练分布会改变，且初始权重组合不同，必须新 run 初始化，不能使用旧 simulation cache。

## 14. 真实状态

```ini
IMPLEMENTATION_COMPLETED = FALSE
TESTS_PASSED = TRUE
BETA_EMPIRICAL_SAMPLING_READY = TRUE
FOUR_OUTER_RUNNER_READY_FOR_REVIEW = FALSE
FORMAL_TRAINING_STARTED = FALSE
SLURM_JOB_SUBMITTED = FALSE
OPTIMIZER_STEPS_EXECUTED = 0
CURRENT_PHASE = WAITING_FOR_USER_REVIEW
```

上述 TESTS_PASSED 只指已运行的32项，不代表所有完整 runner 验收条件已经通过。
本报告不声称训练完成、均衡收敛或可以正式提交。
