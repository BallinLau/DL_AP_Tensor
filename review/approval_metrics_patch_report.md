# bdb58fc 最小审查修复：批准绑定与跨Outer指标

## 范围与状态

基线 `bdb58fcc6a1fcbd74a43267076536066d9c27250`。仅修复批准绑定与结果记录，未改变生产训练方程/预算。
新commit见交付 `git_info.txt`。没有训练、Slurm、GPU优化、独立实验或push，没有生成 `approved=true` 文件。
测试中的有效批准字典仅在内存中模拟，磁盘仅有schema文档和无效批准fixture。

## 1. Approval Artifact Binding

`require_approval()` 保留默认plan-only退出。显式execute开关仍须通过新schema v2：

- `schema_version = 2`
- `approved` 必须是布尔true（只能由未来明确人工授权产生）
- `source_commit` 必须匹配当前clean checkout
- `plan_fingerprint` 必须匹配当前完整计划
- `sampling_signature_fingerprint` 必须匹配实际signature文件的fingerprint
- `sampling_signature_sha256` 必须匹配实际signature文件内容SHA（包括每个bank artifact的hash manifest）
- `initial_checkpoint_sha256` 必须匹配实际combined checkpoint文件SHA

额外绑定signature文件SHA是为了防止保留配置fingerprint字符串、却更换bank hash manifest。
不绑定Mac/HPC绝对路径；相同内容复制/改名后可通过同一批准，内容变化即拒绝。
新schema文件是规范，不是批准。未提供任何可启动生产的批准文件。
旧source/plan-only批准不再有效；未来必须对新commit/new plan及实际部署artifacts重新审查批准。

## 2. 经济指标与分母

新模块 `analysis/finite_direct_outer_metrics.py` 仅在CPU聚合现成逐child记录，零模型forward。
每个D0/D1/D2分别记录Train/Validation × BETA/EMPIRICAL/ALL。
post-SDF阶段仅Validation，Train post-SDF为未执行/NOT_AVAILABLE，不会假称有训练后Train重模拟。

| 指标 | 口径 |
|---|---|
| G_mean/median/p90/p99 | 有效经济节点中有限的saved normalized_G_at_prediction；沿用699894报告口径 |
| G_raw_mean / G_raw_abs_mean | 同样有效节点中的saved G_at_prediction，另保留物理尺度 |
| Hatc_forecast_MAE / LnK_forecast_MAE | saved forecast与saved realized calculated label的绝对误差 |
| FC1_C_R2 / FC1_K_R2 | saved forecast vs realized label的水平R² |
| Delta_LnK_R2 | (LnKF_next−LnK_cal_t) vs (LnK_cal_next−LnK_cal_t) |
| Surviving_firm_count_mean | 所有有定义的saved surviving_mass均值，包含失效节点记录的0；finite firm单位质量下为存活数量 |
| K_mean / C_mean | 所有有定义的saved K_cal_t1 / C_raw_t1，包含失效节点已算出的负消费；不填补未定义后缀 |
| Default/Investment/Entry_rate | sum(saved numerator mass)/sum(saved pre-decision firm_mass)，所有账户有定义的尝试节点 pooled ratio |
| Valid_child_fraction | 有效child / (2×预定来源Parent数) |
| Complete_SDF_pair_fraction | 完整有效双child Parent / 预定来源Parent数 |

Default/Investment是已有gate-weighted mass统计，不重新推断hard决定，不裁剪比率。
Entry分母是该节点saved firm_mass（含已获准进入者），不是另设进入人口规则。
覆盖率同时包含scheduled、attempted、valid Parent/Child、failed Child、complete pair counts。
G有限样本数与forecast有限样本数另报，避免非有限预测被悄悄藏掉。
存活、C/K及mass比率同时提供`_valid`条件子集列与finite count，主指标不因child灭绝就抹去已记录的0存活/违约/负消费。
无数据/常数目标导致不可定义R²时，CSV写 `NOT_AVAILABLE`，不是0。
原逐child明细、失败原因及shard保留不动。

## 3. 真正的post-SDF评价

阶段顺序只新增：最后SDF chunk → **POST_SDF_VALIDATION_RESIMULATION** → cross-block → Outer commit。
调用原 `generate_dataset`，仅使用原固定Validation bank、seed、Parent/Child CRN，不优化、不开Root、不重新训练任何网络。
新阶段名与专属目录区别于D2；cache仍绑定当前Firm/FC1/SDF/Wealth hash，SDF变化后旧结果不能接受。
方法有no-grad，临时eval，并恢复所有module模式和Python/NumPy/Torch/CUDA RNG；结束后核对参数hash不变。
不读取D2缓存作为post-SDF结果。此阶段是正式Outer内嵌评价，不是独立实验；本次未实际执行该正式评价。

## 4. 输出与恢复

从**已提交StageJournal结果**重建三份CSV：

- `outer_training_history.csv`
- `outer_macro_economic_history.csv`
- `outer_sdf_al_history.csv`

SDF复用原已保存每epoch JSON history，通过SHA验证引用，不把整份历史反复复制进每个checkpoint。
CSV在已提交data/SDF/evaluation/Outer边界更新，可幂等重建；未提交阶段不会成为结果行。
StageJournal源码、原子提交协议、resume粒度完全不变。
新commit与新plan自然使旧contract不能静默resume；原恢复测试仍通过。

五张图：`outer_g_residual.png`、`outer_fc1_r2.png`、`outer_firm_residual.png`、`outer_sdf_aio.png`、`outer_survival_capital_consumption.png`。
经济图只画Validation/ALL，各D阶段与post-SDF分开，曲线按同一CRN hash分组，不跨CRN连线。
全部BETA/EMPIRICAL细分保留CSV。Firm图明确标 `FIXED_7D_PROBE`，不冒充新内生Parent上的训练残差。
SDF图标明固定macro-pair TRUE_ONLY validation的normalized Signed AiO，不称其为post-SDF重模拟目标。
无真实数据时显示NOT_AVAILABLE；本轮只有合成单元测试临时图，没有生成正式Outer曲线。

## 5. 明确保留的内容

P0/PI/Q、BP Grid、FC1 loss、Signed AiO、Wealth recursion、AL更新公式、资源/资本会计、冲击过程均不动。
4096/1024 Parent、200 firms、双child、Beta/Empirical 50:50、四轮预算及Firm-first顺序均保留。
Empirical仍是 **RECORDED_INITIAL_PARENT_DISTRIBUTION**（699894的t0），没有升级t≥1 on-policy模板。
原t≥1 capture保留，本任务不开发动态模板更新器。

## 6. 验收

相关非优化测试与静态检查见 `test_results.txt`。不运行全仓优化测试或GPU benchmark。
测试覆盖批准字段替换、路径迁移、invalid approval无法构造backend、分组与账本统计、失败覆盖、
StageJournal恢复、post-SDF缓存隔离/无optimizer/模型及RNG不变、CSV幂等与五图输出。
仅在审查分支记录一次本地commit，完成后停止等待最终审查；不生成批准、不提交作业。
