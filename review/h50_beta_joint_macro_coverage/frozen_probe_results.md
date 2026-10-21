# 最终冻结探针：32候选、H50

生成源码 `f48b2e2a7af02435ded2a43b47e0f30005cd7500`，`tracked_dirty=false`。原始结果完整保存在：

`/private/tmp/h50_joint_coverage_final32_f48b2e2_20261021/`

源 Job702636 Outer11；combined checkpoint SHA256：`9ad7610a1851d06d80c8b2d08052a01532923f7a2808e3a006f5538b404aa94d`。完整参数hash、来源事务/Optimizer/AL审计在证据包JSON中。CUDA Fresh-Pair RNG原字节保留但未当作CPU RNG使用；冻结模拟使用固定identity CRN，没有执行SDF抽样训练。不声称CPU与CUDA训练RNG完全等价恢复。

## 执行与安全性

先单独4候选tiny，再32候选冻结探针；CPU4线程，200企业/Parent，固定50次转移，Parent batch4。未触及服务器。

```ini
FROZEN_PROBE_STATUS = COMPLETE
MODELS_UNCHANGED = TRUE
OPTIMIZER_STEPS = 0
TRAINING_PERFORMED = FALSE
FC2_CALLS = 0
GPU_PERFORMANCE = NOT_TESTED
TRAINED_CONVERGENCE = NOT_TESTED
```

## 候选与真实覆盖

32提出，32严格Root通过、32独立ledger通过，29当前节点经济有效并入选，3当前经济无效未入选。32候选的最大normalized residual为2.791487e-6，低于未改变的1e-5门槛。

Empirical有效容量0：118784个历史模板账本状态未知/无效，未追认。Beta实际29，初始target32仍underfilled3；其中13个Beta是相对初始50%目标的真实补位。不能将未达到32写成补齐成功。

七类实际入选：low5、medium4、high4、broad5、u_shaped4、bimodal3、upper_tail4。初始占14/64个固定joint bins（21.875%），不是整个目标支持域已覆盖。

| 量 | min | median | max |
| --- | ---: | ---: | ---: |
| 物理K总量（全部32候选） | 11.430259 | 76.054344 | 324.557251 |
| proposed h belief | -3.924782 | -1.775594 | 0.310791 |
| proposed ell=log实际K | 2.436264 | 4.331435 | 5.782462 |
| 严格Root h_cal | -1.247252 | -0.915289 | -0.751946 |
| 账本ell_cal | 2.436264 | 4.331435 | 5.782462 |

proposed h belief不是label。原始h_cal及全部分位数在`coverage/coverage_phase_summary.csv`。`coverage_bin_gaps.csv`包含完整空bin；`candidate_acceptance.csv`保留全部32个候选及拒绝原因。

## 动态资格：不得只看初始数

| Child到达t | 可继续主干 | 合法Child0 | 合法Child1 | 完整SDF Pair |
| --- | ---: | ---: | ---: | ---: |
| 0（Root） | 29 | 不适用 | 不适用 | 不适用 |
| 1 | 29 | 17 | 15 | 14 |
| 5 | 8 | 5 | 5 | 5 |
| 10 | 1 | 1 | 1 | 1 |
| 20 | 1 | 1 | 1 | 1 |
| 30 | 1 | 1 | 1 | 1 |
| 50 | 1 | 1 | 1 | 1 |

总合法FC1 Child行197，完整Pair95（190条配对Child，另7条单Child仅可供FC1）。来源均为Beta；不是50:50合法训练分布。

29条初始路径的last_valid_t P10/P50/P90为0/1/6；有1条主干t50右删失。t50该主干predecision37家，post-exit35家；不是空经济体伪续接。其余大多早停，因此不能把197行当作197个独立初始Parent，也不能声称t10–50覆盖充分。

t0 hard default为4795/5800=82.6724%。本轮采样故意覆盖高b等家族，不是人口破产率估计。开发16探针最后合法t6，本次32探针出现1条长期路径，说明小样本选择敏感；两者均不构成收敛或因果证据。

## 性能

CPU峰值RSS3.406GiB，总137.40秒（包含来源认证/加载），候选生成0.030秒、候选严格h-star18.96秒；记录的rollout_seconds60.70秒包含rollout后的诊断收集/绘图。原始输出约14.66MB。有效行/全流程秒1.434。

没有A800显存、GPU吞吐或正式8192×H50性能验收；不得将CPU RSS称为GPU allocated。

## 图与真实阶段

`diagnostics/`有joint coverage、来源对照、family composition、时间有效行/Pair、h/lnK误差、Default/Entry图及逐行CSV。

D0/D1/D2没有训练后快照，阶段比较图明确标记NO_ACTUAL_STAGE_DATA；没有绘制伪造改善曲线。最终报告提交只补充比较表中的endpoint quantiles，不改变上述经济数据生成源码或原始探针结果。

当前科学结论：新覆盖产生合法joint状态，也能在此样本产生至少一条长路径；大多数路径仍快速丧失合法训练资格。长期覆盖充分性及FC1/SDF训练改善均未验证。
