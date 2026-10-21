# Phase G — Matched Firm evaluation

## 可认证输入

Before：Job702636 Outer11，checkpoint SHA `9ad7610a1851d06d80c8b2d08052a01532923f7a2808e3a006f5538b404aa94d`。

After：既有标准Uniform Firm r2，SHA `d26b9004210eef0be224c79d1f1faef6e59692e48ceba45050e616865ef2d325`；只读从THU下载。历史summary/history记录2048 batch、3inner、h[-6,2]/ell[-2,5]，不是cost01变体。Macro完整state_dict（含buffers）与Before逐字节digest相同；Firm hash不同。

历史采样8192行只有1211个独立firm-time donor，替换抽样不创造独立微观状态。本轮新增报告也按path/time/branch/firm_id核对独立donor，不能将Bank行索引数称为微观独立数。

## 最终固定7D网格

1944行：9×9 b/z网格、eta0/1、12个宏观背景；包含常见区域、Uniform四角、低ell及从原H50 t1/t2/t3/t4合法节点选出的真实低K背景。x/i与h/ell在两个checkpoint间完全一致。1296行的早期8背景探针是开发证据，不替代最终1944行结果。

同一Macro、同一Conditional CRN，Child fingerprint完全相同；重新计算原Bellman RHS及BP Grid，未复用旧targets。带regret诊断teacher与原value-only target在首batch的P0/PI/Q最大绝对差均0。eta0不融资，BP head MAE/regret标为不适用，而不是填0。

这是包含已知异常宏观背景的固定诊断probe，不是新留出泛化集。下表为raw absolute residual MAE；逐点CSV同时保留production scales，可重算normalized residual。9×9网格上的boundary roughness仅是粗网格诊断，不声称获得高精度边界。

| eta | 指标MAE | Before | After |
| --- | --- | ---: | ---: |
| 0 | P0 residual | 0.584445 | 0.105081 |
| 0 | PI residual | 0.712705 | 0.087415 |
| 0 | Q residual | 0.224335 | 0.114218 |
| 1 | P0 residual | 0.873960 | 0.297044 |
| 1 | PI residual | 1.029274 | 0.299410 |
| 1 | Q residual | 0.241609 | 0.183997 |
| 1 | BP0 regret | 0.116215 | 0.129216 |
| 1 | BPI regret | 0.114276 | 0.162459 |

P/Q改善，但BP regret恶化；不是全面收敛。default面积eta0：0.830247→0.789095，eta1：0.812757→0.601852。单个背景可不同，不能把面积越小当越正确。

四组Before/After×eta0/1每个背景均同轴、同色、同真实初始scatter。黑色=current checkpoint boundary，未画previous-period boundary。另有Before黑线/After红线直接叠加。保存方向反转、多交叉、离散总变差、Value尺度，以及原i-grid上全部PI−P0变号bracket；未假装得到唯一精确i-star。

## Off-support风险

认证下载的Outer11 D2训练记录16384行中，合法FC1行14067；其Parent h/ell轴包络约[-1.415235,-0.752838]×[3.806767,4.442607]。最终网格91.6667%在该观测轴包络之外；这只是与一批真实训练输入的距离，不证明包络内均有joint support，也不是整个SDF可信域。

同一网格raw M min≈9.38e-9，P90≈62299.7，max≈248663.5；原P-value clamp触发91.6667%。target均finite，但不能据此认定Macro在宽域可靠。本轮不改变clamp、输入范围或objective。

## 相同4初始经济体的冻结H50

共同初始微观hash、Child0/1 CRN、Entry冲击和Exit-v2相同；两模型分别严格求h-star（最大残差<1e-5），不是共用不再合法的根。

| t | Before可推进主干 | After可推进主干 |
| --- | ---: | ---: |
| 1 | 4 | 4 |
| 5 | 2 | 3 |
| 10 | 1 | 3 |
| 20 | 1 | 3 |
| 30 | 1 | 2 |
| 50 | 1 | 1 |

t0 hard default 662/800→571/800；合法FC1行111→217；完整Pair54→108。实际K/C、信念差、Entry及last_valid逐时CSV均保留。t50没有最终路径数改善；n=4且不是总体随机抽样，不能外推。

贡献合法数据的initial lineage两者均3条，但ESS从1.219780增至2.744594；Middle独立lineage从1增至3，Late从1增至2（Late ESS仅1.511331）。因此确有中期贡献分散改善，仍不能把217行当217个独立路径，也不能说长期coverage已解决。

CPU最终49.42秒，峰值RSS0.710GiB；无新optimizer/backward，FC2=0，GPU性能/经济收敛NOT_TESTED。实验生成源码b0d8d73，后续只读指标完善与报告提交不能冒称生成commit。

## 仍未识别的因果问题

没有“原分布再训练3inner”的B控制checkpoint；故H0额外训练步数与H1 Uniform覆盖的贡献不可分。已准备A=0额外inner、B=原分布3inner、C=Uniform3inner的同预算/同donor/同seed/同optimizer初始化接口，但本轮没有训练B/C。

静态同状态变化支持存在Firm直接输出差异；rollout同时改变初始Root与后续宏观路径。因此“减少不合理Default”与“避开危险宏观区域”的定量分解仍NOT_TESTED。
