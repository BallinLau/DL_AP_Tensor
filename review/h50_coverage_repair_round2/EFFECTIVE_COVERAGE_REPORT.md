# Phase F — Effective macro coverage

原始经济数据保持f48b2e2生成版本，本轮只做CPU流式scalar重聚合。invalid终止suffix缺少的数值列补NaN以通过统一审计schema；严格select_training_rows不接受其成为标签。

| Parent-time段 | Child行 | 独立初始lineage | lineage ESS | 完整SDF Pair | 占joint bins |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全体 | 197 | 18 | 3.585127 | 95 | 27（含方向overflow bins） |
| Early 0–9 | 117 | 18 | 11.174694 | 55 | 26 |
| Middle 10–29 | 40 | 1 | 1 | 20 | 2 |
| Late 30–49 | 40 | 1 | 1 | 20 | 1 |

ESS=(Σw)²/Σw²，w为每条初始lineage贡献的Child行数，不把同Parent-time两个Child叫作两个独立Parent。最大lineage100/197=50.7614%，top5=74.1117%，来源只有Beta，source entropy=0。

全体贡献lineage的last_valid_t mean/median=5.6667/3；这是“至少贡献过合法FC1行”的18条lineage口径，不等于全部29条初始路径。Middle/Late唯一lineage的last_valid_t=50（右删失）。多个晚期bins不能替代独立lineage数量。

原二维overflow23/197：新分类H_IN/ELL_LOW=21，H_HIGH/ELL_LOW=2，其余174为H_IN/ELL_IN。原fixed edges未扩张，NaN未入bin；没有把低K挤回原区间。

输出按time/bin/family/source/time-source/时间段分组，同时保存raw rows、unique lineages、ESS、largest/top5 share、source entropy、完整Pair和last-valid统计。`source_event_table`另按outer/stage/split/t/source_type保留主干、hard default、Entry与child/pair数量；仅reductions，无额外NN forward。

CONFIRMED_BY_FROZEN_SAMPLE：中晚期覆盖不足且由单一路径支撑。不能用197行宣称接近197个独立宏观路径。NOT_TESTED：8192规模的最终有效覆盖。
