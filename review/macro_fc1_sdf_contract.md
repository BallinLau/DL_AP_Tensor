# Macro / FC1 / SDF

每个尝试Parent保留两条child审计记录，包括因parent失效而无法模拟的child。
固定初始bank、稳定经济身份键、branch0/1条件独立AR1冲击；不重抽直到成功。
生产entry、维护、资本/债务转移与资源账本不改；没有floor，没有root/FAST correction。

FC1输入：(x_t,x_next,h_cal_t,lnK_cal_t)，目标：(h_cal_next,lnK_cal_next)。
合法单child可训练，完整pair不是FC1前提。
adapter校验h=log(C_raw/K)、lnK=log(K)，相同parent的兄弟true context完全一致。

SDF要求完整有效parent/child0/child1，不能跨parent拼接，不能复制child。
parent true列7/8，child true列8/9；实际production TRUE_ONLY forward读取前者。
child M第7列仅macro layout占位，实际M由当前SDF forward计算；真实损失测试用NaN置换该列确认无消费。
parent/child forecast由真实模拟记录提供；历史仅布局测试缺forecast时的NaN不是生产输入。
真实TRUE_ONLY测试还将parent forecast改NaN确认训练损失只使用true context，不依赖占位值。
生产验证仍使用真实forecast记录，不能向全模式诊断喂伪零/NaN。

parent_index为完整pair的连续索引，val offset在train之后，避免fresh-bank身份混用。
每epoch先按train+val容量准备同一shock bank，原fresh条件抽样规则不改。
固定模拟pair、fresh wealth pair和pooled AL估计来源分别记录，不能把企业行数当宏观样本数。
AL按production约束公式每5个primal epoch更新，两个chunk连续，不用旧strict gate中止四轮预算。

所有失败行保留，coverage不只展示成功子集R²。
无合法FC1行、无完整SDF pair、非有限优化状态为硬错误；低R²/非零G/未收敛不是硬错误。
