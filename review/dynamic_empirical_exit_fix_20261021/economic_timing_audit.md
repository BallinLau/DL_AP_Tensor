# 退出与 Entry 时序契约

已核对函数：_process_node_batched、_expand_branches_batched、_apply_exit_batched、_apply_value_cost_entry_batched、compute_entry_reference、capital_growth_decomposition、apply_refinancing_policy。

| 时点 | 参与集合/数据 | 不变项 |
|---|---|---|
| Parent 节点账本 | 当期入场/存活集合，含当期稍后退出者 | 当期资源与维护、违约清算规则 |
| 锁定历史/展开 | 原 Parent K/alive/ID | Entry reference 仍为转移前规模 |
| b/K/冲击转移 | 原 eta_t 与 bar_i 公式、稳定身份冲击 | 不使用 eta_child 门控旧融资 |
| Child Exit | Parent bar_z 尚未覆盖 | 原 <0.5 阈值，default mass清零 |
| Child Entry | 保存的 transition_previous_K/alive | PI-P0条件、候选冲击、规模不改 |
| pre-decision capture | Child已退出+已入场，但未作当前政策 | 不用 Child bar_z/C/valid筛模板 |
| Child 当前政策/账本 | 不含上一期 default，含新entrant | entry仅一次、无floor |

Entry reference 与 Child actual population 有意不同，不能用两者数量不同就判会计失配。它们各自的口径原代码已明确；修正保留这些口径。

investment_compare 出生状态仍 b=0、eta=0、i=creation cost、K=K_birth；出生期 bar_i_executed=0，维护豁免与原资源账本保持不变。测试同时覆盖 value_cost 和 investment_compare 的生产函数。

历史 raw expansion 仍保留，以便复现 bcb2c5a 的旧反例；正式 finite-direct adapter 现在通过 expand_after_parent_exit 调用修正版。未修改其他模拟模式或 continuum/root solver。
