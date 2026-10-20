# 经济时序审查：FAIL / BLOCKED

直接证据：`timing_reproduction.json`；测试 `tests/test_dynamic_empirical_timing_blocker.py`。

实际链：Parent policy/accounting → Parent bar_z 写回（alive 不变）→ expand（复制 alive、按 eta_t 写 realized b、按 bar_i 转资本）→ Child entry → Child policy/accounting（覆盖 bar_z）→ capture。

所需链：上一期违约与真正退出 → 合法融资/资本/冲击转移 → 正确入场 → 当前期政策前捕获。退出放置点涉及生产经济时序，未经用户确认不能在本任务直接改写。

`_process_node_batched()` 的 endpoint_alive 只供分解诊断，不是已提交的状态更新。`_apply_exit_batched()` 使用调用时当前 bar_z；如果在 Child 决策后调用，就会筛 Child 当期结果，不能逆充 Parent 的退出。

投资比较 entry 的 realized b=0、K=候选 K_birth、eta=0、i=cost，保留原规则；其正确纳入新 capture 的测试尚未实现。

已有 capture 的 eligible 来源是当前 node 的 economic_state_valid，是事后账本信息；既不能以它选择新模板，也不能只根据 alive 认证 post-exit。尚无可认证的动态 pool。
