# Project Structure

- `analysis/`: Python/PyTorch研究执行器、事务与诊断。
- `data/`: finite-direct采样/真实生产模拟适配、账本、transition。
- `training/`, `losses/`: P/Q、BP teacher、FC1、Signed AiO/AL生产实现。
- `tests/`: pytest非生产优化测试；`slurm/`:未来THU部署脚本。

# Core Modules

`data/finite_direct_runtime.py`只负责观察、缓存、Firm bank连接，实际经济计算仍在SimulateTS primitives。
`analysis/finite_direct_outer4_state.py`负责原子block提交，`nested_equilibrium_utils`负责完整训练事务内容。
`training/episode.py`与existing frozen-policy/Q求解器负责损失与优化，不复制其数学算法。

# Entry Files

- `analysis/run_full_nested_finite_direct_outer4.py`: 默认plan-only；明确批准后才执行。
- `analysis/prepare_finite_direct_outer4_review.py`: CPU采样/权重组合，始终不训练。
- `slurm/run_full_nested_finite_direct_outer4_a800.slurm`:未来单A80080GB运行配置，无自身提交动作。

# Dependency Relationships

```text
new main → load_locked_sampling → template/schedule/bank SHA validation
         → ProductionBackend → prior FC1 setup (no old targets)
         → StageJournal → drive → ProductionBackend.execute

Firm source → generate_dataset → simulator / process_node
                               → _apply_entry_batched
                               → _process_node_batched (production ledger + Grid)
                               → _expand_branches_batched
            → capture_node → firm_bank_from_captures (same-node rows)
            → conditional_child_bank → build_child_exogenous_bundle(true context)
P → existing _p_policy_outer → build_problem → _solve_frozen_policy
Q → existing _q_picard_outer → build_q_survival_problem → run_q_adam
BP → Episode._build_bp_target_cache → Episode._run_bp_distillation_stage

D0 → production FC1 loss → D1 → production FC1 loss → D2
   → select_training_rows → production_sdf_tensors → sdf_batches
   → Episode._run_batches → train_step → _compute_sdf_loss(TRUE_ONLY)
   → _estimate_sdf_al_constraints → _update_sdf_al_duals
   → fixed-probe P/Q cross evaluation → committed Outer → next Outer
```

不复用旧full runner的Boundary/root macro-bank构造与candidate gate。
不复用 `run_macro_outer_inner_pq.refresh_transition_bank` 的perceived-parent默认上下文：本任务要求显式同节点true context。
不复用 `run_macro_block_isolation.run_fc1_fit` 的best-weight/last-optimizer潜在组合；只复用生产FC1 loss并用last matched state。
SDF validated-phase外层含gate early stop/resetAL，因此复用其生产损失、batch driver、validator和dual函数，而不复用该停止策略。
通用collocation sampler会混边际，不适合真实同节点Firm bank；使用InnerFP数据结构和其底层conditional transition。

依赖：PyTorch、NumPy、Pandas、Matplotlib、pytest；运行环境沿用THU DL_HL。
现有analysis执行器之间import耦合较重，新入口有意只复用明确列出的函数，不改旧runner的模式分派。
