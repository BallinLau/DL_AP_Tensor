# Macro Parent One-Step 训练流审计

## 1. Overview

本报告追踪 `analysis/run_full_nested_equilibrium.py` 中 Macro conditional solve 的数据、损失和优化路径。已确认本补丁只替换 Macro primary dataset 的生成方式；FC1 loss、SDF loss、网络结构和 optimizer sink 均继续走生产实现。

## 2. Entry Point

- 文件：`analysis/run_full_nested_equilibrium.py`
- 入口：`main()`，MacroSolve 数据准备与主段位于 1691–2151 行。
- Macro inner：1793–2114 行。
- 职责：固定 Firm 后，按同一 source schedule 重建 parent bank，构造一步数据，依次训练 FC1 与 SDF，并比较函数、target 和 parent-bank movement。

## 3. Data Input Pipeline

### 3.1 数据源

- `run_short_rollout_harvest`：`analysis/macro_parent_state_bank.py:329-391`
  - direct call 到生产 `SimulateTS`；horizon 为 8（smoke 为 3）。
  - capture 位于生产 simulation loop 的 node processing 之前；实际调用点为 `data/simulate_ts_parallel.py:514-520`。
  - requested `[0,2,4,6,8]` 按 StateBank schema 约束映射为 effective `[1,2,4,6,7]`，两者均写 metadata。
- `prepare_macro_parent_schedule`：`analysis/macro_parent_state_bank.py:961-1058`
  - 从真实 short-rollout candidate pool 做 deterministic 80% endogenous-frequency + 20% coverage-aware selection。
- `harvest_macro_parent_schedule`：`analysis/macro_parent_state_bank.py:1061-1100`
  - 每个 macro inner 使用相同 initial seeds、shock seeds、path IDs、capture times 重新 harvest；只允许 Macro law 改变 endogenous state。

### 3.2 一步 transition 与 pairing

- `simulate_one_step_chunked`：`analysis/macro_parent_state_bank.py:710-754`
  - direct call 到生产 `SimulateTS(initial_state=stacked_state, horizon=1, branch_num=2)`。
  - parent economy 分块，OOM 顺序为 requested → 128 → 64。
- `build_one_step_macro_dataset`：`analysis/macro_parent_state_bank.py:801-856`
  - direct call 到 `Episode._build_sdf_pairs_from_macro_tensor`。
  - direct call 到 `Episode._create_sdf_batches_from_macro_tensor`。
- Pairing 生产定义：`training/episode.py:2524-2586`。
- Batch 生产定义：`training/episode.py:2588-2792`。
- `audit_one_step_pairing`：`analysis/macro_parent_state_bank.py:757-798`，assert 每个 parent 恰好对应 branch 0/1 两个 child，child local t=1。

### 3.3 下游变量

- FC1 parent tensor：`x_t, Hatcf_t, LnKF_t, Hatc_t, LnK_t`。
- FC1 child/target tensor：`x_t1, Hatc_t1, LnK_t1`。
- SDF group：同一个 parent 下两个独立 child branch。
- Long rollout：只写 `long_diagnostic_start/final`，不传入上述 primary batch。

## 4. High-Level Call Graph

```text
run_full_nested_equilibrium.main
  -> _build_or_refresh_macro_parent_context                 [direct call]
     -> make_initial_state                                  [direct call]
     -> prepare/harvest_macro_parent_schedule               [direct call]
        -> run_short_rollout_harvest                        [direct call]
           -> SimulateTS.simulate_tensor                    [direct call]
              -> StateBank.capture(parent-ready state)      [nested call]
     -> fixed source schedule + coverage artifacts
  -> build_one_step_macro_dataset                           [direct call]
     -> SimulateTS(horizon=1, branch_num=2)                 [direct call]
     -> Episode._build_sdf_pairs_from_macro_tensor          [direct call]
     -> Episode._create_sdf_batches_from_macro_tensor       [direct call]
  -> run_fc1_fit                                            [direct call]
     -> Episode._run_batches                                [direct call]
        -> Episode.train_step                               [nested call]
           -> Episode._compute_sdf_loss (FC1_ONLY)          [nested call]
           -> weighted total_loss -> backward -> step
  -> run_sdf_chunks                                         [direct call]
     -> Episode._run_sdf_phase_with_validation              [direct call]
        -> Episode._run_batches                             [nested call]
           -> Episode.train_step                            [nested call]
              -> Episode._compute_sdf_loss (SDF_TRUE_ONLY)  [nested call]
              -> weighted total_loss -> backward -> step
  -> same source schedule under Theta_(j+1)
  -> parent/target/function movement
```

## 5. Detailed Function Trace

### 5.1 `_build_or_refresh_macro_parent_context`

- 文件：`analysis/run_full_nested_equilibrium.py:903-1077`
- Called from：`main` at 1724。
- Edge：direct call。
- 输出：train/validation initial states、固定 source schedules、当前 selected parent banks。
- 不变量：validation schedule 在整个 run 固定；train schedule 每个 equilibrium outer refresh；source overlap 必须为 0。

### 5.2 `run_macro`

- 文件：`analysis/run_full_nested_equilibrium.py:1842-1966`
- Called from：Macro inner loop at 1793。
- Edge：nested function/direct call by OOM wrapper。
- 输入：当前一步 pair table。
- 输出：FC1/SDF summaries、`Theta_(j+1)` 下重新 harvest 的 parent banks 与一步 tables。

### 5.3 `run_fc1_fit`

- 文件：`analysis/run_macro_block_isolation.py:590-713`
- Called from：`run_macro` at 1871。
- Edge：direct call。
- 训练阶段：`FC1_ONLY`；SDF/value hash 必须不变。
- 监控：joint validation mean R2；Hatc R2 不作为整个 training 的硬终止条件。

### 5.4 `run_sdf_chunks`

- 文件：`analysis/run_macro_block_isolation.py:721-799`
- Called from：`run_macro` at 1880。
- Edge：direct call。
- 阶段：`SDF_TRUE_ONLY`；FC1 hash 必须不变；AL dual state 在 continuation chunks 间保留。

## 6. Loss Function Tracing

### 6.1 FC1 loss

- 入口：`Episode._compute_sdf_loss`，`training/episode.py:4023-4679`。
- 阶段分支：`FC1_ONLY` at 4050。
- 输入来自 production one-step batch，而非 100-period primary trajectory。
- loss aggregation 返回同一 `total_sdf_loss`；本补丁未改 loss 表达式或权重。

### 6.2 SDF true-only loss

- 入口：同一 `Episode._compute_sdf_loss`，`training/episode.py:4023-4679`。
- 阶段分支：`SDF_TRUE_ONLY` at 4064–4066。
- 主 loss/AL aggregation：`training/episode.py:4484-4497`。
- 两个 child branch 来自同一 production parent，且 pairing audit 在训练前硬检查。

## 7. Loss Aggregation and Optimization

- `zero_grad()`：`training/episode.py:3617-3620`。
- SDF/FC1 loss 加入总损失：`training/episode.py:3625-3633`。
- `total_loss.backward()`：`training/episode.py:3670-3672`。
- gradient protection：`training/episode.py:3740-3753`。
- `optimizer.step()`：`training/episode.py:3755-3766`。
- scheduler：`training/episode.py:3772-3782`。
- FC1 outer training loop：`training/episode.py:12150-12327`。
- validated SDF outer loop/rollback：`training/episode.py:14568-14980`，其中实际 primal epoch 在 14760–14770 nested-call `_run_batches`。

## 8. Unresolved / Ambiguous Points

- `macro-bank-candidate-multiplier` 记录为 protocol parameter；当前显式 candidate path counts 决定实际 pool 大小，未用于 synthetic 扩增。
- StateBank 不接受 fresh `t=0` 与 later-state schema 混存；因此有效 capture 从 `t=1` 开始。这是已验证的生产 schema 约束，不是经济时点的重定义。
- CPU smoke 不能给出 A800 peak GPU memory；必须由服务器 smoke 的 metadata/summary 填充。

## 9. Final Summary

- Entry point：full nested `main`。
- Primary data：many complete parent cross-sections × one-step two-branch transitions。
- Losses：原 FC1 loss + 原 SDF_TRUE_ONLY/AL loss。
- Optimizer：原 `Episode.train_step` optimizer/scheduler 路径。
- Long horizon：diagnostic only，未进入 primary training。

```text
fixed source schedule
-> production short rollout
-> full parent-ready state bank
-> production horizon=1, B=2
-> production pair/batch builders
-> unchanged FC1/SDF loss
-> weighted total_loss
-> backward
-> optimizer.step
```

## 10. Structured Trace Table

| Stage | Function | File | Lines | Called By | Edge | Returns | Used By |
|---|---|---|---:|---|---|---|---|
| Candidate | `run_short_rollout_harvest` | `analysis/macro_parent_state_bank.py` | 329–391 | context builder | direct | full StateBank + output | schedule/coverage |
| One-step | `simulate_one_step_chunked` | same | 710–754 | dataset builder | direct | macro table | pairing |
| Pairing | `_build_sdf_pairs_from_macro_tensor` | `training/episode.py` | 2524–2586 | dataset builder | direct | pair table | batch builder |
| Batch | `_create_sdf_batches_from_macro_tensor` | `training/episode.py` | 2588–2792 | dataset builder | direct | parent/children batches | FC1/SDF |
| FC1 | `run_fc1_fit` | `analysis/run_macro_block_isolation.py` | 590–713 | `run_macro` | direct | FC1 summary | movement/gate |
| SDF | `run_sdf_chunks` | same | 721–799 | `run_macro` | direct | SDF summary | movement/gate |
| Loss | `_compute_sdf_loss` | `training/episode.py` | 4023–4679 | `train_step` | nested | scalar | weighted total |
| Update | `train_step` | `training/episode.py` | 3569–3788 | `_run_batches` | nested | metrics | model state |

## 11. Optimization Topology

- FC1 stage updates：`FC1_C`, `FC1_K`；SDF/value frozen。
- SDF stage updates：SDF/value；FC1 frozen。
- Firm model：整个 MacroSolve hash frozen。
- FC2：forward guard count 必须保持 0，参数 hash 必须不变。
- Detach boundary：StateBank 保存 CPU clones；one-step output 在 chunk 完成后移到 CPU，再进入 production pair builder。
