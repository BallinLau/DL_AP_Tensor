# PVBP 阶段改为 Value/Policy 交替训练

日期：2026-04-01

## 背景

在 `Q stage -> PVBP stage` 的两阶段训练里，最新结果暴露出两个稳定问题：

1. `bp` 头没有跟 `V0/VI` 的 value 面对齐。
   - 典型现象：`bp*` 明显偏离 `argmax V0 / argmax VI`
   - log 中 `bp_value_target_mean` 与 `bp_value_pred_mean` 长期相差很大

2. `PVBP` 内部把 value 学习与 policy 学习混在同一个更新里。
   - value/gate 面还没稳定时，`bp0/bpI` 就在跟随一个不断变化的目标
   - 结果容易出现高杠杆区 policy 偏高、`bp0` 与 `bpI` 不分化、policy 与 Bellman 面错位

## 修改思路

不再把 `PVBP` 当成一个单块同步更新，而是把每个 PVBP epoch 拆成两个 pass：

1. `value-only pass`
   - 只更新 `pvbp_model` 中除 `bp0_head / bpI_head` 外的参数
   - 目标：先把 `V0 / VI / chi / bar_z / P` 的 value/gate 面压稳

2. `policy-only pass`
   - 只更新 `bp0_head / bpI_head`
   - 目标：在固定的 value/gate 面上，让 policy 头去跟随 Bellman/FOC/KKT/value target

## 实现位置

- `training/episode.py`
  - 新增并使用 `_set_policy_value_only_freeze(...)`
  - 在 `_run_batches(...)` 中，当进入 `PVBP stage` 时，按 `value pass -> policy pass` 顺序运行

- `config/hyperparams.py`
  - 新增：
    - `pvbp_alternating_enabled`
    - `pvbp_value_steps_per_epoch`
    - `pvbp_policy_steps_per_epoch`

## 默认设置

- `pvbp_alternating_enabled = True`
- `pvbp_value_steps_per_epoch = 1`
- `pvbp_policy_steps_per_epoch = 1`

即每个 PVBP epoch 默认做：

1. 一次完整 `value-only` pass
2. 一次完整 `policy-only` pass

## 预期效果

如果这一改动有效，应该看到：

1. `bp_value_target_mean` 和 `bp_value_pred_mean` 更接近
2. `bp*` 靠近 `argmax V0 / argmax VI`
3. `bp0` 与 `bpI` 重新出现有经济意义的分化
4. `bp_diag_safe` 中 `bp*` 不再长期卡在 `P(bp)` 掉崖附近

## 暂不处理

这一轮只改 `PVBP` 内部训练顺序，不同时引入：

- 新的 `Q` refresh stage
- 新的 `Q` 额外形状约束
- 额外的 `bp` 目标函数重构

目的是先单独识别：`bp` 异常有多少是由“policy/value 同步训练”造成的。
