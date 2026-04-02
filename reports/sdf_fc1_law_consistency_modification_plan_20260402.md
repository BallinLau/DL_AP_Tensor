# `sdf_fc1` Aggregate Law Consistency 修改计划

日期：2026-04-02

## 背景

当前 `macro_hatc` 诊断的含义是：

- 固定当前 `sdf_fc1` 给出的 aggregate consumption law
- 在该 law 下解 `policy/value`
- 再用 simulate implied 的 aggregate `Hatc`
- 检查它是否能回到这条 law

因此，这里真正想检验的是：

- `Hatcf_{pred}` 与 `Hatc_{sim}` 的 **aggregate law consistency**

而不是单纯 one-step forecast 的统计拟合。

## 当前问题

`_compute_sdf_loss()` 里虽然已经有：

- true-state recon
- forecast-state recon
- delta penalty
- jacobian penalty

但还缺少一个**被明确命名并单独监控**的 `aggregate law consistency` 目标。

这会带来两个问题：

1. 训练时看不清 `Hatc/LnK law consistency` 到底有没有改善。
2. 后续要加 `x-response consistency` 时，缺少一个清晰的主损失挂载点。

## 修改总路线

### 第一步：先拆出显式 `law consistency loss`

在 forecast-state 闭环口径下，显式定义：

- `L_hatc_law = E[(Hatcf_pred - Hatc_true)^2]`
- `L_lnk_law = E[(LnKF_pred - LnK_true)^2]`

并新增超参数：

- `fc1_hatc_law_consistency_weight`
- `fc1_lnk_law_consistency_weight`

再定义：

- `L_law = w_hatc * L_hatc_law + w_lnk * L_lnk_law`

然后把现有 `forecast-state recon` 明确解释成：

- `aggregate law consistency loss`

### 第二步：再加 `x-response consistency`

后续会按 `x` 分 bin，比对：

- `E[Hatcf_pred | x-bin]`
- `E[Hatc_true | x-bin]`

以及 `LnK` 的对应响应。

这一步的目标不是先追逐逐点 MSE，而是先把 conditional response 形状对齐。

### 第三步：最后才决定是否下调 `delta penalty`

只有在：

- 已经显式加入 law consistency
- 已经显式加入 x-response consistency

之后，如果 `hatcf` 仍然明显被压错形状，才去下调：

- `fc1_delta_penalty_weight`

## 本次实际落地内容

本次只做第一步：

1. 在 `_compute_sdf_loss()` 中显式拆出：
   - `sdf_law_consistency_loss_hatc`
   - `sdf_law_consistency_loss_lnk`
   - `sdf_law_consistency_loss`
2. 在 `hyperparams.py` 中新增对应权重：
   - `fc1_hatc_law_consistency_weight`
   - `fc1_lnk_law_consistency_weight`
3. 暂时不改：
   - `x-response consistency`
   - `delta penalty`
   - `jacobian penalty`

## 这一步之后要看什么

下一轮实验应重点看：

- `sdf_law_consistency_loss_hatc`
- `sdf_law_consistency_loss_lnk`
- `macro_hatc_branch01`
- `macro_hatc_vs_x`

判据不是只看 `R²`，而要同时看：

- `corr`
- `slope`
- `std_ratio`
- `x-response` 是否更接近
