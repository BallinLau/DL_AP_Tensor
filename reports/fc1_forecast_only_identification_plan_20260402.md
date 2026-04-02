# FC1 纯 Forecast 识别实验计划

## 目的

当前 `macro_hatc` 一致性很差，还不能区分：

1. `FC1` 的状态变量不充分；
2. `FC1` 被 `SDF/Euler/moment/anchor` 等联合目标拖坏；
3. 两者同时存在。

第一步先做最小识别实验：只保留 `FC1` 的 forecast / law consistency 目标，关掉 `SDF` 主方程相关项。

## 本轮修改

新增超参数：

- `fc1_forecast_only_ablation`

当该开关为 `True` 时：

1. `SDF/FC1(stage2)` 里不再优化：
   - `main_loss`
   - `moment_loss`
   - `mean_anchor_loss`
2. 只保留：
   - `recon_loss`
   - `law_consistency_loss`
   - `x_response_loss`
   - `delta_penalty`
   - `jacobian_penalty`
3. `stage2` 也只更新 `fc1_model`，不更新 `sdf_model` / `value_model`

## 识别逻辑

如果在这个 ablation 下，`macro_hatc` 明显改善，说明：

- `FC1` 之前很大程度上是被联合训练目标冲突拖坏；

如果改善仍然很有限，说明：

- 当前 `FC1` state 本身不足，或者 `Hatc` 标签/聚合路径仍有问题。

## 重点观察指标

- `sdf_law_consistency_loss_hatc`
- `sdf_x_response_loss_hatc`
- `macro_hatc_branch01`
- `macro_hatc_vs_x`
- `corr_hatc`
- `slope_hatc`
- `std_ratio_hatc`

## 建议运行方式

建议直接在现有 runner 中加：

```bash
--fc1-forecast-only-ablation
```

先做短程对照，不和其它结构修改混在一起。
