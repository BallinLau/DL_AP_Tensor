# FC1 `LnK` 重建降权修改记录（2026-03-20）

## 背景

在 `df_macro` 的 child-state `M` 分布诊断中，发现 `M` 不是围绕 1 附近的单峰分布，而是表现为：

- 大量样本堆在低 `M` 区
- 一部分样本停在 1 附近
- 右侧还有明显长尾

对 [ep3_stage_modea_macro.pkl](/Users/ballinliu/Desktop/ep3_stage_modea_macro.pkl) 的分组分析显示，`M` 的主要分层变量不是 `Hatc`，而是 `LnK`：

- `corr(M, Δlnkf) ≈ -0.71`
- `corr(M, Δhatcf)` 很弱

这说明当前 child `M` 分布异常，更像是 `FC1_K / LnK` 递推过强导致的 regime splitting。

## 问题定位

`stage2` 的 FC1 重建项此前写成：

```python
(hatcf_pred - hatcf_true)^2 + (lnkf_pred - lnkf_true)^2
```

并且 FC1 已经不再使用 scaler，直接在 physical scale 上训练。这样会带来两个后果：

1. `Hatc` 与 `LnK` 的重建误差在原始尺度上直接相加。
2. 由于 `LnK` 的波动范围通常大于 `Hatc`，`LnK` 项会在训练中自然占优。

结果是 `FC1_K` 更容易主导 `stage2` 的闭环拟合，并通过

```math
M \propto \exp(-4 \Delta \ln K + 3 \Delta \hat c)
```

把 child-state `M` 分布拉坏。

## 本次修改

### 1. 在超参数中引入分目标内部权重

文件：

- [config/hyperparams.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/config/hyperparams.py)

新增：

- `fc1_hatc_recon_weight = 1.0`
- `fc1_lnk_recon_weight = 0.25`

含义：

- `Hatc` 重建项保持基准权重
- `LnK` 重建项在 FC1 重建内部降权到 `0.25`

### 2. 在 `stage2` 中拆分 `Hatc/LnK` 重建损失

文件：

- [training/episode.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/training/episode.py)

修改后：

- `recon_loss_hatc = mse(Hatcf_pred, Hatc_true)`
- `recon_loss_lnk = mse(LnKF_pred, LnK_true)`
- `recon_loss = w_hatc * recon_loss_hatc + w_lnk * recon_loss_lnk`

forecast-state 重建同样拆开：

- `recon_loss_forecast_hatc`
- `recon_loss_forecast_lnk`
- `recon_loss_forecast = w_hatc * ... + w_lnk * ...`

### 3. 补充日志项

现在会额外记录：

- `sdf_recon_loss_hatc`
- `sdf_recon_loss_lnk`
- `sdf_recon_loss_forecast_hatc`
- `sdf_recon_loss_forecast_lnk`
- `sdf_hatc_recon_inner_weight`
- `sdf_lnk_recon_inner_weight`

这样后续可以直接判断：

- 是不是 `LnK` 项仍然在压制 `Hatc`
- 降权后 child `M` 分布有没有明显收敛

### 4. 同步运行入口默认值

文件：

- [experiments/run_utils.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/experiments/run_utils.py)
- [experiments/run_episode0_full.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/experiments/run_episode0_full.py)

避免不同入口把默认值覆盖回旧口径。

## 预期影响

本次修改不改变：

- `stage2` 仍以 forecast-state 闭环监督为主
- `M` 仍来自 `state["hatcf"] / state["lnkf"]` 的 forecast-state 递推

本次修改只改变：

- FC1 重建项内部 `Hatc` 和 `LnK` 的相对梯度权重

预期结果：

1. `FC1_K` 对递推的主导作用下降。
2. child-state `M` 的低值堆积和长尾应有所收敛。
3. 训练日志里 `sdf_recon_loss_lnk` 应仍较大，但其对总 FC1 重建项的影响会减弱。

## 建议后续重点观察

下一轮训练建议优先看：

- `sdf_recon_loss_hatc`
- `sdf_recon_loss_lnk`
- `sdf_recon_loss_forecast_hatc`
- `sdf_recon_loss_forecast_lnk`
- `df_macro` child-only `M` 直方图
- `Δlnkf` 与 `M` 的相关性

如果 child `M` 分布仍然明显双峰/长尾，再继续考虑：

1. 对 `LnK` 重建做标准差归一化，而不只是手工降权。
2. 在 `FC1_K` 上单独加更强的递推稳定约束。
