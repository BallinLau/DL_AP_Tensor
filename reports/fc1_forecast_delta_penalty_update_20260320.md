# FC1 forecast-state 单步增量约束修改报告（2026-03-20）

## 修改目的

在前一轮诊断中，确认了 child-state `M` 分布异常的核心不是画图口径，而是 `forecast-state` 的一步递推映射过于激进：

- 某些区域出现 `Δhatcf < 0`
- 某些区域出现过大的 `Δlnkf`
- 从一个平滑的 parent 分布映射出了多个 child regime

本次修改的目标是：

- 不直接硬编码经济方向
- 只限制单步 forecast-state 递推的“幅度过大”
- 降低 child `M` 分布被一步映射撕裂的概率

## 设计原则

不采用“高 `hatcf_prev` 不允许负 `Δhatcf`”或“低 `lnkf_prev` 不允许大 `Δlnkf`”这种区域性强先验。

改为更弱的幅度约束：

```math
\max(0, |\Delta \hat c| - \bar d_c)^2

\max(0, |\Delta \ln K| - \bar d_k)^2
```

含义：

- 小幅度递推不受惩罚
- 只有单步跳跃超过阈值时才惩罚
- 不直接规定方向，只抑制过强的 local response

## 代码修改

### 1. 新增超参数

文件：

- [config/hyperparams.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/config/hyperparams.py)

新增：

- 初始版本：
  - `fc1_delta_penalty_weight = 1.0`
  - `fc1_delta_hatc_abs_max = 0.75`
  - `fc1_delta_lnk_abs_max = 0.35`
- 调强后的版本：
  - `fc1_delta_penalty_weight = 10.0`
  - `fc1_delta_hatc_abs_max = 0.50`
  - `fc1_delta_lnk_abs_max = 0.30`

解释：

- `Hatc` 单步绝对变动超过 `0.75` 才开始罚
- `LnK` 单步绝对变动超过 `0.35` 才开始罚
- `LnK` 阈值更小，因为此前诊断显示 `LnK` 更容易主导 `M` 的分裂

#### 为什么选 `0.75` 和 `0.35`

这些阈值不是理论常数，而是基于当前 `ep3` 诊断结果给出的 first-pass calibration。

对 [ep3_stage_modea_macro.pkl](/Users/ballinliu/Desktop/ep3_stage_modea_macro.pkl) 的 child macro states 分组后，发现：

- 低 `M` 组：
  - `Δlnkf ≈ 0.84`
  - `Δhatcf ≈ -0.21`
- 中间组：
  - `Δlnkf ≈ 0.34`
  - `Δhatcf ≈ 0.82`
- 高 `M` 组：
  - `Δlnkf ≈ 0.16`
  - `Δhatcf ≈ 1.60`

因此：

- `0.35` 大致对应 `Δlnkf` 从“中间区域”进入“低 `M` 危险区域”的分界附近  
  也就是先允许常规波动，但开始抑制会把系统推向低 `M` regime 的较大资本跳跃。

- `0.75` 大致对应 `Δhatcf` 从“小幅递推”进入“明显 regime 跳跃”的边界附近  
  目标不是压制正常小波动，而是抑制 `0.8~1.8` 这一类把系统快速送进高 `M`/低 `M` 极端区域的大步跳跃。

从响应面扫描看，这两个阈值也与局部异常区间吻合：

- `low M` 区域常伴随 `Δhatcf < 0` 且 `Δlnkf` 偏大
- `high M` 区域常伴随很大的正 `Δhatcf`

因此，初始版阈值的设计原则是：

1. 不压正常小波动
2. 只抑制明显过猛的一步跳跃
3. 先把 child-state 的 regime splitting 压下来，再看是否需要更细的局部平滑约束

如果后续希望进一步减少主观性，可以把这两个阈值改成：

- teacher-forcing 阶段 `Δhatcf` 的经验分位数（如 p90）
- teacher-forcing 阶段 `Δlnkf` 的经验分位数（如 p90）

这样可以把当前手工阈值替换为完全数据驱动的阈值。

## 为什么又进一步调强

在加入初始版增量惩罚后，检查新日志 [dl_tensor_high_mem_output.278534](/Users/ballinliu/Desktop/dl_tensor_high_mem_output.278534) 发现：

- `sdf_delta_penalty` 已经不是 0，说明罚项确实触发了
- 但 `child M` 分布仍然没有明显改善

具体地，`stage2` 末期日志显示：

- `sdf_delta_penalty ≈ 0.185`
- `sdf_delta_penalty_hatc ≈ 0.178`
- `sdf_delta_penalty_lnk ≈ 0.026`

同时：

- `sdf_moment_loss ≈ 1.825`
- `sdf_moment_weight = 5.0`

这意味着 moment 相关项对总损失的贡献大约在 `9` 左右，而增量惩罚只有 `0.185 × 1.0` 的量级。  
换句话说，罚项虽然生效了，但被更强的 SDF 矩约束项淹没，难以真正改变递推映射。

从同一份日志看，极端一步跳跃也仍然存在：

- `sdf_dhatcf_p90 ≈ 1.51`
- `sdf_dlnkf_p90 ≈ 0.62`

而初始阈值只设在：

- `|Δhatcf| <= 0.75`
- `|Δlnkf| <= 0.35`

说明：

1. 超阈值现象很多
2. 但原先的惩罚权重不够大
3. 仅靠初始版参数，不足以在训练中与 moment/HJ 项竞争梯度

因此本次把增量约束进一步调强为：

- `fc1_delta_penalty_weight = 10.0`
- `fc1_delta_hatc_abs_max = 0.50`
- `fc1_delta_lnk_abs_max = 0.30`

调整逻辑是：

1. **提高权重**：让这条约束真正进入主导梯度量级，而不是只做边缘修正
2. **收紧 `Hatc` 阈值**：因为日志显示当前主要超标的是 `Hatc`，不是 `LnK`
3. **小幅收紧 `LnK` 阈值**：继续压制会拉坏 `M` 指数项的资本一步跳跃

这次调强的目标不是追求“立刻得到完美分布”，而是先验证：

- 增量惩罚在足够强时，是否真的能改变 child-state 递推形状

如果调强后仍然无法明显改善 `M` 分布，那么下一步就不该继续只调参数，而应该上更结构性的响应面平滑约束。

### 2. 在 `stage2` 里对 forecast-state 输出加入增量惩罚

文件：

- [training/episode.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/training/episode.py)

实现位置：

- 在 `forecast-state` 重建项计算之后，基于
  - `c_children_forecast - parent[:, 5:6]`
  - `k_children_forecast - parent[:, 6:7]`
  构造 `Δhatcf` 和 `Δlnkf`

惩罚项定义为：

```python
relu(abs(Δhatcf) - hatc_max)^2
relu(abs(Δlnkf) - lnk_max)^2
```

再按现有内部权重聚合：

```python
delta_penalty =
    fc1_hatc_recon_weight * delta_penalty_hatc
  + fc1_lnk_recon_weight  * delta_penalty_lnk
```

并加入总损失：

```python
+ fc1_delta_penalty_weight * delta_penalty
```

### 3. 新增日志项

现在训练日志会额外记录：

- `sdf_delta_penalty`
- `sdf_delta_penalty_hatc`
- `sdf_delta_penalty_lnk`
- `sdf_delta_penalty_weight`
- `sdf_delta_hatc_abs_max`
- `sdf_delta_lnk_abs_max`

这样可以直接判断：

- 惩罚是否真的在起作用
- 主要是 `Hatc` 还是 `LnK` 触发了罚项

### 4. 同步运行入口默认值

文件：

- [experiments/run_utils.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/experiments/run_utils.py)
- [experiments/run_episode0_full.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/experiments/run_episode0_full.py)

避免不同入口覆盖回旧口径。

## 预期效果

本次修改不改变：

- `M` 的定义
- `forecast-state` 递推主路径
- `Hatc/LnK` 重建目标

本次修改只是在 `forecast-state` 下一步预测上增加“过大跳跃”的软惩罚。

预期看到的变化：

1. `sdf_dhatcf_p90` 和 `sdf_dlnkf_p90` 回落
2. `child macro states` 的 `M` 直方图不再那么容易裂成极低值和厚右尾
3. `FC1_K` 对 `M` 分布的主导作用减弱

## 建议下一轮重点观察

1. 日志

- `sdf_delta_penalty`
- `sdf_delta_penalty_hatc`
- `sdf_delta_penalty_lnk`
- `sdf_dhatcf_mean / p90`
- `sdf_dlnkf_mean / p90`

2. 图形

- `child macro states` 的 `M` 直方图
- `parent macro states` 的 `M` 直方图

3. 若仍然异常

如果 `M` 分布仍明显双峰或长尾，下一步再考虑加入更强的局部平滑项，例如 Jacobian / Lipschitz penalty，而不是马上上区域性符号约束。

## 后续追加：Jacobian / 局部平滑约束

在继续实验后，观察到：

- 调强后的增量惩罚确实让 `child M` 分布有所改善
- 但改善仍不足，分布依然呈现混合形态

因此进一步追加了 `forecast-state` 的 Jacobian penalty，用来约束 `FC1` 响应面的局部陡峭度。

### 新增超参数

文件：

- [config/hyperparams.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/config/hyperparams.py)

新增：

- `fc1_jacobian_penalty_weight = 1.0`

### 约束形式

在 forecast-state 输入下，记：

```math
\Delta \hat c = F_c(x_t, x_{t+1}, \hat c_t^f, \ln K_t^f)
```

```math
\Delta \ln K = F_k(x_t, x_{t+1}, \hat c_t^f, \ln K_t^f)
```

则 Jacobian penalty 近似为：

```math
\mathcal L_{\text{jac}}^c
=
\left\|\frac{\partial \Delta \hat c}{\partial \hat c_t^f}\right\|^2
+
\left\|\frac{\partial \Delta \hat c}{\partial \ln K_t^f}\right\|^2
```

```math
\mathcal L_{\text{jac}}^k
=
\left\|\frac{\partial \Delta \ln K}{\partial \hat c_t^f}\right\|^2
+
\left\|\frac{\partial \Delta \ln K}{\partial \ln K_t^f}\right\|^2
```

总项为：

```math
\mathcal L_{\text{jac}}
=
\omega_c \mathcal L_{\text{jac}}^c
+
\omega_k \mathcal L_{\text{jac}}^k
```

其中当前沿用：

- `ω_c = fc1_hatc_recon_weight = 1.0`
- `ω_k = fc1_lnk_recon_weight = 0.25`

### 经济解释

这条约束并不规定：

- 哪个区域 `Δhatcf` 必须为正
- 哪个区域 `Δlnkf` 必须为小

它只要求：

- `FC1` 对当前 forecast state 的局部响应不要过于敏感

如果 parent state 只发生很小扰动，child state 不应被立刻送进完全不同的 regime。

### 为什么需要它

前面的增量惩罚解决的是：

- “一步跳得太远”

而 Jacobian penalty 解决的是：

- “即使跳得没有特别远，但响应面太陡，仍然会把相邻 parent state 映射到不同 child regime”

因此，这两条约束是互补的：

1. `delta penalty` 管幅度
2. `jacobian penalty` 管局部形状
