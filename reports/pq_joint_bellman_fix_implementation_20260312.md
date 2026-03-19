# P/Q 联立贝尔曼修复落地报告（2026-03-12）

基于 `reports/pq_joint_bellman_diagnosis_20260312.md` 的代码改造已完成，本次重点覆盖：支持集、时点一致性、P/bar_z 梯度形态、CF 符号与 FOC 可诊断性。

## 1. 训练分布支持集修复（A）

### 已改动
- 新增可配置杠杆支持集参数：
  - `config/constants.py:57` `SIM_B_INIT_MIN`
  - `config/constants.py:59` `ENTRY_B_MIN`
- simulate 初始 `b` 采样改为走配置区间，而非硬编码：
  - `data/simulate_ts.py:567`
  - `data/simulate_ts.py:587`
- 进入者不再固定 `b=0`，改为区间采样：
  - `data/simulate_ts.py:539`

### 预期影响
- mode2 将不再天然偏向低杠杆子空间；
- `b>=1` 邻域出现概率提升，Q 的高杠杆边界损失可被激活；
- default 区域样本增加后，`bar_z` 不再被动塌缩到接近 0。

## 2. 过渡方程时点一致性修复（B）

### 已改动
- SimulateTS 分支扩展中，`b_{t+1}` 改为使用当期 `eta_t`：
  - `data/simulate_ts.py:405`
- Sample 的 child 杠杆回填改为使用 parent 的 `ETA`：
  - `data/sample.py:756`
  - `data/sample.py:768`
- Episode 中 P0/PI/Q 三个损失构造 child state 时统一使用 parent 的 `eta`：
  - `training/episode.py:625`
  - `training/episode.py:742`
  - `training/episode.py:878`

### 预期影响
- `bp_t -> b_{t+1}` 的映射和理论一致（`eta_t` 口径）；
- FOC 对 `bp` 的识别噪声下降；
- P/Q 联立方程中的状态转移链条更一致。

## 3. P/bar_z 梯度死区缓解（D）

### 已改动
- 新增平滑参数：
  - `config/constants.py:70` `P_SOFTPLUS_BETA`
  - `config/constants.py:71` `BARZ_LOGIT_TEMP`
- `P` 由硬 `clamp` 改为 `softplus`，`bar_z` 改为对 `Phat` 的温和 sigmoid：
  - `models/policy_value.py:176`
  - `models/policy_value.py:178`

### 预期影响
- 避免 `Phat<0` 区域梯度直接截断；
- `bar_z` 不再因超陡温度迅速饱和，边界附近可训练区间扩大；
- 有助于恢复“存在破产区域”的可学习性。

## 4. CF 融资成本符号修复（E）

### 已改动
- `P0Loss` 现金流调整改为：`cf0p = cf0p_raw - kappa_e * relu(-cf0p_raw)`：
  - `losses/p0_loss.py:97`
- `PILoss` 现金流调整改为：`cfip = cfip_raw - kappa_e * relu(-cfip_raw)`：
  - `losses/pi_loss.py:106`

### 预期影响
- 负现金流时，融资成本会让 CF 更负（方向符合经济含义）；
- 避免“融资成本反而抬升股权价值”的偏误；
- 对恢复 default 区域（`P≈0`）有正向作用。

## 5. FOC 梯度可诊断性增强（C）

### 已改动
- P0/PI loss 内记录 FOC 关键梯度统计（含 missing 比率）：
  - `losses/p0_loss.py:70`, `losses/p0_loss.py:221`
  - `losses/pi_loss.py:75`, `losses/pi_loss.py:214`
- Episode 将上述诊断并入训练日志输出：
  - `training/episode.py:136`
  - `training/episode.py:382`
  - `training/episode.py:389`
  - `training/episode.py:685`
  - `training/episode.py:808`

### 新增可观测项（losses 中）
- P0: `p0_cf_grad_abs_mean`, `p0_pgrad_abs_mean`, `p0_cf_grad_missing`, `p0_pgrad_missing_ratio`
- PI: `pi_cf_grad_abs_mean`, `pi_pgrad_abs_mean`, `pi_cf_grad_missing`, `pi_pgrad_missing_ratio`

## 6. 验证

已执行语法检查：
- `python -m py_compile config/constants.py data/simulate_ts.py data/sample.py models/policy_value.py losses/p0_loss.py losses/pi_loss.py training/episode.py`
- 结果：通过（无语法错误）。

> 说明：本地沙盒环境无法稳定跑完整 torch 训练（OpenMP 共享内存限制），行为验证请以你本机 notebook 复现实验为准。

## 7. 建议你下一轮重点看

1. mode2 `b` 支持是否扩展到接近 `[0,1]`，并出现 `b>=0.8` 的有效样本。  
2. `bar_z mean` 是否从接近 0 回升，并出现明显非零区域。  
3. `p0_cf_grad_abs_mean/pi_cf_grad_abs_mean` 与 `p0_pgrad_abs_mean/pi_pgrad_abs_mean` 是否长期远离 0。  
4. `p0_pgrad_missing_ratio/pi_pgrad_missing_ratio` 是否显著下降。  
5. Q 图形是否恢复为：`z` 方向递增，`b` 方向出现“先升后降”而非全域贴近 0。
