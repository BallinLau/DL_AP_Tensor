# DL_AP_Local 方法论（代码对齐版，2026-03-13）

## 1. 研究目标与总体框架

本项目目标是在含融资摩擦、投资决策与违约风险的异质企业动态经济中，联合学习以下对象：

1. 随机贴现因子（SDF）与宏观过渡（FC1）。
2. 企业层面的债券定价与股权定价（Q, P0, PI）及政策函数（bp0, bpI, bar_i, bar_z）。
3. 横截面分布到宏观状态的固定点映射（FC2，可选）。

整体是“宏观-微观双向闭环”：

- 上层：宏观状态决定贴现与跨期环境。
- 下层：企业横截面在给定宏观下做融资/投资/违约决策并反向聚合出宏观量。

---

## 2. 状态、控制与网络对象

### 2.1 状态变量定义

代码中 firm-state 列顺序定义为
\[
s_i=(b_i,z_i,\eta_i,i_i,x,\hat c_f,\ln K_f),
\]
对应 `(b, z, ETA, i, x, Hatcf, LnKF)`（见 `SIMMODEL` 枚举）。

- \(b\)：杠杆/债务状态
- \(z\)：企业 idiosyncratic 冲击
- \(\eta\in\{0,1\}\)：再融资可得性冲击
- \(i\)：投资成本冲击
- \(x\)：宏观冲击
- \(\hat c_f,\ln K_f\)：宏观代理状态

### 2.2 网络模块

1. `SDF+FC1` 组合网络：
- FC1 预测 \((\hat c_f,\ln K_f)\) 的跨期变化；
- Value-W 生成 \(w\)；
- SDF 模块用 \((w_t,w_{t+1},\hat c,\ln K)\) 计算 \(M\)。

2. `PolicyValue` 网络：
- Shared 分支输出 \(Q, bp^0, bp^I\)；
- Combined 分支输出 \(P^0,P^I,\bar i\)；
- 通过 `cal_phats` 给出 \(\hat P, P, \bar z\)。

3. `FC2` 网络（可选）：
- 输入为横截面分位特征（`b` 100 分位 + `z` 100 分位 + `x`）；
- 输出 \((\hat c,\ln K)\)。

---

## 3. 结构方程与损失函数

## 3.1 SDF 与 FC1 联合部分

### 3.1.1 FC1 的增量建模

FC1 不直接预测绝对值，而是预测增量：
\[
\Delta \hat c_{f,t+1},\ \Delta \ln K_{f,t+1},
\]
并更新为
\[
\hat c_{f,t+1}=\hat c_{f,t}+\Delta \hat c_{f,t+1},\qquad
\ln K_{f,t+1}=\ln K_{f,t}+\Delta \ln K_{f,t+1}.
\]

### 3.1.2 Value-W 结构化正值约束

Value-W 采用结构化参数化：
\[
w_t=\exp(\hat c_{f,t})+\text{surplus}_t,
\]
其中 `surplus` 由 softplus 网络输出并加下界，保证
\[
w_t-\exp(\hat c_{f,t})>0.
\]

### 3.1.3 SDF 计算

实现中 SDF 为
\[
M_{t,t+1}^{(j)}=\beta^{\kappa}
\exp\left[-\gamma(\ln K_{t+1}^{(j)}-\ln K_t)-\frac{\kappa}{\sigma}(\hat c_{t+1}^{(j)}-\hat c_t)\right]
\left(\frac{w_{t+1}^{(j)}}{w_t-\exp(\hat c_t)}\right)^{\kappa-1}.
\]

指数项使用 clamp（默认 \([-10,10]\)）以提升数值稳定性。

### 3.1.4 SDF 损失

欧拉残差（每个分支 \(j\)）写作
\[
\varepsilon_{\text{sdf}}^{(j)}
=\exp\left[(\Delta \ln K^{(j)})(1-\gamma)-\frac{\kappa}{\sigma}\Delta \hat c^{(j)}\right]\beta^{\kappa}
\cdot (w_{t+1}^{(j)})^{\kappa}
-(w_t-\exp(\hat c_t))^{\kappa}.
\]

主损失：
\[
\mathcal L_{\text{main}}=\mathbb E\left[\log\left(1+\left|\prod_j \varepsilon_{\text{sdf}}^{(j)}\right|\right)\right].
\]

矩约束：
\[
\log \mathbb E[M]\in[\mu_{lo},\mu_{hi}],\qquad
\log \mathrm{Var}(M)\le var_{hi}.
\]

最终在 episode 实现为
\[
\mathcal L_{\text{SDF-total}}=\mathcal L_{\text{main}}+\omega_{mom}\mathcal L_{mom}+\omega_{recon}\mathcal L_{recon}.
\]

其中：

- 阶段1（`add_FC1loss=False`）使用较小学习率与更高矩约束权重；
- 阶段2（`add_FC1loss=True`）加入 FC1 重建项（监督 `Hatc/LnK`）。

---

## 3.2 企业价值与政策：P0 / PI / bp

## 3.2.1 现金流定义

不投资分支现金流：
\[
CF^0_t
=\pi_t + \eta_t\big((1-\kappa_b)Q_t(b'_t)-Q_t(b_t)\big)-\kappa_e\max(-CF^{0,raw}_t,0).
\]

投资分支现金流：
\[
CF^I_t
=\pi_t-i_t + \eta_t\big((1-\kappa_b)gQ_t(b'_t)-Q_t(b_t)\big)-\kappa_e\max(-CF^{I,raw}_t,0).
\]

其中利润项由 `compute_cashflow` 给出。

## 3.2.2 Bellman 残差

P0 分支：
\[
\varepsilon_{P0}^{(j)}=P_t^0-CF_t^{0,(j)}-M_t^{(j)}P_{t+1}^{(j)}(1-\bar z_{t+1}^{(j)}).
\]

PI 分支：
\[
\varepsilon_{PI}^{(j)}=P_t^I-CF_t^{I,(j)}-gM_t^{(j)}P_{t+1}^{(j)}(1-\bar z_{t+1}^{(j)}).
\]

多分支聚合采用 AIO：
\[
\mathrm{AIO}(\{\varepsilon_j\})=(1-w)\frac{1}{N}\sum_j\varepsilon_j^2+w\left|\prod_j\varepsilon_j\right|.
\]

## 3.2.3 FOC 与 KKT（bp 约束在 P0/PI 内）

`bp` 由 sigmoid 限制在 \([0,1]\)，故使用有界控制 KKT 条件：

- 内点 \(0<bp<1\)：FOC=0
- 下边界 \(bp\approx0\)：FOC\(\le0\)
- 上边界 \(bp\approx1\)：FOC\(\ge0\)

实现细节：

1. FOC 残差通过自动微分计算：
\[
\mathrm{FOC}^{(j)}=\frac{\partial CF^{(j)}}{\partial bp}
+\eta^{(j)}M^{(j)}\frac{\partial P_{t+1}^{(j)}}{\partial bp}(1-\bar z_{t+1}^{(j)}).
\]

2. 条件矩改为保留符号的
\[
\mathbb E[\mathrm{FOC}\mid \eta=1]=0,
\]
并仅在 active 样本上计算 FOC 的 z-penalty。

3. KKT 罚项对上边界违约项加权（`kkt_high_weight`），并用 soft region weight 在内点/边界间平滑过渡。

4. 针对 \(\eta\) 稀疏，bp 相关项使用 active-ratio 重权重：
\[
\text{boost}=\mathrm{clip}\left(\frac{\rho_{target}}{\rho_{active}},1,\rho_{max}\right).
\]

5. 训练 batch 层面对 \(\eta=1\) 样本做条件重采样，提高有效梯度密度。

最终 `P0/PI` 总损失均为“Bellman 主项 + z 惩罚 +（PI 的 b 约束）+ 经过 boost 的 bp 项（FOC+KKT+FOC-z）”。

---

## 3.3 债券定价 Q 方程

## 3.3.1 总债价值口径

Shared 模型中，债券价值采用
\[
Q=b_+\cdot q_{unit},\qquad b_+=\max(b,0),
\]
故当 \(b=0\) 时结构上有 \(Q=0\)。

回收价值采用总债价值口径：
\[
\mathrm{Recovery}_{total}=b_+\cdot\phi\left(1-\delta+e^{x+z}\right).
\]

## 3.3.2 主方程残差

定义
\[
\text{multiplier}=\bar i(G-1)+1,
\qquad
b^{sp}=\frac{b}{\text{multiplier}}.
\]

主残差（每分支）为
\[
\varepsilon_Q^{(j)}=M^{(j)}\Big[(b+Q^{sp,(j)}\cdot \text{multiplier})(1-\bar z^{sp,(j)})
+\mathrm{Recovery}_{total}\cdot \text{multiplier}\cdot\bar z^{sp,(j)}\Big]-Q.
\]

并叠加：

- 违约一致性约束 \((Q-\mathrm{Recovery}_{total})^2\bar z\)；
- 边界约束 \(b\le0\Rightarrow Q\approx0\), \(b\ge1\Rightarrow Q\approx\mathrm{Recovery}_{total}\)；
- z 区域惩罚；
- 形状约束（\(\partial Q/\partial z\ge0\)，低杠杆区 \(\partial Q/\partial b\ge0\)，高杠杆区 \(\partial Q/\partial b\le0\)）；
- 可选 warm-start 监督项（前若干 epoch）。

---

## 3.4 FC2 固定点模块（可选）

FC2 输入为
\[
\phi_t=[q_b(100),q_z(100),x_t]\in\mathbb R^{201},
\]
输出 \((\hat c_t,\ln K_t)\)。

训练主流程中 FC2 使用 `FC2LossPipe`：

1. 用 FC2 预测 parent 宏观；
2. 将宏观输入 PV，得到 \((bp,\bar i,\bar z)\) 并更新 child 状态；
3. 再构造 children 的 FC2 输入与预测；
4. 用 parent+children 的一致性 MSE 作为 `fc2` 损失。

对应思想是
\[
(\hat c,\ln K)=\mathcal A\big(\mathcal P(\hat c,\ln K;\text{cross-section})\big)
\]
的近似固定点求解。

---

## 4. 数据生成与状态转移

## 4.1 Sample 截面生成

`Sample` 用于训练批数据构造，支持 `sample/simulate` 两种规模：

- `build_sdf_fc1_df()`：生成 SDF/FC1 宏观 pair（`x_t,x_{t+1},Hatcf_t,LnKF_t`）；
- `build_policy_value_df()`：生成 policy/value 所需 firm-level parent/children 数据。

child 杠杆更新规则为
\[
b_{t+1}=\eta_{t+1}bp_t+(1-\eta_{t+1})b_t.
\]

## 4.2 SimulateTS 树状时间序列

`SimulateTS` 用于 episode 闭环模拟：

1. parent 节点（branch=-1）计算当前决策并聚合宏观；
2. 向多分支扩展（branch=0,1,...）；
3. 分支内执行进入（entry）与退出（由 \(\bar z\) 控制）；
4. 选主分支进入下一期。

关键转移方程：
\[
x_{t+1}=\rho_x x_t+\sigma_x\varepsilon^x_{t+1},
\qquad
z_{i,t+1}=\rho_z z_{i,t}+\sigma_z\varepsilon^z_{i,t+1},
\]
\[
b_{i,t+1}=\eta_{i,t+1}bp_{i,t}+(1-\eta_{i,t+1})b_{i,t},
\]
\[
K_{i,t+1}=\bar i_{i,t}G K_{i,t}+(1-\bar i_{i,t})K_{i,t}.
\]

资源核算：
\[
Y_i=e^{x+z_i}K_i,
\]
\[
\Phi_i=(1-\phi)(1+e^{x+z_i})K_i\bar z_i,
\]
\[
I_i=\bar i_iK_i i_i-\bar z_iK_i+\delta K_i,
\]
\[
C_i=Y_i-I_i-\Phi_i.
\]

宏观聚合：
\[
K=\sum_i K_i,
\qquad
C=\sum_i C_i,
\qquad
\ln K=\log K,
\qquad
\hat c=\log\left(\frac{C}{K}+\epsilon\right).
\]

---

## 5. Episode 训练制度（三模式）

项目当前支持三种 episode 模式：

### 5.1 Mode0（通常用于 episode 0）

1. 用 `Sample` 训练 SDF stage1（矩条件主导，无 FC1 重建）；
2. 用 `Sample` 训练 Policy/Value；
3. 用 `SimulateTS(h=1)` 生成 `df, df_macro`；
4. 可选训练 FC2；
5. 将 `df_macro` 转 SDF pair，开启 `add_FC1loss=True` 做 SDF stage2（重建阶段），结束后复位。

### 5.2 ModeA（后续 episode 的轻闭环）

1. 用 `Sample` 训练 Policy/Value；
2. 用 `SimulateTS(h=1)` 生成当期 realized macro；
3. 可选 FC2；
4. SDF stage2（`add_FC1loss=True`）对齐 realized macro。

### 5.3 ModeB（后续 episode 的长时序闭环）

1. 先 `SimulateTS(h=T)` 生成长序列 firm/macro；
2. 在该模拟样本上训练 Policy/Value；
3. 在该样本上训练 SDF（当前为 `add_FC1loss=False` 路径）；
4. 可选 FC2；
5. 记录宏观预测/实现 R² 诊断。

`run_multi_episode_job.py` 支持 `modea/modeb/alternate` 作为 episode>0 的策略，并默认 `FC2` 关闭、按需启用。

---

## 6. 稳定化与训练工程机制

### 6.1 数值稳定

- SDF 指数项 clamp；
- `w=exp(c)+surplus` 避免负底数与幂爆炸；
- 梯度裁剪（`gradient_protection`）；
- 非有限值检测与替换（如 SDF recon）。

### 6.2 课程式与冻结策略

- SDF 两阶段（stage1 矩条件，stage2 重建）；
- Q 可选预训练：`q_only` 时可冻结非 Q 参数，仅训练 `q_head` 或 `share+q_head`；
- M 在 Q loss 中可 detach + clamp，降低噪声传导。

### 6.3 稀疏 \(\eta\) 对策

- batch 重采样提升 active 比例；
- FOC/KKT 条件化到 `eta=1` 子样本；
- active-ratio 重权重提高 bp 梯度有效强度。

---

## 7. 收敛判据与诊断输出

### 7.1 Bellman 残差收敛（非 AIO 口径）

对 `P0/PI/Q` 三个方程分别计算 \(|\varepsilon|\) 的：

- 均值 `mean(abs)`
- 90 分位 `p90(abs)`

并与阈值比较（默认）：
\[
\text{mean}<10^{-3},\qquad p90<5\times10^{-3}.
\]

### 7.2 训练期诊断

- SDF：`log(E[M])`, `log(Var(M))`, `Δhatcf`, `Δlnkf` 分布；
- P0/PI：`FOC active ratio`, `signed moment`, `KKT low/high/inner`；
- Q：边界项、形状项、warm-start 项；
- 宏观：`Hatc/LnK` 预测-实现 R²（支持按 branch）。

---

## 8. 代码锚点（主实现位置）

- SDF/FC1 模型：
[models/sdf_fc1.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/models/sdf_fc1.py)
- Policy/Value 模型：
[models/policy_value.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/models/policy_value.py)
- ShareLayer 与 Q=\(b\cdot q_{unit}\)：
[models/share_layer.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/models/share_layer.py)
- 各损失函数：
[losses/sdf_loss.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/losses/sdf_loss.py)
[losses/p0_loss.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/losses/p0_loss.py)
[losses/pi_loss.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/losses/pi_loss.py)
[losses/q_loss.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/losses/q_loss.py)
[losses/FC2losspipe.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/losses/FC2losspipe.py)
- Episode 训练主逻辑（三模式、KKT/FOC、收敛）：
[training/episode.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/training/episode.py)
- 数据生成：
[data/sample.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/data/sample.py)
[data/simulate_ts.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/data/simulate_ts.py)
- 关键超参数：
[config/hyperparams.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/config/hyperparams.py)
- 多 episode 运行入口：
[experiments/run_multi_episode_job.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/experiments/run_multi_episode_job.py)

---

## 9. 可直接放入论文的方法论写作建议

若按 `main4` 风格写论文正文，可直接采用以下章节顺序：

1. 经济环境与状态转移
2. 神经网络参数化（SDF/FC1, Policy/Value, FC2）
3. 定价方程与损失函数（SDF, P0, PI, Q）
4. 约束与可行域（KKT、边界条件、形状约束）
5. 训练制度（Mode0/ModeA/ModeB）与收敛标准
6. 诊断指标与可视化

本文件已与当前代码实现逐项对齐，可作为论文方法论章节的“实现基线版本”。
