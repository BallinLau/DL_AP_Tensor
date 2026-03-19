# 方法论实现一致性审查报告（`main_4.tex` 对照当前代码）

## 1. 方法概览
目标文档是 `/Users/ballinliu/Desktop/PHD/Project1/DL Equilibrium/main_4.tex`，核心方法包括：
- 价值与政策网络联合求解（`Q, P, P0, PI, bp, bar_z, bar_i`）；
- SDF+FC1 联合训练（含 SDF 矩约束）；
- FC2 宏观聚合闭环；
- 按 `Warmup -> PV -> FC2 -> Converge? -> FC1+SDF` 的 episode 循环；
- 结构化预训练与 GradNorm 权重自适应。

---

## 2. 组件映射

| 方法组件 | 期望功能（来自 `main_4.tex`） | 代码位置 | 状态 |
|---|---|---|---|
| 状态/输出定义 | firm state `(b,z,eta,i,x,hatcf,lnkf)`；输出 `Q,bp0,bpI,P0,PI,P,bar_z,bar_i` | `models/policy_value.py`, `models/share_layer.py` | Implemented |
| Warmup（仅 SDF+mom） | 无 FC1 真值时先用 `L_SDF + mom1 + mom2` | `training/episode.py` | Implemented |
| FC1/FC2 宏观模块 | FC1 预测宏观状态，FC2 做截面->宏观映射 | `models/sdf_fc1.py`, `models/fc2.py`, `losses/FC2losspipe.py` | Implemented |
| SDF 核与矩约束 | AiO 型 SDF consistency + 对数矩约束 | `losses/sdf_loss.py`, `training/episode.py` | Partial |
| Q 损失 | `Xi^Q = L_Q^(0)+L_Q^(1)+边界项`，含回收/边界约束 | `losses/q_loss.py`, `training/episode.py` | Partial |
| P0/PI Bellman+FOC | AiO 双抽样 Bellman + FOC 残差 | `losses/p0_loss.py`, `losses/pi_loss.py`, `training/episode.py` | Partial |
| `bar_z^value` / `bar_i^value` 监督 | BCE 分类边界损失（文中 1620-1629） | 仅有模型定义，无训练接入 | Missing |
| 结构化预训练（Q/P） | 用经济学先验函数先预训练 `Q,P0,PI` | 当前仅 Q warm-start，未见 P0/PI 对应流程 | Partial |
| 训练总流程与 R2 收敛闭环 | `PV->FC2->R2 检查->未收敛则 FC1+SDF` 循环 | `training/episode.py` / `experiments/*` | Partial |
| GradNorm 权重更新 | 动态更新多目标权重并归一化 | 采用自定义 adaptive scheduler（非 GradNorm） | Missing |

---

## 3. 详细分析

### 3.1 状态与输出接口（Implemented）
期望：与文档中状态和输出定义对齐。

代码证据：
- firm-state 输入定义：`(b,z,eta,i,x,hatcf,lnkf)`（`models/policy_value.py:110`）。
- 输出包含 `Q,bp0,bpI,P0,PI,bar_i,bar_z,P,Phat,bp`（`models/policy_value.py:18-29,126-137`）。
- `bar_i` 来自 `PI-P0` 的平滑指示（`models/share_layer.py:360-363`）。

结论：基础接口是对齐的。

---

### 3.2 Warmup（Implemented）
期望：先在无 FC1 真值阶段只训练 SDF+矩约束。

代码证据：
- Episode 0 先跑 SDF/FC1（`training/episode.py:1397-1405`）。
- 之后才在 `add_FC1loss=True` 下引入 FC1 重建（`training/episode.py:1451-1458`）。
- `add_FC1loss=False` 时 `recon_loss` 不启用，且矩项权重可单独提高（`training/episode.py:506-541`）。

结论：实现了 warmup 思想。

---

### 3.3 FC1/FC2 模块（Implemented）
期望：FC1 跨期宏观预测，FC2 截面分布映射宏观量。

代码证据：
- FC1 采用增量形式：`y_{t+1}=y_t+Δy`（`models/sdf_fc1.py:390-395`）。
- FC2 输入为 `100(b分位)+100(z分位)+x`（`models/fc2.py:45-48,74-84`）。
- FC2 pipeline 实际接入训练（`training/episode.py:979-1016`）。

结论：主干在代码中可见并被调用。

---

### 3.4 SDF 核与矩约束（Partial）
期望：文中 `L_SDF = E[f(S,S')f(S,S'')]` + 对数矩约束（`main_4.tex:1557-1611,1741-1794`）。

代码证据：
- SDF 结构和矩约束实现存在（`models/sdf_fc1.py:32-130`, `losses/sdf_loss.py:35-79`）。
- 但训练中主损失采用 `log1p(abs(prod(residuals)))`（`training/episode.py:484-487`, `losses/sdf_loss.py:215-225`），与论文表达的原式并不完全等价。

结论：有实现，但目标函数形式存在偏离。

偏差或风险：早期稳定性更好，但可能改变最优点位置与理论残差几何。

---

### 3.5 Q 模块（Partial）
期望：含回收项、边界项与 AiO 结构（`main_4.tex:1390-1404`）。

代码证据：
- 回收函数与 `b<=0 / b>=1` 边界损失已实现（`losses/q_loss.py:60-177`）。
- episode 中用 AiO 聚合并训练（`training/episode.py:891-899`）。
- 但文档中的 FB 约束形式（`main_4.tex:1385-1388`）未见等价实现。

结论：核心约束大体存在，但不是完全同构实现。

---

### 3.6 P0/PI Bellman+FOC（Partial）
期望：AiO 双抽样 Bellman + FOC（`main_4.tex:1418-1504`）。

代码证据：
- `P0/PI` 的 cashflow、Bellman、FOC 函数在 loss 类里齐全（`losses/p0_loss.py:71-221`, `losses/pi_loss.py:76-214`）。
- 但 episode 实际训练时仍采用“分支残差乘积再取绝对值均值”，没有走 AiO 聚合（`training/episode.py:653-659,770-776`）。
- 且 `PI` 分支子状态杠杆更新当前写成 `child_state[:,0]=bp_for_pi`，未使用 `eta*bp + (1-eta)*b`（`training/episode.py:739-741`）。

结论：有主要项，但训练落地与文档公式有关键偏差。

偏差或风险：会直接影响你关心的 `Q/P` 曲面形状和违约边界学习。

---

### 3.7 阈值网络监督（Missing）
期望：`L_{bar_z}`、`L_{bar_i}` BCE 训练（`main_4.tex:1615-1640`）。

代码证据：
- 有 `BarzModel/BariModel` 定义（`models/share_layer.py:371-443`）。
- 但主前向用的是 `cal_phats` 生成 `bar_z`（`models/policy_value.py:122-177`），训练链路未见 BCE 损失接入（全仓库检索无对应 loss/step 调用）。

结论：该组件目前未接入主流程。

---

### 3.8 结构化预训练（Partial）
期望：先用结构化函数预训练 `Q,P0,PI`（`main_4.tex:1654-1699,1873`）。

代码证据：
- 目前新增了 `Q` warm-start（`training/episode.py:947-963`）。
- `build_models` 无 `P0/PI/Q` 的独立 supervised pretrain 流程（`experiments/run_utils.py:21-55`）。

结论：仅 Q 部分有近似实现，P0/PI 预训练缺失。

---

### 3.9 训练总流程与收敛闭环（Partial）
期望：`PV->FC2->R2 检查->未收敛则 FC1+SDF` 循环（`main_4.tex:1869-1924`）。

代码证据：
- `Episode.run_episode` 包含 `SDF -> PV -> FC2 -> SDF` 的阶段串联（`training/episode.py:1397-1458`）。
- 但未看到按 `R2` 阈值驱动的停止/重训闭环逻辑；目前是固定 epoch 与固定阶段执行。

结论：流程骨架存在，但收敛控制机制未按论文闭环落地。

---

### 3.10 GradNorm 权重更新（Missing）
期望：文档多处写明用 GradNorm 自适应损失权重（`main_4.tex:1554,1728,1815`）。

代码证据：
- 当前是 `LossWeightScheduler` 的 fixed/linear/step/adaptive 规则（`training/scheduler.py:10-142`），并非 GradNorm。

结论：方法名义与实现不一致。

---

## 4. 高影响偏差（与当前训练异常最相关）

1. `PHead` 使用 `softplus` 强制 `P0/PI>0`，再加 `P=clamp(Phat,min=0)`，会弱化违约发生。
- 证据：`models/share_layer.py:152-167`, `models/policy_value.py:173-176`。

2. `P0/PI` 在 episode 中未使用 AiO 聚合，而是 product 绝对值损失。
- 证据：`training/episode.py:657-659,773-776`。

3. `PI` 分支杠杆更新未使用 `eta` 混合公式。
- 证据：`training/episode.py:739-741`。

4. FC2 pipeline 内部将 `bar_z` 直接当作“存活概率”使用（内部含义与其他模块的 `1-bar_z` 生存口径不一致）。
- 证据：`losses/FC2losspipe.py:192-199`。

---

## 5. 缺失组件

1. `bar_z^value / bar_i^value` 的 BCE 监督训练链路。  
2. 论文式 GradNorm 多目标权重更新。  
3. `R2` 驱动的闭环收敛判据与“未收敛继续 FC1+SDF”自动迭代机制。  

---

## 6. 修复建议（按优先级）

1. 先修训练目标一致性：
- 将 episode 的 `P0/PI` 聚合从 product 改回 AiO（与文档同口径）。
- 修复 `PI` 分支 `b_{t+1}` 更新为 `eta*bp + (1-eta)*b`。

2. 修口径一致性：
- 统一 `bar_z` 语义（默认=违约概率/指标），FC2 pipeline 使用 `survival=1-bar_z`。

3. 再补理论组件：
- 接入 `bar_z^value / bar_i^value` BCE 训练（或在文档中删除该模块声明）。
- 把权重更新替换为真正 GradNorm（或文档改为 adaptive scheduler）。

4. 收敛闭环工程化：
- 增加 `R2(Hatc), R2(LnK)` 监控与阈值终止；不满足则自动进入下一轮 `FC1+SDF` 迭代。

---

## 7. 实现评分

- Coverage: **55%**
- Implemented: **3**
- Partial: **5**
- Missing: **2**

计算：
- Coverage = `(Implemented + 0.5 * Partial) / 总组件数`
- `= (3 + 0.5*5) / 10 = 55%`

