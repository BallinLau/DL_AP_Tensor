# 方法论实现一致性审查报告（2026-03-11）

## 1. 审查范围

- 方法论来源目录：`/Users/ballinliu/Desktop/PHD/Project1/VibeCoding`
- 重点对照文档：`training.md`、`loss.md`、`modelszoo.md`、`sample.md`、`Simulate_ts.md`、`FC2训练.md`
- 代码仓库：`/Users/ballinliu/Desktop/PHD/Project1/DL-APSSH`
- 判定标准：
  - `Implemented`：核心逻辑存在且已接入训练/模拟主流程
  - `Partial`：实现了部分逻辑，或实现存在但接线与文档不一致
  - `Missing`：未找到可验证实现

总体判断：核心骨架已实现，但尚未达到“方法论文档完整落地”。关键差距集中在 FC2 口径一致性、Episode 闭环顺序、收敛判据与部分训练技巧接线。

---

## 2. 组件映射

| 方法组件 | 期望功能 | 代码证据 | 状态 |
|---|---|---|---|
| ShareLayer 输入约束 | no-invest 头不看 `i`，invest 头看 `i` | `models/share_layer.py:125-137,169-181,255-269,345-361` | Implemented |
| SDF&FC1 联合一步前向 | FC1 预测宏观、Value 生成 `w`、再算 `M` | `models/sdf_fc1.py:456-535` | Implemented |
| FC1 标准化流程接线 | `fit_scaler + inverse_transform` 参与训练闭环 | `models/base.py:109-147`, `models/sdf_fc1.py:447-455`（全仓库仅定义，未见训练调用） | Partial |
| SDF Loss | Euler 残差 + 矩约束 | `losses/sdf_loss.py:208-244`, `training/episode.py:399-426` | Implemented |
| P0 Loss | Bellman + FOC + 单调性 + z 惩罚 | `losses/p0_loss.py:288-339`, `training/episode.py:499-536` | Partial |
| PI Loss | Bellman + FOC + 单调性 + b>1 + z 惩罚 | `losses/pi_loss.py:269-305`, `training/episode.py:624-654` | Partial |
| Q Loss | 主残差 + 边界约束 + z 惩罚 | `losses/q_loss.py:215-250`, `training/episode.py:735-757` | Implemented |
| FC2 fixed-point 损失 | 节点一致性+跨期一致性 | `losses/fc2_loss.py:126-187`，但训练主流程走 `FC2LossPipe`：`training/episode.py:785-794` | Partial |
| Sample triplet 结构 | `(B+1)-tuple` 对齐 | `data/sample.py:113-114,283-341,817-837` | Implemented |
| FlowNet/Real(b,z) 抽样 | `realbz` 可用 | `data/sample.py:85,239`, `data/data_utils.py:231-261` | Partial |
| SimulateTS 树状分支推进 | `parent -> branches -> main_branch` | `data/simulate_ts.py:129-154,361-450` | Implemented |
| SimulateTS 进入/退出机制 | `entry` 与 `bar_z` 筛选 | `data/simulate_ts.py:474-541` | Implemented |
| FC2 输入 202 维口径 | `[q_b, q_z, x, lnK_prev]` | 文档：`modelszoo.md:221`, `FC2训练.md:168`; 代码：`config/constants.py:81`, `models/fc2.py:45-47`, `losses/FC2losspipe.py:114-134` | Partial |
| FC2 alive/entry 机制 | alive mask 语义一致，entry 按空槽位写入 | `losses/FC2losspipe.py:192-199,260-264`, `experiments/fill_fullN_entrants.py:109-276` | Partial |
| Episode 0 闭环步骤 | Step0.1~0.4 顺序一致 | `training/episode.py:1116-1190` | Partial |
| Episode>=1 闭环步骤 | 先 `simulate_ts` 再训练 Step1.1~1.5 | `training/episode.py:1195-1242` | Partial |
| 收敛判据 | NMSE + FC1 contraction 检验 | `training.md:202-206,175`; 代码检索未见实现 | Missing |
| 稳定技巧接线 | `auto_cooldown/episode_guard/L-BFGS/Plateau` | 参数定义在 `config/hyperparams.py:47-49,85-87`，训练代码未见对应执行路径 | Partial |
| 主入口可执行一致性 | `main.py` 与模型/配置签名匹配 | `main.py:94-113,132-137`, `models/policy_value.py:45-50`, `models/sdf_fc1.py:396-403`, `models/fc2.py:35-41` | Partial |

---

## 3. 详细分析

### 3.1 模型结构层面

期望：
- ShareLayer 保证 `i` 仅影响投资分支头。

证据：
- `BpHead/PHead` 在 `requires_i=True` 时拼接 `i`，否则不拼接。
- `SharedModel/CombinedModel` 明确拆出 base_state 与 `i`，并将 `i` 仅传给 `bpI/PI` 头。

结论：
- `Implemented`，与方法论文档一致。

---

### 3.2 SDF/FC1 与损失链路

期望：
- SDF、P0、PI、Q 都按文档组合项训练。

证据与判断：
- SDF：`Implemented`。`SDFLoss` 与 `Episode._compute_sdf_loss` 都有 Euler+moment。
- P0/PI：`Partial`。`Episode` 里走了 Bellman + FOC + z 惩罚，但未接入单调性惩罚路径（未调用 `P0Loss/PILoss.forward` 的 monotonic 分支）。
- Q：`Implemented`。主残差、边界约束、z 惩罚都在 `Episode._compute_q_loss` 组合。

风险：
- 与 `loss.md` 中“P0/PI 含单调性约束”的方法论不完全一致。

---

### 3.3 Sample / SimulateTS 数据层

期望：
- Sample 严格 triplet。
- simulate 侧支持 Real(b,z)/FlowNet。

证据与判断：
- triplet：`Implemented`（Sample 的数据组织和 tensor 切分完整）。
- Real(b,z)/FlowNet：`Partial`。`Sample` 声明支持 `realbz`，但底层 `generate_firm_states` 仅支持 `uniform/feasible`；`SimulateTS` 仅保留 `dist_b` 扩展接口。

---

### 3.4 FC2 口径一致性（关键偏差）

期望（文档）：
- FC2 输入应为 202 维，含 `lnK_prev`。

证据（代码）：
- 常量与模型均按 201 维（`100+100+x`）实现。
- FC2 pipeline parent/children 输入构建也只拼接 `x`。

结论：
- `Partial`，属于“能训练但与方法定义不一致”的偏差。

---

### 3.5 FC2 alive 与 entry 语义

期望（文档）：
- alive mask 与 `bar_z` 语义一致，entry 是写入机制而不是替代 mask。

证据：
- FC2 pipeline 中 `updated_alive = base_alive * bar_z`。
- SimulateTS 退出逻辑是 `bar_z < 0.5` 才存活。

结论：
- `Partial`，FC2 训练链与模拟链语义不一致，存在行为偏差风险。

---

### 3.6 Episode 闭环顺序

期望（`training.md`）：
- Episode>=1 应先有 `simulate_ts` 输入，再进行 Step1.1~1.4 训练。

证据：
- `run_episode` 的 episode>=1 分支在训练完模块后才调用 `simulator.simulate()`。
- 且 episode>=1 若直接训练 `sdf_fc1/policy_value`，依赖 `self.df` 先验存在（新建 Episode 时通常为空），顺序耦合较强。

结论：
- `Partial`，与文档闭环顺序不一致。

---

### 3.7 收敛判据与稳定技巧

期望：
- NMSE 收敛、FC1 contraction、auto_cooldown、episode_guard、L-BFGS 等应有执行逻辑。

证据：
- `HyperParams` 有参数位，但代码检索只见参数定义，未见完整执行链。
- `NMSE/Contraction` 在训练主代码中未找到实现。

结论：
- 收敛判据：`Missing`
- 稳定技巧：`Partial`

---

### 3.8 主入口一致性

期望：
- `main.py` 能按当前模型签名直接构建并训练。

证据：
- `main.py` 使用 `config.SDF_INPUT_DIM / FIRM_STATE_DIM / FC2_N_QUANTILES`（配置中不存在）。
- `main.py` 传入的构造参数与三个模型类签名不匹配。

结论：
- `Partial`（入口可执行性风险高）。

---

## 4. 缺失组件

1. NMSE 收敛判据（`C/K`）与自动停止逻辑。
2. FC1 contraction 检验（Step 1.35）。
3. 文档定义的 202 维 FC2 输入口径（含 `lnK_prev`）在主流程中的统一实现。

---

## 5. 修复建议（按优先级）

1. 高优先：统一 FC2 口径
- 在 `Config`、`FC2Model`、`FC2LossPipe`、`SimulateTS.compute_fc2_features` 同步引入 `lnK_prev`，统一为 202 维。

2. 高优先：修复 FC2 alive 语义
- 将 FC2 pipeline 的 alive 更新改成与 SimulateTS 一致（阈值或 `1-bar_z` 语义），并补充单元测试对齐。

3. 高优先：对齐 Episode>=1 顺序
- 改为“先生成 simulate_ts 数据，再训练该轮模块”，避免先训后生导致的数据时序偏差。

4. 中优先：补齐 P0/PI 的单调性训练项
- 在 `Episode` 调用 `P0Loss/PILoss` 完整 `forward`，或显式补入 monotonic penalty。

5. 中优先：落地收敛控制
- 实现 NMSE 计算、阈值判定、early stop，并把关键指标落盘（CSV）。

6. 中优先：修复主入口
- 统一 `main.py` 与 `Config/HyperParams/Model` 构造签名，保证 README 的命令可运行。

---

## 6. 实现评分

- 总组件数：19
- Implemented：7
- Partial：11
- Missing：1
- Coverage：65.8%

计算方式：
- `Coverage = (Implemented + 0.5 * Partial) / Total * 100%`
- 本次为 `(7 + 0.5 * 11) / 19 = 65.8%`
