# Policy & Value 结果异常原因分析（2026-03-12）

## 1. 你反馈的问题
1. 总体损失不够小。
2. `P/P0/PI` 与 `b,z` 的方向看起来对，但 `P` 基本都大于 0，不合理（几乎无破产）。
3. `Q` 与 `b,z` 的关系不对：理论上应对 `b` 倒 U、对 `z` 递增，且 `b=0` 时 `Q≈0` 但不能全域都很小。

## 2. 方法论期望（来自 VibeCoding 文档 + PDF）
- `P` 与破产关系：PDF 中 `P = max(0, \hat P)`，并用 `\hat P <= 0` 定义违约指示（`DL_Equilibrium.pdf` 提取文本第 20 页附近，见 `/tmp/dl_equilibrium_extracted_clean.txt:590-597`）。
- `Q` 的形状：文档明确写 `Q(b,z,η)` 对 `b` 为倒 U、对 `z` 递增（`/tmp/dl_equilibrium_extracted_clean.txt:1319-1324`）。
- 训练流程：Policy/Value 应分阶段（`Q→P0/PI→bp→联合`），并包含更完整子损失（含边界/门槛相关项）（`/Users/ballinliu/Desktop/PHD/Project1/VibeCoding/training.md:38-41, 75-84`）。

## 3. 代码对照与偏差（核心）

| 组件 | 方法论期望 | 当前实现证据 | 状态 |
|---|---|---|---|
| `P0/PI` 取值域与默认 | 允许穿越 0（至少 `\hat P` 应可为负）以触发默认边界 | `PHead` 输出激活是 `softplus`，强制 `P0/PI>0`（`models/share_layer.py:147-167`） | Partial / 存在结构偏差 |
| `P` 与 `bar_z` 构造 | 通过 `\hat P` 与边界网络/阈值确定默认 | `P = clamp(Phat,min=0)` 且 `bar_z = sigmoid(-50*P)`（`models/policy_value.py:173-176`），`P` 非负使 `bar_z` 很难显著 | Partial / 偏差较大 |
| Barz/Bari 子损失接入 | 文档描述有更多子损失与边界网络 | 训练步只优化 `p0/pi/q`（`training/episode.py:319-335`），无独立 `barz/bari` loss 接入 | Missing |
| P0/PI 单调性与 AIO 组合 | loss 文件中定义了单调性与 AIO 稳定项 | `Episode` 手写了另一套 loss 聚合（`training/episode.py:587-616, 704-734`），未接入 `losses/p0_loss.py` 与 `losses/pi_loss.py` 的单调性项（`losses/p0_loss.py:325-335`, `losses/pi_loss.py:291-301`） | Partial |
| `Q` 经济形状约束 | 对 `b` 倒 U、对 `z` 递增，不应全域塌缩 | 仅有 Euler/边界/`z` 惩罚（`losses/q_loss.py`），无“倒U显式约束”；且 `Q` 输出 `softplus`（`models/share_layer.py:69-85`） | Partial |
| Policy 训练策略 | 分阶段 + 更强稳定策略 | 当前 `train_step` 每步同时加总 `p0+pi+q`（`training/episode.py:319-335`） | Partial |

## 4. 针对你三个问题的原因解释

### 问题2：为什么几乎没有破产（`P` 基本 > 0）
最直接原因是**结构性正值偏置**：
- `P0/PI` 头使用 `softplus`，天生非负（`models/share_layer.py:152`）。
- `P` 再次被 `clamp(min=0)`（`models/policy_value.py:173`）。
- `bar_z` 由 `sigmoid(-50*P)` 给出（`models/policy_value.py:175`），当 `P` 稍大于 0 就接近 0。 

结论：当前实现里“默认状态”几乎只能在 `P≈0` 的极窄区域出现，训练中很容易退化为“几乎无破产”。

### 问题3：为什么 `Q` 形状不对、且全域偏小
是多因素叠加：
1. `Q` 也被 `softplus` 限制为正，表达空间受限（`models/share_layer.py:73`）。
2. `Q` 的边界约束里，低杠杆区 `b<=0.1` 会直接压 `Q→0`（`losses/q_loss.py:152-162`）；但如果样本里高杠杆很少，则 `b>=1` 约束弱。 
3. 当前 `bar_z` 常接近 0（上节），导致 `Q` 方程中“违约回收”分支贡献很弱（`losses/q_loss.py:97-101`），更容易塌到“低风险低价值”局部解。 
4. 缺少“倒U/单调”显式形状约束，只靠残差学习很难稳定恢复你期待的经济曲面。

补充数据证据（本地快速诊断）：
- sample 模式 `b<=0.1` 占比约 `10.9%`，`b>=0.9` 约 `7.9%`；
- simulate 模式 `b<=0.1` 占比约 `53.7%`，`b>=0.9` 约 `0%`（高杠杆几乎没覆盖）。

这会显著削弱 `Q` 在高杠杆区域的学习锚点。

### 问题1：为什么损失下不去
主要是**目标函数与训练流程不一致**：
- 你的方法论里是“分阶段训练 + 更多子损失分解”，现在是一步同时优化 `p0/pi/q`（`training/episode.py:319-335`），优化冲突更大。 
- `Episode` 中 P0/PI/Q 的残差聚合使用“分支残差乘积再取绝对值均值”（如 `training/episode.py:591-593, 708-710, 819-821`），而 `losses/utils.py` 里本来有更稳健的 AIO 稳定项组合（`losses/utils.py:107-141`）。
- loss 文件里定义的单调性惩罚在 `Episode` 当前路径没有被纳入总损失（见上表），形状约束不足会导致网络在“看起来局部有方向、但全局不合经济意义”的解附近停住。

## 5. 额外一致性偏差（对最终形状影响较大）
- 文档里提到的 `Barz/Bari` 相关训练目标没有进入主训练步（`training/episode.py:319-335`）。
- `Sample` 在构造 policy 训练数据时，child 的 `M` 直接来自 `sdf_fc1` 预测（`data/sample.py:267-270, 325-338`）。若该阶段 `sdf_fc1` 尚未充分稳定，会把噪声直接传入 P/Q Bellman 残差。

## 6. 结论（按影响度排序）
1. **最高影响**：`P0/PI/P` 的正值硬约束 + `bar_z` 由 `P` 单向生成，造成“几乎无破产”。
2. **高影响**：Policy loss 接入不完整（未接入 Barz/Bari 与单调性）+ 训练策略与方法论文档不一致（未分阶段）。
3. **中高影响**：`Q` 学习缺少显式形状约束，且样本分布在高杠杆区域覆盖不足，易出现“全域偏小”。
4. **中影响**：`M` 质量对 P/Q 残差敏感，若 SDF 阶段尚未稳定，会进一步恶化收敛。

---

## 7. 说明
- PDF 证据来自本地文件 `/Users/ballinliu/Desktop/PHD/Project1/VibeCoding/DL_Equilibrium.pdf` 的可提取文本（`PyPDF2` 抽取到 `/tmp/dl_equilibrium_extracted_clean.txt`）。
- 结论基于当前仓库代码静态对照与最小量化诊断，不代表你完整大规模训练超参搜索后的最优上限。

## 8. 已落地的 Q 优先修复（本轮代码修改）

### 8.1 训练流程支持 Q-only 预训练
- 文件：`training/episode.py`
- 改动：`_run_batches` 读取 `HyperParams.q_pretrain_epochs`，前若干 epoch 仅优化 `q`，之后回到 `p0+pi+q` 联合训练。
- 目的：先把 Q 曲面学稳，再进入多目标联合阶段。

### 8.2 Q 损失改为 AIO 稳定聚合
- 文件：`training/episode.py`
- 改动：`_compute_q_loss` 主残差从“分支乘积绝对值均值”改为 `compute_aio_residual(...)`。
- 目的：降低分支乘积带来的梯度不稳定与塌缩风险。

### 8.3 Q 阶段使用 M 的 detach + clamp
- 文件：`training/episode.py`、`config/hyperparams.py`
- 改动：新增 `q_use_detached_m/q_m_clamp_min/q_m_clamp_max`，Q loss 默认使用 `clamp(M).detach()`。
- 目的：减少 SDF 噪声直接污染 Q 训练。

### 8.4 Q 形状正则接入
- 文件：`training/episode.py`、`config/hyperparams.py`
- 改动：新增 Q 形状项并纳入总损失：
  - `dQ/dz > 0`
  - 低杠杆区 `dQ/db > 0`
  - 高杠杆区 `dQ/db < 0`
- 新增超参：
  - `q_shape_weight_z`
  - `q_shape_weight_b_low`
  - `q_shape_weight_b_high`
  - `q_shape_b_low`
  - `q_shape_b_high`

### 8.5 训练文档同步
- 文件：`training/README.md`
- 改动：新增 Q-first 训练行为说明与超参数开关说明。

## 9. 变更后最小可运行验证（smoke）
- 测试设置：`epochs=3`, `q_pretrain_epochs=2`, `batch_size=64`, sample 模式小样本。
- 结果：
  - 训练正常完成，无 NaN/崩溃。
  - `final_losses` 中出现新增 Q 诊断项：`q_main/q_shape_z/q_shape_b_low/q_shape_b_high`。
  - `loss_history` 长度：`q=12`、`p0=4`、`pi=4`，验证了前 2 个 epoch 为 `Q-only`，最后 1 个 epoch 为联合训练。

## 10. Notebook 参数同步
- 文件：`tests/sdf_fc1_two_modes_test.ipynb`
- 已把 `hp_pv` 默认参数同步为 Q 优先配置（含 `q_pretrain_epochs`、`M` 截断、Q 形状正则），便于直接复现实验。

## 11. 第二轮 Q 修复（已落地）

### 11.1 Q-only 阶段改为“冻结非 Q 参数”
- 文件：`training/episode.py`
- 逻辑：当进入 Q-only 预训练阶段且 `q_freeze_non_q_in_pretrain=True` 时：
  - 保持 `bar_i/bp/bar_z` 由网络输出，不改写经济方程
  - 冻结非 Q 参数，仅训练 Q 相关参数
  - 可训练范围由 `q_pretrain_trainable_scope` 控制（`q_head_only` / `q_path`）
- 目的：在不破坏方程结构前提下，减少未收敛分支对 Q 的干扰。

### 11.2 加入结构化 warm-start 监督
- 文件：`training/episode.py`、`config/hyperparams.py`
- 新增：`q_warmstart_epochs`, `q_warmstart_weight`, `q_warm_A`, `q_warm_b_star`, `q_warm_sigma`, `q_warm_alpha_z`, `q_warm_alpha_x`
- 逻辑：在 `epoch < q_warmstart_epochs` 时，额外加入 `MSE(Q, Q_warm_target)`。
- 目标函数形状：`Q_warm_target` 对 `b` 是高斯峰并乘 `b_nonneg`，同时对 `z,x` 采用指数风险项。

### 11.3 阶段切换规则升级
- 文件：`training/episode.py`
- 逻辑：Q-only 阶段长度改为 `max(q_pretrain_epochs, q_warmstart_epochs)`，确保 warm-start 期间不被 `p0/pi` 干扰。

### 11.4 新增 Q 诊断项
- 文件：`training/episode.py`
- 每步输出新增：
  - `q_physics`
  - `q_warmstart`
  - `q_warm_weight`
  - `q_pretrain_mode`
  - `q_freeze_mode`

### 11.5 最小验证（smoke）
- 3 个 epoch 小样本测试可运行，`q` 为有限值。
- `loss_history` 显示阶段标记按预期切换：
  - `q_pretrain_mode = [1.0, 1.0, 0.0]`
  - `q_freeze_mode = [1.0, 1.0, 0.0]`
  - `q_warm_weight = [1.0, 1.0, 0.0]`
