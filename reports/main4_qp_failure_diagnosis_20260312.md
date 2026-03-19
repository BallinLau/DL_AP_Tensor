# Q/P 训练异常诊断（结合 `main_4.tex` 与当前实现）

## 1. 你当前看到的现象（来自 notebook）

在 `tests/sdf_fc1_two_modes_test.ipynb` 中：
- `Q` 极小：`PV Mode2` 下 `Q mean=1.8323e-05`；`PV Mode1` 也仅 `5.47e-04`。
- `bar_z` 几乎为 0：`PV Mode2` 下 `bar_z mean=1.9789e-05`，`PV Mode1` 近似 0。
- `bp` 接近 1：`PV Mode2` 下 `bp mean=0.999362`。
- `q_shape` 项几乎为 0（`1e-7` 量级），说明形状约束在数值上几乎不起作用。

证据：
- `/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/tests/sdf_fc1_two_modes_test.ipynb:2046`
- `/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/tests/sdf_fc1_two_modes_test.ipynb:2051`
- `/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/tests/sdf_fc1_two_modes_test.ipynb:2049`
- `/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/tests/sdf_fc1_two_modes_test.ipynb:1989`

---

## 2. 对你第 3 点的直接回应：`residual product` 是否 AiO？

你的说法“两个 child residual 相乘就是 AiO”是 **部分正确** 的。

- 在论文口径里，AiO 核心项是 `E[r(S,S') * r(S,S'')]`（例如 `L_Q^(0), L_P^(0), L_P^(I)`）。
  - 证据：`/Users/ballinliu/Desktop/PHD/Project1/DL Equilibrium/main_4.tex:1361`
  - 证据：`/Users/ballinliu/Desktop/PHD/Project1/DL Equilibrium/main_4.tex:1419`

- 你当前 `episode` 里确实用了“双分支 residual 的乘积”，但实现是 `abs(prod)` 再平均：
  - `P0`: `combined = residuals_stack.prod(...); main_loss = combined.abs().mean()`
    - `/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/training/episode.py:657`
    - `/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/training/episode.py:658`
  - `PI`: 同样是 `abs(prod)`
    - `/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/training/episode.py:774`
    - `/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/training/episode.py:775`

结论：
- “两 child 相乘”这一点是 AiO 结构没错；
- 但 `abs(prod)` 与论文写的“有符号乘积期望”并不等价，优化几何会变（尤其在残差异号时）。
- 所以你说“使用了 AiO”可以成立，但严格来说是 **AiO 的变体目标**，不是论文原式逐字落地。

---

## 3. 为什么会出现你说的三个问题

## 3.1 Q 数值太小、b-z 关系不对

### 经济学角度
`main_4.tex` 中 Q 的结构化先验是：
\[
Q(b,z,x)=\min\{1, A\exp(-(b-b^*)^2/(2\sigma^2))\cdot(1+\alpha_z z+\alpha_x x)\}
\]
这会给一个“有高度的倒 U”初始形状（峰值附近并不接近 0）。
- 证据：`/Users/ballinliu/Desktop/PHD/Project1/DL Equilibrium/main_4.tex:1658`

### 代码角度（关键）
你当前 warm-start 目标多乘了一个 `b_nonneg`：
- `q_warm_target = A * b_nonneg * gaussian_peak * risk_term`
- 证据：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/training/episode.py:959`

这会把整个目标幅度显著压低：
- 在 `b*≈0.05` 附近，目标上限也只到 `~0.03-0.05` 量级；
- 在中高杠杆区，因高斯项迅速衰减，目标几乎贴近 0。

因此 Q 很容易被预热到“近零平台”，后续再拉起来就很难。你的结果与这个机制高度一致。

另外两个放大因素：
- Q-only 阶段你设置了 `q_head_only`，冻结了共享层，Q 头只能在固定随机特征上拟合，学习形状能力受限。
  - 代码：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/training/episode.py:236`
  - notebook 设置：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/tests/sdf_fc1_two_modes_test.ipynb:1248`
- 形状惩罚只约束导数符号，不约束导数大小；“几乎常数的小 Q”也能轻松满足，故 `q_shape_*` 接近 0。
  - 证据：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/training/episode.py:931`

---

## 3.2 P 没有 0 区域（几乎不破产）

### 经济学角度
理论是
\[
P=\max\{0,\int \max(P^0,P^I)dH(i)\}
\]
也就是说 `P^0/P^I` 本身应允许为负，最终再由外层 `max(0,·)` 截断。
- 证据：`/Users/ballinliu/Desktop/PHD/Project1/DL Equilibrium/main_4.tex:399`

### 代码角度（关键）
当前实现把负值通道基本堵住了：
1. `PHead` 输出激活是 `softplus`，结构上强制 `P0, PI > 0`。
   - 证据：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/models/share_layer.py:152`
2. `cal_phats` 再做 `P = clamp(Phat, min=0)`。
   - 证据：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/models/policy_value.py:173`
3. `bar_z` 由 `sigmoid(-50*P)` 推出；当 `P` 总是正时，`bar_z` 必然接近 0。
   - 证据：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/models/policy_value.py:175`

这三步叠加后，“违约边界”几乎不可能出现，与你观测的 `bar_z≈0` 完全一致。

---

## 3.3 P0/PI 的 AiO 与论文一致性争议

### 一致的部分
- 确实使用了双 child residual 乘积结构（AiO 核心思想）。

### 不一致或偏离的部分
1. 使用 `abs(prod)`，不是论文展示的有符号乘积期望。
2. `PI` 分支下一期杠杆更新没有用论文的
   \(b_{t+1}=\eta b'_t+(1-\eta)b_t\)，而是直接 `b_{t+1}=bpI`。
   - 理论：`/Users/ballinliu/Desktop/PHD/Project1/DL Equilibrium/main_4.tex:2331`
   - 代码：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/training/episode.py:739`
3. Q 方程注释口径写的是 `b' = b/(...)`，但实际给 `Qsp` 的 `b` 输入是 `b_parent`（`childsp_state[:,0]=b_parent`），这会扭曲 b 维关系学习。
   - 代码：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/losses/q_loss.py:86`
   - 代码：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/training/episode.py:877`

---

## 4. 额外会“推向错误形状”的训练机制

1. `z_penalty` 对高 z 残差加权（`sigmoid(beta*(z-z0))`, `z0=1`），低 z/高 b 的违约区反而权重较弱。
   - 代码：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/losses/utils.py:37`
2. 当前参数下 `x` 均值约为 `-2`，`eta=1` 概率仅 `0.03`，导致“再融资驱动的债务调整信号”本来就稀疏。
   - 代码：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/config/constants.py:36`
   - 代码：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/config/constants.py:55`
3. 训练样本中 `b` 均匀采样于 `[0,1]`，但缺少“贴边/违约邻域”的额外重采样，会让边界学习慢。
   - 代码：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/data/data_utils.py:232`

---

## 5. 总结（对应你的 3 点）

1. **Q 太小且形状不对**：主要由 warm-start 目标缩放过小（多了 `*b`）+ `q_head_only` 冻结共享层 + 仅符号型形状惩罚共同造成。
2. **P 无 0 区域**：`P0/PI` 被 `softplus` 和 `clamp(min=0)` 双重正值约束，违约通道被结构性抑制，`bar_z` 必然接近 0。
3. **AiO 争议**：你说“用到了 AiO 核心结构”是对的；但当前是 `abs(prod)` 变体，且还有 `PI` 杠杆转移和 `Qsp` 输入口径偏差，和 `main_4.tex` 的严格公式仍有关键差异。

