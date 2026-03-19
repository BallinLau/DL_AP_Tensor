# P/Q 联立贝尔曼训练诊断（基于当前 notebook 与 `main_4.tex`）

日期：2026-03-12

## 1) 当前 notebook 告诉我们的关键信号

来自 `tests/sdf_fc1_two_modes_test.ipynb` 最新输出：

- Mode1（sample）
  - `Q mean ≈ 0.6057`
  - `P0 mean ≈ -0.3465`, `PI mean ≈ -0.5269`
  - `bar_z mean ≈ 0.4744`（有明显破产区）

- Mode2（simulate）
  - `Q mean ≈ 0.0441`（明显偏小）
  - `P0 mean ≈ +0.1697`, `PI mean ≈ -0.0464`
  - `bar_z mean ≈ 0.0231`（几乎无破产）

- Mode2 的 grid 支持集：`b in [0, 0.4998]`（缺失高杠杆区）

结论：不是“模型只会输出不破产”，而是 **mode2 的训练分布把违约区域几乎抹掉了**，Q 的右半边曲率也失去识别。

---

## 2) 从方程看：P 与 Q 为什么会一起偏

理论中（`main_4.tex`）
- Q 方程：`f_Q` 里 survival/default 两条分支由 `I{z' > zbar}` 切换；
- P 方程：`P0/PI` 的 continuation 也乘 `I{z' >= zbar}`；
- FOC：依赖 `∂CF/∂b'` 和 `∂P'/∂b'`。

所以 P/Q 是强耦合系统：
1. `bar_z` 学不好 -> Q 的 default/survival 切换失真；
2. Q 学偏 -> `CF0/CFI` 偏 -> P 继续偏；
3. FOC 梯度不稳 -> `bp` 不可识别 -> 状态转移支持集塌缩 -> 两个方程都在“错误局部均衡”里自洽。

---

## 3) 这次 mode2 异常的主因（代码层）

## A. 训练数据支持集塌缩（最重要）

1. SimulateTS 初始杠杆默认只采样 `[0, 0.5]`
- `data/simulate_ts.py:549`

2. 新进入企业直接 `b=0`
- `data/simulate_ts.py:529`

3. mode2 网格也验证了 `b_max≈0.5`
- `tests/sdf_fc1_two_modes_test.ipynb`（输出中 `grid setup: b in [0, 0.4998]`）

影响：
- Q 的 `b>=1` 边界项几乎无样本（你结果里 `q_bdry_high=0`）；
- 违约高风险区缺样本，`bar_z` 被动收缩到接近 0；
- Q 的倒U右半边无法被识别，只会学到低杠杆左侧。

## B. 过渡方程里 η 的时点使用混乱（会削弱政策可识别）

理论写的是 `b_{t+1} = η_t b'_t + (1-η_t)b_t`（当前期 η）。
但当前数据/训练中多处用了 child 的 `η_{t+1}` 去更新 `b_{t+1}`：
- `data/sample.py:768-769`
- `data/simulate_ts.py:392` + `:405`
- `training/episode.py` 的 child state 构造也使用 `eta_child`

影响：
- `b'` 到下一期价值的映射变得噪声化，FOC 的经济含义被稀释；
- P/Q 对政策变量 `bp` 的梯度有效性下降。

## C. FOC 梯度很容易“静默断开”

在 `compute_foc_residual_from_bp` 中，`allow_unused=True`，一旦图断开就直接置 0：
- `losses/p0_loss.py:191-211`
- `losses/pi_loss.py:185-205`

如果 `η` 稀疏（`ZETA=0.03`）+ 支持集低杠杆，常见结果是 `∂P'/∂bp` 很小或为 0，FOC 项几乎不起作用。

## D. `P/bar_z` 链路有强饱和，梯度区间很窄

- `P = clamp(Phat, min=0)`：当 `Phat<0` 时梯度对 `Phat` 为 0；
- `bar_z = sigmoid(-50*P)`：温度过硬，`P` 远离 0 时几乎饱和，梯度很小。

位置：`models/policy_value.py:173-175`

影响：
- default boundary 附近可学，远离边界梯度弱；
- 在 mode2 低杠杆/低违约分布下，`bar_z` 更容易被推到接近 0 并锁住。

## E. `CF` 的股权融资成本符号实现存在可疑点

当前实现：
- `cf = cf_raw - kappa_e * relu(-cf_raw) * sign(cf_raw)`
- 当 `cf_raw<0` 时，该式会让 `cf`“变得没那么负”（更接近 0）

位置：
- `losses/p0_loss.py:95-97`
- `losses/pi_loss.py:104-106`

这通常会 **抬高** 股权价值、降低破产概率，方向上与你观察到的“mode2 变得不破产”一致。

---

## 4) 训练两个贝尔曼方程时，真正要盯的点

## (1) 支持集覆盖（比总 loss 更关键）
- 训练批次中必须持续覆盖：
  - 高杠杆区（接近 `b=1`）
  - 违约边界邻域（`z≈zbar`）
  - `η=1` 样本（FOC识别关键）
- 否则“看起来 loss 很低”，实际只是学了一个低风险子空间。

## (2) 过渡方程时间索引必须统一
- `b_{t+1}` 更新、损失残差里的 `η`、模拟器里的 `η` 要一致使用同一时点定义。
- 这是联立固定点求解的最基础一致性。

## (3) 避免硬截断导致梯度死区
- `clamp`/超陡 sigmoid 建议改为温和可微近似（或温度退火）。
- 尤其 `bar_z` 作为 Q/P 的耦合桥梁，不能一开始就近似硬阈值。

## (4) FOC 不能“有名无实”
- 监控 `||dCF/db'||`、`||dP'/db'||` 的批均值与分位数；
- 若长期接近 0，说明 FOC 项不可识别，需要先修支持集/过渡映射。

## (5) 训练顺序建议（联立而非混训）
- 阶段A：固定/缓慢更新 `bar_z`，先把 Q 的边界和形状立住；
- 阶段B：固定 Q，训练 P0/PI + FOC，恢复 `bp` 经济含义；
- 阶段C：联合小步交替（Q一步，P一步）做 fixed-point 微调。

这比“一锅端同时更新”更稳，尤其在 endogenous simulation 反馈很强时。

---

## 5) 对你当前现象的直接解释（简版）

- 为什么 mode2 下 P 又“不破产”？
  - 因为 mode2 数据把 `b` 压到 `[0,0.5]`，进入者又是 `b=0`，违约样本天然稀缺；
  - 再叠加 `CF` 符号与 `bar_z` 饱和问题，P 更容易整体偏正。

- 为什么 Q 还不理想？
  - Q 在 mode2 只看到低杠杆左半边，`b>=1` 边界几乎无约束样本；
  - 与 P 的 default 切换信号弱耦合后，倒U右半边学不出来，均值就偏小。

