# P0/PI 中 bp 的 KKT 边界条件：理论推导与代码落地（2026-03-13）

## 1. 问题背景

在本项目中，`bp`（下一期目标杠杆）是通过 Policy 网络输出，并由 `sigmoid` 限制在区间 `[0,1]`。

这意味着在每个给定状态下，P0/PI 子问题本质是一个**有界控制变量优化**：

\[
\max_{0\le b'\le 1} J(b'; s)
\]

其中 `s` 表示状态（如 \(b,z,\eta,i,S\)），\(J\) 是 P0 或 PI 的 Bellman 右端目标。

因此最优条件不能仅使用“无约束 FOC=0”，而应使用 KKT 条件。

---

## 2. KKT 推导（对应你提出的三条条件）

将最大化问题改写为最小化：

\[
\min_{b'} f(b')=-J(b')
\]

约束写为：

\[
c_1(b')=-b'\le 0,\qquad c_2(b')=b'-1\le 0
\]

Lagrangian：

\[
\mathcal{L}(b',\lambda_1,\lambda_2)= -J(b')+\lambda_1(-b')+\lambda_2(b'-1),
\quad \lambda_1,\lambda_2\ge 0
\]

KKT 条件：

1. Stationarity
\[
\frac{\partial \mathcal L}{\partial b'}=0
\Rightarrow -J'(b')-\lambda_1+\lambda_2=0
\Rightarrow J'(b')=\lambda_2-\lambda_1
\]

2. Primal feasibility
\[
0\le b'\le 1
\]

3. Dual feasibility
\[
\lambda_1,\lambda_2\ge 0
\]

4. Complementary slackness
\[
\lambda_1 b'=0,\qquad \lambda_2(b'-1)=0
\]

由此得到三种情形：

- 内点 \(0<b'<1\)：\(\lambda_1=\lambda_2=0\Rightarrow J'(b')=0\)
- 下边界 \(b'=0\)：\(\lambda_2=0,\lambda_1\ge0\Rightarrow J'(0)=-\lambda_1\le0\)
- 上边界 \(b'=1\)：\(\lambda_1=0,\lambda_2\ge0\Rightarrow J'(1)=\lambda_2\ge0\)

即：

- 内点：`FOC = 0`
- 下边界：`FOC <= 0`
- 上边界：`FOC >= 0`

这正是你要求加入的边界最优性条件。

---

## 3. 与当前项目损失的对齐方式

项目中 `P0/PI` 已计算了基于 `bp` 的 FOC 残差（对 \(\partial CF/\partial b'\) 与 \(\partial P'/\partial b'\) 的自动微分组合）。

### 3.1 KKT 边界最优性（bp 的有界控制）

本次改动在 `Episode` 层新增了 KKT 罚项构造：

- 输入：`bp` 与 `foc_residuals`（按分支）
- 先对分支残差取均值，得到 signed 的 \(\overline{FOC}\)
- 使用软边界权重（sigmoid）区分三块区域：
  - `w_inner`：内点区域
  - `w_low`：接近 0 的下边界区域
  - `w_high`：接近 1 的上边界区域

罚项定义：

\[
\mathcal L_{\text{KKT,inner}} = \mathbb E\big[w_{inner}\cdot (\overline{FOC})^2\big]
\]
\[
\mathcal L_{\text{KKT,low}} = \mathbb E\big[w_{low}\cdot \max(\overline{FOC},0)\big]
\]
\[
\mathcal L_{\text{KKT,high}} = \mathbb E\big[w_{high}\cdot \max(-\overline{FOC},0)\big]
\]

总 KKT 项：

\[
\mathcal L_{\text{KKT}}
= \omega_{in}\,\mathcal L_{\text{KKT,inner}}
+ \omega_{bd}\,(\mathcal L_{\text{KKT,low}}+\mathcal L_{\text{KKT,high}})
\]

再分别加到 P0/PI loss：

\[
\Xi^{P0}_{new}=\Xi^{P0}_{old}+\lambda^{P0}_{KKT}\,\mathcal L_{KKT}
\]
\[
\Xi^{PI}_{new}=\Xi^{PI}_{old}+\lambda^{PI}_{KKT}\,\mathcal L_{KKT}
\]

### 3.2 FOC 改为“条件在再融资事件上的 signed moment”

由于理论中的 FOC 项带有 `\eta`（参见主文 FOC 公式），本次把 FOC 损失改为条件矩：

\[
\mathbb E[\text{FOC} \mid \eta=1]=0
\]

实现上：

1. 保留 FOC 的符号（signed），不再使用 `compute_aio_residual` 的平方+`abs(product)` 形式；
2. 先按分支对样本内有效再融资事件做条件平均，得到每个样本的 signed FOC；
3. 再对 active 样本求条件矩并平方作为损失：

\[
\mathcal L_{\text{FOC,cond}}=\left(\mathbb E[\text{FOC}_{\text{signed}}\mid \eta=1]\right)^2.
\]

这样避免了 `eta=0` 样本占多数时对 FOC 梯度的稀释，并保持了符号信息与理论一致。

---

## 4. 代码改动点

1. 超参数（新增）
- `config/hyperparams.py`
  - `p0_kkt_weight`
  - `pi_kkt_weight`
  - `kkt_boundary_eps`
  - `kkt_boundary_temp`
  - `kkt_inner_weight`
  - `kkt_boundary_weight`

2. KKT 核心实现
- `training/episode.py`
  - 新增 `Episode._compute_bp_kkt_penalty(...)`
  - 该函数支持 `eta_children`，KKT 统计也按 active（`eta=1`）样本条件化。

3. FOC 条件矩实现（新增）
- `training/episode.py`
  - 新增 `Episode._compute_conditional_signed_foc_terms(...)`
  - 以 `E[FOC|eta=1]=0` 构造 `loss_foc`，并保留 signed moment 诊断。

4. 接入 P0/PI loss
- `training/episode.py`
  - `_compute_p0_loss(...)` 中将 `kkt_penalty` 加入 `total_loss`
  - `_compute_pi_loss(...)` 中将 `kkt_penalty` 加入 `total_loss`
  - 两处 FOC 项均改为调用 `_compute_conditional_signed_foc_terms(...)`（不再经 `compute_aio_residual` 聚合 FOC）。

5. 训练日志诊断项
- P0 新增：
  - `p0_kkt`, `p0_kkt_inner`, `p0_kkt_low`, `p0_kkt_high`, `p0_kkt_foc_abs_mean`, `p0_kkt_active_ratio`
  - `p0_foc_active_ratio`, `p0_foc_signed_moment`, `p0_foc_cond_abs_mean`
- PI 新增：
  - `pi_kkt`, `pi_kkt_inner`, `pi_kkt_low`, `pi_kkt_high`, `pi_kkt_foc_abs_mean`, `pi_kkt_active_ratio`
  - `pi_foc_active_ratio`, `pi_foc_signed_moment`, `pi_foc_cond_abs_mean`

---

## 5. 为什么这是“理论一致”的改法

- 没有额外引入“把 bp 压小”的经验正则（例如 `||bp||^2` 或 `bp-b` 锚定）作为主导。
- 仅在 P0/PI 的最优性条件内，补齐了有界控制的 KKT 互补条件。
- FOC 使用条件矩 `E[FOC|eta=1]=0`，与主文中 `eta` 因子一致，且保持符号信息。
- 本质是在“FOC + 边界可行域”上完成闭环，符合主文 Bellman/FOC 框架。

---

## 6. 已完成检查

- `python3 -m py_compile config/hyperparams.py training/episode.py` 通过。
