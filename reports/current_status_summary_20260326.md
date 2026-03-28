# 当前阶段总结（2026-03-26）

## 1. 当前已经确认的问题

### 1.1 `M` 分布不合理

最开始观察到：

- `M` 在图上大量堆积在 `0` 附近
- 同时存在 `1` 附近的尖峰和右侧长尾

后续排查确认：

- 最早一部分异常来自可视化口径问题：
  - `firm panel` 重复计数
  - `parent` 与 `child` 混画
- 修正口径后，真正的问题仍然存在：
  - `child macro states` 的 `M` 分布本身就不合理
  - 并不是围绕 `1` 附近的单峰正值分布

### 1.2 `bar_i` 出现在高 `b`、低 `z` 区域

当前结果里：

- `bar_i` 偏向高杠杆、低生产率区域
- 低杠杆、高生产率的安全区域反而投资偏弱

这与正常“低 `b`、高 `z` 更容易投资”的理论直觉不一致。

### 1.3 `bp` 偏大且容易贴边

`bp` 长期偏高、靠近边界，说明当前 `PI/P0` 的比较中混入了较强的融资操作效应，而不只是纯粹的投资价值比较。

### 1.4 当前实现没有清楚区分 conditional / unconditional value

这是目前最核心的理论问题。

现有代码里：

- `P0 / PI` 一边承担 Bellman 递推
- 一边又通过 `Phat -> P / bar_z` 进入当前期 default 逻辑

因此对象定义混在了一起，没有严格区分：

1. 当前期存活条件下的 continuation value
2. 考虑当前期 default 后的总股权价值

## 2. 目前已经完成的修正与诊断

### 2.1 可视化口径修正

已经完成：

- `M` 直方图改为优先使用 `df_macro["M"]`
- `M` 主图只画 `child macro states`
- `parent macro states` 单独成图
- `bp` 只画 parent states
- 新增 `PI/P0` 诊断图：
  - `pidiff`
  - `cfdiff`
  - `contdiff`
- 新增 final simulate 的中段 `b` 分布图：
  - `final_b_hist_t50_t150_parent.png`

此外，投资相关 `(b,z)` 图现在已加 current-survival mask：

```math
P_t > 0 \quad \text{and} \quad \bar z_t < 0.5
```

也就是说：

- `bari`
- `bp`
- `pidiff`
- `cfdiff`
- `contdiff`

只显示当前期仍存活的区域，避免把当期已经 default 的状态误画进投资区。

### 2.1.1 `bp` 截面诊断图现在怎么看

在 `codex/conditional-value-refactor` 分支中，新增了一组固定 parent 状态下的 `bp` 截面诊断图：

- `ep*_bp_diag_safe.png`
- `ep*_bp_diag_mid.png`
- `ep*_bp_diag_risky.png`
- `ep*_bp_diag_distress.png`

每张图固定一个 parent 状态 `(b,z,\eta,i,x,\hat c^f,\ln K^f)`，只沿 `bp \in [0,1]` 扫描，目的是回答：

```math
\text{为什么当前模型会把最优 } bp \text{ 选到很高，甚至把下一期推到 default 边界？}
```

图中的六个子图分别表示：

1. `Q(bp)`  
   - 当前实现中的总债价值，而不是单位债价格。  
   - 若它在高 `bp` 区仍然上升，说明“发更多债的总融资额”仍在增加。

2. `q_unit(bp) = Q(bp)/bp`  
   - 单位债价格。  
   - 这个量比 `Q(bp)` 更直接反映债务定价是否已经显著惩罚高杠杆。

3. `P_{t+1}(bp)`  
   - 固定当前 parent 状态后，给定 child `bp` 时的下一期总股权价值。  
   - 若这个曲线在某个 `bp` 后迅速掉到 0，说明高杠杆会把企业推到下一期股权清零区域。

4. `\bar z_{t+1}(bp)`  
   - 下一期违约概率代理。  
   - 若与 `P_{t+1}(bp)` 同时快速上升/恶化，说明高 `bp` 会显著提高下一期 default 风险。

5. `CF0(bp)` / `CFI(bp)` 与虚线 `V0/VI`  
   - 这里把“当期现金流部分”和“一步总条件价值”放在同一个面板里。  
   - 实线：
     - `CF0(bp)`
     - `CFI(bp)`
   - 虚线：
     - `V0(bp)`
     - `VI(bp)`
   - 若虚线最优点落在高 `bp`，而实线仍在上升，则说明股东很可能是被“当前融资收益”推着往高杠杆走。

6. `cont0(bp)` / `contI(bp)`  
   - continuation 项本身。  
   - 若它们在高 `bp` 已经明显恶化，但总值仍偏向高 `bp`，就说明当前融资收益仍压过未来损失。

后续又新增了两个 `CF decomposition` 面板，用来继续拆 `CF0/CFI` 的来源：

7. `CF0 decomposition`  
   - `prod(bp)`：税后经营利润  
   - `debt_adj0(bp)`：不投资分支的净发债收入  
   - `-eq_cost0(bp)`：股权融资成本的负向贡献  
   - `CF0(bp)`：以上三项合成后的 no-invest 当前现金流

8. `CFI decomposition`  
   - `prod(bp)`：税后经营利润  
   - `debt_adjI(bp)`：投资分支的净发债收入  
   - `-i(bp)`：投资成本  
   - `-eq_costI(bp)`：股权融资成本的负向贡献  
   - `CFI(bp)`：以上几项合成后的 invest 当前现金流

这两个分解面板的作用是判断：

```math
\text{到底是哪一项把 safe 区的 } V(bp) \text{ 推向高杠杆。}
```

如果观察到：

- `prod(bp)` 基本平缓甚至下降
- `debt_adj0(bp)` / `debt_adjI(bp)` 随 `bp` 明显上升
- 而 `CF0(bp)` / `CFI(bp)` 的峰值主要跟着 `debt_adj` 走

那么就可以更明确地判断：

```math
\text{当前 safe 区高 } bp \text{ 的主因是净发债收入，而不是经营利润或 continuation。}
```

后续又继续加入了两个导数面板：

9. `Derivative decomposition: V0`  
   - `dCF0/dbp`
   - `dcont0/dbp`
   - `dV0/dbp`

10. `Derivative decomposition: VI`  
   - `dCFI/dbp`
   - `dcontI/dbp`
   - `dVI/dbp`

这两张图回答的是一个更精确的问题：

```math
\text{最优 } bp \text{ 到底是被 value 的哪个边际项推出来的？}
```

它们比只看 `V(bp)` 的水平更重要，因为最优点取决于：

```math
\frac{dV}{dbp}
=
\frac{dCF}{dbp}
+
\frac{dCont}{dbp}
```

因此：

- 如果低 `bp` 区 `Cont(bp)` 水平很大，但 `dCont/dbp` 始终为负且很快衰减
- 同时在最优点附近 `dCF/dbp` 仍然为正并主导

那么就说明：

```math
\text{continuation 在左侧确实重要，但最优点附近的边际仍主要由 } CF \text{ 决定。}
```

这能避免把“continuation 水平在左边很大”误读成“最优点一定由 continuation 主导”。

图上的三条竖线分别是：

- `bp0*`：不投资分支最优候选
- `bpI*`：投资分支最优候选
- `bp*`：最终混合后的实际执行杠杆

### 2.1.2 如何用这张图判断问题

最关键看三件事：

1. `P_{t+1}(bp)` 是否在 `bp*` 附近已经接近 0。  
   - 如果是，说明当前最优杠杆正在主动把企业推到下一期股权清零边界。

2. `\bar z_{t+1}(bp)` 是否在 `bp*` 附近已经很高。  
   - 如果是，说明模型在选择“高 default 风险的融资点”。

3. `q_unit(bp)` 是否已经明显下降，但 `Q(bp)` 或 `CF` 仍让股东受益。  
   - 这意味着：
     - 债权人已经开始要求更高违约补偿
     - 但股东仍然因为总融资额/有限责任而偏好高 `bp`

如果同时观察到：

- `P_{t+1}(bp*) \approx 0`
- `\bar z_{t+1}(bp*)` 很高
- `CF0/CFI` 仍支撑高 `bp`

那么就可以把问题解释为：

```math
\text{股东在有限责任下做 risk shifting / gambling for resurrection，}
```

也就是：

- 当前融资收益还在增加
- 未来股权损失在 `P_{t+1} \to 0` 后被截断
- 债权损失主要由债权人承担

这类诊断图的价值就在于：它不再只告诉我们“`bp` 很大”，而是明确告诉我们：

- 是债务定价没把高杠杆打下来
- 还是 continuation 被低估
- 还是当期融资现金流项过强，压过了未来违约代价

### 2.1.3 新增的 `argmax(V)` 与 `FOC/KKT` 截面该怎么看

后续又在同一组 `bp` 诊断图中加入了：

- `argmax V0`
- `argmax VI`
- `FOC0(bp)` / `FOCI(bp)`
- `KKT0 point penalty` / `KKTI point penalty`

它们的用途是：判断**网络给出的 `bp0* / bpI*`，到底是不是它自己训练目标意义下的最优解**。

#### `argmax V0 / argmax VI`

这里的：

## 3. `Q` 训练链当前的新判断

从最近的 `safe state` 诊断图看，`Q` 这条线已经暴露出一个独立问题：

- `q_unit(bp)` 几乎是一条接近常数的小正线
- 量级只有 `1e-4 ~ 1e-5`
- `Q(bp)=bp \cdot q_unit(bp)` 只是机械地随 `bp` 线性增加一点
- `CF0/CFI` 里的 `debt_adj` 基本消失，导致 `dCF/dbp \approx 0`

这说明当前问题已经不只是 recovery 规格或 `bp` surrogate 的问题，而是：

```math
q\_head \text{ 很可能没有真正学出 } Q(b,z,\eta,x,\hat c,\ln K)
```

### 3.1 为什么怀疑是 `Q` 自己没学起来

在当前 `q_loss` 里：

- `M` 已被 clamp 到 `[0.5, 1.5]`
- `safe state` 下 `P_{t+1}` 仍为正
- `bar_z` 也未接近 1

按这种方程环境，`Q` 正常不应普遍塌到 `1e-5` 量级。

因此更像是：

- `q_head` 在当前共享特征上学成了近常数的小正值
- 而不是方程自然推出了极小 `Q`

### 3.2 为什么默认 `q_head_only` 容易出问题

此前默认配置是：

- `q_pretrain_epochs = 0`
- `q_warmstart_epochs = 0`
- `q_pretrain_trainable_scope = q_head_only`

这意味着：

- 没有任何 `Q-only` 的独立学习窗口
- 即使开预训练，也只允许 `q_head` 自己在 frozen `share_layer` 特征上拟合

如果共享表征主要是为 `P/bp` 服务的，那么 `Q` 很容易学成：

```math
q_{unit}(h) \approx \text{一个接近常数的小正数}
```

于是 `Q = b \cdot q_{unit}` 就会整体塌到很低量级。

### 3.3 本次默认改动

为了先把 `Q` 这条线单独救起来，当前默认口径改为：

- `q_pretrain_epochs = 10`
- `q_warmstart_epochs = 10`
- `q_pretrain_trainable_scope = q_path`

含义是：

1. 先给 `Q` 一个独立的预训练窗口
2. 预训练期允许 `share_layer + q_head` 一起适配
3. 不再要求 `q_head` 单独在 frozen 特征上硬拟合

### 3.4 为什么这个改动有效

#### 数学上

如果当前共享特征 `h` 本身对债券价值没有可分辨信息，那么即使 `q_head` 非线性再强，
它也只能在一个错误的表示上学出近常数解。

让 `q_path` 一起动，相当于同时优化：

```math
h_\theta(s), \quad q_\psi(h_\theta(s))
```

而不是只优化：

```math
q_\psi(h_{\text{frozen}}(s))
```

这会显著提高 `Q` 对状态和 `bp` 的辨识能力。

#### 经济上

只有当 `Q` 先恢复到合理量级并呈现合理形状时，
后面的：

- `debt_adj`
- `CF0/CFI`
- `bp` 最优债务选择

才有解释价值。

否则如果 `Q` 自己就是一条接近零的常数曲线，
后续所有关于“高杠杆是因为融资收益太强/太弱”的经济解释都会失真。

```math
\arg\max_{bp} V0(bp), \qquad \arg\max_{bp} VI(bp)
```

是用图中 one-step diagnostics 直接扫出来的数值最优点。

如果观察到：

- `bp0*` 很高
- 但 `argmax V0` 很低

或

- `bpI*` 很高
- 但 `argmax VI` 很低

那么这说明：

```math
\text{网络输出的 } bp^* \text{ 并没有实现图上这条 value curve 的 argmax。}
```

这种情况下，问题已经不只是“高杠杆看起来不合理”，而是：

- `bp` 头学到的 surrogate 目标
- 与图里计算出来的值函数目标

两者之间已经发生脱节。

#### `FOC0(bp)` / `FOCI(bp)`

这两条曲线现在直接复用了训练里的 FOC 定义：

- `FOC0(bp)`：不投资分支的债务选择一阶条件
- `FOCI(bp)`：投资分支的债务选择一阶条件

它们不是随手构造的辅助量，而是训练中真正用于 `bp` 学习的对象。

解读方式：

- 若某个内点 `bp` 是最优点，理论上对应的 `FOC(bp)` 应接近 `0`
- 若 `bp` 在下边界附近，理论上应满足下边界符号条件
- 若 `bp` 在上边界附近，理论上应满足上边界符号条件

因此：

- 若 `bp*` 靠近 1，但 `FOC(bp)` 在那附近并不支持上边界最优
- 或者 `FOC` 过零点与 `bp*` 相差很大

就说明训练 surrogate 本身和网络输出并不一致。

#### `KKT point penalty`

这两个量把训练里真正使用的 KKT 逻辑按点展开出来：

- `KKT0 point penalty`
- `KKTI point penalty`

它们越小，表示该 `bp` 越符合当前训练口径下的 KKT 条件。

所以最重要的对比是：

- `bp0* / bpI*` 是否落在 `KKT penalty` 较小的区域
- `argmax V0 / argmax VI` 是否也在这些区域附近

如果出现下面这种情况：

- `bp*` 很高
- `argmax V` 很低
- `KKT penalty` 在高 `bp` 也不低

那么可以直接判断：

```math
\text{当前 } bp \text{ 头既没有贴合 value argmax，也没有真正贴合训练的 KKT/FOC 结构。}
```

这时问题就不再只是经济解释，而是：

- `bp` 输出头
- 与训练损失 / surrogate

之间的实现一致性本身出了问题。

### 2.1.4 `bp` 的真实 value 导数与训练 surrogate 并不相同

后续在 safe-state 截面里进一步确认：

- `argmax V0 = 0`
- `argmax VI = 0`
- 但网络输出的 `bp0* / bpI*` 仍可能落在明显更高的区域

这说明当前 `bp` 头并不一定在实现：

```math
\arg\max_{bp} V_t(bp)
```

更准确地说，当前训练里的 `bp` surrogate 与真实 one-step value 的导数并不完全一致。

若把当前分支的一步价值写成：

```math
V(bp)=CF(bp)+M(bp)\,P(bp)\,(1-\bar z(bp))
```

那么严格导数应为：

```math
\frac{dV}{dbp}
=
\frac{dCF}{dbp}
+
\frac{dM}{dbp}P(1-\bar z)
+
M(1-\bar z)\frac{dP}{dbp}
-
MP\frac{d\bar z}{dbp}
```

而当前实现里的训练 surrogate 更接近：

```math
FOC^{impl}(bp)
=
\frac{dCF}{dbp}
+
M(1-\bar z)\frac{d\tilde P}{dbp}
```

其中：

```math
\tilde P=
\begin{cases}
\hat P, & \text{若 } bp\_foc\_use\_phat\_children=True \\
P, & \text{若 } bp\_foc\_use\_phat\_children=False
\end{cases}
```

因此 surrogate 相比真实导数，至少少了两项：

```math
\frac{dM}{dbp}P(1-\bar z)
```

和

```math
-MP\frac{d\bar z}{dbp}
```

同时在默认旧设置下，还把：

```math
\frac{dP}{dbp}
```

替换成了

```math
\frac{d\hat P}{dbp}
```

这会在 `\hat P<0`、而 `P=\max(0,\hat P)=0` 的区域带来最大的错配。

因此当前我们看到的现象，本质上可以写成：

```math
\arg\max_{bp} V(bp) \neq \arg\max_{bp} \widetilde V(bp)
```

而网络输出的 `bp^*` 更可能被后者推动。

### 2.1.5 当前最小修正：让 `bp` 的 FOC/KKT 先与 Bellman payoff 对齐

基于上面的错配，当前 refactor 分支已经做了一个最小改动：

```math
bp\_foc\_use\_phat\_children = False
```

也就是先让 `bp` 的 FOC/KKT 通道默认改为使用：

```math
P_{t+1}
```

而不是：

```math
\hat P_{t+1}
```

这样做的目的不是一次性把 `bp` 训练完全修正，而是先消掉最明显的一层对象错配：

- Bellman 主目标用的是 `P_{t+1}`
- `bp` 的 surrogate 也先改成用 `P_{t+1}`

这样至少保证：

```math
\text{value payoff object}
\quad \text{和} \quad
\text{bp surrogate object}
```

在最关键的 child-value 层面先一致起来。

### 2.2 `SimulateTS` 递推口径修正

已经修正：

- 不再把 `state["hatcf"] / state["lnkf"]` 回写成 realized `Hatc / LnK`
- `M` 只沿 forecast-state 递推

这样后，`child M` 的异常可以更明确地解释为 forecast-state 一步映射问题，而不是 realized 宏观量回写带来的口径混杂。

### 2.3 FC1 稳定性修正

已经做过的稳定性修改包括：

- 去掉 FC1 scaler 依赖
- stage2 以 forecast-state 为主进行重建
- `LnK` 重建项降权
- 增加 forecast-state 单步增量惩罚
- 增加 Jacobian / 局部平滑惩罚

结果上：

- forecast-state 递推的极端跳跃有所收敛
- `M` 分布有一定改善
- 但仍然没有恢复为理论期望的单峰形状

### 2.4 关于 `PI/P0` 诊断的关键结论

通过新增图形，目前看到：

1. `pidiff` 在高 `b`、低 `z` 区域大于 0
2. `cfdiff` 全局小于 0
3. `contdiff` 全局大于 0

这说明：

- 当期现金流口径下，投资几乎 everywhere 都吃亏
- 投资分支之所以还能在某些区域赢过不投资，主要靠 continuation 优势

但这里需要特别注意：

- continuation 用的是下一期**总股权价值** `P_{t+1}`
- 不是 `P_{t+1}^0 / P_{t+1}^I` 的分头 continuation

更准确地说，当前实现诊断的是：

```math
g\,M_t\,P_{t+1}(bp_I)\,(1-\bar z_{t+1}(bp_I))
-
M_t\,P_{t+1}(bp_0)\,(1-\bar z_{t+1}(bp_0))
```

## 3. 最新修改：`bp` 训练只让 child 存活区主导

### 3.1 问题的新判定

在新增的 `bp_diag_*` 与 `bp_boundary_*` 图里，已经多次观察到同一模式：

- `P_{t+1}(bp)` 在某个边界 `bp_c` 后迅速掉到 `0`
- `\bar z_{t+1}(bp)` 同时升到 `0.5` 以上
- `FOC` 在该边界附近发生明显跳变
- 但 `bp^*` 仍然可能落在 `bp_c` 右侧，甚至接近 `1`

这说明当前 `bp` 训练存在一个结构性问题：

```math
\text{default 区右侧的局部驻点，也被 surrogate 当成了“可接受最优”。}
```

换句话说，训练并没有区分：

1. child 仍然继续经营的存活区
2. child 已经进入下一期股权清零 / 高违约概率的死亡平台区

### 3.2 数学上的修正思路

设 child 存活权重为：

```math
w_{surv}(bp)=\chi_{t+1}(bp)
```

最硬版可写成：

```math
w_{surv}(bp)=\mathbf{1}\{P_{t+1}(bp)>0,\ \bar z_{t+1}(bp)<0.5\}
```

为保持可导，当前实现采用 soft 版：

```math
w_{surv}(bp)
=
\sigma(\tau_P P_{t+1}(bp))
\cdot
\sigma(\tau_z (z_{th}-\bar z_{t+1}(bp)))
```

其中：

- `\tau_P` 控制对 `P_{t+1}` 的门控斜率
- `\tau_z` 控制对 `\bar z_{t+1}` 的门控斜率
- `z_{th}` 当前取 `0.5`

于是 `bp` surrogate 的 pointwise 训练项改成：

```math
L_{bp}^{FOC}
=
w_{surv}\cdot (FOC(bp))^2
```

```math
L_{bp}^{KKT}
=
w_{surv}\cdot KKT(bp)
```

这一步的数学含义是：

- 当某个 `bp` 已经把 child 推到
  ```math
  P_{t+1}\approx 0,\quad \bar z_{t+1}\gtrsim 0.5
  ```
  时，它在 `bp` 训练里的权重会自动下降
- 训练将主要由左侧 still-alive 区域的 FOC/KKT 信号主导

因此它针对的是当前最具体的问题：

```math
\text{不是“平台区更优”，而是“平台区也被当作有效局部解”。}
```

### 3.3 经济学上的合理性

这一修正并不是技术性地“硬压低 `bp`”，而是把 `bp` 的经济意义重新对齐到：

```math
\text{当前企业若继续经营，应当如何选择下一期债务。}
```

如果某个 `bp` 已经把 child 直接推入：

- 下一期股权清零
- 极高违约概率
- continuation 基本消失

那么这个 `bp` 就不应该和正常 continue 区内的内点最优条件并列对待。

所以这一步的经济解释是：

- `bp` 的 FOC/KKT 应由 continue 区内的最优性主导
- default 区右侧的死亡平台，不应继续给 `bp` 输出头提供“正当性”

### 3.4 预期效果

若该修正有效，后续重新训练后应观察到：

1. `bp^*` 更少落在 `P_{t+1}=0` 的 crossing 右侧
2. `bp^*` 更接近 `argmax V0 / argmax VI`
3. `bp_boundary` 图里，右侧死亡平台对训练的牵引明显减弱

### 3.5 诊断图轻量化

为了避免 `bp` 诊断显著拖慢每个 episode，当前 refactor 分支同时把诊断逻辑切到了轻量版：

- 默认只画 `safe` 状态
- `bp_grid` 从 `201` 降到 `101`
- `run_multi_episode_job.py` 中改为每 `5` 个 episode 画一次，最后一轮强制画
- 诊断图中的 `FOC/KKT` 改成有限差分近似，而不再使用训练级别的 `autograd.grad`

这样做的原因是：

- 训练需要高质量梯度
- 诊断只需要看
  ```math
  \text{shape, crossing, argmax, jump}
  ```
  这些局部几何特征

因此有限差分已经足够，不需要继续在诊断阶段保留大计算图。

## 3. 当前最重要的理论澄清

### 3.1 Bellman 方程本身是 conditional-on-survival 的

理论文稿里：

- `P^0`
- `P^I`

本质上是“当前期存活条件下”的价值。

当前期是否 default，不应只乘在 continuation 上，而应在总股权价值这一层处理。

因此：

- 当前期若已破产，未来当然不应再有股权价值
- 但这个逻辑应由当前期 survival gate 作用在整个价值对象上，而不是只作用在未来项上

### 3.2 这正是当前代码最根本的混淆点

现在代码里：

- `P0 / PI` 同时承担了 Bellman 对象和总值对象的语义

这会导致：

- `bar_i = sigmoid(PI - P0)` 在当前其实已 default 的状态上也有数值
- 必须靠后处理 mask 才能把图画对

也就是说，图已经尽量修正了，但模型对象定义本身还没有完全理顺。

## 4. 当前最新方案

下一阶段最推荐的方案，是把对象分成两层：

### 第一层：survival-conditioned values

定义：

- `V0`：当前期存活条件下，不投资价值
- `VI`：当前期存活条件下，投资价值

Bellman 训练对象改为：

```math
V_t^0 = CF_t^0 + E_t[M_{t,t+1} P_{t+1}]
```

```math
V_t^I = CF_t^I + g E_t[M_{t,t+1} P_{t+1}]
```

这里右边继续用下一期总股权价值 `P_{t+1}`，这一点与理论一致。

### 第二层：unconditional total equity value

定义：

```math
\hat V_t = \int \max(V_t^0, V_t^I)\, dH(i)
```

```math
\chi_t = \text{当前期软存活门}
```

```math
P_t = \chi_t \hat V_t
```

数值上最小改法可以先保留：

```math
P_t = \max(0,\hat V_t)
```

同时把 `\chi_t` 与 `bar_z_t` 联系起来。

### 投资边界也分两层

推荐拆成：

```math
\bar i_t^{cond} = \sigma\big(\tau_i (V_t^I - V_t^0)\big)
```

```math
\bar i_t^{eff} = \chi_t \cdot \bar i_t^{cond}
```

含义：

- `bar_i_cond`：如果当前还活着，投不投资
- `bar_i_eff`：真正会被执行的投资决策

这样：

- parent 已经 default 时，`bar_i_eff = 0`
- 但仍保留 `bar_i_cond` 作为理论上的条件比较对象

## 5. 为什么这是目前最好的下一步

因为它同时解决了目前最核心的几个理论问题：

1. current default 与 continuation value 的层次被分开
2. `bar_i` 不再在已 default 区域拥有经济意义
3. `P0/PI` 不再同时承担 conditional 和 unconditional 两种语义
4. 当前图上看到的很多“反常投资区”，可以更清楚地区分究竟来自：
   - conditional investment incentive
   - 还是 current survival gate

## 6. 当前版本管理建议

不建议直接复制整套项目目录作为新版本管理方式。

更推荐：

1. 先把当前状态完整提交并推到 GitHub
2. 在 Git 里新建一个专门用于 conditional/unconditional 分层改造的分支
3. 如果你希望物理上分开两个工作目录，使用 `git worktree`

原因：

- 比直接复制目录更不容易丢失历史
- 更方便比较差异
- 更方便随时 cherry-pick 或回滚
- 不会出现两套目录独立漂移、难以同步的问题

如果需要物理隔离，建议用：

- 当前目录保留为稳定版本
- 新建一个 worktree 目录专门做结构改造

这比手工复制整个项目更适合作为科研代码的版本管理方式。

## 7. 每个 Episode 变慢的原因与优化

### 7.1 主要瓶颈不在训练主循环，而在 episode 末尾诊断

最近每个 episode 的 wall-clock 明显变长，排查后确认主要不是：

- `bp_survival_reweight`
- `FOC/KKT` 本体
- 或主训练 batch 的前向/反向

而是 **每个 episode 结束后的诊断与出图阶段**。

在 [run_multi_episode_job.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/experiments/run_multi_episode_job.py) 中，每轮 `run_episode()` 结束后还会继续执行：

- `save_stage_df`
- `save_models`
- `plot_surfaces`
- `plot_bp_diagnostic_curves`
- `plot_distributions`
- `plot_macro_series`

其中最重的是 `plot_bp_diagnostic_curves`，因为它会：

1. 固定 parent 状态，扫描一整条 `bp_grid`
2. 对每个网格点重新构造 child state
3. 重跑 `policy_value` 前向，必要时还重跑 `sdf_fc1.forward_step`
4. 继续生成 `Q/P/bar_z/CF/cont/V` 等曲线
5. 额外保存大尺寸 `diag` 和 `boundary` 图

这部分是 **episode 末尾额外的诊断成本**，而不是训练 loss 本身导致的 GPU 负担。

### 7.2 已做的轻量化

为了避免诊断拖垮训练，当前已经把 `bp` 诊断改成轻量版：

- 默认只画 `safe` 状态
- `bp_grid` 从 `201` 降到 `101`
- 多 episode 训练时默认每 `5` 个 episode 才画一次，最后一轮强制画
- 诊断用的 `FOC/KKT` 从重型 `autograd.grad` 改成有限差分近似

这些修改的目标不是改变训练结果，而是让：

```math
\text{训练主循环耗时} \gg \text{诊断额外耗时}
```

重新成立。

## 8. 为什么 `bp` child-survival reweight 不够

我们尝试过让 `bp` 的 FOC/KKT 只在 child 仍具继续经营意义的区域有较大权重：

```math
w_{surv}(bp)
=
\sigma(\tau_P P_{t+1}(bp))
\cdot
\sigma(\tau_z(0.5-\bar z_{t+1}(bp)))
```

并在训练中使用：

```math
L_{bp}^{FOC}=w_{surv}(bp)\cdot(FOC(bp))^2
```

```math
L_{bp}^{KKT}=w_{surv}(bp)\cdot KKT(bp)
```

这个改动本身是生效的，但实验结果表明它 **不足以把 `bp` 拉回左侧存活区**。

原因是：

1. 它只削弱了 `FOC/KKT` surrogate
2. 但没有改变当前 `V(bp)` 的形状
3. 如果无约束的 `V(bp)` 本来就在右侧 default 区取得最大值，那么 `bp` 仍会被主目标推向右侧

也就是说：

```math
\text{reweight 只是在 default 区静音 surrogate，}
\quad
\text{却没有给 } bp \text{ 新的全局 value-level 信号。}
```

## 9. 新修改：GPU 粗网格 `argmax V` supervision

### 9.1 修改动机

当前 `V(bp)` 明显不是凹函数，也不是单峰函数。

因此：

- `FOC = 0`
- `KKT penalty` 小

最多只能刻画某个 **局部驻点**，不能保证得到：

```math
\arg\max_{bp} V(bp)
```

这就是为什么单靠 `FOC + KKT`，会反复出现：

```math
bp^* \neq \arg\max V(bp)
```

或者即使相等，那个最大值本身也落在 default 区右边。

### 9.2 修改内容

因此在当前分支中，给 `bp` 新增了一条 **粗网格、GPU 向量化、无梯度目标搜索** 的 value supervision。

对每个 batch，额外做下面这件事：

1. 从当前 batch 中抽取不超过 `sample_cap` 个样本
2. 在 GPU 上构造一个小网格：

```math
\mathcal G = \{0, \tfrac{1}{G-1}, \dots, 1\}
```

3. 对每个样本、每个 `bp \in \mathcal G`，一次性向量化计算近似 one-step value：

不投资：

```math
V_t^0(bp)\approx CF_t^0(bp)+M_t P_{t+1}(bp)(1-\bar z_{t+1}(bp))
```

投资：

```math
V_t^I(bp)\approx CF_t^I(bp)+g\,M_t P_{t+1}(bp)(1-\bar z_{t+1}(bp))
```

4. 若开启 `survival_only`，则只在满足下面条件的粗网格点中找最大值：

```math
P_{t+1}(bp) > 0,\qquad \bar z_{t+1}(bp) < z_{th}
```

5. 得到目标点：

```math
b^\dagger = \arg\max_{bp \in \mathcal G \cap \mathcal S} V(bp)
```

6. 对网络当前输出的 `bp_\theta` 加监督：

```math
L_{bp}^{value} = (bp_\theta - b^\dagger)^2
```

并把它加进 `P0/PI` 的训练损失：

```math
L = L_{main} + L_{FOC/KKT} + \lambda_{value} L_{bp}^{value}
```

### 9.3 为什么数学上有效

因为这条新损失不再依赖：

- 局部导数是否过零
- `FOC/KKT` 是否把右侧平台也视为“可接受”

它直接给出的是：

```math
\text{当前粗网格上最好的 } bp
```

所以它补上的是 **全局 value-level 信息**，而不只是局部最优条件。

在非凹、存在生存/违约 regime switch 的问题里，这比单纯再调 `FOC/KKT` 更对症。

### 9.4 为什么经济上有效

如果用户理论上关心的是：

```math
\text{当前企业在 continue 意义下，应该选什么下一期债务}
```

那么训练时就不该只让 `bp` 满足某个局部 FOC，而应该让它对齐到：

```math
\text{当前状态下真正能带来更高 continue value 的债务选择。}
```

当 `survival_only=True` 时，这条监督进一步对应：

```math
\arg\max_{bp \in \mathcal S} V(bp)
```

也就是：

- 不是允许 `bp` 去追求“明天股权几乎归零”的赌博解
- 而是在仍有继续经营意义的区域里，寻找更合理的最优债务

### 9.5 为什么速度还能接受

这条修改专门按“训练内可承受”的方式实现：

- **target search 在 `torch.no_grad()` 下完成**
  不额外保留大计算图
- **小网格**
  默认 `21` 个 `bp` 点
- **sample cap**
  默认每个 batch 只抽 `256` 个样本做 target search
- **GPU 向量化**
  把 `(sample, bp_grid)` 展平成一个大 batch 一次前向，不在 Python 层逐点循环

因此它的额外成本远低于“把每个样本都做高精度网格搜索”，也不会像之前那样把 episode 末尾诊断的开销搬进训练主循环。
