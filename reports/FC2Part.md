# FC2Part 备忘录

## 1. FC2 在做什么（抽象层面）

FC2 训练的本质不是“直接回归真实宏观标签”，而是学习一个宏观固定点映射。

记：

- 微观横截面分布：\(\mu_t\)
- 宏观状态（代理）：\(m_t=(\hat c_t,\ln K_t)\)
- 分布特征提取：\(\phi_t=\Phi(\mu_t)\)
- FC2 网络：\(f_\theta\)
- 微观决策+转移+聚合算子：\(\mathcal A\)

则 FC2 给出
\[
\hat m_t=f_\theta(\phi_t).
\]

把 \(\hat m_t\) 送入结构模型（企业决策与聚合）得到
\[
\tilde m_t=\mathcal A(\mu_t,\hat m_t).
\]

FC2 的核心损失是固定点残差
\[
\mathcal L_{\mathrm{FC2}}(\theta)=\mathbb E\left[\|\hat m_t-\tilde m_t\|^2\right].
\]

若加入下一期分支一致性（children），可写为
\[
\mathcal L_{\mathrm{FC2}}
=
\mathbb E\!\left[
\|\hat m_t-\tilde m_t\|^2
+\lambda\sum_j\|\hat m_{t+1}^{(j)}-\tilde m_{t+1}^{(j)}\|^2
\right].
\]

所以 FC2 学的是不动点关系：
\[
m^*(\mu)=\mathcal A(\mu,m^*(\mu)).
\]

## 2. 公司数量变化时，如何定义横截面分布

当每期有进入和退出，企业数 \(n_t\) 可变。统一做法是用经验测度：
\[
\mu_t=\sum_{i\in\mathcal I_t}w_{i,t}\,\delta_{s_{i,t}},
\qquad
\sum_{i\in\mathcal I_t}w_{i,t}=1.
\]

- \(\mathcal I_t\)：时点 \(t\) 的存活企业集合（大小可变）
- \(s_{i,t}\)：企业状态（如 \(b,z,\eta,i,\dots\)）
- \(w_{i,t}\)：权重（可等权，也可资本权重）

这一定义天然兼容“企业数量变化”。

## 3. 进入/退出下的分布转移（概念式）

\[
\mu_{t+1}\propto
\sum_{i\in\mathcal I_t}\mathbf 1\{\text{survive}_{i,t+1}\}w_{i,t}\,\delta_{s'_{i,t+1}}
+
\sum_{k\in\mathcal E_{t+1}}w^e_{k,t+1}\,\delta_{s^e_{k,t+1}},
\]
再做归一化使权重和为 1。

- 第一项：存活企业的状态转移
- 第二项：新进入企业

## 4. 给 FC2 的固定维度表示

FC2 需要固定维度输入，因此对 \(\mu_t\) 做特征提取：
\[
\phi_t=\Phi(\mu_t)
=
\big[
Q_b(\tau_1),\dots,Q_b(\tau_M),
Q_z(\tau_1),\dots,Q_z(\tau_M),
x_t
\big],
\]
其中 \(Q_b,Q_z\) 为（可加权）分位数。

实践中常见两种口径：

1. 等权分布特征：每家企业权重相同  
2. 规模加权分布特征：例如按 \(K_{i,t}\) 权重

二者都合法，但必须在“理论解释、训练、诊断”三处保持一致。

## 5. 这一页的用途

本文件只记录 FC2 的方法论抽象，不绑定具体代码实现细节。  
后续若要改代码，先检查是否仍满足上面的固定点定义和分布语义。

