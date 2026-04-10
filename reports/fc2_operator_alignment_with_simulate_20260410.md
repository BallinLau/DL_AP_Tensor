# FC2 Operator Alignment With Simulate

## 1. 目的

这份笔记把 `FC2LossPipe` 中训练用的 operator，和 `simulate_ts_parallel.py` 中系统真实使用的 operator 并排写清楚。

目标不是讨论抽象概念，而是确认：

> `FC2` 在训练时所面对的 aggregate fixed-point operator，是否与系统在运行时生成 `tensor_macro` 的 operator 一致。

---

## 2. 统一记号

对 parent 节点的公司 \(j\)，状态写成：

\[
s_{j,t} = (b_{j,t}, z_{j,t}, \eta_{j,t}, i_{j,t}, K_{j,t})
\]

给定节点级 macro guess：

\[
y_t^f = (\hat c_t^f, \ln K_t^f)
\]

`policy_value` 给出：

\[
g_t(s_{j,t}; y_t^f) = (\bar i_{j,t}, \bar z_{j,t}, bp_{j,t})
\]

资源核算两边共用：

\[
Y_{j,t} = e^{x_t + z_{j,t}} K_{j,t}
\]

\[
\Phi_{j,t} = (1-\phi)(1 + e^{x_t + z_{j,t}}) K_{j,t} \bar z_{j,t}
\]

\[
I_{j,t} = \bar i_{j,t} K_{j,t} i_{j,t} - \bar z_{j,t} K_{j,t} + \delta K_{j,t}
\]

\[
C_{j,t} = Y_{j,t} - I_{j,t} - \Phi_{j,t}
\]

---

## 3. Parent operator

### 3.1 `simulate_ts_parallel.py`

当前 parent 节点的 aggregate 使用当前活跃公司的当前资本：

\[
K^{sim}_{t} = \sum_{j \in A_t} K_{j,t}
\]

\[
C^{sim}_{t} = \sum_{j \in A_t} \max(C_{j,t}, 0)
\]

\[
\ln K^{sim}_{t} = \log(K^{sim}_{t} + 10^{-8})
\]

\[
\hat c^{sim}_{t} = \log\left(\frac{C^{sim}_{t}}{K^{sim}_{t}+10^{-8}} + 10^{-5}\right)
\]

对应代码：

- [data/simulate_ts_parallel.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/data/simulate_ts_parallel.py)

### 3.2 修正后的 `FC2LossPipe`

parent target 现在改成与 `simulate` 同口径：

\[
K^{fc2}_{t} = \sum_{j \in A_t} K_{j,t}
\]

\[
C^{fc2}_{t} = \sum_{j \in A_t} \max(C_{j,t}, 0)
\]

\[
\ln K^{fc2}_{t} = \log(K^{fc2}_{t} + 10^{-8})
\]

\[
\hat c^{fc2}_{t} = \log\left(\frac{C^{fc2}_{t}}{K^{fc2}_{t}+10^{-8}} + 10^{-5}\right)
\]

这一步修正了之前两个关键不一致：

1. 不再用 `bar_z` 对 `K` 做额外加权
2. 不再用 `|C|`，而改回 `max(C, 0)`

---

## 4. Transition operator

从 parent 到 child 的状态转移发生在 branch expansion：

\[
b_{j,t+1}^{(k)} = \eta_{j,t+1}^{(k)} bp_{j,t} + (1-\eta_{j,t+1}^{(k)}) b_{j,t}
\]

\[
K_{j,t+1} = (1-\bar i_{j,t}) K_{j,t} + \bar i_{j,t} g K_{j,t}
          = K_{j,t}\left[1 + (g-1)\bar i_{j,t}\right]
\]

然后配上新 shock：

\[
(z_{j,t+1}^{(k)}, \eta_{j,t+1}^{(k)}, i_{j,t+1}^{(k)}, x_{t+1}^{(k)})
\]

得到 child state：

\[
s_{j,t+1}^{(k)} = \left(b_{j,t+1}^{(k)}, z_{j,t+1}^{(k)}, \eta_{j,t+1}^{(k)}, i_{j,t+1}^{(k)}, K_{j,t+1}\right)
\]

### 修正后的 `FC2LossPipe`

`FC2LossPipe` 现在也按这个口径处理 child incumbent：

- `b` 用 parent 的 `bp` 与 child 的 `eta_{t+1}` 更新
- `K` 用 parent 的 `bar_i` 更新一次：

\[
K_{j,t+1}^{fc2} = K_{j,t}\left[1 + (g-1)\bar i_{j,t}\right]
\]

- entrant 保留 child node 表中的采样值

这一步修正了之前 child 分支里“又额外做一次资本更新”的问题。

---

## 5. Child operator

### 5.1 `simulate_ts_parallel.py`

进入 child 节点后，aggregate 使用已经 transition 完成的 child state：

\[
K^{sim}_{t+1,k} = \sum_{j \in A_{t+1,k}} K_{j,t+1}
\]

\[
C^{sim}_{t+1,k} = \sum_{j \in A_{t+1,k}} \max(C_{j,t+1}^{(k)}, 0)
\]

\[
\ln K^{sim}_{t+1,k} = \log(K^{sim}_{t+1,k} + 10^{-8})
\]

\[
\hat c^{sim}_{t+1,k} = \log\left(\frac{C^{sim}_{t+1,k}}{K^{sim}_{t+1,k}+10^{-8}} + 10^{-5}\right)
\]

### 5.2 修正后的 `FC2LossPipe`

child target 现在改成与 `simulate` 同口径：

\[
K^{fc2}_{t+1,k} = \sum_{j \in A_{t+1,k}} K_{j,t+1}
\]

\[
C^{fc2}_{t+1,k} = \sum_{j \in A_{t+1,k}} \max(C_{j,t+1}^{(k)}, 0)
\]

\[
\ln K^{fc2}_{t+1,k} = \log(K^{fc2}_{t+1,k} + 10^{-8})
\]

\[
\hat c^{fc2}_{t+1,k} = \log\left(\frac{C^{fc2}_{t+1,k}}{K^{fc2}_{t+1,k}+10^{-8}} + 10^{-5}\right)
\]

并且 child 当前节点的 alive 集合改为：

- 使用 child node 当前存在的 firms（`child_present`）
- 而不是错误地用 parent 的 `bar_z` 去加权或筛选

---

## 6. 这次修正的核心含义

修正前，`FC2LossPipe` 和 `simulate` 的差异主要在：

1. parent `K` 被 `bar_z` 重新加权
2. parent/children 的 `C` 用了 `|C|`
3. child aggregate 阶段又重复做了一次资本更新
4. child 当前节点的 row 选择混入了 parent `bar_z` 的错误语义

修正后，目标是把训练用 operator 写成：

\[
T_{fc2} \approx T_{sim}
\]

至少在 parent / transition / child 三段的定义上，尽量与系统真实运行时的 `tensor_macro` 生成逻辑一致。

---

## 7. 剩余注意点

虽然口径已经明显更接近，但还需要后续实验确认：

1. `FC2` 当前 summary（`b/z` quantiles + `x`）是否足够识别 `lnk`
2. `policy_value` 路径的计算成本是否仍然过高
3. `cal_phats()` 的重复计算是否仍然主导 FC2 训练时间

也就是说，这次修正解决的是：

> 训练 operator 与 simulate operator 的定义错位

但不自动保证：

> 当前 `FC2` summary 本身就足以学好这个 operator

