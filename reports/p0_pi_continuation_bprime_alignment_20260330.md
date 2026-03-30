# P0 / PI continuation 中 `b'` 的理论对齐修正

Date: 2026-03-30

## 1. 结论

按 `main_4.tex` 中的 `P^0` / `P^I` Bellman 公式，continuation 项里的 debt argument 应直接对应当期所选 contract `b'`。

因此在当前代码中：

- `P0` 分支 continuation 应使用 `bp0`
- `PI` 分支 continuation 应使用 `bpI`

而不应写成：

```math
\eta' b' + (1-\eta') b
```

---

## 2. 理论依据

主文公式写的是：

```math
P^0(b,z,\eta,S)
=
\max_{b'}
\left\{
\cdots
+
E_S M(S,S')
\int\int_{\bar z(b',\eta',S')} P(b', z', \eta', S') N(dz'|z)\psi(d\eta')
\right\}
```

以及

```math
P^I(b,z,\eta,i,S)
=
\max_{b'}
\left\{
\cdots
+
g E_S M(S,S')
\int\int_{\bar z(b',\eta',S')} P(b', z', \eta', S') N(dz'|z)\psi(d\eta')
\right\}.
```

这里清楚表明：

- continuation 中的状态对象是 `P(b', z', \eta', S')`
- `\eta'` 进入的是 default cutoff / future integration
- `\eta'` 没有改写 debt contract `b'`

因此若代码里的 policy output 分别是：

- `bp0`：不投资时最优 `b'`
- `bpI`：投资时最优 `b'`

则 continuation 中应直接使用：

- `b' = bp0`
- `b' = bpI`

---

## 3. 修改前的问题

此前 `training/episode.py` 中 `P0 / PI` 的 child state 写成：

```python
child_state[:, 0:1] = eta_child * bp_for_p0 + (1 - eta_child) * b_parent
child_state[:, 0:1] = eta_child * bp_for_pi + (1 - eta_child) * b_parent
```

这等价于把 continuation 的 debt state 改成了：

```math
b_{child} = \eta' b' + (1-\eta') b
```

这与主文 Bellman 公式不一致。

---

## 4. 本次修改

文件：

- `training/episode.py`

修改为：

```python
child_state[:, 0:1] = bp_for_p0
child_state[:, 0:1] = bp_for_pi
```

并同步修改 Bellman abs residual 评估辅助路径，保证：

- 训练口径
- 诊断口径

保持一致。

---

## 5. 修改后的解释

现在 `P0 / PI` 中与 `b'` 相关的三处对象分工变为：

1. 当前现金流中的净发债收入项
   - 仍由当前 `\eta` 控制是否发生新债发行
   - 即 `eta * ((1-kappa_b) Q(b') - Q(b))`

2. continuation 项中的 future equity value
   - 使用固定 contract `b'`
   - 即 `P(b', z', \eta', S')`

3. future `\eta'`
   - 仍保留在 `P(..., \eta', S')` 和 `\bar z(b', \eta', S')`
   - 但不再改写 debt argument 本身

这与论文公式一致。

---

## 6. 影响判断

这次修改的直接影响是：

- `P0 / PI` 的 continuation 将更严格围绕“当前选择的 contract”递推
- `bp0 / bpI` 的经济含义更清晰
- `P0 / PI` 的 Bellman 训练对象与 `Q(b')` 的使用口径更一致

同时，这也进一步说明：

- `Qp / QpI` 使用 `bp0 / bpI` 本身是合理的
- 之前更主要的错位，其实在 continuation 中把 `b'` 写成了 `\eta' b' + (1-\eta')b`

---

## 7. 未解决的问题

这次修改只修正了 `P0 / PI` continuation 中 `b'` 的映射。

以下更底层问题仍然存在：

- `Q` 对象在全链条中仍未统一成单一 `\tilde b`
- `Q` 与 `bp0 / bpI / V0 / VI` 仍共享 trunk
- `Q` 与 `P` 仍可能在 joint backward 中互相污染

因此这次修改是理论对齐上的必要一步，但不是最终结构修复。
