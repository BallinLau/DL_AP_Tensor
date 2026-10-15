# P-only Bellman movement channel decomposition

## 结论

这次 exact post-hoc decomposition 的答案很明确：**让 P-only movement 后期不衰减并加速放大的主通道，是 child PI branch value 的 continuation recursion**，不是 Q、cashflow、default switching、investment switching，也不是 leverage argmax switching。

在 10 个 transition 的平均 normalized directional attribution 中，child PI 对 P0/PI target movement 的贡献均为 `0.7673`；后五个 transition 上升到 `0.9017`。child P0 是次要通道，平均 `0.2137`、后五期 `0.0941`。其余通道平均合计不到 2%：investment `1.38%`、default `0.096%`、production BP/search-grid `0.424%`。

实际 NN movement 也主要由前一期 Bellman target movement 驱动：对 `Delta P` 的平均 directional attribution 是 P0 `97.23%`、PI `96.88%`；projection-error change 只有 `2.77%` 和 `3.12%`。因此 neural fitting/projection 不是主要 runaway source。

## 1. Scope 与 provenance

- 分析代码 HEAD：`d69f94d293883ce9e794cf4643fb27f52c3b2df3`
- 原 P-only run：`/WORK/PUBLIC/zhuyingz_work/LiuHao/cachedir/inner_fp_block_p_only_ep1_10s_20261014_210200`
- 使用 checkpoint：`sweep_000.pt` 至 `sweep_010.pt`，共 11 个，逐个独立读取；没有重跑 sweep。
- 初始 combined checkpoint：`.../normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_investment_compare_fresh100_20261014_144528/checkpoints/ep1_combined.pt`
- 初始 checkpoint SHA-256：`97b7bb1a284904027d1ce17df7b73940c00f12fd08bab86fb689496550fc5d5d`
- fixed validation parents：2,048。
- continuous child shocks：16；exact-eta 后每个 parent 32 个加权 child nodes。
- fixed common leverage grid：101 points on `[0,1]`。
- 计算方式：把服务器现有 checkpoint/bank 原样复制到本机，CPU 只读分析；未修改 checkpoint。
- optimizer objects：0；optimizer/training steps：0。

固定对象 fingerprints：

| object | fingerprint |
|---|---|
| validation parent bank | `2f40a1df1aed6ed8a0ea2cf948f09ac744ca5c0a7a3ef8e60ff0cedcb217f6b0` |
| validation child bank | `15645cc7c677a6e5aaa8bec7fe0b15878ffc24a84f59edc3240e84e0df3908cb` |
| Q parameters, every sweep | `04d10e09b9db1ac4867c36b22642946b9e47ca8bf7a00fe96ba6d4d18cd5946d` |

所有 11 个 checkpoint 内嵌的 validation parent/child fingerprints 都与加载对象一致；Q hash 和 BP-head hash 在 sweep 0--10 完全不变。

## 2. Code-confirmed P Bellman operator

真实调用链如下：

1. `BPGridTeacher.compute()` 按 parent 当前 `eta_t` 分流。`eta_t=1` 使用 production coarse + adaptive local-fine search；`eta_t=0` 强制 `b_{t+1}=b_t`。
2. child leverage 严格使用

   $$b_{t+1}=\eta_t b'_t+(1-\eta_t)b_t.$$

   child 自身的 `eta_{t+1}` 不控制当前选择。
3. child equity 通过 `forward_equity()`，继而调用 `cal_phats()`。它在模型自身的完整 internal i-grid 上计算 P0/PI，形成

   $$\widehat P=\frac{1}{N_i}\sum_i\max\{P_0(i),P_I(i)\},\qquad P=\max\{\widehat P,0\}.$$

   `total_equity(..., hard=True)` 已代码确认就是 `clamp_min(Phat,0)`。soft `bar_z` 没有混入 hard-P exact decomposition。
4. direct-Q run 的 issuance price 走 live claim output。candidate Phat 不进入 issuance recovery gate。P0/PI candidate objectives 为

   $$V_0=CF_0+E[M P(child)],$$

   $$V_I=CF_I+gE[M P(child)].$$

5. aggregation 完全复用训练/evaluator 的 M clamp、exact-eta expansion、branch weights 与 weight normalization。
6. production `bp_star` 与 `value_star` 一一对应；若 quadratic refinement 开启，teacher 会在 refined `bp_star` 上重新 evaluate。此次分解没有用错 `bp_star_grid`。

对应实现证据：

- `training/bp_grid_teacher.py:105-136,198-252,1337-1400,1655-1696,1735-1802`
- `models/policy_value.py:350-407,458-471,557-597`
- `models/firm_derived.py:36-40`
- `training/episode.py:1225-1265`
- `utils/firm_transition.py:27-95`

## 3. Hard invariants 与 exact closure

| check | result |
|---|---:|
| cashflow surface fixed | pass; all tested diffs exactly `0` |
| child Phat closure max | `1.62125e-05` |
| hard child-equity closure max | `1.62125e-05` |
| parent continuation closure max | `4.76837e-06` |
| full target closure max | `4.76837e-06` |
| projection movement closure max | `1.19209e-07` |
| eta=0 `C_bp` absolute max | `0` |
| common-grid `C_bp` minimum | `0` |
| Q hash identical | pass |
| optimizer/training step | `0` |

Frozen-Q cashflow surface checks：

| transition | branch | mean abs diff | p90 abs diff | max abs diff |
|---|---|---:|---:|---:|
| 0→1 | P0 | 0 | 0 | 0 |
| 0→1 | PI | 0 | 0 | 0 |
| 4→5 | P0 | 0 | 0 | 0 |
| 4→5 | PI | 0 | 0 | 0 |
| 9→10 | P0 | 0 | 0 | 0 |
| 9→10 | PI | 0 | 0 | 0 |

这些误差是 float32 复算下的逐节点数值误差；closure 成立后才解释 attribution。

## 4. 核心 channel attribution

10 个 transition 的平均结果：

| channel | P0 alignment | PI alignment | P0 norm abs mean | PI norm abs mean | interpretation |
|---|---:|---:|---:|---:|---|
| child P0 value | 0.2137 | 0.2137 | 0.0696 | 0.0793 | secondary；早期较大、随后衰减 |
| child PI value | **0.7673** | **0.7673** | **0.2479** | **0.2826** | primary；后期占绝对主导 |
| investment switch | 0.01385 | 0.01385 | 0.00341 | 0.00389 | small boundary effect |
| default switch | 0.00096 | 0.00096 | 0.00050 | 0.00057 | negligible |
| BP/search-grid | 0.00424 | 0.00424 | 0.00141 | 0.00161 | negligible overall |
| projection-error change | 0.0277 | 0.0312 | — | — | small effect on actual NN movement |

最关键的时间变化：

| window | branch | target abs mean | child P0 align | child PI align | empirical L2 gain |
|---|---|---:|---:|---:|---:|
| sweeps 0→3 | P0 | 0.1810 | 0.4243 | 0.5267 | 0.8874 |
| sweeps 0→3 | PI | 0.2063 | 0.4243 | 0.5267 | 0.8802 |
| sweeps 5→10 | P0 | 0.4465 | 0.0941 | **0.9017** | **1.2696** |
| sweeps 5→10 | PI | 0.5090 | 0.0941 | **0.9017** | **1.2806** |

P0 target normalized abs mean 从 0→1 的 `0.1822` 上升到 9→10 的 `0.6780`；PI 从 `0.2078` 上升到 `0.7729`。9→10 时 child PI alignment 已为 `0.9687`，而 child P0 只有 `0.0284`。

## 5. Continuation vs leverage/search-grid

总 target movement 几乎全部是 continuation。production `C_bp` 平均 alignment 只有 `0.42%`。adaptive local grid 确实导致极少数 production `C_bp<0`：全样本平均 negative share `0.1025%`，主要在早期；所以它在报告中严格称为 **BP/search-grid channel**，不硬解释成经济 re-optimization gain。

101 点 common grid 使用完全相同 candidate set，`C_bp_common_grid >= 0` 逐状态成立，最小值为 0。production/common-grid BP channel 的平均绝对量相差只约 `1e-4` normalized units，相对于总 target movement 小于 0.1%，因此没有必要追加 201 点 sensitivity；无论采用哪个口径，leverage channel 都不可能改变 root-cause 排名。

weak-margin states 的 BP channel 较大，但仍然很小：

| top2-margin quartile | P0 `C_bp` abs mean | PI `C_bp` abs mean | mean `|Delta bp*|` P0/PI |
|---|---:|---:|---:|
| Q1 weakest | 0.00540 | 0.00619 | 0.01474 / 0.01483 |
| Q4 strongest | 0.00143 | 0.00164 | 0.00079 / 0.00078 |

这确认 weak margin 会放大 policy switching，但其总量和 alignment 太小，不是 divergence driver。

## 6. Eta control group

`eta_t=0` 的 `C_bp` 在全部 sweep、两个 branch 都严格为 0；但它的 P target movement 仍明显不衰减：

| branch | eta | mean target movement | 0→1 | 9→10 |
|---|---:|---:|---:|---:|
| P0 | 0 | 0.1369 | 0.0895 | **0.2721** |
| P0 | 1 | 0.5045 | 0.2750 | **1.0838** |
| PI | 0 | 0.1560 | 0.1020 | **0.3102** |
| PI | 1 | 0.5751 | 0.3135 | **1.2355** |

eta=1 states 的幅度更大，说明 leverage/state composition 调节 amplification；但 eta=0 自身也从约 0.09/0.10 增至 0.27/0.31，所以 **re-optimization 不是 movement 不衰减的必要条件**。

## 7. Default 与 investment boundary

Default switching 的确集中在边界附近：平均 `72.6%` 的 default-channel absolute contribution 来自旧 `|Phat|<0.05` 的 child nodes，其中 `[0,0.01)` flip share 为 `46.8%`，`[0.01,0.05)` 为 `36.2%`。但是它对总 movement 的平均 directional attribution 只有 `0.096%`；这是一个真实但量级极小的 boundary effect，不能解释 runaway。

Investment switching 也集中于 near-tie states：平均 `67.5%` 的 investment-channel absolute contribution 来自旧 `|PI-P0|<0.05`，对应 switch share `45.6%`（`<0.01`）和 `26.6%`（`0.01-0.05`）。但 investment channel alignment 只有 `1.38%`，同样不是主要放大器。

## 8. Empirical Bellman gain

`G_P=||Delta Y_k||/||Delta P_k||` 在 0→1 分别只有 P0 `0.598`、PI `0.556`；从 2→3 开始多数 transition 超过 1，5→10 平均 P0 `1.270`、PI `1.281`，9→10 达到 `1.421/1.418`。这说明 realized trajectory 的 P Bellman movement 后期是 non-damping / amplifying。

这只是 **empirical local movement gain**，不是 Jacobian spectral radius 的估计，更不是 `spectral radius = 1.42` 的证明。

## 9. Projection/fitting dynamics

对于 k≥1，逐状态验证：

$$P_{k+1}-P_k=(Y_k-Y_{k-1})+(e_{k+1}-e_k).$$

平均 directional attribution：

| branch | prior target movement → `Delta P` | projection-error change → `Delta P` |
|---|---:|---:|
| P0 | **0.9723** | 0.0277 |
| PI | **0.9688** | 0.0312 |

部分 sweep 的 projection term 会抵消 target movement（例如 6→7、9→10），而不是持续推动 runaway。projection dynamics 是 secondary noise/correction，不是根因。

## 10. 最终判定

- **Primary destabilizing channel:** child PI value continuation recursion。
- **Secondary channel:** child P0 value continuation，早期更重要、后期显著退居次位。
- **Small boundary channels:** investment switching；default switching 更小。
- **Ruled out as primary:** frozen-Q cashflow movement（精确为零）、leverage re-optimization/search-grid、default switching、investment switching、projection-error change。
- **Mechanism interpretation:** P-only operator 的 hard child equity envelope 越来越由 PI values 的跨 sweep movement 驱动；PI continuation 在两个 parent Bellman branches 中传播，PI parent branch 又乘以 `g`。随着 trajectory 推进，empirical gain 多数大于 1，target movement 因而不衰减并扩张。
- **Not established:** 没有计算 Jacobian/eigenvalues，因此不声称 spectral radius 数值；也不把这个结果外推成任意状态分布下的全局稳定性定理。

## 11. Deliverables

核心数据：

- `cachedir/p_only_movement_channels_20261015/p_movement_channel_summary.csv`
- `cachedir/p_only_movement_channels_20261015/p_movement_channel_by_eta.csv`
- `cachedir/p_only_movement_channels_20261015/p_movement_channel_by_default_margin.csv`
- `cachedir/p_only_movement_channels_20261015/p_movement_channel_by_investment_margin.csv`
- `cachedir/p_only_movement_channels_20261015/p_movement_channel_by_bp_margin.csv`
- `cachedir/p_only_movement_channels_20261015/p_movement_channel_statewise.parquet`
- `cachedir/p_only_movement_channels_20261015/cashflow_invariance.csv`
- `cachedir/p_only_movement_channels_20261015/analysis_metadata.json`

图：

- `figures/target_vs_channels_p0.png`, `target_vs_channels_pi.png`
- `figures/directional_attribution_p0.png`, `directional_attribution_pi.png`
- `figures/continuation_vs_bp.png`, `child_channels.png`
- `figures/eta_comparison.png`, `default_buckets.png`, `investment_buckets.png`
- `figures/bp_margin.png`, `empirical_gain.png`
- `figures/target_vs_projection.png`, `closure.png`

复现命令：

```bash
python analysis/analyze_p_only_movement_channels.py \
  --run /WORK/PUBLIC/zhuyingz_work/LiuHao/cachedir/inner_fp_block_p_only_ep1_10s_20261014_210200 \
  --output-dir /WORK/PUBLIC/zhuyingz_work/LiuHao/cachedir/p_only_movement_channels_20261015 \
  --common-grid-size 101 \
  --max-transitions 10 \
  --batch-size 64
```
