# Direct-Q 与分 Regime 训练重构

## 目标

本次重构将债务价值网络从旧参数化

$$Q(s)=b\,q_{unit}(s)$$

改为直接输出总债务价值：

$$Q_\theta(s)=NN_Q(s).$$

同时把每个 outer episode 的 Policy/Value 阶段改为：

$$
Q^{(k)}
\rightarrow P^{(k+1)}
\rightarrow \operatorname{freeze}P^{(k+1)}
\rightarrow Q_0
\rightarrow Q_D
\rightarrow Q_S
\rightarrow Q_{polish}
\rightarrow bp.
$$

整个 Q stage 使用同一个 frozen-P snapshot 对 parent state 分类。Q continuation
target 可以在 phase 之间刷新，但不能改变 frozen `Phat`。

## 模型与 Checkpoint 语义

- `PolicyValueModel._q_output()` 在 `q_parameterization="direct"` 时直接返回
  linear `q_head` 输出，不乘 `b`，也不做 inference clamp。
- direct-Q 的 `QHead` 使用 `output_activation=None`。
- 非负性通过训练项
  $L_+=E[\operatorname{ReLU}(-Q)^2]$ 处理。
- `policy_value_model_spec` 显式保存 `q_parameterization`。
- 没有该字段的旧 checkpoint 被解释为 `b_times_unit`。
- Trainer 拒绝把 `b_times_unit` checkpoint 静默加载到 direct-Q 模型中。

`q_unit=Q/b` 仅保留为 evaluator 的派生报告比率；它不再是网络 head 输出。

## Regime 与 Loss Path

`losses/q_loss.py::classify_q_parent_regimes()` 定义互斥集合，且 zero 优先：

$$
\mathcal B_0=\{b=0\},\quad
\mathcal D=\{b>0,\widehat P^T\le0\},\quad
\mathcal S=\{b>0,\widehat P^T>0\}.
$$

### Q0

`Episode._compute_q_zero_loss()` 将真实 context row 的 `b` 精确覆盖为 `0.0`，调用
`QLoss.compute_zero_debt_objective()`：

$$L_0=E[Q_\theta(s)^2]+\lambda_+L_+.$$

函数没有 children、M、SDF 或 AiO 参数。

### QD

`Episode._compute_q_default_loss()` 调用
`QLoss.compute_default_parent_objective()`：

$$
R(x,z)=\phi(1-\delta+e^{x+z}),\qquad
L_D=E[(Q_\theta-R)^2]+\lambda_+L_+.
$$

该路径同样没有 children、M、SDF 或 AiO 参数，且 recovery 保持
`asset_only`。

### QS

只有 `Episode._compute_q_survival_bellman_loss()` 构造 child states、读取 M、
计算 continuation payoff，并调用
`QLoss.compute_survival_parent_objective()` 中的 `compute_aio_residual()`：

$$
f_Q(s,\epsilon)=M X_{t+1}-Q_\theta(s),\qquad
L_S^{AiO}=E[f_Q(s,\epsilon_1)f_Q(s,\epsilon_2)]+\lambda_+L_+.
$$

parent survival 不会过滤 child default；child payoff 仍使用现有 survival/recovery
分解。outstanding old-bond continuation leverage 保持：

$$b_{sp}=\frac{b}{1+\bar i(G-1)},$$

没有替换为 firm policy transition
$\eta bp+(1-\eta)b$。

### Mixed Polish

polish batch 由三个独立 sampler 构造。每个样本只进入一条 loss path：

$$
b=0\Rightarrow L_0,\qquad
b>0,\widehat P^T\le0\Rightarrow L_D,\qquad
b>0,\widehat P^T>0\Rightarrow L_S^{AiO}.
$$

默认 sampling shares 为 `0.20/0.30/0.50`，只表示训练采样比例，不是经济概率。

## Sampler

### Zero boundary

从当前 parent pool 保留 `(z, eta, i, x, Hatc, LnK)`，仅把 `b` 精确设为零。
低正债务不会归入 Q0。

### Default coverage

`_build_q_default_coverage_states()`：

1. 从真实 parent context 循环选 anchor；
2. 在正债务支持上用 `q_default_b_bins` 个中心分层；
3. 在 `ZBAR +/- 4 * stationary_std(z)` 上构造 z grid；
4. 用 frozen-P snapshot 计算 `Phat`；
5. 正式 QD 保留 `Phat <= -q_default_phat_eps`；
6. 若数量不足，candidate pool 最多连续扩大四次；
7. mixed polish 使用 `Phat <= 0`。

### Survival transitions

`_build_q_survival_batch()` 保留 matched parent-child rows。正式 QS 使用
`Phat > q_default_phat_eps`；mixed polish 使用 `Phat > 0`。coverage replay 只重复
已有 matched transitions，不改变 parent-child 对齐。

## 冻结与 Target 语义

- Stage P 只更新 value/P modules，冻结 `q_encoder` 与 `q_head`。
- P 完成后 deep-copy 一份 frozen-P snapshot。
- Q0/QD/QS/polish 全程使用该 snapshot 的 equity-only forward 计算 `Phat`。
- 每个 Q phase 单独建立 continuation-Q target snapshot；它与 frozen-P classifier
  是两个对象。
- Q phase 只更新 `q_encoder` 和 `q_head`。
- phase 结束时验证 frozen-P hash 及全部 non-Q 参数完全不变。
- nonfinite loss/gradient 会恢复 phase 起点 Q 参数并阻断后续 BP stage。

## 配置与入口

`experiments/run_multi_episode_job.py` 和
`slurm/run_normalized_sdf_aio_bp_logit.slurm` 已接通：

- `Q_PARAMETERIZATION=direct`
- `Q_ZERO_BOUNDARY_EPOCHS=5`
- `Q_DEFAULT_PRETRAIN_EPOCHS=10`
- `Q_SURVIVAL_AIO_EPOCHS=20`
- `Q_MIXED_POLISH_EPOCHS=5`
- `Q_ZERO_SAMPLE_SHARE=0.20`
- `Q_DEFAULT_SAMPLE_SHARE=0.30`
- `Q_SURVIVAL_SAMPLE_SHARE=0.50`
- `Q_DEFAULT_PHAT_EPS=1e-2`
- `Q_ZERO_B_EPS=0.0`
- `Q_DEFAULT_CANDIDATE_MULTIPLIER=4`
- `Q_DEFAULT_B_BINS=10`
- `Q_SURVIVAL_ONDIST_SHARE=0.80`
- 四个 loss weights 默认均为 `1.0`

formal staged runner 对非 direct 参数化、负 epoch/weight、非法 share 或 coverage
配置 fail fast。

## 诊断

Q phase summary 按 epoch 输出：

- zero: n、absolute mean/p90/max；
- default: recovery MSE、absolute mean/p90/max、Q/R mean、逐 b-bin count/MAE；
- survival: Bellman branch residual signed mean、absolute mean/p90；
- global: negative share 与 negative mean absolute value；
- frozen-P hash、non-Q max parameter change、optimizer steps 和 rollback reason。

现有 checkpoint evaluator 已输出 `Q`、`Phat`、default mask、recovery、
`Q-recovery` 和 survival Bellman residual surfaces，并在 metadata 中记录
`q_parameterization`。`q_unit` 明确标记为派生比率。

## 测试覆盖

新增及相关测试覆盖：

1. direct-Q 不乘 b，允许观察负 raw output；
2. 旧 model spec 缺字段时重建为 `b_times_unit`；
3. 旧参数化 checkpoint 不能静默加载为 direct-Q；
4. zero/default/survival masks 互斥；
5. Q0/QD objective 不调用 AiO；
6. Episode mixed dispatcher 在 Q0/QD 不进入 child/SDF/AiO path；
7. Q phase 只改变 Q 参数，frozen `Phat` 完全不变；
8. survival continuation 保持 old-bond leverage；
9. 既有 child-default recovery、recovery normalization 和 staged rollback tests。

本地定向回归结果：`187 passed`。完整测试结果：`548 passed`。

## 尚需 GPU 验证的风险

- default coverage 在某些 frozen-P snapshot 下可能几乎没有 deep-default 样本；日志会
  显示 skipped/count，而不会退化为 survival loss。
- direct linear Q head 初期可能产生较多负输出；需要用新 negative diagnostics 判断
  nonnegative penalty 是否足够，而不是在 forward clamp。
- 默认 Q phase epoch 数是第一版配置，尚未通过正式 multi-episode GPU run 校准。
- P/Q outer iteration 的收敛速度与旧 joint P/Q 不可直接比较，需要新 run root 和新
  checkpoint 语义进行实验。

## 第二轮修复（follow-up）

详见 `reports/q_direct_regime_training_followup.md`。要点：

1. **Episode-0 direct-Q cold-start bootstrap**：在第一次 P stage **之前**运行，
   只更新 `q_encoder + q_head`，把随机 direct-Q 初始化到有限、平滑、非随机状态，
   切断 $Q_{\rm random}^{(0)}\to P^{(1)}\to\widehat P^{T}\to\mathcal D/\mathcal S$
   的污染链。
2. **Q0/QD/QS required-phase gate**：`skipped_no_samples` 不再被吞成 accepted；
   QS 没有 Bellman optimizer step 时 Q stage 显式失败，且不进入 BP。
3. **checkpoint / model-spec semantic metadata**：formal runner 现在写
   `metadata/{hyperparams,config_snapshot,policy_value_model_spec}.json` 与
   `epX_combined.pt`；`build_models(..., ckpt_dir=...)` 在没有 spec 且目标是
   direct-Q 时拒绝静默加载裸 state_dict。
4. **Q shape prior 默认置 0**：`q_shape_weight_{z,b_low,b_high}` 默认 0.0，
   并接入 CLI / Slurm，便于做 shape-prior ablation。

主体（direct-Q、frozen-P 分类、Q0/QD/QS 分流、asset_only recovery、child-default
recovery、old-bond continuation $b_{sp}=b/[1+\bar i(G-1)]$）未改动。
