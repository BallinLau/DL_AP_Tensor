# Codex 全部修改汇总报告（2026-03-12）

> 目的：汇总本次协作中我已实施的全部代码/Notebook 修改，给出修改原因、影响范围、验证情况，并链接到分阶段详细报告。

## 0. 总览

本次修改围绕三条主线展开：

1. **SDF/FC1 两阶段训练稳定性与诊断增强**（解决 NaN、重建列位错配、阶段化学习率/矩约束）。  
2. **P/Q 联立贝尔曼训练一致性修复**（状态转移时点、梯度可识别、Q 形状约束链路）。  
3. **Q 定价语义与边界口径修复**（`Qsp` 输入口径、`eta_{t+1}` 对齐、`b*recovery` 总债价值语义）。

---

## 1. 已改代码文件（本次我实际修改）

- `config/constants.py`
- `data/sample.py`
- `data/simulate_ts.py`
- `losses/p0_loss.py`
- `losses/pi_loss.py`
- `losses/q_loss.py`
- `models/policy_value.py`
- `models/share_layer.py`
- `training/episode.py`
- `tests/sdf_fc1_two_modes_test.ipynb`

说明：仓库中还有其他已变更文件（例如 `config/hyperparams.py`, `models/sdf_fc1.py`, `README` 类文档等）在 `git status` 中显示为修改，但并非都由我在本轮新改；此报告仅归纳我本次实际落地的改动。

---

## 2. 分阶段改动与原因

## 2.1 SDF/FC1 稳定性与两阶段训练支持

### 改动点
- 在 `episode._compute_sdf_loss` 中补充：
  - FC1 重建目标列索引的双布局兼容（含/不含 `M` 列）；
  - 缺失真值列时跳过重建损失并告警；
  - 非有限 `recon_loss` 自动回退为 0；
  - 第一阶段（`add_FC1loss=False`）提高矩约束权重；
  - 记录 `logE[M]`、`logVar[M]`、`Δhatc/Δlnk` 分位数诊断。

- 在训练主循环中把上述诊断写入 `losses/final_losses`，便于 notebook 直接监控。

### 原因
- 解决你之前指出的 `children_t[:,:,8:9]/9:10` 监督列错位导致的 NaN；
- 支持“episode0 先矩条件预训练”的两阶段流程；
- 提供你要求的中间过程监控，而不只看总 loss。

---

## 2.2 模拟支持集与时点一致性（先改后回滚到你指定口径）

### 最终口径（当前代码）
- `b_{t+1}` 使用 **child eta（`eta_{t+1}`）**，与你最新要求一致：
  - `data/sample.py` 子节点杠杆回填；
  - `data/simulate_ts.py` 分支扩展；
  - `training/episode.py` P0/PI/Q 三个 loss 构造 child state。

### 同步修复
- `simulate_ts` 中大量 `squeeze()` 改为 `reshape(-1)`，避免单公司时 0 维张量；
- `_apply_entry` 加入 0 维张量规范化，修复 `torch.cat` 报错。

### 原因
- 对齐你指定的 `eta_{t+1}` 转移语义；
- 修复 notebook mode2 模拟报错：`zero-dimensional tensor cannot be concatenated`。

---

## 2.3 P/bar_z 平滑化与现金流符号修复

### 改动点
- `models/policy_value.py`：
  - `P = softplus(Phat)`，`bar_z = sigmoid(-temp*Phat)`，替代硬 `clamp` + 超陡变换。
- `losses/p0_loss.py` / `losses/pi_loss.py`：
  - 股权融资成本改为 `cf = cf_raw - kappa_e*relu(-cf_raw)`（负现金流更负）；
  - 增加 FOC 梯度诊断字段（均值、缺失率）。
- `training/episode.py`：
  - 将 P0/PI 分解项与 FOC 诊断接入日志。

### 原因
- 缓解 `P/bar_z` 梯度死区；
- 修复“融资成本方向错误会抬高权益价值”的实现偏差；
- 提高 FOC 可观测性，排查 `bp` 不可识别问题。

---

## 2.4 Qsp 输入杠杆口径修复（方程一致性）

### 改动点
- 在 `_compute_q_loss` 中：
  - 将 `childsp_state[:,0]` 从 `b_parent` 改为
    \[
    b_{sp}=\frac{b}{\bar i (G-1)+1}
    \]
  - 并加 `clamp_min(1e-6)` 数值保护。

### 原因
- 对齐 `q_loss` 主方程中 `multiplier = bar_i*(G-1)+1` 的缩放结构；
- 修复“输入状态与方程缩放项不一致”导致的 Q 曲率学习扭曲。

---

## 2.5 Q 的 `b*recovery` 总债价值语义改造（你最新确认）

### 改动点
- `models/share_layer.py`：
  - `Q` 改为结构化输出：`Q = b_+ * q_unit`，硬满足 `b=0 => Q=0`。

- `losses/q_loss.py`：
  - 新增 `compute_total_recovery(b,x,z)=b_+*recovery_unit`；
  - `compute_main_residual` 违约项改为 `b*recovery`；
  - `compute_bar_z_constraint` 改为 `Q≈b*recovery`；
  - 高杠杆边界与 `penalty_z_loss3` 统一到同一语义。

- `training/episode.py`：
  - Q loss 调用与 `penalty_z_loss3` 对接新 `b*recovery` 口径。

### 原因
- 与你确认的“引入 `b*recovery` 语义”一致；
- 解决 `b≈0` 时 `Q` 仍偏正的问题；
- 统一主残差、违约约束、边界条件的债值定义。

---

## 2.6 Notebook 修改

文件：`tests/sdf_fc1_two_modes_test.ipynb`

- 在 mode2 policy/value 模拟单元加入：
  - `importlib.reload(data.simulate_ts)`，防止 notebook 复跑时仍使用旧版类定义；
- 清理该单元历史 error 输出。

原因：消除“代码已修复但 notebook 内存仍引用旧实现”的假阳性报错。

---

## 3. 已生成的分阶段详细报告

- `reports/pq_joint_bellman_fix_implementation_20260312.md`
- `reports/pq_eta_tplus1_sync_and_notebook_fix_20260312.md`
- `reports/qsp_b_transform_fix_20260312.md`
- `reports/q_b_recovery_semantics_fix_20260312.md`

本汇总报告与上述详细报告互补：本文件负责全局串联，分报告负责每轮实施细节。

---

## 4. 验证记录

我已对本次关键改动文件多次执行静态语法检查（`py_compile`），均通过。

受当前运行环境限制（OpenMP SHM），无法在此稳定执行完整 torch 训练流程，因此动态行为（曲线形状、收敛表现）需以你本机 notebook 复跑为准。

---

## 5. 当前状态与后续

当前代码已完成你要求的关键口径统一（尤其 `eta_{t+1}` 与 `b*recovery` 语义），并修复了 mode2 notebook 报错。下一步可继续进入“逐 episode 训练 + 收敛判定”实现阶段。

---

## 6. 本轮增补（同日后续）

按你后续要求，我在同一天补充了两项关键改动，并已落地到代码与文档：

### 6.1 Episode 收敛判定阈值接入（Bellman 非 AIO 口径）

- 阈值固定为：
  - `mean(abs residual) < 0.001`
  - `p90(abs residual) < 0.005`
- 新增了 `P0/PI/Q` 三方程主残差（原始 branch residual，非 AIO）的统计与通过判定，并写入 `summary['convergence']`。
- `Trainer.train` 已支持在 `convergence.passed=True` 时提前结束后续 episode。

对应文件：
- `config/hyperparams.py`
- `training/episode.py`
- `training/trainer.py`

详细报告：
- `reports/episode_bellman_convergence_update_20260312.md`

### 6.2 SDF 中 W 的结构化参数化（相对 c 的正增量）

你提出 `w` 应与 `c` 保持结构关系，我已将 `ValueFunctionW` 从“仅 softplus 正值”改为：
\[
w = \exp(c) + \text{surplus},\quad \text{surplus}=\text{softplus}(\cdot)+\epsilon_w
\]
其中 `\epsilon_w = Config.W_SURPLUS_FLOOR`。

这保证了：
- `w - exp(c) > 0` 在结构上恒成立；
- `w - exp(c)` 具有显式下界，降低分母接近 0 的风险。

对应文件：
- `config/constants.py`
- `models/sdf_fc1.py`
- `models/README.md`

详细报告：
- `reports/sdf_w_surplus_param_update_20260312.md`

---

## 7. 新增优化文档（GPU 安全改造路线）

按你“准备上 H100 训练、先确保不破坏现有功能”的要求，我新增了一份独立修改文档：

- `reports/gpu_optimization_safe_modification_plan_20260312.md`

该文档重点内容：

1. 标注原 GPU 报告中会破坏现有流程的高风险点（FC2 输入契约、FC2->SDF2 回灌、DataFrame 下游兼容、SDF 全链路 AMP 风险）。
2. 给出“先改什么、哪些先不改”的分批方案。
3. 固化回归门槛（功能一致性、数值稳定性、Bellman 阈值、P/Q 经济学形状约束）。
4. 明确 H100 上建议先启用 TF32 + 分模块 AMP（PV/FC2 开，SDF 先关）。

---

## 8. 2026-03-13：Episode 流程对齐 SSH 版本（你本轮要求）

本轮按你提供的差异清单，把本地 `training/episode.py` 回调到 SSH 口径，核心是修复 `add_FC1loss` 状态泄漏和 SDF 宏观 batch 分组差异。

### 8.1 `add_FC1loss` 生命周期

- 在 episode0 的 `FC2 -> SDF(宏观重建)` 段中，保持：
  - 进入该段前设 `self.add_FC1loss = True`
  - 该段结束后显式复位 `self.add_FC1loss = False`

原因：避免该标志泄漏到后续不带重建真值列（7/8列输入）的 SDF batch，触发你观察到的 “need >=9 cols, got 8” warning。

### 8.2 `_create_sdf_batches_from_macro_df` 分组逻辑

- 默认 `batch_size` 改回 `1024`。
- 当 `self.add_FC1loss=True`：
  - 分组键改为 `['path', 't']`（若无 `t` 列则回退 `['path']`）；
  - 当 `self.train_mode != '2time'` 且有 `t` 列时，执行 `t > 2` 过滤。
- 当 `self.add_FC1loss=False`：
  - 维持按 `['path']` 分组。

原因：与 SSH 的宏观 pair 构造逻辑一致，避免不同 `t` 的分支样本被混组。

### 8.3 训练模式参数补齐

- `Episode.__init__` 新增 `self.train_mode = '2time'`。
- `run_episode(...)` 新增参数 `train_mode: str = '2time'`，并在开头写入 `self.train_mode = train_mode`。

原因：为 8.2 的分支过滤逻辑提供与 SSH 一致的模式开关。

### 8.4 默认 batch_size 回调

以下函数默认 `batch_size` 统一回调为 `1024`：

- `create_batches`
- `_create_sdf_batches_from_macro_df`
- `_create_firm_batches_from_df`
- `_create_fc2_batches`
- `run`

### 8.5 `generate_data()` 采样参数口径

- `mode='sample'` 时，`Sample(...)` 改为 `n_samples=None`（而非 `n_samples=n_samples`）。

原因：与 SSH 当前使用口径一致。

### 8.6 验证

- 已执行：`python3 -m py_compile training/episode.py`
- 结果：通过。

---

## 9. 2026-03-13：完整训练流程同步（ep0 与 ep>0 分阶段顺序 + FC2 可选）

按你最新要求，完整训练脚本已与 notebook quick test 的 episode 流程保持一致：

- `episode=0`：`sdf1 -> pv -> sdf2 -> (fc2 可选)`
- `episode>0`：`pv -> sdf2 -> (fc2 可选)`

### 9.1 修改文件

- `experiments/run_multi_episode_job.py`
- `experiments/run_multi_episode.py`

### 9.2 关键改动

1. **stage 顺序改为按 episode 动态生成**
   - 不再所有 episode 固定跑 `sdf1`；
   - 从 `episode>0` 开始默认跳过 `sdf1`，直接进入 `pv -> sdf2`。

2. **FC2 阶段改为可选**
   - 两个脚本都新增 CLI 开关：`--enable-fc2 / --no-enable-fc2`；
   - 默认仍为开启（`--enable-fc2`）。

3. **兼容原输出逻辑**
   - 每个已执行的 stage 仍会保存对应 `stage df`；
   - episode summary 结构保持不变，仅 stage 集合按新策略变化。

### 9.3 使用示例

- 保持 FC2：
  - `python3 experiments/run_multi_episode_job.py --n-episodes 10 --enable-fc2`
- 跳过 FC2：
  - `python3 experiments/run_multi_episode_job.py --n-episodes 10 --no-enable-fc2`

### 9.4 验证

- 已执行：`python3 -m py_compile experiments/run_multi_episode_job.py experiments/run_multi_episode.py`
- 结果：通过。

---

## 10. 2026-03-13：补全 run\_multi\_episode\_job 的 P/bz 图输出

你反馈完整训练生成图缺少 `P` 与 `bz` 相关图。根因是 `plot_surfaces()` 只绘制了：

- `P0`
- `PI`
- `bp`
- `Q`

没有包含 `P` 和 `bar_z`。

### 10.1 修改文件

- `experiments/run_utils.py`

### 10.2 修改内容

在 `plot_surfaces(...)` 的网格前向结果中新增：

- `P = out.P`
- `bar_z = out.bar_z`

并把绘图列表由原来的：

- `("p0", P0), ("pi", PI), ("bp", bp), ("q", Q)`

改为：

- `("p0", P0), ("pi", PI), ("p", P), ("barz", bar_z), ("bp", bp), ("q", Q)`

因此每个 episode 现在会额外产出：

- `ep{ep}_p_heatmap.png`
- `ep{ep}_p_surface.png`
- `ep{ep}_barz_heatmap.png`
- `ep{ep}_barz_surface.png`

### 10.3 验证

- 已执行：`python3 -m py_compile experiments/run_utils.py experiments/run_multi_episode_job.py`
- 结果：通过。

---

## 11. 2026-03-13：临时默认停用 FC2 训练

按你“先不训练 FC2”的要求，已将完整训练入口和 notebook quick test 默认行为改为 **不训练 FC2**，但保留可选开关，后续可随时恢复。

### 11.1 修改文件

- `experiments/run_multi_episode_job.py`
- `experiments/run_multi_episode.py`
- `tests/sdf_fc1_two_modes_test.ipynb`

### 11.2 修改内容

1. `run_multi_episode_job.py`
   - `--enable-fc2` 的默认值从 `True` 改为 `False`。
   - 说明文字同步为“默认关闭，显式 `--enable-fc2` 才开启”。

2. `run_multi_episode.py`
   - `--enable-fc2` 的默认值从 `True` 改为 `False`。
   - 说明文字同步更新。

3. notebook quick test
   - 最后一个 quick test 单元中的 `quick_enable_fc2` 默认从 `True` 改为 `False`。

### 11.3 使用方式

- 默认（不训 FC2）：
  - `python3 experiments/run_multi_episode_job.py ...`
- 需要时临时开启 FC2：
  - `python3 experiments/run_multi_episode_job.py --enable-fc2 ...`

### 11.4 验证

- 已执行：`python3 -m py_compile experiments/run_multi_episode_job.py experiments/run_multi_episode.py`
- 结果：通过。

---

## 12. 2026-03-13：宏观序列新增 Delta 图与 FC1 预测/实现 R² 诊断

按你最新要求，新增两类输出：

1. 宏观增量图：
   - `ΔlnK` 对 `t`
   - `ΔlnC` 对 `t`（其中 `lnC = Hatc + LnK`）
2. 在 `df_macro_realized` 生成后、`sdf_fc1` 第二阶段训练前，记录 `lnK` 与 `hatc` 的预测值-真实值 `R²`。

### 12.1 修改文件

- `experiments/run_utils.py`
- `training/episode.py`

### 12.2 具体改动

1. `plot_macro_series(...)` 增强（`experiments/run_utils.py`）
   - 预测/真实同图：
     - `Hatc true vs pred`
     - `LnK true vs pred`
   - 标题内显示 `R²`（若存在预测列）。
   - 新增图文件：
     - `ep{ep}_macro_delta_lnk.png`
     - `ep{ep}_macro_delta_lnc.png`
   - 预测列名兼容：
     - `hatcf/Hatcf`，`lnkf/LnKF`
   - 真实列名兼容：
     - `Hatc/hatc`，`LnK/lnk`

2. FC2→SDF2 间隙 R² 诊断（`training/episode.py`）
   - 新增 `_macro_forecast_r2(df_macro)` 计算函数。
   - 在 episode0 的 FC2 训练完成后、`build_sdf_pairs_from_macro_ts(...)` 之前执行：
     - 日志打印 `R2(Hatc)` 与 `R2(LnK)`
     - 将结果写入 `module_summaries['macro_diag_before_sdf2']`

### 12.3 验证

- 已执行：`python3 -m py_compile experiments/run_utils.py training/episode.py`
- 结果：通过。

---

## 13. 2026-03-13：Episode 三模式（Mode0/ModeA/ModeB）与交替调度落地

按你最新定义，训练流程升级为三模式，并支持 `episode>0` 的“固定/交替”调度。

### 13.1 修改文件

- `training/episode.py`
- `experiments/run_multi_episode_job.py`
- `experiments/run_multi_episode.py`

### 13.2 `training/episode.py` 关键变更

1. `run_episode(...)` 新增参数：
   - `episode_mode: Optional[str] = None`
   - 支持：`mode0` / `modea` / `modeb`（`None` 自动：`episode0->mode0`，其余默认 `modeb`）。

2. 新增三模式执行逻辑：
   - `mode0`：
     - `Sample.build_sdf_fc1_df()` 训练 SDF(stage1)
     - `Sample.build_policy_value_df()` 训练 Policy/Value
     - `SimulateTS(horizon=1)` 后进行（可选）FC2
     - 再把 `df_macro` 转 SDF pairs，`add_FC1loss=True` 做 SDF(stage2)
   - `modea`：
     - `Sample.build_policy_value_df()` 训练 Policy/Value
     - `SimulateTS(horizon=1)` 后进行（可选）FC2
     - 再做 SDF(stage2, `add_FC1loss=True`)
   - `modeb`：
     - `SimulateTS(horizon=T)` 直接生成训练数据
     - 在该数据上训练 Policy/Value、SDF（以及可选 FC2）

3. 增加内部辅助函数，减少重复并固定流程：
   - `_simulate_df(...)`
   - `_run_fc2_epochs(...)`
   - `_run_sdf_recon_from_macro(...)`
   - `_resolve_episode_mode(...)`

4. `add_FC1loss` 状态保护：
   - 入口先置 `False`
   - SDF 二阶段仅在局部 `True`
   - `try/finally` 确保每次 `run_episode` 结束后复位 `False`
   - 避免阶段间状态串扰（对应你之前看到的 recon 维度 warning 根源之一）。

5. `module_summaries` 扩展：
   - `sdf_fc1_stage1`
   - `sdf_fc1_stage2`
   - 保留兼容键 `sdf_fc1`（指向当前主 SDF 阶段结果）
   - `macro_diag_before_sdf2`（mode0/modeA 的 SDF 二阶段前 R²）

### 13.3 Runner 调度能力（两个脚本）

1. 新增 CLI 参数：
   - `--post0-mode {modea,modeb,alternate}`
   - `--alternate-start {modea,modeb}`

2. 调度规则：
   - `episode 0` 固定 `mode0`
   - `episode > 0`：
     - 固定 `modea` 或 `modeb`
     - 或 `alternate` 按起始模式交替

3. 执行方式调整：
   - 每个 episode 改为一次完整 `run_episode(...)` 调用（不再由外层 stage 反复拼接）
   - `simulate_kwargs` 统一传：
     - `horizon_mode1=1`（mode0/modeA）
     - `horizon=simulate_horizon`（modeB）

4. `run_multi_episode.py` 的 loss 汇图函数增强：
   - 跳过非数值或非字典项，防止 summary 结构变化导致画图报错。

### 13.4 验证

- 已执行：  
  `python3 -m py_compile training/episode.py experiments/run_multi_episode_job.py experiments/run_multi_episode.py`
- 结果：通过。

- 运行级 smoke test 在当前环境被 OpenMP 共享内存限制阻断（非代码语法问题）：
  - 报错：`OMP: Error #179: Function Can't open SHM2 failed`

---

## 14. 2026-03-13：Macro R² 口径改为全样本 + R² 图改散点（含 y=x）

按你的要求，R² 不再只用 `branch=-1`，改为 **全样本口径**；且 `Hatc/LnK` 的 R² 可视化由时间序列改为 **散点图 + 45° 参考线 (`y=x`)**。

### 14.1 修改文件

- `experiments/run_utils.py`
- `training/episode.py`

### 14.2 具体改动

1. `plot_macro_series(...)`（`experiments/run_utils.py`）
   - 删除 `branch==-1` 过滤，统一使用 `df_macro` 全样本。
   - `ep{ep}_macro_hatc.png` / `ep{ep}_macro_lnk.png` 改为：
     - 横轴：true
     - 纵轴：pred
     - 散点：全样本点云
     - 参考线：`y=x`
     - 标题保留 `R²`。
   - `Delta LnK` 与 `Delta lnC` 图仍保留按 `t` 展示，但其构造也改为全样本按 `t` 聚合（不再筛分支）。

2. `_macro_forecast_r2(...)`（`training/episode.py`）
   - 删除 `branch==-1` 过滤。
   - R² 改为全样本逐点计算（不按 `t` 聚合后再算）。
   - 输出新增 `n_obs`，并保留 `n_t` 键用于兼容（其值现在同样表示样本数）。

### 14.3 验证

- 已执行：`python3 -m py_compile experiments/run_utils.py training/episode.py`
- 结果：通过。

---

## 15. 2026-03-13：R² 散点图按 branch 分组显示

根据你最新要求，“散点图还是用 branch 来画”，已将 `Hatc/LnK` 的预测-真实散点图改为按 `branch` 分组着色并显示图例。

### 15.1 修改文件

- `experiments/run_utils.py`

### 15.2 修改内容

1. `plot_macro_series(...)` 中：
   - 读取 `df_macro['branch']`（若存在）。
   - 在 `ep{ep}_macro_hatc.png` 与 `ep{ep}_macro_lnk.png` 中：
     - 仍是 `pred vs true` 散点；
     - 仍保留 `y=x` 参考线；
     - 新增按 `branch` 分组绘制（不同颜色+图例 `branch=...`）。

2. 当前口径说明：
   - R² 仍为全样本计算（上一节修改保持不变）；
   - 可视化点云按 branch 分组展示。

### 15.3 验证

- 已执行：`python3 -m py_compile experiments/run_utils.py`
- 结果：通过。

---

## 16. 2026-03-13：R² 改为按 branch 统计（图与诊断一致）

你补充要求“R方也是分branch”，已补齐为：
- 图上按 branch 展示各自 R²；
- 训练阶段 macro 诊断输出也包含按 branch 的 R²。

### 16.1 修改文件

- `experiments/run_utils.py`
- `training/episode.py`

### 16.2 修改内容

1. `plot_macro_series(...)`（`experiments/run_utils.py`）
   - `Hatc/LnK` 散点图图例改为：
     - `branch=k (R2=...)`
   - 即每个 branch 单独计算并显示 R²（仍保留 `y=x` 参考线）。

2. `_macro_forecast_r2(...)`（`training/episode.py`）
   - 在原有全样本 `r2_hatc`、`r2_lnk` 之外，新增：
     - `r2_hatc_by_branch`
     - `r2_lnk_by_branch`
     - `n_obs_by_branch`
   - 这些字段进入 `module_summaries['macro_diag_before_sdf2']` / `macro_diag_modeb`，便于后续分析。

### 16.3 验证

- 已执行：`python3 -m py_compile experiments/run_utils.py training/episode.py`
- 结果：通过。

---

## 17. 2026-03-13：P0/PI 对齐理论（KKT + 条件在 eta=1 的 signed FOC）

按你的要求，`bp` 只通过 `P0/PI` 方程决定，不做经验正则；同时把 FOC 对齐到“条件在再融资事件上的矩条件”并保留符号信息。

### 17.1 修改文件

- `config/hyperparams.py`
- `training/episode.py`
- `reports/p0_pi_kkt_boundary_theory_and_impl_20260313.md`（新增）

### 17.2 代码改动

1. 新增 KKT 超参数（`config/hyperparams.py`）
- `p0_kkt_weight`, `pi_kkt_weight`
- `kkt_boundary_eps`, `kkt_boundary_temp`
- `kkt_inner_weight`, `kkt_boundary_weight`

2. `Episode` 新增 KKT 构造函数（`training/episode.py`）
- `_compute_bp_kkt_penalty(bp, foc_residuals, eta_children)`：
  - 内点罚项：`FOC^2`
  - 下边界罚项：`relu(FOC)`（对应 `FOC<=0`）
  - 上边界罚项：`relu(-FOC)`（对应 `FOC>=0`）
  - 采用 sigmoid 软边界权重区分内点/边界区域；
  - 使用 `eta_children` 进行 active 过滤，避免 `eta=0` 样本稀释 KKT 信号。

3. `Episode` 新增 FOC 条件矩函数（`training/episode.py`）
- `_compute_conditional_signed_foc_terms(...)`：
  - 改为最小化 \(\left(E[\text{FOC}\mid \eta=1]\right)^2\)；
  - 保留 FOC 符号（signed moment）；
  - 不再对 FOC 使用 `compute_aio_residual` 的平方+`abs(product)` 聚合。

4. 将上述两项接入 `P0/PI` 总损失（`training/episode.py`）
- `_compute_p0_loss(...)`：
  - 使用条件 signed FOC loss；
  - 增加 `+ p0_kkt_weight * L_kkt`
- `_compute_pi_loss(...)`：
  - 使用条件 signed FOC loss；
  - 增加 `+ pi_kkt_weight * L_kkt`
- 同步增加日志诊断键：
  - `p0_kkt_*`, `pi_kkt_*`
  - `p0_foc_active_ratio`, `p0_foc_signed_moment`, `p0_foc_cond_abs_mean`
  - `pi_foc_active_ratio`, `pi_foc_signed_moment`, `pi_foc_cond_abs_mean`

### 17.3 理论说明文档

新增文档：`reports/p0_pi_kkt_boundary_theory_and_impl_20260313.md`

内容包括：
- 从 `max_{0<=b'<=1} J(b')` 出发的 KKT 推导；
- 三条条件（内点 / 下界 / 上界）的数学来源；
- `E[FOC|eta=1]=0` 的条件矩实现与代码对应关系。

### 17.4 验证

- 已执行：`python3 -m py_compile config/hyperparams.py training/episode.py`
- 结果：通过。

---

## 18. 2026-03-13：bp 高位问题修复（eta 稀疏重权重 + KKT 上边界强化 + eta 条件重采样）

针对 `quick_20260313_133151` 中 `bp` 仍贴近上界（尤其 `bp0`）的问题，按“只改估计强度，不改经济方程”的原则进行了三类修复。

### 18.1 修改文件

- `config/hyperparams.py`
- `training/episode.py`

### 18.2 代码改动

1. 强化 KKT 上边界约束（`training/episode.py`）
- 在 `_compute_bp_kkt_penalty(...)` 中新增 `kkt_high_weight`：
  - 总罚项改为：
    - `kkt_inner_weight * L_inner + kkt_boundary_weight * (L_low + kkt_high_weight * L_high)`
  - 目的：针对 `bp≈1` 区域加大纠偏力度。
- 新增 `kkt_high_weight` 诊断输出。

2. FOC 的 z 惩罚改为仅在 eta 活跃子样本上计算（`training/episode.py`）
- `_compute_conditional_signed_foc_terms(...)` 现在使用 `active_bool = (sum_j eta_j > 0)`：
  - `loss_foc = mean(FOC | active)^2`
  - `penalty_z_foc = compute_z_penalty(foc_abs[active], z[active], ...)`
- 不再把 inactive 样本（`eta=0`）以 0 值混入 `mean`，避免信号被稀释。
- 新增诊断：`foc_active_n`。

3. eta 稀疏时对 bp 相关项做条件重权重（`training/episode.py`）
- 新增 `_compute_eta_active_boost(active_ratio)`：
  - `boost = clip(target_ratio / active_ratio, 1, max_boost)`
- 在 `_compute_p0_loss(...)` / `_compute_pi_loss(...)` 中：
  - `bp_terms = boost * (loss_foc + penalty_z_foc + kkt_penalty)`
  - 总损失改为 `main + penalty_z (+ penalty_b) + bp_terms`
- 新增诊断：
  - `p0_bp_terms`, `pi_bp_terms`
  - `p0_eta_active_boost`, `pi_eta_active_boost`

4. Policy/Value batch 增加 eta=1 条件重采样（`training/episode.py`）
- `_create_firm_batches_from_df(..., eta_resample=True)` 新增 `eta_resample` 参数。
- 当启用时，根据 `max_j eta_child_j` 划分 active/inactive，按目标 active 占比重采样。
- 仅用于 Policy/Value 默认路径；ModeB 的 SDF firm-batch 显式传 `eta_resample=False`，避免影响 SDF 数据分布。

5. 新增超参数（`config/hyperparams.py`）
- `kkt_high_weight`
- `eta_active_reweight_enabled`
- `eta_active_target_ratio`
- `eta_active_max_reweight`
- `pv_eta_resample_enabled`
- `pv_eta_resample_active_share`

### 18.3 验证

- 已执行：`python3 -m py_compile config/hyperparams.py training/episode.py`
- 结果：通过。

---

## 19. 2026-03-13：bp 训练增强（二次修复：自适应同量级 + bp-only 精修）

针对新一轮 quick 结果中 `bp` 仍高位的问题，进一步对 `P0/PI` 中的 `bp` 相关项做了“同量级自适应”并新增 `bp-only` 精修阶段。

### 19.1 修改文件

- `config/hyperparams.py`
- `training/episode.py`

### 19.2 代码改动

1. `bp` 项自适应同量级（`training/episode.py`）
- 新增 `_compute_bp_adaptive_scale(main_loss, bp_terms_after_eta)`：
  - 目标：令 `bp` 相关项达到 `bp_target_main_ratio * main_loss` 的量级。
  - 公式：
    - `scale = clip(target/main_bp_terms, min_scale, max_scale)`
- 在 `_compute_p0_loss(...)` / `_compute_pi_loss(...)` 中：
  - `bp_terms_base = loss_foc + penalty_z_foc + kkt_penalty`
  - `bp_terms_after_eta = eta_active_boost * bp_terms_base`
  - `bp_terms = bp_adapt_scale * bp_terms_after_eta`
- 该项进入总损失（替代之前仅 `eta_boost` 的版本）。

2. `bp-only` 精修阶段（`training/episode.py`）
- 新增 `_set_policy_bp_only_freeze(enable)`：
  - 仅解冻 `shared_model.bp0_head` 和 `shared_model.bpI_head`。
- `train_step(...)` 增加 `bp_only_step` 判定，并接入该冻结逻辑。
- `_run_batches(...)` 中每个 policy epoch 结束后追加：
  - `bp_refine_steps_per_epoch` 轮小步训练；
  - 每轮仅用 `policy_loss_terms=['p0','pi']`，并设置 `self._bp_only_stage=True`；
  - 可通过 `bp_refine_batch_cap` 限制每轮精修 batch 数。

3. 新增超参数（`config/hyperparams.py`）
- `bp_adaptive_enabled`
- `bp_target_main_ratio`
- `bp_adaptive_min_scale`
- `bp_adaptive_max_scale`
- `bp_refine_steps_per_epoch`
- `bp_refine_batch_cap`

4. 诊断项扩展（`training/episode.py`）
- `p0_bp_terms_base`, `p0_bp_terms_after_eta`, `p0_bp_adapt_scale`
- `pi_bp_terms_base`, `pi_bp_terms_after_eta`, `pi_bp_adapt_scale`

### 19.3 验证

1. 语法检查：
- `python3 -m py_compile config/hyperparams.py training/episode.py`
- 结果：通过。

2. checkpoint 级分解验证（`quick_20260313_152037`, ep2）：
- 修复前：`bp_terms` 约 `1e-4~1e-3`，显著小于 `main_loss`。
- 修复后（同一 checkpoint 做前向分解）：
  - `p0_main≈0.0386`，`p0_bp_terms≈0.0116`，`p0_bp_adapt_scale≈35.9`
  - `pi_main≈0.0588`，`pi_bp_terms≈0.0176`，`pi_bp_adapt_scale≈50.5`
- 说明 `bp` 项已提升到与主项可比的量级（约 0.3 比例）。

---

## 20. 2026-03-13：可视化补充（新增 Bar_i / bari 图）

按用户要求，在多 episode 自动出图中新增 `Bar_i` 的 b-z 可视化。

### 20.1 修改文件

- `experiments/run_utils.py`

### 20.2 代码改动

1. `plot_surfaces(...)`
- 从 `PolicyValueOutput` 额外提取：
  - `bar_i = out.bar_i.reshape(B.shape)`
- 在绘图循环中新增 `(\"bari\", bar_i)`：
  - 生成 `ep{ep}_bari_heatmap.png`
  - 生成 `ep{ep}_bari_surface.png`

### 20.3 验证

- 已执行：`python3 -m py_compile experiments/run_utils.py`
- 结果：通过。

## 2026-03-13 追加：P 违约区恢复 + 高 M 传导抑制

### 问题
- 现象：`bp` 基本正常，但 `P` 缺少 `=0` 区域，违约不明显；同时 `M` 偏高时会把 `P0/PI` continuation 项抬高。

### 修改
1. `models/policy_value.py`
- `cal_phats` 中 `P` 从 `softplus(Phat)` 改为 `P=max(0,Phat)`（`torch.clamp_min(Phat,0.0)`），恢复显式违约零值区。

2. `training/episode.py`
- 在 `_compute_p0_loss/_compute_pi_loss` 中对输入 `M` 增加可配置裁剪：`pv_use_clipped_m`, `pv_m_clamp_min`, `pv_m_clamp_max`。
- 新增诊断项：
  - `p0_log_mean_M_raw/used`, `p0_M_raw_p90/used_p90`
  - `pi_log_mean_M_raw/used`, `pi_M_raw_p90/used_p90`

3. `config/hyperparams.py`
- 新增超参数：
  - `pv_use_clipped_m=True`
  - `pv_m_clamp_min=0.7`
  - `pv_m_clamp_max=1.3`

4. `experiments/run_utils.py`
- CLI 默认 `build_hyperparams()` 同步稳定口径：
  - `fc1_recon_weight=1.0`
  - `sdf_stage1_lr=1e-4`
  - `sdf_stage1_moment_weight=5.0`
  - `sdf_moment_weight=3.0`
  - `pv_use_clipped_m=True`, `pv_m_clamp_min=0.7`, `pv_m_clamp_max=1.3`

### 预期
- `P` 恢复可见 `=0` 区域；
- `bar_z` 不再被“全域正 P”压缩到近 0；
- 上游 `M` 短期异常不再直接把 `P0/PI` 顶偏。

## 2026-03-13 追加：bp 修复后 P/M 异常的 SDF 稳定化

### 问题
- `bp` 修复后，`M` 在后续 episode（特别是 SDF 第二阶段）偏高，进而通过 `P0/PI` 的 `M*P'*(1-bar_z')` continuation 项放大到 `P`。

### 修改
1. `config/hyperparams.py`
- 新增：
  - `sdf_stage2_lr`
  - `sdf_log_mean_target`（默认 `log(0.98)`）
  - `sdf_log_mean_anchor_weight_stage1`
  - `sdf_log_mean_anchor_weight_stage2`

2. `training/episode.py`
- `_configure_sdf_lr_for_phase`：`add_FC1loss=True` 时支持单独 `sdf_stage2_lr`。
- `_compute_sdf_loss`：新增均值锚损失 `mean_anchor_loss=(log(E[M])-target)^2`，并加入总损失。
- 新增诊断项：`sdf_mean_anchor_loss/weight/target`。
- 新增非有限值保护：`mean_anchor_loss` 非有限时置 0 并告警。

3. `experiments/run_utils.py`
- `build_hyperparams()` 默认同步：
  - `sdf_stage2_lr=2e-4`
  - `sdf_moment_weight=5.0`
  - `sdf_log_mean_anchor_weight_stage1=1.0`
  - `sdf_log_mean_anchor_weight_stage2=5.0`

### 目的
- 在保留 bp 修复的同时，抑制阶段切换后 SDF 均值漂移，降低 `M` 异常向 `P` 的连锁传导。

## 2026-03-13 追加：bp 约束梯度通道修复（FOC/KKT 使用 Phat'）

### 背景
- 诊断中 `p0_pgrad_abs_mean/pi_pgrad_abs_mean` 长期接近 0，导致 `bp` 相关 FOC/KKT 信号偏弱，`bp` 容易贴上边界。
- 直接原因是 `P=max(Phat,0)` 在违约区对 `bp` 的梯度可大面积为 0。

### 修改
1. `config/hyperparams.py`
- 新增开关：`bp_foc_use_phat_children=True`
- 含义：仅在 `bp` 的 FOC/KKT 梯度通道中用 `Phat`，Bellman 主方程仍使用 `P`。

2. `training/episode.py`
- `_compute_p0_loss` / `_compute_pi_loss` 中新增：
  - `P_children_for_foc = Phat_children`（当开关开启）
  - FOC 计算改为使用 `P_children_for_foc`
- 追加诊断项：
  - `p0_bp_foc_use_phat`
  - `pi_bp_foc_use_phat`

### 设计原则
- 不改经济学违约语义：`P=0` 区域保留在 Bellman 主方程。
- 只修复优化信号：让 `∂(continuation)/∂bp` 在违约邻域仍可学习，从而让 `bp` 通过 P0/PI 方程真正被约束。

## 2026-03-13 追加：KKT 上边界软区放宽（针对 bp≈0.8 仍偏高）

### 问题
- 原实现 `kkt_boundary_eps=0.02` 时，上边界权重 `w_high` 主要在 `bp>0.98` 才显著。
- 对你当前常见的 `bp≈0.75~0.9`，KKT 上边界约束几乎不生效。

### 修改
1. `config/hyperparams.py`
- 新增：
  - `kkt_boundary_eps_low: Optional[float] = None`
  - `kkt_boundary_eps_high: Optional[float] = 0.20`
- 语义：允许上下边界软区宽度分离；默认上边界从 `bp>0.8` 起逐步施压。

2. `training/episode.py`
- `_compute_bp_kkt_penalty`：
  - `w_low = sigmoid(temp * (eps_low - bp))`
  - `w_high = sigmoid(temp * (bp - (1 - eps_high)))`
- 兼容逻辑：若 `eps_low/high` 为 `None`，回退到旧参数 `kkt_boundary_eps`。
- 诊断新增：`kkt_eps_low`, `kkt_eps_high`。

### 目的
- 不改变 KKT 形式，只放宽“近上边界”判定区间，使 `bp` 在 0.8 左右时也能收到边界信号，而不是仅在 1.0 附近才受约束。

## 2026-03-13 追加：`Sample` + `SimulateTS` Tensor-first（DL_AP_Tensor Phase 1）

### 背景
- 目标是把数据模拟阶段迁移到 CUDA，避免中间 DataFrame/`.item()` 导致的 CPU/GPU 往返。
- 同时为后续 FC2 重写预留 tensor 数据协议。

### 修改
1. `data/tensor_data.py`（新增）
- `TensorTable`：统一二维张量 + 列名，并提供末端 `to_dataframe()`。
- `TensorSimulationOutput`：统一管理 firm/macro tensor 输出。
- `cat_rows`：安全拼接 path 级结果（支持空张量）。

2. `data/simulate_ts.py`
- 新增 `FIRM_COLUMNS` / `MACRO_COLUMNS`。
- 新增 `simulate_tensor()`：全路径 tensor-native 模拟并返回 `TensorSimulationOutput`。
- `simulate()` 改为内部调用 `simulate_tensor()`，仅末端转 DataFrame。
- 新增 tensor 子流程：
  - `_initialize_path_tensor`
  - `_simulate_path_tensor`
  - `_process_node_tensor`
  - `_expand_branches_tensor`
  - `_predict_macro_fc1_tensor`
  - `_apply_entry_tensor`
- `_resource_accounting` 支持 tensor/float 标量输入。

3. `data/sample.py`
- 新增 `SDF_COLUMNS` / `PV_COLUMNS`。
- 新增 `build_sdf_fc1_tensor()`。
- `build_sdf_fc1_df()` 改为 tensor 末端导出。
- 新增 `build_policy_value_tensor()`（parent+children，tensor 生成）。
- `build_policy_value_df()` 改为走 tensor 生成后再转 DataFrame。

4. `data/__init__.py`
- 导出 `TensorTable` 与 `TensorSimulationOutput`。

### 校验
- `python3 -m py_compile data/tensor_data.py data/simulate_ts.py data/sample.py` 通过。

### 备注
- 本阶段先完成 `Sample/SimulateTS` 的 tensor-first 主路径。
- `episode` 训练端尚未全面切到 tensor batch（下一阶段再接）。

## 2026-03-13 追加：Episode 训练端 tensor batch 接入

### 目标
- 在 `training/episode.py` 中去掉 SDF/PV 训练前的 pandas 拼装，直接从 tensor 数据源构建 batch。

### 修改
1. `config/hyperparams.py`
- 新增 `use_tensor_pipeline=True` 开关（默认启用）。

2. `training/episode.py`
- 新增 tensor 状态：
  - `tensor_firm / tensor_macro / tensor_sdf`
- 新增工具方法：
  - `_create_firm_batches_from_tensor`
  - `_build_sdf_pairs_from_macro_tensor`
  - `_create_sdf_batches_from_macro_tensor`
  - `_build_batches_from_parent_children`（统一打包与 eta 重采样）
  - `_macro_forecast_r2_tensor`
  - `_simulate_tensor`
- parent/child 对齐使用共享 key 基数，修复 child 含 entrant 新 ID 时的潜在错配。
- `mode0/modeA/modeB` 训练流改为：
  - Sample/SimulateTS 的 tensor 输出 -> 直接 batch -> 训练
  - 仅 FC2 需要时才导出 DataFrame。
- `create_batches()` 改为优先 tensor 分支。
- FC2 兼容：`_run_fc2_epochs` 在仅有 tensor 数据时自动末端转 DataFrame。
- 训练后兼容导出：`run_episode` 结束时自动把 tensor 数据落到 `df/df_macro/df_sdf`，不破坏现有绘图/保存脚本。

### 验证
- `python3 -m py_compile training/episode.py config/hyperparams.py` 通过。
- 最小 smoke test 通过：
  - tensor firm batch / tensor sdf(stage1,stage2) batch 构建正常；
  - `run_episode(modeB)` 调度可运行。
