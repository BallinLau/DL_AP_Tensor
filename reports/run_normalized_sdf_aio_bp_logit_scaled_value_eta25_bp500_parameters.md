# `eta25_bp500` Full-Training Slurm 参数手册

本文档说明当前版本的：

```text
slurm/run_normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500.slurm
```

适用分支：`exp/equity-value-xz-scaling`。

实现审计基线（文档编写前）：`b4f738eee47e455a4af90c8e9b90acb0e31f9f89`。

这不是 isolated BP diagnostic。它最终运行：

```text
experiments/run_multi_episode_job.py
```

因此它执行完整 multi-episode equilibrium training，包括 simulation、FC1/SDF、P/Q value training、BP teacher cache 和 BP distillation。

## 1. 调用链与参数优先级

实际调用链为：

```text
run_normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500.slurm
  -> run_normalized_sdf_aio_bp_logit_scaled_value_pilot.slurm
    -> run_normalized_sdf_aio_bp_logit.slurm
      -> experiments/run_multi_episode_job.py
```

只有第一层文件由 `sbatch` 直接提交，因此只有第一层的 `#SBATCH` 指令生效。后两层是由 `exec bash` 执行的普通 shell 脚本，其 `#SBATCH` 行不会重新申请资源。

环境变量按以下规则处理：

1. `VAR="${VAR:-default}"`：可以通过 `sbatch --export=ALL,VAR=value` 覆盖。
2. `export VAR=value`：专用实验强制值，外部同名变量会被覆盖。
3. Python `HyperParams` 默认值：Slurm 没有显式传递时才生效。

## 2. 当前实验的核心语义

默认运行的核心配置为：

| 模块 | 当前设置 | 含义 |
|---|---:|---|
| Value 参数化 | `PV_VALUE_SCALE_MODE=exp_xz` | P0/PI 网络学习缩放后的 latent value，物理 value 通过 `exp(x+z)` 尺度还原 |
| Bellman 标准化 | `PV_BELLMAN_NORMALIZE_BY_VALUE_SCALE=1` | P0/PI Bellman residual 按同一 value scale 标准化 |
| SDF wealth loss | `signed_aio` | 使用 signed AiO 条件矩目标，不使用 legacy absolute-log loss |
| SDF moment constraint | `augmented_lagrangian` | 当前 `build_hyperparams()` 默认使用 pooled PHR inequality AL |
| BP loss | `logit` | P0/PI BP head 在 logit 空间拟合 grid teacher |
| BP current-eta train share | `0.25` | 只对 BP distillation **训练 cache** 重采样，使 current parent `eta_t=1` 约占 25% |
| BP optimizer budget | `500` | 每个 BP stage 最多接受 500 个 successful optimizer updates |
| BP trainable scope | `heads_only` | BP stage 只更新 BP heads，共享 trunk 和 value/Q heads 冻结 |
| BP validation | natural | validation cache 不做 current-eta 重采样；best checkpoint 仍由 natural validation 选择 |
| Economic future eta | exact integration | Bellman/teacher 对 `eta_{t+1}` 使用精确 Bernoulli integration |
| Economic `ZETA` | `0.03` | 未来 eta 的经济概率，不受 25% training sampler 影响 |
| PV mixture | enabled, ratio `0.20` | Episode 1 起，在固定总预算内混合 SimulateTS 与 coverage Sample parent groups |
| Post-0 mode | `modeb` | Episode 1 以后使用 Mode B |
| Treatment | B | PV 更新后重新 simulation，再进入后续 FC1/SDF stage |
| FC2 | disabled | 默认不训练 FC2 |

最重要的口径区别是：

```text
P_train(eta_t = 1) = 0.25
P_economic(eta_{t+1} = 1) = ZETA = 0.03
```

前者只是 BP-head training cache 的 current-state sampling distribution；后者仍进入 Bellman teacher 的 future-state expectation。二者不能混用。

## 3. 推荐提交方式

### 3.1 为什么必须显式传 `REQUIRED_COMMIT`

当前专用 wrapper 的默认逻辑是：

```bash
git log -1 --format=%H -- slurm/run_normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500.slurm
```

也就是取“最后一次修改这个 Slurm 文件的 commit”，而不是当前 `HEAD`。在本文档审计时：

```text
Slurm last-touch commit = 894ce9dfebf263a19cf65be68306a17a1be9cdcb
current HEAD             = b4f738eee47e455a4af90c8e9b90acb0e31f9f89
```

基础脚本要求 `ACTUAL_COMMIT == REQUIRED_COMMIT`，所以当前分支直接执行裸 `sbatch` 会在训练前失败。最稳妥的运行方法是从服务器当前 checkout 动态读取 `HEAD`：

```bash
cd /home/fit/zhuyingz/WORK/LiuHao/DL_AP_Tensor
git switch exp/equity-value-xz-scaling
git pull --ff-only

HEAD_SHA="$(git rev-parse HEAD)"

sbatch --export=ALL,REQUIRED_COMMIT="$HEAD_SHA" \
  slurm/run_normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500.slurm
```

这会同时保留两个保护：

- 作业只运行提交时服务器上实际 checkout 的 commit；
- working tree 非空时基础脚本仍会失败，避免运行未记录代码。

### 3.2 正式运行并自定义输出目录

```bash
cd /home/fit/zhuyingz/WORK/LiuHao/DL_AP_Tensor
HEAD_SHA="$(git rev-parse HEAD)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="/home/fit/zhuyingz/WORK/LiuHao/cachedir/eta25_bp500_${STAMP}"

JOB_ID=$(sbatch --parsable \
  --export=ALL,REQUIRED_COMMIT="$HEAD_SHA",RUN_ROOT="$RUN_ROOT" \
  slurm/run_normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500.slurm)

echo "JOB_ID=$JOB_ID"
echo "RUN_ROOT=$RUN_ROOT"
```

不要在变量值中写 Markdown 转义的 `\_`；shell 路径必须使用普通下划线 `_`。

### 3.3 小规模 smoke

`QUICK_TEST=1` 在这一 wrapper 链中不会可靠缩小规模，因为 pilot 同时显式传递了 `N_PATHS`、`EPOCHS`、`BATCH_SIZE` 和 `SIMULATE_HORIZON`，这些值会在 Python 的 quick-test 默认值之后再次覆盖。应直接覆盖规模参数：

```bash
cd /home/fit/zhuyingz/WORK/LiuHao/DL_AP_Tensor
HEAD_SHA="$(git rev-parse HEAD)"

sbatch --export=ALL,\
REQUIRED_COMMIT="$HEAD_SHA",\
N_EPISODES=2,\
EPOCHS=2,\
N_PATHS=20,\
POST0_N_PATHS=20,\
PV_BATCH_SIZE=512,\
SDF_FC1_BATCH_SIZE=256,\
SIMULATE_GROUP_SIZE=32,\
SIMULATE_HORIZON=10,\
EPISODE0_SDF_EPOCHS_PER_ROUND=5,\
EPISODE0_SDF_MAX_ROUNDS=5,\
SDF_TRUE_ONLY_EPOCHS=5,\
PV_EVAL_EPOCHS=2,\
BP_DISTILL_MAX_OPTIMIZER_STEPS=20 \
slurm/run_normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500.slurm
```

这里显式把 BP budget 改为 20，所以它只是 smoke，不再是正式 `bp500` 实验。

## 4. Slurm 资源参数

| `#SBATCH` 参数 | 当前值 | 作用 | 修改后的结果 |
|---|---:|---|---|
| `-J` | `dl-scaled-eta25-bp500` | 队列中显示的 job name | 只影响识别，不影响训练 |
| `-N` | `1` | 节点数 | 当前训练设计为单节点；增加节点不会自动获得分布式加速 |
| `-p` | `a01` | partition | 必须是服务器存在且允许 GPU 的 partition |
| `-o` | `.../dl_scaled_eta25_bp500_output.%j` | stdout | `%j` 替换为 job id |
| `-e` | `.../dl_scaled_eta25_bp500_error.%j` | stderr | Python warning、traceback 通常在这里 |
| `--ntasks-per-node` | `1` | 单节点 task 数 | 当前只启动一个 Python process |
| `--gres` | `gpu:1` | GPU 数量 | 代码固定使用 `cuda:0`；申请更多 GPU 不会自动使用 |
| `--cpus-per-task` | `8` | CPU worker/thread 预算 | 同时设置 OMP/MKL 线程数；过高可能增加调度等待 |
| `--time` | `1-00:00:00` | 24 小时上限 | 超时会由 Slurm 终止；可提交时用 `--time=...` 覆盖 |

注意：`LOG_DIR` 环境变量不会改写已经由 Slurm 解析的 `#SBATCH -o/-e`。要临时改日志路径，应使用：

```bash
sbatch --output=/path/output.%j --error=/path/error.%j ...
```

## 5. 路径、环境与复现参数

| 参数 | 有效默认值 | 是否可覆盖 | 作用与设置结果 |
|---|---|---|---|
| `WORK_ROOT` | `/home/fit/zhuyingz/WORK/LiuHao` | 是 | 项目、cache、Matplotlib cache 和默认日志根目录 |
| `REPO_DIR` | `$WORK_ROOT/DL_AP_Tensor` | 是 | 代码仓库位置；必须是目标 branch 的 clean checkout |
| `LOG_DIR` | `$WORK_ROOT/logs` | 是，但不改 `#SBATCH` 日志 | 基础脚本创建该目录；stdout/stderr 仍由 `#SBATCH` 固定路径决定 |
| `RUN_ROOT` | `$WORK_ROOT/cachedir/normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_<timestamp>` | 是 | 本次模型、数据、图片、summary 和 failure report 的根目录 |
| `REQUIRED_COMMIT` | Slurm 文件 last-touch commit | 是，且当前建议必须覆盖 | 与运行时 `git rev-parse HEAD` 不一致会在训练前退出 |
| `SEED` | `12345` | 是 | 设置 Python、NumPy、Torch 和 CUDA 的全局 RNG |
| `BP_CURRENT_ETA_RESAMPLE_SEED` | `13579` | **否，wrapper 强制** | 独立 CPU generator；episode `e` 使用 `13579 + e * 100003`，不消耗 global Torch RNG |
| `SDF_CHILD_BANK_SEED` | `12345` | 是 | SDF fresh child bank 的基础随机种子 |
| `PV_MIXTURE_SEED` | `24680` | 是 | mixture selection 的基础种子；episode 间加入 deterministic offset |

基础脚本还固定：

```text
conda environment = DL_HL
device            = cuda:0
OMP_NUM_THREADS   = SLURM_CPUS_PER_TASK (默认 8)
MKL_NUM_THREADS   = SLURM_CPUS_PER_TASK (默认 8)
PYTHONUNBUFFERED  = 1
MPLCONFIGDIR      = $WORK_ROOT/.mplcache
```

CUDA 不可用时，preflight 会直接退出，不会静默切换 CPU。

## 6. Episode、数据与 batch 参数

| 参数 | 有效默认值 | 作用 | 调大/调小的主要结果 |
|---|---:|---|---|
| `N_EPISODES` | `10` | 运行 Episode 0 到 9 | 增大可增加 outer fixed-point 更新次数，也线性增加时间和输出 |
| `EPOCHS` | `100` | `run_episode()` 的通用 epoch 预算 | 影响未被 stage-specific 参数覆盖的阶段；不是所有 stage 都恰好跑 100 |
| `N_PATHS` | `400` | Episode 0 数据路径数，且用于最终 simulation | 增大改善覆盖但增加生成和训练成本 |
| `POST0_N_PATHS` | `400` | Episode > 0 的 simulation path 数 | 增大后 post-0 firm/macro 数据更多 |
| `BATCH_SIZE` | `4096` | legacy/global fallback batch size | 只有仍使用 global batch 的路径受影响 |
| `PV_BATCH_SIZE` | `20480` | firm-level P/Q/BP parent batch size | 增大通常减少 batch 数、提高 GPU 吞吐，同时增加显存 |
| `SDF_FC1_BATCH_SIZE` | `4096` | FC1、Episode-0 SDF、SDF validation、post-refresh gate 的 batch size | 增大减少 batch 数，但 SDF child tensors 会提高显存压力 |
| `SIMULATE_GROUP_SIZE` | `200` | Episode > 0 每条 path 的 firm 数 | 增大 simulation 和 firm dataset 规模 |
| `SIMULATE_HORIZON` | `100` | SimulateTS 时间长度 | 增大可提供更长动态轨迹，但生成、存储和训练时间上升 |
| `POST0_MODE` | `modeb` | Episode > 0 的训练模式，可选 `modea/modeb/alternate` | `modeb` 使用当前 Mode B 顺序；`alternate` 才会读取 `ALTERNATE_START` |
| `ALTERNATE_START` | `modea` | alternate 时 Episode 1 的起始模式 | `POST0_MODE!=alternate` 时无效 |
| `ENABLE_FC2` | `0` | 是否把 FC2 加入每个 episode 的训练模块 | 设为 1 会增加 FC2 stage 和运行时间 |
| `QUICK_TEST` | `0` | Python quick-test flag | 在本 wrapper 中主要规模值均被显式参数覆盖，不能依赖它完成 smoke |

## 7. Episode 0 SDF bootstrap 参数

| 参数 | 默认值 | 作用 | 设置影响 |
|---|---:|---|---|
| `SDF_WEALTH_LOSS_MODE` | `signed_aio` | wealth Euler residual 的聚合目标 | 正式运行应保持 `signed_aio`；`legacy_abs_log1p` 是显式 ablation |
| `SDF_FRESH_PAIR_ENABLED` | `1` | 使用可刷新 fresh aggregate shock pairs | 关闭后回到固定 child pair，改变 signed-AiO sampling 口径 |
| `SDF_CHILD_BANK_SIZE` | `16` | 每个 parent 的 fresh shock bank 大小 | 增大多样性与构造成本 |
| `SDF_CHILD_BANK_REFRESH_EPOCHS` | `1` | 每多少 epoch 刷新 child bank | `1` 表示每 epoch 刷新；更大值复用更久 |
| `EPISODE0_SDF_EPOCHS_PER_ROUND` | `20` | Episode 0 每个 bootstrap gate round 的训练 epoch | 增大每次 gate 前的更新幅度和时间 |
| `EPISODE0_SDF_MAX_ROUNDS` | `20` | bootstrap 最大 round 数 | 最坏名义预算为 20 x 20 epochs；可提前通过 gate |
| `EPISODE0_SDF_LOG_MEAN_ERROR_MAX` | `0.25` | `abs(log E[M] - target)` 上限 | 调小更严格，可能需要更多 rounds 或失败 |
| `EPISODE0_SDF_CLIP_LOW_RATIO_MAX` | `0.20` | raw M 低于 0.7 的最大比例 | 调小更严格限制低尾部 |
| `EPISODE0_SDF_FINITE_RATIO_MIN` | `1.0` | M finite ratio 下限 | `1.0` 要求全部 finite |

Episode 0 的 gate 与 post-bootstrap formal `SDF_TRUE_ONLY` gate 不是同一套阈值。

## 8. FC1 / SDF_TRUE 调度参数

| 参数 | 默认值 | 作用 | 设置影响 |
|---|---:|---|---|
| `SDF_TRUE_START_EPISODE` | `2` | 第一个运行 formal FC1/SDF_TRUE 的 post-bootstrap episode | 默认 Episode 1 只形成首批 simulation-derived calculated macro data，Episode 2 才开始 formal stage |
| `SDF_TRUE_ONLY_EPOCHS` | `100` | 每次 formal SDF_TRUE 的名义 primal epoch 预算 | AL 为保证最后一次 dual update 后还有 primal block，实际最多可进入额外 terminal primal tail |
| `SDF_RECURSIVE_ONLY_EPOCHS` | `0` | recursive SDF-only stage epoch | `0` 表示关闭；不要与 FC1 rollout diagnostic 混淆 |
| `SDF_STAGE2_LR` | `2e-4` | Stage2 FC1/SDF 基础学习率 override | 调大更新更快但更易破坏 M/FC1 稳定性 |
| `SDF_TRUE_ONLY_LR` | `4e-5` | SDF_TRUE_ONLY 的 SDF/value LR | 调小更稳但 progress 更慢 |
| `SDF_TRUE_TARGET_BATCHES` | `20` | SDF_TRUE 希望切成的 parent-group batch 数 | 实际 batch 数受 parent 总数和最小 batch 限制 |
| `SDF_MIN_PARENT_GROUPS_PER_BATCH` | `256` | dynamic batching 的最小 parent groups | 更大时 batch 更少、更稳定，但显存更高 |
| `SDF_TRUE_MOMENT_WEIGHT` | `5e-4` | legacy fixed moment penalty 权重 | 当前 AL 模式下该 fixed penalty 被 PHR constraint 取代，不是主要控制量 |
| `SDF_TRUE_ANCHOR_WEIGHT` | `0.05` | legacy log-mean anchor 权重 | 当前 AL 模式下 fixed anchor 被 pooled PHR inequalities 取代 |
| `SDF_RESET_OPTIMIZER_ON_TRUE_START` | `0` | SDF_TRUE 开始时是否清除 SDF/value optimizer moments | `1` 会丢弃已有 Adam 动量；默认延续 |
| `SDF_RESTORE_BEST_CHECKPOINT` | `1` | validated stage 结束后恢复 accepted/best checkpoint | 关闭后可能保留最后尝试状态，不建议正式实验关闭 |
| `SDF_COLLAPSE_LOG_MEAN_ERROR` | `0.5` | collapse detector 的 log-mean error 阈值 | 调小会更早判定 collapse |
| `SDF_COLLAPSE_MEAN_RATIO` | `0.10` | E[M] 相对目标的最低比例 | 调大更严格，可能拒绝尚在恢复中的状态 |
| `FC1_JACOBIAN_PENALTY_INTERVAL` | `10` | Jacobian penalty 的节流间隔 | 当前 `fc1_jacobian_penalty_weight=0`，因此默认没有实际 penalty；只有权重非零时才影响速度和 loss |

Dynamic SDF_TRUE batching 的实际规则是：

```text
actual_batches = min(
    target_batches,
    floor(n_parents / min_parent_groups_per_batch),
    n_parents,
)
```

因此 `SDF_TRUE_TARGET_BATCHES=20` 是目标，不保证日志中一定出现 20 batches。

## 9. SDF Augmented Lagrangian 与 gate 参数

当前 Slurm 没有显式传 `--sdf-moment-constraint-mode`，但当前 `HyperParams` 和 `build_hyperparams()` 的有效默认均为：

```text
sdf_moment_constraint_mode = augmented_lagrangian
```

### 9.1 Slurm 已暴露参数

| 参数 | 默认值 | 作用 | 设置影响 |
|---|---:|---|---|
| `SDF_AL_RHO` | `2.0` | PHR quadratic penalty 参数 | 增大更强惩罚 constraint violation，但可能让 primal optimization 更硬 |
| `SDF_AL_PRIMAL_EPOCHS_PER_DUAL_UPDATE` | `5` | 每累计多少 accepted primal epochs 更新一次 multipliers | 增大可减慢 dual 变化；减小会更频繁改变 AL objective |
| `SDF_CONTINUE_CONSTRAINT_TOL` | `0.02` | 允许 downstream P/Q/BP 继续的最大 moment violation | 调小会更容易阻断 downstream；它不是 final convergence tolerance |
| `SDF_AIO_PROGRESS_RATIO_MAX` | `0.80` | `abs(AiO_after)/(abs(AiO_before)+eps)` 的 progress 上限 | `0.80` 要求至少约 20% 相对改善；调大更宽松 |
| `SDF_AIO_GOOD_ABS_TOL` | `1e-3` | AiO mean 已足够小时的 progress short-circuit | 即使相对 ratio 不好，只要绝对值小于阈值也可认定有 progress |
| `SDF_FINAL_CONSTRAINT_TOL` | `1e-3` | final SDF convergence 的 moment violation 上限 | 比 continuation 的 `0.02` 严格 |
| `SDF_FINAL_MAX_SIGNED_T_ABS` | `2.0` | final convergence 的 `abs(signed_aio_t)` 上限 | 只决定最终收敛，不再单独作为每个 episode 的 hard kill switch |
| `SDF_FINAL_AIO_MEAN_TOL` | `1e-3` | final convergence 的 `abs(normalized signed-AiO mean)` 上限 | 调小更严格 |

当前三层语义为：

```text
safe_to_continue:
  numerical safety passes
  AND constraints are finite
  AND max constraint violation <= 0.02

sdf_stage_progress:
  abs(AiO_after) <= 1e-3
  OR AiO progress ratio <= 0.80

sdf_converged:
  numerical safety passes
  AND max constraint violation <= 1e-3
  AND abs(signed_aio_t) <= 2
  AND abs(normalized signed-AiO mean) <= 1e-3
```

所以 `abs(t)>2` 只会使 `sdf_converged=False`；只要 continuation safety 和 stage progress 成立，downstream P/Q/BP 仍可运行。

### 9.2 当前有效但未暴露为 Slurm 环境变量的 AL 参数

以下参数由 Python 默认提供，当前基础 Slurm 没有读取对应环境变量。仅在 `sbatch --export` 中设置同名大写变量不会生效：

| Python 参数 | 当前值 | 含义 |
|---|---:|---|
| `sdf_moment_constraint_mode` | `augmented_lagrangian` | AL/legacy 模式选择 |
| `sdf_al_gate_tolerance` | `0.0` | strict train constraint pass tolerance |
| `sdf_al_dual_max_batches` | `0` | dual estimator 使用完整 training split |
| `sdf_al_reset_on_true_start` | `True` | formal SDF_TRUE 开始时重置 multipliers |
| `sdf_al_strict_semantics_guard` | `True` | 要求 signed AiO + normalized residual 的正式 AL 语义 |
| `sdf_al_noise_tax_warn_ratio` | `1.0` | 只发 diagnostic warning，不改 rho 或 stopping rule |

如需改变这些值，应先把对应 CLI 接入基础 Slurm，而不是假设环境变量会自动传入。

## 10. P/Q value 与 BP grid supervision 参数

| 参数 | 有效默认值 | 是否可覆盖 | 作用与设置结果 |
|---|---:|---|---|
| `PV_TRAINING_FLOW` | `staged` | 是 | 先训练 P/Q value，再固定 teacher cache 训练 BP；改为 `joint` 会与当前 mixture/target 配置冲突 |
| `PV_EVAL_EPOCHS` | `100` | 是 | staged P/Q evaluation/value stage 的 epoch 预算；这是日志中 `PV P/Q eval x/100` 较长的原因 |
| `BP_DISTILL_EPOCHS` | `20` | 是 | 无 step budget 时的 BP epoch 预算；有 500-step budget 时内部 epoch limit 会扩展以允许达到 step budget |
| `BP_DISTILL_PATIENCE` | `1000000` | **否，wrapper 强制** | 基本禁用普通 early-stop patience，避免在 500 successful steps 前因 patience 停止 |
| `BP_DISTILL_MIN_DELTA` | `1e-4` | 是 | natural-validation score 至少改善多少才更新 best checkpoint |
| `BP_DISTILL_MAX_OPTIMIZER_STEPS` | `500` | 是 | successful BP optimizer updates 上限；非有限/硬梯度 skip 不计入 successful steps |
| `BP_DISTILL_TRAINABLE_SCOPE` | `heads_only` | **否，wrapper 强制** | 只训练 BP heads；shared trunk、P0/PI/Q/value heads 保持冻结 |
| `BP_GRID_POLICY_LOSS_SPACE` | `logit` | 是 | `logit` 在概率饱和区保留更有效梯度；`output` 是 probability-space ablation |
| `BP_GRID_LOGIT_TARGET_EPS` | `1e-4` | 是 | teacher probability 在做 logit 前 clip 到 `[eps,1-eps]`；eps 更小允许更极端 target logit，也会放大边界 target magnitude |
| `BP_GRID_LOGIT_HUBER_DELTA` | `1.0` | 是 | logit Huber loss 的 transition point；更小更早进入线性区、更抗 outlier，更大更接近平方误差 |
| `PV_ROLLBACK_ON_SOFT_SPIKES` | `0` | 是 | soft gradient spike 是否直接触发 rollback；默认只记录/clip，hard/nonfinite 仍按安全控制处理 |
| `FIRM_TARGET_UPDATE` | `stage_hard` | 是 | staged flow 中 target 在 stage 内冻结，成功后 hard sync；`staged` 只允许 `stage_hard` 或 `none` |

`BP_DISTILL_MAX_OPTIMIZER_STEPS=500` 不表示最终使用第 500 步权重。代码会持续保存 natural-validation best checkpoint，并在 stage 结束时恢复该 checkpoint。日志中应同时检查：

```text
accepted_optimizer_steps_total
best_checkpoint_optimizer_steps
optimizer_steps
max_optimizer_steps_reached
stop_reason
```

当前正式 BP summary 会报告 train cache 重采样前后及 natural validation 的 current-eta count/share；当前版本尚未额外输出 `bp0_eta0_mae`、`bp0_eta1_mae` 等 conditional MAE。

## 11. Current-eta 与 future-eta 参数

| 参数 | 当前值 | 是否可覆盖 | 作用 |
|---|---:|---|---|
| `BP_CURRENT_ETA_RESAMPLE_ENABLED` | `1` | **否，wrapper 强制** | 开启 current parent eta 的 BP train-cache-only resampling |
| `BP_CURRENT_ETA1_TRAIN_SHARE` | `0.25` | **否，wrapper 强制** | 目标训练 share；由于整数 round，实际 share 可能有极小离散误差 |
| `BP_CURRENT_ETA_RESAMPLE_SEED` | `13579` | **否，wrapper 强制** | 独立 CPU RNG，不改变 global Torch RNG state |
| `PV_ETA_RESAMPLE_ENABLED` | `0` | **否，wrapper 强制** | 关闭旧的 future-child-eta cache resampler，避免与 current-eta sampler 混用 |
| `PV_EXACT_ETA_INTEGRATION_ENABLED` | `1` | **否，wrapper 强制** | Bellman/teacher 对 future `eta_{t+1}` 精确求和 |

`BP_CURRENT_ETA1_TRAIN_SHARE` 只影响：

```text
BP distillation train cache
```

它不影响：

```text
validation cache
P0/PI/Q Bellman batches
BPGridTeacher economic objective
simulation eta process
Config.ZETA
future eta exact integration
```

## 12. PV mixture 参数

| 参数 | 默认值 | 是否可覆盖 | 作用与设置影响 |
|---|---:|---|---|
| `PV_MIXTURE_ENABLED` | `1` | **否，wrapper 强制** | Episode > 0 构造 mixed P/Q/BP parent distribution |
| `PV_MIXTURE_RATIO` | `0.20` | 是 | 固定总 parent budget 中 coverage Sample 的目标比例；其余来自 SimulateTS |
| `PV_MIXTURE_START_EPISODE` | `1` | 是 | 从哪个 episode 开始 mixture；Episode 0 保持原逻辑 |
| `PV_MIXTURE_BUDGET_MODE` | `fixed_total` | 是，但当前只支持该值 | mixture 不增加总 parent budget，只重新分配来源 |
| `PV_MIXTURE_SAMPLING_MODE` | `uniform` | 是，可选 `uniform/feasible/realbz` | 控制 coverage pool 的采样策略 |
| `PV_MIXTURE_COVERAGE_GROUP_SIZE` | `2` | 是 | coverage Sample 的 parent grouping 规模；必须为正 |
| `PV_MIXTURE_SEED` | `24680` | 是 | mixture selection 的 deterministic seed |
| `PV_MIXTURE_STRATIFIED_VALIDATION` | `1` | 是 | validation split 保留不同 source 的覆盖 |
| `PV_MIXTURE_PRESERVE_RNG` | `1` | 是 | 构造 coverage pool 后恢复全局 RNG，减少实验间非目标差异 |

约束：mixture 要求 `PV_TRAINING_FLOW=staged` 且 `PV_ETA_RESAMPLE_ENABLED=0`。

## 13. Value scaling 参数

| 参数 | 默认值 | 作用 | 设置影响 |
|---|---:|---|---|
| `PV_VALUE_SCALE_MODE` | `exp_xz` | value scale 使用 `exp(x+z)` | 改为 `none` 就不再是 scaled-value matched experiment |
| `PV_VALUE_SCALE_LOG_MAX` | `20.0` | 对用于指数尺度的 log scale 做上界保护 | 调小会更强地限制极端 scale；调大可能增加数值范围 |
| `PV_BELLMAN_NORMALIZE_BY_VALUE_SCALE` | `1` | Bellman residual 除以 value scale | 关闭会改变 P0/PI loss 的数值尺度 |

当前 `training/episode.py` 中 PI loss 使用 `PILoss(b_penalty_weight=0.0)`，P0 没有对应 high-b penalty；因此这次 full training 的 P0/PI 都没有额外 high-b penalty。

## 14. Mode B / Treatment B 参数

| 参数 | 默认值 | 作用 | 设置结果 |
|---|---:|---|---|
| `POST0_MODE` | `modeb` | Episode > 0 使用 Mode B | 设为 `modea` 会改变 episode 训练顺序和数据语义 |
| `MODEB_RESIMULATE_AFTER_PV` | `1` | PV 更新后重新 simulation | `1` 是 Treatment B；`0` 是不重模拟的 Treatment A |

当前 Treatment B 顺序的关键点是：PV 更新后的 policy/value 会生成 refreshed Mode B data，后续 gate 与 FC1/SDF 使用刷新后的数据。

## 15. 输出位置与检查重点

`RUN_ROOT` 下通常包括：

```text
RUN_ROOT/
  checkpoints/
  data/outputs/
  experiments/figs/
  failure_report.json          # 仅 NumericalStageFailure 时
  gpu_memory_monitoring.json
```

具体 episode stage dataframe 和 checkpoint 名称由 `run_multi_episode_job.py` 的保存 helper 决定。运行后至少检查：

1. Slurm stdout 开头的 branch、actual commit、required commit 和 CUDA 型号；
2. `Run parameters` 是否与实验设计一致；
3. BP stage 的 current-eta before/after share 和 validation natural share；
4. BP `accepted_optimizer_steps_total` 与 `best_checkpoint_optimizer_steps`；
5. SDF 的 `safe_to_continue`、`sdf_stage_progress`、`sdf_converged`；
6. AL dual update 频率、terminal primal tail 和 final constraint violation；
7. 是否生成 `failure_report.json`。

## 16. 常见调整方案

### 16.1 只减少 P/Q 训练时间

```bash
--export=ALL,REQUIRED_COMMIT="$HEAD_SHA",PV_EVAL_EPOCHS=20
```

结果：减少 `PV P/Q eval x/100` 的 epoch 数，不改变 BP 500-step budget。

### 16.2 保持正式 BP 500 steps，但减少 outer episodes

```bash
--export=ALL,REQUIRED_COMMIT="$HEAD_SHA",N_EPISODES=3
```

结果：只运行 Episode 0、1、2；每个实际进入 BP distillation 的 stage 仍最多 500 successful updates。

### 16.3 降低显存

```bash
--export=ALL,REQUIRED_COMMIT="$HEAD_SHA",PV_BATCH_SIZE=4096,SDF_FC1_BATCH_SIZE=1024
```

结果：单 batch 显存降低，但 batch 数和 wall time 通常上升。不要只改 `BATCH_SIZE`，因为 PV/SDF 已有独立 batch size。

### 16.4 增加 simulation 覆盖

```bash
--export=ALL,REQUIRED_COMMIT="$HEAD_SHA",POST0_N_PATHS=800,SIMULATE_HORIZON=150
```

结果：post-0 数据覆盖增加，同时 simulation、pickle、batch construction 和训练成本显著增加。

### 16.5 改变 current-eta share

当前专用 wrapper 使用：

```bash
export BP_CURRENT_ETA1_TRAIN_SHARE=0.25
```

这是强制赋值，外部 `--export=...,BP_CURRENT_ETA1_TRAIN_SHARE=...` 会被覆盖。若要跑其他 share，应新增明确命名的实验 wrapper，而不是临时假装覆盖成功。

## 17. 启动前检查清单

```bash
cd /home/fit/zhuyingz/WORK/LiuHao/DL_AP_Tensor

git branch --show-current
git rev-parse HEAD
git status --short
bash -n slurm/run_normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500.slurm
python3 experiments/run_multi_episode_job.py --help >/dev/null
```

应满足：

- branch 为 `exp/equity-value-xz-scaling`；
- `git status --short` 无输出；
- `REQUIRED_COMMIT` 使用服务器上的 `git rev-parse HEAD`；
- `DL_HL` 中 `torch.cuda.is_available()` 为 `True`；
- `$WORK_ROOT/logs` 已存在或可创建；
- `RUN_ROOT` 不与需要保留的旧实验冲突。

## 18. 当前默认参数总表

以下是专用 wrapper 链最终产生的主要有效值：

```text
SEED=12345
N_EPISODES=10
EPOCHS=100
N_PATHS=400
POST0_N_PATHS=400
BATCH_SIZE=4096
PV_BATCH_SIZE=20480
SDF_FC1_BATCH_SIZE=4096
SIMULATE_GROUP_SIZE=200
SIMULATE_HORIZON=100
POST0_MODE=modeb
ALTERNATE_START=modea
ENABLE_FC2=0
QUICK_TEST=0

SDF_WEALTH_LOSS_MODE=signed_aio
SDF_FRESH_PAIR_ENABLED=1
SDF_CHILD_BANK_SIZE=16
SDF_CHILD_BANK_REFRESH_EPOCHS=1
SDF_CHILD_BANK_SEED=12345
EPISODE0_SDF_EPOCHS_PER_ROUND=20
EPISODE0_SDF_MAX_ROUNDS=20
EPISODE0_SDF_LOG_MEAN_ERROR_MAX=0.25
EPISODE0_SDF_CLIP_LOW_RATIO_MAX=0.20
EPISODE0_SDF_FINITE_RATIO_MIN=1.0
SDF_TRUE_ONLY_EPOCHS=100
SDF_TRUE_START_EPISODE=2
SDF_RECURSIVE_ONLY_EPOCHS=0
SDF_STAGE2_LR=2e-4
SDF_TRUE_ONLY_LR=4e-5
SDF_TRUE_TARGET_BATCHES=20
SDF_MIN_PARENT_GROUPS_PER_BATCH=256
SDF_TRUE_MOMENT_WEIGHT=5e-4
SDF_TRUE_ANCHOR_WEIGHT=0.05
SDF_AL_RHO=2.0
SDF_AL_PRIMAL_EPOCHS_PER_DUAL_UPDATE=5
SDF_CONTINUE_CONSTRAINT_TOL=0.02
SDF_AIO_PROGRESS_RATIO_MAX=0.80
SDF_AIO_GOOD_ABS_TOL=1e-3
SDF_FINAL_CONSTRAINT_TOL=1e-3
SDF_FINAL_MAX_SIGNED_T_ABS=2.0
SDF_FINAL_AIO_MEAN_TOL=1e-3
SDF_RESET_OPTIMIZER_ON_TRUE_START=0
SDF_RESTORE_BEST_CHECKPOINT=1
SDF_COLLAPSE_LOG_MEAN_ERROR=0.5
SDF_COLLAPSE_MEAN_RATIO=0.10
FC1_JACOBIAN_PENALTY_INTERVAL=10

BP_GRID_POLICY_LOSS_SPACE=logit
BP_GRID_LOGIT_TARGET_EPS=1e-4
BP_GRID_LOGIT_HUBER_DELTA=1.0
PV_TRAINING_FLOW=staged
PV_EVAL_EPOCHS=100
BP_DISTILL_EPOCHS=20
BP_DISTILL_PATIENCE=1000000
BP_DISTILL_MIN_DELTA=1e-4
BP_DISTILL_MAX_OPTIMIZER_STEPS=500
BP_CURRENT_ETA_RESAMPLE_ENABLED=1
BP_CURRENT_ETA1_TRAIN_SHARE=0.25
BP_CURRENT_ETA_RESAMPLE_SEED=13579
BP_DISTILL_TRAINABLE_SCOPE=heads_only
PV_ROLLBACK_ON_SOFT_SPIKES=0

PV_MIXTURE_ENABLED=1
PV_MIXTURE_RATIO=0.20
PV_MIXTURE_START_EPISODE=1
PV_MIXTURE_BUDGET_MODE=fixed_total
PV_MIXTURE_SAMPLING_MODE=uniform
PV_MIXTURE_COVERAGE_GROUP_SIZE=2
PV_MIXTURE_SEED=24680
PV_MIXTURE_STRATIFIED_VALIDATION=1
PV_MIXTURE_PRESERVE_RNG=1
PV_ETA_RESAMPLE_ENABLED=0
PV_EXACT_ETA_INTEGRATION_ENABLED=1

FIRM_TARGET_UPDATE=stage_hard
MODEB_RESIMULATE_AFTER_PV=1
PV_VALUE_SCALE_MODE=exp_xz
PV_VALUE_SCALE_LOG_MAX=20.0
PV_BELLMAN_NORMALIZE_BY_VALUE_SCALE=1
```

这份总表用于核对日志，不替代上面关于强制值、可覆盖值和未暴露 Python defaults 的说明。

## 19. Shell 内部变量（不是用户训练参数）

三层脚本还使用以下内部变量。它们用于拼接路径、commit guard 和命令参数，不应通过 `sbatch --export` 调参：

| 内部变量 | 来源与用途 |
|---|---|
| `STAMP` | 启动作业时的时间戳，用于默认 `RUN_ROOT` |
| `EXPERIMENT_SCRIPT` | 专用 wrapper 的仓库相对路径 |
| `DEFAULT_REQUIRED_COMMIT` | `git log` 解析出的 Slurm last-touch commit |
| `ACTUAL_COMMIT` | 运行节点上 `git rev-parse HEAD` 的结果 |
| `EXTRA_ARGS` | 基础脚本根据 0/1 环境变量构造的 Boolean CLI 参数数组 |

这些内部变量和 `#SBATCH` directives 不属于模型超参数。用户实际应通过前文列出的环境变量和 `sbatch` options 控制作业。
