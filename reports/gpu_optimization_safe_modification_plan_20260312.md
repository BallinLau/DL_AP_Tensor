# GPU 优化安全改造文档（不破坏现有经济学训练闭环）

**日期**: 2026-03-12  
**适用工程**: `/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local`  
**目标**: 在 H100 80GB 上提速，同时保持你当前要求的功能语义不变（尤其是 SDF 两阶段与 P/Q 联立贝尔曼训练链路）。

---

## 1. 审查结论（先给结论）

原 `GPU_OPTIMIZATION_REPORT.md` 的方向总体正确，但有 4 个高风险点需要改写后再实施：

1. FC2 预计算方案若直接改成“纯 tensor batch”，会与当前 `_compute_fc2_loss` 的输入契约冲突。  
代码证据：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/training/episode.py:1009`。

2. FC2 优化示例里没有保留你当前必须的 `FC2 -> SDF( add_FC1loss=True )` 回灌步骤，可能破坏两阶段 SDF 训练。  
代码证据：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/training/episode.py:1821`。

3. 将 `Sample/SimulateTS` 直接改为 tensor 输出，会破坏大量 DataFrame 下游（`build_sdf_pairs_from_macro_ts`、notebook 可视化、stage 存盘）。  
代码证据：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/data/data_utils.py:270`。

4. SDF 全链路直接开 AMP 风险高（exp/pow/负指数敏感），会放大 NaN/Inf 风险；应分模块启用。

---

## 2. 不变约束（必须保留）

以下是优化后必须保持的行为约束：

1. `Episode` 阶段流程仍是 `sdf1 -> pv -> sdf2 -> fc2`（以 `run_multi_episode_job.py` 的 stage_order 为准）。
2. `fc2` 阶段训练后，必须继续执行：
   - `build_sdf_pairs_from_macro_ts(..., include_hatc_lnk_t1=True)`
   - `self.add_FC1loss = True`
   - 再训练一次 `sdf_fc1`
3. `SimulateTS.simulate()` 返回结构维持 `(df, df_macro)` 的 DataFrame 合约。
4. 现有 notebook 所依赖字段与图（P/Q、delta hatc/lnk、宏观时序）不改接口。

---

## 3. 第一批改动（建议立即实施，低风险高收益）

### 3.1 删除 FC2 阶段每 step 的 CSV 落盘

问题：`train_step` 在 episode>0 时每个 step 都 `to_csv`，这是明显 CPU+IO 瓶颈。  
代码位置：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/training/episode.py:401-405`。

改法：
- 新增调试开关（如 `fc2_dump_debug_csv=False`），默认关闭。
- 仅在显式调试时导出 CSV。

预期：FC2 每轮时间显著下降，且不影响任何损失定义。

### 3.2 FC2 Pipeline 缓存（保留 DataFrame 输入契约）

问题：每个 epoch 重建 `FC2LossPipe` 成本高。  
代码位置：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/training/episode.py:1035-1043`。

改法：
- 保持 `_compute_fc2_loss(batch)` 仍接收 DataFrame/dict(df)；
- 在 `Episode` 内部缓存“同一 df 语义下”的 pipe 或其静态中间量（ID 索引、mask、固定张量）；
- 每轮只更新真正随网络输出变化的部分。

预期：不破坏现有训练语义的前提下，FC2 加速最明显。

### 3.3 分模块 AMP（只开 PV/FC2，SDF 先保持 FP32）

改法：
- 增加细粒度开关：
  - `amp_policy_value=True`
  - `amp_fc2=True`
  - `amp_sdf=False`（默认）
- `train_step` 根据 `train_modules` 进入 autocast。

原因：SDF 残差链对数值范围更敏感，先避免把 NaN 风险引入最脆弱部分。

### 3.4 H100 打开 TF32

改法：
- 启动时设置：
  - `torch.backends.cuda.matmul.allow_tf32 = True`
  - `torch.backends.cudnn.allow_tf32 = True`

收益：对大矩阵计算吞吐提升明显，数值风险低于 AMP 全开。

---

## 4. 第二批改动（中风险，做前先跑回归）

### 4.1 提升 batch 与样本规模（分步而不是一步到位）

建议节奏：
1. `batch_size: 512 -> 1024`
2. `n_samples: 10000 -> 20000`
3. `n_paths: 1000 -> 1500`

说明：先观察 Bellman 残差与 NaN 率，再继续加。

### 4.2 `torch.compile` 仅用于 FC2 / PolicyValue 前向

说明：
- 可提升吞吐，但先限定模块；
- 避免编译阶段对 SDF 数值稳定性造成额外变量。

---

## 5. 暂不建议改动（当前会破坏功能或高不确定性）

1. 把 `Sample/SimulateTS` 主接口改成“只返回 tensor”。
2. 让 `_compute_fc2_loss` 只接收预计算 tensor dict（不再接受 DataFrame）。
3. 直接对 `sdf_fc1` 全链路启用 AMP。
4. 不经回归验证就把 `n_paths`、`group_size` 同时大幅拉满。

---

## 6. 回归验证清单（每一项都要过）

### 6.1 功能一致性

1. episode0 与 episode>0 都能完整跑完四阶段。
2. FC2 后仍触发 `add_FC1loss=True` 的 SDF 训练。
3. notebook 里的图全部能画出，字段不缺失。

### 6.2 数值稳定性

1. SDF/PV/FC2 loss 全程无 NaN/Inf。
2. 梯度告警（`NaN gradient detected`）显著减少或为 0。
3. Bellman 残差收敛判定满足：`mean < 0.001, p90 < 0.005`。

### 6.3 经济学形状约束

1. Q 对 b 呈倒 U 趋势（至少局部明显）。
2. 存在 `P=0` 区域（破产区域）。
3. `b=0` 附近 Q 接近 0（无发债时债券价值应接近 0）。

---

## 7. 推荐实施顺序（建议按周）

### 第 1 天

1. 关闭 FC2 每 step CSV 导出。  
2. 加 TF32 开关。  
3. 仅给 PV/FC2 开 AMP，SDF 保持 FP32。  
4. 跑 `episode0` 验证稳定性。

### 第 2 天

1. 做 FC2 pipeline 缓存（不改输入契约）。  
2. 跑 2-3 个 episode 做速度与数值对比。  
3. 保留 rollback 开关。

### 第 3 天

1. 分步拉大 batch / samples / paths。  
2. 每步都跑 Bellman 与形状图回归。

---

## 8. 需要同步的文档

实施上述代码改动后，建议同步更新：

1. `/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/reports/codex_all_modifications_summary_20260312.md`  
2. `/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Local/reports/GPU_OPTIMIZATION_REPORT.md`（改成“安全版路线图”）

