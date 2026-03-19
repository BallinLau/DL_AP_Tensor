# Loss 与 Episode 详细差异报告（DL_AP_Local vs DL_AP_SSH）

生成时间：2026-03-11 17:33:19 CST

## 1. 对比范围
- A（本地）：`BallinLau/DL_AP_Local` @ `origin/main`（2d9009a6aaf95ca41bc24bf0910db09acd87fd19）
- B（服务器）：`BallinLau/DL_AP_SSH` @ `sshrepo/main`（061b281dc0f3dea7a12fdb208e8f78af7364b0ac）
- 方向说明：以下“变更后”均指 A（Local）相对 B（SSH）。
- 本报告仅覆盖：`training/episode.py` 与 `losses/*`。

## 2. 文件级差异统计
| 文件 | Local 新增行 | Local 删除行 | 备注 |
|---|---:|---:|---|
| training/episode.py | 119 | 135 | 训练流程、P0/PI loss 计算路径、默认 batch 配置调整 |
| losses/p0_loss.py | 53 | 1 | 新增基于 bp 的 FOC 自动求导接口 |
| losses/pi_loss.py | 66 | 0 | 新增 PI 的 FOC 自动求导接口 |
| losses/FC2losspipe.py | 7 | 8 | merge key 与 full_N/张量尺寸处理调整 |
| losses/README.md | 17 | 1 | 文档同步到新的 FOC 接线 |

## 3. training/episode.py 详细差异

### 3.1 日志与初始化
- 日志实例由项目级名称改为模块级名称：
  - 旧：`logger = logging.getLogger('DL-AP')`
  - 新：`logger = logging.getLogger(__name__)`
  - 新文件位置：`training/episode.py:33`
- 删除了 `self.train_mode` 的初始化字段（旧版在 `__init__` 中存在）。

### 3.2 generate_data 参数行为
- Sample 构造时：
  - 旧：`n_samples=None`
  - 新：`n_samples=n_samples`
- 影响：Local 版本按调用参数控制样本量，SSH 版本在此处固定忽略传入值。

### 3.3 SDF 重建监督索引与调试分支
- 在 `_compute_sdf_loss` 中，FC1 重建监督列索引发生变化：
  - 旧：`hatcf_true=children_t[:,:,7:8]`，`lnkf_true=children_t[:,:,8:9]`
  - 新：`hatcf_true=children_t[:,:,8:9]`，`lnkf_true=children_t[:,:,9:10]`
  - 新文件位置：`training/episode.py:422-423`
- 旧版包含 NaN/Inf 详细告警日志分支，新版删除该分支。

### 3.4 P0 loss 计算链路（核心）
新版本在 `_compute_p0_loss` 中引入了 FOC 路径并改了 b' 取值来源：
- b' 使用：
  - 旧：child 使用 `eta*bp_t + (1-eta)*b_parent`，`childp0_state` 用 `bpI_t`
  - 新：显式使用 `bp_for_p0 = bp0_t`；child 与 childp0 都基于 `bp0`
  - 新文件位置：`training/episode.py:472-487`
- 损失项：
  - 旧：`total_loss = main_loss + penalty_z`
  - 新：新增 FOC 主损失和 FOC 的 z penalty：
    - `loss_foc`
    - `penalty_z_foc`
    - `total_loss = main_loss + penalty_z + loss_foc + penalty_z_foc`
  - 新文件位置：`training/episode.py:519-536`

### 3.5 PI loss 计算链路（核心）
新版本在 `_compute_pi_loss` 中接入 FOC 路径，并将 child 的 b 设置改为固定 `bpI`：
- b' 使用：
  - 旧：`eta*bp_t + (1-eta)*b_parent`
  - 新：`bp_for_pi = bpI_t`，child 直接 `child_state[:,0:1] = bp_for_pi`
  - 新文件位置：`training/episode.py:583-599`
- 损失项：
  - 旧：`total_loss = main_loss + penalty_z + penalty_b`
  - 新：增加 `loss_foc + penalty_z_foc`
  - 新文件位置：`training/episode.py:637-654`

### 3.6 默认 batch_size 调整
以下函数默认 `batch_size` 从 1024 调整到 256：
- `create_batches`（`training/episode.py:800`）
- `_create_sdf_batches_from_macro_df`（`training/episode.py:878`）
- `_create_firm_batches_from_df`（`training/episode.py:940`）
- `_create_fc2_batches`（函数签名已同步）
- `run_episode`（`training/episode.py:1099`）
- `run`（底部签名）

### 3.7 _create_sdf_batches_from_macro_df 逻辑简化
- 旧版：
  - 根据 `self.add_FC1loss` 和 `self.train_mode` 在 `['path']` vs `['path','t']` 间切换分组；
  - 含 `t > 2` 过滤与额外日志。
- 新版：
  - 统一按 `df_sdf.groupby('path')` 分组；
  - 删除 `train_mode` 相关过滤与日志分支。

### 3.8 run_episode 流程重构
- 函数签名删除 `train_mode` 参数。
- 旧版是 `train_mode == '2time'` 的双分支驱动；新版改为按 `episode_id == 0` 与 `else` 分流。
- FC2 训练循环：
  - 旧版多处被注释；
  - 新版在 episode=0 和 episode>=1 分支中均启用训练循环与日志统计。
- 数据流顺序也有所调整（如 episode>=1 分支中模拟与训练调用顺序）。

## 4. losses 详细差异

### 4.1 losses/p0_loss.py
- 新增函数：`compute_foc_residual_from_bp(...)`
  - 位置：`losses/p0_loss.py:172-221`
  - 逻辑：用 autograd 计算
    - `dCF0p/dbp`
    - 各路径 `dP_child/dbp`
  - 再复用 `compute_foc_residual(...)` 生成多分支 FOC 残差。
- 兼容性增强：`allow_unused=True` 下对 `None` 梯度做零张量回退。

### 4.2 losses/pi_loss.py
- 新增函数：
  - `compute_foc_residual(...)`（PI 版，含 `self.g`）
  - `compute_foc_residual_from_bp(...)`
  - 位置：`losses/pi_loss.py:150-214`
- 逻辑与 p0 对称，但公式为：
  - `foc_j = dCFip/dbp + g * M_j * dP_child_j/dbp * (1-bar_z_j) * eta`

### 4.3 losses/FC2losspipe.py
- parent/child 合并键：
  - 旧：`on=['path','ID']`
  - 新：`on=['ID']`
  - 新文件位置：`losses/FC2losspipe.py:51-52`
- `self.full_N` 处理：
  - 旧版在 `_build_tensors` 中覆写 `self.full_N=self.N`
  - 新版删除该覆写，保留传入 `full_N`，并按 `self.full_N` 做 padding 尺寸。

### 4.4 losses/README.md
- 文档补充了 P0/PI 的 FOC 接口与 episode 聚合方式，和源码接线保持一致。

## 5. 核心行为影响（重点）
1. Local 版本的 P0/PI 训练目标比 SSH 版本更强：不再只优化 Bellman，还加入了 FOC 约束。
2. PI 分支 child b 的构造在 Local 变为固定 `bpI`，这会显著改变 PI 训练样本上的状态转移机制。
3. FC2 pipeline 的 merge key 从 `path+ID` 改为 `ID`，如果不同 path 存在相同 ID，可能引入跨 path 连接风险。
4. batch 默认从 1024 降到 256，会改变显存占用、梯度噪声与训练速度。

## 6. 复现实验命令
- 单文件差异：
  - `git diff sshrepo/main origin/main -- training/episode.py`
  - `git diff sshrepo/main origin/main -- losses/p0_loss.py losses/pi_loss.py`
- 行统计：
  - `git diff --numstat sshrepo/main origin/main -- training/episode.py losses/*`
