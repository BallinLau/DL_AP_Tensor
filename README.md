# DL-APSSH

深度学习资产定价实验仓库，包含 SDF、FC1、Policy/Value、FC2 四类核心模块，以及 episode 训练流程和模拟数据生成流程。

## 项目结构

```text
DL-APSSH/
├── config/                 # 经济参数与训练超参数
├── data/                   # Sample / SimulateTS 数据生成与处理
├── models/                 # SDF/FC1、Policy/Value、FC2 网络
├── losses/                 # SDF、P0、PI、Q、FC2 损失
├── training/               # Episode 与 Trainer
├── utils/                  # 日志、checkpoint、指标与可视化
├── experiments/            # 训练/调试脚本
├── tests/                  # 测试
├── main.py                 # 命令行入口
└── requirements.txt
```

## 安装

```bash
cd /Users/ballinliu/Desktop/PHD/Project1/DL-APSSH
pip install -r requirements.txt
```

## 快速开始

```bash
# 联合训练
python main.py --mode train --train_mode joint --n_episodes 10

# 分阶段训练
python main.py --mode train --train_mode staged --n_episodes 10

# 交替训练
python main.py --mode train --train_mode alternating --n_episodes 10

# 评估（读取 checkpoints/best.pt）
python main.py --mode eval --resume best

# 模拟（输出到 checkpoints/simulation/）
python main.py --mode simulate --resume best
```

## CLI 参数（`main.py`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--mode` | `train` | 运行模式：`train` / `eval` / `simulate` |
| `--train_mode` | `joint` | 训练策略：`joint` / `staged` / `alternating` |
| `--n_episodes` | `10` | 训练 episode 数 |
| `--epochs_per_episode` | `10` | 每个 episode 的 epoch 数 |
| `--batch_size` | `256` | 训练批大小 |
| `--n_samples` | `10000` | 每个 episode 的样本数 |
| `--lr` | `1e-3` | 学习率 |
| `--weight_decay` | `1e-4` | 权重衰减 |
| `--hidden_dim` | `128` | 隐层宽度参数 |
| `--n_layers` | `4` | 网络层数参数 |
| `--save_dir` | `./checkpoints` | checkpoint 输出目录 |
| `--log_dir` | `./logs` | 日志目录 |
| `--resume` | `None` | checkpoint 名称或路径 |
| `--device` | `auto` | `auto/cpu/cuda/mps` |

## 核心状态与模型接口

### Firm-state 向量

`(b, z, ETA, i, x, Hatcf, LnKF)`，7 维，索引定义见 `config/constants.py` 中 `SIMMODEL`。

### 公开类（当前代码）

- `models.SDFFC1Combined(sdf_input_dim=4, fc1_input_dim=4, ...)`
- `models.PolicyValueModel(base_state_dim=6, share_hidden_dims=None, share_output_dim=64, dropout=0.0)`
- `models.FC2Model(input_dim=None, hidden_dims=None, output_dim=2, quantile_num=100, dropout=0.1)`
- `training.Trainer(models, config=Config, hyperparams=HyperParams(), save_dir='./checkpoints', log_dir='./logs', device=None)`

## 主要经济参数（`config/constants.py`）

| 参数 | 当前值 |
|---|---|
| `RHO_X` | `0.95` |
| `SIGMA_X` | `0.012` |
| `RHO_Z` | `0.90` |
| `SIGMA_Z` | `0.36` |
| `DELTA` | `0.02` |
| `TAU` | `0.2` |
| `BETA` | `0.942` |
| `GAMMA` | `4.0` |
| `ZETA` | `0.03` |
| `FC2_INPUT_DIM` | `201`（100 个 b 分位点 + 100 个 z 分位点 + x） |

## 训练与输出

- 训练流程由 `training/Episode` 与 `training/Trainer` 协同完成。
- 训练/评估模型权重默认保存到 `checkpoints/`。
- 模拟结果默认保存到 `checkpoints/simulation/firm_panel.csv` 与 `checkpoints/simulation/macro_panel.csv`。

## 子模块文档

### Finite Direct Outer 4–11 continuation（审查阶段）

Job 701387 的完整 Outer 3 状态续训入口为
`analysis/run_finite_direct_large_continuation.py`，默认仅打印计划、不构建训练 backend。
CPU 准备入口为 `analysis/prepare_finite_direct_continuation.py`。
新逻辑规模为 8192/2048 Parent，Beta/Empirical 50:50，实际阶段编号 4–11。
跨 run 导入完整优化器/AL/RNG；`--resume` 仅用于新 run 自身的事务恢复。
本版本修正 Parent default 在 Child 政策前退出，算子版本为
`finite_direct_parent_exit_before_child_v2`。Train Empirical 逐 Outer 更新：
post-SDF Train-only pre-decision realized-b source → 原子 Pool/下一轮 Bank；
Recent/Archive 精确2867/1229，Validation 与 Beta 固定。Outer4 初始模板仍明确为历史t0。
旧 Job701387 只作为完整学习状态初始化，不是新算子的收敛证据。
必须在用户审查后单独批准 `CONTINUATION_APPROVAL_V4_DYNAMIC_EMPIRICAL`，才可使用
`--execute-approved`；旧v3批准、旧缓存和旧run journal不能用于本版本。
新 run 的批准原文留档于 `metadata/approval_v4.json`；最终部署前核对见
[最终训练准备报告](review/final_training_readiness/implementation_report.md)。
本分支不自动训练、提交 Slurm、批准或 push。
完整契约、真实 CLI、源状态审计和验收见
[退出修正与动态 Empirical 审查报告](review/dynamic_empirical_exit_fix_20261021/dynamic_empirical_implementation_report.md)。

- `config/README.md`
- `data/README.md`
- `models/README.md`
- `losses/README.md`
- `training/README.md`
- `utils/README.md`

## Independent H10 + H100 historical macro replay

The 2026-10-22 experiment uses a separate checkout and V6 approval; it never
switches or modifies the running H50 job. See [runtime, data contracts, smoke
and formal launch gates](review/h10_h100_longrun/README.md) and the
[verified implementation status](review/h10_h100_longrun/implementation_report.md).
Default CLI behavior is plan-only. A running bounded smoke is not formal training.
