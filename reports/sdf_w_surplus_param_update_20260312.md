# SDF W 参数化更新报告（2026-03-12）

## 背景
用户提出：`w` 应与 `c` 保持结构关系，确保 `w - exp(c)` 始终为正，而不是仅依赖网络输出“正值但可能很小”。

## 本次改动

### 1) 新增配置项
- 文件：`config/constants.py`
- 新增：
  - `W_SURPLUS_FLOOR = 1e-3`
- 含义：`w` 中 surplus 的最小下界。

### 2) 改造 ValueFunctionW 的输出结构
- 文件：`models/sdf_fc1.py`
- 变更前：
  - `w = softplus(MLP(x))`
- 变更后：
  - `surplus = softplus(MLP(x)) + W_SURPLUS_FLOOR`
  - `w = exp(c) + surplus`，其中 `c = x[..., 1:2]`

由此保证：
- `w - exp(c) >= W_SURPLUS_FLOOR > 0`（结构上恒成立）
- 避免 `w - exp(c)` 过小导致 SDF 分母不稳定。

### 3) 文档同步
- 文件：`models/README.md`
- 增加了新参数化说明和稳定性动机。

## 校验
已执行：
- `python3 -m py_compile config/constants.py models/sdf_fc1.py`
- 结果：通过。

## 说明
该改动满足“`w` 相对 `c` 的增量”要求，且与经济含义一致：
- `exp(c)` 视作基准项；
- surplus 代表超额价值，受 softplus 与 floor 共同约束为正。
