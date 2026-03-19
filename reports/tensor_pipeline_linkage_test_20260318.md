# Tensor Pipeline 联通性测试（2026-03-18）

## 测试目标
验证 `DL_AP_Tensor` 中“tensor 数据生成 -> Episode 训练”链路是否正常。

## 测试环境
- 项目：`/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor`
- 设备：CPU
- 配置：`use_tensor_pipeline=True`
- 测试脚本：`/tmp/test_tensor_pipeline_linkage.py`（临时）

## 一、SDF + Policy/Value（不含 FC2）结果

已覆盖三种模式：`mode0 / modeA / modeB`，每种跑 1 epoch。

结果文件：
- `/Users/ballinliu/Desktop/PHD/Project1/cachedir/tensor_pipeline_linkage_test_20260318.json`

摘要：
- 三种模式均可运行完成；
- `has_nan_or_inf = False`（均为有限值）；
- `tensor_firm/tensor_macro` 均成功生成；
- 训练结束后 `df/df_macro` 也成功导出（兼容现有可视化脚本）。

## 二、FC2 链路（包含 FC2）结果

测试：`mode0` + `train_modules=['sdf_fc1','policy_value','fc2']`。

结论：FC2 当前链路 **有问题**，报错如下：

```text
RuntimeError: The expanded size of the tensor (1000) must match the existing size (4060) at non-singleton dimension 0.
```

报错位置：
- `losses/FC2losspipe.py:76` (`P_s[i, : len(group)] = vals`)

调用链：
- `training/episode.py:_run_fc2_epochs -> train_step -> _compute_fc2_loss -> FC2LossPipe._build_tensors`

## 结论

1. Tensor 数据生成与训练的主链路（SDF/PV）目前联通正常。  
2. FC2 路径仍存在 shape 假设不一致问题，需要单独修复。  
