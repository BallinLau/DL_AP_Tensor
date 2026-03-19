# GPU 高吞吐训练入口修改说明

**日期**: 2026-03-19

## 本次修改

1. 提高默认训练批大小
- 文件: [experiments/run_utils.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/experiments/run_utils.py)
- 修改: `build_hyperparams()` 的默认 `batch_size` 从 `512` 提高到 `4096`
- 原因: 当前模型较小、状态维度较低，`512` 对 80GB GPU 明显过保守。

2. 去掉 `episode > 0` 时强行截断 `n_paths <= 100`
- 文件: [experiments/run_multi_episode_job.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/experiments/run_multi_episode_job.py)
- 修改: 默认令后续 episode 继续使用完整 `hyperparams.n_paths`
- 原因: 这是导致后续 episode GPU 占用极低的最直接原因。

3. 新增训练负载控制参数
- 文件: [experiments/run_multi_episode_job.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/experiments/run_multi_episode_job.py)
- 新增参数:
  - `--batch-size`
  - `--post0-n-paths`
  - `--simulate-group-size`
- 原因: 便于在服务器上直接放大训练负载，而不必改代码。

4. 新增每个 episode 的 GPU 峰值日志
- 文件: [experiments/run_multi_episode_job.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/experiments/run_multi_episode_job.py)
- 输出内容:
  - 当前 episode 的 `mode / batch_size / n_paths / group_size / horizon`
  - `Allocated / Reserved / Max Allocated`
- 原因: 训练后可以立即判断本轮工作集是否真的被放大。

5. 保持 quick test 轻量
- 文件: [experiments/run_multi_episode_job.py](/Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/experiments/run_multi_episode_job.py)
- 修改: `--quick-test` 下把 `batch_size` 限在 `1024`
- 原因: smoke test 的目标是验证链路，不是榨满 GPU。

## 这次修改解决了什么

之前 A800 仅有十几 MB 当前显存、两百 MB 左右峰值，不是 CUDA 没用，而是训练入口把负载设得过小：

1. `episode > 0` 时 `n_paths` 被截到 `100`
2. 默认 `batch_size = 512`
3. 训练数据规模和 batch 都不足以让 GPU 形成大的工作集

本次修改先解决前两项，并把负载参数暴露到 CLI。

## 建议的服务器命令

先从这个量级开始：

```bash
python3 experiments/run_multi_episode_job.py \
  --device cuda:0 \
  --n-episodes 3 \
  --epochs 20 \
  --n-paths 2000 \
  --post0-n-paths 2000 \
  --simulate-group-size 1000 \
  --simulate-horizon 100 \
  --batch-size 8192
```

如果 loss 和数值稳定，再向上试：

```bash
--batch-size 16384
```

## 还没做的事

这次只是把训练入口从“保守小负载”改成“可放大负载”。

真正进一步提高吞吐，还需要后续继续做：

1. 减少 Python list-of-batches 调度
2. 把更多训练阶段改成大张量连续切片
3. 评估 `torch.compile`
4. 视稳定性决定是否引入 AMP
