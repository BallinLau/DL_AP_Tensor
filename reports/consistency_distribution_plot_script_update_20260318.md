# 一致性对比分布图脚本更新（2026-03-18）

## 背景
你要求在一致性对比的 `.py` 文件中，绘制更多变量的分布图（不仅输出统计）。

## 修改内容

新增脚本：
- `experiments/compare_local_tensor_consistency.py`

功能：
1. 同时调用两个仓库（`DL_AP_Local` 与 `DL_AP_Tensor`）生成对比数据：
   - `sample_sdf`
   - `sample_pv`
   - `sim_firm`
   - `sim_macro`
2. 自动保存原始 CSV：
   - `out_dir/local/*.csv`
   - `out_dir/tensor/*.csv`
3. 自动绘制并保存叠加直方图（Local vs Tensor）：
   - `plots/sample_sdf_dist_overlay.png`
   - `plots/sample_pv_dist_overlay.png`
   - `plots/sim_firm_dist_overlay.png`
   - `plots/sim_macro_dist_overlay.png`
4. 自动导出统计摘要：
   - `out_dir/summary.json`

## 新增绘图变量

- `sample_sdf`: `x_t, x_t1, Hatcf_t, LnKF_t`
- `sample_pv`: `b, z, ETA, i, x, Hatcf, LnKF, K, M`
- `sim_firm`:
  - `b, z, ETA, i, x, Hatcf, LnKF, K, M`
  - `Q, P0, PI, Bar_i, Bar_z, P, bp`
  - `Y, I, Phi, C`
- `sim_macro`:
  - `K, C, LnK, Hatc, n_firms, M, x, hatcf, lnkf`

## 校验

1. 语法检查通过：
   - `python3 -m py_compile experiments/compare_local_tensor_consistency.py`
2. 小规模运行通过，并成功生成全部图与 summary：
   - 输出目录：`/Users/ballinliu/Desktop/PHD/Project1/cachedir/consistency_plots_test`

## 使用示例

```bash
/Users/ballinliu/anaconda3/bin/python \
  /Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/experiments/compare_local_tensor_consistency.py \
  --out-dir /Users/ballinliu/Desktop/PHD/Project1/cachedir/consistency_plots_test \
  --n-paths-sdf 128 --n-paths-pv 64 --n-paths-sim 12 \
  --group-size-sim 20 --horizon-sim 3 \
  --enable-entry --enable-exit
```
