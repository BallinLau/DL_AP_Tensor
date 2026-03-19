# SDF 均值锚/阶段2学习率 对比实验（2026-03-13）

配置：3 episodes（ep0=mode0, ep1-2=modeA），`train_modules=[sdf_fc1, policy_value]`，不训练 FC2。

## pre_fix_like

overrides: `{'sdf_stage2_lr': None, 'sdf_moment_weight': 3.0, 'sdf_log_mean_anchor_weight_stage1': 0.0, 'sdf_log_mean_anchor_weight_stage2': 0.0}`

| ep | mode | sdf_log_mean_M | sdf_mean_anchor_loss | parent_p_zero_ratio | parent_barz_mean | parent_bp_mean | grid_p_zero_ratio |
|---:|:-----|---------------:|---------------------:|--------------------:|-----------------:|---------------:|------------------:|
| 0 | mode0 | -0.2159 | 3.8314e-02 | 0.7667 | 0.7250 | 0.9161 | 0.6183 |
| 1 | modea | 0.0493 | 4.8332e-03 | 0.6897 | 0.6756 | 0.9609 | 0.5222 |
| 2 | modea | 0.3544 | 1.4033e-01 | 0.7840 | 0.7640 | 0.9594 | 0.5975 |

## current_new

| ep | mode | sdf_log_mean_M | sdf_mean_anchor_loss | parent_p_zero_ratio | parent_barz_mean | parent_bp_mean | grid_p_zero_ratio |
|---:|:-----|---------------:|---------------------:|--------------------:|-----------------:|---------------:|------------------:|
| 0 | mode0 | 0.1222 | 2.0282e-02 | 0.7667 | 0.7251 | 0.9161 | 0.6186 |
| 1 | modea | 0.0465 | 4.4430e-03 | 0.6986 | 0.6837 | 0.9563 | 0.5192 |
| 2 | modea | -0.0848 | 4.1720e-03 | 0.6284 | 0.6189 | 0.9548 | 0.4786 |
