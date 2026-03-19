# bp FOC 使用 Phat 梯度 对比（20260313_182504）

配置：3 episodes（ep0=mode0, ep1-2=modeA），train_modules=[sdf_fc1, policy_value]，不训练FC2。

## p_foc_grad_off

overrides: `{'bp_foc_use_phat_children': False}`

| ep | mode | p0_pgrad_abs_mean | pi_pgrad_abs_mean | parent_bp_mean | parent_p_zero_ratio | parent_barz_mean |
|---:|:-----|------------------:|------------------:|---------------:|--------------------:|-----------------:|
| 0 | mode0 | 9.7053e-03 | 1.0419e-02 | 0.8038 | 0.0000 | nan |
| 1 | modea | 1.0113e-03 | 1.0262e-03 | 0.7225 | 0.0000 | nan |
| 2 | modea | 2.4604e-03 | 2.5179e-03 | 0.6776 | 0.0006 | nan |

## p_foc_grad_on

overrides: `{'bp_foc_use_phat_children': True}`

| ep | mode | p0_pgrad_abs_mean | pi_pgrad_abs_mean | parent_bp_mean | parent_p_zero_ratio | parent_barz_mean |
|---:|:-----|------------------:|------------------:|---------------:|--------------------:|-----------------:|
| 0 | mode0 | 9.7053e-03 | 1.0419e-02 | 0.8038 | 0.0000 | nan |
| 1 | modea | 1.0113e-03 | 1.0262e-03 | 0.7225 | 0.0000 | nan |
| 2 | modea | 2.4604e-03 | 2.5179e-03 | 0.6776 | 0.0006 | nan |
