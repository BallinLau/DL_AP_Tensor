# PI Continuation Gain Decomposition

## 直接答案：child PI continuation 为什么让 P movement 不衰减？

后五期中，PI channel 的 realized L2 gain 超过 1 的 transition 占比为 **100%**，directional gain 超过 1 的占比为 **100%**；同时 frozen-active-set PI row gain >1 的平均 state share 为 **24.3%**。这说明是否存在不衰减必须同时由下面三组数值判断，不能把 row gain 或 norm ratio 称为 spectral radius。

## 最关键结果

| transition   |   G_II_l2 |   lambda_II_dir |   row_gain_mean |   row_gain_p90 |   row_gain_gt1_share |   row_gain_g1_gt1_share |   row_gain_M1_gt1_share |   PI_active_share |
|:-------------|----------:|----------------:|----------------:|---------------:|---------------------:|------------------------:|------------------------:|------------------:|
| 0->1         |  0.170859 |        0.065745 |        0.162738 |       0.423860 |             0.000977 |                0.000000 |                0.000977 |          0.160077 |
| 1->2         |  0.664870 |        0.588338 |        0.287053 |       0.778730 |             0.038574 |                0.000000 |                0.038574 |          0.267659 |
| 2->3         |  0.882668 |        0.798194 |        0.331936 |       0.862507 |             0.061523 |                0.000000 |                0.061523 |          0.304369 |
| 3->4         |  0.888804 |        0.814781 |        0.363249 |       0.893410 |             0.064941 |                0.000000 |                0.064941 |          0.334055 |
| 4->5         |  0.729763 |        0.682566 |        0.453045 |       1.037614 |             0.120117 |                0.000000 |                0.120117 |          0.411577 |
| 5->6         |  1.198893 |        1.124035 |        0.507225 |       1.089736 |             0.169434 |                0.000000 |                0.169434 |          0.460031 |
| 6->7         |  1.061545 |        1.026053 |        0.526392 |       1.087191 |             0.166992 |                0.000000 |                0.166992 |          0.474599 |
| 7->8         |  1.095670 |        1.067436 |        0.569103 |       1.121151 |             0.225586 |                0.000000 |                0.225586 |          0.504356 |
| 8->9         |  1.107888 |        1.085974 |        0.602038 |       1.127434 |             0.264648 |                0.000000 |                0.264648 |          0.543951 |
| 9->10        |  1.377223 |        1.351266 |        0.659485 |       1.140000 |             0.387695 |                0.000000 |                0.387695 |          0.595904 |

### 后五期平均

|                |   G_II_l2 |   lambda_II_dir |   row_gain_mean |   row_gain_p90 |   row_gain_gt1_share |   row_gain_g1_gt1_share |   row_gain_M1_gt1_share |   PI_active_share |
|:---------------|----------:|----------------:|----------------:|---------------:|---------------------:|------------------------:|------------------------:|------------------:|
| 5->6 ... 9->10 |  1.168244 |        1.130953 |        0.572848 |       1.113102 |             0.242871 |                0.000000 |                0.242871 |          0.515768 |

## Code-confirmed operator

Production `BPGridTeacher._branch_objective_grid` uses `CFI + Config.G * used_M * P(child)` for PI. `PolicyValue.cal_phats` evaluates the exact internal uniform i-grid, averages `max(P0,PI)`, and `FirmDerivedObjects.total_equity(..., hard=True)` applies `clamp_min(0)`. The Episode path clamps used M with the run's configured bounds and enumerates future eta exactly with normalized weights. For this fixed-point path the solver passes a seven-column parent into `_build_policy_m_lists`; therefore the production P backup sets used M to one. The raw SDF stored as child column 8 is not consumed by this path.

## Numerical invariants

- `analysis_date`: `2026-10-15`
- `optimizer_count`: `0`
- `q_hash_identical`: `True`
- `q_hash`: `04d10e09b9db1ac4867c36b22642946b9e47ca8bf7a00fe96ba6d4d18cd5946d`
- `bp_hash_identical`: `True`
- `validation_parent_fingerprint`: `2f40a1df1aed6ed8a0ea2cf948f09ac744ca5c0a7a3ef8e60ff0cedcb217f6b0`
- `validation_child_fingerprint`: `15645cc7c677a6e5aaa8bec7fe0b15878ffc24a84f59edc3240e84e0df3908cb`
- `parameter_hash_unchanged`: `True`
- `max_gain_closure_error`: `2.086162567138672e-07`
- `alpha_min`: `0.0`
- `alpha_max`: `1.0`
- `same_b_g_ratio_max_error`: `0.0`
- `Config.G`: `1.14`
- `used_M_min`: `1.0`
- `used_M_negative_average_mass`: `0.0`
- `finite_difference_enabled`: `True`
- `finite_difference_parents`: `256`
- `hard_survival_semantics`: `1{mean_i max(P0,PI)>0}; P=clamp_min(Phat,0)`
- `source_checkpoint`: `/WORK/PUBLIC/zhuyingz_work/LiuHao/cachedir/normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_investment_compare_fresh100_20261014_144528/checkpoints/ep1_combined.pt`

## Structural attribution

- g threshold effect: late transitions show a lower g=1 >1 share in 100% of cases.
- M one-factor effect on mean PI row gain (actual minus M=1): `0.000000`. Positive means M amplifies; negative means it stabilizes.
- Survival effect on mean gain (survival minus gross): `-0.323293`. Negative means survival attenuates.
- PI-active weighted share moves from `0.160077` to `0.595904` over the path.
- In late-sweep top-decile gain states, eta=1 has mean share `82.9%`; amplification is concentrated in refinancing-active parents.
- These are exact code facts for the frozen active set and empirical facts along one realized P-only trajectory; they are not an eigenvalue proof.

## Final classification

- **Primary amplifier — g=1.14.** With used M=1, every g=1 PI row gain is at most one and its >1 share is zero; all actual >1 row gains cross the threshold through g.
- **Secondary amplifier — declining P0 dilution.** PI-active share rises materially, so the PI loop inherits more of the surviving continuation mass in later sweeps.
- **Stabilizing factors — survival and incomplete PI activity.** Both reduce gross 1.14 gain; survival alone lowers the late mean from 1.14 to about the reported `a_surv_mean`, and PI selection lowers it further to `row_gain_mean`.
- **Ruled out — M amplification.** Used M is identically one in the actual operator, so the M=1 counterfactual is exactly the actual result and all M-selection covariances are zero.
- **Still relevant — state redistribution/function shape.** Late mean row gain remains below one and only a minority of states exceed one, while realized directional gain exceeds one in every late transition. Row gains therefore support but do not fully explain the realized amplification.
- **Non-contraction status.** These diagnostics are strong local empirical evidence of non-damping along the realized P-only direction, but do not prove a global non-contraction or spectral radius above one.

## 对十个问题的逐项回答

1. **Realized gain 是否 >1：是（后五期）。** 后五期 `G_II L2` 平均 `1.168244`，5/5 transitions >1。
2. **Directional gain 是否 >1：是（后五期）。** `lambda_II_dir` 平均 `1.130953`，cosine 平均 `0.967946`。
3. **Row gain 是否大面积 >1：有实质性尾部，但不是多数。** 后五期均值 `0.572848`、p90 `1.113102`、>1 share `24.3%`；9->10 的 >1 share `38.8%`。
4. **g 是否是跨越 1 的关键：是。** g=1 时 >1 share 在所有 transitions 都为零；实际 g=1.14 才产生 >1 rows。
5. **M 是 amplifier 还是 stabilizer：两者都不是。** 当前真实 P operator 的 used M 恒为 1，actual 与 M=1 counterfactual 完全相同。
6. **survival：稳定器。** 后五期把 gross mean `1.14` 降到 `0.816707`。
7. **PI-active share：水平上仍是 attenuation，但 attenuation 在减弱。** share 从 `0.160077` 升到 `0.595904`，所以越来越少 continuation mass 被 P0 稀释。
8. **High-gain states：集中在 eta=1 和高 z。** 后五期 top-decile 中 eta=1 share `82.9%`；平均 `(b,z,i,x,Hatcf,LnKF)=(0.410,1.074,0.236,-2.001,-2.171,4.676)`。
9. **是否足以说明 non-contraction：不足以作全局定理。** 足以说明这条 realized trajectory 的局部 observed direction 在后期不衰减并被放大，但没有识别完整 Jacobian 的 spectral radius。
10. **证据层级：** operator 公式、hard survival、M=1 路径、closure、hash 不变量是 exact code/numerical facts；gain ratios、high-state concentration 和 amplifier 排序是这一次固定 bank/trajectory 上的 empirical evidence。

## High-gain state localization

| transition   | selection   | eta_group   |   count |   population_share |   a_PI_top10_cutoff |   b_mean |   b_p50 |   b_p90 |   z_mean |   z_p50 |   z_p90 |   eta_mean |   eta_p50 |   eta_p90 |   i_mean |   i_p50 |   i_p90 |   x_mean |    x_p50 |    x_p90 |   Hatcf_mean |   Hatcf_p50 |   Hatcf_p90 |   LnKF_mean |   LnKF_p50 |   LnKF_p90 |   a_PI_mean |   a_PI_p50 |   a_PI_p90 |   M_mean_mean |   M_mean_p50 |   M_mean_p90 |   survival_share_mean |   survival_share_p50 |   survival_share_p90 |   alpha_mean |   alpha_p50 |   alpha_p90 |   survival_PI_share_mean |   survival_PI_share_p50 |   survival_PI_share_p90 |   phat_abs_margin_mean |   phat_abs_margin_p50 |   phat_abs_margin_p90 |   pi_p0_abs_margin_mean |   pi_p0_abs_margin_p50 |   pi_p0_abs_margin_p90 |
|:-------------|:------------|:------------|--------:|-------------------:|--------------------:|---------:|--------:|--------:|---------:|--------:|--------:|-----------:|----------:|----------:|---------:|--------:|--------:|---------:|---------:|---------:|-------------:|------------:|------------:|------------:|-----------:|-----------:|------------:|-----------:|-----------:|--------------:|-------------:|-------------:|----------------------:|---------------------:|---------------------:|-------------:|------------:|------------:|-------------------------:|------------------------:|------------------------:|-----------------------:|----------------------:|----------------------:|------------------------:|-----------------------:|-----------------------:|
| 0->1         | top10       | all         |     205 |            0.10010 |             0.42386 |  0.39057 | 0.34558 | 0.87659 |  1.13848 | 1.13295 | 1.72389 |    0.75122 |   1.00000 |   1.00000 |  0.23982 | 0.24237 | 0.44792 | -2.00203 | -2.00190 | -1.97143 |     -2.33412 |    -2.11133 |    -1.70853 |     5.34245 |    5.18618 |    7.28464 |     0.54681 |    0.51786 |    0.67726 |       1.00000 |      1.00000 |      1.00000 |               0.99970 |              1.00000 |              1.00000 |      0.47968 |     0.45426 |     0.59409 |                  0.47965 |                 0.45426 |                 0.59409 |                3.01227 |               2.64767 |               4.78822 |                 0.59523 |                0.55254 |                0.89794 |
| 1->2         | top10       | all         |     205 |            0.10010 |             0.77873 |  0.40768 | 0.37060 | 0.87659 |  1.14528 | 1.06109 | 1.72389 |    0.81463 |   1.00000 |   1.00000 |  0.23073 | 0.22965 | 0.43933 | -2.00127 | -2.00118 | -1.97326 |     -2.13203 |    -2.10848 |    -1.56814 |     5.09266 |    5.06401 |    6.69722 |     0.96059 |    0.95060 |    1.11736 |       1.00000 |      1.00000 |      1.00000 |               1.00000 |              1.00000 |              1.00000 |      0.84262 |     0.83386 |     0.98014 |                  0.84262 |                 0.83386 |                 0.98014 |                3.61955 |               3.27051 |               5.43290 |                 0.20646 |                0.16908 |                0.31107 |
| 2->3         | top10       | all         |     205 |            0.10010 |             0.86251 |  0.38909 | 0.33920 | 0.86688 |  1.16778 | 1.11641 | 1.74052 |    0.78049 |   1.00000 |   1.00000 |  0.24200 | 0.24627 | 0.44403 | -2.00115 | -2.00121 | -1.97326 |     -2.33336 |    -2.14123 |    -1.62343 |     5.28005 |    5.21948 |    7.29929 |     1.02816 |    1.03831 |    1.13981 |       1.00000 |      1.00000 |      1.00000 |               1.00000 |              1.00000 |              1.00000 |      0.90189 |     0.91080 |     0.99983 |                  0.90189 |                 0.91080 |                 0.99983 |                4.38446 |               4.02718 |               6.28617 |                 0.29526 |                0.25763 |                0.44782 |
| 3->4         | top10       | all         |     205 |            0.10010 |             0.89341 |  0.38984 | 0.34416 | 0.85516 |  1.16786 | 1.13091 | 1.74052 |    0.78537 |   1.00000 |   1.00000 |  0.24085 | 0.24259 | 0.45091 | -2.00157 | -2.00172 | -1.97326 |     -2.33864 |    -2.14943 |    -1.59966 |     5.19302 |    5.15121 |    7.00322 |     1.04031 |    1.05048 |    1.14000 |       1.00000 |      1.00000 |      1.00000 |               1.00000 |              1.00000 |              1.00000 |      0.91256 |     0.92148 |     1.00000 |                  0.91256 |                 0.92148 |                 1.00000 |                5.19607 |               4.77167 |               7.60346 |                 0.38861 |                0.34063 |                0.61808 |
| 4->5         | top10       | all         |     205 |            0.10010 |             1.03761 |  0.40823 | 0.36139 | 0.86038 |  1.14214 | 1.08148 | 1.70454 |    0.80976 |   1.00000 |   1.00000 |  0.23395 | 0.23399 | 0.43608 | -2.00102 | -2.00118 | -1.97143 |     -2.28433 |    -2.10495 |    -1.46020 |     5.05196 |    5.02702 |    6.88843 |     1.10750 |    1.11467 |    1.14000 |       1.00000 |      1.00000 |      1.00000 |               1.00000 |              1.00000 |              1.00000 |      0.97149 |     0.97778 |     1.00000 |                  0.97149 |                 0.97778 |                 1.00000 |                6.17474 |               5.63051 |               9.01844 |                 0.56207 |                0.49483 |                0.87209 |
| 5->6         | top10       | all         |     208 |            0.10156 |             1.08974 |  0.40810 | 0.36014 | 0.87350 |  1.12668 | 1.05959 | 1.70332 |    0.82212 |   1.00000 |   1.00000 |  0.23412 | 0.23235 | 0.43590 | -2.00112 | -2.00099 | -1.97120 |     -2.26935 |    -2.09834 |    -1.40082 |     4.96036 |    4.97835 |    6.76321 |     1.12634 |    1.13352 |    1.14000 |       1.00000 |      1.00000 |      1.00000 |               1.00000 |              1.00000 |              1.00000 |      0.98801 |     0.99432 |     1.00000 |                  0.98801 |                 0.99432 |                 1.00000 |                6.90424 |               6.26106 |              10.07557 |                 0.68370 |                0.60518 |                1.07193 |
| 6->7         | top10       | all         |     205 |            0.10010 |             1.08719 |  0.41279 | 0.37060 | 0.86688 |  1.14144 | 1.06777 | 1.70856 |    0.82927 |   1.00000 |   1.00000 |  0.23587 | 0.23546 | 0.43556 | -2.00071 | -2.00121 | -1.97067 |     -2.21030 |    -2.09717 |    -1.33396 |     4.88416 |    4.94998 |    6.69114 |     1.12434 |    1.13352 |    1.14000 |       1.00000 |      1.00000 |      1.00000 |               1.00000 |              1.00000 |              1.00000 |      0.98626 |     0.99432 |     1.00000 |                  0.98626 |                 0.99432 |                 1.00000 |                8.01086 |               7.30737 |              11.62112 |                 0.73176 |                0.63328 |                1.17251 |
| 7->8         | top10       | all         |     209 |            0.10205 |             1.12115 |  0.40942 | 0.36703 | 0.87247 |  1.11865 | 1.05630 | 1.70697 |    0.82775 |   1.00000 |   1.00000 |  0.23450 | 0.23798 | 0.43540 | -2.00089 | -2.00105 | -1.97121 |     -2.16961 |    -2.07254 |    -1.31307 |     4.76344 |    4.78584 |    6.63358 |     1.13547 |    1.14000 |    1.14000 |       1.00000 |      1.00000 |      1.00000 |               1.00000 |              1.00000 |              1.00000 |      0.99602 |     1.00000 |     1.00000 |                  0.99602 |                 1.00000 |                 1.00000 |                9.25918 |               8.53725 |              13.44130 |                 0.93401 |                0.83447 |                1.49816 |
| 8->9         | top10       | all         |     206 |            0.10059 |             1.12743 |  0.41647 | 0.37856 | 0.87556 |  1.06558 | 0.99275 | 1.68460 |    0.83981 |   1.00000 |   1.00000 |  0.23939 | 0.23438 | 0.44358 | -2.00172 | -2.00127 | -1.97074 |     -2.13509 |    -2.00856 |    -1.20684 |     4.44453 |    4.40007 |    6.01585 |     1.13777 |    1.14000 |    1.14000 |       1.00000 |      1.00000 |      1.00000 |               1.00000 |              1.00000 |              1.00000 |      0.99805 |     1.00000 |     1.00000 |                  0.99805 |                 1.00000 |                 1.00000 |               10.90460 |              10.01643 |              15.77300 |                 1.12314 |                0.99658 |                1.73565 |
| 9->10        | top10       | all         |     278 |            0.13574 |             1.14000 |  0.40488 | 0.37345 | 0.85784 |  0.91635 | 0.84523 | 1.64050 |    0.82374 |   1.00000 |   1.00000 |  0.23779 | 0.23638 | 0.44269 | -2.00253 | -2.00151 | -1.97089 |     -2.06974 |    -1.98958 |    -1.19304 |     4.32775 |    4.24871 |    5.92795 |     1.14000 |    1.14000 |    1.14000 |       1.00000 |      1.00000 |      1.00000 |               1.00000 |              1.00000 |              1.00000 |      1.00000 |     1.00000 |     1.00000 |                  1.00000 |                 1.00000 |                 1.00000 |               11.56129 |              10.50632 |              17.54455 |                 1.24379 |                1.13493 |                1.98484 |

## Interpretation hierarchy

- `G_II` is a realized norm gain; `lambda_II_dir` is the observed-direction gain.
- `a_PI` is a local frozen-active-set row gain, not a spectral radius.
- survival and PI share can strongly attenuate gross g*M even when the default-switch movement channel is tiny.
- Function-shape/state redistribution remains relevant whenever directional amplification is not explained by the statewise row gains.

## Function-output finite-difference validation

`P_epsilon` interpolates P0/PI function outputs, not network parameters. Q stays at Q0 and the full production PI teacher re-optimizes b'. This validation uses the first `256` fixed validation parents in unchanged ordering.

| transition   |   fd_eps0.01_cont_cosine |   fd_eps0.01_cont_relative_norm_error |   fd_eps0.01_cont_norm_ratio |   fd_eps0.01_linear_cosine |   fd_eps0.01_linear_relative_norm_error |   fd_eps0.01_linear_norm_ratio |   fd_eps0.05_cont_cosine |   fd_eps0.05_cont_relative_norm_error |   fd_eps0.05_cont_norm_ratio |   fd_eps0.05_linear_cosine |   fd_eps0.05_linear_relative_norm_error |   fd_eps0.05_linear_norm_ratio |   fd_eps0.1_cont_cosine |   fd_eps0.1_cont_relative_norm_error |   fd_eps0.1_cont_norm_ratio |   fd_eps0.1_linear_cosine |   fd_eps0.1_linear_relative_norm_error |   fd_eps0.1_linear_norm_ratio |
|:-------------|-------------------------:|--------------------------------------:|-----------------------------:|---------------------------:|----------------------------------------:|-------------------------------:|-------------------------:|--------------------------------------:|-----------------------------:|---------------------------:|----------------------------------------:|-------------------------------:|------------------------:|-------------------------------------:|----------------------------:|--------------------------:|---------------------------------------:|------------------------------:|
| 0->1         |                 0.997687 |                              0.113879 |                     0.906325 |                   0.999999 |                                0.001211 |                       1.000306 |                 0.997771 |                              0.112463 |                     0.907247 |                   0.999993 |                                0.003884 |                       1.001323 |                0.997986 |                             0.109516 |                    0.908716 |                  0.999948 |                               0.010609 |                      1.002944 |
| 1->2         |                 0.999173 |                              0.041018 |                     0.993683 |                   0.999311 |                                0.037481 |                       1.004549 |                 0.999891 |                              0.017505 |                     0.990476 |                   0.999987 |                                0.005350 |                       1.001307 |                0.999911 |                             0.016002 |                    0.991043 |                  0.999984 |                               0.006029 |                      1.001880 |
| 2->3         |                 0.999956 |                              0.010371 |                     0.995564 |                   1.000000 |                                0.000287 |                       1.000088 |                 0.999957 |                              0.010155 |                     0.995743 |                   1.000000 |                                0.000738 |                       1.000269 |                0.999930 |                             0.012375 |                    0.996410 |                  0.999969 |                               0.007980 |                      1.000939 |
| 3->4         |                 0.999931 |                              0.018039 |                     0.986227 |                   0.999999 |                                0.001171 |                       1.000174 |                 0.999935 |                              0.017431 |                     0.986724 |                   0.999999 |                                0.001656 |                       1.000678 |                0.999939 |                             0.016494 |                    0.987717 |                  0.999997 |                               0.002906 |                      1.001685 |
| 4->5         |                 0.999974 |                              0.010749 |                     0.992031 |                   1.000000 |                                0.000248 |                       1.000047 |                 0.999976 |                              0.010374 |                     0.992295 |                   1.000000 |                                0.000704 |                       1.000313 |                0.999979 |                             0.009797 |                    0.992697 |                  0.999999 |                               0.001368 |                      1.000719 |
| 5->6         |                 0.999981 |                              0.006489 |                     0.997895 |                   1.000000 |                                0.000636 |                       1.000027 |                 0.999982 |                              0.006263 |                     0.997976 |                   0.999999 |                                0.001074 |                       1.000109 |                0.999984 |                             0.006014 |                    0.998086 |                  0.999999 |                               0.001289 |                      1.000219 |
| 6->7         |                 0.999993 |                              0.005246 |                     0.996216 |                   1.000000 |                                0.000187 |                       1.000046 |                 0.999994 |                              0.005032 |                     0.996482 |                   1.000000 |                                0.001010 |                       1.000313 |                0.999993 |                             0.004968 |                    0.996825 |                  0.999998 |                               0.002108 |                      1.000657 |
| 7->8         |                 0.999988 |                              0.005301 |                     0.997967 |                   1.000000 |                                0.000094 |                       1.000016 |                 0.999990 |                              0.004906 |                     0.998063 |                   1.000000 |                                0.000604 |                       1.000113 |                0.999991 |                             0.004670 |                    0.998169 |                  1.000000 |                               0.000832 |                      1.000219 |
| 8->9         |                 0.999987 |                              0.006892 |                     0.995438 |                   1.000000 |                                0.000224 |                       1.000023 |                 0.999988 |                              0.006545 |                     0.995672 |                   1.000000 |                                0.000730 |                       1.000259 |                0.999989 |                             0.006074 |                    0.996035 |                  0.999999 |                               0.001537 |                      1.000623 |
| 9->10        |                 0.999993 |                              0.004231 |                     0.998119 |                   1.000000 |                                0.000082 |                       1.000015 |                 0.999993 |                              0.004079 |                     0.998230 |                   1.000000 |                                0.000399 |                       1.000127 |                0.999993 |                             0.004058 |                    0.998388 |                  0.999999 |                               0.001183 |                      1.000285 |

## Outputs

- `/private/tmp/dl_ap_tensor_fresh100_tol1/cachedir/pi_continuation_gain_20261015/pi_continuation_gain_by_sweep.csv`
- `/private/tmp/dl_ap_tensor_fresh100_tol1/cachedir/pi_continuation_gain_20261015/pi_continuation_gain_statewise.parquet`
- `/private/tmp/dl_ap_tensor_fresh100_tol1/cachedir/pi_continuation_gain_20261015/pi_gain_high_state_summary.csv`
- `/private/tmp/dl_ap_tensor_fresh100_tol1/cachedir/pi_continuation_gain_20261015/analysis_metadata.json`
- `/private/tmp/dl_ap_tensor_fresh100_tol1/cachedir/pi_continuation_gain_20261015/figures/01_realized_G_II.png`
- `/private/tmp/dl_ap_tensor_fresh100_tol1/cachedir/pi_continuation_gain_20261015/figures/02_directional_lambda_II.png`
- `/private/tmp/dl_ap_tensor_fresh100_tol1/cachedir/pi_continuation_gain_20261015/figures/03_row_gain_distribution.png`
- `/private/tmp/dl_ap_tensor_fresh100_tol1/cachedir/pi_continuation_gain_20261015/figures/04_row_gain_gt1_share.png`
- `/private/tmp/dl_ap_tensor_fresh100_tol1/cachedir/pi_continuation_gain_20261015/figures/05_actual_vs_g1.png`
- `/private/tmp/dl_ap_tensor_fresh100_tol1/cachedir/pi_continuation_gain_20261015/figures/06_actual_vs_M1.png`
- `/private/tmp/dl_ap_tensor_fresh100_tol1/cachedir/pi_continuation_gain_20261015/figures/07_gross_survival_pi.png`
- `/private/tmp/dl_ap_tensor_fresh100_tol1/cachedir/pi_continuation_gain_20261015/figures/08_pi_active_share.png`
- `/private/tmp/dl_ap_tensor_fresh100_tol1/cachedir/pi_continuation_gain_20261015/figures/09_survival_share.png`
- `/private/tmp/dl_ap_tensor_fresh100_tol1/cachedir/pi_continuation_gain_20261015/figures/10_M_conditional.png`
- `/private/tmp/dl_ap_tensor_fresh100_tol1/cachedir/pi_continuation_gain_20261015/figures/11_alignment_vs_pi_active.png`
- `/private/tmp/dl_ap_tensor_fresh100_tol1/cachedir/pi_continuation_gain_20261015/figures/12_high_gain_vs_b.png`
- `/private/tmp/dl_ap_tensor_fresh100_tol1/cachedir/pi_continuation_gain_20261015/figures/13_high_gain_vs_z.png`
- `/private/tmp/dl_ap_tensor_fresh100_tol1/cachedir/pi_continuation_gain_20261015/figures/14_high_gain_vs_x.png`
