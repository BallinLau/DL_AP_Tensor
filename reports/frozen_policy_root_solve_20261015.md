# Frozen-policy neural root solve diagnostic

Date: 2026-10-15

Base commit: `ada06a5076228cc14fcb973139658e4973f7f47f`

Branch: `exp/frozen-policy-root-solve`

Operator: `LEGACY_M1_OPERATOR = true`

## Scope and status

This is an analysis-only numerical diagnostic. No production training file,
economic equation, grid, transition law, network width, Q equation, or default
rule was changed. The implementation is contained in
`analysis/test_frozen_policy_root_solve.py`; its tests are in
`tests/test_frozen_policy_root_solve.py`.

The completed local run is the required deterministic K=5 bounded smoke with
2,048 training parents and 2,048 validation parents. It ran ten Adam steps and
one production-like fixed-policy Picard outer step. LBFGS was deliberately set
to zero iterations in this bounded local smoke. Therefore the row named
`Root-Adam+LBFGS` equals the Adam result and is not evidence about LBFGS. The
full 8,192-parent/300-Adam/200-LBFGS command is given below for the server.

## Audited production call chain

1. `Episode._policy_batch_hash_components` converts the parent to the seven
   economic state columns and calls `_build_policy_m_lists`
   (`training/episode.py:8922-8941`). On the historical P-only banks, the
   parent has exactly seven columns, so `_build_policy_m_lists` constructs ones
   (`training/episode.py:1225-1238`). The diagnostic additionally aborts unless
   `max_abs(used_M - 1) < 1e-8`.
2. Exact eta integration comes from production
   `_expand_policy_expectation_children` (`training/episode.py:1245-1265`).
   The historical 16 continuous shocks therefore become 32 weighted eta
   branches.
3. Historical P targets are built by `BPGridTeacher.compute_value_target`
   (`training/episode.py:8972-8985`; implementation at
   `training/bp_grid_teacher.py:846-889`). BP is chosen by the production
   coarse/fine grid and star selection (`training/bp_grid_teacher.py:1337-1395`),
   not by the policy heads.
4. Candidate child leverage follows
   `b_next = eta_current * bp + (1-eta_current) * b_parent`
   (`training/bp_grid_teacher.py:198-234`). Production branch objectives use
   `cashflow0 + M*P_child` for P0 and
   `cashflowI + G*M*P_child` for PI
   (`training/bp_grid_teacher.py:1698-1790`). Here M is frozen to one.
5. The production value network uses the physical/normalized parameterization
   in `_value_outputs` and exact internal i-grid in `cal_phats`; the hard
   limited-liability value is `max(Phat, 0)`
   (`models/policy_value.py:288-338`, `models/policy_value.py:557-597`).
6. The historical Picard P stage stores `value_star.detach()` in its target
   cache (`training/episode.py:9000-9005`), then backpropagates a supervised
   cached-target loss (`training/episode.py:8432-8455`). This is the detached
   target mechanism being compared with the full-residual solve.
7. The diagnostic freezes BP stars, exact-eta weights, investment selector,
   hard survival/default active set, branch cash flow, and Q at theta_k. It
   then forwards current-theta child P on every optimizer/closure evaluation
   (`analysis/test_frozen_policy_root_solve.py:258-310`). The RHS is not cached
   and child P is not detached. The production P-stage parameter set is reused
   through `Episode._policy_value_stage_params("p")`
   (`training/episode.py:924-947`).

## Input snapshot and invariants

- Sweep checkpoint: historical `sweep_005.pt`, format
  `inner_fixed_point_v1`, independently loaded from the same source
  `ep1_combined.pt` as the P-only run.
- Train bank fingerprint:
  `5c2f7c0a4f7cfed287673018dc65c37bab659c571f40fd8b838960160907b7fb`.
- Validation bank fingerprint:
  `2f40a1df1aed6ed8a0ea2cf948f09ac744ca5c0a7a3ef8e60ff0cedcb217f6b0`.
- Train child fingerprint:
  `5fe18cebed78d539c0f838feb6e0d52f40b09cf02fa0a6f33ec3573d412f5389`.
- Validation child fingerprint:
  `15645cc7c677a6e5aaa8bec7fe0b15878ffc24a84f59edc3240e84e0df3908cb`.
- `max_abs(used_M - 1) = 0`.
- Q hash before/after:
  `04d10e09b9db1ac4867c36b22642946b9e47ca8bf7a00fe96ba6d4d18cd5946d`.
- BP/policy hash before/after:
  `85b180644922138b968e0aaa06bf80ee396ecedf7464a948cc9ca005f4e4bf2c`.
- Frozen-policy train and validation hashes are identical before/after.
- The production P scope contains 132,676 parameters. The exact names are in
  `metadata.json`; Q and BP/policy modules are absent.
- Runtime value mode is `exp_xz`; Bellman residuals use the runtime equity
  scale because `pv_bellman_normalize_by_value_scale=true`.

Initialization parity on all 4,096 smoke rows:

| branch | mean abs error | p90 | max | tolerance | pass |
|---|---:|---:|---:|---:|---|
| P0 | 1.05e-7 | 2.38e-7 | 1.91e-6 | 1e-5 | yes |
| PI | 1.26e-7 | 4.77e-7 | 1.91e-6 | 1e-5 | yes |

The gradient tests establish both directions: dynamic child continuation gives
nonzero gradients to the value path, while the detached-child negative control
removes the RHS gradient. A separate test verifies that the full residual
gradient differs from the semi-gradient control.

## Bounded K=5 smoke result

| method | initial val residual | best val residual | final val residual | reduction | P0 norm p90 | PI norm p90 | P max | finite | diagnostic status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Picard-fixed-policy | 0.267793 | 0.267793 | 0.310639 | 1.00x | 0.732611 | 0.832423 | 17.7463 | yes | no damping in this short smoke |
| Root-Adam | 0.267793 | 0.171126 | 0.171126 | 1.56x | 0.412156 | 0.470806 | 15.2590 | yes | optimizer has not found a low-residual root |
| Root-Adam+LBFGS | 0.267793 | 0.171126 | 0.171126 | 1.56x | 0.412156 | 0.470806 | 15.2590 | yes | LBFGS not run in bounded smoke |

The label does **not** mean a fixed point does not exist. It says only that ten
Adam updates did not meet the predeclared 10x/100x success thresholds.

One policy refresh after the short Adam root attempt produced:

- bp0 mean absolute change `4.46e-4`, p90 `0`, p99 `0.00625`;
- bpI mean absolute change `4.49e-4`, p90 `0`, p99 `0.00625`;
- investment-selector flip share `0.13259`;
- default/survival flip share `0.009766`;
- PI-active share `0.49911 -> 0.37993`;
- refreshed greedy normalized MAE: P0 `0.15781`, PI `0.18697`.

Because this is only ten Adam steps, these policy-change numbers are diagnostic
and not a converged-policy result.

## Direct answers

**Q1. Does production-style Picard remain non-damping under the same frozen
policy?** In the one-outer-step bounded smoke it worsened from `0.267793` to
`0.310639`. This is evidence of non-damping in this short run, but five to ten
outer iterations are still required for the intended full comparison.

**Q2. Can the full-residual solver materially reduce the residual?** Yes in the
bounded smoke: ten Adam steps reduced validation normalized MAE by 36.1% to
`0.171126`. This is a real validation-bank reduction, not merely training loss,
but it is only 1.56x and does not satisfy the partial-success threshold.

**Q3. What is the lowest validation residual; is there train-only overfit?** The
lowest observed value is `0.171126`. It is a held-out fixed validation-bank
measurement. The short run shows no validation rebound at its final checkpoint,
but is too short to characterize overfitting.

**Q4. How much does a greedy policy refresh change after the solve?** BP grid
choices barely move for most rows (both p90 values are zero), while the internal
investment selector changes on 13.26% of i-grid decisions and default status on
0.98% of rows. The short root is therefore not policy-self-consistent yet.

**Q5. Which interpretation is currently supported?** The bounded smoke is
closest to **C: inconclusive**. It establishes correct plumbing and a favorable
initial root-optimization direction, but it neither reaches a low-residual root
nor runs the configured LBFGS phase. A/B require the full K=5 run; K=9 should be
run only if K=5 gives a clear reduction.

## Verification and commands

Tests:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider \
  tests/test_frozen_policy_root_solve.py \
  tests/test_policy_value_stage_separation.py
# 120 passed
```

Completed local bounded smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 python analysis/test_frozen_policy_root_solve.py \
  --run /Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/artifacts/inner_fp_block_isolation_20261015/p_only \
  --source-run /Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/artifacts/inner_fp_block_isolation_20261015/p_only/source \
  --source-checkpoint /Users/ballinliu/Desktop/PHD/Project1/DL_AP_Tensor/artifacts/inner_fp_block_isolation_20261015/p_only/source/checkpoints/ep1_combined.pt \
  --sweep 5 --output-dir cachedir/frozen_policy_root_solve_20261015/smoke_k5 \
  --device cpu --train-limit 2048 --validation-limit 2048 --batch-size 64 \
  --adam-steps 10 --eval-every 5 --lbfgs-max-iter 0 \
  --picard-outer 1 --picard-fit-steps 32 --run-optimization
```

Server full K=5 command (run from the checked-out branch):

```bash
python analysis/test_frozen_policy_root_solve.py \
  --run /WORK/PUBLIC/zhuyingz_work/LiuHao/cachedir/inner_fp_block_p_only_ep1_10s_20261014_210200 \
  --source-run /WORK/PUBLIC/zhuyingz_work/LiuHao/cachedir/normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_investment_compare_fresh100_20261014_144528 \
  --source-checkpoint /WORK/PUBLIC/zhuyingz_work/LiuHao/cachedir/normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_investment_compare_fresh100_20261014_144528/checkpoints/ep1_combined.pt \
  --sweep 5 --output-dir cachedir/frozen_policy_root_solve_20261015/full_k5 \
  --device cuda:0 --train-limit 8192 --validation-limit 2048 --batch-size 64 \
  --adam-steps 300 --adam-lr 1e-4 --eval-every 10 \
  --lbfgs-max-iter 200 --lbfgs-history-size 50 \
  --picard-outer 5 --picard-fit-steps 128 --run-optimization
```

If and only if K=5 is clearly successful, repeat with `--sweep 9` and a distinct
`--output-dir`.

## Artifacts

- `cachedir/frozen_policy_root_solve_20261015/smoke_k5/root_solve_summary.csv`
- `cachedir/frozen_policy_root_solve_20261015/smoke_k5/root_solve_history.csv`
- `cachedir/frozen_policy_root_solve_20261015/smoke_k5/picard_history.csv`
- `cachedir/frozen_policy_root_solve_20261015/smoke_k5/policy_change_after_root.csv`
- `cachedir/frozen_policy_root_solve_20261015/smoke_k5/initial_parity.csv`
- `cachedir/frozen_policy_root_solve_20261015/smoke_k5/metadata.json`
- `cachedir/frozen_policy_root_solve_20261015/smoke_k5/checkpoints/`
- `cachedir/frozen_policy_root_solve_20261015/smoke_k5/figures/`
