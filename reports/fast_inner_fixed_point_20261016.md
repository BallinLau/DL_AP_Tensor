# Fast fixed-macro P-Q inner fixed-point diagnostic

Date: 2026-10-16  
Branch: `exp/fast-pq-inner-fixed-point`  
Starting HEAD: `da0b3769bc220fb9bea6915452ca859167ac2113`  
GPU: RTX 4090, 24 GiB  
Train/validation rows: 8192 / 2048

## Executive conclusion

The fixed-macro inner firm problem is numerically promising under the tested operator. The P root closure was accelerated by about 20.6x without changing its Bellman definition, the production-parity Q full-residual solver reached the diagnostic residual gate, and the bounded P-Q block solver reached the full coupled convergence gate at outer 5 without damping, a two-cycle, a macro update, or a bank mutation.

This is evidence for a numerically accessible fixed-macro inner fixed point. It is not a proof of a global equilibrium, uniqueness, or stability after reintroducing macro updates.

## Scope and operator semantics

- P uses the existing production P operator, including its explicitly labelled `LEGACY_M1` continuation convention.
- Q uses corrected child-transition M, detached and clamped exactly as production Q: `[0.5, 1.5]` in train and evaluation.
- Exact eta expansion/collapse is unchanged.
- P, Q, recovery, default, investment, transition, network architecture, and macro economics are unchanged.
- Macro models, ordered parent bank, and child shock bank are immutable across coupled outers.
- The invalid preliminary Q smoke that squared each stochastic branch separately is retained only for audit under `q_full_corrected/`; it is excluded from every conclusion below. All reported Q results use the production branch-expectation target and pass an independent production snapshot parity check.

## Stage A: P speed and parity

The legacy child continuation performed 23 value-network calls per batch: one parent call plus 11 i-grid calls for each of P0 and PI. The fast path caches expanded child states, concatenates P0/PI branches, and uses two calls per batch. Validation is outside strong-Wolfe closures and LBFGS is evaluated in chunks.

### Numerical parity

| Check | Max/absolute error |
|---|---:|
| P0 residual | 9.54e-7 |
| PI residual | 4.77e-7 |
| Root loss | 1.49e-8 |
| Gradient max | 7.15e-7 |
| Gradient L2 | 2.31e-6 |
| Cached vs noncached P0 | 0 |
| Cached vs noncached PI | 0 |

Source: `cachedir/fast_inner_fixed_point_20261016/p_speed/full/p_numerical_parity.csv`.

### Performance

| Stage | Implementation | Batch | sec/closure | sec/optimizer block | peak VRAM MiB | best residual | reduction/min |
|---|---|---:|---:|---:|---:|---:|---:|
| P benchmark | legacy | 64 | 7.5689 | 158.95 | 462.8 | 0.06094 | 0.0781 |
| P benchmark | vectorized | 512 | 0.3783 | 7.94 | 1852.9 | 0.06145 | 1.5585 |
| P benchmark | vectorized | 1024 | 0.3676 | 7.72 | 3428.3 | 0.06187 | 1.6006 |
| P benchmark | vectorized | 4096 | 0.3875 | 8.14 | 12882.9 | 0.06163 | 1.5203 |
| Q full regime | full residual | 1024 | 0.1501 | 5.71 | 229.6 | 0.01718 | 0.5535 |
| Coupled P block | same fast P engine | 1024 | 0.3676 benchmark | 14.0–33.0 | 3470.9–3544.3 | 0.01571–0.01878 after P | n/a |
| Coupled Q block | full residual | 1024 | 0.1476 outer-1 | 5.00–5.21 | 497.4–544.3 | 0.00972–0.01091 | n/a |

Batch 8192 OOMed after trying to allocate another 1.38 GiB with 23.22 GiB already in use. Batch 2048/4096 fit, but neither improved the 20-step wall clock. Batch 1024 is the fastest tested safe choice; 512 is the lower-memory fallback.

The primary bottleneck was repeated i-grid network dispatch and batch-loop overhead, not raw arithmetic or lack of VRAM.

Raw files:

- `p_speed/full/p_speed_profile.csv`
- `p_speed/full/p_lbfgs_benchmark.csv`
- `p_speed/large_batch/p_speed_profile.csv`
- `p_speed/large_batch/p_lbfgs_benchmark.csv`

## Stage B: Q full-residual solver

### Production parity and invariants

For 512 validation rows covering 27 zero, 302 default, and 183 survival parents:

| Metric | Error |
|---|---:|
| Target mean absolute | 1.37e-8 |
| Target p90 absolute | 5.96e-8 |
| Target maximum absolute | 9.54e-7 |
| Scale mean absolute | 1.47e-8 |
| Scale maximum absolute | 9.54e-7 |

Child-Q target gradients are nonzero in the full-residual path and absent in the detached negative control. Only `q_encoder` and `q_head` update. P/BP hashes are unchanged. Q M source is `CORRECTED_CHILD_CLAMPED_0.5_1.5`; min/mean/max are `0.5 / 0.8965 / 1.5`, with hash `9372397fee7c37284fea4a8e2231a4f536b4d2ca1704f53e6e87548e7a44be5f`.

### A/B results

Survival-only diagnostic:

| Method | Initial | Final | Seconds | Q p90 | Q max |
|---|---:|---:|---:|---:|---:|
| Frozen-target Picard | 0.07764 | 0.02370 | 2.93 | 0.7418 | 4.7971 |
| Full-residual Adam+LBFGS | 0.07764 | 0.02118 | 32.06 | 0.6802 | 1.3396 |

Full production zero/default/survival regimes:

| Method | Initial | Adam | Final | Seconds | Peak MiB | Q p90 | Q max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen-target Picard | 0.06983 | n/a | 0.01568 | 3.33 | 109.6 | 0.5829 | 1.8960 |
| Full-residual Adam+LBFGS | 0.06983 | 0.02244 | 0.01718 | 5.71 | 229.6 | 0.5548 | 1.2885 |

Both methods converge in this fixed-P experiment. Frozen-target is faster and has a slightly lower normalized residual; full residual has a lower Q tail maximum and proves that a differentiable current-network Q root path is viable. The evidence does not support the claim that only full-residual can solve Q. It supports a conditional Q fixed point with a small residual floor around 0.016–0.017 under this network/bank.

Full-residual Q trajectory:

`0.06983 → 0.06035 → 0.04881 → 0.03869 → 0.03390 → 0.02998 → 0.02716 → 0.02553 → 0.02448 → 0.02366 → 0.02299 → 0.02244 → 0.01718`.

## Stage C: fixed-macro coupled pilot

Initialization is the full-regime Q root checkpoint. The initial P residual jumps to 0.12415 when the greedy P operator is refreshed under the newly solved Q. This is direct evidence of material P-Q coupling.

### Coupled trajectory

| Outer | P residual | Q residual | P0 move p90 | PI move p90 | Q move p90 | invest flip | default flip | BP0/BPI p90 | P sec | Q sec | total sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.12415 | 0.01718 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 |
| 1 | 0.02321 | 0.01091 | 0.35588 | 0.41509 | 0.02858 | 6.574% | 19.287% | 0.0875 / 0.0875 | 32.96 | 5.08 | 39.24 |
| 2 | 0.01852 | 0.01075 | 0.02774 | 0.02801 | 0.01434 | 1.509% | 1.318% | 0.0500 / 0.0500 | 14.62 | 5.11 | 20.86 |
| 3 | 0.01761 | 0.01010 | 0.01505 | 0.01648 | 0.00886 | 0.648% | 0.928% | 0.0250 / 0.0250 | 14.34 | 5.21 | 20.71 |
| 4 | 0.01655 | 0.00997 | 0.01042 | 0.01155 | 0.00642 | 0.484% | 0.635% | 0.0125 / 0.0125 | 14.40 | 5.18 | 20.77 |
| 5 | 0.01609 | 0.00972 | 0.00801 | 0.00895 | 0.00431 | 0.888% | 0.439% | 0.0125 / 0.0125 | 14.00 | 5.00 | 20.20 |

The outer-5 row passes every diagnostic gate. No P/Q/policy two-cycle was detected. Macro hash remained `32e419...da0`, and parent/child bank hashes remained `ec69ae...9de` / `baf75f...1c18`.

P residual trajectory: `0.12415 → 0.02321 → 0.01852 → 0.01761 → 0.01655 → 0.01609`.

Q residual trajectory: `0.01718 → 0.01091 → 0.01075 → 0.01010 → 0.00997 → 0.00972`.

PI-active share: `17.43% → 12.64% → 13.12% → 12.91% → 12.86% → 12.29%`.

Default share: `57.42% → 75.63% → 74.90% → 75.15% → 74.71% → 74.95%`.

## Figures and raw bundle

Local bundle root:

`cachedir/fast_inner_fixed_point_20261016/`

Key artifacts:

- `coupled_fixed_macro_profiled/coupled_outer_summary.csv`
- `coupled_fixed_macro_profiled/metadata.json`
- `coupled_fixed_macro_profiled/figures/coupled_outer_summary.png`
- `coupled_fixed_macro_profiled/figures/levels/` — 72 outer-level P0/PI/Q/Phat/BP0/BPI 3D+heatmap figures, eta separated.
- `coupled_fixed_macro_profiled/figures/transitions/` — 80 transition heatmaps, eta separated.
- `q_full_regime_profiled/q_root_summary.csv`
- `q_full_regime_profiled/q_root_history.csv`
- `q_full_regime_profiled/q_initial_parity.csv`
- `historical_comparison/historical_picard_vs_coupled.csv`
- `historical_comparison/historical_picard_vs_coupled.png`

The historical artifact is P-only Picard with Q frozen and omits P/Q magnitude fields. The comparison figure therefore labels the algorithms as non-equivalent iteration axes and leaves unavailable historical magnitudes blank instead of fabricating them.

## Answers to Q1–Q14

1. **P speedup:** 20.6x by 20-step wall clock (158.95 s to 7.72 s), about 20.6x per closure.
2. **Main bottleneck:** repeated 11-point i-grid forward dispatch for both branches plus many small batch loops.
3. **Vectorization parity:** yes; forward errors are below 1e-6 and gradient max error is 7.15e-7.
4. **Fastest batch:** 1024. Larger batches fit through 4096 but are slower; 8192 OOMs.
5. **Frozen-target Q non-damping:** no. It converges in both survival-only and full-regime tests.
6. **Full residual better:** not uniformly on final residual/time. It is slightly better in survival-only and slightly worse in full-regime residual, but produces a substantially smaller Q maximum and supplies the required differentiable root path.
7. **Conditional Q fixed point:** yes, numerically promising; full-regime residual reaches 0.01718 and frozen-target reaches 0.01568.
8. **Repeated P block reduction:** yes. Each coupled P solve reaches its inexact target; post-Q P residual declines to 0.01609.
9. **Repeated Q block reduction:** yes. Each Q block reaches its target and the post-block residual declines to 0.00972.
10. **Joint trend within five outers:** yes. Residuals and all p90 movement measures decrease enough to pass the outer-5 gate.
11. **Policy movement:** yes, investment/default flips fall below 1%; BP p90 falls to 0.0125.
12. **Oscillation/two-cycle:** none detected in P, Q, investment, default, BP0, or BPI.
13. **Classification:** **A — clearly promising** for a fixed-macro inner firm fixed point under the tested operator. This classification does not extend to the outer macro equilibrium.
14. **Next bottleneck:** test robustness across macro anchors/seeds and then integrate this bounded inner block into an outer macro update while preserving the same hash/parity diagnostics. The remaining numerical issue is P tail movement/max values, not mean/p90 convergence.
