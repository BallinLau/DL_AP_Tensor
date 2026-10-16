# Neural policy iteration diagnostic (2026-10-15)

## Scope

This analysis-only experiment starts from historical P-only sweep 5, keeps Q,
BP-policy parameters, macro inputs, parent/validation banks, child shock banks,
and the legacy `M=1` operator fixed, and alternates hard greedy policy refreshes
with full-residual frozen-policy P solves. It does not modify the production
economic model.

Outer 0 reuses the independently verified K5 root at
`cachedir/frozen_policy_root_solve_20261015/full_k5`; outers 1--4 warm-start from
the preceding root and use 100 Adam steps followed by at most 200 strong-Wolfe
LBFGS iterations.

## Provenance and invariants

- Experiment implementation branch: `exp/neural-policy-iteration`.
- Implementation base: `942ac6561f881dd575bfa4695ea6ba10d7d901c7`.
- Source checkpoint recorded by the run: `ep1_combined.pt`, source commit
  `959984df64918654aa69b7f5596301df062f8030`.
- Q parameter hash stayed fixed:
  `04d10e09b9db1ac4867c36b22642946b9e47ca8bf7a00fe96ba6d4d18cd5946d`.
- BP/policy parameter hash stayed fixed:
  `85b180644922138b968e0aaa06bf80ee396ecedf7464a948cc9ca005f4e4bf2c`.
- Train, validation, and child-bank fingerprints are recorded in `metadata.json`.
- `legacy_m1_invariant=true` for every outer.
- The stopped, incomplete K9 experiment was not used.

## Outer trajectory

| outer | initial frozen | final frozen | full greedy | investment flip | default flip | PI active old->new | BP p90 (0/I) | P0/PI movement L2 | seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.267793 | 0.023381 | 0.083387 | 0.327947 | 0.213867 | 0.499112 -> 0.193714 | 0.2250 / 0.2250 | n/a | 20* |
| 1 | 0.083387 | 0.021512 | 0.025649 | 0.038130 | 0.064453 | 0.193714 -> 0.208319 | 0.0500 / 0.0500 | 0.091786 / 0.105778 | 1558 |
| 2 | 0.025649 | 0.019370 | 0.023089 | 0.032582 | 0.041992 | 0.208319 -> 0.188699 | 0.0125 / 0.0125 | 0.073135 / 0.084449 | 1705 |
| 3 | 0.023089 | 0.019084 | 0.020285 | 0.026190 | 0.032715 | 0.188699 -> 0.183194 | 0.0125 / 0.0125 | 0.059222 / 0.067607 | 1673 |
| 4 | 0.020285 | 0.018200 | 0.018708 | 0.016602 | 0.012207 | 0.183194 -> 0.174316 | 0.00625 / 0.00625 | 0.031084 / 0.035852 | 1694 |

`*` Outer 0 only loads and verifies the existing K5 solution. The original K5
solve took 1674.95 seconds; its 20-second row is therefore not a solver-speed
comparison.

## Interpretation

The full-greedy residual falls monotonically from `0.083387` after the reused
K5 root to `0.018708`. Function movement, policy flips, and BP movement also
decline. No outer is close to its two-step predecessor while far from the
one-step predecessor, so the implemented two-cycle detector remains false.

This supports the existence of a numerically approachable **conditional P
fixed point given fixed Q and macro inputs**. It does not establish a joint P-Q
fixed point or a full equilibrium. The final investment and default flips are
still above the 1% diagnostic gate, and value maxima continue to rise, so more
P-only outer iterations would be less informative than testing Q with a
matched full-residual solver.

## Artifacts

Raw artifacts are in
`cachedir/neural_policy_iteration_20261015/k5_outer5/`:

- `policy_iteration_summary.csv`
- `outer_transition_metrics.csv`
- `historical_vs_policy_iteration.csv`
- `metadata.json` and `metadata_outer_*.json`
- `checkpoints/outer_*_{initial,root_best}.pt`
- `optimizer_history_outer_*.csv`
- `figures/01_*.png` through `figures/09_*.png`
- `figures/transitions/`, `figures/state_maps/`, `figures/surfaces/`, and
  `figures/boundaries/`

## Direct answers

1. Outer 0 is the verified full K5 root, reused rather than recomputed.
2. Five outer states (0--4) completed.
3. The conditional frozen-policy residual is repeatedly reducible below 0.024.
4. The full-greedy residual declines monotonically to 0.018708.
5. Investment flips decline from 32.79% to 1.66%.
6. Default/survival flips decline from 21.39% to 1.22%.
7. PI-active share moves toward a narrower band and ends at 17.43%.
8. P0 and PI function movement decline materially but are not yet zero.
9. BP p90 movement declines to 0.00625 for both branches.
10. No two-cycle is detected.
11. The evidence supports a conditional P fixed point, not yet a joint P-Q
    fixed point; Q must next be tested with train/eval-matched M semantics and a
    differentiable child-Q full-residual path.
