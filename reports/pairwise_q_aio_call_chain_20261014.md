# Q-survival pairwise AiO call-chain audit

## Scope

This patch changes only the estimator used by the current formal Q-survival
training objective. The economic Bellman equation, target timing, fixed-point
evaluator, shock bank, exact-eta expectation, P/Q networks, and default rule
are unchanged.

## Training call chain

1. `training/episode.py:6115-6117` expands each continuous child shock into
   eta=0/1 states when exact-eta integration is enabled.
2. `training/episode.py:6188-6207` computes the economic Q components, then
   calls `_collapse_policy_eta_pairs` separately for every continuous shock.
3. `training/episode.py:6208-6237` selects the collapsed Q residuals and keeps
   the existing normalization and fixed sweep-start Q target semantics.
4. `training/episode.py:6239-6245` calls
   `QLoss.compute_survival_parent_objective` with the J collapsed continuous
   shock residuals.
5. `losses/q_loss.py:611-639` is the only production call site switched to the
   pairwise-U components.
6. `losses/utils.py:144-191` computes
   `stable=mean(r_j^2)` and
   `U_J=((sum r_j)^2-sum(r_j^2))/(J(J-1))` in O(J), returning
   `(1-w)stable+w|U_J|`.
7. `training/episode.py:6374-6382` records the estimator mode, J, pair count,
   stable, signed U, absolute U, and total. `q_main` and `q_aio_total_mean`
   are both the mean of the same per-parent total tensor.

The legacy product helper remains unchanged at `losses/utils.py:107-141`.
Its P0/PI, Q0/Q-default, historical Q, and SDF-related callers were not
redirected.

## Evaluator isolation

`training/inner_fixed_point.py:536-661` still evaluates Q survival by taking
the signed child residual matrix, averaging across child shocks at line 613,
then taking the absolute conditional mean. It never imports or calls the
pairwise-U helper. Its definition string remains
`current_online_fixed_bank_operator_residual_v1`.

## Bellman gate configuration

The source hyperparameters and `config/hyperparams.py` intentionally use
`None` to mean report-only for the conditional Bellman thresholds. No formal
numeric defaults exist, so this patch does not invent them. The inner solver
now reports `bellman_gate_configured=false`, `bellman_passed=null`, and
`convergence_gate_unconfigured` while continuing all requested sweeps. The
optional CLI threshold pair at `experiments/run_inner_fixed_point.py:70-71`
and `:176-180` must be supplied together if a future experiment wants a
configured convergence verdict.

## Regression invariants

1. Q0 objective unchanged.
2. Q-default recovery objective unchanged.
3. P-stage partition/hash isolation unchanged.
4. BP parameters remain frozen in the inner solver.
5. SDF/FC1 remains frozen and hash-checked.
6. Parent/child bank fingerprints remain immutable.
7. Formal child count remains J=16.
8. Exact-eta expansion and collapse order unchanged.
9. Q target remains the sweep-start fixed snapshot.
10. Gauss-Seidel P then Q order unchanged.
11. Fixed-point evaluator definition and residual calculation unchanged.
12. Outer simulation remains disabled.
13. Damping remains 1.0.

## Resume-only infrastructure correction

The 5-to-10 continuation initially exposed a pre-existing device mismatch:
`torch.load(..., map_location=cuda)` moved saved diagnostic comparison surfaces
to CUDA, while fresh evaluator surfaces are intentionally returned on CPU.
`InnerFixedPointSolver.load_resume` now moves only the saved diagnostic
values/boundaries/policies back to CPU. Model weights still load on the solver
device. This does not alter any fresh-run or training calculation.
