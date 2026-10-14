# Firm-side inner fixed-point call-chain audit

## Scope

The independent entry point is `experiments/run_inner_fixed_point.py:155`. It never calls `Episode.run`, simulation, entry, State Bank refresh, SDF training, FC1 training, FC2 training, or BP distillation.

## Initialization and frozen inputs

1. `experiments/run_inner_fixed_point.py:209` invokes the existing strict combined-checkpoint loader. The runner separately compares the checkpoint's embedded hyperparameters, economic config, and policy-value model spec to the three source metadata files before constructing models.
2. `experiments/run_inner_fixed_point.py:234-246` constructs deterministic train and validation parent banks with different fixed seeds. `training/inner_fixed_point.py:176` implements joint macro-state sampling plus broad micro-state coverage.
3. `experiments/run_inner_fixed_point.py:249-263` constructs the fixed one-step child banks. `training/inner_fixed_point.py:287` uses the frozen `sdf_fc1` and hashes it before and after child creation.
4. `InnerFixedPointSolver.run` starts at `training/inner_fixed_point.py:894`; it freezes `sdf_fc1` and FC2 and checks the SDF and bank fingerprints at sweep boundaries.

## P update

At `training/inner_fixed_point.py:942`, one frozen sweep-start snapshot is passed to `Episode._run_policy_value_evaluation_stage` (`training/episode.py:8302`). That existing stage:

- creates BPGridTeacher value targets at `training/episode.py:8946-8975`;
- calls the canonical `BPGridTeacher.compute_value_target` implementation (`training/bp_grid_teacher.py:846`);
- integrates future eta through `Episode._expand_policy_expectation_children` (`training/episode.py:1245`);
- updates only the P partition under the existing P-only train scope.

The solver hashes P, Q, BP, and SDF before and after this stage and fails if Q, BP, or SDF changes. It never invokes the BP-distillation stage.

## Q update

After P fitting, the solver freezes the new P snapshot and invokes `Episode._run_q_regime_training` at `training/inner_fixed_point.py:968`. The new `fixed_q_target_model` argument is defined at `training/episode.py:11395-11406`. When supplied, the same sweep-start Q snapshot is selected at `training/episode.py:11456` for Q0, QD, QS, and optional polish; its hash is checked again at `training/episode.py:11582-11585`.

The recursive survival equation remains the existing implementation at `training/episode.py:6065`. Current eta controls the debt transition through the existing `apply_refinancing_policy` function (`utils/firm_transition.py:134`); child eta is integrated as a future shock and does not gate the current transition.

The solver hashes the P, BP, and SDF partitions after Q training and fails if any changed. Thus each sweep implements the requested ordering:

`P_(k+1) = T_P(P_k,Q_k)`, then `Q_(k+1) = T_Q(P_(k+1),Q_k)`.

## Current residual and convergence gates

The dedicated evaluator starts at `training/inner_fixed_point.py:536`. It constructs BPGridTeacher with the current model as both equity and Q target (`training/inner_fixed_point.py:552`), computes fresh P0/PI backups (`training/inner_fixed_point.py:566-571`), and uses the existing current-model Q signed-residual helper. Hence sweep-k metrics evaluate `F_k - T(F_k)`, not the stale targets used to train sweep k.

The solver separately records current operator residuals, successive function changes, default-mask/cutoff changes, and implicit grid-policy changes. Passing requires all four gates for the configured number of consecutive sweeps. Module/bank hashes and per-stage checkpoints make a violation fail closed rather than appear as convergence.
