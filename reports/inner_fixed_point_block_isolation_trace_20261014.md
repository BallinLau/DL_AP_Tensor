# Inner fixed-point block-isolation call-chain audit

## Overview

This audit fixes the execution and optimization boundaries used by the three-way `p_only`, `q_only`, and `coupled` experiment. The implementation adds branching and diagnostics only; the existing P and Q training operators remain the optimization sinks.

## Entry Point

- `experiments/run_inner_fixed_point.py:160-325` (`main`) validates the Episode-1 combined checkpoint, restores the source hyperparameters/economic configuration, creates deterministic parent and child banks, records provenance, and calls `InnerFixedPointSolver.run`.
- `experiments/run_inner_fixed_point.py:299-322` passes `fp_mode` into one shared solver. This is a direct call edge.

## Data Input Pipeline

- `experiments/run_inner_fixed_point.py:240-273` reads `ep1_stage_modeb.pkl` and `ep1_stage_modeb_macro.pkl`, builds train/validation parent banks, then builds fixed child banks from frozen SDF/FC1.
- `training/inner_fixed_point.py:347-367` (`make_batches`) preserves parent-bank row order and attaches the fixed child tensors and source indices.
- `training/inner_fixed_point.py:584-730` (`evaluate_inner_operator_snapshot`) uses those same validation parents/children for every start, post-P, and post-Q evaluation.

## High-Level Call Graph

```text
run_inner_fixed_point.main
  -> InnerFixedPointSolver.run
     -> [p_only or coupled] Episode._run_policy_value_evaluation_stage
        -> Episode._compute_cached_pq_loss(include_q=False)
        -> total.backward -> P optimizer.step
     -> [q_only or coupled] Episode._run_q_regime_training
        -> Episode._run_q_regime_phase
        -> Episode._compute_q_survival_bellman_loss
        -> QLoss.compute_main_components
        -> QLoss.compute_survival_parent_objective
        -> loss.backward -> Q optimizer.step
     -> evaluate_inner_operator_snapshot(start/post-P/post-Q)
```

All arrows are direct calls except the Q loss calls reached through phase dispatch, which are nested calls.

## Detailed Function Trace

### Solver orchestration

- `training/inner_fixed_point.py:1186-1514` (`InnerFixedPointSolver.run`) creates `teacher_k=(P_k,Q_k)`, optionally executes P, snapshots `(P_{k+1},Q_k)`, optionally executes Q, and evaluates all three states.
- The `q_only` branch skips `_run_policy_value_evaluation_stage`; the `p_only` branch skips `_run_q_regime_training`. These are direct conditional edges, not zero-learning-rate approximations.
- Component hashes after each stage prove P/Q/BP/SDF ownership. Q-only additionally compares the full default-classification tensor and validation regime counts to their initial values.

### Fresh operator diagnostics

- `training/inner_fixed_point.py:584-730` computes row-aligned `P0`, `PI`, and `Q` outputs and targets.
- `training/bp_grid_teacher.py:846-889` (`BPGridTeacher.compute_value_target`) dispatches P0/PI targets while retaining the grid argmax for eligible parents and the forced branch for ineligible parents.
- `losses/q_loss.py:676-730` (`QLoss.compute_main_components` / `compute_main_residual`) defines each child-specific Q target and signed physical Bellman residual.
- `training/inner_fixed_point.py:683-689` averages signed Q residuals across children before taking absolute values, so the fixed-point diagnostic is the conditional expectation residual, not the AiO training objective.

## Loss Function Tracing

### P loss

- `training/episode.py:8309-8480` (`Episode._run_policy_value_evaluation_stage`) builds frozen targets once for the stage, calls `_compute_cached_pq_loss(..., include_q=False)`, and trains only the P parameter group.
- `training/episode.py:9234-9300` (`Episode._compute_cached_pq_loss`) consumes the cached P0/PI Bellman targets and returns the scalar P objective plus diagnostics.
- `training/bp_grid_teacher.py:846-940` provides the target `value_star` and `bp_star` used by that cache.

### Q loss

- `training/episode.py:11402-11661` (`Episode._run_q_regime_training`) runs Q0, QD, QS, and polish with one frozen P classifier and, for inner-FP sweeps, one fixed previous-sweep Q target.
- `training/episode.py:10783-11191` (`Episode._run_q_regime_phase`) constructs the regime-specific batches and dispatches the corresponding loss.
- `training/episode.py:6065-6472` (`Episode._compute_q_survival_bellman_loss`) constructs child Bellman residuals and records target/recursion diagnostics.
- `losses/q_loss.py:611-639` (`QLoss.compute_survival_parent_objective`) applies the J=16 pairwise U-statistic AiO objective to survival residuals. This objective is training-only and is not reused as the fixed-point convergence residual.

## Loss Aggregation and Optimization

- P optimization sink: `training/episode.py:8440-8477`. The scalar `total` is checked, passed to `total.backward()` at line 8451, clipped, then consumed by `optimizer.step()` at line 8476.
- Q optimization sink: `training/episode.py:11044-11056`. The scalar regime loss is checked, passed to `loss.backward()` at line 11047, clipped, then consumed by `optimizer.step()` at line 11056.
- No optimizer step exists on the frozen block in its isolation mode.

## Unresolved / Ambiguous Points

None in the static path used by this experiment. Runtime acceptance/reversion inside the established P/Q stages remains data-dependent and is therefore verified by smoke jobs and saved stage summaries.

## Final Summary

The implementation preserves the production Gauss-Seidel timing for `coupled`, freezes Q exactly for `p_only`, freezes P and its derived geometry exactly for `q_only`, and evaluates cross effects from row-aligned fresh operator targets on one immutable validation bank.

## Structured Trace Table

| Caller | Callee | Edge | Tensor role | Optimization ownership |
|---|---|---|---|---|
| `main` | `InnerFixedPointSolver.run` | direct call | source checkpoint and fixed banks | orchestration |
| `run` | `_run_policy_value_evaluation_stage` | direct call | `(P_k,Q_k) -> Y_k^P` | P only |
| P stage | `_compute_cached_pq_loss` | direct call | cached P0/PI targets -> scalar `total` | P only |
| `run` | `_run_q_regime_training` | direct call | `(P_{k+1},Q_k) -> Y_k^Q` | Q only |
| Q stage | `_run_q_regime_phase` | direct call | regime batches and fixed targets | Q only |
| Q phase | `_compute_q_survival_bellman_loss` | nested call | child signed residuals | Q only |
| Q loss | `compute_survival_parent_objective` | nested call | pairwise-U AiO scalar | Q only |
| `run` | `evaluate_inner_operator_snapshot` | direct call | fresh outputs, targets, regimes | no gradients/no optimizer |
