# Diff summary

New source certifier/learning-state codec, continuation sampler/preparer,
review-gated eight-Outer runner, fixed-input forward probes and histories,
Slurm plan, non-optimization tests, and review docs. Shared modifications are
limited to an optional initial-model injection in setup, optional full-learning
source hook in ProductionBackend, conditional run identity in cache provenance,
and dynamic Outer ticks in existing plots. No training equations or loss changed.

Implementation change set (pending worktree or final committed diff):
```text
 M README.md
 M analysis/finite_direct_outer_metrics.py
 M analysis/run_fc1_self_consistent_macro.py
 M analysis/run_full_nested_finite_direct_outer4.py
 M data/finite_direct_runtime.py
?? analysis/build_finite_direct_continuation_review.py
?? analysis/finite_direct_continuation_probes.py
?? analysis/finite_direct_continuation_state.py
?? analysis/prepare_finite_direct_continuation.py
?? analysis/run_finite_direct_large_continuation.py
?? data/finite_direct_continuation_sampling.py
?? review/finite_direct_large_continuation_outer4_11_20261020/
?? slurm/run_finite_direct_large_continuation_outer4_11_a800.slurm
?? tests/test_finite_direct_large_continuation.py
```
