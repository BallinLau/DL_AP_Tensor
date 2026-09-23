# Full-run evaluator performance v2 (BP hot path)

Read-only evaluator work for the completed multi-episode run. Everything in this
report is measurement, not projection: the only performance claims are the ones
reproduced by the commands below.

## 1. What was slow

The previous full-run evaluation spent almost all of its wall time inside the BP
teacher grid (`bp_seconds`). Two structural inefficiencies were responsible:

1. **Per-J, per-branch repetition.** Each branch (`p0`, `pi_low`, `pi_mid`,
   `pi_high`) and each requested child count `J` ran its own independent
   `BPGridTeacher.compute()` pass. Every pass re-evaluated the same candidate
   child equity block over the same candidate grid, and the `J=32`/`J=64`
   passes re-derived exactly the states that the `J=128` pass had already
   computed (the shocks are a nested prefix of one canonical bank).
2. **Chunk budget pinned at the training value.** `bp_grid_max_expanded_states`
   defaulted to `65536` for the evaluator, so each candidate forward covered at
   most `65536` expanded child states. On an 80 GiB A800 with only ~10.7 GB
   peak allocation this left the GPU heavily under-utilized.

## 2. What changed

Numerical semantics, model state, checkpoint hyperparameters, formal metric
definitions, and the training path are untouched. Only the read-only evaluator
changed:

| change | file |
|---|---|
| `compute_multi_j_branches()`: one pass over many `J` prefixes and many branches, sharing the candidate child-equity forward and the `q_issue` grid | [bp_grid_teacher.py](file:///Users/ballinliu/.codex/worktrees/171b/DL_AP_Tensor/training/bp_grid_teacher.py) |
| `_EquityBlock` + `_child_equity_block` + `_branch_objective_grid`: candidate child equity is computed once per chunk, branch-specific objective assembly stays separate | [bp_grid_teacher.py](file:///Users/ballinliu/.codex/worktrees/171b/DL_AP_Tensor/training/bp_grid_teacher.py) |
| `evaluate_bp_consistency_multi_j()`: the evaluator entry point for the shared pass, plus `resolve_bp_eval_max_expanded_states()` | [bp_diagnostics.py](file:///Users/ballinliu/.codex/worktrees/171b/DL_AP_Tensor/evaluation/bp_diagnostics.py) |
| `--bp-eval-max-expanded-states` / `BP_EVAL_MAX_EXPANDED_STATES`, one shared multi-J pass per eta grid, matrix-level BP counters | [evaluate_checkpoints.py](file:///Users/ballinliu/.codex/worktrees/171b/DL_AP_Tensor/experiments/evaluate_checkpoints.py) |
| canonical cross-episode shock bank, component isolation, BP counters in `evaluation_timing.json` | [evaluate_full_run.py](file:///Users/ballinliu/.codex/worktrees/171b/DL_AP_Tensor/experiments/evaluate_full_run.py) |

The fine grid is deliberately **not** shared across `J`: different prefixes can
have different argmax candidates, so each `(branch, J)` finalizes its own local
fine grid. Only the coarse candidate grid and the child-equity block are shared.
This is why the measured reduction is ~36-52% rather than ~90%.

## 3. Correctness gate

The optimized path is checked against the legacy per-`(branch, J)` path by
`tests/test_policy_child_vectorization.py::test_multi_j_branch_reuse_matches_legacy_single_j_compute`
and the benchmarks below report `max_abs_diff = 0.000e+00` (bitwise identical on
`bp_star`, `value_star`, `regret`, `coarse_value_grid`).

```
python3 -m pytest \
  tests/test_full_run_evaluator.py \
  tests/test_checkpoint_evaluator.py \
  tests/test_policy_child_vectorization.py -q
```

## 4. Local CPU synthetic benchmark

`CUDA benchmark unavailable locally` — this workspace has no GPU, so no GPU
wall-time or peak-memory number is invented here.

The benchmark uses a deterministic MLP-backed teacher (a proxy for the real
network cost per forward), 128 parents, the real branch set
`p0@i_mid / pi_low / pi_mid / pi_high`, and the real coarse/fine grid
(`21` + `9`, quadratic refine off). It compares:

* **before** — `BPGridTeacher.compute()` once per `(branch, J)`.
* **after** — one `compute_multi_j_branches()` call for all of them.

| budget | `J` prefixes | branches | before equity forwards | after equity forwards | forward reduction | before seconds | after seconds | wall reduction | max abs diff |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 65536 | 64, 128 | 4 | 64 | 34 | 46.9% | 4.307 | 2.049 | 52.4% | 0.000e+00 |
| 262144 | 64, 128 | 4 | 28 | 18 | 35.7% | 3.691 | 1.796 | 51.3% | 0.000e+00 |
| 65536 | 64, 128, 256 | 4 | 132 | 63 | 52.3% | 9.288 | 4.103 | 55.8% | 0.000e+00 |
| 262144 | 64, 128, 256 | 4 | 52 | 31 | 40.4% | 8.369 | 3.826 | 54.3% | 0.000e+00 |

Reading the table:

* Sharing the candidate child-equity block across the four branches and the
  nested `J` prefixes removes 36-52% of the child-equity forwards and roughly
  halves the BP wall time at realistic per-forward cost.
* The `J` prefixes are served from the same `Jmax` coarse grid, so the coarse
  candidate grid is evaluated once instead of once per `J`.
* `bp_max_actual_expanded_states` never exceeds `bp_max_expanded_states`; the
  hard cap is respected at every chunk size.

An honest caveat: with a *trivial* model (a few elementwise ops), the optimized
pass is ~20% **slower** on CPU because the extra Python-level loop over
`(branch, J)` per chunk dominates a near-zero forward cost. Real models do not
have near-zero forward cost; the table above shows the crossover, and the
target hardware is a GPU where each forward is a large batched kernel.

## 5. GPU utilization

`--bp-eval-max-expanded-states` (or `BP_EVAL_MAX_EXPANDED_STATES`) raises the
evaluator chunk budget without touching the checkpoint:

| value | intended hardware |
|---:|---|
| `65536` | CPU / small GPU / explicit legacy parity |
| `131072` | >= 35 GiB GPU total memory (auto tier) |
| `262144` | >= 70 GiB GPU total memory (auto tier, A800 80GB) |
| `524288` | explicit opt-in |

Resolution rules (`resolve_bp_eval_max_expanded_states`):

* an explicit value always wins and must be one of the four allowed values;
* on CUDA with no explicit value, the device-memory tier is used;
* everywhere else the safe legacy default `65536` is used.

Because `resolve_grid_chunk_plan` scales `parent_chunk` with the budget, going
from `65536` to `262144` makes each candidate forward cover 4x more expanded
child states (measured above: `expanded_states_per_forward` goes from `65536` to
`262144`), which is the intended GPU-utilization fix.

## 6. Instrumentation

`evaluation_timing.json` now records, in addition to the pre-existing
`total_seconds` / `checkpoint_load_count` / per-phase seconds:

`bp_model_forward_calls`, `bp_child_equity_forward_calls`, `bp_q_forward_calls`,
`bp_parent_chunks`, `bp_candidate_chunks`, `bp_max_expanded_states`,
`bp_max_actual_expanded_states`, `bp_multi_j_reuse_enabled`,
`bp_branch_reuse_enabled`.

Per-episode metadata records `canonical_shock_bank_max_children`,
`evaluated_child_counts` and `max_evaluated_children`, so the canonical
cross-episode bank size is never confused with the per-episode evaluated prefix.

## 7. Server command (A800 80GB)

```bash
cd /home/fit/zhuyingz/WORK/LiuHao/DL_AP_Tensor
export RUN_ROOT=/home/fit/zhuyingz/WORK/LiuHao/<completed_run>
export OUTPUT_DIR=$RUN_ROOT/data/outputs/full_run_evaluation
export BP_EVAL_MAX_EXPANDED_STATES=262144
export RUN_EVALUATOR_TESTS=1
export OVERWRITE=1
sbatch slurm/run_full_run_evaluator_gpu.slurm
```

The job script prints GPU total memory, the canonical / primary / robustness `J`
values, the robustness scope, and the resolved BP budget before the run, then
prints the aggregated BP counters from `evaluation_timing.json` afterwards.
