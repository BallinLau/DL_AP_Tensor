# Full-run evaluator vectorization benchmark

## Scope

This benchmark isolates the SDF K=32/64/128 robustness path. It uses 512
synthetic parents, a deterministic lightweight SDF/FC1 fixture, common seed
12345, and the same maximum shock bank K=128 in both implementations.

The before path invokes the backward-compatible single-K evaluator three times.
The after path invokes `evaluate_sdf_heldout_multi_k()` once and summarizes the
three nested prefixes from one max-K residual tensor.

## Local CPU result

Run on 2026-09-23 in the local CPU-only workspace:

| implementation | wall time (seconds) | SDF forward calls | transition builds | peak GPU memory |
|---|---:|---:|---:|---:|
| repeated single-K | 0.022393 | 3 | n/a | n/a (CUDA unavailable) |
| max-K nested prefix | 0.015626 | 1 | n/a | n/a (CUDA unavailable) |

Observed fixture speedup: `1.433x`.

For common plus on-distribution evaluation, the structural forward-count change
is from six calls per episode to two. Real GPU peak memory is intentionally not
invented from CPU measurements; `evaluation_timing.json` records
`cuda_peak_memory_mb` during the real Slurm run.

The firm robustness matrix separately records `checkpoint_load_count` and
`transition_build_count`. Its intended counts are one checkpoint load per
episode evaluation and one Jmax transition build per current-eta case, with
smaller J values derived as exact-eta tensor prefixes.
