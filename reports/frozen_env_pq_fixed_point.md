# Frozen-Environment P-Q Fixed-Point Test

## Scope

This experiment repeatedly applies the production fitted P stage followed by the production Q regime stage while holding the EP2 environment fixed. It does not call simulation, train SDF/FC1, train BP, update entry, or modify a production loss. The evaluator is delivered on GitHub base `5195ad0`; the GRID run and frozen batch provenance remain `4b236ea`.

## Production semantics at commit `4b236ea`

- P stage: `Episode._run_policy_value_evaluation_stage()` trains `value_encoder`, `v0_head`, `vi_head`, `barz_model`, and `bari_model`. A cycle-start `PolicyValueModel` snapshot supplies equity continuation and Q pricing. Its target-grid cache is built once and hash-checked throughout the stage.
- Q stage: `Episode._run_q_regime_training()` freezes the post-P P snapshot and executes `zero -> default -> survival -> polish`. The GRID run uses `q_target_refresh_mode=phase`; each phase therefore snapshots the current online Q at phase entry.
- BP heads and `policy_encoder` are excluded from both stages. SDF/FC1 is only used through the already-materialized frozen training batches and the canonical transition bank.

## Exact-data requirement

The GRID run with `MODEB_RESIMULATE_AFTER_PV=1` saves `ep2_stage_modeb.pkl` after the post-PV replay. EP2 P/Q actually consumed the earlier post-SDF-refresh, pre-PV tensor plus deterministic mixture coverage and its frozen train/validation split. That exact bank was not saved by the historical run.

For this reason the runner requires `--frozen-batch-bank` with format `frozen_pq_batch_bank_v1` and strict provenance:

```text
episode = 2
panel_stage = post_sdf_refresh_pre_pv
simulation_reused_without_rerun = true
batch_composition_frozen = true
batch_order_frozen = true
validation_split_frozen = true
source_commit = 4b236eaacf47b2ac7cb506508e54b21a199e89b0
```

Using `ep2_stage_modeb.pkl` to reconstruct the bank would violate the frozen-environment definition and is deliberately rejected.

## Run

```bash
FROZEN_BATCH_BANK=/path/to/exact_ep2_pre_pv_batches.pt \
sbatch slurm/run_frozen_env_pq_fixed_point_gpu.slurm
```

The runner requires CUDA, uses a fixed cycle RNG seed, performs a cycle-1 deterministic replica check, hashes all immutable components, and writes per-cycle checkpoints, stage-fit diagnostics, contemporaneous fixed-point residuals, drift/contraction/oscillation metrics, plots, and an A-F verdict.

Q stage-fit quantiles are the production fixed-validation-bank phase metrics (the production helper aggregates its per-batch diagnostics). They remain separate from canonical post-cycle Bellman self-consistency residuals; the runner never relabels the latter as phase target fit.

The `5195ad0` entry-mechanism commit does not change the five production P/Q staged methods used here. The runner verifies their AST fingerprints against `4b236ea` before loading data, so unrelated entry changes are allowed but a silent P/Q mapping change is rejected.
