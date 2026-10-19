# Final Firm-First Nested Training — command draft

These commands are review material only. They were **not executed** during Phase A.
All options below are implemented by `analysis/run_full_nested_equilibrium.py`.

## 1. Code-review-approved smoke

```bash
CUDA_VISIBLE_DEVICES=0 python analysis/run_full_nested_equilibrium.py \
  --cold-start \
  --smoke \
  --device cuda:0 \
  --output-dir cachedir/final_firm_first_nested_smoke_REVIEWED
```

The smoke override uses 8/4 valid train/validation targets, 16→32 and 8→16
candidate budgets, one equilibrium outer, and minimal optimizer budgets. It is
still a real economic smoke and therefore must wait for code-review approval.

## 2. Firm-first two-outer pilot

```bash
CUDA_VISIBLE_DEVICES=0 python analysis/run_full_nested_equilibrium.py \
  --cold-start \
  --initial-checkpoint /ABSOLUTE/PATH/TO/REVIEWED_INITIAL_CHECKPOINT.pt \
  --device cuda:0 \
  --output-dir cachedir/final_firm_first_nested_pilot_REVIEWED \
  --equilibrium-outer-max 2 \
  --firm-inner-max 8 \
  --p-policy-outer-max 8 \
  --q-picard-outer-max 8 \
  --macro-inner-max 20 \
  --macro-train-target-valid 2048 \
  --macro-validation-target-valid 512 \
  --macro-train-initial-candidate-budget 4096 \
  --macro-train-max-candidate-budget 8192 \
  --macro-validation-initial-candidate-budget 1024 \
  --macro-validation-max-candidate-budget 2048 \
  --macro-minimum-partial-train 1024 \
  --macro-minimum-partial-validation 256 \
  --no-run-long-diagnostics
```

## 3. Resume

Resume accepts either the run directory, a stage checkpoint, or the run's
`latest_checkpoint.json` resolution target:

```bash
CUDA_VISIBLE_DEVICES=0 python analysis/run_full_nested_equilibrium.py \
  --resume /ABSOLUTE/PATH/TO/RUN_OR_CHECKPOINT \
  --device cuda:0
```

Do not add `--allow-code-mismatch` unless an independent review has explicitly
approved the Git/protocol mismatch recorded by the checkpoint loader.

## 4. Full-training candidate

This is only a candidate command. The 10-outer run remains unauthorized until
the smoke/pilot artifacts pass review.

```bash
CUDA_VISIBLE_DEVICES=0 python analysis/run_full_nested_equilibrium.py \
  --cold-start \
  --initial-checkpoint /ABSOLUTE/PATH/TO/REVIEWED_INITIAL_CHECKPOINT.pt \
  --device cuda:0 \
  --output-dir cachedir/final_firm_first_nested_full_REVIEWED \
  --equilibrium-outer-max 10 \
  --macro-train-target-valid 2048 \
  --macro-validation-target-valid 512 \
  --macro-train-initial-candidate-budget 4096 \
  --macro-train-max-candidate-budget 8192 \
  --macro-validation-initial-candidate-budget 1024 \
  --macro-validation-max-candidate-budget 2048 \
  --macro-minimum-partial-train 1024 \
  --macro-minimum-partial-validation 256 \
  --market-clearing-mode market_cleared_experimental_v1 \
  --aggregation-mode continuum_mass_consistent_v1 \
  --disable-fc2 \
  --sdf-recursive-epochs 0 \
  --no-run-long-diagnostics
```

No command in this document was submitted to Slurm or run locally.
