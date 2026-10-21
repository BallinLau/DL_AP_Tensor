# User-authorized H50 Outer 12–20 execution

Base: ca7e727d9aaed2c87b74d4018e09cae483fa8a68. No economics/root tolerance changes.
Authorization: attached TRAINING EXECUTION REQUEST 5d3d25b8, 2026-10-21.

- Full Job702636 Outer11 state is certified against its committed Journal; optimizers,
  AL, schedulers and fresh RNG restored with the existing production interface.
- Explicit `--end-outer 20 --skip-bp-head`; BP stages remain transactional and emit zero
  Head optimizer steps. Production simulation still requires Grid BP.
- Hybrid: 50% Uniform / 25% observed / 25% calculated. Requested 8192 rows, batch2048;
  duplicate firm-time-branch donors are removed from observed draws before mixing;
  balanced distinct capacity may underfill. Cross-context micro donor reuse remains
  disclosed by actual unique counts, never claimed independent.
- Joint candidates retain fixed32768 upper budget,16 candidate batch, seven families,
  original strict roots and deterministic capacity refill. No new adaptive early-stop.
  `coverage_bank/progress.json` logs measured throughput and estimated remaining time.
- Rollout batch32, FC1/SDF parent batch128. These values are approval-bound.
- A missing SDF training pair set skips that chunk without optimizer/AL/RNG changes.
  Missing validation pairs yields NOT_AVAILABLE, never training-data substitution.
- Fixed Validation remains certified; dynamic Train templates activate next Outer only.
- Nine complete Outer schedules; no final-convergence early exit.

CPU preparation/submission (explicitly authorized this task):

```sh
python analysis/prepare_h50_formal_execution.py \
  --deployment-root /PERSISTENT/NEW_LAUNCH \
  --run-root /PERSISTENT/NEW_RUN --submit
```

The helper refuses an existing run or previous submission intent; there is no automatic
retry of uncertain `sbatch`. Source is deployed using a Git bundle, not GitHub push.
No approved=true artifact is committed into the source checkout.

Resume uses the SAME immutable source, approval and environment recorded in
`slurm_environment.json`, with RESUME=1 and the same Slurm script. Inspect latest
Journal and job state before submitting a replacement; do not repeat committed stages.

Tests: targeted CPU suite136 passed; shell syntax and diff whitespace checks passed.
No pre-training A/B, SDF extrapolation study, GPU benchmark or frozen H50 experiment.
