# Independent H10 + H100 replay experiment

Base: `1e05ecde9f6981c54f89deeadde1a55bc5c48296`, including reviewed `ca7e727` and the subsequent actual H50 production fixes. The original H50 checkout/output is not an edit target. See the root `H50_PROTECTION_MANIFEST.json` (read-only observed job 711731, node g53, Outer13).

## Execution and approval boundary

`analysis/run_h10_h100_longrun_macro.py` defaults to **plan-only**. It requires a new `CONTINUATION_APPROVAL_V6_H10_H100` approval, exact source SHA, training-plan fingerprint, sampling signature/artifact hashes, initial checkpoint SHA, source transaction/pointer SHA and runtime partition. H50 V5 is rejected. No approval is created by this runner. The plan is four independent experiment Outers 12–15, not the original H50 Outers.

```sh
python analysis/run_h10_h100_longrun_macro.py --output /NEW_RUN
```

Formal deployment uses `slurm/run_h10_h100_longrun_macro_a800.slurm` only after integration smoke, source review and V6 content approval. Export the required variables listed in that script. Planned submission (NOT an executed job ID):

```sh
sbatch --output="$RUN_ROOT/logs/slurm.%j.out" \
  --error="$RUN_ROOT/logs/slurm.%j.err" \
  slurm/run_h10_h100_longrun_macro_a800.slurm
```

Resume the **same** source/inputs/approval with `RESUME=1`; inspect the previous Slurm state first, never duplicate an active job. Both Outer commits and replay commits are journal stages. A replay failure does not overwrite `outer_complete.pt` or the preceding committed journal. The replay uses atomic per-time shards and a separate immutable `commit.json`, and can be retried through `--resume` without redoing committed optimization.

## Data contracts

- H10: 4096 initial Train / 1024 fixed initial Validation, 200 firms, two actual children. Root only at t0; Exit-v2 and Grid policy unchanged.
- Firm: Hybrid 50% Uniform h[-6,2]/ell[-2,5], 25% observed and 25% calculated. Rebuild conditional children/M/RHS; no old labels. Budget remains three inner cycles, original P/Q optimization; BP head is skipped, not Grid simulation.
- FC1: current D0/D1 legal coverage plus previous committed H100, up to 4096 unique parent-time groups. Historical target remains explicitly historical, not current-policy realization. Requested long-run 2048; any shortfall transfers to coverage; real capacity/lineage caps can underfill. No repeated groups to fake sample size. Fixed bins are fitted once on Train sources plus economic reference bounds; directional overflow is separate.
- SDF: only current D2 complete pairs. One shared vectorized eligibility scan produces FC1 and SDF indices. Single-child replay is never an SDF source.
- H100: 100 root-consistent + 100 off-equilibrium-belief starts, 200 firms, 100 transitions, one Child0 spine, burn-in parent times 0–19. Sobol macro capital/belief/x design; original Beta family/correlation proposals. True capital is rescaled to the requested total. Exploratory h is never relabeled as cleared h. No optimizer or model mutation.
- Child0-only expansion keeps the original two-branch identity namespace; merely setting branch_num=1 would change economic node IDs and Entry CRN, so is deliberately not used.
- Sparse terminal suffix: scheduled denominators remain n*horizon*branches. Actual failed nodes are retained; no repeated synthetic empty rows after permanent termination. CSV sidecars avoid full micro-shard deserialization for selection.
- A legal ledger can permit original next Entry even with zero incumbent survivors. Undefined ledger cannot. Root failure is not relabeled extinction.

## Tests versus science

CPU production-function tests cover Child0 transition/Entry/policy exact parity, H10/H100 interfaces/resume, old H50 contracts, burn-in exclusion, invalid/Validation rejection, lineage caps, empty replay fallback, pool corruption and shared eligibility parity. The separate frozen real-checkpoint smoke is **not** optimizer-pipeline smoke or formal training.

Actual production-shape GPU memory/performance and full optimizer-pipeline smoke require their own completed artifact evidence; do not infer them from CPU unit tests. No claim of long-run stationarity, broad feasible support, balanced effective data or equilibrium convergence is made.

## Bounded smoke

`analysis/run_h10_h100_pipeline_smoke.py` is an explicitly named integration test. It uses 16 initial parents per split, two H100 starts per mode (four total), and two tiny cycles with actual P/Q/FC1/SDF stages. It does not generate formal approval and cannot count as the four formal Outers. A SDF skip due to no valid pairs is recorded and cannot satisfy optimizer-coverage PASS.

Pending work/results must remain labeled pending in the acceptance report. Do not submit the formal budget merely because the code parses or CPU tests pass.
