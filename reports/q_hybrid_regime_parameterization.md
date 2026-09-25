# Hybrid-regime Q parameterization

## Scope

This experiment adds `q_parameterization=hybrid_regime` without changing the
default (`direct`) or reinterpreting legacy `b_times_unit` checkpoints.

## Economic output contract

For firm state `s=(b,z,eta,i,x,...)`:

```text
q_unit(s)     = softplus(q_head(h_q(s)))
Q_claim(s)    = b * q_unit(s)
recovery(s)   = phi * (1 - delta + exp(x + z))

Q_effective(s) = 0             if b == 0
                 recovery(s)   if b > 0 and realized Phat(s) <= 0
                 Q_claim(s)    if b > 0 and realized Phat(s) > 0
```

The zero-debt branch has priority over the default branch. Default Q never
constructs `recovery / b`, so there is no low-debt `1/b` target.

`Q_claim` is the market value of a live debt claim, conditional on the firm
being alive/issuing debt. `Q_effective` is the realized settlement value for a
given current firm state. The realized-default hard gate is therefore not a
candidate-issuance pricing rule.

`PolicyValueOutput.Q` remains effective Q for simulation and plotting. The
explicit model APIs are:

- `_q_unit_output(state)`
- `_q_claim_output(state)`
- `_q_effective_output(state, phat=..., default_mask=...)`

`_q_survival_output` remains only as a compatibility alias. The full public
forward computes equity once and reuses its `Phat` to construct realized
effective Q.

## Teacher and Bellman semantics

`BPGridTeacher` separates:

- `target_model`: frozen equity continuation and candidate `Phat` diagnostics;
- `q_target_model`: Q-head snapshot supplying `q_unit`/`Q_claim`.

Both `q_current` and every candidate-specific `q_issue(b')` use `Q_claim`.
Candidate `Phat` may change continuation/default diagnostics, but it never
replaces issuance value with current recovery. P0/PI are conditional-survival
value functions, so financing cash flow must price a live claim.

The recursive Q Bellman path learns `Q_claim`: parent LHS and future-survival
continuation use `b*q_unit`; future default recovery is added by the existing
Q loss. Future default risk is therefore internalized in the learned claim
price without current candidate `Phat` causing a discontinuous recovery gate.

## Stage schedule

In hybrid mode:

- Q0: structural exact-zero diagnostic; no optimizer and no child/SDF/AiO work.
- QD: structural recovery diagnostic on frozen-P default coverage; no optimizer.
- QS/claim: the required formal `q_unit` Bellman/AiO optimizer phase. Its
  on-distribution parents satisfy `b>0` and frozen parent `Phat>0`; realized
  current-default parents remain settlement diagnostics and do not train the
  claim schedule.
- polish: claim replay only, plus optional boundary diagnostics.

The legacy phase key remains `survival`, but summaries record
`hybrid_learned_object=claim_q`. A deterministic coverage replay draws source
contexts only from realized-survival parents, keeps their non-debt state and
matched child shocks, and replaces only current `b` with bin-centered
candidates. Candidate `Phat` after replacing `b` is diagnostic only: negative
candidate `Phat` never removes or converts a synthetic live-claim sample. By
default, 80% is on-distribution and 20% is synthetic coverage over 10 bins.
The Q Bellman path then recomputes child old-bond leverage from the synthetic
parent debt as `b_sp=b_syn/[1+bar_i*(G-1)]`; it never reuses the child input's
original debt column.

Required gates use `q_structural_zero_tol=1e-8` and QS claim
sample/optimizer-step requirements. If realized-default parents are observed,
QD verifies their effective-Q recovery identity against
`q_structural_recovery_tol=1e-6`; absence of a realized default is reported but
does not reject the stage. Absence of any realized survivor does reject the
claim phase. Direct-Q keeps its existing optimizer-step gates.

Boundary matching remains implemented but disabled by default with
`q_boundary_match_weight=0.0`. Equity indifference `Phat=0` does not imply the
theoretical restriction `Q_claim=recovery`; the gap is diagnostic only.

## Episode-0 timing

The first P target cache uses two snapshots:

- equity continuation: original episode-start frozen firm target;
- Q pricing: post-bootstrap online Q snapshot when bootstrap performed updates.

Later episodes, or an episode-0 resume that skips bootstrap, use the
episode-start target for both roles. This makes the cold-start bootstrap visible
to the first P cash-flow target without replacing its frozen P continuation.
The Q snapshot supplies `Q_claim`; the equity snapshot does not gate issuance.

## Running the matched experiment

```bash
sbatch slurm/run_normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_hybrid_q.slurm
```

The wrapper matches the eta25/BP500 scaled-value experiment and changes only Q
semantics. `Q_BOUNDARY_MATCH_WEIGHT` remains zero unless explicitly overridden.

## Checkpoint safety

`hybrid_regime` is recorded in `policy_value_model_spec`. Raw checkpoint loading
requires an explicit selector from `direct`, `b_times_unit`, or
`hybrid_regime`; tensor-shape compatibility never triggers semantic migration.
