# Transition-SDF M plumbing audit and fix (2026-10-15)

## Overview

This patch fixes a data-plumbing inconsistency in the inner Bellman operator.  It changes only where the transition SDF `M_{t,t+1}` is read.  It does not change the P0, PI, or Q economic equations, clamp ranges, exact-eta integration, BP grid, default/investment rules, AiO reduction, optimizer, damping, network architecture, or bank construction.

- Pre-fix worktree HEAD: `ada06a5076228cc14fcb973139658e4973f7f47f`
- Historical 10-sweep code: `b4a15ad5d7309a455769cdc4a93bcb17b33527b7`
- Source checkpoint code: `959984df64918654aa69b7f5596301df062f8030`
- Source checkpoint SHA-256: `97b7bb1a284904027d1ce17df7b73940c00f12fd08bab86fb689496550fc5d5d`

The historical 10-sweep artifacts were read only and were not overwritten.

## Entry point and data input pipeline

The formal inner entry point remains `experiments/run_inner_fixed_point.py`.  `InnerFPCollocationBank` enforces seven-column model state at `training/inner_fixed_point.py:123-136`; `InnerFPChildBank` enforces `[N,J,8]` children at `training/inner_fixed_point.py:157-174`; `make_batches` keeps parents at width 7 and unbinds the width-8 child bank at `training/inner_fixed_point.py:380-400`.

The eighth child column is the transition edge SDF.  The model input remains the first seven columns only.

## Before: root cause

The old helper selected the SDF source with `parent.shape[1] > 7`.  This confused a parent storage layout with transition-edge availability:

| Path | Parent / child | Before M source | Before used-M semantics |
|---|---|---|---|
| Outer P train | 8 / 8 | `child[:,7]` | P clamp |
| Inner P train | 7 / 8 | unit 1 | P clamp applied to 1 |
| Inner P eval | 7 / 8 | unit 1 | P clamp applied to 1 |
| Inner Q train | 7 / 8 | unit 1 | Q clamp applied to 1 |
| Inner Q eval | 7 / 8 | `child[:,7]` | Q clamp |

Therefore the historical inner P training/evaluation operator differed from the outer operator, and historical Q survival training differed from its snapshot evaluator.

## After: explicit transition-edge contract

`Episode._extract_transition_m_list` at `training/episode.py:1225-1275` now owns raw-M extraction:

- all children width >= 8: use `child[:,7:8]`, source `child_column_7`;
- all children width == 7: unit fallback only when a caller explicitly allows it;
- mixed widths, width < 7, rank mismatch, or batch-size mismatch: fail fast;
- parent width and `parent[:,7]` are never inspected.

P and Q processing remain separate:

- P builder: `training/episode.py:1277-1303`, retaining `pv_fixed_sdf` and the P clamp;
- Q builder: `training/episode.py:1305-1330`, retaining `q_use_detached_m` and the Q clamp.

| Path | Before M source | After M source | Clamp |
|---|---|---|---|
| Outer P train | child M | child M | P `[0.7,1.3]` under source config |
| Inner P train | unit 1 | child M | P `[0.7,1.3]` |
| Inner P eval | unit 1 | child M | P `[0.7,1.3]` |
| Inner Q train | unit 1 | child M | Q `[0.5,1.5]` |
| Inner Q eval | child M | child M | Q `[0.5,1.5]` |

Formal P/Q callers pass `allow_unit_fallback=False`.  The inner evaluator additionally requires `child_column_7` and raises on fallback at `training/inner_fixed_point.py:650-718`.

## High-level call graph

```text
inner fixed bank [parent N,7; child N,J,8]
  -> Episode._extract_transition_m_list(children)
     -> P: _build_policy_m_lists -> P clamp/fixed-SDF
        -> _build_pq_value_target_cache
        -> BPGridTeacher.compute_value_target
        -> cached P loss -> total.backward -> optimizer.step
     -> Q: _build_q_m_lists -> Q detach/clamp
        -> _compute_q_survival_bellman_loss
        -> QLoss.compute_main_components
        -> Q loss -> loss.backward -> optimizer.step
  -> evaluate_inner_operator_snapshot
     -> same P builder
     -> same Q builder
```

All arrows above are direct calls except `QLoss.compute_main_components`, reached through the Q survival loss as a nested call.

## Detailed path audit

| Bellman path | Confirmed code | Parent width | Child width | Raw M | Used M |
|---|---|---:|---:|---|---|
| Outer direct P0 | `training/episode.py:5365-5398` | normally 8 | 8 | child col 7 | P rules |
| Outer direct PI | `training/episode.py:5621-5654` | normally 8 | 8 | child col 7 | P rules |
| P signed residual diagnostics | `training/episode.py:7004-7161` | 7 or 8 | 8 | child col 7 | explicit raw/train mode |
| Q signed residual diagnostics | `training/episode.py:7163-7192` | 7 or 8 | 8 | child col 7 | explicit raw/train mode |
| Target-grid convergence | `training/episode.py:7250-7420` | 7 or 8 | 8 | child col 7 | P rules |
| BP target cache | `training/episode.py:8770-8810` | 7 or 8 | 8 | child col 7 | P rules |
| Frozen P value cache/hash | `training/episode.py:9035-9135` | 7 or 8 | 8 | child col 7 | P rules; used M hashed |
| Inner P evaluator | `training/inner_fixed_point.py:616-675` | 7 | 8 | child col 7 | P rules |
| Inner Q evaluator | `training/inner_fixed_point.py:677-754` | 7 | 8 | child col 7 | Q rules |
| Inner P-only training | same frozen-cache path | 7 | 8 | child col 7 | P rules |
| Inner Q survival training | `training/episode.py:6167-6213` | 7 | 8 | child col 7 | Q rules |

`BPGridTeacher.compute_value_target` remains a consumer of explicit `m_list` at `training/bp_grid_teacher.py:846-940`; it does not infer M from child layout.

## Loss aggregation and optimization sinks

- P: the frozen target is built once at `training/episode.py:9072-9135`; the cached scalar is consumed by `total.backward()` and `optimizer.step()` at `training/episode.py:8562-8587`.
- Q: the survival target components are produced by `QLoss.compute_main_components` at `losses/q_loss.py:676-730`; the regime scalar is consumed by `loss.backward()` and `optimizer.step()` at `training/episode.py:11184-11193`.
- Exact-eta expansion still duplicates each continuous-shock M across its eta=0/1 pair at `training/episode.py:1337-1357`.

## Outer/inner and train/eval parity

The formal audit used the source checkpoint and the first 256 parents from the unchanged historical validation/child bank.

| Check | Max absolute difference |
|---|---:|
| P0 target, outer-like parent8 vs inner parent7 | `0.0` |
| PI target, outer-like parent8 vs inner parent7 | `0.0` |
| Q target total, train builder vs evaluator builder | `0.0` |
| Q continuation component, train vs eval | `0.0` |
| Q recursion gain, train vs eval | `0.0` |

The outer and inner used-M hashes were identical: `010b9d090a012781896fc6b2677fc3936f53b87fb35b53a3a7ca0172e4200279`.

## Historical child-bank M distribution

All-child statistics over 256 parents x 16 continuous child shocks:

| M | mean | p50 | p90 | p99 | min | max | share `|M-1|>1e-6` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw child M | 0.944780 | 0.903915 | 1.151318 | 1.620886 | 0.089266 | 11.522045 | 1.000000 |
| P used M | 0.919672 | 0.903915 | 1.151318 | 1.300000 | 0.700000 | 1.300000 | 1.000000 |
| Q used M | 0.910757 | 0.903915 | 1.151318 | 1.500000 | 0.500000 | 1.500000 | 1.000000 |

The Q snapshot's survival-only subset has raw mean 0.935619 and used mean 0.917360.  The raw bank is therefore materially non-unit; the legacy fallback changed the operator.

## Source-checkpoint residual counterfactual

No optimization step was run.  Values are normalized residual mean/p90 on the same 256-parent validation subset.

| Equation | Legacy M=1 mean | Legacy p90 | Corrected child-M mean | Corrected p90 |
|---|---:|---:|---:|---:|
| P0 | 0.274584 | 0.773363 | 0.263980 | 0.745565 |
| PI | 0.340655 | 0.938288 | 0.328432 | 0.859078 |
| Q survival (n=110) | 0.073773 | 0.182148 | 0.067753 | 0.168003 |

The source checkpoint is slightly closer to the corrected operator on this subset.  This is a counterfactual diagnostic only and does not establish convergence.

## Tests and smoke

- New transition-M tests: 12 passed.
- Existing policy cached-target/stage tests: 106 passed.
- Existing inner/exact-eta/BPGridTeacher tests: 41 passed.
- Existing Q survival/hybrid tests: 56 passed.
- Existing convergence-surface tests: 39 passed.
- Full suite: **919 passed, 1 skipped, 0 failed**; 62 pre-existing warnings.

Read-only smoke result:

- P target builder source: `child_column_7`;
- Q survival builder source: `child_column_7`;
- P used M identically one: false;
- Q used M identically one: false;
- optimizer steps: 0.

Machine-readable evidence:

- `cachedir/transition_m_plumbing_fix_20261015/audit.json`
- `cachedir/transition_m_plumbing_fix_20261015/smoke.json`

## Unresolved / remaining caveats

- No corrected 10-sweep block-isolation experiment was run in this task.
- Historical block-isolation results remain legacy-M results and were not overwritten.
- The corrected operator's multi-sweep convergence remains untested.
- The residual comparison is a 256-parent read-only subset, not a convergence theorem.

## Final summary

The bug was a layout sentinel: transition-M availability incorrectly depended on parent width.  Raw transition M now comes exclusively from child transition data; P and Q retain their separate production processing rules.  Parent state remains seven-dimensional, outer P semantics are unchanged, inner P/Q train/eval semantics now agree, and formal inner execution fails fast rather than silently falling back to unit M.

This fix changes only the source/plumbing of transition M in P and Q Bellman backups.  It does not change the economic Bellman equations themselves.

## Structured trace table

| Caller | Callee | Edge | Tensor role | Optimization owner |
|---|---|---|---|---|
| `_compute_p0_loss` / `_compute_pi_loss` | `_build_policy_m_lists` | direct | child edge -> raw/used P M | P |
| `_build_pq_value_target_cache` | `_policy_batch_hash_components` | direct | used P M and `m_hash` | none during cache build |
| `_build_pq_value_target_cache` | `BPGridTeacher.compute_value_target` | direct | P0/PI frozen Bellman target | none during cache build |
| P stage | cached P loss | direct | P0/PI target -> scalar `total` | P |
| `_compute_q_survival_bellman_loss` | `_build_q_m_lists` | direct | child edge -> raw/used Q M | Q |
| Q survival loss | `QLoss.compute_main_components` | nested | Q target/continuation/recursion gain | Q |
| `evaluate_inner_operator_snapshot` | P/Q M builders | direct | fresh current-operator targets | none |
