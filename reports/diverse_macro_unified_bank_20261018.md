# Distribution-Diverse Short-Horizon + Unified Firm–Macro Bank

## Provenance

- Base/server commit audited: `ab47fd98f0b0072a72cf6d2fdb8eb2eff80c8fa6`
- Implementation commit used by the completed smoke: `7610a1c9eebd904f3a17fc9f484e6deb0f72d099`
- Output-path-only follow-up used by the small pilot: `4c1595b2c6d5d3a0ea9a17d952bd1d50884c2448`
- Firm checkpoint: `fast_inner_fixed_point_20261016/coupled_fixed_macro_profiled/checkpoints/outer_005.pt`
- Firm checkpoint SHA-256: `895c2698300e71d2776abb53da70bed9394b45ebb8093b7097d567aaff4df1e6`
- Macro checkpoint: `full_batch32768_canonical_probe/l1/joint_extended/checkpoints/outer_014.pt`
- Macro checkpoint SHA-256: `a66981ee7b68beaf28827b48d66a6985474255617a5119d199f5ed3b7c1b10b7`
- No optimizer step, FC1/SDF/P/Q retraining, or equilibrium outer was run.

## 1. Where the old Firm transition used f-to-f

**CONFIRMED.** Two old paths treated Firm input columns 5/6 (`Hatcf/LnKF`) as if they were calculated macro state:

- `analysis/run_macro_outer_inner_pq.py::refresh_transition_bank()`;
- `training/inner_fixed_point.py::build_child_bank()`.

Production `SimulateTS` already used `Hatc_cal_t/LnK_cal_t` after resource accounting. The mismatch was in the Firm fixed-point transition/bank path.

## 2. New cal-to-f semantics

**CONFIRMED.** `refresh_transition_bank()` now requires `parent_macro_context[N,2]`. `build_child_bank()` calls `parent_bank.require_macro_context()`. Both call:

```text
MacroForward(x_t, x_t1, Hatc_cal_t, LnK_cal_t)
```

Legacy seven-column banks remain identifiable as `legacy_firm_only_v0`, but a Macro transition on such a bank fails. There is no fallback to f-to-f.

## 3. Firm input dimension

**CONFIRMED.** `firm_parent_state` remains `[N,7] = [b,z,eta,i,x,Hatcf,LnKF]`. Calculated macro state is stored separately as `parent_macro_cal[N,2]`.

## 4. Node matching and coverage

**CONFIRMED.** On-support rows use a many-to-one merge over `path/t/branch` and, when present, `economic_node_id`. The default split is 75% on-support and 25% `synthetic_micro_coverage`. Coverage changes only micro state while retaining the actual node's forecast/calculated macro pairing.

## 5. Cross-section families

**CONFIRMED.** Implemented: low leverage, high leverage, positive/negative b-z correlation, bimodal leverage, capital concentrated among high-z/low-z firms, and on-policy source support. Capital is nonnegative and normalized to the sampled aggregate K.

## 6. Initial forecast and burn-in

**CONFIRMED.** Initial-transition diagnostic mode retains the original forecast. Main short-horizon mode applies an explicit parent-time burn-in; it never sets `Hatcf_t=Hatc_cal_t` or `LnKF_t=LnK_cal_t` as a hidden repair.

## 7. Pair counts

Completed 8+8-path smoke, horizon 5, burn-in 1:

- train: 64 valid branch-level pairs;
- validation: 64 valid branch-level pairs;
- two branches per valid parent.

The 64+64-path small pilot stopped during validation before final pair artifacts were published because of a production resource-feasibility exception.

## 8. Child independence and alignment

**CONFIRMED in tests and smoke.** Child time equals parent time plus one; branches 0 and 1 share the parent state but have distinct shock outcomes. Adjacent time points are not used as a double-sampling substitute.

## 9. SimulateTS vs unified transition parity

**PASS in completed smoke.** Maximum observed differences before the final pass were:

- child Hatcf: `7.7486e-7`;
- child LnKF: `4.7684e-7`;
- raw M: `1.2398e-5`.

The audit uses variable-specific float32 tolerances (`2e-6` for forecasts, `2e-5` for raw M). A controlled calculated-state perturbation still fails the parity test. The M tolerance accounts for production-batch vs audit-batch CUDA kernel rounding; it does not hide a cal/f mismatch.

## 10–11. Old f-to-f vs new cal-to-f

In the tiny validation smoke:

- old f-to-f M clipping rate: `0.296875`;
- new cal-to-f M clipping rate: `0.328125`.

Therefore the timing correction did **not** reduce clipping in this tiny controlled sample. Semantic correctness and numerical improvement are separate claims.

## 12. State support comparison

**UNRESOLVED for the requested three-way comparison.** New train/validation support tables were produced by the tiny smoke. The old Episode-1 and Joint-L1 canonical sources were not silently coerced into the new schema. A three-way conclusion requires explicit extraction of calculated and forecast states from those historical artifacts. No unsupported range-improvement claim is made.

## 13. Low-dimensional state sufficiency

Tiny smoke classification: `STATE_SUFFICIENCY_QUESTIONABLE`. This is only a screening result from eight distributions and is not a general-equilibrium or state-sufficiency proof. The small pilot did not finish, so this conclusion remains preliminary.

## 14. Entry and BP modes

**CONFIRMED.** Actual simulator metadata reported:

- `entry_mode=investment_compare`;
- `entry_spec_version=investment_compare_v1`;
- `entry_capital_ratio=0.05`;
- `entry_cost_max=0.5`, matching `I_THRESHOLD=0.5`;
- `consumption_aggregation_mode=raw`;
- `node_accounting_mode=economic_node_ledger`;
- `bp_action_source=grid` through the existing BPGridTeacher-backed resolver.

The first attempted smoke with `entry_cost_max=1.0` was correctly rejected before simulation because it conflicted with the ordinary investment-cost support. The invariant was not relaxed.

## 15. Small-pilot correctness blocker

**CONFIRMED blocker.** Slurm job `694629` stopped in production validation simulation:

```text
EntryEconomicInfeasibilityError:
raw node resource account is infeasible or non-finite (C_raw<=0)
t=5, branch=0, paths=[54]
C_oper=[-10.524210929870605]
I_entry=[0.0]
C_raw=[-10.524210929870605]
accepted_count=[0.0]
no positive floor was applied
```

Observed path extinction was also recorded for train path 41 at t=5 and validation path 9 at t=1. No path was resampled or removed, and no consumption floor/economic parameter was changed.

## 16. Minimal next tests

1. Diagnose the path-54 node account with the same shocks and checkpoint, without changing its economics.
2. Extract historical Episode-1 and canonical Joint-L1 state support into explicit forecast/calculated columns for the requested three-way support table.
3. Only after the resource blocker is understood, rerun the same 64+64 x 5 pilot unchanged.

## 17. Unchanged model/economics

P0/PI Bellman equations, Q pricing equation, investment/default conditions, household wealth recursion, SDF objective/AL constraints, FC1 loss, seven-dimensional Firm input, entry/exit equations and calibrated parameters were not changed.

## Validation

- Local relevant suite: `122 passed` before launcher-only changes.
- Final targeted suite: `51 passed` locally and `51 passed` on THU A800.
- Completed tiny smoke: Slurm `694623`, exit `0:0`.
- Blocked small pilot: Slurm `694629`, exit `1:0`, preserved logs.

## Final flags

```text
TRANSITION_PARITY = PASS
CAL_TO_F_SEMANTICS = PASS
UNIFIED_BANK_VALID = TRUE
SHORT_HORIZON_DATA_VALID = FALSE
ENTRY_MODE_VERIFIED = TRUE
GRID_POLICY_VERIFIED = TRUE
READY_FOR_MACRO_RETRAINING = FALSE
FULL_EQUILIBRIUM_STARTED = FALSE
```

`SHORT_HORIZON_DATA_VALID` and `READY_FOR_MACRO_RETRAINING` are false because the allowed small pilot did not complete its validation split. The successful tiny smoke alone is insufficient to override that blocker.
