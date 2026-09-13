# Firm-side checkpoint evaluator

`experiments/evaluate_checkpoints.py` performs a read-only evaluation of one
`PolicyValueModel` checkpoint. It does not evaluate Bellman residuals, SDF/FC1,
FC2, episode drift, or simulated-moment convergence.

## Inputs

- A combined checkpoint with `models.policy_value`, `hyperparams`,
  `config_snapshot`, `policy_value_model_spec`, and `value_parameterization`; or
  a raw policy checkpoint plus explicit JSON sidecars.
- A firm-stage pickle or CSV with matched parent/child rows and observed child
  `M`. Parent medians determine fixed `x`, `Hatcf`, and `LnKF`; the grid fixes
  `ETA=1` and scans `b` and `z`.

Strict checkpoint reconstruction is intentional. Legacy checkpoints whose
architecture differs from the current `PolicyValueModel` must be migrated or
evaluated at their source commit; the evaluator never uses `strict=False`.

## Example

```bash
python3 experiments/evaluate_checkpoints.py \
  --checkpoint /path/to/combined_checkpoint.pt \
  --firm-data /path/to/final_stage_firm.pkl \
  --output-dir /path/to/firm_checkpoint_evaluation \
  --device cuda \
  --b-points 101 \
  --z-min -4 \
  --z-max 4 \
  --z-points 101 \
  --i-points 101
```

For a raw state dict, use `--pv-ckpt` and provide `--hyperparams-json`,
`--config-json`, and `--model-spec-json`. Current runtime defaults are accepted
only when the corresponding explicit opt-in flag is passed.

## Output semantics

- `default/default_boundary.csv` uses the first interpolated `Phat=0` crossing.
  Missing crossings remain `NaN` and carry a boundary status.
- `bar_i_cond` is the conditional investment probability; `bar_i_eff` is the
  survival-adjusted executed probability; `i_star` is the first scanned
  `VI(i)-V0=0` crossing. No gap is computed between these distinct objects.
- `Q` is total debt value. `q_unit=Q/b` only for `b>1e-12`; it is `NaN` at zero.
- BP consistency calls `BPGridTeacher.compute()` directly. The reference child
  transition bank is derived deterministically from matched firm transitions.
- `metadata.json` records the grid, reference state, transition bank, checkpoint
  metadata, and before/after parameter hashes.
