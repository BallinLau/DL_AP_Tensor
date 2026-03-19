# SDF_FC1 Stability Update

## Scope
This change set targets the unstable `M` distribution observed in stage-1 SDF training
(`add_FC1loss=False`) when no direct `Hatcf/LnKF` supervision is available.

## Code Changes

### 1) FC1 redesign: level prediction -> increment prediction
- File: `models/sdf_fc1.py`
- `FC1Model` now predicts `ΔHatcf` and `ΔLnKF` and reconstructs next-period levels:
  - `Hatcf_{t+1} = Hatcf_t + ΔHatcf`
  - `LnKF_{t+1} = LnKF_t + ΔLnKF`
- Interface of `forward_fc1`/`forward_step` stays unchanged for callers.

### 2) SDF exponent clamp for numerical stability
- File: `models/sdf_fc1.py`
- `compute_sdf` adds exponent clipping before `exp(...)`.
- Config switch:
  - File: `config/constants.py`
  - `Config.SDF_EXPONENT_CLAMP = 10.0`

### 3) Stage-1 SDF optimizer/loss policy
- File: `training/episode.py`
- Added stage-aware learning-rate behavior for `sdf_fc1`:
  - when `add_FC1loss=False`, use `hyperparams.sdf_stage1_lr`
  - when `add_FC1loss=True`, restore default SDF base LR
- Added stage-aware moment-penalty weight:
  - stage-1 uses `hyperparams.sdf_stage1_moment_weight`
  - later stages use `hyperparams.sdf_moment_weight`
- Hyperparameters:
  - File: `config/hyperparams.py`
  - `sdf_stage1_lr=1e-4`
  - `sdf_stage1_moment_weight=5.0`
  - `sdf_moment_weight=1.0`

### 4) Per-epoch SDF diagnostics (beyond total loss)
- File: `training/episode.py`
- Each SDF step now logs diagnostics that are aggregated by epoch:
  - `log(E[M])`
  - `log(Var(M))`
  - `ΔHatcf` distribution (`mean`, `p10`, `p50`, `p90`)
  - `ΔLnKF` distribution (`mean`, `p10`, `p50`, `p90`)

## Expected Effect
- Stage-1 should be less prone to exploding `M` values.
- Diagnostics make it explicit whether instability comes from:
  - SDF moments drifting (`log(E[M])`, `log(Var(M))`)
  - FC1 transition jumps (`ΔHatcf`, `ΔLnKF`).

## Follow-up Update (Recon Target Alignment)

### 5) FC1 recon target index alignment (user-consistent layout)
- File: `training/episode.py`
- `add_FC1loss=True` branch in `_compute_sdf_loss` now reads recon targets from:
  - `hatcf_true = children_t[:, :, 7:8]`
  - `lnkf_true = children_t[:, :, 8:9]`
- Added explicit shape guard (`>=9` cols). If missing, recon term is skipped with warning.

### 6) SDF macro-batch layout adjusted to match recon indices
- File: `training/episode.py` (`_create_sdf_batches_from_macro_df`)
- For `add_FC1loss=True`, batch rows use 9-column layout:
  - `[b, z, ETA, i, x, Hatcf, LnKF, Hatc_true, LnK_true]`
- Removed the temporary `M` placeholder from this SDF-only batch format to keep
  recon target positions stable and avoid index mismatch.

### 7) NaN safety retained
- Non-finite `recon_loss` is still sanitized to `0.0` with warning in `_compute_sdf_loss`.

## Follow-up Update (Policy/Value Notebook + Batch Builder Robustness)

### 8) Fix sample-mode detection in firm batch builder
- File: `training/episode.py`
- Root cause of `PV Mode1` zero-batch issue:
  - `_create_firm_batches_from_df` previously used `df['t'].dtype == object` to detect sample-mode labels (`'t'`, `'t+1_k'`).
  - In current pandas environment, `t` column is often `string` dtype (not `object`), so code wrongly entered simulate-time branch and returned empty batches.
- Fix:
  - Use robust detection with `pd.api.types.is_string_dtype(df['t'])` plus first non-null value check.
- Effect:
  - `Sample.build_policy_value_df()` now correctly yields non-empty batches under string dtype.

### 9) Continue Policy/Value two-mode tests in notebook
- File: `tests/sdf_fc1_two_modes_test.ipynb`
- Added/updated PV section to run after SDF tests:
  - Mode1: `Sample.build_policy_value_df()` + `Episode._create_firm_batches_from_df()` + policy training.
  - Mode2: `SimulateTS.simulate()` + `Episode._create_firm_batches_from_df()` + policy training.
- Added batch-empty guards in both mode cells to avoid notebook interruption.
- Final summary print cell is now defensive and will not raise `NameError` when an earlier mode is skipped.

### 10) Verification status (executed notebook)
- Notebook rerun completed with no error outputs.
- PV Mode1 final losses are finite.
- PV Mode2 final losses are finite.

## Follow-up Update (Policy/Value Post-Training Visualization)

### 11) Added Policy/Value diagnostics plots in notebook
- File: `tests/sdf_fc1_two_modes_test.ipynb`
- New section appended after PV training:
  - Loss curves (per-step): `total`, `p0`, `pi`, `q`
  - Parent-level `bp` histogram
  - `b-z-Q`, `b-z-P0`, `b-z-PI`, `b-z-P` 3D scatter surfaces
  - Corresponding `b-z` heatmaps (bin-mean)

### 12) Captured mode-specific loss histories
- File: `tests/sdf_fc1_two_modes_test.ipynb`
- In PV mode1 and mode2 training cells:
  - reset `episode_pv.loss_history = {}` before each mode run
  - store snapshots to `pv_mode1_loss_history` and `pv_mode2_loss_history`
- Purpose:
  - Avoid mixing curves across modes and support direct comparison.

### 13) Execution verification
- Notebook rerun completed successfully with no code-cell errors.
- Newly added plotting cell executed and printed finite-ratio diagnostics for mode1/mode2 outputs.

## Follow-up Update (Policy Heatmap Definition Correction)

### 14) Heatmap/surface logic switched to grid re-evaluation (as requested)
- File: `tests/sdf_fc1_two_modes_test.ipynb` (cells 28-29)
- Previous implementation (removed):
  - Heatmap used sample-point bin averages from observed parent predictions.
- New implementation (current):
  - Use parent `b,z` min/max to build a regular grid.
  - Hold other state variables at parent means (`ETA`, `i`, `x`, `Hatcf`, `LnKF`).
  - Recompute model outputs on each grid point:
    - `Q`, `P0`, `PI`, `P`, `bp`.
  - Plot both 3D surfaces and heatmaps from this computed grid.

### 15) Verification
- Notebook rerun completed successfully with `error_count = 0`.
- Plotting cell prints grid setup ranges/means for both mode1/mode2 and finite diagnostics for outputs.

## Follow-up Update (Q-Priority Policy Fixes)

### 16) Q-first staged policy training
- File: `training/episode.py`
- `train_step` now accepts `policy_loss_terms` to selectively optimize `q/p0/pi`.
- `_run_batches` now supports `HyperParams.q_pretrain_epochs`:
  - early epochs: `q` only
  - later epochs: `p0 + pi + q`

### 17) Q loss stabilization and shape regularization
- File: `training/episode.py`
- `_compute_q_loss` updates:
  - Use `compute_aio_residual(...)` for main branch aggregation (replacing pure product aggregation).
  - Optional `M` detach+clamp in Q loss path (`q_use_detached_m`, `q_m_clamp_min`, `q_m_clamp_max`).
  - Add Q-shape penalties:
    - enforce `dQ/dz > 0`
    - enforce `dQ/db > 0` in low-`b` region
    - enforce `dQ/db < 0` in high-`b` region
- Added per-step Q diagnostics in loss dict (`q_main`, boundary losses, shape terms).

### 18) Hyperparameters and docs sync
- File: `config/hyperparams.py`
- Added Q-priority controls:
  - `q_pretrain_epochs`
  - `q_use_detached_m`, `q_m_clamp_min`, `q_m_clamp_max`
  - `q_shape_weight_z`, `q_shape_weight_b_low`, `q_shape_weight_b_high`
  - `q_shape_b_low`, `q_shape_b_high`
- File: `training/README.md`
  - Added a dedicated section documenting Q-first training hooks and Q-loss behavior changes.

### 19) Smoke verification for Q-first flow
- Ran a minimal local smoke test for policy training with:
  - `epochs=3`, `q_pretrain_epochs=2`, `batch_size=64`
- Verified behavior:
  - Training completed without runtime errors.
  - Final loss dict includes new Q diagnostics: `q_main`, `q_shape_z`, `q_shape_b_low`, `q_shape_b_high`.
  - `loss_history` lengths confirm stage switch:
    - `q=12` (all steps)
    - `p0=4`, `pi=4` (only post-pretrain epoch)

### 20) Notebook defaults synced for Q-priority run
- File: `tests/sdf_fc1_two_modes_test.ipynb` (policy setup cell)
- Added default `hp_pv` settings to activate new Q-priority behavior directly from notebook:
  - `q_pretrain_epochs=10`
  - `q_use_detached_m=True`, `q_m_clamp_[min,max]=[0.5,1.5]`
  - Q-shape weights and `b` split thresholds.

## Follow-up Update (Q Freeze + Warm-start)

### 21) Q-only parameter freeze is now active
- File: `training/episode.py`
- In Q-only pretrain stage (`epoch < max(q_pretrain_epochs, q_warmstart_epochs)`):
  - keep original `bar_i/bp/bar_z` network outputs
  - freeze non-Q parameters
  - trainable scope via `q_pretrain_trainable_scope` (`q_head_only` / `q_path`)
- Goal: keep economic equation intact while isolating Q pretraining updates.

### 22) Structured warm-start supervision added to Q loss
- Files: `training/episode.py`, `config/hyperparams.py`
- Added warm-start controls:
  - `q_warmstart_epochs`, `q_warmstart_weight`
  - `q_warm_A`, `q_warm_b_star`, `q_warm_sigma`, `q_warm_alpha_z`, `q_warm_alpha_x`
- Added warm-start term in `_compute_q_loss`:
  - `L_warm = MSE(Q, Q_warm_target(b,z,x))`
  - `Q_warm_target` uses Gaussian peak on `b` and exponential risk scaling on `(z,x)`.

### 23) Additional Q diagnostics
- File: `training/episode.py`
- New logged terms:
  - `q_physics`
  - `q_warmstart`
  - `q_warm_weight`
  - `q_pretrain_mode`
  - `q_freeze_mode`

### 24) SDF recon target index compatibility fix
- File: `training/episode.py`
- FC1 recon target extraction now supports both layouts:
  - with `M` column: true targets from `children[..., 8:10]`
  - without `M` column: true targets from `children[..., 7:9]`
- Goal: avoid empty-slice/NaN issues when macro SDF batch schema differs.

### 25) Smoke re-check
- Ran short policy-only smoke after the new changes.
- Confirmed:
  - finite `q` loss
  - pretrain/warm-start flags switch as expected:
    - `q_pretrain_mode=[1,1,0]`
    - `q_freeze_mode=[1,1,0]`
    - `q_warm_weight=[1,1,0]`
