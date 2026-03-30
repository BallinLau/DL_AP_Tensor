# training

Training orchestration for episodes and modules.

## episode.py
`Episode` encapsulates a single training episode:
- `generate_data` (legacy) and `run_episode` (current flow)
- SDF, Policy/Value, FC2 losses and batch builders

### Policy/Value: Q-first training hooks
- `train_step(..., policy_loss_terms=...)` now supports selective optimization among `['q', 'p0', 'pi']`.
- `_run_batches(...)` now supports separated policy training via
  `HyperParams.policy_separate_q_pvbp_training`:
  - Stage A: `q_stage_epochs` epochs, optimize `q` only
  - Stage B: `pvbp_stage_epochs` epochs, optimize `p0 + pi` only
  - no automatic return to one-step `p0 + pi + q` joint backward
- `Q` and `PV/BP` now use separate optimizers:
  - `policy_value_q`
  - `policy_value_pvbp`
- `_compute_q_loss(...)` now includes:
  - AIO residual aggregation (`compute_aio_residual`) instead of pure branch-product aggregation
  - Optional `M` detach + clamp (`q_use_detached_m`, `q_m_clamp_min`, `q_m_clamp_max`)
  - Q-only parameter freezing (`q_freeze_non_q_in_pretrain`):
    - keep original economic equation (`bar_i`, `bp`, `bar_z` stay model outputs)
    - freeze non-Q parameters during Q stage
    - trainable scope controlled by `q_pretrain_trainable_scope`:
      - `q_head_only` (default, strict freeze)
      - `q_path` (`share_layer + q_head`)
  - Q-shape regularization:
    - `dQ/dz > 0`
    - `dQ/db > 0` on low-`b` region
    - `dQ/db < 0` on high-`b` region
  - Additional diagnostics in loss dict:
    - `q_physics`
    - `q_pretrain_mode`, `q_freeze_mode`

### Policy/Value architecture
- `PolicyValueModel` is now a compatibility wrapper around two independent blocks:
  - `QModel`: old-debt pricing only
  - `PVBPModel`: `bp0 / bpI / V0 / VI / bar_i_cond` and derived `P / bar_z / bp`
- Old checkpoints with monolithic `shared_model.* / combined_model.*` keys are treated as legacy and skipped by split-aware loaders.

### Episode flow
- Episode 0:
  1) Sample-based SDF/FC1 training (`build_sdf_fc1_df`)
  2) Sample-based Policy/Value training (`build_policy_value_df`)
  3) FC2 training using firm-level aggregation and Policy/Value outputs
- Episode >= 1:
  - Use `SimulateTS` to generate firm-time series, then train SDF/FC1, Policy/Value, FC2

### Batch builders
- `_create_sdf_batches_from_macro_df`: macro cross-section for SDF/FC1
- `_create_firm_batches_from_df`: firm triplets for Policy/Value losses
- `_create_fc2_batches`: parent/children firm groups + K for FC2

## trainer.py
`Trainer` orchestrates multi-episode training, checkpoints, and callbacks.

## scheduler.py
Learning rate and loss-weight schedulers used by `Episode`.

## gradient_utils.py
Gradient clipping and NaN protection.
