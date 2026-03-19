# data

Data generation and simulation utilities. This folder produces firm-level panels for
SDF/FC1/Policy-Value training and tree-structured time-series for later episodes.

## sample.py
`Sample` generates firm-level training DataFrames with a strict triplet structure.

### Core outputs
- `build_sdf_fc1_df()`:
  - Columns: `path`, `branch`, `x_t`, `x_t1`, `Hatcf_t`, `LnKF_t`
  - Each path has `branch_num` rows with the same `x_t/hatcf_t/lnkf_t` and
    different `x_t1` draws.
- `build_policy_value_df()`:
  - Firm-level panel with parent + branches per `(path, ID)`.
  - Columns include: `b, z, ETA, i, x, Hatcf, LnKF` plus identifiers.
  - `M` is filled when `sdf_fc1` is provided; otherwise `M` is `None` for children
    and defaults to `1.0` for parents.
- `build_fc2_df()`:
  - For `data_mode='simulate'`, returns `(df_firm, df_macro)` where `df_firm` is
    filtered to `branch==0` and `df_macro` provides aggregated path-level targets.

### Triplet structure (default `branch_num=2`)
For each firm `(path, ID)`, rows are:
- `t` (parent)
- `t+1_0`, `t+1_1`, ... (branches)

The training code assumes this structure for pairing parent with children.

## simulate_ts.py
`SimulateTS` generates tree-structured firm time-series once models are available.

Key features:
- Branching tree per time step (`branch_num`)
- Entry/exit mechanics
- Policy/Value-driven evolution of `b` and `K`
- Macro aggregation per node (C, K, Hatc, LnK)

Outputs:
- `df_firm`: firm-level panel with `path, t, branch, ID` and firm state columns
- `df_macro`: macro-level panel with `path, t, branch` aggregates

## data_utils.py
Sampling and utilities:
- `sample_ar1`, `sample_stationary_ar1`, `sample_uniform`, `sample_bernoulli`
- `generate_initial_macro_proxy`, `generate_firm_states`
- `compute_quantile_features(values, n_quantiles)` used by FC2

## Column conventions
Firm-state order is fixed:
```
(b, z, ETA, i, x, Hatcf, LnKF)
```
All macro quantities stored in DataFrames are in physical scale.
