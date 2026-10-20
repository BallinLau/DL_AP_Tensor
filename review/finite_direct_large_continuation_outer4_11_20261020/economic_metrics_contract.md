# Economics and comparability

Reuse finite_direct_outer_metrics.py, Train/Validation x BETA/EMPIRICAL/ALL.
G is the saved normalized prediction residual, not a solved h* root; signed/raw
G scales are retained. Counts and valid fractions always accompany conditional
metrics. Gate-weighted default/investment rates are distinct from hard survivor
counts. No undefined economic label is imputed; failed rows remain in data audit.
FC1 one-child eligibility is not SDF two-child eligibility.
FC1 pre-fit and post-fit on unchanged labels are in fc1_fit_history.csv; D0/D1/D2
and post-SDF resimulation are distinct in outer_macro_economic_history.csv.
LnK level R2, Delta LnK R2, MAE and sample counts remain separate.

Historical Job701387 Outer0-3 metrics stay read-only in historical_baseline.
They have the OLD Validation bank. Only NEW-bank Outer3 baseline vs Outer4-11
share the new fixed Validation and CRN. Never concatenate old means and claim
same-sample improvement. All post-SDF economics come from a dedicated fresh
Validation-only cache with all current network hashes, no optimizer.
