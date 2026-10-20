# Sampling provenance

8192 Train / 2048 Validation Parent, 200 firms/Parent, two child branches.
4096+4096 and 1024+1024 Beta/Empirical Parent (mixture unit is an economy).
Empirical templates are Job 699894 t0 RECORDED_INITIAL_PARENT_DISTRIBUTION,
not t>=1 on-policy captures. Full original templates: 4096 Train and 1024 Validation.
Each is used once in its own split; b is resampled by the original conditional
CDF-by-capital-quartile rule, not by copying full source firm state.
Template source file SHA f616821c7407230eef850a255ea1205a1df47d3b3e659d1b2820cfd810ebf1d3
was checked equal on Mac and THU. Source paths have no train/validation overlap.
Unchanged supports: alpha [1.8,3.2], beta [4.8,7.8], aggregate K [45,85].
New seed 202610204, new signature and fresh banks; old sampling is not overwritten.
Numeric Parent IDs are scoped by run_id/split/economic_time/branch; dataset
provenance explicitly includes run identity as well as all network/CRN hashes.
