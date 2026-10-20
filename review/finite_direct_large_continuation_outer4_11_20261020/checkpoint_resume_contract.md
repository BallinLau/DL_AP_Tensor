# New journal and disk retention

Fresh initialization imports certified persistent state into a NEW empty output.
It cannot copy the old latest.json or reuse the old writable journal.
--resume requires the new run's own transactions/latest.json; a nonempty fresh
output, wrong code/data/runtime contract or corrupted pointer fails closed.
P/Q/BP commit at block boundaries; FC1 at full epochs; SDF at chunks.
An uncommitted block restarts from its prior checkpoint; committed IDs never rerun.
Global/source counters are retained; new FC1 patience/data progress is independent.

Retention: KEEP_ALL_TRANSACTIONS_NO_AUTOMATIC_DELETION. About 890 journal files
plus eight completed Outer snapshots, before extra metadata. At current ~3MiB
terminal size this starts around 2.7GiB; growing result histories can increase
that materially. Budget 10-20GiB for checkpoints plus datasets, not a fixed cap.
No pruning is performed, and no Job701387 file is deleted. Future pruning needs
separate review and must preserve an atomically verified latest checkpoint.
