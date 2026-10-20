# Continuation contract

Initialization is CROSS_RUN_CONTINUATION_FROM_COMMITTED_OUTER3, not old-run resume.
Start with the authoritative StageJournal checkpoint_00423.pt from Job 701387.
The source terminal stage is outer_003/outer_commit. Eight NEW full Outers are
numbered outer_004 through outer_011 in real stage IDs, not relabeled 0-7.
No Outer 0-3 optimizer block is scheduled. Outer3/new-bank baseline is forward-only.
New sampling, source manifest, approval v3, datasets and journal identify a new run.
Default CLI is plan-only and does not construct a ProductionBackend.
Only explicit --execute-approved plus a fully matching approved v3 can train.
No approved=true artifact was created; no Slurm command was executed.
