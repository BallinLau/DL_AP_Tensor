# Fresh shocks and random state

The archived initial state preserves source bank eps [5120,16,1], source pair RNG,
global RNG, epoch=400 and optimizer step=12800. That old bank is not indexed as
8192/2048 new parents and is never doubled/repeated. After full learning-state
restoration, only active bank storage/capacity/key is invalidated for the new data.
At first SDF epoch 400, the unchanged production constructor creates a fresh bank
for current contiguous complete-pair indices (Train then disjoint Validation).
Production key is episode_id/global epoch/capacity/device/dtype, seed follows
the existing base_seed + 10000*episode_id + epoch rule, and pair reseeding uses
the existing seed+7919 rule. New namespace is the locked sampling signature.
Every committed chunk records actual range, key, bank version, epoch and RNG identity.

This is not bitwise continuation of a hypothetical unchanged old data stream.
Once the new run starts, its committed eps/pair/global RNG restore exactly.
CPU test verifies 10240 capacity at epoch400, no repeated-half bank, and exact
pair draws after fresh_state/restore_fresh roundtrip. CUDA RNG bytes are source-
audited but CUDA execution itself is intentionally not run during review.
