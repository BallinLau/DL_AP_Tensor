# Persistent learning state transfer

The certified source transaction supplies every model, optimizer (including
fc1_conditional), scheduler, AL multiplier/rho, global RNG and SDF runtime counters.
Actual source Adam states: FC1 6728 steps, SDF 12800 steps; global SDF epoch 400.
All optimizer moment tensors and original param-group fields are checked after load.
The Mac's Torch 2.10 adds decoupled_weight_decay metadata to THU Torch 2.5.1 Adam(W)
checkpoints. Only that extra key is permitted, and only if it agrees with the
existing optimizer class. Original LR, parameter order, step, exp_avg and
exp_avg_sq must remain identical. Other changes fail closed. No moments are reset.

P/Q Adam and LBFGS are stage-local in the existing solver; their local optimizer
restart at a new committed block is unchanged production semantics. Episode's
empty policy_value/fc2 optimizer states are copied, not invented as learned state.
The existing production BP stage's own optimizer creation rule is unchanged.

Old backend data, Firm-bank paths, FC1 progress and transaction completion list
are NOT restored into the new mutable backend. Source epoch/learning counters
are restored; previous committed Macro hashes seed the Outer4 boundary check.
No temporary weights-only pseudo-checkpoint is used to initialize production.
The minimal shared setup hook accepts certified model state, then full state
is restored and verified before any stage can execute.
