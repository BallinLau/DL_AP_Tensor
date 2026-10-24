"""Equilibrium convergence evaluator core package.

Exposes the evaluation-only checkpoint loader (with parameter-hash guards and
semantic records) and the fixed ``evaluation_bank_v1`` builders/persistence.
"""

from .checkpoint_io import (
    EvalCheckpoint,
    EvalCheckpointError,
    check_semantic_compatibility,
    compute_param_hashes,
    count_optimizer_steps,
    load_eval_checkpoint,
    model_state_sha256,
    param_hash_guard,
    semantic_record,
    state_dict_sha256,
)
from .eval_bank import (
    CheckpointTransition,
    EvalBankError,
    build_bank_a,
    build_bank_b,
    build_bank_c,
    build_bank_d,
    build_bank_e,
    build_transition_for_checkpoint,
    compute_config_snapshot_hash,
    content_sha256,
    derive_macro_anchors,
    hash_split_lineages,
    load_bank,
    pair_firm_parent_child,
    pair_macro_parent_child,
    save_bank,
    select_parent_view_rows,
    shock_bank_sha256,
)

__all__ = [
    "CheckpointTransition",
    "EvalBankError",
    "EvalCheckpoint",
    "EvalCheckpointError",
    "build_bank_a",
    "build_bank_b",
    "build_bank_c",
    "build_bank_d",
    "build_bank_e",
    "build_transition_for_checkpoint",
    "check_semantic_compatibility",
    "compute_config_snapshot_hash",
    "compute_param_hashes",
    "content_sha256",
    "count_optimizer_steps",
    "derive_macro_anchors",
    "hash_split_lineages",
    "load_bank",
    "load_eval_checkpoint",
    "model_state_sha256",
    "pair_firm_parent_child",
    "pair_macro_parent_child",
    "param_hash_guard",
    "save_bank",
    "select_parent_view_rows",
    "semantic_record",
    "shock_bank_sha256",
    "state_dict_sha256",
]
