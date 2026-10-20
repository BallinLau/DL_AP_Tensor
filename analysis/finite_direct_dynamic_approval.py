"""Content-bound v4, never creates an approval instance."""
from pathlib import Path
from analysis.finite_direct_continuation_state import approval_binding as v3_binding
from analysis.nested_equilibrium_utils import stable_json_hash
from data.finite_direct_sampling import file_hash
from data.finite_direct_dynamic_empirical import CONFIG,VERSION
from data.finite_direct_runtime import OPERATOR


def approval_binding(plan,commit,sampling_dir,initial_checkpoint,source_manifest):
    root=Path(sampling_dir)
    result=v3_binding(plan,commit,root,initial_checkpoint,source_manifest)
    result.update(schema_version=4,schema_name='CONTINUATION_APPROVAL_V4_DYNAMIC_EMPIRICAL',
        dynamic_algorithm_version=VERSION,dynamic_configuration=CONFIG,
        dynamic_configuration_fingerprint=stable_json_hash(CONFIG),economic_operator_version=OPERATOR,
        initial_empirical_pool_sha256=file_hash(root/'initial_dynamic_pool.pt'),
        initial_train_bank_sha256=file_hash(root/'train_initial_bank.pt'),
        fixed_validation_bank_sha256=file_hash(root/'validation_initial_bank.pt'),
        beta_empirical='50:50',recent_archive='70:30',source_results_usage='WEIGHTS_AND_LEARNING_STATE_ONLY_NOT_CORRECTED_CONVERGENCE_EVIDENCE')
    return result


def require_approval(approval,expected):
    if expected.get('schema_version')!=4 or approval.get('approved') is not True or any(approval.get(k)!=v for k,v in expected.items()):
        raise ValueError('invalid dynamic v4 approval; old v3 is forbidden')


def prepare_initial_pool(sampling_dir):
    from data.finite_direct_dynamic_empirical import initial_pool
    from analysis.finite_direct_continuation_state import safe_load,state_digest
    from analysis.nested_equilibrium_utils import atomic_torch_save
    root=Path(sampling_dir);path=root/'initial_dynamic_pool.pt'
    expected=initial_pool(safe_load(root/'empirical_template_bank.pt'))
    if path.exists():
        if state_digest(safe_load(path))!=state_digest(expected): raise ValueError('initial pool changed')
    else: atomic_torch_save(path,expected)
    return path
