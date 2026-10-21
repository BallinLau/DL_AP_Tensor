"""CPU-only certification of the committed Job702636 Outer11 source.

Content identity, not the deployment directory, is the approval boundary.
Never constructs an optimizer or silently picks one of two disagreeing states.
"""
from pathlib import Path
import json
from analysis.finite_direct_continuation_state import safe_load, state_digest
from analysis.nested_equilibrium_utils import atomic_torch_save
from data.finite_direct_sampling import file_hash, state_hash
from data.finite_direct_dynamic_empirical import load_bundle

SOURCE_COMMIT = '97a018118e917fc5bc5d6b7ed3dd662890560595'
SOURCE_SHA = '9ad7610a1851d06d80c8b2d08052a01532923f7a2808e3a006f5538b404aa94d'
SOURCE_STAGE = 'outer_011/outer_commit'
OPERATOR = 'finite_direct_parent_exit_before_child_v2'
CONTRACT = 'H50_CERTIFIED_OUTER11_AND_IMMUTABLE_BANK_V2'
FIXED_VALIDATION_SHA = 'b7be82344320aecf75bf7c57a1580e28b2cd8282f1426bc1363fe5a8dceddab8'
FIXED_VALIDATION_SCHEDULE_SHA = '686a684ed35164050d1cba1dcd93eec7119b3b021445db86fed4031c3944b4b9'


def verify_fixed_validation(sampling_dir):
    for name, expected in (
        ('validation_initial_bank.pt', FIXED_VALIDATION_SHA),
        ('validation_parent_sampling_schedule.json', FIXED_VALIDATION_SCHEDULE_SHA)):
        if file_hash(Path(sampling_dir)/name) != expected:
            raise ValueError('certified fixed Validation changed: '+name)


def certify(checkpoint, journal_checkpoint, pointer, *, source_job, artifact_dir=None):
    if source_job != 702636:
        raise ValueError('wrong source Job')
    digest = file_hash(checkpoint)
    if digest != SOURCE_SHA:
        raise ValueError('wrong Outer11 checkpoint SHA')
    pointer = json.loads(Path(pointer).read_text())
    if file_hash(journal_checkpoint) != pointer['sha256']:
        raise ValueError('source journal pointer SHA mismatch')
    # A local filename may differ, but the declared remote pointer must be a
    # basename. Its SHA binds the actual committed file across deployments.
    if Path(pointer['checkpoint']).name != pointer['checkpoint']:
        raise ValueError('invalid journal checkpoint reference')
    journal = safe_load(journal_checkpoint)
    learning = safe_load(checkpoint)
    if journal['stage'] != SOURCE_STAGE or journal['completed'][-1] != SOURCE_STAGE:
        raise ValueError('source not committed Outer11')
    if journal['contract']['source_commit'] != SOURCE_COMMIT:
        raise ValueError('wrong source commit')
    binding = journal['contract']['approval_binding']
    if binding.get('economic_operator_version') != OPERATOR:
        raise ValueError('wrong source economic operator')
    if state_digest(learning) != state_digest(journal['backend']):
        raise ValueError('STOP_AND_REPORT: Outer11 export differs from committed transaction')
    tx = learning['transaction']; runtime = tx['episode_runtime']
    for key in ('models', 'optimizers', 'episode_runtime', 'rng_state'):
        if not tx.get(key):
            raise ValueError('missing complete learning state: ' + key)
    if not runtime.get('lr_schedulers') or not runtime.get('sdf_al_state'):
        raise ValueError('missing scheduler/AL state')
    if set(tx['rng_state']) != {'python', 'numpy', 'torch', 'cuda'}:
        raise ValueError('missing global RNG snapshot')
    if learning['fresh'].get('pair_rng') is None or 'bank' not in learning['fresh']:
        raise ValueError('missing fresh-pair state')
    if any(not {'state', 'param_groups'} <= set(tx['optimizers'].get(k, {})) for k in ('policy_value', 'sdf_fc1', 'fc1_conditional', 'fc2')):
        raise ValueError('missing persistent optimizer state')
    if any(not tx['optimizers'][k]['state'] for k in ('sdf_fc1','fc1_conditional')):
        raise ValueError('missing persistent SDF/FC1 momentum')
    if set(tx['models']) != {'policy_value','sdf_fc1','fc2'}:
        raise ValueError('incomplete model set')
    if any(not tx['models'][k] for k in tx['models']):
        raise ValueError('empty model parameters')
    state = learning['state']
    if state['active_dynamic_outer'] != 11:
        raise ValueError('wrong active dynamic Outer')
    if any(f'outer_{o:03d}/outer_commit' not in journal['completed'] for o in range(4, 12)):
        raise ValueError('missing completed source Outer')
    record = dict(state['dynamic_banks']['11'])
    # Optional local mirror: paths are relocatable, SHA and state identities
    # remain those of the actual committed source record.
    if artifact_dir is not None:
        record = dict(record, **{k: dict(record[k], path=str(Path(artifact_dir)/Path(record[k]['path']).name))
                                  for k in ('pool', 'bank', 'schedule')})
    pool, bank, schedule = load_bundle(record)
    if record['outer'] != 11 or schedule['split'] != 'train':
        raise ValueError('source dynamic bank is not Outer11 Train')
    audits = dict(source_checkpoint_audit=dict(source_job=source_job, source_commit=SOURCE_COMMIT,
        source_stage=SOURCE_STAGE, operator=OPERATOR, checkpoint_sha256=digest,
        journal_sha256=pointer['sha256'], complete_learning_state_equal=True,
        completed_outers=list(range(4,12)), verified=True),
        source_optimizer_al_audit=dict(verified=True,
            optimizer_state_hashes={k:state_digest(v) for k,v in tx['optimizers'].items()},
            optimizer_state_counts={k:len(v['state']) for k,v in tx['optimizers'].items()},
            scheduler_hash=state_digest(runtime['lr_schedulers']), al_state=runtime['sdf_al_state'],
            rng_hashes={k:state_digest(v) for k,v in tx['rng_state'].items()},
            fresh_hash=state_digest(learning['fresh']), optimizer_steps_performed=0),
        source_dynamic_bank_audit=dict(verified=True, source_bank_outer=11,
            bank_version=bank['bank_version'], source_pool_fingerprint=pool['fingerprint'],
            bank_sha256=record['bank']['sha256'], bank_state_hash=bank['fingerprint'],
            schedule_sha256=record['schedule']['sha256'], pool_sha256=record['pool']['sha256'],
            initial_bank_construction_rule='REUSE_CERTIFIED_OUTER11_BANK_NOT_INDEPENDENT_RESAMPLE',
            independent_new_sample=False, new_bank_state_hash=bank['fingerprint']))
    return learning, bank, schedule, pool, audits


def save_bank_reference(root, bank, schedule, *, outer, pool_version):
    """One immutable artifact per exact bank/schedule; snapshots hold references."""
    value = dict(bank=bank, schedule=schedule, outer=outer, pool_version=pool_version)
    digest = state_digest(value)
    path = Path(root)/'immutable_artifacts'/f'train_bank_{digest}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if state_digest(safe_load(path)) != digest:
            raise ValueError('immutable Train Bank corruption')
    else:
        atomic_torch_save(path, value)
    return dict(path=str(path), sha256=file_hash(path), state_digest=digest,
                outer=outer, bank_version=bank.get('bank_version'), pool_version=pool_version)


def load_bank_reference(record):
    if file_hash(record['path']) != record['sha256']:
        raise ValueError('Train Bank reference SHA mismatch')
    value = safe_load(record['path'])
    if state_digest(value) != record['state_digest'] or value['outer'] != record['outer'] or value['pool_version'] != record['pool_version']:
        raise ValueError('Train Bank Outer/pool mismatch')
    if state_hash(value['bank']['state']) != value['bank']['fingerprint']:
        raise ValueError('Train Bank state corruption')
    return value['bank'], value['schedule']
