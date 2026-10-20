"""Strict, CPU-only source certification and cross-run learning-state transfer.

No model construction, optimizer step, production simulation or Slurm call.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from numpy.core.multiarray import _reconstruct, scalar
from analysis.nested_equilibrium_utils import tensor_hash, stable_json_hash, atomic_write_json, atomic_torch_save
from data.finite_direct_sampling import file_hash

SOURCE_COMMIT = '160a4d18ab410dceb75b46f014dc33b5296526a2'
SOURCE_SHA = '6899b452c14a04a3a3b1837a8903dbb21249ec08d534293ef50046f99f2bc585'
SOURCE_STAGE = 'outer_003/outer_commit'
RUN_ID = 'finite_direct_large_continuation_outer4_11_20261020'
EXPECTED_HASHES = dict(
    firm='03af1fbfb6ca413457f2e24b39e57316e79e8280c172925f8caef2cfe63a4e66',
    fc1_c='101521ab1ebce4c15b90160a720527e0dbbb8a34a0b399e4f608f082c2d981a0',
    fc1_k='0165523b9b205349b6343fbf9e976acbdb38714d0c5cfe10fa9ce9f53156705d',
    sdf='73e4de1bf0abb10066f3f7b1a8e8170ab7b989d0e7cd900fe3a5c1ffbe4f6a8e',
    wealth='b8d66ca7bf3f640d98f7677abc06bd35a6865989498a3e1ffd096cc5301b3b78')


def safe_load(path):
    # Only numeric NumPy containers needed by saved RNG state, never arbitrary
    # pickle globals. No weights_only=False fallback.
    types = [_reconstruct, scalar, np.ndarray, np.dtype]
    types += [type(np.dtype(t)) for t in ('uint32', 'float64', 'float32', 'int64')]
    with torch.serialization.safe_globals(types):
        return torch.load(path, map_location='cpu', weights_only=True)


def state_digest(value):
    """Typed recursive digest, including exact tensor bytes and dictionary keys."""
    h = hashlib.sha256()
    def visit(v):
        h.update(type(v).__name__.encode())
        if torch.is_tensor(v):
            h.update(tensor_hash([('value', v)]).encode())
        elif isinstance(v, np.ndarray):
            h.update(str(v.dtype).encode()); h.update(str(v.shape).encode()); h.update(v.tobytes())
        elif isinstance(v, dict):
            for key in sorted(v, key=lambda k: (type(k).__name__, repr(k))):
                visit(key); visit(v[key])
        elif isinstance(v, (list, tuple)):
            h.update(str(len(v)).encode())
            for item in v: visit(item)
        elif isinstance(v, np.generic):
            h.update(v.tobytes())
        elif isinstance(v, (str, int, float, bool, type(None))):
            h.update(repr(v).encode())
        else:
            raise ValueError(f'unsupported state type: {type(v)}')
    visit(value)
    return h.hexdigest()


def component_hashes(models):
    specs = [('firm', 'policy_value', ''), ('sdf', 'sdf_fc1', 'sdf_model.'),
             ('wealth', 'sdf_fc1', 'value_model.'), ('fc1_c', 'sdf_fc1', 'fc1_model.FC1_C.'),
             ('fc1_k', 'sdf_fc1', 'fc1_model.FC1_K.')]
    result = {}
    for label, module, prefix in specs:
        values = [(k[len(prefix):], v) for k, v in models[module].items() if k.startswith(prefix)]
        if not values: raise ValueError(f'missing model {label}')
        if any(not torch.isfinite(v).all() for _, v in values): raise ValueError(f'nonfinite model {label}')
        result[label] = tensor_hash(sorted(values))
    return result


def certify_payload(payload, *, expected_hashes=EXPECTED_HASHES):
    if payload.get('stage') != SOURCE_STAGE: raise ValueError('source must be committed Outer 3')
    if payload.get('contract', {}).get('source_commit') != SOURCE_COMMIT:
        raise ValueError('wrong source commit')
    committed = [i for i in range(4) if f'outer_{i:03d}/outer_commit' in payload.get('completed', [])]
    if committed != [0, 1, 2, 3] or payload['completed'][-1] != SOURCE_STAGE:
        raise ValueError('source is not final four-Outer state')
    backend = payload['backend']; tx = backend['transaction']
    measured = component_hashes(tx['models'])
    if measured != expected_hashes: raise ValueError('source model hashes do not match')
    opt = tx['optimizers']; runtime = tx['episode_runtime']; fresh = backend['fresh']
    if set(opt) != {'sdf_fc1', 'policy_value', 'fc2', 'fc1_conditional'}:
        raise ValueError('missing persistent optimizer')
    optimizer_audit = {}
    for name, value in opt.items():
        if not value.get('param_groups'): raise ValueError(f'missing optimizer groups: {name}')
        if name in ('sdf_fc1', 'fc1_conditional'):
            if not value.get('state'): raise ValueError(f'missing persistent optimizer momentum: {name}')
            for st in value['state'].values():
                if not {'step', 'exp_avg', 'exp_avg_sq'} <= set(st) or float(st['step']) <= 0:
                    raise ValueError(f'incomplete Adam state: {name}')
                if st['exp_avg'].shape != st['exp_avg_sq'].shape: raise ValueError('moment shape mismatch')
        optimizer_audit[name] = dict(sha256=state_digest(value), state_entries=len(value['state']),
            steps=sorted({float(s['step']) for s in value['state'].values() if 'step' in s}),
            parameter_group_shapes=[len(g['params']) for g in value['param_groups']],
            semantics='PERSISTENT_OPTIMIZER' if name in ('sdf_fc1','fc1_conditional') else 'UNUSED_EPISODE_OPTIMIZER')
    al = runtime.get('sdf_al_state', {})
    if set(al) != {'lambda_mean_low', 'lambda_mean_high', 'lambda_var_high', 'rho'}:
        raise ValueError('missing complete AL state')
    if not all(np.isfinite(v) for v in al.values()) or al['rho'] <= 0: raise ValueError('invalid AL')
    if set(runtime.get('lr_schedulers', {})) != {'sdf_fc1','policy_value','fc2'}:
        raise ValueError('missing scheduler state')
    if runtime.get('sdf_training_phase') != 'sdf_true_only': raise ValueError('wrong SDF phase')
    if set(tx.get('rng_state', {})) != {'python','numpy','torch','cuda'}:
        raise ValueError('missing source RNG')
    if fresh.get('bank') is None or fresh.get('pair_rng') is None: raise ValueError('missing fresh shock/pair RNG')
    if not torch.is_tensor(fresh['pair_rng']) or fresh['pair_rng'].numel() == 0: raise ValueError('empty pair RNG')
    if backend['state'].get('sdf_epoch') != 400 or fresh.get('sdf_fc1_step_count',0) <= 0:
        raise ValueError('source SDF epoch/counters missing')
    return dict(status='PASS', source_job_id=701387, source_terminal_outer=3,
        source_terminal_stage=SOURCE_STAGE, source_git_commit=SOURCE_COMMIT,
        completed_outer_indices=committed, model_hashes=measured, optimizers=optimizer_audit,
        scheduler_hash=state_digest(runtime['lr_schedulers']), source_al_state=al,
        source_al_hash=state_digest(al), episode_runtime_hash=state_digest(runtime),
        source_rng_hash=state_digest(tx['rng_state']), source_fresh_hash=state_digest(fresh),
        source_sdf_epoch=400, source_sdf_optimizer_steps=fresh['sdf_fc1_step_count'],
        firm_optimizer_semantics='P/Q Adam/LBFGS STAGE_LOCAL_OPTIMIZER; BP follows production stage creation',
        production_optimizer_steps=0)


def certify_source(root):
    root = Path(root); pointer = json.loads((root/'transactions/latest.json').read_text())
    if pointer['checkpoint'] != 'checkpoint_00423.pt' or pointer['sha256'] != SOURCE_SHA:
        raise ValueError('wrong latest pointer')
    path = root/'transactions'/pointer['checkpoint']
    if file_hash(path) != SOURCE_SHA: raise ValueError('source checkpoint SHA mismatch')
    payload = safe_load(path); audit = certify_payload(payload)
    summary = json.loads((root/'final_summary.json').read_text())
    if summary.get('FOUR_OUTER_TRAINING_COMPLETED') is not True or summary['final_hashes'] != audit['model_hashes']:
        raise ValueError('source completion manifest mismatch')
    other = safe_load(root/'outer_003/outer_complete.pt')
    # Serialization bytes differ (envelopes differ); actual learning state must match.
    for name in ('transaction','fresh','state'):
        if state_digest(other[name]) != state_digest(payload['backend'][name]):
            raise ValueError(f'outer_complete/StageJournal {name} mismatch')
    audit.update(source_terminal_checkpoint_sha256=SOURCE_SHA,
        outer_complete_sha256=file_hash(root/'outer_003/outer_complete.pt'),
        outer_complete_learning_state_parity='PASS')
    return payload, audit


def extract_learning_state(payload):
    certify_payload(payload)
    backend = payload['backend']
    return dict(transaction=deepcopy(backend['transaction']), fresh=deepcopy(backend['fresh']),
                source_sdf_epoch=backend['state']['sdf_epoch'], source_outer=3,
                source_economic_contract=deepcopy({k:payload['contract'][k] for k in ('economic','hyperparameters')}),
                source_committed_hashes=component_hashes(backend['transaction']['models']))


def new_backend_state(learning):
    # Deliberately excludes ALL old paths, dataset manifests and stage progress.
    return dict(data={}, firm_banks={}, fc1_progress={}, sdf_epoch=learning['source_sdf_epoch'],
        outer_metrics={'3':dict(committed_hashes=learning['source_committed_hashes'])},
        continuation_source=dict(job=701387, outer=3, stage=SOURCE_STAGE))


def approval_binding(plan, commit, sampling_dir, initial_checkpoint, source_manifest):
    signature_path=Path(sampling_dir)/'sampling_signature.json'
    signature=json.loads(signature_path.read_text())
    source=json.loads(Path(source_manifest).read_text())
    return dict(schema_version=3, schema_name='CONTINUATION_APPROVAL_V3', source_commit=commit,
        plan_fingerprint=stable_json_hash(plan), sampling_signature_fingerprint=signature['fingerprint'],
        sampling_signature_sha256=file_hash(signature_path), initial_checkpoint_sha256=file_hash(initial_checkpoint),
        continuation_initial_checkpoint_sha256=file_hash(initial_checkpoint),
        source_manifest_sha256=file_hash(source_manifest), source_job_id=701387,
        source_terminal_outer=3,source_terminal_stage=SOURCE_STAGE,
        source_terminal_checkpoint_sha256=source['source_terminal_checkpoint_sha256'],
        start_outer=4,end_outer=11,train_parents=8192,validation_parents=2048)


def require_continuation_approval(approval, expected):
    if approval.get('approved') is not True or any(approval.get(k) != v for k,v in expected.items()):
        raise ValueError('invalid continuation approval: exact source/state/plan/sampling binding required')


def verify_optimizer_restored(saved, optimizer):
    """Exact saved momentum/groups; permit only documented new Torch metadata.

    Torch 2.10 adds decoupled_weight_decay to Adam(W) state loaded from 2.5.1.
    It must equal the optimizer class' existing algorithm, not change it.
    """
    current=optimizer.state_dict()
    if state_digest(saved['state'])!=state_digest(current['state']): raise ValueError('optimizer momentum/step changed')
    if len(saved['param_groups'])!=len(current['param_groups']): raise ValueError('optimizer groups changed')
    extra=[]
    for source,live in zip(saved['param_groups'],current['param_groups']):
        if any(k not in live or state_digest(v)!=state_digest(live[k]) for k,v in source.items()):
            raise ValueError('source optimizer group/LR/order changed')
        added=set(live)-set(source)
        if added-{'decoupled_weight_decay'}: raise ValueError('unreviewed optimizer metadata migration')
        if 'decoupled_weight_decay' in added:
            expected=isinstance(optimizer,torch.optim.AdamW)
            if live['decoupled_weight_decay'] is not expected: raise ValueError('Adam weight decay semantics changed')
        extra.append(sorted(added))
    return dict(source_state_hash=state_digest(saved),momentum_and_groups='PASS',added_runtime_metadata=extra)


def verify_economic_contract(source, current):
    # Device placement is a recorded runtime contract, not calibration/loss.
    # No other hyperparameter is exempted from exact equality.
    old=deepcopy(source);new=deepcopy(current)
    old_device=old['hyperparameters'].pop('device',None)
    new_device=new['hyperparameters'].pop('device',None)
    if state_digest(old)!=state_digest(new):raise ValueError('economic/hyperparameter drift from certified source')
    return dict(economic_parity='PASS',source_device=old_device,runtime_device=new_device)
