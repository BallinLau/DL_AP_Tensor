"""Final review only: no production optimizer, GPU or Slurm execution."""
import ast
import contextlib
import io
import inspect
import json
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
import pytest
import torch
from analysis.run_finite_direct_large_continuation import main,parse_args,training_plan
from analysis.finite_direct_dynamic_approval import approval_binding,require_approval
from analysis.nested_equilibrium_utils import atomic_write_bytes,atomic_write_json,stable_json_hash

ROOT=Path(__file__).resolve().parents[1]


def test_actual_initialization_body_archives_v4_bytes_and_no_v3(tmp_path):
    # Execute the actual nested metadata writer, not a duplicated test helper.
    # The input is deliberately approved=false: no approved=true file exists.
    tree=ast.parse(inspect.getsource(main))
    body=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='history_init')
    source=tmp_path/'denied_input.json';raw=b'{"approved":false,"schema_version":4}\n';source.write_bytes(raw)
    args=SimpleNamespace(output=tmp_path/'run',approval=source,
        initial_checkpoint=tmp_path/'initial.pt',source_manifest=tmp_path/'manifest.json')
    args.source_manifest.write_text('{}')
    scope=dict(args=args,plan={'review':'only'},source={},
        atomic_write_bytes=atomic_write_bytes,atomic_write_json=atomic_write_json,
        install_historical_baseline=lambda *a,**k:{'optimizer_steps':0})
    exec(compile(ast.Module(body=[body],type_ignores=[]),'<actual history_init>','exec'),scope)
    assert scope['history_init']()['optimizer_steps']==0
    assert (args.output/'metadata/approval_v4.json').read_bytes()==raw
    assert not (args.output/'metadata/approval_v3.json').exists()


def test_plan_only_avoids_backend_and_filesystem_artifacts(tmp_path,monkeypatch):
    import analysis.run_finite_direct_large_continuation as runner
    monkeypatch.setattr(runner,'ContinuationBackend',lambda *a,**k:pytest.fail('backend forbidden'))
    for cls in (torch.optim.Adam,torch.optim.AdamW,torch.optim.LBFGS):
        monkeypatch.setattr(cls,'step',lambda *a,**k:pytest.fail('optimizer forbidden'))
    out=tmp_path/'not_created';stream=io.StringIO()
    with contextlib.redirect_stdout(stream):
        main(['--sampling-dir','missing','--initial-checkpoint','missing','--source-manifest','missing','--output',str(out)])
    payload=json.loads(stream.getvalue())
    assert payload['CURRENT_PHASE']=='WAITING_FOR_USER_REVIEW'
    assert payload['FORMAL_TRAINING_STARTED'] is payload['SLURM_JOB_SUBMITTED'] is False
    assert not out.exists()


@pytest.mark.parametrize('schema',[2,3])
def test_legacy_approval_rejected_before_backend_or_checkpoint_load(tmp_path,monkeypatch,schema):
    import analysis.run_finite_direct_large_continuation as runner
    # In-memory legacy true flag, but disk contains no approved=true artifact.
    denied=tmp_path/'denied.json';denied.write_text('{"approved":false}')
    original=json.loads
    monkeypatch.setattr(runner.json,'loads',lambda value,*a,**k:dict(approved=True,schema_version=schema) if value==denied.read_text() else original(value,*a,**k))
    monkeypatch.setattr(runner,'ContinuationBackend',lambda *a,**k:pytest.fail('backend reached'))
    with pytest.raises(ValueError,match='v4 approval'):
        main(['--sampling-dir','missing','--initial-checkpoint','missing','--source-manifest','missing',
              '--output',str(tmp_path/'run'),'--approval',str(denied),'--execute-approved'])


def bindings(root):
    for name in ('initial_dynamic_pool.pt','train_initial_bank.pt','validation_initial_bank.pt','initial.pt'):
        (root/name).write_bytes(name.encode())
    (root/'sampling_signature.json').write_text(json.dumps(dict(fingerprint='sampling_original')))
    (root/'source.json').write_text(json.dumps(dict(source_terminal_checkpoint_sha256='source')))
    return lambda:approval_binding(training_plan(),'commit',root,root/'initial.pt',root/'source.json')


@pytest.mark.parametrize('artifact',['initial.pt','sampling_signature.json'])
def test_changed_checkpoint_or_sampling_rejects_original_v4(tmp_path,artifact):
    bind=bindings(tmp_path);expected=bind();approval=dict(expected,approved=True)
    require_approval(approval,expected)
    path=tmp_path/artifact
    if artifact.endswith('.json'):path.write_text(json.dumps(dict(fingerprint='different_sampling')))
    else:path.write_bytes(b'different checkpoint')
    with pytest.raises(ValueError,match='v4 approval'):require_approval(approval,bind())


def test_training_and_dynamic_configuration_unchanged():
    p=training_plan();b=p['budget']
    assert (p['start_outer'],p['end_outer'],p['outers_to_execute'])==(4,11,8)
    assert (p['train_parents'],p['validation_parents'],p['firms'],p['children'])==(8192,2048,200,2)
    assert b==dict(equilibrium_outer_max=8,firm_inner_max=3,p_policy_outer_max=3,
        p_adam_steps=200,p_lbfgs_max_iter=100,q_picard_outer_max=3,q_fit_steps=300,
        policy_distill_max_steps=1000,macro_inner_max=2,fc1_min_epochs=25,fc1_max_epochs=40,
        fc1_patience=10,sdf_chunks=2,sdf_epochs_per_chunk=50)
    assert p['operator']=='finite_direct_parent_exit_before_child_v2'
    assert p['root_search'] is p['fc2'] is False
    assert p['dynamic_sampling']['recent_count']==2867 and p['dynamic_sampling']['archive_count']==1229


def test_slurm_syntax_and_real_argparse_match():
    path=ROOT/'slurm/run_finite_direct_large_continuation_outer4_11_a800.slurm'
    subprocess.run(['bash','-n',str(path)],check=True,capture_output=True)
    script=path.read_text()
    assert '#SBATCH --gres=gpu:a800:1' in script
    assert 'metadata/approval_v4.json' in script
    assert 'CONTINUATION_APPROVAL_V4_DYNAMIC_EMPIRICAL' in script
    command=script[script.index('python analysis/run_finite_direct_large_continuation.py'):].replace('\\\n',' ')
    values=dict(REPO_DIR='/repo',SAMPLING_DIR='/sampling',INITIAL_CHECKPOINT='/initial.pt',
        SOURCE_MANIFEST='/manifest.json',RUN_ROOT='/run',APPROVAL_FILE='/future_v4.json')
    for key,value in values.items():command=command.replace('${'+key+'}',value).replace('$'+key,value)
    command=command.replace('"${extra[@]}"','')
    args=parse_args(shlex.split(command)[2:])
    assert args.execute_approved and args.approval==Path('/future_v4.json') and not args.resume
    assert (args.parent_batch_size,args.fit_parent_batch_size,args.firm_batch_size)==(128,128,512)
    assert args.device=='cuda:0'
    assert parse_args(shlex.split(command)[2:]+['--resume']).resume


def test_historical_v3_review_generator_cannot_publish_dynamic_material(tmp_path):
    from analysis.build_finite_direct_continuation_review import write_review
    output=tmp_path/'must_not_exist'
    with pytest.raises(RuntimeError,match='Historical Approval v3'):
        write_review(tmp_path/'no_prepared',output,'no optimizers')
    assert not output.exists()
