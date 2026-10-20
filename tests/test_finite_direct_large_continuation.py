"""Non-optimization continuation tests: no Slurm/GPU/real optimizer step."""
from copy import deepcopy
from pathlib import Path
import inspect
import json
import numpy as np
import pytest
import torch
from analysis.finite_direct_continuation_state import (certify_payload,component_hashes,state_digest,
    extract_learning_state,new_backend_state,require_continuation_approval,SOURCE_COMMIT,SOURCE_STAGE,
    EXPECTED_HASHES,safe_load,verify_optimizer_restored,verify_economic_contract)
from analysis.run_finite_direct_large_continuation import stages,training_plan,drive,main,ContinuationBackend,verify_initial
from analysis.finite_direct_continuation_probes import movement,save_movement
from analysis.finite_direct_outer4_state import StageJournal
from data.finite_direct_continuation_sampling import CONFIG
from data.finite_direct_runtime import verify_cache
from test_finite_direct_outer4_runner import FakeBackend

PREP=Path(__file__).resolve().parents[1]/'cachedir/finite_direct_large_continuation_outer4_11_20261020/prepared_v2'
ACTUAL=PREP.parent/'source/transactions/checkpoint_00423.pt'


@pytest.fixture
def source():
    models={'policy_value':{'w':torch.ones(2)},'sdf_fc1':{p+'w':torch.ones(2)
        for p in ('sdf_model.','value_model.','fc1_model.FC1_C.','fc1_model.FC1_K.')}}
    adam=dict(state={0:dict(step=torch.tensor(4.),exp_avg=torch.ones(2),exp_avg_sq=torch.ones(2))},
              param_groups=[dict(params=[0],lr=.01)])
    blank=dict(state={},param_groups=[dict(params=[0],lr=.01)])
    tx=dict(models=models,optimizers=dict(sdf_fc1=deepcopy(adam),fc1_conditional=deepcopy(adam),
        policy_value=deepcopy(blank),fc2=deepcopy(blank)),episode_runtime=dict(
        sdf_al_state=dict(lambda_mean_low=.1,lambda_mean_high=0.,lambda_var_high=0.,rho=2.),
        lr_schedulers={k:dict(current_step=4) for k in ('sdf_fc1','policy_value','fc2')},
        sdf_training_phase='sdf_true_only'),rng_state=dict(python=1,numpy=2,torch=torch.ones(2),cuda=[torch.ones(2)]))
    return dict(stage=SOURCE_STAGE,contract=dict(source_commit=SOURCE_COMMIT),
        completed=[f'outer_{i:03d}/outer_commit' for i in range(4)],backend=dict(transaction=tx,
        fresh=dict(bank=dict(eps=torch.ones(4,16,1)),pair_rng=torch.ones(16,dtype=torch.uint8),sdf_fc1_step_count=4),
        state=dict(sdf_epoch=400,data={'OLD':'old/path'},probe_bank={'path':'OLD'})))


def certify(source):
    return certify_payload(source,expected_hashes=component_hashes(source['backend']['transaction']['models']))


def test_source_typed_state_and_numeric_loader(source,tmp_path):
    assert certify(source)['status']=='PASS'
    p=tmp_path/'safe.pt';torch.save(source,p)
    assert state_digest(safe_load(p))==state_digest(source)


@pytest.mark.parametrize('field',['stage','commit','completed'])
def test_reject_wrong_source_outer_and_provenance(source,field):
    if field=='stage':source['stage']='outer_002/outer_commit'
    if field=='commit':source['contract']['source_commit']='bad'
    if field=='completed':source['completed']=source['completed'][:-1]
    with pytest.raises(ValueError):certify(source)


@pytest.mark.parametrize('missing',['optimizers','sdf_al_state','lr_schedulers','rng_state','bank','pair_rng'])
def test_reject_weights_only_or_missing_runtime(source,missing):
    b=source['backend'];t=b['transaction']
    if missing in ('bank','pair_rng'):b['fresh'][missing]=None
    elif missing in ('sdf_al_state','lr_schedulers'):t['episode_runtime'].pop(missing)
    else:t.pop(missing)
    with pytest.raises((ValueError,KeyError)):certify(source)


@pytest.mark.parametrize('name',['sdf_fc1','fc1_conditional'])
def test_reject_empty_persistent_momentum(source,name):
    source['backend']['transaction']['optimizers'][name]['state']={}
    with pytest.raises(ValueError,match='momentum'):certify(source)


def test_stage_local_optimizers_are_not_fabricated(source):
    assert certify(source)['optimizers']['policy_value']['state_entries']==0


@pytest.mark.skipif(not ACTUAL.exists(),reason='requires separately certified Job701387 audit fixture')
def test_actual_source_checkpoint_and_all_learning_objects():
    assert ACTUAL.exists(),'Actual source audit fixture must be available for this review'
    p=safe_load(ACTUAL);audit=certify_payload(p)
    assert audit['model_hashes']==EXPECTED_HASHES
    learning=extract_learning_state(p)
    for key in ('transaction','fresh'):
        assert state_digest(learning[key])==state_digest(p['backend'][key])
    new=new_backend_state(learning)
    assert new['data']=={} and new['firm_banks']=={} and new['fc1_progress']=={}
    assert new['sdf_epoch']==400 and new['outer_metrics']['3']['committed_hashes']==EXPECTED_HASHES
    assert 'probe_bank' not in new


@pytest.mark.skipif(not PREP.exists(),reason='requires prepared reviewed continuation artifacts')
def test_actual_prepared_continuation_is_not_weights_only():
    signature=json.loads((PREP/'continuation_sampling/sampling_signature.json').read_text())
    p,s=verify_initial(PREP/'continuation_initial_checkpoint.pt',PREP/'continuation_source_manifest.json',signature['fingerprint'])
    assert p['learning']['transaction']['episode_runtime']['sdf_al_state']==s['source_al_state']
    assert p['learning']['fresh']['bank']['eps'].shape[0]==5120


@pytest.mark.skipif(not PREP.exists(),reason='requires actual CPU constructed 8192/2048 banks')
def test_actual_sampling_size_ratio_templates_and_no_leakage():
    from data.finite_direct_runtime import load_locked_sampling
    banks,schedules,signature=load_locked_sampling(PREP/'continuation_sampling')
    assert (CONFIG.train_parents,CONFIG.validation_parents)==(8192,2048)
    assert signature['config']['firms']==200
    assert signature['fingerprint']!='0afddab4e753063479c5ef36dd4b29bd77eaba8ad259cf1b3fe421ecd440f191'
    ids={}
    for split,n in [('train',8192),('validation',2048)]:
        rows=schedules[split]['parents'];assert banks[split]['state']['b'].shape==(n,200)
        for kind in ('BETA','EMPIRICAL'):assert sum(r['sampling_source']==kind for r in rows)==n//2
        ids[split]={r['empirical_template_id'] for r in rows if r['sampling_source']=='EMPIRICAL'}
        assert len(ids[split])==n//2
        assert all(len(r['child_shock_keys'])==2 and r['child_shock_keys'][0]!=r['child_shock_keys'][1] for r in rows)
    assert not ids['train']&ids['validation']


def test_exact_eight_outer_stage_order_and_real_indices(tmp_path):
    backend=FakeBackend();journal=StageJournal(tmp_path,{'new':'run'},backend)
    drive(backend,journal)
    assert backend.calls[0][:2]==(3,'baseline')
    assert {o for o,k,i in backend.calls if k!='baseline'}==set(range(4,12))
    assert not any(k=='reuse_firm' for o,k,i in backend.calls)
    for o in range(4,12):
        calls=[(k,i) for n,k,i in backend.calls if n==o]
        assert [i['version'] for k,i in calls if k=='data']==[0,1,2]
        assert [i['chunk'] for k,i in calls if k=='SDF']==[0,1]
        assert calls[1][0]=='firm_bank' and calls[-1][0]=='outer_commit'
        assert [k for k,i in calls][-3:]==['post_sdf_validation','cross_block','outer_commit']
    before=len(backend.calls);drive(backend,journal);assert len(backend.calls)==before


def test_budget_and_operator_unchanged():
    p=training_plan();b=p['budget']
    assert (p['start_outer'],p['end_outer'],p['outers_to_execute'])==(4,11,8)
    assert (b['firm_inner_max'],b['p_adam_steps'],b['p_lbfgs_max_iter'],b['q_fit_steps'])==(3,200,100,300)
    assert (b['fc1_min_epochs'],b['fc1_max_epochs'],b['fc1_patience'])==(25,40,10)
    assert b['sdf_chunks']*b['sdf_epochs_per_chunk']==100 and not p['root_search'] and not p['fc2']


def test_plan_only_cannot_construct_production_backend(monkeypatch,capsys):
    import analysis.run_finite_direct_large_continuation as m
    monkeypatch.setattr(m,'ContinuationBackend',lambda *a,**k:pytest.fail('backend constructed'))
    main(['--sampling-dir','MISSING','--initial-checkpoint','MISSING','--source-manifest','MISSING','--output','MISSING'])
    assert json.loads(capsys.readouterr().out)['FORMAL_TRAINING_STARTED'] is False


@pytest.mark.parametrize('field',['source_commit','plan_fingerprint','source_terminal_checkpoint_sha256',
    'continuation_initial_checkpoint_sha256','sampling_signature_sha256','source_manifest_sha256',
    'start_outer','end_outer','train_parents','validation_parents'])
def test_approval_binds_every_input(field):
    expected={k:1 for k in ['source_commit','plan_fingerprint','source_terminal_checkpoint_sha256',
        'continuation_initial_checkpoint_sha256','sampling_signature_sha256','source_manifest_sha256',
        'start_outer','end_outer','train_parents','validation_parents']}
    valid=dict(approved=True,**expected);require_continuation_approval(valid,expected)
    valid[field]='changed'
    with pytest.raises(ValueError):require_continuation_approval(valid,expected)


def test_approval_false_rejected():
    with pytest.raises(ValueError):require_continuation_approval({'approved':False},{})


def test_only_device_is_execution_not_an_economic_relaxation():
    source={'economic':{'gamma':2.},'hyperparameters':{'device':'cuda','sdf_lr':4e-5}}
    current=deepcopy(source);current['hyperparameters']['device']='cpu'
    assert verify_economic_contract(source,current)['economic_parity']=='PASS'
    current['hyperparameters']['sdf_lr']=1e-3
    with pytest.raises(ValueError):verify_economic_contract(source,current)


def test_resume_keeps_committed_state_and_cannot_reuse_old_run(tmp_path):
    a=FakeBackend();j=StageJournal(tmp_path,{'run':'new'},a)
    j.run('outer_004/FC1/epoch1',lambda:a.execute(4,'FC1',{}));snapshot=a.snapshot()
    a.state['AL']=999;torch.rand(10)
    b=FakeBackend();restored=StageJournal(tmp_path,{'run':'new'},b,resume=True)
    assert b.state==snapshot['state'] and torch.equal(torch.get_rng_state(),snapshot['rng'])
    called=[];restored.run('outer_004/FC1/epoch1',lambda:called.append(True));assert called==[]
    with pytest.raises(RuntimeError):StageJournal(tmp_path,{'run':'old'},FakeBackend(),resume=True)
    with pytest.raises(RuntimeError):verify_cache({'signature':'old'},{'signature':'new'})


def test_function_movement_same_inputs_and_two_cycle(tmp_path):
    torch.manual_seed(4);x=torch.randn(8)
    row=movement(x.numpy(),(x+1).numpy(),x.numpy())
    assert row['rms']==pytest.approx(1) and row['two_outer_rms']==0
    assert row['update_direction_cosine']==pytest.approx(-1)
    current=dict(probe_fingerprint='same',firm={'P0':x},fc1={},sdf={})
    previous=deepcopy(current);previous['probe_fingerprint']='other'
    with pytest.raises(ValueError):save_movement(tmp_path,4,current,previous)


def test_post_sdf_uses_dedicated_cache_and_no_optimizer():
    from analysis.run_full_nested_finite_direct_outer4 import ProductionBackend
    body=inspect.getsource(ProductionBackend._post_sdf_validation)
    assert "stage='POST_SDF_VALIDATION_RESIMULATION'" in body
    code='\n'.join(x for x in body.splitlines() if not x.strip().startswith('#'))
    assert "split='validation'" in body and 'self._data(' not in code
    assert '.step(' not in body and 'backward(' not in body


def test_fresh_bank_rebinding_not_repetition():
    body=inspect.getsource(ContinuationBackend.__init__)
    assert '_sdf_shock_bank=None' in body and '_sdf_shock_bank_key=None' in body
    assert 'restore_training_transaction' in body and 'restore_fresh' in body
    assert '.repeat(' not in body


def test_slurm_is_prepared_only_and_existing_production_called():
    root=Path(__file__).resolve().parents[1]
    script=(root/'slurm/run_finite_direct_large_continuation_outer4_11_a800.slurm').read_text()
    assert '#SBATCH --gres=gpu:a800:1' in script and '--execute-approved' in script
    assert '\nsbatch ' not in script and '\nsrun ' not in script
    assert '--source-manifest' in script and 'extra+=(--resume)' in script
    body=inspect.getsource(ContinuationBackend)
    assert 'super()._execute' in body and 'super()._fc1_epoch' in body


@pytest.mark.skipif(not ACTUAL.exists() or not Path('/private/tmp/701387_fixed_probe_bank.pt').exists(),reason='requires actual source forward-only fixtures')
def test_real_checkpoint_cpu_forward_probe_does_not_optimize_or_mutate(monkeypatch):
    import pandas as pd
    from types import SimpleNamespace
    from analysis.run_fc1_self_consistent_macro import setup
    from analysis.finite_direct_continuation_probes import make_macro_probe,evaluate_fixed
    from analysis.nested_equilibrium_utils import capture_rng_state,restore_training_transaction
    from training.inner_fixed_point import InnerFPCollocationBank
    def forbidden(*a,**k):pytest.fail('actual optimizer step in review')
    for cls in (torch.optim.AdamW,torch.optim.Adam,torch.optim.LBFGS):monkeypatch.setattr(cls,'step',forbidden)
    payload=safe_load(ACTUAL)
    models,episode,fc1,_=setup(SimpleNamespace(device='cpu',checkpoint='NOT_LOADED'),
        initial_models=payload['backend']['transaction']['models'])
    tx=payload['backend']['transaction']
    restore_training_transaction(tx,models=models,optimizers={**episode.optimizers,'fc1_conditional':fc1},episode=episode)
    for name,opt in {**episode.optimizers,'fc1_conditional':fc1}.items():
        assert verify_optimizer_restored(tx['optimizers'][name],opt)['momentum_and_groups']=='PASS'
    assert episode._get_sdf_al_state()==tx['episode_runtime']['sdf_al_state']
    probe=safe_load('/private/tmp/701387_fixed_probe_bank.pt')['banks']['validation']
    bank=InnerFPCollocationBank(**{k:v for k,v in probe.items() if k!='fingerprint'})
    frame=pd.read_csv('/private/tmp/701387_old_validation_for_forward_interface.csv')
    ids=sorted(frame.parent_candidate_id.unique())[:16]
    macro=make_macro_probe(frame[frame.parent_candidate_id.isin(ids)])
    backend=SimpleNamespace(models=models,episode=episode)
    before=state_digest({k:v.state_dict() for k,v in models.items()});rng=state_digest(capture_rng_state())
    out=evaluate_fixed(backend,bank,macro);again=evaluate_fixed(backend,bank,macro)
    assert state_digest(out)==state_digest(again)
    assert before==state_digest({k:v.state_dict() for k,v in models.items()})
    assert rng==state_digest(capture_rng_state())
    assert {'M_raw','M_used','W_parent','W_child','M_true_raw'}<=set(out['sdf'])
    # Same production fresh-shock constructor, new bank capacity and source
    # global epoch; no model forward/backward or optimizer is needed here.
    episode._sdf_shock_bank=None;episode._sdf_shock_bank_key=None
    episode._ensure_sdf_bank_capacity(torch.arange(10240),dtype=torch.float32,epoch=400)
    assert episode._sdf_shock_bank.eps.shape==(10240,16,1)
    assert episode._sdf_shock_bank_key[1]==400
    assert not torch.equal(episode._sdf_shock_bank.eps[:5120],episode._sdf_shock_bank.eps[5120:])
    from analysis.run_full_nested_finite_direct_outer4 import fresh_state,restore_fresh
    snapshot=fresh_state(episode)
    original=episode._sdf_shock_bank.sample_pair(torch.arange(16),episode._sdf_pair_generator)
    restore_fresh(episode,snapshot)
    repeated=episode._sdf_shock_bank.sample_pair(torch.arange(16),episode._sdf_pair_generator)
    assert all(torch.equal(x,y) for x,y in zip(original,repeated))


@pytest.mark.skipif(not PREP.exists(),reason='requires historical metrics fixture, not simulated new results')
def test_baseline_only_histories_and_plot_interfaces(tmp_path):
    import pandas as pd
    import analysis.finite_direct_continuation_probes as probes
    source=pd.read_csv(PREP/'historical_baseline/outer_macro_economic_history.csv')
    rows=source[(source.outer==3)&(source.stage=='POST_SDF_VALIDATION_RESIMULATION')].to_dict('records')
    # Interface fixtures only, in pytest tmpdir; never delivered as new results.
    for r in rows:r['stage']='OUTER3_ON_NEW_VALIDATION_BANK_BASELINE'
    functions={g:[] for g in ('firm','fc1','sdf')}
    for group,name in [('firm','bp0'),('firm','bpI'),('firm','default'),('fc1','FC1_C'),('sdf','M_raw')]:
        functions[group].append(dict(outer=3,function=name,mean=1.,variance=0.,p90_value=1.,p99_value=1.,baseline=True))
    result=dict(outer=3,stage='baseline',economic_metrics=rows,function_movement=functions,optimizer_steps=0)
    b=object.__new__(ContinuationBackend);b.root=tmp_path
    b.state={'new_code_commit':'test'};b.signature={'fingerprint':'test'}
    b.schedules={'validation':{'parents':[]}}
    b.write_histories({'baseline':result},figures=True)
    assert (tmp_path/'firm_residual_history.csv').exists()
    assert pd.read_csv(tmp_path/'outer_training_history.csv').optimizer_steps.eq(0).all()


def test_historical_copy_is_idempotent_and_hash_checked(tmp_path):
    from analysis.run_finite_direct_large_continuation import install_historical_baseline
    from data.finite_direct_sampling import file_hash
    prepared=tmp_path/'prepared';source=prepared/'historical_baseline';source.mkdir(parents=True)
    (source/'history.csv').write_text('outer,value\n3,1\n')
    manifest={'historical_artifacts':{'history.csv':file_hash(source/'history.csv')}}
    out=tmp_path/'new_run'
    assert install_historical_baseline(prepared,out,manifest)['optimizer_steps']==0
    install_historical_baseline(prepared,out,manifest)
    install_historical_baseline(prepared,out,manifest,resume=True)
    (out/'historical_baseline/history.csv').write_text('tampered')
    with pytest.raises(ValueError):install_historical_baseline(prepared,out,manifest,resume=True)


@pytest.mark.skipif(not PREP.exists(),reason='requires certified input bindings')
def test_invalid_approval_cannot_enter_real_backend(tmp_path,monkeypatch):
    import analysis.run_finite_direct_large_continuation as m
    monkeypatch.setattr(m,'ContinuationBackend',lambda *a,**k:pytest.fail('production reached'))
    approval=tmp_path/'denied.json';approval.write_text(json.dumps({'approved':False,'schema_version':2}))
    with pytest.raises(ValueError,match='approval'):
        m.main(['--sampling-dir',str(PREP/'continuation_sampling'),
            '--initial-checkpoint',str(PREP/'continuation_initial_checkpoint.pt'),
            '--source-manifest',str(PREP/'continuation_source_manifest.json'),
            '--output',str(tmp_path/'run'),'--execute-approved','--approval',str(approval)])
