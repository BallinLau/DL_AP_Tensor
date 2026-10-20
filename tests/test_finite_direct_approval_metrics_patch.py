"""Scoped, CPU-only regressions. No approved=true file or optimizer steps."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from analysis import run_full_nested_finite_direct_outer4 as runner
from analysis.finite_direct_outer_metrics import economic_metrics,write_histories,POST_SDF
from analysis.nested_equilibrium_utils import stable_json_hash,atomic_write_json
from data.finite_direct_sampling import file_hash
from data.finite_direct_runtime import cache_provenance,verify_cache


def toy_macro():
    m=torch.nn.Module(); m.fc1_model=torch.nn.Module()
    m.fc1_model.FC1_C=torch.nn.Linear(4,1); m.fc1_model.FC1_K=torch.nn.Linear(4,1)
    m.sdf_model=torch.nn.Linear(4,1); m.value_model=torch.nn.Linear(3,1)
    return m


def artifacts(tmp_path,monkeypatch):
    sig=tmp_path/'sampling'; sig.mkdir()
    (sig/'sampling_signature.json').write_text(json.dumps(dict(fingerprint='sampling-A',artifact_sha256={'bank':'aaa'})))
    checkpoint=tmp_path/'init.pt'; checkpoint.write_bytes(b'checkpoint-A')
    plan=runner.training_plan()
    approved=dict(schema_version=2,approved=True,source_commit='reviewed-commit',plan_fingerprint=stable_json_hash(plan),
        sampling_signature_fingerprint='sampling-A',sampling_signature_sha256=file_hash(sig/'sampling_signature.json'),
        initial_checkpoint_sha256=file_hash(checkpoint))
    # Only an in-memory test double, NEVER an approved=true file on disk.
    args=SimpleNamespace(execute_approved=True,approval=SimpleNamespace(read_text=lambda:json.dumps(approved)),
                         sampling_dir=sig,initial_checkpoint=checkpoint)
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**k:'')
    return args,approved,plan


def test_approval_binds_content_not_location(tmp_path,monkeypatch):
    args,_,plan=artifacts(tmp_path,monkeypatch)
    assert runner.require_approval(args,plan,'reviewed-commit')
    moved=tmp_path/'HPC-copy'; moved.mkdir()
    (moved/'sampling_signature.json').write_bytes((args.sampling_dir/'sampling_signature.json').read_bytes())
    cp=moved/'renamed.pt'; cp.write_bytes(args.initial_checkpoint.read_bytes())
    args.sampling_dir=moved; args.initial_checkpoint=cp
    assert runner.require_approval(args,plan,'reviewed-commit')


@pytest.mark.parametrize('change',['checkpoint','signature','manifest_same_fingerprint','legacy_schema','not_approved','source','plan'])
def test_approval_rejects_unapproved_inputs(tmp_path,monkeypatch,change):
    args,approval,plan=artifacts(tmp_path,monkeypatch)
    if change=='checkpoint': args.initial_checkpoint.write_bytes(b'checkpoint-B')
    if change=='signature':
        (args.sampling_dir/'sampling_signature.json').write_text(json.dumps(dict(fingerprint='sampling-B')))
    if change=='manifest_same_fingerprint':
        (args.sampling_dir/'sampling_signature.json').write_text(json.dumps(dict(fingerprint='sampling-A',artifact_sha256={'bank':'bbb'})))
    if change=='legacy_schema': approval.pop('schema_version')
    if change=='not_approved': approval['approved']=False
    if change=='source': approval['source_commit']='other'
    if change=='plan': approval['plan_fingerprint']='other'
    with pytest.raises(RuntimeError,match='approval does not match'):
        runner.require_approval(args,plan,'reviewed-commit')


def test_plan_only_does_not_read_or_execute_artifacts():
    args=SimpleNamespace(execute_approved=False)
    assert runner.require_approval(args,runner.training_plan(),'commit') is False


def test_invalid_approval_never_constructs_production_backend(tmp_path,monkeypatch):
    args,approval,_=artifacts(tmp_path,monkeypatch); approval['approved']=False
    monkeypatch.setattr(runner,'parse_args',lambda argv:args)
    def forbidden(*a,**k): raise AssertionError('must not enter production')
    monkeypatch.setattr(runner,'ProductionBackend',forbidden)
    with pytest.raises(RuntimeError,match='approval does not match'): runner.main([])


def schedule(split='validation'):
    return dict(split=split,seed=123,parents=[dict(parent_id=p,sampling_source=s,
        child_shock_keys=[dict(path=p,branch=b,seed=123) for b in (0,1)])
        for p,s in [(10,'BETA'),(20,'EMPIRICAL')]])


def observations():
    rows=[]
    for p,b,g,valid in [(10,0,.1,True),(10,1,999,False),(20,0,.3,True),(20,1,.5,True)]:
        k=4.+b*.1+(p==20)*.2
        rows.append(dict(split='validation',parent_candidate_id=p,t=1,child_branch=b,
            economic_state_valid=valid,failure_reason='' if valid else 'CHILD_EXTINCT',
            normalized_G_at_prediction=g,G_at_prediction=-g,C_raw_t1=20.+b,K_cal_t1=math.exp(k),
            Hatc_cal_t1=-1.+b*.1,Hatcf_t1=-.9+b*.1,LnK_cal_t1=k,LnKF_t1=k+.05,LnK_cal_t=3.9,
            firm_mass=200.,surviving_mass=180.-b,default_mass=20.,investment_mass=40.,entry_mass=2.))
    return pd.DataFrame(rows)


def test_metrics_grouping_coverage_and_ledger_parity_without_mutation():
    frame=observations(); original=frame.copy(deep=True); s=schedule(); before=deepcopy(s)
    rows={r['sampling_source']:r for r in economic_metrics(frame,s,outer=0,stage='D2')}
    assert rows['BETA']['G_mean']==pytest.approx(.1)
    assert rows['EMPIRICAL']['G_mean']==pytest.approx(.4)
    assert rows['ALL']['G_mean']==pytest.approx(.3)
    assert rows['ALL']['G_p90']==pytest.approx(np.quantile([.1,.3,.5],.9))
    assert rows['ALL']['G_raw_mean']==pytest.approx(-.3)
    assert rows['ALL']['Default_rate']==pytest.approx(.1)
    assert rows['ALL']['Investment_rate']==pytest.approx(.2)
    assert rows['ALL']['Entry_rate']==pytest.approx(.01)
    assert rows['ALL']['Valid_child_fraction']==.75
    assert rows['ALL']['Complete_SDF_pair_fraction']==.5
    assert rows['BETA']['Complete_SDF_pair_fraction']==0
    assert rows['EMPIRICAL']['Complete_SDF_pair_fraction']==1
    assert rows['ALL']['attempted_child_count']==4 and rows['ALL']['failed_child_count']==1
    assert rows['ALL']['Delta_LnK_R2']==pytest.approx(1-3*.05**2/np.var([.1,.3,.4])/3)
    pd.testing.assert_frame_equal(frame,original); assert s==before


def test_undefined_metrics_are_not_available_not_zero(tmp_path):
    frame=observations(); frame['economic_state_valid']=False
    rows=economic_metrics(frame,schedule(),outer=0,stage=POST_SDF)
    assert all(np.isnan(r['G_mean']) for r in rows)
    assert all(r['Valid_child_fraction']==0 for r in rows)
    results={'post':dict(stage='post_sdf_validation',outer=0,economic_metrics=rows)}
    write_histories(tmp_path,results,figures=True)
    text=(tmp_path/'outer_macro_economic_history.csv').read_text()
    assert 'NOT_AVAILABLE' in text and 'POST_SDF_VALIDATION_RESIMULATION' in text
    for name in ['outer_g_residual.png','outer_fc1_r2.png','outer_firm_residual.png','outer_sdf_aio.png','outer_survival_capital_consumption.png']:
        assert (tmp_path/'figures'/name).stat().st_size>0


def test_failed_defined_accounts_remain_in_raw_economic_statistics():
    frame=observations()
    frame.loc[1,['C_raw_t1','surviving_mass','default_mass']]=[-5.,0.,200.]
    all_rows=next(r for r in economic_metrics(frame,schedule(),outer=0,stage=POST_SDF) if r['sampling_source']=='ALL')
    assert all_rows['C_mean']==pytest.approx((20-5+20+21)/4)
    assert all_rows['Default_rate']==pytest.approx(260/800)
    assert all_rows['Default_rate_valid']==pytest.approx(.1)
    assert all_rows['Surviving_firm_count_mean']==pytest.approx((180+0+180+179)/4)
    assert all_rows['C_finite_child_count']==4 and all_rows['G_finite_child_count']==3


def test_post_sdf_is_validation_only_dedicated_cache_no_optimizer(tmp_path,monkeypatch):
    macro=toy_macro()
    b=object.__new__(runner.ProductionBackend); b.root=tmp_path
    b.models=dict(policy_value=torch.nn.Linear(7,1),sdf_fc1=macro)
    b.episode=object(); b.args=SimpleNamespace(parent_batch_size=128)
    b.banks={'validation':{'fingerprint':'fixed-bank','state':{'x':torch.ones(2)}}}
    b.schedules={'validation':schedule()}; b.signature={'fingerprint':'locked'}
    def forbidden(*a,**k): raise AssertionError('no optimizer or D2 consumption allowed')
    for cls in (torch.optim.Adam,torch.optim.AdamW,torch.optim.LBFGS): monkeypatch.setattr(cls,'step',forbidden)
    b._data=forbidden; bank_before=b.banks['validation']['state']['x'].clone(); model_before=runner.hashes(b.models)
    modes=[m.training for model in b.models.values() for m in model.modules()]
    rng=torch.get_rng_state().clone()
    calls=[]
    def generate(models,episode,bank,s,signature,path,**kwargs):
        assert not torch.is_grad_enabled()
        assert not any(m.training for model in models.values() for m in model.modules())
        torch.rand(7)  # Evaluation RNG consumption must not affect later training.
        assert s is b.schedules['validation'] and bank is b.banks['validation']
        assert kwargs['stage']==POST_SDF and kwargs['capture'] is False
        assert 'D2' not in str(path)
        calls.append(kwargs); path.mkdir(parents=True)
        manifest={'provenance':{'crn_hash':'fixed-crn'}}
        atomic_write_json(path/'manifest.json',manifest)
        return observations(),[],manifest
    monkeypatch.setattr(runner,'generate_dataset',generate)
    row=b._post_sdf_validation(0)
    assert len(calls)==1 and row['split']=='validation' and len(row['economic_metrics'])==3
    assert runner.hashes(b.models)==model_before
    assert modes==[m.training for model in b.models.values() for m in model.modules()]
    assert torch.equal(torch.get_rng_state(),rng)
    torch.testing.assert_close(b.banks['validation']['state']['x'],bank_before)
    plan=list(runner.stages()); kinds=[k for o,k,i in plan if o==0]
    assert kinds[-3:]==['post_sdf_validation','cross_block','outer_commit']
    assert kinds[kinds.index('post_sdf_validation')-1]=='SDF'


def test_post_sdf_cannot_accept_d2_or_pre_sdf_model_provenance():
    models=dict(policy_value=torch.nn.Linear(7,1),sdf_fc1=toy_macro())
    kwargs=dict(bank={'fingerprint':'bank'},schedule=schedule(),signature={'fingerprint':'sampling'},outer=0,economic={})
    d2=cache_provenance(models,stage='D2',**kwargs)
    post=cache_provenance(models,stage=POST_SDF,**kwargs)
    with pytest.raises(RuntimeError,match='stale'): verify_cache(d2,post)
    with torch.no_grad(): next(models['sdf_fc1'].sdf_model.parameters()).add_(.01)
    newpost=cache_provenance(models,stage=POST_SDF,**kwargs)
    with pytest.raises(RuntimeError,match='stale'): verify_cache(post,newpost)


def test_three_histories_are_rebuilt_from_committed_results_idempotently(tmp_path):
    val=economic_metrics(observations(),schedule(),outer=0,stage='D2')
    train=observations(); train['split']='train'
    train=economic_metrics(train,schedule('train'),outer=0,stage='D2')
    epochs=[dict(epoch=99,dual_update=True,constraints={'g_mean_low':.01},al={'lambda_mean_low':.2,'rho':2.},
                 validation={'prefix_primary_true_state_normalized_signed_aio_mean':-.003})]
    path=tmp_path/'sdf_history.json'; atomic_write_json(path,epochs)
    results={
        'data':dict(outer=0,stage='data',economic_metrics=train+val),
        'sdf':dict(outer=0,stage='SDF',indices={'chunk':1},optimizer_steps=100,
                   al_history_path=str(path),al_history_sha256=file_hash(path)),
        'cross':dict(outer=0,stage='cross_block',label='post_macro',fixed_parent_hash='fixed',
            P={'p0_norm_mae':.04,'pi_norm_mae':.05},Q={'normalized_mae':.06})}
    write_histories(tmp_path,results,figures=True)
    paths=list(tmp_path.glob('*.csv')); first={p.name:p.read_bytes() for p in paths}
    write_histories(tmp_path,results)
    assert all(p.read_bytes()==first[p.name] for p in paths)
    econ=pd.read_csv(tmp_path/'outer_macro_economic_history.csv')
    assert set(econ.split)=={'train','validation'} and len(econ)==6
    al=pd.read_csv(tmp_path/'outer_sdf_al_history.csv')
    assert al.normalized_signed_aio_mean.iloc[0]==-.003 and al.lambda_mean_low.iloc[0]==.2
    history=pd.read_csv(tmp_path/'outer_training_history.csv')
    assert history.loc[history.stage=='cross_block','probe_kind'].iloc[0]=='FIXED_7D_PROBE'
    path.write_text('[]')
    with pytest.raises(RuntimeError,match='SDF history changed'): write_histories(tmp_path,results)
