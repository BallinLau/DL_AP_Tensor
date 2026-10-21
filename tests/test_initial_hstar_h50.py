"""Non-optimizing contracts; synthetic fixtures are NOT root feasibility data."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import json
import math
import pytest
import torch
import pandas as pd

from analysis.run_initial_hstar_h50_broad_macro import main,plan,require_approval,approval_binding
from analysis.nested_equilibrium_utils import atomic_torch_save,atomic_write_json
from data.finite_direct_initial_hstar import calibrate_initial
from data.finite_direct_long_horizon_runtime import ShardStore,generate_long_horizon_dataset,promote_child
from data.finite_direct_long_horizon_sampling import broad_firm_bank,sample_macro_rows,time_bin,macro_bin
from data.finite_direct_sampling import file_hash


@pytest.fixture(autouse=True)
def no_optimization(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('optimizer forbidden in review tests')
    for cls in (torch.optim.Adam,torch.optim.AdamW,torch.optim.LBFGS):monkeypatch.setattr(cls,'step',forbidden)


def test_plan_only_never_builds_backend(monkeypatch,tmp_path):
    import analysis.run_initial_hstar_h50_broad_macro as r
    monkeypatch.setattr(r,'H50Backend',lambda *a,**k:pytest.fail('backend constructed'))
    assert main(['--output',str(tmp_path)])==0
    assert json.loads((tmp_path/'training_plan.json').read_text())['horizon']==50


def test_plan_sequence_and_firm_first():
    p=plan();assert p['start_outer']==12 and p['end_outer']==15
    for o in range(12,16):
        kinds=[s['kind'] for s in p['stages'] if s['outer']==o]
        assert kinds.index('firm_bank')<kinds.index('P')<kinds.index('data')<kinds.index('SDF')
        assert kinds.index('post_sdf_validation')<kinds.index('EMPIRICAL_POOL_COMMIT')<kinds.index('outer_commit')
        assert kinds.count('SDF')==2 and 'reuse_firm' not in kinds


@pytest.mark.parametrize('schema',[2,3,4])
def test_old_approval_rejected_before_artifact_loading(schema):
    a=SimpleNamespace(execute_approved=True,approval=SimpleNamespace(read_text=lambda:json.dumps({'schema_version':schema})))
    with pytest.raises(ValueError,match='v4 forbidden'):require_approval(a,'commit')


def test_missing_explicit_approval_rejected():
    with pytest.raises(ValueError,match='explicit V5'):require_approval(SimpleNamespace(execute_approved=True,approval=None),'c')
    assert require_approval(SimpleNamespace(execute_approved=False),'c') is False


@pytest.mark.parametrize('artifact',['checkpoint','train_initial_bank.pt','validation_initial_bank.pt','sampling_signature.json'])
def test_approval_binds_actual_content(artifact,tmp_path,monkeypatch):
    # Arbitrary test bytes, never an approved=true file.
    for name in ('checkpoint','train_initial_bank.pt','validation_initial_bank.pt','sampling_signature.json'):
        (tmp_path/name).write_bytes(b'test fixture')
    (tmp_path/'sampling_signature.json').write_text(json.dumps({'fingerprint':'fixture'}))
    import analysis.run_initial_hstar_h50_broad_macro as runner
    monkeypatch.setattr(runner,'safe_load',lambda path:dict(state=dict(dynamic_banks={'11':{
        k:dict(sha256=k) for k in ('pool','bank','schedule')}})))
    a=SimpleNamespace(sampling_dir=tmp_path,initial_checkpoint=tmp_path/'checkpoint',
        source_journal_checkpoint=tmp_path/'checkpoint',source_latest_pointer=tmp_path/'checkpoint')
    first=approval_binding(a,'c');(tmp_path/artifact).write_bytes(b'different fixture')
    if artifact=='sampling_signature.json':(tmp_path/artifact).write_text(json.dumps({'fingerprint':'different'}))
    assert approval_binding(a,'c')!=first


@pytest.mark.parametrize('time',[1,2,49,50])
def test_hstar_forbidden_after_t0(time):
    with pytest.raises(ValueError,match='only at t=0'):
        calibrate_initial(None,{},outer=12,split='train',economic={},crn={},economic_time=time)


@pytest.mark.parametrize('time,expected',[(0,0),(9,0),(10,1),(29,1),(30,2),(49,2)])
def test_time_strata(time,expected):assert time_bin(time)==expected


def test_nonfinite_macro_bins_rejected():
    with pytest.raises(ValueError):macro_bin(float('nan'),4.)


def fixture_store(tmp_path):
    manifest={}
    for t in (0,12,35):
        # Two lineages per stratum; current forecast != calculated input.
        capture=dict(split='train',source_time=t,states=torch.tensor([[.1,0.,0.,.2,-2.,-2.,4.],[.3,1.,1.,.1,-2.,-3.,3.]]),
            context=torch.tensor([[-1.,2.],[-.5,.2]]),eligible=torch.tensor([True,True]),
            path=torch.tensor([10,20]),firm_id=torch.tensor([1,2]))
        rows=[dict(split='train',parent_candidate_id=p,t=t+1,child_branch=b,
            economic_state_valid=True,x_t=-2.,x_t1=-2.+.01*b,Hatc_cal_t=-1.,LnK_cal_t=2.,
            Hatcf_t1=-.7,LnKF_t1=.7,
            Hatc_cal_t1=math.log(.5),LnK_cal_t1=math.log(2),C_raw_t1=1.,K_cal_t1=2.)
            for p in (10,20) for b in (0,1)]
        path=tmp_path/f'time_{t}.pt';atomic_torch_save(path,dict(captures=[capture],rows=rows))
        manifest[path.name]=file_hash(path)
    atomic_write_json(tmp_path/'manifest.json',dict(shards=manifest))
    return ShardStore(tmp_path)


def test_broad_firm_multi_time_and_calculated_inputs(tmp_path):
    store=fixture_store(tmp_path)
    b=broad_firm_bank(store,split='train',max_rows=120,seed=20)
    assert set(b.metadata['source_time'].tolist())=={0,12,35}
    kinds=b.metadata['source_kind'];cal=torch.tensor([k=='CALCULATED_MACRO_COLLOCATION' for k in kinds])
    assert int(cal.sum())==60
    torch.testing.assert_close(b.parent_states[cal,5:7],b.metadata['true_macro_context'][cal])
    assert not torch.equal(b.parent_states[~cal,5:7],b.metadata['true_macro_context'][~cal])
    assert 'REBUILD' in b.metadata['target_contract']
    repeat=broad_firm_bank(store,split='train',max_rows=120,seed=20)
    assert b.fingerprint==repeat.fingerprint


def test_broad_firm_no_validation_leakage(tmp_path):
    with pytest.raises(ValueError,match='lineage mismatch'):
        broad_firm_bank(fixture_store(tmp_path),split='validation',max_rows=8,seed=1)


def test_macro_sampling_pair_time_and_lazy_corruption(tmp_path):
    store=fixture_store(tmp_path);rows,index=sample_macro_rows(store,pair_budget=20,seed=1)
    from data.finite_direct_macro_contract import production_sdf_tensors
    tensor=production_sdf_tensors(rows)
    assert len(tensor['parents'])==6
    assert {t for _,_,t in tensor['lineage']}=={1,13,36}
    assert len(rows)==12 and sum(i['attempted'] for i in index['source_coverage'])==12
    path=next(tmp_path.glob('time_*.pt'));path.write_bytes(b'corrupt fixture')
    with pytest.raises(ValueError,match='corruption'):list(store.chunks())


def test_promotion_has_only_financing_side_effect(monkeypatch):
    import data.finite_direct_long_horizon_runtime as r
    state=dict(alive=torch.tensor([[True,False]]),b=torch.tensor([[.2,0.]]),z=torch.zeros(1,2),
        eta=torch.ones(1,2),i=torch.ones(1,2),x=torch.tensor([-2.]),hatcf=torch.tensor([-1.]),lnkf=torch.tensor([2.]),
        hatc_cal=torch.tensor([-.8]),lnk_cal=torch.tensor([1.9]),bp=torch.zeros(1,2),
        firm_id=torch.tensor([[3,-1]]),economic_node_id=torch.tensor([123]),accounting_C_raw=torch.tensor([4.]))
    before=deepcopy(state);calls=[]
    monkeypatch.setattr(r,'forward_policy_value_for_simulation',lambda *a:SimpleNamespace(bp=torch.tensor([[.3]])))
    def resolve(**kw):
        calls.append(kw);return torch.tensor([[.4]])
    sim=SimpleNamespace(models={'policy_value':None},_resolve_bp_action=resolve)
    promote_child(sim,state,2)
    assert len(calls)==1 and calls[0]['bp_execution_applicable'] is True
    for k in before:
        if k!='bp':torch.testing.assert_close(before[k],state[k],rtol=0,atol=0)
    assert float(state['bp'][0,0])==pytest.approx(.4)


def test_horizon_and_batch_reject_before_models(tmp_path):
    with pytest.raises(ValueError,match='exactly 50'):
        generate_long_horizon_dataset(None,None,None,None,None,tmp_path,outer=12,stage='x',horizon=1)
    with pytest.raises(ValueError,match='batch must be positive'):
        generate_long_horizon_dataset(None,None,None,None,None,tmp_path,outer=12,stage='x',parent_batch_size=0)


def test_actual_calculated_children_are_rebuilt_for_new_bank(tmp_path):
    from config import Config
    from experiments.run_utils import build_models
    from analysis.economic_config import AnalysisEconomicConfig
    from data.finite_direct_runtime import conditional_child_bank
    store=fixture_store(tmp_path);models=build_models(torch.device('cpu'))
    model=models['sdf_fc1'];model.eval().requires_grad_(False)
    on=broad_firm_bank(store,split='train',max_rows=12,seed=2,calculated_fraction=0)
    cal=broad_firm_bank(store,split='train',max_rows=12,seed=2,calculated_fraction=1)
    on_child=conditional_child_bank(model,on,seed=3,n_child_shocks=2,
        economic=AnalysisEconomicConfig.from_current_config(),device='cpu')
    cal_child=conditional_child_bank(model,cal,seed=3,n_child_shocks=2,
        economic=AnalysisEconomicConfig.from_current_config(),device='cpu')
    assert on.fingerprint!=cal.fingerprint
    assert cal_child.metadata['parent_fingerprint']==cal.fingerprint
    assert on_child.metadata['parent_fingerprint']==on.fingerprint
    assert cal_child.children.shape==(12,2,8)
    assert torch.isfinite(cal_child.children).all()
    assert 'raw_m' in cal_child.metadata


def test_zero_survivors_does_not_skip_original_entry_opportunity():
    from data.finite_direct_long_horizon_runtime import continuable_nodes
    state={'hatc_cal':torch.tensor([float('nan')]),'lnk_cal':torch.tensor([float('nan')])}
    row=dict(C_raw_t1=1.,K_cal_t1=2.,Hatc_cal_t1=math.log(.5),LnK_cal_t1=math.log(2.),
        surviving_mass=0.,economic_state_valid=False,failure_reason='ZERO_SURVIVING_MASS')
    assert continuable_nodes(state,[row])==[0]
    assert row['economic_state_valid'] is False
    assert float(state['lnk_cal'][0])==pytest.approx(math.log(2))
    row['K_cal_t1']=0
    assert continuable_nodes(state,[row])==[]


def test_fifty_transitions_one_spine_no_duplicate_node_commit(monkeypatch,tmp_path):
    import data.finite_direct_long_horizon_runtime as r
    from experiments.run_utils import build_hyperparams
    entry=[];processed=[];promoted=[];roots=[]
    sim=SimpleNamespace(device=torch.device('cpu'),group_size=2,
        transition_rng_mode='stable_firm_identity',entry_rng_mode='stable_economic_identity')
    monkeypatch.setattr(r,'simulator',lambda *a:sim)
    monkeypatch.setattr(r,'hashes',lambda *a:{'fixture':'SYNTHETIC_NOT_FEASIBILITY'})
    monkeypatch.setattr(r,'cache_provenance',lambda *a,**k:{'crn_hash':'fixture'})
    state=dict(x=torch.zeros(2),b=torch.zeros(2,2),K=torch.ones(2,2),alive=torch.ones(2,2,dtype=torch.bool),
        transition_path_id=torch.tensor([10,20]),hatc_cal=torch.zeros(2),lnk_cal=torch.zeros(2))
    def observations(s,t,b):
        return [dict(parent_candidate_id=int(pid),t=t,child_branch=b,x_t1=0.,Hatcf_t1=0.,LnKF_t1=math.log(2),
            Hatc_cal_t1=0.,LnK_cal_t1=math.log(2),C_raw_t1=2.,K_cal_t1=2.,
            economic_state_valid=True,failure_reason='') for pid in s['transition_path_id']]
    def initial(sim,s,**kw):
        if not kw['cache'].exists():atomic_torch_save(kw['cache'],{'synthetic_cache_fixture':True})
        roots.append(kw);return s,observations(s,0,-1),[0,1],[dict(root_found=True,fixture=True)]
    monkeypatch.setattr(r,'calibrate_initial',initial)
    monkeypatch.setattr(r,'expand_after_parent_exit',lambda sim,s,**k:[deepcopy(s),deepcopy(s)])
    def enter(sim,s,n):entry.append(1);return s
    monkeypatch.setattr(r,'_apply_entry_batched',enter)
    def process(sim,s,t,b,**kw):
        processed.append((t,b));return s,observations(s,t,b),[0,1]
    monkeypatch.setattr(r,'process_node',process)
    def promote(sim,s,t):promoted.append(t);return s
    monkeypatch.setattr(r,'promote_child',promote)
    e=SimpleNamespace(device=torch.device('cpu'),hyperparams=build_hyperparams())
    bank=dict(state=state,fingerprint='fixture');schedule=dict(seed=1,split='train')
    store=r.generate_long_horizon_dataset({},e,bank,schedule,{},tmp_path,
        outer=12,stage='CPU_SYNTHETIC_INTERFACE_ONLY',parent_batch_size=2,capture=False)
    rows=pd.concat(list(store.rows()))
    assert len(rows)==200 and set(rows.t)==set(range(1,51))
    assert len(roots)==1 and len(entry)==100 and len(processed)==100
    assert len(set(processed))==100 and promoted==list(range(1,50))
    assert not rows.duplicated(['split','parent_candidate_id','t','child_branch']).any()
    old=deepcopy(store.manifest['shards'])
    again=r.generate_long_horizon_dataset({},e,bank,schedule,{},tmp_path,
        outer=12,stage='CPU_SYNTHETIC_INTERFACE_ONLY',parent_batch_size=2,capture=False)
    assert len(processed)==100 and old==again.manifest['shards']


def test_dynamic_pool_long_time_train_only_and_fixed_beta(tmp_path):
    from data.finite_direct_long_horizon_empirical import resample_from_sources
    from data.finite_direct_dynamic_empirical import template
    templates=[]
    for time in (1,20,40):
        for branch in (0,1):
            value=template([.1,.3],[1.,2.],identity=f'fixture:{time}:{branch}',
                source=dict(split='train',cluster_id='original_lineage_1',source_time=time,source_branch=branch))
            templates.append(dict(split='train',dynamic_template=value))
    shard=tmp_path/'templates.pt';atomic_torch_save(shard,dict(templates=templates))
    atomic_write_json(tmp_path/'manifest.json',dict(provenance={'stage':'POST_SDF_TRAIN_EMPIRICAL_SOURCE','split':'train'},
        shards={shard.name:file_hash(shard)}))
    sources=[dict(path=str(tmp_path),manifest_sha256=file_hash(tmp_path/'manifest.json'),outer=12)]
    bank=dict(state={'x':torch.zeros(8192),'b':torch.full((8192,2),.2),'bp':torch.full((8192,2),.2),'K':torch.ones(8192,2)})
    schedule={'parents':[dict(parent_id=j,sampling_source='BETA' if j<4096 else 'EMPIRICAL') for j in range(8192)]}
    archive=[templates[0]['dynamic_template']]
    result=resample_from_sources(bank,schedule,sources,outer=13,initial_archive=archive)
    torch.testing.assert_close(result['state']['b'][:4096],bank['state']['b'][:4096],rtol=0,atol=0)
    assert set(h['source_time'] for h in result['h50_empirical_history'] if 'source_time' in h)=={1,20,40}
    assert sum(h['pool_source']=='RECENT' for h in result['h50_empirical_history'])==2867
    assert sum(h['pool_source']=='ARCHIVE' for h in result['h50_empirical_history'])==1229
    again=resample_from_sources(bank,schedule,sources,outer=13,initial_archive=archive)
    assert result['fingerprint']==again['fingerprint']


def test_all_time_population_coverage_not_one_step_denominator(tmp_path):
    from analysis.h50_diagnostics import streaming_economic_metrics
    s=fixture_store(tmp_path)
    schedule=dict(split='train',parents=[dict(parent_id=p,sampling_source='BETA',child_shock_keys=[0,1]) for p in (10,20)])
    result=streaming_economic_metrics(s,schedule,outer=12,stage='fixture')[-1]
    assert result['scheduled_child_count']==200 and result['valid_child_count']==12
    assert result['Complete_SDF_pair_fraction']==pytest.approx(6/100)


def test_post_sdf_uses_new_validation_store_not_d2(monkeypatch,tmp_path):
    from analysis.run_initial_hstar_h50_broad_macro import H50Backend
    import analysis.h50_diagnostics as diagnostics
    backend=object.__new__(H50Backend);backend.root=tmp_path;backend.schedules={'validation':{}}
    calls=[]
    def store(*args,**kwargs):
        calls.append((args,kwargs));return SimpleNamespace(root=tmp_path)
    backend._store=store
    backend._data=lambda *a:pytest.fail('pre-SDF D2 must not be read')
    monkeypatch.setattr(diagnostics,'write_diagnostics',lambda *a:None)
    monkeypatch.setattr(diagnostics,'streaming_economic_metrics',lambda *a,**k:[])
    result=backend._execute(12,'post_sdf_validation',{})
    assert calls==[((12,'POST_SDF_VALIDATION_RESIMULATION','validation'),{'capture':False})]
    assert result['optimizer_steps']==0


def test_new_approval_rejects_changed_checkpoint_content(tmp_path,monkeypatch):
    for name in ('checkpoint','train_initial_bank.pt','validation_initial_bank.pt'):
        (tmp_path/name).write_bytes(b'fixture')
    (tmp_path/'sampling_signature.json').write_text(json.dumps({'fingerprint':'fixture'}))
    import analysis.run_initial_hstar_h50_broad_macro as runner
    monkeypatch.setattr(runner,'safe_load',lambda path:dict(state=dict(dynamic_banks={'11':{
        k:dict(sha256=k) for k in ('pool','bank','schedule')}})))
    args=SimpleNamespace(sampling_dir=tmp_path,initial_checkpoint=tmp_path/'checkpoint',execute_approved=True,
        source_journal_checkpoint=tmp_path/'checkpoint',source_latest_pointer=tmp_path/'checkpoint')
    expected=approval_binding(args,'commit')
    # Pure in-memory fixture; never create a real approved=true artifact.
    args.approval=SimpleNamespace(read_text=lambda:json.dumps(dict(expected,approved=True)))
    args.initial_checkpoint.write_bytes(b'changed fixture')
    with pytest.raises(ValueError,match='content-bound approval mismatch'):require_approval(args,'commit')


def test_sdf_sampler_uses_complete_source_pairs_not_fc1_subset(tmp_path):
    from analysis.finite_direct_continuation_state import safe_load
    store=fixture_store(tmp_path);path=tmp_path/'time_0.pt';payload=safe_load(path)
    payload['rows'][0]['economic_state_valid']=False
    atomic_torch_save(path,payload)
    store.manifest['shards'][path.name]=file_hash(path)
    atomic_write_json(tmp_path/'manifest.json',store.manifest)
    fc1,_=sample_macro_rows(store,pair_budget=20,seed=1)
    sdf,index=sample_macro_rows(store,pair_budget=20,seed=2,complete_pairs_only=True)
    assert len(fc1)==11 and len(sdf)==10
    assert (sdf.groupby(['parent_candidate_id','t']).size()==2).all()
    assert index['complete_pairs_only'] is True
