from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest
import torch
from data.finite_direct_runtime import expand_after_parent_exit
from data.finite_direct_longrun_replay import select_mixture,fit_edges,commit_pool,load_pool,eligibility
from analysis.run_h10_h100_longrun_macro import plan,require_approval


def rows(n=20,start=0,t=21):
    return pd.DataFrame([dict(split='train',parent_candidate_id=j,t=t,child_branch=0,
        economic_state_valid=True,x_t=-2.,x_t1=-2.,Hatc_cal_t=-1.+j*.01,LnK_cal_t=2.,
        Hatc_cal_t1=np.log(.5),LnK_cal_t1=np.log(2),C_raw_t1=1.,K_cal_t1=2.)
        for j in range(start,start+n)])


def test_production_child0_subset_is_exact_including_entry_crn():
    from test_entry_value_cost import _value_cost_sim
    from data.simulate_ts_parallel import _initialize_batched_state,_apply_entry_batched,_process_node_batched
    from analysis.finite_direct_continuation_state import state_digest
    sim,_=_value_cost_sim(bp_source='grid',stable_identity_rng=True);sim.branch_num=2
    parent=_initialize_batched_state(sim,4)
    _process_node_batched(sim,parent,0,-1)
    before=state_digest(parent)
    two=expand_after_parent_exit(sim,deepcopy(parent),child_t=1)
    one=expand_after_parent_exit(sim,deepcopy(parent),child_t=1,branch_indices=(0,))
    assert len(one)==1 and len(two)==2
    assert state_digest(one[0])==state_digest(two[0])
    a=_apply_entry_batched(sim,one[0],20);b=_apply_entry_batched(sim,two[0],20)
    _process_node_batched(sim,a,1,0);_process_node_batched(sim,b,1,0)
    assert state_digest(a)==state_digest(b)
    assert state_digest(parent)==before and sim.branch_num==2


def test_h10_plan_four_real_outers_and_postcommit_h100():
    p=plan()
    assert p['horizon']==10 and p['train_parents']==4096 and p['validation_parents']==1024
    assert p['firm_macro_coverage']['firm_macro_coverage_fraction']==.5
    for o in range(12,16):
        s=[x['kind'] for x in p['stages'] if x['outer']==o]
        assert s.index('outer_commit')<s.index('LONGRUN_MACRO_POOL_COMMIT')
        assert s.count('SDF')==2 and s.count('data')==3


def test_h50_v5_approval_is_rejected_before_loading():
    from types import SimpleNamespace
    a=SimpleNamespace(execute_approved=True,approval=SimpleNamespace(read_text=lambda:json.dumps(dict(schema_version=5))))
    with pytest.raises(ValueError,match='H50 V5 forbidden'):require_approval(a,'fixture')


def test_h100_branch_does_not_shadow_earlier_stage_hash_helpers():
    from analysis.run_h10_h100_longrun_macro import H50Backend
    assert 'module_hash' not in H50Backend._execute.__code__.co_varnames


def test_mixture_shortfall_and_no_duplicates():
    coverage=rows(100,t=1);replay=rows(10,start=1000)
    f,a=select_mixture(coverage,replay,edges=fit_edges([coverage,replay]),target=40)
    assert a['LONGRUN']['selected']==10 and a['COVERAGE']['selected']==30
    assert not f.duplicated(['parent_candidate_id','t']).any()
    empty,a=select_mixture(coverage,replay.iloc[:0],edges=fit_edges([coverage]),target=40)
    assert len(empty)==40 and a['LONGRUN']['selected']==0


def test_burnin_invalid_and_validation_forbidden():
    c=rows(5,t=1);r=rows(5,start=100,t=20)
    with pytest.raises(ValueError,match='invalid historical'):select_mixture(c,r,edges=fit_edges([c]))
    c['split']='validation'
    with pytest.raises(ValueError,match='Train-only'):select_mixture(c,rows(0),edges=[[-6,2],[-2,5]])


def test_lineage_limit_underfill_truthful():
    c=pd.concat([rows(1,t=t) for t in range(1,101)])
    f,a=select_mixture(c,rows(0),edges=fit_edges([c]),target=40)
    assert len(f)==2 and a['COVERAGE']['underfilled']==38


def test_bin_endpoints_and_directional_overflow():
    from data.finite_direct_longrun_replay import cells
    f=pd.DataFrame(dict(Hatc_cal_t=[-7,-6,2,3],LnK_cal_t=[-2,-3,5,6]))
    assert cells(f,[[-6,0,2],[-2,0,5]]).tolist()==[[-1,0],[0,-1],[1,1],[2,2]]


def test_diagnostic_coverage_and_bin_error_units():
    from analysis.h10_h100_diagnostics import coverage_report,by_bin_errors
    c=rows(20,t=1);r=rows(20,start=100,t=21);edges=fit_edges([c,r])
    mixed,_=select_mixture(c,r,edges=edges,target=20)
    validation=c.copy();validation['split']='validation';validation.parent_candidate_id+=10000
    report=coverage_report(mixed,validation,c,r,edges)
    assert report['train_valid_rows']==len(mixed)
    assert report['validation_unique_initial_lineages']==20
    assert report['longrun_share']+report['coverage_share']==1
    predicted=validation[['Hatc_cal_t1','LnK_cal_t1']].to_numpy()
    errors=by_bin_errors(validation,predicted,edges,'validation')
    assert errors.rows.sum()==20 and errors.h_MAE.max()==0 and errors.ell_RMSE.max()==0


def test_committed_pool_burnin_and_corruption(tmp_path):
    from types import SimpleNamespace
    p=tmp_path/'source';p.mkdir();(p/'manifest.json').write_text('{}')
    frame=pd.concat([rows(2,t=20),rows(2,t=21)])
    store=SimpleNamespace(root=p,manifest={'provenance':{'branches':1,'initial_mode':'ROOT_CONSISTENT'}},rows=lambda:iter([frame]))
    m=commit_pool([store],tmp_path/'pool',source_outer=12,model_hashes=dict(firm='f',fc1='c',sdf='s'),initial_design=[],outer_checkpoint_sha='cp')
    assert m['eligible_parent_times']==2
    f=load_pool(tmp_path/'pool',consumer_outer=13)
    assert set(f.parent_time)=={20} and set(f.replay_age)=={1}
    with pytest.raises(ValueError,match='previous'):load_pool(tmp_path/'pool',consumer_outer=14)
    (tmp_path/'pool'/'macro_rows.csv').write_text('corruption')
    with pytest.raises(ValueError,match='corruption'):load_pool(tmp_path/'pool',consumer_outer=13)


def test_vectorized_shared_eligibility_matches_original_contract():
    from data.finite_direct_macro_contract import select_training_rows
    f=rows(12)
    second=f.copy();second.child_branch=1
    f=pd.concat([f,second],ignore_index=True)
    f.loc[3,'economic_state_valid']=False
    original=select_training_rows(f);fast=eligibility(f)
    pd.testing.assert_frame_equal(original.fc1,fast.fc1)
    pd.testing.assert_frame_equal(original.sdf_pairs,fast.sdf_pairs)
    f.loc[len(f)-1,'Hatc_cal_t']+=1
    with pytest.raises(ValueError,match='siblings'):eligibility(f)


@pytest.mark.parametrize('horizon,branches',[(10,2),(100,1)])
def test_runtime_configured_horizon_sparse_resume_and_columnar(monkeypatch,tmp_path,horizon,branches):
    import data.finite_direct_replay_runtime as r
    from types import SimpleNamespace
    from experiments.run_utils import build_hyperparams
    from analysis.nested_equilibrium_utils import atomic_torch_save
    calls=[]
    sim=SimpleNamespace(device=torch.device('cpu'),group_size=2,
        transition_rng_mode='stable_firm_identity',entry_rng_mode='stable_economic_identity')
    monkeypatch.setattr(r,'simulator',lambda *a:sim)
    monkeypatch.setattr(r,'hashes',lambda *a:{'fixture':'SYNTHETIC_INTERFACE_ONLY'})
    monkeypatch.setattr(r,'cache_provenance',lambda *a,**k:{'crn_hash':'fixture'})
    s=dict(x=torch.zeros(2),b=torch.zeros(2,2),bar_z=torch.zeros(2,2),K=torch.ones(2,2),
        alive=torch.ones(2,2,dtype=torch.bool),transition_path_id=torch.tensor([10,20]),
        hatc_cal=torch.zeros(2),lnk_cal=torch.zeros(2))
    def observations(state,t,b):
        return [dict(parent_candidate_id=int(pid),t=t,child_branch=b,x_t1=0.,Hatcf_t1=0.,LnKF_t1=np.log(2),
            Hatc_cal_t1=0.,LnK_cal_t1=np.log(2),C_raw_t1=2. if t<3 else float('nan'),K_cal_t1=2.,
            economic_state_valid=t<3,failure_reason='' if t<3 else 'UNDEFINED') for pid in state['transition_path_id']]
    def initial(sim,state,**kw):
        if not kw['cache'].exists():atomic_torch_save(kw['cache'],{'synthetic_fixture':True})
        return state,observations(state,0,-1),[0,1],[dict(root_found=True,fixture=True)]
    monkeypatch.setattr(r,'calibrate_initial',initial)
    monkeypatch.setattr(r,'calibrate_exploration',initial)
    def expand(sim,state,**kw):
        calls.append(kw['branch_indices'])
        return [deepcopy(state) for b in kw['branch_indices']]
    monkeypatch.setattr(r,'expand_after_parent_exit',expand)
    monkeypatch.setattr(r,'_apply_entry_batched',lambda sim,state,n:state)
    monkeypatch.setattr(r,'promote_child',lambda sim,state,t:state)
    monkeypatch.setattr(r,'process_node',lambda sim,state,t,b,**kw:(state,observations(state,t,b),[0,1]))
    e=SimpleNamespace(device=torch.device('cpu'),hyperparams=build_hyperparams())
    bank=dict(state=s,fingerprint='fixture');schedule=dict(seed=1,split='train',parents=[dict(parent_id=p,sampling_source='BETA') for p in (10,20)])
    store=r.generate_long_horizon_dataset({},e,bank,schedule,{},tmp_path,
        outer=12,stage='SYNTHETIC_INTERFACE_ONLY',horizon=horizon,branches=branches,capture=False)
    assert len(calls)==3 and all(c==tuple(range(branches)) for c in calls)
    assert store.manifest['scheduled_child_rows']==2*horizon*branches
    assert len(store.manifest['shards'])==3 and store.manifest['terminal_suffixes']
    sidecars=pd.concat(list(store.rows()),ignore_index=True)
    raw=pd.concat([pd.DataFrame(chunk['rows']) for chunk in store.chunks()],ignore_index=True)
    assert len(sidecars)==len(raw)==6*branches
    assert sidecars.economic_state_valid.sum()==4*branches
    r.generate_long_horizon_dataset({},e,bank,schedule,{},tmp_path,
        outer=12,stage='SYNTHETIC_INTERFACE_ONLY',horizon=horizon,branches=branches,capture=False)
    assert len(calls)==3
    if horizon==10:
        dense=r.generate_long_horizon_dataset({},e,bank,schedule,{},tmp_path/'dense',
            outer=12,stage='SYNTHETIC_INTERFACE_ONLY',horizon=10,branches=2,capture=False,sparse_terminal=False)
        dense_rows=pd.concat(list(dense.rows()),ignore_index=True)
        a=raw.loc[raw.economic_state_valid].reset_index(drop=True)
        b=dense_rows.loc[dense_rows.economic_state_valid].reset_index(drop=True)
        for name in ('parent_candidate_id','t','child_branch','C_raw_t1','K_cal_t1','Hatc_cal_t1','LnK_cal_t1'):
            np.testing.assert_allclose(a[name],b[name],rtol=0,atol=0)
        assert len(dense.manifest['shards'])==10


def test_h100_design_real_capital_and_disjoint_stratified_origins():
    from test_entry_value_cost import _value_cost_sim
    from data.simulate_ts_parallel import _initialize_batched_state
    from data.finite_direct_h100_initial import design_initial
    sim,_=_value_cost_sim();sim.group_size=200;state=_initialize_batched_state(sim,200)
    reference=dict(state=state)
    a,sa,ma=design_initial(reference,outer=12,mode='ROOT_CONSISTENT')
    b,sb,mb=design_initial(reference,outer=12,mode='OFF_EQUILIBRIUM_INITIAL_BELIEF')
    assert len(a['state']['x'])==len(b['state']['x'])==100
    assert not set(ma['initial_ids'])&set(mb['initial_ids'])
    torch.testing.assert_close(b['state']['K'].sum(1).log(),b['state']['lnkf'])
    assert ((b['state']['hatcf']>=-6)&(b['state']['hatcf']<=2)).all()
    assert len({s['coverage_family'] for s in sb['parents']})==7
    assert len({s['rank_mode'] for s in sb['parents']})==5
    assert all(k['transition_seed']==sb['seed']+11 and k['entry_seed']==sb['seed']+29
        for p in sb['parents'] for k in p['child_shock_keys'])
    assert ma['root_required'] and not mb['root_required']
    again,_,_=design_initial(reference,outer=12,mode='OFF_EQUILIBRIUM_INITIAL_BELIEF')
    assert again['fingerprint']==b['fingerprint']
