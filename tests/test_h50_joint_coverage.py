"""Non-optimizing contracts. Synthetic ledgers are never frozen evidence."""
from dataclasses import replace,asdict
from types import SimpleNamespace
from copy import deepcopy
from pathlib import Path
import json,math
import numpy as np
import pandas as pd
import pytest
import torch
from data.finite_direct_joint_coverage import (JointCoverageConfig,beta_values,propose_beta_batch,joint_bin,
    source_allocation,JointReservoir,eligible_templates,certify_batch,build_joint_bank)
from data.finite_direct_sampling import state_hash,file_hash
from data.entry import compute_entry_reference
from data.finite_direct_initial_hstar import InitialCalibrationBlocked
from analysis.nested_equilibrium_utils import atomic_torch_save,atomic_write_json
from analysis.finite_direct_continuation_state import state_digest
from data.finite_direct_long_horizon_runtime import node_diagnostics,ShardStore
from analysis.h50_joint_diagnostics import write_dashboard,compare_validation_stages,error_metrics


@pytest.fixture(autouse=True)
def forbid_optimization(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('optimization forbidden')
    for cls in (torch.optim.Adam,torch.optim.AdamW,torch.optim.LBFGS):monkeypatch.setattr(cls,'step',forbidden)


def reference(n=64,firms=200):
    k=torch.linspace(.1,1,firms).repeat(n,1);k*=60/k.sum(1,keepdim=True)
    state=dict(K=k,b=torch.zeros_like(k),bp=torch.zeros_like(k),K_birth=k.clone(),z=torch.zeros_like(k),
        eta=torch.zeros_like(k),i=torch.zeros_like(k),alive=torch.ones_like(k,dtype=torch.bool),
        firm_mass=torch.ones_like(k),x=torch.full((n,),-2.),hatcf=torch.full((n,),-2.),lnkf=k.sum(1).log(),
        transition_path_id=torch.arange(n)+2000000)
    return dict(state=state,seed=1,fingerprint=state_hash(state))


@pytest.mark.parametrize('family',['low','medium','high','broad','u_shaped','bimodal','upper_tail'])
def test_family_statistics_and_legal_support(family):
    c=JointCoverageConfig();b=beta_values(family,20000,np.random.default_rng(8),c)
    assert b.min()>=0 and b.max()<=1
    if family=='low':assert .13<b.mean()<.16
    if family=='medium':assert .48<b.mean()<.52 and b.std()<.15
    if family=='high':assert .84<b.mean()<.88
    if family=='broad':assert .48<b.mean()<.52 and .27<b.std()<.30
    if family=='u_shaped':assert ((b<.1)|(b>.9)).mean()>.4
    if family=='bimodal':assert ((b<.25)|(b>.75)).mean()>.7
    if family=='upper_tail':assert (b>.7).mean()>.1


def test_proposal_reproducible_and_K_entry_reference_scaled():
    c=JointCoverageConfig();base=reference();old=state_digest(base)
    a,specs=propose_beta_batch(base,start=0,count=20,outer=12,cfg=c)
    b,_=propose_beta_batch(base,start=0,count=20,outer=12,cfg=c)
    assert state_hash(a)==state_hash(b) and old==state_digest(base)
    assert set(s['K_scale'] for s in specs)==set(c.k_scales)
    assert torch.allclose(a['lnkf'],a['K'].sum(1).log())
    for j,p in enumerate(specs):
        assert float(a['K'][j].sum())==pytest.approx(p['K_total_requested'],rel=2e-6)
        index=((j*97+c.seed)%len(base['state']['x']))
        r0=compute_entry_reference(base['state']['K'][index:index+1],base['state']['alive'][index:index+1],entry_capital_ratio=.05,entry_size_ratio=1.)
        r1=compute_entry_reference(a['K'][j:j+1],a['alive'][j:j+1],entry_capital_ratio=.05,entry_size_ratio=1.)
        assert torch.allclose(r1.K_per_entrant,r0.K_per_entrant*p['K_scale'],rtol=2e-6,atol=1e-7)
        assert torch.allclose(r1.n_star,r0.n_star)


def test_rank_coupling_does_not_change_z_or_b_marginal():
    base=reference();a,_=propose_beta_batch(base,start=0,count=8,outer=12,cfg=replace(JointCoverageConfig(),rank_modes=('independent',)))
    b,specs=propose_beta_batch(base,start=0,count=8,outer=12,cfg=replace(JointCoverageConfig(),rank_modes=('z_positive',)))
    assert torch.equal(a['z'],b['z']) and torch.equal(a['K'],b['K'])
    assert torch.equal(a['b'].sort(1).values,b['b'].sort(1).values)
    assert min(p['corr_b_z'] for p in specs)>.7


@pytest.mark.parametrize('emp,beta,expected',[(0,8192,(0,8192)),(1000,8192,(1000,7192)),(5000,8192,(4096,4096)),(1000,10,(1000,10))])
def test_empirical_shortfall_counts_unique_capacity(emp,beta,expected):
    a=source_allocation(8192,emp,beta);assert (a['empirical'],a['beta'])==expected
    assert a['underfilled']==8192-sum(expected)


def test_template_lineage_cap_accounting_and_commit():
    template=dict(split='train',source_outer=11,source_time=1,cluster_id='p1',template_id='a',
        post_decision_accounting_valid=True,b_sorted=[.1,.2],b_by_capital_quartile=[[.1]]*4)
    one=deepcopy(template);one['template_id']='b'
    unknown=dict(template,template_id='c',cluster_id='p2',post_decision_accounting_valid='NOT_YET_EVALUATED')
    rows,reasons=eligible_templates([template,one,unknown],outer=12,cfg=JointCoverageConfig(),committed=True)
    assert len(rows)==1 and reasons['LINEAGE_REUSE_CAP']==1 and reasons['ACCOUNTING_UNVERIFIED_OR_INVALID']==1
    assert not eligible_templates([template],outer=12,cfg=JointCoverageConfig(),committed=False)[0]
    assert not eligible_templates([dict(template,split='validation')],outer=12,cfg=JointCoverageConfig(),committed=True)[0]


def test_joint_bins_reservoir_caps_and_determinism():
    cfg=replace(JointCoverageConfig(),bin_cap=3)
    assert joint_bin(-2.2,2.3,cfg)==(0,0)
    assert joint_bin(.4,6.,cfg)==(7,7)
    assert joint_bin(-99.,3.,cfg)==(-1,1)  # H_LOW / ELL_IN, not the old collapsed overflow
    a=JointReservoir(5,cfg);b=JointReservoir(5,cfg)
    for j in range(30):
        r=dict(parent_id=j,coverage_bin=(j%2,0),coverage_family=str(j%3))
        a.add(r);b.add(r)
    assert a.count<=5 and a.selected(5)==b.selected(5)
    assert max(sum(len(v) for (cell,f),v in a.buckets.items() if cell==bin_) for bin_ in [(0,0),(1,0)])<=3


def test_root_failure_isolation_never_fakes_h(monkeypatch):
    import data.finite_direct_joint_coverage as m
    state=reference(3)['state'];calls=[]
    def strict(sim,s,**kw):
        ids=s['transition_path_id'].tolist();calls.append(ids)
        if 2000001 in ids:raise InitialCalibrationBlocked('NO_BRACKET',{'failures':[{'root_status':'NO_BRACKET_ON_TESTED_GRID'}]})
        rows=[dict(parent_candidate_id=int(pid),C_raw_t1=1.,K_cal_t1=2.,Hatc_cal_t1=math.log(.5),LnK_cal_t1=math.log(2),economic_state_valid=True) for pid in ids]
        return s,rows,list(range(len(ids))),[{} for _ in ids]
    monkeypatch.setattr(m,'calibrate_initial',strict)
    good,bad=certify_batch(SimpleNamespace(device='cpu'),state,outer=12,cfg=JointCoverageConfig(),economic={},crn={})
    assert len(good)==2 and len(bad)==1 and bad[0]['parent_id']==2000001
    assert len(calls[0])==3


def test_runtime_reductions_are_read_only_and_hard_not_soft():
    s=reference(1,3)['state'];s['bar_z']=torch.tensor([[.1,.6,.9]])
    old=state_digest(s)
    rows=[dict(parent_candidate_id=2000000,C_raw_t1=1.,K_cal_t1=2.,Hatc_cal_t1=math.log(.5),LnK_cal_t1=math.log(2))]
    node_diagnostics(s,rows,entry_attempted=False)
    assert rows[0]['hard_default_count']==2 and rows[0]['post_exit_alive_firms']==1
    assert rows[0]['entry_candidate_count'] is None
    assert state_digest(s)==old


def diagnostic_fixture(tmp_path):
    def row(pid,t,b,valid):
        return dict(parent_candidate_id=pid,t=t,child_branch=b,split='validation',economic_state_valid=valid,
            node_attempted=True,continuation_allowed=valid,C_raw_t1=1. if valid else 0.,K_cal_t1=2. if valid else 0.,
            Hatc_cal_t1=math.log(.5) if valid else float('nan'),LnK_cal_t1=math.log(2) if valid else float('nan'),
            Hatcf_t1=-.5,LnKF_t1=1.,Hatc_cal_t=-.7,LnK_cal_t=.7,x_t=-2.,x_t1=-2.,
            sampling_source='BETA',b_distribution_family='low',failure_reason='' if valid else 'EXTINCT',
            predecision_alive_firms=2,hard_default_count=0 if valid else 2,post_exit_alive_firms=2 if valid else 0,
            entry_attempted=True,entry_candidate_count=0,entry_accepted_count=0)
    initial=[row(pid,0,-1,True) for pid in (1,2)]
    shards={}
    for t in range(1,51):
        rows=[row(pid,t,b,(t==1 and (pid==1 or b==1))) for pid in (1,2) for b in (0,1)]
        if t>2:
            for r in rows:r['node_attempted']=False
        p=tmp_path/f'time_{t:02d}.pt';atomic_torch_save(p,dict(rows=rows,parent_records=initial if t==1 else [],attempted_parent_ids=[1,2] if t==1 else [1] if t==2 else []))
        shards[p.name]=file_hash(p)
    atomic_write_json(tmp_path/'manifest.json',dict(scheduled_initial_parents=2,shards=shards,
        provenance=dict(outer=12,stage='D0',split='validation',crn_hash='fixed',operator='exitv2',parent_bank_hash='bank',
            model_hashes={k:'hash' for k in ('firm','sdf','wealth','fc1_c','fc1_k')})))
    return ShardStore(tmp_path)


def test_survival_counts_not_pairs_and_late_errors_nan(tmp_path):
    store=diagnostic_fixture(tmp_path);out=tmp_path/'diagnostics'
    f=write_dashboard(store,out)
    assert f.loc[f.t==0,'main_spine_alive_paths'].iloc[0]==2
    assert f.loc[f.t==1,'main_spine_alive_paths'].iloc[0]==1
    assert f.loc[f.t==2,'main_spine_alive_paths'].iloc[0]==0
    assert f.loc[f.t==1,'paths_with_at_least_one_valid_child'].iloc[0]==2
    assert f.loc[f.t==1,'complete_sdf_pairs'].iloc[0]==1
    assert f.loc[f.t==2,'cumulative_terminated_paths'].iloc[0]==2
    errors=pd.read_csv(out/'macro_forecast_error_by_time.csv')
    late=errors[(errors.t==20)&(errors.macro_coverage_bin=='ALL')]
    assert late.metric_valid_count.eq(0).all() and late.RMSE.isna().all()


def test_error_sign_and_no_data():
    m=error_metrics([-2.,1.]);assert m['signed_bias']==-.5 and m['MAE']==1.5
    assert math.isnan(error_metrics([])['RMSE'])


@pytest.mark.parametrize('change',[dict(k_scales=()),dict(x_quantiles=()),dict(rank_modes=('typo',))])
def test_empty_or_unknown_proposal_configuration_rejected(change):
    with pytest.raises(ValueError):replace(JointCoverageConfig(),**change).validate()


def test_uncommitted_stream_is_rejected_without_assuming_list():
    rows,reasons=eligible_templates(({} for _ in range(3)),outer=12,cfg=JointCoverageConfig(),committed=False)
    assert rows==[] and reasons=={'UNCOMMITTED_POOL':3}


def test_candidate_bank_content_resume_and_summary(tmp_path,monkeypatch):
    import data.finite_direct_joint_coverage as m
    import analysis.run_fc1_self_consistent_macro as backend
    monkeypatch.setattr(backend,'hashes',lambda models: {'firm':'unchanged'})
    calls=[]
    def certify(sim,s,**kwargs):
        calls.append(len(s['x']))
        rows=[dict(parent_candidate_id=int(pid),C_raw_t1=1.,K_cal_t1=10.,Hatc_cal_t1=math.log(.1),
            LnK_cal_t1=math.log(10.),economic_state_valid=True) for pid in s['transition_path_id']]
        return s,rows,list(range(len(rows))),[{'normalized_residual':0.} for _ in rows]
    monkeypatch.setattr(m,'calibrate_initial',certify)
    cfg=replace(JointCoverageConfig(),max_candidates=4,candidate_batch_size=2)
    kw=dict(reference=reference(4),templates=[],outer=12,target=4,cfg=cfg,output=tmp_path,
        simulator_factory=lambda n:SimpleNamespace(models={},device='cpu'),economic={})
    bank,plan,stats=build_joint_bank(**kw)
    assert stats['economic_valid']==4 and stats['allocation']['empirical']==0
    phases=pd.read_csv(tmp_path/'coverage_phase_summary.csv')
    assert phases.loc[phases.phase=='PROPOSED','h_cal_p50'].isna().all()
    count=len(calls);again=build_joint_bank(**kw)
    assert len(calls)==count and again[0]['fingerprint']==bank['fingerprint']
    with (tmp_path/'selected_bank.pt').open('ab') as stream:stream.write(b'corrupt')
    with pytest.raises(ValueError,match='corruption'):build_joint_bank(**kw)


def test_comparison_rejects_changed_validation_crn(tmp_path):
    p=tmp_path/'v';p.mkdir();write_dashboard(diagnostic_fixture(p),p/'diagnostics')
    q=tmp_path/'q';q.mkdir();meta=json.loads((p/'diagnostics/stage_provenance.json').read_text());meta['rng_or_crn_identifier']='different'
    atomic_write_json(q/'stage_provenance.json',meta)
    pd.read_csv(p/'diagnostics/path_survival_by_time.csv').to_csv(q/'path_survival_by_time.csv',index=False)
    with pytest.raises(ValueError,match='UNFAIR'):compare_validation_stages({'D0':p/'diagnostics','D1':q},tmp_path/'compare')
